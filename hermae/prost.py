"""Run upstream Pröst's offset-only scorer on frozen Hermae candidates."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import logging
from pathlib import Path

from astro_prost.helpers import GalaxyCatalog, Transient, build_galaxy_array, calc_dlr, calc_shape_props_decals
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.cosmology import Planck18
import numpy as np
import pandas as pd
from scipy.stats import gamma, uniform

from .splits import validation_selection

SAMPLES = 1000
SEED = 42
SHAPE = ["shape_r", "shape_e1", "shape_e2"]
UNCERTAINTY = [f"{name}_ivar" for name in SHAPE]
LOGGER = logging.getLogger(__name__)


def score_field(event, candidates, radius):
    scores = candidates[["event_id", "source_id", "separation_arcsec"]].copy()
    scores["prost_posterior"] = np.nan
    scores["prost_rank"] = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    usable = (
        np.isfinite(candidates[SHAPE]).all(axis=1)
        & candidates.shape_r.gt(0)
        & np.hypot(candidates.shape_e1, candidates.shape_e2).lt(1)
    )
    field = candidates.loc[usable].copy().rename(columns={"source_id": "objID"})
    if field.empty:
        return scores, None
    field = field.sort_values("objID")
    field[UNCERTAINTY] = field[UNCERTAINTY].fillna(0)
    # Shift the RA origin to keep the upstream offset subtraction away from the 0/360 seam.
    field["ra"] = (field.ra - event.ra + 180) % 360
    position = SkyCoord(180, event.dec, unit="deg")
    galaxy_positions = SkyCoord(field.ra.to_numpy(), field.dec.to_numpy(), unit="deg")
    rng = np.random.get_state()
    try:
        seed = int.from_bytes(sha256(f"{SEED}:{event.event_id}".encode()).digest()[:4], "big")
        np.random.seed(seed)
        transient = Transient(event.event_id, position, LOGGER, n_samples=SAMPLES)
        transient.set_prior("offset", uniform(loc=0, scale=10))
        transient.set_likelihood("offset", gamma(a=0.75))
        catalog = GalaxyCatalog("decals", release="dr10", n_samples=SAMPLES)
        catalog.search_rad = Angle(radius * u.arcsec)
        catalog.ngals = len(field)
        catalog.limiting_mag = 26
        catalog.galaxies, _ = build_galaxy_array(field, False, event.event_id, "decals", "dr10", LOGGER)
        with np.errstate(divide="ignore", invalid="ignore"):
            shapes = calc_shape_props_decals(field)
            light_radii = calc_dlr(transient.position_samples, galaxy_positions, *shapes, n_samples=SAMPLES)
            offsets = galaxy_positions[:, None].separation(transient.position_samples).arcsec
            catalog.galaxies["offset_samples"] = list(offsets)
            catalog.galaxies["dlr_samples"] = list(light_radii)
            transient.associate(catalog, Planck18, condition_host_props=["offset"])
    finally:
        np.random.set_state(rng)
    posterior = catalog.galaxies["total_posterior"]
    if not np.isfinite(posterior).all():
        raise ValueError(f"Non-finite Pröst candidate posterior for {event.event_id}")
    scores.loc[field.index, "prost_posterior"] = posterior
    # Match the main benchmark tie order. Preserve Pröst's separate Monte Carlo win choice.
    ranked = scores.loc[scores.prost_posterior.notna()].sort_values(
        ["prost_posterior", "separation_arcsec", "source_id"], ascending=[False, True, True])
    scores.loc[ranked.index, "prost_rank"] = range(1, len(ranked) + 1)
    winner = catalog.galaxies["objID"][transient.best_host] if transient.best_host >= 0 else None
    return scores, winner


def compare(run, split):
    ids, radius = validation_selection(run, split)
    events = pd.read_csv(run / "events.csv", usecols=["event_id", "ra", "dec"])
    candidates = pd.read_csv(run / "candidates.csv", usecols=[
        "event_id", "source_id", "ra", "dec", "separation_arcsec", "in_baseline_cone", *SHAPE, *UNCERTAINTY])
    candidates = candidates.loc[candidates.event_id.isin(ids) & candidates.in_baseline_cone]
    event_scores, choices = [], []
    for event in events.loc[events.event_id.isin(ids)].sort_values("event_id").itertuples():
        pool = candidates.loc[candidates.event_id.eq(event.event_id)]
        scored, winner = score_field(event, pool, radius)
        event_scores.append(scored)
        choices.append(dict(event_id=event.event_id, prost_native_pick=winner,
                            candidate_count=len(pool), candidates_scored=scored.prost_rank.notna().sum()))
    scores = pd.concat(event_scores).sort_values(["event_id", "prost_rank", "source_id"])
    retrieved = f"reference_retrieved_{radius}as"
    evaluation = pd.read_csv(run / "evaluation.csv", usecols=[
        "event_id", "reference_source_id", retrieved, "nearest_reference_rank", "dlr_reference_rank"])
    validation = evaluation.merge(pd.DataFrame(choices), on="event_id", validate="one_to_one")
    reference_ranks = scores[["event_id", "source_id", "prost_rank"]].rename(
        columns={"source_id": "reference_source_id", "prost_rank": "prost_reference_rank"})
    validation = validation.merge(reference_ranks, on=["event_id", "reference_source_id"], how="left",
                                  validate="one_to_one").sort_values("event_id")
    return scores, validation, retrieved, radius


def report(scores, validation, retrieved, radius):
    reference = validation.loc[validation[retrieved].eq(True)]
    lines = [
        "# Pröst offset-only comparison", "",
        f"Pröst {version('astro-prost')} scored {scores.prost_rank.notna().sum():,} of {len(scores):,} candidates "
        f"from the shared {radius}″ cones for {len(validation)} validation events.",
        f"The run used {SAMPLES:,} Monte Carlo draws per field and base seed {SEED}. The offset prior is "
        "uniform on [0, 10], the likelihood is Gamma with shape 0.75 and scale 1, and the nominal "
        "transient coordinate uncertainty is 0.1″ per axis.", "",
        f"Agreement with the published ZTF reference among {len(reference)} uniquely recovered counterparts.",
        "",
    ]
    for method, rank in (("Nearest angular neighbour", "nearest_reference_rank"),
                         ("Tractor half-light DLR", "dlr_reference_rank"),
                         ("Pröst median candidate posterior", "prost_reference_rank")):
        first = reference[rank].eq(1).sum()
        top_three = reference[rank].le(3).sum()
        lines.append(f"- **{method}** ranked {first}/{len(reference)} first and "
                     f"{top_three}/{len(reference)} in the top three.")
    winner_correct = reference.prost_native_pick.eq(reference.reference_source_id).sum()
    abstentions = validation.prost_native_pick.isna().sum()
    lines += ["", f"Pröst's native Monte Carlo winner agreed for {winner_correct}/{len(reference)} recovered "
              f"counterparts and returned no candidate in {abstentions}/{len(validation)} validation fields.",
              "", "This run calls upstream shape sampling and association on the shared Hermae candidates. "
              "The stock DECaLS query joins photo-z data and applies its own cuts. This offset experiment "
              "uses candidate shapes and separation only. Independent spectroscopy coverage is reported separately.",
              "", "[Pröst reference notebook](https://github.com/alexandergagliano/Prost/blob/main/docs/notebooks/associate.ipynb)",
              ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    scores, validation, retrieved, radius = compare(options.run, options.split)
    options.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(options.output / "prost_rankings.csv", index=False)
    validation.to_csv(options.output / "prost_validation.csv", index=False)
    (options.output / "prost_report.md").write_text(report(scores, validation, retrieved, radius), encoding="utf-8")
    print(f"Pröst comparison written to {options.output}.")


if __name__ == "__main__":
    main()
