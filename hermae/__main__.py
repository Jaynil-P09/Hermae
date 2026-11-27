"""Run the seeded ZTF and Legacy Surveys host-matching pilot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .diagnostics import fetch_panel_cutout, panel
from .matching import evaluate
from .survey import (BRICKS, LAYER, VIEWER, ZTF, Cache, SourceError, footprint_bricks,
                     has_brick, load_events, point_coverage)


def select_events(cache, events, size, seed, pool):
    bricks = cache.get(BRICKS, "release", ".fits.gz")
    bounds = footprint_bricks(bricks)
    events = events.sort_values("event_id").copy()
    # A hash permutation is independent of the incoming row order and library RNG.
    events["sort_key"] = events.event_id.map(lambda name: sha256(f"{seed}:{name}".encode()).hexdigest())
    eligible = [
        event for event in events.sort_values("sort_key").itertuples()
        if pd.notna(event.ra) and pd.notna(event.dec) and has_brick(event.ra, event.dec, bounds)
    ]
    rows = []
    selected = []
    for start in range(0, len(eligible), 6):
        batch = eligible[start:start + 6]
        for event, (status, bands, error) in zip(
            batch, pool.map(lambda item: point_coverage(cache, item), batch)
        ):
            include = status == "covered" and len(selected) < size
            rows.append(dict(event_id=event.event_id, coverage_status=status, coverage_bands=bands,
                             coverage_error=error, selected=include))
            if include:
                selected.append(event.event_id)
        if len(selected) == size:
            break
    audit = pd.DataFrame(rows, columns=["event_id", "coverage_status", "coverage_bands",
                                      "coverage_error", "selected"])
    cohort = events.drop(columns="sort_key").set_index("event_id").loc[selected].reset_index()
    cohort = cohort.merge(audit.drop(columns="selected"), on="event_id", validate="one_to_one")
    return cohort, audit, bricks, len(eligible)


def render_panels(cache, path, events, evaluation, candidates, radius, count, workers):
    directory = path / "panels"
    directory.mkdir(exist_ok=True)
    # Give disagreements and failed associations priority in the inspection sample.
    order = evaluation.assign(
        priority=evaluation.review_status.map({"possible_missing_host": 0, "review": 2, "unflagged": 3}),
    )
    order.loc[order.reference_status.ne("unique_galaxy") & order.priority.gt(1), "priority"] = 1
    order = order.sort_values(["priority", "event_id"])
    if not count:
        return []
    events_by_id = {event.event_id: event for event in events.itertuples()}
    records = order.to_dict("records")

    rows = []
    rendered = 0
    start = 0

    def fetch(record):
        event = events_by_id[record["event_id"]]
        try:
            return fetch_panel_cutout(cache, event, radius), ""
        except SourceError as exc:
            return None, str(exc)

    with ThreadPoolExecutor(workers) as pool:
        while start < len(records) and rendered < count:
            batch = records[start:start + count - rendered]
            start += len(batch)
            for record, (cutout, error) in zip(batch, pool.map(fetch, batch)):
                event_id = record["event_id"]
                if cutout is None:
                    rows.append(dict(event_id=event_id, status="failed", error=error))
                    continue
                try:
                    filename = panel(*cutout, directory, events_by_id[event_id], record,
                                     candidates.loc[candidates.event_id == event_id], radius)
                except (SourceError, OSError, ValueError, IndexError, TypeError) as exc:
                    rows.append(dict(event_id=event_id, status="failed", error=str(exc)))
                    continue
                rows.append(dict(event_id=event_id, status="rendered", fits_cache_file=filename,
                                 image=f"panels/{event_id}.png", review_reason=record["review_reason"],
                                 reference_status=record["reference_status"]))
                rendered += 1
                print(f"Rendered {event_id}", flush=True)
    return rows


def counts(series):
    return ", ".join(f"{label} `{count}`" for label, count in series.value_counts().items())


def fraction(numerator, denominator):
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})" if denominator else "n/a"


def report(events, evaluation, audit, radii, panels, options):
    with_host = events.label_provenance.ne("missing")
    success = evaluation.catalogue_status.ne("query_failed")
    comparable = with_host & success & evaluation.reference_status.ne("ambiguous")
    retrieved = evaluation[f"reference_retrieved_{radii[0]}as"].eq(True)
    disagreement = evaluation.nearest_pick.notna() & evaluation.nearest_pick.ne(evaluation.dlr_pick)
    lines = [
        "# Hermae ZTF SN Ia DR2 pilot", "",
        f"Seed {options.seed} selected {len(events)} of {options.events} requested supernovae "
        "with positive optical inverse variance at the transient position in Legacy Surveys DR10 south.",
        f"Footprint checks for {len(audit)} events yielded {counts(audit.coverage_status)}.", "",
        "## Candidate retrieval", "",
        f"Published host coordinates were present for {with_host.sum()} of {len(events)} events.",
        f"Catalogue outcomes were {counts(evaluation.catalogue_status)}.",
        f"Cross-match outcomes at a {options.match_arcsec:g}″ tolerance were "
        f"{counts(evaluation.reference_status)}. "
        f"{evaluation.reference_duplicate_rows.fillna(0).gt(0).sum()} reference positions also had a Gaia "
        "`DUP` row within tolerance. These rows are logged and excluded from counterpart counts.", "",
    ]
    for radius in radii:
        found = evaluation[f"reference_retrieved_{radius}as"].eq(True).sum()
        lines.append(f"- **{radius}″** recovered {fraction(found, int(comparable.sum()))} of references "
                     f"with a determinate match outcome and {fraction(found, int(with_host.sum()))} "
                     "of all published coordinates as confirmed matches.")
    lines += ["", f"## Baseline agreement at {radii[0]}″", "",
              "Each method ranks the same galaxy pool. Agreement is with the ZTF published "
              "DLR-selected assignments. Independent verification was outside this pilot.", ""]
    for method, name in (("nearest", "Nearest angular neighbour"), ("dlr", "Tractor half-light DLR")):
        rank = pd.to_numeric(evaluation[f"{method}_reference_rank"])
        top1, top3 = rank.eq(1).sum(), rank.le(3).sum()
        lines.append(f"- **{name}** top-1 {fraction(top1, int(retrieved.sum()))} and "
                     f"top-3 {fraction(top3, int(retrieved.sum()))} among retrieved references. "
                     f"Across all published coordinates, confirmed top-1 is "
                     f"{fraction(top1, int(with_host.sum()))} and confirmed top-3 is "
                     f"{fraction(top3, int(with_host.sum()))}.")
    lines += ["", f"The methods disagreed on {disagreement.sum()} events. "
              f"DLR used a nearest fallback on {evaluation.dlr_fallback.eq(True).sum()} events.",
              f"Review outcomes were {counts(evaluation.review_status)}.", "",
              "## Redshift provenance and inspection", "",
              f"Published redshift sources were {counts(events.redshift_source.fillna('missing'))}. "
              f"{events.redshift_provenance.eq('unknown').sum()} source codes were missing or unresolved "
              f"and {events.redshift_provenance.eq('transient_template').sum()} identified a transient "
              "template estimate. The baselines used only positions and morphology.", "",
              ]
    drawn = [item for item in panels if item["status"] == "rendered"]
    if drawn:
        lines += [f"{len(drawn)} FITS-based panels prioritize difficult cases for inspection. "
                  "This subset is drawn from the pilot.", ""]
        for item in drawn:
            lines.append(f"- [{item['event_id']}]({item['image']}) · "
                         f"{item['review_reason'].replace('_', ' ').replace('|', ', ') or item['reference_status'].replace('_', ' ')}")
    if any(item["status"] == "failed" for item in panels):
        lines += ["", "Failed panel requests are recorded in the panel manifest."]
    lines += ["", "## Sources", "",
              "- [ZTF SN Ia DR2 overview](https://doi.org/10.1051/0004-6361/202450388)",
              "- [Legacy Surveys DR10 Tractor catalogue](https://www.legacysurvey.org/dr10/catalogs/)",
              "- [Legacy Surveys DR10 image access](https://www.legacysurvey.org/dr10/description/)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=60, help="baseline radius in arcseconds")
    parser.add_argument("--wider", type=int, nargs="*", default=[90, 120], help="wider radii in arcseconds")
    parser.add_argument("--match-arcsec", type=float, default=1.5)
    parser.add_argument("--panels", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, help="defaults to results/pilot or results/single")
    parser.add_argument("--offline", action="store_true", help="use cached downloads only")
    parser.add_argument("--position", metavar=("RA", "DEC"), nargs=2, type=float,
                        help="rank one J2000 position in degrees instead of running the pilot")
    options = parser.parse_args()
    radii = [options.radius, *sorted(set(options.wider))]
    if (options.events < 1 or options.workers < 1 or options.panels < 0 or
            not 0 < options.match_arcsec <= 10 or not 0 < options.radius <= 300 or
            any(not options.radius < radius <= 300 for radius in options.wider)):
        parser.error("use positive events/workers, nonnegative panels, a cross-match tolerance in "
                     "(0, 10], and wider search radii above the baseline radius and at most 300″")
    if options.output is None:
        options.output = Path("results/single" if options.position is not None else "results/pilot")
    options.output.mkdir(parents=True, exist_ok=True)
    cache = Cache(options.cache, options.offline)
    if options.position is not None:
        ra, dec = options.position
        if not (0 <= ra < 360 and -90 <= dec <= 90):
            parser.error("position requires 0 <= RA < 360 and -90 <= Dec <= 90")
        event = SimpleNamespace(event_id="input_position", ra=ra, dec=dec, host_ra=np.nan,
                                host_dec=np.nan)
        coverage, bands, error = point_coverage(cache, event)
        evaluation, candidates = evaluate(cache, event, radii, options.match_arcsec)
        evaluation.update(ra=ra, dec=dec, coverage_status=coverage, coverage_bands=bands,
                          coverage_error=error)
        if coverage != "covered" and evaluation["catalogue_status"] != "query_failed":
            evaluation.update(review_status="review", review_reason=coverage)
        candidates.to_csv(options.output / "candidates.csv", index=False)
        pd.DataFrame([evaluation]).to_csv(options.output / "evaluation.csv", index=False)
        print(f"{evaluation['review_status']} · outputs written to {options.output}", flush=True)
        if evaluation["catalogue_status"] == "query_failed":
            raise SystemExit("Catalogue request failed. See evaluation.csv.")
        return
    release, files = load_events(cache)
    with ThreadPoolExecutor(options.workers) as pool:
        events, audit, bricks, brick_count = select_events(cache, release, options.events, options.seed, pool)
        audit.to_csv(options.output / "coverage_audit.csv", index=False)
        if events.empty:
            raise SystemExit(f"No covered events found. See {options.output / 'coverage_audit.csv'}.")
        print(f"Selected {len(events)} events after {len(audit)} footprint checks", flush=True)
        evaluations, candidate_frames = [], []
        for row, frame in pool.map(lambda event: evaluate(cache, event, radii, options.match_arcsec),
                                   events.itertuples()):
            evaluations.append(row)
            candidate_frames.append(frame)
            print(f"{len(evaluations)}/{len(events)} {row['event_id']} {row['catalogue_status']} "
                  f"{row['reference_status']}", flush=True)

    evaluation = pd.DataFrame(evaluations)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    events["transient_release"] = "ZTF SN Ia DR2"
    events["catalogue_release"] = "Legacy Surveys DR10 south"
    events = events.merge(evaluation[["event_id", "catalogue_status"]], on="event_id", validate="one_to_one")
    events.to_csv(options.output / "events.csv", index=False)
    candidates.to_csv(options.output / "candidates.csv", index=False)
    evaluation.to_csv(options.output / "evaluation.csv", index=False)
    panels = render_panels(cache, options.output, events, evaluation, candidates, radii[0],
                           options.panels, options.workers)
    files["dr10_south_bricks"] = bricks
    hashes = {}
    for name, path in files.items():
        with path.open("rb") as file:
            hashes[name] = file_digest(file, "sha256").hexdigest()
    manifest = dict(seed=options.seed, events_requested=options.events, events_selected=len(events),
                    release_events=len(release), events_in_processed_bricks=brick_count,
                    search_radii_arcsec=radii, reference_tolerance_arcsec=options.match_arcsec,
                    ztf_base=ZTF, catalogue_endpoint=f"{VIEWER}/{LAYER}/cat.fits",
                    fits_endpoint=f"{VIEWER}/fits-cutout", footprint_bricks=BRICKS,
                    release_sha256=hashes, panels=panels)
    (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (options.output / "report.md").write_text(
        report(events, evaluation, audit, radii, panels, options), encoding="utf-8")
    print(f"Pilot outputs written to {options.output}", flush=True)
    if len(events) < options.events or sum(item["status"] == "rendered" for item in panels) < options.panels:
        raise SystemExit("Pilot did not reach the requested event or panel count.")


if __name__ == "__main__":
    main()
