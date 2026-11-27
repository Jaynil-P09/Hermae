"""Train catalogue rankers and compare them on the frozen validation split."""

import argparse
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .splits import input_hashes

GEOMETRY = ["log_separation", "log_dlr"]
PHOTOMETRY = ["r_mag", "g_minus_r", "r_minus_i", "r_minus_z"]
MORPHOLOGY = ["log_shape_r", "axis_ratio", "sersic_index", "type_REX", "type_EXP", "type_DEV", "type_SER"]
FEATURES = {"geometry": GEOMETRY,
            "catalogue": [*GEOMETRY, *PHOTOMETRY, *MORPHOLOGY, "neighbours_5as"]}
METHODS = {"nearest": "Nearest angular neighbour", "dlr": "Tractor half-light DLR",
           "geometry": "Logistic regression, geometry", "catalogue": "Logistic regression, full catalogue"}
RAW_FEATURES = ["event_id", "source_id", "ra", "dec", "separation_arcsec", "dlr", "shape_r",
                "axis_ratio", "sersic", "type",
                *[f"{name}_{band}" for name in ("flux", "flux_ivar", "nobs") for band in "griz"]]


def catalogue_features(rows):
    """Derive features from a single field's retrieved candidate pool."""
    features = pd.DataFrame(index=rows.index)
    for name in ("separation_arcsec", "dlr", "shape_r"):
        safe = rows[name].where(np.isfinite(rows[name]) & (rows[name] >= 0))
        output = "log_separation" if name == "separation_arcsec" else f"log_{name}"
        features[output] = np.log1p(safe)
    mags = {}
    for band in "griz":
        flux = rows[f"flux_{band}"]
        measured = (flux > 0) & (rows[f"flux_ivar_{band}"] > 0) & (rows[f"nobs_{band}"] > 0)
        mags[band] = 22.5 - 2.5 * np.log10(flux.where(measured & np.isfinite(flux)))
    features["r_mag"] = mags["r"]
    features["g_minus_r"] = mags["g"] - mags["r"]
    features["r_minus_i"] = mags["r"] - mags["i"]
    features["r_minus_z"] = mags["r"] - mags["z"]
    features["axis_ratio"] = rows.axis_ratio
    features["sersic_index"] = rows.sersic.where(rows.type.eq("SER"))
    for morphology in ("REX", "EXP", "DEV", "SER"):
        features[f"type_{morphology}"] = rows.type.eq(morphology).astype(float)
    coord = SkyCoord(rows.ra.to_numpy(), rows.dec.to_numpy(), unit="deg")
    left, right, _, _ = search_around_sky(coord, coord, 5 * u.arcsec)
    features["neighbours_5as"] = np.bincount(left[left != right], minlength=len(rows))
    return features


def ranker(columns, features, target, weights):
    estimator = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000),
    )
    estimator.fit(features[columns], target, logisticregression__sample_weight=weights)
    return estimator


def compare(run, manifest):
    assignment = pd.DataFrame(manifest["assignments"]).set_index("event_id")
    files = input_hashes(run)
    if files != manifest["source_sha256"]:
        raise ValueError("Export files differ from the frozen benchmark inputs.")
    radius = manifest["search_radii_arcsec"][0]
    retrieved = f"reference_retrieved_{radius}as"
    eval_columns = ["event_id", "reference_source_id", "reference_status", "host_ra", "host_dec",
                    retrieved, "nearest_reference_rank", "dlr_reference_rank"]
    train_ids = assignment.index[assignment.split.eq("train")]
    val_ids = assignment.index[assignment.split.eq("validation")]
    evaluation = pd.read_csv(run / "evaluation.csv", usecols=eval_columns, index_col="event_id")
    train_eval = evaluation.loc[train_ids]
    validation = evaluation.loc[val_ids].join(assignment[["group_id"]]).copy()
    candidates = pd.read_csv(run / "candidates.csv", usecols=[*RAW_FEATURES, "in_baseline_cone"])
    candidates = candidates.loc[candidates.event_id.isin(train_ids.union(val_ids))].copy()
    features = pd.concat([catalogue_features(field)
                          for _, field in candidates.groupby("event_id", sort=False)])
    rows = candidates.loc[candidates.in_baseline_cone].copy()
    training = rows.loc[rows.event_id.isin(train_eval.index[train_eval[retrieved].eq(True)])].copy()
    in_event = training.groupby("event_id").source_id.transform("size")
    training = training.loc[in_event > 1]
    target = training.source_id.eq(training.event_id.map(train_eval.reference_source_id))
    if target.nunique() != 2 or target.sum() != training.event_id.nunique():
        raise ValueError("Training requires one valid counterpart and at least one negative per event.")
    weights = np.where(target, 1.0, 1.0 / (in_event.loc[training.index] - 1))
    val_rows = rows.loc[rows.event_id.isin(val_ids)].copy()
    if val_rows.empty:
        raise ValueError("The validation split has no baseline-cone candidates.")
    scored = val_rows[["event_id", "source_id", "separation_arcsec"]].copy()
    for method, columns in FEATURES.items():
        estimator = ranker(columns, features.loc[training.index], target, weights)
        score = f"{method}_score"
        rank = f"{method}_rank"
        scored[score] = estimator.decision_function(features.loc[scored.index, columns])
        scored = scored.sort_values(["event_id", score, "separation_arcsec", "source_id"],
                                    ascending=[True, False, True, True])
        scored[rank] = scored.groupby("event_id").cumcount() + 1
        rank_lookup = scored.set_index(["event_id", "source_id"])[rank]
        reference_keys = pd.MultiIndex.from_arrays([validation.index, validation.reference_source_id])
        validation[f"{method}_reference_rank"] = rank_lookup.reindex(reference_keys).to_numpy()
    return (validation, scored.sort_values(["event_id", "catalogue_rank"]),
            int(training.event_id.nunique()), retrieved)


def paired_interval(validation, method, seed):
    """Group-bootstrap the paired change in first-rank agreement against DLR."""
    method_hit = validation[f"{method}_reference_rank"].eq(1).astype(int)
    dlr_hit = validation.dlr_reference_rank.eq(1).astype(int)
    paired = (method_hit - dlr_hit).groupby(validation.group_id).agg(["sum", "count"])
    if len(paired) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(paired), size=(4000, len(paired)))
    differences = paired["sum"].to_numpy()[samples].sum(axis=1)
    total = paired["count"].to_numpy()[samples].sum(axis=1)
    return np.quantile(differences / total, [0.025, 0.975])


def validation_report(manifest, validation, train_count, retrieved_column):
    total = len(validation)
    with_reference = validation[["host_ra", "host_dec"]].notna().all(axis=1).sum()
    recovered = validation.loc[validation[retrieved_column].eq(True)]
    count = len(recovered)
    radius = manifest["search_radii_arcsec"][0]
    held_out = sum(row["split"] == "test" for row in manifest["assignments"])
    lines = [
        "# Hermae catalogue ranking validation", "",
        f"The frozen split reserved {held_out} test events. This run scored {total} validation events "
        f"and trained both logistic rankers on {train_count} events with one recovered match "
        "and at least one alternative candidate.", "",
        f"Within {radius}″, {count}/{with_reference} validation events with published coordinates "
        "had a uniquely recovered catalogue counterpart. The comparisons use the same candidate pool. "
        "Agreement is measured against ZTF's published DLR-selected reference.", "",
        "## Agreement with the published reference", "",
    ]
    for method, name in METHODS.items():
        ranks = recovered[f"{method}_reference_rank"]
        top1, top3 = ranks.eq(1).sum(), ranks.le(3).sum()
        if count:
            line = (f"- **{name}** ranked {top1}/{count} ({top1 / count:.1%}) first and "
                    f"{top3}/{count} ({top3 / count:.1%}) in the top three. Confirmed first ranks "
                    f"across all published coordinates were {top1}/{with_reference}.")
        else:
            line = f"- **{name}** had no uniquely recovered reference to score."
        lines.append(line)
    lines += ["", "## Paired change versus DLR", ""]
    for method, name in METHODS.items():
        if method == "dlr" or not count:
            continue
        change = recovered[f"{method}_reference_rank"].eq(1).mean() - recovered.dlr_reference_rank.eq(1).mean()
        interval = paired_interval(recovered, method, manifest["split_seed"])
        uncertainty = "" if interval is None else f" (95% group bootstrap interval {interval[0]:+.1%} to {interval[1]:+.1%})"
        lines.append(f"- **{name}** {change * 100:+.1f} percentage points in first-rank agreement{uncertainty}.")
    return "\n".join([*lines, "", "Both models used only Tractor candidate observables and geometric offsets. "
                          "Output scores are uncalibrated logistic decision values.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="survey pipeline output directory")
    parser.add_argument("--split", type=Path, required=True, help="frozen split manifest JSON")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    try:
        manifest = json.loads(options.split.read_text(encoding="utf-8"))
        validation, rankings, train_count, retrieved = compare(options.run, manifest)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    options.output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(options.output / "validation_events.csv")
    rankings.to_csv(options.output / "validation_rankings.csv", index=False)
    report = validation_report(manifest, validation, train_count, retrieved)
    (options.output / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"Validation results written to {options.output}")


if __name__ == "__main__":
    main()
