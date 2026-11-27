"""Compare catalogue-only and catalogue-plus-FITS ranking on validation events."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import compare, paired_interval
from .imaging import PIXEL_FEATURES, fetch_images, pixel_features, read_image
from .splits import input_hashes
from .survey import Cache


def inputs(run, manifest):
    if input_hashes(run) != manifest["source_sha256"]:
        raise ValueError("Export files differ from the frozen benchmark inputs.")
    events = pd.DataFrame(manifest["assignments"])
    events = events.loc[events.split.isin(["train", "validation"]), ["event_id", "split"]]
    coords = pd.read_csv(run / "events.csv", usecols=["event_id", "ra", "dec", "coverage_bands"])
    events = events.merge(coords, on="event_id", validate="one_to_one").sort_values("event_id")
    candidates = pd.read_csv(run / "candidates.csv",
                             usecols=["event_id", "source_id", "ra", "dec", "in_baseline_cone"])
    candidates = candidates.loc[candidates.event_id.isin(events.event_id) & candidates.in_baseline_cone]
    return events, candidates


def measure(cache, events, candidates, audit):
    records = audit.set_index("event_id")
    rows = []
    for event in events.itertuples():
        sources = candidates.loc[candidates.event_id.eq(event.event_id)]
        record = records.loc[event.event_id]
        if record.status == "ok":
            path = cache.root / "image_ablation" / record.cache_file
            image, valid, wcs = read_image(path, event, record.pixel_scale_arcsec)
            rows.append(pixel_features(image, valid, wcs, event, sources, record.pixel_scale_arcsec))
        else:
            block = sources[["event_id", "source_id"]].copy()
            block[PIXEL_FEATURES] = np.nan
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def report(manifest, audit, image_features, evaluation, train_count, retrieved_column):
    candidate_counts = image_features[PIXEL_FEATURES].notna()
    recovered = evaluation.loc[evaluation[retrieved_column].eq(True)]
    discovered = recovered.image_reference_rank.eq(1)
    baseline = recovered.catalogue_reference_rank.eq(1)
    with_pixels = set(image_features.loc[candidate_counts.any(axis=1), "event_id"])
    lines = [
        "# FITS image ranking ablation", "",
        f"The same {train_count} training events and {len(evaluation)} validation events were used by "
        "the catalogue-only and catalogue-plus-image rankers. The frozen test split remained reserved.", "",
        "## Image coverage", "",
    ]
    for split, rows in audit.groupby("split", sort=False):
        lines.append(f"- **{split.title()}** had {rows.status.eq('ok').sum()}/{len(rows)} usable FITS fields. "
                     f"{rows.event_id.isin(with_pixels).sum()} had at least one measured candidate feature.")
    skipped = audit.loc[audit.status.ne("ok")]
    for row in skipped.itertuples():
        lines.append(f"- **{row.event_id}** had status `{row.status}`.")
    lines += ["", f"At least one pixel feature was measured for {candidate_counts.any(axis=1).sum():,} "
              f"of {len(image_features):,} train and validation candidates.", ""]
    for feature in PIXEL_FEATURES:
        lines.append(f"- **{feature.removeprefix('image_').replace('_', ' ')}** "
                     f"{candidate_counts[feature].sum():,}/{len(image_features):,} measurements.")
    lines += ["", "## Agreement with the published ZTF reference", "",
              f"{len(recovered)} validation events had a uniquely recovered reference counterpart. "
              "The comparison retains the same events and candidates when pixel features are missing.", ""]
    for name, column in (("Catalogue only", "catalogue_reference_rank"),
                         ("Catalogue plus images", "image_reference_rank")):
        ranks = recovered[column]
        lines.append(f"- **{name}** ranked {ranks.eq(1).sum()}/{len(recovered)} first and "
                     f"{ranks.le(3).sum()}/{len(recovered)} in the top three.")
    if not recovered.empty:
        change = paired_interval(recovered, "image", "catalogue", manifest["split_seed"])
        lines += ["", f"The paired first-rank change was {change}. "
                  f"The image model improved {int((discovered & ~baseline).sum())} and worsened "
                  f"{int((baseline & ~discovered).sum())} of the first-rank outcomes."]
    lines += ["", "The image model adds four measurements of r-band light to the same regularized "
              "logistic ranker. Inverse variance masks unusable pixels. Measurements use the candidate "
              "and SN coordinates, with a 1.4″ mask around the transient to reduce contamination from "
              "the SN itself. The DR10 imaging dates overlap the transient search. The training label "
              "remains ZTF's published DLR-selected reference.",
              "", "[Legacy Surveys DR10 images](https://www.legacysurvey.org/dr10/description/)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be at least one")
    manifest = json.loads(options.split.read_text(encoding="utf-8"))
    events, candidates = inputs(options.run, manifest)
    radius = manifest["search_radii_arcsec"][0]
    cache = Cache(options.cache, options.offline)
    audit = fetch_images(cache, events, radius, options.workers)
    options.output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(options.output / "image_inputs.csv", index=False)
    if audit.status.eq("query_or_fits_failed").any():
        raise SystemExit("Some FITS requests failed. See image_inputs.csv and retry.")
    features = measure(cache, events, candidates, audit)
    if not features[PIXEL_FEATURES].notna().any().any():
        raise SystemExit("No usable image features were measured. See image_inputs.csv.")
    features.to_csv(options.output / "image_features.csv", index=False)
    evaluation, rankings, train_count, retrieved = compare(options.run, manifest, features)
    evaluation.to_csv(options.output / "image_validation.csv")
    rankings.to_csv(options.output / "image_rankings.csv", index=False)
    summary = report(manifest, audit, features, evaluation, train_count, retrieved)
    (options.output / "image_report.md").write_text(summary, encoding="utf-8")
    print(f"Image validation written to {options.output}")


if __name__ == "__main__":
    main()
