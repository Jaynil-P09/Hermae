"""Freeze the cohort and reserve sky-grouped validation and test events."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import numpy as np
import pandas as pd

REGION_DEG = 10
CUTOUT_HALF_DIAGONAL_ARCSEC = 60
TARGETS = {"train": 0.6, "validation": 0.2, "test": 0.2}
INPUT_FILES = ("events.csv", "evaluation.csv", "candidates.csv", "coverage_audit.csv", "manifest.json")


def text_hash(path):
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def input_hashes(run):
    return {name: text_hash(run / name) for name in INPUT_FILES}


def sky_regions(events):
    band = np.minimum(((events.dec + 90) // REGION_DEG).astype(int), 179 // REGION_DEG)
    center_dec = (band + 0.5) * REGION_DEG - 90
    sectors = np.maximum(1, np.rint(360 / REGION_DEG * np.cos(np.deg2rad(center_dec))).astype(int))
    sector = (events.ra * sectors / 360).astype(int)
    return band.astype(str) + "-" + sector.astype(str)


def coordinate_pairs(events, ra, dec, arcsec):
    rows = events.loc[events[[ra, dec]].notna().all(axis=1)]
    if len(rows) < 2:
        return
    coords = SkyCoord(rows[ra].to_numpy(), rows[dec].to_numpy(), unit="deg")
    indices_a, indices_b, _, _ = search_around_sky(coords, coords, arcsec * u.arcsec)
    names = rows.event_id.to_numpy()
    for left, right in zip(indices_a, indices_b):
        if left < right:
            yield names[left], names[right]


def group_events(events, evaluation, candidates, overlap_arcsec, match_arcsec, development=None):
    """Take the transitive union of region, source-ID and angular-overlap links."""
    # Include prior fields even when they fell outside the expanded cohort.
    cohort_ids = set(events.event_id)
    if development is not None:
        events = pd.concat([events, development.loc[~development.event_id.isin(cohort_ids)]],
                           ignore_index=True)
    parents = {name: name for name in events.event_id}

    def root(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def join(left, right):
        first, second = sorted((root(left), root(right)))
        parents[second] = first

    regions = sky_regions(events)
    source_links = pd.concat([
        candidates[["event_id", "source_id"]],
        evaluation[["event_id", "reference_source_id"]].rename(columns={"reference_source_id": "source_id"}),
    ]).dropna(subset="source_id")
    first_in_region = {}
    for event_id, region in zip(events.event_id, regions):
        join(event_id, first_in_region.setdefault(region, event_id))
    first_for_source = {}
    for event_id, source_id in source_links.itertuples(index=False, name=None):
        join(event_id, first_for_source.setdefault(source_id, event_id))
    for ra, dec, radius in (("ra", "dec", overlap_arcsec), ("host_ra", "host_dec", match_arcsec)):
        for left, right in coordinate_pairs(events, ra, dec, radius):
            join(left, right)
    return pd.DataFrame({"event_id": events.event_id, "sky_region": regions,
                         "group_id": events.event_id.map(root)})


def assign_splits(groups, cohort_ids, development, seed):
    anchored = set(groups.loc[groups.event_id.isin(development), "group_id"])
    groups = groups.loc[groups.event_id.isin(cohort_ids)].copy()
    sizes = groups.group_id.value_counts().to_dict()
    assigned = {group: "train" for group in sizes if group in anchored}
    counts = dict.fromkeys(TARGETS, 0)
    counts["train"] = sum(sizes[group] for group in assigned)
    remaining = sorted(
        set(sizes) - anchored,
        key=lambda group: (-sizes[group], sha256(f"{seed}:{group}".encode()).hexdigest()),
    )
    for group in remaining:
        split = max(TARGETS, key=lambda name: TARGETS[name] * len(groups) - counts[name])
        assigned[group] = split
        counts[split] += sizes[group]
    if not all(counts.values()):
        raise ValueError(f"Unable to populate all splits with these group constraints: {counts}")
    groups["split"] = groups.group_id.map(assigned)
    return groups.sort_values("event_id").to_dict("records")


def freeze(run, development_file, seed):
    meta = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(run / "coverage_audit.csv")
    events = pd.read_csv(run / "events.csv")
    evaluation = pd.read_csv(run / "evaluation.csv", usecols=["event_id", "reference_source_id", "catalogue_status"])
    candidates = pd.read_csv(run / "candidates.csv", usecols=["event_id", "source_id"])
    if len(events) != meta["events_requested"] or set(events.event_id) != set(audit.loc[audit.selected, "event_id"]):
        raise ValueError("Cohort export does not match the requested or selected events.")
    last_selected = audit.index[audit.selected].max()
    if audit.loc[:last_selected, "coverage_status"].eq("coverage_query_failed").any():
        raise ValueError("Retry failed coverage probes before freezing this cohort.")
    if evaluation.catalogue_status.eq("query_failed").any():
        raise ValueError("Retry failed catalogue queries before freezing this cohort.")
    if (events.event_id.duplicated().any() or evaluation.event_id.duplicated().any()
            or set(events.event_id) != set(evaluation.event_id)
            or not set(candidates.event_id).issubset(events.event_id)):
        raise ValueError("Inconsistent event IDs in the survey exports.")
    development = pd.read_csv(development_file)
    if development.event_id.duplicated().any():
        raise ValueError("Duplicate event in the development catalogue.")
    # Candidate-centered crops are safe up to this half-diagonal around any queried source.
    overlap = 2 * (max(meta["search_radii_arcsec"]) + CUTOUT_HALF_DIAGONAL_ARCSEC)
    groups = group_events(events, evaluation, candidates, overlap, meta["reference_tolerance_arcsec"],
                          development)
    assignments = assign_splits(groups, events.event_id, development.event_id, seed)
    return dict(version=1, split_seed=seed, cohort_seed=meta["seed"],
                search_radii_arcsec=meta["search_radii_arcsec"], region_scale_degrees=REGION_DEG,
                reference_tolerance_arcsec=meta["reference_tolerance_arcsec"],
                max_cutout_half_diagonal_arcsec=CUTOUT_HALF_DIAGONAL_ARCSEC,
                overlap_arcsec=overlap, target_fractions=TARGETS,
                development_event_ids=sorted(development.event_id),
                development_sha256=text_hash(development_file), source_sha256=input_hashes(run),
                assignments=assignments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="survey pipeline output directory")
    parser.add_argument("--development-events", type=Path, required=True, help="CSV of previously inspected sky fields")
    parser.add_argument("--output", type=Path, required=True, help="new immutable split manifest JSON")
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    try:
        manifest = freeze(options.run, options.development_events, options.seed)
        content = json.dumps(manifest, indent=2) + "\n"
        options.output.parent.mkdir(parents=True, exist_ok=True)
        if options.output.exists():
            if options.output.read_text(encoding="utf-8") != content:
                raise ValueError(f"Frozen manifest differs from new inputs: {options.output}")
        else:
            with options.output.open("x", encoding="utf-8") as file:
                file.write(content)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    distribution = pd.DataFrame(manifest["assignments"]).split.value_counts()
    print(f"{options.output}: " + ", ".join(f"{count} {split}" for split, count in distribution.items()))


if __name__ == "__main__":
    main()
