"""Export and score the v1 measured-FITS baseline using its own environment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def fit(root, path):
    from hermae.benchmark import FEATURES, RAW_FEATURES, catalogue_features, ranker
    from hermae.imaging import PIXEL_FEATURES

    run = root / "results/benchmark"
    manifest = json.loads((root / "benchmark/split-v1.json").read_text(encoding="utf-8"))
    train_ids = {row["event_id"] for row in manifest["assignments"] if row["split"] == "train"}
    candidates = pd.read_csv(run / "candidates.csv", usecols=[*RAW_FEATURES, "in_baseline_cone"])
    candidates = candidates[candidates.event_id.isin(train_ids)]
    labels = pd.read_csv(run / "evaluation.csv", usecols=["event_id", "reference_source_id",
                                                         "reference_retrieved_60as"]).set_index("event_id")
    features = pd.concat([catalogue_features(frame) for _, frame in candidates.groupby("event_id", sort=False)])
    image = pd.read_csv(run / "images/image_features.csv").set_index(["event_id", "source_id"])
    cone = candidates[candidates.in_baseline_cone].copy()
    keys = pd.MultiIndex.from_frame(cone[["event_id", "source_id"]])
    features = features.join(image.reindex(keys).set_axis(cone.index)[PIXEL_FEATURES])
    train = cone[cone.event_id.isin(labels.index[labels.reference_retrieved_60as.eq(True)])]
    counts = train.groupby("event_id").source_id.transform("size")
    train = train[counts > 1]
    target = train.source_id.eq(train.event_id.map(labels.reference_source_id))
    if target.sum() != train.event_id.nunique():
        raise ValueError("V1 baseline training labels are not unique.")
    columns = [*FEATURES["catalogue"], *PIXEL_FEATURES]
    impute, scale, model = ranker(columns, features.loc[train.index], target,
                                 np.where(target, 1., 1. / (counts.loc[train.index] - 1))).steps
    impute, scale, model = impute[1], scale[1], model[1]
    state = dict(columns=columns, median=impute.statistics_.tolist(),
                 missing_indicators=impute.indicator_.features_.tolist(),
                 mean=scale.mean_.tolist(), scale=scale.scale_.tolist(), coefficient=model.coef_[0].tolist(),
                 intercept=float(model.intercept_[0]), train_ids=sorted(set(train.event_id)))
    path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def predict(root, path, image_manifest, output):
    from hermae.benchmark import RAW_FEATURES, catalogue_features
    from hermae.imaging import PIXEL_FEATURES, read_image, pixel_features

    state = json.loads(path.read_text(encoding="utf-8"))
    paths = {r["event"]: r for r in json.loads(image_manifest.read_text(encoding="utf-8"))}
    run = root / "results/benchmark"
    events = pd.read_csv(run / "events.csv")
    events = events[events.event_id.isin(paths)]
    rows = pd.read_csv(run / "candidates.csv", usecols=[*RAW_FEATURES, "in_baseline_cone"])
    results = {}
    for event in events.itertuples():
        candidates = rows[rows.event_id.eq(event.event_id)]
        cone = candidates[candidates.in_baseline_cone]
        if cone.empty:
            results[event.event_id] = None
            continue
        features = catalogue_features(candidates).loc[cone.index]
        image_path = paths[event.event_id].get("path")
        if image_path:
            image, valid, wcs = read_image(Path(image_path), event, .262)
            pixels = pixel_features(image, valid, wcs, event, cone, .262)
            features = features.join(pixels.set_axis(cone.index)[PIXEL_FEATURES])
        else:
            features[PIXEL_FEATURES] = np.nan
        values = features[state["columns"]].to_numpy()
        missing = np.isnan(values)
        values = np.column_stack((np.where(missing, state["median"], values),
                                  missing[:, state["missing_indicators"]]))
        values = (values - state["mean"]) / state["scale"]
        scores = values @ state["coefficient"] + state["intercept"]
        i = int(np.lexsort((cone.source_id, cone.separation_arcsec, -scores))[0])
        pick = cone.iloc[i]
        results[event.event_id] = dict(source_id=pick.source_id, ra_deg=float(pick.ra),
                                       dec_deg=float(pick.dec), score=float(scores[i]))
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    if args.image_manifest:
        predict(args.root, args.checkpoint, args.image_manifest, args.out)
    else:
        fit(args.root, args.checkpoint)
