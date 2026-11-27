"""Train fieldwise host ranking CNNs and measure candidate retrieval."""

import argparse
import json
from pathlib import Path

from astropy.coordinates import SkyCoord
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .detection import DetectorConfig, detect, measurements
from .development import Benchmark, digest
from .fits_input import sn_position
from .galaxy import GalaxyDetector, pixels
from .host import rank_hosts
from .ranker import HostRanker, candidate_inputs, score
from .training_device import add_device_argument, resolve_device

POOLS = ("catalogue", "image", "union")


def positions(rows):
    return SkyCoord([r["ra_deg"] for r in rows], [r["dec_deg"] for r in rows], unit="deg")


def candidate_pool(benchmark, event, frame, sn, image_rows):
    catalogue = []
    for raw, x, y, galaxy, _ in benchmark.red_sources(event, frame):
        if not galaxy or not frame.valid[int(y), int(x)]:
            continue
        if float(raw["separation_arcsec"]) > 60:
            continue
        catalogue.append((dict(ra_deg=float(raw["ra"]), dec_deg=float(raw["dec"]), x_px=x, y_px=y,
                               separation_arcsec=float(raw["separation_arcsec"]), size_scaled_offset=None,
                               shape_fallback=True), raw))
    xmatch = {}
    if image_rows and catalogue:
        distance = positions(image_rows)[:, None].separation(positions([row for row, _ in catalogue])[None]).arcsec
        ii, jj = linear_sum_assignment(np.where(distance <= 1., distance, 1e6))
        xmatch = {int(j): int(i) for i, j in zip(ii, jj) if distance[i, j] <= 1.}
    combined = list(image_rows)
    catalog_indices = []
    features = {}
    for j, (row, raw) in enumerate(catalogue):
        if j in xmatch:
            i = xmatch[j]
        else:
            i = len(combined)
            combined.append(dict(row, source_id=-(j + 1)))
        catalog_indices.append(i)
        features[combined[i]["source_id"]] = raw
    return combined, features, dict(catalogue=np.array(sorted(catalog_indices), dtype=int),
                                    image=np.arange(len(image_rows)), union=np.arange(len(combined)))


def field_examples(benchmark, event, backend):
    frame = benchmark.frame(event)
    event_row = benchmark.events.get(event)
    if frame is None or event_row is None:
        return None, "unusable_fits"
    sn, x, y, coverage = sn_position(frame, float(event_row["ra"]), float(event_row["dec"]))
    if coverage != "covered":
        return None, coverage
    labels, signal, variance = detect(frame, DetectorConfig(), backend)
    detections = measurements(frame, labels, signal, variance)
    rows = rank_hosts(frame, detections, sn, x, y, 60.)
    union, catalogue, pools = candidate_pool(benchmark, event, frame, sn, rows)
    reference = benchmark.references[event]
    if not (reference["host_ra"] and reference["host_dec"]):
        return None, "missing_published_position"
    truth = SkyCoord(float(reference["host_ra"]), float(reference["host_dec"]), unit="deg")
    offsets = positions(union).separation(truth).arcsec if union else np.empty(0)
    reference_indices = {name: idx[offsets[idx] <= benchmark.tolerance] for name, idx in pools.items()}
    tensors = candidate_inputs(pixels(frame.valid, signal, variance), labels, union,
                               {r["source_id"]: r for r in detections}, x, y, catalogue)
    return (event, tensors, pools, reference_indices), None


def prepare(benchmark, ids, backend):
    fields, excluded = [], {}
    for event in ids:
        entry, error = field_examples(benchmark, event, backend)
        if error:
            excluded[event] = error
        else:
            fields.append(entry)
    return fields, excluded


def fit(fields, use_catalogue, seed, epochs, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HostRanker(use_catalogue).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=.001)
    eligible = [field for field in fields if len(field[3]["union"]) == 1 and len(field[2]["union"]) > 1]
    if not eligible:
        raise ValueError("No training fields had a unique published counterpart in their candidate pool.")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = []
        for batch in np.array_split(rng.permutation(len(eligible)), (len(eligible) + 7) // 8):
            losses = []
            for idx in batch:
                _, tensors, _, reference = eligible[idx]
                logits = model(*(value.to(device) for value in tensors))
                losses.append(torch.logsumexp(logits, 0) - logits[int(reference["union"][0])])
            loss = torch.stack(losses).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
        print(f"{'Catalogue fusion' if use_catalogue else 'Image-first'} epoch {epoch}: {history[-1]:.4f}",
              flush=True)
    return model.eval(), history, [field[0] for field in eligible]


def measure(fields, model, device):
    result = {name: [] for name in POOLS}
    for event, tensors, pools, references in fields:
        scores = score(model, tensors, device) if len(pools["union"]) else np.empty(0)
        for name, idx in pools.items():
            order = idx[np.argsort(-scores[idx], kind="stable")]
            targets = references[name]
            gap = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None
            result[name].append(dict(event=event, candidates=len(idx), recovered=len(targets),
                                     correct=bool(len(targets) == 1 and order[0] == targets[0]), score_gap=gap))
    return result


def summary(rows):
    found = sum(row["recovered"] == 1 for row in rows)
    correct = sum(row["correct"] for row in rows)
    return dict(fields=len(rows), unique_recovered=found, missing=sum(row["recovered"] == 0 for row in rows),
                ambiguous=sum(row["recovered"] > 1 for row in rows), top1=correct,
                recall=found / len(rows) if rows else None,
                conditional_top1=correct / found if found else None,
                end_to_end_top1=correct / len(rows) if rows else None)


def review_threshold(rows):
    # Select the most development coverage at at least 90% empirical correctness.
    options = sorted({0., *(row["score_gap"] for row in rows if row["score_gap"] is not None)})
    for cut in options:
        accepted = [row for row in rows if row["candidates"] and (
            row["score_gap"] is None or row["score_gap"] >= cut)]
        if accepted and sum(row["correct"] for row in accepted) / len(accepted) >= .90:
            return cut, len(accepted)
    return None, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--galaxy-weights", type=Path, help="frozen CNN detector; omit for the classical detector")
    add_device_argument(parser)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    args.device, args.device_name = resolve_device(parser, args.device)
    torch.set_num_threads(4)
    benchmark = Benchmark(args.v1)
    out = benchmark.output(args.out)
    backend = GalaxyDetector.load(args.galaxy_weights, args.device) if args.galaxy_weights else None
    train, train_exclusions = prepare(benchmark, benchmark.ids("train"), backend)
    valid, valid_exclusions = prepare(benchmark, benchmark.ids("validation"), backend)
    print(f"Prepared {len(train)} training and {len(valid)} validation fields", flush=True)
    report = dict(split_sha256=benchmark.split_hash, detector_sha256=digest(args.galaxy_weights)
                  if args.galaxy_weights else None, seed=args.seed, epochs=args.epochs,
                  device=args.device, device_name=args.device_name,
                  train_exclusions=train_exclusions, validation_exclusions=valid_exclusions, models={})
    for mode in ("image_only", "catalogue_fusion"):
        use_catalogue = mode == "catalogue_fusion"
        model, history, ids = fit(train, use_catalogue, args.seed, args.epochs, args.device)
        field_results = measure(valid, model, args.device)
        threshold, accepted = review_threshold(field_results["image"])
        torch.save(dict(kind="hermae-host-v1", model=model.cpu().state_dict(), use_catalogue=use_catalogue,
                        review_gap=threshold, detector_sha256=report["detector_sha256"],
                        split_sha256=benchmark.split_hash), out / f"{mode}.pt")
        report["models"][mode] = dict(history=history, train_ids=ids, review_gap=threshold,
                                      accepted_validation=accepted, fields=field_results,
                                      summaries={p: summary(rows) for p, rows in field_results.items()})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
