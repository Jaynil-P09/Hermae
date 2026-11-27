"""Build weak catalogue labels, inject galaxies, and train/evaluate the detector."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch.nn import functional as F

from .detection import DetectorConfig, detect, detection_inputs
from .development import Benchmark
from .galaxy import GalaxyDetector, GalaxyNet, instances, pixels
from .training_device import add_device_argument, resolve_device

CROP = 96
SETTINGS = ((.35, .15), (.35, .30), (.45, .20), (.50, .40), (.60, .45),
            (.60, .55), (.60, .60), (.60, .65), (.60, .70), (.70, .65), (.70, .75))


def region(shape, x, y, radius):
    return np.s_[max(0, int(y - radius)):min(shape[0], int(y + radius + 2)),
                 max(0, int(x - radius)):min(shape[1], int(x + radius + 2))]


def ellipse(shape, x, y, a, b, angle, extent=1.):
    bounds = region(shape, x, y, extent * max(a, b))
    yy, xx = np.ogrid[bounds[0].start:bounds[0].stop, bounds[1].start:bounds[1].stop]
    along = (xx - x) * np.cos(angle) + (yy - y) * np.sin(angle)
    across = (yy - y) * np.cos(angle) - (xx - x) * np.sin(angle)
    return bounds, np.hypot(along / a, across / b)


def weak_labels(data, event, frame):
    shape = frame.valid.shape
    targets = np.zeros((2, *shape), dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    scale = float(np.median(proj_plane_pixel_scales(frame.wcs)) * 3600)
    galaxies = []
    for row, x, y, galaxy, snr in data.red_sources(event, frame):
        if not frame.valid[int(y), int(x)] or snr < 7:
            continue
        if galaxy:
            a = np.clip(float(row["shape_r"]) / scale * 1.8, 2., 32.)
            ellipticity = min(.85, np.hypot(float(row["shape_e1"]), float(row["shape_e2"])))
            b = max(1.6, a * (1. - ellipticity) / (1. + ellipticity))
            sky = frame.wcs.pixel_to_world(x, y)
            pa = float(row["major_pa_deg"]) if row["major_pa_deg"] else 0.
            tx, ty = frame.wcs.world_to_pixel(sky.directional_offset_by(pa * u.deg, scale * u.arcsec))
            bounds, radius = ellipse(shape, x, y, a, b, np.arctan2(ty - y, tx - x), 1.8)
            targets[0][bounds] = np.maximum(targets[0][bounds], radius <= 1)
            weight[bounds] = np.maximum(weight[bounds], .4 * (radius < 1.8))
            galaxies.append((x, y, snr, a))
            iy, ix = int(np.floor(y + .5)), int(np.floor(x + .5))
            if iy < shape[0] and ix < shape[1]:
                targets[1, iy, ix] = 1.
        elif row["type"].strip() == "PSF":
            bounds, radius = ellipse(shape, x, y, 4., 4., 0.)
            weight[bounds] = np.maximum(weight[bounds], .8 * (radius <= 1))
    # Unknown regions are ignored. Keep high-confidence quiet pixels away from known objects.
    signal, variance = detection_inputs(frame, DetectorConfig())
    quiet = (np.abs(signal) < 1.5 * np.sqrt(variance)) & frame.valid
    quiet &= ~ndimage.binary_dilation(weight > 0, iterations=3)
    weight[quiet] = .06
    weight[~frame.valid] = 0
    return signal, variance, targets, weight, galaxies


def inject(signal, variance, valid, targets, weight, known, rng):
    truth = []
    for _ in range(60):
        if len(truth) >= 10:
            break
        x, y = rng.integers(16, signal.shape[1] - 16), rng.integers(16, signal.shape[0] - 16)
        if not valid[y-12:y+13, x-12:x+13].all() or weight[y, x] > .1:
            continue
        if any(np.hypot(x - px, y - py) < max(12, a + 5) for px, py, _, a in known):
            continue
        blend = len(truth) % 3 == 0 and x + 7 < signal.shape[1] - 12 and valid[y, x + 7]
        for cx in ((x, x + 7) if blend else (x,)):
            if any(np.hypot(cx - px, y - py) < 6 for px, py, *_ in truth):
                continue
            a, b = rng.uniform(3.8, 6.5), rng.uniform(2.8, 3.7)
            amplitude = float(rng.uniform(5, 25))
            bounds, radius = ellipse(signal.shape, cx, y, a, b, rng.uniform(0, np.pi), 2.)
            profile = amplitude * np.sqrt(variance[y, cx]) * np.exp(-2.0 * radius) * (radius < 2)
            signal[bounds] += profile
            targets[0][bounds] = np.maximum(targets[0][bounds], radius <= 1)
            targets[1, y, cx] = 1
            weight[bounds] = np.maximum(weight[bounds], (radius < 1.6) * valid[bounds])
            truth.append((cx, y, amplitude, a, blend))
    return truth


def crops(benchmark, ids, rng):
    features, labels, weights, used = [], [], [], []
    for event in ids:
        frame = benchmark.frame(event)
        if frame is None:
            continue
        signal, variance, target, weight, galaxies = weak_labels(benchmark, event, frame)
        height, width = frame.valid.shape
        if min(height, width) < CROP + 2:
            continue
        injected = inject(signal, variance, frame.valid, target, weight, galaxies, rng)
        centres = [(x, y) for x, y, *_ in galaxies + injected]
        if not centres:
            continue
        used.append(event)
        image = pixels(frame.valid, signal, variance)
        for i in rng.choice(len(centres), min(16, len(centres)), replace=False):
            x, y = centres[i]
            x = int(np.clip(x - rng.integers(CROP // 4, 3 * CROP // 4), 0, width - CROP))
            y = int(np.clip(y - rng.integers(CROP // 4, 3 * CROP // 4), 0, height - CROP))
            patch = np.s_[y:y+CROP, x:x+CROP]
            features.append(image[:, patch[0], patch[1]])
            labels.append(target[:, patch[0], patch[1]])
            weights.append(weight[patch])
    return tuple(torch.from_numpy(np.array(values, dtype=np.float32)) for values in (features, labels, weights)), used


def loss(logits, target, weight):
    mask, centre = logits.unbind(1)
    footprint, peak = target.unbind(1)
    prediction = mask.sigmoid()
    dice = 1 - (2 * (prediction * footprint * weight).sum() + 1) / (
        ((prediction + footprint) * weight).sum() + 1)
    bce = F.binary_cross_entropy_with_logits(mask, footprint, reduction="none")
    p = centre.sigmoid()
    pt = torch.where(peak > 0, p, 1 - p)
    alpha = torch.where(peak > 0, 20., 1.)
    focal = F.binary_cross_entropy_with_logits(centre, peak, reduction="none") * (1 - pt).square() * alpha
    return dice + ((bce + focal) * weight).sum() / weight.sum().clamp_min(1)


def fit(tensors, epochs, batch_size, seed, device):
    torch.manual_seed(seed)
    model = GalaxyNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=.002)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = []
        for indices in torch.randperm(len(tensors[0]), generator=generator).split(batch_size):
            image, label, weight = (t[indices].to(device) for t in tensors)
            if indices[0] % 2 == 0:
                image, label, weight = (t.flip((-1,)) for t in (image, label, weight))
            if indices[-1] % 2 == 0:
                image, label, weight = (t.transpose(-1, -2) for t in (image, label, weight))
            score = loss(model(image), label, weight)
            optimiser.zero_grad()
            score.backward()
            optimiser.step()
            totals.append(float(score.detach()))
        history.append(float(np.mean(totals)))
        print(f"Epoch {epoch}/{epochs}  loss {history[-1]:.4f}", flush=True)
    return model.eval(), history


def objects(labels):
    if labels.max() == 0:
        return []
    centres = ndimage.center_of_mass(labels > 0, labels, range(1, int(labels.max()) + 1))
    return [(x, y) for y, x in centres]


def match(truth, predictions, tolerance):
    if not (truth and predictions):
        return np.array([], dtype=int), np.array([], dtype=int)
    distance = cdist(np.array(truth)[:, :2], np.array(predictions)[:, :2])
    cost = np.where(distance <= tolerance, distance, tolerance * (max(distance.shape) + 1))
    i, j = linear_sum_assignment(cost)
    accepted = distance[i, j] <= tolerance
    return i[accepted], j[accepted]


def assess(benchmark, model, ids, seed, device):
    detector = GalaxyDetector(model, device, 0, 0)
    methods = ("classical", *SETTINGS)
    metrics = {method: {"real": [], "synthetic": []} for method in methods}
    rng = np.random.default_rng(seed)
    for event in ids:
        frame = benchmark.frame(event)
        if frame is None:
            continue
        signal, variance, target, weight, galaxies = weak_labels(benchmark, event, frame)
        scale = np.median(proj_plane_pixel_scales(frame.wcs)) * 3600
        tolerance = benchmark.tolerance / scale
        detect_config = DetectorConfig()
        baseline, _, _ = detect(frame, detect_config)
        real_prob = detector.predict(pixels(frame.valid, signal, variance))
        synthetic_signal = signal.copy()
        truth = inject(synthetic_signal, variance, frame.valid, target.copy(), weight.copy(), galaxies, rng)
        synthetic_frame = replace(frame, science=frame.science + synthetic_signal - signal)
        synth_base, _, _ = detect(synthetic_frame, detect_config)
        synth_prob = detector.predict(pixels(frame.valid, synthetic_signal, variance))
        for method in methods:
            if method == "classical":
                real, synthetic = objects(baseline), objects(synth_base)
                mask = baseline > 0
            else:
                real = [centre for _, _, centre in instances(*real_prob, frame.valid, 7, *method)]
                synthetic = [centre for _, _, centre in instances(*synth_prob, frame.valid, 7, *method)]
                mask = real_prob[0] >= method[0]
            found, _ = match(galaxies, real, tolerance)
            near = cdist(np.array(synthetic).reshape(-1, 2), np.array(truth).reshape(-1, 5)[:, :2])
            close = [point for point, use in zip(synthetic, (near < 14).any(axis=1)) if use]
            recovered, _ = match(truth, close, min(4., tolerance))
            faint = {i for i, row in enumerate(truth) if row[2] < 10}
            blended = {i for i, row in enumerate(truth) if row[4]}
            edge = {i for i, (x, y, *_) in enumerate(galaxies)
                    if min(x, y, frame.valid.shape[1] - x, frame.valid.shape[0] - y) <= 16}
            # Report mask overlap solely against weak ellipses on weighted real pixels.
            valid = weight >= .4
            label = target[0] > 0
            metrics[method]["real"].append((len(found), len(galaxies), len(real) - len(found),
                                            int(np.sum(mask & label & valid)), int(np.sum((mask | label) & valid)),
                                            len(set(found) & edge), len(edge)))
            metrics[method]["synthetic"].append((len(recovered), len(truth), len(close) - len(recovered),
                                                 len(set(recovered) & faint), len(faint),
                                                 len(set(recovered) & blended), len(blended)))
        print(f"Assessed {event}", flush=True)

    def totals(result):
        real = np.sum(result["real"], axis=0).tolist()
        synthetic = np.sum(result["synthetic"], axis=0).tolist()
        return {"fields": len(result["real"]),
                "real": dict(zip(("matched", "reference", "unmatched", "weak_intersection", "weak_union",
                                  "edge_matched", "edge_reference"), real)),
                "synthetic": dict(zip(("matched", "injected", "unmatched_near_injections", "faint_matched",
                                        "faint_injected", "blend_matched", "blend_injected"), synthetic))}

    return {method: totals(result) for method, result in metrics.items()}


def train(args):
    data = Benchmark(args.v1)
    out = data.output(args.out)
    torch.set_num_threads(4)
    ids = data.ids("train")
    validation_ids = data.ids("validation")
    if args.limit:
        ids, validation_ids = ids[:args.limit], validation_ids[:args.limit]
    tensors, used = crops(data, ids, np.random.default_rng(args.seed))
    if not used:
        raise ValueError("No supervised development crops were available.")
    print(f"Training on {len(tensors[0])} crops from {len(used)} fields", flush=True)
    model, history = fit(tensors, args.epochs, args.batch_size, args.seed, args.device)
    assessment = assess(data, model, validation_ids, args.seed + 1, args.device)
    baseline = assessment["classical"]["real"]["unmatched"]
    eligible = [setting for setting in SETTINGS if assessment[setting]["real"]["unmatched"] <= baseline]
    if eligible:
        chosen = max(eligible, key=lambda setting: assessment[setting]["real"]["matched"])
    else:
        chosen = min(SETTINGS, key=lambda setting: assessment[setting]["real"]["unmatched"])
    report = {"split_sha256": data.split_hash, "seed": args.seed, "epochs": args.epochs,
              "batch_size": args.batch_size, "device": args.device, "device_name": args.device_name,
              "training_crops": len(tensors[0]), "train_ids": used,
              "validation_ids": validation_ids, "loss": history, "thresholds": chosen,
              "thresholds_meeting_baseline_unmatched_count": eligible,
              "test_kind": "weak catalogue real-field labels plus separate synthetic injections",
              "results": {str(key): value for key, value in assessment.items()}}
    torch.save(dict(kind="hermae-galaxy-v1", model=model.cpu().state_dict(), footprint_threshold=chosen[0],
                    centre_threshold=chosen[1], split_sha256=data.split_hash), out / "galaxy.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selected footprint/centre thresholds {chosen}; outputs in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    add_device_argument(parser)
    parser.add_argument("--limit", type=int, help="optional smoke-test field limit per split")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error("Epochs, batch size and field limit must be positive.")
    args.device, args.device_name = resolve_device(parser, args.device)
    train(args)


if __name__ == "__main__":
    main()
