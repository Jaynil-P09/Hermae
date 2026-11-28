"""Train the UV localizer on simulated changes in real GALEX fields."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .development import digest
from .training_device import add_device_argument, resolve_device
from .uv import GALEX, SIZE, UVLocalizer, candidates, features, probability_map, read_uv, robust_scale, write_run


MARGIN = 9
YY, XX = np.indices((SIZE, SIZE))
INTERIOR = np.s_[MARGIN:-MARGIN, MARGIN:-MARGIN]


def make_pair(image, rng):
    height, width = image.shape
    for _ in range(120):
        x, y = rng.integers(width - SIZE + 1), rng.integers(height - SIZE + 1)
        patch = image[y:y + SIZE, x:x + SIZE]
        if np.isfinite(patch).all() and np.mean(patch > 0) >= .9:
            break
    else:
        raise ValueError("Could not find a covered UV patch.")
    baseline = gaussian_filter(patch, .7)
    _, level = robust_scale(patch.ravel())
    noise = rng.uniform(.20, .85) * level
    reference = gaussian_filter(baseline, rng.uniform(.05, .6)) + rng.normal(0, noise, patch.shape)
    science = shift(gaussian_filter(baseline, rng.uniform(.05, .6)), rng.uniform(-.55, .55, 2), order=1)
    science = science * rng.uniform(.75, 1.25) + rng.normal(0, noise, patch.shape)
    # Include isolated detector spikes and smooth gradients in both classes.
    if rng.random() < .7:
        iy, ix = rng.integers(3, SIZE - 3, size=2)
        science[iy, ix] += rng.uniform(4, 18) * noise
    science += rng.normal(0, .25) * noise * ((XX + YY) / SIZE - 1)
    label = np.zeros_like(patch)
    truth = None
    if rng.random() < .66:
        tx, ty = rng.uniform(MARGIN + 1, SIZE - MARGIN - 1, 2)
        d2 = (XX - tx)**2 + (YY - ty)**2
        psf = np.exp(-d2 / (2 * rng.uniform(1.0, 1.8)**2))
        science += rng.uniform(1.8, 12) * noise * psf
        label[d2 <= 2. ** 2] = 1.
        truth = (float(tx), float(ty))
    return reference, science, label, truth, (x, y)


def read_fields(manifest):
    fields = {}
    for row in json.loads(manifest.read_text(encoding="utf-8"))["images"]:
        path = Path(row["path"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"UV input changed after collection: {path}")
        try:
            image, wcs, band, _, _ = read_uv(path, 0)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        if row["band"] != band or band not in GALEX or min(image.shape) < SIZE:
            raise ValueError(f"Unexpected passband or image size in {path}")
        fields.setdefault(row["event"], []).append((row, image, wcs, band))
    return fields


def split_fields(fields, rng):
    complete = sorted(event for event in fields if len({row[3] for row in fields[event]}) == 2)
    if len(complete) < 6:
        raise ValueError("Provide at least six independent fields with both NUV and FUV coverage.")
    rng.shuffle(complete)
    test = complete[:2]
    if "2023ixf" in complete and "2023ixf" not in test:
        test[0] = "2023ixf"
    validation = [event for event in complete if event not in test][:2]
    train = sorted(set(fields) - set(test) - set(validation))
    return dict(train=train, validation=validation, test=test)


def simulate(fields, ids, count, rng, *, retain=False):
    inputs, labels, examples = [], [], []
    valid = np.ones((SIZE, SIZE), bool)
    for event in ids:
        for row, image, wcs, band in fields[event]:
            for _ in range(count):
                ref, sci, label, truth, origin = make_pair(image, rng)
                stack, _ = features(ref, sci, valid, band)
                inputs.append(stack)
                labels.append(label[None])
                examples.append(dict(event=event, band=band, truth=truth))
                if retain:
                    examples[-1].update(row=row, wcs=wcs, ref=ref, sci=sci, origin=origin)
    return np.stack(inputs), np.stack(labels), examples


def predict(net, inputs, device):
    with torch.inference_mode():
        return np.concatenate([net(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy()[:, 0]
                               for x in np.array_split(inputs, max(1, (len(inputs) + 63) // 64))])


def threshold_from_quiet(maps, examples):
    quiet_peaks = [np.max(a[INTERIOR]) for a, ex in zip(maps, examples) if ex["truth"] is None]
    return float(np.quantile(quiet_peaks, .95, method="higher"))


def metrics(maps, examples, threshold):
    distances, recovered, quiet = [], [], []
    for scores, ex in zip(maps, examples):
        region = scores[INTERIOR]
        y, x = np.unravel_index(np.argmax(region), region.shape)
        peak = float(region[y, x])
        if ex["truth"] is None:
            quiet.append(peak >= threshold)
        else:
            tx, ty = ex["truth"]
            distance = float(np.hypot(x + MARGIN - tx, y + MARGIN - ty))
            distances.append(distance)
            recovered.append(peak >= threshold and distance <= 3.)
    return dict(injected=len(recovered), quiet=len(quiet), localized_within_3px=int(sum(recovered)),
                quiet_with_flags=int(sum(quiet)), median_top_peak_error_px=float(np.median(distances)))


def showcase(out, net, inputs, examples, threshold, device):
    chosen = set()
    valid = np.ones((SIZE, SIZE), bool)
    for stack, ex in zip(inputs, examples):
        key = ex["event"], ex["band"]
        if key in chosen or ex["truth"] is None:
            continue
        scores = probability_map(net, stack, valid, device)
        peaks = candidates(scores, threshold, stack[2])
        tx, ty = ex["truth"]
        if not peaks or np.hypot(peaks[0][0] - tx, peaks[0][1] - ty) > 3:
            continue
        chosen.add(key)
        ox, oy = ex["origin"]
        grid = ex["wcs"].slice((slice(oy, oy + SIZE), slice(ox, ox + SIZE)))
        positions = [grid.pixel_to_world(x, y) for x, y, _ in peaks]
        truth_sky = grid.pixel_to_world(tx, ty)
        metadata = dict(display_name=f"SN {key[0]} field / {key[1].upper()} injection",
                        evidence="synthetic_injection", threshold=threshold,
                        injection_ra_deg=float(truth_sky.icrs.ra.deg), injection_dec_deg=float(truth_sky.icrs.dec.deg),
                        localization_error_px=float(np.hypot(peaks[0][0] - tx, peaks[0][1] - ty)),
                        coverage=float(np.isfinite(scores).mean()), background=ex["row"],
                        selection="First recovered injection per held-out field and band. See evaluation JSON for all trials.")
        write_run(out / f"{key[0]}-{key[1]}", ex["ref"], ex["sci"], stack, scores, peaks,
                  grid, positions, ex["band"], metadata, (tx, ty))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--samples", type=int, default=128, help="Training patches per GALEX image")
    parser.add_argument("--seed", type=int, default=42)
    add_device_argument(parser)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 4:
        parser.error("Use positive epochs and at least four samples per training image.")
    device, device_name = resolve_device(parser, args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    fields = read_fields(args.manifest)
    groups = split_fields(fields, rng)
    x, y, _ = simulate(fields, groups["train"], args.samples, rng)
    vx, _, validation = simulate(fields, groups["validation"], 96, rng)
    tx, _, test = simulate(fields, groups["test"], 128, rng, retain=True)
    print(f"{len(x)} training, {len(vx)} validation and {len(tx)} test patches", flush=True)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=48, shuffle=True)
    net = UVLocalizer().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015)
    loss_weight = torch.tensor(35., device=device)
    history = []
    best_key = (-1, float("-inf"))
    best = args.out / "uv-localizer.pt"
    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = 0.
        for batch, labels in loader:
            optimizer.zero_grad()
            logits = net(batch.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(device), pos_weight=loss_weight)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        maps = predict(net.eval(), vx, device)
        cutoff = threshold_from_quiet(maps, validation)
        scores = metrics(maps, validation, cutoff)
        key = scores["localized_within_3px"], -scores["median_top_peak_error_px"]
        if key > best_key:
            best_key = key
            torch.save(dict(model=net.state_dict(), threshold=cutoff, epoch=epoch), best)
        history.append(dict(epoch=epoch, mean_training_loss=loss_sum / len(x), threshold=cutoff, **scores))
        print(json.dumps(history[-1]), flush=True)
    checkpoint = torch.load(best, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model"])
    test_maps = predict(net, tx, device)
    baseline_validation = gaussian_filter(vx[:, 2], (0, 1.3, 1.3))
    baseline_test = gaussian_filter(tx[:, 2], (0, 1.3, 1.3))
    baseline_threshold = threshold_from_quiet(baseline_validation, validation)
    threshold = checkpoint["threshold"]
    report = dict(evidence="injections_on_real_galex_backgrounds", device=device_name, seed=args.seed,
                  field_split=groups, photos_by_band={band: sum(row[3] == band for rows in fields.values()
                                                                for row in rows) for band in GALEX},
                  training_patches=len(x), validation_patches=len(vx), test_patches=len(tx),
                  checkpoint_epoch=checkpoint["epoch"], threshold=threshold, history=history,
                  test={band: metrics(test_maps[[ex["band"] == band for ex in test]],
                                      [ex for ex in test if ex["band"] == band], threshold)
                        for band in GALEX},
                  test_all=metrics(test_maps, test, threshold),
                  matched_filter_baseline=metrics(baseline_test, test, baseline_threshold))
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    showcase(args.out / "examples", net, tx, test, threshold, device)
    print(json.dumps(report["test_all"]), flush=True)


if __name__ == "__main__":
    main()
