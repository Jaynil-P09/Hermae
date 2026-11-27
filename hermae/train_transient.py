"""Train on archived alert positions and subtraction artifacts in development pairs."""

import argparse
import json
from pathlib import Path

from astropy.coordinates import SkyCoord
import numpy as np
import torch
from torch.nn import functional as F

from .development import Benchmark, digest
from .pair import difference, frames, residual_candidates
from .transient import TransientNet, classify, stamps
from .training_device import add_device_argument, resolve_device

BASELINE = 4.
MATCH = 2.5


def load_pair(pair, sigma):
    ref, sci = pair["reference"], pair["science"]
    ref_path, sci_path = Path(ref["path"]), Path(sci["path"])
    if any(digest(path) != row["sha256"] for path, row in ((ref_path, ref), (sci_path, sci))):
        raise ValueError(f"Image pair did not match its recorded hashes: {pair['event']}")
    reference, science = frames(ref_path, sci_path)
    delta = difference(reference, science, ref["seeing_arcsec"], sci["seeing_arcsec"])
    candidates = residual_candidates(reference, delta, sigma)
    target = None
    covered = delta is not None
    if pair["kind"] == "positive" and covered:
        truth = SkyCoord(pair["alert_ra"], pair["alert_dec"], unit="deg")
        x, y = reference.wcs.world_to_pixel(truth)
        height, width = delta[4].shape
        ix, iy = int(round(float(x))), int(round(float(y)))
        covered = 0 <= ix < width and 0 <= iy < height and bool(delta[4][iy, ix])
        if candidates and covered:
            distances = SkyCoord([r["ra_deg"] for r in candidates], [r["dec_deg"] for r in candidates],
                                 unit="deg").separation(truth).arcsec
            if distances.min() <= MATCH:
                target = int(distances.argmin())
    return pair, delta, candidates, target, covered


def artifact(stamp):
    # A strong nearby negative lobe is evidence of a subtraction dipole.
    diff = stamp[2]
    peak = diff[10:15, 10:15].max()
    return diff[6:19, 6:19].min() < -max(1.7, peak * .55)


def training_examples(training):
    images, target, audit = [], [], []
    for pair, delta, candidates, positive, _ in training:
        if not candidates:
            continue
        for i, image in enumerate(stamps(delta, candidates)):
            label = 1 if i == positive else (0 if artifact(image) else None)
            if label is not None:
                target.append(label)
                images.append(image)
                audit.append(dict(event=pair["event"], pair_kind=pair["kind"], label=label,
                                  candidate=candidates[i]["transient_id"]))
    if set(target) != {0, 1}:
        raise ValueError("Training requires recovered alerts and dipole artifact examples.")
    return torch.from_numpy(np.stack(images)), torch.tensor(target, dtype=torch.float32), audit


def fit(images, target, epochs, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TransientNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=.0015)
    groups = [(target == i).nonzero().flatten().numpy() for i in (0, 1)]
    history = []
    for epoch in range(1, epochs + 1):
        one_each = np.stack([rng.choice(group, size=max(map(len, groups)), replace=True) for group in groups], axis=1)
        totals = []
        for batch in torch.from_numpy(one_each.reshape(-1)).split(32):
            batch_images = images[batch].to(device)
            batch_images = torch.rot90(batch_images, int(rng.integers(4)), (-2, -1))
            if rng.integers(2):
                batch_images = batch_images.flip((-1,))
            loss = F.binary_cross_entropy_with_logits(model(batch_images), target[batch].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
        print(f"Transient CNN epoch {epoch}: {history[-1]:.4f}", flush=True)
    return model.eval(), history


def evaluate(dataset, guesses, threshold):
    rows = []
    for (pair, _, candidates, target, covered), scores in zip(dataset, guesses):
        survived = [idx for idx, val in enumerate(scores) if val >= threshold]
        rows.append(dict(event=pair["event"], kind=pair["kind"], covered=covered, detections=len(candidates),
                         survived=len(survived), recovered=target in survived))
    pos = [row for row in rows if row["kind"] == "positive"]
    controls = [row for row in rows if row["kind"] == "pre_event_control"]
    control_alerts = sum(row["survived"] for row in controls)
    covered_controls = sum(row["covered"] for row in controls)
    return dict(positive_pairs=len(pos), covered_positive_pairs=sum(row["covered"] for row in pos),
                recovered=sum(row["recovered"] for row in pos), control_pairs=len(controls),
                covered_controls=covered_controls, control_alerts=control_alerts,
                alerts_per_covered_control=control_alerts / covered_controls if covered_controls else None,
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_device_argument(parser)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    args.device, args.device_name = resolve_device(parser, args.device)
    data = Benchmark(args.v1)
    out = data.output(args.out)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["split_sha256"] != data.split_hash:
        raise ValueError("Pair manifest does not use the audited development split.")
    pairs = manifest["pairs"]
    if any(p["split"] not in ("train", "validation") or data.splits.get(p["event"]) != p["split"] for p in pairs):
        raise ValueError("Pair manifest contains an unknown, misassigned, or reserved sky field.")
    torch.set_num_threads(4)
    training = [load_pair(pair, BASELINE) for pair in pairs if pair["split"] == "train"]
    validation = [load_pair(pair, BASELINE) for pair in pairs if pair["split"] == "validation"]
    images, labels, audit = training_examples(training)
    print(f"Using {int(labels.sum())} real alerts and {len(labels)-int(labels.sum())} candidate dipoles", flush=True)
    model, history = fit(images, labels, args.epochs, args.seed, args.device)
    guessed = [classify(model, delta, candidates, args.device) if candidates else []
               for _, delta, candidates, _, _ in validation]
    baseline = evaluate(validation, [[1.] * len(pair[2]) for pair in validation], 0.)
    thresholds = (.25, .4, .5, .6, .75)
    options = {threshold: evaluate(validation, guessed, threshold) for threshold in thresholds}
    threshold = max(thresholds, key=lambda value: (options[value]["recovered"], -options[value]["control_alerts"],
                                                    -abs(value - .5)))
    torch.save(dict(kind="hermae-transient-v1", model=model.cpu().state_dict(), baseline_sigma=BASELINE,
                    threshold=threshold, split_sha256=data.split_hash), out / "transient.pt")
    report = dict(manifest_sha256=digest(args.manifest), split_sha256=data.split_hash, seed=args.seed,
                  epochs=args.epochs, device=args.device, device_name=args.device_name,
                  baseline_sigma=BASELINE, threshold=threshold, loss=history, training_examples=audit,
                  baseline=baseline, classifier=options[threshold], thresholds=options)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
