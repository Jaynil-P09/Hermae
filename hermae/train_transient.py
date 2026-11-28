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
from .training import BestCheckpoint, optimization_arguments, validate_optimization
from .transient import TransientNet, classify, stamps
from .training_device import add_device_argument, resolve_device

BASELINE = 4.
MATCH = 2.5
THRESHOLDS = (.25, .4, .5, .6, .75)


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
                                  reference_sha256=pair["reference"]["sha256"], science_sha256=pair["science"]["sha256"],
                                  candidate=candidates[i]["transient_id"]))
    if set(target) != {0, 1}:
        raise ValueError("Training requires recovered alerts and dipole artifact examples.")
    return torch.from_numpy(np.stack(images)), torch.tensor(target, dtype=torch.float32), audit


def fit(images, target, validation, args, rates=None):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TransientNet().to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    groups = [(target == i).nonzero().flatten().numpy() for i in (0, 1)]
    checkpoint = BestCheckpoint(args.patience, rates)
    per_class = max(map(len, groups))
    for epoch in range(1, args.epochs + 1):
        checkpoint.start_epoch(optimiser)
        if rates is None:
            balanced = [rng.choice(group, size=per_class, replace=True) for group in groups]
        else:
            # Visit every labelled stamp before repeating examples from the smaller class.
            balanced = [rng.permutation(np.concatenate((group, rng.choice(group, per_class - len(group)))))
                        for group in groups]
        one_each = np.stack(balanced, axis=1)
        totals = []
        for batch in torch.from_numpy(one_each.reshape(-1)).split(args.batch_size):
            batch_images = images[batch].to(args.device)
            batch_images = torch.rot90(batch_images, int(rng.integers(4)), (-2, -1))
            if rng.integers(2):
                batch_images = batch_images.flip((-1,))
            loss = F.binary_cross_entropy_with_logits(model(batch_images), target[batch].to(args.device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            totals.append(float(loss.detach()))
        train_loss = float(np.mean(totals))
        if rates is not None:
            print(f"Transient CNN epoch {epoch}/{args.epochs}  loss {train_loss:.4f}", flush=True)
            checkpoint.observe(model, optimiser, (-train_loss,), train_loss=train_loss)
            continue
        guessed, threshold, options = validation_metrics(model.eval(), validation, args.device)
        operating_point = options[threshold]
        positive = [guesses[index] for (_, _, _, index, _), guesses in zip(validation, guessed)
                    if index is not None]
        control = [value for (pair, _, _, _, covered), guesses in zip(validation, guessed)
                   if pair["kind"] == "pre_event_control" and covered for value in guesses]
        recovered, alerts = operating_point["recovered"], operating_point["control_alerts"]
        control_error = float(np.mean(np.square(control))) if control else 0.
        error = float(np.mean((1 - np.asarray(positive))**2)) + control_error if positive else None
        key = (recovered, -alerts, -error) if error is not None else (-train_loss,)
        print(f"Transient CNN epoch {epoch}/{args.epochs}  loss {train_loss:.4f}"
              f"  validation {recovered} recovered, {alerts} control alerts", flush=True)
        if checkpoint.observe(model, optimiser, key, train_loss=train_loss, validation_recovered=recovered,
                              validation_control_alerts=alerts, validation_score_error=error, threshold=threshold):
            break
    return checkpoint.restore(model), checkpoint


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


def validation_metrics(model, validation, device):
    guessed = [classify(model, delta, candidates, device) if candidates else []
               for _, delta, candidates, _, _ in validation]
    options = {threshold: evaluate(validation, guessed, threshold) for threshold in THRESHOLDS}
    threshold = max(THRESHOLDS, key=lambda value: (options[value]["recovered"], -options[value]["control_alerts"],
                                                   -abs(value - .5)))
    return guessed, threshold, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_device_argument(parser)
    optimization_arguments(parser, epochs=100, learning_rate=.0015, patience=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_optimization(parser, args)
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("Batch size must be a positive even number for balanced batches.")
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
    model, checkpoint = fit(images, labels, validation, args)
    _, threshold, options = validation_metrics(model, validation, args.device)
    baseline = evaluate(validation, [[1.] * len(pair[2]) for pair in validation], 0.)
    torch.save(dict(kind="hermae-transient-v1", model=model.cpu().state_dict(), baseline_sigma=BASELINE,
                    threshold=threshold, split_sha256=data.split_hash), out / "transient.pt")
    report = dict(manifest_sha256=digest(args.manifest), split_sha256=data.split_hash, seed=args.seed,
                  epochs=args.epochs, device=args.device, device_name=args.device_name,
                  learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                  patience=args.patience, batch_size=args.batch_size, selected_epoch=checkpoint.best_epoch,
                  history=checkpoint.history, loss=[row["train_loss"] for row in checkpoint.history],
                  baseline_sigma=BASELINE, threshold=threshold, training_examples=audit,
                  baseline=baseline, classifier=options[threshold], thresholds=options)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
