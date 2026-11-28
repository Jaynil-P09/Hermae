"""Refit models on all eligible development images using preselected schedules."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import train_galaxy, train_ranker, train_transient
from .development import Benchmark, digest
from .galaxy import GalaxyDetector
from .training import DEFAULT_EPOCHS
from .training_device import add_device_argument, resolve_device
from .training_preview import add_preview_argument


def save_report(path, **report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prior_report(path, split_hash):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["split_sha256"] != split_hash:
        raise ValueError(f"Schedule source uses a different sky split: {path}")
    return report


def schedule(prior, model, history_key, device, fields, epochs):
    count = model["selected_epoch"]
    rows = model[history_key][:count]
    rates = [r["learning_rate"] for r in rows]
    if count < 1 or len(rows) != count or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("The prior report must contain every learning rate through its selected epoch.")
    rates = (rates + [rates[-1]] * max(0, epochs - count))[:epochs]
    args = SimpleNamespace(device=device, epochs=epochs, learning_rate=rates[0], seed=prior["seed"],
                           weight_decay=prior["weight_decay"], patience=prior["patience"],
                           **{name: prior[name] for name in fields})
    return args, rates


def refit_galaxy(data, ids, source, folder, device, epochs, *, live_preview=None):
    prior = prior_report(source, data.split_hash)
    args, rates = schedule(prior, prior, "history", device, ("batch_size", "crops_per_field"), epochs)
    args.live_preview = live_preview
    tensors, used = train_galaxy.crops(data, ids, np.random.default_rng(args.seed), args.crops_per_field)
    if not used:
        raise ValueError("No galaxy crops could be prepared.")
    unused = set(ids).difference(used)
    print(f"Galaxy U-Net | {len(tensors[0])} training crops from {len(used)} development images", flush=True)
    model, progress = train_galaxy.fit(tensors, None, args, rates)
    footprint, centre = prior["thresholds"]
    path = folder / "galaxy.pt"
    torch.save(dict(kind="hermae-galaxy-v1", model=model.cpu().state_dict(), footprint_threshold=footprint,
                    centre_threshold=centre, split_sha256=data.split_hash), path)
    save_report(folder / "report.json", source_sha256=digest(source), split_sha256=data.split_hash,
                device=device, train_ids=used, training_crops=len(tensors[0]), history=progress.history,
                dropped={e: data.images.get(e, {}).get("status", "no audited image")
                         if data.images.get(e, {}).get("status") != "ok" else "no valid crops"
                         for e in ids if e in unused}, transferred_thresholds=[footprint, centre])
    return path, len(used)


def refit_rankers(data, ids, source, folder, device, epochs, detector_path=None, *, live_preview=None):
    prior = prior_report(source, data.split_hash)
    backend = GalaxyDetector.load(detector_path, device) if detector_path else None
    fields, excluded = train_ranker.prepare(data, ids, backend)
    print(f"Host rankers ({folder.name}) | {len(fields)} development images with usable host-search inputs",
          flush=True)
    checkpoint_hash = backend.checkpoint_sha256 if backend else None
    result = {}
    for mode in ("image_only", "catalogue_fusion"):
        use_catalogue = mode == "catalogue_fusion"
        args, rates = schedule(prior, prior["models"][mode], "epoch_metrics", device, ("batch_fields",), epochs)
        args.live_preview = live_preview
        args.preview_backend = "U-Net candidates" if detector_path else "Classical candidates"
        model, progress, used = train_ranker.fit(fields, None, use_catalogue, args, rates)
        torch.save(dict(kind="hermae-host-v1", model=model.cpu().state_dict(), use_catalogue=use_catalogue,
                        review_gap=None, detector_sha256=checkpoint_hash, split_sha256=data.split_hash),
                   folder / f"{mode}.pt")
        dropped = dict(excluded)
        for event, _, pools, references in fields:
            if len(references["union"]) != 1:
                dropped[event] = "missing_counterpart" if not len(references["union"]) else "ambiguous_counterpart"
            elif len(pools["union"]) < 2:
                dropped[event] = "no_negative_candidate"
        print(f"{mode} fitted on {len(used)} fields with a unique reference and a negative candidate", flush=True)
        result[mode] = dict(train_ids=used, dropped=dropped, history=progress.history)
    save_report(folder / "report.json", source_sha256=digest(source), split_sha256=data.split_hash,
                device=device, detector_sha256=checkpoint_hash, review_gap=None, models=result)
    return {mode: len(record["train_ids"]) for mode, record in result.items()}


def refit_transient(data, manifest_path, source, folder, device, epochs, supplemental=None, *, live_preview=None):
    prior = prior_report(source, data.split_hash)
    manifest = prior_report(manifest_path, data.split_hash)
    if prior["manifest_sha256"] != digest(manifest_path):
        raise ValueError("The ZTF manifest differs from the one used for schedule selection.")
    pairs = list(manifest["pairs"])
    additional = prior_report(supplemental, data.split_hash) if supplemental else {}
    pairs.extend(additional.get("pairs", []))
    if any(p["split"] not in ("train", "validation") or data.splits.get(p["event"]) != p["split"] for p in pairs):
        raise ValueError("Pair manifest contains an unknown, misassigned, or reserved sky field.")
    identities = [(p["reference"]["sha256"], p["science"]["sha256"]) for p in pairs]
    if len(set(identities)) != len(pairs):
        raise ValueError("The ZTF manifests contain overlapping exposure pairs.")
    pair_inputs = [train_transient.load_pair(pair, prior["baseline_sigma"]) for pair in pairs]
    images, target, audit = train_transient.training_examples(pair_inputs)
    print(f"Transient CNN | processed {len(pairs)} development pairs and found {len(audit)} labelled residuals",
          flush=True)
    args, rates = schedule(prior, prior, "history", device, ("batch_size",), epochs)
    args.live_preview = live_preview
    model, progress = train_transient.fit(images, target, None, args, rates, audit=audit)
    torch.save(dict(kind="hermae-transient-v1", model=model.cpu().state_dict(), baseline_sigma=prior["baseline_sigma"],
                    threshold=prior["threshold"], split_sha256=data.split_hash), folder / "transient.pt")
    counts = dict(pairs=len(pairs), exposures=len({p[k]["sha256"] for p in pairs for k in ("reference", "science")}),
                  labelled_residuals=len(audit), confirmed_alerts=int(target.sum()))
    # A pair can contribute a confirmed alert or a labelled dipole to the supervised loss.
    used_pairs = {(row["reference_sha256"], row["science_sha256"]) for row in audit}
    counts["pairs_with_labelled_residuals"] = len(used_pairs)
    inventory = [dict(event=p["event"], kind=p["kind"], candidates=len(rows), covered=covered,
                      labelled=identity in used_pairs)
                 for (p, _, rows, _, covered), identity in zip(pair_inputs, identities)]
    save_report(folder / "report.json", source_sha256=digest(source), manifest_sha256=digest(manifest_path),
                supplemental_sha256=digest(supplemental) if supplemental else None,
                unpaired=additional.get("unpaired", {}),
                split_sha256=data.split_hash, device=device, history=progress.history, training_examples=audit,
                inventory=inventory, counts=counts, transferred_threshold=prior["threshold"])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--schedules", required=True, type=Path, help="directory containing the four quality reports")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--supplemental-manifest", type=Path, help="extra verified development exposure pairs")
    parser.add_argument("--transient-only", action="store_true", help="refit only the transient classifier")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="epochs per model")
    add_device_argument(parser)
    add_preview_argument(parser)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    device, device_name = resolve_device(parser, args.device)
    torch.set_num_threads(4)
    data = Benchmark(args.v1)
    ids = data.ids("train") + data.ids("validation")
    schedules = {stage: args.schedules / stage / "report.json"
                 for stage in ("galaxy", "ranker-classical", "ranker-unet", "transient")}
    stages = ("transient",) if args.transient_only else tuple(schedules)
    for stage in stages:
        path = schedules[stage]
        prior_report(path, data.split_hash)
    out = data.output(args.out)
    if any(out.iterdir()):
        parser.error("Choose a new or empty output directory for the refit.")
    if args.transient_only:
        refit_transient(data, args.manifest, schedules["transient"], out, device, args.epochs,
                        args.supplemental_manifest, live_preview=args.live_preview)
        print(f"Transient refit finished. Checkpoint and report are in {out}", flush=True)
        return
    for stage in stages:
        (out / stage).mkdir()

    detector, galaxy_count = refit_galaxy(data, ids, schedules["galaxy"], out / "galaxy", device, args.epochs,
                                         live_preview=args.live_preview)
    classical = refit_rankers(data, ids, schedules["ranker-classical"], out / "ranker-classical", device, args.epochs,
                              live_preview=args.live_preview)
    unet = refit_rankers(data, ids, schedules["ranker-unet"], out / "ranker-unet", device, args.epochs, detector,
                         live_preview=args.live_preview)
    transient = refit_transient(data, args.manifest, schedules["transient"], out / "transient", device, args.epochs,
                                args.supplemental_manifest, live_preview=args.live_preview)
    save_report(out / "report.json", split_sha256=data.split_hash, device=device, device_name=device_name,
                epochs_per_model=args.epochs, development_fields=len(ids),
                reserved_fields=sum(s == "test" for s in data.splits.values()),
                galaxy_fields=galaxy_count, ranker_classical_fields=classical, ranker_unet_fields=unet,
                transient=transient, evaluation="none; this refit used both development partitions for training")
    print(f"Full development refit finished. Checkpoints and reports are in {out}", flush=True)


if __name__ == "__main__":
    main()
