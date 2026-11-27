"""Freeze a release configuration and evaluate the two reserved cohorts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import urlencode

from astropy.coordinates import SkyCoord
import numpy as np

from .development import Benchmark, digest, records
from .detection import DetectorConfig, detect, measurements
from .fits_input import read_frame, sn_position
from .host import rank_hosts
from .ztf_pairs import get, pairs_for

SCOPE = ("hermae", "tools", "pyproject.toml")
CUTOUT = "https://www.legacysurvey.org/viewer/fits-cutout"
COHORT_INPUTS = (".cache/ztf/snia_data.csv", ".cache/ztf/globalhost_data.csv", "benchmark/development_events.csv")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*words):
    return subprocess.check_output(["git", *words], text=True).strip()


def freeze(args, data):
    if git("status", "--porcelain", "--", *SCOPE):
        raise ValueError("Commit the pipeline and evaluation code before freezing.")
    assets = dict(galaxy=args.galaxy, ranker=args.ranker, cnn_ranker=args.cnn_ranker,
                  classifier=args.classifier, v1_baseline=args.v1_baseline,
                  pair_development_manifest=args.development_pairs)
    value = dict(format=1, code_commit=git("rev-parse", "HEAD"), split_sha256=data.split_hash,
                 match_arcsec=data.tolerance, cone_arcsec=60., pair_sigma=4., pair_match_arcsec=2.5,
                 independent_pair_candidate_count=36, independent_pair_seed=793,
                 default_known_detector="classical", default_pair_detector="classical",
                 default_pair_workflow="subtraction", assets={},
                 external_cohort_inputs={name: digest(data.root / name) for name in COHORT_INPUTS})
    for name, path in assets.items():
        path = path.resolve()
        value["assets"][name] = dict(path=path.relative_to(Path.cwd()).as_posix(), sha256=digest(path))
    write_json(args.manifest, value)


def verify(path, data):
    value = json.loads(path.read_text(encoding="utf-8"))
    committed = git("show", f"HEAD:{path.resolve().relative_to(Path.cwd()).as_posix()}")
    if json.loads(committed) != value:
        raise ValueError("Commit the freeze manifest before accessing reserved data.")
    subprocess.run(["git", "diff", "--exit-code", value["code_commit"], "--", *SCOPE], check=True,
                   stdout=subprocess.DEVNULL)
    if value["split_sha256"] != data.split_hash:
        raise ValueError("Evaluation requires the frozen field split.")
    for artifact in value["assets"].values():
        if digest(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"Frozen artifact has changed: {artifact['path']}")
    for name, expected in value["external_cohort_inputs"].items():
        if digest(data.root / name) != expected:
            raise ValueError(f"Frozen cohort source has changed: {name}")
    return value


def fetch_known(row, path):
    if not path.exists():
        params = urlencode(dict(ra=f"{float(row['ra']):.8f}", dec=f"{float(row['dec']):.8f}",
                                layer="ls-dr10-south", bands="r", invvar=1, size=511, pixscale=.262))
        temp = path.with_suffix(".part")
        temp.write_bytes(get(f"{CUTOUT}?{params}"))
        read_frame(temp, ivar_hdu=1)
        temp.replace(path)
    return read_frame(path, ivar_hdu=1)


def separation(row, truth):
    return float(SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg").separation(truth).arcsec)


def result(ranked, truth, tolerance):
    if truth is None:
        return dict(outcome="missing_published_position", top1=False, found=0)
    offsets = [separation(row, truth) for row in ranked]
    found = [i + 1 for i, distance in enumerate(offsets) if distance <= tolerance]
    failure = ("source_retrieval" if not found else "ambiguous_reference" if len(found) > 1
               else "host_ranking" if found[0] != 1 else None)
    return dict(outcome=failure or "agrees", top1=failure is None, found=len(found),
                published_rank=found[0] if len(found) == 1 else None, top_offset_arcsec=offsets[0] if offsets else None,
                top_candidate=ranked[0] if ranked else None)


def image_rank(frame, ra, dec, cone, backend, rank_path, pipeline, device):
    from .ranker import rank_image_hosts

    sn, x, y, coverage = sn_position(frame, float(ra), float(dec))
    if coverage != "covered":
        return [], coverage, None
    labels, signal, variance = detect(frame, DetectorConfig(), backend)
    detected = measurements(frame, labels, signal, variance)
    ranked = rank_hosts(frame, detected, sn, x, y, cone)
    info = rank_image_hosts(rank_path, device, frame, labels, signal, variance, ranked, detected,
                            x, y, backend, pipeline) if rank_path else None
    return ranked, coverage, info


def reference(row, ra="host_ra", dec="host_dec"):
    return SkyCoord(float(row[ra]), float(row[dec]), unit="deg") if row.get(ra) and row.get(dec) else None


def known(args, data, policy, out):
    import torch
    from .galaxy import GalaxyDetector
    from .ranker import load_ranker

    torch.set_num_threads(4)
    ids = sorted(e for e, split in data.splits.items() if split == "test")
    directory = out / "known-inputs"
    directory.mkdir(exist_ok=True)
    host_ranks = dict(classical_geometry=(None, None, None))
    galaxy = GalaxyDetector.load(Path(policy["assets"]["galaxy"]["path"]), args.device)
    for mode, backbone, key in (("classical_cnn_ranker", None, "ranker"), ("cnn_detector_and_ranker", galaxy, "cnn_ranker")):
        path = Path(policy["assets"][key]["path"])
        host_ranks[mode] = backbone, path, load_ranker(path, args.device)

    def acquire(event):
        row = data.events[event]
        path = directory / f"{event}.fits"
        try:
            fetch_known(row, path)
            return dict(event=event, path=str(path), sha256=digest(path))
        except (OSError, ValueError, IndexError, KeyError) as exc:
            return dict(event=event, path=None, error=str(exc))

    with ThreadPoolExecutor(max_workers=3) as pool:
        inputs = list(pool.map(acquire, ids))
    input_manifest = out / "known-inputs.json"
    write_json(input_manifest, inputs)
    predictor = out / "v1-measured-picks.json"
    subprocess.run([str(data.root / ".venv/Scripts/python.exe"), "tools/frozen_v1.py", str(data.root),
                    policy["assets"]["v1_baseline"]["path"], "--image-manifest", str(input_manifest),
                    "--out", str(predictor)], check=True)
    picks = json.loads(predictor.read_text(encoding="utf-8"))
    table = []
    for image in inputs:
        event = image["event"]
        row, published = data.events[event], data.references[event]
        truth = reference(published)
        dlr_pick = next((r for r in data.sources.get(event, []) if r["source_id"] == published["dlr_pick"]), None)
        dlr = dict(ra_deg=float(dlr_pick["ra"]), dec_deg=float(dlr_pick["dec"]), source_id=dlr_pick["source_id"]) if dlr_pick else None
        correct_reference = published["reference_source_id"] if published["reference_retrieved_60as"] == "True" else ""
        trial = dict(event=event, input=image, catalogue_reference_status=published["reference_status"],
                     published_position=[float(truth.ra.deg), float(truth.dec.deg)] if truth is not None else None,
                     coverage="fits_unavailable", methods={})
        for key, pick in (("v1_dlr", dlr), ("v1_measured_fits", picks.get(event))):
            hit = bool(correct_reference and pick and correct_reference == pick["source_id"])
            trial["methods"][key] = dict(top1=hit, found=int(bool(correct_reference)), top_candidate=pick,
                outcome="agrees" if hit else "host_ranking" if correct_reference else published["reference_status"],
                top_offset_arcsec=separation(pick, truth) if pick and truth is not None else None)
        if image["path"]:
            frame = read_frame(Path(image["path"]), ivar_hdu=1)
            for method, (backend, path, model) in host_ranks.items():
                ranked, coverage, info = image_rank(frame, row["ra"], row["dec"], policy["cone_arcsec"],
                                                    backend, path, model, args.device)
                trial["coverage"] = coverage
                entry = result(ranked, truth, policy["match_arcsec"])
                if coverage != "covered":
                    entry["outcome"] = "input_coverage"
                entry["ranker_review"] = info
                trial["methods"][method] = entry
        else:
            for method in host_ranks:
                trial["methods"][method] = dict(outcome="input_coverage", top1=False, found=0)
        table.append(trial)
        print(f"{event}: {trial['coverage']}", flush=True)
    totals = {}
    for method in table[0]["methods"]:
        results = [trial["methods"][method] for trial in table]
        paired = [(row["top1"], trial["methods"]["v1_measured_fits"]["top1"]) for row, trial in zip(results, table)]
        totals[method] = dict(top1=sum(row["top1"] for row in results), all_reserved=len(ids),
                              uniquely_recovered=sum(row["found"] == 1 for row in results),
                              improved_over_v1_measured=sum(a and not b for a, b in paired),
                              worsened_vs_v1_measured=sum(b and not a for a, b in paired))
    write_json(out / "known-position.json", dict(usable=sum(t["coverage"] == "covered" for t in table),
                                                 reserved=len(ids), methods=totals, fields=table))


def group(ra, dec):
    band = min(int((dec + 90) // 10), 17)
    sectors = max(1, int(np.rint(36 * np.cos(np.deg2rad((band + .5) * 10 - 90)))))
    return band, int(ra * sectors / 360)


def independent(args, data, policy, out):
    import torch
    from .ranker import load_ranker
    from .train_transient import load_pair
    from .transient import classify, load_classifier

    torch.set_num_threads(4)
    raw = data.root / ".cache/ztf"
    published = {row["ztfname"]: row for row in records(raw / "globalhost_data.csv")}
    forbidden = {group(float(row["ra"]), float(row["dec"])) for row in data.events.values()}
    # Prior pilot fields may be outside the expanded v1 cohort, but their sky is not independent.
    pilot = data.root / "benchmark/development_events.csv"
    pilot_rows = records(pilot)
    prior = {row["event_id"] for row in pilot_rows}
    catalogue = records(raw / "snia_data.csv")
    known_positions = SkyCoord([float(row["ra"]) for row in [*data.events.values(), *pilot_rows]],
                               [float(row["dec"]) for row in [*data.events.values(), *pilot_rows]], unit="deg")
    forbidden.update(group(float(r["ra"]), float(r["dec"])) for r in catalogue
                     if r["ztfname"] in prior and r["ra"] and r["dec"])
    eligible = []
    for row in catalogue:
        event = row["ztfname"]
        host = published.get(event)
        if (event in data.splits or event in prior or host is None or not row["t0"] or
                not host["ra_host"] or not host["dec_host"]):
            continue
        region = group(float(row["ra"]), float(row["dec"]))
        sky = SkyCoord(float(row["ra"]), float(row["dec"]), unit="deg")
        if region not in forbidden and known_positions.separation(sky).min().arcsec > 360:
            eligible.append((event, row, host, region))
    rng = np.random.default_rng(policy["independent_pair_seed"])
    sample, regions = [], set()
    for idx in rng.permutation(len(eligible)):
        event, row, host, region = eligible[idx]
        if region not in regions:
            sample.append((event, row, host))
            regions.add(region)
        if len(sample) == policy["independent_pair_candidate_count"]:
            break
    archive = out / "pair-inputs"
    archive.mkdir(exist_ok=True)

    def acquire(item):
        event, row, host = item
        try:
            return pairs_for(event, dict(row, host_ra=host["ra_host"], host_dec=host["dec_host"]),
                             float(row["t0"]) + 2400000.5, archive), None
        except (OSError, ValueError, KeyError, IndexError) as exc:
            return [], str(exc)

    pairs, failed = [], {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for (event, _, _), (found, error) in zip(sample, pool.map(acquire, sample)):
            pairs.extend(found)
            if error:
                failed[event] = error
            print(f"Independent pair {event}: {error or len(found)}", flush=True)
    write_json(out / "independent-pairs.json", dict(requested=[e for e, _, _ in sample], rejected=failed, pairs=pairs))
    path = Path(policy["assets"]["ranker"]["path"])
    ranker = load_ranker(path, args.device)
    model, state, _ = load_classifier(Path(policy["assets"]["classifier"]["path"]), args.device)
    details = []
    for pair in pairs:
        _, delta, candidates, target, covered = load_pair(pair, policy["pair_sigma"])
        scores = classify(model, delta, candidates, args.device) if candidates else []
        indices = [i for i, score in enumerate(scores) if score >= state["threshold"]]
        truth = reference(pair)
        host_result = None
        if target is not None and truth is not None:
            frame = read_frame(Path(pair["reference"]["path"]))
            trial = candidates[target]
            ranked, _, _ = image_rank(frame, trial["ra_deg"], trial["dec_deg"], policy["cone_arcsec"],
                                       None, path, ranker, args.device)
            host_result = result(ranked, truth, policy["match_arcsec"])
        details.append(dict(event=pair["event"], kind=pair["kind"], covered=covered, baseline_alerts=len(candidates),
                            cnn_alerts=len(indices), baseline_recovered=target is not None,
                            cnn_recovered=target in indices, host_result=host_result,
                            first_failure="input_coverage" if not covered else "transient_detection"
                            if target is None and pair["kind"] == "positive" else
                            host_result["outcome"] if host_result else None))
    positive = [row for row in details if row["kind"] == "positive"]
    control = [row for row in details if row["kind"] != "positive" and row["covered"]]
    metrics = {}
    for mode in ("baseline", "cnn"):
        recovered = [row for row in positive if row[f"{mode}_recovered"]]
        hit = sum(bool(row["host_result"] and row["host_result"]["top1"]) for row in recovered)
        metrics[mode] = dict(recovered=len(recovered), all_positive=len(positive), covered_positive=sum(
            row["covered"] for row in positive), host_top1=hit, host_conditional_denominator=len(recovered),
            end_to_end_denominator=len(positive), control_alerts=sum(row[f"{mode}_alerts"] for row in control),
            covered_controls=len(control))
    write_json(out / "image-pair.json", dict(requested_fields=len(sample), unavailable=len(failed),
                                             positive_pairs=len(positive), metrics=metrics, fields=details))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "known", "pairs"))
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    for name in ("galaxy", "ranker", "cnn-ranker", "classifier", "v1-baseline", "development-pairs"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    required = (args.galaxy, args.ranker, args.cnn_ranker, args.classifier, args.v1_baseline, args.development_pairs)
    if args.operation == "freeze" and not all(required):
        parser.error("Freezing requires every model checkpoint and the development pair manifest.")
    if args.operation != "freeze" and not args.out:
        parser.error("Reserved evaluation requires --out.")
    data = Benchmark(args.v1)
    if args.operation == "freeze":
        freeze(args, data)
        return
    policy = verify(args.manifest, data)
    data = Benchmark(args.v1, include_reserved=True)
    out = data.output(args.out)
    if args.operation == "known":
        known(args, data, policy, out)
    else:
        independent(args, data, policy, out)


if __name__ == "__main__":
    main()
