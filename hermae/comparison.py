"""Read v1 comparison ranks for a verified development image."""

import csv
from hashlib import sha256
import json

from astropy.coordinates import SkyCoord


def rows(path, event):
    with path.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["event_id"] == event]


def comparison(root, event, image_sha, sn):
    audit = json.loads((root / "benchmark/v1-preservation.json").read_text(encoding="utf-8"))
    split_path = root / audit["split_manifest"]["path"]
    if sha256(split_path.read_bytes()).hexdigest() != audit["split_manifest"]["sha256"]:
        raise ValueError("The v1 split differs from its preservation audit.")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split = next((row["split"] for row in manifest["assignments"] if row["event_id"] == event), None)
    if split not in {"train", "validation"}:
        raise ValueError("v1 comparisons accept development events only.")
    run = root / audit["data_paths"]["benchmark"]
    for name in ("events.csv", "evaluation.csv", "candidates.csv"):
        if sha256((run / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != audit["frozen_benchmark_sha256"][name]:
            raise ValueError(f"v1 benchmark input differs from the audit: {name}")
    inputs = run / "images/image_inputs.csv"
    rankings = run / "images/image_rankings.csv"
    for path in (inputs, rankings):
        if sha256(path.read_bytes()).hexdigest() != audit["replayed_outputs"]["images"]["files"][path.name]:
            raise ValueError(f"v1 FITS output differs from the audit: {path.name}")
    [image] = rows(inputs, event)
    if image_sha != image["sha256"] or image["split"] != split:
        raise ValueError("The supplied FITS file is not the audited v1 image for this event.")
    [transient] = rows(run / "events.csv", event)
    sky = SkyCoord(float(transient["ra"]), float(transient["dec"]), unit="deg")
    if sn.separation(sky).arcsec > 0.001:
        raise ValueError("The supplied SN coordinates differ from the v1 event.")
    [reference] = rows(run / "evaluation.csv", event)
    fit = next((row for row in rows(rankings, event) if row["image_rank"] == "1"), None)
    sources = {row["source_id"]: row for row in rows(run / "candidates.csv", event)}

    def pick(source_id, score):
        source = sources[source_id]
        return dict(source_id=source_id, ra_deg=float(source["ra"]), dec_deg=float(source["dec"]),
                    score=score)

    dlr = reference["dlr_pick"]
    return dict(event_id=event, split=split, catalogue_reference_status=reference["reference_status"],
                reference_tolerance_arcsec=manifest["reference_tolerance_arcsec"],
                catalogue_reference_id=reference["reference_source_id"] or None,
                published_position_deg=[float(reference["host_ra"]), float(reference["host_dec"])]
                if reference["host_ra"] and reference["host_dec"] else None,
                dlr=pick(dlr, float(sources[dlr]["dlr"]) if sources[dlr]["dlr"] else None) if dlr else None,
                measured_fits=pick(fit["source_id"], float(fit["image_score"])) if fit else None,
                provenance="v1 preserved development outputs")
