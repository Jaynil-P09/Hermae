"""Read-only access to audited v1 data with the frozen sky split."""

import csv
from hashlib import file_digest, sha256
import json

import numpy as np
from astropy.coordinates import SkyCoord

from .fits_input import read_frame


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class Benchmark:
    def __init__(self, root, *, include_reserved=False):
        self.root = root.resolve()
        audit = json.loads((self.root / "benchmark/v1-preservation.json").read_text(encoding="utf-8"))
        path = self.root / audit["split_manifest"]["path"]
        self.split_hash = digest(path)
        if self.split_hash != audit["split_manifest"]["sha256"]:
            raise ValueError("The frozen split manifest does not match its audit.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.splits = {row["event_id"]: row["split"] for row in manifest["assignments"]}
        self.tolerance = manifest["reference_tolerance_arcsec"]
        self.run = self.root / audit["data_paths"]["benchmark"]
        self.cache = self.root / audit["data_paths"]["fits_cache"]
        paths = {name: self.run / name for name in ("events.csv", "candidates.csv", "evaluation.csv")}
        for name, path in paths.items():
            if sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != audit["frozen_benchmark_sha256"][name]:
                raise ValueError(f"The frozen benchmark table has changed: {name}")
        allowed = {event for event, split in self.splits.items() if include_reserved or split in ("train", "validation")}
        self.events = {row["event_id"]: row for row in records(paths["events.csv"]) if row["event_id"] in allowed}
        self.references = {row["event_id"]: row for row in records(paths["evaluation.csv"])
                           if row["event_id"] in allowed}
        self.sources = {}
        for row in records(paths["candidates.csv"]):
            if row["event_id"] in allowed:
                self.sources.setdefault(row["event_id"], []).append(row)
        path = self.root / audit["data_paths"]["fit_manifest"]
        if digest(path) != audit["image_audit_sha256"]:
            raise ValueError("The frozen FITS manifest does not match its audit.")
        self.images = {row["event_id"]: row for row in records(path)}

    def output(self, path):
        resolved = path.resolve()
        if resolved.is_relative_to(self.root):
            raise ValueError("Write outputs outside the preserved v1 worktree.")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def ids(self, split):
        if split not in ("train", "validation"):
            raise ValueError("Development jobs accept only train and validation fields.")
        return sorted(event for event, part in self.splits.items() if part == split)

    def frame(self, event):
        if self.splits.get(event) not in ("train", "validation"):
            raise ValueError("Reserved FITS access is disabled during development.")
        image = self.images.get(event)
        if image is None or image["status"] != "ok":
            return None
        path = self.cache / image["cache_file"]
        if digest(path) != image["sha256"] or image["split"] != self.splits[event]:
            raise ValueError(f"FITS identity or split differs from the audit: {event}")
        return read_frame(path, ivar_hdu=1)

    def red_sources(self, event, frame):
        rows = [row for row in self.sources.get(event, ()) if float(row["flux_r"] or 0) > 0
                and float(row["flux_ivar_r"] or 0) > 0]
        if not rows:
            return []
        sky = SkyCoord([float(row["ra"]) for row in rows], [float(row["dec"]) for row in rows], unit="deg")
        xs, ys = frame.wcs.world_to_pixel(sky)
        height, width = frame.science.shape
        result = []
        for row, x, y in zip(rows, xs, ys):
            if not (0 <= x < width and 0 <= y < height):
                continue
            galaxy = row["type"].strip() in ("EXP", "DEV", "SER", "REX") and float(row["shape_r"] or 0) > 0
            snr = float(row["flux_r"]) * np.sqrt(float(row["flux_ivar_r"]))
            result.append((row, float(x), float(y), galaxy, float(snr)))
        return result
