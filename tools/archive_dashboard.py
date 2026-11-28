"""Build the combined results dashboard and a standalone archive dashboard."""

import argparse
import json
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from hermae.dashboard import dashboard
from hermae.development import digest
from tools.expand_archive import TARGETS


PROJECT_RUNS = (
    "all-development-training/known-example", "all-development-training/pair-example",
    "examples/known", "examples/pair", "my-host-run", "phase2/ZTF18abvtiko",
    "phase4/baseline", "phase4/ZTF19aartuli", "quality-training/host-smoke", "ZTF18abvtiko",
)
NASA_EVENTS = ("2023ixf", "2021rhu", "2020oi", "2022hrs", "2024ggi")
ARCHIVE_RUNS = [*(f"{event}/mast-r" for event in NASA_EVENTS[:4]), "2023ixf/galex-nuv", "2024ggi/dss2-red"]


def reference_run(entry, out, positions):
    source = Path(entry["path"])
    if not source.is_file() or digest(source) != entry["sha256"]:
        raise ValueError(f"Missing or modified UV reference: {source}")
    sky = positions[entry["event"]]
    wcs = WCS(fits.getheader(source)).celestial
    x, y = (float(v) for v in wcs.world_to_pixel(sky))
    height, width = entry["shape"]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Event outside GALEX reference: {source}")
    band = {"nuv": "near ultraviolet", "fuv": "far ultraviolet"}[entry["band"]]
    run = dict(mode="survey_image", display_name=f"{entry['host']} / {entry['band'].upper()}",
               view_title=f"{entry['host']} in {band}", status="survey_context",
               finding_title="GALEX reference field", reported_host=entry["host"],
               finding="This archival mosaic shows UV structure at the published event position. "
                       "The UV localizer uses two documented observations in the same filter to estimate a change.",
               files=["run.json"], sn=dict(ra_deg=sky.ra.deg, dec_deg=sky.dec.deg, pixel=[x, y]),
               input=dict(path=str(source.resolve()), sha256=entry["sha256"], science_hdu=0,
                          shape_yx=entry["shape"], valid_fraction=entry["positive_fraction"],
                          coverage_metric="Positive pixels", band=band, source="NASA SkyView / GALEX", url=entry["url"]))
    target = out / f"{entry['event']}-{entry['band']}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=Path("results"), type=Path)
    args = parser.parse_args()
    results = args.results
    root = results / "expanded-archive"
    nasa = results / "nasa-fields"
    positions = {event: SkyCoord(ra, dec, unit=("hourangle", "deg")) for event, _, ra, dec, _ in TARGETS}
    optical = json.loads((root / "optical.json").read_text(encoding="utf-8"))["images"]
    uv = json.loads((root / "uv.json").read_text(encoding="utf-8"))["images"]
    groups = {
        "Optical / color FITS": [Path(row["path"]) for row in optical],
        "UV / Swift epochs": sorted(p.parent for p in (root / "swift").glob("*/run.json")),
        "UV / injection tests": sorted(p.parent for p in (root / "uv-model/examples").glob("*/run.json")),
        "UV / GALEX references": [reference_run(row, root / "uv-references", positions) for row in uv],
    }
    expanded = [(name, path) for name, paths in groups.items() for path in paths]
    dashboard([], root / "dashboard.html", collections=expanded)
    previous = {
        "Hermae runs": [results / name for name in PROJECT_RUNS],
        "NASA fields": [nasa / "runs" / event / view for event in NASA_EVENTS for view in ("host-trained", "host-wide")],
        "NASA archive images": [nasa / "runs" / name for name in ARCHIVE_RUNS],
    }
    original = [(name, path) for name, paths in previous.items() for path in paths if (path / "run.json").is_file()]
    preview = nasa / "field-preview.png"
    dashboard([], results / "dashboard.html", collections=[*expanded, *original],
              previews=[("NASA fields", preview), ("NASA archive images", preview)] if preview.is_file() else [])
    print(f"{len(optical)} optical runs, {len(uv)} GALEX references and "
          f"{len(groups['UV / Swift epochs'])} Swift epoch comparisons")


if __name__ == "__main__":
    main()
