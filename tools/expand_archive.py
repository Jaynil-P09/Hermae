"""Download registered optical cutouts and GALEX UV backgrounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import numpy as np
from PIL import Image

from hermae.development import Benchmark, digest, records
from hermae.fits_input import read_frame, sn_position


TARGETS = (
    ("2020jfo", "M61", "12:21:50.480", "+04:28:54.05", 1.4),
    ("2020fqv", "NGC 4568", "12:36:33.260", "+11:13:53.87", 1.0),
    ("2022xxf", "NGC 3705", "11:30:05.890", "+09:16:57.30", 1.1),
    ("2018ivc", "M77", "02:42:41.29", "-00:00:31.4", 1.3),
    ("2016gkg", "NGC 613", "01:34:14.46", "-29:26:25.0", 1.3),
    ("2013ej", "M74", "01:36:48.16", "+15:45:31.0", 1.8),
    ("2011dh", "M51", "13:30:05.08", "+47:10:11.2", 1.9),
    ("2014J", "M82", "09:55:42.22", "+69:40:26.6", 1.8),
    ("1993J", "M81", "09:55:24.77", "+69:01:13.7", 3.0),
    ("2017eaw", "NGC 6946", "20:34:44.24", "+60:11:36.2", 1.9),
    ("2022ffv", "NGC 1380", "03:36:29.678", "-35:17:21.66", .9),
    ("2012aw", "M95", "10:43:53.72", "+11:40:17.9", 1.5),
    ("2007af", "NGC 5584", "14:22:21.03", "-00:23:37.6", 1.1),
    ("2017ein", "NGC 3938", "11:52:53.25", "+44:07:26.2", 1.2),
    ("2023ixf", "M101", "14:03:38.562", "+54:18:41.94", 0),
    ("2024ggi", "NGC 3621", "11:18:22.087", "-32:50:15.27", 0),
)
LEGACY = "https://www.legacysurvey.org/viewer/fits-cutout"
SKYVIEW = "https://skyview.gsfc.nasa.gov/cgi-bin/images"
BANDS = {"nuv": "GALEX Near UV", "fuv": "GALEX Far UV"}


def fetch(url, path, validate):
    if path.exists():
        return validate(path)
    for delay in (2, 7, None):
        try:
            with urlopen(url, timeout=32) as response:
                content = response.read(32 * 1024**2 + 1)
            if len(content) > 32 * 1024**2:
                raise ValueError("Unexpected FITS size")
            result = validate(BytesIO(content))
            path.write_bytes(content)
            return result
        except (HTTPError, TimeoutError) as exc:
            if delay is None or isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            retry = exc.headers.get("Retry-After", "") if isinstance(exc, HTTPError) else ""
            sleep(max(delay, int(retry)) if retry.isdecimal() else delay)


def read_cube(path):
    with fits.open(path, memmap=False) as hdus:
        cube = np.asarray(hdus[0].data, dtype=np.float32)
        header = hdus[0].header
        bands = str(header.get("BANDS", "")).replace(",", "").replace(" ", "").lower()
        if cube.ndim != 3 or cube.shape[0] != 3 or bands != "grz":
            raise ValueError(f"Expected the Legacy g/r/z plane ordering, received {bands!r}")
        if any(np.isfinite(p).mean() < .9 or np.nanstd(p) <= 0 for p in cube):
            raise ValueError("Insufficient optical science coverage")
        if not WCS(header).has_celestial:
            raise ValueError("Missing celestial WCS")
        return cube


def uv_info(path):
    with fits.open(path, memmap=False) as hdus:
        a = np.asarray(hdus[0].data)
        if a.ndim != 2 or not WCS(hdus[0].header).has_celestial:
            raise ValueError("Expected a celestial UV image")
        valid = np.isfinite(a) & (a > 0)
        if valid.mean() < .4 or a[valid].std() <= 0:
            raise ValueError("Insufficient positive UV coverage")
        return dict(shape=list(a.shape), positive_fraction=float(valid.mean()))


def optical(target, sky, root, args):
    name, host, _, _, scale = target
    folder = root / "optical" / name
    folder.mkdir(parents=True, exist_ok=True)
    for survey in ("ls-dr10", "ls-dr9"):
        query = dict(ra=sky.ra.deg, dec=sky.dec.deg, layer=survey, size=511)
        color_url = LEGACY + "?" + urlencode(dict(query, bands="grz", pixscale=scale))
        native_url = LEGACY + "?" + urlencode(dict(query, bands="r", pixscale=.262, invvar=1))
        cube_path, native_path = folder / f"{survey}-grz.fits", folder / f"{survey}-r.fits"
        try:
            cube = fetch(color_url, cube_path, read_cube)
            frame = fetch(native_url, native_path, read_frame)
            if sn_position(frame, sky.ra.deg, sky.dec.deg)[3] != "covered":
                raise ValueError("Native image misses the transient coordinates")
            break
        except (OSError, ValueError):
            if survey == "ls-dr9":
                raise
    command = [sys.executable, "-m", "hermae", "host", "--fits", str(native_path),
               "--ra", str(sky.ra.deg), "--dec", str(sky.dec.deg), "--out", str(folder),
               "--radius", "63", "--device", args.device, "--ranker-weights", str(args.weights)]
    with fits.open(native_path) as hdus:
        if len(hdus) > 1 and hdus[1].data.shape == frame.science.shape:
            command += ["--ivar-hdu", "1"]
    receipt_path = folder / "run.json"
    if not receipt_path.is_file():
        subprocess.run(command, check=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    backgrounds = np.nanpercentile(cube, 35, axis=(1, 2))
    g, r, z = np.nan_to_num(cube - backgrounds[:, None, None])
    stretch = max(float(np.percentile((r + g + z) / 3, 99.4)) * .55, .004)
    color_path = folder / "grz-composite.png"
    Image.fromarray(make_lupton_rgb(z, r, g, stretch=stretch, Q=7)[::-1]).save(color_path)
    receipt.update(display_name=f"SN {name} / {host}", survey_origin=f"Legacy Surveys {survey.upper()}",
                   color_view=dict(file=color_path.name, sha256=digest(color_path), fits=cube_path.name,
                                   url=color_url, pixel_scale_arcsec=scale),
                   coordinate_reference=f"https://www.wis-tns.org/object/{name}", archive_host=host)
    receipt["files"] = list(dict.fromkeys([*receipt["files"], color_path.name, cube_path.name]))
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return dict(event=name, host=host, path=str(folder), color_url=color_url, native_url=native_url,
                color_sha256=digest(cube_path), native_sha256=digest(native_path),
                detections=receipt["detections"], candidates=receipt["candidates"], status=receipt["status"])


def ultraviolet(target, sky, root, band):
    name, host, *_ = target
    folder = root / "uv" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"galex-{band}.fits"
    query = dict(Survey=BANDS[band], Position=f"{sky.ra.deg},{sky.dec.deg}", Return="FITS",
                 Pixels=480, Size=.20, Coordinates="J2000", Projection="Tan")
    url = SKYVIEW + "?" + urlencode(query)
    info = fetch(url, path, uv_info)
    return dict(event=name, host=host, band=band, path=str(path), url=url, sha256=digest(path), **info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/expanded-archive"))
    parser.add_argument("--weights", type=Path, default=Path("results/all-development-training/ranker-classical/image_only.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--v1", type=Path, default=Path("../Hermae_v1"))
    parser.add_argument("--independent-pairs", type=Path, default=Path("results/release/independent-pairs.json"))
    parser.add_argument("--part", choices=["optical", "uv"], required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark(args.v1, include_reserved=True)
    previous = list(benchmark.events.values()) + list(records(benchmark.root / "benchmark/development_events.csv"))
    positions = [(float(row["ra"]), float(row["dec"])) for row in previous]
    positions += [(p["alert_ra"], p["alert_dec"]) for p in
                  json.loads(args.independent_pairs.read_text())["pairs"] if p["kind"] == "positive"]
    old = SkyCoord(*np.array(positions).T, unit="deg")
    jobs, excluded = [], []
    for target in TARGETS:
        sky = SkyCoord(target[2], target[3], unit=("hourangle", "deg"))
        separation = float(old.separation(sky).min().arcsec)
        if separation < max(630, target[4] * 511 / 1.414):
            excluded.append(dict(event=target[0], reason="Overlapping benchmark field", separation_arcsec=separation))
        elif args.part == "uv":
            jobs.extend((target, sky, band) for band in BANDS)
        elif target[4]:
            jobs.append((target, sky, None))

    def run(job):
        target, sky, band = job
        try:
            result = ultraviolet(target, sky, root, band) if band else optical(target, sky, root, args)
            print(f"{target[0]} {band or 'optical'} OK", flush=True)
            return True, result
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"{target[0]} {band or 'optical'}: {exc}", flush=True)
            return False, dict(event=target[0], band=band, reason=str(exc))

    completed = []
    with ThreadPoolExecutor(max_workers=4 if args.part == "uv" else 1) as pool:
        for result in pool.map(run, jobs):
            completed.append(result)
            manifest = dict(images=[row for ok, row in completed if ok],
                            unavailable=[row for ok, row in completed if not ok], excluded=excluded)
            (root / f"{args.part}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
