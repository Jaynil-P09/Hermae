"""Known-position host ranking from full-frame detections."""

import csv
from dataclasses import asdict
from hashlib import file_digest
from importlib.metadata import version
import json

from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np

from .comparison import comparison
from .detection import detect, measurements
from .fits_input import read_frame, sn_position
from .overlay import render

DETECTION_COLUMNS = (
    "source_id", "x_px", "y_px", "ra_deg", "dec_deg", "footprint_pixels", "xmin", "xmax_exclusive",
    "ymin", "ymax_exclusive", "flux", "flux_noise", "major_sigma_px", "minor_sigma_px",
    "major_angle_px_rad", "major_pa_deg", "classification", "frame_edge", "coverage_edge", "reliable_shape",
)
RANK_COLUMNS = (
    "source_id", "ra_deg", "dec_deg", "x_px", "y_px", "classification", "separation_arcsec",
    "directional_sigma_arcsec", "size_scaled_offset", "shape_fallback", "nearest_rank", "rank",
)


def rank_hosts(frame, detections, sn, x, y, radius):
    candidates = []
    for row in detections:
        sky = SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg")
        separation = float(sky.separation(sn).arcsec)
        if separation > radius:
            continue
        radius_arcsec = ratio = None
        if row["reliable_shape"]:
            theta = np.arctan2(y - row["y_px"], x - row["x_px"])
            delta = theta - row["major_angle_px_rad"]
            directional = 1 / np.hypot(np.cos(delta) / row["major_sigma_px"],
                                       np.sin(delta) / row["minor_sigma_px"])
            end = frame.wcs.pixel_to_world(row["x_px"] + directional * np.cos(theta),
                                           row["y_px"] + directional * np.sin(theta))
            radius_arcsec = float(sky.separation(end).arcsec)
            if np.isfinite(radius_arcsec) and radius_arcsec > 0:
                ratio = separation / radius_arcsec
            else:
                radius_arcsec = None
        candidates.append(dict(**{key: row[key] for key in RANK_COLUMNS[:6]},
                               separation_arcsec=separation, directional_sigma_arcsec=radius_arcsec,
                               size_scaled_offset=ratio, shape_fallback=ratio is None))
    candidates.sort(key=lambda r: (r["separation_arcsec"], r["source_id"]))
    for rank, row in enumerate(candidates, start=1):
        row["nearest_rank"] = rank
    candidates.sort(key=lambda r: (r["shape_fallback"], r["size_scaled_offset"]
                                   if r["size_scaled_offset"] is not None else r["separation_arcsec"],
                                   r["nearest_rank"]))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    return candidates


def write_table(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def host(args, config):
    with args.fits.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    sn = SkyCoord(args.ra, args.dec, unit="deg")
    external = comparison(args.v1, args.event, digest, sn) if args.v1 else None
    if args.v1 and args.out.resolve().is_relative_to(args.v1.resolve()):
        raise ValueError("Write v2 outputs outside the v1 data worktree.")
    files = ["detections.csv", "hosts.csv", "segmentation.fits", "overlay.png", "run.json"]
    if external:
        files.append("comparison.json")
    if args.fits.resolve() in {(args.out / name).resolve() for name in files}:
        raise ValueError("Output names collide with the science FITS input.")
    frame = read_frame(args.fits, args.hdu, mask_hdu=args.mask_hdu, ivar_hdu=args.ivar_hdu,
                       variance_hdu=args.variance_hdu)
    sn, x, y, coverage = sn_position(frame, args.ra, args.dec)
    backend = None
    if getattr(args, "detector", "classical") == "cnn":
        from .galaxy import GalaxyDetector
        backend = GalaxyDetector.load(args.galaxy_weights, args.device)
    labels, signal, variance = detect(frame, config, backend)
    detections = measurements(frame, labels, signal, variance)
    ranked = rank_hosts(frame, detections, sn, x, y, args.radius) if coverage == "covered" else []
    rank_info = None
    if getattr(args, "ranker_weights", None):
        from .ranker import rank_image_hosts
        rank_info = rank_image_hosts(args.ranker_weights, args.device, frame, labels, signal, variance,
                                     ranked, detections, x, y, backend)
    status = "host_proposed" if ranked else ("no_plausible_host_detected" if coverage == "covered"
                                             else "no_usable_coverage")
    if rank_info and rank_info["review_required"]:
        status = "multiple_plausible_hosts" if len(ranked) > 1 else "host_requires_review"
    summary = dict(
        format_version=1, mode="known_position", status=status, detector="cnn" if backend else "classical",
        input=dict(path=str(args.fits.resolve()), sha256=digest, shape_yx=list(frame.science.shape),
                   band=frame.band, flux_units=frame.units, hdu=frame.hdu),
        sn=dict(ra_deg=args.ra, dec_deg=args.dec, x_px=x if np.isfinite(x) else None,
                y_px=y if np.isfinite(y) else None, coverage=coverage),
        valid_pixel_fraction=float(frame.valid.mean()), radius_arcsec=args.radius, config=asdict(config),
        dependencies={name: version(name) for name in ("astropy", "numpy", "scipy", "photutils", "scikit-image")},
        detections=len(detections), candidates=len(ranked), provisional_host=ranked[0] if ranked else None,
        files=files,
    )
    if backend:
        summary["galaxy_checkpoint"] = dict(sha256=backend.checkpoint_sha256,
                                            footprint_threshold=backend.footprint_threshold,
                                            centre_threshold=backend.centre_threshold)
    if rank_info:
        summary["ranker"] = rank_info
    args.out.mkdir(exist_ok=True, parents=True)
    write_table(args.out / "detections.csv", detections, DETECTION_COLUMNS)
    write_table(args.out / "hosts.csv", ranked, RANK_COLUMNS + (("ranker_score",) if rank_info else ()))
    fits.writeto(args.out / "segmentation.fits", labels, frame.wcs.to_header(), overwrite=True)
    render(frame, labels, x, y, ranked, external, args.out / "overlay.png",
           "CNN galaxy detection" if backend else "Classical source detection")
    if external:
        external["v2_provisional_host"] = ranked[0] if ranked else None
        if ranked:
            pick = SkyCoord(ranked[0]["ra_deg"], ranked[0]["dec_deg"], unit="deg")
            for key in ("dlr", "measured_fits"):
                obj = external[key]
                if obj:
                    obj["separation_from_v2_arcsec"] = float(
                        pick.separation(SkyCoord(obj["ra_deg"], obj["dec_deg"], unit="deg")).arcsec)
        if external["published_position_deg"]:
            published = SkyCoord(*external["published_position_deg"], unit="deg")
            matches = []
            for row in ranked:
                offset = float(published.separation(SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg")).arcsec)
                if offset <= external["reference_tolerance_arcsec"]:
                    matches.append(dict(source_id=row["source_id"], rank=row["rank"], offset_arcsec=offset))
            external["v2_reference_centroid_matches"] = matches
        (args.out / "comparison.json").write_text(json.dumps(external, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    (args.out / "run.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
