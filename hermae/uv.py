"""Experimental UV point-transient localization on same-filter image pairs."""

import csv
import json
from pathlib import Path
import re

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import Sip, WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, map_coordinates, maximum_filter
import torch
from torch import nn

from .development import digest


SIZE = 64
GALEX = {"nuv": "near UV", "fuv": "far UV"}
FILTERS = {
    "nuv": "nuv", "galex near uv": "nuv", "fuv": "fuv", "galex far uv": "fuv",
    "uvw1": "uvw1", "uvm2": "uvm2", "uvw2": "uvw2",
}


class UVLocalizer(nn.Sequential):
    def __init__(self):
        # Receptive field 15 px, so real search excludes an 8 px coverage rim.
        super().__init__(
            nn.Conv2d(4, 24, 5, padding=2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 32, 5, padding=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 24, 5, padding=2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )


def robust_scale(values):
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    if scale <= 0:
        scale = np.std(values)
    return float(center), max(float(scale), 1e-10)


def features(reference, science, valid, band):
    """Flux-match with clipped regression and express the residual in robust noise units."""
    r = reference[valid].astype(float)
    s = science[valid].astype(float)
    if len(r) < 64:
        raise ValueError("Insufficient usable overlap for UV subtraction.")
    rc, rs = robust_scale(r)
    sc, ss = robust_scale(s)
    x, y = np.clip((r - rc) / rs, -12, 12), np.clip((s - sc) / ss, -12, 12)
    xx = x - np.mean(x)
    initial = np.clip(np.dot(xx, y - np.mean(y)) / max(np.dot(xx, xx), 1e-10), .1, 2.)
    offset, scatter = robust_scale(y - initial * x)
    keep = np.abs(y - initial * x - offset) < 3 * scatter
    slope = initial
    if keep.sum() > 64:
        kx, ky = x[keep], y[keep]
        kx = kx - np.mean(kx)
        slope = np.clip(np.dot(kx, ky - np.mean(ky)) / max(np.dot(kx, kx), 1e-10), .1, 2.)
    factor = float(slope * ss / rs)
    residual = (science - sc) - factor * (reference - rc)
    dc, ds = robust_scale(residual[valid])
    channel = -1 if band == "fuv" else 1
    stack = np.stack(((reference - rc) / rs, (science - sc) / ss,
                      (residual - dc) / ds, np.full(reference.shape, channel)))
    stack = np.where(valid, np.clip(stack, -30, 30), 0).astype(np.float32)
    return stack, dict(reference_gain=factor, residual_noise=ds)


def read_uv(path, hdu):
    with fits.open(path, memmap=False) as hdus:
        image = hdus[hdu]
        values = np.array(image.data, dtype=np.float32)
        header = hdus[0].header.copy()
        header.update(image.header)
        wcs = WCS(image.header, fobj=hdus).celestial
    if values.ndim != 2 or not wcs.has_celestial:
        raise ValueError("Select a two-dimensional UV science image with celestial WCS.")
    labels = [str(header.get(key, "")).lower().strip() for key in ("FILTER", "SURVEY", "BANDS")]
    band = next((FILTERS[label] for label in labels if label in FILTERS), None)
    if band is None:
        for line in header.get("HISTORY", []):
            match = re.search(r"GALEX (?:Near|Far) UV", line, re.I)
            if match:
                band = FILTERS[match[0].lower()]
                break
    if band is None:
        raise ValueError("Missing supported UV filter metadata in the science HDU.")
    date = header.get("DATE-OBS")
    time = Time(date).mjd if date else header.get("MJD-OBS")
    return values, wcs, band, None if time is None else float(time), header


def aligned_pair(reference, science, rhdu, shdu, ref_fwhm, sci_fwhm, center, size):
    ref, rw, rb, rt, rh = read_uv(reference, rhdu)
    data, sw, sb, st, sh = read_uv(science, shdu)
    if rb != sb:
        raise ValueError(f"Use identical UV filters for both epochs ({rb} and {sb}).")
    if rt is None or st is None or not (np.isfinite(rt) and np.isfinite(st) and rt < st):
        raise ValueError("Both FITS images need observation dates in increasing order.")
    for label, header in (("reference", rh), ("science", sh)):
        if float(header.get("EXPOSURE", 1)) <= 0:
            raise ValueError(f"Invalid {label} exposure.")
    for key in ("TELESCOP", "INSTRUME", "BUNIT"):
        rvalue, svalue = [str(h.get(key, "")).strip().lower() for h in (rh, sh)]
        if rvalue and svalue and rvalue != svalue:
            raise ValueError(f"Use matching {key} metadata in both images.")
    cy, cx = (np.asarray(data.shape) - 1) / 2
    if center is not None:
        from astropy.coordinates import SkyCoord
        cx, cy = (float(v) for v in sw.world_to_pixel(SkyCoord(*center, unit="deg")))
    if not (np.isfinite(cx) and np.isfinite(cy) and 0 <= cx < data.shape[1] and 0 <= cy < data.shape[0]):
        raise ValueError("The field center is outside the science image.")
    scale = float(np.mean(proj_plane_pixel_scales(sw)) * 3600)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The science WCS needs a usable pixel scale.")
    # Set the matched PSF to ~3 px, comparable to the injection distribution.
    stride = max(1, round(max(ref_fwhm, sci_fwhm) / (3 * scale)))
    x0, y0 = round(cx) - (size // 2) * stride, round(cy) - (size // 2) * stride
    grid = sw.deepcopy()
    grid.wcs.crpix -= [x0, y0]
    grid.wcs.crpix = (grid.wcs.crpix - 1) / stride + 1
    if grid.wcs.has_cd():
        grid.wcs.cd = grid.wcs.cd * stride
    else:
        grid.wcs.cdelt *= stride
    if sw.sip is not None:
        coefficients = []
        for poly in (sw.sip.a, sw.sip.b, sw.sip.ap, sw.sip.bp):
            coefficients.append(None if poly is None else
                                poly * float(stride)**(np.indices(poly.shape).sum(axis=0) - 1))
        grid.sip = Sip(*coefficients, (sw.sip.crpix - [x0, y0] - 1) / stride + 1)
    # Use the original WCS for projection, including any distortion table.
    y, x = np.indices((size, size), dtype=float)
    sy, sx = y * stride + y0, x * stride + x0
    rx, ry = rw.world_to_pixel(sw.pixel_to_world(sx, sy))

    def sample(values, yy, xx, wcs, fwhm):
        native_scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600)
        blur = np.sqrt(max(ref_fwhm, sci_fwhm)**2 - fwhm**2) / (2.355 * native_scale)
        valid = np.isfinite(values)
        if "swift" in str(sh.get("TELESCOP", "")).lower():
            # Preserve zero-count interior pixels while excluding detector padding.
            valid &= binary_fill_holes(values > 0)
        clean = np.where(valid, values, 0)
        coverage = valid.astype(np.float32)
        if blur > .15:
            clean = gaussian_filter(clean, blur)
            coverage = gaussian_filter(coverage, blur)
        pixels = map_coordinates(clean, [yy, xx], order=1, cval=0, prefilter=False)
        usable = map_coordinates(coverage, [yy, xx], order=1, cval=0, prefilter=False) > .98
        return pixels, usable

    ref, ref_ok = sample(ref, ry, rx, rw, ref_fwhm)
    sci, sci_ok = sample(data, sy, sx, sw, sci_fwhm)
    return ref, sci, ref_ok & sci_ok, grid, rb, (rt, st), (x0, y0, stride), sw


def probability_map(model, stack, valid, device):
    with torch.inference_mode():
        scores = model(torch.from_numpy(stack[None]).to(device)).sigmoid().cpu().numpy()[0, 0]
    interior = binary_erosion(valid, structure=np.ones((17, 17), bool))
    return np.where(interior, scores, np.nan)


def candidates(scores, threshold, residual=None):
    clean = np.nan_to_num(scores, nan=-1.)
    y, x = np.where((clean >= threshold) & (clean == maximum_filter(clean, size=7)))
    # Break saturated-score ties by PSF-smoothed signal and suppress duplicate peaks.
    signal = gaussian_filter(residual, 1.3)[y, x] if residual is not None else np.zeros(len(x))
    order = np.lexsort((-signal, -clean[y, x]))
    peaks = []
    blocked = np.zeros(scores.shape, bool)
    for i in order:
        px, py = int(x[i]), int(y[i])
        if blocked[py, px]:
            continue
        peaks.append((px, py, float(clean[py, px])))
        blocked[max(0, py - 3):py + 4, max(0, px - 3):px + 4] = True
        if len(peaks) == 25:
            break
    return peaks


def write_run(out, reference, science, stack, scores, peaks, grid, positions, band, metadata, truth=None):
    """Save WCS-tagged FITS planes, point candidates and linked dashboard overlays."""
    from .uv_view import save_uv_views

    out.mkdir(parents=True, exist_ok=True)
    files = save_uv_views(out, reference, science, stack[2], scores, peaks, truth)
    header = grid.to_header(relax=True)
    fits.HDUList([fits.PrimaryHDU(scores, header), fits.ImageHDU(stack[2], name="RESIDUAL", header=header),
                  fits.ImageHDU(np.isfinite(scores).astype("uint8"), name="COVERAGE", header=header)
                  ]).writeto(out / "uv-scores.fits", overwrite=True)
    rows = [dict(rank=i, x=x, y=y, score=score, ra_deg=float(sky.icrs.ra.deg), dec_deg=float(sky.icrs.dec.deg))
            for i, ((x, y, score), sky) in enumerate(zip(peaks, positions), 1)]
    with (out / "uv-candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "x", "y", "score", "ra_deg", "dec_deg"])
        writer.writeheader()
        writer.writerows(rows)
    result = dict(mode="uv_search", display_name=metadata.pop("display_name"), band=band,
                  status="candidates_for_review" if rows else "no_residual_above_threshold",
                  files=[*files, "uv-candidates.csv", "uv-scores.fits"], candidates=len(rows),
                  top_candidate=rows[0] if rows else None, **metadata)
    (out / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def search(args):
    ref, sci, valid, grid, band, dates, crop, native = aligned_pair(
        args.reference, args.science, args.reference_hdu, args.science_hdu,
        args.reference_fwhm, args.science_fwhm, args.center, args.size,
    )
    checkpoint = torch.load(args.weights, map_location=args.device, weights_only=True)
    net = UVLocalizer().to(args.device).eval()
    net.load_state_dict(checkpoint["model"])
    stack, info = features(ref, sci, valid, band)
    scores = probability_map(net, stack, valid, args.device)
    if not np.isfinite(scores).any():
        raise ValueError("The epochs do not overlap enough for a UV search.")
    threshold = checkpoint["threshold"]
    peaks = candidates(scores, threshold, stack[2])
    x0, y0, stride = crop
    sky = [native.pixel_to_world(x0 + x * stride, y0 + y * stride) for x, y, _ in peaks]
    info.update(display_name=args.name or f"{args.science.stem} / {band.upper()}",
                evidence="archival_epoch_pair", filter_transfer=band not in GALEX, epoch_mjd=list(dates),
                threshold=threshold, coverage=float(np.isfinite(scores).mean()), candidate_limit=25,
                pixel_scale_arcsec=float(np.mean(proj_plane_pixel_scales(grid)) * 3600),
                inputs={key: dict(path=str(Path(path).resolve()), sha256=digest(path), hdu=hdu, fwhm_arcsec=fwhm)
                        for key, path, hdu, fwhm in (
                            ("reference", args.reference, args.reference_hdu, args.reference_fwhm),
                            ("science", args.science, args.science_hdu, args.science_fwhm))})
    mark = None
    if args.compare_position:
        from astropy.coordinates import SkyCoord
        expected = SkyCoord(*args.compare_position, unit="deg", frame="icrs")
        offset = [float(expected.separation(position).arcsec) for position in sky]
        info["comparison_position"] = dict(ra_deg=expected.ra.deg, dec_deg=expected.dec.deg,
                                           top_arcsec=offset[0] if offset else None,
                                           nearest_arcsec=min(offset) if offset else None,
                                           nearest_rank=offset.index(min(offset)) + 1 if offset else None)
        px, py = native.world_to_pixel(expected)
        x, y = float((px - x0) / stride), float((py - y0) / stride)
        if 0 <= x < scores.shape[1] and 0 <= y < scores.shape[0]:
            mark = x, y
    return write_run(args.out, ref, sci, stack, scores, peaks, grid, sky, band, info, mark)
