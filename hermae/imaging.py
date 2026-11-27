"""Retrieve and verify DR10 south r-band science and weight cutouts."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd

from .survey import BANDS, LAYER, PIX_SCALE, VIEWER, SourceError

SIZE = 511
PADDING_ARCSEC = 6


def image_request(event, radius):
    scale = max(PIX_SCALE, 2 * (radius + PADDING_ARCSEC) / (SIZE - 3))
    # On fields without r-band weight at the SN, a grz request can still deliver
    # the surrounding r-band coverage or a valid zero-weight plane.
    bands = "r" if "r" in event.coverage_bands else BANDS
    return dict(ra=f"{event.ra:.8f}", dec=f"{event.dec:.8f}", layer=LAYER, bands=bands,
                size=SIZE, pixscale=scale, invvar=1)


def read_image(path, event, scale):
    with fits.open(path) as hdus:
        header = hdus[0].header
        bands = header.get("BANDS")
        if len(hdus) != 2 or bands not in {"r", BANDS} or hdus[1].header.get("BANDS") != bands:
            raise SourceError("Expected one r-band science image and its inverse-variance map.")
        shape = (SIZE, SIZE) if bands == "r" else (len(bands), SIZE, SIZE)
        if any(hdu.data is None or hdu.data.shape != shape for hdu in hdus):
            raise SourceError("Unexpected FITS image dimensions.")
        image, weights = (np.array(hdu.data if bands == "r" else hdu.data[bands.index("r")],
                                   dtype=float) for hdu in hdus)
        wcs = WCS(header, naxis=2)
        weights_wcs = WCS(hdus[1].header, naxis=2)
    sn = SkyCoord(event.ra, event.dec, unit="deg")
    expected = (SIZE - 1) / 2
    x, y = wcs.world_to_pixel(sn)
    pixel_scales = wcs.proj_plane_pixel_scales()
    if not (np.isfinite([x, y]).all() and np.hypot(x - expected, y - expected) <= 0.1
            and np.allclose([item.to_value("arcsec") for item in pixel_scales], scale, atol=0.001)):
        raise SourceError("FITS WCS does not match the requested centre and scale.")
    expected_x = [0, expected, SIZE - 1, 0]
    expected_y = [SIZE - 1, expected, 0, 0]
    controls = wcs.pixel_to_world(expected_x, expected_y)
    wx, wy = weights_wcs.world_to_pixel(controls)
    if not (np.allclose(wx, expected_x, atol=0.01)
            and np.allclose(wy, expected_y, atol=0.01)):
        raise SourceError("Science and inverse-variance WCS are misaligned.")
    valid = np.isfinite(image) & np.isfinite(weights) & (weights > 0)
    return image, valid, wcs


def fetch_image(cache, event, radius):
    params = image_request(event, radius)
    record = dict(event_id=event.event_id, split=event.split, status="ok", error="",
                  cache_file="", sha256="", requested_bands=params["bands"],
                  pixel_scale_arcsec=params["pixscale"], valid_fraction=0.)
    try:
        path = cache.get(f"{VIEWER}/fits-cutout", "image_ablation", ".fits", params)
        record["cache_file"] = path.name
        with path.open("rb") as stream:
            record["sha256"] = file_digest(stream, "sha256").hexdigest()
        _, valid, _ = read_image(path, event, params["pixscale"])
        record["valid_fraction"] = float(valid.mean())
        if not valid.any():
            record["status"] = "no_r_band_weight"
    except (SourceError, OSError, ValueError, TypeError, IndexError) as exc:
        record.update(status="query_or_fits_failed", error=str(exc))
    return record


def fetch_images(cache, events, radius, workers):
    records = []
    with ThreadPoolExecutor(workers) as pool:
        for record in pool.map(lambda event: fetch_image(cache, event, radius), events.itertuples()):
            records.append(record)
            if record["status"] != "ok" or len(records) % 10 == 0 or len(records) == len(events):
                print(f"Images {len(records)}/{len(events)} {record['event_id']} {record['status']}", flush=True)
    return pd.DataFrame(records)


PIXEL_FEATURES = ["image_core", "image_envelope", "image_sn_to_core", "image_bridge_contrast"]


def aperture(image, valid, x, y, inner, outer):
    """Mean pixel values in circular apertures or annuli, with a coverage threshold."""
    pad = int(np.ceil(outer)) + 1
    grid_y, grid_x = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    px = np.rint(x).astype(int)[:, None, None] + grid_x
    py = np.rint(y).astype(int)[:, None, None] + grid_y
    distance2 = (px - x[:, None, None]) ** 2 + (py - y[:, None, None]) ** 2
    region = (distance2 >= inner**2) & (distance2 <= outer**2)
    inside = (px >= 0) & (py >= 0) & (px < image.shape[1]) & (py < image.shape[0])
    safe_x = np.clip(px, 0, image.shape[1] - 1)
    safe_y = np.clip(py, 0, image.shape[0] - 1)
    mask = region & inside & valid[safe_y, safe_x]
    total = region.sum(axis=(1, 2))
    good = mask.sum(axis=(1, 2))
    flux = np.where(mask, image[safe_y, safe_x], 0).sum(axis=(1, 2))
    measured = (total > 0) & (good >= 0.8 * total)
    return np.divide(flux, good, out=np.full(len(x), np.nan), where=measured)


def pixel_features(image, valid, wcs, event, candidates, scale):
    """Measure local light using only candidate and transient coordinates."""
    result = candidates[["event_id", "source_id"]].copy()
    result[PIXEL_FEATURES] = np.nan
    if not valid.any() or candidates.empty:
        return result
    sn_x, sn_y = wcs.world_to_pixel(SkyCoord(event.ra, event.dec, unit="deg"))
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    valid = valid & (np.hypot(xx - sn_x, yy - sn_y) * scale > 1.4)
    if not valid.any():
        return result
    sky = np.median(image[valid])
    noise = 1.4826 * np.median(np.abs(image[valid] - sky))
    if not np.isfinite(noise) or noise <= 0:
        return result
    flux = (image - sky) / noise
    coords = SkyCoord(candidates.ra.to_numpy(), candidates.dec.to_numpy(), unit="deg")
    x, y = wcs.world_to_pixel(coords)
    core = aperture(flux, valid, x, y, 0, 0.8 / scale)
    envelope = aperture(flux, valid, x, y, 2 / scale, 4.5 / scale)
    sn = aperture(flux, valid, np.array([sn_x]), np.array([sn_y]), 1.8 / scale, 3 / scale)[0]
    dx, dy = (sn_x - x) / 2, (sn_y - y) / 2
    bridge = aperture(flux, valid, x + dx, y + dy, 0, 0.8 / scale)
    rotate = np.sqrt(0.5)
    left = aperture(flux, valid, x + (dx - dy) * rotate, y + (dx + dy) * rotate, 0, 0.8 / scale)
    right = aperture(flux, valid, x + (dx + dy) * rotate, y + (dy - dx) * rotate, 0, 0.8 / scale)
    contrast = bridge - (left + right) / 2
    contrast[np.hypot(dx, dy) * scale < 2.2] = np.nan
    result["image_core"] = np.arcsinh(core)
    result["image_envelope"] = np.arcsinh(envelope)
    result["image_sn_to_core"] = np.arcsinh(sn / (1 + np.abs(core)))
    result["image_bridge_contrast"] = np.arcsinh(contrast)
    return result
