"""Retrieve and verify DR10 south r-band science and weight cutouts."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd

from .survey import LAYER, PIX_SCALE, VIEWER, SourceError

SIZE = 511
PADDING_ARCSEC = 6


def image_request(event, radius):
    scale = max(PIX_SCALE, 2 * (radius + PADDING_ARCSEC) / (SIZE - 3))
    return dict(ra=f"{event.ra:.8f}", dec=f"{event.dec:.8f}", layer=LAYER, bands="r",
                size=SIZE, pixscale=scale, invvar=1)


def read_image(path, event, scale):
    with fits.open(path) as hdus:
        header = hdus[0].header
        if len(hdus) != 2 or any(h.header.get("BANDS") != "r" for h in hdus):
            raise SourceError("Expected one r-band science image and its inverse-variance map.")
        image, weights = (np.array(hdu.data, dtype=float) for hdu in hdus)
        if image.shape != (SIZE, SIZE) or weights.shape != image.shape:
            raise SourceError("Unexpected FITS image dimensions.")
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
                  cache_file="", sha256="", pixel_scale_arcsec=params["pixscale"], valid_fraction=0.)
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
