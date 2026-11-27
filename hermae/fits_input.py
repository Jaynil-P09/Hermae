"""Read a celestial science frame and establish usable pixel coverage."""

from dataclasses import dataclass

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


@dataclass
class Frame:
    science: np.ndarray
    valid: np.ndarray
    wcs: WCS
    variance: np.ndarray | None
    units: str | None
    band: str
    hdu: dict


def hdu_key(value):
    return int(value) if value.isdecimal() else value


def read_frame(path, science_hdu=0, *, mask_hdu=None, ivar_hdu=None, variance_hdu=None):
    if ivar_hdu is not None and variance_hdu is not None:
        raise ValueError("Choose inverse variance or variance.")
    with fits.open(path, mode="readonly", memmap=False) as hdus:
        science = hdus[science_hdu]
        values = np.array(science.data, dtype=float, copy=True)
        if values.ndim != 2 or min(values.shape) < 2:
            raise ValueError("Select a two-dimensional science HDU.")
        band = str(science.header.get("BANDS", science.header.get("FILTER", ""))).strip().lower()
        if band and band.split()[0] not in {"r", "zr", "ztf_r"}:
            raise ValueError(f"Expected an r-band image, got {band!r}.")
        # Passing the HDU list retains any WCS lookup tables carried in the FITS.
        wcs = WCS(science.header, fobj=hdus).celestial
        if not wcs.has_celestial or wcs.pixel_n_dim != 2:
            raise ValueError("The science HDU needs a two-dimensional celestial WCS.")
        height, width = values.shape
        xs, ys = [0., 0., width / 2., width - 1.], [0., height - 1., height / 2., height - 1.]
        sky = wcs.pixel_to_world(xs, ys)
        px, py = wcs.world_to_pixel(sky)
        if not (np.allclose(px, xs, atol=0.02, rtol=0) and np.allclose(py, ys, atol=0.02, rtol=0)):
            raise ValueError("Science WCS failed its pixel–sky round trip.")

        def companion(key):
            hdu = hdus[key]
            plane = np.array(hdu.data, dtype=float, copy=True)
            if plane.shape != values.shape:
                raise ValueError(f"HDU {key!r} has different dimensions from the science image.")
            alternative = WCS(hdu.header, fobj=hdus)
            if alternative.has_celestial:
                x, y = alternative.celestial.world_to_pixel(sky)
                if not (np.allclose(x, xs, atol=0.05, rtol=0) and np.allclose(y, ys, atol=0.05, rtol=0)):
                    raise ValueError(f"HDU {key!r} is misaligned with the science WCS.")
            return plane

        names = {hdu.name.upper() for hdu in hdus}
        if mask_hdu is None and "MASK" in names:
            mask_hdu = "MASK"
        if ivar_hdu is None and variance_hdu is None:
            ivar_hdu = next((name for name in ("IVAR", "INVVAR") if name in names), None)
            if ivar_hdu is None and "VARIANCE" in names:
                variance_hdu = "VARIANCE"
        valid = np.isfinite(values)
        if mask_hdu is not None:
            mask = companion(mask_hdu)
            mask_header = hdus[mask_hdu].header
            if (mask_header.get("ORIGIN") == "Zwicky Transient Facility"
                    and "BIT01" in mask_header and "BIT11" in mask_header
                    and "DETECTION" in mask_header.comments["BIT01"]
                    and "SOURCE POSITION" in mask_header.comments["BIT11"]):
                # ZTF's detection flags identify astrophysical signal in good pixels.
                finite = np.isfinite(mask)
                valid &= finite & ((np.where(finite, mask, 0).astype(np.int32) & 6141) == 0)
            else:
                valid &= mask == 0
        variance = None
        key = ivar_hdu if ivar_hdu is not None else variance_hdu
        if key is not None:
            weights = companion(key)
            usable = np.isfinite(weights) & (weights > 0)
            variance = (np.divide(1, weights, out=np.full_like(weights, np.nan), where=usable)
                        if ivar_hdu is not None else np.where(usable, weights, np.nan))
            valid &= usable & np.isfinite(variance)
        return Frame(values, valid, wcs, variance, science.header.get("BUNIT"),
                     "r" if band else "assumed_r",
                     dict(science=science_hdu, mask=mask_hdu, inverse_variance=ivar_hdu,
                          variance=variance_hdu))


def sn_position(frame, ra, dec):
    sky = SkyCoord(ra, dec, unit="deg", frame="icrs")
    x, y = (float(v) for v in frame.wcs.world_to_pixel(sky))
    height, width = frame.science.shape
    if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
        return sky, x, y, "outside_frame"
    if not frame.valid[int(np.floor(y + 0.5)), int(np.floor(x + 0.5))]:
        return sky, x, y, "masked_sn_pixel"
    return sky, x, y, "covered"
