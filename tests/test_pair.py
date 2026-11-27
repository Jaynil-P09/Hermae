"""Regression tests for WCS registration, positive residuals and mask decoding."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from hermae.fits_input import Frame, read_frame
from hermae.pair import difference, residual_candidates


def grid(ra, x):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [x + 1, 33.]
    wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
    wcs.wcs.crval = [ra, 12.]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class PairTest(unittest.TestCase):
    def test_shifted_transient_and_no_overlap(self):
        y, x = np.mgrid[:64, :64]
        rng = np.random.default_rng(42)
        valid = np.ones(x.shape, dtype=bool)
        ref = Frame(rng.normal(size=x.shape), valid, grid(70., 32), None, None, "r", {})
        image = rng.normal(size=x.shape) + 25 * np.exp(-((x - 38)**2 + (y - 32)**2) / 3.)
        science = Frame(image, valid, grid(70., 38), None, None, "r", {})
        result = difference(ref, science, 2.5, 2.5)
        rows = residual_candidates(ref, result, 4.)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["x_px"], 32, delta=.5)
        self.assertAlmostEqual(rows[0]["y_px"], 32, delta=.5)
        science.wcs = grid(71., 38)
        self.assertIsNone(difference(ref, science, 2.5, 2.5))

    def test_ztf_detection_bits_are_usable(self):
        header = grid(70., 32).to_header()
        header["FILTER"] = "ZTF_r"
        mask_header = header.copy()
        mask_header["ORIGIN"] = "Zwicky Transient Facility"
        mask_header["BIT01"] = (1, "CONTAINS SEXTRACTOR DETECTION")
        mask_header["BIT11"] = (11, "CONTAINS PSF-EXTRACTED SOURCE POSITION")
        mask = np.zeros((64, 64), dtype=np.int16)
        mask[0, :3] = (2, 2050, 2050 + 256)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "science.fits"
            fits.HDUList([fits.PrimaryHDU(np.zeros(mask.shape), header),
                          fits.ImageHDU(mask, mask_header, name="MASK")]).writeto(path)
            frame = read_frame(path)
        self.assertEqual(frame.valid[0, :3].tolist(), [True, True, False])
