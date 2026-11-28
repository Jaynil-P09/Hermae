import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install the ml extras for UV tests")
class UVPairTest(unittest.TestCase):
    def test_align_shifted_wcs_and_preserve_result_coordinates(self):
        from hermae.uv import aligned_pair

        y, x = np.indices((89, 89))
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("ref.fits", "sci.fits")]
            for delta, date, path in zip((0, 4), ("2011-01-01", "2023-05-21"), paths):
                wcs = WCS(naxis=2)
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                wcs.wcs.crpix = [45 + delta, 45]
                wcs.wcs.crval = [150., 20.]
                wcs.wcs.cdelt = [-.5 / 3600, .5 / 3600]
                header = wcs.to_header()
                header["FILTER"], header["DATE-OBS"] = "UVW2", date
                img = 1 + 20 * np.exp(-((x - 44 - delta)**2 + (y - 44)**2) / 10)
                fits.writeto(path, img, header)
            ref, sci, valid, grid, band, dates, _, _ = aligned_pair(*paths, 0, 0, 3., 3., (150., 20.), 40)
            self.assertEqual(band, "uvw2")
            self.assertLess(dates[0], dates[1])
            self.assertTrue(valid.all())
            self.assertTrue(np.allclose(ref, sci, atol=1e-5))
            sky = grid.pixel_to_world(20, 20)
            self.assertAlmostEqual(sky.ra.deg, 150., places=7)
            self.assertAlmostEqual(sky.dec.deg, 20., places=7)
            with self.assertRaisesRegex(ValueError, "increasing order"):
                aligned_pair(*paths[::-1], 0, 0, 3., 3., None, 40)
            fits.setval(paths[1], "FILTER", value="UVM2")
            with self.assertRaisesRegex(ValueError, "identical UV filters"):
                aligned_pair(*paths, 0, 0, 3., 3., None, 40)

    def test_adjacent_saturated_pixels_report_one_peak(self):
        from hermae.uv import candidates

        scores = np.zeros((32, 32))
        scores[11, 10:12] = 1.
        scores[22, 25] = .96
        residual = np.zeros_like(scores)
        residual[11, 11] = 15
        self.assertEqual(candidates(scores, .9, residual), [(11, 11, 1.), (25, 22, .96)])
