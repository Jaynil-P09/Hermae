import unittest

import numpy as np

from hermae.imaging import aperture


class ApertureTest(unittest.TestCase):
    def test_masks_pixels_and_counts_coverage_at_cutout_boundary(self):
        science = np.full((11, 11), 2.0)
        science[5, 5] = 900
        valid = np.ones_like(science, dtype=bool)
        valid[5, 5] = False
        result = aperture(science, valid, np.array([5., 0.]), np.array([5., 5.]), 0, 2)
        self.assertEqual(result[0], 2)
        self.assertTrue(np.isnan(result[1]))

    def test_subpixel_aperture_uses_sky_projected_centre(self):
        science = np.arange(25, dtype=float).reshape(5, 5)
        result = aperture(science, np.ones_like(science, dtype=bool), np.array([2.65]),
                          np.array([1.]), 0, 0.4)
        self.assertEqual(result[0], 8)
