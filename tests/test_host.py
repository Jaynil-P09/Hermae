from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

from hermae.comparison import comparison
from hermae.detection import DetectorConfig, detect, measurements
from hermae.fits_input import read_frame, sn_position


class HostTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "source.fits"
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [56., 46.]
        self.wcs.wcs.cdelt = [-0.262 / 3600, 0.262 / 3600]
        self.wcs.wcs.crval = [179., -2.]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    def write(self, data, mask=None):
        header = self.wcs.to_header()
        header["BANDS"] = "r"
        images = [fits.PrimaryHDU(), fits.ImageHDU(data, header, name="SCI")]
        if mask is not None:
            images.append(fits.ImageHDU(mask.astype(np.uint8), header, name="MASK"))
        fits.HDUList(images).writeto(self.path)

    def run_cli(self, ra, dec):
        output = self.root / "run"
        subprocess.run([sys.executable, "-m", "hermae", "host", "--fits", str(self.path),
                        "--hdu", "SCI", "--ra", str(ra), "--dec", str(dec), "--out", str(output)],
                       check=True, capture_output=True)
        summary = json.loads((output / "run.json").read_text())
        self.assertTrue((output / "overlay.png").is_file())
        return summary

    def test_wcs_round_trip_and_mask(self):
        image = np.ones((90, 110))
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[45, 55] = 1
        self.write(image, mask)
        frame = read_frame(self.path, "SCI")
        sky, x, y, status = sn_position(frame, 179, -2)
        self.assertAlmostEqual(x, 55., places=5)
        self.assertAlmostEqual(y, 45., places=5)
        self.assertLess(sky.separation(frame.wcs.pixel_to_world(x, y)).arcsec, 1e-5)
        self.assertEqual(status, "masked_sn_pixel")

    def test_tiled_deblending_keeps_adjacent_sources(self):
        y, x = np.mgrid[:90, :110]
        data = np.random.default_rng(11).normal(0, .08, x.shape)
        for cx, cy in [(60.2, 38.4), (67.2, 38.6), (63.8, 65.3)]:
            data += 5. * np.exp(-((x - cx)**2 + (y - cy)**2)/(2 * 1.55**2))
        self.write(data)
        frame = read_frame(self.path, "SCI")
        labels, signal, var = detect(frame, DetectorConfig(tile=64, overlap=16))
        sources = measurements(frame, labels, signal, var)
        local = sorted(row["x_px"] for row in sources if 33 < row["y_px"] < 45 and 52 < row["x_px"] < 74)
        self.assertEqual(len(local), 2, sources)
        np.testing.assert_allclose(local, [60.2, 67.2], atol=1)
        self.assertNotEqual(labels[38, 60], labels[38, 67])
        seam = [row for row in sources if 59 < row["x_px"] < 68 and 60 < row["y_px"] < 70]
        self.assertEqual(len(seam), 1)
        self.assertEqual(labels[65, 63], seam[0]["source_id"])
        self.assertEqual(labels[65, 64], seam[0]["source_id"])

    def test_sn_outside_still_exports_full_frame_detections(self):
        y, x = np.mgrid[:90, :110]
        self.write(np.exp(-((x-32)**2 + (y-40)**2) / 6))
        summary = self.run_cli(180, -2)
        self.assertEqual(summary["status"], "no_usable_coverage")
        self.assertEqual(summary["sn"]["coverage"], "outside_frame")
        self.assertGreater(summary["detections"], 0)
        self.assertEqual(summary["candidates"], 0)

    def test_empty_covered_field(self):
        self.write(np.full((90, 110), 3.))
        summary = self.run_cli(179, -2)
        self.assertEqual(summary["status"], "no_plausible_host_detected")
        self.assertEqual(summary["detections"], 0)
        self.assertIsNone(summary["provisional_host"])

    def test_v1_comparison_excludes_reserved_event_before_reading_outputs(self):
        benchmark = self.root / "benchmark"
        benchmark.mkdir()
        split = benchmark / "split.json"
        split.write_text(json.dumps({"assignments": [{"event_id": "reserved", "split": "test"}]}))
        (benchmark / "v1-preservation.json").write_text(json.dumps({
            "split_manifest": {"path": "benchmark/split.json", "sha256": sha256(split.read_bytes()).hexdigest()},
        }))
        with self.assertRaisesRegex(ValueError, "development events only"):
            comparison(self.root, "reserved", "unused", SkyCoord(1, 2, unit="deg"))
