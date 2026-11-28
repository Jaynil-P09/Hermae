"""Regression coverage for the offline gallery."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from hermae.dashboard import dashboard


class DashboardTest(unittest.TestCase):
    def test_combined_gallery_ignores_stale_overlay_and_embeds_safe_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            known, pair = root / "known", root / "pair"
            known.mkdir()
            pair.mkdir()
            (known / "run.json").write_text(json.dumps(dict(
                mode="known_position", status="no_plausible_host_detected", detections=0, candidates=0,
                input=dict(path=str(root / "missing.fits"), sha256="missing", shape_yx=[2, 2]),
                files=["hosts.csv", "detections.csv", "run.json"])), encoding="utf-8")
            (known / "hosts.csv").write_text("source_id,rank\n", encoding="utf-8")
            (known / "detections.csv").write_text("source_id\n", encoding="utf-8")
            camera = dict(path=str(root / "missing.fits"), sha256="missing")
            (pair / "run.json").write_text(json.dumps(dict(
                mode="pair_search", status="no_usable_coverage", detections=0, candidates=0,
                inputs=dict(reference=camera, science=camera), files=["transients.csv", "run.json"],
                metadata="</script><script>stale</script>")), encoding="utf-8")
            (pair / "transients.csv").write_text("transient_id,host_run\n", encoding="utf-8")
            (pair / "overlay.png").write_bytes(b"stale image")

            content = dashboard([known, pair], root / "gallery.html").read_text(encoding="utf-8")
            self.assertNotIn("</script><script>stale", content)
            payload = re.search(r'id="run-data">(.*?)</script>', content).group(1)
            runs = json.loads(payload)["runs"]
            self.assertEqual([item["summary"]["mode"] for item in runs], ["known_position", "pair_search"])
            self.assertIsNone(runs[1]["layers"][0]["image"])
            self.assertNotIn("overlay.png", [item["name"] for item in runs[1]["artifacts"]])
            self.assertEqual(runs[0]["artifacts"][0]["url"], "known/hosts.csv")

            previews = root / "preview.png"
            previews.write_bytes(b"\x89PNG\r\n\x1a\n")
            content = dashboard([], root / "grouped.html", collections=[("Hermae runs", known), ("NASA fields", pair)],
                                previews=[("NASA fields", previews)]).read_text(encoding="utf-8")
            payload = re.search(r'id="run-data">(.*?)</script>', content).group(1)
            menus = json.loads(payload)["collections"]
            self.assertEqual([(item["name"], item["runs"]) for item in menus],
                             [("Hermae runs", [0]), ("NASA fields", [1])])
            self.assertIsNone(menus[0]["preview"])
            self.assertTrue(menus[1]["preview"].startswith("data:image/png;base64,"))
