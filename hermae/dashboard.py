"""Create an offline dashboard from a saved Hermae run."""

import base64
from html import escape
from importlib.resources import files
import json
from pathlib import Path

from .development import records


def png(path):
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def dashboard(folder, out=None):
    folder = folder.resolve()
    summary = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    pair = summary["mode"] == "pair_search"
    if not pair and summary["mode"] != "known_position":
        raise ValueError("Select the top-level directory for a known-position or pair-search run.")
    data = dict(mode=summary["mode"], status=summary["status"], counts=dict(
        detections=summary["detections"], candidates=summary["candidates"]),
        image=png(folder / "overlay.png") if "overlay.png" in summary["files"] else None,
        tables={}, hosts={})
    if pair:
        data["tables"]["Transient candidates"] = records(folder / "transients.csv")
        for candidate in data["tables"]["Transient candidates"]:
            name = candidate["host_run"]
            if not name:
                continue
            child = (folder / name).resolve()
            if child.parent != folder:
                raise ValueError("A host result must be a direct subdirectory of its pair run.")
            data["hosts"][candidate["transient_id"]] = dict(image=png(child / "overlay.png"),
                table=records(child / "hosts.csv"), status=candidate["host_status"])
    else:
        data["tables"]["Ranked hosts"] = records(folder / "hosts.csv")
        data["tables"]["Full-frame detections"] = records(folder / "detections.csv")
    data["overlay_alt"] = ("Aligned reference, newer exposure and difference image with candidate boxes."
                           if pair else "Full-frame FITS and detail showing footprints, the supernova and provisional host.")
    out = out or folder / "dashboard.html"
    if out.suffix.lower() != ".html":
        raise ValueError("A dashboard output must have an .html extension.")
    title = escape(folder.name)
    serialised = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    template = files("hermae").joinpath("dashboard.html").read_text(encoding="utf-8")
    before, after = template.replace("@@TITLE@@", title).split("@@DATA@@")
    out.write_text(before + serialised + after, encoding="utf-8")
    return out
