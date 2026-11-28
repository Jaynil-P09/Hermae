"""Build a portable overlay gallery and data explorer from saved runs."""

import base64
from html import escape
from importlib.resources import files
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.parse import quote

from astropy.io import fits
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave
import numpy as np

from .development import digest, records
from .fits_input import read_frame


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def png_url(content):
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


def saved_png(folder, summary):
    path = folder / "overlay.png"
    return png_url(path.read_bytes()) if "overlay.png" in summary["files"] and path.is_file() else None


RESIDUAL = LinearSegmentedColormap.from_list("hermae_residual", ("#4f9ce9", "#121b2b", "#f88b81"))


def residual_boxes(colours, candidates, stride):
    height, width = colours.shape[:2]
    size = max(3, min(height, width) // 26)
    for candidate in candidates:
        x, y = (round(float(candidate[key]) / stride) for key in ("x_px", "y_px"))
        x0, x1 = max(0, x - size), min(width - 1, x + size)
        y0, y1 = max(0, y - size), min(height - 1, y + size)
        if x0 >= x1 or y0 >= y1:
            continue
        colour = (.45, 1., .70, 1) if candidate["survived"].lower() == "true" else (1., .74, .42, 1)
        colours[[y0, y1], x0:x1 + 1] = colour
        colours[y0:y1 + 1, [x0, x1]] = colour


def mark_position(colours, pixel, stride):
    x, y = (round(float(value) / stride) for value in pixel)
    height, width = colours.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return
    length = max(5, min(height, width) // 42)
    background = (0.04, 0.07, 0.12, 1.)
    centre = (.52, .98, .89, 1.)
    for dx, dy in ((1, 0), (0, 1)):
        along = np.arange(-length, length + 1)
        for offset in (-1, 0, 1):
            xs, ys = x + dx * along + dy * offset, y + dy * along + dx * offset
            valid = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
            colours[ys[valid], xs[valid]] = background
    colours[y, max(0, x-length):x+length+1] = centre
    colours[max(0, y-length):y+length+1, x] = centre


def raster(values, kind="science", candidates=(), target=None):
    # Display samples in their original orientation. Full-resolution measurements stay in FITS.
    stride = max(1, int(np.ceil(max(values.shape) / 1300)))
    values = values[::stride, ::stride]
    if kind == "labels":
        colours = colormaps["turbo"]((values.astype(np.int64) * 137 % 257) / 256)
        colours[values == 0] = (0.032, 0.050, 0.082, 1)
    elif kind == "coverage":
        colours = np.where(values[..., None], [0.28, 0.81, 0.73, 1], [0.07, 0.09, 0.15, 1])
    else:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return None
        if kind == "difference":
            limit = max(float(np.percentile(abs(finite), 99.7)), np.finfo(float).eps)
            scaled = (1 + np.clip(values / limit, -1, 1)) / 2
            palette = RESIDUAL
        else:
            low, high = np.percentile(finite, [6, 99.75])
            span = max(high - low, np.finfo(float).eps)
            scaled = np.arcsinh(np.clip((values - low) / span, 0, 1) * 10) / np.arcsinh(10)
            palette = colormaps["magma" if kind == "ultraviolet" else "gray"]
        colours = palette(scaled)
        colours[~np.isfinite(values)] = (.07, .09, .15, 1)
    if candidates:
        residual_boxes(colours, candidates, stride)
    if target is not None:
        mark_position(colours, target, stride)
    stream = BytesIO()
    imsave(stream, colours, format="png", origin="lower")
    return png_url(stream.getvalue())


def science_preview(entry):
    path = Path(entry["path"])
    try:
        if not path.is_file() or digest(path) != entry["sha256"]:
            return None
        hdu = entry.get("hdu", 0)
        if isinstance(hdu, dict):
            frame = read_frame(path, hdu.get("science", 0), mask_hdu=hdu.get("mask"),
                               ivar_hdu=hdu.get("inverse_variance"), variance_hdu=hdu.get("variance"))
        else:
            frame = read_frame(path, hdu)
        return raster(np.where(frame.valid, frame.science, np.nan))
    except (OSError, ValueError, KeyError, IndexError):
        return None


def layer(title, group, image, reading, *, summary=None, legend=(), table=0, transient=None):
    return dict(title=title, group=group, image=image, reading=reading, summary=summary,
                legend=legend, table=table, transient=transient)


HOST_LEGEND = (("cyan", "Detected footprint"), ("white", "Target position"), ("lime", "Provisional host"))


def table(folder, filename, title):
    path = folder / filename
    return dict(title=title, filename=filename, rows=records(path))


def comparison_table(comparison):
    rows = []
    for name, key in (("Hermae v2", "v2_provisional_host"), ("v1 DLR", "dlr"), ("v1 measured FITS", "measured_fits")):
        entry = comparison.get(key)
        if entry:
            row = dict(method=name, **{field: entry.get(field) for field in
                       ("source_id", "ra_deg", "dec_deg", "separation_from_v2_arcsec")})
            rows.append({field: str(value) if value is not None else "" for field, value in row.items()})
    return dict(title="Method comparison", filename="method-comparison.csv", rows=rows)


def known(folder, run):
    path = folder / "comparison.json"
    comparison = json_file(path) if "comparison.json" in run["files"] and path.is_file() else None
    legend = HOST_LEGEND + ((("amber", "Published counterpart"),) if comparison and
                            comparison.get("published_position_deg") else ())
    layers = [layer("Host association", "Overlay", saved_png(folder, run),
                    "The full field shows the source footprints. The detail connects the supernova position with the highest-ranked host candidate.",
                    summary=run, legend=legend)]
    view = science_preview(run["input"])
    if view:
        layers.append(layer("Original r-band field", "Science pixels", view,
                            "This intensity-stretched view shows the input pixels before drawing footprints and candidate markers.",
                            table=1))
    if "segmentation.fits" in run["files"] and (folder / "segmentation.fits").is_file():
        labels = fits.getdata(folder / "segmentation.fits")
        if labels.ndim == 2:
            layers.append(layer("Segmentation map", "Detection layer", raster(labels, "labels"),
                                "Each coloured region is a labelled footprint. Source IDs in the detection table link the measurements to the original FITS segmentation.",
                                legend=(("cyan", "Detected pixels use source-specific colours"),), table=1))
    tables = [table(folder, "hosts.csv", "Ranked hosts"), table(folder, "detections.csv", "Full-frame detections")]
    if comparison:
        tables.append(comparison_table(comparison))
    return layers, tables, comparison


def pair(folder, run):
    tables = [table(folder, "transients.csv", "Transient candidates")]
    layers = [layer("Image subtraction", "Overlay", saved_png(folder, run),
                    "Read the aligned reference, newer exposure, and difference from left to right. A box marks a positive residual considered for transient review.",
                    legend=(("green", "Retained residual"), ("amber", "Rejected residual")))]
    for key, title, reading in (
        ("reference", "Earlier exposure", "The earlier image shows the field before the transient candidate. The host search measures sources on this reference frame."),
        ("science", "Newer exposure", "The newer observation supplies the incoming signal. This preview is on its native pixel grid before alignment."),
    ):
        image = science_preview(run["inputs"][key])
        if image:
            layers.append(layer(title, "Native science pixels", image, reading))
    if "difference.fits" in run["files"] and (folder / "difference.fits").is_file():
        with fits.open(folder / "difference.fits") as hdus:
            difference = raster(hdus[0].data, "difference", tables[0]["rows"])
            layers.append(layer("Signed difference", "Residual layer", difference,
                                "Warm pixels were brighter in the newer exposure after alignment and scaling. Cool pixels were fainter. A residual can also come from subtraction artifacts.",
                                legend=(("warm", "Positive residual"), ("blue", "Negative residual"),
                                        ("green", "Retained residual box"), ("amber", "Rejected residual box"))))
            if "MASK" in hdus:
                layers.append(layer("Usable overlap", "Coverage layer", raster(hdus["MASK"].data == 0, "coverage"),
                                    "Teal shows where both images provide pixels usable for the subtraction.",
                                    legend=(("green", "Usable pixels"), ("muted", "Excluded pixels"))))
            if "VARIANCE" in hdus:
                view = raster(hdus["VARIANCE"].data)
                if view:
                    layers.append(layer("Difference variance", "Noise layer", view,
                                        "Brighter pixels have a larger estimated variance. The residual search accounts for this noise when measuring detection significance."))
    for candidate in tables[0]["rows"]:
        name = candidate["host_run"]
        if not name:
            continue
        child = (folder / name).resolve()
        if child.parent != folder:
            raise ValueError("A host result must be a direct subdirectory of its pair run.")
        child_summary = json_file(child / "run.json")
        layers.append(layer(f"Transient {candidate['transient_id']} host search", "Host overlay",
                            saved_png(child, child_summary),
                            "The cross marks the transient candidate on the earlier exposure. The highlighted galaxy is the highest-ranked nearby source.",
                            summary=child_summary, legend=HOST_LEGEND, table=len(tables), transient=candidate))
        tables.append(table(child, "hosts.csv", f"Transient {candidate['transient_id']} hosts"))
    return layers, tables, None


def survey(run):
    entry = run["input"]
    path = Path(entry["path"])
    preview = None
    if path.is_file() and digest(path) == entry["sha256"]:
        values = fits.getdata(path, entry.get("science_hdu", 0))
        if values.ndim == 2:
            kind = "ultraviolet" if entry["band"] == "near ultraviolet" else "science"
            preview = raster(values, kind, target=run["sn"]["pixel"])
    layers = [layer(run["view_title"], "Survey image", preview,
                    "The cross marks the published supernova coordinates on the archive pixels.",
                    legend=(("green", "Published supernova position"),))]
    height, width = entry["shape_yx"]
    x, y = run["sn"]["pixel"]
    rows = [("RA", run["sn"]["ra_deg"], "deg"), ("Declination", run["sn"]["dec_deg"], "deg"),
            ("Image width", width, "px"), ("Image height", height, "px"),
            ("Target x", x, "px"), ("Target y", y, "px"),
            ("Finite pixels", 100 * entry["valid_fraction"], "%"),
            ("Passband", entry["band"], "")]
    table_data = dict(title="Survey measurements", filename="survey-measurements.csv", rows=[
        dict(metric=name, value=str(value), unit=unit) for name, value, unit in rows])
    return layers, [table_data], None


def file_link(path, output_parent):
    path = path.resolve()
    try:
        relative = Path(os.path.relpath(path, output_parent)).as_posix()
        return dict(name=path.name, url=quote(relative))
    except ValueError:  # Windows drives can differ.
        return dict(name=path.name, url=path.as_uri())


def artifact_links(folder, names, output_parent):
    result = []
    for name in names:
        path = folder / name
        if path.parent != folder or not path.is_file():
            continue
        result.append(file_link(path, output_parent))
    return result


def run_payload(folder, output_parent):
    summary = json_file(folder / "run.json")
    mode = summary["mode"]
    if mode == "survey_image":
        layers, tables, comparison = survey(summary)
    elif mode in {"known_position", "pair_search"}:
        layers, tables, comparison = (pair if mode == "pair_search" else known)(folder, summary)
    else:
        raise ValueError("Select a top-level known-position, pair-search or survey-image result directory.")
    artifacts = artifact_links(folder, summary["files"], output_parent)
    if summary.get("survey_origin") or mode == "survey_image":
        source = Path(summary["input"]["path"])
        if source.is_file() and digest(source) == summary["input"]["sha256"]:
            artifacts.append(file_link(source, output_parent))
    return dict(name=summary.get("display_name", folder.name), summary=summary, layers=layers, tables=tables,
                comparison=comparison, artifacts=artifacts)


def dashboard(folder, out=None, *, collections=(), previews=()):
    folders = [folder] if isinstance(folder, Path) else list(folder)
    groups = {"Saved runs": folders} if folders else {}
    for name, path in collections:
        if not name.strip():
            raise ValueError("Each collection needs a name.")
        groups.setdefault(name, []).append(Path(path))
    groups = {name: list(dict.fromkeys(path.resolve() for path in paths)) for name, paths in groups.items()}
    folders = list(dict.fromkeys(path for paths in groups.values() for path in paths))
    if not folders or (len(folders) > 1 and out is None):
        raise ValueError("Provide at least one run and specify --out for a combined dashboard.")
    out = out or folders[0] / "dashboard.html"
    if out.suffix.lower() != ".html":
        raise ValueError("A dashboard output must have an .html extension.")
    title = folders[0].name if len(folders) == 1 else "Run gallery"
    runs = [run_payload(path, out.resolve().parent) for path in folders]
    for folder, run in zip(folders, runs):
        if len(folders) > 1 and not run["summary"].get("display_name"):
            run["name"] = f"{folder.parent.name}/{folder.name}"
    images = {}
    for name, path in previews:
        if name not in groups:
            raise ValueError(f"No collection named {name!r} for this preview.")
        if name in images:
            raise ValueError(f"Provide one preview for collection {name!r}.")
        content = Path(path).read_bytes()
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("A collection preview must be a PNG image.")
        images[name] = png_url(content)
    positions = {path: index for index, path in enumerate(folders)}
    data = {"runs": runs, "collections": [dict(name=name, runs=[positions[path] for path in paths],
                                              preview=images.get(name)) for name, paths in groups.items()]}
    serialised = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = files("hermae").joinpath("dashboard.html").read_text(encoding="utf-8")
    before, after = template.replace("@@TITLE@@", escape(title)).split("@@DATA@@")
    out.write_text(before + serialised + after, encoding="utf-8")
    return out
