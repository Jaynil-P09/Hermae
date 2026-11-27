"""Render the frozen validation rankings and their apertures on real FITS fields."""

import argparse
from hashlib import file_digest
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from .imaging import (CORE_ARCSEC, ENVELOPE_ARCSEC, SN_ANNULUS_ARCSEC, SN_MASK_ARCSEC,
                      bridge_centres, read_image)

BACKGROUND = "#101820"
TEXT = "#eff6fb"
COLORS = ("#55d8de", "#fa97c7", "#b2ed88")
REFERENCE = "#ffd36c"
METHODS = (
    ("Tractor geometry", "dlr_rank", "dlr", "DLR"),
    ("Catalogue model", "catalogue_rank", "catalogue_score", "score"),
    ("Catalogue + FITS model", "image_rank", "image_score", "score"),
)
CNN_METHODS = (
    METHODS[0],
    ("Matched catalogue control", "catalogue_control_rank", "catalogue_control_score", "score"),
    ("Catalogue + CNN", "cnn_fusion_rank", "cnn_fusion_score", "score"),
)
KEYS = ["event_id", "source_id"]


def display_image(cache, event, audit):
    path = cache / "image_ablation" / audit.cache_file
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != audit.sha256:
            raise ValueError(f"FITS bytes differ from the image audit for {event.event_id}.")
    scale = audit.pixel_scale_arcsec
    for band in "rgz":
        if band in audit.requested_bands:
            image, valid, wcs = read_image(path, event, scale, band=band)
            if valid.any():
                return np.where(valid, image, np.nan), wcs, band
    return np.full_like(image, np.nan), wcs, None


def ellipse(axis, wcs, row, color):
    if not (np.isfinite(row.shape_r) and np.isfinite(row.axis_ratio)):
        return
    angle = np.linspace(0, 2 * np.pi, 121)
    major, minor = row.shape_r * np.cos(angle), row.shape_r * row.axis_ratio * np.sin(angle)
    pa = (row.major_pa_deg if pd.notna(row.major_pa_deg) else 0.) * u.deg
    sky = SkyCoord(row.ra, row.dec, unit="deg").directional_offset_by(
        pa + np.arctan2(minor, major) * u.rad, np.hypot(major, minor) * u.arcsec)
    axis.plot(*wcs.world_to_pixel(sky), color=color, linewidth=1.4, zorder=5)


def draw_background(axis, image, sn, limits, scale, band):
    x0, x1, y0, y1 = limits
    sample = image[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1]
    finite = sample[np.isfinite(sample)]
    axis.set_facecolor("#202c38")
    if finite.size:
        low, high = np.percentile(finite, [8, 99.5])
        axis.imshow(image, origin="lower", cmap="gray", interpolation="nearest",
                    norm=ImageNormalize(vmin=low, vmax=max(high, low + 1e-9),
                                        stretch=AsinhStretch(0.10)))
    else:
        axis.text(.5, .52, "No usable pixels in this view", transform=axis.transAxes,
                  color=TEXT, ha="center", fontsize=10)
    axis.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
    axis.tick_params(colors=TEXT, labelsize=8)
    for index, dim in enumerate((axis.xaxis, axis.yaxis)):
        dim.set_major_locator(MaxNLocator(5))
        dim.set_major_formatter(FuncFormatter(lambda v, _, i=index: f"{(v - sn[i]) * scale:+.0f}″"))
    axis.text(.97, .035, f"{band} band" if band else "No image", transform=axis.transAxes,
              color=TEXT, ha="right", fontsize=8, bbox=dict(facecolor=BACKGROUND, edgecolor="none"))


def limits_around(points, scale, margin, minimum):
    bounds = np.asarray(points, dtype=float)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    half = max(minimum, float((high - low).max()) * scale / 2 + margin) / scale
    cx, cy = (low + high) / 2
    return cx - half, cx + half, cy - half, cy + half


def num(value):
    return f"{value:.2f}" if pd.notna(value) else "missing"


def render(event, audit, evaluation, field, cache, output, *, cnn=False):
    image, wcs, band = display_image(cache, event, audit)
    scale = audit.pixel_scale_arcsec
    sn = wcs.world_to_pixel(SkyCoord(event.ra, event.dec, unit="deg"))
    field = field.copy()
    field["x"], field["y"] = wcs.world_to_pixel(
        SkyCoord(field.ra.to_numpy(dtype=float), field.dec.to_numpy(dtype=float), unit="deg"))
    ref = None
    if pd.notna(evaluation.host_ra) and pd.notna(evaluation.host_dec):
        xy = wcs.world_to_pixel(SkyCoord(evaluation.host_ra, evaluation.host_dec, unit="deg"))
        if all(-0.5 <= v < image.shape[0] - 0.5 for v in xy):
            ref = xy
    methods = CNN_METHODS if cnn else METHODS
    picks = [field.nsmallest(3, rank) for _, rank, _, _ in methods]
    overview = limits_around([sn, *(p for pick in picks for p in pick[["x", "y"]].to_numpy()),
                              *([ref] if ref is not None else [])], scale, margin=5, minimum=10)
    figure, grid = plt.subplots(2, 2, figsize=(10, 10.9), facecolor=BACKGROUND)
    figure.subplots_adjust(left=.075, right=.97, top=.885, bottom=.14, wspace=.19, hspace=.35)
    try:
        for axis, (name, rank, score, label), color, pick in zip(grid.flat, methods, COLORS, picks):
            draw_background(axis, image, sn, overview, scale, band)
            axis.set_title(name, color=color, fontsize=13, fontweight="bold")
            axis.scatter(field.x, field.y, s=9, facecolors="none", edgecolors="#a5b9c6",
                         linewidths=.5, alpha=.7, zorder=3)
            axis.scatter(*sn, marker="+", s=75, color=TEXT, linewidths=1.45, zorder=8)
            if ref is not None:
                axis.scatter(*ref, marker="*", s=170, facecolors="none", edgecolors=REFERENCE,
                             linewidths=1.2, zorder=6)
            rows = []
            for index, row in enumerate(pick.itertuples()):
                n = getattr(row, rank)
                if pd.isna(n):
                    continue
                axis.scatter(row.x, row.y, s=78 if n == 1 else 42, marker="s",
                             facecolors="none", edgecolors=color, linewidths=1.35, zorder=7)
                axis.annotate(str(int(n)), (row.x, row.y), xytext=(7, 6 if index % 2 == 0 else -12),
                              textcoords="offset points", fontsize=9, color=color, fontweight="bold",
                              bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=.2), zorder=9)
                rows.append(f"#{int(n)}  {label} {num(getattr(row, score))}  ·  {row.separation_arcsec:.1f}″")
                if rank == "dlr_rank" and n == 1:
                    ellipse(axis, wcs, row, color)
                    axis.plot([sn[0], row.x], [sn[1], row.y], "--", color=color, lw=.8, zorder=4)
            axis.text(0, -.10, "    ".join(rows[:2]) + ("\n" + rows[2] if len(rows) > 2 else ""),
                      transform=axis.transAxes, va="top", color=TEXT, fontsize=8.5)

        axis = grid[1, 1]
        title = "CNN input stamps for fusion #1" if cnn else "FITS measurements for image-model #1"
        axis.set_title(title, color=COLORS[2], fontsize=12,
                       fontweight="bold")
        if band != "r" or picks[2].empty:
            draw_background(axis, image, sn, overview, scale, band)
            reason = "No candidates to measure"
            if band != "r":
                detail = "The CNN score term is zero" if cnn else "The model imputed the image features"
                reason = f"No usable r-band input\n{detail}"
            axis.text(.5, .83, reason, transform=axis.transAxes, ha="center", color=TEXT,
                      bbox=dict(facecolor=BACKGROUND, edgecolor="none"))
        elif cnn:
            from .cnn import STAMP, stamp_centres

            row = picks[2].iloc[0]
            centres = np.rint(stamp_centres(np.array([[row.x, row.y]]), np.array(sn))[0])
            draw_background(axis, image, sn, limits_around(centres, scale, 5, 7), scale, band)
            for offset, ((x, y), name, color) in enumerate(
                zip(centres, ("Candidate", "SN", "Midpoint"), (COLORS[2], REFERENCE, COLORS[0]))
            ):
                axis.add_patch(Rectangle((x - STAMP / 2, y - STAMP / 2), STAMP, STAMP,
                                         fill=False, edgecolor=color, linewidth=1.5, zorder=5))
                axis.annotate(name, (x, y), xytext=(6, (offset - 1) * 13), textcoords="offset points",
                              color=color, fontsize=8, zorder=8,
                              bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=1))
            axis.add_patch(Circle(sn, SN_MASK_ARCSEC / scale, fill=False, edgecolor=TEXT,
                                  linestyle="--", linewidth=1.25, zorder=5))
            term = row.cnn_fusion_image_term
            note = (f"{STAMP} × {STAMP} pixels per stamp" if row.cnn_has_pixels
                    else "Image term suppressed by the coverage threshold")
            axis.text(0, -.10,
                      f"Catalogue term {row.cnn_fusion_score - term:.2f}  ·  CNN term {term:+.2f}\n{note}",
                      transform=axis.transAxes, va="top", color=TEXT, fontsize=8.7)
        else:
            row = picks[2].iloc[0]
            centre = row.x, row.y
            midpoint, control_a, control_b = bridge_centres(*centre, *sn)
            draw_background(axis, image, sn,
                            limits_around([sn, centre, midpoint, control_a, control_b], scale, 6, 8),
                            scale, band)
            for xy, radii, color, style in (
                (centre, (CORE_ARCSEC, *ENVELOPE_ARCSEC), COLORS[2], "-"),
                (sn, (SN_MASK_ARCSEC,), TEXT, "--"),
                (sn, SN_ANNULUS_ARCSEC, REFERENCE, "-"),
                (midpoint, (CORE_ARCSEC,), COLORS[0], "-"),
                (control_a, (CORE_ARCSEC,), COLORS[0], "--"),
                (control_b, (CORE_ARCSEC,), COLORS[0], "--"),
            ):
                for size in radii:
                    axis.add_patch(Circle(xy, size / scale, fill=False, edgecolor=color, linestyle=style,
                                          linewidth=1.25, zorder=5))
            axis.scatter(*sn, marker="+", s=55, color=TEXT, linewidths=1.3, zorder=8)
            axis.text(0, -.10,
                      f"Core {num(row.image_core)}  ·  Envelope {num(row.image_envelope)}\n"
                      f"SN / core {num(row.image_sn_to_core)}  ·  Bridge contrast {num(row.image_bridge_contrast)}",
                      transform=axis.transAxes, va="top", color=TEXT, fontsize=8.7)
        figure.suptitle(event.event_id, color=TEXT, fontsize=23, fontweight="bold", y=.972)
        match = evaluation.reference_status.replace("_", " ")
        figure.text(.5, .927, f"Legacy Surveys DR10 south · {len(field)} candidates · Reference {match}",
                    ha="center", color=TEXT, fontsize=11)
        handles = [
            Line2D([], [], marker="+", color=TEXT, linestyle="none", label="SN / dashed mask"),
        ]
        if cnn:
            entries = ((REFERENCE, "Published position / SN stamp"),
                       (COLORS[2], "Candidate stamp"), (COLORS[0], "Midpoint stamp"))
        else:
            entries = ((REFERENCE, "Published position / SN annulus"),
                       (COLORS[2], "Candidate core and envelope"), (COLORS[0], "Bridge / dashed controls"))
        for index, (color, name) in enumerate(entries):
            handles.append(Line2D([], [], color=color, label=name, marker="*" if index == 0 else None,
                                  markerfacecolor="none", linestyle="none" if index == 0 else "-"))
        legend = figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
                               ncols=2, frameon=False, fontsize=9)
        for label in legend.get_texts():
            label.set_color(TEXT)
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--images", type=Path, default=Path("results/benchmark/images"))
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cnn", action="store_true", help="show CNN fusion ranks and input stamps")
    parser.add_argument("--cnn-results", type=Path, default=Path("results/benchmark/cnn"))
    parser.add_argument("--event", action="append", help="repeat to select validation event IDs")
    options = parser.parse_args()
    if options.output is None:
        options.output = options.cnn_results / "overlays" if options.cnn else Path("results/benchmark/overlays")
    data = options.cnn_results if options.cnn else options.images
    prefix = "cnn" if options.cnn else "image"
    comparisons = pd.read_csv(data / f"{prefix}_validation.csv").set_index("event_id")
    events = sorted(set(options.event or comparisons.index))
    if missing := set(events) - set(comparisons.index):
        parser.error(f"Events outside the scored validation split: {', '.join(sorted(missing))}")
    transients = pd.read_csv(options.run / "events.csv", usecols=["event_id", "ra", "dec"]).set_index("event_id")
    audit = pd.read_csv(options.images / "image_inputs.csv").set_index("event_id")
    rankings = pd.read_csv(data / f"{prefix}_rankings.csv")
    sources = pd.read_csv(options.run / "candidates.csv", usecols=[
        *KEYS, "ra", "dec", "dlr", "dlr_rank", "shape_r", "axis_ratio", "major_pa_deg"])
    selected = rankings.loc[rankings.event_id.isin(events)]
    field = selected.merge(sources, on=KEYS, validate="one_to_one")
    if not options.cnn:
        field = field.merge(pd.read_csv(options.images / "image_features.csv"), on=KEYS, validate="one_to_one")
    if len(field) != len(selected):
        parser.error("Some ranked candidates are missing catalogue rows or FITS features.")
    options.output.mkdir(parents=True, exist_ok=True)
    gallery = ["# CNN analysis overlays" if options.cnn else "# FITS analysis overlays", "",
               "Numbers mark candidate ranks within each model. Model scores are uncalibrated.",
               "Axes show cutout offsets from the SN. DLR is separation divided by the directional half-light radius."]
    if options.cnn:
        gallery += ["The boxes show the CNN input stamps. Scores split into catalogue and CNN terms. "
                    "The g- or z-band backgrounds provide context for fields with missing r-band pixels.", ""]
    else:
        gallery += ["Aperture values are transformed, field-normalized measurements. A missing value indicates insufficient coverage or an excluded aperture.",
                    "When r-band pixels are unavailable, the image-model panel shows its scores with imputed image features. The g- or z-band background provides visual context.", ""]
    reference_a, reference_b = (("catalogue_control_reference_rank", "cnn_fusion_reference_rank") if options.cnn
                                else ("catalogue_reference_rank", "image_reference_rank"))
    def reference_changed(event_id):
        row = comparisons.loc[event_id]
        return pd.notna(row[reference_a]) and pd.notna(row[reference_b]) and row[reference_a] != row[reference_b]

    for event_id in sorted(events, key=lambda name: (not reference_changed(name), name)):
        event = transients.loc[[event_id]].reset_index().iloc[0]
        evaluation = comparisons.loc[event_id]
        render(event, audit.loc[event_id], evaluation,
               field.loc[field.event_id.eq(event_id)], options.cache, options.output / f"{event_id}.png",
               cnn=options.cnn)
        note = ""
        if reference_changed(event_id):
            note = (f" · Published counterpart rank {int(evaluation[reference_a])} "
                    f"→ {int(evaluation[reference_b])} with {'CNN' if options.cnn else 'FITS'}")
        gallery.append(f"- [{event_id}]({event_id}.png){note}")
    (options.output / "gallery.md").write_text("\n".join([*gallery, ""]), encoding="utf-8")
    print(f"Rendered {len(events)} FITS overlays in {options.output}.")


if __name__ == "__main__":
    main()
