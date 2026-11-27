"""Render WCS-based inspection panels from Legacy Surveys image cutouts."""

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .survey import BANDS, PIX_SCALE, SourceError, cutout


def fetch_panel_cutout(cache, event, radius):
    scale = max(PIX_SCALE, 2.25 * radius / 510)
    size = min(512, int(np.ceil(2.25 * radius / scale)) + 1)
    return cutout(cache, event.ra, event.dec, size, pixscale=scale), size


def panel(path, size, directory, event, evaluation, candidates, radius):
    with fits.open(path) as hdus:
        header = hdus[0].header.copy()
        cube = hdus[0].data
        if header.get("BANDS") != BANDS or cube.shape != (3, size, size):
            raise SourceError(f"Unexpected image bands or size in {path}")
        usable = [band for band, plane in zip(BANDS, cube) if np.isfinite(plane).any()
                  and np.any(np.nan_to_num(plane) != 0)]
        if not usable:
            raise SourceError(f"No usable optical plane in {path}")
        band = next((name for name in "rgz" if name in usable))
        image = cube[BANDS.index(band)].copy()

    wcs = WCS(header, naxis=2)
    figure = plt.figure(figsize=(7.4, 7.3), layout="constrained", facecolor="#101820")
    white = "#eef4f9"
    try:
        axis = figure.add_subplot(projection=wcs)
        norm = ImageNormalize(image[np.isfinite(image)], interval=PercentileInterval(99.6),
                              stretch=AsinhStretch(0.09))
        axis.imshow(image, origin="lower", cmap="gray", norm=norm)
        axis.set(xlim=(-0.5, size - 0.5), ylim=(-0.5, size - 0.5))
        for coord, label in zip(axis.coords, ("Right ascension (J2000)", "Declination (J2000)")):
            coord.set_axislabel(label, color=white)
            coord.set_ticklabel(color=white, size=9)
            coord.set_ticks(color=white)
        center = SkyCoord(event.ra, event.dec, unit="deg")
        x, y = wcs.world_to_pixel(center)
        expected_center = (size - 1) / 2
        if np.hypot(x - expected_center, y - expected_center) > 0.1:
            raise SourceError(f"Cutout WCS did not center the requested event: {path}")
        boundary = center.directional_offset_by(np.linspace(0, 360, 361) * u.deg,
                                               radius * u.arcsec)
        bx, by = wcs.world_to_pixel(boundary)
        axis.plot(bx, by, "--", color="#b5c3cf", linewidth=0.7, alpha=0.75,
                  label=f"{radius}″ search")
        shown = candidates.loc[candidates.in_baseline_cone]
        sx, sy = wcs.world_to_pixel(SkyCoord(shown.ra.to_numpy(dtype=float),
                                           shown.dec.to_numpy(dtype=float), unit="deg"))
        axis.scatter(sx, sy, s=12, facecolors="none", edgecolors="#a9c9da",
                     linewidths=0.65, alpha=0.8, label="Tractor galaxy")
        axis.scatter(x, y, s=65, color="white", marker="+", linewidths=1.15, label="Supernova",
                     zorder=7)
        off_frame = []

        def mark(ra, dec, name, color, marker, size):
            px, py = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
            if -0.5 <= px < image.shape[1] - 0.5 and -0.5 <= py < image.shape[0] - 0.5:
                axis.scatter(px, py, s=size, facecolors="none", edgecolors=color, marker=marker,
                             linewidths=1.4, label=name)
            else:
                off_frame.append(name)

        if pd.notna(event.host_ra) and pd.notna(event.host_dec):
            mark(event.host_ra, event.host_dec, "Published host", "#ffd16b", "o", 300)
        for method, name, color, marker, area in (
            ("nearest", "Nearest", "#65e0ff", "s", 74),
            ("dlr", "DLR", "#ff8aba", "D", 125),
        ):
            source = shown.loc[shown.source_id == evaluation[f"{method}_pick"]]
            if not source.empty:
                row = source.iloc[0]
                mark(row.ra, row.dec, name, color, marker, area)
        figure.suptitle(event.event_id, color=white, fontsize=17, fontweight="bold")
        subtitle = (f"Legacy Surveys DR10 south · {band} band · {len(shown)} candidates\n"
                    f"Reference match {evaluation['reference_status'].replace('_', ' ')}"
                    f"  ·  {evaluation['review_status'].replace('_', ' ')}")
        if off_frame:
            subtitle += "\nOutside cutout: " + ", ".join(off_frame)
        axis.set_title(subtitle, color=white, fontsize=9.5)
        handles, labels = axis.get_legend_handles_labels()
        legend = axis.legend(handles, labels, ncols=2, loc="lower left", fontsize=8,
                             facecolor="#101820", edgecolor="#536471", framealpha=0.92)
        for label in legend.get_texts():
            label.set_color(white)
        figure.savefig(directory / f"{event.event_id}.png", dpi=150)
    finally:
        plt.close(figure)
    return path.name
