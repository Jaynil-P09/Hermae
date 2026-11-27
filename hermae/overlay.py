"""Draw the detected footprints on the original FITS frame."""

from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def render(frame, labels, x, y, hosts, comparison, path, title="Classical source detection"):
    canvas = "#0e1722"
    ink = "#eaf4fb"
    valid_data = frame.science[frame.valid]
    low, high = np.percentile(valid_data, [8, 99.7]) if valid_data.size else (0., 1.)
    norm = ImageNormalize(vmin=low, vmax=max(high, low + 1e-12), stretch=AsinhStretch(.12))
    height, width = labels.shape
    # A thin inner boundary keeps touching deblended masks visible.
    edge = (labels > 0) & (ndimage.minimum_filter(labels, 3) != ndimage.maximum_filter(labels, 3))
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    rgba[edge] = [.32, .74, .84, .68]
    if hosts:
        rgba[edge & (labels == hosts[0]["source_id"])] = [.8, 1., .47, 1.]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.9), layout="constrained", facecolor=canvas)
    try:
        inside = np.isfinite([x, y]).all() and -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
        published = None
        if comparison and comparison["published_position_deg"]:
            ref = SkyCoord(*comparison["published_position_deg"], unit="deg")
            px, py = frame.wcs.world_to_pixel(ref)
            if -0.5 <= px < width - 0.5 and -0.5 <= py < height - 0.5:
                published = px, py
        for ax in axes:
            ax.set_facecolor("#212d3a")
            ax.imshow(np.where(frame.valid, frame.science, np.nan), origin="lower", cmap="gray", norm=norm)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
            if inside:
                ax.scatter(x, y, marker="+", s=115, color=canvas, linewidths=3.4, zorder=6)
                ax.scatter(x, y, marker="+", s=115, color="#ffffff", linewidths=1.5,
                           zorder=7, label="Supernova")
            if hosts:
                host = hosts[0]
                ax.scatter(host["x_px"], host["y_px"], marker="D", s=110, facecolors="none",
                           edgecolors="#cbff78", linewidths=1.5, zorder=5, label="Provisional host")
            if published is not None:
                ax.scatter(*published, marker="*", s=200, facecolors="none", edgecolors="#ffc876",
                           linewidths=1.4, zorder=4, label="Published counterpart")
            ax.set(xlim=(-0.5, width - 0.5), ylim=(-0.5, height - 0.5))
            ax.tick_params(colors=ink, labelsize=9)
            ax.set_xlabel("FITS x (pixel)", color=ink)
        axes[0].set_ylabel("FITS y (pixel)", color=ink)
        axes[0].set_title("Full frame · detected footprints", color=ink, fontsize=12)
        if inside:
            hx, hy = (hosts[0]["x_px"], hosts[0]["y_px"]) if hosts else (x, y)
            points = np.array([(x, y), (hx, hy), *([published] if published is not None else [])])
            cx, cy = (points.min(axis=0) + points.max(axis=0)) / 2
            pad = max(19., float(np.ptp(points, axis=0).max()) / 2 + 10)
            axes[1].set(xlim=(cx - pad, cx + pad), ylim=(cy - pad, cy + pad))
            axes[1].set_title("Detail · supernova and host ranking", color=ink, fontsize=12)
        else:
            axes[1].set_title("SN outside usable image bounds", color=ink, fontsize=12)
        if axes[1].get_legend_handles_labels()[0]:
            legend = axes[1].legend(loc="lower left", facecolor=canvas, edgecolor="#465563", fontsize=9)
            for label in legend.get_texts():
                label.set_color(ink)
        fig.suptitle(f"Hermae v2 · {title}", fontsize=20, color=ink, fontweight="bold")
        fig.savefig(path, dpi=150, facecolor=canvas)
    finally:
        plt.close(fig)
