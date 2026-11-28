"""Project a saved UV localization onto the original GALEX image grid."""

import json
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .development import digest


def position(record):
    return SkyCoord(record["ra_deg"], record["dec_deg"], unit="deg", frame="icrs")


def save_projection(folder, source):
    receipt = folder / "run.json"
    survey = json.loads(receipt.read_text(encoding="utf-8"))
    model = json.loads(source.read_text(encoding="utf-8"))
    if model["mode"] != "uv_search" or model["evidence"] != "archival_epoch_pair":
        raise ValueError("A survey projection needs a saved archival UV localization.")
    published, estimate = position(survey["sn"]), position(model["top_candidate"])
    offset = float(published.separation(estimate).arcsec)
    entry = survey["input"]
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"GALEX FITS changed after collection: {path}")
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[entry["science_hdu"]]
        wcs = WCS(hdu.header, fobj=hdus).celestial
        values = np.array(hdu.data, dtype=float)
    pixels = [tuple(float(v) for v in wcs.world_to_pixel(p)) for p in (published, estimate)]
    height, width = values.shape
    for x, y in pixels:
        if not (-.5 <= x < width - .5 and -.5 <= y < height - .5
                and np.isfinite(values[round(y), round(x)])):
            raise ValueError("The published or estimated position falls outside usable GALEX coverage.")

    image = folder / "uv-model-estimate.png"
    render_projection(image, values, pixels, model["band"].upper(), offset)
    survey["projection"] = dict(file=image.name, sha256=digest(image), source_run=str(source.resolve()),
                                source_sha256=digest(source), source_filter=model["band"].upper(),
                                epoch_mjd=model["epoch_mjd"], offset_arcsec=offset,
                                estimate=model["top_candidate"], estimate_pixel=list(pixels[1]),
                                published_pixel=list(pixels[0]))
    survey["files"] = list(dict.fromkeys([*survey["files"], image.name]))
    receipt.write_text(json.dumps(survey, indent=2, allow_nan=False), encoding="utf-8")


def render_projection(out, values, pixels, passband, offset):
    low, high = np.nanpercentile(values, (6, 99.75))
    scaled = np.arcsinh(10 * np.clip((values - low) / max(high - low, 1e-12), 0, 1)) / np.arcsinh(10)
    fig = Figure(figsize=(10, 5.4), dpi=150, facecolor="#111512")
    FigureCanvasAgg(fig)
    full, detail = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.065, right=.978, bottom=.15, top=.86, wspace=.20)
    for ax, title in ((full, "GALEX NUV / FULL FIELD"), (detail, "GALEX NUV / PIXEL DETAIL")):
        ax.imshow(scaled, cmap="magma", origin="lower", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, color="#e5eae4", loc="left", fontsize=10, fontfamily="monospace", pad=9)
        ax.tick_params(colors="#a7b2a8", labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_color("#56645a")
    (px, py), (x, y) = pixels
    full.scatter(x, y, marker="s", s=545, facecolors="none", edgecolors="#9ae3b6", linewidths=1.4)
    detail.set(xlim=(px-3.5, px+3.5), ylim=(py-3.5, py+3.5), xlabel="FITS X (px)", ylabel="FITS Y (px)")
    detail.xaxis.label.set_color("#a7b2a8")
    detail.yaxis.label.set_color("#a7b2a8")
    detail.plot([px, x], [py, y], color="#e5eae4", alpha=.7, linewidth=1, zorder=2)
    published = detail.scatter(px, py, marker="+", s=210, color="#ffc778", linewidths=1.6, zorder=4)
    estimated = detail.scatter(x, y, marker="D", s=95, facecolors="none", edgecolors="#9ae3b6",
                               linewidths=1.7, zorder=3)
    fig.suptitle(f"M101 / SN 2023ixf   ·   SWIFT {passband} ESTIMATE ON GALEX NUV",
                 x=.065, y=.95, ha="left", color="#e5eae4", fontsize=14, fontfamily="monospace")
    fig.legend([estimated, published], [f"Model estimate · {offset:.2f}″ from published", "Published position"],
               loc="lower center", bbox_to_anchor=(.5, .025), ncols=2, frameon=False,
               labelcolor="#e5eae4", fontsize=9)
    fig.savefig(out)
