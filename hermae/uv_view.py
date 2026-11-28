"""Render UV localization layers in FITS pixel orientation."""

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw


def save_uv_views(out, reference, science, residual, scores, peaks, truth, threshold):
    valid = np.isfinite(scores)
    limits = np.percentile(np.concatenate((reference[valid], science[valid])), [5, 99.6])
    low, high = limits

    def intensity(values):
        return np.arcsinh(12 * np.clip((values - low) / max(high - low, 1e-10), 0, 1)) / np.arcsinh(12)

    files = []
    for name, values, cmap, marks in (
        ("uv-reference.png", intensity(reference), "bone", False),
        ("uv-science.png", intensity(science), "bone", True),
        ("uv-residual.png", np.clip(residual / 12 + .5, 0, 1), "RdBu_r", True),
        ("uv-score.png", np.nan_to_num(scores), "magma", True),
    ):
        rgb = colormaps[cmap](np.nan_to_num(values[::-1]), bytes=True)[..., :3]
        if name in {"uv-score.png", "uv-residual.png"}:
            rgb[~valid[::-1]] = (17, 21, 20)
        height, width = values.shape
        scale = max(1, min(8, 768 // width))
        image = Image.fromarray(rgb).resize((width * scale, height * scale), Image.Resampling.NEAREST)
        if marks:
            draw = ImageDraw.Draw(image)
            for rank, (x, y, _) in enumerate(peaks[:12], 1):
                px, py = (x + .5) * scale, (height - y - .5) * scale
                draw.rectangle((px - 4 * scale, py - 4 * scale, px + 4 * scale, py + 4 * scale),
                               outline="#9ae3b6", width=2)
                draw.text((px + 4 * scale + 3, py - 4 * scale), str(rank), fill="#9ae3b6",
                          stroke_width=1, stroke_fill="#0b0d0d")
            if truth is not None:
                x, y = truth
                px, py = (x + .5) * scale, (height - y - .5) * scale
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    draw.line((px + 2*scale*dx, py + 2*scale*dy, px + 7*scale*dx, py + 7*scale*dy),
                              fill="#ffd388", width=2)
        image.save(out / name)
        files.append(name)
    return files
