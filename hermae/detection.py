"""Background estimation, tiled segmentation, and footprint measurements."""

from dataclasses import dataclass

from astropy.stats import SigmaClip, mad_std
import numpy as np
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import deblend_sources, detect_sources
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DetectorConfig:
    tile: int = 512
    overlap: int = 64
    threshold_sigma: float = 3.
    min_pixels: int = 7
    background_box: int = 64
    deblend_contrast: float = 0.005


@dataclass
class Patch:
    tile: int
    pixels: np.ndarray
    centre: tuple[float, float]


def background(frame, box):
    sample = frame.science[frame.valid]
    if not sample.size:
        return np.zeros_like(frame.science), np.zeros_like(frame.science)
    # A global robust fallback also supports small frames and sparse footprints.
    median, rms = np.median(sample), mad_std(sample)
    if not np.isfinite(rms) or not rms > 0:
        rms = max(np.finfo(float).eps * max(1., float(np.max(np.abs(sample)))), 1e-12)
    shape = tuple(min(box, size) for size in frame.science.shape)
    if min(shape) < 8 or min(frame.science.shape) < 2 * box:
        return np.full_like(frame.science, median), np.full_like(frame.science, rms)
    try:
        estimated = Background2D(frame.science, shape, mask=~frame.valid, sigma_clip=SigmaClip(sigma=3.),
                                 bkg_estimator=MedianBackground())
    except ValueError:
        return np.full_like(frame.science, median), np.full_like(frame.science, rms)
    noise = estimated.background_rms
    return estimated.background, np.where(np.isfinite(noise) & (noise > 0), noise, rms)


def tile_starts(length, tile, overlap):
    starts = [0]
    while starts[-1] + tile < length:
        starts.append(starts[-1] + tile - overlap)
    return starts


def weighted_centre(pixels, width, signal):
    y, x = np.divmod(pixels, width)
    weight = np.maximum(signal[y, x], 0)
    if not weight.sum() > 0:
        weight = np.ones(len(x))
    return float(np.average(x, weights=weight)), float(np.average(y, weights=weight))


def find_patches(frame, config, signal, noise):
    sigma = 1.0
    weight = ndimage.gaussian_filter(frame.valid.astype(float), sigma)
    smoothed = np.divide(ndimage.gaussian_filter(signal, sigma), weight,
                         out=np.zeros_like(signal), where=weight > 0)
    height, width = signal.shape
    patches = []
    tile_id = 0
    for y in tile_starts(height, config.tile, config.overlap):
        for x in tile_starts(width, config.tile, config.overlap):
            region = np.s_[y:y + config.tile, x:x + config.tile]
            valid = frame.valid[region]
            tile_id += 1
            data = smoothed[region]
            threshold = noise[region] * config.threshold_sigma
            if not np.any(valid & (data > threshold)):
                continue
            seg = detect_sources(data, threshold, config.min_pixels, mask=~valid)
            if seg is None:
                continue
            seg = deblend_sources(data, seg, config.min_pixels, n_levels=32,
                                  contrast=config.deblend_contrast, progress_bar=False)
            for segment in seg.segments:
                sy, sx = segment.slices
                yy, xx = np.nonzero(seg.data[sy, sx] == segment.label)
                pixels = (yy + sy.start + y) * width + xx + sx.start + x
                centre = weighted_centre(pixels, width, signal)
                patches.append(Patch(tile_id, pixels, centre))
    return patches


def merge_patches(patches, shape):
    """Join compatible cross-tile footprints without collapsing same-tile neighbours."""
    height, width = shape
    labels = np.zeros(shape, dtype=np.int32)
    if not patches:
        return labels
    size = np.array([len(p.pixels) for p in patches])
    radii = np.maximum(3., 0.6 * np.sqrt(size / np.pi))
    coords = np.array([patch.centre for patch in patches])
    edges = []
    for a, b in cKDTree(coords).query_pairs(float(radii.max())):
        if (patches[a].tile == patches[b].tile or
                np.linalg.norm(coords[a] - coords[b]) > max(radii[a], radii[b])):
            continue
        common = len(np.intersect1d(patches[a].pixels, patches[b].pixels, assume_unique=True))
        overlap = common / min(size[a], size[b])
        if overlap >= 0.55:
            edges.append((-overlap, a, b))
    parents = list(range(len(patches)))
    tiles = [{p.tile} for p in patches]

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for _, a, b in sorted(edges):
        a, b = root(a), root(b)
        if a != b and tiles[a].isdisjoint(tiles[b]):
            parents[b] = a
            tiles[a].update(tiles[b])
    groups = {}
    for index, patch in enumerate(patches):
        groups.setdefault(root(index), []).append(patch)
    best = np.full(shape, np.inf, dtype=np.float32)
    for number, group in enumerate(groups.values(), start=1):
        pixels = np.unique(np.concatenate([p.pixels for p in group]))
        yy, xx = np.divmod(pixels, width)
        x, y = np.mean([p.centre for p in group], axis=0)
        distance = ((xx - x)**2 + (yy - y)**2).astype(np.float32)
        update = distance < best[yy, xx]
        labels[yy[update], xx[update]] = number
        best[yy[update], xx[update]] = distance[update]
    # Renumber surviving footprints after resolving conflicting boundary pixels.
    _, inverse = np.unique(labels, return_inverse=True)
    if labels.min() != 0:
        inverse += 1
    return inverse.reshape(height, width).astype(np.int32)


def detect(frame, config):
    sky, noise = background(frame, config.background_box)
    signal = np.where(frame.valid, frame.science - sky, 0.)
    variance = np.square(noise)
    if frame.variance is not None:
        variance = np.where(frame.valid, np.maximum(variance, frame.variance), variance)
    patches = find_patches(frame, config, signal, np.sqrt(variance))
    return merge_patches(patches, frame.science.shape), signal, variance


def measurements(frame, labels, signal, variance):
    height, width = labels.shape
    mask_edge = ndimage.binary_dilation(~frame.valid)
    rows = []
    for source_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        sy, sx = bounds
        dy, dx = np.nonzero(labels[bounds] == source_id)
        x, y = dx + sx.start, dy + sy.start
        weights = np.maximum(signal[y, x], 0)
        if not weights.sum() > 0:
            weights = np.ones(len(x))
        cx, cy = np.average(x, weights=weights), np.average(y, weights=weights)
        offsets = np.vstack((x - cx, y - cy))
        cov = (offsets * weights) @ offsets.T / weights.sum()
        vals, vec = np.linalg.eigh(cov)
        minor, major = np.sqrt(np.maximum(vals, 0))
        direction = vec[:, 1]
        sky = frame.wcs.pixel_to_world(cx, cy).icrs
        coords = frame.wcs.pixel_to_world(cx + direction[0], cy + direction[1]).icrs
        edge = sx.start == 0 or sy.start == 0 or sx.stop == width or sy.stop == height
        coverage_edge = bool(mask_edge[y, x].any())
        resolved = bool(minor >= 0.8)
        reliable = bool(resolved and not edge and not coverage_edge)
        rows.append(dict(source_id=source_id, x_px=float(cx), y_px=float(cy),
                         ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg),
                         footprint_pixels=len(x), xmin=sx.start, xmax_exclusive=sx.stop,
                         ymin=sy.start, ymax_exclusive=sy.stop, flux=float(signal[y, x].sum()),
                         flux_noise=float(np.sqrt(variance[y, x].sum())),
                         major_sigma_px=float(major), minor_sigma_px=float(minor),
                         major_angle_px_rad=float(np.arctan2(direction[1], direction[0]) % np.pi),
                         major_pa_deg=float(sky.position_angle(coords).deg % 180),
                         classification="extended" if resolved else "unresolved",
                         frame_edge=edge, coverage_edge=coverage_edge, reliable_shape=reliable))
    return rows
