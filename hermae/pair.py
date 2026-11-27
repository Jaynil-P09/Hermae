"""Align and subtract two r-band exposures, then rank hosts on the reference."""

import json

from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs.utils import proj_plane_pixel_area
from matplotlib.figure import Figure
from matplotlib import colormaps
import numpy as np
from photutils.segmentation import detect_sources, deblend_sources
from scipy import ndimage

from .detection import background, detect, measurements
from .development import digest
from .fits_input import read_frame, sn_position
from .host import RANK_COLUMNS, rank_hosts, write_table
from .overlay import render

COLUMNS = ("transient_id", "ra_deg", "dec_deg", "x_px", "y_px", "peak_sigma", "footprint_pixels",
           "classifier_score", "survived", "host_status", "host_run")


def frames(reference_path, science_path, reference_hdu=0, science_hdu=0):
    reference = read_frame(reference_path, reference_hdu)
    science = read_frame(science_path, science_hdu)
    return reference, science


def context(path, hdu, supplied):
    header = fits.getheader(path, hdu)
    seeing = supplied if supplied is not None else header.get("SEEING")
    if seeing is None or not np.isfinite(seeing) or seeing <= 0:
        raise ValueError("Supply a positive PSF FWHM in arcseconds for both exposures.")
    return float(seeing), header.get("MJD-OBS")


def psf_match(value, variance, valid, sigma):
    if sigma <= .05:
        return value, variance, valid
    radius = max(1, int(np.ceil(4 * sigma)))
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (axis / sigma)**2)
    kernel /= kernel.sum()

    def separable(image, weights):
        return ndimage.convolve1d(ndimage.convolve1d(image, weights, axis=0, mode="constant"),
                                 weights, axis=1, mode="constant")

    coverage = separable(valid.astype(float), kernel)
    return (separable(np.where(valid, value, 0.), kernel),
            separable(np.where(valid, variance, 0.), kernel**2), valid & (coverage >= .99))


def photometry(ref, new, coverage):
    usable = np.flatnonzero(coverage)
    if len(usable) > 100_000:
        usable = usable[::int(np.ceil(len(usable) / 100_000))]
    x, y = ref.ravel()[usable], new.ravel()[usable]
    background_rms = max(mad_std(x), np.finfo(float).eps)
    # Fit on bright reference pixels and clip objects that changed between epochs.
    signal = x > np.median(x) + 4 * background_rms
    scale = 1.
    if signal.sum() >= 20:
        use = signal
        for _ in range(4):
            denom = np.sum(x[use]**2)
            if use.sum() < 10 or denom <= 0:
                break
            scale = np.clip(np.sum(x[use] * y[use]) / denom, .05, 20.)
            residual = y - scale * x
            scatter = max(mad_std(residual[use]), np.finfo(float).eps)
            use = signal & (abs(residual - np.median(residual[use])) < 3 * scatter)
    intercept = float(np.median(y - scale * x))
    return float(scale), intercept, int(signal.sum())


def difference(reference, science, ref_seeing, sci_seeing):
    ref_sky, ref_rms = background(reference, 48)
    sci_sky, sci_rms = background(science, 48)
    new = np.zeros(reference.science.shape)
    new_var = np.zeros_like(new)
    coverage = np.zeros(new.shape, dtype=bool)
    ref = np.where(reference.valid, reference.science - ref_sky, 0.)
    ref_var = ref_rms**2 if reference.variance is None else reference.variance
    sci_var = sci_rms**2 if science.variance is None else science.variance
    values = np.where(science.valid, science.science - sci_sky, 0.)
    sci_var = np.where(science.valid, sci_var, 0.)
    weights = science.valid.astype(float)
    # Work in strips to avoid several full-frame WCS intermediate arrays.
    for y in range(0, new.shape[0], 128):
        yy, xx = np.mgrid[y:min(y + 128, new.shape[0]), :new.shape[1]]
        x, ys = science.wcs.world_to_pixel(reference.wcs.pixel_to_world(xx, yy))
        coords = np.stack((ys, x))
        section = np.s_[y:y + 128, :]
        new[section] = ndimage.map_coordinates(values, coords, order=1, mode="constant", cval=0)
        new_var[section] = ndimage.map_coordinates(sci_var, coords, order=1, mode="constant", cval=0)
        coverage[section] = ndimage.map_coordinates(weights, coords, order=1, mode="constant", cval=0) >= .999
    if not (coverage & reference.valid).any():
        return None
    size = float(np.sqrt(proj_plane_pixel_area(reference.wcs)) * 3600)
    ref_blur = np.sqrt(max(0., sci_seeing**2 - ref_seeing**2)) / (2.35482 * size)
    new_blur = np.sqrt(max(0., ref_seeing**2 - sci_seeing**2)) / (2.35482 * size)
    ref, ref_var, ref_valid = psf_match(ref, ref_var, reference.valid, ref_blur)
    new, new_var, new_valid = psf_match(new, new_var, coverage, new_blur)
    coverage = ref_valid & new_valid
    if not coverage.any():
        return None
    scale, bias, samples = photometry(ref, new, coverage)
    aligned = ref * scale + bias
    residual = np.where(coverage, new - aligned, 0.)
    variance = new_var + ref_var * scale**2
    floor = max(mad_std(residual[coverage]), np.finfo(float).eps)
    variance = np.where(coverage, np.maximum(variance, floor**2), 0.)
    calibration = dict(photometric_scale=scale, offset=bias, positive_reference_pixels=samples,
                       gaussian_reference_sigma_px=ref_blur, gaussian_science_sigma_px=new_blur,
                       difference_noise_floor=float(floor))
    return aligned, new, residual, variance, coverage, calibration


def residual_candidates(reference, delta, sigma):
    if delta is None:
        return []
    _, _, residual, variance, valid, _ = delta
    significance = np.divide(residual, np.sqrt(variance), out=np.zeros_like(residual), where=valid)
    smoothed = ndimage.gaussian_filter(significance, .75)
    if not (smoothed[valid] >= sigma).any():
        return []
    segments = detect_sources(smoothed, sigma, 3, mask=~valid)
    if segments is None:
        return []
    segments = deblend_sources(smoothed, segments, 3, n_levels=16, progress_bar=False)
    rows = []
    for obj in segments.segments:
        sy, sx = obj.slices
        y, x = np.nonzero(segments.data[sy, sx] == obj.label)
        y, x = y + sy.start, x + sx.start
        weights = smoothed[y, x]
        cx, cy = np.average(x, weights=weights), np.average(y, weights=weights)
        sky = reference.wcs.pixel_to_world(cx, cy).icrs
        rows.append(dict(ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg),
                         x_px=float(cx), y_px=float(cy), peak_sigma=float(significance[y, x].max()),
                         footprint_pixels=len(x)))
    rows.sort(key=lambda row: (-row["peak_sigma"], row["y_px"], row["x_px"]))
    return [dict(transient_id=i, **row) for i, row in enumerate(rows, 1)]


def difference_overlay(path, delta, candidates):
    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    coverage = delta[4]
    grey = colormaps["gray"].with_extremes(bad="#202832")
    for ax, values, title in zip(axes, delta[:3], ("Reference aligned", "Newer aligned", "Science − reference")):
        median = np.median(values[coverage])
        scale = max(mad_std(values[coverage]), np.finfo(float).eps)
        ax.imshow(np.where(coverage, values, np.nan), origin="lower", cmap=grey,
                  vmin=median - 3 * scale, vmax=median + 8 * scale)
        ax.set_title(title)
        for item in candidates:
            ax.scatter(item["x_px"], item["y_px"], s=34, marker="s", facecolors="none",
                       edgecolors="#61df93" if item["survived"] else "#f2ad64")
            if item["survived"]:
                ax.annotate(item["transient_id"], (item["x_px"], item["y_px"]), color="#61df93")
        ax.set_axis_off()
    fig.savefig(path, dpi=130)


def search(args, config):
    ref_path, sci_path = args.reference.resolve(), args.science.resolve()
    if ref_path == sci_path:
        raise ValueError("Reference and newer science must be different exposures.")
    if any(path.is_relative_to(args.out.resolve()) for path in (ref_path, sci_path)):
        raise ValueError("Store pair outputs outside the directory containing the inputs.")
    reference, science = frames(ref_path, sci_path, args.reference_hdu, args.science_hdu)
    ref_psf, ref_mjd = context(ref_path, args.reference_hdu, args.reference_fwhm)
    sci_psf, sci_mjd = context(sci_path, args.science_hdu, args.science_fwhm)
    if ref_mjd is not None and sci_mjd is not None and float(ref_mjd) >= float(sci_mjd):
        raise ValueError("The reference exposure must be earlier than the science exposure.")
    delta = difference(reference, science, ref_psf, sci_psf)
    rows = residual_candidates(reference, delta, args.transient_sigma)
    classifier = None
    if args.transient_weights:
        from .transient import load_classifier, classify
        model, state, identity = load_classifier(args.transient_weights, args.device)
        if state["baseline_sigma"] != args.transient_sigma:
            raise ValueError("Use the subtraction threshold recorded in the classifier checkpoint.")
        classifier = dict(sha256=identity, threshold=state["threshold"])
        scores = classify(model, delta, rows, args.device) if rows else []
    else:
        scores = [None] * len(rows)
    surviving = []
    for row, score in zip(rows, scores):
        row.update(classifier_score=score, survived=score is None or score >= classifier["threshold"],
                   host_status=None, host_run=None)
        if row["survived"]:
            surviving.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    files = ["transients.csv", "run.json"]
    if delta is not None:
        _, _, residual, variance, coverage, _ = delta
        header = reference.wcs.to_header()
        hdu = fits.HDUList([fits.PrimaryHDU(np.where(coverage, residual, np.nan), header),
                            fits.ImageHDU((~coverage).astype(np.uint8), header, name="MASK"),
                            fits.ImageHDU(np.where(coverage, variance, np.nan), header, name="VARIANCE")])
        hdu.writeto(args.out / "difference.fits", overwrite=True)
        difference_overlay(args.out / "overlay.png", delta, rows)
        files.extend(("difference.fits", "overlay.png"))
    backend = None
    rank_pipeline = None
    if args.galaxy_weights:
        from .galaxy import GalaxyDetector
        backend = GalaxyDetector.load(args.galaxy_weights, args.device)
    if args.ranker_weights:
        from .ranker import load_ranker, rank_image_hosts
        rank_pipeline = load_ranker(args.ranker_weights, args.device)
    if surviving:
        labels, signal, variance = detect(reference, config, backend)
        detections = measurements(reference, labels, signal, variance)
        for row in surviving:
            sn, x, y, coverage = sn_position(reference, row["ra_deg"], row["dec_deg"])
            ranked = rank_hosts(reference, detections, sn, x, y, args.radius)
            info = (rank_image_hosts(args.ranker_weights, args.device, reference, labels, signal, variance,
                                     ranked, detections, x, y, backend, rank_pipeline) if rank_pipeline else None)
            status = "host_proposed" if ranked else "no_plausible_host_detected"
            if info and info["review_required"]:
                status = "multiple_plausible_hosts" if len(ranked) > 1 else "host_requires_review"
            relative = f"transient-{row['transient_id']:03d}"
            folder = args.out / relative
            folder.mkdir(exist_ok=True)
            row.update(host_status=status, host_run=relative)
            write_table(folder / "hosts.csv", ranked, RANK_COLUMNS + (("ranker_score",) if info else ()))
            render(reference, labels, x, y, ranked, None, folder / "overlay.png", "Reference-frame host",
                   position_label="Transient candidate")
            result = dict(format_version=1, mode="pair_host", status=status, candidates=len(ranked), ranker=info,
                          provisional_host=ranked[0] if ranked else None, files=["hosts.csv", "overlay.png"])
            (folder / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_table(args.out / "transients.csv", rows, COLUMNS)
    summary = dict(format_version=1, mode="pair_search", status=("no_usable_coverage" if delta is None else
                   "transients_detected" if surviving else "no_transient_detected"), detections=len(rows),
                   candidates=len(surviving), classifier=classifier, inputs={
                       "reference": dict(path=str(ref_path), sha256=digest(ref_path), seeing=ref_psf, mjd=ref_mjd),
                       "science": dict(path=str(sci_path), sha256=digest(sci_path), seeing=sci_psf, mjd=sci_mjd)},
                   overlap_fraction=float(delta[4].mean()) if delta is not None else 0.,
                   calibration=delta[5] if delta is not None else None, transient_sigma=args.transient_sigma,
                   files=files)
    (args.out / "run.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
