"""Find supernova hosts in a FITS image or search earlier and newer exposures."""

import argparse
from pathlib import Path

from .detection import DetectorConfig
from .fits_input import hdu_key
from .host import host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("host", help="detect full-frame sources and rank candidates near a known SN")
    run.add_argument("--fits", required=True, type=Path)
    run.add_argument("--ra", type=float, required=True, help="ICRS right ascension in degrees")
    run.add_argument("--dec", type=float, required=True, help="ICRS declination in degrees")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--hdu", type=hdu_key, default=0, help="science HDU index or name")
    run.add_argument("--mask-hdu", type=hdu_key, help="mask HDU, zero means valid")
    weights = run.add_mutually_exclusive_group()
    weights.add_argument("--ivar-hdu", type=hdu_key)
    weights.add_argument("--variance-hdu", type=hdu_key)
    run.add_argument("--radius", type=float, default=60., help="host cone radius in arcseconds")
    run.add_argument("--tile", type=int, default=512)
    run.add_argument("--overlap", type=int, default=64)
    run.add_argument("--sigma", type=float, default=3., help="detection threshold above local sky")
    run.add_argument("--min-pixels", type=int, default=7)
    run.add_argument("--detector", choices=("classical", "cnn"), default="classical")
    run.add_argument("--galaxy-weights", type=Path, help="trained galaxy checkpoint for the CNN detector")
    run.add_argument("--ranker-weights", type=Path, help="optional image-only host ranker checkpoint")
    run.add_argument("--device", default="cpu", help="model inference device, e.g. cpu or cuda")
    run.add_argument("--v1", type=Path, help="read-only path to the preserved v1 worktree")
    run.add_argument("--event", help="frozen development event identifier for a v1 comparison")
    pair = commands.add_parser("search", help="search an earlier and newer exposure for positive transients")
    pair.add_argument("--reference", required=True, type=Path)
    pair.add_argument("--science", required=True, type=Path)
    pair.add_argument("--out", required=True, type=Path)
    pair.add_argument("--reference-hdu", type=hdu_key, default=0)
    pair.add_argument("--science-hdu", type=hdu_key, default=0)
    pair.add_argument("--reference-fwhm", type=float, help="PSF FWHM in arcseconds, defaults to SEEING")
    pair.add_argument("--science-fwhm", type=float, help="PSF FWHM in arcseconds, defaults to SEEING")
    pair.add_argument("--transient-sigma", type=float, default=4.)
    pair.add_argument("--radius", type=float, default=60.)
    pair.add_argument("--galaxy-weights", type=Path)
    pair.add_argument("--ranker-weights", type=Path)
    pair.add_argument("--transient-weights", type=Path)
    pair.add_argument("--device", default="cpu")
    uv = commands.add_parser("uv-search", help="localize UV residuals in a same-filter epoch pair")
    uv.add_argument("--reference", required=True, type=Path)
    uv.add_argument("--science", required=True, type=Path)
    uv.add_argument("--reference-hdu", type=hdu_key, default=0)
    uv.add_argument("--science-hdu", type=hdu_key, default=0)
    uv.add_argument("--reference-fwhm", required=True, type=float, help="reference PSF FWHM in arcseconds")
    uv.add_argument("--science-fwhm", required=True, type=float, help="science PSF FWHM in arcseconds")
    uv.add_argument("--center", type=float, nargs=2, metavar=("RA", "DEC"), help="optional field center in ICRS degrees")
    uv.add_argument("--compare-position", type=float, nargs=2, metavar=("RA", "DEC"),
                    help="published coordinate for post-inference overlay and offset")
    uv.add_argument("--size", type=int, default=64, help="output cutout width in model pixels")
    uv.add_argument("--name")
    uv.add_argument("--device", default="cpu")
    uv.add_argument("--weights", type=Path, required=True)
    uv.add_argument("--out", type=Path, required=True)
    view = commands.add_parser("dashboard", help="build an offline HTML view from saved run outputs")
    view.add_argument("--run", action="append", type=Path, default=[],
                      help="saved run directory; repeat to combine results")
    view.add_argument("--collection", action="append", nargs=2, default=[], metavar=("NAME", "RUN"),
                      help="add a saved run to a named menu option")
    view.add_argument("--collection-preview", action="append", nargs=2, default=[], metavar=("NAME", "PNG"),
                      help="attach a PNG preview to a named collection")
    view.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "uv-search":
        if not 40 <= args.size <= 512 or any(
            not 0 < fwhm < float("inf") for fwhm in (args.reference_fwhm, args.science_fwhm)
        ):
            uv.error("Use a 40–512 pixel cutout and positive finite PSF widths.")
        for position in (args.center, args.compare_position):
            if position and not (0 <= position[0] < 360 and -90 <= position[1] <= 90):
                uv.error("Use valid ICRS coordinates.")
        from .uv import search
        try:
            result = search(args)
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            uv.exit(1, f"hermae: {exc}\n")
        print(f"{result['status']} · {result['candidates']} UV candidates · {args.out}")
        return
    if args.command == "dashboard":
        from .dashboard import dashboard
        try:
            path = dashboard(args.run, args.out, collections=args.collection, previews=args.collection_preview)
        except (OSError, ValueError, KeyError) as exc:
            view.exit(1, f"hermae: {exc}\n")
        print(f"Dashboard written to {path}")
        return
    if args.command == "search":
        if not (0 < args.radius <= 648000 and 0 < args.transient_sigma < float("inf")):
            pair.error("Use finite, positive thresholds and a valid radius.")
        from .pair import search
        try:
            result = search(args, DetectorConfig())
        except (OSError, ValueError, KeyError, IndexError) as exc:
            pair.exit(1, f"hermae: {exc}\n")
        print(f"{result['status']} · {result['detections']} residuals · {result['candidates']} candidates")
        print(f"Results written to {args.out}")
        return
    if not (0 <= args.ra < 360 and -90 <= args.dec <= 90 and 0 < args.radius <= 648000):
        run.error("Use finite ICRS coordinates and a positive radius in arcseconds.")
    if not (args.tile >= 8 and 0 <= args.overlap < args.tile and 1 <= args.min_pixels <= args.tile**2
            and 0 < args.sigma < float("inf")):
        run.error("Tile size must be at least 8 and overlap smaller than the tile. Use positive detection settings.")
    if bool(args.v1) != bool(args.event):
        run.error("--v1 and --event must be supplied together.")
    if (args.detector == "cnn") != bool(args.galaxy_weights):
        run.error("CNN detection requires --galaxy-weights; omit it for classical detection.")
    try:
        result = host(args, DetectorConfig(args.tile, args.overlap, args.sigma, args.min_pixels))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        run.exit(1, f"hermae: {exc}\n")
    print(f"{result['status']} · {result['detections']} detections · {result['candidates']} candidates")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
