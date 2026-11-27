"""Known-position full-FITS host detection command."""

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
    run.add_argument("--v1", type=Path, help="read-only path to the preserved v1 worktree")
    run.add_argument("--event", help="frozen development event identifier for a v1 comparison")
    args = parser.parse_args()
    if not (0 <= args.ra < 360 and -90 <= args.dec <= 90 and 0 < args.radius <= 648000):
        run.error("Use finite ICRS coordinates and a positive radius in arcseconds.")
    if not (args.tile >= 8 and 0 <= args.overlap < args.tile and 1 <= args.min_pixels <= args.tile**2
            and 0 < args.sigma < float("inf")):
        run.error("Tile size must be at least 8 and overlap smaller than the tile. Use positive detection settings.")
    if bool(args.v1) != bool(args.event):
        run.error("--v1 and --event must be supplied together.")
    try:
        result = host(args, DetectorConfig(args.tile, args.overlap, args.sigma, args.min_pixels))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        run.exit(1, f"hermae: {exc}\n")
    print(f"{result['status']} · {result['detections']} detections · {result['candidates']} candidates")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
