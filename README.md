# Hermae v2

Hermae v2 is an image-first host-galaxy pipeline. The first milestone starts from an r-band FITS image and known supernova coordinates. It detects and deblends sources across the entire image, then ranks detections in the host search cone. The separate `version_1` checkout holds the preserved catalogue benchmark, data cache, and checkpoint audit.

## Install

Run from this worktree in PowerShell.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e .
```

## Find a provisional host

```powershell
.venv/Scripts/hermae.exe host --fits field.fits --ra 71.884741 --dec -2.306897 --out results/field
```

Supply one two-dimensional celestial image and ICRS coordinates in degrees. The science image defaults to HDU 0. Select another with `--hdu` by name or index. An explicit `--mask-hdu` uses zero for valid pixels. Supply `--ivar-hdu` or `--variance-hdu` for an uncertainty plane. The command also looks for named `MASK`, `IVAR`, `INVVAR`, and `VARIANCE` extensions when their options are omitted. Unlabelled science frames are treated as r band. Nonfinite pixels and invalid weights are masked. The command checks the WCS, HDU alignment, and coverage at the SN.

Defaults are a 60″ cone, 512 × 512 tiles, 64-pixel overlap, and a 3σ detection threshold. Use `--radius`, `--tile`, `--overlap`, `--sigma`, and `--min-pixels` to change these. Detection uses a robust local background, Gaussian-smoothed signal, and Photutils watershed deblending. Cross-tile footprints merge when at least 55% of the smaller mask overlaps and the centroids are close. A merge cannot combine two footprints already separated in the same tile. [Photutils segmentation](https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html)

The provisional ranking divides sky separation by the directional size from the footprint's flux-weighted second moments. Reported sizes are moment dispersions in pixels. Centroid and position angle use the celestial WCS. Edge, mask-adjacent, and unresolved shapes fall back to separation and follow candidates with reliable shape measurements. `extended` and `unresolved` describe the footprint. They carry no galaxy or star identity. [Astropy WCS](https://docs.astropy.org/en/stable/wcs/wcsapi.html)

## Outputs

- `detections.csv` contains every full-frame detection with centroid, bounds, pixel area, flux in the input's native units, approximate flux noise, shape, and coverage flags.
- `segmentation.fits` maps each labelled footprint back to the original pixel grid. A positive label equals the detection's `source_id`.
- `hosts.csv` contains candidates inside the cone, nearest rank, shape-scaled rank, and shape fallback flags.
- `overlay.png` shows the real FITS pixels, detected footprints, SN position, and provisional host.
- `run.json` records the input SHA-256, HDU choices, configuration, coverage, counts, and status.
- `comparison.json` is written when an audited v1 development input is supplied. It reports the preserved DLR and measured-FITS picks and checks the proximity of v2 centroids to the published counterpart.

A field with an SN outside coverage retains its full-frame detections and returns no host candidates. A covered field with an empty cone reports `no_plausible_host_detected`.

## Reproduce the development example

```powershell
$v1 = Resolve-Path -LiteralPath '../Hermae_v1'
$event = 'ZTF18abvtiko'
$sn = Import-Csv (Join-Path $v1 'results/benchmark/events.csv') | Where-Object event_id -eq $event
$image = Import-Csv (Join-Path $v1 'results/benchmark/images/image_inputs.csv') | Where-Object event_id -eq $event
$fits = Join-Path $v1 (Join-Path '.cache/image_ablation' $image.cache_file)
.venv/Scripts/hermae.exe host --fits $fits --ivar-hdu 1 --ra $sn.ra --dec $sn.dec --out "results/$event" --v1 $v1 --event $event
```

The comparison path verifies the FITS and the preserved split and benchmark hashes. It accepts development events only and writes results to the v2 checkout. See the [development run record](docs/development-run.md) and [remaining phases](docs/roadmap.md).

```powershell
.venv/Scripts/python -m unittest discover -s tests -v
```
