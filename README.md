# Hermae v2

Hermae v2 is an image-first host-galaxy pipeline. The first milestone starts from an r-band FITS image and known supernova coordinates. It detects and deblends sources across the entire image, then ranks detections in the host search cone. The separate `version_1` checkout holds the preserved catalogue benchmark, data cache, and checkpoint audit.

## Install

Run from this worktree in PowerShell.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e .
```

Install the optional neural models with `.venv/Scripts/python -m pip install -e ".[ml]"`.
Training selects CUDA automatically with a CUDA-enabled PyTorch installation.
Use `--device cuda` to require CUDA or `--device cpu` to train on the CPU. The classical
detector, subtraction baseline, and dashboard can run without PyTorch.

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

Add `--ranker-weights results/phase3-classical/image_only.pt` to use the
release host ranker with classical detections. For U-Net detection, use
`--detector cnn --galaxy-weights results/phase2/galaxy.pt`. Pair those U-Net
weights with `results/phase3/image_only.pt` if you also want neural ranking.
Ranker scores are uncalibrated. The run summary records the review threshold and
returns `multiple_plausible_hosts` when several candidates need review.

## Reproduce the development example

```powershell
$v1 = Resolve-Path -LiteralPath '../Hermae_v1'
$event = 'ZTF18abvtiko'
$sn = Import-Csv (Join-Path $v1 'results/benchmark/events.csv') | Where-Object event_id -eq $event
$image = Import-Csv (Join-Path $v1 'results/benchmark/images/image_inputs.csv') | Where-Object event_id -eq $event
$fits = Join-Path $v1 (Join-Path '.cache/image_ablation' $image.cache_file)
.venv/Scripts/hermae.exe host --fits $fits --ivar-hdu 1 --ra $sn.ra --dec $sn.dec --out "results/$event" --v1 $v1 --event $event
```

The comparison path verifies the FITS and the preserved split and benchmark hashes. It accepts development events only and writes results to the v2 checkout. See the [development run record](docs/development-run.md) and [project roadmap](docs/roadmap.md).

## Search two epochs

`search` uses the earlier exposure for source measurements and host ranking.
Supply two r-band FITS images with overlapping celestial WCS. A PSF FWHM in
arcseconds is required for each exposure. It comes from the `SEEING` header or
the `--reference-fwhm` and `--science-fwhm` flags. Explicit science HDU names or
indices are supported. Named `MASK`, `IVAR` and `VARIANCE` planes carry
coverage and noise across alignment and subtraction.

```powershell
$pair = (Get-Content results/ztf-development/manifest.json -Raw | ConvertFrom-Json).pairs |
  Where-Object { $_.event -eq 'ZTF19aartuli' -and $_.kind -eq 'positive' }
.venv/Scripts/hermae.exe search --reference $pair.reference.path --science $pair.science.path --out results/example-search --ranker-weights results/phase3-classical/image_only.pt
.venv/Scripts/hermae.exe dashboard --run results/example-search
.venv/Scripts/hermae.exe dashboard --run results/ZTF18abvtiko
```

Open the generated `dashboard.html` directly in a browser. It contains the
original field or aligned image triplet, sortable detection tables, and
reference-frame host results for each surviving transient. `difference.fits`
contains the signed residual, coverage mask and variance estimate. Add
`--transient-weights results/phase4/transient.pt` to try the experimental CNN
filter. The difference detections are transient candidates pending independent
classification.

## Train and evaluate

The training commands, development results and known limitations are in the
[model and pair-search notes](docs/models.md). The [release report](docs/release.md)
records the reserved evaluations and points to the local release bundle.

```powershell
.venv/Scripts/python -m unittest discover -s tests -v
```
