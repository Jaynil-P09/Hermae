# Hermae

Hermae compares nearest angular neighbour and directional light radius (DLR) host ranking on real ZTF SN Ia DR2 supernovae. The pilot uses the southern, DECam portion of Legacy Surveys DR10.

## Run the pilot

From PowerShell in the repository root:

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m hermae
```

The defaults select 50 events with seed 42, rank in a 60″ cone, check retrieval again at 90″ and 120″, and render 10 inspection panels. Use `--help` for run settings. Add `--offline` to reuse the cached survey files under `.cache/`. Use `--output` for a separate run.

To rank a single position, supply J2000 RA and Dec in degrees and choose a separate output directory:

```powershell
.venv/Scripts/python -m hermae --position 350.455457 3.828344 --output results/single
```

## Outputs

The default output directory is `results/pilot/`.

- `events.csv` contains ZTF event and host positions, published redshift and provenance, subtype, footprint coverage, and catalogue status.
- `candidates.csv` contains galaxy candidates out to the widest search radius. Both baseline ranks apply inside the baseline cone only. Tractor IDs use `release:brickid:objid`. Coordinates and position angles are in degrees. Offsets and model radii are in arcseconds. Flux is in AB nanomaggies and flux inverse variance is in inverse squared nanomaggies.
- `evaluation.csv` records both picks, the published host cross-match and rank, retrieval at each radius, and review status. A blank retrieval result identifies an unknown outcome. A false entry records a determinate failure to retrieve a rankable reference.
- `coverage_audit.csv` records every point-coverage probe in the checked prefix of the seeded event order. It includes failed requests and any extra events checked in the final batch.
- `panels/` holds the annotated PNGs. `manifest.json` records the corresponding cached FITS files, release checksums, and run settings. `report.md` gives the baseline comparison and links to the panels.

## Methods

- **Selection.** Left-join the published ZTF transient and global-host tables by `ztfname`. Order events by the SHA-256 of `seed:event_id`. Preselect events in processed DR10 south bricks and verify positive `g/r/z` inverse variance at the transient pixel in at least one band. The first 50 verified events form the pilot. A covered position can still have a partially covered search cone.
- **Candidate search.** Fetch DR10 south Tractor records using the survey viewer's padded bounding rectangles and apply an exact spherical cone locally. Split rectangles at RA 0° and retain primary rows. Both methods use `REX`, `EXP`, `DEV`, and `SER` galaxy fits. Keep point-like sources for diagnosing reference matches. Retain native `maskbits`, `fitbits`, and per-band quality fields. Both baselines use the same pool without a photometry or mask cut.
- **Reference cross-match.** Look for primary Tractor rows within 1.5″ of the published host position. Count distinct fitted sources. Tractor `DUP` rows are Gaia entries without their own optical fit, so record their IDs but exclude them from the counterpart count. Mark duplicate-only, multiple fitted matches, and unique point-source matches separately. A narrow buffer outside the widest cone allows a cross-match at the boundary. Candidates outside the widest cone are excluded from retrieval counts.
- **DLR.** Let `e = hypot(e1, e2)` and `b/a = (1-e)/(1+e)`. The Tractor basis gives `atan2(e2, e1)/2` as the major-axis angle east of north modulo 180°. For offset angle `Δ` from the major axis, the directional half-light radius is `shape_r / hypot(cos Δ, sin Δ / (b/a))`. DLR divides spherical separation by this radius. REX fits are round. Finite shapes with a radius-limit or an uncertainty flag remain eligible. Invalid shapes sort after finite scores, in nearest-first order. Ties break by separation and source ID.
- **Review.** Flag empty galaxy cones or best DLR above 7 as possible missing hosts. Flag nearest fallback, flagged best-DLR shapes, disagreements, or best-two DLR scores within 20% for review. These are pilot heuristics. The cut at 7 is borrowed from the ZTF procedure, and its behavior depends on the radius definition. Panels emphasize difficult cases within the pilot.
- **Interpretation.** Compare against the ZTF published DLR-selected reference assignments. Ranking uses no redshift or published-host features. The output records galaxy-catalogue and galaxy-emission-line redshifts as provenance, and identifies transient-template redshifts for potential later use. Unknown and override codes require provenance verification. Independent evaluation will need independently corroborated host labels and must withhold evidence used to establish those labels from the corresponding ranking inputs.

## Expand the benchmark

Build the first 150-event cohort with the same survey settings as the pilot and save the split manifest before fitting any model:

```powershell
.venv/Scripts/python -m hermae --events 150 --panels 0 --output results/benchmark
.venv/Scripts/python -m hermae.splits --run results/benchmark --development-events benchmark/development_events.csv --output benchmark/split-v1.json
```

The split freezes event IDs and fingerprints the survey exports. It targets a 60/20/20 train, validation, and test allocation by whole sky group. Groups link events in the same roughly 10° sky block, events sharing a retrieved or published catalogue source, events whose published host coordinates agree within the reference match tolerance, and events whose candidate cutouts could overlap. The 120″ outer search and a 60″ maximum crop half-diagonal give a conservative 360″ separation limit for possible overlap. All groups linked to a previously inspected field are forced into training. The development list includes provisional pilot runs and the single-position example, even if a field fell outside the expanded cohort.

The split command refuses to overwrite a manifest with different inputs or to freeze a cohort with a failed catalogue query or a failed coverage probe preceding the last selected event. The training and evaluation inputs are regenerated from the survey pipeline and cache.

Fit the two catalogue rankers and compare them against nearest neighbour and DLR on the validation split:

```powershell
.venv/Scripts/python -m hermae.benchmark --run results/benchmark --split benchmark/split-v1.json --output results/benchmark/validation
```

Both models fit regularized logistic regression. One uses separation and DLR. The other adds optical magnitudes and colours, Tractor morphology, and the number of other candidates within 5″ of each candidate. Median imputation, missingness indicators, and standardization fit on training rows. Each informative training event contributes equal total weight to its matched source and to its alternative candidates. The training label is the uniquely retrieved ZTF reference counterpart.

The command exports per-candidate validation scores and ranks, an event-level comparison, and a summary with first-rank and top-three agreement. The paired first-rank differences use a bootstrap that resamples whole validation sky groups. Scores are uncalibrated. The test assignments remain reserved for a later evaluation.

The [v1 split manifest](benchmark/split-v1.json) and [v1 validation report](benchmark/validation-v1.md) record the 150-event run. For the next experiment, audit candidate spectra and their targeting provenance, then compare redshift and FITS features as separate additions to the catalogue ranker. Evaluate Pröst on the same validation events and document any differences in the candidate pool. [DESI DR1 data](https://data.desi.lbl.gov/doc/releases/dr1/) · [Pröst](https://github.com/alexandergagliano/Prost)

## Release documentation

- [ZTF SN Ia DR2 overview](https://doi.org/10.1051/0004-6361/202450388) and [release reader](https://github.com/ZwickyTransientFacility/ztfcosmo/blob/main/ztfcosmo/io.py)
- [Legacy Surveys DR10 data and cutout service](https://www.legacysurvey.org/dr10/description/), [Tractor fields](https://www.legacysurvey.org/dr10/catalogs/), and [mask definitions](https://www.legacysurvey.org/dr10/bitmasks/)
- [Tractor ellipse basis](https://github.com/dstndstn/tractor/blob/main/tractor/ellipses.py)
