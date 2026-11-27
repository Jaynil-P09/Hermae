# Hermae v2 development release

The local release bundle is `results/release/hermae-v2-release.zip`.
It contains an installable Python wheel, model checkpoints, example FITS
inputs, saved runs with offline dashboards, and the field-level benchmark.
The CLI defaults to classical galaxy detection and, in pair mode,
subtraction-based transient detection. Both neural detectors remain
selectable. The release uses the image-only host ranker for both example runs.

## Run the bundled examples

Extract the release bundle and run these commands from its root in PowerShell.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install (Get-ChildItem *.whl | Select-Object -First 1 -ExpandProperty FullName)
.venv/Scripts/python.exe -m pip install torch==2.10.0
$example = Get-Content examples/coordinates.json -Raw | ConvertFrom-Json
.venv/Scripts/hermae.exe host --fits examples/inputs/known.fits --ivar-hdu 1 --ra $example.ra --dec $example.dec --ranker-weights models/host-classical.pt --out rerun/known
.venv/Scripts/hermae.exe search --reference examples/inputs/reference.fits --science examples/inputs/newer.fits --ranker-weights models/host-classical.pt --out rerun/pair
.venv/Scripts/hermae.exe dashboard --run rerun/known
.venv/Scripts/hermae.exe dashboard --run rerun/pair
```

The saved examples use the development event `ZTF20aahqbib`. Open
`examples/known/dashboard.html` or `examples/pair/dashboard.html` in a browser
to inspect the saved outputs. The known-position model placed the top
candidate 0.030″ from the published counterpart. Blind search recovered the
transient and placed the top host candidate 0.164″ from that counterpart
using the earlier ZTF exposure.

## Pre-evaluation freeze

Commit `e40ef89` recorded code commit `d8eae9d`, the sky split, checkpoints,
external cohort source hashes, and thresholds before either reserved
evaluation. The v1 baseline was exported with its training-only features.
The export reproduced all 26 archived v1 validation picks and their scores
exactly. The pair cohort was sampled with a fixed seed and excludes v1
cohort and pilot sky regions plus fields within 360″ of those prior fields.
[Freeze manifest](release-freeze.json)

Run the following from the `hermae-v2-frozen-eval` tag checkout with the local
checkpoints and v1 data in the paths specified by the freeze. The benchmark
scripts validate the code and model hashes before loading held-out data.
The packaged CLI includes a later presentation fix that labels blind-search
positions as transient candidates. That fix leaves the score calculations and
frozen results intact.

```powershell
.venv/Scripts/python.exe -m hermae.release known --v1 ../Hermae_v1 --manifest docs/release-freeze.json --out results/release --device cuda
.venv/Scripts/python.exe -m hermae.release pairs --v1 ../Hermae_v1 --manifest docs/release-freeze.json --out results/release --device cuda
```

## Known-position benchmark

All 26 reserved FITS images were acquired. The initial request hit the
survey's rate limit, and paced retries recovered the outstanding images.
Two images had masked SN pixels, leaving 24 covered fields. One covered
field lacked a published counterpart position.

| Method | Top one, all reserved fields | Top one given unique retrieval |
| --- | ---: | ---: |
| v1 DLR | 20/26 | 20/24 |
| v1 measured FITS | 23/26 | 23/24 |
| v2 classical detector and geometric rank | 21/26 | 21/23 |
| v2 classical detector and image-only CNN ranker | 22/26 | 22/23 |
| v2 U-Net detector and image-only CNN ranker | 7/26 | 7/7 |

The release image-only path agreed with the published counterpart for
22/24 covered fields. Its four unsuccessful fields were two coverage
failures, one missing reference, and one ranking error. The experimental
U-Net missed 16 published positions at retrieval.
Against measured FITS, the v2 release path improved one field and worsened
two. In `ZTF20actlvpy`, it picked the single image centroid within the
published tolerance in a field with an ambiguous catalogue reference.
In `ZTF20abpubll`, the SN pixel was masked. In `ZTF19abrilok`, the
published candidate was recovered and a different object ranked first.
The field-level JSON includes paired outcomes and input SHA-256 values.

## Blind image-pair benchmark

Of 36 independently sampled fields, 27 yielded verified alert exposure
pairs. Nine lacked a usable public pairing. There were 26 positive pairs
with usable coverage at the alert location, and 17 pre-event control pairs
with overlapping coverage.

| Measure | Subtraction | Subtraction + CNN |
| --- | ---: | ---: |
| Alert recovery, all usable archive pairs | 19/27 | 19/27 |
| Alert recovery, covered alert positions | 19/26 | 19/26 |
| Alerts per covered pre-event control | 81/17 (4.76) | 25/17 (1.47) |
| Host agreement given transient recovery | 12/19 | 12/19 |
| End-to-end host agreement | 12/27 | 12/27 |

At the first failed stage, one positive pair lacked coverage, seven
missed the transient, five missed the published host at retrieval, and
two ranked another candidate first. Control alerts are a proxy for false
alerts because pre-event fields can contain unrelated real variability.
The held-out improvement is reported with the frozen default unchanged.

For `ZTF19abtnftd`, the published counterpart was missed during retrieval
and the first-ranked source was 11.48″ away. For `ZTF20abpyxja`, both sources
were recovered and the ranker preferred a much brighter object 48.09″
away. Independent Pan-STARRS r-band stacks show distinct sources at both
published and selected positions in both fields. The checked stacks and
annotated cutouts are included under `benchmark/disagreements/`. These
image checks describe detections and proposed associations. Independent
host reassignment would require additional evidence.
[Pan-STARRS image cutout service](https://outerspace.stsci.edu/spaces/PANSTARRS/pages/298812251/PS1%2BImage%2BCutout%2BService)

The pair benchmark reports recovery at the known alert positions after
running the search without position inputs. The FITS cutouts are selected
around known events to build the benchmark and cover 170″ fields. Rates
apply to this archive sample and area.

## Data sources

The known-position example uses a Legacy Surveys DR10 south coadd. The
image-pair examples use public ZTF r-band science and mask exposures from
IRSA and alert metadata from ALeRCE. The package includes source URLs and
recorded FITS identities for the pair inputs. [Legacy Surveys acknowledgments](https://www.legacysurvey.org/acknowledgment/)
[IRSA ZTF archive](https://irsa.ipac.caltech.edu/Missions/ztf.html)
[ALeRCE pipeline](https://science.alerce.online/alerce-pipeline/)
