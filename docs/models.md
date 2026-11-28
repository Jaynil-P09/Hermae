# Development models

Train in `Hermae_v2` with the audited `Hermae_v1` checkout alongside it.
The training commands select CUDA automatically when available. Pass
`--device cuda` to require an NVIDIA GPU, `--device cuda:1` to select a specific
GPU, or `--device cpu` for CPU execution. Each run records the selected device
and its name in `report.json`. The commands below write new checkpoints to
`results/quality-training`. Use a new output folder to keep multiple runs.

To run all seven v2 checkpoints for 150 epochs each, open PowerShell and run
`powershell -NoExit -ExecutionPolicy Bypass -File tools/train_150.ps1`.
The console prints each epoch, and `results/training-150` receives both logs,
checkpoints, and a `completed.json` count for every model. A repeat run adds a
timestamp to the output directory.

```powershell
.venv/Scripts/python.exe -m hermae.train_galaxy --v1 ../Hermae_v1 --out results/quality-training/galaxy --device cuda
.venv/Scripts/python.exe -m hermae.train_ranker --v1 ../Hermae_v1 --out results/quality-training/ranker-classical --device cuda
.venv/Scripts/python.exe -m hermae.train_ranker --v1 ../Hermae_v1 --out results/quality-training/ranker-unet --device cuda --galaxy-weights results/quality-training/galaxy/galaxy.pt
.venv/Scripts/python.exe -m hermae.train_transient --v1 ../Hermae_v1 --manifest results/ztf-development/manifest.json --out results/quality-training/transient --device cuda
```

If you need to acquire ZTF pairs again, run
`.venv/Scripts/python.exe -m hermae.ztf_pairs --v1 ../Hermae_v1 --out results/ztf-new --limit 35`
and pass the new manifest to the transient trainer. Use the archived acquisition
manifest and its FITS hashes for the exact release cohort.

| Trainer | Default epochs | Plateau window | Batch | Initial learning rate |
| --- | ---: | ---: | --- | ---: |
| Galaxy U-Net | 150 | 12 | 32 crops | 0.002 |
| Host ranker | 150 | 18 | 8 fields | 0.001 |
| Optical transient classifier | 150 | 15 | 32 residuals | 0.0015 |
| UV localizer | 150 | Fixed rate | 48 patches | 0.0015 |

The galaxy trainer samples up to 32 crops per field. The default seed is 42
and AdamW weight decay is 0.01. The trainers halve the rate during a
validation plateau and restore the selected checkpoint after all 150
epochs. The learning rate can fall to 1% of its initial value. The
galaxy trainer selects by loss on separate validation crops. Host rankers
select by top-one validation accuracy and mean reciprocal rank on the
image or union pool. The transient trainer prioritizes confirmed recovery
and fewer pre-event control alerts on the development pairs. The small
transient validation cohort limits comparisons. Use each trainer's
`--help` for overrides. Reports keep the validation metrics, epoch selected,
and learning-rate history.

The CUDA rerun in `results/quality-training` selected galaxy epoch 39 of
51 and doubled the training crops to 3,072. The detector matched 524/1,330
catalogue centres with 322 unmatched detections, compared with 501/1,330
and 458 for the release U-Net. At the selected operating point, the new
U-Net missed more published supernova hosts in the image-only pool.
The classical detector with the new image-only ranker had the strongest
FITS-only host result at 20/22 development fields. The U-Net route had
4/22, with only four unique published hosts retrieved. The transient
classifier selected epoch 5 of 20 and reproduced the development recovery
and control-alert counts. The rerun used the development split for model
selection.

## Refit on every eligible development image

To refit after selecting schedules on the original development validation
partition, run the command below. It combines the training and validation
partitions, replays the saved learning rates through each selected epoch,
and holds the last rate for the remainder of the 150-epoch run.
Reserved fields and the independent image-pair benchmark stay outside this
fit. The refit reports training coverage and exclusions. Its host scores
remain subject to review because the new weights lack a separately measured
score-gap threshold.

```powershell
.venv/Scripts/python.exe -u -m hermae.refit --v1 ../Hermae_v1 --schedules results/quality-training --manifest results/ztf-development/manifest.json --out results/full-refit --device cuda
```

Pass `--supplemental-manifest results/ztf-cached-supplement/manifest.json` to
include the recovered local ZTF pairs, and use `--transient-only` when
you want to refit only that classifier. The supplement uses earlier
same-quadrant references and matches the newer exposures to positive r-band
alert timestamps. [ALeRCE detection fields](https://alerce.readthedocs.io/en/latest/models/detection.html)

The local run used 119 usable coadds for the galaxy U-Net and 109–111
uniquely labelled fields per host ranker. The expanded transient run
processed 38 exposures in 24 pairs. Sixteen pairs supplied 49 labelled
residuals, including 10 confirmed alerts. The 13 older multiband previews
cover fields already in the coadd set. Five audited coadds lacked r-band
weight, and one cached ZTF exposure lacked a verified later companion.
Run reports are under `results/all-development-training`, with the expanded
transient checkpoint in its `transient-expanded` subdirectory.

## Galaxy detector

The figures in the remaining sections describe the frozen release runs in
the tagged `hermae-v2-frozen-eval` checkout.

The two-head U-Net learned footprint pixels and centre peaks from 1,536 crops
across 96 development training fields. Training used Legacy Surveys galaxy
centres, weak footprints derived from fitted shapes, known point sources, and
simulated faint and blended galaxies injected onto real backgrounds. The
masked loss combines Dice, binary cross-entropy and focal loss. Inference
uses centre peaks to partition footprints, then passes them through the
same overlap merge and measurement code as the classical detector.
[Legacy Surveys catalogue format](https://www.legacysurvey.org/dr10/catalogs/)

Both methods had **458 unmatched detections across 23 fields** at the
selected operating point. With the same unmatched count, the classical
detector matched 539/1,330 weak real-field references and the U-Net matched
501/1,330. Unmatched detections can include genuine sources missing from the
weak reference. Pixel overlap against the catalogue ellipses, measured over
labelled regions, was 0.285 and 0.390 respectively. The classical detector
recovered 205/253 artificial galaxies, including 16/51 faint and 140/182
blended. The U-Net recovered 168/253, including 15/51 faint and 105/182
blended. A field-edge subset had 70/165 and 68/165 matches respectively.
Classical detection stays the default.

The U-Net did recover the published counterpart of `ZTF18abvtiko`, where
classical detection missed. At the frozen footprint and centre thresholds
of 0.70 and 0.65, it placed that source first with a centroid 0.375″ from
the published host. Dense, independently drawn real galaxy masks and
exhaustive false-positive labels remain unavailable.

## Host ranker

The model scores each candidate from a close cutout, a wider SN-to-host
cutout, the candidate footprint, valid-pixel and SN-position masks, and
image-derived geometry and brightness. A second model adds available
catalogue features with missing-data indicators. Both train with a
fieldwise softmax objective and supervise only fields with a unique
published-counterpart match in the candidate pool. The published position
is never injected into a candidate pool.

<table>
<thead><tr><th>Candidate detector</th><th>Ranker</th><th>Catalogue pool</th><th>Image pool</th><th>Union pool</th></tr></thead>
<tbody>
<tr><td>Classical</td><td>Image</td><td>19/22</td><td>19/22</td><td>18/22</td></tr>
<tr><td>Classical</td><td>Catalogue fusion</td><td>20/22</td><td>20/22</td><td>19/22</td></tr>
<tr><td>U-Net</td><td>Image</td><td>20/22</td><td>10/22</td><td>20/22</td></tr>
<tr><td>U-Net</td><td>Catalogue fusion</td><td>20/22</td><td>10/22</td><td>20/22</td></tr>
</tbody></table>

These are end-to-end top-one counts on 22 evaluable validation fields. The
other four fields had unusable FITS coverage or missing published positions.
Unique retrieval counts for catalogue, image and union were 21, 20 and 20
for classical detection, and 21, 10 and 21 for U-Net detection. Conditional
top-one accuracy uses those retrieval counts as denominators. For the
deployed classical plus image-only path, conditional agreement was 19/20.

The review policy chooses a development score gap at which at least 90% of
accepted recommendations agree with the reference. A field with no
candidates or unusable coverage receives a separate status. The CLI's
image-only score works on candidates with no catalogue entry.

## Blind transient search

The pipeline aligns newer pixels onto the reference WCS, estimates the
background and relative flux scale, matches the sharper exposure with a
Gaussian kernel, and propagates the usable footprint and an approximate
variance estimate. It detects positive residuals and sends each surviving
location to the host ranker using the earlier exposure. ZTF mask bits that
record clean sources remain usable. [IRSA ZTF image API](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html)
[ZTF mask definitions](https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf)

The development acquisition used ALeRCE red-band positive alerts and public
IRSA science and mask pairs. Earlier pairs from the same sky fields served as
pre-event controls, which can include unrelated variability. The classifier
used 5 recovered real alerts and 29 likely dipoles from training fields.
Those negatives are labelled from the subtraction signatures. The classifier
reads aligned reference, newer, difference and valid-mask cutouts.
[ALeRCE pipeline](https://science.alerce.online/alerce-pipeline/)

Both workflows recovered **2/3** position-confirmed development alerts
within 2.5″, with **2 alerts over 3** pre-event controls. The reference-frame
ranker agreed with the published host for both recovered events (2/2
conditional, 2/3 end to end). The CNN showed no benefit on this small
selection, so subtraction remains the default. Gaussian PSF matching and
bilinear resampling can leave residuals near bright or structured objects.
