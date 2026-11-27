# Development models

Train in `Hermae_v2` with the audited `Hermae_v1` checkout alongside it.
Commands below use the verified RTX 5070 environment. Omit `--device cuda` for
CPU execution. All training and operating-point selection used development
fields from the frozen v1 split. The archived ZTF pair acquisition also used
that split.

```powershell
.venv/Scripts/python.exe -m hermae.train_galaxy --v1 ../Hermae_v1 --out results/phase2 --epochs 12 --device cuda
.venv/Scripts/python.exe -m hermae.train_ranker --v1 ../Hermae_v1 --out results/phase3-classical --epochs 20 --device cuda
.venv/Scripts/python.exe -m hermae.train_ranker --v1 ../Hermae_v1 --out results/phase3 --epochs 20 --device cuda --galaxy-weights results/phase2/galaxy.pt
.venv/Scripts/python.exe -m hermae.ztf_pairs --v1 ../Hermae_v1 --out results/ztf-development --limit 35
.venv/Scripts/python.exe -m hermae.train_transient --v1 ../Hermae_v1 --manifest results/ztf-development/manifest.json --out results/phase4 --epochs 20 --device cuda
```

The archive commands are live queries. Availability can change. Use the
archived acquisition manifest and its FITS hashes to reproduce the exact pair
cohort in this release.

## Galaxy detector

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
