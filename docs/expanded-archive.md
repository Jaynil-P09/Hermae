# Optical and ultraviolet archive runs

The combined dashboard is `results/dashboard.html`. The standalone archive view
is `results/expanded-archive/dashboard.html`. Each optical field includes
registered color, a host overlay, the native r-band image, a segmentation
layer and the underlying candidate measurements.

## Optical images

The run accepted 13 Legacy Surveys fields, including M51, M74, M81, M82,
M61 and M77. Each has a wide g/r/z FITS cube and a separate 0.262 arcsec/pixel
r-band FITS cutout. The color stretch maps z/r/g to RGB. The existing host
ranker ran on the 13 native r-band frames on CUDA with a 63 arcsec search
cone. All 13 outputs call for human host review. The Legacy service did not
return a usable optical FITS product for the NGC 6946 request.

The optical and UV manifests record source URLs, checksums and unavailable
requests. The new collection avoids the frozen Hermae benchmarks.
[Legacy Surveys FITS cutouts](https://www.legacysurvey.org/dr10/description/)

## UV localization

The independent UV CNN takes an aligned reference, a newer image, the
normalized difference and a passband channel. Training used simulated
positive point sources and quiet controls on real NASA SkyView GALEX
backgrounds. This batch has 15 NUV images and 10 FUV images. Both images
from a given sky field stay in the same split. M101 and M77 are held out.
[NASA SkyView](https://skyview.gsfc.nasa.gov/current/cgi/titlepage.pl)

The RTX 5070 run used 3,264 training patches over 24 epochs. Epoch 22 was
selected using the validation fields. On 512 held-out patches it localized
312 of 342 injected sources within three pixels at the selected threshold
and flagged 7 of 170 quiet patches. NUV recovered 164 of 179 injections.
FUV recovered 148 of 163. The quiet-patch flag rate describes these
64-pixel simulations. Larger image searches and Swift observations can have
different flag rates. The dashboard shows selected examples, and
`uv-model/evaluation.json` reports every test group.

## Real Swift example

In `UV / Swift epochs`, the model searched an M101 UVW2 cutout from each of
two epochs. It put the top candidate about 0.85 arcsec from the published
SN 2023ixf position. The published coordinate was used for the comparison
marker after inference. The output reached the limit of 25 review
candidates. UVW2 is a transfer test for the GALEX-trained checkpoint.
Each candidate can be inspected on the science image, signed residual and
score map. A single archival GALEX reference documents spatial structure.
Localization of a new source requires a second observation in the same
filter. [Swift UVOT instrument log](https://heasarc.gsfc.nasa.gov/W3Browse/swift/swiftuvlog.html)
[Published SN 2023ixf coordinates](https://www.wis-tns.org/object/2023ixf)

Open **M101 in near ultraviolet** to see the same top Swift candidate
projected through the WCS of the original GALEX NUV FITS. The full field
shows its location on the galaxy. The nearest-neighbor inset separates
the model diamond from the published cross at their fractional-pixel
coordinates. The 0.85 arcsec separation measures agreement with the
published position. The marker size is for display.

## Run locally

Run these PowerShell commands from the v2 root. The optical job expects the
existing host ranker checkpoint. Both jobs read the v1 and independent-pair
index to exclude overlapping coordinates.

```powershell
.\.venv\Scripts\python.exe -m tools.expand_archive --part optical --device cuda
.\.venv\Scripts\python.exe -m tools.expand_archive --part uv
.\.venv\Scripts\python.exe -m hermae.train_uv --manifest results/expanded-archive/uv.json --out results/expanded-archive/uv-model --epochs 24 --samples 192 --device cuda
```

Download the two original archive files into `results/expanded-archive/swift`
with their original names. Reference uses HDU 2. Science uses HDU 1.
The supplied seeing is the NASA UVW2 calibration FWHM.
[Reference obsid 00032094014](https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/2011_09/00032094014/uvot/image/sw00032094014uw2_sk.img.gz)
[Science obsid 00032481004](https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/2023_05/00032481004/uvot/image/sw00032481004uw2_sk.img.gz)
[UVOT PSF calibration](https://swift.gsfc.nasa.gov/caldb/docs/uvot/uvot_caldb_psf_02.pdf)

```powershell
.\.venv\Scripts\python.exe -m hermae uv-search --reference results/expanded-archive/swift/sw00032094014uw2_sk.img.gz --reference-hdu 2 --science results/expanded-archive/swift/sw00032481004uw2_sk.img.gz --science-hdu 1 --reference-fwhm 2.92 --science-fwhm 2.92 --center 210.895 54.318 --size 128 --compare-position 210.910675 54.31165 --name "SN 2023ixf / Swift UVW2" --weights results/expanded-archive/uv-model/uv-localizer.pt --out results/expanded-archive/swift/2023ixf-uvw2 --device cuda
.\.venv\Scripts\python.exe -m tools.archive_dashboard
```
