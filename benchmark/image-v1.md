# FITS image ranking ablation

The same 92 training events and 26 validation events were used by the catalogue-only and catalogue-plus-image rankers. The frozen test split remained reserved.

## Image coverage

- **Train** had 96/98 usable FITS fields. 96 had at least one measured candidate feature.
- **Validation** had 23/26 usable FITS fields. 23 had at least one measured candidate feature.
- **ZTF19aarqmjs** had status `no_r_band_weight`.
- **ZTF19aaznoct** had status `no_r_band_weight`.
- **ZTF19acrddcz** had status `no_r_band_weight`.
- **ZTF19acrmmpq** had status `no_r_band_weight`.
- **ZTF20abzayoe** had status `no_r_band_weight`.

At least one pixel feature was measured for 7,145 of 7,300 train and validation candidates.

- **core** 7,089/7,300 measurements.
- **envelope** 7,145/7,300 measurements.
- **sn to core** 7,089/7,300 measurements.
- **bridge contrast** 6,998/7,300 measurements.

## Agreement with the published ZTF reference

23 validation events had a uniquely recovered reference counterpart. The comparison retains the same events and candidates when pixel features are missing.

- **Catalogue only** ranked 22/23 first and 23/23 in the top three.
- **Catalogue plus images** ranked 23/23 first and 23/23 in the top three.

The paired first-rank change was +4.3 percentage points (95% group bootstrap interval +0.0 to +15.0). The image model improved 1 and worsened 0 of the first-rank outcomes.

The image model adds four measurements of r-band light to the same regularized logistic ranker. Inverse variance masks unusable pixels. Measurements use the candidate and SN coordinates, with a 1.4″ mask around the transient to reduce contamination from the SN itself. The DR10 imaging dates overlap the transient search. The training label remains ZTF's published DLR-selected reference.

[Legacy Surveys DR10 images](https://www.legacysurvey.org/dr10/description/)
