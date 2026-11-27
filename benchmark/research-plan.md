# Hermae research plan

**Research question.** Do telescope images and independently measured redshifts improve Type Ia supernova host ranking and help identify unreliable associations?

The groundwork is complete. The catalogue benchmark uses 150 real ZTF SN Ia DR2 events in Legacy Surveys DR10 south. The frozen cohort contains 98 training, 26 validation, and 26 test events. The comparison includes nearest neighbour, Tractor DLR, an offset-only Pröst run, and a logistic catalogue model. The initial DESI audit found insufficient overlap with transient-template redshifts. [Frozen split](split-v1.json) · [Catalogue validation](validation-v1.md) · [Pröst](prost-v1.md) · [Spectroscopy audit](spectroscopy-v1.md)

## Phase 1 — Test FITS image features

Completed on the frozen v1 cohort. [Image validation report](image-v1.md) · [FITS input audit](image-inputs-v1.csv)

Fetch a DR10 south r-band science image and inverse-variance map for each training and validation field. Verify WCS alignment and log missing or failed cutouts. Derive simple candidate-centred features from masked pixel apertures at the galaxy centre and toward the SN. Retain missing measurements and the full catalogue candidate pool.

Compare catalogue-only and catalogue-plus-image logistic rankers on the same validation events, using identical training events, labels, and ranking rules. Report coverage, top-1 and top-3 agreement, and the paired top-1 change with a sky-group bootstrap interval. Phase completion requires versioned code and results, a runnable command, an offline replay, and a position/coverage check. The test split stays reserved.

## Phase 2 — Broaden the spectroscopy audit

Search public spectroscopic releases beyond DESI Main Survey bright and dark, beginning with other DESI programmes and SDSS where coverage applies. Track candidate match ambiguity, uncertainties, quality flags, reference frames, and target-selection provenance. Admit SN-side redshifts only when derived from the transient independently of its assigned host. Run a redshift comparison on the same covered validation events if the evidence supports it. Otherwise report the coverage constraints and keep the redshift result exploratory.

## Phase 3 — Freeze a decision policy and run the reserved test

Use validation to specify the feature sets, missing-data behaviour, and rules for automatic assignment, review, and possible missing host. Expose the frozen ranking and review policy through the existing RA and Dec entry point. Record the complete protocol before scoring the reserved events. Compare retrieval and ranking separately, report uncertainty and accuracy versus coverage, and track crowded, faint, high-offset, and missing-data cases. Keep simulated source removal separate from evidence for real missing hosts. Use probability language only for scores with demonstrated calibration. If the reserved sample is too small for a stable comparison, collect another untouched cohort with a versioned sky-group split.

## Phase 4 — Investigate disputed hosts and publish the benchmark

Build an evidence gallery of about 20–30 disagreements or unresolved fields. Check deeper imaging, independent spectroscopy, and discovery or survey records. Keep the published reference, model proposal, and independent evidence traceable. Where independent evidence establishes a label, withhold that evidence from the corresponding ranking evaluation. Record supported, plausible, and unresolved verdicts. Publish the reproducible benchmark, an ablation report, and the gallery with its provenance and limitations.
