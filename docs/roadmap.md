# Hermae v2 phases

0. Preserve the v1 worktree and audit the frozen split, benchmark inputs, model preprocessing, and checkpoint eligibility.
1. Deliver known-position full-frame FITS detection, provisional ranking, and a verified development overlay.
2. Create real and synthetic galaxy footprint and centre labels. Benchmark a two-head U-Net against classical segmentation and select thresholds on development data.
3. Train an image-first host ranker over detected candidates and compare image-only, catalogue-only, and union candidate pools. Freeze a review and missing-host policy on development data.
4. Align reference and science exposures, detect difference-image transients, and compare a classifier against the subtraction baseline. Rank hosts using the reference image.
5. Connect saved CLI outputs to a dashboard for known-position and image-pair modes.
6. Lock code, weights, and thresholds. Open the reserved known-position cohort and a separate reserved image-pair cohort for final evaluation, failure analysis, and release.

Use the preserved v1 sky-group split for v2 development. Keep the 26 reserved events out of fitting, normalization across fields, model selection, thresholds, synthetic backgrounds, and manual debugging. WCS positions and supervised references can validate a pipeline stage after its detections are produced.
