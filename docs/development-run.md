# Phase 1 development run

`ZTF18abvtiko` is in the frozen v1 development validation split. The 511 × 511 r-band FITS uses science HDU 0 and inverse-variance HDU 1. The input SHA-256 is `54679b197fc38e5fbf999e284e81add4f44e7b39898ee8015fc86ea4669bc6f3`. The SN position (RA 71.884741°, Dec −2.306897° ICRS) falls at approximately (255, 255) on the zero-based FITS array. The run used the v2 detector from `fbdd113` and verified v1 references from `52b92cc`.

Follow the [README example](../README.md#reproduce-the-development-example) from the v2 worktree with the two checkouts side by side. The audited FITS cache sits in the v1 checkout. The command writes all six outputs to `results/ZTF18abvtiko/` in the v2 checkout. These generated files stay local.

## Result

- The full frame contained **63 detected footprints**, with **47 candidates** inside the 60″ host cone.
- Both provisional rankings placed source 58 first. Its centroid is **0.124″** from the SN and **0.176″** from the v1 DLR pick.
- The v1 measured-FITS pick agrees with the published counterpart, **16.134″** from the v2 provisional host. Zero v2 centroids matched the published position at the recorded 1.5″ tolerance. The overlay shows segmentation boundaries through the bright region containing that counterpart. This is a centroid retrieval miss under the matching rule, so the field cannot measure ranking accuracy conditional on host recovery.
- With `--tile 256 --overlap 64`, the same input produced **75 detections** and **59 candidates**, kept its first pick within 0.101″ of the SN, and still had zero centroid matches to the published counterpart. Source counts on this field depend on tiling.

## Verification

The overlay was visually checked for footprint alignment and the placement of the SN cross, provisional-host diamond, and published-counterpart star. Reading back the segmentation FITS confirmed all 63 label IDs and pixel counts against the detection table. The maximum sky-to-pixel centroid round-trip error was below `3e-10` pixels. A second run at the default settings reproduced all six files byte for byte on the same machine.

The test suite covers WCS and masks, adjacent-source deblending and a footprint that crosses a tile boundary, full-frame output when the SN is outside coverage, a covered field with an empty host cone, and rejection of reserved v1 comparisons. The 26 reserved fields were kept out of the v2 development run.
