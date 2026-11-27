# Hermae catalogue ranking validation

The frozen split reserved 26 test events. This run scored 26 validation events and trained both logistic rankers on 92 events with one recovered match and at least one alternative candidate.

Within 60″, 23/25 validation events with published coordinates had a uniquely recovered catalogue counterpart. The comparisons use the same candidate pool. Agreement is measured against ZTF's published DLR-selected reference.

## Agreement with the published reference

- **Nearest angular neighbour** ranked 17/23 (73.9%) first and 20/23 (87.0%) in the top three. Confirmed first ranks across all published coordinates were 17/25.
- **Tractor half-light DLR** ranked 20/23 (87.0%) first and 23/23 (100.0%) in the top three. Confirmed first ranks across all published coordinates were 20/25.
- **Logistic regression, geometry** ranked 20/23 (87.0%) first and 23/23 (100.0%) in the top three. Confirmed first ranks across all published coordinates were 20/25.
- **Logistic regression, full catalogue** ranked 22/23 (95.7%) first and 23/23 (100.0%) in the top three. Confirmed first ranks across all published coordinates were 22/25.

## Paired change in first-rank agreement

- **Nearest angular neighbour versus Tractor half-light DLR** -13.0 percentage points (95% group bootstrap interval -27.3 to +0.0).
- **Logistic regression, geometry versus Tractor half-light DLR** +0.0 percentage points (95% group bootstrap interval +0.0 to +0.0).
- **Logistic regression, full catalogue versus Tractor half-light DLR** +8.7 percentage points (95% group bootstrap interval +0.0 to +21.4).
- **Logistic regression, full catalogue versus Logistic regression, geometry** +8.7 percentage points (95% group bootstrap interval +0.0 to +21.4).

Both models used only Tractor candidate observables and geometric offsets. Output scores are uncalibrated logistic decision values.
