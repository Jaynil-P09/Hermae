# Pröst offset-only comparison

Pröst 1.2.13 scored 1,425 of 1,425 candidates from the shared 60″ cones for 26 validation events.
The run used 1,000 Monte Carlo draws per field and base seed 42. The offset prior is uniform on [0, 10], the likelihood is Gamma with shape 0.75 and scale 1, and the nominal transient coordinate uncertainty is 0.1″ per axis.

Agreement with the published ZTF reference among 23 uniquely recovered counterparts.

- **Nearest angular neighbour** ranked 17/23 first and 20/23 in the top three.
- **Tractor half-light DLR** ranked 20/23 first and 23/23 in the top three.
- **Pröst median candidate posterior** ranked 19/23 first and 23/23 in the top three.

Pröst's native Monte Carlo winner agreed for 19/23 recovered counterparts and returned no candidate in 0/26 validation fields.

This run calls upstream shape sampling and association on the shared Hermae candidates. The stock DECaLS query joins photo-z data and applies its own cuts. This offset experiment uses candidate shapes and separation only. Independent spectroscopy coverage is reported separately.

[Pröst reference notebook](https://github.com/alexandergagliano/Prost/blob/main/docs/notebooks/associate.ipynb)
