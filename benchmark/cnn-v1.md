# Pixel CNN and catalogue fusion

The three listwise rankers used 92 informative training fields and scored the same 26 validation fields and 1,425 candidates. The reserved test set remained untouched.
The CNN received usable pixels for 1,319 validation candidates. For missing stamps the CNN score term is zero. CNN-only ties use angular separation.

## Agreement among 23 uniquely recovered published counterparts

- **Catalogue logistic regression** ranked 22/23 first and 23/23 in the top three.
- **Catalogue + measured apertures** ranked 23/23 first and 23/23 in the top three.
- **Matched catalogue control** ranked 19/23 first and 20/23 in the top three.
- **CNN only** ranked 22/23 first and 23/23 in the top three.
- **Catalogue + CNN** ranked 20/23 first and 23/23 in the top three.

The paired first-rank change from adding the CNN to the matched control was +4.3 percentage points (95% group bootstrap interval -10.5 to +16.7).

The CPU experiment used PyTorch 2.10.0+cpu, seed 42, 14 epochs, AdamW with learning rate 0.003, weight decay 0.01, and 8 fields per update.
It used r-band stamps centred on the candidate, SN and midpoint, each with a coverage mask. The three stamps share two convolutional layers. The fused score adds a learned image term to the catalogue score. Redshift was omitted because the initial audit found insufficient independent overlap. Published ZTF counterparts remain catalogue references.
