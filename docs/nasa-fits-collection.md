# Supernova FITS field collection

The local collection is under `results/nasa-fields`. It contains 16 public
science FITS images and a [field manifest](nasa-fits-manifest.json) with the
archive URLs, checksums, target coordinates, and available uncertainty HDUs.
Use `results/nasa-fields/dashboard.html` for the five wide-field Hermae
overlays. The contact sheet at `results/nasa-fields/field-preview.png`
compares the wide optical fields and the ultraviolet view of M101.

| Event and host | What the field shows | FITS products |
| --- | --- | --- |
| [SN 2023ixf in M101](https://www.wis-tns.org/object/2023ixf) | Spiral arms and an off-centre supernova | Wide and detailed Legacy r, Pan-STARRS r, GALEX near UV |
| [SN 2021rhu in NGC 7814](https://www.wis-tns.org/object/2021rhu) | An edge-on galaxy and dust lane | Wide and detailed Legacy r, Pan-STARRS r |
| [SN 2020oi in M100](https://www.wis-tns.org/object/2020oi/discovery-cert) | Bright spiral nucleus close to the supernova | Wide and detailed Legacy r, Pan-STARRS r |
| [SN 2022hrs in NGC 4647](https://www.wis-tns.org/object/2022hrs) | Two bright galaxies in the same field | Wide and detailed Legacy r, Pan-STARRS r |
| [SN 2024ggi in NGC 3621](https://www.wis-tns.org/object/2024ggi) | An offset supernova in a southern spiral | Wide and detailed Legacy r, DSS2 Red |

The archived science images have celestial WCS, finite pixels at the SN
positions, and a sky-coordinate round-trip error below 0.001″. The five
detailed Legacy r cutouts have inverse-variance data in HDU 1. The wide
cutouts are resampled views and use the science image alone. The Pan-STARRS
images came from NASA MAST, and the GALEX and DSS2 products came from NASA
Goddard SkyView. [Pan-STARRS at MAST](https://archive.stsci.edu/missions-and-data/pan-starrs) ·
[NASA SkyView FITS service](https://skyview.gsfc.nasa.gov/current/docs/batchpage.html) ·
[Legacy Surveys imaging](https://www.legacysurvey.org/dr10/description/)

The collection is staged for visual exploration outside the frozen
training and evaluation cohorts. Dashboard host picks are provisional.
The NASA/IPAC IRSA queries for ZTF science pairs returned a metadata
database connection error on follow-up requests. The collection contains
coadds and survey images for known-position workflows. [NASA/IPAC ZTF archive](https://irsa.ipac.caltech.edu/Missions/ztf.html)
