"""Audit DESI DR1 main-program spectroscopy for frozen validation candidates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path

from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord, search_around_sky
from astropy.io import fits
from astropy_healpix import HEALPix
import numpy as np
import pandas as pd

from .splits import validation_selection
from .survey import Cache, SourceError

ARCHIVE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/main"
HEALPIX = HEALPix(nside=64, order="nested", frame=ICRS())
PROGRAMS = ("bright", "dark")
TOLERANCE = 1.5 * u.arcsec
REDSHIFTS = ["TARGETID", "Z", "ZERR", "ZWARN", "SPECTYPE", "DELTACHI2"]
FIBERS = ["TARGET_RA", "TARGET_DEC", "DESI_TARGET", "BGS_TARGET", "SCND_TARGET", "COADD_FIBERSTATUS"]
OBS_COLS = [field.lower() for field in REDSHIFTS + FIBERS] + ["program", "healpix"]


def archive_file(cache, program, pixel):
    url = f"{ARCHIVE}/{program}/{pixel // 100}/{pixel}/redrock-main-{program}-{pixel}.fits"
    audit = dict(program=program, healpix=pixel, url=url, status="downloaded", sha256="", error="")
    try:
        path = cache.get(url, "desi_dr1", ".fits", missing_ok=True)
        if path is None:
            audit["status"] = "no_public_file"
            return audit, None
        with fits.open(path) as hdus:
            redshifts, fibers = hdus["REDSHIFTS"].data, hdus["FIBERMAP"].data
            if not np.array_equal(redshifts["TARGETID"], fibers["TARGETID"]):
                raise SourceError("DESI REDSHIFTS and FIBERMAP target IDs are misaligned.")
            columns = {
                field.lower(): np.asarray(table[field], dtype=table[field].dtype.newbyteorder("="))
                for table, fields in ((redshifts, REDSHIFTS), (fibers, FIBERS)) for field in fields
            }
        observations = pd.DataFrame(columns).assign(program=program, healpix=pixel)
        observations["targetid"] = observations.targetid.astype(str)
        observations["spectype"] = observations.spectype.str.strip()
        with path.open("rb") as stream:
            audit["sha256"] = file_digest(stream, "sha256").hexdigest()
        return audit, observations
    except (SourceError, OSError, ValueError, KeyError) as exc:
        audit.update(status="query_failed", error=str(exc))
        return audit, None


def spatial_matches(candidates, observations):
    columns = ["event_id", "source_id", *OBS_COLS, "offset_arcsec", "usable_redshift"]
    if candidates.empty or observations.empty:
        return pd.DataFrame(columns=columns)
    galaxy = SkyCoord(candidates.ra.to_numpy(), candidates.dec.to_numpy(), unit="deg")
    spectra = SkyCoord(observations.target_ra.to_numpy(), observations.target_dec.to_numpy(), unit="deg")
    a, b, distance, _ = search_around_sky(galaxy, spectra, TOLERANCE)
    matches = observations.iloc[b].reset_index(drop=True)
    matches["event_id"] = candidates.event_id.iloc[a].to_numpy()
    matches["source_id"] = candidates.source_id.iloc[a].to_numpy()
    matches["offset_arcsec"] = distance.arcsec
    matches["usable_redshift"] = (
        matches.spectype.eq("GALAXY") & matches.zwarn.eq(0) & matches.coadd_fiberstatus.eq(0)
        & np.isfinite(matches.z) & matches.z.gt(0) & np.isfinite(matches.zerr) & matches.zerr.gt(0)
    )
    return matches[columns]


def candidate_audit(candidates, matches, complete):
    output = candidates[["event_id", "source_id"]].copy()
    targets = matches.groupby("source_id").targetid.nunique()
    shared = matches.groupby("targetid").source_id.nunique()
    ambiguous = set(matches.loc[matches.targetid.isin(shared[shared > 1].index), "source_id"])
    rows = []
    for key, group in matches.groupby("source_id"):
        if targets[key] != 1 or key in ambiguous:
            rows.append(dict(source_id=key, status="ambiguous_match"))
            continue
        good = group.loc[group.usable_redshift].sort_values(["deltachi2", "program", "healpix"],
                                                           ascending=[False, True, True])
        if good.empty:
            rows.append(dict(source_id=key, status="no_usable_galaxy_redshift"))
            continue
        best = good.iloc[0]
        discordant = (good.z.sub(best.z).abs() > np.maximum(0.001, 5 * np.hypot(good.zerr, best.zerr))).any()
        if discordant:
            rows.append(dict(source_id=key, status="discordant_redshifts"))
            continue
        secondary = (group.scnd_target.ne(0) | ((group.desi_target.astype("int64") & (1 << 62)) != 0)).any()
        rows.append(dict(source_id=key, status="usable_secondary" if secondary else "usable_no_secondary_flag",
                         targetid=best.targetid, z=best.z, zerr=best.zerr, program=best.program))
    columns = ["source_id", "status", "targetid", "z", "zerr", "program"]
    output = output.merge(pd.DataFrame(rows, columns=columns), on="source_id", how="left",
                          validate="one_to_one")
    output["status"] = output.status.fillna("no_match")
    if not complete:
        output["status"] = "incomplete_archive"
        output[["targetid", "z", "zerr", "program"]] = None
    return output


def audit(run, split, cache, workers):
    ids, radius = validation_selection(run, split)
    events = pd.read_csv(run / "events.csv", usecols=["event_id", "ra", "dec", "redshift_provenance"])
    events = events.set_index("event_id").loc[ids].reset_index()
    candidates = pd.read_csv(run / "candidates.csv",
                             usecols=["event_id", "source_id", "ra", "dec", "in_baseline_cone"])
    candidates = candidates.loc[candidates.event_id.isin(ids) & candidates.in_baseline_cone]
    pixels = {row.event_id: HEALPIX.cone_search_skycoord(SkyCoord(row.ra, row.dec, unit="deg"),
                                                       radius * u.arcsec + TOLERANCE)
              for row in events.itertuples()}
    requests = sorted({(program, pixel) for members in pixels.values() for pixel in members
                       for program in PROGRAMS})
    with ThreadPoolExecutor(workers) as pool:
        archive = dict(zip(requests, pool.map(lambda item: archive_file(cache, *item), requests)))
    files, summaries, pairs, fields = [item[0] for item in archive.values()], [], [], []
    for event in events.itertuples():
        blocks = [archive[(program, pixel)] for program in PROGRAMS for pixel in pixels[event.event_id]]
        frames = [frame for _, frame in blocks if frame is not None]
        observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OBS_COLS)
        complete = all(record["status"] != "query_failed" for record, _ in blocks)
        pool_rows = candidates.loc[candidates.event_id.eq(event.event_id)]
        matches = spatial_matches(pool_rows, observations)
        summary = candidate_audit(pool_rows, matches, complete)
        summaries.append(summary)
        pairs.append(matches)
        fields.append(dict(event_id=event.event_id, archive_complete=complete,
                           redshift_provenance=event.redshift_provenance, candidates=len(pool_rows),
                           candidates_with_usable_redshift=int(summary.status.str.startswith("usable_").sum()),
                           candidates_without_secondary_flag=int(summary.status.eq("usable_no_secondary_flag").sum())))
    return pd.DataFrame(files), pd.concat(summaries), pd.concat(pairs), pd.DataFrame(fields)


def report(files, candidates, fields):
    independent_sn = fields.redshift_provenance.eq("transient_template")
    available = fields.candidates_without_secondary_flag.gt(0)
    alternatives = fields.candidates_without_secondary_flag.ge(2)
    missing = files.status.eq("no_public_file").sum()
    failed = files.status.eq("query_failed").sum()
    return "\n".join([
        "# DESI DR1 spectroscopy coverage", "",
        f"The audit searched {len(fields)} validation fields and {len(candidates):,} candidates in "
        "their baseline cones. It used the public DESI DR1 Main Survey bright and dark Redrock files.",
        f"The archive returned {files.status.eq('downloaded').sum()} files, {missing} HTTP 404 responses "
        f"for absent files, and {failed} failed requests.", "",
        f"- {fields.candidates_with_usable_redshift.sum()} candidates in "
        f"{fields.candidates_with_usable_redshift.gt(0).sum()} fields had an unambiguous, usable galaxy redshift.",
        f"- {fields.candidates_without_secondary_flag.sum()} candidates in {available.sum()} fields "
        "had a usable redshift and no DESI secondary-target flag.",
        f"- {independent_sn.sum()} fields had a ZTF transient-template redshift. "
        f"{(independent_sn & available).sum()} of them also had at least one usable candidate redshift "
        f"without a secondary-target flag, and {(independent_sn & alternatives).sum()} had two or more.", "",
        "A missing secondary flag is a provenance screen. Field-level independence still requires checking "
        "targeting and the SN measurement. These counts describe coverage and leave host identities unchanged.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be at least one")
    files, candidates, matches, fields = audit(options.run, options.split,
                                                Cache(options.cache, options.offline), options.workers)
    options.output.mkdir(parents=True, exist_ok=True)
    for name, table in (("archive", files), ("candidate_spectroscopy", candidates),
                        ("spectral_matches", matches), ("field_spectroscopy", fields)):
        table.to_csv(options.output / f"{name}.csv", index=False)
    (options.output / "spectroscopy_report.md").write_text(report(files, candidates, fields), encoding="utf-8")
    if files.status.eq("query_failed").any():
        raise SystemExit("Some DESI archive requests failed. See archive.csv and retry.")
    print(f"Spectroscopy coverage written to {options.output}.")


if __name__ == "__main__":
    main()
