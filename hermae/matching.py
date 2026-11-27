"""Nearest-neighbour and Tractor half-light DLR baselines."""

from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

from .survey import CATALOG_COLUMNS, GALAXIES, SourceError, fetch_sources

CANDIDATE_COLUMNS = [
    "event_id", "source_id", *CATALOG_COLUMNS, "separation_arcsec", "in_baseline_cone",
    "position_angle_deg", "axis_ratio", "major_pa_deg", "directional_radius_arcsec",
    "dlr", "shape_status", "nearest_rank", "dlr_rank",
]


def rank_sources(sources, ra, dec, radius):
    candidates = sources.loc[sources.type.isin(GALAXIES)].copy()
    galaxy = SkyCoord(candidates.ra.to_numpy(dtype=float), candidates.dec.to_numpy(dtype=float),
                      unit="deg")
    angle = galaxy.position_angle(SkyCoord(ra, dec, unit="deg")).rad
    circular = candidates.type.eq("REX")
    e1 = candidates.shape_e1.astype(float).where(~circular, 0.)
    e2 = candidates.shape_e2.astype(float).where(~circular, 0.)
    ellipticity = np.hypot(e1, e2)
    ratio = (1 - ellipticity) / (1 + ellipticity)
    major = np.arctan2(e2, e1) / 2
    size = candidates.shape_r.astype(float)
    valid = np.isfinite(size) & (size > 0) & (ellipticity < 1)
    radius_limit = (candidates.fitbits.astype(int) & (1 << 2)) != 0
    size_measured = np.isfinite(candidates.shape_r_ivar.astype(float)) & (candidates.shape_r_ivar > 0)
    ellipse_measured = (
        np.isfinite(candidates.shape_e1_ivar.astype(float)) & (candidates.shape_e1_ivar > 0)
        & np.isfinite(candidates.shape_e2_ivar.astype(float)) & (candidates.shape_e2_ivar > 0)
    )
    unmeasured = ~size_measured | (~circular & ~ellipse_measured)
    candidates["shape_status"] = np.select(
        [~valid, radius_limit, unmeasured],
        ["invalid", "radius_limit", "unmeasured_uncertainty"], default="ok")
    delta = angle - major
    directional = size.where(valid) / np.hypot(np.cos(delta), np.sin(delta) / ratio.where(valid))
    candidates["position_angle_deg"] = np.degrees(angle)
    candidates["axis_ratio"] = ratio.where(valid)
    candidates["major_pa_deg"] = (np.degrees(major) % 180).where(valid & (ellipticity > 0))
    candidates["directional_radius_arcsec"] = directional
    candidates["dlr"] = candidates.separation_arcsec / directional
    candidates["in_baseline_cone"] = candidates.separation_arcsec <= radius
    baseline = candidates.loc[candidates.in_baseline_cone]
    nearest = baseline.sort_values(["separation_arcsec", "source_id"]).index
    dlr = baseline.sort_values(["dlr", "separation_arcsec", "source_id"], na_position="last").index
    for name, order in (("nearest_rank", nearest), ("dlr_rank", dlr)):
        candidates[name] = pd.Series(range(1, len(order) + 1), index=order, dtype="Int64")
    return candidates.sort_values(["separation_arcsec", "source_id"])


def reference_match(sources, event, tolerance, max_radius):
    result = dict(reference_status="missing_coordinates", reference_source_id=None,
                  reference_brick=None, reference_type=None, reference_offset_arcsec=None,
                  host_separation_arcsec=None, reference_matches=None, reference_match_ids=None,
                  reference_duplicate_rows=None)
    if pd.isna(event.host_ra) or pd.isna(event.host_dec):
        return result
    host = SkyCoord(event.host_ra, event.host_dec, unit="deg")
    offset = host.separation(SkyCoord(event.ra, event.dec, unit="deg")).arcsec
    result["host_separation_arcsec"] = offset
    if offset > max_radius + tolerance:
        result["reference_status"] = "outside_max_search"
        return result
    sky = SkyCoord(sources.ra.to_numpy(dtype=float), sources.dec.to_numpy(dtype=float), unit="deg")
    distances = host.separation(sky).arcsec
    matches = sources.assign(host_offset=distances).loc[distances <= tolerance]
    matches = matches.sort_values(["host_offset", "source_id"])
    result["reference_match_ids"] = "|".join(matches.source_id)
    duplicate = matches.type.eq("DUP")
    result["reference_duplicate_rows"] = int(duplicate.sum())
    matches = matches.loc[~duplicate]
    result["reference_matches"] = len(matches)
    result["reference_status"] = "unmatched"
    if matches.empty and duplicate.any():
        result.update(reference_status="duplicate_only", reference_type="DUP")
    if len(matches) > 1:
        result["reference_status"] = "ambiguous"
    if len(matches) == 1:
        source = matches.iloc[0]
        status = "unique_galaxy" if source.type in GALAXIES else "excluded_morphology"
        if source.separation_arcsec > max_radius:
            status = "outside_max_search"
        result.update(reference_status=status, reference_source_id=source.source_id,
                      reference_brick=source.brickname, reference_type=source.type,
                      reference_offset_arcsec=source.host_offset)
    return result


def pick(candidates, method):
    selected = candidates.loc[candidates[f"{method}_rank"] == 1]
    return None if selected.empty else selected.iloc[0]


def review(candidates, nearest, dlr):
    if nearest is None:
        return "possible_missing_host", "no_galaxy_in_baseline_cone"
    if pd.isna(dlr.dlr):
        return "review", "dlr_uses_nearest_fallback"
    if dlr.dlr > 7:
        return "possible_missing_host", "all_dlr_scores_above_7"
    ranked = candidates.loc[candidates.in_baseline_cone, "dlr"].dropna().nsmallest(2)
    reasons = []
    if dlr.shape_status != "ok":
        reasons.append("dlr_pick_shape_flagged")
    if nearest.source_id != dlr.source_id:
        reasons.append("methods_disagree")
    if len(ranked) == 2 and ranked.iloc[1] <= 1.2 * ranked.iloc[0]:
        reasons.append("close_dlr_scores")
    return ("review", "|".join(reasons)) if reasons else ("unflagged", "")


def evaluate(cache, event, radii, tolerance):
    baseline, maximum = radii[0], max(radii)
    row = {"event_id": event.event_id, "host_ra": event.host_ra, "host_dec": event.host_dec,
           "catalogue_status": "ok", "query_error": "",
           "source_count_max_radius": None, "reference_status": "query_failed",
           "reference_source_id": None, "reference_brick": None, "reference_type": None,
           "reference_offset_arcsec": None, "host_separation_arcsec": None,
           "reference_matches": None, "reference_match_ids": None, "reference_duplicate_rows": None}
    for radius in radii:
        row[f"candidate_count_{radius}as"] = None
        row[f"reference_retrieved_{radius}as"] = None
    row.update(nearest_pick=None, nearest_ra=None, nearest_dec=None, dlr_pick=None,
               dlr_ra=None, dlr_dec=None, best_dlr=None, dlr_fallback=None, nearest_reference_rank=None,
               dlr_reference_rank=None, review_status="review", review_reason="catalogue_query_failed")
    try:
        sources = fetch_sources(cache, event.ra, event.dec, maximum + max(1.5, tolerance))
    except SourceError as exc:
        row.update(catalogue_status="query_failed", query_error=str(exc))
        return row, pd.DataFrame(columns=CANDIDATE_COLUMNS)

    row["source_count_max_radius"] = int((sources.separation_arcsec <= maximum).sum())
    if not row["source_count_max_radius"]:
        row["catalogue_status"] = "empty_sources"
    row.update(reference_match(sources, event, tolerance, maximum))
    sources = sources.loc[sources.separation_arcsec <= maximum]
    candidates = rank_sources(sources, event.ra, event.dec, baseline)
    candidates["event_id"] = event.event_id
    reference = candidates.loc[candidates.source_id == row["reference_source_id"]]
    for radius in radii:
        row[f"candidate_count_{radius}as"] = int((candidates.separation_arcsec <= radius).sum())
        if row["reference_status"] not in {"missing_coordinates", "ambiguous"}:
            row[f"reference_retrieved_{radius}as"] = bool((reference.separation_arcsec <= radius).any())
    if row["reference_status"] == "unique_galaxy":
        for method in ("nearest", "dlr"):
            row[f"{method}_reference_rank"] = reference.iloc[0][f"{method}_rank"]
    nearest = pick(candidates, "nearest")
    dlr = pick(candidates, "dlr")
    if nearest is not None:
        row.update(nearest_pick=nearest.source_id, nearest_ra=nearest.ra, nearest_dec=nearest.dec,
                   dlr_pick=dlr.source_id, dlr_ra=dlr.ra, dlr_dec=dlr.dec,
                   dlr_fallback=bool(pd.isna(dlr.dlr)), best_dlr=dlr.dlr)
    row["review_status"], row["review_reason"] = review(candidates, nearest, dlr)
    return row, candidates[CANDIDATE_COLUMNS]
