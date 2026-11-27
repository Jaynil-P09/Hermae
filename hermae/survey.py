"""Public ZTF SN Ia DR2 and Legacy Surveys DR10 south data access."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic, sleep

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

ZTF = "https://ztfcosmo.in2p3.fr/download"
VIEWER = "https://www.legacysurvey.org/viewer"
LAYER = "ls-dr10-south"
ARCHIVE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
BRICKS = f"{ARCHIVE}/dr10/south/survey-bricks-dr10-south.fits.gz"
PIX_SCALE = 0.262
BANDS = "grz"
GALAXIES = {"REX", "EXP", "DEV", "SER"}
CATALOG_COLUMNS = [
    "release", "brickid", "brickname", "objid", "brick_primary", "ra", "dec", "type",
    "shape_r", "shape_e1", "shape_e2", "shape_r_ivar", "shape_e1_ivar", "shape_e2_ivar",
    "sersic", "maskbits", "fitbits", "ref_cat", "ref_id",
    *[f"{field}_{band}" for field in
      ("flux", "flux_ivar", "nobs", "anymask", "allmask", "fracflux") for band in "griz"],
]


class SourceError(RuntimeError):
    """The upstream result could not be retrieved or validated."""


class Cache:
    def __init__(self, root, offline=False):
        self.root = Path(root)
        self.offline = offline
        self._viewer_lock = Lock()
        self._viewer_next = 0.

    def _pace_viewer(self, delay=0.):
        with self._viewer_lock:
            now = monotonic()
            if delay:
                self._viewer_next = max(self._viewer_next, now + delay)
                return
            sleep(max(0., self._viewer_next - now))
            self._viewer_next = monotonic() + 1.5

    def get(self, url, group, suffix, params=None, missing_ok=False):
        url = requests.Request("GET", url, params=params).prepare().url
        path = self.root / group / (sha256(url.encode()).hexdigest() + suffix)
        if path.is_file():
            return path
        missing = path.with_suffix(path.suffix + ".not-found")
        if missing_ok and missing.is_file():
            return None
        if self.offline:
            raise SourceError(f"Not cached: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(502, 503, 504),
                      respect_retry_after_header=False)
        try:
            with requests.Session() as session:
                session.mount("https://", HTTPAdapter(max_retries=retry))
                for attempt in range(4):
                    if url.startswith(VIEWER + "/"):
                        self._pace_viewer()
                    with session.get(url, timeout=(10, 60), stream=True) as response:
                        if missing_ok and response.status_code == 404:
                            missing.touch()
                            return None
                        if response.status_code == 429 and url.startswith(VIEWER + "/") and attempt < 3:
                            retry_after = response.headers.get("Retry-After", "")
                            delay = max(5 * 2**attempt, int(retry_after) if retry_after.isdigit() else 0)
                            self._pace_viewer(delay)
                            continue
                        response.raise_for_status()
                        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
                            temp = Path(output.name)
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                        break
            temp.replace(path)
        except (OSError, requests.RequestException) as exc:
            raise SourceError(f"{url}: {exc}") from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return path


def load_events(cache):
    files = {}
    for name in ("snia_data", "globalhost_data"):
        files[name] = cache.get(f"{ZTF}/tables__{name}.csv", "release", ".csv")
    sn_columns = ["ztfname", "ra", "dec", "iau_name", "sn_type", "sub_type",
                  "redshift", "redshift_err", "source"]
    host_columns = ["ztfname", "ra_host", "dec_host"]
    sn = pd.read_csv(files["snia_data"], usecols=sn_columns)
    hosts = pd.read_csv(files["globalhost_data"], usecols=host_columns)
    events = sn.merge(hosts, on="ztfname", how="left", validate="one_to_one",
                      indicator="host_table_join")
    events = events.rename(columns={"ztfname": "event_id", "source": "redshift_source",
                                    "ra_host": "host_ra", "dec_host": "host_dec"})
    events["redshift_provenance"] = events.redshift_source.map({
        "z_gal": "galaxy_catalogue",
        "z_nonSEDm": "galaxy_lines_non_SEDm",
        "z_SEDm": "galaxy_lines_SEDm",
        "z_snid": "transient_template",
    }).fillna("unknown")
    events["label_provenance"] = np.where(
        events[["host_ra", "host_dec"]].notna().all(axis=1), "ZTF_DR2_DLR", "missing")
    return events, files


def footprint_bricks(path):
    with fits.open(path) as hdus:
        table = hdus[1].data
        order = np.argsort(table["dec1"])
        ra1, ra2, dec1, dec2 = (np.array(table[key][order]) for key in ("ra1", "ra2", "dec1", "dec2"))
    return ra1, (ra2 - ra1) % 360, dec1, dec2, np.max(dec2 - dec1)


def has_brick(ra, dec, bounds):
    ra1, span, dec1, dec2, max_height = bounds
    start = np.searchsorted(dec1, dec - max_height - 1e-10)
    stop = np.searchsorted(dec1, dec, side="right")
    rows = slice(start, stop)
    return np.any((dec <= dec2[rows]) & ((ra - ra1[rows]) % 360 <= span[rows]))


def cutout(cache, ra, dec, size, pixscale=PIX_SCALE, coverage=False):
    params = dict(ra=f"{ra:.8f}", dec=f"{dec:.8f}", layer=LAYER, bands=BANDS,
                  size=size, pixscale=pixscale)
    if coverage:
        params["invvar"] = 1
    return cache.get(f"{VIEWER}/fits-cutout", "coverage" if coverage else "images",
                     ".fits", params)


def point_coverage(cache, event):
    """Check for positive inverse variance at the SN in any optical grz plane."""
    try:
        path = cutout(cache, event.ra, event.dec, 9, coverage=True)
        with fits.open(path) as hdus:
            wcs = WCS(hdus[0].header, naxis=2)
            pixel = wcs.world_to_pixel(SkyCoord(event.ra, event.dec, unit="deg"))
            x, y = np.rint(pixel).astype(int)
            weights = hdus[1].data
            if (hdus[1].header.get("BANDS") != BANDS or weights.shape != (3, 9, 9)
                    or not (0 <= x < 9 and 0 <= y < 9)):
                raise SourceError("Unexpected inverse-variance cutout geometry or bands")
            present = "".join(band for band, weight in zip(BANDS, weights[:, y, x])
                              if np.isfinite(weight) and weight > 0)
    except (SourceError, OSError, ValueError, IndexError, TypeError) as exc:
        return "coverage_query_failed", "", str(exc)
    return ("covered" if present else "no_grz_weight"), present, ""


def cone_boxes(ra, dec, radius):
    degrees = radius / 3600
    low, high = max(-90, dec - degrees), min(90, dec + degrees)
    if low <= -90 or high >= 90:
        return [(start, start + 1, low, high) for start in range(360)]
    half_width = np.degrees(np.arcsin(np.sin(np.radians(degrees)) / np.cos(np.radians(dec))))
    left, right = ra - half_width, ra + half_width
    if left < 0:
        return [(left + 360, 360, low, high), (0, right, low, high)]
    if right > 360:
        return [(left, 360, low, high), (0, right - 360, low, high)]
    return [(left, right, low, high)]


def fetch_sources(cache, ra, dec, radius):
    """Retrieve DR10 Tractor rows in bounding boxes, then apply a spherical cone."""
    frames = []
    for left, right, low, high in cone_boxes(ra, dec, radius):
        # Pad the viewer's TAN rectangle before applying the exact cone locally.
        pad = 0.001
        params = dict(ralo=max(0., left - pad), rahi=min(360., right + pad),
                      declo=max(-90., low - pad), dechi=min(90., high + pad))
        path = cache.get(f"{VIEWER}/{LAYER}/cat.fits", "tractor", ".fits", params)
        try:
            with fits.open(path) as hdus:
                table = hdus[1].data
                frame = pd.DataFrame({key: np.asarray(table[key], dtype=table[key].dtype.newbyteorder("="))
                                      for key in CATALOG_COLUMNS})
        except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
            if path.stat().st_size <= 128 and path.read_bytes().strip() in {
                b"No sources", b"No catalog sources in layer and RA,Dec box",
            }:
                continue
            raise SourceError(f"Invalid Tractor response {path}: {exc}") from exc
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*CATALOG_COLUMNS, "source_id", "separation_arcsec"])
    sources = pd.concat(frames, ignore_index=True)
    sources = sources.loc[sources.brick_primary].drop_duplicates(["release", "brickid", "objid"]).copy()
    sources["source_id"] = (sources.release.astype(str) + ":" + sources.brickid.astype(str)
                            + ":" + sources.objid.astype(str))
    for column in ("type", "brickname", "ref_cat"):
        sources[column] = sources[column].str.strip()
    sky = SkyCoord(sources.ra.to_numpy(), sources.dec.to_numpy(), unit="deg")
    sources["separation_arcsec"] = SkyCoord(ra, dec, unit="deg").separation(sky).to_value(u.arcsec)
    return sources.loc[sources.separation_arcsec <= radius].reset_index(drop=True)
