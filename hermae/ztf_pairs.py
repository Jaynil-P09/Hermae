"""Prepare public ZTF development pairs using confirmed positive r-band alerts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from astropy.io import fits

from .development import Benchmark, digest, records

IRSA = "https://irsa.ipac.caltech.edu/ibe"
ALERCE = "https://api.alerce.online/ztf/v1/objects"


def get(url):
    with urlopen(url, timeout=30) as response:
        return response.read()


def field_archive(ra, dec, peak):
    query = urlencode(dict(POS=f"{ra},{dec}", INTERSECT="CENTER",
                          WHERE=f"obsjd>{peak-150} AND obsjd<{peak+55} AND filtercode='zr' AND infobits<33554432",
                          COLUMNS="field,ccdid,qid,filtercode,filefracday,imgtypecode,obsjd,seeing", CT="csv"))
    return list(csv.DictReader(io.StringIO(get(f"{IRSA}/search/ztf/products/sci?{query}").decode())))


def cutout_url(row, ra, dec, product):
    date = row["filefracday"]
    file = (f"ztf_{date}_{int(row['field']):06d}_zr_c{int(row['ccdid']):02d}_"
            f"{row['imgtypecode']}_q{row['qid']}_{product}.fits")
    return (f"{IRSA}/data/ztf/products/sci/{date[:4]}/{date[4:8]}/{date[8:]}/{file}?"
            f"center={ra},{dec}&size=170arcsec&gzip=false")


def exposure(out, row, ra, dec):
    path = out / f"{row['filefracday']}-{row['field']}-{row['ccdid']}-{row['qid']}.fits"
    urls = [cutout_url(row, ra, dec, name) for name in ("sciimg", "mskimg")]
    if not path.exists():
        with fits.open(io.BytesIO(get(urls[0])), memmap=False) as science, fits.open(
                io.BytesIO(get(urls[1])), memmap=False) as mask:
            if science[0].data.shape != mask[0].data.shape:
                raise ValueError("IRSA science and mask shape mismatch.")
            science[0].header["FILTER"] = "r"
            science[0].header["SEEING"] = float(row["seeing"])
            science[0].header["MJD-OBS"] = float(row["obsjd"]) - 2400000.5
            merged = fits.HDUList([science[0], fits.ImageHDU(mask[0].data, mask[0].header, name="MASK")])
            temp = path.with_suffix(".part")
            merged.writeto(temp, overwrite=True)
            temp.replace(path)
    return dict(path=str(path), sha256=digest(path), url=urls[0], mask_url=urls[1],
                obsjd=float(row["obsjd"]), seeing_arcsec=float(row["seeing"]))


def pairs_for(event, row, peak, out):
    ra, dec = row["ra"], row["dec"]
    archive = field_archive(ra, dec, peak)
    detections = json.loads(get(f"{ALERCE}/{event}/detections"))
    alerts = [r for r in detections if r["fid"] == 2 and int(r["isdiffpos"]) > 0 and r["rb"] >= .55
              and abs(r["mjd"] + 2400000.5 - peak) <= 55]
    alerts.sort(key=lambda a: (a["magpsf"], abs(a["mjd"] + 2400000.5 - peak)))
    for alert in alerts:
        matches = [r for r in archive if abs(float(r["obsjd"]) - (alert["mjd"] + 2400000.5)) < .00015
                   and 0 < float(r["seeing"]) < 4]
        for rec in matches:
            early = [r for r in archive if all(r[key] == rec[key] for key in ("field", "ccdid", "qid"))
                     and float(r["obsjd"]) < peak - 40 and 0 < float(r["seeing"]) < 4]
            early.sort(key=lambda r: abs(float(r["seeing"]) - float(rec["seeing"])))
            if not early:
                continue
            folder = out / event
            folder.mkdir(exist_ok=True)
            reference = exposure(folder, early[0], ra, dec)
            science = exposure(folder, rec, ra, dec)
            pairs = [dict(kind="positive", event=event, reference=reference, science=science,
                          alert_id=str(alert["candid"]), alert_ra=alert["ra"], alert_dec=alert["dec"],
                          host_ra=row["host_ra"], host_dec=row["host_dec"])]
            quiet = [r for r in early[1:] if float(r["obsjd"]) > float(early[0]["obsjd"]) + .1]
            if quiet:
                pairs.append(dict(kind="pre_event_control", event=event, reference=reference,
                                  science=exposure(folder, quiet[0], ra, dec)))
            return pairs
    raise ValueError("No public r-band alert exposure with a matching pre-event quadrant was found.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="maximum number of requested fields per development split")
    args = parser.parse_args()
    data = Benchmark(args.v1)
    out = data.output(args.out)
    raw = data.root / ".cache/ztf/snia_data.csv"
    peak = {r["ztfname"]: float(r["t0"]) + 2400000.5 for r in records(raw) if r["t0"]}
    jobs = [(split, event) for split in ("validation", "train") for event in data.ids(split)[:args.limit]]

    def prepare(job):
        split, event = job
        try:
            pairs = pairs_for(event, data.events[event], peak[event], out)
            return [dict(split=split, **pair) for pair in pairs], None
        except (OSError, ValueError, KeyError, IndexError) as exc:
            return [], str(exc)

    pairs, rejected = [], {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for (_, event), (found, error) in zip(jobs, executor.map(prepare, jobs)):
            pairs.extend(found)
            if error:
                rejected[event] = error
            print(f"{event}: {error or f'{len(found)} verified archived pairs'}", flush=True)
    manifest = dict(split_sha256=data.split_hash, peak_table_sha256=digest(raw), pairs=pairs, rejected=rejected)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
