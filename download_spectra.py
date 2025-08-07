
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_spectra.py

Download APOGEE DR17 apStar/asStar spectra, GALAH DR3 HERMES (B,G,R,I) spectra,
and Gaia-ESO (UVES) Phase 3 spectra for a list of cross-matched stars.

Inputs: a CSV/TSV with columns for the relevant identifiers:
- APOGEE_ID   (e.g., 2M17335483-2753043)
- GALAH_ID    (GALAH DR3 sobject_id, e.g., 170418003701205)
- GES_ID      (Gaia-ESO "CNAME" or target name; or provide RA/Dec columns)

Public endpoints used (no credentials required):
- SDSS DR17 SAS (apStar/asStar): see bulk download docs.
- Data Central SSA for GALAH DR3: https://datacentral.org.au/vo/slink/links
- ESO Archive (Phase 3) via astroquery.eso for Gaia-ESO

Examples:
    python download_spectra.py \
        --input crossmatch.csv \
        --apogee-col APOGEE_ID \
        --galah-col sobject_id \
        --gaiaeso-col GES_CNAME \
        --out ./data \
        --surveys apogee galah gaiaeso \
        --limit 100

Notes:
- APOGEE paths require TELESCOPE (apo25m/lco25m) and FIELD (e.g. 000+02).
  We obtain these from the DR17 allStar summary file (downloaded once and cached)
  then construct the SAS URL accordingly.
- GALAH fetches up to four camera files per sobject_id using the SSA "slink" endpoint.
- Gaia-ESO uses astroquery.eso to query Phase 3 "GaiaESO" survey and retrieve the files.
  For 30k objects you should batch (see --chunk-size) and consider running overnight.

Author: (your name)
"""
import argparse
import csv
import os
import sys
import time
import math
import json
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x

# Optional imports (Gaia-ESO)
try:
    from astroquery.eso import Eso
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except Exception:
    ASTROQUERY_AVAILABLE = False

# -------------------------------
# Utilities
# -------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunked(seq: Iterable, n: int) -> Iterable[List]:
    """Yield successive n-sized chunks from seq."""
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def make_session(timeout: int = 60, retries: int = 5, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request = _request_with_timeout(s.request, timeout=timeout)
    return s

def _request_with_timeout(original_request, timeout: int):
    def wrapper(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)
    return wrapper

def sha1sum(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------------
# APOGEE DR17: apStar/asStar downloader
# -------------------------------

APOGEE_ALLSTAR_URL = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits"
APOGEE_STARS_BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars"

def load_or_download_allstar(cache_dir: Path) -> Path:
    """
    Download the DR17 allStar table once (if not already cached).
    Returns the local path to the FITS file.
    """
    ensure_dir(cache_dir)
    target = cache_dir / "allStar-dr17-synspec_rev1.fits"
    if target.exists() and target.stat().st_size > 10_000_000:  # sanity
        return target
    print(f"[APOGEE] Downloading allStar to {target} ...")
    with requests.get(APOGEE_ALLSTAR_URL, stream=True) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return target

def build_apogee_url(telescope: str, field: str, apogee_id: str) -> str:
    # telescope is 'apo25m' or 'lco25m' (from allStar).
    # File prefix depends on telescope: apStar for apo25m, asStar for lco25m.
    prefix = "apStar" if "apo25m" in telescope else "asStar"
    return f"{APOGEE_STARS_BASE}/{telescope}/{field}/{prefix}-dr17-{apogee_id}.fits"

def map_apogee_to_path(allstar_fits: Path, apogee_ids: List[str]) -> pd.DataFrame:
    """
    Read the allStar FITS and return a DataFrame with APOGEE_ID, TELESCOPE, FIELD, URL.
    """
    from astropy.io import fits
    with fits.open(allstar_fits, memmap=True) as hdul:
        data = hdul[1].data  # Table
        # Column names vary in case; normalize
        cols = {c.lower(): c for c in data.columns.names}
        # Expect columns: APOGEE_ID, TELESCOPE, FIELD (sometimes FIELD appears twice)
        apogee_col = cols.get("apogee_id", "APOGEE_ID")
        telescope_col = cols.get("telescope", "TELESCOPE")
        field_col = cols.get("field", "FIELD")
        # Extract minimal mapping
        df = pd.DataFrame({
            "APOGEE_ID": data[apogee_col].astype(str),
            "TELESCOPE": data[telescope_col].astype(str),
            "FIELD": data[field_col].astype(str)
        })
    # Filter to requested IDs
    sub = df[df["APOGEE_ID"].isin(set(apogee_ids))].drop_duplicates("APOGEE_ID")
    # Build URLs
    sub["URL"] = [build_apogee_url(tel, fld, aid) for tel, fld, aid in sub[["TELESCOPE","FIELD","APOGEE_ID"]].itertuples(index=False, name=None)]
    return sub

def download_apogee(apogee_ids: List[str], outdir: Path, cache_dir: Path, limit: Optional[int] = None, sleep: float = 0.0):
    ensure_dir(outdir)
    session = make_session()
    allstar = load_or_download_allstar(cache_dir)
    mapping = map_apogee_to_path(allstar, apogee_ids)
    if limit:
        mapping = mapping.head(limit)
    if mapping.empty:
        print("[APOGEE] No matching APOGEE_ID found in allStar file.")
        return
    print(f"[APOGEE] Will download {len(mapping)} apStar/asStar files...")
    for row in tqdm(mapping.itertuples(index=False), total=len(mapping)):
        aid, tel, field, url = row
        # Adjust tuple unpacking (since row has columns in order of mapping)
        # Work-around in case tuple order differs:
        if isinstance(row, tuple) and len(row) == 4 and url.startswith("http"):
            pass
        else:
            # regenerate safely
            url = build_apogee_url(row.TELESCOPE, row.FIELD, row.APOGEE_ID)
        fname = url.split("/")[-1]
        dest = outdir / fname
        if dest.exists():
            continue
        try:
            r = session.get(url, stream=True)
            if r.status_code == 404:
                print(f"[APOGEE] 404 for {url}")
                continue
            r.raise_for_status()
            with open(dest, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            if sleep > 0:
                time.sleep(sleep)
        except Exception as e:
            print(f"[APOGEE] Failed {url}: {e}")

# -------------------------------
# GALAH DR3: HERMES spectra via Data Central SSA
# -------------------------------

GALAH_SSA = "https://datacentral.org.au/vo/slink/links"

def galah_camera_suffixes():
    # 1,2,3,4 correspond to B,G,R,I cameras; Data Central FILTER expects 'B','G','R','I'
    return [("B", 1), ("G", 2), ("R", 3), ("I", 4)]

def galah_url_for(sobject_id: str, filt: str) -> str:
    # Build SSA slink URL for a specific camera
    return f"{GALAH_SSA}?ID={sobject_id}&DR=galah_dr3&IDX=0&FILT={filt}&RESPONSEFORMAT=fits"

def download_galah(sobject_ids: List[str], outdir: Path, limit: Optional[int] = None, sleep: float = 0.1):
    ensure_dir(outdir)
    session = make_session()
    to_process = sobject_ids[:limit] if limit else sobject_ids
    print(f"[GALAH] Will download up to {len(to_process)} sobject_id x 4 cameras ...")
    for sid in tqdm(to_process):
        star_dir = outdir / str(sid)
        ensure_dir(star_dir)
        for filt, camnum in galah_camera_suffixes():
            url = galah_url_for(str(sid), filt)
            dest = star_dir / f"{sid}{camnum}.fits"
            if dest.exists():
                continue
            try:
                r = session.get(url, stream=True)
                if r.status_code == 404:
                    # Missing camera (known gaps); skip
                    continue
                r.raise_for_status()
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                if sleep > 0:
                    time.sleep(sleep)
            except Exception as e:
                print(f"[GALAH] Failed {sid} {filt}: {e}")

# -------------------------------
# Gaia-ESO (UVES): ESO Archive via astroquery
# -------------------------------

def _init_eso(row_limit: int = 1000):
    if not ASTROQUERY_AVAILABLE:
        raise RuntimeError("astroquery is required for Gaia-ESO downloads. Install with: pip install astroquery astropy")
    eso = Eso()
    # Most Gaia-ESO Phase 3 data are public; anonymous access is fine.
    eso.ROW_LIMIT = row_limit
    return eso

def query_gaiaeso_targets(eso, target_names: List[str]) -> Dict[str, List[str]]:
    """
    For each target name (e.g., GES CNAME or recognizable target string), query Phase 3 'GaiaESO'
    and collect ARCFILE IDs for UVES 1D spectra.
    Returns mapping: target_name -> list of ARCFILE strings.
    """
    result = {}
    for name in tqdm(target_names, desc="[Gaia-ESO] query by name"):
        try:
            tbl = eso.query_surveys('GaiaESO', cache=False, target=name)
            if tbl is None or len(tbl) == 0:
                result[name] = []
                continue
            # Filter to UVES instrument if column exists
            cols = {c.lower(): c for c in tbl.colnames}
            arc_col = cols.get("arcfile", "ARCFILE")
            ins_col = cols.get("instrument", None)
            if ins_col and ins_col in tbl.colnames:
                mask = [("UVES" in str(x)) for x in tbl[ins_col]]
                tbl = tbl[mask]
            result[name] = [str(x) for x in tbl[arc_col]]
        except Exception as e:
            print(f"[Gaia-ESO] query failed for {name}: {e}")
            result[name] = []
    return result

def download_gaiaeso_arcfiles(eso, arcfiles: List[str], outdir: Path, sleep: float = 0.2):
    ensure_dir(outdir)
    # Use astroquery to retrieve. Public data do not require login.
    # eso.retrieve_data() will create a request and download products.
    for chunk in chunked(arcfiles, 50):
        try:
            eso.retrieve_data(chunk, destination=str(outdir), unzip=True)
            time.sleep(sleep)
        except Exception as e:
            print(f"[Gaia-ESO] retrieve_data failed for a chunk: {e}")

def download_gaiaeso(ges_ids: List[str], outdir: Path, limit: Optional[int] = None, sleep: float = 0.2):
    if not ASTROQUERY_AVAILABLE:
        print("[Gaia-ESO] astroquery not available; skip. Install: pip install astroquery astropy")
        return
    eso = _init_eso(row_limit=5000)
    names = ges_ids[:limit] if limit else ges_ids
    mapping = query_gaiaeso_targets(eso, names)
    # Flatten arcfiles
    arcfiles = []
    for k, v in mapping.items():
        arcfiles.extend(v)
    if not arcfiles:
        print("[Gaia-ESO] No ARCFILEs found. Check your identifiers or try RA/Dec search via --ra-col/--dec-col.")
        return
    print(f"[Gaia-ESO] Will download {len(arcfiles)} files to {outdir} ...")
    download_gaiaeso_arcfiles(eso, arcfiles, outdir, sleep=sleep)

# -------------------------------
# Main CLI
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download APOGEE DR17, GALAH DR3, and Gaia-ESO spectra for a cross-matched list of stars.")
    p.add_argument("--input", required=True, help="Path to CSV/TSV with cross-match IDs")
    p.add_argument("--sep", default=",", help="CSV separator (default ',')")
    p.add_argument("--apogee-col", default="APOGEE_ID", help="Column with APOGEE_ID (e.g. 2M00000002+7417074)")
    p.add_argument("--galah-col", default="sobject_id", help="Column with GALAH DR3 sobject_id")
    p.add_argument("--gaiaeso-col", default="GES_CNAME", help="Column with Gaia-ESO target/CNAME")
    p.add_argument("--ra-col", default=None, help="Optional RA column (deg) for Gaia-ESO")
    p.add_argument("--dec-col", default=None, help="Optional Dec column (deg) for Gaia-ESO")
    p.add_argument("--out", default="./data", help="Output root directory")
    p.add_argument("--surveys", nargs="+", default=["apogee","galah","gaiaeso"], choices=["apogee","galah","gaiaeso"], help="Which surveys to download")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows for testing")
    p.add_argument("--sleep", type=float, default=0.05, help="Sleep (seconds) between HTTP requests")
    p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for large queries/downloads")
    p.add_argument("--cache", default="./.cache_sdss", help="Cache directory (for APOGEE allStar)")
    return p.parse_args()

def main():
    args = parse_args()
    outroot = Path(args.out).resolve()
    ensure_dir(outroot)

    df = pd.read_csv(args.input, sep=args.sep)
    if args.limit:
        df = df.head(args.limit)

    # Extract columns if present
    apogee_ids = df[args.apogee_col].dropna().astype(str).tolist() if args.apogee_col in df.columns else []
    galah_ids  = df[args.galah_col].dropna().astype(str).tolist() if args.galah_col in df.columns else []
    gaiaeso_ids = df[args.gaiaeso_col].dropna().astype(str).tolist() if args.gaiaeso_col in df.columns else []

    if "apogee" in args.surveys and apogee_ids:
        apogee_out = outroot / "apogee_dr17"
        ensure_dir(apogee_out)
        download_apogee(apogee_ids, apogee_out, cache_dir=Path(args.cache), limit=None, sleep=args.sleep)

    if "galah" in args.surveys and galah_ids:
        galah_out = outroot / "galah_dr3"
        ensure_dir(galah_out)
        download_galah(galah_ids, galah_out, limit=None, sleep=args.sleep)

    if "gaiaeso" in args.surveys and (gaiaeso_ids or (args.ra_col and args.dec_col)):
        gaiaeso_out = outroot / "gaia_eso_uves"
        ensure_dir(gaiaeso_out)
        if gaiaeso_ids:
            download_gaiaeso(gaiaeso_ids, gaiaeso_out, limit=None, sleep=args.sleep)
        # Optional RA/Dec-based search (not implemented in detail to keep script simpler)
        # You can add a SkyCoord-based cone search per row if needed:
        # coords = SkyCoord(df[args.ra_col].values * u.deg, df[args.dec_col].values * u.deg)
        # for c in coords: ... eso.query_surveys('GaiaESO', coord=c, radius=1*u.arcsec)

    print("Done.")

if __name__ == "__main__":
    main()
