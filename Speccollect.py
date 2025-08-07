#!/usr/bin/env python3
"""
build_hdf5.py

Collect continuum‑normalized spectra for the ~30 000 stars common to
APOGEE, GALAH, and Gaia‑ESO and store them in a single HDF5 file.
"""

import glob
import os
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

# ------------------------------------------------------------------
# User‑configurable paths
# ------------------------------------------------------------------
CROSSMATCH_CSV = "common_star_list.csv"  # cross‑match file with IDs
APOGEE_DIR     = "/path/to/apogee/dr17/apStar/"   # apStar DR17 directory
GALAH_DIR      = "/path/to/galah/dr3/spectra/"    # normalized spectra
GAIAESO_DIR    = "/path/to/gaia_eso/uves/"        # Gaia‑ESO UVES spectra
OUTPUT_H5      = "combined_spectra.h5"

# ------------------------------------------------------------------
# Continuum‑normalization helpers (import from spec_collect.py or define here)
# ------------------------------------------------------------------
# from spec_collect import continuum_normalize, continuum_normalize_parallel
# For illustration, a placeholder normalization function is shown:
def continuum_normalize(flux, wave):
    """
    Basic continuum normalization using a running median.
    Replace with survey‑appropriate approach from spec_collect.py.
    """
    from scipy.ndimage import median_filter
    cont = median_filter(flux, size=301, mode="nearest")
    return flux / cont

# ------------------------------------------------------------------
# 1. Load cross‑matched star list
# ------------------------------------------------------------------
cross = pd.read_csv(CROSSMATCH_CSV)
apogee_ids = cross["APOGEE_ID"].astype(str).tolist()
galah_ids  = cross["GALAH_ID"].astype(str).tolist()
ges_ids    = cross["GES_CNAME"].astype(str).tolist()

# Optional: keep a dataframe index aligned with these lists
n_stars = len(cross)
print(f"Processing {n_stars} common stars")

# ------------------------------------------------------------------
# 2. Build lookup dictionaries for quick file access
# ------------------------------------------------------------------
# APOGEE: apStar-dr17-<APOGEE_ID>.fits (possibly nested in subdirectories)
apogee_files = glob.glob(os.path.join(APOGEE_DIR, "**", "apStar-dr17-*.fits"),
                         recursive=True)
apogee_map = {
    os.path.basename(f).split("apStar-dr17-")[1].replace(".fits", ""): f
    for f in apogee_files
}

# GALAH: <sobject_id>*.fits (normalized spectra)
galah_files = glob.glob(os.path.join(GALAH_DIR, "*.fits"))
# Each star may have up to 4 camera files; group by sobject_id prefix.
galah_map = {}
for f in galah_files:
    sid = os.path.basename(f).split("_")[0]  # adjust if naming differs
    galah_map.setdefault(sid, []).append(f)

# Gaia‑ESO: UVES FITS files, OBJECT/CNAME in primary header
ges_files = glob.glob(os.path.join(GAIAESO_DIR, "*.fits"))
# Map Gaia‑ESO CNAME to filename
ges_map = {}
for f in ges_files:
    with fits.open(f, memmap=False) as hdul:
        cname = hdul[0].header.get("OBJECT", "").strip()
    if cname:
        ges_map[cname] = f

# ------------------------------------------------------------------
# 3. Loop over stars and collect spectra
# ------------------------------------------------------------------
flux_apogee, wave_apogee = [], None
flux_galah,  wave_galah  = [], None  # concatenated four cameras
flux_ges,    wave_ges    = [], None

missing_apogee = []
missing_galah  = []
missing_ges    = []

for i in range(n_stars):
    aid  = apogee_ids[i]
    gid  = galah_ids[i]
    ges  = ges_ids[i]

    # ---------------- APOGEE ----------------
    if aid in apogee_map:
        with fits.open(apogee_map[aid], memmap=False) as hdul:
            ap_flux = hdul[1].data.astype(np.float64)
            ap_wave = 10 ** (hdul[1].header["CRVAL1"] +
                             np.arange(hdul[1].header["NAXIS1"]) *
                             hdul[1].header["CDELT1"])  # example for log-lambda
        ap_flux = continuum_normalize(ap_flux, ap_wave)
        flux_apogee.append(ap_flux)
        if wave_apogee is None:
            wave_apogee = ap_wave
    else:
        missing_apogee.append(aid)
        continue  # skip star entirely if any survey data is missing

    # ---------------- GALAH ----------------
    if gid in galah_map:
        seg_flux, seg_wave = [], []
        for fname in sorted(galah_map[gid]):  # ensure consistent order
            with fits.open(fname, memmap=False) as hdul:
                # Replace indices if GALAH stores normalized flux elsewhere
                gflux = hdul[0].data.astype(np.float64)
                gwave = hdul[1].data.astype(np.float64)
            seg_flux.append(gflux)
            seg_wave.append(gwave)
        gflux_all = np.concatenate(seg_flux)
        gwave_all = np.concatenate(seg_wave)
        gflux_all = continuum_normalize(gflux_all, gwave_all)
        flux_galah.append(gflux_all)
        if wave_galah is None:
            wave_galah = gwave_all
    else:
        missing_galah.append(gid)
        continue

    # ---------------- Gaia-ESO ----------------
    if ges in ges_map:
        with fits.open(ges_map[ges], memmap=False) as hdul:
            data = hdul[1].data
            ges_flux = data["FLUX"].astype(np.float64)
            ges_wave = data["WAVE"].astype(np.float64)
        ges_flux = continuum_normalize(ges_flux, ges_wave)
        flux_ges.append(ges_flux)
        if wave_ges is None:
            wave_ges = ges_wave
    else:
        missing_ges.append(ges)
        continue

    if (i + 1) % 1000 == 0:
        print(f"Processed {i+1}/{n_stars} stars")

# ------------------------------------------------------------------
# 4. Convert to arrays and write HDF5
# ------------------------------------------------------------------
flux_apogee = np.asarray(flux_apogee)
flux_galah  = np.asarray(flux_galah)
flux_ges    = np.asarray(flux_ges)

with h5py.File(OUTPUT_H5, "w") as h5:
    h5.create_dataset("spectra_apogee", data=flux_apogee, compression="gzip")
    h5.create_dataset("spectra_galah",  data=flux_galah,  compression="gzip")
    h5.create_dataset("spectra_ges",    data=flux_ges,    compression="gzip")
    h5.create_dataset("wave_apogee",    data=wave_apogee)
    h5.create_dataset("wave_galah",     data=wave_galah)
    h5.create_dataset("wave_ges",       data=wave_ges)
    h5.create_dataset("APOGEE_ID", data=np.array(apogee_ids, dtype="S"))
    h5.create_dataset("GALAH_ID",  data=np.array(galah_ids,  dtype="S"))
    h5.create_dataset("GES_ID",    data=np.array(ges_ids,    dtype="S"))

print(f"Saved combined spectra to {OUTPUT_H5}")

if missing_apogee or missing_galah or missing_ges:
    print("Stars with missing data:")
    print(f"APOGEE: {len(missing_apogee)}  GALAH: {len(missing_galah)}  "
          f"Gaia‑ESO: {len(missing_ges)}")
