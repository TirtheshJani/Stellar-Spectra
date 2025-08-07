## Stellar-Spectra

This repository provides a single script, `build_hdf5.py`, which demonstrates
how to gather continuum-normalized spectra for the ~30,000 stars common to
APOGEE, GALAH, and Gaia-ESO and store them in one HDF5 file.

To use the script:

1. Supply a cross-match CSV listing the star identifiers from each survey.
2. Download the relevant FITS spectra for all three surveys and update the path
   constants near the top of the script.
3. Run the script to create `combined_spectra.h5`, which will contain the
   normalized spectra and wavelength grids for the matched stars.

The normalization routine included here is a simple median filter; replace it
with the more robust functions from `spec_collect.py` or other tools as needed.

