#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Beam-cube I/O helpers."""

import os

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits


def _load_beam_npz(datafile):
    archive = np.load(datafile)
    data = np.asarray(archive["data"])
    x = np.asarray(archive["x"])
    y = np.asarray(archive["y"])
    freq_arr = np.asarray(archive["freq"])
    error = np.asarray(archive["error"]) if "error" in archive.files else None
    return x, y, freq_arr, error, data


def _load_beam_hdf5(datafile):
    with h5py.File(datafile, "r") as f:
        data = np.asarray(f["data"])
        x = np.asarray(f["x"])
        y = np.asarray(f["y"])
        freq_arr = np.asarray(f["freq"])
        error = np.asarray(f["error"]) if "error" in f else None
    return x, y, freq_arr, error, data


_LOADERS = {
    ".npz": _load_beam_npz,
    ".h5": _load_beam_hdf5,
    ".hdf5": _load_beam_hdf5,
}


def _validate_beam_cube(x, y, freq_arr, data):
    """Raise ``ValueError`` if the loaded arrays are inconsistent."""
    if data.ndim != 3:
        raise ValueError(
            f"data must be 3-D (n_freq, ny, nx); got shape {data.shape}")
    n_freq, ny, nx = data.shape
    if len(freq_arr) != n_freq:
        raise ValueError(
            f"freq has {len(freq_arr)} entries but data has {n_freq} channels")
    if len(y) != ny:
        raise ValueError(
            f"y has {len(y)} entries but data.shape[1]={ny}")
    if len(x) != nx:
        raise ValueError(
            f"x has {len(x)} entries but data.shape[2]={nx}")
    for i in range(n_freq):
        if np.all(np.isnan(data[i])):
            raise ValueError(
                f"Channel {i} (freq={freq_arr[i]:.1f} MHz) is all-NaN — "
                "check your input file")


def load_beam(datafile):
    """Load a beam cube from an ``.npz`` or ``.h5``/``.hdf5`` file.

    Works on any cube with this layout, regardless of its size or resolution.
    Expected contents, as arrays (``.npz``) or datasets (HDF5): ``data``
    (n_chan, ny, nx), ``x``, ``y``, ``freq`` (in GHz), and optionally ``error``.
    ``freq`` is converted to MHz on load so it matches the ``frequency`` value
    used throughout the configs.

    Returns
    -------
    x, y, freq_arr, nchan, error, data
    """
    ext = os.path.splitext(datafile)[1].lower()
    try:
        loader = _LOADERS[ext]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported beam-cube format {ext!r} for {datafile}. "
            f"Supported formats: {', '.join(sorted(_LOADERS))}"
        ) from None

    x, y, freq_arr, error, data = loader(datafile)
    data = data.astype(float, copy=True)
    data[~np.isfinite(data)] = 0.0
    freq_arr = freq_arr * 1e3  # GHz -> MHz
    nchan = freq_arr.shape[0]

    _validate_beam_cube(x, y, freq_arr, data)

    return x, y, freq_arr, nchan, error, data


def save_npz(output_dir, name, **arrays):
    """Save ``arrays`` to ``<output_dir>/<name>.npz``, creating the directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    np.savez(path, **arrays)
    return path


def save_fits(output_path, x, y, freq_arr_mhz, data, overwrite=True):
    """Save a beam cube to a FITS file.

    The cube is written as a primary HDU with shape ``(n_freq, ny, nx)``.
    WCS keywords describe the x/y pixel scale and the frequency axis.

    Parameters
    ----------
    output_path : str
        Destination ``.fits`` file path.
    x, y : 1-D arrays
        Coordinate axes in degrees (Cartesian) or r / theta (polar).
        Assumed evenly spaced; only the step size is used for WCS.
    freq_arr_mhz : 1-D array
        Frequencies in MHz.
    data : 3-D array, shape (n_freq, ny, nx)
        Beam amplitude cube.
    overwrite : bool
        Overwrite an existing file (default True).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    hdr = fits.Header()
    hdr["SIMPLE"]  = True
    hdr["BITPIX"]  = -64
    hdr["NAXIS"]   = 3
    hdr["NAXIS1"]  = data.shape[2]   # x
    hdr["NAXIS2"]  = data.shape[1]   # y
    hdr["NAXIS3"]  = data.shape[0]   # freq

    # Spatial axes (degrees)
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    hdr["CTYPE1"] = "X--DEG"
    hdr["CRPIX1"] = (len(x) + 1) / 2.0
    hdr["CRVAL1"] = float(x[len(x) // 2])
    hdr["CDELT1"] = dx
    hdr["CUNIT1"] = "deg"

    hdr["CTYPE2"] = "Y--DEG"
    hdr["CRPIX2"] = (len(y) + 1) / 2.0
    hdr["CRVAL2"] = float(y[len(y) // 2])
    hdr["CDELT2"] = dy
    hdr["CUNIT2"] = "deg"

    # Frequency axis
    df = float(freq_arr_mhz[1] - freq_arr_mhz[0]) if len(freq_arr_mhz) > 1 else 1.0
    hdr["CTYPE3"] = "FREQ"
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL3"] = float(freq_arr_mhz[0]) * 1e6   # MHz → Hz
    hdr["CDELT3"] = df * 1e6
    hdr["CUNIT3"] = "Hz"

    hdr["BUNIT"]  = "AMPLITUDE"
    hdr["ORIGIN"] = "BeamLab21"

    hdu = fits.PrimaryHDU(data=data.astype(np.float64), header=hdr)
    hdu.writeto(output_path, overwrite=overwrite)
    print(f"FITS cube written → {output_path}  (shape {data.shape})")


def load_coefficients(coeffile):
    """Load Noll-indexed Zernike coefficients from a CSV with columns j, n, m, coef.

    Returns ``(j, n, m, coef)`` as numpy arrays (``j``/``n``/``m`` as int).
    """
    df = pd.read_csv(coeffile)
    j = df["j"].to_numpy().astype(int)
    n = df["n"].to_numpy().astype(int)
    m = df["m"].to_numpy().astype(int)
    coef = df["coef"].to_numpy()
    return j, n, m, coef


def load_scale_params(spfile):
    """Load beam scale parameters from a CSV with columns sigx, sigy.

    Returns ``(sigx, sigy)`` as numpy arrays.
    """
    df = pd.read_csv(spfile)
    return df["sigx"].to_numpy(), df["sigy"].to_numpy()
