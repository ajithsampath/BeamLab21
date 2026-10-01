#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Beam-cube I/O helpers."""

import os

import numpy as np
import pandas as pd


def load_beam(datafile):
    """Load a beam cube from an ``.npz`` file.

    The archive is expected to contain ``data`` (n_chan, ny, nx), ``x``, ``y``,
    ``freq`` (in GHz), and optionally ``error``. ``freq`` is converted to MHz on
    load so it matches the ``frequency`` value used throughout the configs.

    Returns
    -------
    x, y, freq_arr, nchan, error, data
    """
    if not datafile.endswith(".npz"):
        raise NotImplementedError(
            f"load_beam currently only supports .npz cubes, got: {datafile}"
        )

    archive = np.load(datafile)
    data = archive["data"]
    data[np.isnan(data)] = 0.0

    x = archive["x"]
    y = archive["y"]
    freq_arr = archive["freq"] * 1e3  # GHz -> MHz
    nchan = freq_arr.shape[0]
    error = archive["error"] if "error" in archive.files else None

    return x, y, freq_arr, nchan, error, data


def save_npz(output_dir, name, **arrays):
    """Save ``arrays`` to ``<output_dir>/<name>.npz``, creating the directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    np.savez(path, **arrays)
    return path


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
