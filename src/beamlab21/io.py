#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Beam-cube I/O helpers."""

import os

import numpy as np


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
