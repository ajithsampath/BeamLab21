#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""CST end-to-end pipeline: convert + fit."""

import tempfile
from pathlib import Path

import numpy as np

from beamlab21 import fit as _fit
from beamlab21.cst.load import load_cst


def run(cst_file, freq_mhz, config_path="configs/config_fit.yaml",
        base_dir=None, column="copol", size=1501, xy_max=75.0,
        cube_output=None, skip_fit=False):
    """Convert a CST farfield export and (optionally) fit it.

    Parameters
    ----------
    cst_file : str or Path
        Path to the CST ``.txt`` file.
    freq_mhz : float
        Frequency in MHz.
    config_path : str
        Path to the fit YAML config (default: ``configs/config_fit.yaml``).
    base_dir : str or None
        Base directory for config-relative paths.
    column : {"copol", "e", "cross"}
        Amplitude column to convert.
    size : int
        Cartesian grid resolution.
    xy_max : float
        Half-width of the output window in degrees.
    cube_output : str or None
        Where to save the converted ``.npz`` beam cube.  If ``None`` a
        temporary file is used and deleted after fitting (unless
        ``skip_fit=True``, in which case it is saved next to the CST file).
    skip_fit : bool
        If ``True``, stop after conversion and save the cube; skip the fit.
    """
    x, y, freq_arr, data = load_cst(cst_file, freq_mhz, column=column,
                                    size=size, xy_max=xy_max)

    if cube_output is not None:
        cube_path = Path(cube_output)
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cube_path), x=x, y=y, freq=freq_arr, data=data)
        print(f"Beam cube saved → {cube_path}\n")
        _keep_cube = True
    elif skip_fit:
        cube_path = Path(cst_file).with_suffix(".npz")
        np.savez(str(cube_path), x=x, y=y, freq=freq_arr, data=data)
        print(f"Beam cube saved → {cube_path}\n")
        _keep_cube = True
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        _tmp.close()
        cube_path = Path(_tmp.name)
        np.savez(str(cube_path), x=x, y=y, freq=freq_arr, data=data)
        _keep_cube = False

    if skip_fit:
        return

    try:
        _fit.run(config_path, base_dir=base_dir,
                 coord_type="cartesian", datafile=str(cube_path))
    finally:
        if not _keep_cube:
            cube_path.unlink(missing_ok=True)
