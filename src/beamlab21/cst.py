#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""CST far-field text export → Cartesian beam cube → fit.

Workflow
--------
1. Load a CST farfield ``.txt`` export.
2. Project each ``(theta, phi)`` sample to Cartesian ``(x, y)`` via
   ``x = theta * cos(phi)``, ``y = theta * sin(phi)`` (where theta is the
   polar angle from boresight in degrees).
3. Interpolate the chosen amplitude column (Copol or total-E, dB → linear)
   onto a regular Cartesian grid.
4. Save the resulting beam cube as ``.npz``.
5. Run the standard Gaussian + Zernike fit pipeline on it.
"""

import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from beamlab21 import fit as _fit

# CST column indices (0-based, after skipping 2 header rows)
# Theta [deg.]  Phi [deg.]  Abs(E)[dBV/m]  Abs(Cross)[dBV/m]  Phase(Cross)[deg]
# Abs(Copol)[dBV/m]  Phase(Copol)[deg]  Ax.Ratio[dB]
_COL = {
    "copol": 5,
    "e": 2,
    "cross": 3,
}

_COL_LABEL = {
    "copol": "Abs(Copol) [dB(V/m)]",
    "e": "Abs(E) [dB(V/m)]",
    "cross": "Abs(Cross) [dB(V/m)]",
}


def load_cst(file, freq_mhz, column="copol", size=1501, xy_max=75.0):
    """Load a CST far-field text export and return a Cartesian beam cube.

    Parameters
    ----------
    file : str or Path
        Path to the CST farfield ``.txt`` export.
    freq_mhz : float
        Frequency of this export in MHz (e.g. 400).
    column : {"copol", "e"}
        Which amplitude column to use.  ``"copol"`` (default) uses
        ``Abs(Copol)``; ``"e"`` uses the total ``Abs(E)``.
    size : int
        Output grid resolution: ``(size, size)`` pixels.
    xy_max : float
        Half-width of the Cartesian output window in degrees.
        The grid spans ``[-xy_max, xy_max]`` in both axes.

    Returns
    -------
    x : ndarray, shape (size,)
        Cartesian x axis in degrees.
    y : ndarray, shape (size,)
        Cartesian y axis in degrees.
    freq_arr : ndarray, shape (1,)
        Frequency in GHz (for the beam-cube convention used by ``load_beam``).
    data : ndarray, shape (1, size, size)
        Linear amplitude on the Cartesian grid.  Pixels outside the CST
        coverage are set to zero.
    """
    col_idx = _COL.get(column)
    if col_idx is None:
        raise ValueError(
            f"column must be one of {list(_COL)}; got {column!r}"
        )

    raw = np.loadtxt(file, skiprows=2)

    theta_deg = raw[:, 0]   # polar angle from boresight — used as radius
    phi_deg   = raw[:, 1]   # azimuth
    amp_dB    = raw[:, col_idx]

    phi_rad = np.deg2rad(phi_deg)
    x_scat = theta_deg * np.cos(phi_rad)
    y_scat = theta_deg * np.sin(phi_rad)

    mask = (np.abs(x_scat) <= xy_max) & (np.abs(y_scat) <= xy_max)
    x_scat = x_scat[mask]
    y_scat = y_scat[mask]
    amp_dB = amp_dB[mask]

    amp_lin = 10 ** (amp_dB / 20.0)

    xi = np.linspace(-xy_max, xy_max, size)
    yi = np.linspace(-xy_max, xy_max, size)
    Xg, Yg = np.meshgrid(xi, yi)

    Zg = griddata(
        (x_scat, y_scat),
        amp_lin,
        (Xg, Yg),
        method="cubic",
        fill_value=np.nan,
    )
    # Cubic interpolation can produce negative overshoots and infs near the
    # convex-hull boundary; amplitude must be finite and non-negative.
    Zg = np.where(np.isfinite(Zg), Zg, 0.0)
    Zg = np.maximum(Zg, 0.0)

    data = Zg[np.newaxis, :, :]               # (1, size, size)
    freq_arr = np.array([freq_mhz / 1e3])     # MHz → GHz

    print(f"CST file loaded: {Path(file).name}")
    print(f"  Column : {_COL_LABEL[column]}")
    print(f"  Freq   : {freq_mhz} MHz")
    print(f"  Grid   : {size}×{size}  (xy_max={xy_max} deg)")
    print(f"  Data range (linear): {Zg.min():.4f} – {Zg.max():.4f}\n")

    return xi, yi, freq_arr, data


def stack(files, freqs_mhz, column="copol", size=1501, xy_max=75.0):
    """Load multiple single-frequency CST exports and stack into one cube.

    Parameters
    ----------
    files : list of str
        Paths to the CST ``.txt`` files, one per frequency.
    freqs_mhz : list of float
        Corresponding frequencies in MHz.  Must have the same length as
        ``files``.
    column : {"copol", "e"}
        Amplitude column to convert.
    size : int
        Cartesian grid resolution (same for all files).
    xy_max : float
        Half-width of the Cartesian window in degrees (same for all files).

    Returns
    -------
    x : ndarray, shape (size,)
    y : ndarray, shape (size,)
    freq_arr : ndarray, shape (n_files,)  — in GHz
    data : ndarray, shape (n_files, size, size)  — linear amplitude
    """
    if len(files) != len(freqs_mhz):
        raise ValueError(
            f"files ({len(files)}) and freqs_mhz ({len(freqs_mhz)}) must have the same length")

    slices = []
    ref_x = ref_y = None

    for i, (f, freq) in enumerate(zip(files, freqs_mhz)):
        xi, yi, _, Zi = load_cst(f, freq, column=column, size=size, xy_max=xy_max)
        if ref_x is None:
            ref_x, ref_y = xi, yi
        elif not (np.allclose(xi, ref_x) and np.allclose(yi, ref_y)):
            raise ValueError(
                f"File {i} ({Path(f).name}) produced a grid that differs from file 0 — "
                "ensure all files use the same size and xy_max")
        slices.append(Zi[0])    # shape (size, size)

    freq_arr = np.array(freqs_mhz) / 1e3   # MHz → GHz
    data = np.stack(slices, axis=0)          # (n_files, size, size)

    print(f"Stacked {len(files)} CST files into cube {data.shape}")
    return ref_x, ref_y, freq_arr, data


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
        Base directory for config-relative paths (same meaning as in
        ``beamlab21 fit --base-dir``).
    column : {"copol", "e"}
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

    # Determine where to save the cube
    if cube_output is not None:
        cube_path = Path(cube_output)
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cube_path), x=x, y=y, freq=freq_arr, data=data)
        print(f"Beam cube saved → {cube_path}\n")
        _keep_cube = True
    elif skip_fit:
        # save beside the CST file
        cube_path = Path(cst_file).with_suffix(".npz")
        np.savez(str(cube_path), x=x, y=y, freq=freq_arr, data=data)
        print(f"Beam cube saved → {cube_path}\n")
        _keep_cube = True
    else:
        # temp file — cleaned up after fitting
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
