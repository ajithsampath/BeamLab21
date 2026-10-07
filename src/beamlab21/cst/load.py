#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""CST far-field text export → Cartesian beam cube.

Workflow
--------
1. Load a CST farfield ``.txt`` export.
2. Project each ``(theta, phi)`` sample to Cartesian ``(x, y)`` via
   ``x = theta * cos(phi)``, ``y = theta * sin(phi)`` (where theta is the
   polar angle from boresight in degrees).
3. Interpolate the chosen amplitude column (Copol or total-E, dB → linear)
   onto a regular Cartesian grid.
4. Save the resulting beam cube as ``.npz``.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

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
        Frequency in MHz (used only to build the ``freq`` axis of the cube;
        the file itself contains a single frequency).
    column : {"copol", "e", "cross"}
        Which amplitude column to extract:

        * ``"copol"``  — ``Abs(Copol)`` (column 5, 0-based)
        * ``"e"``      — total ``Abs(E)`` (column 2)
        * ``"cross"``  — ``Abs(Cross)`` (column 3)
    size : int
        Number of pixels along each Cartesian axis of the output grid.
    xy_max : float
        Half-width (degrees) of the square Cartesian window.

    Returns
    -------
    x : ndarray, shape (size,)
    y : ndarray, shape (size,)
    freq_arr : ndarray, shape (1,)   — ``[freq_mhz / 1e3]`` (GHz)
    data : ndarray, shape (1, size, size)  — linear amplitude
    """
    if column not in _COL:
        raise ValueError(
            f"column must be one of {list(_COL)}, got {column!r}")

    raw = np.loadtxt(str(file), skiprows=2)
    theta = raw[:, 0]        # polar angle from boresight [deg]
    phi   = raw[:, 1]        # azimuth [deg]
    amp_db = raw[:, _COL[column]]

    # dB → linear amplitude
    amp_lin = 10.0 ** (amp_db / 20.0)

    # Project to Cartesian
    phi_rad = np.deg2rad(phi)
    px = theta * np.cos(phi_rad)
    py = theta * np.sin(phi_rad)

    # Interpolate onto a regular grid
    xi = np.linspace(-xy_max, xy_max, size)
    yi = np.linspace(-xy_max, xy_max, size)
    Xi, Yi = np.meshgrid(xi, yi)

    Zg = griddata((px, py), amp_lin, (Xi, Yi), method="cubic")
    Zg = np.where(np.isfinite(Zg), Zg, 0.0)
    Zg = np.maximum(Zg, 0.0)   # cubic overshoot → negative clipped to 0

    freq_arr = np.array([freq_mhz / 1e3])   # MHz → GHz
    data = Zg[np.newaxis, :, :]             # (1, size, size)

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
