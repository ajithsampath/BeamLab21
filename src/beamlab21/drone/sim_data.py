#Author: Ajith Sampath
#Affiliation: University of Geneva

#Simulate drone-based beam mapping: generate a flight path and evaluate
#a beam model at the (scattered, non-gridded) path coordinates.
#Complements lib.py's grid-based twoD_Gaussian/GenZTBeam, which evaluate
#on a regular pixel grid rather than along an arbitrary track.

import numpy as np
from tqdm import tqdm

from beamlab21.io import load_coefficients, load_scale_params
from beamlab21.models import gaussian_pointwise
from beamlab21.zernike import zernike_mode


def create_drone_path(
    width,
    height,
    dx=1.0,
    dy=1.0,
    ds=0.1,
    jitter=0.0,
    direction="EW",  # "EW" or "NS"
):
    """
    Generate (x, y) coordinates for a continuous zigzag scan
    with selectable primary flight direction.

    direction:
        "EW" -> East-West scan lines stepping North-South
        "NS" -> North-South scan lines stepping East-West
    """

    coords = []

    if direction.upper() == "EW":
        nx = int(width / dx) + 1
        ny = int(height / dy) + 1

        for j in range(ny):
            if j % 2 == 0:
                x_start, x_end = 0, (nx - 1) * dx
            else:
                x_start, x_end = (nx - 1) * dx, 0

            y = j * dy

            x_line = np.linspace(
                x_start, x_end,
                int(abs(x_end - x_start) / ds) + 1
            )
            y_line = np.full_like(x_line, y)
            coords.extend(np.column_stack((x_line, y_line)))

            if j < ny - 1:
                y_next = (j + 1) * dy
                y_vert = np.linspace(
                    y, y_next,
                    int(abs(y_next - y) / ds) + 1
                )[1:]
                x_vert = np.full_like(y_vert, x_end)
                coords.extend(np.column_stack((x_vert, y_vert)))

    elif direction.upper() == "NS":
        nx = int(width / dx) + 1
        ny = int(height / dy) + 1

        for i in range(nx):
            if i % 2 == 0:
                y_start, y_end = 0, (ny - 1) * dy
            else:
                y_start, y_end = (ny - 1) * dy, 0

            x = i * dx

            y_line = np.linspace(
                y_start, y_end,
                int(abs(y_end - y_start) / ds) + 1
            )
            x_line = np.full_like(y_line, x)
            coords.extend(np.column_stack((x_line, y_line)))

            if i < nx - 1:
                x_next = (i + 1) * dx
                x_vert = np.linspace(
                    x, x_next,
                    int(abs(x_next - x) / ds) + 1
                )[1:]
                y_vert = np.full_like(x_vert, y_end)
                coords.extend(np.column_stack((x_vert, y_vert)))

    else:
        raise ValueError("direction must be 'EW' or 'NS'")

    coords = np.array(coords)
    coords += np.random.uniform(-jitter, jitter, coords.shape)
    return coords


def evaluate_gaussian_on_path(x, y, params):
    """
    Evaluate a 2D Gaussian beam pointwise at scattered (x, y) coordinates,
    e.g. along a drone flight path. Thin wrapper around
    beamlab21.models.gaussian_pointwise (the same formula used by
    beamlab21.models.twoD_Gaussian and twoD_Gaussian_polar_track).

    params: (amp, sigx, sigy, xo, yo, tilt) with tilt in degrees,
    matching beamlab21.models.twoD_Gaussian's convention.
    """
    return gaussian_pointwise(x, y, params)


def load_zernike_coef(coeffile, spfile):
    """Load Zernike (Noll-indexed) coefficients and beam scale parameters from CSV."""
    _, n, m, coef = load_coefficients(coeffile)
    sigx, sigy = load_scale_params(spfile)
    return n, m, coef, sigx, sigy


def evaluate_zernike_on_path(x, y, coeffile, spfile):
    """
    Evaluate a Zernike-basis beam pointwise at scattered (x, y) coordinates,
    e.g. along a drone flight path. Cartesian analogue of
    beamlab21.models.GenZTBeam.basisfunc, which evaluates on a regular grid;
    both build on the shared beamlab21.zernike.zernike_mode basis function.
    """
    n_arr, m_arr, coef, sigx, sigy = load_zernike_coef(coeffile, spfile)
    xm, ym = x / sigx, y / sigy
    rm = np.hypot(xm, ym)
    rm[rm == 0] = 1e-10
    thetam = np.arctan2(ym, xm)

    basis = np.zeros((len(coef), rm.size))
    print("Constructing beam for drone coordinates...")
    with tqdm(total=100, bar_format='{l_bar}{bar}| [{elapsed}] {postfix}') as pbar:
        for idx in range(coef.shape[0]):
            n = n_arr[idx]
            m = m_arr[idx]
            basis[idx] = zernike_mode(n, m, rm, thetam).flatten()
            pbar.update(100 / coef.shape[0])
            pbar.set_postfix_str(f'{round(pbar.n, 1)}%')

    return basis.T @ coef
