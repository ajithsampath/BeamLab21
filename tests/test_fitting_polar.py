#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Test polar-coordinate fitting round-trip.

Generates a synthetic Gaussian beam on a polar (r, theta) grid, saves it as a
beam cube, fits with coord_type="polar", and checks the recovered parameters
match the known truth.  No real data required.
"""

import os
import tempfile

import numpy as np
import pytest

from beamlab21.fitting import GaussianFit, ZernikeFit
from beamlab21.models import gaussian_pointwise

# True beam parameters used to generate synthetic data
TRUE_AMP = 1.0
TRUE_SIGX = 20.0
TRUE_SIGY = 18.0
TRUE_XO = 0.0
TRUE_YO = 0.0
TRUE_TILT = 0.0
TRUE_PARAMS = [TRUE_AMP, TRUE_SIGX, TRUE_SIGY, TRUE_XO, TRUE_YO, TRUE_TILT]

FREQ_GHZ = 0.4  # single-frequency cube


def _make_polar_cube(r_max=75.0, nr=51, ntheta=64, tmpdir=None):
    """Create a beam cube in polar (r, theta) coordinates and save as .npz."""
    r = np.linspace(0.0, r_max, nr)        # (nr,)  — x axis
    theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)  # (ntheta,) — y axis
    freq = np.array([FREQ_GHZ])

    R, THETA = np.meshgrid(r, theta)       # both (ntheta, nr)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    beam_slice = gaussian_pointwise(X, Y, TRUE_PARAMS)  # (ntheta, nr)
    # data cube: (n_freq, ny, nx) = (1, ntheta, nr)
    data = beam_slice[np.newaxis, :, :]

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "polar_beam.npz")
    np.savez(path, x=r, y=theta, freq=freq, data=data)
    return path, r, theta


@pytest.fixture(scope="module")
def polar_cube_path(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("polar"))
    path, _, _ = _make_polar_cube(tmpdir=tmpdir)
    return path


# ---------------------------------------------------------------------------
# GaussianFit polar
# ---------------------------------------------------------------------------

def test_gaussian_fit_polar_auto_detection(polar_cube_path):
    """auto coord_type should correctly identify polar data."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="auto")
    assert gfit.coord_type == "polar"


def test_gaussian_fit_polar_x_y_unchanged(polar_cube_path):
    """self.x / self.y must remain the original 1D r / theta axes after init."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="polar")
    # x should be non-negative (r axis); y should be in [0, 2pi] (theta axis)
    assert np.all(gfit.x >= 0)
    assert np.all(gfit.y >= 0) and np.all(gfit.y <= 2 * np.pi)


def test_gaussian_fit_polar_cart_grid_shape(polar_cube_path):
    """The internal 2D Cartesian grid must match the data shape."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="polar")
    assert gfit._X_cart.shape == gfit.data.shape
    assert gfit._Y_cart.shape == gfit.data.shape


def test_gaussian_fit_polar_recovers_sigma(polar_cube_path):
    """Fitted sigx/sigy should be close to the true values."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="polar")
    init = [1.0, TRUE_SIGX * 0.8, TRUE_SIGY * 0.8, 0.0, 0.0, 0.0]
    result = gfit.optimize_Gauss(init, minimize_method="Nelder-Mead",
                                 xtol=1e-6, verbose=False)
    _, _, xo, yo, _, _, sigx, sigy, _, _, _ = result
    assert abs(sigx - TRUE_SIGX) < 5.0, f"sigx={sigx:.2f}, expected {TRUE_SIGX}"
    assert abs(sigy - TRUE_SIGY) < 5.0, f"sigy={sigy:.2f}, expected {TRUE_SIGY}"


# ---------------------------------------------------------------------------
# ZernikeFit polar
# ---------------------------------------------------------------------------

def test_zernike_fit_polar_basis_shape(polar_cube_path):
    """Zernike basis columns must equal the number of data pixels."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="polar")
    init = [1.0, TRUE_SIGX * 0.8, TRUE_SIGY * 0.8, 0.0, 0.0, 0.0]
    x, y, xo, yo, freq_arr, freq, sigx, sigy, _, data, _ = gfit.optimize_Gauss(
        init, minimize_method="Nelder-Mead", xtol=1e-6, verbose=False)

    ztfit = ZernikeFit(x, y, xo, yo, freq_arr, freq, data, N=6,
                       error_type="uniform", normalize_data=False,
                       coord_type="polar")
    ztfit.basis_N([sigx, sigy])

    n_pixels = data.size
    assert ztfit.Basis.shape[1] == n_pixels, (
        f"Basis columns {ztfit.Basis.shape[1]} != data pixels {n_pixels}")


def test_zernike_fit_polar_no_optimize_runs(polar_cube_path):
    """NO_optimize_ZT should complete without error in polar mode."""
    gfit = GaussianFit(polar_cube_path, freq=400, error_type="uniform",
                       normalize_data=True, coord_type="polar")
    init = [1.0, TRUE_SIGX * 0.8, TRUE_SIGY * 0.8, 0.0, 0.0, 0.0]
    x, y, xo, yo, freq_arr, freq, sigx, sigy, _, data, _ = gfit.optimize_Gauss(
        init, minimize_method="Nelder-Mead", xtol=1e-6, verbose=False)

    ztfit = ZernikeFit(x, y, xo, yo, freq_arr, freq, data, N=6,
                       error_type="uniform", normalize_data=False,
                       coord_type="polar")
    sp, sp2, coef, model = ztfit.NO_optimize_ZT([sigx, sigy], fac=3.0)
    assert model.shape == data.shape
    assert coef is not None


# ---------------------------------------------------------------------------
# Equivalence: polar cube == Cartesian cube (same beam, different grid)
# ---------------------------------------------------------------------------

def test_polar_vs_cartesian_gaussian_consistent():
    """Fitting the same beam in Cartesian and polar grids should give similar sigmas."""
    # Cartesian grid
    x_c = np.linspace(-60.0, 60.0, 61)
    y_c = np.linspace(-60.0, 60.0, 61)
    Xc, Yc = np.meshgrid(x_c, y_c)
    beam_c = gaussian_pointwise(Xc, Yc, TRUE_PARAMS)
    freq = np.array([FREQ_GHZ])

    # Polar grid
    r = np.linspace(0.0, 70.0, 51)
    theta = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    R, THETA = np.meshgrid(r, theta)
    beam_p = gaussian_pointwise(R * np.cos(THETA), R * np.sin(THETA), TRUE_PARAMS)

    with tempfile.TemporaryDirectory() as tmpdir:
        cart_path = os.path.join(tmpdir, "cart.npz")
        polar_path = os.path.join(tmpdir, "polar.npz")
        np.savez(cart_path, x=x_c, y=y_c, freq=freq,
                 data=beam_c[np.newaxis, :, :])
        np.savez(polar_path, x=r, y=theta, freq=freq,
                 data=beam_p[np.newaxis, :, :])

        init = [1.0, TRUE_SIGX * 0.8, TRUE_SIGY * 0.8, 0.0, 0.0, 0.0]

        gfit_c = GaussianFit(cart_path, freq=400, error_type="uniform",
                             normalize_data=True, coord_type="cartesian")
        _, _, _, _, _, _, sigx_c, sigy_c, _, _, _ = gfit_c.optimize_Gauss(
            init, minimize_method="Nelder-Mead", xtol=1e-6, verbose=False)

        gfit_p = GaussianFit(polar_path, freq=400, error_type="uniform",
                             normalize_data=True, coord_type="polar")
        _, _, _, _, _, _, sigx_p, sigy_p, _, _, _ = gfit_p.optimize_Gauss(
            init, minimize_method="Nelder-Mead", xtol=1e-6, verbose=False)

    # Both should recover the truth within a few degrees
    assert abs(sigx_c - TRUE_SIGX) < 5.0
    assert abs(sigy_c - TRUE_SIGY) < 5.0
    assert abs(sigx_p - TRUE_SIGX) < 5.0
    assert abs(sigy_p - TRUE_SIGY) < 5.0
    # And agree with each other
    assert abs(sigx_c - sigx_p) < 3.0, f"sigx: cart={sigx_c:.2f} polar={sigx_p:.2f}"
    assert abs(sigy_c - sigy_p) < 3.0, f"sigy: cart={sigy_c:.2f} polar={sigy_p:.2f}"
