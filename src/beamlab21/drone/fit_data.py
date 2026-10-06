#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Fit a beam model to real drone-track measurements (scattered, non-gridded
``x``, ``y``, ``data``), as a counterpart to :mod:`beamlab21.drone.sim_data`'s
simulate/evaluate helpers.

:class:`beamlab21.fitting.GaussianFit` / :class:`beamlab21.fitting.ZernikeFit`
build their model on a ``meshgrid(x, y)``, so they can't be used directly on
scattered track coordinates. This module instead reuses the lower-level,
grid-agnostic building blocks those classes are themselves built on —
:func:`beamlab21.models.gaussian_pointwise` and
:func:`beamlab21.zernike.zernike_mode` — the same functions
:mod:`beamlab21.drone.sim_data` uses to *evaluate* a model on a track; fitting
adds the optimizer / least-squares coefficient solve on top.

Developed and validated against real (private, not in this repo) drone
flight-track data in ``tests/drone_fit_dev.ipynb``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from beamlab21.models import gaussian_pointwise
from beamlab21.zernike import NollToQuantum, find_min_full_N_for_Nprime, zernike_mode


@dataclass
class GaussianPathFit:
    """Result of :func:`fit_gaussian_on_path`."""

    params: np.ndarray  # amp, sigx, sigy, xo, yo, tilt_deg
    model: np.ndarray
    chisq: float
    success: bool
    message: str

    @property
    def amp(self):
        return self.params[0]

    @property
    def sigx(self):
        return self.params[1]

    @property
    def sigy(self):
        return self.params[2]

    @property
    def xo(self):
        return self.params[3]

    @property
    def yo(self):
        return self.params[4]

    @property
    def tilt(self):
        return self.params[5]


@dataclass
class ZernikePathFit:
    """Result of :func:`fit_zernike_on_path`."""

    coef: np.ndarray
    model: np.ndarray
    chisq: float
    sigx: float
    sigy: float


@dataclass
class DronePathFit:
    """Result of :func:`fit_beam_on_path`: a Gaussian main lobe plus a Zernike
    basis fit on top of it."""

    gaussian: GaussianPathFit
    zernike: ZernikePathFit


def fit_gaussian_on_path(x, y, data, error=None, init_params=None,
                         minimize_method="Nelder-Mead", maxiter=5000, maxfev=5000):
    """Fit a rotated 2D Gaussian to scattered track data ``(x, y, data)``.

    Pointwise analogue of :meth:`beamlab21.fitting.GaussianFit.g_chisq` /
    ``optimize_Gauss`` — same chi-square, but evaluated directly at the given
    coordinates via :func:`beamlab21.models.gaussian_pointwise` instead of on a
    meshgrid.

    ``error`` defaults to a uniform array of ones (no per-point error available
    from the track data); ``init_params`` defaults to
    ``[max(data), 20.0, 20.0, 0.0, 0.0, 0.0]``.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)
    error = np.ones_like(data) if error is None else np.asarray(error)
    if init_params is None:
        init_params = [float(np.max(data)), 20.0, 20.0, 0.0, 0.0, 0.0]

    def chisq(params):
        model = gaussian_pointwise(x, y, params)
        resid = (data - model) / error
        dof = len(params)
        return np.vdot(resid, resid) / (len(data) - dof - 2)

    opt = minimize(chisq, init_params, method=minimize_method,
                   options={"maxiter": maxiter, "maxfev": maxfev})
    model = gaussian_pointwise(x, y, opt.x)
    return GaussianPathFit(params=opt.x, model=model, chisq=opt.fun,
                           success=opt.success, message=opt.message)


def fit_zernike_on_path(x, y, data, xo, yo, sigx, sigy, N=20, fac=3.0, error=None):
    """Least-squares fit of a Zernike/Bessel basis to scattered track data.

    Pointwise analogue of :meth:`beamlab21.fitting.ZernikeFit.zt_chisq` (skip-
    minimise variant, like ``NO_optimize_ZT``): the scale parameters are
    derived from the Gaussian fit's ``sigx``/``sigy`` divided by ``fac``
    (``config_fit.yaml``'s ``N``/``fac`` convention) rather than
    independently optimised, and the basis is built directly at the scattered
    ``(x, y)`` coordinates via :func:`beamlab21.zernike.zernike_mode` instead
    of on a meshgrid.

    ``error`` defaults to a uniform array of ones. A proportional error (e.g.
    ``0.1 * abs(data)``, as the gridded ``ZernikeFit`` uses by default) blows
    up near zero-crossings in background-subtracted track data and is not a
    good default here — see ``tests/drone_fit_dev.ipynb``.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)
    error = np.ones_like(data) if error is None else np.asarray(error)

    zsigx, zsigy = sigx / fac, sigy / fac
    xc, yc = x - xo, y - yo
    rm = np.hypot(xc / zsigx, yc / zsigy)
    rm = np.where(rm == 0, 1e-10, rm)
    thetam = np.arctan2(yc / zsigy, xc / zsigx)

    N_full = find_min_full_N_for_Nprime(N)
    basis = np.zeros((N_full, len(x)))
    count = 0
    for j in range(N_full):
        n, m = NollToQuantum(j)
        if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0:
            basis[count] = zernike_mode(n, m, rm, thetam)
            count += 1
    basis = basis[:count]

    w = 1 / error ** 2
    bw = basis.T * np.sqrt(w[:, None])
    cw = data * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(bw, cw, rcond=None)

    model = basis.T @ coef
    resid = (data - model) / error
    chisq = np.vdot(resid, resid) / (len(data) - len(coef))
    return ZernikePathFit(coef=coef, model=model, chisq=chisq, sigx=zsigx, sigy=zsigy)


def fit_beam_on_path(x, y, data, error=None, N=20, fac=3.0,
                     init_gparams=None, minimize_method="Nelder-Mead"):
    """Fit a beam model to scattered drone-track measurements ``(x, y, data)``.

    Two-stage fit, mirroring :func:`beamlab21.fit.run`'s workflow for gridded
    cubes: a Gaussian main lobe first (:func:`fit_gaussian_on_path`), then a
    Zernike basis on top using its fitted centre and (``fac``-scaled) width
    (:func:`fit_zernike_on_path`).

    Returns a :class:`DronePathFit` with both sub-fits.
    """
    gaussian_fit = fit_gaussian_on_path(
        x, y, data, error=error, init_params=init_gparams, minimize_method=minimize_method,
    )
    zernike_fit = fit_zernike_on_path(
        x, y, data, gaussian_fit.xo, gaussian_fit.yo, gaussian_fit.sigx, gaussian_fit.sigy,
        N=N, fac=fac, error=error,
    )
    return DronePathFit(gaussian=gaussian_fit, zernike=zernike_fit)
