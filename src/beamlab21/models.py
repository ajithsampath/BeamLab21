#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Analytic beam models: 2D Gaussian and the generative Zernike-transform beam."""

import numpy as np
from tqdm import tqdm

from beamlab21.io import load_coefficients
from beamlab21.zernike import zernike_mode


def gaussian_pointwise(x, y, params):
    """Evaluate a rotated 2D Gaussian pointwise at Cartesian coordinates ``(x, y)``.

    ``x``/``y`` may be scalars, a grid (e.g. from ``meshgrid``), or scattered
    arrays of matching shape. ``params`` = ``[amp, sigx, sigy, xo, yo, tilt_deg]``.
    This is the shared formula behind :func:`twoD_Gaussian`, :func:`twoD_Gaussian_polar_track`,
    and :func:`beamlab21.drone.sim_data.evaluate_gaussian_on_path`.
    """
    amp, sigx, sigy, xo, yo, tilt = params
    xo = float(xo)
    yo = float(yo)
    tilt = np.radians(tilt)

    a = (np.cos(tilt) ** 2) / (2 * sigx ** 2) + (np.sin(tilt) ** 2) / (2 * sigy ** 2)
    b = -(np.sin(2 * tilt)) / (4 * sigx ** 2) + (np.sin(2 * tilt)) / (4 * sigy ** 2)
    c = (np.sin(tilt) ** 2) / (2 * sigx ** 2) + (np.cos(tilt) ** 2) / (2 * sigy ** 2)

    dx = x - xo
    dy = y - yo
    return amp * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))


def twoD_Gaussian(x, y, params):
    """Evaluate a rotated 2D Gaussian on the grid ``meshgrid(x, y)``.

    ``params`` = ``[amp, sigx, sigy, xo, yo, tilt_deg]``. Output shape is
    ``(len(y), len(x))``.
    """
    xg, yg = np.meshgrid(x, y)
    return gaussian_pointwise(xg, yg, params)


def twoD_Gaussian_polar_track(r, theta, params):
    """Rotated 2D Gaussian evaluated point-wise at track coordinates ``(r, theta)``.

    ``r`` and ``theta`` must share the same shape; ``params`` =
    ``[amp, sigx, sigy, xo, yo, tilt_deg]``.
    """
    r = np.asarray(r)
    theta = np.asarray(theta)
    if r.shape != theta.shape:
        raise ValueError("r and theta must have the same shape for track coordinates")

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return gaussian_pointwise(x, y, params)


class GenZTBeam:
    """Generate a beam from stored Zernike-transform coefficients + scale params."""

    def __init__(self, freq, x, y, dtype):
        self.freq = freq
        self.x = x
        self.y = y
        self.dtype = dtype

    def load_coef(self, coeffile):
        """Load coefficients and (j, n, m) indices from a CSV file."""
        if not coeffile.endswith(".csv"):
            raise ValueError("Unsupported file format. Please use .csv files for Coefficients.\n")
        self.j, self.n, self.m, self.coef = load_coefficients(coeffile)
        return None

    def basisfunc(self, sigx, sigy):
        """Build the Bessel-derived basis matrix for the loaded coefficients."""
        self.sigx, self.sigy = sigx, sigy
        xm, ym = np.meshgrid(self.x / self.sigx, self.y / self.sigy)
        rm = np.hypot(xm, ym)
        rm[rm == 0] = 1e-10
        thetam = np.arctan2(ym, xm)

        self.Basis = np.zeros((len(self.coef), len(rm.flatten())), dtype=self.dtype)
        print("Constructing the basis set for the given coefficients and scale parameters...")
        with tqdm(total=100, bar_format="{l_bar}{bar}| [{elapsed}] {postfix}") as pbar:
            for idx in range(self.coef.shape[0]):
                n = self.n[idx]
                m = self.m[idx]
                self.Basis[idx] = zernike_mode(n, m, rm, thetam).flatten()
                pbar.update(100 / self.coef.shape[0])
                pbar.set_postfix_str(f"{round(pbar.n, 1)}%")
        return self.Basis
