#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Analytic beam models: 2D Gaussian and the generative Zernike-transform beam."""

import numpy as np
import pandas as pd
from scipy.special import jn
from tqdm import tqdm


def twoD_Gaussian(x, y, params):
    """Evaluate a rotated 2D Gaussian on the grid ``meshgrid(x, y)``.

    ``params`` = ``[amp, sigx, sigy, xo, yo, tilt_deg]``. Output shape is
    ``(len(y), len(x))``.
    """
    amp, sigx, sigy, xo, yo, tilt = params
    xo = float(xo)
    yo = float(yo)
    tilt = np.radians(tilt)
    x, y = np.meshgrid(x, y)
    a = (np.cos(tilt) ** 2) / (2 * sigx ** 2) + (np.sin(tilt) ** 2) / (2 * sigy ** 2)
    b = -(np.sin(2 * tilt)) / (4 * sigx ** 2) + (np.sin(2 * tilt)) / (4 * sigy ** 2)
    c = (np.sin(tilt) ** 2) / (2 * sigx ** 2) + (np.cos(tilt) ** 2) / (2 * sigy ** 2)

    return amp * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )


def twoD_Gaussian_polar_track(r, theta, params):
    """Rotated 2D Gaussian evaluated point-wise at track coordinates ``(r, theta)``.

    ``r`` and ``theta`` must share the same shape; ``params`` =
    ``[amp, sigx, sigy, xo, yo, tilt_deg]``.
    """
    amp, sigx, sigy, xo, yo, tilt = params
    xo = float(xo)
    yo = float(yo)
    tilt = np.radians(tilt)

    r = np.asarray(r)
    theta = np.asarray(theta)
    if r.shape != theta.shape:
        raise ValueError("r and theta must have the same shape for track coordinates")

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ct = np.cos(tilt)
    st = np.sin(tilt)
    a = (ct ** 2) / (2 * sigx ** 2) + (st ** 2) / (2 * sigy ** 2)
    b = -(np.sin(2 * tilt)) / (4 * sigx ** 2) + (np.sin(2 * tilt)) / (4 * sigy ** 2)
    c = (st ** 2) / (2 * sigx ** 2) + (ct ** 2) / (2 * sigy ** 2)

    dx = x - xo
    dy = y - yo
    return amp * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))


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
        df = pd.read_csv(coeffile)
        self.coef = df["coef"].values
        self.j = df["j"].to_numpy().astype(int)
        self.n = df["n"].to_numpy().astype(int)
        self.m = df["m"].to_numpy().astype(int)
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
                Bes = (jn(n + 1, rm)) / rm
                nc = np.abs(((2 * n + 1) * (2 * n + 3) * (2 * n + 5)) / (-1) ** n) ** 0.5
                phase = np.exp(1j * m * thetam) / ((1j ** m) * 2 * np.pi)
                temp = np.real(nc * phase * (-1) ** ((n - m) / 2) * Bes)
                self.Basis[idx] = temp.flatten()
                pbar.update(100 / self.coef.shape[0])
                pbar.set_postfix_str(f"{round(pbar.n, 1)}%")
        return self.Basis
