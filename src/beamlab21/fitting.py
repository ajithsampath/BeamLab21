#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Gaussian and Zernike-transform fitting of a single-frequency beam slice."""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
from tqdm import tqdm

from beamlab21.io import load_beam
from beamlab21.models import gaussian_pointwise, twoD_Gaussian
from beamlab21.zernike import NollToQuantum, find_min_full_N_for_Nprime, zernike_mode


class GaussianFit:
    """Fit a rotated 2D Gaussian to one frequency slice of a beam cube."""

    def __init__(self, datafile, freq, error_type="uniform",
                 normalize_data=True, coord_type="auto"):
        self.freq = freq
        self.x, self.y, self.freq_arr, self.nchan, self.error, self.data_cube = \
            load_beam(datafile)

        if coord_type == "auto":
            if np.all(self.x >= 0) and np.all((self.y >= 0) & (self.y <= 2 * np.pi)):
                coord_type = "polar"
            else:
                coord_type = "cartesian"
        self.coord_type = coord_type

        # For polar data: build the 2D Cartesian grid from the 1D r/theta axes.
        # self.x/self.y stay as the original axis arrays so ZernikeFit receives them.
        if self.coord_type == "polar":
            r, theta = self.x, self.y
            R, THETA = np.meshgrid(r, theta)      # shape (ny, nx)
            self._X_cart = R * np.cos(THETA)      # 2D Cartesian x
            self._Y_cart = R * np.sin(THETA)      # 2D Cartesian y

        chan = int(np.argmin(np.abs(self.freq_arr - freq)))
        self.data = self.data_cube[chan]
        if normalize_data:
            self.data = self.data / np.max(self.data)

        if self.error is None:
            if error_type == "uniform":
                self.error = np.ones_like(self.data)
                print("Proceeding with uniform error array...\n")
            else:
                raise ValueError(
                    "Error array not found. Provide error array or set error_type='uniform'.\n"
                )

    def callback(self, xk):
        if getattr(self, "pbar", None) is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"Chi-square": f"{xk[0]:.4f}"})

    def g_chisq(self, params):
        """Reduced chi-square for the Gaussian fit."""
        if self.coord_type == "polar":
            self.gExpected = gaussian_pointwise(self._X_cart, self._Y_cart, params)
        else:
            self.gExpected = twoD_Gaussian(self.x, self.y, params)
        Expected = self.gExpected.flatten()
        data = self.data.flatten()
        k = len(params)
        resid = (data - Expected) / self.error.flatten()
        chi = np.vdot(resid, resid) / (len(data) - k - 2)
        return np.abs(chi)

    def optimize_Gauss(self, init_gparams, minimize_method="Nelder-Mead",
                       xtol=1e-8, verbose=True):
        """Minimise :meth:`g_chisq` and return the fit summary tuple."""
        self.init_gparams = init_gparams
        print("Initial Gaussian parameters:", self.init_gparams)
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.gopt = minimize(self.g_chisq, self.init_gparams,
                             callback=self.callback,
                             method=minimize_method,
                             options={"disp": verbose})
        self.pbar.close()

        _, self.sigx_gopt, self.sigy_gopt, self.xo, self.yo, _ = self.gopt.x
        return (self.x, self.y, self.xo, self.yo, self.freq_arr,
                self.freq, self.sigx_gopt, self.sigy_gopt,
                self.gExpected, self.data, self.gopt.fun)


class ZernikeFit:
    """Least-squares fit of a Bessel/Zernike-transform basis to a beam slice."""

    def __init__(self, x, y, xo, yo, freq_arr, freq, data, N,
                 error_type="proportional", normalize_data=True, coord_type="auto"):
        self.pbar = None
        self.coord_type = coord_type

        if self.coord_type == "auto":
            if np.all(x >= 0) and np.all((y >= 0) & (y <= 2 * np.pi)):
                self.coord_type = "polar"
            else:
                self.coord_type = "cartesian"

        if self.coord_type == "polar":
            self.r = x
            self.theta = y
            self.x = None
            self.y = None
        else:
            self.x = x
            self.y = y
            self.r = None
            self.theta = None

        self.xo = xo
        self.yo = yo
        self.freq_arr = freq_arr
        self.freq = freq
        self.data = data
        self.N = N
        self.N_full = find_min_full_N_for_Nprime(self.N, NollToQuantum)

        if normalize_data:
            self.data = self.data / np.max(self.data)
            print("Data normalized to maximum value.\n")
        else:
            print("Data not normalized.\n")

        if error_type == "proportional":
            self.error = np.abs(self.data) * 0.1
        elif error_type == "uniform":
            self.error = np.ones_like(self.data)
        else:
            raise ValueError("Unsupported error type. Use 'proportional' or 'uniform'.\n")

    def basis_N(self, params):
        """Generate the Bessel-derived basis for the given scale parameters."""
        self.sigx, self.sigy = params

        if self.coord_type == "cartesian":
            xm, ym = np.meshgrid(self.x / self.sigx, self.y / self.sigy)
            rm = np.hypot(xm, ym)
            rm[rm == 0] = 1e-10
            thetam = np.arctan2(ym, xm)
        elif self.coord_type == "polar":
            R, THETA = np.meshgrid(self.r, self.theta)    # shape (ny, nx)
            X_cart = R * np.cos(THETA)
            Y_cart = R * np.sin(THETA)
            rm = np.hypot(X_cart / self.sigx, Y_cart / self.sigy)
            rm[rm == 0] = 1e-10
            thetam = np.arctan2(Y_cart / self.sigy, X_cart / self.sigx)

        self.Basis = np.zeros((int(self.N_full), int(len(rm.flatten()))), dtype="float32")
        count = 0
        print(
            f"Constructing the full basis set for the given N={self.N} and scaling "
            f"parameters = [{np.round(self.sigx, 2), np.round(self.sigy, 2)}]..."
        )
        with tqdm(total=100, bar_format="{l_bar}{bar}| [{elapsed}] {postfix}") as pbar:
            for j in range(0, self.N_full):
                n, m = NollToQuantum(j)
                if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0:
                    self.Basis[count] = zernike_mode(n, m, rm, thetam).flatten()
                    count += 1
                pbar.update(100 / self.N_full)
                pbar.set_postfix_str(f"{round(pbar.n, 1)}%")

    def zt_chisq(self, params):
        """Reduced chi-square after the linear least-squares coefficient solve."""
        self.basis_N(params)
        w = 1 / self.error.flatten() ** 2
        Bw = self.Basis.T * np.sqrt(w[:, np.newaxis])
        Cw = self.data.flatten() * np.sqrt(w)
        self.coef, _, _, _ = sp.linalg.lstsq(Bw, Cw)
        self.Expected = np.dot(self.Basis.T, self.coef).reshape(self.data.shape)
        Expected = self.Expected.flatten()
        data = self.data.flatten()
        k = len(self.coef)
        resid = (data - Expected) / np.sqrt(self.error.flatten())
        chi_red = np.vdot(resid, resid).real / (len(data) - k)
        return np.abs(chi_red)

    def callback(self, xk):
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"Chi-square": f"{xk[0]:.4f}"})

    def optimize_ZT(self, init_ztparams, minimize_method="Nelder-Mead", xtol=1e-8, maxiter=100):
        """Optimise the scale parameters, refitting coefficients at each step."""
        self.init_ztparams = init_ztparams
        print("Is this a HPC machine? If so, good choice :). "
              "If not run this script with 'skip_minimise' set to True.\n")
        self.pbar = tqdm(desc="Optimizing", unit="iter", dynamic_ncols=True)
        self.ztopt = minimize(self.zt_chisq, self.init_ztparams,
                              callback=self.callback,
                              method=minimize_method,
                              options={"disp": True, "maxiter": maxiter, "xtol": xtol})
        self.pbar.close()
        self.sigx_ztopt, self.sigy_ztopt = self.ztopt.x
        return self.sigx_ztopt, self.sigy_ztopt, self.coef, self.Expected, self.ztopt.fun

    def NO_optimize_ZT(self, init_ztparams, fac):
        """Skip scale-parameter optimisation; derive it from the Gaussian sigmas."""
        print("Skipping scaling parameter optimization for Zernike basis.....\n")
        print("Good choice if you are running in a laptop :) \n")
        self.init_ztparams = [init_ztparams[0] / fac, init_ztparams[1] / fac]
        self.zt_chisq(self.init_ztparams)
        print("No optimization done for scaling parameter... "
              "and the ZT model is constructed using Gaussian sigma!!\n")
        return self.init_ztparams[0], self.init_ztparams[1], self.coef, np.abs(self.Expected)
