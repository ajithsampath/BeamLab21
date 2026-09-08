#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Deprecated aggregate module.

``beamlab21.lib`` has been split into focused modules. Import from those instead:

    beamlab21.config    -> load_config
    beamlab21.paths     -> resolve_path, get_project_root
    beamlab21.io        -> load_beam, save_npz
    beamlab21.zernike   -> NollToQuantum, QuantumToNoll, find_min_full_N_for_Nprime, reorder_coef
    beamlab21.models    -> twoD_Gaussian, twoD_Gaussian_polar_track, GenZTBeam
    beamlab21.fitting   -> GaussianFit, ZernikeFit
    beamlab21.plotting  -> plot_results_cart, plot_results_polar

This shim re-exports the public names for backwards compatibility and will be
removed in a future release.
"""

import warnings

from beamlab21.config import load_config
from beamlab21.fitting import GaussianFit, ZernikeFit
from beamlab21.io import load_beam, save_npz
from beamlab21.models import GenZTBeam, twoD_Gaussian, twoD_Gaussian_polar_track
from beamlab21.paths import get_project_root, resolve_path
from beamlab21.plotting import plot_results_cart, plot_results_polar
from beamlab21.zernike import (
    NollToQuantum,
    QuantumToNoll,
    find_min_full_N_for_Nprime,
    reorder_coef,
)

warnings.warn(
    "beamlab21.lib is deprecated; import from beamlab21.config / .io / .zernike / "
    ".models / .fitting / .plotting / .paths instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "load_config",
    "load_beam",
    "save_npz",
    "GenZTBeam",
    "twoD_Gaussian",
    "twoD_Gaussian_polar_track",
    "get_project_root",
    "resolve_path",
    "plot_results_cart",
    "plot_results_polar",
    "GaussianFit",
    "ZernikeFit",
    "NollToQuantum",
    "QuantumToNoll",
    "find_min_full_N_for_Nprime",
    "reorder_coef",
]
