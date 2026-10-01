"""Drone-based beam mapping (under development).

- :mod:`beamlab21.drone.sim_data` — generate a flight path and evaluate a beam
  model at its (scattered, non-gridded) coordinates.
- :mod:`beamlab21.drone.fit_data` — fit a beam model to real drone-track
  measurements (stub; not yet implemented).

The names below are re-exported at the package level for convenience.
"""

from beamlab21.drone.fit_data import fit_beam_on_path
from beamlab21.drone.sim_data import (
    create_drone_path,
    evaluate_gaussian_on_path,
    evaluate_zernike_on_path,
    load_zernike_coef,
)

__all__ = [
    "create_drone_path",
    "evaluate_gaussian_on_path",
    "evaluate_zernike_on_path",
    "load_zernike_coef",
    "fit_beam_on_path",
]
