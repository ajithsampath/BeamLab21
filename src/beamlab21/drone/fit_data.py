#Author: Ajith Sampath
#Affiliation: University of Geneva

#Fit a beam model to real drone-track measurements (scattered, non-gridded
#x, y, data), as a counterpart to beamlab21.drone.sim_data's simulate/evaluate
#helpers. Placeholder until real flight-track data is available to validate against.


def fit_beam_on_path(x, y, data, *args, **kwargs):
    """Fit a beam model to scattered drone-track measurements ``(x, y, data)``.

    STATUS: under development, not yet implemented.

    :class:`beamlab21.fitting.GaussianFit` / :class:`beamlab21.fitting.ZernikeFit`
    already operate on arbitrary (non-gridded) coordinates mathematically, so this
    will likely wire them up directly against drone-track arrays instead of going
    through :func:`beamlab21.fit.run`'s cube-loading path. Left as a stub until
    real flight-track data exists to validate against.
    """
    raise NotImplementedError(
        "Fitting drone-track data is not implemented yet; see this function's docstring."
    )
