import numpy as np
import pytest

from beamlab21.drone.fit_data import (
    DronePathFit,
    GaussianPathFit,
    ZernikePathFit,
    fit_beam_on_path,
    fit_gaussian_on_path,
    fit_zernike_on_path,
)
from beamlab21.models import gaussian_pointwise

TRUE_PARAMS = [1.0, 10.0, 15.0, 3.0, -2.0, 0.0]  # amp, sigx, sigy, xo, yo, tilt_deg


def _scattered_xy(n=2000, extent=100.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-extent, extent, n)
    y = rng.uniform(-extent, extent, n)
    return x, y


def test_fit_gaussian_on_path_recovers_known_params():
    x, y = _scattered_xy()
    data = gaussian_pointwise(x, y, TRUE_PARAMS)

    result = fit_gaussian_on_path(x, y, data)

    assert isinstance(result, GaussianPathFit)
    assert result.success
    # amp, sigx, sigy, xo, yo recover tightly; tilt is only weakly identifiable
    # near its true value here (sigx/sigy aren't different enough to pin it down
    # to better than Nelder-Mead's default tolerance), so it's checked loosely.
    np.testing.assert_allclose(result.params[:5], TRUE_PARAMS[:5], atol=0.05)
    assert abs(result.tilt - TRUE_PARAMS[5]) < 5.0 or abs(result.tilt - TRUE_PARAMS[5] - 180) < 5.0
    assert result.model.shape == data.shape
    assert result.chisq < 1e-6  # noise-free data, should fit almost exactly


def test_fit_gaussian_on_path_robust_to_noise():
    x, y = _scattered_xy(seed=1)
    rng = np.random.default_rng(2)
    data = gaussian_pointwise(x, y, TRUE_PARAMS) + rng.normal(0, 0.01, len(x))

    result = fit_gaussian_on_path(x, y, data)

    assert result.success
    np.testing.assert_allclose(result.params[:5], TRUE_PARAMS[:5], atol=0.5)


def test_fit_zernike_on_path_shapes_and_improves_on_gaussian_alone():
    x, y = _scattered_xy(seed=3)
    # a beam the pure Gaussian model can't fully capture: a Gaussian core
    # plus some higher-order structure, so the Zernike fit has something to
    # pick up beyond what fit_gaussian_on_path already explains.
    core = gaussian_pointwise(x, y, TRUE_PARAMS)
    ripple = 0.05 * np.cos(0.2 * np.hypot(x, y))
    data = core + ripple

    gfit = fit_gaussian_on_path(x, y, data)
    zfit = fit_zernike_on_path(x, y, data, gfit.xo, gfit.yo, gfit.sigx, gfit.sigy, N=20)

    assert isinstance(zfit, ZernikePathFit)
    assert zfit.model.shape == data.shape
    assert zfit.coef.shape == (20,)

    gaussian_resid_std = (data - gfit.model).std()
    zernike_resid_std = (data - zfit.model).std()
    assert zernike_resid_std < gaussian_resid_std


def test_fit_beam_on_path_returns_combined_result():
    x, y = _scattered_xy(seed=4)
    data = gaussian_pointwise(x, y, TRUE_PARAMS)

    result = fit_beam_on_path(x, y, data, N=10)

    assert isinstance(result, DronePathFit)
    assert isinstance(result.gaussian, GaussianPathFit)
    assert isinstance(result.zernike, ZernikePathFit)
    assert np.isfinite(result.gaussian.chisq)
    assert np.isfinite(result.zernike.chisq)


def test_fit_beam_on_path_default_error_is_uniform():
    x, y = _scattered_xy(n=200, seed=5)
    data = gaussian_pointwise(x, y, TRUE_PARAMS)

    # should not raise, and should behave the same as an explicit uniform error
    result_default = fit_beam_on_path(x, y, data, N=5)
    result_explicit = fit_beam_on_path(x, y, data, N=5, error=np.ones_like(data))

    np.testing.assert_allclose(result_default.gaussian.params, result_explicit.gaussian.params)
    np.testing.assert_allclose(result_default.zernike.coef, result_explicit.zernike.coef)


def test_fit_zernike_on_path_rejects_mismatched_shapes():
    x, y = _scattered_xy(n=50, seed=6)
    data = gaussian_pointwise(x, y, TRUE_PARAMS)
    with pytest.raises(ValueError):
        fit_zernike_on_path(x, y, data[:-1], 0, 0, 10.0, 10.0, N=5)
