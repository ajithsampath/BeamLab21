import numpy as np
import pytest

from beamlab21.models import twoD_Gaussian, twoD_Gaussian_polar_track


def test_gaussian_shape():
    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 30)
    g = twoD_Gaussian(x, y, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert g.shape == (len(y), len(x))


def test_gaussian_peak_at_offset_and_amplitude():
    x = np.linspace(-10, 10, 201)
    y = x
    amp, xo, yo = 3.0, 2.0, -1.0
    g = twoD_Gaussian(x, y, [amp, 1.5, 2.5, xo, yo, 0.0])
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    assert abs(x[ix] - xo) < 0.2
    assert abs(y[iy] - yo) < 0.2
    assert np.isclose(g.max(), amp, rtol=1e-3)


def test_gaussian_symmetric_when_round():
    x = np.linspace(-4, 4, 81)
    g = twoD_Gaussian(x, x, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(g, g.T, atol=1e-12)


def test_polar_track_matches_cartesian_pointwise():
    params = [2.0, 1.3, 0.8, 0.2, -0.3, 15.0]
    x = np.linspace(-3, 3, 25)
    xx, yy = np.meshgrid(x, x)
    cart = twoD_Gaussian(x, x, params)
    r = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    polar = twoD_Gaussian_polar_track(r, theta, params)
    assert np.allclose(cart, polar, atol=1e-12)


def test_polar_track_shape_mismatch_raises():
    with pytest.raises(ValueError):
        twoD_Gaussian_polar_track(np.zeros(4), np.zeros(5), [1, 1, 1, 0, 0, 0])
