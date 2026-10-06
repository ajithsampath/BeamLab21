#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Tests for beamlab21.metrics."""

import numpy as np
import pytest

from beamlab21.metrics import (
    _FWHM_FACTOR,
    beam_solid_angle,
    beam_summary,
    directivity,
    hpbw,
    main_lobe_efficiency,
)


def test_hpbw_formula():
    hx, hy = hpbw(10.0, 8.0)
    assert abs(hx - _FWHM_FACTOR * 10.0) < 1e-10
    assert abs(hy - _FWHM_FACTOR * 8.0) < 1e-10


def test_hpbw_symmetric():
    hx, hy = hpbw(5.0, 5.0)
    assert abs(hx - hy) < 1e-10


def test_beam_solid_angle_positive():
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    Xg, Yg = np.meshgrid(x, y)
    data = np.exp(-(Xg**2 + Yg**2) / (2 * 10**2))
    sa = beam_solid_angle(data, x, y)
    assert sa > 0


def test_beam_solid_angle_zero_data():
    x = np.linspace(-5, 5, 11)
    y = np.linspace(-5, 5, 11)
    data = np.zeros((11, 11))
    assert beam_solid_angle(data, x, y) == 0.0


def test_beam_solid_angle_peak_normalised():
    # For a flat (constant) beam the solid angle equals the grid area
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    data = np.ones((3, 3))
    sa = beam_solid_angle(data, x, y)
    assert abs(sa - 9.0) < 1e-10


def test_main_lobe_efficiency_range():
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    Xg, Yg = np.meshgrid(x, y)
    sigx, sigy = 10.0, 10.0
    data = np.exp(-(Xg**2 + Yg**2) / (2 * sigx**2))
    eff = main_lobe_efficiency(data, x, y, 0.0, 0.0, sigx, sigy, fac=1.5)
    assert 0.0 < eff <= 1.0


def test_main_lobe_efficiency_increases_with_fac():
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    Xg, Yg = np.meshgrid(x, y)
    data = np.exp(-(Xg**2 + Yg**2) / (2 * 10.0**2))
    eff1 = main_lobe_efficiency(data, x, y, 0.0, 0.0, 10.0, 10.0, fac=1.0)
    eff2 = main_lobe_efficiency(data, x, y, 0.0, 0.0, 10.0, 10.0, fac=3.0)
    assert eff2 > eff1


def test_main_lobe_efficiency_zero_data():
    x = np.linspace(-5, 5, 11)
    y = np.linspace(-5, 5, 11)
    data = np.zeros((11, 11))
    assert np.isnan(main_lobe_efficiency(data, x, y, 0, 0, 2.0, 2.0))


def test_directivity_uniform():
    data = np.ones((10, 10))
    assert directivity(data) == pytest.approx(1.0)


def test_directivity_peaked():
    data = np.ones((10, 10))
    data[5, 5] = 10.0
    d = directivity(data)
    assert d > 1.0


def test_directivity_zero():
    data = np.zeros((5, 5))
    assert np.isnan(directivity(data))


def test_beam_summary_keys():
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    Xg, Yg = np.meshgrid(x, y)
    data = np.exp(-(Xg**2 + Yg**2) / (2 * 10**2))
    s = beam_summary(10.0, 10.0, data, x, y)
    for k in ("hpbw_x", "hpbw_y", "beam_solid_angle", "main_lobe_efficiency", "directivity"):
        assert k in s
