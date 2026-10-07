#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Tests for new metrics: chromaticity fit, aperture efficiency."""

import numpy as np

from beamlab21.metrics import (
    aperture_efficiency,
    fit_beam_chromaticity,
)

# ---------------------------------------------------------------------------
# fit_beam_chromaticity
# ---------------------------------------------------------------------------

class TestFitBeamChromaticity:
    def _make_series(self, alpha=-1.0, sigma0=20.0, nu0=400.0):
        freq = np.array([350.0, 400.0, 450.0, 500.0, 550.0])
        sigx = sigma0 * (freq / nu0) ** alpha
        sigy = sigma0 * 0.9 * (freq / nu0) ** alpha
        return freq, sigx, sigy

    def test_recovers_alpha(self):
        freq, sigx, sigy = self._make_series(alpha=-1.0)
        res = fit_beam_chromaticity(freq, sigx, sigy, nu0_mhz=400.0)
        assert abs(res["alpha_x"] - (-1.0)) < 0.01
        assert abs(res["alpha_y"] - (-1.0)) < 0.01

    def test_recovers_sigma0(self):
        freq, sigx, sigy = self._make_series(sigma0=25.0)
        res = fit_beam_chromaticity(freq, sigx, sigy, nu0_mhz=400.0)
        assert abs(res["sigma0_x"] - 25.0) < 0.1

    def test_model_evaluated(self):
        freq, sigx, sigy = self._make_series()
        res = fit_beam_chromaticity(freq, sigx, sigy)
        assert res["sigma_fit_x"].shape == (len(freq),)
        assert res["sigma_fit_y"].shape == (len(freq),)

    def test_default_nu0_is_geometric_mean(self):
        freq = np.array([400.0, 600.0])
        sigx = np.array([20.0, 18.0])
        sigy = np.array([19.0, 17.0])
        res = fit_beam_chromaticity(freq, sigx, sigy)
        expected_nu0 = float(np.exp(np.mean(np.log(freq))))
        assert abs(res["nu0_mhz"] - expected_nu0) < 1e-6

    def test_positive_alpha_increasing_beam(self):
        freq = np.array([400.0, 500.0, 600.0])
        sigx = freq / 400.0 * 20.0    # alpha = +1
        sigy = sigx
        res = fit_beam_chromaticity(freq, sigx, sigy, nu0_mhz=400.0)
        assert res["alpha_x"] > 0


# ---------------------------------------------------------------------------
# aperture_efficiency
# ---------------------------------------------------------------------------

class TestApertureEfficiency:
    def test_returns_expected_keys(self):
        res = aperture_efficiency(400.0, 20.0, 18.0, 6.0)
        for k in ("lambda_m", "omega_a_sr", "eta_ap", "A_eff_m2"):
            assert k in res

    def test_lambda_correct(self):
        res = aperture_efficiency(400.0, 20.0, 18.0, 6.0)
        expected = 299.792458 / 400.0
        assert abs(res["lambda_m"] - expected) < 1e-6

    def test_eta_between_0_and_1(self):
        # HIRAX: ~6 m dish at 400 MHz, ~20 deg sigma
        res = aperture_efficiency(400.0, 20.0, 20.0, 6.0)
        assert 0.0 < res["eta_ap"] <= 1.0

    def test_wider_beam_lower_efficiency(self):
        narrow = aperture_efficiency(400.0, 10.0, 10.0, 6.0)
        wide   = aperture_efficiency(400.0, 30.0, 30.0, 6.0)
        # Wider beam → larger Ω_A → smaller A_eff → lower η
        assert narrow["eta_ap"] > wide["eta_ap"]

    def test_higher_freq_shorter_lambda(self):
        low  = aperture_efficiency(400.0, 10.0, 10.0, 6.0)
        high = aperture_efficiency(600.0, 10.0, 10.0, 6.0)
        assert high["lambda_m"] < low["lambda_m"]
