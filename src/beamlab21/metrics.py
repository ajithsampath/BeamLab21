#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Diagnostic beam metrics derived from Gaussian / Zernike fit outputs."""

import numpy as np
from scipy.optimize import curve_fit

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # ≈ 2.3548


def hpbw(sigx, sigy):
    """Half-power beam width from Gaussian sigma values.

    Parameters
    ----------
    sigx, sigy : float
        Gaussian sigma in x and y (same angular units as the beam cube axes).

    Returns
    -------
    hpbw_x, hpbw_y : float
    """
    return float(_FWHM_FACTOR * sigx), float(_FWHM_FACTOR * sigy)


def beam_solid_angle(data, x, y):
    """Numerical beam solid angle: ∫∫ (B / B_max) dx dy.

    Returns the solid angle in the same units as ``x`` and ``y`` squared
    (e.g. deg²).  The integrand is normalised to the peak so the result is
    independent of amplitude scaling.
    """
    dx = float(np.abs(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = float(np.abs(y[1] - y[0])) if len(y) > 1 else 1.0
    peak = float(np.max(data))
    if peak == 0:
        return 0.0
    return float(np.sum(data / peak)) * dx * dy


def main_lobe_efficiency(data, x, y, xo, yo, sigx, sigy, fac=1.5):
    """Fraction of total beam power within ``fac`` sigma of the beam centre.

    Uses power (amplitude²) to weight each pixel.  ``fac=1.5`` includes
    roughly the main lobe; ``fac=1.0`` corresponds to the half-power ellipse.

    Parameters
    ----------
    data : 2-D array
        Beam amplitude slice.
    x, y : 1-D arrays
        Coordinate axes (same length as ``data.shape[1]`` and ``data.shape[0]``).
    xo, yo : float
        Beam centre (from the Gaussian fit).
    sigx, sigy : float
        Gaussian sigmas.
    fac : float
        Ellipse radius in units of sigma.
    """
    Xg, Yg = np.meshgrid(x, y)
    inside = ((Xg - xo) ** 2 / sigx ** 2 + (Yg - yo) ** 2 / sigy ** 2) <= fac ** 2
    total = float(np.sum(data ** 2))
    if total == 0:
        return float("nan")
    return float(np.sum(data[inside] ** 2) / total)


def directivity(data):
    """Peak-to-mean ratio over the full data grid."""
    mean = float(np.mean(data))
    if mean == 0:
        return float("nan")
    return float(np.max(data) / mean)


def beam_summary(sigx, sigy, data, x, y, xo=0.0, yo=0.0, fac=1.5):
    """Compute all metrics and return them in a single dict.

    Parameters
    ----------
    sigx, sigy : float
        Gaussian sigmas from the fit.
    data : 2-D array
        Beam amplitude slice.
    x, y : 1-D arrays
        Coordinate axes.
    xo, yo : float
        Beam centre (default 0).
    fac : float
        Main-lobe ellipse radius in sigma units (default 1.5).

    Returns
    -------
    dict with keys ``hpbw_x``, ``hpbw_y``, ``beam_solid_angle``,
    ``main_lobe_efficiency``, ``directivity``.
    """
    hx, hy = hpbw(sigx, sigy)
    return {
        "hpbw_x": hx,
        "hpbw_y": hy,
        "beam_solid_angle": beam_solid_angle(data, x, y),
        "main_lobe_efficiency": main_lobe_efficiency(
            data, x, y, xo, yo, sigx, sigy, fac),
        "directivity": directivity(data),
    }


# ---------------------------------------------------------------------------
# Frequency-dependent beam width
# ---------------------------------------------------------------------------

def _power_law(nu, sigma0, alpha):
    return sigma0 * nu ** alpha


def _powerlaw_ripple(nu_norm, sigma0, alpha, A, P_norm, phi):
    """Power-law envelope with a multiplicative sinusoidal ripple.

    σ(ν) = σ₀ · (ν/ν₀)^α · [1 + A · sin(2π · (ν/ν₀) / P_norm + φ)]
    """
    return sigma0 * nu_norm ** alpha * (1.0 + A * np.sin(2.0 * np.pi * nu_norm / P_norm + phi))


def _fft_ripple_guess(nu_norm, resid):
    """Estimate ripple amplitude, period and phase from FFT of fractional residuals."""
    n = len(nu_norm)
    if n < 4:
        return 0.05, 0.5, 0.0
    dnu = float(nu_norm[1] - nu_norm[0]) if n > 1 else 1.0
    fft_vals = np.fft.rfft(resid)
    freqs = np.fft.rfftfreq(n, d=dnu)
    # Ignore DC (index 0); find dominant frequency
    peak_idx = int(np.argmax(np.abs(fft_vals[1:]))) + 1
    A_init = float(2.0 * np.abs(fft_vals[peak_idx]) / n)
    P_init = float(1.0 / freqs[peak_idx]) if freqs[peak_idx] > 0 else 0.5
    phi_init = float(np.angle(fft_vals[peak_idx]))
    return A_init, P_init, phi_init


def _lambda_over_D_model(freq_mhz, k, dish_diam_m):
    """σ(ν) = k · λ/D converted to degrees."""
    lam_m = 299.792458 / freq_mhz        # wavelength in metres
    return np.rad2deg(k * lam_m / dish_diam_m)


def fit_beam_chromaticity(freq_mhz, sigx_arr, sigy_arr, nu0_mhz=None,
                          model="powerlaw", dish_diameter_m=None):
    """Fit a chromatic beam-width model σ(ν) to Gaussian sigmas vs frequency.

    Three models are supported:

    ``"powerlaw"`` (default)
        σ(ν) = σ₀ · (ν/ν₀)^α  —  two free parameters per axis.

    ``"powerlaw_ripple"``
        σ(ν) = σ₀ · (ν/ν₀)^α · [1 + A · sin(2π·(ν/ν₀)/P + φ)]

        Captures sinusoidal frequency structure from reflections or standing
        waves.  Five free parameters per axis; A, P and φ are seeded from an
        FFT of the power-law residuals.

    ``"lambda_over_D"``
        σ(ν) = k · λ(ν) / D  (degrees),  λ(ν) = c/ν.

        Theoretical diffraction-limited beamwidth for a circular aperture.
        ``dish_diameter_m`` is required.  Fits a single dimensionless
        illumination coefficient k per axis (k ≈ 1 for uniform illumination).
        α is implicitly −1; no ν₀ normalisation needed.

    Parameters
    ----------
    freq_mhz : array-like, shape (n,)
        Frequencies in MHz.
    sigx_arr, sigy_arr : array-like, shape (n,)
        Gaussian sigma values in x and y at each frequency.
    nu0_mhz : float or None
        Reference frequency in MHz (used by ``"powerlaw"`` and
        ``"powerlaw_ripple"``).  Defaults to the geometric mean of
        ``freq_mhz``.
    model : {"powerlaw", "powerlaw_ripple", "lambda_over_D"}
        Which chromatic model to fit.
    dish_diameter_m : float or None
        Dish diameter in metres.  Required when ``model="lambda_over_D"``.

    Returns
    -------
    dict
        Always contains: ``nu0_mhz``, ``sigma0_x``, ``alpha_x``,
        ``sigma0_y``, ``alpha_y``, ``sigma_fit_x``, ``sigma_fit_y``,
        ``freq_mhz``, ``model``.

        ``"powerlaw_ripple"`` additionally contains per-axis keys
        ``A_x``, ``P_mhz_x``, ``phi_x``, ``A_y``, ``P_mhz_y``, ``phi_y``.
    """
    freq = np.asarray(freq_mhz, dtype=float)
    sigx = np.asarray(sigx_arr, dtype=float)
    sigy = np.asarray(sigy_arr, dtype=float)

    if model not in ("powerlaw", "powerlaw_ripple", "lambda_over_D"):
        raise ValueError(
            f"model must be 'powerlaw', 'powerlaw_ripple', or 'lambda_over_D', got {model!r}")
    if model == "lambda_over_D" and dish_diameter_m is None:
        raise ValueError("dish_diameter_m is required when model='lambda_over_D'")

    if nu0_mhz is None:
        nu0_mhz = float(np.exp(np.mean(np.log(freq))))

    nu_norm = freq / nu0_mhz

    # --- Stage 1: power-law fit (used as envelope / initial guess) ---
    (s0x, ax), _ = curve_fit(_power_law, nu_norm, sigx,
                              p0=[float(np.mean(sigx)), -1.0])
    (s0y, ay), _ = curve_fit(_power_law, nu_norm, sigy,
                              p0=[float(np.mean(sigy)), -1.0])

    if model == "powerlaw":
        return {
            "nu0_mhz": float(nu0_mhz),
            "sigma0_x": float(s0x), "alpha_x": float(ax),
            "sigma0_y": float(s0y), "alpha_y": float(ay),
            "sigma_fit_x": _power_law(nu_norm, s0x, ax),
            "sigma_fit_y": _power_law(nu_norm, s0y, ay),
            "freq_mhz": freq,
            "model": "powerlaw",
        }

    if model == "lambda_over_D":
        D = float(dish_diameter_m)
        k0 = float(np.mean(np.deg2rad(sigx) * D / (299.792458 / freq)))
        _lod = lambda nu, k: _lambda_over_D_model(nu, k, D)  # noqa: E731
        (kx,), _ = curve_fit(_lod, freq, sigx, p0=[k0])
        (ky,), _ = curve_fit(_lod, freq, sigy, p0=[k0])
        return {
            "k_x": float(kx),
            "k_y": float(ky),
            "dish_diameter_m": D,
            "sigma_fit_x": _lambda_over_D_model(freq, kx, D),
            "sigma_fit_y": _lambda_over_D_model(freq, ky, D),
            "freq_mhz": freq,
            "model": "lambda_over_D",
        }

    # --- Stage 2: ripple fit ---
    # Fractional residuals from the power-law envelope
    resid_x = sigx / _power_law(nu_norm, s0x, ax) - 1.0
    resid_y = sigy / _power_law(nu_norm, s0y, ay) - 1.0

    Ax0, Px0, phx0 = _fft_ripple_guess(nu_norm, resid_x)
    Ay0, Py0, phy0 = _fft_ripple_guess(nu_norm, resid_y)

    bounds = ([0, -np.inf, 0, 0, -np.pi],
              [np.inf, np.inf, 1.0, np.inf, np.pi])

    (s0x_r, ax_r, Ax, Px, phx), _ = curve_fit(
        _powerlaw_ripple, nu_norm, sigx,
        p0=[s0x, ax, Ax0, Px0, phx0], bounds=bounds, maxfev=10000)
    (s0y_r, ay_r, Ay, Py, phy), _ = curve_fit(
        _powerlaw_ripple, nu_norm, sigy,
        p0=[s0y, ay, Ay0, Py0, phy0], bounds=bounds, maxfev=10000)

    return {
        "nu0_mhz": float(nu0_mhz),
        "sigma0_x": float(s0x_r), "alpha_x": float(ax_r),
        "A_x": float(Ax), "P_mhz_x": float(Px * nu0_mhz), "phi_x": float(phx),
        "sigma0_y": float(s0y_r), "alpha_y": float(ay_r),
        "A_y": float(Ay), "P_mhz_y": float(Py * nu0_mhz), "phi_y": float(phy),
        "sigma_fit_x": _powerlaw_ripple(nu_norm, s0x_r, ax_r, Ax, Px, phx),
        "sigma_fit_y": _powerlaw_ripple(nu_norm, s0y_r, ay_r, Ay, Py, phy),
        "freq_mhz": freq,
        "model": "powerlaw_ripple",
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Aperture efficiency
# ---------------------------------------------------------------------------

def aperture_efficiency(freq_mhz, sigx, sigy, dish_diameter_m):
    """Compute aperture efficiency from the fitted beam solid angle.

    η_ap = λ² / (4π · Ω_A)

    where ``Ω_A`` is computed analytically from the Gaussian sigmas
    (Ω_A = π · σx · σy / ln 2  for a 2D Gaussian beam in steradians,
    converted from the input angular units assumed to be degrees).

    Parameters
    ----------
    freq_mhz : float
        Frequency in MHz.
    sigx, sigy : float
        Gaussian sigmas in degrees.
    dish_diameter_m : float
        Physical dish diameter in metres (used for the geometric area A_geom).

    Returns
    -------
    dict with keys:
        ``lambda_m``, ``omega_a_sr``, ``eta_ap``, ``A_eff_m2``.
    """
    lam = 299.792458 / freq_mhz  # wavelength in metres
    # σ in radians
    sx_rad = np.deg2rad(float(sigx))
    sy_rad = np.deg2rad(float(sigy))
    # Gaussian beam solid angle (steradians)
    omega_a = np.pi * sx_rad * sy_rad / np.log(2)
    A_eff = lam ** 2 / (4.0 * np.pi * omega_a) if omega_a > 0 else float("nan")
    A_geom = np.pi * (dish_diameter_m / 2.0) ** 2
    eta = A_eff / A_geom if A_geom > 0 else float("nan")

    return {
        "lambda_m": float(lam),
        "omega_a_sr": float(omega_a),
        "eta_ap": float(eta),
        "A_eff_m2": float(A_eff),
    }
