#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Diagnostic beam metrics derived from Gaussian / Zernike fit outputs."""

import numpy as np

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
