#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Zernike / Noll index bookkeeping and basis evaluation."""

import numpy as np
from scipy.special import jn


def NollToQuantum(j):
    """Map a Noll linear index ``j`` to radial/azimuthal quantum numbers ``(n, m)``."""
    n = int(np.ceil((-3 + np.sqrt(9 + (8 * j))) / 2))
    m = int((2 * j) - (n * (n + 2)))
    return (n, m)


def QuantumToNoll(n, m):
    """Inverse of :func:`NollToQuantum`."""
    j = int((n * (n + 2) + m) / 2)
    return j


def find_min_full_N_for_Nprime(Nprime, noll_to_quantum=NollToQuantum):
    """Smallest full basis size containing ``Nprime`` valid (n >= m >= 0) modes."""
    count = 0
    j = 0
    while True:
        n, m = noll_to_quantum(j)
        if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0:
            count += 1
            if count == Nprime:
                return j + 1  # +1 because j is 0-based index
        j += 1


def zernike_mode(n, m, rm, thetam):
    """Evaluate one Noll-indexed Bessel/Zernike-transform basis mode.

    ``rm``, ``thetam`` are (scaled) polar coordinates, scalar or array; ``n``,
    ``m`` are the quantum numbers for a single valid mode (``n >= abs(m) >= 0``,
    ``n - abs(m)`` even). This is the shared basis building block used when
    fitting (:class:`beamlab21.fitting.ZernikeFit`), generating on a grid
    (:class:`beamlab21.models.GenZTBeam`), and evaluating along scattered
    points (:mod:`beamlab21.drone.sim_data`).
    """
    bes = jn(n + 1, rm) / rm
    nc = np.abs(((2 * n + 1) * (2 * n + 3) * (2 * n + 5)) / (-1) ** n) ** 0.5
    phase = np.exp(1j * m * thetam) / ((1j ** m) * 2 * np.pi)
    return np.real(nc * phase * (-1) ** ((n - m) / 2) * bes)


def reorder_coef(coef):
    """Re-insert zeros for negative-m modes so ``coef`` aligns with the full Noll order."""
    neg_m_list = []
    for j in range(coef.shape[0]):
        n, m = NollToQuantum(j)
        if m < 0:
            neg_m_list.append(j)
    for g in neg_m_list:
        coef = np.delete(coef, -1)
        coef = np.insert(coef, g, 0.0)
    return coef
