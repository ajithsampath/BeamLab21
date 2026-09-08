#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Zernike / Noll index bookkeeping."""

import numpy as np


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
