import numpy as np

from beamlab21.zernike import (
    NollToQuantum,
    QuantumToNoll,
    find_min_full_N_for_Nprime,
    reorder_coef,
)


def test_noll_quantum_roundtrip():
    for j in range(200):
        n, m = NollToQuantum(j)
        assert QuantumToNoll(n, m) == j


def test_find_min_full_N_monotonic_and_covers():
    prev = 0
    for nprime in range(1, 30):
        full = find_min_full_N_for_Nprime(nprime)
        assert full >= prev
        prev = full
        # exactly `nprime` valid modes in [0, full)
        valid = sum(
            1
            for j in range(full)
            for (n, m) in [NollToQuantum(j)]
            if n >= 0 and n >= abs(m) and (n - abs(m)) % 2 == 0 and m >= 0
        )
        assert valid == nprime


def test_reorder_coef_preserves_length():
    coef = np.arange(1, 11, dtype=float)
    out = reorder_coef(coef.copy())
    assert len(out) == len(coef)
    assert np.count_nonzero(out) <= len(coef)
