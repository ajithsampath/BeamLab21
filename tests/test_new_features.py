#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Tests for: input validation, multi-frequency fit, CST stack, compute polar."""


import numpy as np
import pytest

from beamlab21.cli import build_parser
from beamlab21.cst import load_cst, stack
from beamlab21.io import _validate_beam_cube, load_beam
from beamlab21.models import GenZTBeam, gaussian_pointwise

# ---------------------------------------------------------------------------
# Helpers shared across sections
# ---------------------------------------------------------------------------

def _make_npz(path, n_freq=3, ny=31, nx=31):
    x = np.linspace(-30, 30, nx)
    y = np.linspace(-30, 30, ny)
    freq = np.linspace(0.4, 0.6, n_freq)
    Xg, Yg = np.meshgrid(x, y)
    data = np.stack([gaussian_pointwise(Xg, Yg, [1.0, 10.0, 10.0, 0, 0, 0])] * n_freq)
    np.savez(path, x=x, y=y, freq=freq, data=data)
    return x, y, freq * 1e3, data  # freq in MHz


def _write_mini_cst(path, n_theta=15, n_phi=20):
    header = (
        "Theta [deg.]  Phi   [deg.]  Abs(E   )[dB(V/m)]   "
        "Abs(Cross)[dB(V/m)]  Phase(Cross)[deg.]  "
        "Abs(Copol)[dB(V/m)]  Phase(Copol)[deg.]  Ax.Ratio[dB    ]    \n"
        + "-" * 90 + "\n"
    )
    thetas = np.linspace(-8.0, 8.0, n_theta)
    phis   = np.linspace(-90.0, 90.0, n_phi)
    rows = []
    for theta in thetas:
        for phi in phis:
            copol_lin = np.exp(-(theta**2) / (2 * 4.0**2))
            copol_dB  = 20 * np.log10(max(copol_lin, 1e-10))
            e_dB = copol_dB + 0.1
            rows.append(
                f"{theta:12.3f} {phi:12.3f} {e_dB:20.3e} {-40.0:20.3e} "
                f"{0.0:20.3f} {copol_dB:20.3e} {0.0:20.3f} {50.0:12.3e}\n"
            )
    with open(path, "w") as f:
        f.write(header)
        f.writelines(rows)


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_good_cube_passes(self):
        x = np.arange(5, dtype=float)
        y = np.arange(4, dtype=float)
        freq = np.array([400.0, 500.0])
        data = np.ones((2, 4, 5))
        _validate_beam_cube(x, y, freq, data)   # must not raise

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="3-D"):
            _validate_beam_cube(np.arange(5.0), np.arange(4.0),
                                np.array([400.0]), np.ones((4, 5)))

    def test_freq_mismatch(self):
        with pytest.raises(ValueError, match="freq has"):
            _validate_beam_cube(np.arange(5.0), np.arange(4.0),
                                np.array([400.0, 500.0]),
                                np.ones((3, 4, 5)))

    def test_y_mismatch(self):
        with pytest.raises(ValueError, match="y has"):
            _validate_beam_cube(np.arange(5.0), np.arange(3.0),
                                np.array([400.0]),
                                np.ones((1, 4, 5)))

    def test_x_mismatch(self):
        with pytest.raises(ValueError, match="x has"):
            _validate_beam_cube(np.arange(6.0), np.arange(4.0),
                                np.array([400.0]),
                                np.ones((1, 4, 5)))

    def test_all_nan_channel(self):
        x = np.arange(5, dtype=float)
        y = np.arange(4, dtype=float)
        freq = np.array([400.0])
        data = np.full((1, 4, 5), np.nan)
        with pytest.raises(ValueError, match="all-NaN"):
            _validate_beam_cube(x, y, freq, data)

    def test_load_beam_bad_shape(self, tmp_path):
        bad = tmp_path / "bad.npz"
        np.savez(str(bad),
                 x=np.arange(5.0), y=np.arange(4.0),
                 freq=np.array([0.4]),
                 data=np.ones((4, 5)))  # missing freq dimension
        with pytest.raises(ValueError, match="3-D"):
            load_beam(str(bad))


# ---------------------------------------------------------------------------
# 2. Multi-frequency fit (run_all) — CLI parsing only; full pipeline would
#    need the real example cube, so we only test the argument wiring here.
# ---------------------------------------------------------------------------

class TestRunAllCLI:
    def test_fit_all_defaults(self):
        args = build_parser().parse_args(["fit-all", "configs/config_fit.yaml"])
        assert args.command == "fit-all"
        assert args.channels is None
        assert args.coord is None
        assert args.datafile is None

    def test_fit_all_channels(self):
        args = build_parser().parse_args([
            "fit-all", "configs/config_fit.yaml",
            "--channels", "400", "500", "600",
        ])
        assert args.channels == pytest.approx([400.0, 500.0, 600.0])

    def test_fit_all_with_coord_and_datafile(self):
        args = build_parser().parse_args([
            "fit-all", "configs/config_fit.yaml",
            "--coord", "cartesian",
            "--datafile", "/tmp/cube.npz",
        ])
        assert args.coord == "cartesian"
        assert args.datafile == "/tmp/cube.npz"


# ---------------------------------------------------------------------------
# 3. CST stack
# ---------------------------------------------------------------------------

class TestCSTStack:
    def test_stack_shape(self, tmp_path):
        f1 = str(tmp_path / "f1.txt")
        f2 = str(tmp_path / "f2.txt")
        _write_mini_cst(f1)
        _write_mini_cst(f2)
        x, y, freq_arr, data = stack([f1, f2], freqs_mhz=[400, 500],
                                     size=31, xy_max=8.0)
        assert data.shape == (2, 31, 31)
        assert len(freq_arr) == 2
        assert abs(freq_arr[0] - 0.4) < 1e-9
        assert abs(freq_arr[1] - 0.5) < 1e-9

    def test_stack_mismatched_lengths(self, tmp_path):
        f1 = str(tmp_path / "f1.txt")
        _write_mini_cst(f1)
        with pytest.raises(ValueError, match="same length"):
            stack([f1], freqs_mhz=[400, 500], size=31, xy_max=8.0)

    def test_stack_x_y_axes_match_single_load(self, tmp_path):
        f1 = str(tmp_path / "f1.txt")
        _write_mini_cst(f1)
        x_single, y_single, _, _ = load_cst(f1, 400, size=31, xy_max=8.0)
        x_stacked, y_stacked, _, _ = stack([f1], freqs_mhz=[400],
                                           size=31, xy_max=8.0)
        assert np.allclose(x_single, x_stacked)
        assert np.allclose(y_single, y_stacked)

    def test_cst_stack_cli_args(self):
        args = build_parser().parse_args([
            "cst-stack", "a.txt", "b.txt",
            "--freqs", "400", "500",
            "--output", "/tmp/stack.npz",
            "--size", "256",
        ])
        assert args.command == "cst-stack"
        assert args.cst_files == ["a.txt", "b.txt"]
        assert args.freqs == pytest.approx([400.0, 500.0])
        assert args.output == "/tmp/stack.npz"
        assert args.size == 256

    def test_cst_stack_cli_missing_freqs(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["cst-stack", "a.txt", "--output", "x.npz"])


# ---------------------------------------------------------------------------
# 4. GenZTBeam polar (compute polar path)
# ---------------------------------------------------------------------------

class TestGenZTBeamPolar:
    def _make_coef_files(self, tmp_path, sigx=10.0, sigy=10.0):
        import pandas as pd
        coef_path = str(tmp_path / "coef.csv")
        sp_path   = str(tmp_path / "sp.csv")
        pd.DataFrame({"j": [0], "n": [0], "m": [0], "coef": [1.0]}).to_csv(
            coef_path, index=False)
        pd.DataFrame({"sigx": [sigx], "sigy": [sigy]}).to_csv(sp_path, index=False)
        return coef_path, sp_path

    def test_basisfunc_polar_shape(self, tmp_path):
        r     = np.linspace(0, 30, 20)
        theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
        coef_path, _ = self._make_coef_files(tmp_path)
        ztgen = GenZTBeam(400, r, theta, "float32", coord_type="polar")
        ztgen.load_coef(coef_path)
        basis = ztgen.basisfunc(10.0, 10.0)
        # Basis shape: (n_coef, n_theta * n_r)
        assert basis.shape == (1, len(theta) * len(r))

    def test_basisfunc_cartesian_unchanged(self, tmp_path):
        x = np.linspace(-30, 30, 25)
        y = np.linspace(-30, 30, 25)
        coef_path, _ = self._make_coef_files(tmp_path)
        ztgen = GenZTBeam(400, x, y, "float32", coord_type="cartesian")
        ztgen.load_coef(coef_path)
        basis = ztgen.basisfunc(10.0, 10.0)
        assert basis.shape == (1, len(y) * len(x))

    def test_compute_coord_cli(self):
        args = build_parser().parse_args(
            ["compute", "configs/config_compute.yaml", "--coord", "polar"])
        assert args.coord == "polar"

    def test_compute_coord_cli_default(self):
        args = build_parser().parse_args(["compute", "configs/config_compute.yaml"])
        assert args.coord is None
