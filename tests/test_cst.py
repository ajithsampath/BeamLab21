#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Tests for beamlab21.cst: CST far-field loader and converter."""

import os
import tempfile

import numpy as np
import pytest

from beamlab21.cli import build_parser
from beamlab21.cst import load_cst

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cst_txt(path, n_theta=20, n_phi=36):
    """Write a minimal synthetic CST farfield text file."""
    header = (
        "Theta [deg.]  Phi   [deg.]  Abs(E   )[dB(V/m)]   "
        "Abs(Cross)[dB(V/m)]  Phase(Cross)[deg.]  "
        "Abs(Copol)[dB(V/m)]  Phase(Copol)[deg.]  Ax.Ratio[dB    ]    \n"
        + "-" * 90 + "\n"
    )
    thetas = np.linspace(-10.0, 10.0, n_theta)
    phis   = np.linspace(-90.0, 90.0, n_phi)
    rows = []
    for theta in thetas:
        for phi in phis:
            # copol: Gaussian in theta, dB
            copol_lin = np.exp(-(theta**2) / (2 * 4.0**2))
            copol_dB  = 20 * np.log10(max(copol_lin, 1e-10))
            e_dB = copol_dB + 0.1   # slightly different total-E
            rows.append(
                f"{theta:12.3f} {phi:12.3f} {e_dB:20.3e} {-40.0:20.3e} "
                f"{0.0:20.3f} {copol_dB:20.3e} {0.0:20.3f} {50.0:12.3e}\n"
            )
    with open(path, "w") as f:
        f.write(header)
        f.writelines(rows)


@pytest.fixture(scope="module")
def cst_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("cst") / "farfield.txt"
    _write_cst_txt(str(p))
    return str(p)


# ---------------------------------------------------------------------------
# load_cst
# ---------------------------------------------------------------------------

def test_load_cst_output_shapes(cst_file):
    x, y, freq_arr, data = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    assert x.shape == (51,)
    assert y.shape == (51,)
    assert freq_arr.shape == (1,)
    assert data.shape == (1, 51, 51)


def test_load_cst_freq_stored_in_ghz(cst_file):
    _, _, freq_arr, _ = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    assert abs(freq_arr[0] - 0.4) < 1e-9


def test_load_cst_x_y_axes_symmetric(cst_file):
    x, y, _, _ = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    assert abs(x[0] + x[-1]) < 1e-10
    assert abs(y[0] + y[-1]) < 1e-10


def test_load_cst_data_nonnegative(cst_file):
    _, _, _, data = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    assert np.all(data >= 0)


def test_load_cst_no_nan(cst_file):
    _, _, _, data = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    assert not np.any(np.isnan(data))


def test_load_cst_column_e(cst_file):
    _, _, _, data_copol = load_cst(cst_file, freq_mhz=400, column="copol",
                                   size=51, xy_max=10.0)
    _, _, _, data_e     = load_cst(cst_file, freq_mhz=400, column="e",
                                   size=51, xy_max=10.0)
    # e and copol columns differ in the synthetic file — results should differ
    assert not np.allclose(data_copol, data_e)


def test_load_cst_bad_column(cst_file):
    with pytest.raises(ValueError, match="column must be one of"):
        load_cst(cst_file, freq_mhz=400, column="crosspol", size=51, xy_max=10.0)


def test_load_cst_peak_near_centre(cst_file):
    x, y, _, data = load_cst(cst_file, freq_mhz=400, size=51, xy_max=10.0)
    peak_idx = np.unravel_index(np.argmax(data[0]), data[0].shape)
    centre = data[0].shape[0] // 2
    assert abs(peak_idx[0] - centre) <= 5
    assert abs(peak_idx[1] - centre) <= 5


# ---------------------------------------------------------------------------
# CLI parsing for the cst subcommand
# ---------------------------------------------------------------------------

def test_cst_cli_required_freq():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["cst", "farfield.txt"])  # missing --freq


def test_cst_cli_defaults():
    args = build_parser().parse_args(["cst", "farfield.txt", "--freq", "400"])
    assert args.command == "cst"
    assert args.cst_file == "farfield.txt"
    assert args.freq == 400.0
    assert args.column == "copol"
    assert args.size == 1501
    assert args.xy_max == 75.0
    assert args.cube_output is None
    assert args.skip_fit is False


def test_cst_cli_all_options():
    args = build_parser().parse_args([
        "cst", "farfield.txt",
        "--freq", "800",
        "--column", "e",
        "--size", "256",
        "--xy-max", "45",
        "--cube-output", "/tmp/cube.npz",
        "--config", "my_config.yaml",
        "--base-dir", "/data",
        "--skip-fit",
    ])
    assert args.freq == 800.0
    assert args.column == "e"
    assert args.size == 256
    assert args.xy_max == 45.0
    assert args.cube_output == "/tmp/cube.npz"
    assert args.config == "my_config.yaml"
    assert args.base_dir == "/data"
    assert args.skip_fit is True


def test_cst_cli_invalid_column():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["cst", "farfield.txt", "--freq", "400",
                                   "--column", "crosspol"])


# ---------------------------------------------------------------------------
# run() with skip_fit=True (conversion only — no fit config needed)
# ---------------------------------------------------------------------------

def test_cst_run_skip_fit_saves_npz(cst_file):
    from beamlab21.cst import run as cst_run

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "cube.npz")
        cst_run(cst_file, freq_mhz=400, cube_output=out_path,
                skip_fit=True, size=51, xy_max=10.0)

        assert os.path.exists(out_path)
        cube = np.load(out_path)
        assert "x" in cube and "y" in cube and "data" in cube and "freq" in cube
        assert cube["data"].shape == (1, 51, 51)
