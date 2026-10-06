import pytest

from beamlab21.cli import build_parser


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_fit_subcommand_parsed():
    args = build_parser().parse_args(["fit", "configs/config_fit.yaml", "--base-dir", "/tmp"])
    assert args.command == "fit"
    assert args.config == "configs/config_fit.yaml"
    assert args.base_dir == "/tmp"
    assert args.coord is None


def test_fit_coord_flag_polar():
    args = build_parser().parse_args(["fit", "configs/config_fit.yaml", "--coord", "polar"])
    assert args.coord == "polar"


def test_fit_coord_flag_cartesian():
    args = build_parser().parse_args(["fit", "configs/config_fit.yaml", "--coord", "cartesian"])
    assert args.coord == "cartesian"


def test_fit_coord_flag_auto():
    args = build_parser().parse_args(["fit", "configs/config_fit.yaml", "--coord", "auto"])
    assert args.coord == "auto"


def test_fit_coord_flag_invalid():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fit", "configs/config_fit.yaml", "--coord", "spherical"])


def test_compute_subcommand_parsed():
    args = build_parser().parse_args(["compute", "configs/config_compute.yaml"])
    assert args.command == "compute"
    assert args.config == "configs/config_compute.yaml"
    assert args.base_dir is None
