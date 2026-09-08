import pytest

from beamlab21.cli import build_parser
from beamlab21.data import resolve_data_url


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_fit_subcommand_parsed():
    args = build_parser().parse_args(["fit", "configs/config_fit.yaml", "--base-dir", "/tmp"])
    assert args.command == "fit"
    assert args.config == "configs/config_fit.yaml"
    assert args.base_dir == "/tmp"


def test_fetch_data_flags_parsed():
    args = build_parser().parse_args(["fetch-data", "--url", "http://example/x.npz", "--force"])
    assert args.command == "fetch-data"
    assert args.url == "http://example/x.npz"
    assert args.force is True


def test_resolve_data_url_prefers_explicit(monkeypatch):
    monkeypatch.delenv("BEAMLAB21_DATA_URL", raising=False)
    assert resolve_data_url("http://example/cube.npz") == "http://example/cube.npz"


def test_resolve_data_url_env(monkeypatch):
    monkeypatch.setenv("BEAMLAB21_DATA_URL", "http://env/cube.npz")
    assert resolve_data_url() == "http://env/cube.npz"
