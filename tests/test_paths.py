from pathlib import Path

from beamlab21.paths import default_base_dir, resolve_path


def test_resolve_absolute_is_unchanged(tmp_path):
    abs_in = tmp_path / "x" / "y.npz"
    assert resolve_path(abs_in, "/somewhere/else") == abs_in


def test_resolve_relative_joins_base(tmp_path):
    out = resolve_path("outputs/model.npz", tmp_path)
    assert out == (tmp_path / "outputs" / "model.npz").resolve()


def test_default_base_dir_for_configs_layout(tmp_path):
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    cfg = cfgdir / "config_fit.yaml"
    cfg.write_text("frequency: 400\n")
    assert default_base_dir(cfg) == tmp_path


def test_default_base_dir_falls_back_to_cwd(tmp_path):
    cfg = tmp_path / "config_fit.yaml"
    cfg.write_text("frequency: 400\n")
    assert default_base_dir(cfg) == Path.cwd()
