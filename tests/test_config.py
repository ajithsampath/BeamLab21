import textwrap

import pytest

from beamlab21.config import load_config


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.yaml")


def test_frequency_templated_into_names(tmp_path):
    cfg = _write(
        tmp_path,
        """
        frequency: 550
        out_name: "coefficients_{{ frequency }}.csv"
        """,
    )
    parsed = load_config(cfg)
    assert parsed["frequency"] == 550
    assert parsed["out_name"] == "coefficients_550.csv"


def test_explicit_context_overrides_file(tmp_path):
    cfg = _write(
        tmp_path,
        """
        frequency: 400
        out_name: "model_{{ frequency }}"
        """,
    )
    parsed = load_config(cfg, context={"frequency": 700})
    assert parsed["out_name"] == "model_700"
