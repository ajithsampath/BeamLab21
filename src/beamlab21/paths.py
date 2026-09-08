#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Path helpers.

All input/output locations in BeamLab21 are resolved relative to a ``base_dir``
that the caller controls, so the package works whether it is run from a source
checkout, a pip-installed environment, or an arbitrary working directory.
"""

import os
import warnings
from pathlib import Path


def resolve_path(path, base_dir):
    """Return ``path`` as an absolute :class:`~pathlib.Path`.

    Absolute paths are returned unchanged; relative paths are joined onto
    ``base_dir``.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


def resolve_under(base_dir, dir_value, file_value):
    """Resolve ``file_value`` inside ``dir_value``, both taken relative to ``base_dir``."""
    return resolve_path(file_value, resolve_path(dir_value, base_dir))


def default_base_dir(config_path):
    """Best-effort project root for a given config file.

    If the config lives in a ``configs/`` directory (the layout shipped with
    this repo) the parent of that directory is used; otherwise the current
    working directory is returned.
    """
    cfg = Path(config_path).resolve()
    if cfg.parent.name == "configs":
        return cfg.parent.parent
    return Path.cwd()


def get_project_root():
    """Deprecated: use :func:`resolve_path` / :func:`default_base_dir` instead."""
    warnings.warn(
        "get_project_root() is deprecated; paths are now resolved relative to a "
        "caller-supplied base_dir. This shim returns the current working directory.",
        DeprecationWarning,
        stacklevel=2,
    )
    return os.getcwd()
