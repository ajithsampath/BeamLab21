#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

"""Fetch the example beam cube used by the tutorial and default configs.

The cube (~39 MB) is not stored in the repository (removed for confidentiality; it
can be provided on request via the collaboration). If you have a copy, place it at
``data/Example_cube.npz``. To download it from a location you control, the URL is
looked up in this order:

1. an explicit ``url`` argument,
2. the ``BEAMLAB21_DATA_URL`` environment variable,
3. a ``data/DATA_URL.txt`` file (one line, the URL),
4. the :data:`EXAMPLE_DATA_URL` constant below.

TODO(maintainer): once ``Example_cube.npz`` is published to a stable location
(Zenodo / GitHub release asset / institutional storage), paste the URL into
:data:`EXAMPLE_DATA_URL` so ``beamlab21 fetch-data`` works with no extra setup.
"""

import os
import urllib.request
from pathlib import Path

EXAMPLE_DATA_URL = ""  # <-- paste the Example_cube.npz download URL here

def _default_dest():
    return Path.cwd() / "data" / "Example_cube.npz"


def resolve_data_url(url=None):
    """Return the first configured download URL, or ``None`` if none is set.

    Order: explicit ``url`` -> ``BEAMLAB21_DATA_URL`` -> ``data/DATA_URL.txt``
    (relative to the current directory) -> :data:`EXAMPLE_DATA_URL`.
    """
    if url:
        return url.strip()
    env = os.environ.get("BEAMLAB21_DATA_URL")
    if env:
        return env.strip()
    url_file = Path.cwd() / "data" / "DATA_URL.txt"
    if url_file.is_file():
        text = url_file.read_text().strip()
        if text:
            return text
    return EXAMPLE_DATA_URL.strip() or None


def fetch_example_data(dest=None, url=None, force=False):
    """Download the example cube to ``dest`` (default ``data/Example_cube.npz``).

    Returns the destination :class:`~pathlib.Path`. Raises ``RuntimeError`` if no
    URL is configured.
    """
    dest = Path(dest) if dest is not None else _default_dest()

    if dest.is_file() and not force:
        print(f"{dest} already exists; nothing to do (use force=True to re-download).")
        return dest

    resolved = resolve_data_url(url)
    if not resolved:
        raise RuntimeError(
            "No data URL configured. Set BEAMLAB21_DATA_URL, create data/DATA_URL.txt, "
            "or set EXAMPLE_DATA_URL in beamlab21/data.py. See data/README.md."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {resolved}\n     -> {dest}")
    urllib.request.urlretrieve(resolved, dest)  # noqa: S310 - trusted, user-supplied URL
    print(f"Done ({dest.stat().st_size / 1e6:.1f} MB).")
    return dest
