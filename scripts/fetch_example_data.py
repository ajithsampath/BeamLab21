#!/usr/bin/env python3
"""Download the example beam cube used by the default configs.

Thin wrapper around ``beamlab21.data.fetch_example_data`` / ``beamlab21 fetch-data``.
Provide the URL via ``BEAMLAB21_DATA_URL``, ``data/DATA_URL.txt``, or
``EXAMPLE_DATA_URL`` in ``src/beamlab21/data.py``. See ``data/README.md``.

Usage:
    python scripts/fetch_example_data.py
"""

import sys

from beamlab21.data import fetch_example_data

if __name__ == "__main__":
    try:
        fetch_example_data()
    except RuntimeError as exc:
        sys.exit(str(exc))
