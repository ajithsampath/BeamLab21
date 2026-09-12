"""BeamLab21 - beam computation & characterization tools for 21cm arrays."""

from importlib.metadata import PackageNotFoundError, version

from beamlab21 import compute, fit

try:
    __version__ = version("beamlab21")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.1.0"

__all__ = ["fit", "compute", "__version__"]
