"""Top-level package for the MLCast source data validator."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("mlcast-dataset-validator")
except PackageNotFoundError:  # pragma: no cover - package metadata unavailable
    __version__ = "0.0.0"
