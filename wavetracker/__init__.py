"""Detect and track wave-type electric fish in multi-electrode recordings."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wavetracker")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
