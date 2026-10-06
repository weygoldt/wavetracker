"""Typed configuration for the wavetracker pipeline.

All parameters live in a single :class:`Config` made of one dataclass per
pipeline stage. A config can be written to / read from YAML; unknown keys are
rejected so typos do not silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SpectrogramConfig:
    nfft: int = 2**15
    """Samples per FFT window (frequency resolution = rate / nfft)."""
    overlap_frac: float = 0.9
    """Overlap of consecutive FFT windows (0-1)."""
    block_duration: float = 60.0
    """Seconds of data processed per block on the device."""
    exclude_channels: list[int] = field(default_factory=list)
    """Electrodes to ignore (e.g. broken ones picking up interference)."""


@dataclass
class HarmonicGroupsConfig:
    min_freq: float = 400.0
    """Lowest fundamental frequency considered [Hz]."""
    max_freq: float = 1200.0
    """Highest fundamental frequency considered [Hz]."""
    low_threshold: float | None = None
    """Peak detection threshold [dB]; estimated from the noise floor if None."""
    high_threshold: float | None = None
    """Threshold for 'good' peaks [dB]; estimated from the noise floor if None."""
    low_thresh_factor: float = 6.0
    """Multiple of the noise std used for the low threshold."""
    high_thresh_factor: float = 10.0
    """Multiple of the noise std used for the high threshold."""
    max_freq_tol: float = 1.0
    """Tolerance when matching a peak to a harmonic, in fundamental units [Hz]."""
    mains_freq: float = 50.0
    """Mains frequency; its harmonics are excluded [Hz]."""
    mains_freq_tol: float = 1.0
    """Tolerance around mains harmonics [Hz]."""
    max_divisor: int = 3
    """Peaks are tested as harmonics 1..max_divisor of a fundamental."""
    min_group_size: int = 3
    """Number of lowest harmonics that must all be present."""
    min_good_peak_power: float = -100.0
    """Minimum power of a fundamental [dB]."""
    exclusive_harmonics: str = "core"
    """Which peaks of an accepted fish are unavailable to further fish:
    "core" (its `min_group_size` lowest harmonics) or "all" (every harmonic,
    as in Raab et al. 2022; drops fish whose harmonics coincide by chance)."""
    max_groups_per_frame: int = 64
    """Upper bound on fish detected in a single spectrum."""
    refine_frequency: bool = True
    """Refine fundamentals with parabolic interpolation (sub-bin precision)."""


@dataclass
class TrackingConfig:
    freq_tolerance: float = 2.5
    """Maximum frequency difference of two detections to be linked [Hz]."""
    max_dt: float = 10.0
    """Maximum time difference of two detections to be linked [s]."""


@dataclass
class OutputConfig:
    save_fine_spec: bool = False
    """Store the full-resolution summed spectrogram (large!)."""
    fine_spec_max_freq: float = 2000.0
    """Upper frequency limit of the stored fine spectrogram [Hz]."""
    sparse_spec_max_freq: float = 2000.0
    """Upper frequency limit of the overview spectrogram [Hz]."""
    sparse_spec_time_bins: int = 4000
    """Approximate number of time bins of the overview spectrogram."""
    sparse_spec_freq_res: float = 2.0
    """Approximate frequency resolution of the overview spectrogram [Hz]."""


@dataclass
class Config:
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    harmonic_groups: HarmonicGroupsConfig = field(default_factory=HarmonicGroupsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        data = data or {}
        sections = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"Unknown config section(s): {sorted(unknown)}")
        kwargs = {}
        for name, f in sections.items():
            section_cls = f.default_factory  # type: ignore[misc]
            values = data.get(name) or {}
            valid = {sf.name for sf in fields(section_cls)}
            bad = set(values) - valid
            if bad:
                raise ValueError(f"Unknown key(s) in '{name}': {sorted(bad)}")
            kwargs[name] = section_cls(**values)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load a YAML config; returns the defaults if `path` is None."""
        if path is None:
            return cls()
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
