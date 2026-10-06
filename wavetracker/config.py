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
    """Mains frequency; its harmonics are excluded [Hz]. 0 disables this
    (battery-powered field recordings)."""
    mains_freq_tol: float = 1.0
    """Tolerance around mains harmonics [Hz]."""
    max_divisor: int = 3
    """Peaks are tested as harmonics 1..max_divisor of a fundamental."""
    max_harmonics: int | None = None
    """Cap on the harmonics collected per group. Without it the count is
    max_freq / min_freq * min_group_size - 1, which explodes for wide
    frequency ranges (e.g. 199 for 20-2000 Hz)."""
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
class InterferenceConfig:
    """Removal of stationary interference combs (see wavetracker.interference)."""

    enabled: bool = True
    line_threshold: float = 10.0
    """Persistent excess over the noise baseline for a line [dB]."""
    persistence_quantile: float = 0.2
    """Quantile over time defining the persistent spectrum (0.2: a line must
    be present in >80% of a block's frames)."""
    subtract_quantile: float = 0.9
    """Quantile over a block's frames defining a tooth's gate level."""
    subtract_margin: float = 3.0
    """Margin added to the gate level [dB]."""
    level_history_blocks: int = 5
    """Gate level: minimum over this many recent blocks. Long protects fish
    resting near a tooth (they cannot raise the level); short follows
    changes of the interference level faster (fewer leaks). 2 instead of 5
    halved hum leaks in 2022-06-02 but cost 3% of a fish's detections near a
    tooth in 2022-06-14."""
    history_blocks: int = 5
    """A line must be persistent at the same bin in this many consecutive
    blocks (protects resting fish that slowly drift past a tooth)."""
    neighbour_distance: float = 4.0
    """Two detections of a frame this close [Hz] with a tooth between them
    (or within `neighbour_tolerance` of one) are one fish split by the
    tooth; the weaker is dropped."""
    neighbour_tolerance: float = 0.4
    """Distance to a tooth counting as 'on the tooth' [Hz]."""
    tooth_neighbour_distance: float = 8.0
    """A detection on a tooth with a stronger non-tooth detection this close
    [Hz] is interference lifted by the fish's spectral leakage; dropped."""
    tooth_memory_blocks: int = 30
    """A tooth found on an electrode keeps being removed for this many blocks."""
    baseline_width: int = 301
    """Width of the running median giving the noise baseline [bins]."""
    min_spacing: float = 20.0
    """Lowest comb fundamental [Hz]."""
    max_spacing: float = 300.0
    """Highest comb fundamental [Hz]; must be well below the lowest fish
    frequency, otherwise a fish's own harmonic series counts as a comb."""
    min_run: int = 4
    """Minimum number of consecutive teeth."""
    tooth_tolerance: float = 0.3
    """Maximum deviation of a tooth from k * spacing [Hz]."""
    search_max_freq: float = 3000.0
    """Combs are searched among lines below this frequency, then extended
    to all lines [Hz]."""
    max_line_freq: float = 10000.0
    """Lines above this frequency are ignored [Hz]."""
    min_frames: int = 30
    """Blocks with fewer frames reuse the previous block's combs."""
    frame_stride: int = 4
    """Use every n-th frame to estimate the persistent spectrum (speed)."""


@dataclass
class TrackingConfig:
    freq_tolerance: float = 2.5
    """Maximum frequency difference of two detections to be linked [Hz]."""
    max_dt: float = 10.0
    """Maximum time difference of two detections to be linked [s]."""


@dataclass
class StitchingConfig:
    """Joining track fragments across rises (see wavetracker.stitching)."""

    enabled: bool = True
    max_gap: float = 30.0
    """Maximum time between the end of one fragment and the next [s]."""
    max_overlap: float = 30.0
    """Maximum temporal overlap of joined fragments [s]."""
    overlap_tolerance: float = 2.5
    """Overlapping fragments must have the same median frequency within this
    [Hz] (as for linking detections in tracking.freq_tolerance)."""
    max_rise: float = 100.0
    """Largest upward frequency jump at a join (a rise onset) [Hz]."""
    max_drop: float = 2.5
    """Largest downward frequency jump at a join [Hz]."""
    baseline_window: float = 60.0
    """Window for the baseline frequency at a fragment's end [s]; twice as
    long at its start, where a rise may still decay."""
    baseline_quantile: float = 0.1
    """Quantile defining the baseline (low: ignores upward rises)."""
    baseline_tolerance: float = 3.0
    """Maximum baseline difference of joined fragments [Hz]."""
    rise_max_gap: float = 2.0
    """A join with a gap up to this [s] ..."""
    rise_min_jump: float = 3.0
    """... and an upward jump of at least this [Hz] is a rise onset ..."""
    rise_baseline_tolerance: float = 6.0
    """... and may differ in baseline by up to this [Hz]."""
    max_dropout: float = 900.0
    """Longest gap bridged when nothing else is at that frequency [s]."""
    dropout_tolerance: float = 1.5
    """Maximum baseline difference across a dropout [Hz]."""
    pattern_window: float = 30.0
    """Window for the electrode amplitude pattern at fragment edges [s]."""
    max_pattern_distance: float | None = None
    """Maximum RMS difference of the max-normalized electrode amplitude
    patterns (None: not checked). In the 2022 tube recordings it did not
    separate the two fish, so it is off by default."""
    min_detections: int = 10
    """Fragments with fewer detections are not joined."""


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
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    harmonic_groups: HarmonicGroupsConfig = field(default_factory=HarmonicGroupsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
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
