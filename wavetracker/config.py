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
    min_freq: float = 80.0
    """Lowest fundamental frequency considered [Hz]. The default excludes the
    50/60 Hz mains fundamental but includes low-frequency species (e.g.
    Sternopygus). Set the range of your species if known (e.g. 400-1200 Hz for
    Apteronotus leptorhynchus): it is faster and lets the interference filter
    remove combs with wider spacing."""
    max_freq: float = 2400.0
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
    max_harmonics: int | None = 10
    """Cap on the harmonics collected per group. The count is otherwise
    max_freq / min_freq * min_group_size - 1 (as in the original), which
    explodes for wide ranges (89 for the default 80-2400 Hz). Ranges needing
    fewer (e.g. 8 for 400-1200 Hz) are unaffected. None: no cap."""
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
    max_spacing: float | None = None
    """Highest comb fundamental [Hz]; must be below the lowest fish frequency,
    otherwise a resting fish's own harmonic series counts as a comb. None:
    min(300, 0.9 * harmonic_groups.min_freq)."""
    min_run: int = 4
    """Minimum number of consecutive teeth."""
    tooth_tolerance: float = 0.3
    """Maximum deviation of a tooth from k * spacing [Hz]."""
    search_max_freq: float = 3000.0
    """Combs are searched among lines below this frequency, then extended
    to all lines [Hz]."""
    max_line_freq: float | None = None
    """Lines above this frequency are ignored [Hz]. None: up to Nyquist."""
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
    gap_tolerance: float | None = None
    """If set, two detections may differ by at most gap_tolerance +
    gap_tolerance_rate * gap [Hz] (and freq_tolerance): stricter links across
    gaps for dense populations. None: original behaviour."""
    gap_tolerance_rate: float = 0.0
    """Growth of the gap tolerance [Hz/s]."""
    amplitude_feature: str = "minmax"
    """Electrode pattern used in the link error: "minmax" (original) or "db"
    (level ratios; use with few electrodes, e.g. 2)."""
    min_support: int = 0
    """Only track detections with at least this many other detections within
    support_window and support_freq (0: off). Removes isolated noise
    detections that otherwise bridge tracks of neighbouring fish."""
    support_window: float = 1.0
    """Time window for min_support [s] (+-)."""
    support_freq: float = 1.0
    """Frequency window for min_support [Hz] (+-)."""


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
    fine_spec_max_freq: float | None = None
    """Upper frequency limit of the stored fine spectrogram [Hz]. None: 1.25 x
    harmonic_groups.max_freq (the tracked range plus room for rises)."""
    sparse_spec_max_freq: float | None = None
    """Upper frequency limit of the overview spectrogram [Hz]. None: as for
    fine_spec_max_freq."""
    sparse_spec_time_bins: int = 4000
    """Approximate number of time bins of the overview spectrogram."""
    sparse_spec_freq_res: float = 2.0
    """Approximate frequency resolution of the overview spectrogram [Hz]."""


@dataclass
class PositionMergingConfig:
    """Grouping of track segments into fish by frequency and source position
    for moving electrodes (see wavetracker.position). Only used by
    ``wavetracker merge-by-position``; it does not affect ``run``."""

    # --- electrodes ---
    reference: int | None = None
    """Reference electrode: channel i measures electrode i (skipping the
    reference) minus the reference electrode. None: see channel_pairs."""
    channel_pairs: list[list[int]] | None = None
    """Explicit [plus, minus] electrode indices per channel (minus -1: distant
    ground). None and no reference: channel i = electrode i vs. ground."""
    t_start: float | None = None
    """Start of the survey [s, recording time]; None: start of the electrode
    track."""
    t_end: float | None = None
    """End of the survey [s]; None: end of the electrode track."""

    # --- segments and frequency grouping ---
    split_gap: float = 3.0
    """Identities are split into segments at gaps longer than this [s]."""
    min_segment_duration: float = 3.0
    """Shorter segments are clutter [s]."""
    min_segment_detections: int = 15
    """Segments with fewer detections are clutter."""
    freq_tolerance: float = 1.0
    """Segments join a candidate if their median frequency is this close [Hz]."""
    max_overlap: float = 1.0
    """Segments of one fish may overlap in time by at most this [s]."""
    resid_abs: float = 1.0
    """Position check: a segment is split off a candidate if its mean |log
    amplitude residual| exceeds max(resid_abs, resid_rel * median) ..."""
    resid_rel: float = 2.0
    """... (see resid_abs)."""
    merge_freq_tolerance: float = 1.5
    """Candidates are merged if the median frequencies of a segment of each are
    this close [Hz] (fish drift slowly) ..."""
    merge_se_factor: float = 2.0
    """... and their positions differ by less than this many combined
    standard errors ..."""
    merge_min_se: float = 0.1
    """... with each standard error at least this [m]."""

    # --- source model (wavetracker.position.efield) ---
    model: str = "monopoles"
    """"monopoles" (line of monopoles) or "dipole" (point dipole)."""
    body_length: float = 0.2
    """Length of the monopole line [m]."""
    n_poles: int = 10
    """Number of monopoles."""
    water_depth: float | None = None
    """Depth of the insulating bottom [m]; None: half-space."""
    n_images: int = 2
    """Truncation of the image series."""
    min_depth: float = 0.02
    """Minimum fish depth [m]."""
    max_depth: float | None = None
    """Maximum fish depth [m]; None: water_depth - 0.05 (or detection_range
    without a bottom)."""

    # --- observation model ---
    noise_quantile: float = 10.0
    """Percentile over frames of the per-channel spectrum giving the noise
    floor (from the recording), or of the detection powers near the fish's
    frequency (fallback without recording) [%]."""
    sign_weight: float = 1.0
    """Weight of the relative-sign residual."""
    strong_snr: float = 3.0
    """Channels this many times above the noise floor (amplitude) enter the
    relative-sign term."""
    censor_weight: float = 0.5
    """Weight of the censored-frame residual."""
    censor_freq_tolerance: float = 1.5
    """A survey frame is censored if no detection (of any identity) is within
    this of the fish's frequency [Hz]."""
    censor_stride: int = 2
    """Use every n-th censored frame."""
    censor_range_factor: float = 3.0
    """Censored frames farther than this times detection_range from all of a
    fish's detections are ignored (they cannot constrain it)."""
    detection_range: float = 2.5
    """Maximum 3-D distance of a detected fish to the nearest midpoint of an
    electrode pair at its detections [m]."""
    bound_weight: float = 300.0
    """Weight of the detection-range penalty."""

    # --- fit ---
    start_half_width: float = 1.5
    """Multi-start grid: half width around the strongest detections [m] ..."""
    start_grid: int = 7
    """... with this many points per axis ..."""
    start_headings: int = 8
    """... this many headings ..."""
    start_depths: list[float] = field(default_factory=lambda: [0.3, 0.8])
    """... and these depths [m]."""
    stage1_stride: int = 4
    """The multi-start runs on every n-th detection and censored frame ..."""
    n_refine: int = 6
    """... and the best n distinct basins are refined on all data."""
    alt_min_distance: float = 0.5
    """A competing basin must be this far from the best fit [m] ..."""
    ambiguity_dcost: float = 25.0
    """... and a fit is ambiguous if its cost is within this of the best."""
    jackknife_blocks: int = 8
    """Delete-a-block jackknife over this many blocks of a fish's detection
    times."""


@dataclass
class Config:
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    harmonic_groups: HarmonicGroupsConfig = field(default_factory=HarmonicGroupsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    position_merging: PositionMergingConfig = field(
        default_factory=PositionMergingConfig
    )

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
