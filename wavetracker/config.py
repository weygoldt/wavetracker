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


def _fast_fft_size(target: float) -> int:
    """The even number without prime factors above 5 closest to `target`."""
    best, top = 2, 2.0 * max(target, 2.0)
    p2 = 2
    while p2 <= top:
        p3 = p2
        while p3 <= top:
            p5 = p3
            while p5 <= top:
                if abs(p5 - target) < abs(best - target):
                    best = p5
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return best


@dataclass
class SpectrogramConfig:
    freq_resolution: float = 20000.0 / 2**15
    """Frequency resolution [Hz], the same at any sampling rate: the FFT window
    lasts 1 / freq_resolution (1.64 s) and nfft is the fast FFT size closest to
    rate / freq_resolution (2**15 at 20 kHz, 78732 at 48 kHz)."""
    nfft: int | None = None
    """Samples per FFT window; overrides `freq_resolution` when set. The
    config stored with the results (wavetracker.json) holds the nfft used."""
    overlap_frac: float = 0.9
    """Overlap of consecutive FFT windows (0-1)."""
    block_duration: float = 60.0
    """Seconds of data processed per block on the device."""
    exclude_channels: list[int] = field(default_factory=list)
    """Electrodes to ignore (e.g. broken ones picking up interference)."""

    def resolve_nfft(self, rate: float) -> int:
        """Samples per FFT window at sampling rate `rate`."""
        if self.nfft is not None:
            return int(self.nfft)
        if not self.freq_resolution > 0:
            raise ValueError("spectrogram.freq_resolution must be positive")
        return _fast_fft_size(rate / self.freq_resolution)


@dataclass
class HarmonicGroupsConfig:
    min_freq: float = 80.0
    """Lowest fundamental frequency considered [Hz]. The default 80-2400 Hz
    is a broad catch-all (excludes the mains fundamental, includes
    Sternopygus to fast Apteronotus). If the fish frequencies or their
    expected range are known, set min_freq/max_freq explicitly to a narrow
    band (e.g. 400-1200 Hz for A. leptorhynchus): fewer false fish, better
    interference removal, faster."""
    max_freq: float = 2400.0
    """Highest fundamental frequency considered [Hz]."""
    low_threshold: float | None = None
    """Peak detection threshold [dB]; follows the noise of the recording if None."""
    high_threshold: float | None = None
    """Threshold for 'good' peaks [dB]; follows the noise of the recording if None."""
    threshold_window: float | None = 10.0
    """The noise that sets the thresholds is the median over this many seconds
    around a frame, so that they follow changes such as electrodes going into
    the water. None: estimate once from the first block and keep them for the
    whole recording (as in the original)."""
    threshold_band: float | None = 100.0
    """Thresholds also follow the noise over frequency, in bands of this width
    [Hz]. Noise at low frequencies is often common to all electrodes, and then
    has several times the spread of the high band the thresholds are based on.
    None: one value per frame (requires `threshold_window`)."""
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
    max_missing_harmonics: int = 0
    """Gaps allowed among the lowest harmonics: a fish needs its fundamental
    and `min_group_size` of its first `min_group_size + max_missing_harmonics`
    harmonics. 1 with min_group_size 2 ("2 of the first 3") finds
    odd-harmonic fish without a 2nd harmonic (common below ~350 Hz in the
    field). Peaks are then visited by power and first tried as the
    fundamental of a group (a good peak); groups with gaps only use
    unclaimed peaks, so sub-harmonics of accepted fish are not reported.
    0: original behaviour."""
    min_good_peak_power: float | None = -100.0
    """Minimum absolute power of a fundamental [dB] (as in the original, for
    fishgrid amplifier levels). Its scale depends on the recording's gain:
    in quiet audio recordings (e.g. field recorders, floor around -120 dB)
    it removes weak fish that pass the relative thresholds; a warning is
    logged then. None: no absolute limit."""
    exclusive_harmonics: str = "core"
    """Which peaks of an accepted fish are unavailable to further fish:
    "core" (its `min_group_size` lowest harmonics; for groups with gaps the
    first `min_group_size + max_missing_harmonics`) or "all" (every harmonic,
    as in Raab et al. 2022; drops fish whose harmonics coincide by chance)."""
    max_groups_per_frame: int = 64
    """Upper bound on fish detected in a single spectrum."""
    refine_frequency: bool = True
    """Refine fundamentals with parabolic interpolation (sub-bin precision)."""


@dataclass
class DetectorConfig:
    method: str = "classic"
    """How fundamentals are found in each frame: "classic" (spectral peaks
    grouped into harmonic series by rules, see `harmonic_groups`) or
    "learned" (a neural network's salience map, see `wavetracker.salience`).
    Both use the fish range harmonic_groups.min_freq/max_freq."""
    model: str | None = None
    """Learned detector: path to the model file; None uses the packaged model.
    The spectrogram resolution and frame step must match the model's."""
    threshold: float = 0.6
    """Learned detector: minimum salience (0-1) of a detection (0.6 was best
    for the packaged model on the dense benchmark, lab and field noise)."""
    pair_threshold: float | None = None
    """Learned detector with a pair head: minimum pair probability (0-1) at
    which a salience peak becomes two detections, two fish closer than the
    spectrogram resolves. None: the same as `threshold`. Models without a
    pair head ignore it."""

    def __post_init__(self):
        if self.method not in ("classic", "learned"):
            raise ValueError(
                f"detector.method must be 'classic' or 'learned', not {self.method!r}"
            )
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("detector.threshold must be between 0 and 1")
        if self.pair_threshold is not None and not 0.0 < self.pair_threshold <= 1.0:
            raise ValueError("detector.pair_threshold must be between 0 and 1")


@dataclass
class ClickConfig:
    """Removal of impulsive noise before the spectrogram (see wavetracker.clicks)."""

    enabled: bool = True
    threshold: float = 12.0
    """A click must reach this many robust standard deviations of the channel's high
    band, which Gaussian noise never does. Crackle in the water that stays below it
    raises the noise estimate, and with it the thresholds. Measured on real fish at
    night and on lab recordings: 24 found few of the fish; 8 found the most, but
    broke the tracks of 2 of 16 lab fish apart."""
    extent_threshold: float = 8.0
    """A click lasts for as long as it stays above this many standard deviations."""
    high_band: float = 0.2
    """Clicks are searched above this fraction of the sampling rate, ..."""
    high_band_edge: float = 0.05
    """... reached over this further fraction. Fish must not reach it; the noise
    estimate of the thresholds relies on fish-free spectra above 0.25 too."""
    max_duration: float = 0.5
    """Events longer than this [s] are real signals (a boat, rain), not clicks."""
    guard: float = 0.002
    """Also repaired on both sides of a click: its decay below the threshold [s]."""
    resolution: float = 0.001
    """Time resolution of the search [s]."""
    edge: float = 0.005
    """Not searched at the very ends of a block [s]: see wavetracker.clicks."""
    line_margin: float = 15.0
    """What stands this far above the spectral floor of a block counts as
    narrowband (fish, their chirps, hum), which is never repaired [dB]."""
    line_width: float = 10.0
    """Frequency resolution of the spectral floor [Hz]."""
    floor_width: float = 1000.0
    """Range the spectral floor is the `floor_quantile` of [Hz]: wider than a
    fish's harmonics sweep in a block, including chirps."""
    floor_quantile: float = 0.2
    """Quantile over that range: low, because a loud fish can fill most of it."""
    local_window: float = 0.05
    """Window over which the narrowband part in a hole is scaled to what the good
    samples around it hold [s]: long enough for some periods of the lowest fish."""
    fade: float = 0.5
    """Length of the fade that makes a block periodic for the spectral split [s]."""
    sample_stride: int = 16
    """Use every n-th sample to estimate the noise of the high band."""


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
    start_salience: float | None = None
    """If set, only detections with at least this salience (0-1, `sal_v`)
    count as evidence for a new identity; weaker ones may still join existing
    identities (ByteTrack-style association). None: all detections are alike."""
    start_evidence: float = 0.5
    """With start_salience: a new identity needs confident detections in this
    many seconds of frames (round(start_evidence / frame step), at least 1)
    of the temporary track it starts from, which spans at most 3 * max_dt [s]."""
    motion_window: float | None = None
    """If set, links follow each fish's frequency slope: a first pass tracks as
    usual, every detection gets the slopes of its identity over the
    motion_window [s] before and after it, and the second pass compares the
    later detection of a link with the frequency predicted from the earlier
    one's past slope, and the earlier with the later one's future slope (the
    mean of both errors), in the link error and the tolerances. Slopes from
    either side of a crossing are not mixed, so two fish that cross keep their
    own. Each slope is shrunk by its own uncertainty, and a prediction reaches
    at most motion_window ahead. None: off (frequencies compared directly)."""
    motion_max_slope: float = 2.0
    """Slopes are clipped to +-this [Hz/s] (rise onsets are jumps, not slopes)."""
    min_identity_salience: float | None = None
    """If set, identities whose mean detection salience (0-1, `sal_v`) is
    below this are dropped after linking (or stitching): lines of harmonics
    and clutter that no confident detection supports. Weak detections inside a
    confident identity stay. None: off."""
    method: str = "classic"
    """"classic": the rule-based tracker (Raab et al. 2022, with the options
    above). "learned": a trained network scores the links between detections
    of the next few seconds and a frame-by-frame linking builds the tracks
    (wavetracker.assocnet); of the options above only min_identity_salience
    applies. Needs assoc_model."""
    assoc_mode: str = "link"
    """Learned association: "link" builds tracks from the learned links alone
    (frame by frame, links up to the model's horizon of a few seconds); "cost"
    keeps the rule-based tracker (windows, temporary identities, gaps up to
    max_dt and the options above) and replaces its error for links within the
    model's horizon by 1 - the learned probability, dropping links below
    assoc_threshold."""
    assoc_model: str | None = None
    """Weights of the learned association (method "learned"), trained with
    benchmarks/dense/assocnet.py; not packaged."""
    assoc_threshold: float = 0.5
    """Learned association: minimum link probability (0-1)."""
    assoc_min_detections: int = 1
    """Learned association: tracks with fewer detections stay unassigned."""
    assoc_device: str = "cpu"
    """Learned association: torch device of the network."""

    def __post_init__(self):
        for key in ("start_salience", "min_identity_salience"):
            v = getattr(self, key)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"tracking.{key} must be between 0 and 1")
        if self.method not in ("classic", "learned"):
            raise ValueError(
                f"tracking.method must be 'classic' or 'learned', not {self.method!r}"
            )
        if self.assoc_mode not in ("link", "cost"):
            raise ValueError(
                f"tracking.assoc_mode must be 'link' or 'cost', not {self.assoc_mode!r}"
            )
        if self.method == "learned" and not self.assoc_model:
            raise ValueError("tracking.method 'learned' needs tracking.assoc_model")
        if not 0.0 <= self.assoc_threshold <= 1.0:
            raise ValueError("tracking.assoc_threshold must be between 0 and 1")


@dataclass
class StitchingConfig:
    """Joining track fragments across rises (see wavetracker.stitching);
    only used with linking.enabled: false."""

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
class LinkingConfig:
    """Joining fragments with a learned link score (see wavetracker.linking);
    replaces `stitching` when enabled (the default)."""

    enabled: bool = True
    """Join the tracker's fragments with the learned linker instead of the
    rule-based stitching (`stitching`, used when this is false)."""
    threshold: float = 0.9
    """Minimum probability of a link (0-1). Higher: purer identities, more
    fragments. 0.9 keeps identities as pure as the stitching on the
    benchmark; 0.5 joins more (IDF1 +0.02) at a purity cost of ~0.3 %."""
    max_gap: float = 180.0
    """Longest time between two fragments that may be joined [s]."""
    max_df: float = 5.0
    """Candidate links: the frequencies at the junction, the baselines or the
    medians of the two fragments differ by at most this [Hz]."""
    rise_df: float = 0.0
    """Also consider jumps up to this [Hz] across gaps up to rise_gap (rise
    onsets). 0: off; they were rarely needed and joined different fish on
    field data."""
    rise_gap: float = 30.0
    """See rise_df [s]."""
    max_shared: float = 0.05
    """Two identities may share at most this fraction of the smaller one's
    frames (one fish is detected once per frame)."""
    parallel_df: float = 3.0
    """Two identities that run at the same time (detections within
    parallel_dt) at frequencies more than this apart [Hz] are two fish ..."""
    parallel_dt: float = 0.3
    """... [s] ..."""
    max_parallel: float = 0.3
    """... if that holds for more than this fraction of their simultaneous
    detections (and at least 3): they are not joined."""
    min_detections: int = 10
    """Fragments with fewer detections are not joined."""
    model: str | None = None
    """Path to model weights (.npz); None: the model shipped with wavetracker."""


@dataclass
class RisesConfig:
    """Joining identities across rise onsets (see wavetracker.rises), after
    linking (or stitching) and before `tracking.min_identity_salience`.

    At a rise the frequency jumps up by a few to about 60 Hz within about a
    second, too far for the tracker and the linker's candidates, and the fish
    goes on as a new identity. An identity that stops is joined to one that
    starts shortly after, a rise-sized jump above it, with the same electrode
    pattern."""

    enabled: bool = False
    stop_gap: float = 2.0
    """An identity stops (or starts) where it has no detection for this long
    after (before) a detection [s]."""
    max_gap: float = 5.0
    """The later identity starts at most this long after the earlier one
    stops [s] ..."""
    max_overlap: float = 4.0
    """... or at most this long before it [s]: detectors with temporal
    context report the old frequency for a few seconds after the onset. The
    earlier identity's detections after the later one's start are dropped."""
    min_jump: float = 1.5
    """Smallest jump up from the earlier identity's last to the later one's
    first frequencies (medians over edge_window) [Hz] ..."""
    max_jump: float = 35.0
    """... and the largest [Hz]."""
    edge_window: float = 1.0
    """Frequencies at a stop or start: median over this [s]."""
    pattern_window: float = 5.0
    """Electrode patterns: mean power over this before the stop and after the
    start [s]."""
    max_pattern_rms: float = 7.0
    """Largest rms difference of the two electrode patterns [dB] ..."""
    min_pattern_corr: float = 0.5
    """... and the smallest correlation."""
    min_duration: float = 1.0
    """The later identity spans at least this after its start [s]."""
    min_salience: float | None = 0.4
    """Both identities need a mean detection salience (0-1, `sal_v`) of at
    least this over their pattern windows: the electrode pattern of a fish
    near the noise is noise, and two faint fish look alike. None: not
    checked."""
    max_shared: float = 0.3
    """Two identities may share at most this fraction of the smaller one's
    frames (one fish is detected once per frame)."""


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


YAML_HEADER = """\
# wavetracker configuration, see
# https://github.com/weygoldt/wavetracker/blob/master/docs/configuration.md
#
# IMPORTANT: set harmonic_groups.min_freq / max_freq to the frequency band of
# your fish, as narrow as you can (with some margin for drift and rises),
# e.g. 400-1200 Hz for Apteronotus leptorhynchus. The default 80-2400 Hz is a
# broad catch-all: slower, more false fish, and hum combs spaced wider than
# the lowest fish frequency are not removed.

"""


@dataclass
class Config:
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    clicks: ClickConfig = field(default_factory=ClickConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    harmonic_groups: HarmonicGroupsConfig = field(default_factory=HarmonicGroupsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    linking: LinkingConfig = field(default_factory=LinkingConfig)
    rises: RisesConfig = field(default_factory=RisesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    position_merging: PositionMergingConfig = field(
        default_factory=PositionMergingConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return YAML_HEADER + yaml.safe_dump(self.to_dict(), sort_keys=False)

    def default_frequency_range(self) -> bool:
        """True if the fish frequency range was left at the broad default."""
        d = HarmonicGroupsConfig()
        hc = self.harmonic_groups
        return hc.min_freq == d.min_freq and hc.max_freq == d.max_freq

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
