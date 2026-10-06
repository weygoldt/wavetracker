# wavetracker

[![Frontiers in Integrative Neuroscience](https://img.shields.io/badge/Published%20in-Frontiers%20in%20Integrative%20Neuroscience-blue)](https://doi.org/10.3389/fnint.2022.965211)

Detect and track the EOD frequencies of individual **wave-type electric fish**
in long, multi-electrode recordings.

The pipeline:

1. **Spectrogram** – one-sided power spectral densities of every electrode,
   computed block-wise on the GPU with PyTorch (CPU works too).
   Stationary interference combs are detected and removed per electrode.
2. **Harmonic groups** – peaks of the electrode-summed spectrum are grouped
   into harmonic series; each group is one fish (parallel numba code).
3. **Tracking** – detections are linked into identities using their
   frequency and their amplitude pattern across electrodes
   ([Raab et al. 2022](https://doi.org/10.3389/fnint.2022.965211)).
4. **Stitching** – track fragments are joined across rises, dropouts and
   short double detections.

It runs at several hundred times realtime on an 11-channel, 20 kHz recording
(a 4 h recording takes about a minute on an RTX 4080, mostly disk IO).

## Installation

Requires Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/weygoldt/wavetracker.git
cd wavetracker
uv sync                 # add --extra gui for the EOD sorter GUI
uv run wavetracker --help
```

or with pip into an existing environment: `pip install -e ".[gui]"`.

PyTorch from PyPI ships with CUDA support on Linux; the GPU is used
automatically when available (`--device cpu` forces the CPU).

## Usage

```bash
wavetracker info  /data/2022-05-10-10_00                 # channels, rate, duration
wavetracker run   /data/2022-05-10-10_00 -o results/0510 # detect + track
wavetracker run   /data/2022-05-10-10_00 -o test --start 3600 --duration 300
wavetracker summary results/0510                          # table of identities
wavetracker plot    results/0510 -o tracks.png            # tracks on spectrogram
wavetracker track   results/0510 -c my_config.yaml        # re-track only
```

`run` accepts a single file, a fishgrid recording directory
(`traces-grid*.raw` + `fishgrid.cfg`) or a directory with a sequence of audio
files (read as one continuous recording via
[thunderlab](https://github.com/bendalab/thunderlab)). Several inputs can be
given at once; each gets a subdirectory of `--output`. Existing results are
skipped unless `-f/--overwrite` is given.

Post-processing and tools:

| command | purpose |
|---|---|
| `wavetracker cleanup DIR -n 2` | join/filter tracks, keep the N most prominent fish |
| `wavetracker concat DAY1 DAY2 … -o OUT` | concatenate consecutive recordings |
| `wavetracker freq-analysis DIRS…` | top-N frequencies at fixed times of day |
| `wavetracker sorter DIR` | GUI for manual track correction (`--extra gui`) |
| `wavetracker synth out.wav -n 4` | synthetic recording with ground truth |
| `wavetracker config [FILE]` | print or write the default configuration |

### Configuration

All parameters live in one YAML file; `wavetracker config cfg.yaml` writes the
defaults, pass your edited copy with `-c cfg.yaml`. Unknown keys are an error.
The most relevant parameters:

| parameter | default | meaning |
|---|---|---|
| `spectrogram.nfft` | 32768 | FFT window (0.61 Hz resolution at 20 kHz) |
| `spectrogram.overlap_frac` | 0.9 | window overlap (0.16 s frame step) |
| `spectrogram.exclude_channels` | `[]` | broken/noisy electrodes to ignore (`run -x 8`) |
| `interference.enabled` | true | remove interference combs (`run --no-interference`) |
| `interference.level_history_blocks` | 5 | gate-level memory; 2 follows changing hum faster, costs fish detections near teeth |
| `harmonic_groups.min_freq` / `max_freq` | 400 / 1200 | fundamental frequency range [Hz] |
| `harmonic_groups.low_thresh_factor` / `high_thresh_factor` | 6 / 10 | peak thresholds in units of the noise std |
| `harmonic_groups.min_group_size` | 3 | harmonics 1..n that must all be present |
| `harmonic_groups.max_harmonics` | none | cap on harmonics per group (needed for wide frequency ranges) |
| `harmonic_groups.mains_freq` | 50 | mains harmonics are excluded; 0 disables (battery-powered recordings) |
| `harmonic_groups.exclusive_harmonics` | `core` | `all` reproduces the original grouping (see below) |
| `tracking.freq_tolerance` | 2.5 | max. frequency jump between linked detections [Hz] |
| `tracking.max_dt` | 10 | max. gap between linked detections [s] |
| `stitching.enabled` | true | join fragments across rises, dropouts and double detections |
| `stitching.max_dropout` | 900 | longest gap bridged when nothing else is at that frequency [s] |
| `output.save_fine_spec` | false | also store the full-resolution spectrogram |

### Output

Each results directory holds plain NumPy files, compatible with the
post-processing tools and the sorter GUI:

| file | content |
|---|---|
| `fund_v.npy` | fundamental frequency of every detection [Hz] |
| `idx_v.npy` | frame index of every detection (into `times`) |
| `ident_v.npy` | identity of every detection (NaN = unassigned) |
| `sign_v.npy` | power at the fundamental on each electrode, `(n, channels)` |
| `times.npy` | time of each frame, relative to the recording start [s] |
| `sparse_spectra.npy`, `sparse_freq.npy`, `sparse_time.npy` | overview spectrogram (freq × time, power) |
| `fine_spec.npy`, `fine_freqs.npy`, `fine_times.npy` | optional full spectrogram (time × freq, `np.load(..., mmap_mode="r")`) |
| `wavetracker.json` | input, config, estimated thresholds, timings, version |

```python
from wavetracker.results import Results

r = Results.load("results/0510")
for fish in r.ids():
    m = r.ident_v == fish
    t, f = r.times[r.idx_v[m]], r.fund_v[m]
```

### Python API

```python
from wavetracker.config import Config
from wavetracker.pipeline import detect, track_results

cfg = Config.load("cfg.yaml")            # or Config()
out = detect("/data/rec", "results/rec", cfg, start=0, duration=600)
track_results(out.results, cfg)
out.results.save("results/rec")
```

`wavetracker.synthetic` generates recordings of fish with known frequency
traces and `wavetracker.evaluation.evaluate` scores results against them
(precision, recall, identity coverage and purity).

## Differences to the original implementation

The detection and tracking algorithms follow Raab et al. (2022). The tracker
is a numba port that reproduces the original `freq_tracking_v6` assignments
exactly in compatibility mode (verified in the test-suite), while being roughly 25× faster and
linear in recording length. Deliberate changes:

* **Harmonic exclusivity** – the original rejects a fish if *any* of its up to
  8 harmonics coincides with a peak of an already accepted fish. Because the
  harmonic tolerance grows with the harmonic number, high harmonics of one
  fish regularly capture a harmonic of another, which makes that fish
  disappear. By default only the `min_group_size` lowest harmonics are now
  exclusive (this still rejects in-range harmonics of other fish as ghosts).
  `exclusive_harmonics: all` restores the original behaviour.
* **Window-check bug** – when attaching a window's links to established
  identities, the original compared a detection's *position in the window*
  with the window's *frame count*, so the skipped span depended on the
  number of fish. It now checks the target's frame. In practice this rarely
  matters (on a 4 h recording it assigns ~100 more detections and changes no
  existing assignment); `track(..., v6_compat=True)` reproduces the original.
* **Interference removal** – new, see below.
* **Sub-bin frequencies** – fundamentals are refined by parabolic
  interpolation (`refine_frequency`), giving ~0.01 Hz precision instead of the
  0.61 Hz bin spacing.
* **Seamless blocks** – blocks are aligned to the STFT frame grid, so there
  are no edge artifacts or duplicate frames between blocks.
* Spectra are true PSDs (mlab `scale_by_freq` convention). They are 3 dB
  above the old hand-calibrated scale, which does not affect the
  noise-relative thresholds.

## Interference removal

Electrical interference often forms a *comb* of persistent lines at
consecutive multiples of a low fundamental (in the 2022 tube recordings:
95.4 Hz on all electrodes, plus 25, 55.5, 100 and 125 Hz combs). Each tooth's
harmonics are other teeth, so the harmonic-group detector cannot tell them
from fish. `wavetracker.interference` removes them before the channels are
summed. For every electrode and 60 s block it:

1. estimates the persistent spectrum: the 20th percentile over time, then
   the minimum over the last 5 blocks. A line must sit on the same bin for
   5 minutes.
2. finds lines ≥10 dB above a running-median noise baseline;
3. searches them for combs: spacing 20–300 Hz, at least 4 *consecutive* teeth
   within 0.3 Hz;
4. remembers every tooth found on an electrode for 30 minutes (single teeth
   intermittently fail the line test, which made the hum leak back), and
5. gates the remembered teeth: in each frame a tooth bin at or below the
   tooth's level (90th percentile + 3 dB, minimum over the history) is set to
   the noise floor; a bin above it contains a fish and is left untouched.

Where a fish overlaps a tooth (within the ~2 Hz main lobe), the gating can
split the fish's peak in two; two detections in one frame within 4 Hz with a
tooth between them are therefore merged into the stronger one. A strong fish
up to ~8 Hz away can also lift a tooth above the gate through its spectral
leakage, and the tracker then hands the fish's identity to the constant
tooth; a detection on a tooth with a stronger non-tooth detection within
8 Hz is therefore dropped.

The gate level is the minimum over the last 5 blocks. That protects fish
resting near a tooth but lags when the interference gets stronger, so some
hum still leaks as separate constant-frequency tracks (e.g. in the first hour
of 2022-06-02); `cleanup` discards them. `level_history_blocks: 2` follows the
interference faster at the cost of fish detections near teeth.

A fish is never a comb: its harmonics are spaced by its own fundamental
(≥ `min_freq`), and sub-multiples of it match only every 2nd/3rd tooth. So a
resting fish that is stable for hours and seen on a single electrode is kept
(tested with synthetic data). Gating rather than notching keeps a fish
visible while it passes a tooth, as long as it is stronger than the tooth.
The remaining blind spot is a fish that is **weaker than a tooth and stays
within ~1 Hz of it for 5+ minutes** on the same electrodes. At 0.6 Hz
resolution the two cannot be separated.

On the 4 h recording 2022-06-14 (no channel excluded) it removes the
interference detections (4.1 → 2.0 detections per frame for two fish) without
losing fish detections. The detected combs are listed in `wavetracker.json`
(`interference_combs`). It costs ~40 ms per 60 s block.

Because a stationary fish's harmonic series is itself a comb with spacing
equal to its fundamental, combs are only searched below
`interference.max_spacing` (300 Hz). When tracking fish below ~300 Hz, switch
the filter off (`interference.enabled: false` or `run --no-interference`).

## Stitching

The tracker breaks a track at every rise: the onset jump exceeds
`freq_tolerance` and, by the time the frequency has decayed back, the gap
exceeds `max_dt`. `wavetracker.stitching` joins fragment A to a later
fragment B when their *baselines* (10th percentile of the frequency, which
ignores the upward rises) agree:

| situation | condition |
|---|---|
| gap ≤ 30 s | baselines within 3 Hz; a clear rise onset (gap ≤ 2 s, jump ≥ +3 Hz) may differ by 6 Hz |
| dropout ≤ 15 min | baselines within 1.5 Hz and no other track at that frequency during the gap |
| overlap ≤ 30 s (double detection) | same median frequency during the overlap (2.5 Hz) |

Joins are made greedily, best first; frames that end up with two detections
of one identity keep the one closer to the track. An electrode amplitude
pattern check is available (`max_pattern_distance`) but off by default: in
the tube recordings it did not separate the two fish. With many fish close in
frequency (field recordings) lower `max_dropout`, e.g. to 60 s.

## Post-processing: cleanup

`cleanup -n N` keeps the N identities with the most detections. Leftover
identities of at least a minute are not discarded outright: each is assigned
to the kept fish whose baseline frequency it continues (no shared frames;
tolerance grows by 0.2 Hz per minute of gap), unless two fish fit about
equally well (`assign_leftovers` in the cleanup config). Parameters are read
from `cleanup_config.cfg` in the results directory or the packaged default.

## Field recordings and dense populations

The defaults are tuned for long grid recordings with few fish. For short
field recordings with many fish (e.g. a moving electrode pair, ~40 fish in a
chorus, battery powered) a reasonable starting point is:

```yaml
spectrogram:
  nfft: 65536            # 0.73 Hz at 48 kHz: resolves fish a few Hz apart
  overlap_frac: 0.9
interference:
  enabled: false         # required when tracking fish below ~300 Hz
harmonic_groups:
  min_freq: 20.0
  max_freq: 2000.0
  mains_freq: 0.0        # no mains in battery-powered recordings
  low_thresh_factor: 3.0 # the chorus fills the troughs between fish peaks;
  high_thresh_factor: 5.0  # the defaults (6/10) miss most fish
  min_group_size: 2      # fish often show only two harmonics
  max_harmonics: 10      # otherwise ~200 harmonics per candidate
stitching:
  max_dropout: 60.0
output:
  save_fine_spec: true
  fine_spec_max_freq: 2050.0
```

Observations on a 16 min, 2-channel, ~40-fish recording (Iriri 2026):

* Detections follow the visible fish lines from 350 Hz to 2 kHz; harmonics
  of lower fish rarely appear as extra fish (≤ 4 % of detections above
  1 kHz in excess of chance).
* Many fish with fundamentals below ~350 Hz are missed: their 2nd harmonic
  is often absent while the 3rd is present (odd-harmonic waveforms), and the
  detector requires harmonics 1..`min_group_size` to all be present.
* Broadband noise (boat motor, contact) produces short clutter tracks.
* With moving electrodes each fish is in range for seconds to minutes, so
  many short tracks are expected; identity counts are not fish counts.
* Recorders that split a take into several files may write the take's start
  time into every file; thunderlab then refuses to read them as one
  recording. Concatenate them first (e.g. with `audioio`).

## Known limitations / next steps

* The amplitude-error distribution used for tracking is estimated once from
  the first `3 * max_dt` seconds only.
* Detection thresholds are estimated once from the first block and frozen
  (as in the original), so slow changes in noise over days are not followed.
* Fish within ±1 Hz of a mains harmonic (multiples of 50 Hz) are not detected
  (set `mains_freq: 0` if there is no mains).
* A fish needs harmonics 1..`min_group_size`; fish with a missing 2nd
  harmonic (odd-harmonic waveforms, common below ~350 Hz in the field) are
  missed. Accepting "2 of the first 3 harmonics" would fix this.
* Detection thresholds are relative to the global noise floor; in a dense
  chorus the troughs between fish are far above it, so thresholds have to be
  lowered by hand (see field recordings).
* Interference that is a single stationary harmonic series with a
  fundamental in the fish range (not a comb) is indistinguishable from a
  resting fish and is not removed.
* Gated comb teeth are set to the persistent noise baseline, which is a few
  dB below the median noise; they show as faint white lines in spectrogram
  plots.
* While a fish crosses a comb tooth its frequency estimate can be off by up
  to ~1 Hz (the merged detection is the stronger of two split peaks).
* Tracking runs on the CPU; for weeks of data with many fish it should be
  chunked/parallelized.

## Benchmarks

`benchmarks/tube_competition.py` scores results on the 2022 tube-competition
recordings (two fish per trial, no ground truth traces). It builds a
pseudo ground truth from the detections, independent of tracking, and reports
purity, fragmentation and coverage for the raw tracks and for `cleanup -n 2`
(see the module docstring for usage).

Results on 10 full 4 h trials (all six pairings), means over trials:

| | first rewrite | current |
|---|---|---|
| identities with ≥ 300 detections (truth: 2) | 32.2 | 9.9 |
| identities covering 90 % of the loser | 13.0 | 3.0 |
| largest track, winner / loser [% of frames] | 50 / 34 | 64 / 57 |
| after `cleanup -n 2`, winner / loser [% of frames] | 91 / 84 | 95 / 92 |
| purity (detections assigned to the right fish) | 1.000 | 1.000 |

Detection itself covers 95 % (winner) and 92 % (loser) of the frames; the rest is
mostly rises, which leave the pseudo-ground-truth band. Remaining
fragmentation is concentrated in losers with many rises (pairing 4a: 79 %
after cleanup).

## Development

```bash
uv sync --all-extras
uv run pytest            # tests using recordings in /mnt/data2 are skipped if absent
uv run ruff check . && uv run ruff format .
```

`tests/legacy_tracking.py` holds the original tracker as a reference for the
equivalence tests. `wavetracker/gui` and `wavetracker/postprocessing` contain
older and contributed code that is wrapped by the CLI but not yet refactored.

## Citation

Raab T, Madhav MS, Jayakumar RP, Henninger J, Cowan NJ, Benda J (2022).
*Advances in non-invasive tracking of wave-type electric fish in natural and
laboratory settings.* Front. Integr. Neurosci. 16:965211.
[doi:10.3389/fnint.2022.965211](https://doi.org/10.3389/fnint.2022.965211)
