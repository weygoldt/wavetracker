# wavetracker

[![Frontiers in Integrative Neuroscience](https://img.shields.io/badge/Published%20in-Frontiers%20in%20Integrative%20Neuroscience-blue)](https://doi.org/10.3389/fnint.2022.965211)

Detect and track the EOD frequencies of individual **wave-type electric fish**
in long, multi-electrode recordings.

The pipeline:

1. **Spectrogram** – one-sided power spectral densities of every electrode,
   computed block-wise on the GPU with PyTorch (CPU works too).
2. **Harmonic groups** – peaks of the electrode-summed spectrum are grouped
   into harmonic series; each group is one fish (parallel numba code).
3. **Tracking** – detections are linked into identities using their
   frequency and their amplitude pattern across electrodes
   ([Raab et al. 2022](https://doi.org/10.3389/fnint.2022.965211)).

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
| `harmonic_groups.min_freq` / `max_freq` | 400 / 1200 | fundamental frequency range [Hz] |
| `harmonic_groups.low_thresh_factor` / `high_thresh_factor` | 6 / 10 | peak thresholds in units of the noise std |
| `harmonic_groups.exclusive_harmonics` | `core` | `all` reproduces the original grouping (see below) |
| `tracking.freq_tolerance` | 2.5 | max. frequency jump between linked detections [Hz] |
| `tracking.max_dt` | 10 | max. gap between linked detections [s] |
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
* **Sub-bin frequencies** – fundamentals are refined by parabolic
  interpolation (`refine_frequency`), giving ~0.01 Hz precision instead of the
  0.61 Hz bin spacing.
* **Seamless blocks** – blocks are aligned to the STFT frame grid, so there
  are no edge artifacts or duplicate frames between blocks.
* Spectra are true PSDs (mlab `scale_by_freq` convention). They are 3 dB
  above the old hand-calibrated scale, which does not affect the
  noise-relative thresholds.

## Known limitations / next steps

* Tracking fragments a fish during fast frequency excursions (rises) whose
  jump exceeds `freq_tolerance`; `cleanup` is currently needed to rejoin them.
* The amplitude-error distribution used for tracking is estimated once from
  the first `3 * max_dt` seconds only.
* Detection thresholds are estimated once from the first block and frozen
  (as in the original), so slow changes in noise over days are not followed.
* Fish within ±1 Hz of a mains harmonic (multiples of 50 Hz) are not detected.
* Stationary interference combs look exactly like fish to the harmonic-group
  detector. In the 2022 tube-competition recordings electrode 8 picks up a
  95.4 Hz comb (lines at 477, 573, 669, 764, 859 Hz); exclude it with `-x 8`.
  An automatic filter (zero frequency variance, single-electrode spatial
  pattern) would be a good addition.
* Tracking runs on the CPU; for weeks of data with many fish it should be
  chunked/parallelized.

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
