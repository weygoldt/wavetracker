# Configuration and output

All parameters live in one YAML file; `wavetracker config cfg.yaml` writes the
defaults, pass your edited copy with `-c cfg.yaml`. Unknown keys are an error.

## Two pipelines

**The default is the learned pipeline**: a neural network finds the
fundamentals ([Learned detector](algorithms.md#learned-detector)), a second
one links detections into tracks ([Learned
association](algorithms.md#learned-association)), the fragment linker and the
rise joiner join the tracks of a fish, identities without confident
detections are dropped, and `review.json` gives a calibrated confidence for
every identity and join with the places most likely wrong
([Review flags](review-flags.md)). All its models are shipped with
wavetracker (`wavetracker/models/`). On the dense benchmark (51 mixtures of
real two-fish recordings, held out from training) it reaches IDF1 0.78
against 0.64 for the classic pipeline, in lab and in field noise
([benchmarks](benchmarks.md#detection-threshold-05)).

**The classic pipeline** (rule-based harmonic groups and tracker, the
classic linker; the defaults before the learned pipeline) is selected with

```yaml
detector:
  method: classic
```

A config file only lists what differs: the keys it leaves out come from the
classic pipeline if `detector.method` is `classic`, otherwise from the
default pipeline. `wavetracker config --classic` writes the classic one in
full. In Python, `Config()` is the default pipeline, `Config.classic()` the
classic one. A model path may be a file name of a shipped model
(`assocnet.pt`, `linker-assocnet.npz`, `review-assocnet.npz`, `salience.pt`,
`linker.npz`).

The default pipeline needs a GPU for long recordings: on an RTX 4080 it
runs at 62-154x realtime on 11 electrodes at 20 kHz (classic 139-173x), on a
CPU at about realtime. Everything on PyTorch (spectrograms, the detector, the
association) is GPU first: `run --device auto` (the default) takes CUDA or
MPS if there is one, the association follows the detection's device
(`tracking.assoc_device: auto`, also for `wavetracker track`), and both fall
back to the CPU with a warning if the GPU runs out of memory. Models trained for other
recordings (another species, more electrodes) can be given with
`detector.model`, `tracking.assoc_model`, `linking.model` and
`review.model`; the review model is only calibrated for the pipeline it was
fitted on.

## Parameters

The most relevant parameters, with the default pipeline's value and, where
it differs, the classic one's:

| parameter | default (classic) | meaning |
|---|---|---|
| `detector.method` | `learned` (`classic`) | how fundamentals are found: the network, or spectral peaks grouped into harmonic series by rules (`harmonic_groups`) |
| `detector.threshold` | 0.5 | learned detector: minimum salience (0–1) of a detection |
| `detector.model` | none | learned detector: model file; none: the shipped `salience.pt` |
| `spectrogram.freq_resolution` | 0.61 | frequency resolution [Hz] at any sampling rate (1.64 s FFT window; nfft 32768 at 20 kHz, 78732 at 48 kHz) |
| `spectrogram.nfft` | `null` | FFT window in samples; overrides `freq_resolution` (the results' `wavetracker.json` holds the nfft used) |
| `spectrogram.overlap_frac` | 0.9 | window overlap (0.16 s frame step) |
| `spectrogram.exclude_channels` | `[]` | broken/noisy electrodes to ignore (`run -x 8`) |
| `clicks.enabled` | true | remove static and crackle from the traces (`run --no-clicks`) |
| `clicks.threshold` | 12 | a click must reach this many standard deviations of the channel's high band |
| `interference.enabled` | true | remove interference combs (`run --no-interference`) |
| `interference.level_history_blocks` | 5 | gate-level memory; 2 follows changing hum faster, costs fish detections near teeth |
| `harmonic_groups.min_freq` / `max_freq` | 80 / 2400 | fundamental frequency range [Hz]; set it for your species (see [Usage](../README.md#usage)) |
| `harmonic_groups.low_thresh_factor` / `high_thresh_factor` | 4 / 7 (6 / 10) | peak thresholds in units of the noise std |
| `harmonic_groups.threshold_window` | 10 | the noise setting the thresholds is the median over this many seconds around a frame; `null` estimates once from the first block, as in the original |
| `harmonic_groups.threshold_band` | 100 | thresholds also follow the noise over frequency, in bands of this width [Hz]; `null`: one value per frame |
| `harmonic_groups.min_group_size` | 2 (3) | harmonics 1..n that must all be present |
| `harmonic_groups.max_missing_harmonics` | 0 | gaps allowed: fundamental + `min_group_size` of the first `min_group_size + n` harmonics (odd-harmonic fish) |
| `harmonic_groups.min_good_peak_power` | -100 | absolute power limit for fundamentals [dB]; `null` for quiet audio recordings (a warning says when it binds) |
| `harmonic_groups.max_harmonics` | 10 | cap on harmonics per group (the original formula gives 89 for 80–2400 Hz) |
| `harmonic_groups.mains_freq` | 50 | mains harmonics are excluded; 0 disables (battery-powered recordings) |
| `harmonic_groups.exclusive_harmonics` | `core` | `all` reproduces the original grouping (see [Differences to the original implementation](algorithms.md#differences-to-the-original-implementation)) |
| `tracking.freq_tolerance` | 2.5 | max. frequency jump between linked detections [Hz] |
| `tracking.max_dt` | 10 | max. gap between linked detections [s] |
| `tracking.gap_tolerance` / `gap_tolerance_rate` | none / 0 | gap-dependent frequency tolerance `tol + rate * gap` (dense populations) |
| `tracking.amplitude_feature` | `minmax` | electrode pattern in the link error; `db` (level ratios) for few electrodes |
| `tracking.min_support` | 0 | only track detections with ≥ n neighbours within `support_window` s and `support_freq` Hz |
| `tracking.start_salience` / `start_evidence` | none / 0.5 | new identities need detections of at least this salience (0–1) in `start_evidence` s of frames; weaker ones only extend identities (see [Starting identities from confident detections](algorithms.md#starting-identities-from-confident-detections)) |
| `tracking.motion_window` / `motion_max_slope` | none / 2 | second pass that predicts links along each fish's frequency slope, measured over this many s before and after a detection, clipped to ± `motion_max_slope` Hz/s (see [Following the frequency slope](algorithms.md#following-the-frequency-slope-motion-model)) |
| `tracking.min_identity_salience` | 0.75 (none) | after linking, drop identities whose mean salience (0–1) is below this |
| `tracking.method` | `learned` (`classic`) | `learned`: a trained network scores the links between detections of the next few seconds (see [Learned association](algorithms.md#learned-association)); `classic`: the rule-based tracker with the options above |
| `tracking.assoc_model` | none | weights of the learned association (`benchmarks/dense/assocnet.py`); none: the shipped `assocnet.pt` |
| `tracking.assoc_mode` | `link` | `link`: tracks from the learned links alone; `cost`: the rule-based tracker with 1 − the learned probability as the error of short links |
| `tracking.assoc_threshold` / `assoc_min_detections` | 0.7 / 10 (0.5 / 1) | minimum link probability; shorter tracks stay unassigned (`link` mode) |
| `tracking.assoc_device` | `auto` | torch device of the network; `auto`: the device the detections were computed on if available, else GPU first; CPU if the GPU runs out of memory |
| `stitching.enabled` | true | rule-based joining of fragments across rises, dropouts and double detections; only used with `linking.enabled: false` |
| `stitching.max_dropout` | 900 | longest gap bridged when nothing else is at that frequency [s] |
| `linking.enabled` | true | join fragments with the learned linker instead of the stitching (see [Learned linking](algorithms.md#learned-linking)) |
| `linking.threshold` | 0.6 (0.9) | minimum link probability; higher: purer identities, more fragments |
| `linking.model` | `linker-assocnet.npz` (none) | linker weights; none: the shipped classic linker `linker.npz` |
| `linking.max_shared` / `max_parallel` | 0.3 / 0.6 (0.05 / 0.3) | two identities sharing more of their frames, or running in parallel for more of their simultaneous detections, are not joined |
| `linking.max_gap` / `max_df` | 180 / 5 | longest gap [s] and largest frequency mismatch [Hz] of a candidate link |
| `linking.parallel_df` | 3 | identities running at the same time this far apart [Hz] are two fish |
| `rises.enabled` | true (false) | join identities across rise onsets by their electrode patterns (see [Joining across rises](algorithms.md#joining-across-rises)) |
| `pairs.enabled` | false | handle the second detections a learned detector's pair head adds to peaks that hide two fish (see [Two fish in one peak](algorithms.md#two-fish-in-one-peak-the-pair-heads-detections)); no effect without a pair head |
| `pairs.partners` / `fill_gap` / `fill_df` | `fill` / 2 / 0.4 | `fill`: those detections are tracked last and only extend identities through their gaps (steps up to `fill_gap` s and `fill_df` Hz); `track`: like any detection |
| `pairs.partner_tracks` | none | tracks of the left-over second detections at least this long [s] become identities (fish hidden in a neighbour's peak for minutes) |
| `pairs.resolve` | false | decide every episode of two identities closer than `pairs.max_df` by their electrode patterns before and after it |
| `review.enabled` | true (false) | confidences of every identity, join and rejected join, and the places most likely wrong with a suggested fix, in `review.json`; the identities do not change (see [Review flags](review-flags.md)) |
| `review.model` | `review-assocnet.npz` (none) | calibrated confidence models for the pipeline (`benchmarks/dense/review.py fit`); none: the stages' own scores, uncalibrated |
| `review.missed_joins` / `min_missed_prob` | true / 0.05 | also flag rejected candidate joins with at least this linker probability |
| `review.max_confidence` | none | write only items below this confidence |
| `output.save_fine_spec` | false | also store the full-resolution spectrogram |
| `output.fine_spec_max_freq` / `sparse_spec_max_freq` | none | upper limit of the stored spectrograms; none = 1.25 × `harmonic_groups.max_freq` |
| `interference.max_line_freq` | none | upper limit of the hum-line search; none = Nyquist |

## Output

Each results directory holds plain NumPy files, compatible with the
post-processing tools and [audian](../README.md#correcting-tracks-interactively):

| file | content |
|---|---|
| `fund_v.npy` | fundamental frequency of every detection [Hz] |
| `idx_v.npy` | frame index of every detection (into `times`) |
| `ident_v.npy` | identity of every detection (NaN = unassigned) |
| `sign_v.npy` | power at the fundamental on each electrode, `(n, channels)` |
| `sal_v.npy` | salience of every detection (0–1; learned detector: the network's, classic: from the level over the noise floor) |
| `split_v.npy` | only from a model with a pair head: the detection is the second fish of a split peak |
| `cplx_v.npy` | complex spectrum at the fundamental on each electrode (`abs(cplx_v)**2 == sign_v`); the phase difference gives the relative sign of electrodes: `np.angle(cplx_v[:, 1] * np.conj(cplx_v[:, 0]))` |
| `times.npy` | time of each frame, relative to the recording start [s] |
| `review.json`, `frag_v.npy` | only with `review.enabled`: confidences and review flags, and the tracker fragment of every detection that their splits name (see [Review flags](review-flags.md)) |
| `sparse_spectra.npy`, `sparse_freq.npy`, `sparse_time.npy` | overview spectrogram (freq × time, power) |
| `fine_spec.npy`, `fine_freqs.npy`, `fine_times.npy` | optional full spectrogram (time × freq, `np.load(..., mmap_mode="r")`) |
| `wavetracker.json` | input, config, median thresholds, noise of each block (`noise_std_blocks`), fraction of samples repaired (`click_fraction`, `click_fraction_blocks`), timings, version |
