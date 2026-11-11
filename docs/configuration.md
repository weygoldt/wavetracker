# Configuration and output

All parameters live in one YAML file; `wavetracker config cfg.yaml` writes the
defaults, pass your edited copy with `-c cfg.yaml`. Unknown keys are an error.
The most relevant parameters:

| parameter | default | meaning |
|---|---|---|
| `spectrogram.freq_resolution` | 0.61 | frequency resolution [Hz] at any sampling rate (1.64 s FFT window; nfft 32768 at 20 kHz, 78732 at 48 kHz) |
| `spectrogram.nfft` | `null` | FFT window in samples; overrides `freq_resolution` (the results' `wavetracker.json` holds the nfft used) |
| `spectrogram.overlap_frac` | 0.9 | window overlap (0.16 s frame step) |
| `spectrogram.exclude_channels` | `[]` | broken/noisy electrodes to ignore (`run -x 8`) |
| `clicks.enabled` | true | remove static and crackle from the traces (`run --no-clicks`) |
| `clicks.threshold` | 12 | a click must reach this many standard deviations of the channel's high band |
| `interference.enabled` | true | remove interference combs (`run --no-interference`) |
| `interference.level_history_blocks` | 5 | gate-level memory; 2 follows changing hum faster, costs fish detections near teeth |
| `harmonic_groups.min_freq` / `max_freq` | 80 / 2400 | fundamental frequency range [Hz]; set it for your species (see [Usage](../README.md#usage)) |
| `harmonic_groups.low_thresh_factor` / `high_thresh_factor` | 6 / 10 | peak thresholds in units of the noise std |
| `harmonic_groups.threshold_window` | 10 | the noise setting the thresholds is the median over this many seconds around a frame; `null` estimates once from the first block, as in the original |
| `harmonic_groups.threshold_band` | 100 | thresholds also follow the noise over frequency, in bands of this width [Hz]; `null`: one value per frame |
| `harmonic_groups.min_group_size` | 3 | harmonics 1..n that must all be present |
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
| `tracking.min_identity_salience` | none | after linking, drop identities whose mean salience (0–1) is below this |
| `stitching.enabled` | true | rule-based joining of fragments across rises, dropouts and double detections; only used with `linking.enabled: false` |
| `stitching.max_dropout` | 900 | longest gap bridged when nothing else is at that frequency [s] |
| `linking.enabled` | true | join fragments with the learned linker instead of the stitching (see [Learned linking](algorithms.md#learned-linking)) |
| `linking.threshold` | 0.9 | minimum link probability; higher: purer identities, more fragments (0.5: IDF1 +0.02, purity -0.3 %) |
| `linking.max_gap` / `max_df` | 180 / 5 | longest gap [s] and largest frequency mismatch [Hz] of a candidate link |
| `linking.parallel_df` | 3 | identities running at the same time this far apart [Hz] are two fish |
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
| `cplx_v.npy` | complex spectrum at the fundamental on each electrode (`abs(cplx_v)**2 == sign_v`); the phase difference gives the relative sign of electrodes: `np.angle(cplx_v[:, 1] * np.conj(cplx_v[:, 0]))` |
| `times.npy` | time of each frame, relative to the recording start [s] |
| `sparse_spectra.npy`, `sparse_freq.npy`, `sparse_time.npy` | overview spectrogram (freq × time, power) |
| `fine_spec.npy`, `fine_freqs.npy`, `fine_times.npy` | optional full spectrogram (time × freq, `np.load(..., mmap_mode="r")`) |
| `wavetracker.json` | input, config, median thresholds, noise of each block (`noise_std_blocks`), fraction of samples repaired (`click_fraction`, `click_fraction_blocks`), timings, version |
