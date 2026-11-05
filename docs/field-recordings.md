# Field recordings and dense populations

The defaults are tuned for long grid recordings with few fish. For short
field recordings with many fish (e.g. a moving electrode pair, ~40 fish in a
chorus, battery powered) a reasonable starting point is:

```yaml
spectrogram:
  nfft: 65536            # 0.73 Hz at 48 kHz: resolves fish a few Hz apart
  overlap_frac: 0.9
interference:
  enabled: false         # no mains/hum on battery; avoids any risk to resting low fish
harmonic_groups:
  min_freq: 20.0
  max_freq: 2000.0
  mains_freq: 0.0        # no mains in battery-powered recordings
  low_thresh_factor: 3.0 # the chorus fills the troughs between fish peaks;
  high_thresh_factor: 5.0  # the defaults (6/10) miss most fish
  min_group_size: 2      # fish often show only two harmonics
  max_harmonics: 10      # (the default) otherwise ~200 harmonics per candidate
  min_good_peak_power: null  # absolute -100 dB limit removes weak (low) fish
  max_missing_harmonics: 0   # 1: "2 of the first 3" for fish without a 2nd
                             # harmonic (see below; neutral on the Iriri data)
tracking:
  min_support: 3         # isolated noise detections bridge neighbouring fish
  amplitude_feature: db  # min-max patterns are meaningless with 2 electrodes
  gap_tolerance: 0.5     # allowed frequency jump grows with the gap:
  gap_tolerance_rate: 0.1  # 0.5 Hz + 0.1 Hz/s (capped at freq_tolerance)
stitching:
  max_dropout: 60.0
  baseline_tolerance: 1.0  # neighbouring fish can be 3 Hz apart
  rise_baseline_tolerance: 2.0
  dropout_tolerance: 0.5
  overlap_tolerance: 1.0
output:
  save_fine_spec: true
  fine_spec_max_freq: 2050.0
```

The threshold factors are in units of the noise, and the noise depends on the number
of electrodes: about 3.3 dB on 2 channels, but 1 to 2 dB on 16. So 3 and 5 are 10 and
16 dB here, but far less on a 16-channel grid. There the factors decide between missed
fish and false ones. On a real grid with `min_group_size: 2`, for the fish of a night,
and for the 8 h of empty day and 10 min of static before it:

| factors | fish found | false detections per frame, day / static |
|---|---|---|
| 6 and 10 (default) | 35 % | 0.02 / 0.02 |
| 4 and 7 | 76 % | 0.17 / 0.27 |
| 3 and 5 | 92 % | 1.4 / 5.5 |

4 and 7 are a good start for grids. False detections do not last: at these factors
all tracks of 30 s or more on that grid were fish (see the figure in
[Benchmarks](benchmarks.md#recordings-that-start-without-fish)).

The tracking and stitching settings matter when neighbouring fish are only
a few Hz apart. On the Iriri recording (identities ≥ 30 s; shift of the
10th-percentile frequency in 20 s windows) they reduced identities that
drift by > 2 Hz, i.e. that continue on a different fish, from 55 % to 19 %
(median shift 2.4 → 0.9 Hz), while 89 % of the detections are still tracked.
Identities can still jump between fish 3–4 Hz apart after gaps; treat
identities as track segments and merge them into fish with additional
information, e.g. position
([`merge-by-position`](post-processing.md#merging-by-position-moving-electrodes)).

On the [dense-mixture benchmark](benchmarks.md#learned-fragment-linking),
the learned linker (the default; it replaces the `stitching`
settings above, which apply with `linking: {enabled: false}`) joins
fragments far better than those settings: IDF1 0.56 → 0.64
(lab noise) and 0.52 → 0.61 (field noise) with 4-30 fish, at the same purity;
a fish needs 1.4 identities instead of 2.4. With two electrodes its
identities are slightly less pure (0.3-0.8 %), and it has not been validated
on fish that move across the electrodes; check its output before relying on
it there.

Observations on a 16 min, 2-channel, ~40-fish recording (Iriri 2026):

* Detections follow the visible fish lines from 350 Hz to 2 kHz; harmonics
  of lower fish rarely appear as extra fish (≤ 4 % of detections above
  1 kHz in excess of chance).
* With `min_good_peak_power: -100` (the default) most fish below ~300 Hz
  were missed: the recording's noise floor is at about -122 dB and their
  fundamentals at -95 to -121 dB, so they never passed the absolute limit,
  although their 2nd harmonics were mostly present. Without the limit 17
  instead of 3 of 29 visible lines between 110 and 300 Hz are detected in
  more than 20 % of their frames (Site A), detections below 350 Hz rise
  from 2.4 % to 9.2 %, and all fish of an independent co-modulation
  catalogue (125–166 Hz) are found. Low detections near f/2 or f/3 of a
  stronger chorus detection occur at chance level (2f) or come from a few
  persistent near-3:1 pairs that are not co-modulated, i.e. independent
  fish (3f).
* `max_missing_harmonics: 1` adds little on this recording once the
  absolute limit is off (13.6k instead of 15.8k detections below 350 Hz in
  Site A); it helps where the 2nd harmonic is really absent (synthetic
  [benchmark](benchmarks.md#low-frequency-fish-without-a-2nd-harmonic)).
* Broadband noise (boat motor, contact) produces short clutter tracks.
* The relative sign of the electrodes (from `cplx_v`) is bimodal as
  expected for a dipole field: 86 % of strong detections at 150–180°,
  8 % at 0–30°.
* With moving electrodes each fish is in range for seconds to minutes, so
  many short tracks are expected; identity counts are not fish counts.
* Recorders that split a take into several files may write the take's start
  time into every file; thunderlab then refuses to read them as one
  recording. Concatenate them first (e.g. with `audioio`).
