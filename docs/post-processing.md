# Post-processing

Commands that refine tracking results. All of them work on a results
directory (see [output](configuration.md#output)).

## Cleanup

`cleanup -n N` keeps the N identities with the most detections. Leftover
identities of at least a minute are not discarded outright: each is assigned
to the kept fish whose baseline frequency it continues (no shared frames;
tolerance grows by 0.2 Hz per minute of gap), unless two fish fit about
equally well (`assign_leftovers` in the cleanup config). Parameters are read
from `cleanup_config.cfg` in the results directory or the packaged default.

Cleanup assumes a known number of fish with persistent frequencies (grid
recordings). For moving electrodes, where every fish appears in short passes,
use [`merge-by-position`](#merging-by-position-moving-electrodes) instead.

## Harmonics tracked as fish

With few required harmonics (`min_group_size: 2`) and many fish, higher
harmonics of a fish are often tracked as additional "fish" (23 % of the
fish-related detections in dense tube mixtures, see
[benchmarks](benchmarks.md#dense-mixtures-of-tube-recordings)). A harmonic follows
its fundamental exactly, f_h(t) = h f_1(t), including its fast modulations,
while two fish near an integer ratio share at most slow drifts (temperature
changes every fish by the same factor, Q10):

```bash
wavetracker harmonics DIR             # writes DIR/harmonics.csv
wavetracker harmonics DIR --remove    # also untracks them (ident_v = NaN)
```

For every pair of overlapping identities (>= `--min-overlap` 10 s) within
3 Hz of an integer ratio it computes the offset from the exact ratio and the
correlation of the fast frequency modulations: each trace minus its running
median over `--timescale` seconds. A pair is a harmonic if it is within
0.25 Hz of the exact ratio and the correlation is >= 0.5. Amplitude
co-modulation (`--min-amp-corr`) and electrode-pattern similarity
(`--min-pattern`, grids) can be added as alternative evidence; with moving
electrodes neighbouring fish share amplitude changes, so they are off by
default.

The timescale matters: below ~10 s the modulations of resting fish are
mostly estimation noise; above ~100 s slow drifts shared by all fish make
different fish correlate. On [dense tube mixtures](benchmarks.md#dense-mixtures-of-tube-recordings) (103 harmonic pairs,
1645 pairs of different fish) the share of harmonic pairs found at 5 %
false positives was 0.71 (5 s), 0.84 (10 s), **0.90 (30 s)**, 0.88 (60 s),
0.86 (120 s) and 0.83 (600 s), the threshold needed rising from 0.23 (5 s)
to 0.89 (600 s). 30 s is the default. On the Iriri recording true harmonics
correlate with >= 0.94 at 30 s, while independent fish within 0.15 Hz of
3:1 (e.g. 307.4 / 921.5 Hz) are at -0.11 (30 s) but rise to 0.3 at
120-600 s.

Results on three [tube mixtures](benchmarks.md#dense-mixtures-of-tube-recordings) (`min_group_size: 2`, 20 min, 16 fish each):
70 of 79 harmonic identities that overlap their fundamental are found (74
with amplitude, 78 with electrode pattern as additional evidence), no fish
identity is flagged. Removing them removes 75 % of the harmonic detections
and 2 of 293k fish detections. Not covered: octave errors, where the
fundamental is not detected and its harmonic group is reported instead (no
simultaneous identity to compare with; 16 % of the harmonic detections with
`min_group_size: 2`, 65 % with 3).

## Merging by position (moving electrodes)

With moving electrodes (e.g. a boat towing an electrode pair over resting
fish) each fish is in range only during passes, identities are track
segments, and the neighbouring fish is often only 3–4 Hz away, so frequency
alone cannot tell whether two segments separated by a gap are one fish.
Resting fish keep their position, which survives the gaps.
`merge-by-position` groups segments into fish by frequency *and* a source
position estimated from the amplitudes (`sign_v`) and relative signs
(`cplx_v`) of the detections and the electrode positions:

```bash
wavetracker merge-by-position results/take -e electrodes.npz -c cfg.yaml -j 16
```

```yaml
position_merging:
  reference: 2       # channel i = electrode i - electrode 2 (two tips, hull reference)
  water_depth: 1.2   # insulating bottom [m]; omit for a half-space
```

**Electrode positions** (`-e`) are on the recording's time base: an `.npz`
with `time` (T,) [s] and `positions` (T, E, 3) [m] (or `x`, `y`, `z`, each
(T, E)), or a `.csv` with columns `time, x0, y0, z0, x1, …`. `z` is relative
to the water surface (negative under water). NaN rows mark times without a
valid geometry; the survey consists of the frames with valid positions
within `t_start`–`t_end`. Channels (columns of `sign_v`) are electrode
pairs: `reference: k` (channel i = electrode i − k, skipping k),
`channel_pairs: [[0, 2], [1, 2]]`, or neither (each electrode against a
distant ground).

**Noise floor**: per channel and frequency, a low percentile
(`noise_quantile`, 10 %) of the power spectrum over survey frames, computed
from the recording in `wavetracker.json` (or `--recording`) with the
spectrogram's scaling, and cached as `noise_floor.npz`
(`--noise-floor` reuses it). Without a recording it is estimated from the
detection powers near the fish's frequency (less reliable).

**Method** (`wavetracker.position`):

1. Segments: identities split at gaps > `split_gap` (3 s); segments
   < `min_segment_duration` (3 s) or < `min_segment_detections` (15) are
   clutter.
2. Candidates: segments, strongest first, join the candidate with the
   closest median frequency within `freq_tolerance` (1 Hz) unless they
   overlap one of its segments by > `max_overlap` (1 s).
3. Fit (per candidate, in parallel): forward model
   (`efield.SourceModel`) = a horizontal line of 10 monopoles over 0.2 m
   (Chen et al. 2005; `model: dipole` for a point dipole) in water between
   an insulating surface and optional insulating bottom (method of images,
   `n_images`). Parameters x, y, depth, heading, log strength. Residuals:
   log amplitudes with the noise floor (`log sqrt(V² + n²) − log A`);
   relative sign of channel pairs above `strong_snr`; censored survey frames
   (no detection within `censor_freq_tolerance` of the fish) must not be
   predicted above the fish's lowest detected level; distance to the nearest
   electrode-pair midpoint at a detection ≤ `detection_range` (2.5 m); depth
   ≤ `max_depth`. Soft-L1 least squares with an analytic Jacobian; a
   multi-start grid (7 × 7 positions over ±1.5 m, 8 headings, 2 depths) on
   every 4th frame, the 6 best distinct basins refined on all data. A
   competing basin > 0.5 m away within Δcost 25 marks the fit ambiguous.
4. Position check: segments whose mean |log residual| (channels above the
   noise floor) exceeds max(`resid_abs`, `resid_rel` × median), or that lie
   beyond the detection range of the fitted fish, are split off, regrouped
   by frequency and fitted as new candidates.
5. Standard errors: delete-a-block jackknife over 8 blocks of the
   candidate's detection times.
6. Merge: candidates without temporal overlap whose segments agree in
   frequency within `merge_freq_tolerance` (1.5 Hz) and whose positions
   agree within `merge_se_factor` (2) combined standard errors are refitted
   jointly; the merge is kept if the joint fit passes the check of step 4.
   The standard error is at least `merge_min_se` (0.1 m), and for ambiguous
   fits at least the spread of the competing basins.

**Output** (in the results directory; `ident_v.npy` is not changed):

| file | content |
|---|---|
| `fish_v.npy` | fish of each detection (NaN: clutter, outside the survey, untracked) |
| `fish.csv` | per fish: `freq`, `x`, `y`, `depth`, `heading` [deg], `log_strength`, jackknife `se_major`/`se_minor`/`se_angle`/`se_x`/`se_y`, `ambiguous`, `alt_x`/`alt_y`/`alt_dcost`, `basin_radius`, `r_fit`, `cost`, `n_det`, `n_cens`, `n_segments`, `at_bound`, `at_depth_cap`, `t_first`/`t_last`, `segments` |
| `fish_segments.csv` | per segment: identity, time span, frequency, clutter flag, fish, residual and distance under its fish |
| `position_merging.json` | counts, timings and the configuration used |

**Assumptions and limits**:

* Fish are stationary during the survey (resting). Moving fish are split
  into several "fish" or poorly fitted.
* Line-of-monopoles and point-dipole models are valid beyond about one body
  length; closer passes are fitted with systematic errors. The model's
  heading has a near-180° ambiguity (the field of a reversed fish is almost
  the same), which shifts the position by a few cm.
* Surface and bottom are taken as flat insulating planes in homogeneous
  water; rocks, banks and conductivity gradients are not modelled.
* Depth is weakly constrained (trades off against strength and horizontal
  distance) and often ends at `max_depth`; positions of fish seen in a
  single weak pass are often ambiguous (mirror positions across the path).
* Frequency must stay within `merge_freq_tolerance` between segments of one
  fish; rises are handled by the tracker's stitching, not here.
* Electrode positions must be accurate to well below the fish–electrode
  distance; timing errors between video and audio shift fish along the path.

**Validation** (`tests/test_position_merging.py`, synthetic boat survey
from `wavetracker.synthetic.boat_survey`/`simulate_survey`: 10 × 6 m, lanes
1 m apart, 6 fish; one identity joins two fish 3 Hz apart and 6 m apart,
two fish are 0.5 Hz apart, one drifts 2.5 Hz over the survey; 20 clutter
tracks): every fish ends up in exactly one group, no two fish are merged,
the 3-Hz pair is separated, clutter stays unassigned; position errors 1–10
mm, ≈ 6 cm when the heading-reversed solution wins; depth within 0.05 m.
Runtime: ~15 s with a reduced start grid on 4 processes, ~45 s with the
default grid (~20 s per candidate of ~300 detections on one core).

On the Iriri recording (Site A, 6 min survey, 2 channels, 107 600 survey
detections; 30 processes, 12.5 min): 2614 segments (1585 clutter, 79 % of
the survey detections kept), 361 frequency candidates, 140 segments split
off, 102 merges → 352 groups with 85 000 detections. Most groups are small:
118 have < 50 detections (92 % ambiguous), 85 have ≥ 300 (12–26 %
ambiguous); `r_fit` is about 0.5, as for the catalogue-based fits of the
habitat-mapping project (other detections, same model). Of 59 of its fish
matched by frequency, 14 agree within 0.15 m, the median distance is
0.85 m. Treat small and ambiguous groups as unresolved fragments rather
than fish.
