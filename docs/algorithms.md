# How it works

Where the pipeline departs from the published method, and how its newer
stages work. For the method itself see
[Raab et al. 2022](https://doi.org/10.3389/fnint.2022.965211).

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
* **Interference removal** – new, see [below](#interference-removal).
* **Sub-bin frequencies** – fundamentals are refined by parabolic
  interpolation (`refine_frequency`), giving ~0.01 Hz precision instead of the
  0.61 Hz bin spacing.
* **Seamless blocks** – blocks are aligned to the STFT frame grid, so there
  are no edge artifacts or duplicate frames between blocks.
* Spectra are true PSDs (mlab `scale_by_freq` convention). They are 3 dB
  above the old hand-calibrated scale, which does not affect the
  noise-relative thresholds.

## Learned detector

`detector.method: learned` finds fundamentals with a neural network instead
of spectral peaks and harmonic rules (`wavetracker/salience.py`). Each frame
of the electrode-summed log spectrogram is normalised by its median and
spread in 100 Hz bands, then read as a *harmonic stack*: on a linear axis
starting at 0 Hz, harmonic h of bin k is bin h*k, so channel h at candidate k
holds the feature at bin h*k (with a max over the bins an off-centre
harmonic can move to), for h = 1/3, 1/2, 1, ..., 5. A true fundamental is
bright from f upward and dark at f/2 and f/3; a harmonic of a lower fish has
its fundamental there. A fully convolutional network without frequency
pooling (time dilations give it seconds of context) outputs a salience per
bin and a sub-bin offset; peaks above `detector.threshold` become
detections, with their electrode powers read from the spectrogram as for
classic detections, and the salience is stored as `sal_v`. Blocks are
evaluated with their neighbours as context, so block boundaries leave no
trace. The model stores the spectrogram resolution and frame step it was
trained on; a different resolution is refused. Training and benchmark:
docs/benchmarks.md, "Learned detector (v1)".

**Two fish in one peak (optional; not in the packaged model).** Two
fundamentals closer than about 1.2 Hz make one peak in the 1.64 s window,
and peak picking gives one detection for both. A model can carry a *pair
head* (`SalienceSpec.pair_head`): a second output per bin, the probability
that two fish are there, with the offsets of both. A salience peak without
another peak within 2 bins whose pair probability reaches
`detector.pair_threshold` (default: `detector.threshold`) gets a second
detection at the pair offset farther from its own, with the pair probability
as its salience; with `tracking.start_salience` such detections can extend an
identity but not start one. A model can also read the electrodes
(`SalienceSpec.electrode_channels`): for every electrode, its share of the
summed power at the candidate bins and its phase relative to the strongest
electrode (one fish: fixed shares and phases near 0 or pi, as the field is
quasi-static; two fish in one bin: both move with their beat). One encoder
with shared weights reads every electrode; the mean and maximum over the
electrodes (any number, in any order) join the network through a gate that
starts at zero, either into the pair head's branch or into the trunk. Both
are exact block by block like the rest of the detector. Whether they help:
docs/benchmarks.md, "Two fish in one peak".

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
`interference.max_spacing`, by default `min(300, 0.9 × harmonic_groups.min_freq)`
(72 Hz for the default range, 300 Hz for 400–1200 Hz). A comb spaced wider
than the lowest fish frequency cannot be told apart from a resting fish and
is kept; restrict the fish range to remove it.

## Starting identities from confident detections

A detector that finds faint fish also finds more doubtful peaks. If the
tracker treats all detections alike, a high detection threshold misses faint
fish and a low one builds false identities out of clutter. With
`tracking.start_salience` (off by default) the tracker splits the two roles,
as ByteTrack does (Zhang et al., ECCV 2022): only confident detections may
start an identity, but any detection may extend one.

Every detection has a salience between 0 and 1 (`sal_v`). The learned
detector (`detector.method: learned`) gives its network's. The classic
detector gives a stand-in from the fundamental's level over the local noise
floor (the median of the frame's spectrum in the detection's threshold band,
or over the fish range without bands), in units of the high threshold:
`1 - 2^(-snr / high_threshold)`. It is 0.5 for a fundamental one high
threshold above the floor, the prominence a "good" peak needs, and 0.75 at
twice that.

The tracker forms temporary identities in windows of `3 * max_dt` and
attaches them to the identities it has. A temporary identity that attaches to
none starts a new identity only if it holds confident detections (salience at
least `start_salience`) in `start_evidence` seconds of frames (0.5 s: 3
detections at the default frame step). Weak detections still link into
temporary identities, so a faint fish is followed throughout once a few
confident detections have started it, from up to about `2 * max_dt` before
them. Clutter that is never confident makes no identities. The detections of a
temporary identity that may not start stay unassigned, even if the identity
starts in a later window. Links are scored as before, without the salience:
a weak detection that fits an identity better than a confident one takes the
link. With `start_salience: null` the tracker is the same as before, also in
compatibility mode.

A related filter acts at the end: with `tracking.min_identity_salience`,
identities whose mean detection salience is below it are dropped after
linking (or stitching). Lines of harmonics and clutter that no confident
detection supports go; weak detections inside a confident identity stay. On
the dense benchmark, identities of the learned detector with a mean salience
below 0.75 are 79 % harmonic or false detections. Dropping them before
linking would also drop weak fragments of real fish that the linker joins to
their identities.

## Following the frequency slope (motion model)

The tracker scores links between pairs of detections, not between tracks and
detections: the link error is 2/3 the rank of the electrode-pattern distance
among all pairs and 1/3 a sigmoid of the frequency difference. Where two
fish cross, their patterns can be alike and their frequencies are equal for a
while, and the tracker may swap them. Object trackers for look-alike objects
(DanceTrack, OC-SORT) found that motion is then the better cue.

With `tracking.motion_window` (off by default) the tracker runs twice. The
first pass tracks as before; every detection then gets the slope of its
identity over `motion_window` seconds *before* it and over as many *after*
it. In the second pass, a link from an earlier detection a to a later
detection b compares b with the frequency predicted from a along a's past
slope, and a with the frequency predicted back from b along b's future
slope; the mean of the two errors replaces the frequency difference in the
link error and in the tolerances. One-sided slopes keep the two fish of a
crossing apart even where the first pass swapped them: before the crossing
each detection's past belongs to its own fish, after it each detection's
future does (OC-SORT keeps the motion estimate from before an occlusion for
the same reason). Two guards keep noisy slopes from doing harm: each slope is
shrunk towards 0 by its own uncertainty, `s * s^2 / (s^2 + var(s))`, so the
scatter of a flat, faint track is not taken for a slope, and a prediction
reaches at most `motion_window` ahead. Slopes are clipped to
`motion_max_slope` (2 Hz/s); rise onsets are jumps, not slopes.

## Stitching

The rule-based stitching joins fragments when the learned linker is off
(`linking.enabled: false`, see [Learned linking](#learned-linking)).
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

## Learned linking

In dense populations the tracker's fragments are almost always pure (one
fish), but each fish is spread over many of them: it breaks at rises, where a
fish is not detected for longer than `max_dt`, and next to close neighbours.
On the [dense-mixture benchmark](benchmarks.md#learned-fragment-linking),
joining every fragment to its true fish would raise IDF1 from 0.56 (the
rule-based stitching above) to 0.69 (lab noise); the linker reaches 0.64 at
the same purity. By default (`linking.enabled: true`), `wavetracker.linking`
replaces the stitching:

1. **Fragments**: every tracker identity with at least 10 detections,
   summarised at its start and end: frequency, slope and scatter, baseline
   (10th percentile), median, the electrode power pattern (short and long
   window, with its own scatter from split halves), the relative signs between
   electrodes (`cplx_v`) and the level.
2. **Candidates**: pairs (A, B) with B starting at most `max_gap` (180 s)
   after A ends, or overlapping it, whose junction frequencies, baselines or
   medians agree within `max_df` (5 Hz).
3. **Score**: a small neural network (2 hidden layers, 64 and 32 units)
   gives the probability that A and B are one fish, from 40 features of the
   pair: gap, overlap and shared frames, frequency jump and extrapolation
   errors, baseline and median differences, pattern distances and relative
   sign agreement, level change, how crowded the frequency is between them,
   the nearest other detection at both ends, and how the pair ranks among the
   competing candidates of A and of B. The weights
   (`wavetracker/models/linker.npz`) were trained on the benchmark's training
   mixtures (`benchmarks/dense/linker.py`), whose truth says which fragments
   belong to one fish.
4. **Association**: pairs are accepted most probable first, down to
   `threshold` (0.9), into identities (union-find), unless the two identities
   would share more than 5 % of their frames, or run at the same time (within
   0.3 s) at frequencies more than 3 Hz apart for more than 30 % of those
   detections (a fish is never at two frequencies at once).

The threshold sets the balance: higher thresholds keep identities purer and
leave more fragments. The features are measured in seconds and Hz, not in
frames or detections, and the model was trained on 9 and 2 electrodes and on
0.61 and 1.22 Hz resolution, on fish that move (their electrode patterns
change by about 4.5 dB rms within a minute, see
[benchmarks](benchmarks.md#learned-fragment-linking)).

## Joining across rises

At a rise a fish's frequency jumps up by a few to about 60 Hz within about a
second and decays back over 10-60 s. The jump is far beyond the tracker's
tolerances and beyond the linker's candidates (`linking.max_df`, 5 Hz), so
the fish goes on as a new identity, or the linker leaves it on two
identities that take turns at its rises. With `rises.enabled` (off by
default), `wavetracker.rises` joins such identities after linking (or
stitching) and before `tracking.min_identity_salience`:

1. **Stops and starts**: an identity *stops* where it has no detection for
   `stop_gap` (2 s) after a detection, and *starts* where it has none for as
   long before one. A start must be followed by at least `min_duration`
   (1 s) of the identity.
2. **Candidates**: a stop of A and a start of B (another identity) that
   begins at most `max_gap` (5 s) after it, or up to `max_overlap` (4 s)
   before it, between `min_jump` and `max_jump` (1.5-35 Hz) higher
   (medians over `edge_window`, 1 s, at the two edges).
3. **One fish?** The electrode patterns (mean power per electrode in dB,
   relative to their mean) over `pattern_window` (5 s) before the stop and
   after the start agree: rms difference at most `max_pattern_rms` (7 dB)
   and correlation at least `min_pattern_corr` (0.5). A fish hardly moves
   during a rise, while another fish that starts there usually sits
   elsewhere on the grid. Near the noise a pattern is noise and two faint
   fish look alike, so both sides need a mean detection salience of
   `min_salience` (0.4).
4. **Joins**: best pattern first, unless the two identities would share
   more than `max_shared` (30 %) of the smaller one's frames.

Detectors with temporal context report the old frequency for a few seconds
after the onset (a ghost tail), and sometimes the new one a little before
it, hence the overlap: when such a pair is joined, A's detections after B's
start are dropped. The decay after the rise is not used: many rise pieces
end, or the next rise starts, before they decay, and requiring a falling
frequency lost more true joins than false ones. On the dense benchmark this
raises IDF1 by 0.02 with the learned detector; about 1 in 11 of its joins
puts two different fish together (see
[benchmarks](benchmarks.md#rises-step-b)).

## Two fish in one peak: the pair head's detections

A pair head (see [Learned detector](#learned-detector)) adds a second
detection to a peak that hides two fish. Tracked like any detection, the two
detections of one peak hurt: they carry the electrode spectra of the same
peak, so two thirds of the tracker's link error cannot tell them apart, and
below about 0.2 Hz the frequency term cannot either. Within a pair episode
the tracker hands the two fish back and forth, and the fish often leave the
episode on each other's identity, or on new ones, and stay there. On the
dense benchmark even a perfect pair head lowered IDF1, through the frames
*after* the episodes (see
[benchmarks](benchmarks.md#keeping-two-close-fish-apart-step-c)).

The detector flags its second detections (`Results.split_v`, saved as
`split_v.npy`). With `pairs.enabled` and `pairs.partners: fill`
(`wavetracker.pairs`), they come last, as in ByteTrack's second association:

1. Tracking, linking, the rise joiner and `tracking.min_identity_salience`
   run without them, so every other detection gets exactly the identity it
   would get from a detector without the pair head.
2. Identities then grow into their gaps: in one sweep forward in time and
   one backward, a second detection goes to an identity that has no
   detection in its frame and whose latest detection (or a second detection
   it already took) lies at most `fill_gap` (2 s) and `fill_df` (0.4 Hz)
   away, the closest first. A fish whose own detection is missing while it
   shares its neighbour's peak keeps its identity through those frames.
3. With `pairs.partner_tracks` (seconds), the second detections no identity
   took are tracked on their own; their tracks of at least that duration
   become identities (they pass `tracking.min_identity_salience` like any
   identity). These are fish hidden in another's peak for minutes, which
   the confident detections alone never start.

Results without flags (older runs, detectors without a pair head) are
tracked as before. Any two detections closer than 1.2 Hz are *not* taken
for a split: the learned detector also resolves neighbours that close, and
holding them back cost IDF1.

`pairs.resolve` (with `pairs.enabled`; off in the recommended configs) is a
second, pattern-based tool: for every episode of two identities closer than
`max_df` (2 Hz) in a frame, it compares their electrode patterns from
detections clear of other lines (`clear_df`) just before and just after it,
and if the fish fit better exchanged (by `margin`), the identities swap
everything after the closest approach; `order` also orders the detections
inside the episode by frequency. It helped a little when every split was
tracked, but too few episodes have clean patterns on both sides (fish that
weave among several neighbours have none), and on detections without a pair
head it decided no better than the tracker and linker.

## Recordings without fish

Field loggers start on the boat. For minutes the electrodes are out of the water,
then in empty water, and fish may come hours later, at dusk. Nothing estimated from
the start of a recording may be trusted, and noise must not make fish. Four things
went wrong here, each found on real deployments.

**Static is clicks.** Electrodes out of the water pick up clicks of a few
milliseconds, up to 40000 times the noise, about ten per second. One click in an FFT
window gives a smooth spectrum, so the noise estimate collapses (0.02 dB), and two
give an evenly spaced ripple, which is a harmonic series to the detector.
`wavetracker.clicks` removes them from the traces before the spectrogram:

1. Each channel is split by its spectrum over the block. Clicks are searched in the
   band above 20 % of the sampling rate, which fish do not reach, so that even a
   chirp right at an electrode does not look like one.
2. A click must reach 12 robust standard deviations of that band and lasts for as
   long as it stays above 8. Events longer than 0.5 s are real signals (a boat, rain).
3. Clicks are bridged by straight lines, but only in the part of the signal that does
   not stand out of the spectral floor. A hole in a loud fish is itself a click, of
   the fish's amplitude. In a hole, what stands out is scaled to what the samples
   around it hold, or a fish arriving later in the block would appear before it did.

Channels without clicks stay exactly as they are. `click_fraction` in
`wavetracker.json` tells how much was repaired, and the stored spectrograms show the
repaired traces. On 60 s of static 0.6 % of the samples change:

![Real static before and after click removal](images/static_click_removal.png)

**Thresholds were frozen** from the first block. They now follow the median noise of
the 10 s around each frame (`threshold_window`), and cannot fall below what noise on
that many channels has, 0.77 dB on 16 channels. A frozen threshold could be wrong
either way: too low for the water if estimated in the air, too high for the rest of a
recording if estimated during crackle.

**Noise at low frequencies is not the noise of the high band.** Below about 1 kHz the
noise in the water is often common to all electrodes: 16 of them spread like one,
4.7 dB against 1.1 dB. Thresholds therefore also follow the noise over frequency
(`threshold_band`). It is measured as how much a bin changes between windows that do
not overlap, so that fish, which are stable, do not raise it even in a chorus. Bands
only ever raise thresholds, and not where the noise is the same over frequency.

**One stray detection could ruin the tracker.** The amplitude error distribution came
from the 30 s after the first detection. A lone false detection left it empty, which
removed the electrode pattern from every link of the recording: two crossing fish
became 23 identities with 118 swaps. It now comes from the first window with
detections in at least half of its frames. `track(..., v6_compat=True)` keeps the
original.

Results on real field recordings are in
[Benchmarks](benchmarks.md#recordings-that-start-without-fish): what a recording
starts with no longer changes what is found. This has a price. On clean lab recordings
fish lose about 0.5 % of their frames (up to 5 % after `cleanup` on one trial), and
where the water is noisy at low frequencies a fish near 150 Hz needs up to 10 dB more
than before; above about 650 Hz little changes. To reproduce results of earlier
versions, set `clicks.enabled: false`, `harmonic_groups.threshold_window: null` and
`harmonic_groups.threshold_band: null`.

## Known limitations / next steps

* The amplitude-error distribution used for tracking is estimated once, from the
  first window of `3 * max_dt` seconds with detections in half of its frames.
* The default threshold factors keep empty recordings clean, but miss weak fish in
  noisy water: on a 16-channel field grid they found 35 % of the fish of a night. Lower
  them for such recordings, see [field recordings](field-recordings.md).
* Static that lasts longer than `clicks.max_duration` without a pause stays.
* Fish within ±1 Hz of a mains harmonic (multiples of 50 Hz) are not detected
  (set `mains_freq: 0` if there is no mains).
* A fish needs harmonics 1..`min_group_size` unless
  `max_missing_harmonics` > 0. Groups with gaps are judged by the same
  greedy assignment; a fish with gaps whose 3rd harmonic coincides (within
  a bin) with a stronger fish is lost.
* `min_good_peak_power` is an absolute level whose meaning depends on the
  recording's gain; it is not derived from the data.
* Detection thresholds are relative to the global noise floor; in a dense
  chorus the troughs between fish are far above it, so thresholds have to be
  lowered by hand (see [field recordings](field-recordings.md)).
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
* [`merge-by-position`](post-processing.md#merging-by-position-moving-electrodes) resolves synthetic surveys correctly, but on the Iriri
  boat survey (~40 fish) it still leaves many small, ambiguous groups (352);
  filter its fish table by detection count, ambiguity and fit quality.
