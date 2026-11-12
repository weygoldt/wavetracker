# Learned detector: open items

Running list for the `learned-detector` branch (newest first). Results and
numbers live in `docs/benchmarks.md`.

## Fish whose harmonics are hidden (2026-11-12, boat take DR0000_0209) - decided: not pursued

Decision (user, 2026-11-12): do not optimise for fish with no visible
harmonics; that would re-introduce false tracks of noise bands and stray
lines. The explanation below is kept for reference; the noise-shelf
option stays in `salience_train.py`, off by default, and is not part of
the planned retraining.


The strongest fish of the 2-electrode boat take, at 306-309 Hz for the
whole take (a rise at 289.5 s; absent before immersion), is detected in
19 % of its frames by `learned-rises` (v1 21 %, classic 61 %). Tracking
loses nothing (one identity); v1's identity is dropped by
`min_identity_salience` (mean 0.74). The loss is the network's salience
(median 0.07 along the fish). The fish sits in a quiet band just below the
chorus (350-2000 Hz), whose floor is ~20 dB higher, so none of its
harmonics is visible. The network wants harmonics: adding a synthetic 2nd
harmonic to its input raises the frames above 0.6 from 19 % to 63 % (with a
3rd 84 %), while a stronger fundamental lowers them (8 %); sub-harmonics
(a weak unrelated line near f/2 in 100-200 s) and the 100-Hz normalisation
matter little. In `ltrain`, fish as strong as this one have a visible
harmonic in 86-95 % of frames (this one: 28 %). The same failure on held-out
`ldev` crops whose harmonics are put under a +20 dB noise shelf: recall of
visible fish 0.91 / 0.94 -> 0.30 / 0.31 (lab / field).
Diagnosis, figures and numbers: `benchmarks/boatfish.py`,
`$WT_BENCHMARK_DATA/field_check/C-bf-NOTES.md`,
`$WT_BENCHMARK_DATA/figures/boatfish/`.

Tried: the noise-shelf augmentation of
`salience_train.py` (`--shelf-p 0.3 --shelf-db 10,30 --shelf-edge
150,1200`): white noise above a random edge raises the floor there, so
fish below the edge lose their harmonics and keep their fundamental.
Checked with a 2000-step fine-tune of the step-B detector (`C-bf-shelf-s0`,
not a candidate): normal `ldev` recall unchanged (0.913 / 0.933 against
0.911 / 0.936), recall under a +20 dB shelf 0.30 / 0.31 -> 0.66 / 0.70, the
boat fish 19 % -> 58 % of its frames, on one identity (classic 59 %). Its
salience is calibrated higher (precision at 0.5: 0.91 / 0.93 against
0.96 / 0.96), so the detection threshold has to be re-tuned with it.

Through `learned-rises` on `ldev` + `ldev-field` (model swapped, against
learned-rises and against a fine-tune without the shelf) IDF1 is unchanged
(-0.001 [-0.007, +0.006]), but harmonics reported as fish come back (0.14
-> 0.32 per frame), precision drops by 0.029 [-0.048, -0.012] and false
identities >= 30 s rise from 0.5 to 1.4 per mixture: once missing
harmonics are no reason to reject, a fish's own 2nd-5th harmonic looks like
a fish. The network cannot tell "no harmonic because the floor there is
20 dB higher" from "no harmonic on the same floor" (a harmonic line, a pure
tone), since the 100-Hz normalisation removes the floor level. So the
retraining should also give it the floor: per harmonic channel, the band
median of the normalisation at h*k minus that at k (a `SalienceSpec`
field, off by default; untested). Check on `ldev` / `ldev2` with harmonics
per frame and false identities, not only IDF1. Lowering the threshold or
`start_salience` does not help this fish (at most 37-48 % of its frames,
nothing in 125-225 s).

## Detector traces only part of steep rises (2026-11-12, future, not now)

The detector retrained on the corrected rise truth (`D-rt-tcn64aug-s0/s1`)
finds only about 1 point more of the steep-part frames than the step-B
detector at the same threshold 0.5 (+0.011 / +0.009), IDF1 unchanged; the
+0.038 first reported came mostly from the lower threshold (0.5 against
0.6). Not adopted. The steepest climbs (about 10 Hz in 0.2 s) are still
missed in the middle. Likely reasons: rise frames
are 0.35 % of the targets, and the 1.64 s window smears a fast climb into a
vertical stripe. Future options (user, 2026-11-12: not top priority):
- weight the rise frames more in the loss;
- give the network a second, short-window spectrogram (e.g. 0.2 s) as
  input, where the climb is visible.

## Truth misses the steep part of rises (2026-11-11, noticed by the user) - truth fixed, retrained (see above)

Done (branch learned-risetruth, 2026-11-12): steps 1-3 and the rise-onset
stratum. `rise_truth.py` follows 271 of 415 onsets of the dev sources by the
fish's instantaneous frequency (87 were traced already); `truth_rises.npz`
of `ltrain`, `ldev`, `ldev2` (+ field) flags ~520 new samples per mixture,
a quarter of which the old truth had more than 2 Hz off. Every current
detector finds fewer than half of these frames. All scripts take
`--truth rises`. Left: step 4, retraining (GPU; commands in
`$WT_BENCHMARK_DATA/salience/D-rt-NOTES.md`), and then whether the rise
joiner is still needed. See `docs/benchmarks.md`, "Truth through the steep
part of rises".

Original note:

At most rises the tracks stop at the onset and continue only after the
steep part, which stays untracked; the rise joiner bridges the identity but
not the frequency excursion. Cause: the benchmark truth comes from tracking
each source recording alone with the original wavetracker, which does the
same, so the steep part is missing from the truth (or bridged by the
ridge completion along a straight line). The detector learned exactly
that: those frames are ignored in the loss, or the target bump lies on the
bridge instead of the ridge, and the benchmark does not count them as
missed.

Fix sketch:
1. In each source (two fish, high SNR), find rise onsets in the truth
   (a jump of more than a few Hz within one or two frames, or a gap
   followed by a higher frequency that decays back).
2. Follow the spectral ridge of that fish through the onset frame by frame
   in the source's own spectrogram (peak search in a window that moves with
   the slope, maybe on a finer time grid), and replace the truth there; flag
   the new frames so they can be scored apart.
3. Rebuild the mixture truths from the source truths (the warp and gain of
   every source are in `mixture.json`; the audio does not need to be
   regenerated).
4. Retrain the detector; score with a "rise onset" stratum. The detector's
   temporal context should then learn to trace the whole rise.

Check first on a few sources by eye (truth over the spectrogram at rises).

## Confidence of tracks, not just of detections (2026-11-11, idea from the user)

Report how sure we are that a track is a fish and that its identity is
right, so users (and the eodsorter GUI) can see where to look. Today there
is only the detector's salience per detection. Ideas, not decided:
- per identity and per segment: mean and lowest salience, fraction of
  frames detected, harmonic or line-like evidence (the network could get a
  "line but not a fundamental" output);
- per join: the linker's probability and the rise joiner's pattern match,
  stored with the result, so weak joins can be reviewed;
- calibration: check on the dev mixtures that a stated confidence matches
  how often it is right (reliability curves), per detection and per join;
- uncertainty from several models (two seeds or an ensemble) or from a
  learned association model's link probabilities;
- a confidence map over the spectrogram (salience plus association
  confidence) shown in the GUI next to the tracks.

## Close pairs (done: learned association and pair filling merged)

Two fish closer than about 1.2 Hz are limited by association: even a
perfect pair detector lowers IDF1 because the tracker swaps the two
detections of a split peak (see `docs/benchmarks.md`, "Two fish in one
peak"). The pair head and the electrode branch are built and off by default.

## Smaller items

- Truth SNR of a fish in a close pair includes the partner's power
  (`mixture.py`), so close-pair strata by SNR are mislabeled.
- Retrain the fragment linker on the step-B detector's fragments (it was
  trained on v1 fragments).
- Detector threshold 0.5 is now used (2026-11-12); the association network
  and its linker were trained on 0.6 detections: retrain them on 0.5
  detections, and try thresholds below 0.5.
- Ghost tails: the detector keeps reporting the old frequency for 1-3 s
  after a rise onset (its +-26 s temporal context); a narrower or causal
  context may remove them at the source.
- Check the rise joiner's joins on real grid recordings.
- Final model: train long on all data (including new recordings when they
  come) and score the `test` split once. The step-B detector with the
  learned association (`learned-assocnet.yaml` at threshold 0.5) is the
  packaged default since 2026-11-12 (user's request); the `test` split is
  still unscored.
