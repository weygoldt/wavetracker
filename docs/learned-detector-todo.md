# Learned detector: open items

Running list for the `learned-detector` branch (newest first). Results and
numbers live in `docs/benchmarks.md`.

## Fish whose harmonics are hidden (2026-11-12, boat take DR0000_0209)

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

Fix for the planned retraining: the noise-shelf augmentation of
`salience_train.py` (`--shelf-p 0.3 --shelf-db 10,30 --shelf-edge
150,1200`): white noise above a random edge raises the floor there, so
fish below the edge lose their harmonics and keep their fundamental.
Checked with a 2000-step fine-tune of the step-B detector (`C-bf-shelf-s0`,
not a candidate): normal `ldev` recall unchanged (0.913 / 0.933 against
0.911 / 0.936), recall under a +20 dB shelf 0.30 / 0.31 -> 0.66 / 0.70, the
boat fish 19 % -> 58 % of its frames, on one identity (classic 59 %). Its
salience is calibrated higher (precision at 0.5: 0.91 / 0.93 against
0.96 / 0.96), so the detection threshold has to be re-tuned with it.

## Truth misses the steep part of rises (2026-11-11, noticed by the user)

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

## Close pairs (in progress, branches learned-pairtrack, learned-assocnet)

Two fish closer than about 1.2 Hz are limited by association: even a
perfect pair detector lowers IDF1 because the tracker swaps the two
detections of a split peak (see `docs/benchmarks.md`, "Two fish in one
peak"). The pair head and the electrode branch are built and off by default.

## Smaller items

- Truth SNR of a fish in a close pair includes the partner's power
  (`mixture.py`), so close-pair strata by SNR are mislabeled.
- Retrain the fragment linker on the step-B detector's fragments (it was
  trained on v1 fragments).
- Ghost tails: the detector keeps reporting the old frequency for 1-3 s
  after a rise onset (its +-26 s temporal context); a narrower or causal
  context may remove them at the source.
- Check the rise joiner's joins on real grid recordings.
- Final model: train long on all data (including new recordings when they
  come), score the `test` split once, then make it the packaged default
  (`detector.method: learned` still uses v1).
