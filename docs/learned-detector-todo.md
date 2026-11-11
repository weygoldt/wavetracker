# Learned detector: open items

Running list for the `learned-detector` branch (newest first). Results and
numbers live in `docs/benchmarks.md`.

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

## Why the boat take's constant fish is missed (in progress, branch learned-boatfish)

The one fish present for the whole of `DR0000_0209` (probably the ~308 Hz
line: v1 found it in 19 % of its frames, classic in 60 %) is barely
tracked. Diagnosis only; the boat takes are not evaluation data.

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
