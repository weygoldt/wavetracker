# Association on the dense benchmark (step B, `learned-assoc`)

Working notes: where the tracker and linker lose identities, and what was
tried. All numbers are means over the 17 mixtures of `ldev` / `ldev-field`
(lab / field noise), with detections fixed. "Learned" is learned v1 with
mirrored edges (`A-v1m-relinked`, config `configs/learned-relinked.yaml`);
"oracle" is truth detections >= 0 dB (`A-oracle-snr0`, `configs/dense-linked.yaml`).

## Diagnosis (`assoc_diag.py`)

`assoc_diag.py` reruns the tracker on a run's detections and compares its raw
fragments (before linking) with the truth.

**Where the IDF1 goes** (lab / field):

| | learned | oracle |
|---|---|---|
| raw tracker fragments | 0.577 / 0.549 | 0.652 / 0.605 |
| tracker + linker (baseline) | **0.739 / 0.723** | **0.775 / 0.740** |
| raw fragments, perfectly joined (each labelled by its majority fish) | 0.769 / 0.754 | 0.850 / 0.809 |
| fragments cut at every fish switch, perfectly joined | 0.781 / 0.765 | 0.881 / 0.838 |
| truth identities on the detections the tracker kept | 0.788 / 0.773 | 0.881 / 0.838 |
| truth identities on all detections | 0.790 / 0.775 | 0.887 / 0.843 |

So, for learned v1 (lab), the 0.051 between the baseline and the truth
identities splits into:

* **joining the tracker's fragments: 0.030** (perfect joining of the raw
  fragments minus the linker);
* **fish switches inside fragments: 0.012**;
* false and harmonic detections inside fragments: 0.008 (2.6 % of the
  assigned detections; the truth identities leave them out);
* detections the tracker leaves unassigned: 0.002.

With oracle detections, joining costs 0.075 and switches 0.031. Fragment
purity is lower there (0.954 vs 0.981) because every close fish is detected,
so more pairs can be confused.

**Switches** (two consecutive detections of a fragment on different fish).
The situation of each, first match wins: crossing (the two truth lines cross
within +-5 s), rise (either fish is above 0.3 Hz/s or more than 1.5 Hz above
its baseline), gap (more than 1 s between the two), then separation.
Cost = IDF1 gained by cutting the fragments at those switches, with perfect
joining:

| situation | learned: per mixture | cost | oracle: per mixture | cost |
|---|---|---|---|---|
| crossing | 131 (78 %) | 0.008 / 0.008 | 256 (90 %) | 0.025 / 0.024 |
| rise | 17 | 0.004 / 0.004 | 21 | 0.006 / 0.006 |
| parallel < 1.2 Hz | 19 | 0.003 / 0.003 | 4.5 | 0.002 / 0.001 |
| gap > 1 s | 0.3 | 0.000 | 2.3 | 0.002 / 0.001 |
| parallel 1.2-3 Hz, >= 3 Hz | 0.7 | 0.000 | 0.1 | 0.000 |

Half of all switches happen while the two fish are less than 0.25 Hz apart
(82 of 168 per mixture, learned). With learned detections the two fish are a
single line there, and the scorer gives each detection to whichever fish is
nearer: no tracker can get those right, so the truth-identity ceiling
overstates what association can win. Example: `ldev/density-30-s0`, fish 3,
5 and 17 (different sources) wander within the same 20 Hz and are within
0.5 Hz for minutes; one fragment carries fish 3 and 5 alternately for 250 s.

With the real linker, cutting the fragments at the crossings (where the truth
says they switch) gains nothing for learned (-0.001 / -0.003), since the
linker then fails to join the pieces, and +0.011 / +0.007 for oracle.

**Breaks** (consecutive detections of a fish in different raw fragments).
Cost = IDF1 gained by joining the final identities at the breaks the linker
left open, where both identities belong mostly to that fish (an oracle join):

| break | learned: per mixture | joined by linker | cost | oracle: per mixture | joined | cost |
|---|---|---|---|---|---|---|
| rise (onset, decay, fast modulation) | 73 | 38 % | **0.013 / 0.015** | 87 | 25 % | 0.015 / 0.016 |
| gap 1-10 s | 44 | 33 % | 0.007 / 0.006 | 26 | 24 % | 0.003 / 0.007 |
| crossing | 24 | 12 % | 0.005 / 0.003 | 386 | 2 % | **0.030 / 0.027** |
| dropout >= 10 s | 40 | 60 % | 0.004 / 0.006 | 49 | 26 % | 0.006 / 0.005 |
| next to a line < 3 Hz | 3 | 20 % | 0.001 / 0.002 | 23 | 2 % | 0.009 / 0.010 |
| alone | 6 | 24 % | 0.000 / 0.001 | 1.5 | 0 % | 0.002 / 0.001 |
| all | 190 | 37 % | 0.023 / 0.024 | 571 | 9 % | 0.050 / 0.045 |

Rise breaks on learned detections are mostly *inside the decay*: 600 of 1248
have a jump of 0.5-2 Hz between consecutive frames (the fish 4.5 Hz above
baseline, falling at 0.7 Hz/s) and break the 0.5 Hz short-gap tolerance
(`gap_tolerance`); 144 are onsets (jumps > 2 Hz); the rest interleave (two
fragments alternating on one wiggling fish).

**Why the linker leaves breaks open** (learned, lab; per mixture, cost of an
oracle join of those breaks):

| reason | per mixture | cost |
|---|---|---|
| blocked in association, p >= 0.7 (mostly `max_shared`) | 16 | **0.010** |
| no candidate (`max_df`, `max_gap`) | 7 | 0.007 |
| probability below 0.7 | 5 | 0.006 |
| fragment shorter than 10 detections | 42 | 0.004 |
| one side impure | 43 | 0.002 |

Of the pairs above the threshold that the association refuses, 26 per
mixture are refused for frames shared with earlier joins and 6 for frames
shared within the pair: one shared frame is more than 5 % of a 17-detection
fragment. 64 % and 77 % of them are the same fish. The parallel rule refuses
4 per mixture, 98 % of them the same fish. On oracle detections the largest
item is the threshold 0.9 of the packaged linker (0.033).

Near-duplicates (two detections of one fish in a frame) are not a cause:
pairs closer than 1.25 Hz in a frame are 0.07 % of the detections.

The rest of the rise breaks interleave: the fish's frequency wobbles by about
1.5 Hz within one 1.64 s analysis window, the spectrogram shows both
frequencies at once, and the detector reports two lines for one fish (e.g.
`ldev/atten-dyad-30-s0`, fish 23, 125-130 s and 152-156 s); the tracker
builds two parallel fragments, and the linker may not join them because they
share frames.

## Experiments (retracking the same detections)

Run tags `B-assoc-<name>` (learned) and `B-assoc-<name>-oracle`. Retracking
is deterministic and the baseline reproduces bit for bit, but the tracker is
chaotic: a small change flips crossings and moves single mixtures by up to
0.04 IDF1, so differences of the 17-mixture means below about 0.005 are not
reliable. A change counts if it holds in lab and field noise.

| change | learned IDF1 | oracle IDF1 |
|---|---|---|
| baseline (`base`) | 0.739 / 0.723 | 0.775 / 0.740 |
| `linking.max_shared` 0.15 | 0.741 / 0.725 | 0.779 / 0.742 |
| `linking.max_shared` 0.3 | 0.741 / 0.726 | 0.781 / 0.746 |
| `linking.max_shared` 0.6 | 0.741 / 0.726 | |
| `max_shared` 0.3 + `max_parallel` 0.6 | 0.742 / 0.728 | |
| `max_shared` 0.3 + `min_detections` 5 | 0.741 / 0.726 | |
| `max_shared` 0.3 + `threshold` 0.6 / 0.8 | 0.742 / 0.726, 0.740 / 0.724 | |
| `max_shared` 0.3 + rise candidates (`rise_df` 30, `rise_gap` 5) | 0.736 / 0.723 | |
| wider tolerance for isolated detections (1.0 Hz) | 0.739 / 0.721 | 0.777 / 0.744 |
| the same, 1.5 Hz | 0.739 / 0.719 | 0.778 / 0.744 |
| identity salience >= 0.75 after linking (offline) | 0.743 / 0.728 | |
| motion, symmetric slopes +-2 s (first version) | 0.743 / 0.720 | **0.795 / 0.754** |
| motion, symmetric slopes +-5 s | 0.740 / 0.719 | 0.790 / 0.750 |
| motion, one-sided slopes, 1 s | 0.744 / 0.718 | 0.791 / 0.750 |
| motion, one-sided slopes, 2 s | 0.742 / 0.720 | 0.790 / 0.746 |
| motion, one-sided, shrunk slopes, horizon (`motion_window` 1) | 0.743 / 0.723 | 0.786 / 0.749 |
| the same, `motion_window` 3 | 0.744 / 0.722 | 0.788 / 0.749 |

* Isolated tolerance (`$WT_BENCHMARK_DATA/assoc/isolated-tolerance.patch`,
  not kept): a detection with no other line within +-3 Hz for +-1 s may link
  across up to 1-1.5 Hz instead of 0.5 Hz. Rise breaks fall from 73 to 56
  per mixture, but only those the linker already joined; the IDF1 does not
  move.
* Rise candidates: the linker was never trained on such pairs, and the
  candidates change the competition features of all pairs.
* Identity salience: identities whose mean salience is below 0.75 are 79 %
  harmonic or false detections. Dropping them before linking loses weak
  fragments of real fish (-0.002); after linking it gains +0.004.

**Motion model** (`tracking.motion_window`). The tracker compares detection
pairs: its link error is 2/3 the *rank* of the electrode-pattern distance
among all pairs plus 1/3 a sigmoid of the frequency difference (0.35 Hz
midpoint). A first pass tracks as before; every detection then gets the slope
of its identity, and the second pass predicts along it. Lessons:

* Slopes over a window around the detection (first version) mix both fish at
  a crossing, where the first pass may have swapped them. Slopes from the
  detection's own side (the past of the earlier detection of a link, the
  future of the later) are not mixed (OC-SORT's lesson: keep the motion
  estimate from before the occlusion).
* One-sided slopes over 1-2 s are noisy for faint fish in field noise
  (0.3 Hz/s error for 0.2 Hz frequency scatter, 1.5 Hz after 5 s): field IDF1
  fell by 0.003-0.005. Shrinking each slope by its own variance
  (s * s^2 / (s^2 + var s)) and predicting at most `motion_window` ahead fixed
  that.
* A synthetic crossing (two fish, same pattern, 0.1 Hz/s apart, among eight
  others) is swapped in 6 of 8 draws without motion and kept apart in 7 of 8
  with it (`tests/test_tracking.py`). With only the two fish the pattern
  ranks span 0-1 and drown the frequency term; in real mixtures most pairs
  are between different fish, so the ranks of same-pattern pairs stay small.
* With the linker trained on the old fragments, motion raises purity (learned
  field 0.981 -> 0.984) and the raw tracker's IDF1, but the linker joins fewer
  of the new fragments (on `ldev-field/density-20,30`: joined 37 % -> 31 %
  of the breaks, more interleaved fragments at rises). Hence the retrained
  linker below.

**Retraining the linker on the fragments of a new tracker.** `linker.py
build` re-tracks a base run's detections with the tracking settings of `-c`
and caches fragments and pairs under `$WT_BENCHMARK_DATA/linker/<base>/`, so
every tracker variant needs its own base tag holding the `ltrain` detections
of `A-learned` (step A's model, `salience/linker-learned-v1.npz`, is
reproduced bit for bit by `train --base A-learned`):

```bash
python benchmarks/dense/run.py retrack --tag B-assoc-lt-motion2 --source A-learned -c benchmarks/dense/configs/learned-assoc.yaml --split ltrain   # and ltrain-field
python benchmarks/dense/linker.py build --base B-assoc-lt-motion2 -c benchmarks/dense/configs/learned-assoc.yaml --splits ltrain ltrain-field
python benchmarks/dense/linker.py train --base B-assoc-lt-motion2 --train-splits ltrain ltrain-field --val-splits ltrain-field --out $WT_BENCHMARK_DATA/assoc/linker-learned-v1-motion2.npz
```

(The runs here used base tags whose split folders link to the `A-learned`
and `A-v1m-relinked` runs instead of copying them, and validated on `ldev`;
the validation split does not change the model.)
