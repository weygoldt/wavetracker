# Review flags: where a tracked result most likely needs a correction

Goal: cut manual annotation by pointing the user to the few places where
the tracking is most likely wrong, most doubtful first, each with a
suggested fix. wavetracker writes the flags (`wavetracker.review`, config
section `review`); a reader such as the eodsorter plugin in claudian
(`src/audian_plugins/eodsorter/`) can show them and apply the suggestion.

## What gets a confidence, from which evidence

Every tracking stage decides with evidence it used to throw away. With
`review.enabled`, `pipeline.track_results` keeps it (a `review.Trace`):

| stage | kept |
|---|---|
| tracker | the fragment (tracker identity) of every detection |
| learned association (`tracking.method: learned`) | per detection: the probability of the link from its predecessor on its track, and of the best link into it from another track |
| fragment linker | every candidate pair: its 40 features, its probability, joined or not |
| rise joiner | every candidate pair: pattern rms difference, start and stop times, joined or not, tail dropped |
| identity salience filter | (the final identities) |

From it, every place where the result may be wrong becomes an *item*:

| kind | source | what may be wrong | fix (`action`) |
|---|---|---|---|
| `join` | `linker`, `rise` | two fragments joined into one identity are two fish | `split`: cut the identity where the join was |
| `track` | `identity` | an identity is not a fish (harmonic, clutter) | `delete` |
| `missed_join` | `linker`, `rise` | two identities that were candidates for a join (rejected by the threshold, or because they would share frames or run in parallel) are one fish | `merge` |

A join is undone exactly by removing it from the identity's join tree: the
joins of an identity connect its fragments without cycles (each joined two
identities that were apart), so removing one leaves two parts. In
`review.json` the cut is given as the frame `k` where the later part starts
(a reader that cuts in time); the benchmark applies the exact two parts.
Missed joins are the best candidate per pair of identities; linker
candidates need a probability of at least `review.min_missed_prob`.

Features (names in `wavetracker.review.FEATURES`):

* **joins and missed joins**: sizes and durations of both sides, their mean
  salience, the electrode patterns of the two sides (rms difference,
  correlation), the difference of their median frequencies, shared frames
  and simultaneous detections more than 3 Hz apart; for the linker the
  logit of its probability, the best competing candidate of either end, and
  its own features (gap, frequency jump, extrapolation error, pattern
  distances, relative signs, crowding, nearest neighbours, margins); for the
  rise joiner its pattern rms and correlation, the jump, gap or overlap,
  both sides' saliences and the competing candidates.
* **identities**: number of detections, duration, fill, salience (mean, 10th
  and 90th percentile), frequency and its spread, level relative to all
  identities, number of fragments and rise joins, the weakest linker join
  in it, the fraction of its detections with another identity at f / h
  (it is a harmonic) and at h f (h = 2..5), with another identity within
  1 Hz, the median distance to the nearest other identity, the drift of its
  electrode pattern from minute to minute; with the learned association
  also the mean and 5th percentile of its link probabilities and the mean
  best competing link.
* every item also carries its raw score (`raw_logit`).

## Calibrated confidence

`confidence` is the probability that the result is right at that place
(the join is right; the identity is a fish; not joining is right). A
`ReviewModel` (`.npz`, `review.model`) holds one model per group
(`join_linker`, `join_rise`, `track`, `missed_linker`, `missed_rise`):
standardised features (missing ones at the training mean) -> logistic
regression or a small MLP -> probability, optionally followed by a monotone
map. Without a model, `confidence` is the stage's own score, uncalibrated
(linker probability, mean salience, 1 - rms / 14 dB for rises) and
`review.json` says `"calibration": null`.

The models are fitted per pipeline on the `ltrain` mixtures
(`benchmarks/dense/review.py fit`): for every group, an isotonic map of the
raw score, logistic regressions (three L2 strengths) and an MLP (16 hidden
units) are compared by out-of-fold log loss (5 folds by mixture, lab and
field variants together); the best is refitted on all. Groups with fewer
than 40 items or 5 of either label get their base rate. See
[benchmarks](benchmarks.md#review-confidence-step-e) for how they calibrate
on dev and how much review they save.

## Config (`review`)

| key | default | meaning |
|---|---|---|
| `enabled` | `true` (classic pipeline: `false`) | keep the evidence and write `review.json`; the identities are the same with it on or off |
| `model` | `review-assocnet.npz` (classic: `null`) | path of the `ReviewModel` (.npz), or the name of a shipped one; null: raw scores, uncalibrated. The shipped model is calibrated for the default pipeline only |
| `missed_joins` | `true` | also flag rejected candidate joins |
| `min_missed_prob` | `0.05` | ... with at least this linker probability |
| `max_confidence` | `null` | write only items below this confidence (null: all) |

## File: `review.json` in the results directory

Written by `Results.save` when `review.enabled`. Readers must ignore unknown
keys and treat a missing file as "no flags".

```json
{
  "version": 1,
  "calibration": "/path/model-v1.npz",
  "items": [
    {
      "kind": "join",
      "source": "linker",
      "confidence": 0.31,
      "t": 341.2, "f": 512.3,
      "t0": 331.2, "t1": 351.2,
      "ids": [12.0],
      "action": "split",
      "k": 2083,
      "impact": 412,
      "why": "linker p = 0.71 across 2.1 s, df 0.4 Hz, patterns 6.1 dB apart"
    }
  ]
}
```

Fields of an item (items are sorted by `confidence`, lowest first):

| key | meaning |
|---|---|
| `kind` | `join`, `track` or `missed_join` (see above) |
| `source` | `linker` or `rise` (joins and missed joins), `identity` (tracks) |
| `confidence` | calibrated probability that the current result is right at this place |
| `t`, `f` | where to look: time [s, recording time as in `times.npy`] and frequency [Hz]: the junction of a join (the later side's start), an identity's first detection |
| `t0`, `t1` | time span to show around it [s] (an identity: its span) |
| `ids` | identities involved, as in `ident_v` (one for `join` and `track`, two for `missed_join`, earlier first) |
| `action` | suggested fix: `split` (cut the identity at frame `k`), `merge` (merge `ids[1]` into `ids[0]`), `delete` (unassign the identity) |
| `k` | frame index (into `times`) where a `split` cuts: frames `>= k` of the identity get a new id. Only for `split`. |
| `impact` | detections the fix would move (the smaller side of a join or missed join, the identity's detections) |
| `why` | one short line for the reader |

Every identity has a `track` item and every join a `join` item, so the
file also gives a confidence for each identity and each join. Ordering by
`(1 - confidence) * impact` (expected detections gained) visits the costly
errors first and was slightly better than plain confidence order on the
benchmark.

Identities can change after the file was written (edits in the GUI). A
reader skips an item whose ids no longer exist, and an item whose suggested
fix has already been done (e.g. the two ids of a `missed_join` are already
one).

## How it is judged

On the 51 dev mixtures (`benchmarks/dense/review.py evaluate`): a user
fixes the flagged places in review order, simulated by applying the right
fix where the item is really wrong at that moment (judged on the current
identities): a wrong join keeps the side on the identity's own fish and
moves the other side to the identity its fish is mostly on (or a new one;
deleted if it is not a fish), a false identity is deleted, a missed join is
merged. IDF1 after every item gives the review-efficiency curve, against
random order, time order, the raw scores and the eodsorter's rule-based
issues in time order (join, gap, short, crossing, harmonic).
