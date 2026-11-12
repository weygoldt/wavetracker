# Review flags: where a tracked result most likely needs a correction

Goal: cut manual annotation by pointing the user to the few places where
the tracking is most likely wrong, most doubtful first, each with a
suggested fix. wavetracker writes the flags; the eodsorter plugin in
claudian (`src/audian_plugins/eodsorter/`) shows them in its Issues
section (`G` / `Shift+G` visit them, Enter applies the suggestion).

## File: `review.json` in the results directory

Written after tracking (and after the rise joiner and the identity
salience filter) when `review.enabled` is true. Readers must ignore unknown
keys and treat a missing file as "no flags".

```json
{
  "version": 1,
  "calibration": "name or path of the calibration used, or null",
  "items": [
    {
      "kind": "join",
      "confidence": 0.31,
      "t": 341.2, "f": 512.3,
      "t0": 338.0, "t1": 344.0,
      "ids": [12],
      "k": 2083,
      "action": "split",
      "why": "linker p = 0.41 over a 2.1 s gap, electrode patterns differ"
    }
  ]
}
```

Fields of an item:

| key | meaning |
|---|---|
| `kind` | `join` (two fragments the linker or rise joiner joined), `link` (two consecutive detections the tracker put on one identity: a possible switch inside a track), `missed_join` (two identities that were *not* joined but maybe should be), `track` (a whole identity that may not be a fish: harmonic, stray line, clutter) |
| `confidence` | calibrated probability that the current result is right at this place (the join is right; the link is right; not joining is right; the track is a real fish). Lowest first is the review order. |
| `t`, `f` | where to look: time [s, recording time as in `times.npy`] and frequency [Hz] |
| `t0`, `t1` | time span to show around it [s] |
| `ids` | identities involved, as in `ident_v` (one for `join`, `link`, `track`; two for `missed_join`) |
| `k` | frame index (into `times`) where a `split` cuts: frames `>= k` of the identity get a new id. Only for `action: split`. |
| `action` | suggested fix: `split` (cut the identity at frame `k`), `merge` (merge `ids[1]` into `ids[0]`), `delete` (unassign the identity), or `none` (look only) |
| `why` | one short line for the reader |

Identities can change after the file was written (edits in the GUI). A
reader skips an item whose ids no longer exist, and an item whose suggested
fix has already been done (e.g. the two ids of a `missed_join` are already
one).

## Calibration

`confidence` is calibrated on benchmark data with known truth (the `ltrain`
mixtures, never `ldev`/`ldev2` or `test`): per kind, a monotone map from the
raw score (linker probability, association probability, rise-joiner pattern
match, identity salience and similar) to the observed rate of being right.
Reliability is checked on the dev mixtures. The calibration is stored with
the package or the model it belongs to, so that results say which was used.

## How it is judged

On the 51 dev mixtures: if a user fixes the flagged places in review order
(simulated by applying the suggested fix when it is right), how much of the
gap to the truth does each reviewed item close? The curve "IDF1 against
number of items reviewed" (per mixture and per hour of recording) is the
measure, compared with the GUI's existing rule-based issues in time order.
