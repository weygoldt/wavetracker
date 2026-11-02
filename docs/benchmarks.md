# Benchmarks

## Tube competition recordings

`benchmarks/tube_competition.py` scores results on the 2022 tube-competition
recordings (two fish per trial, no ground truth traces). It builds a
pseudo ground truth from the detections, independent of tracking, and reports
purity, fragmentation and coverage for the raw tracks and for `cleanup -n 2`
(see the module docstring for usage).

Results on 10 full 4 h trials (all six pairings) with the species range
`benchmarks/tube_competition.yaml` (400–1200 Hz), means over trials:

| | first rewrite | current |
|---|---|---|
| identities with ≥ 300 detections (truth: 2) | 32.2 | 9.9 |
| identities covering 90 % of the loser | 13.0 | 3.0 |
| largest track, winner / loser [% of frames] | 50 / 34 | 64 / 57 |
| after `cleanup -n 2`, winner / loser [% of frames] | 91 / 84 | 95 / 92 |
| purity (detections assigned to the right fish) | 1.000 | 1.000 |

Detection itself covers 95 % (winner) and 92 % (loser) of the frames; the rest is
mostly rises, which leave the pseudo-ground-truth band. Remaining
fragmentation is concentrated in losers with many rises (pairing 4a: 79 %
after cleanup).

![Coverage per fish and identities per trial](images/tube_benchmark_summary.png)

Tracks of a rise-heavy trial before and after `cleanup -n 2` (successive
identities of one fish alternate light/dark):

![Tracks of trial 2022-06-20](images/tube_tracks_2022-06-20.png)

## Recordings that start without fish

Results for a fish must not depend on what a recording starts with.
`benchmarks/fish_free_start.py` puts real fish-free data in front of a real recording
with fish, and compares the fish one to one with the recording alone, so that no
ground truth is needed.

The deployment below (16 channels, 48 kHz, `min_group_size: 2`, no absolute power
limit, no mains) was started at 11:18, and fish come at dusk, around 19:00. The first
measure is the part of 262 minutes of fish tracks in 2.5 h of night that is detected,
the fish taken from the most sensitive run. The night is analysed alone, after the 8 h
of day, and after 10 min of real static:

| threshold factors | fish found, before: alone / day / static | after |
|---|---|---|
| 6 and 10 (default) | 2 / 100 / 93 % | 35 / 35 / 35 % |
| 4 and 7 | 21 / 93 / 88 % | 76 / 77 / 76 % |
| 3 and 5 | 62 / 89 / 86 % | 92 / 92 / 92 % |

Before, the result depended on the noise of the first block. The price of that
sensitivity in the empty day and in the static, per frame:

| threshold factors | false detections, before: day / static | after |
|---|---|---|
| 6 and 10 (default) | 3.3 / 3.6 | 0.02 / 0.02 |
| 4 and 7 | 13 / 8.5 | 0.17 / 0.27 |
| 3 and 5 | 29 / 24 | 1.4 / 5.5 |

With factors 4 and 7, before and after:

![A day without fish, then fish at night, before and after](images/fish_free_day_night.png)

The night's tracks of 30 s or more number 50 alone, 53 after the day and 55 after the
static, and 87 % of them stay one identity. Before they numbered 24, 604 and 2055.
Fish-free recordings, at the default settings:

| 35 to 70 min, no fish | detections per frame | identities |
|---|---|---|
| static | 2.83 → 0.01 | 6400 → 23 |
| quiet, then in the water | 1.65 → 0.02 | 1850 → 63 |
| quiet, then in the water | 0.58 → 0.03 | 1290 → 128 |
| in the water, with noise bursts | 1.13 → 0.01 | 2273 → 25 |

What it costs, on recordings with fish:

| | before | after |
|---|---|---|
| tube trials (8), fish detected, winner / loser [% of frames] | 96.1 / 92.0 | 95.6 / 91.4 |
| ... after `cleanup -n 2` | 95.0 / 91.4 | 94.4 / 91.0 |
| ... identities | 485 | 186 |
| ... purity | 1.000 | 1.000 |
| Iriri chorus, detections per frame | 60.4 | 57.7 |
| runtime of a 4 h tube trial [s] | 62 | 67 |

The worst single tube trial lost 5 points after cleanup (6a, winner, 96.3 to 91.1 %),
others gained up to 6 (6c, winner). Two settings are trade-offs, measured there. A
click threshold of 8 finds more of the night's fish at 4 and 7 (81 % of a stretch
against 58 %), but broke the tracks of 2 of the 16 lab fish apart. Stronger thresholds
over frequency remove more false detections in noise bursts at the cost of fish: on
the fish of that night, 0.7 instead of the 0.6 of `DetectionThresholds.BAND_SCALE`
lost 6 points of what is found without bands, and gained 0.01 per frame.

## Dense mixtures of tube recordings

`benchmarks/tube_mixture.py` sums 8 two-fish tube recordings electrode by
electrode. Each source is time-warped (frequency scale 0.4-1.6, plus a slow
drift common to all sources, std 0.5 % over ~5 min) so fish spread over
250-1300 Hz with natural modulations and harmonics land among other fish;
hum combs are removed first. Ground truth comes from tracking each source
alone. `benchmarks/harmonic_comodulation.py` scans the timescale,
`benchmarks/harmonic_rule.py` evaluates the rule per identity.

## Low-frequency fish without a 2nd harmonic

`benchmarks/low_frequency.py` synthesizes odd-harmonic fish (no even
harmonics) at 120–300 Hz in a dense 30-fish chorus at 350–900 Hz (2 channels,
48 kHz, field-recording settings) and compares `max_missing_harmonics` 0 and 1.
60 s, three seeds:

| | `max_missing_harmonics: 0` | `1` |
|---|---|---|
| recall of the low fish (fraction of frames) | 0.29 / 0.22 / 0.21 | 0.89 / 0.85 / 0.82 |
| recall of the chorus fish | 0.92 / 0.91 / 0.93 | 0.96 / 0.93 / 0.96 |
| unmatched detections at f/2 of a chorus fish | 57 / 25 / 81 | 31 / 25 / 62 |
| unmatched detections at f/3 of a chorus fish | 64 / 47 / 45 | 29 / 27 / 27 |

(430 frames each.) Groups with gaps cost about two extra unmatched
detections per frame above 900 Hz (chance groups among chorus harmonics).
