"""Join and filter tracks, keeping the N most prominent fish.

The implementation of ``wavetracker cleanup``. It assumes a known number of
fish with persistent, well separated frequencies (grid recordings); for moving
electrodes use ``wavetracker merge-by-position``.
"""

import configparser
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from thunderlab.powerspectrum import decibel

#: Show the intermediate tracks in figures (``wavetracker cleanup --show``).
show_results = True


def load_config(config_path=None, folder=None):
    """
    Load cleanup configuration from .cfg file.

    Search order:
    1. Explicit config_path if provided
    2. cleanup_config.cfg in data folder
    3. cleanup_config_default.cfg next to this module
    4. Hardcoded defaults

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to config file
    folder : str or Path, optional
        Data folder to search for cleanup_config.cfg

    Returns
    -------
    dict
        Configuration parameters
    """
    # Default values
    defaults = {
        "stride_minutes": 10,
        "overlap_frac": 0.2,
        "freq_tolerance": 2.5,
        "time_tolerance_minutes": 5,
        "density_threshold": 0.1,
        "n_fish": 2,
        "assign_leftovers": 1,
    }

    # Try to find config file
    config_file = None

    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Specified config file not found: {config_path}")
            config_file = None

    if not config_file and folder:
        # Look in data folder
        candidate = Path(folder) / "cleanup_config.cfg"
        if candidate.exists():
            config_file = candidate

    if not config_file:
        # Look in script directory
        script_dir = Path(__file__).parent
        candidate = script_dir / "cleanup_config_default.cfg"
        if candidate.exists():
            config_file = candidate

    if not config_file:
        print("No config file found, using hardcoded defaults")
        return defaults

    # Load config
    print(f"Loading config from: {config_file}")
    config = configparser.ConfigParser()
    config.read(config_file)

    if "parameters" not in config:
        print("Warning: [parameters] section not found in config, using defaults")
        return defaults

    # Parse parameters
    params = {}
    for key in defaults:
        if key in config["parameters"]:
            value = config["parameters"][key]
            # Convert to appropriate type
            if key in ("n_fish", "assign_leftovers"):
                params[key] = int(value)
            else:
                params[key] = float(value)
        else:
            params[key] = defaults[key]

    return params


def frequency_kde(ff, convolve_f, sigma):
    """Sum of one area-normalised Gaussian per frequency in `ff`, sampled on
    the regular grid `convolve_f`, and the peak value of a single Gaussian.

    Frequencies are snapped to the grid, which keeps memory linear: evaluating
    every Gaussian on the grid needs len(ff) x len(convolve_f) floats (tens of
    GB for dense field data).
    """
    df = convolve_f[1] - convolve_f[0]
    edges = np.append(convolve_f - df / 2, convolve_f[-1] + df / 2)
    counts = np.histogram(ff, bins=edges)[0].astype(float)
    half = int(np.ceil(5 * sigma / df))
    kernel = np.exp(-((np.arange(-half, half + 1) * df / sigma) ** 2) / 2)
    kernel /= kernel.sum()
    # "full" cropped to the grid: "same" has the length of the longer input, and
    # the kernel is longer than the grid around a single integer frequency.
    kde = np.convolve(counts, kernel, mode="full")[half : half + len(counts)]
    return kde, kernel.max()


def get_valid_ids_by_freq_dist(
    times,
    idx_v,
    ident_v,
    fund_v,
    valid_v,
    old_valid_ids,
    i0,
    stride,
    f_th,
    kde_th,
):
    """Mark the identities of the window [i0, i0 + stride) that lie near a
    frequency with many detections, or that were valid before, in `valid_v`.

    Returns the detection-density threshold (estimated from the first window
    unless `kde_th` is given) and a table of the valid identities: (id, median
    frequency in the window, time of the identity's first detection).
    """
    window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

    ff = fund_v[(~np.isnan(ident_v)) & (window_t_mask)]

    if len(ff) == 0:
        # no detections in this window (e.g. electrodes out of the water):
        # nothing is valid here; keep the threshold. (This used to return the
        # flat id list `old_valid_ids` in place of the (n, 3) table.)
        return kde_th, np.empty((0, 3))

    # frequency grid spanning the detections of this window (was fixed to
    # 400-1200 Hz: detections outside gave NaN densities and no valid ids)
    margin = 5 * 2 * f_th
    convolve_f = np.arange(np.floor(ff.min() - margin), np.ceil(ff.max() + margin), 0.1)
    kde, g_max = frequency_kde(ff, convolve_f, sigma=2 * f_th)

    if not kde_th:
        kde_th = (
            g_max * len(times[(times >= times[0]) & (times < times[0] + stride)]) * 0.05
        )

    valid_f = convolve_f[kde > kde_th]
    valid_ids = []

    for id in np.unique(ident_v[(~np.isnan(ident_v)) & (window_t_mask)]):
        f = fund_v[(ident_v == id) & (window_t_mask)]
        if len(f) <= 1:
            continue

        # no valid frequency in this window: inf fails the test below safely
        min_abs_diff = (
            np.min(np.abs(valid_f - f[:, np.newaxis])) if valid_f.size > 0 else np.inf
        )
        if min_abs_diff <= (convolve_f[1] - convolve_f[0]) / 2 or id in old_valid_ids:
            valid_ids.append([id, np.median(f), times[idx_v[(ident_v == id)]][0]])
            valid_v[ident_v == id] = 1

    return kde_th, np.array(valid_ids)


def connect_by_similarity(
    times,
    idx_v,
    ident_v,
    fund_v,
    sign_v,
    valid_v,
    valid_ids,
    f_th,
    i0,
    stride,
    tick=None,
):
    """Join valid identities of one window whose median frequencies agree.

    Closest pairs first, within `f_th`. A pair is skipped if the two share more
    than 1 % of their frames, or if the joined track would jump by more than
    `f_th` where it changes from one identity to the other. Returns the valid
    ids after joining and `ident_v`.
    """
    if len(valid_ids) == 0:
        return valid_ids, ident_v

    window_t_mask = (times[idx_v] >= i0) & (times[idx_v] < i0 + stride)

    d_med_f = np.abs(valid_ids[:, 1] - valid_ids[:, 1][:, np.newaxis])

    med_power_id = np.zeros(len(valid_ids))
    for enu, id in enumerate(valid_ids[:, 0]):
        med_power_id[enu] = np.median(
            np.max(sign_v[(ident_v == id) & (window_t_mask)], axis=1)
        )

    idx0s, idx1s = np.unravel_index(np.argsort(d_med_f, axis=None), np.shape(d_med_f))
    similar_mask = idx0s == idx1s
    idx0s = idx0s[~similar_mask]
    idx1s = idx1s[~similar_mask]
    # the loop stops at the first pair further apart than f_th: that many
    n_pairs = int(np.count_nonzero(d_med_f[idx0s, idx1s] <= f_th))
    every = max(1, n_pairs // 200)

    for enu, (idx0, idx1) in enumerate(zip(idx0s, idx1s, strict=False)):
        if tick is not None and enu % every == 0:
            tick(enu, n_pairs)
        if np.abs(valid_ids[idx0, 1] - valid_ids[idx1, 1]) > f_th:
            break
        id0 = valid_ids[idx0, 0]
        id1 = valid_ids[idx1, 0]

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]

        more_id = id0 if len(taken_idxs0) > len(taken_idxs1) else id1
        less_id = id1 if more_id == id0 else id0

        double_idx = np.intersect1d(taken_idxs0, taken_idxs1)

        if (
            len(double_idx) > len(taken_idxs1) * 0.01
            and len(double_idx) > len(taken_idxs0) * 0.01
        ):
            continue

        fund_v_m = fund_v[(ident_v == more_id) & (window_t_mask)]
        fund_v_l = fund_v[(ident_v == less_id) & (window_t_mask)]

        idx_v_m = idx_v[(ident_v == more_id) & (window_t_mask)]
        idx_v_l = idx_v[(ident_v == less_id) & (window_t_mask)]

        d_idx_v = np.intersect1d(idx_v_m, idx_v_l)

        join_idx_v = np.concatenate((idx_v_m, idx_v_l[~np.isin(idx_v_l, d_idx_v)]))
        help_v = np.ones_like(join_idx_v)
        help_v[: len(idx_v_m)] = 0
        join_fund_v = np.concatenate((fund_v_m, fund_v_l[~np.isin(idx_v_l, d_idx_v)]))

        sorter = np.argsort(join_idx_v)
        help_v = help_v[sorter]
        join_fund_v = join_fund_v[sorter]
        join_idx_v = join_idx_v[sorter]

        if np.any(np.abs(np.diff(join_fund_v)[np.diff(help_v != 0)]) > f_th):
            continue

        ident_v[(np.isin(idx_v, np.array(double_idx))) & (ident_v == less_id)] = np.nan
        valid_v[(np.isin(idx_v, np.array(double_idx))) & (ident_v == less_id)] = 0

        if valid_ids[idx0, 2] < valid_ids[idx1, 2]:
            valid_ids[:, 0][valid_ids[:, 0] == id1] = id0
            ident_v[ident_v == id1] = id0
        else:
            valid_ids[:, 0][valid_ids[:, 0] == id0] = id1
            ident_v[ident_v == id0] = id1

    previous_valid_ids = np.unique(valid_ids[:, 0])
    if tick is not None:
        tick(n_pairs, n_pairs)

    return previous_valid_ids, ident_v


def connect_with_overlap(
    fund_v, ident_v, valid_v, idx_v, times, time_tol=5 * 60, freq_tol=2.5, tick=None
):
    """Join valid identities that are close in time and frequency, closest first.

    Pairs within `time_tol` [s] and `freq_tol` [Hz] that share at most 1 % of
    their frames are candidates. They join if they barely overlap, or if one
    of them is sparse in the overlap, in which case its detections there are
    dropped. Returns `ident_v`.
    """
    connections_candidates = []
    unique_ids = np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)])
    n_pairs = len(unique_ids) * (len(unique_ids) - 1) // 2
    every = max(1, n_pairs // 200)
    for enu, (id0, id1) in enumerate(itertools.combinations(unique_ids, r=2)):
        if tick is not None and enu % every == 0:
            tick(enu, n_pairs)

        t0 = times[idx_v[ident_v == id0]]
        t1 = times[idx_v[ident_v == id1]]

        t0_span = [t0[0] - time_tol, t0[-1] + time_tol]
        t1_span = [t1[0] - time_tol, t1[-1] + time_tol]

        t_overlap = (t0_span[0] <= t1_span[0] and t0_span[1] > t1_span[0]) or (
            t1_span[0] <= t0_span[0] and t1_span[1] > t0_span[0]
        )
        if not t_overlap:
            continue

        overlap_t = np.array(t0_span + t1_span)
        overlap_t = overlap_t[np.argsort(overlap_t)][1:3]

        f0 = fund_v[
            (ident_v == id0)
            & (times[idx_v] > overlap_t[0])
            & (times[idx_v] < overlap_t[1])
        ]
        f1 = fund_v[
            (ident_v == id1)
            & (times[idx_v] > overlap_t[0])
            & (times[idx_v] < overlap_t[1])
        ]

        if len(f0) <= 1 or len(f1) <= 1:
            continue

        f0_span = [np.min(f0) - freq_tol, np.max(f0) + freq_tol]
        f1_span = [np.min(f1) - freq_tol, np.max(f1) + freq_tol]

        f_overlap = (f0_span[0] <= f1_span[0] and f0_span[1] > f1_span[0]) or (
            f1_span[0] <= f0_span[0] and f1_span[1] > f0_span[0]
        )
        if not f_overlap:
            continue

        taken_idxs0 = idx_v[(ident_v == id0)]
        taken_idxs1 = idx_v[(ident_v == id1)]
        double_idx = np.isin(taken_idxs0, taken_idxs1)

        if (
            np.sum(double_idx) > len(taken_idxs1) * 0.01
            and np.sum(double_idx) > len(taken_idxs0) * 0.01
        ):
            continue

        i0 = idx_v[
            (ident_v == id0)
            & (times[idx_v] > overlap_t[0])
            & (times[idx_v] < overlap_t[1])
        ]
        i1 = idx_v[
            (ident_v == id1)
            & (times[idx_v] > overlap_t[0])
            & (times[idx_v] < overlap_t[1])
        ]

        freq_dists = np.abs(f0 - f1[:, np.newaxis])
        time_dist = times[np.abs(i0 - i1[:, np.newaxis])]
        valid_freq_dists = freq_dists[time_dist <= time_tol]
        mean_freq_dist = np.mean(valid_freq_dists)

        connections_candidates.append([id0, id1, mean_freq_dist])

    # (n, 3) even when no pair qualifies; np.array([]) would be 1-D.
    connections_candidates = np.array(connections_candidates, dtype=float).reshape(
        -1, 3
    )

    for pair_no in np.argsort(connections_candidates[:, 2]):
        id0 = connections_candidates[pair_no, 0]
        id1 = connections_candidates[pair_no, 1]

        idx_v0 = idx_v[(ident_v == id0)]
        idx_v1 = idx_v[(ident_v == id1)]
        if len(idx_v0) == 0 or len(idx_v1) == 0:
            continue

        first_last_idxs = np.array([idx_v0[0], idx_v0[-1], idx_v1[0], idx_v1[-1]])

        overlap_category = np.array([0, 0, 1, 1])
        overlap_category = overlap_category[np.argsort(first_last_idxs)]
        sorted_first_last_idxs = first_last_idxs[np.argsort(first_last_idxs)]

        overlap_count0 = len(
            idx_v0[
                (idx_v0 >= sorted_first_last_idxs[1])
                & (idx_v0 <= sorted_first_last_idxs[2])
            ]
        )
        overlap_count1 = len(
            idx_v1[
                (idx_v1 >= sorted_first_last_idxs[1])
                & (idx_v1 <= sorted_first_last_idxs[2])
            ]
        )

        double_idx = np.intersect1d(idx_v0, idx_v1)

        density0 = overlap_count0 / (
            sorted_first_last_idxs[2] - sorted_first_last_idxs[1] + 1
        )
        density1 = overlap_count1 / (
            sorted_first_last_idxs[2] - sorted_first_last_idxs[1] + 1
        )

        n_overlap_ratio0 = (
            len(double_idx) / overlap_count0 if overlap_count0 > 0 else np.nan
        )
        n_overlap_ratio1 = (
            len(double_idx) / overlap_count1 if overlap_count1 > 0 else np.nan
        )

        merge = False
        id_for_overlap = None
        del_id = None
        if overlap_count0 <= 1 and overlap_count1 <= 1:
            merge = True

        if (
            not merge
            and (n_overlap_ratio0 <= 0.1 or np.isnan(n_overlap_ratio0))
            and (n_overlap_ratio1 <= 0.1 or np.isnan(n_overlap_ratio1))
        ):
            merge = True
            id_for_overlap = id0
            del_id = id1
        if not merge:
            if density0 <= 0.1 and density0 * 3 < density1:
                merge = True
                id_for_overlap = id1
                del_id = id0
            elif density1 <= 0.1 and density1 * 0.3 < density0:
                merge = True
                id_for_overlap = id0
                del_id = id1
            elif (
                density0 <= 0.3 and n_overlap_ratio0 >= 0.7 and density0 * 2 < density1
            ):
                merge = True
                id_for_overlap = id1
                del_id = id0
            elif (
                density1 <= 0.3 and n_overlap_ratio1 >= 0.7 and density1 * 2 < density0
            ):
                merge = True
                id_for_overlap = id0
                del_id = id1

        if not merge:
            continue

        first_id = id0 if overlap_category[0] == 0 else id1
        not_first_id = id1 if first_id == id0 else id0
        last_id = id0 if overlap_category[-1] == 0 else id1

        if id_for_overlap is None:
            ident_v[ident_v == not_first_id] = first_id
        else:
            ident_v[
                (ident_v == del_id)
                & (idx_v >= sorted_first_last_idxs[1])
                & (idx_v <= sorted_first_last_idxs[2])
            ] = np.nan
            ident_v[
                (ident_v == id_for_overlap)
                & (idx_v >= sorted_first_last_idxs[1])
                & (idx_v <= sorted_first_last_idxs[2])
            ] = first_id
            if last_id != first_id:
                ident_v[(ident_v == last_id) & (idx_v > sorted_first_last_idxs[2])] = (
                    first_id
                )

        connections_candidates[:, 0][connections_candidates[:, 0] == not_first_id] = (
            first_id
        )
        connections_candidates[:, 1][connections_candidates[:, 1] == not_first_id] = (
            first_id
        )

    return ident_v


def power_density_filter(valid_v, sign_v, ident_v, idx_v, times, density_th=0.1):
    """Invalidate identities that are sparse or quiet (mean power <= -100 dB).

    A dense identity longer than 10 minutes that is quiet on average, but above
    the limit for more than 10 % of its detections, keeps the stretches where
    its smoothed power is above the limit as new identities. Returns `valid_v`.
    """
    dps = 1 / (times[1] - times[0])
    dB_th = -100

    for id in np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)]):
        i = idx_v[ident_v == id]
        density = len(i) / (i[-1] - i[0] + 1)
        p = np.max(sign_v[(ident_v == id)], axis=1)
        mean_power = np.mean(p)
        if density < density_th or mean_power <= dB_th:
            valid_v[ident_v == id] = 0
            if (
                mean_power <= dB_th
                and density > density_th
                and (i[-1] - i[0]) / dps > 10 * 60
            ):
                ii = np.arange(i[0], i[-1] + 1)
                pp = np.interp(ii, i, p)

                ppp = np.convolve(
                    pp,
                    np.ones(int(60 * dps)) / int(10 * 60 * dps),
                    mode="same",
                )
                corr_fac = np.convolve(
                    np.ones(len(pp)),
                    np.ones(int(60 * dps)) / int(10 * 60 * dps),
                    mode="same",
                )
                ppp *= 1 / corr_fac

                if len(p[p > dB_th]) / len(p) > 0.1:
                    up_idx = ii[:-1][(ppp[:-1] < dB_th) & (ppp[1:] > dB_th)]
                    down_idx = ii[1:][(ppp[:-1] > dB_th) & (ppp[1:] < dB_th)]

                    if len(up_idx) == 0:
                        up_idx = [i[0]]
                    if len(down_idx) == 0:
                        down_idx = [i[-1]]

                    if up_idx[0] > down_idx[0]:
                        up_idx = np.concatenate((np.array([ii[0] - 1]), up_idx))
                    if down_idx[-1] < up_idx[-1]:
                        down_idx = np.concatenate((down_idx, np.array([ii[-1] + 1])))
                    next_ident = np.nanmax(ident_v) + 1
                    for ui, di in zip(up_idx, down_idx, strict=False):
                        ident_v[(ident_v == id) & (idx_v >= ui) & (idx_v < di)] = (
                            next_ident
                        )
                    valid_v[ident_v == next_ident] = 1

    return valid_v


def assign_leftovers(
    fund_v,
    idx_v,
    ident_v,
    times,
    kept_ids,
    freq_tol=2.5,
    min_minutes=1.0,
    drift_per_minute=0.2,
    max_gap_minutes=30.0,
    max_conflict=0.05,
    baseline_window=60.0,
):
    """Assign identities outside the `kept_ids` to one of the kept fish.

    A leftover identity (with at least `min_minutes` of detections, largest
    first) joins kept fish k if it shares at most `max_conflict` of its frames
    with k and its baseline frequency (10th percentile; rises only go up)
    continues k's baseline before and/or after it within
    ``freq_tol + drift_per_minute * gap``. If several fish fit about equally
    well (cost ratio < 2) the leftover stays unassigned.

    Returns a copy of `ident_v` with assigned leftovers relabelled.
    """
    ident_v = ident_v.copy()
    t_det = times[idx_v]
    dt = times[1] - times[0]
    ids, counts = np.unique(ident_v[~np.isnan(ident_v)], return_counts=True)
    leftovers = [i for i in ids[np.argsort(counts)[::-1]] if i not in kept_ids]
    min_count = min_minutes * 60 / dt

    def baseline(f):
        return np.quantile(f, 0.1)

    for lid in leftovers:
        lm = ident_v == lid
        if lm.sum() < min_count:
            continue
        lt, lf = t_det[lm], fund_v[lm]
        l_frames = idx_v[lm]
        costs = {}
        for k in kept_ids:
            km = ident_v == k
            if np.isin(l_frames, idx_v[km]).mean() > max_conflict:
                continue
            kt, kf = t_det[km], fund_v[km]
            sides = []
            before = kt < lt[0]
            if before.any():
                tb = kt[before][-1]
                gap = lt[0] - tb
                if gap <= max_gap_minutes * 60:
                    ref = baseline(kf[before & (kt >= tb - baseline_window)])
                    own = baseline(lf[lt <= lt[0] + 2 * baseline_window])
                    sides.append(
                        abs(own - ref) / (freq_tol + drift_per_minute * gap / 60)
                    )
            after = kt > lt[-1]
            if after.any():
                ta = kt[after][0]
                gap = ta - lt[-1]
                if gap <= max_gap_minutes * 60:
                    ref = baseline(kf[after & (kt <= ta + 2 * baseline_window)])
                    own = baseline(lf[lt >= lt[-1] - baseline_window])
                    sides.append(
                        abs(own - ref) / (freq_tol + drift_per_minute * gap / 60)
                    )
            if sides and max(sides) <= 1.0:
                costs[k] = max(sides)
        if not costs:
            continue
        ranked = sorted(costs.items(), key=lambda kv: kv[1])
        if len(ranked) > 1 and ranked[1][1] < 2 * ranked[0][1]:
            continue  # ambiguous
        ident_v[lm] = ranked[0][0]
    return ident_v


def _plot_tracks(title, times, idx_v, ident_v, fund_v, ids, labels=True):
    """Draw the tracks of the identities `ids` in a new figure."""
    _, ax = plt.subplots(figsize=(30 / 2.54, 18 / 2.54))
    for id in ids:
        m = ident_v == id
        if labels:
            ax.text(
                times[idx_v[m]][0], fund_v[m][0], f"{id:.0f}", ha="center", va="bottom"
            )
        ax.plot(times[idx_v[m]], fund_v[m], marker=".")
    ax.set_title(title)


def main(
    folder,
    n_fish=None,
    stride_minutes=None,
    overlap_frac=None,
    freq_tolerance=None,
    time_tolerance_minutes=None,
    density_threshold=None,
    config_path=None,
    progress=None,
):
    """
    Run cleanup on wavetracker output.

    Parameters
    ----------
    folder : str or Path
        Path to folder containing wavetracker output files
    n_fish : int, optional
        Number of fish to track (overrides config)
    stride_minutes : float, optional
        Sliding window size in minutes (overrides config)
    overlap_frac : float, optional
        Window overlap fraction (overrides config)
    freq_tolerance : float, optional
        Frequency tolerance in Hz (overrides config)
    time_tolerance_minutes : float, optional
        Time gap tolerance in minutes (overrides config)
    density_threshold : float, optional
        Minimum detection density (overrides config)
    config_path : str or Path, optional
        Path to config file
    progress : callable, optional
        ``progress(stage, fraction)`` is called as the cleanup runs, with a
        short description of the current step and the fraction of the whole
        run done (0 to 1, non-decreasing).  The window pass and joining
        overlapping tracks take nearly all the time; their pair loops report
        about 200 times each.
    """
    # Load configuration
    config = load_config(config_path, folder)

    # Override with command-line arguments if provided
    if n_fish is not None:
        config["n_fish"] = n_fish
    if stride_minutes is not None:
        config["stride_minutes"] = stride_minutes
    if overlap_frac is not None:
        config["overlap_frac"] = overlap_frac
    if freq_tolerance is not None:
        config["freq_tolerance"] = freq_tolerance
    if time_tolerance_minutes is not None:
        config["time_tolerance_minutes"] = time_tolerance_minutes
    if density_threshold is not None:
        config["density_threshold"] = density_threshold

    # Extract config values
    n_fish = config["n_fish"]
    stride_minutes = config["stride_minutes"]
    overlap_frac = config["overlap_frac"]
    freq_tolerance = config["freq_tolerance"]
    time_tolerance_minutes = config["time_tolerance_minutes"]
    density_threshold = config["density_threshold"]
    assign_leftover_tracks = bool(config["assign_leftovers"])

    print("\n" + "=" * 60)
    print("CLEANUP PARAMETERS")
    print("=" * 60)
    print(f"  Stride: {stride_minutes} min ({stride_minutes * 60} s)")
    print(f"  Overlap: {overlap_frac:.0%}")
    print(f"  Frequency tolerance: {freq_tolerance} Hz")
    print(f"  Time gap tolerance: {time_tolerance_minutes} min")
    print(f"  Density threshold: {density_threshold:.2%}")
    print(f"  Target fish count: {n_fish}")
    print(f"  Assign leftover tracks: {assign_leftover_tracks}")
    print("=" * 60 + "\n")

    # Load data
    fund_v = np.load(os.path.join(folder, "fund_v.npy"))
    idx_v = np.load(os.path.join(folder, "idx_v.npy"))
    ident_v = np.load(os.path.join(folder, "ident_v.npy"))
    sign_v = np.load(os.path.join(folder, "sign_v.npy"))
    if np.nanmin(sign_v) > 0:
        sign_v = decibel(sign_v)
    loaded_ident_v = ident_v.copy()
    times = np.load(os.path.join(folder, "times.npy"))

    # parameters
    valid_v = np.zeros_like(ident_v)
    stride = stride_minutes * 60
    overlap = overlap_frac
    f_th = freq_tolerance
    kde_th = None
    previous_valid_ids = np.array([])

    # share of the run each step takes (measured on a dense field recording:
    # the window pass and the overlap joining are ~50 % each, the rest ~2 %)
    w_windows, w_density, w_overlap = 0.5, 0.01, 0.47

    def report(stage, fraction):
        if progress is not None:
            progress(stage, float(min(max(fraction, 0.0), 1.0)))

    starts = np.arange(0, times[-1], int(stride * (1 - overlap)))
    n_windows = max(len(starts), 1)
    for k, i0 in enumerate(starts):

        def tick(done, total, k=k):
            inner = done / total if total else 1.0
            report(
                f"window {k + 1}/{n_windows}",
                w_windows * (k + 0.5 + 0.5 * inner) / n_windows,
            )

        report(f"window {k + 1}/{n_windows}", w_windows * k / n_windows)
        kde_th, valid_ids = get_valid_ids_by_freq_dist(
            times,
            idx_v,
            ident_v,
            fund_v,
            valid_v,
            previous_valid_ids,
            i0,
            stride,
            f_th,
            kde_th,
        )

        previous_valid_ids, ident_v = connect_by_similarity(
            times,
            idx_v,
            ident_v,
            fund_v,
            sign_v,
            valid_v,
            valid_ids,
            f_th,
            i0,
            stride,
            tick=tick,
        )

    report("density filter", w_windows)
    if show_results:
        _plot_tracks(
            "after snippet connections",
            times,
            idx_v,
            ident_v,
            fund_v,
            np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)]),
        )
        plt.show()

    if not np.any(valid_v == 1):
        raise ValueError(
            "cleanup: no identity passed the frequency-density selection, so "
            "there is nothing to clean up. The selection assumes a few fish "
            "with well separated, persistent frequencies; it does not suit "
            "dense populations or short track segments (e.g. moving "
            "electrodes). Check --stride/--freq-tol, or use merge-by-position."
        )
    valid_v = power_density_filter(
        valid_v, sign_v, ident_v, idx_v, times, density_th=density_threshold
    )

    if show_results:
        _plot_tracks(
            "after power_density filter",
            times,
            idx_v,
            ident_v,
            fund_v,
            np.unique(ident_v[(~np.isnan(ident_v)) & (valid_v == 1)]),
        )

    base = w_windows + w_density
    report("joining overlapping tracks", base)
    ident_v = connect_with_overlap(
        fund_v,
        ident_v,
        valid_v,
        idx_v,
        times,
        time_tol=time_tolerance_minutes * 60,
        freq_tol=freq_tolerance,
        tick=lambda done, total: report(
            "joining overlapping tracks",
            base + w_overlap * (done / total if total else 1.0),
        ),
    )
    report(f"keeping the {n_fish} best fish", base + w_overlap)

    # Keep the n_fish identities with the most detections.
    idents, counts = np.unique(ident_v[~np.isnan(ident_v)], return_counts=True)
    valid_idents = idents[np.argsort(counts)][-n_fish:]
    # Leftover tracks that continue one of the kept fish are assigned to it
    # instead of being discarded (e.g. a fish split into two large tracks).
    if assign_leftover_tracks:
        ident_v = assign_leftovers(
            fund_v, idx_v, ident_v, times, valid_idents, freq_tol=freq_tolerance
        )

    # set all values in ident_v that are not in valid_idents to nan
    ident_v[~np.isin(ident_v, valid_idents)] = np.nan

    if show_results:
        ids = np.unique(ident_v[~np.isnan(ident_v)])
        _plot_tracks("after global connect", times, idx_v, ident_v, fund_v, ids)
        for id in ids:
            i = idx_v[ident_v == id]
            density = len(i) / (i[-1] - i[0] + 1)
            mean_power = np.mean(np.max(sign_v[ident_v == id], axis=1))
            print(f"{id}; density: {density:.3f}; power: {mean_power:.2f}dB")
        _plot_tracks(
            "original",
            times,
            idx_v,
            loaded_ident_v,
            fund_v,
            np.unique(loaded_ident_v[~np.isnan(loaded_ident_v)]),
            labels=False,
        )
        plt.show()

    # save data
    report("saving", 0.99)
    np.save(os.path.join(folder, f"ident_v_cleaned_n{n_fish}.npy"), ident_v)
    np.save(os.path.join(folder, f"idx_v_cleaned_n{n_fish}.npy"), idx_v)
    np.save(os.path.join(folder, f"fund_v_cleaned_n{n_fish}.npy"), fund_v)
    report("done", 1.0)
