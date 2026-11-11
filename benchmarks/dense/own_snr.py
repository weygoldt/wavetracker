"""Each fish's own SNR, also where another fish shares its spectral peak.

The truth's ``snr_db`` is the mixture's power at a fish's bin (the maximum
over +-1 bin of 0.61 Hz) over the noise, so where another line lies within
about 1.2 Hz it holds the power of both fish, and a faint fish next to a
strong one reads as strong. Close-pair strata by that SNR are mislabeled.

Here a fish's own power is taken from the samples where it is *clear*: no
fundamental or harmonic (1-5) of another fish within ``CLEAR_HZ``. In
between, its power is interpolated in time after removing the gain the
mixture applied to it (``gain_db`` plus its electrode gains, both known
exactly), and that gain is put back: an attenuated fish stays attenuated in
the pair. The own SNR is that power over the variant's noise power at the
fish's bin (``noise_power``, so lab and field variants each get theirs).
On clear samples it equals the truth's SNR.

The result is a sidecar ``truth_own_snr.npz`` next to each ``truth.npz``
(which is left as it is), on the truth's ``snr_times``:

* ``own_snr_db`` (fish, T): the own SNR [dB];
* ``clear`` (fish, T): the sample is clear, ``own_snr_db == snr_db``;
* ``gap_s`` (fish, T): time to the nearest clear sample of the fish [s]
  (inf if it is never clear; ``own_snr_db`` is NaN then).

``check`` hides random blocks of clear samples and predicts them, which
gives the error of the interpolation by gap length.

    python benchmarks/dense/own_snr.py write --splits ldev ldev-field ldev2 ldev2-field
    python benchmarks/dense/own_snr.py check --splits ldev ldev-field
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA

MIXTURES = DATA / "mixtures"
SIDECAR = "truth_own_snr.npz"
CLEAR_HZ = 3.0
MAX_H = 5


def other_line_distance(F: np.ndarray) -> np.ndarray:
    """Distance [Hz] of every fish (fish, T) to the nearest fundamental or
    harmonic (1..MAX_H) of any other fish."""
    d = np.full(F.shape, np.inf)
    for k in range(len(F)):
        for j in range(len(F)):
            if j != k:
                for h in range(1, MAX_H + 1):
                    d[k] = np.fmin(d[k], np.abs(h * F[j] - F[k]))
    return d


def effective_gain_db(truth: dict, t: np.ndarray) -> np.ndarray:
    """Gain [dB] of every fish's electrode-summed power at times t: its
    common gain plus the mean (in power) of its electrode gains."""
    tt = truth["times"]
    chan = truth["channel_gain_db"].astype(np.float64)  # (fish, electrodes, T)
    g = truth["gain_db"][:, None, :] + chan
    eff = 10 * np.log10(np.mean(10 ** (g / 10), axis=1))
    return np.stack([np.interp(t, tt, e) for e in eff])


def _interp_from(t, x, ok):
    """x interpolated in time from the samples `ok` (held at the ends) and
    the time to the nearest such sample; NaN / inf without any."""
    if ok.sum() == 0:
        return np.full(len(t), np.nan), np.full(len(t), np.inf)
    ti = t[ok]
    v = np.interp(t, ti, x[ok])
    pos = np.clip(np.searchsorted(ti, t), 1, max(len(ti) - 1, 1))
    if len(ti) == 1:
        gap = np.abs(t - ti[0])
    else:
        gap = np.minimum(np.abs(ti[pos] - t), np.abs(ti[pos - 1] - t))
    return v, gap


def compute(truth: dict) -> dict:
    """The sidecar's arrays for one mixture variant."""
    t = truth["snr_times"]
    F = np.stack([np.interp(t, truth["times"], f) for f in truth["freqs_filled"]])
    d = other_line_distance(F)
    fp = truth["fish_power"].astype(np.float64)
    nb = truth["noise_power"].astype(np.float64)
    g = effective_gain_db(truth, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 10 * np.log10(fp) - g  # the fish's power without the mixture's gain
    clear = (d >= CLEAR_HZ) & np.isfinite(F) & np.isfinite(x)
    own = np.full(F.shape, np.nan)
    gap = np.full(F.shape, np.inf)
    for k in range(len(F)):
        v, gk = _interp_from(t, x[k], clear[k])
        own_p = 10 ** ((v + g[k]) / 10)
        with np.errstate(divide="ignore", invalid="ignore"):
            own[k] = 10 * np.log10(own_p / nb[k])
        gap[k] = gk
    own[clear] = truth["snr_db"][clear]  # exactly the truth where clear
    own[~np.isfinite(F)] = np.nan
    return {
        "snr_times": t,
        "own_snr_db": own.astype(np.float32),
        "clear": clear,
        "gap_s": gap.astype(np.float32),
    }


def load(mix_dir: Path, write: bool = True) -> dict:
    """The sidecar of a mixture variant, computed (and written) if missing."""
    path = Path(mix_dir) / SIDECAR
    if path.exists():
        return dict(np.load(path))
    out = compute(dict(np.load(Path(mix_dir) / "truth.npz")))
    if write:
        np.savez_compressed(path, **out)
    return out


def own_snr_at(mix_dir: Path, times: np.ndarray) -> np.ndarray:
    """Own SNR (fish, frames) at result frame times, as score.py reads the
    truth's SNR (linear interpolation, NaN outside)."""
    s = load(mix_dir)
    return np.stack(
        [
            np.interp(times, s["snr_times"], v, left=np.nan, right=np.nan)
            for v in s["own_snr_db"]
        ]
    )


def variants(splits) -> list[Path]:
    out = []
    for sp in splits:
        out += sorted(p.parent for p in (MIXTURES / sp).glob("*/band.yaml"))
    return out


def write(args):
    for mdir in variants(args.splits):
        path = mdir / SIDECAR
        if path.exists() and not args.overwrite:
            continue
        out = compute(dict(np.load(mdir / "truth.npz")))
        np.savez_compressed(path, **out)
        truth = np.load(mdir / "truth.npz")
        known = np.isfinite(out["own_snr_db"])
        lift = (truth["snr_db"] - out["own_snr_db"])[known & ~out["clear"]]
        print(
            f"{mdir.relative_to(MIXTURES)}: clear {out['clear'][known].mean():.2f}, "
            f"truth SNR above own where not clear: median {np.nanmedian(lift):.1f} dB",
            flush=True,
        )


def check(args):
    """Hide blocks of clear samples of a fish, predict them from the rest."""
    rng = np.random.default_rng(0)
    rows = []
    for mdir in variants(args.splits):
        truth = dict(np.load(mdir / "truth.npz"))
        t = truth["snr_times"]
        clear_all = compute(truth)["clear"]
        g_all = effective_gain_db(truth, t)
        if args.no_gain:
            g_all = np.zeros_like(g_all)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_all = 10 * np.log10(truth["fish_power"].astype(np.float64)) - g_all
        for block in args.blocks:
            n = max(1, round(block / (t[1] - t[0])))
            for k in range(len(clear_all)):
                clear = clear_all[k]
                if clear.sum() < 4 * n:
                    continue
                for _ in range(args.draws):
                    i0 = rng.integers(0, len(t) - n)
                    hide = np.zeros(len(t), bool)
                    hide[i0 : i0 + n] = True
                    test = hide & clear
                    if not test.any() or not (clear & ~hide).any():
                        continue
                    v, _ = _interp_from(t, x_all[k], clear & ~hide)
                    pred = 10 * np.log10(
                        10 ** ((v + g_all[k]) / 10) / truth["noise_power"][k]
                    )
                    err = pred[test] - truth["snr_db"][k][test]
                    rows.append((block, np.median(np.abs(err)), np.median(err)))
    rows = np.array(rows)
    gain = "without" if args.no_gain else "with"
    print(f"{gain} the mixture's gain removed before interpolating")
    print("hidden block [s]  median |error| [dB]  median error [dB]  blocks")
    for block in args.blocks:
        r = rows[rows[:, 0] == block]
        print(
            f"{block:16.0f}  {np.median(r[:, 1]):19.2f}  {np.median(r[:, 2]):17.2f}  "
            f"{len(r)}"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--splits", nargs="+", default=["ldev", "ldev-field"])
    w.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--splits", nargs="+", default=["ldev"])
    c.add_argument("--blocks", nargs="+", type=float, default=[5, 15, 30, 60])
    c.add_argument("--draws", type=int, default=3)
    c.add_argument("--no-gain", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "write":
        write(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
