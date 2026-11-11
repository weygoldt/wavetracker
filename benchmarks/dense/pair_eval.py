"""How well a pair head finds the second fish, on fixed crops of the dev
mixtures (lab and field noise, all electrodes), before any tracking.

At every salience peak (threshold ``--threshold``) without another peak
within 2 bins: is a second truth fish (>= -3 dB) within 1.2 Hz of the
peak's detection (a hidden pair), or not? Reports the AUC of the pair
probability, and per pair threshold the share of hidden pairs split (and
the second detection within 1.2 Hz of the second fish) against the share
of single-fish peaks split.

    python benchmarks/dense/pair_eval.py --models salience/B-pair-fz-s0/last.pt ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS
from pair_cues import wauc
from salience_train import DF, Batches, electrode_window

from wavetracker.salience import PAIR_SPACING, load_model

PAIR_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def evaluate(path: Path, n_crops: int, frames: int, bins: int, thr: float, dev) -> dict:
    model, spec, _ = load_model(path, dev)
    ctx = model.context_frames
    el_frames, el_start = electrode_window(model, frames)
    data = Batches(
        "ldev",
        frames,
        ctx,
        bins,
        -3.0,
        dev,
        2,
        harmonics=spec.harmonics,
        workers=0,
        el_frames=el_frames,
    )
    rng = np.random.default_rng(3)
    T = frames + 2 * ctx
    res = {}
    for gain, name in ((0.0, "lab"), (1.0, "field")):
        score, label, hit = [], [], []
        for m in data.mixes:
            for _ in range(n_crops):
                j0 = int(rng.integers(0, m.n_frames - T))
                k_lo = int(rng.integers(int(150 / DF), int(1300 / DF) - bins))
                ex = data.example(m, j0, k_lo, list(range(MIX_CHANNELS)), gain)
                with torch.no_grad():
                    out = model.outputs(
                        ex["feats"][None],
                        ex["el"][None] if "el" in ex else None,
                        ex["el_mask"][None] if "el" in ex else None,
                        el_start,
                    )
                sl = slice(ctx, ctx + frames)
                p = torch.sigmoid(out["logits"][0, sl].float())
                left = torch.nn.functional.pad(p, (1, 0), value=-1.0)[:, :-1]
                right = torch.nn.functional.pad(p, (0, 1), value=-1.0)[:, 1:]
                peak = (p >= thr) & (p >= left) & (p > right)
                r = PAIR_SPACING
                n_near = torch.nn.functional.avg_pool1d(
                    torch.nn.functional.pad(peak.float(), (r, r))[:, None], 2 * r + 1, 1
                )[:, 0] * (2 * r + 1)
                lone = peak & (n_near < 1.5)
                fr, kk = torch.nonzero(lone, as_tuple=True)
                if not len(fr):
                    continue
                own = (kk + k_lo).double() + out["offset"][0, sl][fr, kk].double()
                q = torch.sigmoid(out["pair_logits"][0, sl][fr, kk].float())
                po = out["pair_offset"][0, :, sl][:, fr, kk].double()
                o = out["offset"][0, sl][fr, kk].double()
                far = torch.where((po[0] - o).abs() >= (po[1] - o).abs(), po[0], po[1])
                second = ((kk + k_lo).double() + far) * DF
                f_own = own.cpu().numpy() * DF
                f2 = second.cpu().numpy()
                rows = j0 + ctx + fr.cpu().numpy()
                tf = m.freqs[:, rows]  # (fish, peaks)
                snr = m.snr_at(gain)[:, rows]
                ok = np.isfinite(tf) & (snr >= -3)
                d = np.where(ok, np.abs(tf - f_own[None]), np.inf)
                near = np.sort(d, 0)
                # the peak's fish is the nearest one (within 1.2 Hz), a
                # second fish within 1.2 Hz of the detection makes a pair
                has_fish = near[0] <= 1.2
                is_pair = has_fish & (near[1] <= 1.2)
                i2 = np.argsort(d, 0)[1]
                f_second = tf[i2, np.arange(tf.shape[1])]
                score.append(q.cpu().numpy()[has_fish])
                label.append(is_pair[has_fish])
                hit.append((np.abs(f2 - f_second) <= 1.2)[has_fish])
        score, label, hit = (np.concatenate(v) for v in (score, label, hit))
        r = {
            "auc": wauc(score[label], score[~label], np.ones((~label).sum())),
            "n_pair": int(label.sum()),
            "n_single": int((~label).sum()),
        }
        for t in PAIR_THRESHOLDS:
            r[f"pairs split @{t}"] = float((score[label] >= t).mean())
            r[f"second found @{t}"] = float(((score >= t) & hit)[label].mean())
            r[f"singles split @{t}"] = float((score[~label] >= t).mean())
        res[name] = r
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--crops", type=int, default=4, help="crops per mixture")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--bins", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    dev = torch.device(args.device)
    for m in args.models:
        path = Path(m) if Path(m).is_absolute() else DATA / m
        res = evaluate(path, args.crops, args.frames, args.bins, args.threshold, dev)
        for name, r in res.items():
            cells = " ".join(
                f"{t}: {r[f'second found @{t}']:.2f}/{r[f'singles split @{t}']:.3f}"
                for t in PAIR_THRESHOLDS
            )
            print(
                f"{m} {name}: AUC {r['auc']:.3f} ({r['n_pair']} pair / {r['n_single']} "
                f"single peaks); second found / singles split: {cells}",
                flush=True,
            )


if __name__ == "__main__":
    main()
