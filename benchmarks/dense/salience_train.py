"""Train the learned (salience) detector on the `ltrain` mixtures.

Every training example is a fresh crop of a training mixture (lab or field
noise), turned into a spectrogram on the GPU exactly as wavetracker does it
(so beats and leakage between close fish are as in a recording), summed over
a random subset of electrodes. Targets and loss weights come from the
mixture's truth:

* positives: a Gaussian bump (sigma 0.6 Hz) at every truth fundamental,
  merged by maximum;
* ignored (weight 0): fish frames below ``--snr-ignore`` dB (the fish was
  attenuated below the noise: neither fish nor no-fish), stretches where the
  truth does not know the fish, and persistent lines the truth does not know
  (found once per mixture on its lab variant, away from every truth
  fundamental and harmonic, see ``unknown_lines``);
* half weight: frames the truth completed by ridge following;
* everything else, harmonics of truth fish included, is a negative.

The loss is binary cross-entropy on the salience plus an L1 loss on the
sub-bin offset at the truth bins. Each example uses a random window of
candidate frequencies (the network is the same at every frequency), with the
harmonic stack gathered from the whole spectrum.

Usage::

    python benchmarks/dense/salience_train.py --tag v1 --steps 6000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, MIX_CHANNELS, RATE

from wavetracker.salience import (
    SalienceNet,
    SalienceSpec,
    decode,
    harmonic_stack,
    normalise,
    save_model,
)
from wavetracker.spectrogram import PowerSpectrogram, decibel

NFFT = 2**15
STEP = int(NFFT * 0.1)  # overlap_frac 0.9, as wavetracker's default
DF = RATE / NFFT
FT = STEP / RATE  # frame step [s]
SIGMA_HZ = 0.6
F_MIN, F_MAX = 40.0, 1500.0  # candidate fundamentals seen in training [Hz]
LINE_SMOOTH_S = 10.0  # unknown lines: mean feature over this many seconds
LINE_Z = 2.0  # ... above this
LINE_CLEAR_HZ = 2.5  # unknown lines farther than this from truth (harmonics)
LINE_HARMONICS = 12
MIXTURES = DATA / "mixtures"
CACHE = DATA / "salience" / "cache"
OUT = DATA / "salience"


def mixture_dirs(split: str) -> list[Path]:
    return sorted(p for p in (MIXTURES / split).glob("*") if (p / "band.yaml").exists())


class Mixture:
    """Raw traces and truth of one lab mixture and its field-noise variant
    (lab traces plus scaled field noise), resampled to spectrogram frames.
    ``traces(j0, n, g)`` mixes in the field noise with any gain g (0: lab,
    1: the field variant) and ``snr_at(g)`` gives the matching SNR."""

    def __init__(self, path: Path):
        self.path = path
        n = (path / "traces-grid1.raw").stat().st_size // 4 // MIX_CHANNELS
        self.raw = np.memmap(
            path / "traces-grid1.raw",
            dtype=np.float32,
            mode="r",
            shape=(n, MIX_CHANNELS),
        )
        fpath = path.parent.parent / (path.parent.name + "-field") / path.name
        self.field = None
        if (fpath / "band.yaml").exists():
            self.field = np.memmap(
                fpath / "traces-grid1.raw",
                dtype=np.float32,
                mode="r",
                shape=(n, MIX_CHANNELS),
            )
            zf = np.load(fpath / "truth.npz")
        self.n_frames = (n - NFFT) // STEP + 1
        t = (np.arange(self.n_frames) * STEP + NFFT / 2) / RATE
        z = np.load(path / "truth.npz")
        self.freqs = np.stack(
            [np.interp(t, z["times"], f, left=np.nan, right=np.nan) for f in z["freqs"]]
        )
        # interpolation across NaN gaps gives NaN; keep where the truth knows the fish
        known = np.stack(
            [
                np.interp(t, z["times"], np.isfinite(f).astype(float)) > 0.999
                for f in z["freqs"]
            ]
        )
        self.freqs[~known] = np.nan
        # where the fish is unknown, still keep a guess of its frequency to ignore it
        self.guess = np.stack([_fill(t, z["times"], f) for f in z["freqs"]])
        det = z["freqs_detected"] if "freqs_detected" in z.files else z["freqs"]
        detk = np.stack(
            [
                np.interp(t, z["times"], np.isfinite(f).astype(float)) > 0.999
                for f in det
            ]
        )
        self.completed = known & ~detk
        st = z["snr_times"]

        def frames(a):
            return np.stack([np.interp(t, st, np.nan_to_num(r, nan=0.0)) for r in a])

        self.fish_power = frames(z["fish_power"])
        self.noise_lab = frames(z["noise_power"])
        self.noise_field = None if self.field is None else frames(zf["noise_power"])
        self.snr = self.snr_at(0.0)

    def snr_at(self, g: float) -> np.ndarray:
        """SNR [dB] per fish and frame with the field noise at gain g."""
        noise = self.noise_lab
        if g and self.noise_field is not None:
            noise = noise + g**2 * (self.noise_field - self.noise_lab)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 10 * np.log10(self.fish_power / noise)
        return np.nan_to_num(snr, nan=-99.0, neginf=-99.0)

    def traces(self, j0: int, n_frames: int, g: float) -> np.ndarray:
        a, b = j0 * STEP, (j0 + n_frames - 1) * STEP + NFFT
        x = np.asarray(self.raw[a:b])
        if g and self.field is not None:
            x = x + np.float32(g) * (np.asarray(self.field[a:b]) - x)
        return x

    def lab_name(self) -> str:
        split, name = self.path.parent.name, self.path.name
        return f"{split.removesuffix('-field')}/{name}"


def _fill(t, tt, f):
    ok = np.isfinite(f)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    return np.interp(t, tt[ok], f[ok])


def spectrogram(spec: PowerSpectrogram, x: np.ndarray, electrodes, dev) -> torch.Tensor:
    """Raw (samples, channels) -> summed log power (bins, frames) [dB]."""
    tr = torch.from_numpy(np.ascontiguousarray(x[:, electrodes].T)).to(dev)
    return decibel(spec(tr).sum(0))


def unknown_lines(mix: Mixture, spec: PowerSpectrogram, dev) -> np.ndarray:
    """(frames, bins up to F_MAX) bool: persistent lines in the lab variant of
    the mixture that are no truth fundamental or harmonic. Cached per mixture."""
    lab = mix.lab_name()
    cache = CACHE / (lab.replace("/", "__") + ".npz")
    kmax = int(F_MAX / DF) + 1
    if cache.exists():
        z = np.load(cache)
        return np.unpackbits(z["bits"], axis=1, count=kmax).astype(bool)
    lab_mix = (
        mix if mix.path.parent.name == lab.split("/")[0] else Mixture(MIXTURES / lab)
    )
    rows = []
    block = 400
    for f0 in range(0, lab_mix.n_frames, block):
        f1 = min(lab_mix.n_frames, f0 + block)
        x = lab_mix.raw[f0 * STEP : (f1 - 1) * STEP + NFFT]
        log = spectrogram(spec, x, list(range(MIX_CHANNELS)), dev)
        rows.append(normalise(log[:kmax].T, DF).cpu())
    zf = torch.cat(rows).numpy()  # (frames, kmax)
    n = max(1, round(LINE_SMOOTH_S / FT))
    cs = np.cumsum(np.vstack([np.zeros((1, kmax)), zf]), axis=0)
    i = np.arange(len(zf))
    a, b = np.clip(i - n // 2, 0, len(zf)), np.clip(i - n // 2 + n, 0, len(zf))
    smooth = (cs[b] - cs[a]) / (b - a)[:, None]
    line = smooth > LINE_Z
    fk = np.arange(kmax) * DF
    clear = int(np.ceil(LINE_CLEAR_HZ / DF))
    for g in lab_mix.guess:
        for h in range(1, LINE_HARMONICS + 1):
            k = np.round(h * g / DF)
            ok = np.isfinite(k) & (k < kmax + clear)
            for d in range(-clear, clear + 1):
                kk = (k[ok] + d).astype(int)
                m = (kk >= 0) & (kk < kmax)
                line[np.flatnonzero(ok)[m], kk[m]] = False
    line[:, fk < F_MIN / 2] = False
    line = line | np.roll(line, 1, 1) | np.roll(line, -1, 1)  # +-1 bin
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, bits=np.packbits(line, axis=1))
    return line


class Batches:
    """Training examples drawn from the mixtures of a split, each with field
    noise at a random gain (0: lab, 1: the field variant, up to 3)."""

    def __init__(
        self,
        split: str,
        frames: int,
        ctx: int,
        bins: int,
        snr_ignore: float,
        dev,
        seed=0,
        limit: int | None = None,
    ):
        self.mixes = [Mixture(p) for p in mixture_dirs(split)[:limit]]
        if not self.mixes:
            raise SystemExit(f"no mixtures in {MIXTURES / split}")
        self.frames, self.ctx, self.bins = frames, ctx, bins
        self.snr_ignore = snr_ignore
        self.dev = dev
        self.spec = PowerSpectrogram(NFFT, STEP, RATE, dev)
        self.rng = np.random.default_rng(seed)
        self.lines = {}
        for m in self.mixes:
            key = m.lab_name()
            if key not in self.lines:
                self.lines[key] = unknown_lines(m, self.spec, dev)

    def example(
        self,
        mix: Mixture | None = None,
        j0: int | None = None,
        k_lo: int | None = None,
        electrodes=None,
        gain: float | None = None,
    ):
        rng = self.rng
        mix = mix if mix is not None else self.mixes[rng.integers(len(self.mixes))]
        T = self.frames + 2 * self.ctx
        if j0 is None:
            j0 = int(rng.integers(0, mix.n_frames - T))
        if electrodes is None:
            n_el = (
                MIX_CHANNELS
                if rng.random() < 0.5
                else int(rng.integers(2, MIX_CHANNELS))
            )
            electrodes = sorted(rng.choice(MIX_CHANNELS, n_el, replace=False).tolist())
        if k_lo is None:
            k_lo = int(rng.integers(int(F_MIN / DF), int(F_MAX / DF) - self.bins))
        k_hi = k_lo + self.bins - 1
        if gain is None:
            r = rng.random()
            gain = 0.0 if r < 0.25 else 1.0 if r < 0.5 else float(rng.uniform(0, 3))
        x = mix.traces(j0, T, gain)
        snr_all = mix.snr_at(gain)
        log = spectrogram(self.spec, x, electrodes, self.dev)
        k_need = min(log.shape[0], 5 * k_hi + 4)
        z = normalise(log[:k_need].T, DF)
        feats = harmonic_stack(z, k_lo, k_hi, SPEC_HARMONICS)
        # targets on the candidate bins, for the frames the loss sees
        fr = slice(j0 + self.ctx, j0 + self.ctx + self.frames)
        fk = (np.arange(k_lo, k_hi + 1)) * DF
        target = np.zeros((self.frames, self.bins), np.float32)
        weight = np.ones((self.frames, self.bins), np.float32)
        off_t = np.zeros((self.frames, self.bins), np.float32)
        off_w = np.zeros((self.frames, self.bins), np.float32)
        half = int(np.ceil(1.5 / DF))
        for i in range(len(mix.freqs)):
            f = mix.freqs[i, fr]
            g = mix.guess[i, fr]
            snr = snr_all[i, fr]
            comp = mix.completed[i, fr]
            near = np.abs(fk[None, :] - np.where(np.isfinite(g), g, -1e9)[:, None])
            gauss = np.exp(-0.5 * ((fk[None, :] - f[:, None]) / SIGMA_HZ) ** 2)
            ok = np.isfinite(f) & (snr >= self.snr_ignore)
            target[ok] = np.maximum(target[ok], gauss[ok])
            around = near <= half * DF
            ignore = (~ok)[:, None] & around  # unknown or below the noise
            weight[ignore] = 0.0
            weight[(ok & comp)[:, None] & around & (weight > 0)] = 0.5
            k = np.round(f / DF) - k_lo
            m = ok & (k >= 0) & (k < self.bins)
            rows = np.flatnonzero(m)
            kk = k[m].astype(int)
            off_t[rows, kk] = (f[m] / DF - np.round(f[m] / DF)).astype(np.float32)
            off_w[rows, kk] = 1.0
        lines = self.lines[mix.lab_name()][fr, k_lo : k_hi + 1]
        weight[lines] = 0.0
        t = lambda a: torch.from_numpy(a).to(self.dev)  # noqa: E731
        return feats, t(target), t(weight), t(off_t), t(off_w)

    def batch(self, n: int):
        ex = [self.example() for _ in range(n)]
        return [torch.stack(c) for c in zip(*ex, strict=True)]


SPEC_HARMONICS = SalienceSpec(DF, FT).harmonics


def losses(model, batch, ctx):
    feats, target, weight, off_t, off_w = batch
    logits, offset = model(feats)
    sl = slice(ctx, ctx + target.shape[1])
    logits, offset = logits[:, sl].float(), offset[:, sl].float()
    bce = F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, reduction="sum"
    )
    bce = bce / weight.sum().clamp_min(1.0)
    l1 = ((offset - off_t).abs() * off_w).sum() / off_w.sum().clamp_min(1.0)
    return bce, l1


def frame_scores(model, data: Batches, examples, ctx, threshold=0.5, tol=1.2):
    """Recall per SNR band and precision of salience peaks on fixed crops."""
    hits = {"faint": [0, 0], "visible": [0, 0]}
    n_det = n_match = 0
    for mix, j0, k_lo, el, gain in examples:
        feats, *_ = data.example(mix, j0, k_lo, el, gain)
        with (
            torch.no_grad(),
            torch.autocast("cuda", torch.bfloat16, enabled=feats.is_cuda),
        ):
            logits, offset = model(feats[None])
        sl = slice(ctx, ctx + data.frames)
        fr, _, f, _ = decode(
            logits[0, sl].float(), offset[0, sl].float(), threshold, k_lo, DF
        )
        rows = slice(j0 + ctx, j0 + ctx + data.frames)
        tf, snr = mix.freqs[:, rows], mix.snr_at(gain)[:, rows]
        lo, hi = k_lo * DF + 1.5, (k_lo + data.bins - 1) * DF - 1.5
        for j in range(data.frames):
            d = f[fr == j]
            t = tf[:, j]
            inb = np.isfinite(t) & (t > lo) & (t < hi)
            used = np.zeros(len(d), bool)
            for i in np.flatnonzero(inb):
                band = (
                    "visible" if snr[i, j] >= 6 else "faint" if snr[i, j] >= 0 else None
                )
                hit = False
                if len(d):
                    dist = np.where(used, np.inf, np.abs(d - t[i]))
                    m = int(np.argmin(dist))
                    if dist[m] <= tol:
                        used[m], hit = True, True
                if band:
                    hits[band][0] += hit
                    hits[band][1] += 1
            dd = d[(d > lo) & (d < hi)]
            n_det += len(dd)
            tt = t[np.isfinite(t)]
            if len(dd) and len(tt):
                n_match += int(np.sum(np.abs(dd[:, None] - tt[None, :]).min(1) <= tol))
    return {
        "recall_visible": hits["visible"][0] / max(hits["visible"][1], 1),
        "recall_faint": hits["faint"][0] / max(hits["faint"][1], 1),
        "precision": n_match / max(n_det, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--frames", type=int, default=300, help="frames per crop seen by the loss"
    )
    p.add_argument("--bins", type=int, default=512, help="candidate bins per crop")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--snr-ignore", type=float, default=-3.0)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--train-split", default="ltrain")
    p.add_argument("--val-split", default="ldev")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--limit", type=int, help="use only the first N mixtures per split (tests)"
    )
    args = p.parse_args(argv)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    spec = SalienceSpec(freq_resolution=DF, frame_step=FT, channels=args.channels)
    model = SalienceNet(spec).to(dev)
    ctx = model.context_frames
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    train = Batches(
        args.train_split,
        args.frames,
        ctx,
        args.bins,
        args.snr_ignore,
        dev,
        1,
        args.limit,
    )
    val = Batches(
        args.val_split, args.frames, ctx, args.bins, args.snr_ignore, dev, 2, args.limit
    )
    vr = np.random.default_rng(3)
    T = args.frames + 2 * ctx
    val_examples = {0.0: [], 1.0: []}  # lab and field variant
    for m in val.mixes:
        for _ in range(2):
            j0 = int(vr.integers(0, m.n_frames - T))
            k_lo = int(vr.integers(int(150 / DF), int(1300 / DF) - args.bins))
            for g, ex in val_examples.items():
                ex.append((m, j0, k_lo, list(range(MIX_CHANNELS)), g))
    n_par = sum(p.numel() for p in model.parameters())
    print(
        f"{args.tag}: {n_par} parameters, context +-{ctx} frames ({ctx * FT:.1f} s), "
        f"{len(train.mixes)} train / {len(val.mixes)} val mixtures",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )
    log, best, t0 = [], -1.0, time.perf_counter()
    t_mark = t0
    for step in range(1, args.steps + 1):
        model.train()
        batch = train.batch(args.batch)
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            bce, l1 = losses(model, batch, ctx)
            loss = bce + 0.5 * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            now = time.perf_counter()
            print(  # speed over the last 50 steps, without validation
                f"step {step}: bce {bce.item():.4f} l1 {l1.item():.3f} "
                f"({(now - t_mark) / 50:.2f} s/step)",
                flush=True,
            )
            t_mark = now
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            sc = {}
            for g, name in ((0.0, "lab"), (1.0, "field")):
                r = frame_scores(model, val, val_examples[g], ctx)
                sc |= {f"{k}_{name}": v for k, v in r.items()}
            row = {
                "step": step,
                "bce": bce.item(),
                "l1": l1.item(),
                **sc,
                "seconds": time.perf_counter() - t0,
            }
            log.append(row)
            t_mark = time.perf_counter()
            print(json.dumps(row), flush=True)
            meta = {"tag": args.tag, "step": step, "args": vars(args), "val": sc}
            save_model(out / "last.pt", model, spec, meta)
            score = sum(sc.values())  # recall faint + visible + precision, lab + field
            if score > best:
                best = score
                save_model(out / "model.pt", model, spec, meta)
            (out / "log.json").write_text(json.dumps(log, indent=1))
    print(
        f"done in {(time.perf_counter() - t0) / 60:.1f} min; best model {out / 'model.pt'}"
    )


if __name__ == "__main__":
    main()
