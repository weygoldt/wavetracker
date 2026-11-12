"""Why the learned detector misses the strong 308 Hz fish of the boat take.

A diagnosis, not an evaluation: the 2-electrode boat take DR0000_0209 has no
truth and is never used to tune anything (DR0000_0208 and its corrections are
never used at all). The fish is followed by a reference trace taken from the
spectrogram alone (a Viterbi path through 296-343.5 Hz), and every stage of
the pipeline is checked along it:

    python benchmarks/boatfish.py run        # detect (salience maps kept) + tracking stages
    python benchmarks/boatfish.py trace      # reference trace of the fish
    python benchmarks/boatfish.py stages     # per-stage counts along the trace
    python benchmarks/boatfish.py plot       # figures

Runs (tags ``C-bf-*`` in ``$WT_BENCHMARK_DATA/field_check/runs_extra/
boat_DR0000_0209/``): ``benchmarks/dense/configs/learned-rises.yaml`` with
the boat's field settings of ``field_check.py`` (no mains, no interference
removal, 80-2000 Hz, dB electrode patterns), the same with the packaged v1
detector, and the field check's classic run for reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = Path(
    os.environ.get("WT_BENCHMARK_DATA", "~/wrk/data/wavetracker_benchmark")
).expanduser()
FC = DATA / "field_check"
OUT = FC / "runs_extra" / "boat_DR0000_0209"
FIG = DATA / "figures" / "boatfish"
CLASSIC = FC / "runs" / "boat_DR0000_0209" / "classic"
BOAT = Path(
    os.environ.get(
        "WT_FIELD_BOAT",
        "/run/media/weygoldt/data1/efish/fielddata/iriri2026/experiments/"
        "wavefish_habitatmapping/wav/DR0000_0209.wav",
    )
)
if "0208" in BOAT.name:
    raise SystemExit("DR0000_0208 is not to be used")

# the fish band of the trace and the band whose maps are kept
TRACE_BAND = (296.0, 343.5)
KEEP_BAND = (40.0, 2000.0)
VISIBLE_DB = 6.0  # a trace frame counts if the line stands this far above the band
MATCH_HZ = 1.5  # a detection this close to the trace is on the fish
START = 25.0  # electrodes in the water from here (handling clutter before)

# the boat's field settings (field_check.py BASE["boat"] + _COMMON)
BOAT_SETTINGS = {
    # 20-s blocks: the GPU is shared (the salience does not depend on blocks)
    "spectrogram": {"block_duration": 20.0},
    "interference": {"enabled": False},
    "harmonic_groups": {
        "mains_freq": 0.0,
        "min_group_size": 2,
        "max_harmonics": 10,
        "min_good_peak_power": None,
        "max_missing_harmonics": 0,
        "min_freq": 80.0,
        "max_freq": 2000.0,
        "low_thresh_factor": 3.0,
        "high_thresh_factor": 5.0,
    },
    "tracking": {"amplitude_feature": "db"},
}


def _merge(base: dict, extra: dict) -> dict:
    out = json.loads(json.dumps(base))
    for sec, vals in extra.items():
        out.setdefault(sec, {}).update(vals)
    return out


def variants() -> dict[str, dict]:
    import yaml

    rises = yaml.safe_load((HERE / "dense/configs/learned-rises.yaml").read_text())
    rises = _merge(rises, BOAT_SETTINGS)
    v1 = json.loads(json.dumps(rises))
    v1["detector"] = {
        "method": "learned",
        "threshold": 0.6,
        "model": str(DATA / "salience/v1/last.pt"),
    }
    # learned-rises with the detector fine-tuned with the noise shelf
    shelf = json.loads(json.dumps(rises))
    shelf["detector"]["model"] = str(DATA / "salience/C-bf-shelf-s0/last.pt")
    return {"C-bf-rises": rises, "C-bf-v1": v1, "C-bf-shelf": shelf}


def config(tag: str, overrides: dict | None = None):
    from wavetracker.config import Config

    return Config.from_dict(_merge(variants()[tag], overrides or {}))


# ------------------------------------------------------------------ capture


class Capture:
    """Keeps the salience maps (frames, candidate bins) of every block, the
    normalised features (frames, bins) and the electrode power around the
    fish, by wrapping `StreamingSalience`."""

    def __init__(self, keep_features: bool):
        self.keep_features = keep_features
        self.sal: dict[int, np.ndarray] = {}
        self.z: list[np.ndarray] = []
        self.log_power: list[np.ndarray] = []
        self.power: dict[int, np.ndarray] = {}
        self.k_lo = None

    def install(self):
        import torch

        from wavetracker import salience as S

        cap = self
        orig_features = S.StreamingSalience.features
        orig_pop = S.StreamingSalience._pop

        def features(self, log_power, stft=None):
            z = orig_features(self, log_power, stft)
            if cap.keep_features:
                zz = z[0] if isinstance(z, tuple) else z
                cap.z.append(zz.half().cpu().numpy())
                k = zz.shape[1]
                cap.log_power.append(log_power[:k].T.float().cpu().numpy())
            return z

        @torch.no_grad()
        def run(self, left, z, right):
            parts = [p for p in (left, z, right) if p is not None and S._frames(p)]
            x = S._cat(parts)
            n_left = 0 if left is None else S._frames(left)
            pad_l = self.ctx - n_left
            pad_r = self.ctx - (0 if right is None else S._frames(right))
            x = S._mirror(x, pad_l, pad_r)
            n_left += max(pad_l, 0)
            stack = S.harmonic_stack(x, self.k_lo, self.k_hi, self.spec.harmonics)
            out = self.model.outputs(stack[None])
            sl = slice(n_left, n_left + S._frames(z))
            cap._last = torch.sigmoid(out["logits"][0, sl].float()).cpu().numpy()
            cap.k_lo = self.k_lo
            return S.decode(
                out["logits"][0, sl],
                out["offset"][0, sl],
                self.threshold,
                self.k_lo,
                self.df,
            )

        def pop(self):
            det, payload = orig_pop(self)
            first = payload[0]
            cap.sal[first] = cap._last.astype(np.float16)
            power = payload[2]  # (electrodes, bins, frames)
            k0, k1 = round(280 / self.df), round(400 / self.df)
            cap.power[first] = power[:, k0:k1].cpu().numpy()
            return det, payload

        S.StreamingSalience.features = features
        S.StreamingSalience._run = run
        S.StreamingSalience._pop = pop

    def save(self, out: Path, df: float):
        firsts = sorted(self.sal)
        np.save(out / "sal_map.npy", np.concatenate([self.sal[f] for f in firsts]))
        np.save(
            out / "power_band.npy",
            np.concatenate([self.power[f] for f in firsts], axis=-1),
        )
        meta = {"k_lo": int(self.k_lo), "df": df, "power_k0": round(280 / df)}
        (out / "maps.json").write_text(json.dumps(meta))
        if self.z:
            np.save(out / "features.npy", np.concatenate(self.z))
            np.save(out / "log_power.npy", np.concatenate(self.log_power))


def run_stages(res, cfg, out: Path):
    """Tracking stage by stage, the identities after each saved."""
    from wavetracker.linking import link
    from wavetracker.rises import join_rises
    from wavetracker.tracking import drop_weak_identities, track

    tc = cfg.tracking
    stages = {}
    ident = track(
        res.fund_v,
        res.idx_v,
        res.sign_v,
        res.times,
        tc,
        min_freq=cfg.harmonic_groups.min_freq,
        max_freq=cfg.harmonic_groups.max_freq,
        sal_v=res.sal_v,
    )
    stages["track"] = ident.copy()
    if cfg.linking.enabled:
        ident = link(
            res.fund_v,
            res.idx_v,
            res.sign_v,
            res.cplx_v,
            ident,
            res.times,
            cfg.linking,
        )
        stages["link"] = ident.copy()
    if cfg.rises.enabled:
        ident = join_rises(
            res.fund_v,
            res.idx_v,
            res.sign_v,
            ident,
            res.times,
            cfg.rises,
            sal_v=res.sal_v,
        )
        stages["rises"] = ident.copy()
    if tc.min_identity_salience is not None:
        ident = drop_weak_identities(ident, res.sal_v, tc.min_identity_salience)
        stages["drop"] = ident.copy()
    np.savez(out / "stages.npz", **stages)
    res.ident_v = ident
    return stages


def cmd_run(args):
    from wavetracker.pipeline import detect

    for tag in args.tags:
        out = OUT / tag
        out.mkdir(parents=True, exist_ok=True)
        cfg = config(tag)
        if not (out / "fund_v.npy").exists() or args.force:
            cap = Capture(keep_features=(tag == "C-bf-rises"))
            cap.install()
            res = detect(BOAT, out, cfg, device=args.device).results
            cap.save(out, float(res.meta["freq_resolution"]))
        from wavetracker.results import Results

        res = Results.load(out)
        run_stages(res, cfg, out)
        res.save(out)
        cfg.save(out / "config.yaml")
        print(f"{tag}: {len(res.fund_v)} detections, {res.n_ids} ids", flush=True)


# ------------------------------------------------------------------- trace


def fine_spec():
    from wavetracker.results import load_fine_spectrogram

    spec, freqs, times = load_fine_spectrogram(CLASSIC)
    return np.asarray(spec, np.float64), freqs, times


def cmd_trace(args):
    """Viterbi path of the fish through TRACE_BAND: emission the dB above the
    band median (clipped), upward jumps cheap (rise onsets), downward ones
    dear (decays are slow)."""
    spec, f, t = fine_spec()
    db = 10 * np.log10(spec + 1e-30)
    j = (f >= TRACE_BAND[0]) & (f <= TRACE_BAND[1])
    fb, v = f[j], db[:, j]
    ex = np.clip(v - np.median(v, axis=1, keepdims=True), -5, 30)
    T, K = ex.shape
    D = fb[:, None] - fb[None, :]  # to - from
    P = np.where(D > 0, 0.4 * D, 2.5 * -D)
    cost = -ex[0]
    back = np.zeros((T, K), int)
    for i in range(1, T):
        trans = cost[None, :] + P
        back[i] = trans.argmin(1)
        cost = trans.min(1) - ex[i]
    path = np.zeros(T, int)
    path[-1] = cost.argmin()
    for i in range(T - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    freq = fb[path]
    excess = ex[np.arange(T), path]
    np.savez(OUT / "trace.npz", times=t, freq=freq, excess=excess)
    vis = (excess >= VISIBLE_DB) & (t >= START)
    print(
        f"trace: {freq[t >= START].min():.1f}-{freq[t >= START].max():.1f} Hz, "
        f"visible in {vis.sum()} of {(t >= START).sum()} frames after {START} s"
    )


def load_trace():
    d = np.load(OUT / "trace.npz")
    return d["times"], d["freq"], d["excess"]


# ------------------------------------------------------------------ stages


def on_fish(fund, idx, trace_f, tol=MATCH_HZ):
    """Detections within tol of the trace in their frame."""
    return np.abs(fund - trace_f[idx]) <= tol


def stage_counts(run: Path, trace_f, frames: np.ndarray) -> dict:
    """Per stage: share of the fish frames with a detection on the fish that
    has an identity, the identities holding the fish and the largest one's
    share and purity."""
    fund = np.load(run / "fund_v.npy")
    idx = np.load(run / "idx_v.npy")
    sal = np.load(run / "sal_v.npy") if (run / "sal_v.npy").exists() else None
    sel = np.isin(idx, frames)
    hit = sel & on_fish(fund, idx, trace_f)
    n = len(frames)
    out = {"frames": n, "detected": len(np.unique(idx[hit])) / n}
    if sal is not None:
        out["salience_on_fish_median"] = (
            float(np.median(sal[hit])) if hit.any() else None
        )
    stages = (
        dict(np.load(run / "stages.npz"))
        if (run / "stages.npz").exists()
        else {"final": np.load(run / "ident_v.npy")}
    )
    for name, ident in stages.items():
        ok = hit & ~np.isnan(ident)
        ids = np.unique(ident[ok])
        # frames per identity (a frame once per identity)
        fr = {i: len(np.unique(idx[ok & (ident == i)])) for i in ids}
        big = max(fr, key=fr.get) if fr else None
        s = {
            "with_identity": len(np.unique(idx[ok])) / n,
            "identities": int(sum(v >= 0.01 * n for v in fr.values())),
            "largest_share": (fr[big] / n) if big is not None else 0.0,
        }
        if big is not None:
            m = ident == big
            s["largest_purity"] = float(hit[m].mean())
            s["largest_mean_salience"] = (
                float(sal[m].mean()) if sal is not None else None
            )
        out[name] = s
    return out


def cmd_stages(args):
    t, f, ex = load_trace()
    frames = np.flatnonzero((ex >= VISIBLE_DB) & (t >= START))
    rows = {}
    for run in [CLASSIC, *(OUT / tag for tag in args.tags)]:
        name = "classic (field_check)" if run == CLASSIC else run.name
        rows[name] = stage_counts(run, f, frames)
    (OUT / "stages.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tags", nargs="+", default=["C-bf-rises", "C-bf-v1"])
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("trace")
    p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("stages")
    p.add_argument("--tags", nargs="+", default=["C-bf-rises", "C-bf-v1"])
    p.set_defaults(fn=cmd_stages)
    p = sub.add_parser("ablate")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_ablate)
    p = sub.add_parser("plot")
    p.set_defaults(fn=cmd_plot)
    p = sub.add_parser("fishsal")
    p.add_argument("models", nargs="+")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_fishsal)
    p = sub.add_parser("shelfval")
    p.add_argument("models", nargs="+")
    p.add_argument("--crops", type=int, default=2)
    p.add_argument("--raise-db", type=float, nargs="+", default=[20.0])
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_shelfval)
    p = sub.add_parser("coverage")
    p.add_argument("--crops", type=int, default=3)
    p.add_argument("--frames", type=int, default=300)
    p.set_defaults(fn=cmd_coverage)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    args.fn(args)


# ---------------------------------------------------------------- ablation


MODELS = {
    "rises": DATA / "salience/B-det-tcn64aug-s0/last.pt",
    "v1": DATA / "salience/v1/last.pt",
}


def load(name: str, device):
    from wavetracker.salience import default_model_path, load_model

    return load_model(MODELS[name] or default_model_path(), device)[:2]


def salience_crop(model, spec, Z, k0, k1, device, edit=None):
    """Salience (frames, k0..k1) of the whole recording at once, with the
    recording's ends mirrored as `StreamingSalience` does. The network is
    convolutional along frequency (a few bins of reach), so a crop of
    candidate bins gives the same values away from its edges. `edit(stack)`
    may change the harmonic stack (channels, frames, bins) before the
    network; frames there include the mirrored context."""
    import torch

    from wavetracker.salience import _mirror, harmonic_stack

    ctx = model.context_frames
    with torch.no_grad():
        x = _mirror(torch.as_tensor(Z, dtype=torch.float32), ctx, ctx)
        stack = harmonic_stack(x, k0, k1, spec.harmonics)
        if edit is not None:
            stack = edit(stack, ctx)
        out = model.outputs(stack[None].to(device))
        T = len(Z)
        return torch.sigmoid(out["logits"][0, ctx : ctx + T]).cpu().numpy()


def along(p, trace_f, df, k0, reach=2):
    """Largest salience within `reach` bins of the trace, per frame."""
    kk = np.round(trace_f / df).astype(int) - k0
    return np.stack(
        [p[np.arange(len(p)), kk + o] for o in range(-reach, reach + 1)]
    ).max(0)


# ---------------------------------------------------------------- coverage


def fish_profile(z, freqs_fish, df, harmonics=(1, 2, 3, 4, 5)):
    """Normalised features (frames, bins) and fish frequencies (fish, frames)
    -> (fish, frames, harmonics) values of the harmonic stack at each fish
    (max over the radius the stack uses, ±1 bin at the fundamental)."""
    from wavetracker.salience import _radius

    out = np.full((*freqs_fish.shape, len(harmonics)), np.nan)
    T, K = z.shape
    for i, h in enumerate(harmonics):
        r = max(1, _radius(h))
        k = np.round(h * freqs_fish / df)
        ok = np.isfinite(k) & (k + r < K)
        kk = np.where(ok, k, 0).astype(int)
        rows = np.broadcast_to(np.arange(T), kk.shape)
        v = np.max([z[rows, np.clip(kk + o, 0, K - 1)] for o in range(-r, r + 1)], 0)
        out[..., i] = np.where(ok, v, np.nan)
    return out


def cmd_coverage(args):
    """Harmonic profiles of the `ltrain` fish (all 9 electrodes summed, a
    few crops per mixture, lab and field noise) against the boat fish's."""
    import torch

    sys.path.insert(0, str(HERE / "dense"))
    from salience_train import NFFT, STEP, Mixture, mixture_dirs

    from wavetracker.salience import normalise
    from wavetracker.spectrogram import PowerSpectrogram, decibel

    dev = torch.device("cpu")
    rate = 20000.0
    spec = PowerSpectrogram(NFFT, STEP, rate, dev)
    df = rate / NFFT
    rng = np.random.default_rng(0)
    rows = []
    for split in ("ltrain", "ltrain-field"):
        for path in mixture_dirs(split):
            mix = Mixture(path)
            g = 1.0 if split.endswith("field") else 0.0
            snr = mix.snr_at(g)
            for _ in range(args.crops):
                j0 = int(rng.integers(0, mix.n_frames - args.frames))
                x = mix.traces(j0, args.frames, g)
                tr = torch.from_numpy(np.ascontiguousarray(x.T))
                log = decibel(spec(tr).sum(0))
                z = normalise(log.T, df).numpy()
                ff = mix.freqs[:, j0 : j0 + args.frames]
                prof = fish_profile(z, ff, df)
                s = snr[:, j0 : j0 + args.frames]
                for k in range(len(ff)):
                    ok = np.isfinite(ff[k])
                    rows.append(
                        np.column_stack(
                            [
                                np.full(ok.sum(), split.endswith("field")),
                                ff[k, ok],
                                s[k, ok],
                                prof[k, ok],
                            ]
                        )
                    )
            print(path.name, split, flush=True)
    tab = np.concatenate(rows)
    np.save(OUT / "ltrain_profiles.npy", tab)
    print(tab.shape)


def _edit_sub(spec, mode):
    """Sub-harmonic channels (h < 1) set to 0 or to their median per frame."""

    def edit(st, ctx):
        st = st.clone()
        for i, h in enumerate(spec.harmonics):
            if h < 1:
                st[i] = (
                    0
                    if mode == "zero"
                    else st[i].median(dim=1, keepdim=True).values.expand_as(st[i])
                )
        return st

    return edit


def _edit_fish(spec, f, df, k0, change):
    """`change(stack, rows, cols, channel index)` at the fish's candidate bins
    (±1), on the frames with mirrored context."""
    import torch

    from wavetracker.salience import _mirror

    def edit(st, ctx):
        st = st.clone()
        tr = _mirror(torch.from_numpy(f[:, None]), ctx, ctx)[:, 0].numpy()
        kb = np.round(tr / df).astype(int) - k0
        rows = np.arange(st.shape[1])
        for o in (-1, 0, 1):
            change(st, rows, kb + o, list(spec.harmonics).index)
        return st

    return edit


def cmd_ablate(args):
    """Salience along the fish with the harmonic stack changed: what makes
    the network accept or reject it."""
    import torch

    from wavetracker.salience import normalise

    t, f, ex = load_trace()
    frames = np.flatnonzero((ex >= VISIBLE_DB) & (t >= START))
    run = OUT / "C-bf-rises"
    df = json.loads((run / "maps.json").read_text())["df"]
    Z = np.load(run / "features.npy").astype(np.float32)
    L = np.load(run / "log_power.npy")
    dev = torch.device(args.device)
    k0, k1 = round(250 / df), round(420 / df)

    def boost(st, r, c, ix):
        st[ix(1), r, c] *= 2.2

    def harm2(st, r, c, ix):
        st[ix(2), r, c] = torch.maximum(st[ix(2), r, c], st[ix(1), r, c] - 1)

    def harm23(st, r, c, ix):
        harm2(st, r, c, ix)
        st[ix(3), r, c] = torch.maximum(st[ix(3), r, c], st[ix(1), r, c] - 2)

    res = {}
    for name in ("rises", "v1"):
        model, spec = load(name, dev)
        runs = {
            "as recorded": (Z, None),
            "sub-harmonics = 0": (Z, _edit_sub(spec, "zero")),
            "sub-harmonics = median": (Z, _edit_sub(spec, "median")),
            "100-Hz bands -> 50 Hz": (
                normalise(torch.from_numpy(L), df, 50.0).numpy(),
                None,
            ),
            "100-Hz bands -> 25 Hz": (
                normalise(torch.from_numpy(L), df, 25.0).numpy(),
                None,
            ),
            "fundamental x 2.2": (Z, _edit_fish(spec, f, df, k0, boost)),
            "+ 2nd harmonic (f - 1)": (Z, _edit_fish(spec, f, df, k0, harm2)),
            "+ 2nd, 3rd (f - 1, f - 2)": (Z, _edit_fish(spec, f, df, k0, harm23)),
        }
        res[name] = {}
        for label, (z, edit) in runs.items():
            p = salience_crop(model, spec, z, k0, k1, dev, edit=edit)
            a = along(p, f, df, k0)
            res[name][label] = {
                "median": float(np.median(a[frames])),
                "frac_0.6": float((a[frames] >= 0.6).mean()),
            }
            if label == "as recorded" or label.startswith("+ 2nd, 3rd"):
                np.save(
                    OUT
                    / f"along_{name}_{'base' if label == 'as recorded' else 'harm'}.npy",
                    a,
                )
            print(name, label, res[name][label], flush=True)
    (OUT / "ablate.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ figures


def _spec_panel(ax, L, df, t, t0, t1, f0, f1, pct=(20, 99.7)):
    it = np.flatnonzero((t >= t0) & (t <= t1))
    k0, k1 = round(f0 / df), round(f1 / df)
    v = np.asarray(L[it[0] : it[-1] + 1, k0 : k1 + 1], np.float32)
    lo, hi = np.percentile(v, pct)
    ax.pcolormesh(
        t[it],
        np.arange(k0, k1 + 1) * df,
        v.T,
        vmin=lo,
        vmax=hi,
        cmap="gray_r",
        shading="nearest",
        rasterized=True,
    )
    ax.set_xlim(t0, t1)
    ax.set_ylim(f0, f1)


def cmd_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = OUT / "C-bf-rises"
    df = json.loads((run / "maps.json").read_text())["df"]
    L = np.load(run / "log_power.npy", mmap_mode="r")
    t, f, ex = load_trace()
    frames = np.flatnonzero((ex >= VISIBLE_DB) & (t >= START))
    blue, orange, green, grey = "#2a78d6", "#eb6834", "#1baf7a", "#888888"

    # --- figure 1: it is a fish ------------------------------------------
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    _spec_panel(ax, L, df, t, 0, t[-1], 250, 420)
    ax.set_title(
        "a  DR0000_0209, electrode sum, 250-420 Hz: the 307 Hz line from immersion "
        "(~24 s) to the end; the chorus starts at ~350 Hz",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("frequency [Hz]")
    ax = fig.add_subplot(gs[1, 0])
    _spec_panel(ax, L, df, t, 0, 40, 250, 420)
    ax.set_title(
        "b  handling clutter: a 400 Hz tone, no 307 Hz line", loc="left", fontsize=10
    )
    ax.set_ylabel("frequency [Hz]")
    ax.set_xlabel("time [s]")
    ax = fig.add_subplot(gs[1, 1])
    _spec_panel(ax, L, df, t, 284, 300, 295, 350, pct=(30, 99.9))
    ax.set_title("c  a rise: +34 Hz in 0.5 s, decay in ~3 s", loc="left", fontsize=10)
    ax.set_xlabel("time [s]")
    ax = fig.add_subplot(gs[1, 2])
    _spec_panel(ax, L, df, t, 284, 300, 590, 700, pct=(30, 99.9))
    ax.set_title(
        "d  the same at 2f (590-700 Hz): no harmonic visible", loc="left", fontsize=10
    )
    ax.set_xlabel("time [s]")
    ax = fig.add_subplot(gs[2, :2])
    sel = np.flatnonzero(t >= 40)
    P = np.median(np.asarray(L[sel, : round(2050 / df)]), 0)
    fk = np.arange(len(P)) * df
    ax.plot(fk, P, color=grey, lw=0.5)
    fm = float(np.median(f[frames]))
    for h in range(1, 7):
        k = round(h * fm / df)
        ax.axvline(
            h * fm, color=orange if h == 1 else blue, lw=0.8, ls="-" if h == 1 else ":"
        )
        ax.text(
            h * fm,
            P[k - 30 : k + 30].max() + 3,
            f"{h}f",
            color=blue,
            fontsize=8,
            ha="center",
        )
    ax.set_xlim(50, 2050)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("median power [dB]")
    ax.set_title(
        "e  median spectrum (40-349 s): the fish's harmonics (dotted) fall on a floor "
        "~20 dB above the floor at its fundamental",
        loc="left",
        fontsize=10,
    )
    ax = fig.add_subplot(gs[2, 2])
    pk0 = json.loads((run / "maps.json").read_text())["power_k0"]
    Pb = np.load(run / "power_band.npy")
    kk = np.round(f / df).astype(int) - pk0
    e = 10 * np.log10(
        np.stack(
            [
                Pb[:, np.clip(kk + o, 0, Pb.shape[1] - 1), np.arange(len(t))]
                for o in (-1, 0, 1)
            ]
        ).max(0)
    )
    ax.plot(t, e[0], color=blue, lw=0.5, label="electrode 1")
    ax.plot(t, e[1], color=orange, lw=0.5, label="electrode 2")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("power at the fish [dB]")
    ax.legend(fontsize=8)
    ax.set_title("f  level per electrode along the trace", loc="left", fontsize=10)
    p = FIG / "boatfish_is_fish.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)

    # --- figure 2: where and why it is lost ------------------------------
    stages = json.loads((OUT / "stages.json").read_text())
    abl = json.loads((OUT / "ablate.json").read_text())
    fig = plt.figure(figsize=(16, 15), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 0.8, 0.8, 1.1])
    ax = fig.add_subplot(gs[0, :])
    _spec_panel(ax, L, df, t, 0, t[-1], 296, 324, pct=(30, 99.8))
    for name, color, dy in [("C-bf-rises", orange, 0.0)]:
        r = OUT / name
        fund, idx = np.load(r / "fund_v.npy"), np.load(r / "idx_v.npy")
        m = (fund > 296) & (fund < 324)
        ax.plot(
            t[idx[m]],
            fund[m] + dy,
            ".",
            ms=1.5,
            color=color,
            label="learned-rises detections",
        )
    fund, idx = np.load(CLASSIC / "fund_v.npy"), np.load(CLASSIC / "idx_v.npy")
    m = (fund > 296) & (fund < 324)
    ax.plot(
        t[idx[m]],
        fund[m] - 4,
        ".",
        ms=1.0,
        color=blue,
        label="classic detections (drawn 4 Hz lower)",
    )
    ax.legend(loc="upper right", fontsize=8, markerscale=6)
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(
        "a  the fish (296-324 Hz) and the detections on it", loc="left", fontsize=10
    )
    ax = fig.add_subplot(gs[1, :])
    for name, color, lab in [
        ("rises", orange, "learned-rises (B-det-tcn64aug-s0)"),
        ("v1", green, "v1"),
    ]:
        a = np.load(OUT / f"along_{name}_base.npy")
        ax.plot(t, a, color=color, lw=0.7, label=lab)
    a = np.load(OUT / "along_rises_harm.npy")
    ax.plot(
        t,
        a,
        color=grey,
        lw=0.6,
        ls="--",
        label="learned-rises, synthetic 2nd + 3rd harmonic added",
    )
    shelf = OUT / "along_C-bf-shelf-s0.npy"
    if shelf.exists():
        ax.plot(
            t,
            np.load(shelf),
            color="#6250d6",
            lw=0.7,
            label="fine-tuned with noise shelf",
        )
    ax.axhline(0.6, color="k", lw=0.8, ls=":")
    ax.set_xlim(0, t[-1])
    ax.set_ylabel("salience at the fish")
    ax.legend(fontsize=8, loc="lower right", ncol=4)
    ax.set_title(
        "b  network salience along the fish (max within 2 bins of the trace); threshold 0.6",
        loc="left",
        fontsize=10,
    )
    ax = fig.add_subplot(gs[2, :])
    V = np.load(OUT / "trace_stack.npy")
    H = ["f/5", "f/4", "f/3", "f/2", "f", "2f", "3f", "4f", "5f"]
    ax.pcolormesh(
        t,
        np.arange(len(H)),
        V,
        vmin=-1,
        vmax=5,
        cmap="magma",
        shading="nearest",
        rasterized=True,
    )
    ax.set_yticks(np.arange(len(H)), H)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time [s]")
    ax.set_title(
        "c  harmonic stack along the fish (normalised level, -1 to 5): only the fundamental stands out",
        loc="left",
        fontsize=10,
    )
    ax = fig.add_subplot(gs[3, 0])
    names = ["classic (field_check)", "C-bf-v1", "C-bf-rises", "C-bf-shelf"]
    names = [n for n in names if n in stages]
    colors = [blue, green, orange, "#6250d6"][: len(names)]
    w = 0.8 / len(names)
    cols = ["detected", "track", "link", "rises", "drop"]
    x = np.arange(len(cols))
    for i, (n, c) in enumerate(zip(names, colors, strict=True)):
        s = stages[n]
        vals = [s["detected"]] + [
            s[k]["with_identity"]
            if k in s
            else s.get("final", {}).get("with_identity", np.nan)
            for k in cols[1:]
        ]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, color=c, label=n)
    ax.set_xticks(
        x, ["detected", "tracked", "linked", "rise\njoiner", "salience\ndrop"]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the fish's frames")
    ax.legend(fontsize=7)
    ax.set_title("d  per stage (classic: its own tracking)", loc="left", fontsize=10)
    ax = fig.add_subplot(gs[3, 1])
    labels = list(abl["rises"])
    y = np.arange(len(labels))
    ax.barh(
        y + 0.2,
        [abl["rises"][k]["frac_0.6"] for k in labels],
        0.4,
        color=orange,
        label="learned-rises",
    )
    ax.barh(
        y - 0.2,
        [abl["v1"][k]["frac_0.6"] for k in labels],
        0.4,
        color=green,
        label="v1",
    )
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of fish frames with salience >= 0.6")
    ax.legend(fontsize=7)
    ax.set_title("e  changing the input along the fish", loc="left", fontsize=10)
    ax = fig.add_subplot(gs[3, 2])
    tab = np.load(OUT / "ltrain_profiles.npy")
    zf, mx = tab[:, 3], np.nanmax(tab[:, 4:], 1)
    s = (zf >= 2.5) & (zf < 4.5)
    zb, mb = V[4, frames], V[5:, frames].max(0)
    bins = np.linspace(-1, 10, 45)
    ax.hist(
        np.clip(mx[s], -1, 10),
        bins,
        density=True,
        color=grey,
        alpha=0.7,
        label=f"ltrain fish, f at 2.5-4.5 ({s.sum()} frames)",
    )
    sb = (zb >= 2.5) & (zb < 4.5)
    ax.hist(
        np.clip(mb[sb], -1, 10),
        bins,
        density=True,
        histtype="step",
        color=orange,
        lw=1.5,
        label=f"boat fish, f at 2.5-4.5 ({sb.sum()} frames)",
    )
    ax.set_xlabel("strongest of 2f-5f channels (clipped at 10)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("f  harmonics of equally strong fish", loc="left", fontsize=10)
    p = FIG / "boatfish_cause.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)


def cmd_fishsal(args):
    """Salience along the fish for other models (the stored features)."""
    import torch

    from wavetracker.salience import load_model

    t, f, ex = load_trace()
    frames = np.flatnonzero((ex >= VISIBLE_DB) & (t >= START))
    run = OUT / "C-bf-rises"
    df = json.loads((run / "maps.json").read_text())["df"]
    Z = np.load(run / "features.npy").astype(np.float32)
    dev = torch.device(args.device)
    k0, k1 = round(250 / df), round(420 / df)
    res = (
        json.loads((OUT / "fishsal.json").read_text())
        if (OUT / "fishsal.json").exists()
        else {}
    )
    for path in args.models:
        model, spec = load_model(path, dev)[:2]
        if spec.band_hz != 100.0:
            raise SystemExit("the stored features are for 100-Hz bands")
        p = salience_crop(model, spec, Z, k0, k1, dev)
        a = along(p, f, df, k0)
        name = Path(path).parent.name
        np.save(OUT / f"along_{name}.npy", a)
        res[name] = {
            "median": float(np.median(a[frames])),
            "frac_0.6": float((a[frames] >= 0.6).mean()),
            "frac_0.6_per_50s": [
                float(
                    (a[frames][(t[frames] >= w) & (t[frames] < w + 50)] >= 0.6).mean()
                )
                for w in range(25, 350, 50)
            ],
        }
        print(name, res[name], flush=True)
    (OUT / "fishsal.json").write_text(json.dumps(res, indent=1))


def cmd_shelfval(args):
    """The boat fish's situation on held-out `ldev` mixtures: fixed crops
    whose candidate window (156 Hz from f_lo, 200-900 Hz) lies below a noise
    shelf (edge 20 Hz above the window, +raise dB), so every fundamental
    keeps its level and every harmonic is under the raised floor. Recall at
    threshold 0.6 of fish frames >= 6 dB (truth SNR) more than 15 Hz below
    the edge, and precision, with and without the shelf."""
    import torch

    sys.path.insert(0, str(HERE / "dense"))
    from salience_train import DF, MIX_CHANNELS, Batches, Crop

    from wavetracker.salience import decode, load_model

    dev = torch.device(args.device)
    rng = np.random.default_rng(7)
    res = {}
    for path in args.models:
        model, spec = load_model(path, dev)[:2]
        ctx = model.context_frames
        data = Batches(
            "ldev", 300, ctx, 256, -3.0, dev, 2, harmonics=spec.harmonics, workers=0
        )
        if not res:  # the same crops for every model
            crops = []
            for m in data.mixes:
                for _ in range(args.crops):
                    j0 = int(rng.integers(0, m.n_frames - 300 - 2 * ctx))
                    f_lo = float(rng.uniform(200, 900))
                    crops.append((m.path.name, j0, f_lo))
        name = Path(path).parent.name
        res[name] = {}
        by_name = {m.path.name: m for m in data.mixes}
        for raise_db in (0.0, *args.raise_db):
            for gain in (0.0, 1.0):
                hit = tot = n_det = n_ok = 0
                torch.manual_seed(0)
                for mname, j0, f_lo in crops:
                    mix = by_name[mname]
                    k_lo = round(f_lo / DF)
                    edge = (k_lo + 255) * DF + 20.0
                    crop = Crop(mix, j0, list(range(MIX_CHANNELS)), gain, [k_lo])
                    if raise_db:
                        crop.shelf = (edge, raise_db, 20.0)
                    ex = data.build(crop, data.read(crop))
                    with torch.no_grad():
                        out = model.outputs(ex["feats"][:1])
                    sl = slice(ctx, ctx + 300)
                    fr, _, fq, _ = decode(
                        out["logits"][0, sl].float(),
                        out["offset"][0, sl].float(),
                        0.6,
                        k_lo,
                        DF,
                    )
                    rows = slice(j0 + ctx, j0 + ctx + 300)
                    tf, snr = mix.freqs[:, rows], mix.snr_at(gain)[:, rows]
                    lo, hi = k_lo * DF + 1.5, min((k_lo + 255) * DF, edge - 15) - 1.5
                    for j in range(300):
                        d = fq[fr == j]
                        t = tf[:, j]
                        vis = np.isfinite(t) & (t > lo) & (t < hi) & (snr[:, j] >= 6)
                        tot += vis.sum()
                        if len(d):
                            hit += sum(np.abs(d - x).min() <= 1.2 for x in t[vis])
                            dd = d[(d > lo) & (d < hi)]
                            tt = t[np.isfinite(t)]
                            n_det += len(dd)
                            if len(tt):
                                n_ok += (
                                    np.abs(dd[:, None] - tt[None]).min(1) <= 1.2
                                ).sum()
                key = f"raise {raise_db:.0f} dB, {'field' if gain else 'lab'}"
                res[name][key] = {
                    "recall_visible": hit / max(tot, 1),
                    "precision": n_ok / max(n_det, 1),
                    "frames": int(tot),
                }
                print(name, key, res[name][key], flush=True)
    out = OUT / "shelfval.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps(old | res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
