"""Spectrogram crop of a dev mixture with truth, harmonic lines and classic detections.
Writes a preview PNG and a compact JSON for the explainer page."""
import base64, json, sys
import numpy as np

B = "/home/weygoldt/wrk/data/wavetracker_benchmark"
mix, t0, f0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
W, BW = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0, float(sys.argv[5]) if len(sys.argv) > 5 else 160.0
out = sys.argv[6] if len(sys.argv) > 6 else None
RATE, NFFT, CH = 20000.0, 2**15, 9
STEP = int(NFFT * 0.1)
d = f"{B}/mixtures/dev/{mix}"
n = (d and None)
raw = np.memmap(f"{d}/traces-grid1.raw", dtype=np.float32, mode="r").reshape(-1, CH)
a = int(t0 * RATE) - NFFT // 2
b = int((t0 + W) * RATE) + NFFT // 2
x = np.asarray(raw[max(a, 0):b], dtype=np.float64)
win = np.hanning(NFFT)
freqs = np.fft.rfftfreq(NFFT, 1 / RATE)
k0, k1 = np.searchsorted(freqs, f0), np.searchsorted(freqs, f0 + BW)
starts = np.arange(0, len(x) - NFFT, STEP)
P = np.empty((len(starts), k1 - k0))
Pe = np.empty((len(starts), CH, k1 - k0), dtype=np.float32)
for i, s in enumerate(starts):
    X = np.fft.rfft(x[s:s + NFFT] * win[:, None], axis=0)[k0:k1]
    pe = np.abs(X) ** 2
    Pe[i] = pe.T
    P[i] = pe.sum(1)
times = (max(a, 0) + starts + NFFT / 2) / RATE
fz = freqs[k0:k1]
L = 10 * np.log10(P + 1e-30)
noise = np.median(L, axis=0, keepdims=True)  # per-bin noise floor over the window
noise = np.median(L)                          # scalar is enough for display
tz = np.load(f"{d}/truth.npz")
Tt, Ft = tz["times"], tz["freqs"]
lines = []
for k in range(Ft.shape[0]):
    fk = np.interp(times, Tt, Ft[k], left=np.nan, right=np.nan)
    snr = np.interp(times, tz["snr_times"], tz["snr_db"][k])
    for h in (1, 2, 3, 4):
        fh = h * fk
        m = (fh > f0) & (fh < f0 + BW)
        if m.sum() > 5:
            lines.append({"fish": k, "h": h, "f": np.where(m, np.round(fh, 2), None).tolist(),
                          "snr": float(np.nanmedian(snr[m])) if h == 1 else None})
run = f"{B}/runs/dense-linked/dev/{mix}"
rt = np.load(f"{run}/times.npy"); fv = np.load(f"{run}/fund_v.npy"); iv = np.load(f"{run}/idx_v.npy"); idv = np.load(f"{run}/ident_v.npy")
tv = rt[iv.astype(int)]
m = (tv >= times[0]) & (tv <= times[-1]) & (fv > f0) & (fv < f0 + BW)
det = {"t": np.round(tv[m], 3).tolist(), "f": np.round(fv[m], 2).tolist(), "assigned": np.isfinite(idv[m]).tolist()}
lo, hi = noise - 3, noise + 32
q = np.clip((L - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
print(mix, "frames", q.shape, "noise", round(noise, 1), "lines", len(lines), "detections", int(m.sum()))
for ln in lines:
    print("  fish", ln["fish"], "h", ln["h"], "median f", round(float(np.nanmedian(np.array(ln["f"], dtype=float))), 1), "snr", ln["snr"])
if out:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(q.T, origin="lower", aspect="auto", extent=[times[0], times[-1], fz[0], fz[-1]], cmap="magma")
    for ln in lines:
        ax.plot(times, np.array(ln["f"], dtype=float), lw=0.8, color="cyan" if ln["h"] == 1 else "lime", ls="-" if ln["h"] == 1 else ":")
    ax.scatter(det["t"], det["f"], s=2, color="white")
    fig.savefig(out + ".png", dpi=110); plt.close(fig)
    json.dump({"mix": mix, "t0": float(times[0]), "dt": float(times[1] - times[0]), "f0": float(fz[0]), "df": float(fz[1] - fz[0]),
               "nt": q.shape[0], "nf": q.shape[1], "img": base64.b64encode(q.T[::-1].tobytes()).decode(),
               "lines": lines, "det": det, "db_range": [round(lo, 1), round(hi, 1)]}, open(out + ".json", "w"))
    np.save(out + "_elec.npy", Pe)
