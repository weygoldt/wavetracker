"""Per-electrode zoom on the close pair of crop B (3x3 grid layout) + compact JSON."""
import base64, json, sys
import numpy as np
E = "/home/weygoldt/wrk/tools/wavetracker-salience/research/explainer"
c = json.load(open(f"{E}/cropB.json")); Pe = np.load(f"{E}/cropB_elec.npy")  # (frames, 9, bins)
lo_hz, hi_hz = float(sys.argv[1]), float(sys.argv[2])
fz = c["f0"] + c["df"] * np.arange(Pe.shape[2])
k = (fz >= lo_hz) & (fz <= hi_hz)
L = 10 * np.log10(Pe[:, :, k] + 1e-30)
noise = np.median(L)
lo, hi = noise - 3, noise + 32
q = np.clip((L - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)  # frames, 9, bins
S = 10 * np.log10(Pe[:, :, k].sum(1) + 1e-30)
qs = np.clip((S - (np.median(S) - 3)) / 35 * 255, 0, 255).astype(np.uint8)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 4, figsize=(14, 7))
for e in range(9):
    axs[e // 3, e % 3].imshow(q[:, e].T, origin="lower", aspect="auto", cmap="magma", extent=[0, 60, fz[k][0], fz[k][-1]]); axs[e // 3, e % 3].set_title(f"electrode {e}")
axs[0, 3].imshow(qs.T, origin="lower", aspect="auto", cmap="magma", extent=[0, 60, fz[k][0], fz[k][-1]]); axs[0, 3].set_title("sum")
fig.savefig(f"{E}/elecB.png", dpi=80); plt.close(fig)
json.dump({"f0": float(fz[k][0]), "df": c["df"], "nt": q.shape[0], "nf": int(k.sum()), "t0": c["t0"], "dt": c["dt"],
           "elec": [base64.b64encode(q[:, e].T[::-1].tobytes()).decode() for e in range(9)],
           "sum": base64.b64encode(qs.T[::-1].tobytes()).decode()}, open(f"{E}/elecB.json", "w"))
print(q.shape, round(noise, 1))
