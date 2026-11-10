"""Find a window of a dense dev mixture that shows the failure modes in one picture:
a close pair (< 3 Hz), a faint fish (< 6 dB), a harmonic of a lower fish, and classic misses."""
import json, sys
import numpy as np

B = "/home/weygoldt/wrk/data/wavetracker_benchmark"
mix = sys.argv[1] if len(sys.argv) > 1 else "crowded-30-s0"
tz = np.load(f"{B}/mixtures/dev/{mix}/truth.npz")
T, F = tz["times"], tz["freqs"]            # 0.1 s grid
st, snr = tz["snr_times"], tz["snr_db"]
run = f"{B}/runs/dense-linked/dev/{mix}"
fr = np.load(f"{run}/frames.npz")
best = []
W, BW = 60.0, 160.0
for t0 in np.arange(0, 600 - W, 10.0):
    sel = (T >= t0) & (T < t0 + W)
    ssel = (st >= t0) & (st < t0 + W)
    med = np.nanmedian(F[:, sel], axis=1)
    msnr = np.nanmedian(snr[:, ssel], axis=1)
    for f0 in np.arange(300, 1150 - BW, 10.0):
        inb = np.flatnonzero((med >= f0 + 8) & (med < f0 + BW - 8))
        if not 5 <= len(inb) <= 12:
            continue
        close = 0
        for i in inb:
            for j in inb:
                if i < j:
                    d = np.nanmedian(np.abs(F[i, sel] - F[j, sel]))
                    if 0.8 < d < 3.0:
                        close += 1
        faint = int(np.sum((msnr[inb] < 6) & (msnr[inb] > -2)))
        harm = 0
        for k in np.flatnonzero(np.isfinite(med)):
            for h in (2, 3):
                if f0 + 8 < h * med[k] < f0 + BW - 8 and msnr[k] > 10:
                    harm += 1
        if close and faint and harm:
            best.append((close + faint + harm, t0, f0, len(inb), close, faint, harm))
best.sort(reverse=True)
for b in best[:8]:
    print(mix, b)
