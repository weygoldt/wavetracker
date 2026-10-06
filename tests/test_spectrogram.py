import numpy as np
import torch
from audioio import write_audio
from matplotlib.mlab import specgram

from wavetracker.io import FrameLayout, iter_blocks, open_recording
from wavetracker.spectrogram import PowerSpectrogram, decibel, estimate_noise_std

RATE = 20000.0
NFFT = 4096
STEP = 410


def test_scaling_matches_mlab():
    rng = np.random.default_rng(0)
    t = np.arange(int(2 * RATE)) / RATE
    x = np.sin(2 * np.pi * 600 * t) + 0.1 * rng.standard_normal(len(t))
    ours = PowerSpectrogram(NFFT, STEP, RATE, torch.device("cpu"))(
        x[None].astype(np.float32)
    )[0].numpy()
    ref, _, _ = specgram(
        x,
        NFFT=NFFT,
        Fs=RATE,
        noverlap=NFFT - STEP,
        window=np.hanning(NFFT),
        scale_by_freq=True,
        detrend="mean",
    )
    n = min(ours.shape[1], ref.shape[1])
    # mlab detrends every segment, we remove the block mean: compare above DC
    np.testing.assert_allclose(ours[5:, :n], ref[5:, :n], rtol=2e-3, atol=1e-9)
    peak = np.argmax(ours[:, 0])
    assert abs(peak * RATE / NFFT - 600) < RATE / NFFT


def test_blocks_reproduce_one_shot_spectrogram(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.standard_normal((int(3 * RATE), 3)).astype(np.float32) * 0.1
    write_audio(str(tmp_path / "x.wav"), data, RATE)
    spec = PowerSpectrogram(NFFT, STEP, RATE, torch.device("cpu"))

    with open_recording(tmp_path / "x.wav") as rec:
        layout = FrameLayout(1000, len(rec), NFFT, STEP, RATE)
        full = spec(np.asarray(rec[1000:].T, dtype=np.float32))
        blocks = [spec(b.data) for b in iter_blocks(rec, layout, frames_per_block=17)]
    # per-block mean removal differs slightly from the one-shot spectrogram,
    # so compare away from DC
    assert full.shape[-1] == layout.n_frames == len(layout.times())
    np.testing.assert_allclose(
        torch.cat(blocks, dim=-1)[:, 5:].numpy(),
        full[:, 5:].numpy(),
        rtol=1e-3,
        atol=1e-10,
    )


def test_noise_std_of_white_noise():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, int(10 * RATE))).astype(np.float32)
    power = PowerSpectrogram(NFFT, STEP, RATE, torch.device("cpu"))(x)[0]
    std = estimate_noise_std(decibel(power))
    # log of an exponential distribution (chi^2, 2 dof) has a std of ~5.6 dB
    assert 3.0 < std < 7.0
