#!/usr/bin/env python3
"""
Cross-correlation video + correlation assessment for HackRF .cs8 files

- Hardcoded file paths (from user)
- Uses ONLY the first 1,000,000 complex samples
- Block size: 32768 complex samples
- Per-block linear cross-correlation magnitude (normalized), NOT dB
- Saves video: xcorr_32k.mp4
- Prints an assessment of whether the two channels look correlated
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =========================
# HARDCODED FILE PATHS
# =========================
fid_x_path = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_184044/ref.cs8"
fid_y_path = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_184044/sur.cs8"

OUTPUT_VIDEO = "xcorr_32k.mp4"

BLOCK_SIZE = 5*4096
MAX_COMPLEX_SAMPLES = 10_000000
FPS = 30
DPI = 150

# Plot settings (linear magnitude, not dB)
Y_MIN, Y_MAX = 0.0, 1.0


# =========================
# Helpers
# =========================
def cs8_to_complex(iq_i8: np.ndarray) -> np.ndarray:
    """Convert interleaved int8 IQ -> complex64 in approx [-1, 1]."""
    i = iq_i8[0::2].astype(np.float32)
    q = iq_i8[1::2].astype(np.float32)
    return (i + 1j * q) / 128.0


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def linear_xcorr_fft(x: np.ndarray, y: np.ndarray, nfft: int) -> np.ndarray:
    """
    Linear cross-correlation using FFT.
    Returns lags from -(N-1) ... (N-1), length (2N-1).
    """
    N = len(x)
    X = np.fft.fft(x, nfft)
    Y = np.fft.fft(y, nfft)
    r_circ = np.fft.ifft(X * np.conj(Y))
    pos = r_circ[:N]          # lag 0..N-1
    neg = r_circ[-(N - 1):]   # lag -(N-1)..-1
    return np.concatenate([neg, pos])


def get_block(mm: np.memmap, block_idx: int) -> np.ndarray:
    """Read one block of complex samples from interleaved int8 memmap."""
    c0 = block_idx * BLOCK_SIZE
    c1 = c0 + BLOCK_SIZE
    i0 = c0 * 2
    i1 = c1 * 2
    iq = np.array(mm[i0:i1], copy=False)
    return cs8_to_complex(iq)


# =========================
# Load data
# =========================
if not os.path.isfile(fid_x_path):
    raise FileNotFoundError(fid_x_path)
if not os.path.isfile(fid_y_path):
    raise FileNotFoundError(fid_y_path)

x_raw = np.memmap(fid_x_path, dtype=np.int8, mode="r")
y_raw = np.memmap(fid_y_path, dtype=np.int8, mode="r")

# Ensure even IQ length (I/Q pairs)
x_len = (x_raw.size // 2) * 2
y_len = (y_raw.size // 2) * 2

x_complex_samples = x_len // 2
y_complex_samples = y_len // 2

usable_samples = min(x_complex_samples, y_complex_samples, MAX_COMPLEX_SAMPLES)
n_blocks = usable_samples // BLOCK_SIZE
if n_blocks < 1:
    raise ValueError("Not enough samples for one 32k block within first 1,000,000 samples.")

# FFT length for linear correlation
nfft = next_pow2(2 * BLOCK_SIZE - 1)
lags = np.arange(-(BLOCK_SIZE - 1), BLOCK_SIZE, dtype=np.int32)  # length 2N-1

print("=== Setup ===")
print("X file:", fid_x_path)
print("Y file:", fid_y_path)
print("Using complex samples:", usable_samples)
print("Block size:", BLOCK_SIZE)
print("Blocks:", n_blocks)
print("FFT size:", nfft)


# =========================
# Correlation assessment
# =========================
def assess_correlation():
    peak_vals = []
    peak_lags = []
    pm_ratios = []

    for bi in range(n_blocks):
        x = get_block(x_raw, bi)
        y = get_block(y_raw, bi)

        # Remove DC offset (helps suppress bias/direct-path DC)
        x = x - np.mean(x)
        y = y - np.mean(y)

        r = linear_xcorr_fft(x, y, nfft)
        mag = np.abs(r).astype(np.float32)

        # Normalize so results are comparable across blocks
        mag /= (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

        peak_idx = int(np.argmax(mag))
        peak_val = float(mag[peak_idx])
        peak_lag = int(lags[peak_idx])

        med = float(np.median(mag))
        pm = peak_val / (med + 1e-12)

        peak_vals.append(peak_val)
        peak_lags.append(peak_lag)
        pm_ratios.append(pm)

    peak_vals = np.array(peak_vals, dtype=np.float32)
    peak_lags = np.array(peak_lags, dtype=np.int32)
    pm_ratios = np.array(pm_ratios, dtype=np.float32)

    # Mode (most common peak lag)
    offset = -int(lags.min())
    counts = np.bincount(peak_lags + offset)
    mode_lag = int(np.argmax(counts) - offset)

    # Stability: how often peak lag stays close to mode
    stable_frac = float(np.mean(np.abs(peak_lags - mode_lag) <= 2))

    # Summaries
    mean_peak = float(np.mean(peak_vals))
    med_peak = float(np.median(peak_vals))
    mean_pm = float(np.mean(pm_ratios))
    med_pm = float(np.median(pm_ratios))

    # Print everything
    print("\n=== Correlation assessment (using all processed blocks) ===")
    print(f"Blocks checked:                 {len(peak_vals)}")
    print(f"Mean peak magnitude (norm):     {mean_peak:.6f}")
    print(f"Median peak magnitude (norm):   {med_peak:.6f}")
    print(f"Mean peak-to-median ratio:      {mean_pm:.3f}")
    print(f"Median peak-to-median ratio:    {med_pm:.3f}")
    print(f"Most common peak lag (samples): {mode_lag}")
    print(f"Peak lag stability (±2 samples): {stable_frac*100:.1f}%")

    # Simple heuristic interpretation
    print("\nHeuristic interpretation (rules of thumb):")
    if (med_peak > 0.10) and (med_pm > 5.0) and (stable_frac > 0.80):
        verdict = "STRONGLY correlated (likely common signal / same-direction, stable delay)."
    elif (med_peak > 0.03) and (med_pm > 2.0) and (stable_frac > 0.50):
        verdict = "SOMEWHAT correlated (shared component, but weaker/less stable)."
    else:
        verdict = "WEAKLY correlated (could be different pointing, low SNR, interference, or lack of coherence)."
    print("-", verdict)

    print("\nNotes:")
    print("- If you used two independent SDR clocks (two separate HackRFs), correlation can smear/drift even with same pointing.")
    print("- If antennas are truly same direction and channels are coherent, you usually see a stable peak lag across blocks.")

    return mode_lag, stable_frac, med_peak, med_pm


mode_lag, stable_frac, med_peak, med_pm = assess_correlation()


# =========================
# Video setup
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.plot([], [], lw=1.2)

ax.set_xlabel("Lag (samples)")
ax.set_ylabel("|Rxy| (linear, normalized)")
ax.grid(True, alpha=0.3)
ax.set_xlim(int(lags.min()), int(lags.max()))
ax.set_ylim(Y_MIN, Y_MAX)

title = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", va="bottom")
subtitle = ax.text(0.5, 0.98, "", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def init():
    line.set_data([], [])
    title.set_text("")
    subtitle.set_text("")
    return line, title, subtitle


def update(frame_idx: int):
    x = get_block(x_raw, frame_idx)
    y = get_block(y_raw, frame_idx)

    # DC removal
    x = x - np.mean(x)
    y = y - np.mean(y)

    r = linear_xcorr_fft(x, y, nfft)
    mag = np.abs(r).astype(np.float32)

    # Normalize
    mag /= (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

    line.set_data(lags, mag)
    title.set_text(f"Cross-correlation magnitude | Block {frame_idx + 1}/{n_blocks}")
    subtitle.set_text(f"Mode peak lag: {mode_lag} samples | Stability ±2: {stable_frac*100:.1f}% | Median peak: {med_peak:.3f} | Median P/M: {med_pm:.2f}")
    return line, title, subtitle


anim = FuncAnimation(fig, update, frames=n_blocks, init_func=init, blit=True)

writer = FFMpegWriter(fps=FPS, bitrate=1800)
anim.save(OUTPUT_VIDEO, writer=writer, dpi=DPI)

print("\nSaved video:", OUTPUT_VIDEO)
