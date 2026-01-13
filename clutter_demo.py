#!/usr/bin/env python3
"""
PASSIVE RADAR PROCESSING (FAST) + GIF OUTPUT (NO FFMPEG REQUIRED)

- Reads cs8 (int8 IQ interleaved) blocks from reference + surveillance files
- Mean removal per block
- Injects simulated delayed + Doppler target AFTER adapt_blocks
- NLMS clutter cancellation (Numba JIT)
- Cross-correlation before/after (FFT-based)
- Records first max_record blocks
- Generates an animated GIF using Matplotlib PillowWriter (no ffmpeg/opencv needed)

Dependencies:
  pip install numpy matplotlib numba pillow
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # IMPORTANT: offscreen backend (works on macOS reliably)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from numba import njit


# ----------------------------
# CS8 reader (int8 IQ interleaved)
# ----------------------------
def read_cs8_block(f, num_complex: int):
    """
    Reads num_complex complex samples from a cs8 file:
      I,Q are int8 interleaved: [I0,Q0,I1,Q1,...]
    Returns complex64 array length num_complex, or None on EOF.
    """
    raw = np.fromfile(f, dtype=np.int8, count=2 * num_complex)
    if raw.size < 2 * num_complex:
        return None
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return i.astype(np.complex64) + 1j * q.astype(np.complex64)


# ----------------------------
# FFT-based full cross-correlation (xcorr-like)
# ----------------------------
def xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Full cross-correlation via FFT for equal-length arrays:
      r = ifft( FFT(a) * conj(FFT(b)) )
    Returned in "full" lag order: lags = -(N-1) ... (N-1)

    Note: If your peak appears mirrored, swap (a,b) or conjugate an input.
    """
    n = a.size
    m = 2 * n - 1
    nfft = 1 << (m - 1).bit_length()

    A = np.fft.fft(a, nfft)
    B = np.fft.fft(b, nfft)
    r = np.fft.ifft(A * np.conj(B))

    # negative lags then non-negative
    return np.concatenate((r[nfft - (n - 1): nfft], r[0:n]))


# ----------------------------
# Numba-accelerated NLMS
# ----------------------------
@njit(cache=True)
def nlms_block(x: np.ndarray, d: np.ndarray, w: np.ndarray, mu: float, eps: float):
    """
    Complex NLMS adaptive filter, one block.
    x: reference (complex64)
    d: desired/surveillance (complex64)
    w: weights (complex64), updated in-place and returned

    Returns (y_hat, e, w):
      y_hat: estimated clutter (filter output)
      e: cleaned signal = d - y_hat
    """
    L = w.size
    N = d.size
    y_hat = np.zeros(N, dtype=np.complex64)
    e = np.zeros(N, dtype=np.complex64)

    # circular buffer for x
    xbuf = np.zeros(L, dtype=np.complex64)
    idx = 0

    for n in range(N):
        xbuf[idx] = x[n]

        y = 0.0 + 0.0j
        norm = 0.0
        p = idx
        for k in range(L):
            xv = xbuf[p]
            y += np.conj(w[k]) * xv
            norm += (xv.real * xv.real + xv.imag * xv.imag)
            p -= 1
            if p < 0:
                p = L - 1

        y_hat[n] = y
        err = d[n] - y
        e[n] = err

        g = (mu / (norm + eps)) * np.conj(err)
        p = idx
        for k in range(L):
            w[k] += g * xbuf[p]
            p -= 1
            if p < 0:
                p = L - 1

        idx += 1
        if idx == L:
            idx = 0

    return y_hat, e, w


def main():
    # ----------------------------
    # File paths (edit)
    # ----------------------------
    fid_x_path = "/Users/ibrahimsweidan/Downloads/500M/63_500M_01_11_car_2.cs8"  # Reference
    fid_y_path = "/Users/ibrahimsweidan/Downloads/500M/63_500M_01_11_car_2.cs8"  # Surveillance (same for demo ok)

    if not os.path.exists(fid_x_path) or not os.path.exists(fid_y_path):
        raise FileNotFoundError("Could not find one or both input files. Check paths.")

    # ----------------------------
    # Parameters (match MATLAB)
    # ----------------------------
    length = 2**15
    Fs = 20e6

    L = 2**12
    mu = 0.3
    eps = 1e-6

    adapt_blocks = 40
    max_record = 50

    delay_samp = 700
    doppler_hz = 30
    atten = 0.0008

    # Output animation
    fps = 20
    gif_filename = "PassiveRadar_Corr_Combined.gif"

    # Precompute Doppler phasor (same each block)
    t = (np.arange(length, dtype=np.float32) / np.float32(Fs)).astype(np.float32)
    doppler_phasor = np.exp(1j * 2 * np.pi * np.float32(doppler_hz) * t).astype(np.complex64)

    # NLMS weights
    w = np.zeros(L, dtype=np.complex64)

    # Storage (store dB arrays)
    R_before_db_all = [None] * max_record
    R_after_db_all = [None] * max_record

    # Lags (us)
    lags_us = (np.arange(-(length - 1), length, dtype=np.int64) / Fs) * 1e6

    print("Starting...")

    # ----------------------------
    # Processing loop
    # ----------------------------
    with open(fid_x_path, "rb") as fx, open(fid_y_path, "rb") as fy:
        for block_idx in range(1, max_record + 1):
            pct = (block_idx / max_record) * 100
            bar_len = 30
            filled = int(round((pct / 100) * bar_len))
            empty = bar_len - filled
            print(f"[{'#'*filled}{'-'*empty}] {pct:5.1f}% ({block_idx}/{max_record} blocks)", end="\r", flush=True)

            x = read_cs8_block(fx, length)
            y = read_cs8_block(fy, length)
            if x is None or y is None:
                print("\nEOF reached early.")
                break

            # Remove mean per block (like MATLAB)
            x = x - x.mean()
            y = y - y.mean()

            # Inject simulated target after NLMS "converges"
            if block_idx > adapt_blocks:
                y_delayed = np.zeros_like(y)
                if delay_samp < length:
                    y_delayed[delay_samp:] = y[:-delay_samp]
                y = y + (atten * (y_delayed * doppler_phasor)).astype(np.complex64)

            # NLMS clutter cancel
            _, y_clean, w = nlms_block(x, y, w, mu, eps)

            # Correlations (FFT)
            R_before = xcorr_fft(x, y)
            R_after = xcorr_fft(x, y_clean)

            # dB magnitude
            R_before_db_all[block_idx - 1] = 20 * np.log10(np.abs(R_before) + 1e-12)
            R_after_db_all[block_idx - 1] = 20 * np.log10(np.abs(R_after) + 1e-12)

    print("\nRecording complete. Generating GIF...")

    # ----------------------------
    # Fast GIF generation (no ffmpeg)
    # - Use line updates + blitting for speed
    # - Crop x-axis to [-100, 100] us to reduce work
    # ----------------------------
    mask = (lags_us >= -100) & (lags_us <= 100)
    x_crop = lags_us[mask]

    before_frames = []
    after_frames = []
    for k in range(max_record):
        if R_before_db_all[k] is None:
            break
        before_frames.append(R_before_db_all[k][mask])
        after_frames.append(R_after_db_all[k][mask])

    nframes = len(before_frames)
    if nframes == 0:
        raise RuntimeError("No frames recorded. Check file paths/EOF.")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
    line_before, = ax.plot([], [], linewidth=1.2, label="Before NLMS")
    line_after,  = ax.plot([], [], linewidth=1.2, label="After NLMS")

    ax.set_title("Cross-Correlation Before/After NLMS")
    ax.set_xlabel("Lag (µs)")
    ax.set_ylabel("Correlation (dB)")
    ax.grid(True)
    ax.set_ylim([-200, 200])
    ax.set_xlim([-100, 100])
    ax.legend()

    def init():
        line_before.set_data([], [])
        line_after.set_data([], [])
        return line_before, line_after

    def update(i):
        line_before.set_data(x_crop, before_frames[i])
        line_after.set_data(x_crop, after_frames[i])
        ax.set_title(f"Cross-Correlation Before/After NLMS (Block {i+1})")
        return line_before, line_after

    anim = FuncAnimation(
        fig,
        update,
        frames=nframes,
        init_func=init,
        blit=True,
        interval=int(1000 / fps),
    )

    writer = PillowWriter(fps=fps)
    anim.save(gif_filename, writer=writer)

    plt.close(fig)
    print(f"Finished! GIF saved to: {gif_filename}")


if __name__ == "__main__":
    main()
