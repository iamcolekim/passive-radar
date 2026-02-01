#!/usr/bin/env python3
"""
PASSIVE RADAR PROCESSING
- NLMS clutter suppression
- Correlation MP4
- Range–Doppler CAF MP4 (post-NLMS)
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from numba import njit


def read_cs8_block(f, n_complex: int):
    raw = np.fromfile(f, dtype=np.int8, count=2 * n_complex)
    if raw.size < 2 * n_complex:
        return None
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return (i + 1j * q).astype(np.complex64)


def xcorr_fft(a, b):
    n = a.size
    m = 2 * n - 1
    nfft = 1 << (m - 1).bit_length()
    r = np.fft.ifft(np.fft.fft(a, nfft) * np.conj(np.fft.fft(b, nfft)))
    return np.concatenate((r[nfft - (n - 1): nfft], r[:n]))


@njit(cache=True)
def nlms_block(x, d, w, mu, eps):
    L = w.size
    N = d.size

    y_hat = np.zeros(N, dtype=np.complex64)
    e = np.zeros(N, dtype=np.complex64)

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
            norm += xv.real * xv.real + xv.imag * xv.imag
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


def save_correlation_mp4(x_lag_us, before_frames, after_frames, out_path, fps):
    fig, ax = plt.subplots(figsize=(7, 7))
    (l1,) = ax.plot([], [], label="Before NLMS")
    (l2,) = ax.plot([], [], label="After NLMS")

    ax.set_xlim(-100, 100)
    ax.set_ylim(-200, 200)
    ax.set_xlabel("Lag (µs)")
    ax.set_ylabel("Correlation (dB)")
    ax.legend()
    ax.grid(True)

    def upd(i):
        l1.set_data(x_lag_us, before_frames[i])
        l2.set_data(x_lag_us, after_frames[i])
        ax.set_title(f"Correlation Block {i+1}")
        return l1, l2

    FuncAnimation(fig, upd, frames=len(before_frames), blit=True).save(
        out_path, writer=FFMpegWriter(fps=fps)
    )
    plt.close(fig)


def save_caf_mp4(caf_corr, x_lag_us, Nslow, Tblock, out_path, fps):
    """
    Builds Range–Doppler CAF frames from caf_corr (list of complex correlation vectors)
    and saves an MP4. Matches original behavior:
      - sliding window of Nslow
      - FFT over slow-time axis
      - fftshift
      - dB magnitude
    """
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(Nslow, d=Tblock))

    caf_frames = []
    for i in range(Nslow - 1, len(caf_corr)):
        S = np.stack(caf_corr[i - (Nslow - 1): i + 1], axis=0)
        RD = np.fft.fftshift(np.fft.fft(S, axis=0), axes=0)
        caf_frames.append(20 * np.log10(np.abs(RD) + 1e-12))

    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [x_lag_us[0], x_lag_us[-1], doppler_axis[0], doppler_axis[-1]]

    im = ax.imshow(
        caf_frames[0],
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    ax.set_xlabel("Lag (µs)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title("Range–Doppler CAF (After NLMS)")

    def upd(i):
        im.set_data(caf_frames[i])
        ax.set_title(f"Range–Doppler CAF (Frame {i+1})")
        return (im,)

    FuncAnimation(fig, upd, frames=len(caf_frames), blit=True).save(
        out_path, writer=FFMpegWriter(fps=fps)
    )
    plt.close(fig)


def main():
    fid_x_path = "/Users/ibrahimsweidan/Downloads/drive-download-20260117T214613Z-3-001/63_99.1MHz_20260117_161719.cs8"
    fid_y_path = "/Users/ibrahimsweidan/Downloads/drive-download-20260117T214613Z-3-001/63_99.1MHz_20260117_161719.cs8"
    if not os.path.exists(fid_x_path):
        raise FileNotFoundError("Input file not found")

    length = 2**15
    Fs = 10e6

    Nslow = 50
    L = 2**14
    mu = 0.6
    eps = 1e-6

    adapt_blocks = 100
    max_record = 150

    delay_samp = 350
    doppler_hz = 30.0
    atten = 0.008

    fps = 20
    corr_mp4 = "PassiveRadar_Correlation.mp4"
    caf_mp4 = "PassiveRadar_RangeDoppler_CAF.mp4"

    lags_us = (np.arange(-(length - 1), length) / Fs) * 1e6
    t = np.arange(length, dtype=np.float32) / Fs
    doppler_phasor = np.exp(1j * 2 * np.pi * doppler_hz * t).astype(np.complex64)

    w = np.zeros(L, dtype=np.complex64)
    R_before_db = [None] * max_record
    R_after_db = [None] * max_record
    R_after_cplx = [None] * max_record

    step_size = 0
    Tblock = length / Fs

    with open(fid_x_path, "rb") as fx, open(fid_y_path, "rb") as fy:
        for bi in range(1, max_record + 1):
            x = read_cs8_block(fx, length)
            y = read_cs8_block(fy, length)
            if x is None or y is None:
                break

            x -= x.mean()
            y -= y.mean()

            if bi > adapt_blocks:
                if step_size > Nslow:
                    step_size = 0
                    delay_samp += 50
                step_size += 1

                y_delayed = np.zeros_like(x)
                y_delayed[delay_samp:] = x[:-delay_samp]

                phase0 = np.exp(1j * 2 * np.pi * doppler_hz * (bi - 1) * Tblock).astype(
                    np.complex64
                )
                y += atten * y_delayed * doppler_phasor * phase0

            _, y_clean, w = nlms_block(x, y, w, mu, eps)

            Rb = xcorr_fft(x, y)
            Ra = xcorr_fft(x, y_clean)

            idx = bi - 1
            R_before_db[idx] = 20 * np.log10(np.abs(Rb) + 1e-12)
            R_after_db[idx] = 20 * np.log10(np.abs(Ra) + 1e-12)
            R_after_cplx[idx] = Ra.astype(np.complex64)

    mask = (lags_us >= -100) & (lags_us <= 100)
    x_crop = lags_us[mask]

    before_frames, after_frames, caf_corr = [], [], []
    for k in range(max_record):
        if R_before_db[k] is None:
            break
        before_frames.append(R_before_db[k][mask])
        after_frames.append(R_after_db[k][mask])
        caf_corr.append(R_after_cplx[k][mask])

    save_correlation_mp4(x_crop, before_frames, after_frames, corr_mp4, fps)
    save_caf_mp4(caf_corr, x_crop, Nslow, Tblock, caf_mp4, fps)

    print("Finished:")
    print(" ", corr_mp4)
    print(" ", caf_mp4)


if __name__ == "__main__":
    main()
