#!/usr/bin/env python3
"""
Doppler-vs-time assuming lag = 0 (tau = 0).

Per CPI slice:
  - Read CPI chunk (ref, sur) at full FS
  - Split into overlapping blocks (NFAST, HOP_FAST)
  - For each block:
      corr = IFFT( FFT(sur)*conj(FFT(ref)) )
      take corr[0] only  # lag = 0
  - Stack corr[0] across blocks -> slow-time sequence (length M)
  - Optional: remove slow-time mean (suppresses 0-Doppler ridge)
  - Doppler FFT across slow-time
  - Append to rolling spectrogram and write MP4 frames

Deps:
  pip install numpy matplotlib imageio imageio-ffmpeg tqdm
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
from tqdm import tqdm

# --------------------------
# USER SETTINGS
# --------------------------

REF_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_183539/ref.cs8"
SUR_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_183539/sur.cs8"

FS = 10_000_000          # full-rate
FC_HZ = 509e6

# Time slicing (for Doppler-vs-time)
CPI_SEC = 0.50
HOP_SEC = 0.10
FPS = 10
WINDOW_SEC_ON_SCREEN = 6.0
OUT_MP4 = "doppler_vs_time_lag0.mp4"

# Blocked correlation (CAF-style)
NFAST = 4096
HOP_FAST = NFAST // 4           # overlap
NFFT_RANGE = 16384              # zero-pad correlation (smooth)

# Ridge suppression
REMOVE_SLOWTIME_MEAN = True     # MTI-ish on the lag=0 slow-time
NOTCH_ZERO_DOPPLER_BINS = 1     # 0 disables; 1-3 typical

# Plot zoom
MAX_DOPPLER_HZ = 80
DB_FLOOR = -40
DB_CEIL = 15
NORM_PERCENTILE = 98.0
IM_INTERP = "bilinear"

# Debug prints
PRINT_PEAK = True
PEAK_EXCLUDE_HZ = 4.0
PEAK_SEARCH_MAX_HZ = 80

# --------------------------
# IO
# --------------------------

def file_num_complex_samples(path: str) -> int:
    return os.path.getsize(path) // 2

def read_cs8_iq(path: str, start_samp: int, num_samp: int) -> np.ndarray:
    """HackRF .cs8: interleaved int8 I,Q."""
    offset_bytes = start_samp * 2
    count_i8 = num_samp * 2
    with open(path, "rb") as f:
        f.seek(offset_bytes, os.SEEK_SET)
        raw = np.fromfile(f, dtype=np.int8, count=count_i8)
    if raw.size < count_i8:
        raw = np.pad(raw, (0, count_i8 - raw.size), mode="constant")
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return ((i + 1j * q) / 128.0).astype(np.complex64)

# --------------------------
# DSP
# --------------------------

def doppler_axis_hz(M: int, pri: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(M, d=pri))

def render_spectrogram(fd_hz, t_axis, spec_db, max_dopp_hz, fc_hz, title) -> np.ndarray:
    mask = np.abs(fd_hz) <= max_dopp_hz
    fd = fd_hz[mask]
    S = spec_db[:, mask]         # (T, Fzoom)
    img_mat = S.T                # (Fzoom, T)

    fig = plt.figure(figsize=(9, 4.8), dpi=140)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    extent = [t_axis[0], t_axis[-1], fd[0], fd[-1]]
    im = ax.imshow(
        img_mat,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation=IM_INTERP,
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler (Hz)")

    c = 299_792_458.0
    lam = c / fc_hz
    def hz_to_v(x): return x * lam
    def v_to_hz(v): return v / lam
    sec = ax.secondary_yaxis("right", functions=(hz_to_v, v_to_hz))
    sec.set_ylabel("Radial velocity (m/s, approx)")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("dB (lag=0, normalized)")

    fig.tight_layout()
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    out = buf[..., :3].copy()
    plt.close(fig)
    return out

def peak_doppler(fd_hz, mag_lin, exclude_hz, max_hz):
    m = (np.abs(fd_hz) >= exclude_hz) & (np.abs(fd_hz) <= max_hz)
    if not np.any(m):
        return float("nan")
    idx = np.argmax(mag_lin[m])
    return float(fd_hz[m][idx])

# --------------------------
# Main
# --------------------------

def main():
    n_total = min(file_num_complex_samples(REF_PATH), file_num_complex_samples(SUR_PATH))

    cpi_samp = int(CPI_SEC * FS)
    hop_samp = int(HOP_SEC * FS)
    if cpi_samp <= 0 or hop_samp <= 0:
        raise ValueError("CPI_SEC and HOP_SEC must be > 0")

    # How many CPI slices
    n_slices = 1 + max(0, (n_total - cpi_samp) // hop_samp)

    # Rolling display window
    slices_per_sec = 1.0 / HOP_SEC
    slices_on_screen = max(5, int(WINDOW_SEC_ON_SCREEN * slices_per_sec))

    # Doppler details
    pri = HOP_FAST / FS

    print("=== Doppler-vs-time (lag = 0 / tau = 0) ===")
    print(f"FS={FS} Hz, CPI={CPI_SEC:.2f}s ({cpi_samp} samples), hop={HOP_SEC:.2f}s")
    print(f"NFAST={NFAST}, HOP_FAST={HOP_FAST}, NFFT_RANGE={NFFT_RANGE}, PRI={pri*1e6:.2f} us")
    print("Assuming lag bin = 0 (no range gating)")
    print(f"Output: {OUT_MP4}")

    # Storage for rolling spectrogram
    spec_ring = []
    time_ring = []

    writer = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8)

    with tqdm(total=n_slices, desc="Processing CPI slices") as pbar:
        start = 0
        for k in range(n_slices):
            if start + cpi_samp > n_total:
                break

            t0 = start / FS

            ref = read_cs8_iq(REF_PATH, start, cpi_samp)
            sur = read_cs8_iq(SUR_PATH, start, cpi_samp)

            # DC removal helps correlation / ridge suppression
            ref = ref - np.mean(ref)
            sur = sur - np.mean(sur)

            L = len(ref)
            if L < NFAST:
                break

            M = 1 + (L - NFAST) // HOP_FAST
            slow = np.zeros(M, dtype=np.complex64)

            win = np.hanning(NFAST).astype(np.float32)

            idx = 0
            for m in range(M):
                rb = (ref[idx:idx+NFAST] * win).astype(np.complex64)
                sb = (sur[idx:idx+NFAST] * win).astype(np.complex64)

                R = np.fft.fft(rb, n=NFFT_RANGE)
                S = np.fft.fft(sb, n=NFFT_RANGE)

                corr = np.fft.ifft(S * np.conj(R))  # correlation vs delay (circular)
                slow[m] = corr[0]                   # <<< lag = 0 only

                idx += HOP_FAST

            # Optional MTI-ish: remove slow-time mean (kills strong 0-Doppler ridge)
            if REMOVE_SLOWTIME_MEAN:
                slow = slow - np.mean(slow)

            # Doppler FFT
            wslow = np.hanning(M).astype(np.float32)
            D = np.fft.fftshift(np.fft.fft(slow * wslow))  # (M,)
            mag = np.abs(D).astype(np.float32)

            fd_hz = doppler_axis_hz(M, pri)

            # Optional notch at 0 Hz (center bin after fftshift)
            if NOTCH_ZERO_DOPPLER_BINS and NOTCH_ZERO_DOPPLER_BINS > 0:
                cbin = M // 2
                mag[cbin-NOTCH_ZERO_DOPPLER_BINS:cbin+NOTCH_ZERO_DOPPLER_BINS+1] = 0.0

            # Peak debug
            if PRINT_PEAK:
                pk = peak_doppler(fd_hz, mag, PEAK_EXCLUDE_HZ, PEAK_SEARCH_MAX_HZ)
                print(f"t={t0:7.2f}s  peak_doppler={pk:7.2f} Hz  M={M}")

            # Convert to dB for display (stable contrast)
            mag_db = 20.0 * np.log10(mag + 1e-12)
            mag_db -= np.percentile(mag_db, NORM_PERCENTILE)
            mag_db = np.clip(mag_db, DB_FLOOR, DB_CEIL)

            spec_ring.append(mag_db.astype(np.float32))
            time_ring.append(t0)

            if len(spec_ring) > slices_on_screen:
                spec_ring.pop(0)
                time_ring.pop(0)

            if len(spec_ring) == slices_on_screen:
                Sspec = np.stack(spec_ring, axis=0)  # (T, F)
                t_axis = np.array(time_ring, dtype=np.float32)

                title = (
                    f"Doppler vs Time (lag=0)  "
                    f"t={t_axis[0]:.2f}–{t_axis[-1]+CPI_SEC:.2f}s  CPI={CPI_SEC:.2f}s"
                )
                img = render_spectrogram(fd_hz, t_axis, Sspec, MAX_DOPPLER_HZ, FC_HZ, title)
                writer.append_data(img)

            start += hop_samp
            pbar.update(1)

    writer.close()
    print("Done. Output:", OUT_MP4)

if __name__ == "__main__":
    main()