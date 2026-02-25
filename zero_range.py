#!/usr/bin/env python3
"""
Range-gated Doppler-vs-time (3 bins) using wideband DVB/ATSC bandwidth.

Pipeline per time-slice:
  - Read CPI chunk (ref, sur) at full FS (no raw decimation)
  - Split into overlapping blocks (NFAST, HOP_FAST)
  - For each block:
      R = FFT(ref_block, NFFT_RANGE)
      S = FFT(sur_block, NFFT_RANGE)
      corr = IFFT(S * conj(R))    # wideband cross-correlation vs delay
      pick 3 delay bins (lag indices corresponding to chosen meters)
  - Stack selected bins across blocks => slow-time sequences (M x 3)
  - Doppler FFT across slow-time for each bin
  - Combine bins (max or sum) => 1 Doppler spectrum for that CPI slice
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

REF_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_183000/ref.cs8"
SUR_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_183000/sur.cs8"

FS = 10_000_000          # keep full-rate to preserve bandwidth
FC_HZ = 509e6

# Time slicing (for Doppler-vs-time)
CPI_SEC = 0.50
HOP_SEC = 0.10
FPS = 10
WINDOW_SEC_ON_SCREEN = 6.0
OUT_MP4 = "doppler_vs_time_3rangebins.mp4"

# Blocked correlation (CAF Method-3 style)
NFAST = 4096
HOP_FAST = NFAST // 4           # overlap helps Doppler smoothness
NFFT_RANGE = 16384              # zero-pad correlation (smooth delay axis)

# Pick 3 "range bins" in meters (bistatic range-difference approx)
# Train ~100–200 m away: try 60/120/180 or 90/150/210 etc.
RANGE_BINS_M = (60.0, 120.0, 180.0)

# Only keep lags up to this to guard against picking nonsense (meters)
MAX_LAG_M = 400.0

# Combine 3 bins into one Doppler spectrum for display
COMBINE_MODE = "max"   # "max" or "sum"

# Optional: suppress residual direct-path ridge a bit
REMOVE_SLOWTIME_MEAN = True     # MTI per delay-bin
NOTCH_ZERO_DOPPLER_BINS = 1     # 0 disables; 1-3 typical

# Plot zoom
MAX_DOPPLER_HZ = 80             # train-focused
DB_FLOOR = -40
DB_CEIL = 15
NORM_PERCENTILE = 98.0          # stabilizes contrast
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

def meters_to_lag_samp(r_m: float, fs: float) -> int:
    c = 299_792_458.0
    tau = r_m / c
    return int(np.round(tau * fs))

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

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
    cb.set_label("dB (range-gated, normalized)")

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

    # Convert range bins (m) -> lag samples
    lag_max_samp = meters_to_lag_samp(MAX_LAG_M, FS)
    lag_bins = []
    for r in RANGE_BINS_M:
        s = meters_to_lag_samp(r, FS)
        if 0 <= s <= lag_max_samp:
            lag_bins.append(s)
        else:
            raise ValueError(f"Range bin {r} m -> lag {s} samples exceeds MAX_LAG_M limit.")
    lag_bins = tuple(lag_bins)

    # How many CPI slices
    n_slices = 1 + max(0, (n_total - cpi_samp) // hop_samp)

    # Rolling display window
    slices_per_sec = 1.0 / HOP_SEC
    slices_on_screen = max(5, int(WINDOW_SEC_ON_SCREEN * slices_per_sec))

    # Doppler details
    pri = HOP_FAST / FS

    print("=== Range-gated Doppler-vs-time (3 bins) ===")
    print(f"FS={FS} Hz, CPI={CPI_SEC:.2f}s ({cpi_samp} samples), hop={HOP_SEC:.2f}s")
    print(f"NFAST={NFAST}, HOP_FAST={HOP_FAST}, NFFT_RANGE={NFFT_RANGE}, PRI={pri*1e6:.2f} us")
    print(f"Range bins (m) = {RANGE_BINS_M}")
    print(f"Lag bins (samples) = {lag_bins}  (1 samp ~ {299_792_458.0/FS:.1f} m)")
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

            # DC removal helps both correlation and ridge suppression
            ref = ref - np.mean(ref)
            sur = sur - np.mean(sur)

            # Build slow-time sequences for 3 lag bins
            L = len(ref)
            if L < NFAST:
                break

            M = 1 + (L - NFAST) // HOP_FAST
            slow = np.zeros((M, len(lag_bins)), dtype=np.complex64)

            win = np.hanning(NFAST).astype(np.float32)

            idx = 0
            for m in range(M):
                rb = (ref[idx:idx+NFAST] * win).astype(np.complex64)
                sb = (sur[idx:idx+NFAST] * win).astype(np.complex64)

                R = np.fft.fft(rb, n=NFFT_RANGE)
                S = np.fft.fft(sb, n=NFFT_RANGE)

                corr = np.fft.ifft(S * np.conj(R))  # delay bins

                # pick only our 3 lag bins
                for j, lag in enumerate(lag_bins):
                    slow[m, j] = corr[lag]

                idx += HOP_FAST

            # Optional MTI per lag: remove slow-time mean
            if REMOVE_SLOWTIME_MEAN:
                slow = slow - np.mean(slow, axis=0, keepdims=True)

            # Doppler FFT per lag bin
            wslow = np.hanning(M).astype(np.float32)[:, None]
            D = np.fft.fftshift(np.fft.fft(slow * wslow, axis=0), axes=0)  # (M, 3)

            # Combine the 3 bins into one spectrum
            mag_bins = np.abs(D).astype(np.float32)  # (M, 3)
            if COMBINE_MODE == "sum":
                mag = np.sum(mag_bins, axis=1)
            else:  # "max"
                mag = np.max(mag_bins, axis=1)

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
                    f"Doppler vs Time (3 range bins, gated)  "
                    f"t={t_axis[0]:.2f}–{t_axis[-1]+CPI_SEC:.2f}s  CPI={CPI_SEC:.2f}s  "
                    f"bins={RANGE_BINS_M}m  combine={COMBINE_MODE}"
                )
                img = render_spectrogram(fd_hz, t_axis, Sspec, MAX_DOPPLER_HZ, FC_HZ, title)
                writer.append_data(img)

            start += hop_samp
            pbar.update(1)

    writer.close()
    print("Done. Output:", OUT_MP4)

if __name__ == "__main__":
    main()