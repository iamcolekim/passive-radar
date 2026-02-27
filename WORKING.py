#!/usr/bin/env python3
"""
Passive radar (ATSC/DTV) offline processing (HackRF .cs8), optimized + simplified.

Pipeline (unchanged):
mmap CS8 -> initial delay estimate/correction -> per-frame:
  read CPI -> (optional) decimate -> DC remove -> Wiener cancel (freq)
  -> overlapped range profiles (batched) -> Range-Doppler (slow-time FFT)
  -> crop -> percentile normalize -> render -> MP4

Deps:
  pip install numpy matplotlib imageio imageio-ffmpeg tqdm
"""

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

REF_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_184044/ref.cs8"
SUR_PATH = "/Users/ibrahimsweidan/Downloads/509.0MHz_20260217_184044/sur.cs8"

FS = 10_000_000
DECIM = 1
FC_HZ = 509e6

# Video framing
CPI_SEC = 0.5
FRAME_HOP_SEC = 0.10
FPS = 10

# Wiener regularization
WIENER_BETA = 3e-3

# Range profile params
NFAST = 4096
HOP_FAST = NFAST // 4
NFFT_RANGE = 2 * NFAST

# Plot limits (train-focused)
MAX_RANGE_M = 1000
MAX_DOPPLER_HZ = 80

# Clutter ridge handling
REMOVE_ZERO_DOPPLER = True
ZERO_DOPPLER_NOTCH_BINS = 1

# Display dynamics
DB_FLOOR = -45
DB_CEIL = 0
IM_INTERP = "bilinear"

# Contrast normalization
NORM_PERCENTILE = 99.5

# Output
OUT_MP4 = "range_doppler_train_zoom.mp4"


# --------------------------
# IO: mmap + CS8 -> complex64
# --------------------------

def mmap_i8(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.int8, mode="r")

def n_cplx(raw_i8: np.ndarray) -> int:
    return raw_i8.size // 2

def read_cs8(raw_i8: np.ndarray, start_samp: int, n_samp: int) -> np.ndarray:
    """
    Read interleaved int8 IQ from memmap and return complex64 normalized by 128.
    Pads with zeros if near EOF (same behavior as your script).
    """
    a = start_samp * 2
    b = a + n_samp * 2
    blk = raw_i8[a:b]
    need = n_samp * 2
    if blk.size < need:
        blk = np.pad(blk, (0, need - blk.size), mode="constant")

    iq = blk.reshape(-1, 2).astype(np.float32, copy=False)
    return ((iq[:, 0] + 1j * iq[:, 1]) / 128.0).astype(np.complex64, copy=False)

def decimate(x: np.ndarray, r: int) -> np.ndarray:
    return x if r <= 1 else x[::r]


# --------------------------
# Delay estimation (unchanged math)
# --------------------------

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def estimate_initial_delay(ref_raw: np.ndarray,
                           sur_raw: np.ndarray,
                           block_size: int = 32768,
                           max_complex_samples: int = 1_000_000) -> int:
    """
    Integer lag estimate near the start using FFT xcorr over multiple blocks.
    Convention (same as your script):
      lag < 0 => sur delayed vs ref by -lag samples
      lag > 0 => sur leads ref by +lag samples
    """
    usable = min(n_cplx(ref_raw), n_cplx(sur_raw), max_complex_samples)
    n_blocks = usable // block_size
    if n_blocks < 1:
        return 0

    nfft = next_pow2(2 * block_size - 1)
    lags = np.arange(-(block_size - 1), block_size, dtype=np.int32)
    peak_lags = np.empty(n_blocks, dtype=np.int32)

    for bi in range(n_blocks):
        start = bi * block_size
        x = read_cs8(ref_raw, start, block_size)
        y = read_cs8(sur_raw, start, block_size)

        x = x - x.mean()
        y = y - y.mean()

        X = np.fft.fft(x, nfft)
        Y = np.fft.fft(y, nfft)
        r_circ = np.fft.ifft(X * np.conj(Y))

        # linear xcorr ordering: [-N+1..-1, 0..N-1]
        r = np.concatenate([r_circ[-(block_size - 1):], r_circ[:block_size]])
        mag = np.abs(r).astype(np.float32)

        mag /= (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
        peak_lags[bi] = int(lags[int(np.argmax(mag))])

    # mode via bincount
    offset = -int(lags.min())
    counts = np.bincount(peak_lags + offset)
    return int(np.argmax(counts) - offset)


# --------------------------
# Signal processing (unchanged)
# --------------------------

def wiener_cancel_freq(ref: np.ndarray, sur: np.ndarray, beta: float, win: np.ndarray) -> np.ndarray:
    X = np.fft.fft(ref * win)
    Y = np.fft.fft(sur * win)
    Sxx = (X * np.conj(X)).real
    Syx = Y * np.conj(X)
    reg = beta * np.mean(Sxx)
    H = Syx / (Sxx + reg + 1e-12)
    y_hat = np.fft.ifft(H * X)
    return (sur - y_hat).astype(np.complex64, copy=False)

def overlapped_range_profiles(ref: np.ndarray, sur_res: np.ndarray,
                              nfast: int, hop: int, nfft: int, win_fast: np.ndarray) -> np.ndarray:
    L = ref.size
    if L < nfast:
        raise ValueError("CPI too short for NFAST")

    M = 1 + (L - nfast) // hop
    ref_v = np.lib.stride_tricks.sliding_window_view(ref, nfast)[::hop][:M]
    sur_v = np.lib.stride_tricks.sliding_window_view(sur_res, nfast)[::hop][:M]

    X = np.fft.fft(ref_v * win_fast, n=nfft, axis=1)
    Y = np.fft.fft(sur_v * win_fast, n=nfft, axis=1)
    return np.fft.ifft(Y * np.conj(X), axis=1).astype(np.complex64, copy=False)

def range_doppler_mag(RP: np.ndarray, remove_zero: bool, notch_bins: int, win_slow: np.ndarray) -> np.ndarray:
    if remove_zero:
        RP = RP - RP.mean(axis=0, keepdims=True)

    RD = np.fft.fftshift(np.fft.fft(RP * win_slow, axis=0), axes=0)

    if notch_bins and notch_bins > 0:
        c = RD.shape[0] // 2
        RD[c - notch_bins:c + notch_bins + 1, :] = 0

    return np.abs(RD).astype(np.float32, copy=False)


# --------------------------
# Plot renderer (reused figure)
# --------------------------

class RDRenderer:
    def __init__(self, extent, fc_hz: float):
        self.fc_hz = fc_hz
        self.fig = plt.figure(figsize=(8, 4.5), dpi=160)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasAgg(self.fig)

        self.im = self.ax.imshow(
            np.zeros((10, 10), dtype=np.float32),
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation=IM_INTERP,
            vmin=DB_FLOOR,
            vmax=DB_CEIL,
        )

        self.ax.set_xlabel("Bistatic range difference (m, approx)")
        self.ax.set_ylabel("Doppler (Hz)")

        c0 = 299_792_458.0
        lam = c0 / self.fc_hz
        self.ax.secondary_yaxis(
            "right",
            functions=(lambda hz: hz * lam, lambda v: v / lam),
        ).set_ylabel("Radial velocity (m/s, approx)")

        cb = self.fig.colorbar(self.im, ax=self.ax)
        cb.set_label("dB")
        self.fig.tight_layout()

    def render(self, mag_db: np.ndarray, title: str) -> np.ndarray:
        self.im.set_data(mag_db)
        self.ax.set_title(title)
        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())
        return buf[..., :3].copy()


# --------------------------
# Main
# --------------------------

def main():
    fs_eff = FS / DECIM

    ref_raw = mmap_i8(REF_PATH)
    sur_raw = mmap_i8(SUR_PATH)

    # delay estimate + correction
    lag0 = estimate_initial_delay(ref_raw, sur_raw)
    ref_off = max(0, lag0)
    sur_off = max(0, -lag0)

    print(f"Estimated start delay (ref vs sur) lag: {lag0} samples ({lag0/FS:.6f} s)")
    if lag0 < 0:
        print(f"Applying correction: shifting SUR forward by {sur_off} samples")
    elif lag0 > 0:
        print(f"Applying correction: shifting REF forward by {ref_off} samples")
    else:
        print("Applying correction: none (lag=0)")

    n_total = min(n_cplx(ref_raw) - ref_off, n_cplx(sur_raw) - sur_off)
    n_total = n_total
    cpi_samp = int(CPI_SEC * FS)
    hop_samp = int(FRAME_HOP_SEC * FS)
    if cpi_samp <= 0 or hop_samp <= 0:
        raise ValueError("CPI_SEC and FRAME_HOP_SEC must be > 0")

    # windows (decimation-aware)
    cpi_len_eff = cpi_samp // DECIM
    if cpi_len_eff <= 0:
        raise ValueError("DECIM too large for CPI length")

    win_cpi = np.hanning(cpi_len_eff).astype(np.float32)
    win_fast = np.hanning(NFAST).astype(np.float32)

    M = 1 + (cpi_len_eff - NFAST) // HOP_FAST
    if M < 2:
        raise ValueError("CPI too short for chosen NFAST/HOP_FAST after decimation")
    win_slow = np.hanning(M).astype(np.float32)[:, None]

    n_frames = 1 + max(0, (n_total - cpi_samp) // hop_samp)

    # axes + constant crop masks
    c0 = 299_792_458.0
    dt = 1.0 / fs_eff
    rng_m_full = c0 * (np.arange(NFFT_RANGE) * dt)
    max_bin = int(np.searchsorted(rng_m_full, MAX_RANGE_M))
    max_bin = int(np.clip(max_bin, 16, NFFT_RANGE))
    rng_m = rng_m_full[:max_bin]

    PRI = HOP_FAST / fs_eff
    fd_full = np.fft.fftshift(np.fft.fftfreq(M, d=PRI))
    dopp_mask = np.abs(fd_full) <= MAX_DOPPLER_HZ
    fd_hz = fd_full[dopp_mask]

    extent = [rng_m[0], rng_m[-1], fd_hz[0], fd_hz[-1]]
    renderer = RDRenderer(extent=extent, fc_hz=FC_HZ)

    # info (same as your script)
    lam = c0 / FC_HZ
    print(f"Rule-of-thumb Doppler for 10 m/s at {FC_HZ/1e6:.1f} MHz: ~{10.0/lam:.1f} Hz")

    writer = imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8)

    start = 0
    with tqdm(total=n_frames, desc="Processing frames") as pbar:
        for _ in range(n_frames):
            if start + cpi_samp > n_total:
                break

            frame_t0 = start / FS

            ref = read_cs8(ref_raw, start + ref_off, cpi_samp)
            sur = read_cs8(sur_raw, start + sur_off, cpi_samp)

            ref = decimate(ref, DECIM)
            sur = decimate(sur, DECIM)

            ref = ref - ref.mean()
            sur = sur - sur.mean()

            sur_res = wiener_cancel_freq(ref, sur, WIENER_BETA, win_cpi)

            RP = overlapped_range_profiles(ref, sur_res, NFAST, HOP_FAST, NFFT_RANGE, win_fast)

            mag = range_doppler_mag(RP, REMOVE_ZERO_DOPPLER, ZERO_DOPPLER_NOTCH_BINS, win_slow)

            mag = mag[:, :max_bin]
            mag = mag[dopp_mask, :]

            mag_db = 20.0 * np.log10(mag + 1e-12)
            ref_level = np.percentile(mag_db, NORM_PERCENTILE)
            mag_db = np.clip(mag_db - ref_level, DB_FLOOR, DB_CEIL)

            title = f"Range–Doppler (Wiener suppressed)  t={frame_t0:.2f}s  CPI={CPI_SEC:.2f}s"
            writer.append_data(renderer.render(mag_db, title))

            start += hop_samp
            pbar.update(1)

    writer.close()
    print("Done. Output:", OUT_MP4)


if __name__ == "__main__":
    main()
