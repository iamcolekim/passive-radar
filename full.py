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
from tqdm import tqdm

# User Imports
from obj_tracking import SimpleTracker


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


# --------------------------
# Added: initial offset handling (correlate-style)
# --------------------------

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

def file_num_complex_samples(path: str) -> int:
    return os.path.getsize(path) // 2

def read_cs8_iq(path: str, start_samp: int, num_samp: int) -> np.ndarray:
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

def estimate_initial_delay_samples(ref_path: str,
                                  sur_path: str,
                                  block_size: int = 32768,
                                  max_complex_samples: int = 1_000_000) -> int:
    """
    Estimate integer lag (samples) between ref and sur near the start of the files.
    Uses multiple blocks and returns the MODE of peak lags.

    Convention for returned lag:
      - lag < 0 => sur is delayed vs ref by -lag samples
      - lag > 0 => sur leads ref by +lag samples
    """
    n_total = min(file_num_complex_samples(ref_path), file_num_complex_samples(sur_path))
    usable = min(n_total, max_complex_samples)
    n_blocks = usable // block_size
    if n_blocks < 1:
        return 0

    nfft = next_pow2(2 * block_size - 1)
    lags = np.arange(-(block_size - 1), block_size, dtype=np.int32)

    peak_lags = []
    for bi in range(n_blocks):
        start = bi * block_size
        x = read_cs8_iq(ref_path, start, block_size)
        y = read_cs8_iq(sur_path, start, block_size)

        x = x - np.mean(x)
        y = y - np.mean(y)

        r = linear_xcorr_fft(x, y, nfft)
        mag = np.abs(r).astype(np.float32)
        mag /= (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

        peak_idx = int(np.argmax(mag))
        peak_lags.append(int(lags[peak_idx]))

    peak_lags = np.array(peak_lags, dtype=np.int32)
    offset = -int(lags.min())
    counts = np.bincount(peak_lags + offset)
    mode_lag = int(np.argmax(counts) - offset)
    return mode_lag

def estimate_delay_from_open_files(fx,
                                   fy,
                                   block_size: int = 32768,
                                   max_complex_samples: int = 1_000_000) -> int:
    """
    Estimate integer lag (samples) between ref and sur using data starting
    from the CURRENT file positions of already-open file handles.

    Convention:
      - lag < 0 => sur is delayed vs ref by -lag samples
      - lag > 0 => sur leads ref by +lag samples
    """
    # Save current positions
    pos_x = fx.tell()
    pos_y = fy.tell()

    try:
        # Available complex samples from current positions
        avail_x = (os.fstat(fx.fileno()).st_size - pos_x) // 2
        avail_y = (os.fstat(fy.fileno()).st_size - pos_y) // 2
        usable = min(avail_x, avail_y, max_complex_samples)

        n_blocks = usable // block_size
        if n_blocks < 1:
            return 0

        nfft = next_pow2(2 * block_size - 1)
        lags = np.arange(-(block_size - 1), block_size, dtype=np.int32)

        peak_lags = []
        for _ in range(n_blocks):
            x = read_cs8_block(fx, block_size)
            y = read_cs8_block(fy, block_size)
            if x is None or y is None:
                break

            x = x - np.mean(x)
            y = y - np.mean(y)

            r = linear_xcorr_fft(x, y, nfft)
            mag = np.abs(r).astype(np.float32)
            mag /= (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)

            peak_idx = int(np.argmax(mag))
            peak_lags.append(int(lags[peak_idx]))

        if len(peak_lags) == 0:
            return 0

        peak_lags = np.array(peak_lags, dtype=np.int32)
        offset = -int(lags.min())
        counts = np.bincount(peak_lags + offset)
        mode_lag = int(np.argmax(counts) - offset)
        return mode_lag

    finally:
        # Restore positions so this estimate does not consume data
        fx.seek(pos_x, os.SEEK_SET)
        fy.seek(pos_y, os.SEEK_SET)


def apply_relative_file_shift(fx, fy, lag_samples: int):
    """
    Apply integer-sample correction at the CURRENT file positions.

    lag_samples convention:
      - lag < 0 => sur delayed vs ref by -lag => advance SUR
      - lag > 0 => sur leads ref by +lag => advance REF
    """
    if lag_samples < 0:
        shift = -lag_samples
        fy.seek(shift * 2, os.SEEK_CUR)
        print(f"  Applying periodic correction: shifting SUR forward by {shift} samples")
    elif lag_samples > 0:
        shift = lag_samples
        fx.seek(shift * 2, os.SEEK_CUR)
        print(f"  Applying periodic correction: shifting REF forward by {shift} samples")
    else:
        print("  Applying periodic correction: none (lag=0)")

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
        err = d[n] - 0.005*y
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

def wiener_block(x, d, reg=1e-3):
    """
    Drop-in Wiener canceller with:
      - power-normalized regularization (reg becomes dimensionless)
      - mild smoothing of H(f) across blocks (persistent internal state)

    Call signature unchanged: wiener_block(x, d, reg)
    Returns unchanged: y_hat, e, state_dict
    """

    # --- persistent state stored as function attributes (no wrapper/main changes needed) ---
    if not hasattr(wiener_block, "_H_prev") or wiener_block._H_prev is None:
        wiener_block._H_prev = None

    # Smoothing factor (fixed here since you don't want to change wrapper/main)
    alpha = 0.95   # 0.90..0.98 (higher = smoother, fewer bursts)

    eps = 1e-12
    N = d.size
    X = np.fft.fft(x, n=N)
    D = np.fft.fft(d, n=N)

    Sxx = (X * np.conj(X)).real          # |X|^2
    Sdx = D * np.conj(X)                 # D X*

    # --- power-normalized diagonal loading ---
    # reg is now dimensionless; effective loading scales with average reference power
    pxx = float(np.mean(Sxx)) + eps
    reg_eff = float(reg) * pxx

    H_new = Sdx / (Sxx + reg_eff)

    # --- smooth across blocks to reduce "bursty" artifacts ---
    H_prev = wiener_block._H_prev
    if H_prev is None or H_prev.shape != H_new.shape:
        H = H_new
    else:
        H = alpha * H_prev + (1.0 - alpha) * H_new

    wiener_block._H_prev = H.astype(np.complex64)

    Yhat = np.fft.ifft(H * X).astype(np.complex64)
    e = (d - Yhat).astype(np.complex64)

    # Return extra debug info (ignored by your pipeline unless you use it)
    return Yhat, e, {"H": H.astype(np.complex64), "reg_eff": reg_eff, "pxx": pxx, "alpha": alpha}


def eca_block(x, d, K=32, delay0=0):
    """
    Time-domain ECA (Extended Cancellation Algorithm) via least squares.
    Builds a tap-delay matrix from x and solves min_w ||d - Xmat w||.

    K:     number of taps (columns)
    delay0: starting delay in samples (>=0). Use delay0>0 to avoid self-leakage at 0 lag.
    Returns: y_hat, e, state
    """
    N = d.size
    if delay0 < 0:
        raise ValueError("delay0 must be >= 0")
    if K < 1:
        raise ValueError("K must be >= 1")
    if delay0 + K > N:
        # Not enough samples in this block to build the matrix
        return np.zeros_like(d), d.copy(), {"w": np.zeros(K, dtype=np.complex64)}

    # Build Xmat: each column is a delayed version of x
    # Xmat[n, k] = x[n - (delay0 + k)]  (with valid indices only)
    Xmat = np.empty((N, K), dtype=np.complex64)
    Xmat[:] = 0.0 + 0.0j

    for k in range(K):
        sh = delay0 + k
        Xmat[sh:, k] = x[: N - sh]

    # Least squares solve (complex)
    w, _, _, _ = np.linalg.lstsq(Xmat, d, rcond=None)
    y_hat = (Xmat @ w).astype(np.complex64)
    e = (d - y_hat).astype(np.complex64)

    return y_hat, e, {"w": w.astype(np.complex64)}


def clutter_cancel_block(x, d, state, method, params):
    """
    Unified clutter canceller wrapper.

    Inputs:
      x: ref block (complex)
      d: sur block (complex)
      state: dict (can be None on first call)
      method: "nlms" | "wiener" | "eca"
      params: dict of method-specific params

    Returns:
      y_hat, e_clean, new_state
    """
    if state is None:
        state = {}

    method = method.lower()

    if method == "nlms":
        # Reuse your existing nlms_block and its weight vector w
        w = state.get("w", None)
        if w is None:
            L = int(params.get("L", 1))
            w = np.zeros(L, dtype=np.complex64)

        mu = float(params.get("mu", 0.1))
        eps = float(params.get("eps", 1e-6))

        y_hat, e, w = nlms_block(x, d, w, mu, eps)
        return y_hat, e, {"w": w}

    elif method == "wiener":
        reg = float(params.get("reg", 1e-3))
        y_hat, e, st = wiener_block(x, d, reg=reg)
        return y_hat, e, st

    elif method == "eca":
        K = int(params.get("K", 32))
        delay0 = int(params.get("delay0", 0))
        y_hat, e, st = eca_block(x, d, K=K, delay0=delay0)
        return y_hat, e, st

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nlms', 'wiener', or 'eca'.")

def save_correlation_mp4(x_axis_m, before_frames, after_frames, out_path, fps):
    fig, ax = plt.subplots(figsize=(7, 7))
    (l1,) = ax.plot([], [], label="Before NLMS")
    (l2,) = ax.plot([], [], label="After NLMS")

    ax.set_xlim(-500, 500)
    ax.set_ylim(-20, 200000)
    ax.set_xlabel("Range (m, approx from lag)")
    ax.set_ylabel("Correlation (dB)")
    ax.legend()
    ax.grid(True)

    def upd(i):
        l1.set_data(x_axis_m, before_frames[i])
        l2.set_data(x_axis_m, after_frames[i])
        ax.set_title(f"Correlation Block {i+1}")
        return l1, l2

    FuncAnimation(fig, upd, frames=len(before_frames), blit=True).save(
        out_path, writer=FFMpegWriter(fps=fps)
    )
    plt.close(fig)


def save_caf_mp4(caf_corr, x_axis_m, Nslow, Tblock, out_path, fps, tracker=None):
    """
    Builds Range–Doppler CAF frames from caf_corr (list of complex correlation vectors)
    and saves an MP4 without storing all frames in memory.

    Behavior:
      - sliding window of Nslow
      - FFT over slow-time axis
      - fftshift
      - Doppler crop
      - dB magnitude
    """
    if len(caf_corr) < Nslow:
        print(f"Not enough blocks for CAF: have {len(caf_corr)}, need Nslow={Nslow}. Skipping CAF MP4.")
        return

    doppler_axis = np.fft.fftshift(np.fft.fftfreq(Nslow, d=Tblock))

    # Doppler zoom to +/- 60 Hz
    MAX_DOPPLER_HZ = 60.0
    dmask = np.abs(doppler_axis) <= MAX_DOPPLER_HZ
    doppler_zoom = doppler_axis[dmask]

    n_caf_frames = len(caf_corr) - Nslow + 1
    frame_step = 5  # keep every 10th frame in the video

    # Build first frame so we can initialize imshow
    S0 = np.stack(caf_corr[0:Nslow], axis=0)
    RD0 = np.fft.fftshift(np.fft.fft(S0, axis=0), axes=0)
    RD0 = RD0[dmask, :]
    frame0 = (20 * np.log10(np.abs(RD0) + 1e-12)).astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [x_axis_m[0], x_axis_m[-1], doppler_zoom[0], doppler_zoom[-1]]

    im = ax.imshow(
        frame0,
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    ax.set_xlabel("Range (m, approx from lag)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title("Range–Doppler CAF (After NLMS)")

    # a marker to represent our Kalman Filter estimate
    tracker_point, = ax.plot([], [], 'ro', markersize=12, markeredgewidth=3, label='Kalman Track')
    ax.legend(loc="upper right")

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, out_path, dpi=100):
        for i in tqdm(
            range(Nslow - 1, len(caf_corr), frame_step),
            total=(n_caf_frames + frame_step - 1) // frame_step,
            desc="Building CAF frames",
        ):
            S = np.stack(caf_corr[i - (Nslow - 1): i + 1], axis=0)
            RD = np.fft.fftshift(np.fft.fft(S, axis=0), axes=0)
            RD = RD[dmask, :]  # crop Doppler
            
            # Genearte visual frame (db)
            frame_db = (20 * np.log10(np.abs(RD) + 1e-12)).astype(np.float32)

            im.set_data(frame_db)

            # run tracker logic
            if tracker is not None:
                caf_power = np.abs(RD.T)**2 
                
                # Find the exact index of 0 Hz in the cropped array
                d_zero_idx = int(np.argmin(np.abs(doppler_zoom)))
                
                track_res = tracker.process_caf_frame(caf_power, i, doppler_0_idx=d_zero_idx)
                
                if track_res['active']:
                    # Pluck the SI units directly from the physics state
                    r_meters = track_res['state']['bistatic_range_m']
                    d_hz = track_res['state']['doppler_hz']
                    
                    # Plot the hollow red dot
                    tracker_point.set_data([r_meters], [d_hz])
                else:
                    tracker_point.set_data([], []) # Hide marker if lost
            writer.grab_frame()

    plt.close(fig)

def main():
    fid_x_path = "/Users/colekim/Documents/Y4_Work/Capstone/DevWork/509.0MHz_20260217_184044/ref.cs8"
    fid_y_path = "/Users/colekim/Documents/Y4_Work/Capstone/DevWork/509.0MHz_20260217_184044/surv.cs8"
    if not os.path.exists(fid_x_path):
        raise FileNotFoundError("Input file not found")

    length = 2 * 4096
    Fs = 10e6

    Nslow = 1000
    L = 80
    mu = 0.1
    eps = 1e-6

    CANCEL_METHOD = "nlms"   # "nlms" | "wiener" | "eca"

    cancel_params = {
        # NLMS params
        "L": L,
        "mu": mu,
        "eps": eps,

        # Wiener params (only used if method=="wiener")
        "reg": 1e-2,

        # ECA params (only used if method=="eca")
        "K": 3,
        "delay0": 0,
    }

    cancel_state = None


    fps = 1000
    corr_mp4 = "PassiveRadar_Correlation.mp4"
    caf_mp4 = "PassiveRadar_RangeDoppler_CAF.mp4"

    # Re-check offset about once per second of data
    seconds_per_recheck = 1.0
    blocks_per_recheck = max(1, int(round(seconds_per_recheck * Fs / length)))

    # Use ~1 second of data for the offset estimator too
    recheck_window_samples = int(Fs * 1.0)
    recheck_corr_block_size = 32768

    # Optional: ignore tiny jittery corrections
    min_recheck_shift_samples = 2

    # Added: fast-time Hann window for correlation
    win = np.hanning(length).astype(np.float32)

    w = np.zeros(L, dtype=np.complex64)
    R_after_cplx = []

    Tblock = length / Fs

    # --- Estimate & apply initial file offset ---
    lag0 = estimate_initial_delay_samples(fid_x_path, fid_y_path)
    ref_off = max(0, lag0)
    sur_off = max(0, -lag0)

    print(f"Estimated start delay (ref vs sur) lag: {lag0} samples  ({lag0 / Fs:.6f} s)")
    if lag0 < 0:
        print(f"Applying correction: shifting SUR forward by {sur_off} samples")
    elif lag0 > 0:
        print(f"Applying correction: shifting REF forward by {ref_off} samples")
    else:
        print("Applying correction: none (lag=0)")

    # Cap max_record to available full blocks after initial offsets
    n_ref = file_num_complex_samples(fid_x_path) - ref_off
    n_sur = file_num_complex_samples(fid_y_path) - sur_off
    n_avail = min(n_ref, n_sur)
    max_blocks = int(n_avail // length)
    if max_blocks <= 0:
        raise RuntimeError("No full blocks available after applying initial offset.")

    with open(fid_x_path, "rb") as fx, open(fid_y_path, "rb") as fy:
        # Apply initial offset
        if ref_off:
            fx.seek(ref_off * 2, os.SEEK_SET)
        if sur_off:
            fy.seek(sur_off * 2, os.SEEK_SET)

        bi = 0
        with tqdm(total=max_blocks, desc="Processing blocks") as pbar:
            while True:
                # Periodic offset re-check
                if bi > 0 and (bi % blocks_per_recheck == 0):
                    lag_now = estimate_delay_from_open_files(
                        fx,
                        fy,
                        block_size=recheck_corr_block_size,
                        max_complex_samples=recheck_window_samples,
                    )

                    print(
                        f"\nPeriodic offset check at block {bi}: "
                        f"lag = {lag_now} samples ({lag_now / Fs:.6f} s)"
                    )

                    if abs(lag_now) >= min_recheck_shift_samples:
                        apply_relative_file_shift(fx, fy, lag_now)
                    else:
                        print("  Skipping correction: lag below threshold")

                x = read_cs8_block(fx, length)
                y = read_cs8_block(fy, length)
                if x is None or y is None:
                    break

                x -= x.mean()
                y -= y.mean()

                _, y_clean, cancel_state = clutter_cancel_block(
                    x, y, cancel_state, CANCEL_METHOD, cancel_params
                )

                # Apply Hann window before correlation
                xw = x 
                yw = y_clean 

                Ra = xcorr_fft(yw, xw)
                R_after_cplx.append(Ra.astype(np.complex64))

                bi += 1
                pbar.update(1)

    if len(R_after_cplx) == 0:
        raise RuntimeError("No processed blocks produced any CAF data.")

    # --- Convert lag axis -> range (meters), and crop to 0..1000 m ---
    c = 299_792_458.0
    lags_s = np.arange(-(length - 1), length) / Fs
    range_m = c * lags_s / 2.0

    mask = (range_m >= 0.0) & (range_m <= 1000.0)
    x_crop = range_m[mask]

    caf_corr = []
    for k in range(len(R_after_cplx)):
        caf_corr.append(R_after_cplx[k][mask])
    
    # Instantiate a tracker and pass it to the CAF MP4 generator
    target_tracker = SimpleTracker(fs=Fs, N=Nslow, fc=509e6, Tblock=Tblock)
    save_caf_mp4(caf_corr, x_crop, Nslow, Tblock, caf_mp4, fps, tracker=target_tracker)

    print("Finished:")
    print(" ", corr_mp4)
    print(" ", caf_mp4)


if __name__ == "__main__":
    main()
