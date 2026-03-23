#!/usr/bin/env python3
"""
PASSIVE RADAR PROCESSING
- NLMS / Wiener / ECA clutter suppression
- Correlation MP4 (before and after suppression)
- Range–Doppler CAF MP4 (post-suppression)
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from numba import njit
from tqdm import tqdm


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
# Initial offset handling
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
    pos = r_circ[:N]
    neg = r_circ[-(N - 1):]
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


def estimate_initial_delay_samples(
    ref_path: str,
    sur_path: str,
    block_size: int = 32768,
    max_complex_samples: int = 1_000_000,
) -> int:
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


def estimate_delay_from_open_files(
    fx,
    fy,
    block_size: int = 32768,
    max_complex_samples: int = 1_000_000,
) -> int:
    """
    Estimate integer lag (samples) between ref and sur using data starting
    from the CURRENT file positions of already-open file handles.

    Convention:
      - lag < 0 => sur is delayed vs ref by -lag samples
      - lag > 0 => sur leads ref by +lag samples
    """
    pos_x = fx.tell()
    pos_y = fy.tell()

    try:
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
def nlms_block(x, d, w, mu, eps, adapt):
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

        if adapt:
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

def wiener_block(x, d, L=32, reg=1e-3, win=None):
    """
    Wiener FIR clutter canceller using:

        w    = (R_RR + lambda*I)^(-1) r_RS
        R_RR = (1/Nv) X^H X
        r_RS = (1/Nv) X^H d

    where each row of X is:
        [x[n], x[n-1], ..., x[n-L+1]]
    """
    x = np.asarray(x, dtype=np.complex64).ravel()
    d = np.asarray(d, dtype=np.complex64).ravel()



    N = d.size
    L = int(L)



    Nv = N - L + 1

    # Build delayed reference matrix
    X = np.empty((Nv, L), dtype=np.complex64)
    for i in range(Nv):
        n = i + L - 1
        X[i, :] = x[n - L + 1 : n + 1][::-1]

    d_valid = d[L - 1:]


    R_RR = (X.conj().T @ X) / Nv
    r_RS = (X.conj().T @ d_valid) / Nv

    reg_eff = float(reg) * float(np.real(np.trace(R_RR)) / max(L, 1))
    R_loaded = R_RR + reg_eff * np.eye(L, dtype=np.complex64)

    w = np.linalg.solve(R_loaded, r_RS).astype(np.complex64, copy=False)

    y_valid = (X @ w).astype(np.complex64, copy=False)
    e_valid = (d_valid - y_valid).astype(np.complex64, copy=False)

    # Keep output same length as input block
    y_hat = np.zeros(N, dtype=np.complex64)
    e = d.copy()

    y_hat[L - 1:] = y_valid
    e[L - 1:] = e_valid

    return y_hat, e, {
        "w": w,
        "R_RR": R_RR,
        "r_RS": r_RS,
        "reg_eff": reg_eff,
        "L": L,
    }

def eca_block(x, d, K=32, delay0=0):
    """
    Time-domain ECA via least squares.
    """
    N = d.size
    if delay0 < 0:
        raise ValueError("delay0 must be >= 0")
    if K < 1:
        raise ValueError("K must be >= 1")
    if delay0 + K > N:
        return np.zeros_like(d), d.copy(), {"w": np.zeros(K, dtype=np.complex64)}

    Xmat = np.empty((N, K), dtype=np.complex64)
    Xmat[:] = 0.0 + 0.0j

    for k in range(K):
        sh = delay0 + k
        Xmat[sh:, k] = x[: N - sh]

    w, _, _, _ = np.linalg.lstsq(Xmat, d, rcond=None)
    y_hat = (Xmat @ w).astype(np.complex64)
    e = (d - y_hat).astype(np.complex64)

    return y_hat, e, {"w": w.astype(np.complex64)}


def clutter_cancel_block(x, d, state, method, params):
    if state is None:
        state = {}

    method = method.lower()

    if method == "nlms":
        w = state.get("w", None)
        if w is None:
            L = int(params.get("L", 1))
            w = np.zeros(L, dtype=np.complex64)

        mu = float(params.get("mu", 0.1))
        eps = float(params.get("eps", 1e-6))
        adapt_blocks = int(params.get("adapt_blocks", -1))

        block_idx = int(state.get("block_idx", 0))
        adapt = (adapt_blocks < 0) or (block_idx < adapt_blocks)

        y_hat, e, w = nlms_block(x, d, w, mu, eps, adapt)

        return y_hat, e, {
            "w": w,
            "block_idx": block_idx + 1,
        }

    elif method == "wiener":
        L = int(params.get("L", 32))
        reg = float(params.get("reg", 1e-3))
        win = params.get("win", None)

        y_hat, e, st = wiener_block(x, d, L=L, reg=reg, win=win)
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
    (l1,) = ax.plot([], [], label="Before suppression")
    (l2,) = ax.plot([], [], label="After suppression")

    ax.set_xlim(float(x_axis_m[0]), float(x_axis_m[-1]))

    all_vals = []
    if len(before_frames) > 0:
        all_vals.append(np.concatenate(before_frames))
    if len(after_frames) > 0:
        all_vals.append(np.concatenate(after_frames))

    if len(all_vals) == 0:
        y_min, y_max = -120.0, 0.0
    else:
        all_vals = np.concatenate(all_vals)
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        if y_max - y_min < 1.0:
            y_min -= 1.0
            y_max += 1.0

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Range (m, approx from lag)")
    ax.set_ylabel("Correlation (dB)")
    ax.legend()
    ax.grid(True)

    def upd(i):
        l1.set_data(x_axis_m, before_frames[i])
        l2.set_data(x_axis_m, after_frames[i])
        ax.set_title(f"Correlation Block {i + 1}/{len(before_frames)}")
        return l1, l2

    FuncAnimation(fig, upd, frames=len(before_frames), blit=True).save(
        out_path, writer=FFMpegWriter(fps=fps)
    )
    plt.close(fig)


def save_caf_mp4(caf_corr, x_axis_m, Nslow, Tblock, out_path, fps):
    if len(caf_corr) < Nslow:
        print(f"Not enough blocks for CAF: have {len(caf_corr)}, need Nslow={Nslow}. Skipping CAF MP4.")
        return

    doppler_axis = np.fft.fftshift(np.fft.fftfreq(Nslow, d=Tblock))

    MAX_DOPPLER_HZ = 80.0
    dmask = np.abs(doppler_axis) <= MAX_DOPPLER_HZ
    doppler_zoom = doppler_axis[dmask]

    zero_bin = np.argmin(np.abs(doppler_zoom))
    print("Zero-Doppler bin:", zero_bin, "freq:", doppler_zoom[zero_bin], "Hz")

    n_caf_frames = len(caf_corr) - Nslow + 1
    frame_step = 1

    S0 = np.stack(caf_corr[0:Nslow], axis=0)
    RD0 = np.fft.fftshift(np.fft.fft(S0, axis=0), axes=0)
    RD0 = RD0[dmask, :]

    lo0 = max(0, zero_bin - 4)
    hi0 = min(RD0.shape[0], zero_bin + 6)
    RD0[lo0:hi0, :] = 0

    frame0 = 20 * np.log10(np.abs(RD0) + 1e-12)
    frame0 -= np.max(frame0)

    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [x_axis_m[0], x_axis_m[-1], doppler_zoom[0], doppler_zoom[-1]]

    im = ax.imshow(
        frame0.astype(np.float32),
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-15.0,
        vmax=0.0,
    )

    ax.set_xlabel("Range (m, approx from lag)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title("Range–Doppler CAF (After suppression)")

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, out_path, dpi=100):
        for i in tqdm(
            range(Nslow - 1, len(caf_corr), frame_step),
            total=(n_caf_frames + frame_step - 1) // frame_step,
            desc="Building CAF frames",
        ):
            S = np.stack(caf_corr[i - (Nslow - 1): i + 1], axis=0)
            RD = np.fft.fftshift(np.fft.fft(S, axis=0), axes=0)
            RD = RD[dmask, :]

            lo = max(0, zero_bin -4)
            hi = min(RD.shape[0], zero_bin+6)
            RD[lo:hi, :] = 0

            frame_db = 20 * np.log10(np.abs(RD) + 1e-12)
            #print(np.max(frame_db))
            frame_db -= np.max(frame_db)
            

            im.set_data(frame_db.astype(np.float32))
            writer.grab_frame()

    plt.close(fig)


def main():
    date = "bev"
    fid_x_path = "/Users/ibrahimsweidan/Downloads/10_509.0MHz_20260310_193719/ref.cs8"
    fid_y_path = "/Users/ibrahimsweidan/Downloads/10_509.0MHz_20260310_193719/sur.cs8"
    if not os.path.exists(fid_x_path):
        raise FileNotFoundError("Input file not found")

    cpi = 0.5
    length = 32 * 1024
    Fs = 10e6
    Nslow = int(cpi * Fs / length)
    print(Nslow)

    win = np.hanning(length).astype(np.float32)

    CANCEL_METHOD = "wiener"   # "nlms" | "wiener" | "eca"

    cancel_params = {
        # NLMS params
        "L": 30,
        "mu": 0.1,
        "eps": 1e-6,
        "adapt_blocks": -1,

        # Wiener params
        "L": 15,
        "reg": 1e-2,
        "win": win,

        # ECA params
        "K": 15,
        "delay0": 0,
    }

    cancel_state = None

    corr_mp4 = "PassiveRadar_Correlation.mp4"
    caf_mp4 = date + CANCEL_METHOD + ".mp4"

    seconds_per_recheck = 1
    blocks_per_recheck = max(1, int(round(seconds_per_recheck * Fs / length)))

    recheck_window_samples = int(Fs * 1.0)
    recheck_corr_block_size = 32768
    min_recheck_shift_samples = 2

    R_after_cplx = []
    R_before_db = []
    R_after_db = []

    Tblock = length / Fs

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

    n_ref = file_num_complex_samples(fid_x_path) - ref_off
    n_sur = file_num_complex_samples(fid_y_path) - sur_off
    n_avail = min(n_ref, n_sur)
    max_blocks = int(n_avail // length)
    if max_blocks <= 0:
        raise RuntimeError("No full blocks available after applying initial offset.")

    with open(fid_x_path, "rb") as fx, open(fid_y_path, "rb") as fy:
        if ref_off:
            fx.seek(ref_off * 2, os.SEEK_SET)
        if sur_off:
            fy.seek(sur_off * 2, os.SEEK_SET)

        bi = 0
        with tqdm(total=max_blocks, desc="Processing blocks") as pbar:
            while bi < max_blocks:
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

                xw = x
                yw = y_clean

                Rb = xcorr_fft(y, xw)    # before suppression
                Ra = xcorr_fft(yw, xw)   # after suppression

                R_before_db.append(20 * np.log10(np.abs(Rb) + 1e-12))
                R_after_db.append(20 * np.log10(np.abs(Ra) + 1e-12))
                R_after_cplx.append(Ra.astype(np.complex64))

                bi += 1
                pbar.update(1)

    if len(R_after_cplx) == 0:
        raise RuntimeError("No processed blocks produced any CAF data.")

    c = 299_792_458.0
    lags_s = np.arange(-(length - 1), length) / Fs
    range_m = c * lags_s / 2.0

    mask = (range_m >= 0.0) & (range_m <= 200.0)
    x_crop = range_m[mask]

    caf_corr = [R_after_cplx[k][mask] for k in range(len(R_after_cplx))]

    if len(caf_corr) < Nslow:
        raise RuntimeError(f"Not enough CAF blocks: have {len(caf_corr)}, need {Nslow}")

    fps = (len(caf_corr) - Nslow + 1) / (len(caf_corr) * length / Fs)

    save_caf_mp4(caf_corr, x_crop, Nslow, Tblock, caf_mp4, fps)

    mask = (range_m >= -512.0) & (range_m <= 512.0)
    x_crop = range_m[mask]
    before_frames = [R_before_db[k][mask] for k in range(len(R_before_db))]
    after_frames = [R_after_db[k][mask] for k in range(len(R_after_db))]

    save_correlation_mp4(x_crop, before_frames, after_frames, corr_mp4, fps)

    print("Finished:")
    print(" ", corr_mp4)
    print(" ", caf_mp4)


if __name__ == "__main__":
    main()