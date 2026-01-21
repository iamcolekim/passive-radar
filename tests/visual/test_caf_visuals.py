import os
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# MUST INSTALL 'ffmpeg' FOR VIDEO OUTPUT
from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys

# Ensure project root is on sys.path for direct script runs
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from caf import cross_ambiguity


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _ensure_artifact_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _artifact_name(prefix: str, params: dict, ext: str) -> str:
    param_str = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{prefix}_{param_str}_{_timestamp()}.{ext}"


def _make_signal(n: int, fs: float, delay: int, doppler_hz: float):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    t = np.arange(n, dtype=np.float64) / fs
    y = np.zeros_like(x)
    if delay >= 0:
        y[delay:] = x[: n - delay] * np.exp(1j * 2 * np.pi * doppler_hz * t[: n - delay])
    else:
        y[: n + delay] = x[-delay:] * np.exp(1j * 2 * np.pi * doppler_hz * t[-delay:])
    return x, y


def _apply_frequency_shift(x: np.ndarray, fs: float, offset_hz: float, start_sample: int = 0) -> np.ndarray:
    if not offset_hz:
        return x
    n = x.shape[0]
    t = (np.arange(n, dtype=np.float64) + start_sample) / fs
    return x * np.exp(-1j * 2 * np.pi * offset_hz * t)


def test_caf_heatmap_artifact():
    _ensure_artifact_dir()

    fs = 2000.0
    n = 256
    delay = 8
    doppler_hz = 80.0
    method = "batch"

    x, y = _make_signal(n, fs, delay, doppler_hz)
    result = cross_ambiguity(x, y, fs, method=method, convention="centered")

    caf_db = 20 * np.log10(result.caf_mag + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(
        caf_db,
        aspect="auto",
        origin="lower",
        extent=[result.delay_samples[0], result.delay_samples[-1], result.doppler_hz[0], result.doppler_hz[-1]],
        cmap="viridis",
    )
    ax.set_title("CAF Heatmap (dB)")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")
    fig.colorbar(im, ax=ax)

    params = {"m": method, "N": n, "fs": int(fs), "d": delay, "fd": int(doppler_hz)}
    filename = _artifact_name("caf_heatmap", params, "png")
    fig.savefig(os.path.join(ARTIFACT_DIR, filename), bbox_inches="tight")
    plt.close(fig)


def test_caf_video_artifact():
    _ensure_artifact_dir()

    fs = 2000.0
    n = 256
    method = "fourier_lag_product"

    delays = [2, 6, 10, 14, 18]
    dopplers = [20, 40, 60, 80, 100]

    frames = []
    total_frames = len(delays)
    for frame_idx, (delay, doppler_hz) in enumerate(zip(delays, dopplers), start=1):
        x, y = _make_signal(n, fs, delay, doppler_hz)
        result = cross_ambiguity(x, y, fs, method=method, convention="centered")
        caf_db = 20 * np.log10(result.caf_mag + 1e-12)
        frames.append((caf_db, result))

        pct = (frame_idx / total_frames) * 100.0
        bar_len = 30
        filled = int(round((pct / 100.0) * bar_len))
        empty = bar_len - filled
        print(
            f"[{'#' * filled}{'-' * empty}] {pct:5.1f}% ({frame_idx}/{total_frames} frames)",
            end="\r",
            flush=True,
        )
        if frame_idx == total_frames:
            print()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(
        frames[0][0],
        aspect="auto",
        origin="lower",
        extent=[frames[0][1].delay_samples[0], frames[0][1].delay_samples[-1], frames[0][1].doppler_hz[0], frames[0][1].doppler_hz[-1]],
        cmap="viridis",
    )
    ax.set_title("CAF Heatmap (dB)")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")

    def update(i):
        caf_db, result = frames[i]
        im.set_data(caf_db)
        im.set_extent(
            [result.delay_samples[0], result.delay_samples[-1], result.doppler_hz[0], result.doppler_hz[-1]]
        )
        ax.set_title(f"CAF Heatmap (dB) frame {i+1}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=400, blit=True)

    params = {"m": method, "N": n, "fs": int(fs), "frames": len(frames)}
    filename = _artifact_name("caf_heatmap", params, "mp4")
    anim.save(os.path.join(ARTIFACT_DIR, filename), writer=FFMpegWriter(fps=2))
    plt.close(fig)


def _read_cs8_block(f, num_complex: int):
    raw = np.fromfile(f, dtype=np.int8, count=2 * num_complex)
    if raw.size < 2 * num_complex:
        return None
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return i.astype(np.complex64) + 1j * q.astype(np.complex64)


def test_caf_cs8_visual_artifact():
    ref_path = os.environ.get("CS8_REF_PATH")
    surv_path = os.environ.get("CS8_SURV_PATH")
    prompt = os.environ.get("CS8_PROMPT", "0") == "1"
    if prompt and (not ref_path or not surv_path):
        ref_path = input("Enter CS8 reference file path: ").strip()
        surv_path = input("Enter CS8 surveillance file path: ").strip()

    if not ref_path or not surv_path:
        print("Skipping CS8 test: CS8_REF_PATH or CS8_SURV_PATH not set.")
        return

    if not os.path.exists(ref_path) or not os.path.exists(surv_path):
        print("Skipping CS8 test: CS8 reference or surveillance file not found.")
        return

    _ensure_artifact_dir()

    fs_env = os.environ.get("CS8_FS", "20000000")
    n_env = os.environ.get("CS8_BLOCK", "4096")
    center_env = os.environ.get("CS8_CENTER_FREQ", "")
    method = os.environ.get("CS8_METHOD", "batch")

    if prompt:
        fs_env = input(f"Enter sample rate Hz (default {fs_env}): ").strip() or fs_env
        n_env = input(f"Enter block size (default {n_env}): ").strip() or n_env
        center_env = input("Enter center frequency offset Hz (default 0): ").strip() or center_env
        method = input(f"Enter method (default {method}): ").strip() or method

    fs = float(fs_env)
    n = int(n_env)
    center_offset = float(center_env) if center_env else 0.0

    with open(ref_path, "rb") as fx, open(surv_path, "rb") as fy:
        x = _read_cs8_block(fx, n)
        y = _read_cs8_block(fy, n)

    if x is None or y is None:
        print("Skipping CS8 test: files ended before requested block size.")
        return

    x = x - x.mean()
    y = y - y.mean()
    if center_offset:
        x = _apply_frequency_shift(x, fs, center_offset, start_sample=0)
        y = _apply_frequency_shift(y, fs, center_offset, start_sample=0)

    result = cross_ambiguity(x, y, fs, method=method, convention="centered")
    caf_db = 20 * np.log10(result.caf_mag + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(
        caf_db,
        aspect="auto",
        origin="lower",
        extent=[result.delay_samples[0], result.delay_samples[-1], result.doppler_hz[0], result.doppler_hz[-1]],
        cmap="viridis",
    )
    ax.set_title("CAF Heatmap (dB) - CS8")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")
    fig.colorbar(im, ax=ax)

    params = {"m": method, "N": n, "fs": int(fs), "cs8": "1"}
    if center_offset:
        params["foff"] = int(center_offset)
    filename = _artifact_name("caf_cs8", params, "png")
    fig.savefig(os.path.join(ARTIFACT_DIR, filename), bbox_inches="tight")
    plt.close(fig)


def test_caf_cs8_visual_video():
    ref_path = os.environ.get("CS8_REF_PATH")
    surv_path = os.environ.get("CS8_SURV_PATH")
    prompt = os.environ.get("CS8_PROMPT", "0") == "1"
    if prompt and (not ref_path or not surv_path):
        ref_path = input("Enter CS8 reference file path: ").strip()
        surv_path = input("Enter CS8 surveillance file path: ").strip()

    if not ref_path or not surv_path:
        print("Skipping CS8 video: CS8_REF_PATH or CS8_SURV_PATH not set.")
        return

    if not os.path.exists(ref_path) or not os.path.exists(surv_path):
        print("Skipping CS8 video: CS8 reference or surveillance file not found.")
        return

    _ensure_artifact_dir()

    fs_env = os.environ.get("CS8_FS", "20000000")
    n_env = os.environ.get("CS8_BLOCK", "4096")
    step_env = os.environ.get("CS8_STEP", "2048")
    max_frames_env = os.environ.get("CS8_MAX_FRAMES", "200")
    center_env = os.environ.get("CS8_CENTER_FREQ", "")
    method = os.environ.get("CS8_METHOD", "batch")

    if prompt:
        fs_env = input(f"Enter sample rate Hz (default {fs_env}): ").strip() or fs_env
        n_env = input(f"Enter block size (default {n_env}): ").strip() or n_env
        step_env = input(f"Enter step size (default {step_env}): ").strip() or step_env
        max_frames_env = input(f"Enter max frames (default {max_frames_env}): ").strip() or max_frames_env
        center_env = input("Enter center frequency offset Hz (default 0): ").strip() or center_env
        method = input(f"Enter method (default {method}): ").strip() or method

    fs = float(fs_env)
    n = int(n_env)
    step = int(step_env)
    max_frames = int(max_frames_env)
    center_offset = float(center_env) if center_env else 0.0

    if step <= 0 or n <= 0:
        print("Skipping CS8 video: block and step must be positive.")
        return

    frames = []
    total_bytes = min(os.path.getsize(ref_path), os.path.getsize(surv_path))
    total_samples = total_bytes // 2
    if max_frames > 0:
        total_samples_target = min(total_samples, (max_frames - 1) * step + n)
    else:
        total_samples_target = total_samples

    def _print_progress(frame_idx: int):
        if total_samples_target <= 0:
            return
        processed_samples = min((frame_idx - 1) * step + n, total_samples_target)
        pct = (processed_samples / total_samples_target) * 100.0
        bar_len = 30
        filled = int(round((pct / 100.0) * bar_len))
        empty = bar_len - filled
        print(
            f"[{'#' * filled}{'-' * empty}] {pct:5.1f}% ({frame_idx}/{max_frames} frames)",
            end="\r",
            flush=True,
        )

    with open(ref_path, "rb") as fx, open(surv_path, "rb") as fy:
        frame_idx = 0
        while frame_idx < max_frames:
            x = _read_cs8_block(fx, n)
            y = _read_cs8_block(fy, n)
            if x is None or y is None:
                break

            x = x - x.mean()
            y = y - y.mean()
            if center_offset:
                x = _apply_frequency_shift(x, fs, center_offset, start_sample=frame_idx * step)
                y = _apply_frequency_shift(y, fs, center_offset, start_sample=frame_idx * step)
            result = cross_ambiguity(x, y, fs, method=method, convention="centered")
            caf_db = 20 * np.log10(result.caf_mag + 1e-12)
            frames.append((caf_db, result))

            if step != n:
                rewind = n - step
                fx.seek(-2 * rewind, os.SEEK_CUR)
                fy.seek(-2 * rewind, os.SEEK_CUR)

            frame_idx += 1
            _print_progress(frame_idx)
    if frames:
        print()

    if not frames:
        print("Skipping CS8 video: no frames collected.")
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(
        frames[0][0],
        aspect="auto",
        origin="lower",
        extent=[frames[0][1].delay_samples[0], frames[0][1].delay_samples[-1], frames[0][1].doppler_hz[0], frames[0][1].doppler_hz[-1]],
        cmap="viridis",
    )
    ax.set_title("CAF Heatmap (dB) - CS8")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")

    def update(i):
        caf_db, result = frames[i]
        im.set_data(caf_db)
        im.set_extent(
            [result.delay_samples[0], result.delay_samples[-1], result.doppler_hz[0], result.doppler_hz[-1]]
        )
        ax.set_title(f"CAF Heatmap (dB) - CS8 frame {i+1}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=True)

    params = {
        "m": method,
        "N": n,
        "fs": int(fs),
        "step": step,
        "frames": len(frames),
        "cs8": "1",
    }
    if center_offset:
        params["foff"] = int(center_offset)
    filename = _artifact_name("caf_cs8", params, "mp4")
    anim.save(os.path.join(ARTIFACT_DIR, filename), writer=FFMpegWriter(fps=4))
    plt.close(fig)


def _run_nonpytest(cs8_only: bool, prompt: bool, ref_path: str, surv_path: str, fs: str, block: str, method: str, center_freq: str):
    if cs8_only:
        if prompt:
            os.environ["CS8_PROMPT"] = "1"
        if ref_path:
            os.environ["CS8_REF_PATH"] = ref_path
        if surv_path:
            os.environ["CS8_SURV_PATH"] = surv_path
        if fs:
            os.environ["CS8_FS"] = fs
        if block:
            os.environ["CS8_BLOCK"] = block
        if method:
            os.environ["CS8_METHOD"] = method
        if center_freq:
            os.environ["CS8_CENTER_FREQ"] = center_freq
        if os.environ.get("CS8_VIDEO", "0") == "1":
            test_caf_cs8_visual_video()
        else:
            test_caf_cs8_visual_artifact()
        return

    test_caf_heatmap_artifact()
    test_caf_video_artifact()
    if prompt or ref_path or surv_path:
        if prompt:
            os.environ["CS8_PROMPT"] = "1"
        if ref_path:
            os.environ["CS8_REF_PATH"] = ref_path
        if surv_path:
            os.environ["CS8_SURV_PATH"] = surv_path
        if fs:
            os.environ["CS8_FS"] = fs
        if block:
            os.environ["CS8_BLOCK"] = block
        if method:
            os.environ["CS8_METHOD"] = method
        if center_freq:
            os.environ["CS8_CENTER_FREQ"] = center_freq
        try:
            if os.environ.get("CS8_VIDEO", "0") == "1":
                test_caf_cs8_visual_video()
            else:
                test_caf_cs8_visual_artifact()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CAF visual artifact generators.")
    parser.add_argument("--cs8", action="store_true", help="Run only the CS8 visual test.")
    parser.add_argument("--prompt", action="store_true", help="Prompt for CS8 paths/params.")
    parser.add_argument("--ref", default="", help="CS8 reference file path.")
    parser.add_argument("--surv", default="", help="CS8 surveillance file path.")
    parser.add_argument("--fs", default="", help="Sample rate (Hz).")
    parser.add_argument("--block", default="", help="Block size (samples).")
    parser.add_argument("--method", default="", help="CAF method.")
    parser.add_argument("--center-freq", default="", help="Center frequency offset (Hz) to mix to baseband.")
    parser.add_argument("--video", action="store_true", help="Generate CS8 sliding-block video.")
    parser.add_argument("--step", default="", help="Step size (samples).")
    parser.add_argument("--max-frames", default="", help="Maximum frames to render.")

    args = parser.parse_args()
    if args.video:
        os.environ["CS8_VIDEO"] = "1"
    if args.step:
        os.environ["CS8_STEP"] = args.step
    if args.max_frames:
        os.environ["CS8_MAX_FRAMES"] = args.max_frames
    _run_nonpytest(
        cs8_only=args.cs8,
        prompt=args.prompt,
        ref_path=args.ref,
        surv_path=args.surv,
        fs=args.fs,
        block=args.block,
        method=args.method,
        center_freq=args.center_freq,
    )
