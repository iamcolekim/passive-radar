import os
from datetime import datetime
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# MUST INSTALL 'ffmpeg' FOR VIDEO OUTPUT
from matplotlib.animation import FuncAnimation, FFMpegWriter

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


def _load_sigmf(meta_path: str):
    try:
        from sigmf import SigMFFile, sigmffile
    except ImportError as exc:
        raise ImportError(
            "sigmf package not installed. Install with 'pip install sigmf' or add it to your env."
        ) from exc

    sigmf_file = sigmffile.fromfile(meta_path)
    return sigmf_file, SigMFFile


def _derive_data_path(meta_path: str) -> str:
    if meta_path.endswith(".sigmf-meta"):
        return meta_path[: -len(".sigmf-meta")] + ".sigmf-data"
    return meta_path + ".sigmf-data"


def _resolve_data_path(sigmf_file, meta_path: str) -> str:
    for attr in ("get_data_path", "get_data_file"):
        if hasattr(sigmf_file, attr):
            candidate = getattr(sigmf_file, attr)()
            if candidate:
                return candidate
    if hasattr(sigmf_file, "_data_file") and sigmf_file._data_file:
        return sigmf_file._data_file
    return _derive_data_path(meta_path)


def _bytes_per_sample(datatype: str) -> int | None:
    if not datatype:
        return None
    dt = datatype.lower()
    match = re.match(r"^(c|r)(i|f)(\d+)", dt)
    if not match:
        return None
    base = match.group(1)
    bits = int(match.group(3))
    bytes_per = bits // 8
    if bytes_per <= 0:
        return None
    if base == "c":
        return 2 * bytes_per
    return bytes_per


def _read_sigmf_block(sigmf_file, start: int, num_samples: int) -> np.ndarray | None:
    block = sigmf_file.read_samples(start, num_samples)
    if block is None or block.size < num_samples:
        return None
    if not np.iscomplexobj(block):
        block = block.astype(np.float32) + 1j * np.zeros_like(block, dtype=np.float32)
    return np.asarray(block, dtype=np.complex64)


def _apply_frequency_shift(x: np.ndarray, fs: float, offset_hz: float, start_sample: int = 0) -> np.ndarray:
    if not offset_hz:
        return x
    n = x.shape[0]
    t = (np.arange(n, dtype=np.float64) + start_sample) / fs
    return x * np.exp(-1j * 2 * np.pi * offset_hz * t)


def _get_center_freq(sigmf_file, SigMFFile):
    center = sigmf_file.get_global_field(SigMFFile.FREQUENCY_KEY)
    if center is not None:
        return center
    captures = sigmf_file.get_captures()
    if captures:
        return captures[0].get(SigMFFile.FREQUENCY_KEY)
    return None


def test_caf_sigmf_visual_artifact():
    ref_meta = os.environ.get("SIGMF_REF_META")
    surv_meta = os.environ.get("SIGMF_SURV_META")
    ref_data = os.environ.get("SIGMF_REF_DATA", "")
    surv_data = os.environ.get("SIGMF_SURV_DATA", "")
    prompt = os.environ.get("SIGMF_PROMPT", "0") == "1"

    if prompt and (not ref_meta or not surv_meta):
        ref_meta = input("Enter SigMF reference .sigmf-meta path: ").strip()
        surv_meta = input("Enter SigMF surveillance .sigmf-meta path: ").strip()

    if not ref_meta or not surv_meta:
        print("Skipping SigMF test: SIGMF_REF_META or SIGMF_SURV_META not set.")
        return

    if not os.path.exists(ref_meta) or not os.path.exists(surv_meta):
        print("Skipping SigMF test: SigMF meta files not found.")
        return

    try:
        sigmf_ref, SigMFFile = _load_sigmf(ref_meta)
        sigmf_surv, _ = _load_sigmf(surv_meta)
    except ImportError as exc:
        print(str(exc))
        return

    ref_data = ref_data or _resolve_data_path(sigmf_ref, ref_meta)
    surv_data = surv_data or _resolve_data_path(sigmf_surv, surv_meta)

    if not os.path.exists(ref_data) or not os.path.exists(surv_data):
        print("Skipping SigMF test: SigMF data files not found.")
        return

    _ensure_artifact_dir()

    fs_env = os.environ.get("SIGMF_FS", "")
    n_env = os.environ.get("SIGMF_BLOCK", "4096")
    center_env = os.environ.get("SIGMF_CENTER_FREQ", "")
    method = os.environ.get("SIGMF_METHOD", "batch")

    if prompt:
        fs_env = input(f"Enter sample rate Hz (default {fs_env or 'from meta'}): ").strip() or fs_env
        n_env = input(f"Enter block size (default {n_env}): ").strip() or n_env
        center_env = input("Enter target center frequency Hz (default from metadata): ").strip() or center_env
        method = input(f"Enter method (default {method}): ").strip() or method

    if fs_env:
        fs = float(fs_env)
    else:
        fs = sigmf_ref.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        if fs is None:
            fs = sigmf_surv.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    if fs is None:
        print("Skipping SigMF test: sample rate missing; set SIGMF_FS.")
        return

    meta_center = _get_center_freq(sigmf_ref, SigMFFile)
    target_center = float(center_env) if center_env else meta_center
    center_offset = 0.0
    if target_center is not None and meta_center is not None:
        center_offset = float(target_center) - float(meta_center)

    n = int(n_env)

    x = _read_sigmf_block(sigmf_ref, 0, n)
    y = _read_sigmf_block(sigmf_surv, 0, n)
    if x is None or y is None:
        print("Skipping SigMF test: files ended before requested block size.")
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
    ax.set_title("CAF Heatmap (dB) - SigMF")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")
    fig.colorbar(im, ax=ax)

    params = {"m": method, "N": n, "fs": int(fs), "sigmf": "1"}
    if target_center is not None:
        params["fc"] = int(target_center)
    if center_offset:
        params["foff"] = int(center_offset)
    filename = _artifact_name("caf_sigmf", params, "png")
    fig.savefig(os.path.join(ARTIFACT_DIR, filename), bbox_inches="tight")
    plt.close(fig)


def test_caf_sigmf_visual_video():
    ref_meta = os.environ.get("SIGMF_REF_META")
    surv_meta = os.environ.get("SIGMF_SURV_META")
    ref_data = os.environ.get("SIGMF_REF_DATA", "")
    surv_data = os.environ.get("SIGMF_SURV_DATA", "")
    prompt = os.environ.get("SIGMF_PROMPT", "0") == "1"

    if prompt and (not ref_meta or not surv_meta):
        ref_meta = input("Enter SigMF reference .sigmf-meta path: ").strip()
        surv_meta = input("Enter SigMF surveillance .sigmf-meta path: ").strip()

    if not ref_meta or not surv_meta:
        print("Skipping SigMF video: SIGMF_REF_META or SIGMF_SURV_META not set.")
        return

    if not os.path.exists(ref_meta) or not os.path.exists(surv_meta):
        print("Skipping SigMF video: SigMF meta files not found.")
        return

    try:
        sigmf_ref, SigMFFile = _load_sigmf(ref_meta)
        sigmf_surv, _ = _load_sigmf(surv_meta)
    except ImportError as exc:
        print(str(exc))
        return

    ref_data = ref_data or _resolve_data_path(sigmf_ref, ref_meta)
    surv_data = surv_data or _resolve_data_path(sigmf_surv, surv_meta)

    if not os.path.exists(ref_data) or not os.path.exists(surv_data):
        print("Skipping SigMF video: SigMF data files not found.")
        return

    _ensure_artifact_dir()

    fs_env = os.environ.get("SIGMF_FS", "")
    n_env = os.environ.get("SIGMF_BLOCK", "4096")
    step_env = os.environ.get("SIGMF_STEP", "2048")
    max_frames_env = os.environ.get("SIGMF_MAX_FRAMES", "200")
    center_env = os.environ.get("SIGMF_CENTER_FREQ", "")
    method = os.environ.get("SIGMF_METHOD", "batch")

    if prompt:
        fs_env = input(f"Enter sample rate Hz (default {fs_env or 'from meta'}): ").strip() or fs_env
        n_env = input(f"Enter block size (default {n_env}): ").strip() or n_env
        step_env = input(f"Enter step size (default {step_env}): ").strip() or step_env
        max_frames_env = input(f"Enter max frames (default {max_frames_env}): ").strip() or max_frames_env
        center_env = input("Enter target center frequency Hz (default from metadata): ").strip() or center_env
        method = input(f"Enter method (default {method}): ").strip() or method

    if fs_env:
        fs = float(fs_env)
    else:
        fs = sigmf_ref.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        if fs is None:
            fs = sigmf_surv.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    if fs is None:
        print("Skipping SigMF video: sample rate missing; set SIGMF_FS.")
        return

    meta_center = _get_center_freq(sigmf_ref, SigMFFile)
    target_center = float(center_env) if center_env else meta_center
    center_offset = 0.0
    if target_center is not None and meta_center is not None:
        center_offset = float(target_center) - float(meta_center)

    n = int(n_env)
    step = int(step_env)
    max_frames = int(max_frames_env)

    if step <= 0 or n <= 0:
        print("Skipping SigMF video: block and step must be positive.")
        return

    datatype_ref = sigmf_ref.get_global_field(SigMFFile.DATATYPE_KEY)
    datatype_surv = sigmf_surv.get_global_field(SigMFFile.DATATYPE_KEY)
    bytes_per_ref = _bytes_per_sample(datatype_ref)
    bytes_per_surv = _bytes_per_sample(datatype_surv)

    total_samples = 0
    if bytes_per_ref and bytes_per_surv:
        total_bytes = min(os.path.getsize(ref_data), os.path.getsize(surv_data))
        bytes_per = min(bytes_per_ref, bytes_per_surv)
        total_samples = total_bytes // bytes_per

    if max_frames > 0 and total_samples > 0:
        total_samples_target = min(total_samples, (max_frames - 1) * step + n)
    else:
        total_samples_target = total_samples

    frames = []

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

    frame_idx = 0
    start = 0
    while frame_idx < max_frames:
        x = _read_sigmf_block(sigmf_ref, start, n)
        y = _read_sigmf_block(sigmf_surv, start, n)
        if x is None or y is None:
            break

        x = x - x.mean()
        y = y - y.mean()
        if center_offset:
            x = _apply_frequency_shift(x, fs, center_offset, start_sample=start)
            y = _apply_frequency_shift(y, fs, center_offset, start_sample=start)
        result = cross_ambiguity(x, y, fs, method=method, convention="centered")
        caf_db = 20 * np.log10(result.caf_mag + 1e-12)
        frames.append((caf_db, result))

        frame_idx += 1
        _print_progress(frame_idx)
        start += step
    if frames:
        print()

    if not frames:
        print("Skipping SigMF video: no frames collected.")
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(
        frames[0][0],
        aspect="auto",
        origin="lower",
        extent=[frames[0][1].delay_samples[0], frames[0][1].delay_samples[-1], frames[0][1].doppler_hz[0], frames[0][1].doppler_hz[-1]],
        cmap="viridis",
    )
    ax.set_title("CAF Heatmap (dB) - SigMF")
    ax.set_xlabel("Delay (samples)")
    ax.set_ylabel("Doppler (Hz)")

    def update(i):
        caf_db, result = frames[i]
        im.set_data(caf_db)
        im.set_extent(
            [result.delay_samples[0], result.delay_samples[-1], result.doppler_hz[0], result.doppler_hz[-1]]
        )
        ax.set_title(f"CAF Heatmap (dB) - SigMF frame {i+1}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=250, blit=True)

    params = {
        "m": method,
        "N": n,
        "fs": int(fs),
        "step": step,
        "frames": len(frames),
        "sigmf": "1",
    }
    if target_center is not None:
        params["fc"] = int(target_center)
    if center_offset:
        params["foff"] = int(center_offset)
    filename = _artifact_name("caf_sigmf", params, "mp4")
    anim.save(os.path.join(ARTIFACT_DIR, filename), writer=FFMpegWriter(fps=4))
    plt.close(fig)


def _run_nonpytest(sigmf_only: bool, prompt: bool, ref_meta: str, surv_meta: str, ref_data: str, surv_data: str, fs: str, block: str, method: str, center_freq: str):
    if sigmf_only:
        if prompt:
            os.environ["SIGMF_PROMPT"] = "1"
        if ref_meta:
            os.environ["SIGMF_REF_META"] = ref_meta
        if surv_meta:
            os.environ["SIGMF_SURV_META"] = surv_meta
        if ref_data:
            os.environ["SIGMF_REF_DATA"] = ref_data
        if surv_data:
            os.environ["SIGMF_SURV_DATA"] = surv_data
        if fs:
            os.environ["SIGMF_FS"] = fs
        if block:
            os.environ["SIGMF_BLOCK"] = block
        if method:
            os.environ["SIGMF_METHOD"] = method
        if center_freq:
            os.environ["SIGMF_CENTER_FREQ"] = center_freq
        if os.environ.get("SIGMF_VIDEO", "0") == "1":
            test_caf_sigmf_visual_video()
        else:
            test_caf_sigmf_visual_artifact()
        return

    if center_freq:
        os.environ["SIGMF_CENTER_FREQ"] = center_freq
    test_caf_sigmf_visual_artifact()
    test_caf_sigmf_visual_video()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CAF visual artifact generators for SigMF filesets.")
    parser.add_argument("--sigmf", action="store_true", help="Run only the SigMF visual test.")
    parser.add_argument("--prompt", action="store_true", help="Prompt for SigMF paths/params.")
    parser.add_argument("--ref-meta", default="", help="SigMF reference .sigmf-meta path.")
    parser.add_argument("--surv-meta", default="", help="SigMF surveillance .sigmf-meta path.")
    parser.add_argument("--ref-data", default="", help="SigMF reference .sigmf-data path override.")
    parser.add_argument("--surv-data", default="", help="SigMF surveillance .sigmf-data path override.")
    parser.add_argument("--fs", default="", help="Sample rate (Hz).")
    parser.add_argument("--block", default="", help="Block size (samples).")
    parser.add_argument("--method", default="", help="CAF method.")
    parser.add_argument("--center-freq", default="", help="Target center frequency (Hz); defaults to metadata.")
    parser.add_argument("--video", action="store_true", help="Generate SigMF sliding-block video.")
    parser.add_argument("--step", default="", help="Step size (samples).")
    parser.add_argument("--max-frames", default="", help="Maximum frames to render.")

    args = parser.parse_args()
    if args.video:
        os.environ["SIGMF_VIDEO"] = "1"
    if args.step:
        os.environ["SIGMF_STEP"] = args.step
    if args.max_frames:
        os.environ["SIGMF_MAX_FRAMES"] = args.max_frames
    _run_nonpytest(
        sigmf_only=args.sigmf,
        prompt=args.prompt,
        ref_meta=args.ref_meta,
        surv_meta=args.surv_meta,
        ref_data=args.ref_data,
        surv_data=args.surv_data,
        fs=args.fs,
        block=args.block,
        method=args.method,
        center_freq=args.center_freq,
    )
