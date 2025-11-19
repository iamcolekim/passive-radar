"""hackrf.py
Multi-HackRF passive radar capture & processing using SoapySDR + NumPy.

Implements (approximate) functionality inspired by `test3.grc`:
  * Dual device simultaneous IQ capture (reference & surveillance channels)
  * DC blocking, subtraction, scaling (mult), optional delay (sample roll)
  * Vectorization into size N blocks
  * FFT-based cross-correlation / ambiguity (X1 * conj(X2) -> IFFT)
  * Magnitude output per block

Hardware timing & triggering notes:
HackRF does NOT provide true sample-synchronous multi-device triggering like
USRP time-spec scheduled streaming. We approximate alignment by:
  1. Resetting (if supported) hardware time to 0 on both devices.
  2. Starting both streams in quick succession.
  3. Cropping to equal length & aligning by discarding the initial imbalance.

For more rigorous synchronization you would need shared clock + external gating
or migrate to hardware (e.g., USRP) supporting timed activation.

CLI usage examples:
  python hackrf.py --duration 2 --freq 99.1e6 --rate 20e6 \
      --ref-serial 000000000000000066a062dc226b169f \
      --surv-serial 0000000000000000436c63dc38284c63 \
      --vector 32768 --mult 0.2 --delay 0 --gain 20 \
      --xcorr-out xcorr.npy --raw-out-prefix capture

File mode (process previously captured .cs8 files):
    python hackrf.py --mode file \
            --ref-file 63_880M_01_11_car_3.cs8 --surv-file 9f_880M_01_11_car_3.cs8 \
            --vector 32768 --mult 0.2 --delay 0 --scale 127 --xcorr-out xcorr.npy

Optional plotting (matplotlib):
    python hackrf.py --mode file --ref-file a.cs8 --surv-file b.cs8 \\
            --plot --xcorr-out xcorr.npy

Outputs (optional):
  capture_ref.npy / capture_surv.npy : raw complex streams (cropped equal length)
  xcorr.npy                          : list of cross-correlation magnitudes per block

Future enhancements (not implemented here):
  * Windowing & optional low-pass filtering as in GRC (currently disabled blocks)
  * Streaming to disk with chunked memory-map for long durations
  * GPU acceleration (CuPy / numba) for high-rate real-time processing
  * True synchronization with hardware supporting time-spec activation
"""

from __future__ import annotations

import argparse
import time
import threading
import queue
from typing import List, Optional, Iterator
import os

import numpy as np
try:  # Optional plotting dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
except ImportError:
    # We allow running in file mode without SoapySDR installed.
    SoapySDR = None  # type: ignore
    SOAPY_SDR_RX = SOAPY_SDR_CF32 = None  # type: ignore


def list_hackrf_devices() -> List[dict]:
    """Enumerate HackRF devices via SoapySDR and return result dictionaries."""
    if SoapySDR is None:
        return []
    return [d for d in SoapySDR.Device.enumerate() if "hackrf" in str(d).lower()]


def open_device(serial: Optional[str] = None):
    """Open a HackRF device by optional serial, else first available."""
    if SoapySDR is None:
        raise RuntimeError("SoapySDR not available; cannot open live devices (use --mode file).")
    if serial:
        args = f"hackrf=serial={serial}"  # serial match attempt
        try:
            return SoapySDR.Device(args)
        except Exception:
            # Fallback: some Soapy modules use hackrf=<serial>
            args_alt = f"hackrf={serial}"
            return SoapySDR.Device(args_alt)
    devs = list_hackrf_devices()
    if not devs:
        raise RuntimeError("No HackRF devices found.")
    return SoapySDR.Device(devs[0])


def configure(device, rate: float, freq: float, gain: float):
    if SoapySDR is None:
        raise RuntimeError("configure() called without SoapySDR present.")
    device.setSampleRate(SOAPY_SDR_RX, 0, rate)
    device.setFrequency(SOAPY_SDR_RX, 0, freq)
    try:
        device.setGain(SOAPY_SDR_RX, 0, gain)
    except Exception:
        pass


def setup_stream(device):
    if SoapySDR is None:
        raise RuntimeError("setup_stream() called without SoapySDR present.")
    stream = device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
    device.activateStream(stream)
    return stream


def close_stream(device: SoapySDR.Device, stream):
    try:
        device.deactivateStream(stream)
    finally:
        device.closeStream(stream)


def attempt_time_reset(device):
    """Attempt to reset hardware time to zero (may be unsupported)."""
    if SoapySDR is None:
        return
    try:
        device.setHardwareTime(0)
    except Exception:
        pass


class CaptureThread(threading.Thread):
    """Thread that continuously reads samples into a queue until stop signal."""

    def __init__(self, device: SoapySDR.Device, stream, chunk: int, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.device = device
        self.stream = stream
        self.chunk = chunk
        self.out_queue = out_queue
        self.stop_event = stop_event

    def run(self):  # pragma: no cover (hardware dependent)
        buf = np.empty(self.chunk, dtype=np.complex64)
        while not self.stop_event.is_set():
            # readStream signature: (ret, buffers, flags, timeNs)
            try:
                ret = self.device.readStream(self.stream, [buf], self.chunk)
            except Exception as e:
                print(f"[WARN] readStream error: {e}")
                break
            rc, _, flags, time_ns = ret
            if rc > 0:
                # copy to avoid overwrite in next call
                self.out_queue.put(buf[:rc].copy())
            else:
                # small sleep to avoid busy spin on errors
                time.sleep(0.001)


def dc_block(x: np.ndarray) -> np.ndarray:
    """Simple DC blocker: subtract mean."""
    return x - np.mean(x)


def process_block(v_ref: np.ndarray, v_surv: np.ndarray, mult: float, delay: int) -> dict:
    """Process one block of samples matching GRC-inspired pipeline.

    Steps:
      1. DC block each input.
      2. Difference: diff = ref - surv.
      3. Apply multiplier.
      4. Optional delay (circular roll) on diff.
      5. Cross-correlation:
            X1 = FFT(ref), X2 = FFT(surv)
            R = IFFT(X1 * conj(X2))  (time-domain correlation / ambiguity)
            mag = |R| (optionally fftshift for centering)
    Returns dictionary with intermediate arrays for debugging.
    """
    v_ref = dc_block(v_ref)
    v_surv = dc_block(v_surv)
    diff = v_ref - v_surv
    if delay:
        diff = np.roll(diff, delay)
    diff_scaled = diff * mult

    # Cross-correlation (unshifted). You may choose fftshift depending on analysis.
    X1 = np.fft.fft(v_ref)
    X2 = np.fft.fft(v_surv)
    R = np.fft.ifft(X1 * np.conj(X2))
    mag = np.abs(R)
    mag_shifted = np.fft.fftshift(mag)

    return {
        "ref": v_ref,
        "surv": v_surv,
        "diff_scaled": diff_scaled,
        "xcorr_time": R,
        "xcorr_mag": mag,
        "xcorr_mag_shifted": mag_shifted,
    }


def capture_and_process(args):  # pragma: no cover (runtime/hardware path)
    if args.mode == "file":
        return capture_from_files(args)

    # Open devices
    dev_ref = open_device(args.ref_serial)
    dev_surv = open_device(args.surv_serial)

    # Basic configuration
    configure(dev_ref, args.rate, args.freq, args.gain)
    configure(dev_surv, args.rate, args.freq, args.gain)

    attempt_time_reset(dev_ref)
    attempt_time_reset(dev_surv)

    stream_ref = setup_stream(dev_ref)
    stream_surv = setup_stream(dev_surv)

    # Queues & thread control
    q_ref: queue.Queue[np.ndarray] = queue.Queue()
    q_surv: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    chunk = args.chunk
    t_ref = CaptureThread(dev_ref, stream_ref, chunk, q_ref, stop_event)
    t_surv = CaptureThread(dev_surv, stream_surv, chunk, q_surv, stop_event)
    t_ref.start(); t_surv.start()

    target_samples = int(args.duration * args.rate)
    ref_buffer = []  # list of arrays to concatenate later
    surv_buffer = []
    processed_blocks = []
    samples_collected = 0

    vector_N = args.vector
    leftover_ref = np.empty(0, dtype=np.complex64)
    leftover_surv = np.empty(0, dtype=np.complex64)

    start_time = time.time()

    try:
        while samples_collected < target_samples:
            # Pull from queues (non-blocking small waits)
            try:
                r_chunk = q_ref.get(timeout=0.1)
                ref_buffer.append(r_chunk)
            except queue.Empty:
                pass
            try:
                s_chunk = q_surv.get(timeout=0.1)
                surv_buffer.append(s_chunk)
            except queue.Empty:
                pass

            # Update sample count as min of lengths (we only use aligned portion)
            total_ref = sum(len(x) for x in ref_buffer)
            total_surv = sum(len(x) for x in surv_buffer)
            samples_collected = min(total_ref, total_surv)

            # Build contiguous arrays for block processing when possible
            if samples_collected >= vector_N:
                # Concatenate incremental until we have at least one full block
                ref_cat = np.concatenate(ref_buffer)
                surv_cat = np.concatenate(surv_buffer)
                usable = min(len(ref_cat), len(surv_cat))
                ref_cat = ref_cat[:usable]
                surv_cat = surv_cat[:usable]

                # Prepend leftovers
                if leftover_ref.size:
                    ref_cat = np.concatenate([leftover_ref, ref_cat])
                if leftover_surv.size:
                    surv_cat = np.concatenate([leftover_surv, surv_cat])

                blocks_available = len(ref_cat) // vector_N
                if blocks_available:
                    for b in range(blocks_available):
                        start = b * vector_N
                        end = start + vector_N
                        v_r = ref_cat[start:end]
                        v_s = surv_cat[start:end]
                        out = process_block(v_r, v_s, args.mult, args.delay)
                        processed_blocks.append(out["xcorr_mag_shifted"])
                    # Save leftovers
                    leftover_ref = ref_cat[blocks_available * vector_N:]
                    leftover_surv = surv_cat[blocks_available * vector_N:]
                    # Reset buffers (already consumed)
                    ref_buffer = [leftover_ref]
                    surv_buffer = [leftover_surv]

            # Periodic progress print
            if time.time() - start_time > 1.0:
                start_time = time.time()
                pct = 100 * samples_collected / target_samples
                print(f"[Capture] {samples_collected}/{target_samples} samples (~{pct:.1f}%)")
    finally:
        stop_event.set()
        t_ref.join(timeout=1)
        t_surv.join(timeout=1)
        close_stream(dev_ref, stream_ref)
        close_stream(dev_surv, stream_surv)

    # Final crop & save raw if requested
    total_ref = sum(len(x) for x in ref_buffer)
    total_surv = sum(len(x) for x in surv_buffer)
    min_len = min(total_ref, total_surv)
    raw_ref = np.concatenate(ref_buffer)[:min_len]
    raw_surv = np.concatenate(surv_buffer)[:min_len]

    if args.raw_out_prefix:
        np.save(f"{args.raw_out_prefix}_ref.npy", raw_ref)
        np.save(f"{args.raw_out_prefix}_surv.npy", raw_surv)
        print(f"Saved raw captures: {args.raw_out_prefix}_ref.npy, {args.raw_out_prefix}_surv.npy")

    xcorr_array = np.stack(processed_blocks) if processed_blocks else np.empty((0, vector_N))
    if args.xcorr_out:
        np.save(args.xcorr_out, xcorr_array)
        print(f"Saved cross-correlation magnitudes: {args.xcorr_out} (shape={xcorr_array.shape})")

    print("Capture complete.")
    return raw_ref, raw_surv, xcorr_array


def _cs8_chunk_reader(path: str, chunk_complex: int, scale: float) -> Iterator[np.ndarray]:
    """Yield successive chunks of complex samples from a .cs8 file.

    Each complex sample = 2 signed int8 (I,Q). A chunk of N complex samples
    therefore uses 2N bytes.
    """
    bytes_per_complex = 2
    bytes_per_chunk = chunk_complex * bytes_per_complex
    with open(path, "rb") as f:
        while True:
            data = f.read(bytes_per_chunk)
            if not data:
                break
            # Ensure even number of bytes
            if len(data) % 2 == 1:
                data = data[:-1]
            arr = np.frombuffer(data, dtype=np.int8)
            if arr.size == 0:
                break
            iq = arr.reshape(-1, 2)
            # scale to float32 (-1..1 approx)
            c = (iq[:, 0].astype(np.float32) / scale) + 1j * (iq[:, 1].astype(np.float32) / scale)
            yield c.astype(np.complex64)


def capture_from_files(args):  # pragma: no cover
    if not args.ref_file or not args.surv_file:
        raise SystemExit("--mode file requires --ref-file and --surv-file")
    for p in (args.ref_file, args.surv_file):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}")

    # Determine min sample count to align lengths without loading all into memory first.
    size_ref = os.path.getsize(args.ref_file)
    size_surv = os.path.getsize(args.surv_file)
    # Each complex sample is 2 bytes
    samples_ref = size_ref // 2
    samples_surv = size_surv // 2
    target_samples = min(samples_ref, samples_surv)

    vector_N = args.vector
    chunk = args.chunk
    scale = args.scale

    gen_ref = _cs8_chunk_reader(args.ref_file, chunk, scale)
    gen_surv = _cs8_chunk_reader(args.surv_file, chunk, scale)

    processed_blocks = []
    ref_used = 0
    surv_used = 0
    leftover_ref = np.empty(0, dtype=np.complex64)
    leftover_surv = np.empty(0, dtype=np.complex64)
    ref_accum_chunks = []
    surv_accum_chunks = []

    print(f"[File Mode] Processing up to {target_samples} complex samples from each file.")

    while ref_used < target_samples and surv_used < target_samples:
        try:
            r_chunk = next(gen_ref)
            s_chunk = next(gen_surv)
        except StopIteration:
            break
        # Respect target_samples limit
        remaining = target_samples - min(ref_used, surv_used)
        if len(r_chunk) > remaining:
            r_chunk = r_chunk[:remaining]
        if len(s_chunk) > remaining:
            s_chunk = s_chunk[:remaining]
        ref_used += len(r_chunk)
        surv_used += len(s_chunk)
        ref_accum_chunks.append(r_chunk)
        surv_accum_chunks.append(s_chunk)

        ref_cat = np.concatenate(ref_accum_chunks)
        surv_cat = np.concatenate(surv_accum_chunks)
        usable = min(len(ref_cat), len(surv_cat))
        ref_cat = ref_cat[:usable]
        surv_cat = surv_cat[:usable]
        if leftover_ref.size:
            ref_cat = np.concatenate([leftover_ref, ref_cat])
        if leftover_surv.size:
            surv_cat = np.concatenate([leftover_surv, surv_cat])

        blocks_available = len(ref_cat) // vector_N
        if blocks_available:
            for b in range(blocks_available):
                start = b * vector_N
                end = start + vector_N
                out = process_block(ref_cat[start:end], surv_cat[start:end], args.mult, args.delay)
                processed_blocks.append(out["xcorr_mag_shifted"])
            leftover_ref = ref_cat[blocks_available * vector_N:]
            leftover_surv = surv_cat[blocks_available * vector_N:]
            ref_accum_chunks = [leftover_ref]
            surv_accum_chunks = [leftover_surv]

        pct = 100 * min(ref_used, surv_used) / target_samples
        print(f"[File Mode] {min(ref_used, surv_used)}/{target_samples} samples (~{pct:.1f}%)")

    # Final crop & save raw if requested
    raw_ref = np.concatenate(ref_accum_chunks)
    raw_surv = np.concatenate(surv_accum_chunks)
    min_len = min(len(raw_ref), len(raw_surv))
    raw_ref = raw_ref[:min_len]
    raw_surv = raw_surv[:min_len]

    if args.raw_out_prefix:
        np.save(f"{args.raw_out_prefix}_ref.npy", raw_ref)
        np.save(f"{args.raw_out_prefix}_surv.npy", raw_surv)
        print(f"Saved raw captures: {args.raw_out_prefix}_ref.npy, {args.raw_out_prefix}_surv.npy")

    xcorr_array = np.stack(processed_blocks) if processed_blocks else np.empty((0, vector_N))
    if args.xcorr_out:
        np.save(args.xcorr_out, xcorr_array)
        print(f"Saved cross-correlation magnitudes: {args.xcorr_out} (shape={xcorr_array.shape})")

    print("File processing complete.")
    return raw_ref, raw_surv, xcorr_array


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dual HackRF passive radar capture & processing (live or file mode)")
    p.add_argument("--mode", choices=["live", "file"], default="live", help="Capture mode: live HackRF or process .cs8 files")
    # Live mode options
    p.add_argument("--ref-serial", help="Reference HackRF serial (live mode)", default=None)
    p.add_argument("--surv-serial", help="Surveillance HackRF serial (live mode)", default=None)
    p.add_argument("--freq", type=float, default=99.1e6, help="Center frequency (Hz) (live mode)")
    p.add_argument("--rate", type=float, default=20e6, help="Sample rate (live mode)")
    p.add_argument("--gain", type=float, default=20.0, help="Gain setting (live mode)")
    p.add_argument("--duration", type=float, default=2.0, help="Capture duration seconds (live mode)")
    # File mode options
    p.add_argument("--ref-file", help="Reference .cs8 file (file mode)", default=None)
    p.add_argument("--surv-file", help="Surveillance .cs8 file (file mode)", default=None)
    p.add_argument("--scale", type=float, default=127.0, help="Scale factor for int8 IQ to float (file mode)")
    # Common processing options
    p.add_argument("--vector", type=int, default=32768, help="Vector length N for block processing")
    p.add_argument("--delay", type=int, default=0, help="Sample delay (circular roll) applied to diff")
    p.add_argument("--mult", type=float, default=0.2, help="Scaling multiplier applied to (ref - surv)")
    p.add_argument("--chunk", type=int, default=4096, help="Read chunk size per stream iteration or file chunk (complex samples)")
    p.add_argument("--xcorr-out", default=None, help="Output .npy file for cross-correlation magnitudes")
    p.add_argument("--raw-out-prefix", default=None, help="Prefix for raw ref/surv .npy output files")
    p.add_argument("--plot", action="store_true", help="Show matplotlib summary plots after processing")
    return p

def plot_results(raw_ref: np.ndarray, raw_surv: np.ndarray, xcorr_array: np.ndarray, args):  # pragma: no cover
    """Generate matplotlib plots for debugging.

    Plots:
      1. First block cross-correlation magnitude (shifted)
      2. Average cross-correlation over all blocks (if >1)
      3. Waterfall (image) of cross-correlation magnitudes (blocks vs delay bins) if >=4 blocks
      4. Power spectral density estimate (FFT magnitude) of reference & surveillance raw captures (first N samples)
    """
    if xcorr_array.size == 0:
        print("[Plot] No cross-correlation data to plot.")
        return
    vector_N = args.vector
    fig_rows = 2
    if xcorr_array.shape[0] >= 4:
        fig_rows = 3
    fig, axes = plt.subplots(fig_rows, 2 if fig_rows == 3 else 1, figsize=(10, 6 if fig_rows==2 else 9))

    # Normalize axes handling
    def get_ax(r, c=0):
        if fig_rows == 2:
            if isinstance(axes, np.ndarray):
                return axes[r]
            return axes
        else:
            return axes[r][c]

    # 1. First block
    first = xcorr_array[0]
    ax1 = get_ax(0)
    ax1.plot(first)
    ax1.set_title("XCorr Magnitude (First Block)")
    ax1.set_xlabel("Delay bin")
    ax1.set_ylabel("Mag")

    # 2. Average (if multiple blocks)
    ax2 = get_ax(1)
    if xcorr_array.shape[0] > 1:
        avg = xcorr_array.mean(axis=0)
        ax2.plot(avg)
        ax2.set_title("XCorr Magnitude (Average)")
    else:
        ax2.plot(first)
        ax2.set_title("XCorr Magnitude (Only Block)")
    ax2.set_xlabel("Delay bin")
    ax2.set_ylabel("Mag")

    # 3. Waterfall if enough blocks
    if fig_rows == 3:
        ax3_left = get_ax(2, 0)
        im = ax3_left.imshow(xcorr_array, aspect='auto', origin='lower', interpolation='nearest')
        ax3_left.set_title("XCorr Waterfall (block vs delay)")
        ax3_left.set_xlabel("Delay bin")
        ax3_left.set_ylabel("Block index")
        fig.colorbar(im, ax=ax3_left, fraction=0.046, pad=0.04)

        # 4. PSD of raw captures (same row right side)
        ax3_right = get_ax(2, 1)
        N_spec = min(vector_N, raw_ref.size, raw_surv.size)
        if N_spec > 0:
            ref_fft = np.fft.fftshift(np.abs(np.fft.fft(raw_ref[:N_spec])))
            surv_fft = np.fft.fftshift(np.abs(np.fft.fft(raw_surv[:N_spec])))
            ax3_right.plot(ref_fft, label='Ref')
            ax3_right.plot(surv_fft, label='Surv', alpha=0.7)
            ax3_right.set_title("FFT Magnitude (First N Samples)")
            ax3_right.set_xlabel("Frequency bin")
            ax3_right.set_ylabel("|X(f)|")
            ax3_right.legend(loc='upper right')
        else:
            ax3_right.set_title("Insufficient samples for FFT plot")
            ax3_right.axis('off')
    else:
        # Add a simple PSD subplot appended below if only two rows wanted
        fig_psd, ax_psd = plt.subplots(figsize=(8,3))
        N_spec = min(vector_N, raw_ref.size, raw_surv.size)
        if N_spec > 0:
            ref_fft = np.fft.fftshift(np.abs(np.fft.fft(raw_ref[:N_spec])))
            surv_fft = np.fft.fftshift(np.abs(np.fft.fft(raw_surv[:N_spec])))
            ax_psd.plot(ref_fft, label='Ref')
            ax_psd.plot(surv_fft, label='Surv', alpha=0.7)
            ax_psd.set_title("FFT Magnitude (First N Samples)")
            ax_psd.set_xlabel("Frequency bin")
            ax_psd.set_ylabel("|X(f)|")
            ax_psd.legend(loc='upper right')
        else:
            ax_psd.set_title("Insufficient samples for FFT plot")
            ax_psd.axis('off')

    plt.tight_layout()
    plt.show()

def main():  # pragma: no cover
    parser = build_arg_parser()
    args = parser.parse_args()

    # Basic validation
    if args.vector <= 0 or (args.vector & (args.vector - 1)) != 0:
        print("[WARN] Vector length N is recommended to be a power of two for FFT efficiency.")

    if args.delay < 0:
        raise SystemExit("Delay must be >= 0")
    if args.chunk <= 0:
        raise SystemExit("Chunk size must be > 0")

    print("Starting capture with parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    raw_ref, raw_surv, xcorr_array = capture_and_process(args)

    if args.plot:
        if plt is None:
            print("[WARN] matplotlib not available; install with: conda install -c conda-forge matplotlib")
        else:
            plot_results(raw_ref, raw_surv, xcorr_array, args)


if __name__ == "__main__":  # pragma: no cover
    main()
