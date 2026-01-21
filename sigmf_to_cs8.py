#!/usr/bin/env python3
"""SigMF (.sigmf-meta + .sigmf-data) to .cs8 converter with progress bar and sidecar JSON."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


BYTES_PER_SAMPLE = 2  # ci8: I,Q int8


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = (done / total) * 100.0
    bar_len = 30
    filled = int(round((pct / 100.0) * bar_len))
    empty = bar_len - filled
    print(f"[{'#' * filled}{'-' * empty}] {pct:5.1f}%", end="\r", flush=True)


def _load_sigmf(meta_path: str):
    try:
        from sigmf import SigMFFile, sigmffile
        # required library: official 'sigmf' package, for working with online datasets.
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


def _convert(
    meta_path: str,
    data_path: str,
    output_path: str,
    start: int,
    count: Optional[int],
) -> dict:
    sigmf_file, SigMFFile = _load_sigmf(meta_path)

    datatype = sigmf_file.get_global_field(SigMFFile.DATATYPE_KEY)
    if not datatype or not datatype.startswith("ci8"):
        raise ValueError(f"Unsupported datatype {datatype!r}; only 'ci8' is supported.")

    data_path = data_path or _resolve_data_path(sigmf_file, meta_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")

    file_size = os.path.getsize(data_path)
    start_bytes = start * BYTES_PER_SAMPLE
    if start_bytes >= file_size:
        raise ValueError("start is beyond end of data file.")

    if count is None:
        total_bytes = file_size - start_bytes
    else:
        total_bytes = min(count * BYTES_PER_SAMPLE, file_size - start_bytes)

    if total_bytes <= 0:
        raise ValueError("No data to convert (total_bytes <= 0).")

    total_samples = total_bytes // BYTES_PER_SAMPLE
    chunk_samples = max(1, (1024 * 1024) // BYTES_PER_SAMPLE)
    done_samples = 0

    with open(output_path, "wb") as fout:
        sample_idx = start
        while done_samples < total_samples:
            to_read = min(chunk_samples, total_samples - done_samples)
            chunk = sigmf_file.read_samples(sample_idx, to_read)
            if chunk is None or chunk.size == 0:
                break

            i = np.real(chunk)
            q = np.imag(chunk)
            scale = 1.0
            if np.issubdtype(i.dtype, np.floating) or np.issubdtype(q.dtype, np.floating):
                max_abs = float(np.max(np.abs(i)))
                max_abs = max(max_abs, float(np.max(np.abs(q))))
                if max_abs <= 1.5:
                    scale = 127.0

            interleaved = np.empty(chunk.size * 2, dtype=np.int8)
            interleaved[0::2] = np.clip(np.rint(i * scale), -128, 127).astype(np.int8)
            interleaved[1::2] = np.clip(np.rint(q * scale), -128, 127).astype(np.int8)
            interleaved.tofile(fout)

            done_samples += chunk.size
            sample_idx += chunk.size
            _print_progress(done_samples, total_samples)

    print()

    sample_rate = sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    captures = sigmf_file.get_captures()
    center_frequency = None
    if captures:
        center_frequency = captures[0].get(SigMFFile.FREQUENCY_KEY)

    sidecar = {
        "output_cs8": os.path.abspath(output_path),
        "source_meta": os.path.abspath(meta_path),
        "source_data": os.path.abspath(data_path),
        "core:datatype": datatype,
        "core:sample_rate": sample_rate,
        "core:frequency": center_frequency,
        "core:author": sigmf_file.get_global_field(SigMFFile.AUTHOR_KEY),
        "core:hw": sigmf_file.get_global_field(SigMFFile.HW_KEY),
        "core:recorder": sigmf_file.get_global_field(SigMFFile.RECORDER_KEY),
        "core:collection": sigmf_file.get_global_field(SigMFFile.COLLECTION_KEY),
        "start_sample": start,
        "sample_count": done_samples,
        "bytes_written": done_samples * BYTES_PER_SAMPLE,
    }

    sidecar_path = output_path + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SigMF (ci8) to .cs8 with progress bar.")
    parser.add_argument("--meta", required=True, help="Path to .sigmf-meta JSON file.")
    parser.add_argument("--data", default="", help="Path to .sigmf-data file (optional).")
    parser.add_argument("--out", required=True, help="Output .cs8 path.")
    parser.add_argument("--start", type=int, default=0, help="Start sample index.")
    parser.add_argument("--count", type=int, default=None, help="Number of samples to convert.")

    args = parser.parse_args()
    data_path = args.data or ""

    sidecar = _convert(
        meta_path=args.meta,
        data_path=data_path,
        output_path=args.out,
        start=args.start,
        count=args.count,
    )

    print("Conversion complete.")
    print(f"Sidecar JSON: {args.out}.json")
    if sidecar.get("core:sample_rate") is not None:
        print(f"Sample rate: {sidecar['core:sample_rate']} Hz")
    if sidecar.get("core:frequency") is not None:
        print(f"Center frequency: {sidecar['core:frequency']} Hz")


if __name__ == "__main__":
    main()
