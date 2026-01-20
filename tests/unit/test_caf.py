import os
import sys

import numpy as np

# Ensure project root is on sys.path for direct script runs
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from caf import cross_ambiguity


def _make_signal(n: int, fs: float, delay: int, doppler_hz: float):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    t = np.arange(n, dtype=np.float64) / fs
    y = np.zeros_like(x)
    if delay >= 0:
        y[delay:] = x[: n - delay] * np.exp(1j * 2 * np.pi * doppler_hz * t[: n - delay])
    else:
        y[: n + delay] = x[-delay:] * np.exp(1j * 2 * np.pi * doppler_hz * t[-delay:])
    return x, y


def test_caf_peak_location():
    fs = 1000.0
    n = 256
    delay = 7
    doppler_hz = 60.0

    x, y = _make_signal(n, fs, delay, doppler_hz)
    result = cross_ambiguity(x, y, fs, method="fourier_lag_product", convention="centered")

    caf_mag = result.caf_mag
    peak_idx = np.unravel_index(np.argmax(caf_mag), caf_mag.shape)
    doppler_peak = result.doppler_hz[peak_idx[0]]
    delay_peak = result.delay_samples[peak_idx[1]]

    doppler_err = np.abs(doppler_peak - doppler_hz)
    assert delay_peak == delay
    assert doppler_err < fs / n


def test_caf_methods_consistency():
    fs = 2000.0
    n = 128
    delay = 4
    doppler_hz = 120.0

    x, y = _make_signal(n, fs, delay, doppler_hz)

    res_flp = cross_ambiguity(x, y, fs, method="fourier_lag_product", convention="centered")
    res_fb = cross_ambiguity(x, y, fs, method="filter_bank", convention="centered")
    res_batch = cross_ambiguity(x, y, fs, method="batch", convention="centered")

    mag_flp = res_flp.caf_mag / (np.max(res_flp.caf_mag) + 1e-12)
    mag_fb = res_fb.caf_mag / (np.max(res_fb.caf_mag) + 1e-12)
    mag_batch = res_batch.caf_mag / (np.max(res_batch.caf_mag) + 1e-12)

    assert np.allclose(mag_flp, mag_fb, rtol=1e-3, atol=1e-3)
    assert np.allclose(mag_flp, mag_batch, rtol=1e-3, atol=1e-3)


def test_centered_convention_axis():
    fs = 1000.0
    n = 64
    x = np.ones(n, dtype=np.complex64)
    y = np.ones(n, dtype=np.complex64)

    result = cross_ambiguity(x, y, fs, method="batch", convention="centered")
    doppler = result.doppler_hz

    assert doppler.size > 0
    assert doppler[0] < 0
    assert np.isclose(doppler[doppler.size // 2], 0.0)


def _run_nonpytest(run_all: bool, tests: list[str]):
    available = {
        "peak": test_caf_peak_location,
        "consistency": test_caf_methods_consistency,
        "centered": test_centered_convention_axis,
    }

    selected = available.values() if run_all else [available[name] for name in tests]
    failures = 0

    for fn in selected:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {fn.__name__}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CAF unit tests without pytest.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (default if no --tests provided).",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=[],
        choices=["peak", "consistency", "centered"],
        help="Select specific tests to run.",
    )

    args = parser.parse_args()
    run_all = args.all or not args.tests
    _run_nonpytest(run_all=run_all, tests=args.tests)
