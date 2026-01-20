"""
Cross-Ambiguity Function (CAF) utilities.

Provides multiple CAF computation methods with a consistent API:
- Fourier lag product
- Filter bank
- Batch (vectorized filter bank)

Default Doppler convention is centered (fftshift).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[complex]]


@dataclass
class CafResult:
	caf: Optional[np.ndarray]
	caf_mag: np.ndarray
	delay_samples: np.ndarray
	delay_s: np.ndarray
	doppler_hz: np.ndarray


def _as_complex_1d(x: ArrayLike, name: str) -> np.ndarray:
	arr = np.asarray(x)
	if arr.ndim != 1:
		raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
	if not np.iscomplexobj(arr):
		arr = arr.astype(np.complex64)
	return arr


def _next_pow2(n: int) -> int:
	return 1 << (n - 1).bit_length()


def _validate_convention(convention: str) -> None:
	if convention == "ask":
		raise ValueError(
			"CAF convention is set to 'ask'. "
			"Please choose 'centered' or 'uncentered'."
		)
	if convention not in {"centered", "uncentered"}:
		raise ValueError("convention must be 'centered' or 'uncentered'.")


def _make_doppler_bins(
	fs: float,
	nfft: int,
	convention: str,
) -> np.ndarray:
	bins = np.fft.fftfreq(nfft, d=1.0 / fs)
	if convention == "centered":
		bins = np.fft.fftshift(bins)
	return bins


def _apply_convention(
	caf: np.ndarray, doppler_bins: np.ndarray, convention: str
) -> Tuple[np.ndarray, np.ndarray]:
	if convention == "centered":
		caf = np.fft.fftshift(caf, axes=0)
		doppler_bins = np.fft.fftshift(doppler_bins)
	return caf, doppler_bins


def _full_lag_indices(n: int, delays: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
	full_lags = np.arange(-(n - 1), n, dtype=np.int64)
	if delays is None:
		return full_lags, np.arange(full_lags.size)
	delays = np.asarray(delays, dtype=np.int64)
	index_map = {lag: idx for idx, lag in enumerate(full_lags)}
	try:
		indices = np.array([index_map[int(d)] for d in delays], dtype=np.int64)
	except KeyError as exc:
		raise ValueError("delays contain values outside full lag range") from exc
	return delays, indices


def xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""Full cross-correlation via FFT for equal-length arrays."""
	n = a.size
	m = 2 * n - 1
	nfft = _next_pow2(m)
	A = np.fft.fft(a, nfft)
	B = np.fft.fft(b, nfft)
	r = np.fft.ifft(A * np.conj(B))
	return np.concatenate((r[nfft - (n - 1) : nfft], r[0:n]))


def _fourier_lag_product(
	x: np.ndarray,
	y: np.ndarray,
	fs: float,
	delays: np.ndarray,
	doppler_bins: Optional[np.ndarray],
	convention: str,
	nfft: int,
) -> Tuple[np.ndarray, np.ndarray]:
	n = x.size
	if doppler_bins is None:
		doppler_bins = _make_doppler_bins(fs, nfft, "uncentered")
		use_fft_bins = True
		caf = np.zeros((nfft, delays.size), dtype=np.complex64)
	else:
		doppler_bins = np.asarray(doppler_bins, dtype=np.float64)
		use_fft_bins = False
		caf = np.zeros((doppler_bins.size, delays.size), dtype=np.complex64)

	for i, d in enumerate(delays):
		if d >= 0:
			x_seg = x[d:]
			y_seg = y[: n - d]
		else:
			x_seg = x[: n + d]
			y_seg = y[-d:]

		prod = x_seg * np.conj(y_seg)
		if use_fft_bins:
			spec = np.fft.fft(prod, nfft)
		else:
			t = np.arange(prod.size, dtype=np.float64) / fs
			exp_mat = np.exp(-1j * 2 * np.pi * doppler_bins[:, None] * t)
			spec = exp_mat @ prod
		caf[:, i] = spec

	if use_fft_bins:
		caf, doppler_bins = _apply_convention(caf, doppler_bins, convention)

	return caf, doppler_bins


def _filter_bank(
	x: np.ndarray,
	y: np.ndarray,
	fs: float,
	delays: np.ndarray,
	doppler_bins: np.ndarray,
	convention: str,
) -> Tuple[np.ndarray, np.ndarray]:
	n = x.size
	delays, delay_indices = _full_lag_indices(n, delays)

	t = np.arange(n, dtype=np.float64) / fs
	caf = np.zeros((doppler_bins.size, delays.size), dtype=np.complex64)

	for k, fd in enumerate(doppler_bins):
		y_mix = y * np.exp(-1j * 2 * np.pi * fd * t)
		r = xcorr_fft(x, y_mix)
		caf[k, :] = r[delay_indices]

	if convention == "centered":
		caf = np.fft.fftshift(caf, axes=0)
		doppler_bins = np.fft.fftshift(doppler_bins)

	return caf, doppler_bins


def _batch_filter_bank(
	x: np.ndarray,
	y: np.ndarray,
	fs: float,
	delays: np.ndarray,
	doppler_bins: np.ndarray,
	convention: str,
) -> Tuple[np.ndarray, np.ndarray]:
	n = x.size
	delays, delay_indices = _full_lag_indices(n, delays)

	t = np.arange(n, dtype=np.float64) / fs
	phase = np.exp(-1j * 2 * np.pi * doppler_bins[:, None] * t[None, :])
	y_mix = y[None, :] * phase

	m = 2 * n - 1
	nfft = _next_pow2(m)
	X = np.fft.fft(x, nfft)
	Y = np.fft.fft(y_mix, nfft, axis=1)
	r = np.fft.ifft(X[None, :] * np.conj(Y), axis=1)
	full = np.concatenate((r[:, nfft - (n - 1) : nfft], r[:, 0:n]), axis=1)
	caf = full[:, delay_indices]

	if convention == "centered":
		caf = np.fft.fftshift(caf, axes=0)
		doppler_bins = np.fft.fftshift(doppler_bins)

	return caf, doppler_bins


def cross_ambiguity(
	x: ArrayLike,
	y: ArrayLike,
	fs: float,
	*,
	delays: Optional[Iterable[int]] = None,
	doppler_bins: Optional[Union[int, Iterable[float]]] = None,
	method: str = "fourier_lag_product",
	convention: str = "centered",
	return_complex: bool = True,
	window: Optional[np.ndarray] = None,
	normalize: bool = False,
) -> CafResult:
	"""
	Compute the cross-ambiguity function (CAF).

	Args:
		x: Reference signal (complex).
		y: Surveillance signal (complex).
		fs: Sample rate in Hz.
		delays: Iterable of delay samples (ints). Default is full lags.
		doppler_bins: None (auto), int (FFT length), or iterable of Hz bins.
		method: 'fourier_lag_product', 'filter_bank', or 'batch'.
		convention: 'centered', 'uncentered', or 'ask'.
		return_complex: If True, returns complex CAF alongside magnitude.
		window: Optional window applied to x and y before processing.
		normalize: If True, normalize magnitude by max value.

	Returns:
		CafResult containing CAF (complex if requested), magnitude, and axes.
	"""
	_validate_convention(convention)

	x = _as_complex_1d(x, "x")
	y = _as_complex_1d(y, "y")
	if x.size != y.size:
		raise ValueError("x and y must be the same length.")
	if fs <= 0:
		raise ValueError("fs must be positive.")

	n = x.size
	if window is not None:
		window = np.asarray(window, dtype=np.float32)
		if window.shape != (n,):
			raise ValueError("window must be the same length as x and y.")
		x = x * window
		y = y * window

	if delays is None:
		delay_samples = np.arange(-(n - 1), n, dtype=np.int64)
	else:
		delay_samples = np.asarray(list(delays), dtype=np.int64)

	if doppler_bins is None:
		nfft = _next_pow2(n)
		doppler_axis = None
	elif isinstance(doppler_bins, int):
		if doppler_bins <= 0:
			raise ValueError("doppler_bins int must be positive.")
		nfft = doppler_bins
		doppler_axis = None
	else:
		nfft = _next_pow2(n)
		doppler_axis = np.asarray(list(doppler_bins), dtype=np.float64)

	method = method.lower()
	if method == "fourier_lag_product":
		caf, doppler_axis = _fourier_lag_product(
			x, y, fs, delay_samples, doppler_axis, convention, nfft
		)
	elif method == "filter_bank":
		if doppler_axis is None:
			doppler_axis = _make_doppler_bins(fs, nfft, "uncentered")
		caf, doppler_axis = _filter_bank(
			x, y, fs, delay_samples, doppler_axis, convention
		)
	elif method == "batch":
		if doppler_axis is None:
			doppler_axis = _make_doppler_bins(fs, nfft, "uncentered")
		caf, doppler_axis = _batch_filter_bank(
			x, y, fs, delay_samples, doppler_axis, convention
		)
	else:
		raise ValueError("method must be 'fourier_lag_product', 'filter_bank', or 'batch'.")

	caf_mag = np.abs(caf)
	if normalize:
		max_val = np.max(caf_mag)
		if max_val > 0:
			caf_mag = caf_mag / max_val

	if not return_complex:
		caf = None

	delay_s = delay_samples.astype(np.float64) / fs
	return CafResult(caf=caf, caf_mag=caf_mag, delay_samples=delay_samples, delay_s=delay_s, doppler_hz=doppler_axis)
