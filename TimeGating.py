"""TimeGating
==============

This module provides utility functions for applying a Tukey‑window
time gate to frequency response data and extracting the resulting
far‑field radiation pattern.  The implementation is adapted from
the original LCPAR project and exposes a set of stateless helpers
that operate on NumPy arrays.  A Tukey window smoothly tapers the
edges of the gated region to reduce spectral leakage while still
retaining most of the direct‑path energy.

Key functions include:

* ``impulse_response``: convert a frequency response into its
  corresponding impulse response via an IFFT with optional
  zero‑padding.
* ``apply_time_gate``: apply a Tukey window centred on the strongest
  early peak of the impulse response.  The width and shape of the
  window are configurable.  Optionally the centre index may be
  provided directly.
* ``gated_frequency_response``: transform the gated impulse response
  back to the frequency domain via an FFT.
* ``extract_pattern``: extract a polar radiation pattern in dB using
  the magnitude of the DC bin (bin 0) for each angle.
* ``apply_time_gating_matrix``: a convenience function combining the
  above into a single pipeline operating on a matrix of complex
  responses across angles and frequencies【205607447554643†L10-L60】.
"""

from __future__ import annotations

import numpy as np
# Import Savitzky–Golay filter from SciPy.  Some SciPy distributions do
# not expose the Tukey window function; we therefore import only
# savgol_filter and implement our own Tukey window below.
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# Window functions
# -----------------------------------------------------------------------------
def tukey(M: int, alpha: float = 0.5) -> np.ndarray:
    """
    Construct a Tukey (tapered cosine) window.

    This implementation mirrors ``scipy.signal.windows.tukey`` for cases
    where SciPy may not provide it.  The Tukey window is essentially a
    cosine taper at both ends of a rectangular window.  The parameter
    ``alpha`` specifies the fraction of the window inside the cosine
    tapered regions.  ``alpha=0`` yields a rectangular window and
    ``alpha=1`` yields a Hann window.

    Parameters
    ----------
    M : int
        Number of samples in the window.  ``M`` must be positive.
    alpha : float, optional
        Fraction of the window length occupied by the taper on each
        side.  ``alpha`` must satisfy ``0 <= alpha <= 1``.

    Returns
    -------
    np.ndarray
        The Tukey window of length ``M``.
    """
    if M <= 0:
        return np.zeros(0, dtype=float)
    if alpha <= 0:
        return np.ones(M, dtype=float)
    if alpha >= 1:
        # Hann window when alpha == 1
        return np.hanning(M)
    # Create symmetric taper
    n = np.arange(M, dtype=float)
    # Half period of cosine portion
    per = alpha * (M - 1) / 2.0
    w = np.ones(M, dtype=float)
    # Rising cosine portion
    first = int(np.floor(per))
    if first > 0:
        t = n[:first]
        w[:first] = 0.5 * (1 + np.cos(np.pi * ((2.0 * t) / (alpha * (M - 1)) - 1)))
    # Falling cosine portion
    last = int(np.floor(per))
    if last > 0:
        t = n[-last:]
        w[-last:] = 0.5 * (1 + np.cos(np.pi * ((2.0 * (t - (M - last))) / (alpha * (M - 1)) - 1)))
    return w


def impulse_response(freq_resp: np.ndarray, N_fft: int) -> np.ndarray:
    """Convert frequency response (angles × freqs) to impulse response.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape (N_angles, N_freqs).
    N_fft : int
        FFT size for zero‑padding before the IFFT.

    Returns
    -------
    np.ndarray
        Complex matrix of shape (N_angles, N_fft) containing the
        impulse response for each angle.
    """
    return np.fft.ifft(freq_resp, n=N_fft, axis=1)


def _find_gate_center_idx(h_t: np.ndarray, fs: float) -> int:
    """Locate the strongest early peak in the impulse response.

    The direct path arrival is assumed to occur early in the impulse
    response.  This helper computes the mean power across angles in
    the first quarter of the time axis and returns the index of the
    maximum power.  This value is used to centre the Tukey gate.

    Parameters
    ----------
    h_t : np.ndarray
        Impulse response of shape (N_angles, N_time).
    fs : float
        Sampling rate (Hz).  Currently unused but kept for API
        consistency with ``apply_time_gate``.

    Returns
    -------
    int
        Index of the direct‑path peak within the time axis.
    """
    N_angles, N_time = h_t.shape
    early = max(1, N_time // 4)
    power_mean = np.mean(np.abs(h_t[:, :early])**2, axis=0)
    return int(np.argmax(power_mean))


def apply_time_gate(
    h_t: np.ndarray,
    fs: float,
    *,
    gate_ns: float = 10.0,
    alpha: float = 0.5,
    center_idx: int | None = None,
) -> np.ndarray:
    """Apply a Tukey window time gate around the strongest early peak.

    Parameters
    ----------
    h_t : np.ndarray
        Impulse response of shape (N_angles, N_time).
    fs : float
        Sampling rate (Hz) of the impulse response.
    gate_ns : float, optional
        Gate width in nanoseconds.  Converted to a sample count via
        ``fs``.  Defaults to 10 ns.
    alpha : float, optional
        Tukey window shape parameter.  ``alpha=0`` gives a rectangular
        window, ``alpha=1`` gives a Hann window.  Defaults to 0.5.
    center_idx : int or None, optional
        Optional index to centre the gate.  If ``None``, the centre is
        automatically determined via ``_find_gate_center_idx``.

    Returns
    -------
    np.ndarray
        Gated impulse response with the same shape as ``h_t``.
    """
    N_angles, N_time = h_t.shape
    gate_len = max(1, int(np.ceil((gate_ns * 1e-9) * fs)))
    if center_idx is None:
        center_idx = _find_gate_center_idx(h_t, fs)
    start = max(0, center_idx - gate_len // 2)
    end = min(N_time, start + gate_len)
    if end <= start:
        start, end = 0, min(N_time, gate_len)
    win = tukey(end - start, alpha=alpha) if (end - start) > 1 else np.ones(1)
    g = np.zeros(N_time, dtype=float)
    g[start:end] = win
    return h_t * g[np.newaxis, :]


def gated_frequency_response(h_t_gated: np.ndarray, N_fft: int) -> np.ndarray:
    """Transform the gated impulse response back to the frequency domain."""
    return np.fft.fft(h_t_gated, n=N_fft, axis=1)


def extract_pattern(H_gated: np.ndarray) -> np.ndarray:
    """Extract a polar pattern in dB from the gated frequency response.

    The DC bin (bin 0) of the frequency response corresponds to the
    complex mean over the original frequency sweep.  Its magnitude is
    used as the per‑angle value.  The result is normalised so that
    the maximum value is 0 dB【205607447554643†L10-L60】.
    """
    mags = np.abs(H_gated[:, 0])
    mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
    pattern_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
    if pattern_db.size:
        pattern_db = pattern_db - np.max(pattern_db)
    return pattern_db


def denoise_pattern(pattern_db: np.ndarray, *, window: int = 11, poly: int = 3) -> np.ndarray:
    """Smooth a jagged gated pattern using a Savitzky–Golay filter.

    This helper provides a lightweight alternative to wavelet denoising
    and has no external dependencies beyond SciPy.  If the pattern is
    shorter than the specified window size, it is returned unchanged.
    """
    if len(pattern_db) < window:
        return pattern_db
    return savgol_filter(pattern_db, window, poly)


def _next_pow2(n: int) -> int:
    """Return the next power of two greater than or equal to ``n``."""
    p = 1
    while p < n:
        p <<= 1
        
    return p


def _infer_fs_from_freqs(freq_list: np.ndarray, N_fft: int) -> float:
    """Infer a nominal sampling rate from the frequency grid spacing.

    The frequency spacing ``df`` is assumed to be uniform and the
    sampling rate ``fs`` is given by ``fs = df * N_fft``.  At least two
    frequency points are required to perform this inference.
    """
    f = np.asarray(freq_list, dtype=float).ravel()
    if f.size < 2:
        raise ValueError("Need at least 2 frequency points to infer fs; provide fs explicitly.")
    df = float(np.median(np.diff(np.sort(f))))
    return df * float(N_fft)


def apply_time_gating_matrix(
    freq_resp: np.ndarray,
    freq_list: np.ndarray,
    *,
    gate_width_s: float = 25e-9,
    fs: float | None = None,
    tukey_alpha: float = 0.5,
    N_fft: int | None = None,
    denoise_wavelet: bool = True,
) -> np.ndarray:
    """Convenience helper to apply Tukey time gating to a matrix of responses.

    This function performs the following steps:

    1. Zero‑pad the frequency response to length ``N_fft`` and take the
       IFFT to obtain the impulse response.
    2. Apply a Tukey window time gate of width ``gate_width_s`` and
       shape ``tukey_alpha``.  If ``fs`` is not provided it is
       inferred from the frequency spacing.
    3. FFT back to the frequency domain and extract the DC bin
       magnitude for each angle.
    4. Optionally apply a Savitzky–Golay filter to smooth the
       resulting pattern.
    5. Normalise so that the peak is at 0 dB.

    Parameters
    ----------
    freq_resp : np.ndarray
        Complex matrix of shape (N_angles, N_freqs).
    freq_list : np.ndarray
        1‑D array of frequencies (Hz) associated with the columns of
        ``freq_resp``.
    gate_width_s : float, optional
        Width of the time gate in seconds.  Defaults to 25 ns.
    fs : float, optional
        Sampling rate (Hz).  If ``None``, the rate is inferred.
    tukey_alpha : float, optional
        Shape parameter for the Tukey window.
    N_fft : int, optional
        FFT length.  If ``None``, the next power of two greater than
        twice the number of frequency points is used (minimum 256).
    denoise_wavelet : bool, optional
        If ``True``, apply Savitzky–Golay filtering to the pattern.

    Returns
    -------
    np.ndarray
        1‑D array of length ``N_angles`` containing the gated pattern in
        dB, normalised to a peak of 0 dB.
    """
    # Convert input to a 2D complex array
    freq_resp = np.asarray(freq_resp, dtype=np.complex128)
    if freq_resp.ndim != 2:
        raise ValueError("freq_resp must be a 2D array [N_angles, N_freqs].")
    N_angles, N_freqs = freq_resp.shape

    # Handle degenerate single‑frequency case by returning the normalised
    # magnitude pattern without time gating.  Time gating cannot be
    # performed with only one frequency point.  This prevents
    # ValueError from _infer_fs_from_freqs when called below.
    if N_freqs < 2:
        mags = np.abs(freq_resp[:, 0])
        denom = np.max(mags) if np.max(mags) > 0 else 1.0
        mags = mags / denom
        pat_db = 20.0 * np.log10(np.clip(mags, 1e-12, None))
        if pat_db.size:
            pat_db = pat_db - np.max(pat_db)
        # optional smoothing
        if denoise_wavelet:
            pat_db = denoise_pattern(pat_db)
        return pat_db

    # Choose FFT size if not provided.  At least twice the number of
    # frequency points is required to avoid circular overlap when
    # applying the gate.  Round up to the next power of two for
    # efficiency.
    if N_fft is None:
        N_fft = _next_pow2(max(256, 2 * N_freqs))

    # Infer sampling rate from frequency spacing if not supplied
    if fs is None:
        fs = _infer_fs_from_freqs(np.asarray(freq_list, float), N_fft)

    # IFFT to obtain the impulse response
    h_t = impulse_response(freq_resp, N_fft)
    # Apply the Tukey window time gate (gate width specified in seconds)
    h_t_g = apply_time_gate(
        h_t,
        fs,
        gate_ns=(gate_width_s * 1e9),
        alpha=tukey_alpha,
    )
    # FFT back to the frequency domain
    H_g = gated_frequency_response(h_t_g, N_fft)
    # Extract per‑angle pattern and normalise to peak 0 dB
    pat_db = extract_pattern(H_g)
    # Optional Savitzky–Golay smoothing
    if denoise_wavelet:
        pat_db = denoise_pattern(pat_db)
    return pat_db


def print_and_return_data(data):
    """Legacy helper for debugging.

    Prints the length of ``data`` and returns it unchanged.  This function is
    kept for backwards compatibility with code in ``RadioFunctions.py`` that
    expects a passthrough interface when time gating is enabled in
    acquisition modes that do not actually support gating on a single
    frequency.
    """
    arr = np.asarray(data, dtype=float)
    print(f"[TimeGating] Data length: {arr.size}, dtype={arr.dtype}")
    return arr