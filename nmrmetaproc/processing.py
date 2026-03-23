"""
Core spectral processing functions.

Pipeline order:
    apodization -> zero-fill -> FFT -> phase correction ->
    chemical-shift referencing -> baseline correction ->
    negative-value handling -> water-region exclusion

All functions operate on 1-D numpy arrays unless stated otherwise.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    import nmrglue as ng
except ImportError as e:
    raise ImportError("nmrglue is required: pip install nmrglue") from e

from nmrmetaproc.utils import (
    TSP_SEARCH_RANGE,
    TSP_PPM,
    WATER_REGION,
    next_power_of_two,
    ppm_range_to_slice,
    ppm_to_index,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Apodization
# ---------------------------------------------------------------------------

def apodize_exponential(
    fid: np.ndarray,
    lb: float,
    sw: float,
) -> np.ndarray:
    """Apply exponential line-broadening (EM apodization).

    Parameters
    ----------
    fid:
        Complex time-domain FID.
    lb:
        Line-broadening factor in Hz.
    sw:
        Spectral width in Hz.

    Returns
    -------
    np.ndarray
        Apodized FID.
    """
    n = len(fid)
    t = np.arange(n) / sw
    window = np.exp(-np.pi * lb * t)
    return fid * window


# ---------------------------------------------------------------------------
# Zero-filling
# ---------------------------------------------------------------------------

def zero_fill(fid: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-fill FID to the next power of 2 at least *factor* x original size.

    Parameters
    ----------
    fid:
        Complex time-domain FID.
    factor:
        Minimum multiplier (default 2x).

    Returns
    -------
    np.ndarray
        Zero-padded FID.
    """
    target = next_power_of_two(len(fid) * factor)
    padded = np.zeros(target, dtype=complex)
    padded[: len(fid)] = fid
    return padded


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

def fourier_transform(fid: np.ndarray) -> np.ndarray:
    """Apply FFT and shift zero-frequency component to center.

    Parameters
    ----------
    fid:
        Complex zero-filled FID.

    Returns
    -------
    np.ndarray
        Complex frequency-domain spectrum (real part is the absorption signal).
    """
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    return spectrum


def build_ppm_axis(params: dict, n_points: int) -> np.ndarray:
    """Construct a chemical-shift axis in ppm.

    Parameters
    ----------
    params:
        Parsed acqus parameters (must contain SW, SFO1, O1, SF).
    n_points:
        Number of spectral points after FFT.

    Returns
    -------
    np.ndarray
        ppm axis, descending (high ppm on left, standard NMR convention).
    """
    # SW is spectral width in ppm; SFO1 is transmitter frequency in MHz
    sw_ppm = float(params.get("SW", 20.0))
    sfo1 = float(params.get("SFO1", 600.0))
    o1 = float(params.get("O1", 0.0))       # offset in Hz
    sf = float(params.get("SF", sfo1))      # reference frequency

    # Centre of spectrum in ppm
    centre_ppm = o1 / sf

    ppm_start = centre_ppm + sw_ppm / 2.0
    ppm_end = centre_ppm - sw_ppm / 2.0
    ppm_axis = np.linspace(ppm_start, ppm_end, n_points)
    return ppm_axis


# ---------------------------------------------------------------------------
# Phase correction
# ---------------------------------------------------------------------------

def auto_phase(spectrum: np.ndarray) -> np.ndarray:
    """Automatic phase correction using the ACME algorithm via nmrglue.

    Parameters
    ----------
    spectrum:
        Complex frequency-domain spectrum.

    Returns
    -------
    np.ndarray
        Phase-corrected real spectrum (absorption mode).
    """
    # nmrglue autops expects a real spectrum; we pass the complex array and
    # work on the result.  The function optimises zero-order (p0) and
    # first-order (p1) phase corrections.
    phased = ng.proc_autophase.autops(spectrum, "acme")
    return np.real(phased)


# ---------------------------------------------------------------------------
# Chemical-shift referencing
# ---------------------------------------------------------------------------

def reference_to_tsp(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Calibrate the spectrum so TSP is at 0.00 ppm.

    Searches for the tallest peak in the TSP region (-0.05 to 0.05 ppm),
    then shifts the entire ppm axis by the offset.

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    ppm_axis:
        Corresponding ppm axis.

    Returns
    -------
    ppm_axis_corrected:
        Shifted ppm axis.
    shift_ppm:
        Applied shift (ppm).
    """
    lo, hi = TSP_SEARCH_RANGE
    start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
    region = spectrum[start:stop]

    if region.size == 0:
        logger.warning("TSP search region is empty; skipping referencing.")
        return ppm_axis, 0.0

    peak_idx_local = int(np.argmax(region))
    peak_ppm = ppm_axis[start + peak_idx_local]
    shift_ppm = peak_ppm - TSP_PPM

    return ppm_axis - shift_ppm, shift_ppm


# ---------------------------------------------------------------------------
# Baseline correction — Asymmetric Least Squares
# ---------------------------------------------------------------------------

def baseline_als(
    spectrum: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    max_iter: int = 10,
) -> np.ndarray:
    """Asymmetric least-squares baseline correction (Eilers & Boelens 2005).

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    lam:
        Smoothness parameter (larger = smoother baseline).
    p:
        Asymmetry parameter (0 < p < 1; small values favour negative residuals
        which pulls the baseline to the bottom).
    max_iter:
        Number of ALS iterations.

    Returns
    -------
    np.ndarray
        Baseline-corrected spectrum (spectrum - estimated_baseline).
    """
    n = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = sparse.csc_matrix(lam * D.T.dot(D))
    w = np.ones(n)

    for _ in range(max_iter):
        W = sparse.diags(w)
        Z = W + H
        baseline = spsolve(Z, w * spectrum)
        w = np.where(spectrum > baseline, p, 1 - p)

    return spectrum - baseline


# ---------------------------------------------------------------------------
# Negative-value handling
# ---------------------------------------------------------------------------

def handle_negatives(
    spectrum: np.ndarray,
    sample_id: str = "",
    noise_factor: float = 3.0,
) -> Tuple[np.ndarray, bool]:
    """Set residual negatives below the noise floor to zero.

    After proper phasing and baseline correction, small negative values
    are noise artefacts.  Negatives deeper than *noise_factor* x std(noise)
    indicate a processing problem and are flagged.

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    sample_id:
        Label for logging.
    noise_factor:
        Threshold multiplier on noise std.

    Returns
    -------
    spectrum_cleaned:
        Spectrum with sub-threshold negatives zeroed.
    flag:
        True if >5 % of points were negative (warning condition).
    """
    # Estimate noise from the last 10% of the spectrum (typically flat region)
    noise_region = spectrum[int(0.9 * len(spectrum)):]
    noise_std = float(np.std(noise_region)) if noise_region.size > 0 else 1e-6
    threshold = -noise_factor * noise_std

    neg_mask = spectrum < threshold
    frac_neg = neg_mask.sum() / len(spectrum)
    flag = frac_neg > 0.05

    if flag:
        logger.warning(
            "Sample %s: %.1f%% of points are negative after processing.",
            sample_id,
            frac_neg * 100,
        )

    cleaned = spectrum.copy()
    cleaned[cleaned < 0] = 0.0
    return cleaned, flag


# ---------------------------------------------------------------------------
# Region exclusion
# ---------------------------------------------------------------------------

def exclude_regions(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    regions: List[Tuple[float, float]],
) -> np.ndarray:
    """Zero out specified ppm regions (e.g. water, TSP).

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    ppm_axis:
        Corresponding ppm axis.
    regions:
        List of (low_ppm, high_ppm) tuples to zero out.

    Returns
    -------
    np.ndarray
        Spectrum with excluded regions set to zero.
    """
    out = spectrum.copy()
    for lo, hi in regions:
        start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
        out[start:stop] = 0.0
    return out


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_spectrum(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    bin_width: float = 0.01,
    ppm_min: float = 0.5,
    ppm_max: float = 9.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform bucket integration (rectangular binning).

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    ppm_axis:
        Corresponding ppm axis.
    bin_width:
        Width of each bin in ppm.
    ppm_min:
        Low end of the region to retain.
    ppm_max:
        High end of the region to retain.

    Returns
    -------
    bin_values:
        Integrated intensity per bin.
    bin_centres:
        ppm value at the centre of each bin.
    """
    bins = np.arange(ppm_min, ppm_max + bin_width, bin_width)
    n_bins = len(bins) - 1
    bin_values = np.zeros(n_bins)
    bin_centres = (bins[:-1] + bins[1:]) / 2.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
        if stop > start:
            seg = spectrum[start:stop]
            # np.trapezoid available in NumPy >= 2.0; fall back to np.trapz
            if hasattr(np, 'trapezoid'):
                bin_values[i] = np.trapezoid(seg)
            else:
                bin_values[i] = np.trapz(seg)

    return bin_values, bin_centres
