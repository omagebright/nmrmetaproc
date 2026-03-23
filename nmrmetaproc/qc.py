"""
Quality-control metrics for processed NMR spectra.

Computed per-sample:
    snr         — Signal-to-noise ratio (signal peak / noise std)
    linewidth   — Full width at half maximum (FWHM) of TSP peak, in Hz
    water_supp  — Water suppression quality (residual water intensity ratio)
    passed      — Boolean gate (snr >= threshold and linewidth <= threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nmrmetaproc.utils import TSP_SEARCH_RANGE, WATER_REGION, ppm_range_to_slice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """Quality-control metrics for a single spectrum."""

    sample_id: str
    snr: float
    linewidth_hz: float
    water_suppression_score: float
    passed: bool
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual metric calculators
# ---------------------------------------------------------------------------

def _signal_region(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    lo: float = 0.5,
    hi: float = 9.5,
) -> np.ndarray:
    """Extract the main spectral signal region."""
    start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
    return spectrum[start:stop]


def compute_snr(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    noise_region: Tuple[float, float] = (9.0, 9.5),
) -> float:
    """Estimate signal-to-noise ratio.

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    ppm_axis:
        Corresponding ppm axis.
    noise_region:
        ppm range assumed to contain only noise.

    Returns
    -------
    float
        Peak-to-noise ratio (max signal in 0.5-8.5 ppm / std of noise region).
    """
    n_start, n_stop = ppm_range_to_slice(ppm_axis, *noise_region)
    noise = spectrum[n_start:n_stop]
    noise_std = float(np.std(noise)) if noise.size > 0 else 1e-9
    if noise_std == 0:
        noise_std = 1e-9

    sig = _signal_region(spectrum, ppm_axis, 0.5, 8.5)
    peak = float(np.max(sig)) if sig.size > 0 else 0.0
    return peak / noise_std


def compute_linewidth(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    sw_hz: float,
) -> float:
    """Estimate FWHM of the TSP peak in Hz.

    Parameters
    ----------
    spectrum:
        1-D real spectrum.
    ppm_axis:
        Corresponding ppm axis.
    sw_hz:
        Spectral width in Hz (used to convert ppm to Hz).

    Returns
    -------
    float
        Linewidth in Hz; returns NaN if TSP peak cannot be located.
    """
    lo, hi = TSP_SEARCH_RANGE
    start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
    region = spectrum[start:stop]

    if region.size < 3:
        return float("nan")

    peak_idx_local = int(np.argmax(region))
    peak_height = region[peak_idx_local]
    half_max = peak_height / 2.0

    # Walk left from peak
    left = peak_idx_local
    while left > 0 and region[left] > half_max:
        left -= 1

    # Walk right from peak
    right = peak_idx_local
    while right < len(region) - 1 and region[right] > half_max:
        right += 1

    fwhm_pts = right - left
    if fwhm_pts == 0:
        return float("nan")

    # Convert points to ppm then to Hz
    ppm_per_point = abs(ppm_axis[1] - ppm_axis[0]) if len(ppm_axis) > 1 else 0.001
    hz_per_ppm = sw_hz / abs(ppm_axis[0] - ppm_axis[-1])
    fwhm_hz = fwhm_pts * ppm_per_point * hz_per_ppm
    return float(fwhm_hz)


def compute_water_suppression(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
) -> float:
    """Score water suppression quality (lower = better suppression).

    Returns the ratio of maximum residual intensity in the water region
    to the maximum of the main signal region.

    Returns
    -------
    float
        Score in [0, inf); 0 = perfect suppression; > 0.1 is poor.
    """
    w_start, w_stop = ppm_range_to_slice(ppm_axis, *WATER_REGION)
    water_intensity = float(np.max(np.abs(spectrum[w_start:w_stop]))) if w_stop > w_start else 0.0

    sig = _signal_region(spectrum, ppm_axis)
    sig_max = float(np.max(sig)) if sig.size > 0 else 1.0
    if sig_max <= 0:
        return 0.0
    return water_intensity / sig_max


# ---------------------------------------------------------------------------
# Per-sample QC
# ---------------------------------------------------------------------------

def evaluate_sample(
    spectrum: np.ndarray,
    ppm_axis: np.ndarray,
    sample_id: str,
    sw_hz: float,
    snr_threshold: float = 10.0,
    linewidth_threshold: float = 2.5,
) -> QCResult:
    """Compute all QC metrics and determine pass/fail.

    Parameters
    ----------
    spectrum:
        1-D real spectrum (post-processing).
    ppm_axis:
        Corresponding ppm axis.
    sample_id:
        Label used in warnings.
    sw_hz:
        Spectral width in Hz (from acqus SW_h or derived).
    snr_threshold:
        Minimum acceptable SNR.
    linewidth_threshold:
        Maximum acceptable FWHM in Hz.

    Returns
    -------
    QCResult
    """
    snr = compute_snr(spectrum, ppm_axis)
    lw = compute_linewidth(spectrum, ppm_axis, sw_hz)
    ws = compute_water_suppression(spectrum, ppm_axis)

    warnings: List[str] = []
    if snr < snr_threshold:
        warnings.append(f"LOW SNR ({snr:.1f} < {snr_threshold})")
    if not np.isnan(lw) and lw > linewidth_threshold:
        warnings.append(f"BROAD LINEWIDTH ({lw:.2f} Hz > {linewidth_threshold} Hz)")
    if ws > 0.1:
        warnings.append(f"POOR WATER SUPPRESSION (score={ws:.3f})")

    passed = (snr >= snr_threshold) and (np.isnan(lw) or lw <= linewidth_threshold)

    return QCResult(
        sample_id=sample_id,
        snr=round(snr, 2),
        linewidth_hz=round(lw, 3) if not np.isnan(lw) else float("nan"),
        water_suppression_score=round(ws, 4),
        passed=passed,
        warnings=warnings,
    )
