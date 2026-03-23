"""
Spectral alignment algorithms.

Two strategies are available:
    1. Reference-peak alignment: shift each spectrum so a chosen reference
       peak (e.g. TSP) aligns exactly. Fast and interpretable.
    2. Icoshift-style correlation-optimized shifting: maximise cross-
       correlation with a reference (or mean) spectrum over a defined
       shifting window. More robust for complex metabolite regions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum shift window (in number of bins) for icoshift-style alignment
DEFAULT_MAX_SHIFT: int = 50


# ---------------------------------------------------------------------------
# Reference-peak alignment
# ---------------------------------------------------------------------------

def align_to_reference_peak(
    spectra: np.ndarray,
    ppm_axis: np.ndarray,
    ref_ppm: float,
    search_width: float = 0.1,
) -> Tuple[np.ndarray, List[int]]:
    """Shift each spectrum so the tallest peak near *ref_ppm* lands exactly there.

    Parameters
    ----------
    spectra:
        2-D array (n_samples x n_points).
    ppm_axis:
        1-D ppm axis.
    ref_ppm:
        Target chemical shift for the reference peak.
    search_width:
        Half-width of the search window in ppm.

    Returns
    -------
    aligned:
        Shifted spectra (same shape as input).
    shifts:
        Integer shift applied to each spectrum (in data-point units).
    """
    from nmrmetaproc.utils import ppm_range_to_slice, ppm_to_index

    lo = ref_ppm - search_width
    hi = ref_ppm + search_width
    start, stop = ppm_range_to_slice(ppm_axis, lo, hi)
    target_idx = ppm_to_index(ppm_axis, ref_ppm)

    aligned = np.zeros_like(spectra)
    shifts: List[int] = []

    for i, sp in enumerate(spectra):
        region = sp[start:stop]
        if region.size == 0:
            aligned[i] = sp
            shifts.append(0)
            continue
        peak_local = int(np.argmax(region))
        peak_global = start + peak_local
        shift = target_idx - peak_global
        aligned[i] = np.roll(sp, shift)
        shifts.append(int(shift))

    return aligned, shifts


# ---------------------------------------------------------------------------
# Icoshift-style cross-correlation alignment
# ---------------------------------------------------------------------------

def icoshift_align(
    spectra: np.ndarray,
    reference: Optional[np.ndarray] = None,
    max_shift: int = DEFAULT_MAX_SHIFT,
) -> Tuple[np.ndarray, List[int]]:
    """Correlation-optimized spectral alignment inspired by icoshift.

    Aligns each spectrum to *reference* (or the column-mean if None) by
    finding the integer shift that maximises normalised cross-correlation,
    within a window of [-max_shift, +max_shift] data points.

    Parameters
    ----------
    spectra:
        2-D array (n_samples x n_points).
    reference:
        Target spectrum.  If None, the column-mean is used.
    max_shift:
        Maximum allowed shift in data points.

    Returns
    -------
    aligned:
        Aligned spectra.
    shifts:
        Integer shift applied to each spectrum.
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)

    ref_norm = _normalise_for_xcorr(reference)
    aligned = np.zeros_like(spectra)
    shifts: List[int] = []

    for i, sp in enumerate(spectra):
        sp_norm = _normalise_for_xcorr(sp)
        xcorr = np.correlate(sp_norm, ref_norm, mode="full")
        n = len(sp)
        lags = np.arange(-(n - 1), n)

        # Restrict to allowed window
        mask = np.abs(lags) <= max_shift
        best_lag = lags[mask][int(np.argmax(xcorr[mask]))]
        aligned[i] = np.roll(sp, -int(best_lag))
        shifts.append(int(-best_lag))

    return aligned, shifts


def _normalise_for_xcorr(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm normalisation."""
    a = arr - arr.mean()
    norm = np.linalg.norm(a)
    return a / norm if norm > 0 else a
