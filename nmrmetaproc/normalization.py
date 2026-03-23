"""
Normalization strategies for NMR spectral matrices.

Supported methods:
    pqn   — Probabilistic Quotient Normalization (Dieterle et al. 2006)
    total — Total-area (sum) normalization
    tsp   — TSP-peak reference normalization
    none  — No normalization (return as-is)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NormMethod = Literal["pqn", "total", "tsp", "none"]


def normalize(
    matrix: np.ndarray,
    method: NormMethod = "pqn",
    tsp_bin_index: Optional[int] = None,
) -> np.ndarray:
    """Normalise a spectral matrix.

    Parameters
    ----------
    matrix:
        2-D array (n_samples x n_bins).  All values should be >= 0.
    method:
        Normalisation method ('pqn', 'total', 'tsp', 'none').
    tsp_bin_index:
        Column index of the TSP reference peak bin (required for method='tsp').

    Returns
    -------
    np.ndarray
        Normalised matrix (same shape).

    Raises
    ------
    ValueError
        If method is unknown or tsp_bin_index is missing when method='tsp'.
    """
    if method == "none":
        return matrix.copy()
    if method == "total":
        return _total_area(matrix)
    if method == "pqn":
        return _pqn(matrix)
    if method == "tsp":
        if tsp_bin_index is None:
            raise ValueError("tsp_bin_index must be provided for method='tsp'")
        return _tsp_reference(matrix, tsp_bin_index)
    raise ValueError(f"Unknown normalisation method: {method!r}")


def _total_area(matrix: np.ndarray) -> np.ndarray:
    """Divide each sample by its total spectral area."""
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0  # avoid divide-by-zero for empty spectra
    return matrix / totals


def _pqn(matrix: np.ndarray) -> np.ndarray:
    """Probabilistic Quotient Normalization (PQN).

    Algorithm (Dieterle et al. 2006, Anal. Chem.):
        1. Total-area normalize all spectra.
        2. Compute a reference spectrum (column-wise median).
        3. For each sample, compute quotients (sample / reference) for all bins
           where the reference > 0.
        4. The normalisation factor is the median of those quotients.
        5. Divide the original (not total-area) spectrum by that factor.

    Parameters
    ----------
    matrix:
        2-D array (n_samples x n_bins), values >= 0.

    Returns
    -------
    np.ndarray
        PQN-normalised matrix.
    """
    # Step 1: total-area normalise for reference construction only
    ta = _total_area(matrix)

    # Step 2: reference spectrum
    reference = np.median(ta, axis=0)

    # Steps 3-5: compute PQN factor per sample
    normed = np.zeros_like(matrix)
    nonzero_ref = reference > 0

    for i, sample in enumerate(ta):
        if not nonzero_ref.any():
            normed[i] = matrix[i]
            continue
        quotients = sample[nonzero_ref] / reference[nonzero_ref]
        pqn_factor = float(np.median(quotients))
        if pqn_factor <= 0:
            logger.warning(
                "Sample %d: PQN factor <= 0 (%.4g), skipping normalisation.", i, pqn_factor
            )
            normed[i] = matrix[i]
        else:
            # Divide the original (un-normalised) sample by the PQN factor
            normed[i] = matrix[i] / pqn_factor

    return normed


def _tsp_reference(matrix: np.ndarray, tsp_bin_index: int) -> np.ndarray:
    """Normalise each sample to the intensity of the TSP reference peak."""
    tsp_intensities = matrix[:, tsp_bin_index].copy()
    tsp_intensities[tsp_intensities == 0] = 1.0
    return matrix / tsp_intensities[:, np.newaxis]
