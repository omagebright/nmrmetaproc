"""Utility helpers used across modules."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WATER_REGION: Tuple[float, float] = (4.5, 5.0)
TSP_SEARCH_RANGE: Tuple[float, float] = (-0.05, 0.05)
TSP_PPM: float = 0.00


def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def ppm_to_index(ppm_axis: np.ndarray, ppm_value: float) -> int:
    """Return the array index closest to *ppm_value* on *ppm_axis*.

    Parameters
    ----------
    ppm_axis:
        1-D array of chemical-shift values (can be ascending or descending).
    ppm_value:
        Target chemical-shift in ppm.

    Returns
    -------
    int
        Index of the closest element.
    """
    return int(np.argmin(np.abs(ppm_axis - ppm_value)))


def ppm_range_to_slice(
    ppm_axis: np.ndarray,
    low: float,
    high: float,
) -> Tuple[int, int]:
    """Return (start, stop) indices enclosing the ppm range [low, high].

    Works whether the axis runs high->low or low->high.
    """
    idx_low = ppm_to_index(ppm_axis, low)
    idx_high = ppm_to_index(ppm_axis, high)
    start = min(idx_low, idx_high)
    stop = max(idx_low, idx_high) + 1
    return start, stop


def format_time(seconds: float) -> str:
    """Format elapsed seconds as 'Xm Ys' or 'X.Xs'."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def elapsed_str(self) -> str:
        return format_time(self.elapsed())
