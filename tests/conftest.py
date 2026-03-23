"""
Pytest fixtures providing synthetic NMR data for testing.

Synthetic FID generation:
    FID = sum of damped sinusoids at known chemical shifts
    Written to disk in Bruker format (fid binary + acqus text)
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic FID generation
# ---------------------------------------------------------------------------

SYNTHETIC_SF = 600.13  # MHz
SYNTHETIC_SW_HZ = 12019.23  # Hz
SYNTHETIC_SW_PPM = SYNTHETIC_SW_HZ / SYNTHETIC_SF  # ~20.03 ppm
SYNTHETIC_TD = 32768
SYNTHETIC_AQ = SYNTHETIC_TD / (2 * SYNTHETIC_SW_HZ)  # acquisition time

# Known peaks: (chemical shift ppm, intensity, T2* s)
SYNTHETIC_PEAKS: List[Tuple[float, float, float]] = [
    (0.00, 5000.0, 0.5),   # TSP
    (1.33, 2000.0, 0.3),   # Lactate CH3
    (2.02, 1500.0, 0.3),   # Acetate CH3
    (3.04, 1800.0, 0.2),   # Creatine N-CH3
    (3.55, 1200.0, 0.25),  # Glycine
    (7.83, 800.0, 0.2),    # Formate
]


def generate_fid(
    peaks: List[Tuple[float, float, float]] = SYNTHETIC_PEAKS,
    td: int = SYNTHETIC_TD,
    sw_hz: float = SYNTHETIC_SW_HZ,
    sf: float = SYNTHETIC_SF,
    noise_level: float = 20.0,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """Generate a synthetic complex FID.

    Parameters
    ----------
    peaks:
        List of (ppm, intensity, T2_star_s) tuples.
    td:
        Number of complex points.
    sw_hz:
        Spectral width in Hz.
    sf:
        Spectrometer frequency in MHz.
    noise_level:
        Gaussian noise standard deviation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    fid:
        Complex 1-D array (td points).
    params:
        Dict matching key acqus parameters.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(td) / (2.0 * sw_hz)  # time axis

    fid = np.zeros(td, dtype=complex)
    for ppm, intensity, t2s in peaks:
        freq_hz = ppm * sf  # frequency relative to carrier
        decay = np.exp(-t / t2s)
        oscillation = np.exp(1j * 2 * np.pi * freq_hz * t)
        fid += intensity * decay * oscillation

    # Add complex noise
    fid += noise_level * (rng.standard_normal(td) + 1j * rng.standard_normal(td))

    params = {
        "TD": td,
        "SW": SYNTHETIC_SW_PPM,
        "SW_h": sw_hz,
        "SFO1": sf,
        "SF": sf,
        "O1": 0.0,
        "BYTORDA": 0,
        "DTYPA": 0,
        "NS": 64,
        "RG": 101,
        "PULPROG": "noesygppr1d",
        "TE": 298.0,
    }
    return fid, params


def write_bruker_sample(
    sample_dir: Path,
    fid: np.ndarray,
    params: dict,
) -> None:
    """Write synthetic FID as Bruker-format files.

    Writes:
        fid     — interleaved int32 little-endian real/imag
        acqus   — minimal Bruker parameter file
    """
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Write fid
    flat = np.empty(len(fid) * 2, dtype=np.int32)
    flat[0::2] = np.real(fid).astype(np.int32)
    flat[1::2] = np.imag(fid).astype(np.int32)
    (sample_dir / "fid").write_bytes(flat.tobytes())

    # Write acqus
    lines = ["##TITLE= Synthetic NMR Data", "##JCAMP-DX= 5.00 Bruker JCAMP library"]
    for key, val in params.items():
        if isinstance(val, str):
            lines.append(f"##${key}= <{val}>")
        else:
            lines.append(f"##${key}= {val}")
    lines.append("##END=")
    (sample_dir / "acqus").write_text("\n".join(lines), encoding="latin-1")


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_fid_and_params():
    """Return a single synthetic (fid, params) tuple."""
    return generate_fid()


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Create a temp directory with 5 synthetic Bruker samples."""
    root = tmp_path_factory.mktemp("nmr_data")
    for i in range(5):
        sdir = root / f"Sample_{i+1:03d}"
        fid, params = generate_fid(noise_level=20.0 + i * 5, seed=i)
        write_bruker_sample(sdir, fid, params)
    return root


@pytest.fixture(scope="session")
def synthetic_data_dir_large(tmp_path_factory):
    """10-sample dataset for integration tests."""
    root = tmp_path_factory.mktemp("nmr_data_large")
    for i in range(10):
        sdir = root / f"Sample_{i+1:03d}"
        fid, params = generate_fid(noise_level=15.0 + i * 3, seed=100 + i)
        write_bruker_sample(sdir, fid, params)
    return root
