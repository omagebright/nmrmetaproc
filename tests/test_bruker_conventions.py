"""Regression tests for the v1.0.1 fixes:
   1. Bruker digital filter group-delay removal in read_fid
   2. Spectrum reversal in fourier_transform (Bruker convention: high ppm at idx 0)
   3. ALS baseline default lam scaled appropriately for 65k-pt spectra

These bugs collectively caused the 1B7 cross-platform parity check vs MNova v17
to return r ~ 0 instead of r >= 0.98 on the CRT NMR cohort (2026-04-26).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from nmrmetaproc.io import read_fid
from nmrmetaproc.processing import fourier_transform, baseline_als
from nmrmetaproc.processor import NMRProcessor


def _make_minimal_bruker(tmp_path: Path, *, grpdly: int, decim: int = 1680, dspfvs: int = 21) -> Path:
    sample = tmp_path / "sample"
    sample.mkdir()
    n = 32
    data = np.zeros(n * 2, dtype=np.int32)
    data[0] = 100
    (sample / "fid").write_bytes(data.tobytes())
    acqus_lines = [
        "##TITLE= test", "##JCAMP-DX= 5.00",
        f"##$TD= {n * 2}", "##$SW= 20.0", "##$SW_h= 12000.0",
        "##$SFO1= 600.0", "##$SF= 600.0", "##$O1= 0.0",
        "##$BYTORDA= 0", "##$DTYPA= 0",
        f"##$GRPDLY= {grpdly}", f"##$DECIM= {decim}", f"##$DSPFVS= {dspfvs}",
        "##$NS= 1", "##$RG= 1", "##$PULPROG= <zg30>", "##$TE= 298.0", "##END=",
    ]
    (sample / "acqus").write_text("\n".join(acqus_lines), encoding="latin-1")
    return sample


def test_read_fid_removes_digital_filter_when_grpdly_positive(tmp_path):
    sample = _make_minimal_bruker(tmp_path, grpdly=76)
    fid, params = read_fid(sample)
    assert params["GRPDLY"] == 76
    assert len(fid) < 32, "DF removal should have shortened the FID"


def test_read_fid_passthrough_when_grpdly_zero_or_missing(tmp_path):
    sample = _make_minimal_bruker(tmp_path, grpdly=0)
    fid, _ = read_fid(sample)
    assert len(fid) == 32, "FID with GRPDLY=0 must not be shortened"


def test_fourier_transform_returns_bruker_high_ppm_first():
    """A FID with a single positive-frequency tone should produce a peak in
    the FIRST half of the array after FFT+fftshift+reverse (high ppm).
    Without the v1.0.1 reversal the peak appears in the second half — that's the bug."""
    n = 1024
    sw_hz = 12000.0
    t = np.arange(n) / sw_hz
    tone = np.exp(2j * np.pi * 500.0 * t) * np.exp(-t / 0.5)
    fid = np.zeros(2048, dtype=complex)
    fid[:n] = tone
    spec = fourier_transform(fid)
    real = np.real(spec)
    argmax = int(np.argmax(real))
    assert argmax < len(spec) // 2, (
        f"Peak at idx {argmax}/{len(spec)} — expected first half (high ppm). "
        "This means spectrum is NOT reversed and is left-right flipped vs its ppm axis."
    )


def test_default_als_lam_preserves_peak_relative_intensity():
    """v1.0 default lam=1e5 crushes peaks on 65k-pt spectra. v1.0.1 default lam=1e9 must preserve them."""
    n = 65536
    x = np.arange(n)
    centre = n // 4
    width = 200
    peak_height = 1e6
    spectrum = 100.0 + peak_height * np.exp(-((x - centre) ** 2) / (2 * width ** 2))
    corrected = baseline_als(spectrum)
    peak_after = float(corrected[centre])
    flat_after = float(np.median(corrected[: n // 8]))
    assert peak_after > 0.5 * peak_height, (
        f"Default ALS removed too much peak intensity: peak_after={peak_after:.3e} "
        f"(expected > {0.5 * peak_height:.3e}). Default lam may be too small."
    )
    assert abs(flat_after) < peak_height * 0.05


def test_processor_als_default_matches_processing_default():
    import inspect
    from nmrmetaproc import processing
    bl_default = inspect.signature(processing.baseline_als).parameters["lam"].default
    proc_default = inspect.signature(NMRProcessor.__init__).parameters["als_lam"].default
    assert bl_default == proc_default, (
        f"als_lam defaults disagree: baseline_als={bl_default}, NMRProcessor={proc_default}"
    )
    assert bl_default >= 1e8, f"als_lam default ({bl_default}) too small for typical NMR"
