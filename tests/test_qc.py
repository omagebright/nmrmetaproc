"""Unit tests for QC metrics."""

import numpy as np
import pytest

from nmrmetaproc.qc import compute_snr, compute_linewidth, compute_water_suppression, evaluate_sample
from tests.conftest import SYNTHETIC_SW_HZ


def make_spectrum_with_peak(n=65536, ppm_lo=0.0, ppm_hi=10.0, peak_ppm=0.0, peak_height=1000.0):
    """Utility: flat spectrum with a single Gaussian peak."""
    ppm = np.linspace(ppm_hi, ppm_lo, n)
    spec = np.zeros(n)
    peak_idx = int(np.argmin(np.abs(ppm - peak_ppm)))
    width = max(1, n // 500)
    for i in range(max(0, peak_idx - width), min(n, peak_idx + width + 1)):
        spec[i] = peak_height * np.exp(-((i - peak_idx) ** 2) / (2 * (width / 2) ** 2))
    return spec, ppm


class TestSNR:
    def test_high_snr_clean_spectrum(self):
        # Build a spectrum with a strong signal peak and very low noise
        n = 65536
        ppm = np.linspace(10.0, 0.0, n)
        spec = np.zeros(n)
        # Strong peak at 2 ppm (well away from noise estimate region 9.0-9.5)
        peak_idx = int(np.argmin(np.abs(ppm - 2.0)))
        spec[peak_idx] = 5000.0
        # Add tiny noise in the 9.0-9.5 ppm region used by compute_snr
        rng = np.random.default_rng(42)
        n_start = int(np.argmin(np.abs(ppm - 9.5)))
        n_stop = int(np.argmin(np.abs(ppm - 9.0)))
        if n_stop > n_start:
            spec[n_start:n_stop] = rng.standard_normal(n_stop - n_start) * 0.5
        snr = compute_snr(spec, ppm)
        assert snr > 10, f"Expected SNR > 10, got {snr:.1f}"

    def test_pure_noise_low_snr(self):
        rng = np.random.default_rng(42)
        ppm = np.linspace(10, 0, 65536)
        spec = rng.standard_normal(65536) * 5
        snr = compute_snr(spec, ppm)
        # Should be low (close to 1, not > 50)
        assert snr < 50


class TestLinewidth:
    def test_narrow_peak_small_lw(self):
        spec, ppm = make_spectrum_with_peak(peak_height=5000.0, peak_ppm=0.0)
        lw = compute_linewidth(spec, ppm, sw_hz=SYNTHETIC_SW_HZ)
        # Should be finite and positive
        assert np.isfinite(lw)
        assert lw > 0

    def test_empty_tsp_region(self):
        ppm = np.linspace(10, 5, 1000)  # no TSP region
        spec = np.ones(1000)
        lw = compute_linewidth(spec, ppm, sw_hz=SYNTHETIC_SW_HZ)
        assert np.isnan(lw)


class TestWaterSuppression:
    def test_no_water_perfect_score(self):
        spec, ppm = make_spectrum_with_peak(peak_ppm=1.0, peak_height=1000.0)
        score = compute_water_suppression(spec, ppm)
        assert score < 0.05  # essentially zero water

    def test_large_water_peak_high_score(self):
        spec, ppm = make_spectrum_with_peak(peak_ppm=4.7, peak_height=5000.0)
        score = compute_water_suppression(spec, ppm)
        assert score > 0.1


class TestEvaluateSample:
    def test_good_sample_passes(self):
        # Use the same reliable spectrum as the SNR test
        n = 65536
        ppm = np.linspace(10.0, 0.0, n)
        spec = np.zeros(n)
        peak_idx = int(np.argmin(np.abs(ppm - 2.0)))
        spec[peak_idx] = 5000.0
        rng = np.random.default_rng(42)
        n_start = int(np.argmin(np.abs(ppm - 9.5)))
        n_stop = int(np.argmin(np.abs(ppm - 9.0)))
        if n_stop > n_start:
            spec[n_start:n_stop] = rng.standard_normal(n_stop - n_start) * 0.5
        qr = evaluate_sample(spec, ppm, "test", sw_hz=SYNTHETIC_SW_HZ, snr_threshold=10.0)
        assert qr.sample_id == "test"
        assert qr.snr > 10

    def test_noisy_sample_fails_snr(self):
        rng = np.random.default_rng(0)
        ppm = np.linspace(10, 0, 65536)
        spec = rng.standard_normal(65536) * 5  # pure noise
        qr = evaluate_sample(spec, ppm, "noisy", sw_hz=SYNTHETIC_SW_HZ, snr_threshold=10.0)
        assert not qr.passed
        assert any("SNR" in w for w in qr.warnings)
