"""Unit tests for spectral processing functions."""

import numpy as np
import pytest

from tests.conftest import generate_fid, SYNTHETIC_SW_HZ, SYNTHETIC_SF, SYNTHETIC_TD, SYNTHETIC_PEAKS
from nmrmetaproc.processing import (
    apodize_exponential,
    auto_phase,
    baseline_als,
    bin_spectrum,
    build_ppm_axis,
    exclude_regions,
    fourier_transform,
    handle_negatives,
    reference_to_tsp,
    zero_fill,
)


@pytest.fixture
def fid_and_params():
    return generate_fid()


class TestApodization:
    def test_output_shape(self, fid_and_params):
        fid, params = fid_and_params
        result = apodize_exponential(fid, lb=0.3, sw=SYNTHETIC_SW_HZ)
        assert result.shape == fid.shape

    def test_exponential_decay(self, fid_and_params):
        fid, _ = fid_and_params
        result = apodize_exponential(fid, lb=0.3, sw=SYNTHETIC_SW_HZ)
        # First point unchanged (exp(0) = 1), last point attenuated
        assert abs(result[0]) <= abs(fid[0]) * 1.01  # within rounding
        assert abs(result[-1]) < abs(result[0])

    def test_zero_lb_passthrough(self, fid_and_params):
        fid, _ = fid_and_params
        result = apodize_exponential(fid, lb=0.0, sw=SYNTHETIC_SW_HZ)
        np.testing.assert_allclose(np.abs(result), np.abs(fid), rtol=1e-5)


class TestZeroFill:
    def test_power_of_two(self, fid_and_params):
        fid, _ = fid_and_params
        result = zero_fill(fid, factor=2)
        n = len(result)
        assert n >= len(fid) * 2
        assert (n & (n - 1)) == 0, "Length must be a power of 2"

    def test_original_preserved(self, fid_and_params):
        fid, _ = fid_and_params
        result = zero_fill(fid)
        np.testing.assert_array_equal(result[: len(fid)], fid)

    def test_padding_is_zero(self, fid_and_params):
        fid, _ = fid_and_params
        result = zero_fill(fid)
        assert np.all(result[len(fid) :] == 0)


class TestFFT:
    def test_output_length(self, fid_and_params):
        fid, _ = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        assert len(spectrum) == len(zf)

    def test_tsp_peak_present(self, fid_and_params):
        fid, params = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        ppm = build_ppm_axis(params, len(spectrum))
        spec_real = auto_phase(spectrum)
        # TSP peak should be within the expected region
        from nmrmetaproc.utils import ppm_range_to_slice
        start, stop = ppm_range_to_slice(ppm, -0.1, 0.1)
        region = spec_real[start:stop]
        # Must have a clear positive peak
        assert np.max(region) > 0


class TestPpmAxis:
    def test_length(self, fid_and_params):
        _, params = fid_and_params
        axis = build_ppm_axis(params, 65536)
        assert len(axis) == 65536

    def test_descending(self, fid_and_params):
        _, params = fid_and_params
        axis = build_ppm_axis(params, 65536)
        # Standard NMR: high ppm on left (first element)
        assert axis[0] > axis[-1]


class TestPhaseCorrection:
    def test_real_output(self, fid_and_params):
        fid, _ = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        phased = auto_phase(spectrum)
        assert phased.dtype == np.float64 or np.isrealobj(phased)

    def test_no_nan(self, fid_and_params):
        fid, _ = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        phased = auto_phase(spectrum)
        assert not np.any(np.isnan(phased))


class TestChemicalShiftReferencing:
    def test_tsp_at_zero(self, fid_and_params):
        fid, params = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        ppm = build_ppm_axis(params, len(spectrum))
        spec_real = auto_phase(spectrum)
        ppm_corr, shift = reference_to_tsp(spec_real, ppm)

        from nmrmetaproc.utils import ppm_to_index
        idx = ppm_to_index(ppm_corr, 0.0)
        # The corrected ppm axis should have 0.0 close to the TSP peak
        assert abs(ppm_corr[idx]) < 0.02  # within 0.02 ppm

    def test_returns_tuple(self, fid_and_params):
        fid, params = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        ppm = build_ppm_axis(params, len(spectrum))
        spec_real = auto_phase(spectrum)
        result = reference_to_tsp(spec_real, ppm)
        assert len(result) == 2


class TestBaselineCorrection:
    def test_output_shape(self, fid_and_params):
        fid, params = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        spec_real = auto_phase(spectrum)
        corrected = baseline_als(spec_real)
        assert corrected.shape == spec_real.shape

    def test_no_nan_or_inf(self, fid_and_params):
        fid, _ = fid_and_params
        zf = zero_fill(fid)
        spectrum = fourier_transform(zf)
        spec_real = auto_phase(spectrum)
        corrected = baseline_als(spec_real)
        assert not np.any(np.isnan(corrected))
        assert not np.any(np.isinf(corrected))


class TestNegativeHandling:
    def test_no_negatives_in_output(self):
        spectrum = np.array([1.0, -0.001, 2.0, -0.5, 3.0, -0.002])
        cleaned, _ = handle_negatives(spectrum, "test")
        assert np.all(cleaned >= 0), "Output must not contain negative values"

    def test_flag_for_many_negatives(self):
        # >5% negatives should set flag=True
        spectrum = np.full(100, -1.0)
        spectrum[0] = 100.0  # one strong peak
        _, flag = handle_negatives(spectrum, "test")
        assert bool(flag) is True

    def test_no_flag_for_few_negatives(self):
        spectrum = np.ones(100) * 10.0
        spectrum[50] = -0.001  # tiny negative
        _, flag = handle_negatives(spectrum, "test")
        assert bool(flag) is False


class TestRegionExclusion:
    def test_water_region_zeroed(self):
        ppm = np.linspace(10, 0, 1000)
        spectrum = np.ones(1000)
        result = exclude_regions(spectrum, ppm, [(4.5, 5.0)])
        from nmrmetaproc.utils import ppm_range_to_slice
        start, stop = ppm_range_to_slice(ppm, 4.5, 5.0)
        assert np.all(result[start:stop] == 0)
        # Outside region untouched
        assert result[0] == 1.0

    def test_multiple_regions(self):
        ppm = np.linspace(10, 0, 1000)
        spectrum = np.ones(1000)
        result = exclude_regions(spectrum, ppm, [(4.5, 5.0), (0.0, 0.5)])
        assert not np.all(result == 0)  # not everything zeroed


class TestBinning:
    def test_output_length(self):
        ppm = np.linspace(10, 0, 65536)
        spectrum = np.random.rand(65536)
        bv, bc = bin_spectrum(spectrum, ppm, bin_width=0.01, ppm_min=0.5, ppm_max=9.5)
        expected_bins = int((9.5 - 0.5) / 0.01)
        assert len(bv) == expected_bins
        assert len(bc) == expected_bins

    def test_no_negatives_from_positive_input(self):
        ppm = np.linspace(10, 0, 65536)
        spectrum = np.abs(np.random.rand(65536))
        bv, _ = bin_spectrum(spectrum, ppm, bin_width=0.01, ppm_min=0.5, ppm_max=9.5)
        assert np.all(bv >= 0)
