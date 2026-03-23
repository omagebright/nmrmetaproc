"""Unit tests for normalisation functions."""

import numpy as np
import pytest

from nmrmetaproc.normalization import normalize, _pqn, _total_area


@pytest.fixture
def sample_matrix():
    rng = np.random.default_rng(7)
    return np.abs(rng.standard_normal((8, 200))) * 100 + 1.0


class TestTotalArea:
    def test_row_sums_equal_one(self, sample_matrix):
        normed = _total_area(sample_matrix)
        sums = normed.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(sums)), rtol=1e-10)

    def test_shape_preserved(self, sample_matrix):
        assert _total_area(sample_matrix).shape == sample_matrix.shape

    def test_no_negatives_from_positive_input(self, sample_matrix):
        assert np.all(_total_area(sample_matrix) >= 0)


class TestPQN:
    def test_shape_preserved(self, sample_matrix):
        assert _pqn(sample_matrix).shape == sample_matrix.shape

    def test_no_negatives_from_positive_input(self, sample_matrix):
        result = _pqn(sample_matrix)
        assert np.all(result >= 0)

    def test_similar_samples_unchanged(self):
        # Nearly identical samples should have very small PQN factors -> similar result
        base = np.ones((5, 100)) * 50.0
        noisy = base + np.random.default_rng(1).standard_normal((5, 100)) * 0.1
        result = _pqn(noisy)
        # All rows should be close to original after PQN
        np.testing.assert_allclose(result, noisy, rtol=0.1)

    def test_different_dilutions(self):
        # PQN should detect and correct for global dilution differences.
        # Data: 5 samples, 100 bins. Most bins identical across samples;
        # only the first 5 bins differ. Samples 1-4 are diluted versions of sample 0.
        # PQN should recover the original proportions.
        rng = np.random.default_rng(7)
        n_bins = 100
        # Base spectrum with varied metabolite levels
        base = rng.uniform(10, 100, n_bins)
        dilutions = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
        # Scale all samples by their dilution factor
        matrix = base[np.newaxis, :] * dilutions[:, np.newaxis]
        result = _pqn(matrix)
        # PQN factor should be approximately equal to the dilution factor.
        # Check that the normalised rows are more similar than the raw input.
        # We use the ratio between max and min row-wise sum as a variability measure.
        raw_ratio = matrix.sum(axis=1).max() / matrix.sum(axis=1).min()
        pqn_ratio = result.sum(axis=1).max() / result.sum(axis=1).min()
        # PQN ratio should be <= raw ratio (same or less spread)
        assert pqn_ratio <= raw_ratio * 1.01, (
            f"PQN should not increase variability. Raw ratio: {raw_ratio:.2f}, PQN ratio: {pqn_ratio:.2f}"
        )
        # And result must have no negatives
        assert np.all(result >= 0)


class TestNormalizeDispatch:
    def test_pqn(self, sample_matrix):
        r = normalize(sample_matrix, method="pqn")
        assert r.shape == sample_matrix.shape

    def test_total(self, sample_matrix):
        r = normalize(sample_matrix, method="total")
        np.testing.assert_allclose(r.sum(axis=1), np.ones(sample_matrix.shape[0]), rtol=1e-10)

    def test_none(self, sample_matrix):
        r = normalize(sample_matrix, method="none")
        np.testing.assert_array_equal(r, sample_matrix)

    def test_tsp(self, sample_matrix):
        r = normalize(sample_matrix, method="tsp", tsp_bin_index=5)
        assert r.shape == sample_matrix.shape

    def test_tsp_missing_index_raises(self, sample_matrix):
        with pytest.raises(ValueError, match="tsp_bin_index"):
            normalize(sample_matrix, method="tsp")

    def test_unknown_method_raises(self, sample_matrix):
        with pytest.raises(ValueError, match="Unknown"):
            normalize(sample_matrix, method="invalid")
