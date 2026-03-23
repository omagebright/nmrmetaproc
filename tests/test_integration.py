"""
End-to-end integration tests using synthetic Bruker data.

Verifies the complete pipeline: FID read -> processing -> normalisation ->
output CSV with no negative values, correct dimensions, and valid QC report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nmrmetaproc.processor import NMRProcessor


class TestFullPipeline:
    def test_process_returns_results(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        assert results.n_total == 5

    def test_spectral_matrix_no_negatives(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        if not results.spectral_matrix.empty:
            values = results.spectral_matrix.values
            assert np.all(values >= 0), "Spectral matrix must not contain negative values"

    def test_spectral_matrix_correct_columns(self, synthetic_data_dir):
        proc = NMRProcessor(bin_width=0.01, ppm_range=(0.5, 9.5), align=None)
        results = proc.process(synthetic_data_dir)
        if not results.spectral_matrix.empty:
            expected_bins = int((9.5 - 0.5) / 0.01)
            assert results.spectral_matrix.shape[1] == expected_bins

    def test_qc_report_has_all_samples(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        assert len(results.qc_report) == 5

    def test_qc_report_columns(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        required = {"sample_id", "snr", "linewidth_hz", "passed", "warnings"}
        assert required.issubset(set(results.qc_report.columns))

    def test_log_contains_version(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        assert "nmrmetaproc" in results.log

    def test_save_creates_files(self, synthetic_data_dir, tmp_path):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        results.save(tmp_path)
        assert (tmp_path / "spectral_matrix.csv").exists()
        assert (tmp_path / "qc_report.csv").exists()
        assert (tmp_path / "acquisition_parameters.csv").exists()
        assert (tmp_path / "processing_log.txt").exists()

    def test_pqn_normalization(self, synthetic_data_dir):
        proc = NMRProcessor(normalization="pqn", align=None)
        results = proc.process(synthetic_data_dir)
        if not results.spectral_matrix.empty:
            values = results.spectral_matrix.values
            assert np.all(values >= 0)

    def test_total_normalization(self, synthetic_data_dir):
        proc = NMRProcessor(normalization="total", align=None)
        results = proc.process(synthetic_data_dir)
        if not results.spectral_matrix.empty:
            values = results.spectral_matrix.values
            assert np.all(values >= 0)
            # Each row should sum to approximately 1
            row_sums = values.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_acquisition_params_has_sample_ids(self, synthetic_data_dir):
        proc = NMRProcessor(align=None)
        results = proc.process(synthetic_data_dir)
        assert "sample_id" in results.acquisition_parameters.columns

    def test_icoshift_alignment(self, synthetic_data_dir):
        proc = NMRProcessor(align="icoshift")
        results = proc.process(synthetic_data_dir)
        if not results.spectral_matrix.empty:
            assert np.all(results.spectral_matrix.values >= 0)


class TestEdgeCases:
    def test_empty_directory(self, tmp_path):
        proc = NMRProcessor()
        results = proc.process(tmp_path)
        assert results.n_total == 0
        assert results.spectral_matrix.empty

    def test_single_sample(self, tmp_path):
        from tests.conftest import generate_fid, write_bruker_sample
        fid, params = generate_fid(seed=999)
        write_bruker_sample(tmp_path / "only_sample", fid, params)
        proc = NMRProcessor(align=None)
        results = proc.process(tmp_path)
        assert results.n_total == 1
