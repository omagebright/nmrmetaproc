"""Unit tests for I/O functions."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nmrmetaproc.io import find_sample_dirs, read_acqus, read_fid
from tests.conftest import generate_fid, write_bruker_sample


@pytest.fixture
def bruker_sample(tmp_path):
    fid, params = generate_fid()
    write_bruker_sample(tmp_path / "sample1", fid, params)
    return tmp_path / "sample1", fid, params


class TestFindSampleDirs:
    def test_finds_valid_dirs(self, tmp_path):
        for i in range(3):
            sdir = tmp_path / f"sample{i}"
            fid, params = generate_fid(seed=i)
            write_bruker_sample(sdir, fid, params)
        found = find_sample_dirs(tmp_path)
        assert len(found) == 3

    def test_ignores_dirs_without_fid(self, tmp_path):
        (tmp_path / "empty").mkdir()
        found = find_sample_dirs(tmp_path)
        assert len(found) == 0

    def test_nested_discovery(self, tmp_path):
        deep = tmp_path / "group" / "subgroup" / "sample1"
        fid, params = generate_fid()
        write_bruker_sample(deep, fid, params)
        found = find_sample_dirs(tmp_path)
        assert len(found) == 1


class TestReadAcqus:
    def test_returns_dict(self, bruker_sample):
        sdir, _, _ = bruker_sample
        params = read_acqus(sdir / "acqus")
        assert isinstance(params, dict)

    def test_key_params_present(self, bruker_sample):
        sdir, _, _ = bruker_sample
        params = read_acqus(sdir / "acqus")
        assert "SW" in params
        assert "SFO1" in params

    def test_numeric_values_parsed(self, bruker_sample):
        sdir, _, _ = bruker_sample
        params = read_acqus(sdir / "acqus")
        assert isinstance(params["SW"], float)
        assert isinstance(params["TD"], int)


class TestReadFid:
    def test_returns_complex(self, bruker_sample):
        sdir, orig_fid, _ = bruker_sample
        fid, params = read_fid(sdir)
        assert np.iscomplexobj(fid)

    def test_length_matches_td(self, bruker_sample):
        sdir, orig_fid, orig_params = bruker_sample
        fid, params = read_fid(sdir)
        assert len(fid) == orig_params["TD"]

    def test_signal_roundtrip(self, bruker_sample):
        sdir, orig_fid, _ = bruker_sample
        fid, _ = read_fid(sdir)
        # Values should be close (int32 quantisation limits precision)
        np.testing.assert_allclose(np.real(fid), np.real(orig_fid).astype(np.int32), atol=1.0)

    def test_missing_fid_raises(self, tmp_path):
        sdir = tmp_path / "bad_sample"
        sdir.mkdir()
        (sdir / "acqus").write_text("##$TD= 1024\n##END=", encoding="latin-1")
        with pytest.raises(FileNotFoundError):
            read_fid(sdir)

    def test_missing_acqus_raises(self, tmp_path):
        sdir = tmp_path / "bad_sample2"
        sdir.mkdir()
        (sdir / "fid").write_bytes(b"\x00" * 8)
        with pytest.raises(FileNotFoundError):
            read_fid(sdir)
