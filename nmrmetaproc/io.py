"""
I/O module: reading Bruker FID data and writing output files.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import nmrglue as ng
except ImportError as exc:  # nmrglue is declared in pyproject.toml
    raise ImportError("nmrglue is required: pip install nmrglue") from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bruker FID reader
# ---------------------------------------------------------------------------

def find_sample_dirs(root: Path) -> List[Path]:
    """Walk *root* and return directories that contain both ``fid`` and ``acqus``.

    Bruker raw data typically lives in numbered experiment directories
    (e.g. ``sample01/1/``, ``sample01/10/``).  We accept any layout as long
    as both files are present.
    """
    hits: List[Path] = []
    for p in sorted(root.rglob("fid")):
        if (p.parent / "acqus").exists():
            hits.append(p.parent)
    return hits


def read_acqus(acqus_path: Path) -> Dict[str, Any]:
    """Parse a Bruker ``acqus`` parameter file.

    Returns a flat dict with parameter names (without leading ``$``) as keys.
    Multiline arrays are stored as lists.
    """
    params: Dict[str, Any] = {}
    current_key: Optional[str] = None
    array_buf: List[str] = []

    with open(acqus_path, "r", encoding="latin-1") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("$$"):
                continue

            if line.startswith("##$"):
                # flush previous array if any
                if current_key and array_buf:
                    params[current_key] = _parse_value(" ".join(array_buf))
                    array_buf = []
                # parse new key
                rest = line[3:]
                if "=" in rest:
                    key, val = rest.split("=", 1)
                    current_key = key.strip()
                    val = val.strip()
                    if val.startswith("("):
                        # array opener - may continue on next lines
                        array_buf = [val]
                    else:
                        params[current_key] = _parse_value(val)
                        current_key = None
            elif current_key:
                array_buf.append(line)

    # flush any trailing array
    if current_key and array_buf:
        params[current_key] = _parse_value(" ".join(array_buf))

    return params


def _parse_value(val: str) -> Any:
    """Try to cast a string value to int, float, or leave as str."""
    val = val.strip()
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # strip angle-bracket strings like <PROTON>
    if val.startswith("<") and val.endswith(">"):
        return val[1:-1]
    return val


def read_fid(sample_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a Bruker FID file and return the complex time-domain signal.

    Parameters
    ----------
    sample_dir:
        Directory containing ``fid`` and ``acqus``.

    Returns
    -------
    fid_data:
        Complex 1-D numpy array (time-domain FID).
    params:
        Parsed acquisition parameters from ``acqus``.

    Raises
    ------
    FileNotFoundError
        If expected files are missing.
    ValueError
        If byte order or data type cannot be determined.
    """
    fid_path = sample_dir / "fid"
    acqus_path = sample_dir / "acqus"

    if not fid_path.exists():
        raise FileNotFoundError(f"fid not found: {fid_path}")
    if not acqus_path.exists():
        raise FileNotFoundError(f"acqus not found: {acqus_path}")

    params = read_acqus(acqus_path)

    # Determine byte order
    dtypa = params.get("DTYPA", 0)  # 0=int32, 2=float64
    bytorda = params.get("BYTORDA", 0)  # 0=little, 1=big

    if bytorda == 0:
        endian = "<"
    else:
        endian = ">"

    if dtypa == 2:
        dtype = np.dtype(f"{endian}f8")  # float64
    else:
        dtype = np.dtype(f"{endian}i4")  # int32

    raw = np.frombuffer(fid_path.read_bytes(), dtype=dtype).astype(np.float64)

    # Interleaved real/imag
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    fid = raw[0::2] + 1j * raw[1::2]

    # Remove Bruker digital-filter group delay. Without this step, modern Bruker
    # FIDs (DSPFVS=10..21) produce a phase ramp across the spectrum after FFT,
    # creating a phantom intensity ridge near one edge that dominates the bin
    # table.  nmrglue.bruker.remove_digital_filter handles all DSPFVS variants.
    grpdly = params.get("GRPDLY")
    decim = params.get("DECIM")
    dspfvs = params.get("DSPFVS")
    if grpdly is not None and float(grpdly) > 0:
        ng_dic = {"acqus": {"GRPDLY": grpdly, "DECIM": decim, "DSPFVS": dspfvs}}
        try:
            fid = ng.bruker.remove_digital_filter(ng_dic, fid)
        except Exception as exc:  # noqa: BLE001 — fall back so non-Bruker FIDs still work
            logger.warning("remove_digital_filter failed (%s); using raw FID.", exc)

    return fid, params


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    spectral_matrix: pd.DataFrame,
    qc_report: pd.DataFrame,
    acq_params: pd.DataFrame,
    log_text: str,
) -> None:
    """Write all output files to *output_dir*.

    Parameters
    ----------
    output_dir:
        Destination directory (created if absent).
    spectral_matrix:
        Rows = samples, columns = ppm bins.
    qc_report:
        One row per sample with QC metrics.
    acq_params:
        Acquisition parameters table.
    log_text:
        Full processing log as a single string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_matrix.to_csv(output_dir / "spectral_matrix.csv")
    qc_report.to_csv(output_dir / "qc_report.csv", index=False)
    acq_params.to_csv(output_dir / "acquisition_parameters.csv", index=False)

    log_path = output_dir / "processing_log.txt"
    log_path.write_text(log_text, encoding="utf-8")
