"""
Main NMRProcessor class — orchestrates the full processing pipeline.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from nmrmetaproc import __version__, __author__, __orcid__
from nmrmetaproc.io import find_sample_dirs, read_fid, save_results
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
from nmrmetaproc.alignment import icoshift_align
from nmrmetaproc.normalization import NormMethod, normalize
from nmrmetaproc.qc import QCResult, evaluate_sample
from nmrmetaproc.utils import WATER_REGION, Timer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProcessingResults:
    """Container returned by :meth:`NMRProcessor.process`.

    Attributes
    ----------
    spectral_matrix:
        Rows = samples that passed QC, columns = ppm bins.
    qc_report:
        Full QC table (one row per sample, including failed ones).
    acquisition_parameters:
        Acquisition parameters extracted from ``acqus`` files.
    log:
        Full processing log as a string.
    n_passed:
        Number of samples that passed QC.
    n_total:
        Total samples found.
    """

    spectral_matrix: pd.DataFrame
    qc_report: pd.DataFrame
    acquisition_parameters: pd.DataFrame
    log: str
    n_passed: int = 0
    n_total: int = 0

    def save(self, output_dir: str | Path) -> None:
        """Write all result files to *output_dir*."""
        save_results(
            Path(output_dir),
            self.spectral_matrix,
            self.qc_report,
            self.acquisition_parameters,
            self.log,
        )


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class NMRProcessor:
    """Full-pipeline NMR metabolomics spectral processor.

    Parameters
    ----------
    lb:
        Exponential line-broadening in Hz (default 0.3).
    bin_width:
        Binning resolution in ppm (default 0.01).
    normalization:
        Normalisation method: 'pqn', 'total', 'tsp', or 'none'.
    ppm_range:
        (low, high) ppm limits for the retained spectral window.
    exclude_regions:
        Additional ppm regions to zero out, as list of (low, high) tuples.
        The water region (4.5-5.0 ppm) is always excluded.
    snr_threshold:
        Minimum SNR for a sample to pass QC.
    linewidth_threshold:
        Maximum TSP FWHM (Hz) for a sample to pass QC.
    align:
        Spectral alignment strategy: 'icoshift', 'reference', or None.
    als_lam:
        Lambda smoothness parameter for ALS baseline correction.
    als_p:
        Asymmetry parameter for ALS baseline correction.
    """

    def __init__(
        self,
        lb: float = 0.3,
        bin_width: float = 0.01,
        normalization: NormMethod = "pqn",
        ppm_range: Tuple[float, float] = (0.5, 9.5),
        exclude_regions_extra: Optional[List[Tuple[float, float]]] = None,
        snr_threshold: float = 10.0,
        linewidth_threshold: float = 2.5,
        align: Optional[str] = "icoshift",
        als_lam: float = 1e9,
        als_p: float = 0.01,
    ) -> None:
        self.lb = lb
        self.bin_width = bin_width
        self.normalization = normalization
        self.ppm_range = ppm_range
        self.snr_threshold = snr_threshold
        self.linewidth_threshold = linewidth_threshold
        self.align = align
        self.als_lam = als_lam
        self.als_p = als_p

        # Always exclude water; user may add more
        self._exclude: List[Tuple[float, float]] = [WATER_REGION]
        if exclude_regions_extra:
            self._exclude.extend(exclude_regions_extra)

        self._log = StringIO()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, data_dir: str | Path) -> ProcessingResults:
        """Process all Bruker FID samples found under *data_dir*.

        Parameters
        ----------
        data_dir:
            Root directory to scan for FID files.

        Returns
        -------
        ProcessingResults
        """
        timer = Timer()
        data_dir = Path(data_dir)
        self._log = StringIO()

        self._banner()
        self._log_line(f"Scanning {data_dir} ...")
        sample_dirs = find_sample_dirs(data_dir)

        if not sample_dirs:
            self._log_line("No valid FID directories found.")
            return self._empty_results()

        n_total = len(sample_dirs)
        self._log_line(f"Found {n_total} sample directories with valid FID files\n")
        self._log_processing_params()

        # --- per-sample processing ---
        raw_spectra: List[np.ndarray] = []
        raw_ppm_axes: List[np.ndarray] = []
        qc_results: List[QCResult] = []
        acq_rows: List[dict] = []
        sample_ids: List[str] = []

        self._log_line("Processing samples...\n")

        for idx, sdir in enumerate(
            tqdm(sample_dirs, desc="Processing", unit="sample", file=sys.stdout), start=1
        ):
            # Use parent/name to avoid duplicate IDs across groups
            parent = sdir.parent.name
            sample_id = f"{parent}_{sdir.name}" if parent != data_dir.name else sdir.name
            sample_ids.append(sample_id)

            try:
                spectrum, ppm_axis, qc, acq_row = self._process_one(sdir, sample_id)
                raw_spectra.append(spectrum)
                raw_ppm_axes.append(ppm_axis)
                qc_results.append(qc)
                acq_rows.append(acq_row)

                warn_str = " ".join(qc.warnings) if qc.warnings else ""
                status = "?" if qc.passed else "?"
                self._log_line(
                    f"  [{idx:>4d}/{n_total}] {sample_id:<30s} "
                    f"{status} SNR={qc.snr:.1f} LW={qc.linewidth_hz:.1f}Hz "
                    f"{warn_str}"
                )

            except Exception as exc:
                logger.exception("Failed to process %s", sdir)
                qc_results.append(
                    QCResult(
                        sample_id=sample_id,
                        snr=0.0,
                        linewidth_hz=float("nan"),
                        water_suppression_score=float("nan"),
                        passed=False,
                        warnings=[f"PROCESSING ERROR: {exc}"],
                    )
                )
                acq_rows.append({"sample_id": sample_id})
                self._log_line(f"  [{idx:>4d}/{n_total}] {sample_id:<30s} ? ERROR: {exc}")

        # --- alignment (on valid spectra only) ---
        if raw_spectra and self.align:
            raw_spectra = self._align_spectra(raw_spectra, raw_ppm_axes)

        # --- build matrix from passed samples ---
        # Use the first valid ppm axis as the common grid
        if not raw_spectra:
            return self._empty_results()

        ref_ppm = raw_ppm_axes[0] if raw_ppm_axes else np.array([])

        bin_centres: Optional[np.ndarray] = None
        binned_rows: List[np.ndarray] = []
        passed_ids: List[str] = []
        passed_mask = [qr.passed for qr in qc_results]

        for i, (sp, ppm, qr) in enumerate(zip(raw_spectra, raw_ppm_axes, qc_results)):
            if not passed_mask[i]:
                continue
            bv, bc = bin_spectrum(
                sp, ppm,
                bin_width=self.bin_width,
                ppm_min=self.ppm_range[0],
                ppm_max=self.ppm_range[1],
            )
            binned_rows.append(bv)
            if bin_centres is None:
                bin_centres = bc
            passed_ids.append(sample_ids[i])

        n_passed = len(binned_rows)

        if n_passed > 0 and bin_centres is not None:
            matrix_np = np.vstack(binned_rows)
            matrix_np = normalize(matrix_np, method=self.normalization)
            col_labels = [f"{c:.4f}" for c in bin_centres]
            spectral_matrix = pd.DataFrame(matrix_np, index=passed_ids, columns=col_labels)
            spectral_matrix.index.name = "sample_id"
        else:
            spectral_matrix = pd.DataFrame()

        # --- summary ---
        n_warn = sum(1 for qr in qc_results if qr.warnings and qr.passed)
        n_fail = n_total - n_passed
        elapsed = timer.elapsed_str()

        self._log_line(f"\nQuality Summary:")
        self._log_line(f"  Passed:   {n_passed}/{n_total} ({100*n_passed/max(n_total,1):.1f}%)")
        self._log_line(f"  Warnings: {n_warn} (passed with warnings)")
        self._log_line(f"  Failed:   {n_fail}")
        self._log_line(f"\nProcessing complete in {elapsed}.")

        qc_df = pd.DataFrame([
            {
                "sample_id": qr.sample_id,
                "snr": qr.snr,
                "linewidth_hz": qr.linewidth_hz,
                "water_suppression_score": qr.water_suppression_score,
                "passed": qr.passed,
                "warnings": "; ".join(qr.warnings),
            }
            for qr in qc_results
        ])
        acq_df = pd.DataFrame(acq_rows)

        return ProcessingResults(
            spectral_matrix=spectral_matrix,
            qc_report=qc_df,
            acquisition_parameters=acq_df,
            log=self._log.getvalue(),
            n_passed=n_passed,
            n_total=n_total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_one(
        self, sdir: Path, sample_id: str
    ) -> Tuple[np.ndarray, np.ndarray, QCResult, dict]:
        """Process a single sample directory. Returns (spectrum, ppm_axis, qc, acq_row)."""
        fid, params = read_fid(sdir)

        sw_hz = float(params.get("SW_h", params.get("SW", 20.0) * float(params.get("SFO1", 600.0))))

        # 1. Apodization
        spectrum_td = apodize_exponential(fid, lb=self.lb, sw=sw_hz)

        # 2. Zero-fill
        spectrum_td = zero_fill(spectrum_td, factor=2)

        # 3. FFT
        spectrum_fd = fourier_transform(spectrum_td)
        n_pts = len(spectrum_fd)

        # 4. Build ppm axis
        ppm_axis = build_ppm_axis(params, n_pts)

        # 5. Automatic phase correction
        spectrum_real = auto_phase(spectrum_fd)

        # 6. Chemical-shift referencing
        ppm_axis, _ = reference_to_tsp(spectrum_real, ppm_axis)

        # 7. Baseline correction
        spectrum_real = baseline_als(spectrum_real, lam=self.als_lam, p=self.als_p)

        # 8. Negative value handling
        spectrum_real, _ = handle_negatives(spectrum_real, sample_id=sample_id)

        # 9 & 11. Region exclusion
        spectrum_real = exclude_regions(spectrum_real, ppm_axis, self._exclude)

        # 10. QC
        qc = evaluate_sample(
            spectrum_real,
            ppm_axis,
            sample_id=sample_id,
            sw_hz=sw_hz,
            snr_threshold=self.snr_threshold,
            linewidth_threshold=self.linewidth_threshold,
        )

        # Acquisition parameters row
        acq_row = {
            "sample_id": sample_id,
            "SW_hz": sw_hz,
            "SFO1_MHz": params.get("SFO1"),
            "O1_Hz": params.get("O1"),
            "TD": params.get("TD"),
            "NS": params.get("NS"),
            "RG": params.get("RG"),
            "PULPROG": params.get("PULPROG"),
            "TE": params.get("TE"),
        }

        return spectrum_real, ppm_axis, qc, acq_row

    def _align_spectra(
        self,
        spectra: List[np.ndarray],
        ppm_axes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Apply spectral alignment; returns aligned list."""
        if not spectra:
            return spectra

        # Interpolate to a common ppm grid before alignment
        ref_axis = ppm_axes[0]
        n = len(ref_axis)
        matrix = np.zeros((len(spectra), n))

        for i, (sp, ppm) in enumerate(zip(spectra, ppm_axes)):
            if len(sp) == n and np.allclose(ppm, ref_axis, atol=1e-4):
                matrix[i] = sp
            else:
                matrix[i] = np.interp(ref_axis, ppm[::-1], sp[::-1])

        if self.align == "icoshift":
            aligned_matrix, _ = icoshift_align(matrix)
        elif self.align == "reference":
            from nmrmetaproc.alignment import align_to_reference_peak
            aligned_matrix, _ = align_to_reference_peak(matrix, ref_axis, ref_ppm=0.0)
        else:
            aligned_matrix = matrix

        return [aligned_matrix[i] for i in range(len(spectra))]

    def _banner(self) -> None:
        line = "?" * 60
        self._log_line(f"nmrmetaproc v{__version__} — NMR Metabolomics Processor")
        self._log_line(f"Author: {__author__} (ORCID: {__orcid__})")
        self._log_line(line)

    def _log_processing_params(self) -> None:
        self._log_line("Processing Parameters:")
        self._log_line(f"  Line broadening:   {self.lb} Hz")
        self._log_line(f"  Zero-fill:         2x")
        self._log_line(f"  Phase correction:  Automatic (ACME)")
        self._log_line(f"  Baseline:          Asymmetric Least Squares")
        self._log_line(f"  Reference:         TSP (0.00 ppm)")
        self._log_line(f"  Normalization:     {self.normalization.upper()}")
        self._log_line(f"  Binning:           {self.bin_width} ppm ({self.ppm_range[0]:.2f}–{self.ppm_range[1]:.2f} ppm)")
        self._log_line(f"  Alignment:         {self.align or 'none'}")
        self._log_line("")

    def _log_line(self, msg: str) -> None:
        self._log.write(msg + "\n")
        print(msg)

    def _empty_results(self) -> ProcessingResults:
        return ProcessingResults(
            spectral_matrix=pd.DataFrame(),
            qc_report=pd.DataFrame(),
            acquisition_parameters=pd.DataFrame(),
            log=self._log.getvalue(),
            n_passed=0,
            n_total=0,
        )
