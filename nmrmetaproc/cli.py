"""
Command-line interface for nmrmetaproc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nmrglue")

from nmrmetaproc import __version__, __author__, __orcid__
from nmrmetaproc.io import find_sample_dirs


def _parse_regions(regions_str: str) -> List[Tuple[float, float]]:
    """Parse comma-separated ppm range string like '4.5-5.0,0.0-0.5'."""
    result = []
    for part in regions_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            lo_str, hi_str = part.split("-", 1)
            result.append((float(lo_str), float(hi_str)))
        except ValueError:
            print(f"  Warning: could not parse region '{part}', skipping.", file=sys.stderr)
    return result


def cmd_process(args: argparse.Namespace) -> int:
    """Run the full processing pipeline."""
    from nmrmetaproc.processor import NMRProcessor

    extra_regions = []
    if args.exclude_regions:
        extra_regions = _parse_regions(args.exclude_regions)

    processor = NMRProcessor(
        lb=args.lb,
        bin_width=args.bin_width,
        normalization=args.normalization,
        ppm_range=(args.ppm_min, args.ppm_max),
        exclude_regions_extra=extra_regions if extra_regions else None,
        snr_threshold=args.snr_threshold,
        linewidth_threshold=args.linewidth_threshold,
        align=args.align,
    )

    results = processor.process(args.data_dir)

    output_dir = Path(args.output)
    results.save(output_dir)

    print(f"\nOutput:")
    if not results.spectral_matrix.empty:
        nr, nc = results.spectral_matrix.shape
        print(f"  ? {output_dir}/spectral_matrix.csv ({nr} samples × {nc} bins)")
    print(f"  ? {output_dir}/qc_report.csv")
    print(f"  ? {output_dir}/acquisition_parameters.csv")
    print(f"  ? {output_dir}/processing_log.txt")

    return 0 if results.n_passed > 0 else 1


def cmd_qc(args: argparse.Namespace) -> int:
    """Quick QC-only scan (no full processing)."""
    from nmrmetaproc.io import read_fid
    from nmrmetaproc.processing import (
        apodize_exponential, auto_phase, baseline_als,
        build_ppm_axis, fourier_transform, zero_fill,
    )
    from nmrmetaproc.qc import evaluate_sample
    import pandas as pd

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = find_sample_dirs(data_dir)
    if not sample_dirs:
        print("No valid FID directories found.")
        return 1

    print(f"nmrmetaproc v{__version__} — QC scan")
    print(f"Found {len(sample_dirs)} samples\n")

    rows = []
    for sdir in sample_dirs:
        sid = sdir.name
        try:
            fid, params = read_fid(sdir)
            sw_hz = float(params.get("SW_h", params.get("SW", 20.0) * float(params.get("SFO1", 600.0))))
            fid = apodize_exponential(fid, lb=0.3, sw=sw_hz)
            fid = zero_fill(fid)
            spec = fourier_transform(fid)
            ppm = build_ppm_axis(params, len(spec))
            spec_r = auto_phase(spec)
            spec_r = baseline_als(spec_r)
            qc = evaluate_sample(spec_r, ppm, sid, sw_hz)
            status = "PASS" if qc.passed else "FAIL"
            warn = "; ".join(qc.warnings)
            print(f"  {sid:<30s} {status}  SNR={qc.snr:.1f}  LW={qc.linewidth_hz:.2f}Hz  {warn}")
            rows.append({
                "sample_id": sid,
                "snr": qc.snr,
                "linewidth_hz": qc.linewidth_hz,
                "water_suppression_score": qc.water_suppression_score,
                "passed": qc.passed,
                "warnings": warn,
            })
        except Exception as exc:
            print(f"  {sid:<30s} ERROR: {exc}")
            rows.append({"sample_id": sid, "passed": False, "warnings": str(exc)})

    pd.DataFrame(rows).to_csv(output_dir / "qc_report.csv", index=False)
    print(f"\nQC report saved to {output_dir}/qc_report.csv")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show available samples."""
    data_dir = Path(args.data_dir)
    sample_dirs = find_sample_dirs(data_dir)

    print(f"nmrmetaproc v{__version__} — Data Info")
    print(f"Root: {data_dir}\n")

    if not sample_dirs:
        print("No valid FID directories found.")
        return 1

    print(f"Found {len(sample_dirs)} samples:")
    for sdir in sample_dirs:
        size = (sdir / "fid").stat().st_size
        print(f"  {sdir.name:<40s}  FID size: {size:>10,} bytes")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nmrmetaproc",
        description=(
            f"nmrmetaproc v{__version__} — NMR Metabolomics Spectral Processor\n"
            f"Author: {__author__} (ORCID: {__orcid__})"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # --- process ---
    p_proc = sub.add_parser("process", help="Full spectral processing pipeline")
    p_proc.add_argument("data_dir", metavar="DATA_DIR", help="Root directory with Bruker FID data")
    p_proc.add_argument("--output", "-o", default="./results", metavar="DIR", help="Output directory (default: ./results)")
    p_proc.add_argument("--lb", type=float, default=0.3, metavar="HZ", help="Line broadening in Hz (default: 0.3)")
    p_proc.add_argument("--bin-width", type=float, default=0.01, metavar="PPM", dest="bin_width", help="Bin width in ppm (default: 0.01)")
    p_proc.add_argument("--normalization", choices=["pqn", "total", "tsp", "none"], default="pqn", help="Normalisation method (default: pqn)")
    p_proc.add_argument("--ppm-min", type=float, default=0.5, dest="ppm_min", help="Lower ppm limit (default: 0.5)")
    p_proc.add_argument("--ppm-max", type=float, default=9.5, dest="ppm_max", help="Upper ppm limit (default: 9.5)")
    p_proc.add_argument("--snr-threshold", type=float, default=10.0, dest="snr_threshold", help="Minimum SNR to pass QC (default: 10)")
    p_proc.add_argument("--linewidth-threshold", type=float, default=2.5, dest="linewidth_threshold", help="Max TSP linewidth Hz to pass QC (default: 2.5)")
    p_proc.add_argument("--exclude-regions", metavar="RANGES", dest="exclude_regions", help="Additional ppm regions to exclude, e.g. '4.5-5.0,0.0-0.5'")
    p_proc.add_argument("--align", choices=["icoshift", "reference", "none"], default="icoshift", help="Spectral alignment method (default: icoshift)")
    p_proc.set_defaults(func=cmd_process)

    # --- qc ---
    p_qc = sub.add_parser("qc", help="QC scan only (no full processing)")
    p_qc.add_argument("data_dir", metavar="DATA_DIR")
    p_qc.add_argument("--output", "-o", default="./qc_results", metavar="DIR")
    p_qc.set_defaults(func=cmd_qc)

    # --- info ---
    p_info = sub.add_parser("info", help="List available samples")
    p_info.add_argument("data_dir", metavar="DATA_DIR")
    p_info.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "align") and args.align == "none":
        args.align = None

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
