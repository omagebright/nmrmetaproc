# nmrmetaproc

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://github.com/omagebright/nmrmetaproc)

**NMR Metabolomics Spectral Processor**

`nmrmetaproc` converts raw Bruker NMR FID files into clean, analysis-ready spectral matrices (CSV format) suitable for PCA, PLS-DA, pathway analysis, and other downstream metabolomics workflows. It implements a rigorous, reproducible processing pipeline with automatic phase correction, chemical-shift referencing, robust baseline correction, spectral alignment, and Probabilistic Quotient Normalization (PQN).

**Author:** Folorunsho Bright Omage, Ph.D.
**ORCID:** [0000-0002-9750-5034](https://orcid.org/0000-0002-9750-5034)
**Email:** omagefolorunsho@gmail.com

---

## Features

- Reads raw Bruker FID files (`fid` + `acqus`) directly, no conversion needed
- Full processing pipeline in correct order:
  1. Exponential apodization (line broadening)
  2. Zero-filling
  3. Fast Fourier Transform
  4. **Automatic phase correction** (ACME algorithm, no fixed phase values)
  5. Chemical-shift referencing to TSP (0.00 ppm, auto-detected)
  6. Asymmetric least-squares (ALS) baseline correction
  7. Negative-value handling with per-sample logging
  8. Water region exclusion (4.5-5.0 ppm)
  9. Spectral alignment (icoshift-style cross-correlation or reference-peak)
  10. Configurable region exclusion
  11. Uniform binning
  12. **PQN normalization** (default), or total area, TSP reference, none
- Per-sample quality control: SNR, TSP linewidth, water suppression score
- Clean CSV outputs ready for MetaboAnalyst, R, MATLAB
- Works on Windows, macOS, and Linux

---

## Installation

```bash
pip install nmrmetaproc
```

Or from source:

```bash
git clone https://github.com/omagebright/nmrmetaproc.git
cd nmrmetaproc
pip install -e .
```

**Dependencies:** `nmrglue`, `numpy`, `scipy`, `pandas`, `tqdm`

---

## Command-Line Usage

### Full Processing Pipeline

```bash
nmrmetaproc process /path/to/bruker/data --output ./results
```

```bash
nmrmetaproc process /path/to/data \
    --output ./results \
    --lb 0.5 \
    --bin-width 0.005 \
    --normalization pqn \
    --snr-threshold 10 \
    --exclude-regions "4.5-5.0,0.0-0.5"
```

### QC Scan Only

```bash
nmrmetaproc qc /path/to/data --output ./qc_results
```

### Inspect Available Samples

```bash
nmrmetaproc info /path/to/data
```

---

## Python API

```python
from nmrmetaproc import NMRProcessor

processor = NMRProcessor(
    lb=0.3,
    bin_width=0.01,
    normalization="pqn",
    ppm_range=(0.5, 9.5),
    snr_threshold=10.0,
    linewidth_threshold=2.5,
    align="icoshift",
)

results = processor.process("/path/to/bruker/data")

print(results.spectral_matrix)   # rows=samples, columns=ppm bins
print(results.qc_report)         # SNR, linewidth, pass/fail per sample

results.save("./output")
```

---

## Output Files

| File | Description |
|------|-------------|
| `spectral_matrix.csv` | Rows = samples (passed QC), columns = ppm bin centres |
| `qc_report.csv` | SNR, linewidth (Hz), water suppression score, pass/fail per sample |
| `acquisition_parameters.csv` | SW, SFO1, TD, NS, RG, pulse program, temperature per sample |
| `processing_log.txt` | Full processing log with all parameters and per-sample status |

---

## Data Format

Each sample must be in its own directory containing:
- `fid` - binary FID data (interleaved real/imaginary int32)
- `acqus` - acquisition parameter file

```
data_root/
|-- sample_001/
|   |-- fid
|   `-- acqus
`-- sample_002/
    |-- fid
    `-- acqus
```

Nested layouts are also supported and discovered automatically.

---

## Citing

If you use `nmrmetaproc` in your research, please cite:

```
Omage, F. B. (2026). nmrmetaproc: NMR Metabolomics Spectral Processor (Version 1.0.0).
GitHub. https://github.com/omagebright/nmrmetaproc
```

The PQN normalization method:

> Dieterle, F., Ross, A., Schlotterbeck, G., & Senn, H. (2006). Probabilistic quotient
> normalization as robust method to account for dilution of complex biological mixtures.
> *Analytical Chemistry*, 78(13), 4281-4290. https://doi.org/10.1021/ac051632c

---

## Development

```bash
git clone https://github.com/omagebright/nmrmetaproc.git
cd nmrmetaproc
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
