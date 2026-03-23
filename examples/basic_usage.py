"""
Basic usage example for nmrmetaproc.

This script demonstrates how to use nmrmetaproc as a Python library
to process a directory of Bruker NMR FID files.
"""

from pathlib import Path
from nmrmetaproc import NMRProcessor

# Initialise the processor with desired parameters
processor = NMRProcessor(
    lb=0.3,                    # Line broadening: 0.3 Hz
    bin_width=0.01,            # Binning: 0.01 ppm
    normalization="pqn",       # PQN normalisation
    ppm_range=(0.5, 9.5),      # Spectral window
    snr_threshold=10.0,        # QC: minimum SNR
    linewidth_threshold=2.5,   # QC: maximum TSP linewidth (Hz)
    align="icoshift",          # Spectral alignment
)

# Point to your data directory (containing numbered experiment folders)
data_dir = Path("/path/to/bruker/data")

# Run the pipeline
results = processor.process(data_dir)

# Access the spectral matrix (pandas DataFrame)
print(f"Spectral matrix: {results.spectral_matrix.shape}")
print(results.spectral_matrix.head())

# Access the QC report
print(results.qc_report[["sample_id", "snr", "linewidth_hz", "passed"]])

# Save all outputs
results.save("./output")
print("Saved to ./output/")
