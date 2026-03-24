---
title: 'nmrmetaproc: A Python Package for Automated Processing of Raw Bruker NMR FID Data in Metabolomics Studies'
tags:
  - Python
  - NMR spectroscopy
  - metabolomics
  - spectral processing
  - chemometrics
  - Bruker
authors:
  - name: Folorunsho Bright Omage
    orcid: 0000-0002-9750-5034
    corresponding: true
    affiliation: "1, 2"
  - name: Toyin Bright Omage
    affiliation: 1
  - name: Ljubica Tasic
    orcid: 0000-0003-3002-3796
    corresponding: true
    affiliation: 1
affiliations:
  - name: Institute of Chemistry, University of Campinas (UNICAMP), Campinas, SP, Brazil
    index: 1
    ror: 04wffgt70
  - name: Department of Computer Science, University of Oxford, Oxford, United Kingdom
    index: 2
    ror: 052gg0110
date: 24 March 2026
bibliography: paper.bib
---

# Summary

`nmrmetaproc` is an open-source Python package that converts raw Bruker NMR free induction decay (FID) files into clean, analysis-ready spectral matrices for metabolomics studies. The package implements a complete, reproducible processing pipeline: exponential apodization, zero-filling, fast Fourier transform (FFT), entropy-minimization automatic phase correction (ACME), chemical-shift referencing to sodium 3-(trimethylsilyl)propionate-2,2,3,3-d4 (TSP-d4), asymmetric least-squares (ALS) baseline correction, spectral alignment, water-region exclusion, uniform binning, and Probabilistic Quotient Normalization (PQN). Each sample undergoes automated quality control (QC) assessment for signal-to-noise ratio (SNR), TSP linewidth, and water suppression quality. `nmrmetaproc` provides both a Python API and a command-line interface, producing CSV outputs directly compatible with MetaboAnalyst [@Pang2024], R, and MATLAB.

# Statement of Need

Proton (^1^H) NMR spectroscopy is a cornerstone of untargeted metabolomics, providing quantitative, non-destructive, and highly reproducible measurements of biological mixtures [@Emwas2019]. However, the path from raw FID data to a statistical analysis-ready matrix requires a multi-step processing pipeline where each step introduces user decisions that affect downstream results. In many laboratories, this processing relies on proprietary software such as TopSpin (Bruker), MestReNova (Mestrelab Research), or Chenomx NMR Suite [@Weljie2006], which are expensive, platform-specific, and often require extensive manual intervention. This manual processing creates two problems: (1) it is slow, particularly for large cohort studies with hundreds of samples, and (2) it hinders reproducibility because processing parameters are not systematically recorded.

Open-source alternatives exist at the component level. `nmrglue` [@Helmus2013] provides excellent low-level I/O and processing primitives but requires users to assemble their own pipeline. MetaboAnalyst [@Pang2024] offers web-based processing but does not accept raw Bruker FID files directly. These tools either demand substantial programming effort or require pre-processed data as input.

`nmrmetaproc` fills this gap by providing a single-command, end-to-end pipeline that reads raw Bruker FID data and outputs publication-ready spectral matrices. It is designed for NMR metabolomics researchers, bioinformaticians, and laboratory staff who need reproducible, automated processing without writing custom scripts for each experiment.

# State of the Field

Several software tools address NMR metabolomics processing. TopSpin (Bruker) and MestReNova (Mestrelab Research) are widely used commercial packages that provide comprehensive processing capabilities but are closed-source and licensed. Chenomx NMR Suite [@Weljie2006] combines processing with targeted profiling but also requires a commercial license. Among open-source tools, `nmrglue` [@Helmus2013] provides a Python library for reading, writing, and processing NMR data but operates at the function level rather than as a cohort-processing pipeline. BATMAN [@Hao2012] offers Bayesian deconvolution in R but focuses on metabolite quantification rather than spectral matrix preparation. MetaboAnalyst [@Pang2024] provides web-based statistical analysis and accepts pre-binned NMR data but does not process raw FID files.

`nmrmetaproc` differs from these tools in three ways. First, it operates directly on raw Bruker FID/acqus files, eliminating the need for pre-processing in TopSpin or other software. Second, it integrates the complete processing chain (from apodization through normalization) with automated QC in a single reproducible command. Third, it is designed for batch processing of entire cohort directories, automatically discovering sample folders and applying consistent parameters across all samples.

# Software Design

The architecture of `nmrmetaproc` follows a modular pipeline design with six independent modules orchestrated by a central `NMRProcessor` class.

**I/O module** (`io.py`): Recursively discovers Bruker sample directories by locating `fid` + `acqus` file pairs, reads binary FID data (interleaved int32 real/imaginary), and extracts acquisition parameters (spectral width, observe frequency, transmitter offset, number of scans).

**Processing module** (`processing.py`): Implements the core spectral processing chain in strict order: (1) exponential apodization with configurable line broadening; (2) zero-filling to the next power of two; (3) FFT; (4) automatic phase correction using the ACME entropy-minimization algorithm [@Chen2002] as implemented in `nmrglue`; (5) chemical-shift referencing by detecting the TSP peak near 0.0 ppm via peak-picking within a configurable search window; (6) ALS baseline correction [@Eilers2005] using a sparse-matrix implementation for efficiency; (7) water-region exclusion (default 4.5--5.0 ppm); and (8) uniform spectral binning.

**Alignment module** (`alignment.py`): Implements an icoshift-style [@Savorani2010] cross-correlation alignment algorithm that corrects minor chemical-shift variations across samples using FFT-based circular cross-correlation against a median reference spectrum.

**Normalization module** (`normalization.py`): Supports four normalization strategies: PQN [@Dieterle2006] (default), total spectral area, TSP-referenced, or none. PQN is the recommended default for biological samples as it is robust to dilution effects.

**QC module** (`qc.py`): Evaluates each processed spectrum for SNR (computed from a signal region versus a noise region), TSP linewidth at half-maximum, and water suppression quality. Samples failing user-defined thresholds are flagged but retained in the QC report for transparency.

**Processor orchestrator** (`processor.py`): The `NMRProcessor` class accepts all configurable parameters, processes all discovered samples with progress reporting, and returns a `ProcessingResults` dataclass containing the spectral matrix, QC report, acquisition parameters, and a full processing log. Results can be saved to disk with a single method call.

The command-line interface wraps the Python API and supports three subcommands: `process` (full pipeline), `qc` (QC scan only), and `info` (list available samples). All parameters are exposed as command-line flags.

Key design decisions include: (1) using `nmrglue` for low-level I/O and the ACME phase correction implementation rather than re-implementing these well-tested components; (2) employing sparse matrices for ALS baseline correction to handle high-resolution spectra efficiently; and (3) outputting plain CSV files to maximize interoperability with downstream tools.

# Research Impact Statement

`nmrmetaproc` is currently being used in an active clinical NMR metabolomics study investigating metabolic perturbations in venous thromboembolism (VTE) at the Biological Chemistry Laboratory, Institute of Chemistry, UNICAMP. In this study, the package processes over 270 plasma ^1^H NMR spectra from Bruker 600 MHz data, replacing manual TopSpin processing and reducing processing time from hours of manual work to minutes. The package has been shared with laboratory members for routine use. The software includes a comprehensive test suite (unit and integration tests), a Google Colab demonstration notebook, and is installable via PyPI (`pip install nmrmetaproc`), making it immediately accessible to the metabolomics community.

# AI Usage Disclosure

Generative AI tools (Claude, Anthropic) were used to assist with code generation during development and to help draft sections of this manuscript. All AI-generated code was manually reviewed, tested against known datasets, and validated for scientific correctness by the first author. All AI-generated text was reviewed and edited by the authors. The scientific content, experimental design decisions, algorithm selection, and validation are entirely the work of the authors.

# Acknowledgements

F.B.O. acknowledges financial support from the São Paulo Research Foundation (FAPESP, grants 2023/02691-2 and 2025/23708-6). The authors thank Prof. Goran Neshich (Embrapa Digital Agriculture) for research support.

# References
