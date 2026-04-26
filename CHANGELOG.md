# Changelog

All notable changes to nmrmetaproc are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-04-26

### Fixed

Three Bruker-pipeline correctness bugs that together caused the cohort-level
output of v1.0.0 to be unsuitable for downstream PCA/PLS-DA/ML analysis. The
bugs were discovered during a cross-platform parity check against MNova v17 on
the CRT pediatric thrombosis cohort (Toyin Bright Omage, 2026-04-26):

- **`nmrmetaproc.io.read_fid` now removes the Bruker digital-filter group
  delay** (`GRPDLY`/`DECIM`/`DSPFVS`). Modern Bruker spectrometers (DSPFVS
  ≥ 10) record a digital-filter response in the first ~70 FID points; without
  removing it, FFT produces a phase ramp that creates a phantom intensity
  ridge near one spectral edge. The fix delegates to
  `nmrglue.bruker.remove_digital_filter`, which handles all DSPFVS variants.

- **`nmrmetaproc.processing.fourier_transform` now reverses the spectrum
  after `np.fft.fftshift`** to match the Bruker convention (high ppm at index
  0). Without the reversal, the spectrum array was left-right flipped relative
  to the descending ppm axis built by `build_ppm_axis`, putting metabolite
  peaks at the wrong ppm positions.

- **Default ALS baseline parameter `als_lam` raised from `1e5` to `1e9`**
  (and aligned in `NMRProcessor.__init__`). The previous default was three to
  four orders of magnitude too small for typical 65 536-point NMR spectra,
  causing the smoother to over-fit and crush peak intensities.

### Verified

- All 68 pre-existing tests still pass.
- 5 new regression tests added in `tests/test_bruker_conventions.py`
  exercising each fix.
- End-to-end parity vs MNova v17 on sample 1B7 (CRT cohort): Pearson r
  improves from −0.01 (broken) → 0.984 (passes the 0.98 cross-platform parity
  threshold defined for the Tasic-group pediatric thrombosis manuscript).

### Authors

- Folorunsho Bright Omage (root-cause analysis, fixes, tests)
- Toyin Bright Omage (independent MNova reference data, parity feedback)
- Ljubica Tasic (review)

## [1.0.0] — 2026-04-13

Initial release. See `paper.md` for the full algorithmic description and
intended use.
