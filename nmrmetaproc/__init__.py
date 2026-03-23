# Silence scipy.optimize.fmin convergence output used internally by nmrglue
import scipy.optimize as _scipy_opt
_scipy_opt_fmin_orig = _scipy_opt.fmin
def _fmin_silent(fn, x0, *args, **kwargs):
    kwargs.setdefault("disp", False)
    return _scipy_opt_fmin_orig(fn, x0, *args, **kwargs)
_scipy_opt.fmin = _fmin_silent
del _scipy_opt  # clean up namespace

"""
nmrmetaproc: NMR Metabolomics Spectral Processor
=================================================
A Python package for processing raw Bruker NMR FID files into
analysis-ready spectral matrices for metabolomics studies.

Author: Folorunsho Bright Omage
ORCID: https://orcid.org/0000-0002-9750-5034
Email: omagefolorunsho@gmail.com
License: MIT
"""

from nmrmetaproc.version import __version__, __author__, __email__, __orcid__
from nmrmetaproc.processor import NMRProcessor, ProcessingResults

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__orcid__",
    "NMRProcessor",
    "ProcessingResults",
]
