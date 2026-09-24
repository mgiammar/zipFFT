"""zipFFT: Fast FFT operations using CUDA and cuFFTDx."""

import warnings
from typing import List


# # Try to import each extension
# try:
#     from . import padded_rconv2d
# except ImportError:
#     padded_rconv2d = None

from . import padded_rconv2d
from . import padded_cconv2d

# Load build config if available. VERSION/CUDA_ARCHITECTURES/ENABLED_EXTENSIONS are
# generated at build time from pyproject.toml's [project] version (the single source
# of truth) by setup.py.
try:
    from .build_config import CUDA_ARCHITECTURES, ENABLED_EXTENSIONS, VERSION
except ImportError:
    CUDA_ARCHITECTURES = []
    ENABLED_EXTENSIONS = []
    VERSION = "0.0.0+unknown"

__version__ = VERSION


def is_extension_available(extension_name: str) -> bool:
    """Check if a specific extension is available."""
    return extension_name in ENABLED_EXTENSIONS


def get_available_extensions() -> List[str]:
    """Get list of available extensions."""
    return list(ENABLED_EXTENSIONS)


def get_cuda_architectures() -> List[str]:
    """Get list of CUDA architectures this package was compiled for."""
    return CUDA_ARCHITECTURES.copy()


__all__ = [
    "is_extension_available",
    "get_available_extensions",
    "get_cuda_architectures",
    "padded_rconv2d",
    "padded_cconv2d",
]
