"""zipFFT: Fast FFT operations using CUDA and cuFFTDx."""

import warnings
from typing import List

__version__ = "0.0.2alpha"

# Simple availability checking - just try to import each module
_available_modules = {}

# Try to import each extension
try:
    from . import cfft1d

    _available_modules["cfft1d"] = cfft1d
except ImportError:
    cfft1d = None

try:
    from . import rfft1d

    _available_modules["rfft1d"] = rfft1d
except ImportError:
    rfft1d = None

try:
    from . import padded_rfft1d

    _available_modules["padded_rfft1d"] = padded_rfft1d
except ImportError:
    padded_rfft1d = None

try:
    from . import strided_cfft1d

    _available_modules["strided_cfft1d"] = strided_cfft1d
except ImportError:
    strided_cfft1d = None

# Load build config if available
try:
    from .build_config import CUDA_ARCHITECTURES, ENABLED_EXTENSIONS
except ImportError:
    CUDA_ARCHITECTURES = []
    ENABLED_EXTENSIONS = []


def is_extension_available(extension_name: str) -> bool:
    """Check if a specific extension is available."""
    return extension_name in _available_modules


def get_available_extensions() -> List[str]:
    """Get list of available extensions."""
    return list(_available_modules.keys())


def get_cuda_architectures() -> List[str]:
    """Get list of CUDA architectures this package was compiled for."""
    return CUDA_ARCHITECTURES.copy()


__all__ = [
    "is_extension_available",
    "get_available_extensions",
    "get_cuda_architectures",
    "cfft1d",
    "rfft1d",
    "padded_rfft1d",
    "strided_cfft1d",
]
