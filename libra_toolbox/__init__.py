try:
    # Python 3.8+
    from importlib import metadata
except ImportError:
    try:
        # For older Python versions
        import importlib_metadata as metadata
    except ImportError:
        # If both imports fail, set version to unknown
        __version__ = "unknown"

try:
    # Attempt to get the version of the package
    __version__ = metadata.version("libra_toolbox")
except Exception:
    # If it fails, set version to unknown
    __version__ = "unknown"

from . import tritium
from . import neutron_detection
from . import neutronics
