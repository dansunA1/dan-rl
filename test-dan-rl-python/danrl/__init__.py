"""Test dan-rl-python package."""

__version__ = "0.1.0"

# Import the Rust extension module
try:
    from . import _test_dan_rl_python
    __all__ = ["_test_dan_rl_python"]
except ImportError as e:
    # Module not built yet - run 'maturin develop' to build
    raise ImportError(f"Module not built yet - run 'maturin develop' to build: {e}")

