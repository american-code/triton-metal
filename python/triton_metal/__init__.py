"""triton-metal: Triton backend plugin for Apple Metal.

LANGUAGE POLICY: this package is a thin ctypes shim, kept only because Triton's
backend discovery imports a Python module. All real work (MSL emission, metallib
compilation, device queries, kernel launch) lives in the Swift core
(Sources/TritonMetalCore), reached through the `tm_*` C ABI in libtritonmetal.dylib.
Do not add logic here.
"""

__version__ = "0.0.1"
