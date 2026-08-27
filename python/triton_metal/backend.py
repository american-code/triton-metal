"""Plugin entry point: what Triton's backend discovery imports."""

from .compiler import MetalBackend
from .driver import MetalDriver

compiler = MetalBackend
driver = MetalDriver
