"""Compiler shim: stage functions delegate straight to the Swift core."""

from . import _core


class MetalOptions:
    def __init__(self, num_simdgroups: int = 4):
        self.num_simdgroups = num_simdgroups


class MetalBackend:
    """Triton-facing backend class. ttir/ttgir stages reuse Triton's own MLIR
    passes (they run in Triton's compiled pass pipeline, not in Python); the
    Metal-specific stages are one-line calls into libtritonmetal."""

    binary_ext = "metallib"

    @staticmethod
    def supports_target(target) -> bool:
        return getattr(target, "backend", None) == "metal"

    def add_stages(self, stages: dict, options: MetalOptions) -> None:
        stages["ttir"] = self._make_ttir
        stages["ttgir"] = self._make_ttgir
        stages["msl"] = lambda mod, metadata: _core.emit_msl(mod, options.num_simdgroups)
        stages["metallib"] = self._make_metallib

    def _make_ttir(self, mod, metadata):
        raise NotImplementedError("reuse Triton's canonicalization passes (Triton-side)")

    def _make_ttgir(self, mod, metadata):
        raise NotImplementedError("Triton GPU passes with Metal target profile (Triton-side)")

    def _make_metallib(self, src, metadata):
        # Runtime compilation in the Swift core; returns an opaque library handle.
        # Serialized .metallib caching waits on pinning a Triton release, which
        # decides what `binary_ext` payloads the cache expects.
        return _core.compile_msl(src)
