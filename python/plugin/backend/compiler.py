"""Triton 3.7.1 `BaseBackend` adapter for the Metal backend.

This file exists because `triton.backends` imports `triton.backends.<name>.compiler`
and instantiates the one concrete `BaseBackend` subclass it finds there. Its
signatures are vendored from the pinned release (see docs/ARCHITECTURE.md
§Compatibility); it contains no compilation logic of its own.

Division of labour:

* the `ttir` stage is *Triton's own* MLIR pass pipeline, run in Triton's compiled
  pass manager — nothing here parses or rewrites IR;
* the `msl` stage hands the resulting IR **text** to `tm_emit_msl` and the launch
  metadata question to `tm_kernel_info`, both of which live in the Swift core.

If the release's IR ever spells something our emitter does not accept, the fix
goes in the Swift parser, never here.
"""

import functools
import hashlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget, Language

from triton_metal import _core


@dataclass(frozen=True)
class MetalOptions:
    """Metal's slice of Triton's option space.

    `num_warps` keeps Triton's name; on Metal a warp is a simdgroup of 32 lanes,
    so `num_warps` is the simdgroup count the Swift emitter is asked for.
    """

    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 32
    debug: bool = False
    sanitize_overflow: bool = True
    arch: str = "metal"
    backend_name: str = "metal"
    extern_libs: Optional[dict] = None
    ir_override: Optional[str] = None
    # Metal has one precision per element type: no tf32 analogue to choose.
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee", )
    # `mathMode = .safe`: the emitter never contracts or reassociates (§Floating-point policy).
    enable_fp_fusion: bool = False
    max_num_imprecise_acc_default: int = 0
    # No fp8 in the emitter's type table (§Types).
    supported_fp8_dtypes: Tuple[str, ...] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    launch_cooperative_grid: bool = False
    instrumentation_mode: str = ""

    def __post_init__(self):
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

    def hash(self):
        key = "_".join(f"{name}-{val}" for name, val in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MetalBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "metal"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # The payload the launcher receives is MSL source: Metal's own compiler
        # runs inside the process (`MTLDevice.makeLibrary(source:)`), so there is
        # no offline object file to cache unless `xcrun metal` is installed —
        # which Command Line Tools alone do not provide.
        self.binary_ext = "msl"

    def get_target_name(self, options) -> str:
        return f"metal:{self.target.arch}"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in MetalOptions.__dataclass_fields__ if k in opts and opts[k] is not None}
        args["arch"] = str(self.target.arch)
        return MetalOptions(**args)

    def pack_metadata(self, metadata):
        # Handed verbatim to the launcher as `packed_metadata`.
        return (
            metadata.num_warps,
            metadata.threads_per_threadgroup,
            metadata.metal_args,
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # Nothing to add: the Metal-specific half of the pipeline is not MLIR.
        pass

    @staticmethod
    def make_ttir(mod, metadata, options):
        """Triton's own canonicalization pipeline, unchanged.

        Deliberately the same passes the in-tree backends run, minus the ones
        that only make sense for a target with tensor descriptors.
        """
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_msl(mod, metadata, options):
        """Swift core: IR text in, MSL source and launch metadata out."""
        import json

        ttir = str(mod)
        info = json.loads(_core.kernel_info(ttir, options.num_warps))
        kernel = info["kernels"][0]
        metadata["name"] = kernel["name"]
        metadata["threads_per_threadgroup"] = kernel["threads_per_threadgroup"]
        metadata["block_shape"] = kernel["block_shape"]
        metadata["metal_args"] = [(a["index"], a["kind"], a["dtype"]) for a in kernel["args"]]
        # Threadgroup memory is declared statically inside the emitted MSL, so no
        # dynamically sized allocation is requested at dispatch time.
        metadata["shared"] = 0
        return _core.emit_msl(ttir, options.num_warps).encode("utf-8")

    def add_stages(self, stages, options, language=Language.TRITON):
        if language == Language.GLUON:
            raise NotImplementedError("triton-metal does not implement the Gluon frontend")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return f"{_core.version()}-{self.target.arch}"
