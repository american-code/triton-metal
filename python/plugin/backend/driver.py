"""Triton 3.7.1 `DriverBase` adapter for the Metal backend.

Vendored signatures from the pinned release (docs/ARCHITECTURE.md §Compatibility).
Every method is a call into `libtritonmetal.dylib`: device queries, library and
pipeline creation, allocation and dispatch all happen in Swift. What is left here
is argument marshalling — turning the Python objects Triton hands the launcher
into the two parallel arrays `tm_launch` documents — and nothing else.

Note we subclass `DriverBase`, not `GPUDriver`: `GPUDriver.__init__` reaches
straight into `torch.cuda`.
"""

import time

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

from triton_metal import _core

#: Scalar element types `tm_launch` can carry (see its table in
#: docs/ARCHITECTURE.md §C ABI reference). Anything wider is refused rather than
#: silently truncated.
_INT_DTYPES = {"i1", "i8", "i16", "i32"}
_INT64_DTYPES = {"i64"}
_FLOAT_DTYPES = {"f32"}


class MetalUtils:
    """The `driver.active.utils` surface Triton's `CompiledKernel` calls."""

    def load_binary(self, name, kernel, shared, device):
        """MSL source bytes -> (library handle, pipeline handle, regs, spills, max threads).

        Metal compiles from source in-process, so this is where the MSL text
        becomes a compute pipeline. Register and spill counts have no Metal
        equivalent that is readable without Xcode's tooling, so they are 0.
        """
        library = _core.compile_msl(kernel.decode("utf-8"))
        pipeline = _core.load_kernel(library, name)
        return library, pipeline, 0, 0, _core.kernel_max_threads(pipeline)

    def get_device_properties(self, device):
        # Threadgroup memory is declared statically in the emitted MSL; the value
        # is only used by Triton to reject over-large dynamic allocations.
        return {"max_shared_mem": 32 * 1024}


class MetalLauncher:
    """Marshals one launch. Created once per compiled kernel."""

    def __init__(self, src, metadata):
        signature = getattr(src, "signature", {})
        # `bound_args.values()` includes constexpr parameters; the emitted
        # `tt.func` does not. This maps Metal buffer index -> position in the
        # Python call.
        self.positions = [i for i, ty in enumerate(signature.values()) if ty != "constexpr"]

    def __call__(self, grid_0, grid_1, grid_2, stream, function, packed_metadata, launch_metadata, enter_hook,
                 exit_hook, *args):
        num_warps, threads, metal_args = packed_metadata
        if enter_hook is not None:
            enter_hook(launch_metadata)
        kinds = []
        values = []
        for index, kind, dtype in metal_args:
            arg = args[self.positions[index]]
            if kind == "pointer":
                handle = getattr(arg, "tm_handle", None)
                if handle is None:
                    raise TypeError(
                        f"argument {index} ({type(arg).__name__}) is not backed by Metal memory; "
                        "allocate it with triton_metal.buffer.MetalBuffer (see python/examples)")
                kinds.append(_core.ARG_BUFFER)
                values.append(handle)
            elif dtype in _FLOAT_DTYPES:
                kinds.append(_core.ARG_F32)
                values.append(_core.float_bits(float(arg)))
            elif dtype in _INT_DTYPES:
                kinds.append(_core.ARG_I32)
                values.append(int(arg))
            elif dtype in _INT64_DTYPES:
                kinds.append(_core.ARG_I64)
                values.append(int(arg))
            else:
                # f64 lands here and stays here: Metal has no `double`, so the
                # Swift emitter refuses the kernel rather than narrowing it.
                raise TypeError(
                    f"argument {index} has scalar type {dtype!r}; tm_launch carries 32-bit "
                    "scalars and i64 (docs/ARCHITECTURE.md §C ABI reference)")
        _core.launch(function, (grid_0, grid_1, grid_2), threads, kinds, values)
        if exit_hook is not None:
            exit_hook(launch_metadata)


def _do_bench(kernel_call, quantiles=None, warmup=25, rep=100, **kwargs):
    """`tm_launch` submits and waits, so wall clock around the call is GPU time
    plus one command-buffer round trip. Reported as-is; no CUDA-event analogue."""
    for _ in range(3):
        kernel_call()
    n = max(1, int(rep))
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        kernel_call()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    if quantiles is None:
        return sum(samples) / len(samples)
    return [samples[min(len(samples) - 1, int(q * len(samples)))] for q in quantiles]


class MetalDriver(DriverBase):

    def __init__(self):
        self.utils = MetalUtils()
        self.launcher_cls = MetalLauncher
        super().__init__()

    @classmethod
    def is_active(cls):
        return bool(_core.is_active())

    def get_current_target(self):
        return GPUTarget("metal", _core.device_name(), 32)

    def get_current_device(self):
        return 0

    def set_current_device(self, device):
        assert device == 0, "one Metal device is supported"

    def get_current_stream(self, device=None):
        # Dispatch is synchronous in the Swift core; there is no stream handle.
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def get_device_interface(self):
        return self

    def get_benchmarker(self):
        return _do_bench

    def map_python_to_cpp_type(self, ty: str) -> str:
        return {
            "i1": "int8_t", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
            "u1": "uint8_t", "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
            "fp16": "float", "bf16": "float", "fp32": "float", "f32": "float", "fp64": "double",
        }[ty] if not ty.startswith("*") else "void*"

    def clear_cache(self, cache):
        cache.zero_()

    def empty_cache(self):
        pass


__all__ = ["MetalDriver", "MetalLauncher", "MetalUtils"]
