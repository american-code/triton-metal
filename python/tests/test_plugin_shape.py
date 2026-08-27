"""Smoke tests: shim shape + the ctypes bridge into the Swift core.

Requires `swift build` to have been run at the repo root first.

Note the division of labour: these tests move bytes and compare numbers, but the
shim itself only marshals arguments — every compile, launch and allocation below
happens inside libtritonmetal.dylib.
"""

import array
import ctypes
import json
import math
import platform
from pathlib import Path

import pytest

from triton_metal import _core
from triton_metal.buffer import MetalBuffer

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugin"

VECTOR_ADD_TTIR = """
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""

SOFTMAX_TTIR = """
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                 %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<128xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.load %5, %7, %cst : tensor<128xf32>
    %9 = "tt.reduce"(%8) <{axis = 0 : i32}> ({
    ^bb0(%arg5: f32, %arg6: f32):
      %30 = arith.maxnumf %arg5, %arg6 : f32
      tt.reduce.return %30 : f32
    }) : (tensor<128xf32>) -> f32
    %10 = tt.splat %9 : f32 -> tensor<128xf32>
    %11 = arith.subf %8, %10 : tensor<128xf32>
    %12 = math.exp %11 : tensor<128xf32>
    %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      %31 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %31 : f32
    }) : (tensor<128xf32>) -> f32
    %14 = tt.splat %13 : f32 -> tensor<128xf32>
    %15 = arith.divf %12, %14 : tensor<128xf32>
    %16 = arith.muli %0, %arg3 : i32
    %17 = tt.addptr %arg1, %16 : !tt.ptr<f32>, i32
    %18 = tt.splat %17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %19 = tt.addptr %18, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %19, %15, %7 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
"""

requires_metal = pytest.mark.skipif(
    not _core.is_usable(),
    reason="no usable Metal device: " + (_core.unusable_reason() or "unknown"),
)


# --- Shim shape -------------------------------------------------------------


def test_plugin_directory_is_what_triton_discovers():
    """`TRITON_PLUGIN_DIRS` points at `python/plugin`; Triton's setup.py asserts
    `backend/compiler.py` and `backend/driver.py` exist and reads the backend's
    name out of `backend/name.conf` (Triton 3.7.1, setup.py:BackendInstaller)."""
    assert (PLUGIN_DIR / "backend" / "name.conf").read_text().strip() == "metal"
    assert (PLUGIN_DIR / "backend" / "compiler.py").exists()
    assert (PLUGIN_DIR / "backend" / "driver.py").exists()
    # Triton's main.cc expands the backend name into `init_triton_metal`.
    assert "void init_triton_metal(" in (PLUGIN_DIR / "metal.cc").read_text()


def test_core_version_round_trips_through_c_abi():
    assert _core.version() == "0.0.1"


def test_core_is_active_on_apple_silicon_mac():
    expected = platform.system() == "Darwin" and platform.machine() == "arm64"
    assert bool(_core.is_active()) == expected


# --- Compiler bridge --------------------------------------------------------


def test_emit_msl_surfaces_swift_core_error():
    with pytest.raises(RuntimeError, match="no tt.func"):
        _core.emit_msl("module {}")


def test_unsupported_op_error_names_the_op():
    ir = """
    module {
      tt.func public @k(%arg0: !tt.ptr<f32>) {
        %0 = tt.histogram %arg0 : !tt.ptr<f32>
        tt.return
      }
    }
    """
    with pytest.raises(RuntimeError, match=r"unsupported op 'tt.histogram'"):
        _core.emit_msl(ir)


def test_emit_msl_lowers_vector_add():
    source = _core.emit_msl(VECTOR_ADD_TTIR, 4)
    assert "kernel void add_kernel(" in source
    assert "device float *varg0 [[buffer(0)]]" in source


def test_kernel_info_is_computed_in_the_core():
    info = json.loads(_core.kernel_info(VECTOR_ADD_TTIR, 4))
    kernel = info["kernels"][0]
    assert kernel["name"] == "add_kernel"
    assert kernel["block_size"] == 1024
    assert kernel["block_shape"] == [1024]
    assert kernel["threads_per_threadgroup"] == 128
    assert [a["kind"] for a in kernel["args"]] == ["pointer", "pointer", "pointer", "scalar"]


def test_reductions_and_math_lower_through_the_shim():
    source = _core.emit_msl(SOFTMAX_TTIR, 4)
    assert "simd_shuffle_down" in source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in source
    assert "precise::exp(" in source


# --- End-to-end through the C ABI -------------------------------------------


def _upload(values):
    """Allocate a unified-memory buffer in the core and fill it from a host array."""
    handle = _core.alloc_buffer(values.itemsize * len(values))
    address, count = values.buffer_info()
    _core.buffer_write(handle, 0, address, count * values.itemsize)
    return handle


def _download(handle, count):
    out = array.array("f", [0.0]) * count
    address, _ = out.buffer_info()
    _core.buffer_read(handle, 0, address, count * out.itemsize)
    return out


@requires_metal
def test_vector_add_round_trips_through_the_c_abi():
    n = 5000
    a = array.array("f", [i * 0.5 for i in range(n)])
    b = array.array("f", [float(i % 13) for i in range(n)])
    expected = [x + y for x, y in zip(a, b)]

    live_before = _core.live_handle_count()
    kernel_meta = json.loads(_core.kernel_info(VECTOR_ADD_TTIR, 4))["kernels"][0]
    library = _core.compile_msl(_core.emit_msl(VECTOR_ADD_TTIR, 4))
    kernel = _core.load_kernel(library, kernel_meta["name"])
    assert _core.kernel_max_threads(kernel) >= kernel_meta["threads_per_threadgroup"]

    buffer_a = _upload(a)
    buffer_b = _upload(b)
    buffer_out = _core.alloc_buffer(4 * n)
    assert _core.buffer_length(buffer_out) == 4 * n

    block = kernel_meta["block_size"]
    _core.launch(
        kernel,
        ((n + block - 1) // block, 1, 1),
        kernel_meta["threads_per_threadgroup"],
        [_core.ARG_BUFFER, _core.ARG_BUFFER, _core.ARG_BUFFER, _core.ARG_I32],
        [buffer_a, buffer_b, buffer_out, n],
    )

    assert list(_download(buffer_out, n)) == expected

    for handle in (buffer_a, buffer_b, buffer_out):
        _core.free_buffer(handle)
    _core.release_kernel(kernel)
    _core.release_library(library)
    assert _core.live_handle_count() == live_before


@requires_metal
def test_softmax_round_trips_through_the_c_abi():
    """The fused-softmax kernel — reductions, math.exp, masked load/store — end to
    end over `tm_*`. The shim still computes nothing: the block size, threadgroup
    size and MSL all come out of the Swift core."""
    rows, n_cols, block = 12, 100, 128
    stride = n_cols
    data = array.array("f", [((i * 17) % 53) * 0.4 - 10 for i in range(rows * stride)])

    live_before = _core.live_handle_count()
    meta = json.loads(_core.kernel_info(SOFTMAX_TTIR, 4))["kernels"][0]
    assert meta["block_size"] == block
    library = _core.compile_msl(_core.emit_msl(SOFTMAX_TTIR, 4))
    kernel = _core.load_kernel(library, meta["name"])

    buffer_in = _upload(data)
    buffer_out = _core.alloc_buffer(4 * rows * stride)
    _core.launch(
        kernel,
        (rows, 1, 1),
        meta["threads_per_threadgroup"],
        [_core.ARG_BUFFER, _core.ARG_BUFFER, _core.ARG_I32, _core.ARG_I32, _core.ARG_I32],
        [buffer_in, buffer_out, stride, stride, n_cols],
    )

    out = _download(buffer_out, rows * stride)
    for row in range(rows):
        values = data[row * stride : row * stride + n_cols]
        biggest = max(values)
        exponentials = [math.exp(v - biggest) for v in values]
        total = sum(exponentials)
        for column in range(n_cols):
            assert out[row * stride + column] == pytest.approx(
                exponentials[column] / total, rel=1e-5, abs=1e-7
            )

    for handle in (buffer_in, buffer_out):
        _core.free_buffer(handle)
    _core.release_kernel(kernel)
    _core.release_library(library)
    assert _core.live_handle_count() == live_before


@requires_metal
def test_buffer_contents_exposes_unified_memory():
    handle = _core.alloc_buffer(16)
    try:
        pointer = ctypes.cast(_core.buffer_contents(handle), ctypes.POINTER(ctypes.c_float))
        pointer[0] = 1.25
        out = _download(handle, 4)
        assert out[0] == pytest.approx(1.25)
    finally:
        _core.free_buffer(handle)


@requires_metal
def test_load_binary_path_delegates_to_the_core():
    """What `MetalUtils.load_binary` does when Triton hands it the compiled
    payload: MSL source bytes -> library -> compute pipeline."""
    library = _core.compile_msl(_core.emit_msl(VECTOR_ADD_TTIR, 4))
    kernel = _core.load_kernel(library, "add_kernel")
    try:
        assert kernel != 0
        assert _core.kernel_max_threads(kernel) >= 32
    finally:
        _core.release_kernel(kernel)
        _core.release_library(library)


@requires_metal
def test_metal_buffer_exposes_data_ptr_for_triton():
    """Triton decides an argument is a pointer by looking for `data_ptr`
    (`triton/python/src/specialize.cc`), and specializes on 16-byte alignment."""
    live_before = _core.live_handle_count()
    buffer = MetalBuffer((4, 8), "float32")
    try:
        assert buffer.nbytes == 4 * 8 * 4
        assert buffer.data_ptr() % 16 == 0
        assert str(buffer.dtype) == "float32"
        source = array.array("f", [1.5] * 32)
        buffer.copy_from(source.buffer_info()[0])
        assert buffer.tolist()[:3] == [1.5, 1.5, 1.5]
    finally:
        buffer.free()
    assert _core.live_handle_count() == live_before


def test_metal_buffer_rejects_a_dtype_the_emitter_has_no_type_for():
    with pytest.raises(TypeError, match="unsupported dtype"):
        MetalBuffer(4, "float64")


def test_invalid_handles_raise_with_the_core_message():
    with pytest.raises(RuntimeError, match="is not live"):
        _core.load_kernel(999_999, "add_kernel")
    with pytest.raises(RuntimeError, match="is not live"):
        _core.buffer_length(999_999)
