# Using triton-metal

A practical guide to driving the backend: from Swift, from C/ctypes, and from the
Python plugin shim. Every example here was executed against the Swift core on an
Apple M1 Pro before it was written down; the outputs quoted are the real ones.

For *why* the backend is built this way, see [ARCHITECTURE.md](ARCHITECTURE.md)
and [WHITEPAPER.md](WHITEPAPER.md).

**Contents**

1. [Building](#building) — including
   [Triton with the Metal backend](#building-triton-with-the-metal-backend)
2. [The Swift API](#the-swift-api)
3. [The emitted MSL](#the-emitted-msl)
4. [A second kernel: softmax](#a-second-kernel-softmax)
5. [The C ABI](#the-c-abi)
6. [The Python shim](#the-python-shim)
7. [Supported IR subset](#supported-ir-subset)
8. [Error behaviour](#error-behaviour)
9. [Implementing what's missing](#implementing-whats-missing)

---

## Building

```
swift build                                     # builds .build/debug/libtritonmetal.dylib
swift test                                      # 144 cases, on the real GPU where relevant
cd python && PYTHONPATH=. python3 -m pytest tests/ -q   # 15 cases; needs the dylib above
```

`swift build -c release` puts the dylib in `.build/release/`. **Never
`xcodebuild`** — the offline Metal path shells out to `xcrun metal` directly.

Requirements: macOS 14+, an Apple-silicon Metal device. Nothing else — the
primary compile path is `MTLDevice.makeLibrary(source:)`, which needs no Xcode
command-line tools.

### Building Triton with the Metal backend

Everything above works without Triton. To run actual `@triton.jit` kernels you
need Triton itself, and **there is no macOS wheel** — `pip install triton` fails
on macOS arm64. Triton has to be built from source, with this repository handed to
it as an out-of-tree plugin. It is less painful than it sounds: the prebuilt
`macos-arm64` LLVM that Triton's CI publishes is downloaded automatically, and the
whole build is **about 9 minutes** on an M1 Max with 10 cores (allow ~25 on a
4-performance-core laptop). No Xcode, no CUDA toolchain, no patch to Triton.

```
# 1. This repo's dylib and shim.
swift build -c release
pip install -e python/                       # the ctypes shim, `triton_metal`

# 2. Triton, pinned, with the plugin.
git clone https://github.com/triton-lang/triton.git
cd triton && git checkout v3.7.1
pip install cmake ninja wheel setuptools pybind11

export TRITON_PLUGIN_DIRS=/abs/path/to/triton-metal/python/plugin
export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF"
# Skip the CUDA redistributables: setup.py maps Darwin -> "linux" downloads, and
# download_and_copy() short-circuits when the destination variable is already set.
for v in PTXAS CUOBJDUMP NVDISASM CUDACRT CUDART; do export TRITON_${v}_PATH=/nonexistent; done
export TRITON_CUPTI_INCLUDE_PATH=/nonexistent TRITON_CUPTI_LIB_PATH=/nonexistent

pip install -e .                             # or `python setup.py bdist_wheel` for a wheel
```

Python 3.10+ is required (3.11 is what this was built and tested on; macOS's
system 3.9 is too old). Then:

```
cd /path/to/triton-metal
python python/examples/vector_add.py
python python/examples/fused_softmax.py
python python/examples/matmul.py
PYTHONPATH=python python -m pytest python/tests -q    # 21 cases once Triton is present
```

Two caveats worth knowing before you start:

* The NVIDIA and AMD in-tree backends **are** built, and cannot be turned off:
  3.7.1's core sources include their tablegen'd headers. Only their runtime
  downloads are skipped. This costs build time, nothing else — neither driver is
  active on a Mac, so `triton.runtime.driver.active` resolves to Metal.
* `python setup.py bdist_wheel` yields
  `triton-3.7.1+gitf797708c-cp311-cp311-macosx_26_0_arm64.whl` (69.4 MB) with
  `metal = triton.backends.metal` in its entry points. It is a normal wheel: it
  can be installed on another Mac with the same Python minor version, but it does
  **not** contain `libtritonmetal.dylib` — the shim finds that through
  `TRITON_METAL_CORE_LIB` or a repo-relative `swift build` output.

---

## The Swift API

Four types carry the whole flow:

| Type | Role |
| --- | --- |
| `MetalCompiler` | `emit` / `emitMSL` (IR text → MSL), `compileMSL` (→ `MTLLibrary` or `MTLComputePipelineState`), `compileMSLOffline` (→ `.metallib` bytes) |
| `EmissionResult` | `.source` (the MSL) and `.kernels` |
| `EmittedKernel` | `.name`, `.blockShape`, `.blockSize`, `.threadsPerThreadgroup`, `.arguments` — everything a launcher needs |
| `MetalRuntime` | `makeBuffer`, `loadKernel`, `launch` |

The core never chooses a grid for you, and it never asks you to compute a
threadgroup size. Block shape and threadgroup size are properties of the lowered
kernel, so they come *out* of `emit`.

### A complete, runnable vector-add

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "usagecheck",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../triton-metal")],
    targets: [
        .executableTarget(
            name: "usagecheck",
            dependencies: [.product(name: "tritonmetal", package: "triton-metal")]
        )
    ]
)
```

```swift
import Foundation
import Metal
import TritonMetalCore

// Triton's vector-add tutorial kernel, as the `ttir` text Triton prints.
let vectorAddIR = """
    module {
      tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                 %arg2: !tt.ptr<f32>, %arg3: i32) {
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

// Upload a host array into a unified-memory buffer.
func upload(_ values: [Float]) throws -> MTLBuffer {
    let buffer = try MetalRuntime.makeBuffer(
        length: values.count * MemoryLayout<Float>.stride)
    values.withUnsafeBytes {
        buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
    }
    return buffer
}

func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(
        UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

// 1. Lower IR text to MSL source plus per-kernel launch metadata.
let emission = try MetalCompiler.emit(ttir: vectorAddIR, options: .init(numSimdgroups: 4))
let addKernel = emission.kernels[0]
print(emission.source)

// 2. Compile the MSL and resolve the kernel's compute pipeline.
let addPipeline = try MetalCompiler.compileMSL(emission.source, kernelName: addKernel.name)

// 3. Allocate unified-memory buffers and fill the inputs.
let n = 5000
let a = (0..<n).map { Float($0) * 0.5 }
let b = (0..<n).map { Float($0 % 13) }
let bufferA = try upload(a)
let bufferB = try upload(b)
let bufferOut = try MetalRuntime.makeBuffer(length: n * MemoryLayout<Float>.stride)

// 4. One threadgroup per Triton program; the core reports the threadgroup size.
try MetalRuntime.launch(
    pipeline: addPipeline,
    threadgroups: MTLSize(
        width: (n + addKernel.blockSize - 1) / addKernel.blockSize, height: 1, depth: 1),
    threadsPerThreadgroup: addKernel.threadsPerThreadgroup,
    arguments: [
        .buffer(bufferA), .buffer(bufferB), .buffer(bufferOut), .int32(Int32(n)),
    ])

// 5. Read results straight out of unified memory — no copy back.
let out = read(bufferOut, count: n)
print("out[0]=\(out[0]) out[1]=\(out[1]) out[4999]=\(out[4999])")
```

`swift run` prints the MSL (below), then:

```
name=add_kernel blockShape=[1024] blockSize=1024 threads=128
out[0]=0.0 out[1]=1.5 out[4999]=2506.5
vector add worst absolute error: 0.0
```

Notes on the five steps:

* **`numSimdgroups` is Triton's `num_warps`.** A Metal simdgroup is 32 lanes, so
  `num_warps=4` asks for 128 threads. The emitter clamps that to the innermost
  block dimension and rounds *up* to a whole simdgroup, so what you get back in
  `threadsPerThreadgroup` may not be `numSimdgroups * 32`. Use the reported value.
* **The grid is yours.** `blockSize` is the number of elements one Triton program
  handles, so `cdiv(n, blockSize)` threadgroups is the vector-add convention —
  exactly what `triton.cdiv` computes in the tutorial's `grid` lambda.
* **Argument order is buffer index.** Argument `i` of `tt.func` binds at
  `[[buffer(i)]]`. `kernel.arguments` tells you which are pointers and which are
  scalars, and the Triton spelling of each element type.
* **Elementwise kernels are threadgroup-size-agnostic** (tested from
  `num_warps=1` to `32`). Kernels containing a `tt.reduce` or a `tt.dot` are
  **not**: `simd_shuffle_down` reads lanes that must be live, so those must be
  launched at the reported `threadsPerThreadgroup`.
* **Unified memory means no copy back.** `.storageModeShared` buffers are visible
  to CPU and GPU alike; `launch` blocks until the command buffer completes, so
  the results are readable the moment it returns.

### Offline `.metallib`

`compileMSL` is the primary path. When you want cacheable bytes instead:

```swift
let metallib = try MetalCompiler.compileMSLOffline(emission.source)   // 5028 bytes
let offlinePipeline = try MetalRuntime.loadKernel(metallib: metallib, kernelName: "add_kernel")
```

This shells out to `xcrun metal -std=metal3.1` (pass `standard:` to change it)
and throws `CoreError.metal` carrying the front end's own diagnostics if the
toolchain is missing or the source is rejected.

---

## The emitted MSL

Textual MSL is emitted rather than AIR precisely so that this is readable. Here
is `emission.source` for the kernel above, verbatim:

```metal
// Generated by triton-metal (TritonMetalCore). Do not edit.
//
// Execution model: one threadgroup per Triton program. A kernel's tensors
// share one block index space; the innermost block dimension is strided
// across the threadgroup's threads, outer dimensions are looped uniformly.
#include <metal_stdlib>
using namespace metal;

// kernel add_kernel: BLOCK=1024, threadgroup=128 threads
kernel void add_kernel(
    device float *varg0 [[buffer(0)]],
    device float *varg1 [[buffer(1)]],
    device float *varg2 [[buffer(2)]],
    constant int &varg3 [[buffer(3)]],
    uint3 tm_program_id [[threadgroup_position_in_grid]],
    uint3 tm_thread_id [[thread_position_in_threadgroup]],
    uint3 tm_threadgroup_size [[threads_per_threadgroup]]
) {
    int vc1024_i32 = 1024;
    int v0 = int(tm_program_id.x);
    int v1 = v0 * vc1024_i32;
    for (uint tm_i0 = tm_thread_id.x; tm_i0 < 1024u; tm_i0 += tm_threadgroup_size.x) {
        int v2 = int(tm_i0);
        int v3 = v1;
        int v4 = v3 + v2;
        int v5 = varg3;
        bool v6 = v4 < v5;
        device float *v7 = varg0;
        device float *v8 = v7 + v4;
        float v9 = v6 ? *v8 : 0.0f;
        device float *v10 = varg1;
        device float *v11 = v10 + v4;
        float v12 = v6 ? *v11 : 0.0f;
        float v13 = v9 + v12;
        device float *v14 = varg2;
        device float *v15 = v14 + v4;
        if (v6) { *v15 = v13; }
    }
}
```

Things worth reading out of it, because they are the whole execution model:

* Every SSA name survives. `%13 = arith.addf %9, %12` is `float v13 = v9 + v12;`,
  so a diff between the IR and the MSL is a line-for-line diff.
* `%0 = tt.get_program_id x` and `%1 = arith.muli %0, %c1024_i32` are emitted
  **before** the loop. They do not vary along the block, so they are computed
  once per program rather than once per element. That placement is inferred, not
  annotated — see [ARCHITECTURE.md §Execution model](ARCHITECTURE.md#execution-model).
* The one loop is the distributed one: `tm_i0` starts at this thread's id and
  strides by the threadgroup size, so the 1024-wide block is covered by 128
  threads in 8 passes.
* The mask becomes a ternary on load (`v6 ? *v8 : 0.0f`) and a guarded store
  (`if (v6) { *v15 = v13; }`). `tt.load`'s optional `other` operand replaces the
  `0.0f`.
* `tt.splat` emits an assignment and nothing else (`int v3 = v1;`); `tt.expand_dims`
  and `tt.broadcast` emit *no code at all* — they are relabellings in the layout,
  which is why a rank-2 kernel's row arithmetic ends up hoisted out of the inner
  loop rather than repeated per column.

---

## A second kernel: softmax

The fused-softmax tutorial adds two `tt.reduce`s and a `math.exp`. Nothing about
the calling convention changes; only the launch geometry does — one program per
row instead of one per block of elements.

```swift
let softmax = try MetalCompiler.emit(ttir: softmaxIR, options: .init(numSimdgroups: 4))
let softmaxKernel = softmax.kernels[0]      // blockShape=[128], threads=128
let softmaxPipeline = try MetalCompiler.compileMSL(softmax.source, kernelName: softmaxKernel.name)

let rows = 12, columns = 100, stride = columns
let input = (0..<(rows * stride)).map { Float(($0 * 17) % 53) * 0.4 - 10 }
let bufferIn = try upload(input)
let bufferSoftmax = try MetalRuntime.makeBuffer(
    length: rows * stride * MemoryLayout<Float>.stride)

// A kernel containing a `tt.reduce` must be launched at the reported
// `threadsPerThreadgroup`: `simd_shuffle_down` reads lanes that must be live.
try MetalRuntime.launch(
    pipeline: softmaxPipeline,
    threadgroups: MTLSize(width: rows, height: 1, depth: 1),   // one program per row
    threadsPerThreadgroup: softmaxKernel.threadsPerThreadgroup,
    arguments: [
        .buffer(bufferIn), .buffer(bufferSoftmax),
        .int32(Int32(stride)), .int32(Int32(stride)), .int32(Int32(columns)),
    ])
```

with `softmaxIR` being Triton's kernel, printed the way MLIR prints an op that
carries a region — `tt.reduce` is the one op the parser accepts only in generic
form, because that is how Triton prints it:

```mlir
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                 %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<128xf32>   // -inf
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
      %20 = arith.maxnumf %arg5, %arg6 : f32
      tt.reduce.return %20 : f32
    }) : (tensor<128xf32>) -> f32
    %10 = tt.splat %9 : f32 -> tensor<128xf32>
    %11 = arith.subf %8, %10 : tensor<128xf32>
    %12 = math.exp %11 : tensor<128xf32>
    %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      %21 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %21 : f32
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
```

Measured against a CPU reference over 12 rows of 100 columns:

```
softmax: blockShape=[128] threads=128
softmax worst absolute error vs CPU: 7.4505806e-08
```

That is one float ULP at these magnitudes, and it is not an accident: fast math
is switched **off** (Metal defaults it on), and `math.exp` lowers to
`precise::exp`. `fast::` is never emitted.

Two consequences of a reduction show up in the generated source rather than the
API. First, the reduction *closes* the distributed loop and opens a fresh one, so
softmax emits three lane loops over the same row and **recomputes** `%8` in each
— cheaper than spilling `3 x BLOCK` floats to threadgroup memory, and legal
because every lowered op is pure. Second, a `-inf` seed spelled as MLIR's raw bit
pattern (`dense<0xFF800000>`) parses; you do not have to rewrite it.

---

## The C ABI

Everything above is reachable from any language that can call C. The exports
live in `libtritonmetal.dylib` and follow three conventions:

* `char *` results are **malloc'd**; the caller frees them with `tm_free`.
* Handle-returning calls (`int64`) return **0** on failure; `int32` calls return **−1**.
* On any failure the reason is in `tm_last_error()`, which **clears** it and
  returns a malloc'd string. It reads `"no error"` when nothing is pending.

Handles are opaque `int64` values from one process-wide table. Handle 0 is never
valid, which is what makes "0 means failure" unambiguous.

The full symbol table is in
[ARCHITECTURE.md §C ABI reference](ARCHITECTURE.md#c-abi-reference). The flow is:

```
tm_emit_msl    ─┐
tm_kernel_info ─┴─→ tm_compile_msl ─→ library handle ─→ tm_load_kernel ─→ kernel handle
                    (or tm_load_metallib for cached bytes)                     │
tm_alloc_buffer ─→ buffer handle ─→ tm_buffer_write ────────────────────────→ tm_launch
                                                       tm_buffer_read ←────────┘
                   tm_free_buffer / tm_release_kernel / tm_release_library
                   tm_live_handle_count   (leak check)
```

`tm_kernel_info` exists so that no caller ever computes launch geometry itself:

```json
{"kernels": [{"name": "add_kernel", "block_size": 1024, "block_shape": [1024],
              "threads_per_threadgroup": 128,
              "args": [{"index": 0, "kind": "pointer", "dtype": "f32"},
                       {"index": 1, "kind": "pointer", "dtype": "f32"},
                       {"index": 2, "kind": "pointer", "dtype": "f32"},
                       {"index": 3, "kind": "scalar",  "dtype": "i32"}]}]}
```

### A minimal ctypes walkthrough

Nothing here imports `triton_metal`; this is the raw ABI.

```python
import array, ctypes, json, sys

lib = ctypes.CDLL(sys.argv[1])   # .build/debug/libtritonmetal.dylib

i32, i64, ptr, cstr = ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p, ctypes.c_char_p
for name, restype, argtypes in [
    ("tm_free", None, [ptr]),
    ("tm_last_error", ptr, []),
    ("tm_emit_msl", ptr, [cstr, i32]),
    ("tm_kernel_info", ptr, [cstr, i32]),
    ("tm_compile_msl", i64, [cstr]),
    ("tm_load_kernel", i64, [i64, cstr]),
    ("tm_alloc_buffer", i64, [i64]),
    ("tm_buffer_write", i32, [i64, i64, ptr, i64]),
    ("tm_buffer_read", i32, [i64, i64, ptr, i64]),
    ("tm_launch", i32, [i64, i64, i64, i64, i64,
                        ctypes.POINTER(i32), ctypes.POINTER(i64), i32]),
    ("tm_free_buffer", i32, [i64]),
    ("tm_release_kernel", i32, [i64]),
    ("tm_release_library", i32, [i64]),
    ("tm_live_handle_count", i64, []),
]:
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = restype, argtypes


def take(p):
    """Every `char *` the core returns is malloc'd; the caller frees it."""
    if not p:
        return None
    try:
        return ctypes.cast(p, cstr).value.decode()
    finally:
        lib.tm_free(p)


def fail(what):
    raise RuntimeError("%s: %s" % (what, take(lib.tm_last_error())))


TTIR = open(sys.argv[2]).read()
live_before = lib.tm_live_handle_count()

# 1. Lower. NULL back means failure; the reason is in tm_last_error.
msl = take(lib.tm_emit_msl(TTIR.encode(), 4))
if msl is None:
    fail("tm_emit_msl")

# 2. Ask the core for the launch geometry — the caller never computes it.
info = take(lib.tm_kernel_info(TTIR.encode(), 4))
if info is None:
    fail("tm_kernel_info")
meta = json.loads(info)["kernels"][0]
print("kernel metadata:", json.dumps(meta))

# 3. Compile and load. Handle-returning calls return 0 on failure.
library = lib.tm_compile_msl(msl.encode())
if library == 0:
    fail("tm_compile_msl")
kernel = lib.tm_load_kernel(library, meta["name"].encode())
if kernel == 0:
    fail("tm_load_kernel")

# 4. Unified-memory buffers, one per pointer argument.
n = 5000
a = array.array("f", [i * 0.5 for i in range(n)])
b = array.array("f", [float(i % 13) for i in range(n)])


def upload(values):
    handle = lib.tm_alloc_buffer(values.itemsize * len(values))
    if handle == 0:
        fail("tm_alloc_buffer")
    address, count = values.buffer_info()
    if lib.tm_buffer_write(handle, 0, address, count * values.itemsize) != 0:
        fail("tm_buffer_write")
    return handle


buf_a, buf_b = upload(a), upload(b)
buf_out = lib.tm_alloc_buffer(4 * n)

# 5. Launch. `kinds`/`values` are parallel arrays; argument i binds at buffer i.
ARG_BUFFER, ARG_I32, ARG_F32 = 0, 1, 2
kinds = (i32 * 4)(ARG_BUFFER, ARG_BUFFER, ARG_BUFFER, ARG_I32)
values = (i64 * 4)(buf_a, buf_b, buf_out, n)
block = meta["block_size"]
status = lib.tm_launch(
    kernel,
    (n + block - 1) // block, 1, 1,      # grid of threadgroups
    meta["threads_per_threadgroup"],     # reported by the core
    kinds, values, 4,
)
if status != 0:
    fail("tm_launch")

# 6. Read back.
out = array.array("f", [0.0]) * n
address, _ = out.buffer_info()
if lib.tm_buffer_read(buf_out, 0, address, 4 * n) != 0:
    fail("tm_buffer_read")
assert list(out) == [x + y for x, y in zip(a, b)]
print("vector add matches: out[4999] =", out[4999])

# 7. Release everything; tm_live_handle_count is the leak check.
for handle in (buf_a, buf_b, buf_out):
    lib.tm_free_buffer(handle)
lib.tm_release_kernel(kernel)
lib.tm_release_library(library)
assert lib.tm_live_handle_count() == live_before
print("handles balanced:", lib.tm_live_handle_count())
```

Run against the vector-add IR from earlier:

```
kernel metadata: {"args": [...], "block_shape": [1024], "block_size": 1024,
                  "name": "add_kernel", "threads_per_threadgroup": 128}
vector add matches: out[4999] = 2506.5
handles balanced: 0
```

**Launch argument kinds.** `tm_launch` takes two parallel arrays. `kinds[i]`
says how to read `values[i]`:

| kind | `values[i]` |
| --- | --- |
| 0 | a buffer handle from `tm_alloc_buffer` |
| 1 | an `i32` scalar, sign-extended into the low 32 bits |
| 2 | an `f32` scalar, **as its bit pattern** in the low 32 bits |

Kind 2 is the one that trips people: pass `struct.unpack("<I", struct.pack("<f", x))[0]`,
not `int(x)`.

**Zero-copy writes.** `tm_buffer_contents(handle)` returns the host pointer into
unified memory, so a caller that would rather build its data in place can skip
`tm_buffer_write` entirely. `tm_buffer_write`/`tm_buffer_read` are the
bounds-checked copies; both refuse an out-of-range offset or length rather than
scribbling on the host.

---

## The Python shim

```
swift build                                             # produces the dylib
cd python && PYTHONPATH=. python3 -m pytest tests/ -q   # 14 passed
```

`_core.py` finds the dylib in this order: `$TRITON_METAL_CORE_LIB`,
`.build/release/`, `.build/debug/`, then next to the installed package. Set the
environment variable when embedding a wheel-installed build elsewhere.

### Using `_core` directly

`triton_metal._core` is a one-to-one binding for the `tm_*` exports. It marshals
arguments, translates the documented sentinel into a `RuntimeError` carrying
`tm_last_error`, and computes nothing:

```python
import array, json
from triton_metal import _core

print(_core.version(), _core.is_active(), _core.device_name())
# 0.0.1 True Apple M1 Pro

meta = json.loads(_core.kernel_info(TTIR, num_simdgroups=4))["kernels"][0]
library = _core.compile_msl(_core.emit_msl(TTIR, 4))
kernel = _core.load_kernel(library, meta["name"])

n = 4096
a = array.array("f", [i * 0.5 for i in range(n)])
b = array.array("f", [float(i % 7) for i in range(n)])

def upload(values):
    h = _core.alloc_buffer(values.itemsize * len(values))
    address, count = values.buffer_info()
    _core.buffer_write(h, 0, address, count * values.itemsize)
    return h

buf_a, buf_b = upload(a), upload(b)
buf_out = _core.alloc_buffer(4 * n)

block = meta["block_size"]
_core.launch(
    kernel,
    ((n + block - 1) // block, 1, 1),
    meta["threads_per_threadgroup"],
    [_core.ARG_BUFFER, _core.ARG_BUFFER, _core.ARG_BUFFER, _core.ARG_I32],
    [buf_a, buf_b, buf_out, n],
)

out = array.array("f", [0.0]) * n
address, _unused = out.buffer_info()
_core.buffer_read(buf_out, 0, address, 4 * n)
assert list(out) == [x + y for x, y in zip(a, b)]
print("ok, out[-1] =", out[-1])

for h in (buf_a, buf_b, buf_out):
    _core.free_buffer(h)
_core.release_kernel(kernel)
_core.release_library(library)
print("live handles:", _core.live_handle_count())
```

Output:

```
0.0.1 True Apple M1 Pro
ok, out[-1] = 2047.5
live handles: 0
```

`_core.launch` takes the grid as a 3-tuple and the kinds/values as ordinary
lists; `ARG_BUFFER`, `ARG_I32`, `ARG_F32` mirror the table above.

### The plugin surface

`python/plugin/` is what `TRITON_PLUGIN_DIRS` points at. Triton reads the backend
name out of `backend/name.conf`, links `backend/` in as `triton.backends.metal`,
and builds the directory as a CMake subproject:

| File | What it is |
| --- | --- |
| `backend/name.conf` | The single word `metal`. Read by both `setup.py` and `CMakeLists.txt`. |
| `backend/compiler.py` | `MetalBackend(BaseBackend)` for the pinned release: `add_stages` fills `ttir` (Triton's own passes) then `msl` (`tm_emit_msl` + `tm_kernel_info`); `binary_ext = "msl"`; `supports_target` matches `target.backend == "metal"`. |
| `backend/driver.py` | `MetalDriver(DriverBase)`, `MetalUtils.load_binary` (MSL bytes -> `MTLLibrary` -> pipeline), `MetalLauncher` (marshals to `tm_launch`), and a wall-clock `do_bench`. |
| `metal.cc` | Ten lines of C++ defining `init_triton_metal`, which Triton's `main.cc` declares and calls for every backend name. Without it libtriton does not link. |
| `CMakeLists.txt` | Registers that one file as a Triton plugin object. |

The Metal-specific stage is a one-liner over the Swift core:

```python
stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
# make_msl: ttir = str(mod); tm_kernel_info(ttir) -> launch metadata;
#           tm_emit_msl(ttir) -> MSL source bytes.
```

The `ttir` stage is **Triton's** MLIR pass pipeline, run in Triton's compiled pass
manager — this backend has no business reimplementing canonicalization. There is
no `ttgir` stage at all; see
[ARCHITECTURE.md §Compatibility](ARCHITECTURE.md#compatibility) for why that fork
point is deliberate.

**The language policy is a hard rule.** The shim exists only because Triton's
discovery imports a Python module. Any new functionality goes in
`Sources/TritonMetalCore` behind a `tm_*` export, never here — including anything
the *real* release's IR turned out to spell differently, which is why the two
gaps the tutorials exposed were fixed in the Swift parser and emitter rather than
worked around in Python. The Python tests enforce the visible half of this:
`test_jit_kernel_compiles_through_the_swift_core` asserts the stage names, and
`test_kernel_info_is_computed_in_the_core` asserts that block size and threadgroup
size arrive as JSON from Swift rather than being derived in Python.

### Passing tensors

Triton decides an argument is a pointer by looking for `data_ptr()`
(`python/src/specialize.cc`), and reads `dtype` to type the pointee. Torch is the
usual provider, but macOS torch wheels are CPU-only and a CPU allocation is not an
`MTLBuffer`, so kernel arguments come from `triton_metal.buffer.MetalBuffer`:

```python
from triton_metal.buffer import MetalBuffer

x = MetalBuffer.from_numpy(a)          # or MetalBuffer.from_torch(t) — copies
out = MetalBuffer(a.shape, "float32")
add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
print(out.numpy())
```

`from_torch`/`to_torch` **copy**. `MTLDevice.makeBuffer(bytesNoCopy:)` requires
page-aligned memory and torch does not guarantee it; a zero-copy path would have
to allocate the tensor's storage through Metal in the first place. For v1 the copy
is explicit and cheap enough that it is not the thing to optimise first.

---

## Supported IR subset

Condensed from
[ARCHITECTURE.md §Supported IR subset](ARCHITECTURE.md#supported-ir-subset),
which has the full detail and the reasoning.

### Types

| Triton | MSL | Notes |
| --- | --- | --- |
| `i1`, `i8`, `i16`, `i32`, `i64` | `bool`, `char`, `short`, `int`, `long` | |
| `f16`, `f32` | `half`, `float` | no `f64`, no `bf16` |
| `!tt.ptr<T>` | `device T *` | unified memory; no address-space variants |
| `tensor<D0x…xT>` | one per-lane value of `T` per block point | ranks 1 and 2 exercised end to end; higher ranks lower through the same machinery, untested |
| a `tt.dot` operand/accumulator | `threadgroup T[]`, moved as 8x8 `simdgroup_{half,float}8x8` fragments | |

Block pointers (`!tt.ptr<tensor<…>>`) are not lowered. ttgir layout encodings are
parsed and then ignored.

### Operations

| Group | Ops | Notes |
| --- | --- | --- |
| Structure | `tt.func`, `tt.return` | one kernel per function, must return void; argument position == buffer index |
| Program grid | `tt.get_program_id`, `tt.get_num_programs` | axes `x`/`y`/`z`, both the keyword and `{axis = N}` spellings |
| Tensor shape | `tt.make_range`, `tt.splat`, `tt.expand_dims`, `tt.broadcast`, `tt.trans` | `make_range` is 1-D, as in Triton; `expand_dims`/`broadcast`/`trans` emit no code |
| Memory | `tt.addptr`, `tt.load`, `tt.store` | any rank; `ptr[, mask[, other]]` → `mask ? *p : other`; store → `if (mask) …`; cache/eviction attributes parsed and ignored |
| Integer arithmetic | `arith.addi subi muli divsi divui remsi remui andi ori xori shli shrsi shrui maxsi minsi maxui minui` | unsigned variants cast to `uint`/`ulong`; `i1` `andi`/`ori`/`xori` become `&&`/`\|\|`/`!=` |
| Float arithmetic | `arith.addf subf mulf divf maximumf minimumf maxnumf minnumf` | |
| Compare / select | `arith.cmpi` (`eq ne slt sle sgt sge ult ule ugt uge`), `arith.cmpf`, `arith.select` | ordered and unordered `cmpf`; no explicit NaN-ordering yet |
| Conversions | `arith.sitofp uitofp fptosi fptoui extsi extui trunci extf truncf bitcast` | widths checked; `trunci` to `i1` takes the low bit |
| Math (`precise::`) | `math.exp exp2 log log2 sqrt rsqrt sin cos` | |
| Math (default ns) | `math.tanh erf floor ceil absf absi` | `erf` via a generated `tm_erf` (A&S 7.1.26, ~1.5e-7 abs) |
| Constants | `arith.constant` | scalars and `dense<…>` splats, including `0xFF800000` bit-pattern non-finites |
| Control flow | `scf.for`, `scf.if`, `scf.yield` | `iter_args` and multi-result `%r:2` / `%r#0`; a body containing a cross-lane op hoists the loop (below) |
| Reduction | `tt.reduce` | single operand, **innermost axis only**, combiner `add`/`max`/`min`; generic MLIR form only; folds within a row's lane group when there is an axis outside the reduced one, across the threadgroup when there is not |
| Matmul | `tt.dot` | rank 2, `f16`/`f32` with the accumulator at least as wide; `M`/`N`/`K` need not be multiples of 8 |

### Restrictions worth memorising

* **`tt.reduce` is generic-form only.** `"tt.reduce"(%x) <{axis = 0 : i32}> ({ … })`
  — that is how Triton prints an op carrying a region, so it costs nothing. Every
  *other* op must be in pretty form; generic syntax elsewhere is rejected.
* **Reductions are last-axis only**, and the axis must be the distributed one.
* **A cross-lane op inside an `scf.for` hoists the whole loop** to a
  threadgroup-uniform level. It may carry scalars, a `tt.dot` accumulator it
  yields directly (which stays in simdgroup registers for the whole loop), a
  contraction-space pointer (strength-reduced rather than carried), and any other
  per-lane tensor — the last of which is **spilled** to a threadgroup tile and
  updated through a shadow. A carried tensor spanning more than two block axes is
  refused.
* **A cross-lane op inside an `scf.if` is refused outright** — an `scf.if` is
  per-lane by construction, and the barriers would not be reached by every thread.
* **A `tt.trans` whose result is materialised is refused.** A transposed `tt.dot`
  operand is free (it is a relabelling, staged over its own tile's axes); a
  transposed value that has to be walked by an elementwise nest asks for two
  contradictory nestings of the same two block dimensions.
* **One `tl.arange` may not index both a row and a column** of the same value —
  that describes a diagonal, not a tile, and is refused by name. Emit a separate
  `tt.make_range` for each; Triton's CSE will otherwise merge them whenever the
  two block sizes are equal.

### Not supported

Each of these fails by name: `tt.cat`, `tt.atomic_*`, `tt.call`, `tt.histogram`,
`tt.scan`, multi-operand `tt.reduce` (argmax/argmin), reductions over a
non-innermost axis, cross-lane ops inside `scf.if`, a materialised `tt.trans`,
and generic op syntax for anything but `tt.reduce`.

Two things this list used to contain and no longer does: a per-lane tensor carried
across a cross-lane `scf.for` (now spilled to a threadgroup tile), and a `tt.dot`
whose operand comes from another `tt.dot`'s result (the first result is already a
tile, and the second dot's staging loops read it). Both were what
FlashAttention-2 was waiting on.

---

## Error behaviour

Nothing outside the subset is silently mis-compiled. Every rejection names the
offending op and its source position, and the position is a real line and column
in the IR text you handed in.

`CoreError` cases and their rendering:

| Case | Message shape |
| --- | --- |
| `.unsupportedOp` | `unsupported op '<name>' at line L, col C: the Metal backend does not lower this operation yet (see docs/ARCHITECTURE.md §Supported IR subset)` |
| `.unsupportedType` | `unsupported type '<name>' at line L, col C` |
| `.parse` | `parse error at line L, col C: <what was expected>` |
| `.lowering` | `lowering error at line L, col C: <what the emitter could not do>` |
| `.metal` | `metal error: <Metal's own diagnostic>` |
| `.invalidHandle` | `invalid handle: <buffer\|kernel\|library> handle N is not live` |
| `.invalidArgument` | `invalid argument: <what was wrong>` |

Swift throws them:

```swift
do {
    _ = try MetalCompiler.emitMSL(ttir: unsupportedIR, options: .init())
} catch {
    print(error)
}
// unsupported op 'tt.trans' at line 3, col 5: the Metal backend does not lower
// this operation yet (see docs/ARCHITECTURE.md §Supported IR subset)
```

C callers get the sentinel plus `tm_last_error`:

```
error: unsupported op 'tt.cat' at line 1, col 32: the Metal backend does not
lower this operation yet (see docs/ARCHITECTURE.md §Supported IR subset)
```

Python raises `RuntimeError` carrying the same string:

```python
>>> _core.emit_msl("module { tt.func public @k() { %0 = tt.scan %0 : i32 tt.return } }")
RuntimeError: unsupported op 'tt.scan' at line 1, col 32: the Metal backend does
not lower this operation yet (see docs/ARCHITECTURE.md §Supported IR subset)
```

Lowering errors that are *not* about a missing op are just as specific. A
`tl.arange` that reached both a row and a column position, for instance, does not
say "unsupported":

```
lowering error at line L, col C: '%12' indexes one block dimension with two of its
own dimensions; a value reached both a row and a column position (a tl.arange
expanded along axis 0 in one place and axis 1 in another), which describes a
diagonal rather than a tile
```

Others in the same family: a block shape whose tiles overrun Metal's 32 KB
threadgroup budget is refused **with the byte count**; a `tt.reduce` whose
recomputed region would contain a store or an `scf` region is refused rather than
duplicating the store; a `tt.dot` with mismatched contraction dimensions prints
both operand types.

MSL that Metal's own front end rejects surfaces as
`metal error: MSL compilation failed: <Metal's diagnostic>`. If you see one of
those from generated source, that is a backend bug — please keep the emitted MSL,
which is exactly why it is emitted as text.

---

## Implementing what's missing

The first two entries were the FlashAttention-2 blockers and are **done** — kept
here because what they turned out to need was not what this section predicted, and
that is the useful part. The next three are the matmul performance gap and are all
local to one function; the last is the Triton pin, which is now the most important
open task in the project.

### 1–2. ~~The FlashAttention-2 blockers~~ — done

Both landed, and the write-up of what they actually needed is in
[ARCHITECTURE.md §Cross-lane regions](ARCHITECTURE.md#cross-lane-regions) and
§Execution model. The short version, because the estimate in this section used to
be wrong in an instructive way:

* **A per-lane tensor carried across a cross-lane loop** is now spilled to a
  threadgroup tile and updated through a shadow tile, written where the new value
  is *defined* rather than at the `scf.yield` — the nest that computed it is
  closed by then, and the old value is usually still being read. The one special
  case worth having: when the loop yields a `tt.dot`'s result and the dot's
  accumulator is a per-lane function of the carried value
  (`acc = tt.dot(p, v, acc * alpha[:, None])`), the carried tile *is* the dot's
  accumulator tile and the rescale becomes the staging pass that fills it — no
  copy at all.
* **`tt.trans`** turned out cheaper than the `transpose_matrix` route this section
  proposed: it is a relabelling of which block axis each dimension indexes, so
  `K^T` is staged over `(HEAD_DIM, BLOCK_N)` and **no code is emitted**. Setting
  `simdgroup_load`'s flag is still worth doing later — staging `K^T` this way reads
  `K` down its columns, so the loads are not coalesced — but that is throughput,
  not correctness.
* **What the estimate missed.** FA-2's `m_i`/`l_i` are not scalars but
  `BLOCK_M`-wide vectors, so they needed the spill too. And two larger changes
  were not on the list at all: a kernel's tensors stopped sharing one loop nest
  (`p` spans `(M, N)`, `acc` spans `(M, HEAD_DIM)`, and neither contains the
  other), and axis assignment became unification rather than identity seeding,
  because FA-2's two dots share their axes crosswise.

The kernel itself is `AttentionKernel.forward(blockM:blockN:headDim:element:)` in
`TritonMetalBench`, tested in `AttentionTests` and benchmarked by
`tmbench --attn`.

### What to do next on attention

**A. Keep the score tile out of threadgroup memory.** The f32 accumulator
(`BLOCK_M x HEAD_DIM`) and the f32 score tile (`BLOCK_M x BLOCK_N`) are both live
for the whole iteration, which is why `BLOCK_M` above 32 does not fit at
`HEAD_DIM = 64` and why the kernel stages 10240 elements to do 131072 MACs at
`16x64`. This is the same change the GEMM wants for its accumulator, and it is the
one that lifts the cap on arithmetic intensity.

**B. Hoist loop-invariant dot operands.** `Q` is restaged on every iteration
although nothing in it depends on the induction variable — 1024 redundant device
loads per iteration at `16x64`. The deferral model rebuilds an operand inside the
dot's own loops and does not yet notice that this one need not move.

**C. Stop computing `P` twice.** It is materialised once for the row sum and
rebuilt once inside the second dot's staging — one extra `exp2` per element per
iteration. Using its spill tile as the dot's operand tile would remove that, at
the cost of one more `BLOCK_M x BLOCK_N` tile.

### 3–5. The matmul performance gap

The kernel reaches 76% of `MPSMatrixMultiplication` at 1024, 2048 and 4096
square (4.66 TFLOP/s f32 at 2048 on an M1 Max; 2.33 TFLOP/s on an M1 Pro), up from
~33% at the first working version. That clears the >50% milestone and lands inside
the 62–82% band published Triton reaches on its own target. The per-change
attribution — and, more usefully, the two CUDA-playbook techniques that did *not*
transfer to M1-generation Apple silicon (all of it measured on M1 Max and M1 Pro;
nothing here has been re-run on M2/M3/M4), plus the one with no CUDA analogue that
beat both — is in
[ARCHITECTURE.md §Matmul throughput](ARCHITECTURE.md#matmul-throughput).

What landed, all local to `emitDot` and its helpers: register blocking of the
output fragments, accumulators resident in simdgroup registers across the whole K
loop, the accumulator's tile doubling as the arena its operand tiles stage into
(which is what lets a 64x128 block shape exist at all), staging in runs of
consecutive columns, compile-time staging trip counts, a zero accumulator that
never enters threadgroup memory, a 2-D-distributed epilogue, a wave-to-fragment
mapping that lets every wave of a simdgroup share one B fragment (+15%), and
`float4` staging runs behind a runtime contiguity check (+20%), and two fragment
rows per simdgroup where the blocking score ties (+8%).

What is left, in the order to try it:

**A. Keeping the accumulator out of threadgroup memory entirely.** A 128x128 tile
would halve staging traffic again, and its f32 accumulator alone is 64KB — twice
the whole budget. The only route is an epilogue that streams register fragments
out in panels instead of through one full-size tile, which means giving the
epilogue its own loop over panels. Much the largest of the three, and untried: the
throughput target was reached without it.

**B. Vector staging for narrow tiles.** A tile is only vectorised when a run of
four columns leaves no thread idle. At the winning `64x64x16` on 8 simdgroups both
operand tiles clear that bar; at `64x128x16` on 16 the `64x16` A tile does not and
stages scalar. Forcing runs of four there measures 3069 GFLOP/s against 4223 — the
occupancy loss dwarfs the vector win. What is needed is a staging distribution that
lets a *narrow* tile hand out runs of four without idling threads (one thread
taking several rows), not a longer run.

**C. Specialising the run mask away.** The vector fast path re-checks the mask at
the run's last column on every K step. When `BLOCK_N` divides `N` — which the
launcher knows and the emitter does not — it is statically true, and so is most of
the masking in the scalar path behind it.

Measure with `swift run -c release tmbench` (works without Xcode), or
`TM_BENCH=1 swift test --filter MatmulBenchmark`. `tmbench --attn` is the same
sweep for attention, against the unfused MPS composite;
`tmbench --attn-shapes 1,8,512,64 --verbose` prints every configuration and
`tmbench --emit-attn 32,32,16,64` prints the kernel. `tmbench --sweep full` sweeps
block shapes, `num_warps`, register blocking, tile padding and double buffering;
`tmbench --config 64,128,16,16,0,0,0,-1,0,0` turns vector staging off;
`tmbench --config 64,128,16,16` pins one configuration, which is how a change
should be attributed — one axis at a time. `tmbench --emit 64,64,16,16` prints the
kernel. Ignore the 512 row when judging a change: that GEMM is under a millisecond
and MPS's own timing swings by 1.5x run to run.

### 6. ~~Pinning a Triton release~~ — done

Pinned at **v3.7.1**; `python/plugin/backend/` carries that tag's
`BaseBackend`/`DriverBase` signatures, and the three tutorials above run through
it. The recipe is [Building Triton with the Metal backend](#building-triton-with-the-metal-backend);
the findings are [ARCHITECTURE.md §Compatibility](ARCHITECTURE.md#compatibility).
What is left of this task:

1. **CI against the pin.** `python/tests/test_triton_roundtrip.py` asserts
   `triton.__version__ == 3.7.1` and drives a real kernel, but skips when Triton
   is absent — which is every machine that has not spent the 9 minutes. A CI job
   that does spend them turns the next interface churn into a test failure rather
   than a mystery.
2. **Axis identity independent of extent.** The one constraint the matmul
   tutorial exposed: block sizes must be pairwise distinct, because
   `LayoutInference` identifies a block axis by its extent. `(64, 64, 32)` is
   refused. This is the highest-value correctness item now.
3. **Zero-copy tensors.** `MetalBuffer.from_torch` copies. Either allocate torch
   storage through Metal, or accept a page-aligned external allocation via
   `makeBuffer(bytesNoCopy:)` and add the `tm_*` export for it.
4. **Cacheable binaries.** `binary_ext` is `msl`, so every process re-runs
   `MTLDevice.makeLibrary(source:)`. `tm_load_metallib` already exists; what is
   missing is a way to *produce* the bytes without `xcrun metal`, which Command
   Line Tools do not ship.
5. **Conformance.** Port a subset of Triton's own `test_core.py` against numpy
   references — the signal this backend still does not have. Now that real Triton
   drives the backend, that suite is runnable rather than hypothetical.
6. **i64 and f64 kernel arguments.** `tm_launch` carries `i32`/`f32` scalars only;
   a 64-bit scalar argument is refused by the launcher with a clear message.
