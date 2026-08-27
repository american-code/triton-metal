# triton-metal architecture

## Language policy

Swift core (`Sources/TritonMetalCore`), C ABI (`tm_*` in libtritonmetal.dylib),
Python only as the unavoidable Triton-plugin shim (`python/triton_metal` — ctypes
bindings, no logic). Python here is justified solely because Triton's backend
discovery imports a Python module and rebuilding Triton's frontend is a multi-month
project. Any new functionality goes in the Swift core and gets a `tm_*` export.

## Compatibility

Triton's backend plugin interface has churned across releases. First real task:
pin a Triton release (whatever ships with current PyTorch), vendor its
`BaseBackend`/`GPUDriver` signatures into the stubs, and add a CI check against
that pin. Everything below assumes the ~3.x layout (`triton.backends` discovery,
stage-dict compiler, driver with `load_binary`/`launch`).

## Lowering pipeline

```
ttir      Triton IR — reuse Triton's canonicalization passes unchanged
ttgir     TritonGPU IR — needs a Metal target profile:
            - warp size 32 == simdgroup size 32 (convenient!)
            - no cross-threadgroup barrier (no grid.sync equivalent);
              kernels requiring it must be split or use atomics
            - shared memory == threadgroup memory (32KB+ per threadgroup)
msl       Emit Metal Shading Language source from ttgir ops:
            - tl.load/store → device pointer arithmetic (unified memory)
            - tl.dot → simdgroup_matrix (simdgroup_float8x8) ops
            - reductions → simd_shuffle_down intrinsics
            - barriers → threadgroup_barrier(mem_flags)
metallib  MTLDevice.makeLibrary(source:) at runtime (primary path — no Xcode
          command-line tools required, no temp files), with
          `xcrun metal -std=metal3.1` available for offline .metallib bytes.
          xcodebuild is never used.
```

Emitting textual MSL first (rather than AIR/LLVM bitcode) trades some compile
speed for debuggability — you can read every kernel the backend produces. AIR
emission is a later optimization once correctness is established.

## Supported IR subset

What the emitter lowers **today**. Anything outside this list is rejected by name
and source line (`unsupported op 'tt.trans' at line 12, col 10: ...`) rather than
silently mis-compiled.

### Execution model

One Metal threadgroup per Triton program (`tt.get_program_id x` ==
`threadgroup_position_in_grid.x`).

Every tensor in a kernel is a slice of one **block**: an index space of rank R
with `BLOCK[d]` elements along dimension `d`. The emitter walks it with R nested
loops. Only the **innermost** dimension is distributed over the threadgroup's
threads; outer dimensions are looped uniformly by every thread:

```metal
for (uint tm_i0 = 0u; tm_i0 < BLOCK_M; ++tm_i0) {                 // uniform
    for (uint tm_i1 = tm_thread_id.x; tm_i1 < BLOCK_N;            // distributed
         tm_i1 += tm_threadgroup_size.x) { ... }
}
```

A value is emitted at the **shallowest depth its own dimensions allow**: one past
the deepest block dimension it actually varies along, or deeper if an operand
forces it. So `offs_m[:, None] * stride` (rank 2, but constant along the columns)
is computed once per row, `tt.get_program_id` once per program, and only
column-varying work lands in the innermost loop. Statements shallower than the
distributed loop run redundantly in every thread, so a `tt.store` at such a depth
is guarded with `tm_thread_id.x == 0u`.

Which block dimension a value spans is **inferred**, not assumed: `tl.arange(0, M)`
is rank 1 even inside a rank-2 kernel, and only the `tt.expand_dims` that consumes
it says whether it indexes rows or columns. `Layout.swift` seeds every full-rank
tensor with the identity map and propagates backwards through `tt.expand_dims`,
`tt.broadcast` and the elementwise ops to a fixed point; a rank-deficient value
that never reaches one is an error, not a guess.

`threadsPerThreadgroup` is `num_warps * 32`, clamped to the innermost block
dimension, rounded **up to a whole simdgroup** and capped at Metal's 1024. The
rounding matters: `simd_shuffle_down` reads from lanes that must be live, so a
threadgroup is never a partial simdgroup. A kernel containing a `tt.dot` skips the
innermost-dimension clamp — that clamp is an occupancy heuristic for elementwise
kernels, and a dot spreads 8x8 fragments and tile staging across the whole
threadgroup, so clamping would only starve it.

Kernels that are purely elementwise stay correct at any threadgroup size (tested
from `num_warps=1` to `32`); kernels with a `tt.reduce` or a `tt.dot` must be
launched at the reported `threads_per_threadgroup`.

`scf` regions lower *inside* the block loops by default, so an `scf.for` carrying
a tensor becomes a per-lane loop and each lane carries its own accumulator — no
spilling to threadgroup memory. A region that performs a cross-lane operation
cannot stay there; see §Cross-lane regions.

A `tt.reduce` does the opposite: it **closes** the distributed loop, folds the
per-thread partials, and opens a fresh loop for whatever follows. Tensor values
that were live across the reduction are *recomputed* in the new loop rather than
spilled (all lowered ops are pure). Softmax therefore emits three lane loops over
the same row — the redundancy is real, and cheaper than 3 x BLOCK floats of
threadgroup memory. Recomputation is refused with a precise error when the
recomputed region would contain a store or an `scf` region.

### Cross-lane regions

A `tt.dot` and a `tt.reduce` both need every thread in the threadgroup to reach
the same `threadgroup_barrier`. An `scf.for` whose body contains either is
therefore **hoisted out of the block loops entirely** and lowered
threadgroup-uniformly; the block loops are then opened and closed *inside* its
body, as the cross-lane ops require. Its recompute bookkeeping is scoped to the
region: it inherits the enclosing `history` (a lane loop opened inside the body
rebuilds whatever was live when the loops closed around the loop) but does not
leak its own statements back out.

Hoisting costs the loop its ability to carry per-lane tensors, so exactly three
shapes of `iter_args` are lowered:

* **scalars**, which are uniform anyway (a reduction's result is broadcast to
  every thread) and stay ordinary variables — this is what an online softmax's
  running maximum and running sum are;
* a **`tt.dot` accumulator**, which becomes a threadgroup tile *and* a block of
  simdgroup registers per simdgroup: the tile is initialised before the loop, read
  into the registers once, updated only in registers for the whole loop, and
  written back to the tile after it for the per-lane epilogue (§`tt.dot`);
* a **contraction-space pointer** (`a_ptrs += BLOCK_K * stride_ak`), which is not
  carried at all: it is strength-reduced back to `init + trip_count * delta` so
  that the dot can rebuild it from scratch (see §`tt.dot`).

A per-lane tensor carried across such a loop is refused by name — that is the
remaining FlashAttention-2 blocker, described precisely in §Hard parts 3. An
`scf.if` is per-lane by construction and still refuses both ops.

### `tt.dot`, the contraction dimension, and the matrix value class

Everything above is *one scalar per lane per block point*. A
`simdgroup_float8x8` is not that: it is a per-simdgroup object whose 64 elements
sit in an opaque distribution across 32 lanes, populated by `simdgroup_load` from
memory rather than assembled from per-lane registers. `tt.dot` therefore adds
three things.

**A contraction dimension.** `tt.dot`'s operands are `MxK` and `KxN` while its
result is `MxN`, so the block index space grows a dimension that is deliberately
*not* iterated by the elementwise lane distribution. `BlockLayout` keeps the
iteration axes (`rank`, `shape`) separate from `contractions`, one per dot, at
axis index `rank + i`. Inference seeds the dot's operands first — `a` spans
`(M, K)`, `b` spans `(K, N)`, accumulator and result span `(M, N)` — and only
then identity-seeds the ordinary elementwise tensors, so that everything feeding
an operand inherits the operand's axes rather than being forced onto the
iteration space.

**A second value class.** A tile is a tensor value living in a threadgroup array
rather than in per-lane registers. Tiles are padded up to whole 8x8 fragments and
the padding is zero-filled during staging, which is what makes `BLOCK_M`,
`BLOCK_N` and `BLOCK_K` that are not multiples of 8 work without a separate
edge-tile path. Once the emitter is back inside the block loops a tile reads as
an ordinary per-lane value — the SSA name is bound to `tile[i0 * padN + i1]` —
so every consumer downstream of a dot (casts, elementwise math, masked stores)
needs no special case at all.

**Deferral.** A dot operand is never materialised where it appears: it is rebuilt
inside the dot's own staging loops, over an index space that includes the
contraction dimension. A backward liveness walk (`DotDeferral`) marks a value
*materialised* when some use other than "an operand of a `tt.dot`" needs it, and
everything else is deferred. Deferred values are recorded, not emitted; the dot
replays their producer subgraph in program order inside its staging nest, at the
shallower of the two staging loops each one can live in. A value that spans the
contraction axis *and* is materialised — stored, say — is refused by name.

The lowering itself. The accumulator's fragments are handed out **once**, before
the contraction loop, and each simdgroup keeps a `tilesM x tilesN` block of them
in registers for the whole of it:

```metal
<stage the accumulator into tm_dot_c, over (M, N)>
threadgroup_barrier(mem_flags::mem_threadgroup);
uint bm = tm_simd_group / BN, bn = tm_simd_group % BN;   // this simdgroup's block
simdgroup_float8x8 c_0_0, c_0_1, c_1_0, c_1_1;           // tilesM x tilesN of them
simdgroup_load(c_0_0, tm_dot_c + ...); ...               // once, not once per step
threadgroup_barrier(mem_flags::mem_threadgroup);         // tm_dot_c is now free

for (int k = 0; k < K_TILES; ++k) {                      // the scf.for over K
    <stage A into tm_dot_a, over (M, K)>                 // spread over both dims
    <stage B into tm_dot_b, over (K, N)>
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0u; s < FK; ++s) {
        simdgroup_load(a_0, ...); simdgroup_load(a_1, ...);       // tilesM
        simdgroup_load(b_0, ...); simdgroup_load(b_1, ...);       // tilesN
        simdgroup_multiply_accumulate(c_0_0, a_0, b_0, c_0_0);    // tilesM * tilesN
        simdgroup_multiply_accumulate(c_0_1, a_0, b_1, c_0_1);
        simdgroup_multiply_accumulate(c_1_0, a_1, b_0, c_1_0);
        simdgroup_multiply_accumulate(c_1_1, a_1, b_1, c_1_1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(c_0_0, tm_dot_c + ...); ...              // once, for the epilogue
threadgroup_barrier(mem_flags::mem_threadgroup);
```

Four things in that shape are worth naming, because each was measured
independently (§Matmul throughput).

**Register blocking.** A `simdgroup_multiply_accumulate` needs two operand
fragments, so one output fragment per simdgroup costs two `simdgroup_load`s per
unit of arithmetic. A `tilesM x tilesN` block amortises them: the A fragment of a
fragment row feeds every output in that row and the B fragment of a column feeds
every output in that column, so `tilesM + tilesN` loads feed `tilesM * tilesN`
multiply-accumulates. The blocking is **chosen**, not fixed, by a score over three
quantities — intensity (`tilesM * tilesN` per `tilesM + tilesN`), occupancy (the
last wave over the block grid should not idle most of the threadgroup) and fit (a
block that does not divide the fragment grid pads the tile out to whole blocks,
which costs both memory and arithmetic on zeros). It picks 2x2 for a 64x64
accumulator on 16 simdgroups and 4x4 for the same accumulator on 4. The autotuner
can override it.

**Register residency.** The fragments never touch threadgroup memory between K
steps. This needs the simdgroup count to be a compile-time constant — it is, it is
the reported `threads_per_threadgroup`, which a `tt.dot` kernel's launcher already
has to honour — so every fragment gets its own *named* variable rather than a slot
in an array, which a `simdgroup_float8x8` cannot be indexed out of without
spilling.

**The accumulator's tile doubles as the operand arena.** The accumulator tile is
written before the loop, read straight into registers, and untouched until the
registers go back after it — which is exactly the window in which the operand
tiles exist. So they share storage, and the barrier after the register load is
what makes that safe. A 128x64 f32 accumulator is the entire 32KB budget on its
own; without the sharing, no block shape that large could stage anything.

**Staging in runs.** Each thread stages a run of consecutive columns rather than a
single one. Every staged element costs a masked device load plus its share of the
address arithmetic, and with one column per thread none of that arithmetic is
shared — the row half is recomputed per element although it is loop-invariant, and
the column half has no loop to strength-reduce along. A run fixes both. The row
stride is spelled as a literal rather than `tm_threadgroup_size`, which gives every
staging loop a compile-time trip count.

f16 operands with an f32 accumulator — the shape that matters for ML — mix
`simdgroup_half8x8` operands with a `simdgroup_float8x8` accumulator, which Metal
supports directly. When the accumulator is f32 and the operands f16 the operand
tiles are cast pointers into the f32 arena.

Staging spreads the threadgroup over **both** tile dimensions rather than only
the innermost one: a `BLOCK_M x BLOCK_K` tile is usually much narrower than the
threadgroup, and striding only its columns leaves most threads idle. `lanes`
consecutive threads take consecutive runs of columns, which keeps the device reads
coalesced, and the rest walk the rows. This is worth ~4.7x on the matmul tutorial
— it was, by a distance, the single biggest thing in the kernel.

Tile storage is checked against Metal's 32KB threadgroup budget and over-large
block shapes are refused with the byte count.

Two further knobs exist and are **off by default**, because both measured as
losses or wash on Apple silicon (§Matmul throughput): `dotTilePadding`, which
skews tile rows across threadgroup-memory banks, and `dotDoubleBuffer`, which
ping-pongs the operand tiles between two halves of the arena so a contraction
step's staging need not wait on the previous step's arithmetic.

### Floating-point policy

Fast math is **off** (`MTLCompileOptions.mathMode = .safe`), which Metal does not
default to. `math.*` maps to Metal's `precise::` namespace where one exists —
`exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `sin`, `cos` — and to the default
namespace where it does not (`tanh`, `floor`, `ceil`, `fabs`, `abs`); `fast::` is
never emitted. Metal has no `erf` at all, so `math.erf` lowers to a generated
`tm_erf` helper (Abramowitz & Stegun 7.1.26, ~1.5e-7 absolute — float precision
over erf's range). Every one of these is checked against a CPU reference on the
GPU with a per-function tolerance.

### Types

| Triton | MSL |
| --- | --- |
| `i1`, `i8`, `i16`, `i32`, `i64` | `bool`, `char`, `short`, `int`, `long` |
| `f16`, `f32` | `half`, `float` |
| `!tt.ptr<T>` | `device T *` (unified memory; no address-space variants yet) |
| `tensor<D0x...xDNxT>` | one per-lane value of `T` per point of the block |
| a `tt.dot` operand / accumulator | a `threadgroup T[]` tile, moved in 8x8 `simdgroup_{half,float}8x8` fragments |

`f64`, `bf16`, block pointers (`!tt.ptr<tensor<...>>`) and ttgir layout encodings
(parsed, then ignored) are not lowered. Ranks 1 and 2 are exercised end to end;
higher ranks lower through the same nested-loop machinery but are untested.

### Operations

| Op | Notes |
| --- | --- |
| `tt.func` | One kernel per function; must return void. Argument attributes ignored. Argument position == Metal buffer index. |
| `tt.return` | Terminator only; returning a value is an error. |
| `tt.get_program_id` | Axes `x`/`y`/`z`, both the keyword and `{axis = N}` spellings. |
| `tt.get_num_programs` | Lowers to `[[threadgroups_per_grid]]`. |
| `tt.make_range` | 1-D only (as in Triton); `start`/`end` must match the result length. |
| `tt.splat` | Scalar -> tensor (including pointer splats). |
| `tt.expand_dims` | Inserts a size-1 dimension. A relabelling — no code emitted. |
| `tt.broadcast` | Expands size-1 dimensions. A relabelling — no code emitted. |
| `tt.addptr` | Pointer + integer offset, scalar or tensor, any rank. |
| `tt.load` | `ptr[, mask[, other]]`, any rank. Masked loads become `mask ? *p : other` (`other` defaults to zero). Cache/eviction attributes are parsed and ignored. |
| `tt.store` | `ptr, value[, mask]` -> `if (mask) { *p = v; }`, any rank. |
| `tt.reduce` | Single-operand, **last axis only** (the distributed one), combiner `add`/`max`/`min`. Lowers to `simd_shuffle_down` within each simdgroup plus threadgroup memory across them. Parsed in MLIR's generic form, which is how Triton prints ops with regions. Allowed inside an `scf.for` (see §Cross-lane regions), not inside an `scf.if`. |
| `tt.dot` | `%d = tt.dot %a, %b, %acc[, inputPrecision = ...] : tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxU>`, rank 2, `T`/`U` in `{f16, f32}` with `U` at least as wide. `M`, `N`, `K` need not be multiples of 8. Lowers to threadgroup tiles + `simdgroup_multiply_accumulate` over 8x8 fragments (see §`tt.dot`). The old `{allowTF32}` attribute spelling is accepted and ignored — Metal has one precision per element type. |
| `arith.constant` | Scalar and `dense<...>` splat integer/float constants, including MLIR's `0xFF800000` bit-pattern spelling of non-finite floats. |
| `arith.addi/subi/muli/divsi/divui/remsi/remui` | Unsigned variants cast to `uint`/`ulong`. |
| `arith.andi/ori/xori/shli/shrsi/shrui/maxsi/minsi/maxui/minui` | `i1` `andi`/`ori`/`xori` become `&&`/`\|\|`/`!=`. |
| `arith.addf/subf/mulf/divf/maximumf/minimumf/maxnumf/minnumf` | |
| `arith.cmpi` | `eq ne slt sle sgt sge ult ule ugt uge`. |
| `arith.cmpf` | Ordered and unordered comparisons (no explicit NaN-ordering yet). |
| `arith.select` | Scalar or per-lane condition. |
| `arith.sitofp/uitofp/fptosi/fptoui` | Unsigned forms go through an explicit `uint`/`ushort` cast. |
| `arith.extsi/extui/trunci/extf/truncf/bitcast` | Widths are checked; `trunci` to `i1` takes the low bit. |
| `math.exp/exp2/log/log2/sqrt/rsqrt/sin/cos` | Metal's `precise::` namespace. |
| `math.tanh/erf/floor/ceil/absf/absi` | Default namespace; `erf` via the generated `tm_erf`. |
| `scf.for` | `iter_args` and results, scalar or tensor, including the multi-result `%r:2 = ... -> (T, U)` / `%r#0` spelling. Results alias the carried variables. A body containing a `tt.dot` or `tt.reduce` hoists the whole loop to a threadgroup-uniform level and restricts what it may carry — see §Cross-lane regions. |
| `scf.if` | With or without results; results need an `else` region. |
| `scf.yield`, `tt.reduce.return` | Region terminators. |

Not supported (each fails with the op name): `tt.trans`, `tt.cat`,
`tt.atomic_*`, `tt.call`, `tt.histogram`, `tt.scan`, multi-operand `tt.reduce`
(argmax/argmin), reductions over a non-innermost axis, cross-lane operations
inside an `scf.if`, a per-lane tensor carried across an `scf.for` that contains a
cross-lane operation, a `tt.dot` whose operand is another `tt.dot`'s result, and
MLIR's generic op syntax for anything except `tt.reduce` — dump the pretty form
instead.

## C ABI reference

All exports live in `libtritonmetal.dylib`. Conventions: `char *` results are
malloc'd and freed by the caller with `tm_free`; handle-returning calls return **0**
on failure, `int32` calls return **-1**; the reason is always retrievable from
`tm_last_error` (which clears it). Handles are opaque `int64` values from one
process-wide table.

| Symbol | Signature | Purpose |
| --- | --- | --- |
| `tm_version` | `char *()` | Core version string. |
| `tm_free` | `void(void *)` | Free a returned string. |
| `tm_last_error` | `char *()` | Last error message; clears it. |
| `tm_is_active` | `int32()` | 1 when a Metal device exists. |
| `tm_device_name` | `char *()` | Default device name. |
| `tm_emit_msl` | `char *(const char *ttir, int32 num_simdgroups)` | Lower IR text to MSL source. |
| `tm_kernel_info` | `char *(const char *ttir, int32 num_simdgroups)` | Launch metadata as JSON (see below). |
| `tm_compile_msl` | `int64(const char *source)` | Runtime-compile MSL -> library handle. |
| `tm_load_metallib` | `int64(const void *bytes, int64 len)` | Load offline `.metallib` -> library handle. |
| `tm_release_library` | `int32(int64 library)` | |
| `tm_load_kernel` | `int64(int64 library, const char *name)` | Build a compute pipeline -> kernel handle. |
| `tm_kernel_max_threads` | `int64(int64 kernel)` | Hardware threadgroup-size limit. |
| `tm_release_kernel` | `int32(int64 kernel)` | |
| `tm_alloc_buffer` | `int64(int64 nbytes)` | `.storageModeShared` (unified) buffer. |
| `tm_buffer_contents` | `void *(int64 buffer)` | Host pointer into unified memory. |
| `tm_buffer_length` | `int64(int64 buffer)` | |
| `tm_buffer_write` | `int32(int64 buffer, int64 offset, const void *src, int64 len)` | Bounds-checked copy in. |
| `tm_buffer_read` | `int32(int64 buffer, int64 offset, void *dst, int64 len)` | Bounds-checked copy out. |
| `tm_free_buffer` | `int32(int64 buffer)` | |
| `tm_launch` | `int32(int64 kernel, int64 gx, int64 gy, int64 gz, int64 threads, const int32 *kinds, const int64 *values, int32 argc)` | Dispatch + wait. |
| `tm_live_handle_count` | `int64()` | Live handles (leak check for tests). |

`tm_launch` binds argument `i` at buffer index `i` (== the `tt.func` argument
position), described by two parallel arrays:

| kind | `values[i]` |
| --- | --- |
| 0 | buffer handle from `tm_alloc_buffer` |
| 1 | `i32` scalar, sign-extended into the low 32 bits |
| 2 | `f32` scalar, as its bit pattern in the low 32 bits |

`tm_kernel_info` returns everything the caller needs to size a launch, so the
Python shim never computes it:

```json
{"kernels": [{"name": "add_kernel", "block_size": 1024, "block_shape": [1024],
              "threads_per_threadgroup": 128,
              "args": [{"index": 0, "kind": "pointer", "dtype": "f32"},
                       {"index": 3, "kind": "scalar", "dtype": "i32"}]}]}
```

`block_size` is the product of `block_shape`; a scalar-only kernel reports an
empty shape and one thread.

## Matmul throughput

The matmul tutorial kernel, lowered by this backend, against
`MPSMatrixMultiplication`, f32, square, best configuration per size out of the
`tmbench` sweep. Dispatches are packed into one command buffer until each timed
sample is ~25ms of GPU work, so submission overhead is not most of what either
side is measured doing; median of three samples after a warm-up.

Two machines, because the ratio depends on both. **lab-02** is a Mac Studio
**M1 Max** (measured peaks 6.2 TFLOP/s f32, 371.5 GB/s) and is the machine to
believe: it has no Xcode, which is why `tmbench` exists at all. The **M1 Pro** is
a laptop and its MPS readings move with its thermal state.

### Apple M1 Max (lab-02)

| size | best configuration | before | after | MPS | before | after |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | `64x64x16`, `num_warps=8` | 1.44 TF | **2.01 TF** | ~2.18 TF | 62% | 92% |
| 1024 | `64x64x16`, `num_warps=8` | 1.85 TF | **2.75 TF** | ~5.54 TF | 33% | **50%** |
| 2048 | `64x128x16`, `num_warps=16` | 1.99 TF | **3.06 TF** | ~6.10 TF | 33% | **50%** |
| 4096 | `64x128x16`, `num_warps=16` | 1.97 TF | **3.21 TF** | ~6.13 TF | 32% | **52%** |

### Apple M1 Pro (laptop)

| size | best configuration | before | after | MPS | before | after |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | `64x128x16`, `num_warps=16` | 1.02 TF | **1.57 TF** | ~3.17 TF | 34% | **50%** |
| 2048 | `64x128x16`, `num_warps=16` | 1.11 TF | **1.69 TF** | ~3.37 TF | 33% | **50%** |

**About 1.55x throughput on both chips, and ~33% of MPS becomes ~50%.** The
milestone's >50% is met at the sizes where the measurement is trustworthy; the
60–80% band this round was aimed at is **not** reached. The 512 row is not
evidence of anything: that GEMM is under a millisecond and MPS's own timing swings
by 1.5x run to run at that size — it is reported for completeness, not for its
ratio.

Reproduce with `.build/release/tmbench --sweep full`, or
`TM_BENCH=1 swift test --filter MatmulBenchmark` where XCTest exists.

### What each change was worth

Measured one at a time on lab-02 at 2048, each on top of the one before it, by
pinning a configuration with `tmbench --config`.

| change | GFLOP/s | delta |
| --- | --- | --- |
| baseline (one output fragment per simdgroup, tile round trip per K step) | 1993 | — |
| + accumulator resident in registers across the K loop | 2257 | +13% |
| + register blocking (2x2 fragments per simdgroup) | 2485 | +10% |
| + staging in runs of consecutive columns | 2726 | +10% |
| + literal (compile-time) staging trip counts | 2902 | +6% |
| + accumulator tile doubling as the operand arena, `64x128` block | 3062 | +6% |
| + bank-conflict padding of 4 elements per tile row | ~2990 at `64x64` | +2%, size-dependent |

The last row is the odd one out: padding helps `64x64` slightly and is a net loss
at `64x128`, because the slack it adds is what stops that block shape fitting. It
is off by default and swept.

### What did not help, and why that is interesting

**Double buffering is a wash on Apple silicon.** With two operand buffers a
contraction step stages into the half the previous step is not reading, so the
trailing barrier can go and the staging need not wait on the arithmetic. On lab-02
at 2048 this measured 2863 GFLOP/s against 2900 without it — inside the noise, and
if anything negative. Metal has no `cp.async`: the prefetch is issued by the same
threads, into the same issue slots, so nothing is actually asynchronous. What CUDA
buys with a dedicated copy engine, Apple's GPU buys with occupancy instead — and
doubling the operand tiles *costs* occupancy, which is roughly what cancels it. It
is kept as `dotDoubleBuffer`, off by default, because it is cheap to re-measure on
a future chip.

**Bank-conflict padding is marginal and not free.** Four elements of slack per tile
row was worth ~2–3% at `64x64`, and the slack comes out of the same 32KB budget
that decides how large a block shape can be — with padding on, a `128x64` f32
accumulator no longer fits at all. Left off by default and swept.

**Register blocking past 1x1 stopped helping, and then started hurting.** It was
worth 10% when we added it. After the staging work, the full sweep separated 1x1
from 2x2 by under 1% at `64x64` (2711 vs 2691 at 2048) — and at `64x128`, 1x1 runs
**28% faster** than 2x2 (3066 vs 2391) *with the same number of accumulator
fragments per simdgroup*, so it is not register pressure. The operand-load traffic
a bigger block saves was never the binding constraint on this chip, and the
coarser fragment-to-simdgroup mapping it produces evidently costs more than the
loads it removes. The emitter's default was changed to the smallest blocking that
fits the register budget, which is the opposite of the CUDA playbook's advice and
is worth 2% at `64x64` and 28% at `64x128` over the largest-that-fits rule we
shipped first.

**Bigger blocks did not pay either.** Sharing the accumulator's storage unlocked
`128x64`, `64x128` and `64x64x64`, which cut staged elements per output element by
a third. The measured gain was ~6%, not the ~25% arithmetic intensity predicts.

### Where the remaining gap is

The measurement now says something the instruction-slot account did not: this
kernel is not short of arithmetic slots and not short of bandwidth. At 3.06 TF a
2048-cube GEMM takes 5.6ms and moves ~0.8 GB, which is ~145 GB/s against a 371
GB/s part — comfortably under. Three candidates remain, in the order they should
be tried:

1. **The epilogue and prologue.** Every dot still stages a zero accumulator into
   threadgroup memory and reads its result back per lane through the tile, and the
   per-lane block loops that do it spread only over the innermost dimension
   (§Hard parts 6). For a 64x128 tile that is a lot of poorly-distributed work
   either side of a loop that is itself only ~128 steps long.
2. **Vector (`float4`) staging.** Staging in runs amortised the address arithmetic
   but each element is still its own device load. A real `float4` load needs the
   emitter to prove the run is contiguous — an affine analysis of the staging
   subgraph, which is small and closed (`tt.make_range`, `tt.splat`, `arith.muli`,
   `arith.addi`, `tt.addptr`) and therefore tractable, plus a runtime check that
   the innermost stride is 1.
3. **Keeping the accumulator out of threadgroup memory entirely.** A 128x128 tile
   would halve the staging traffic again, and its accumulator alone is 64KB. The
   only way there is an epilogue that streams register fragments to device memory
   in panels rather than through one full-size tile — which means giving the
   epilogue its own loop over panels, the largest of the three by some way.

## Hard parts, ranked

1. ~~**`tl.dot`**, and the value model it needs.~~ Done — see §`tt.dot`. The
   contraction axis, the tile value class and the deferral of dot operands landed
   together, because none of them works without the other two.
2. **Layout conversions** in ttgir assume CUDA's blocked/mma layouts; Metal needs
   its own simdgroup-matrix layout and the convert-layout lowerings. The inference
   in `Layout.swift` is the honest, small version of this — it maps tensor
   dimensions to block dimensions, but says nothing about *which lane holds which
   element*, which is exactly what an mma layout has to pin down. It is also why
   §`tt.dot` stages through threadgroup memory rather than shuffling fragments
   between lanes: the emitter does not know where a fragment's elements live.
3. **A per-lane tensor carried across a cross-lane loop.** An `scf.for` containing
   a `tt.dot` or `tt.reduce` is lowered threadgroup-uniformly, and a loop-carried
   *scalar* survives that fine — which is why an online softmax now works
   end to end. A loop-carried **tensor** does not: it would have to be spilled to
   a threadgroup tile, and unlike a dot accumulator (which the dot updates in
   place inside its own barriers) an arbitrary carried tensor is read and written
   by ordinary per-lane code. Making that correct means a tile write at every
   `scf.yield`, a barrier discipline around each cross-lane op that also covers
   those writes, and a decision — per carried value — between spilling and the
   recomputation the emitter uses everywhere else. FlashAttention-2's `acc`
   (`BLOCK_M x HEAD_DIM`) is exactly this case; its `m_i`/`l_i` are not, they are
   the scalars that already work. This is the one thing standing between the
   current backend and the FA-2 forward pass.
4. **Atomics coverage**: Metal lacks some of CUDA's atomic dtypes (e.g. fp16 atomics).
5. **num_warps > actual concurrency**: occupancy model differs; autotune must
   re-learn its search space (feeds metalscope's roofline data back in here).
6. **Occupancy of outer block dimensions**: only the innermost dimension is spread
   across threads, so a `BLOCK_M x BLOCK_N` tile with a small `BLOCK_N` leaves most
   of the threadgroup idle. A `tt.dot`'s staging loops already spread over both
   tile dimensions — doing the same for the elementwise nest is the obvious next
   step, and would speed up a dot kernel's accumulator initialisation and
   read-back too.
7. **`tt.trans`**, which FlashAttention needs for `K^T`. On this backend it is
   cheaper than it looks: `simdgroup_load` takes a `transpose_matrix` flag, so a
   transposed dot operand is a staging-time decision rather than a data movement.

## Test strategy

`swift test` covers these layers, all on the real GPU where relevant:

1. **Parser** — op/type/attribute shapes, both Triton spellings of
   `tt.get_program_id`, rank-N tensor types, multi-result `%r:2`/`%r#0`, comments
   and `loc(...)`, and every error path (unsupported op, mixed block sizes,
   undefined values, type mismatches, generic syntax) asserting the message names
   the offender and its line.
2. **Emitter** — the exact MSL emitted for vector-add, uniform/row/lane
   partitioning, threadgroup sizing vs. `num_warps`, kernel metadata JSON, that
   `tt.expand_dims`/`tt.broadcast` emit no code at all, and that every fixture
   compiles in Metal's own front end.
3. **Casts and math** — one kernel per `arith` conversion and per `math.*` op,
   run on the GPU against a CPU reference with a per-function tolerance, plus a
   check that `precise::` is used exactly where Metal has it and `fast::` never.
4. **Control flow** — a strided `scf.for` accumulation into per-program partials
   (tensor `iter_args`), a zero-trip loop, multi-result loops in a scalar-only
   kernel, and a tensor-yielding `scf.if`.
5. **Rank 2** — tiled add and copy at sizes that divide neither block dimension,
   a padded-stride guard test proving masks protect the gaps between rows, and
   assertions that row-uniform work really is hoisted out of the inner loop.
6. **Reductions** — add/max/min over 1-D blocks, swept across `num_warps` 1..32 so
   the cross-simdgroup path is exercised; row-wise reduction over a rank-2 tile;
   fused softmax at four shapes against a CPU reference, plus a rows-sum-to-one
   check on inputs large enough to overflow a naive `exp`; and an **online
   softmax** that streams the row through an `scf.for` with both reductions inside
   the loop body, at row lengths that are not multiples of the block.
7. **`tt.dot`** — a single-tile product at six shapes including `5x3x7` and
   `12x20x12` (neither a multiple of 8, so the zero-padded edge fragments carry
   the result); the matmul tutorial against a CPU reference at six shapes
   including `129x257x65`; f16 operands with an f32 accumulator; the same kernel
   at `num_warps` 1..32; a sentinel test proving masked stores leave the padding
   between rows alone; assertions on the emitted tile sizes, barrier placement,
   accumulator residency and pointer strength reduction; assertions that the
   output fragments stay in simdgroup registers for the whole K loop, that an
   explicit register blocking overrides the emitter's choice, that a resident
   accumulator's tile doubles as its operands' staging arena with a barrier
   between the last read and the first write of the shared storage, and that a
   stand-alone dot does *not* share it; and the error paths —
   a two-operand `tt.dot`, a mismatched contraction, an integer dot, a dot inside
   an `scf.if`, a contraction-space value used outside the dot, an unrelated
   tensor carried across a dot loop, and a block shape that overruns threadgroup
   memory.
8. **End-to-end** — copy / add / mul / scale-bias / integer kernels against CPU
   references at non-multiple-of-BLOCK sizes; a guard-region test proving masked
   stores don't write out of bounds; the same kernel at `num_warps` 1..32.
9. **C ABI** — the whole spine (emit -> compile -> load -> alloc -> write -> launch
   -> read -> release) through `tm_*` only, plus handle-validity, bounds and
   leak checks.

`MatmulBenchmark` is a tenth suite that stays off unless `TM_BENCH=1` is set: it
measures a machine, not a contract. The measurement itself lives in the
`TritonMetalBench` library and is shared with the **`tmbench`** executable
(`swift run -c release tmbench`), because XCTest does not exist on a machine with
only the command-line tools installed — which is where the numbers in
§Matmul throughput were taken. `tmbench` sweeps block shapes, `num_warps`,
register blocking, tile padding and double buffering; pins a single configuration
with `--config M,N,K,W[,RM,RN[,U[,P[,DB]]]]`; prints a kernel with `--emit`; and
checks its winners against a CPU reference at `129x257x65` and other awkward
shapes before reporting them.

Next: port Triton's own backend conformance tests (`test_core.py` subset) against
numpy references. FlashAttention-2 forward remains the integration milestone
kernel; §Hard parts 3 and 7 are what it is still waiting on.

## Milestones

1. ~~`is_active()` + device query working.~~ Done; pinning Triton still open.
2. ~~Trivial kernels end-to-end: copy, add, masked load/store.~~ Done.
3. ~~Casts and `math.*` unary ops.~~ Done — `precise::` namespace, fast math off.
4. ~~`scf.for`/`scf.if` with `iter_args` and results.~~ Done.
5. ~~Rank-2 tensors, `tt.expand_dims`/`tt.broadcast`, per-dimension block sizes.~~
   Done — see §Execution model for the layout inference this forced.
6. ~~`tt.reduce` -> simdgroup reductions; fused softmax end to end.~~ Done.
7. ~~`tt.dot` on simdgroup matrices, then the matmul tutorial kernel.~~ Correct —
   f32 and f16-in/f32-out, at sizes divisible by nothing in particular. The
   **>50% of MPS** half of this milestone is now met: ~50% at 1024, 2048 and 4096
   on an M1 Max, up from ~33%, via register blocking, register-resident
   accumulators, storage sharing between the accumulator and its operand tiles,
   and cheaper staging. §Matmul throughput has the per-change attribution, what
   did *not* help on Apple silicon, and what is left before the 60–80% band.
   `tt.atomic_*` and `tt.trans` still open.
8. ~~Reductions inside an `scf.for`.~~ Done for loop-carried **scalars** — an
   online softmax streams its row through the loop with both reductions inside it.
   Loop-carried *tensors* remain refused (§Hard parts 3).
9. **Next up**: FlashAttention-2 forward. Needs §Hard parts 3 (spilling a
   loop-carried tensor accumulator) and §Hard parts 7 (`tt.trans`, or a
   transposing `simdgroup_load`); then benchmark vs. MLX fast attention.
