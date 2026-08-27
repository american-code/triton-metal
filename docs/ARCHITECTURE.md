# triton-metal architecture

## Language policy

Swift core (`Sources/TritonMetalCore`), C ABI (`tm_*` in libtritonmetal.dylib),
Python only as the unavoidable Triton-plugin shim (`python/triton_metal` — ctypes
bindings, no logic). Python here is justified solely because Triton's backend
discovery imports a Python module and rebuilding Triton's frontend is a multi-month
project. Any new functionality goes in the Swift core and gets a `tm_*` export.

## Compatibility

**Pinned: `triton-lang/triton` v3.7.1** (commit `f797708c`), the release
`pytorch/pytorch` release/2.12 and release/2.13 pin in `.ci/docker/triton_version.txt`
(main is on 3.8.0, which has no tag yet). Everything below — the stage dict, the
vendored `BaseBackend`/`DriverBase` signatures in `python/plugin/backend/`, the
launcher's argument order — is that tag's interface, read off the tag rather than
assumed.

### Building it

Triton publishes **no macOS wheel**: `pip install triton` fails on macOS arm64 at
resolution. The backend is therefore reached by building Triton from source with
this repository passed as an out-of-tree plugin. On studio-b (M1 Max, 10 cores,
Command Line Tools only — no Xcode) that is **≈9 minutes** and needs no CUDA
toolchain:

```
git clone https://github.com/triton-lang/triton.git && cd triton && git checkout v3.7.1
export TRITON_PLUGIN_DIRS=/path/to/triton-metal/python/plugin
export TRITON_BUILD_PROTON=OFF          # no CUPTI / roctracer
export TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF"
export TRITON_PTXAS_PATH=/nonexistent   # ... and the other six TRITON_CUDA*/CUPTI vars:
                                        # download_and_copy() skips a download whose
                                        # destination variable is already set, which keeps
                                        # ~1 GB of Linux CUDA redistributables out of a
                                        # macOS build (setup.py maps Darwin -> "linux")
pip install -e .                        # or: python setup.py bdist_wheel
```

**No patch to Triton is required.** Two things were tried and are worth recording
as dead ends: building with `TRITON_CODEGEN_BACKENDS=""` (plugin only, no NVIDIA
or AMD) fails, because 3.7.1's *core* sources include the in-tree backends'
tablegen'd headers — `lib/Conversion/TritonInstrumentToLLVM/InstrumentationToLLVM.cpp`
includes `nvidia/include/Dialect/NVGPU/IR/Dialect.h.inc`, `python/src/gluon_ir.cc`
includes the AMD dialect, and `examples/plugins` hard-codes
`TritonNVIDIAGPUConversionPassIncGen`. The NVIDIA and AMD backends must be built;
only their *runtime* dependencies can be skipped. The prebuilt LLVM does exist for
this platform (`llvm-1f126a6d-macos-arm64`), which is what makes the build short.

`setup.py bdist_wheel` produces
`triton-3.7.1+gitf797708c-cp311-cp311-macosx_26_0_arm64.whl`, **69.4 MB**, whose
`entry_points.txt` carries `metal = triton.backends.metal` next to `amd` and
`nvidia`, and which contains `triton/backends/metal/{compiler,driver}.py`.

### How the plugin attaches

`TRITON_PLUGIN_DIRS` is a semicolon-separated list of directories; each must
contain `backend/name.conf` (the backend name — ours says `metal`),
`backend/compiler.py` and `backend/driver.py`, plus a `CMakeLists.txt`, because
the plugin is also a CMake subproject. That last part is not optional: CMake joins
the plugin names into `TRITON_BACKENDS_TUPLE`, and `python/src/main.cc` expands
that tuple into one `void init_triton_<name>(pybind11::module &&)` **declaration
and call** per backend. A plugin that provides no such symbol does not link. So
`python/plugin/metal.cc` exists and is the entire C++ surface of this project:
a registration stub that reports `linked()`. Everything else is Swift.

Discovery at import time is by entry point (`triton.backends` group), which
`setup.py` generates from the same list; `triton/backends/__init__.py` then
imports `<pkg>.compiler` and `<pkg>.driver` and requires **exactly one** concrete
`BaseBackend` and one concrete `DriverBase` in each.

### What the adapter does and does not do

| Piece | 3.7.1 expects | Ours |
| --- | --- | --- |
| stages | ordered dict `ext -> f(module, metadata)`, last returns `bytes` | `ttir` (Triton's own passes) then `msl` (`tm_emit_msl` + `tm_kernel_info`) |
| `binary_ext` | names the cached payload | `msl` — Metal compiles from source in-process (`MTLDevice.makeLibrary(source:)`); `xcrun metal` is absent from Command Line Tools, so there are no offline `.metallib` bytes to cache |
| driver base | `GPUDriver` | `DriverBase` — `GPUDriver.__init__` reaches straight into `torch.cuda` |
| target | `GPUTarget(backend, arch, warp_size)` | `("metal", <device name>, 32)`; a simdgroup is 32 lanes |
| launcher | `launcher_cls(src, metadata)`, called as `(gx, gy, gz, stream, function, packed_metadata, launch_metadata, enter_hook, exit_hook, *args)` | marshals to `tm_launch`; the kind/index table comes from `tm_kernel_info`, computed in Swift |
| tensors | any object with `data_ptr()` and `dtype` (`python/src/specialize.cc`) | `triton_metal.buffer.MetalBuffer` over `tm_alloc_buffer`; a CPU torch tensor is **copied** in, since `makeBuffer(bytesNoCopy:)` needs page-aligned memory |

There is no `ttgir` stage: Triton's TritonGPU passes lower toward LLVM with a
target profile this backend does not have, and the Swift emitter does its own
layout inference from `ttir` (§Execution model). That is a deliberate fork point,
not an omission — it is also why `num_warps` reaches the emitter as a simdgroup
count rather than through a `ttg` encoding.

### What real Triton IR looked like, versus the fixtures

The fixtures in `Tests/` were hand-written from Triton's documentation. Real
3.7.1 output differs in exactly two ways, one cosmetic and one that mattered:

1. **Locations everywhere.** Triton's `MLIRModule.__str__` prints with debug info
   on, so the module is bracketed by `#loc` alias definitions *before and after*
   it, every operation carries a trailing `loc(#locN)`, and — the part the parser
   had not seen — every function argument does too:
   `%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x_ptr"(#loc))`.
   The parser already skipped trailing locations on operations, module and
   function; it now skips them on function arguments as well (`Parser.swift`,
   one call to `skipTrailingLocation`). That was the only parse failure in the
   entire exercise.
2. **Named SSA values.** Real output is `%pid`, `%offsets_1`, `%accumulator_31`,
   not `%0`, `%1`, `%arg0`. The lexer already accepted those; nothing keys on
   argument *names*, only positions, so nothing broke.

Two *emitter* gaps did show up, both in the matmul tutorial, and both are cases
where Triton's canonicalizer rewrites a source-level `ptrs += BLOCK_K * stride`
into a shape the `tt.dot` pointer-advance check had not anticipated:

* the advance folded into a **dense splat constant** (`dense<16> : tensor<64x16xi32>`)
  rather than a `tt.splat` — uniform by construction, now accepted;
* the advance recomputed **inside** the loop body as a scalar product of
  loop-invariant operands (`arith.muli %stride_bk, %c16_i32` then `tt.splat`) —
  now folded into an inline uniform expression (`uniformExpression` in
  `MSLEmitter.swift`), which also covers `+ - * / % << >>` over such operands.

One real constraint survives, and the matmul tutorial hits it: the emitter
identifies a block axis **by its extent**, so `BLOCK_M == BLOCK_N` (or either
equal to `BLOCK_K`) makes a row index indistinguishable from a column index. The
kernel is refused by name and line rather than mis-compiled. Pairwise-distinct
block sizes work: `(128, 64, 32)` runs and matches numpy; `(64, 64, 32)` is
rejected. Lifting this means carrying axis identity independently of extent
through `LayoutInference` — the next real piece of work in the layout model.

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

Every tensor in a kernel is a slice of one **block**: an index space whose
dimensions are called **axes**. A value is emitted inside a nest of loops over
exactly the axes it varies along — its **path** — and the nests of different
values share whatever prefix they have in common. The last axis of a nest is
distributed over the threadgroup's threads; the axes outside it are walked by
every thread:

```metal
for (uint tm_i0 = 0u; tm_i0 < BLOCK_M; ++tm_i0) {                 // uniform
    for (uint tm_i1 = tm_thread_id.x; tm_i1 < BLOCK_N;            // distributed
         tm_i1 += tm_threadgroup_size.x) { ... }
}
```

Nests form a **tree**, not a chain, and that is what FlashAttention-2 forced. Its
`p` spans `(BLOCK_M, BLOCK_N)` and its `acc` spans `(BLOCK_M, HEAD_DIM)`; neither
contains the other, and walking both under one three-deep nest would recompute
each of them `HEAD_DIM` or `BLOCK_N` times over. So `p` lives under `(M, N)`,
`acc` lives under `(M, HEAD_DIM)`, and moving between them closes the inner loop
and opens the other while the shared `M` loop stays put — which is also what
keeps a value defined in the `M` loop (an online softmax's running maximum) in
scope across both.

Whether an axis is uniform or distributed is a property of the *kernel*, not of
the statement: an axis that any value nests another axis inside is uniform
everywhere. That is what lets the `M` loop be shared rather than reopened.

A value's path is its own varying axes, extended with every axis a consumer nests
outside them, so that its variable is still in scope where it is read — Triton's
`offs_n < N` mask spans only the column axis but is consumed inside the row loop.
So `offs_m[:, None] * stride` (rank 2, but constant along the columns) is still
computed once per row, `tt.get_program_id` once per program, and only
column-varying work lands in the innermost loop. A statement whose own nest ends
on a *uniform* axis runs redundantly in every thread, so a `tt.store` there is
guarded — with `tm_thread_id.x == 0u`, or with `tm_thread_id.x % lanes == 0u`
when the threadgroup is split two-dimensionally (§Cross-lane regions).

Which block axis a value spans is **inferred**, not assumed: `tl.arange(0, M)`
is rank 1 even inside a rank-2 kernel, and only the `tt.expand_dims` that consumes
it says whether it indexes rows or columns. `Layout.swift` gives every dimension
of every tensor its own axis *variable* and **unifies** them: elementwise ops
dimension-wise, `tt.expand_dims`/`tt.reduce` around the inserted or dropped
dimension, `tt.trans` through its permutation, and `tt.dot` by pinning `a` to
(M, K), `b` to (K, N) and the accumulator/result to (M, N). Two full-rank tensors
that no chain relates are then merged onto one index space, which is what the old
identity seeding did for every tensor and is still what an ordinary elementwise
kernel wants. A rank-deficient value that never reaches a full-rank one is an
error, not a guess.

Unification rather than seeding is what makes FlashAttention-2 expressible at
all: its two dots share axes *crosswise* — the first contracts over the head
dimension its accumulator iterates, the second over the key block its softmax
iterates — which no fixed (M, N, fresh-K) assignment can describe. Two rules earn
their keep, and both were found by running real IR through it. A **size-1
dimension carries no axis identity**: the matmul tutorial broadcasts one
`tensor<BLOCK_M x 1 x i1>` row mask into both a `BLOCK_M x BLOCK_K` and a
`BLOCK_M x BLOCK_N` tensor, and unifying its second dimension with both would
declare the contraction axis and the column axis to be the same one. And **two
varying dimensions of one value must land on two different axes**: they collapse
when a single `tl.arange` is expanded once into a row index and once into a
column index — which Triton's CSE hands you whenever `BLOCK_M == BLOCK_N` — and
the value would then be a diagonal of itself. Refused by name.

The axes are then **ordered**: a materialised value's own dimension order says
which of its axes is outside which, and the order is a topological sort of those
constraints. Deferred `tt.dot` operands do not constrain it — they are staged
over their own two dimensions in whichever order they are written, and `K^T` and
`V` disagree about the head dimension while both are right. A cycle among
*materialised* values is refused by name, and a materialised `tt.trans` is how to
make one.

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

Four shapes of `iter_args` are lowered:

* **scalars**, which are uniform anyway (a reduction's result is broadcast to
  every thread) and stay ordinary variables — this is what a rank-1 online
  softmax's running maximum and running sum are;
* a **`tt.dot` accumulator the loop yields directly**, which becomes a
  threadgroup tile *and* a block of simdgroup registers per simdgroup: the tile is
  initialised before the loop, read into the registers once, updated only in
  registers for the whole loop, and written back to the tile after it for the
  per-lane epilogue (§`tt.dot`). This is the cheap case and the one Triton's
  matmul tutorial writes;
* a **contraction-space pointer** (`a_ptrs += BLOCK_K * stride_ak`), which is not
  carried at all: it is strength-reduced back to `init + trip_count * delta` so
  that the dot can rebuild it from scratch (see §`tt.dot`);
* any other **per-lane tensor**, which is *spilled* to a threadgroup tile.

### Spilling a carried tensor

The spill is what FlashAttention-2 needed, and it has two shapes.

The **general** one: the value gets a tile and a **shadow** tile. It is seeded
before the loop (a flat pass, so the fragment padding is covered too — no
per-lane nest walks it, and a `simdgroup_load` of an edge fragment still reads
it), read as an ordinary tile everywhere inside the loop, and updated once per
iteration. The new value is written into the shadow *where it is defined*, not at
the `scf.yield`, for two reasons that both bite: by the time the yield is reached
the nest that computed it has been closed, and a `tt.reduce` result cannot be
rebuilt in a reopened one; and the *old* value is usually still being read at that
point — FA-2's `alpha = exp2(m_i - m_ij)` reads `m_i` after `m_ij` is known. The
shadow is copied over the tile once, at the end of the body, between two barriers:
the first orders every read of the old value before the overwrite, the second
makes the new value visible to the threads that did not write it. `m_i` and `l_i`
are `BLOCK_M` floats each and take this path.

The **special** one, which is worth having because it costs nothing and the
general one costs a full `BLOCK_M x HEAD_DIM` copy per iteration: when the loop
yields a `tt.dot`'s **result** but the dot's accumulator is a per-lane function of
the carried value rather than the carried value itself —
`acc = tt.dot(p, v, acc * alpha[:, None])`, which is exactly FA-2's rescale — the
carried tile *is* the dot's accumulator tile, and the rescale is lowered as the
staging pass that fills it. Each staging thread reads and writes the same element,
so the read-modify-write needs no ordering that staging does not already have,
there is no copy at all, and the tile exists once. What it gives up is register
residency: an accumulator per-lane code has to touch every iteration cannot stay
in `simdgroup_float8x8`s across the loop.

Writes into a spilled tile are guarded by whoever owns the element. Where the
tile's innermost axis is the distributed one, every element has exactly one owner
and no guard is needed. Where the nest ends on a uniform axis, every thread
computes the *same* value for the same slot and writes it redundantly — which is
what the emitter already does at that depth for everything else.

A reduction result a `tt.dot`'s staging loops need is spilled the same way, for
the same reason: staging rebuilds its operands from their producers, and a
`tt.reduce` is the one thing that cannot be rebuilt there — the fold has already
happened. FA-2's `p = exp2(qk * scale - m_ij)` is the second dot's left operand
and `m_ij` is a row reduction, so the row maximum goes into a `BLOCK_M`-float
array where it is computed and is read out of it in the staging nest, behind the
barrier every dot with spilled tiles now emits in front of its staging.

An `scf.if` is per-lane by construction and still refuses both cross-lane ops.

### Reductions across a row, not across the threadgroup

A `tt.reduce` whose operand nest has an axis outside the reduced one hands each
row a **group of lanes** rather than the whole threadgroup: `tid / lanes` walks
the rows and `tid % lanes` walks the reduced axis, with `lanes` capped at the
32-wide simdgroup so a row's lanes always share one. The fold is then
`simd_shuffle_down` within the group and a `simd_shuffle` to broadcast the total
back across it — no threadgroup memory, no barrier, and every row folding at once.

A rank-1 reduction has no outer axis to hand a lane group to, so its row *is* the
threadgroup and it still folds through `threadgroup` scratch and two barriers.

This was worth **3.9x** on the attention kernel (§Attention throughput), which is
where the cost of the old shape became obvious: with `BLOCK_M` rows and two
reductions per key block, an online softmax over blocks was running
`2 * BLOCK_M` threadgroup-wide folds per iteration, one after another, with 128
threads cooperating on 32 values at a time.

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
uint bn = tm_simd_group % BN;                            // shared by every wave
uint bm0 = 0u, bm1 = 1u, ...;                            // one row block per wave
simdgroup_float8x8 c_0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c_1 = ...;

for (int k = 0; k < K_TILES; ++k) {                      // the scf.for over K
    <stage A into tm_dot_a, over (M, K)>                 // spread over both dims
    <stage B into tm_dot_b, over (K, N)>                 // float4 runs where it can
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0u; s < FK; ++s) {
        simdgroup_load(b_0, tm_dot_b + s * ... + bn * 8u, ...);   // once per step
        simdgroup_load(a_0, tm_dot_a + bm0 * ... + s * 8u, ...);  // once per wave
        simdgroup_multiply_accumulate(c_0, a_0, b_0, c_0);
        simdgroup_load(a_1, ...);
        simdgroup_multiply_accumulate(c_1, a_1, b_0, c_1);        // same b_0
        ...
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_store(c_0, tm_dot_c + ...); ...                // once, for the epilogue
threadgroup_barrier(mem_flags::mem_threadgroup);
<read tm_dot_c back per lane, spread over (M, N)>
```

Six things in that shape are worth naming, because each was measured
independently (§Matmul throughput).

**A zero accumulator has no prologue.** Triton seeds every matmul accumulator with
`arith.constant dense<0.0>`, and the emitter detects exactly that: the fragments
are born zero in registers (`make_filled_simdgroup_matrix`) instead of being
written into the tile by a per-lane pass, read back one `simdgroup_load` per
fragment, and fenced by a barrier on each side. A non-zero seed still takes the
long way, because the value has to reach the fragments somehow.

**Every wave of a simdgroup shares a column of the block grid**, so it shares the
B fragment of a contraction step. That follows from the block grid's width
dividing the simdgroup count, which is the usual case; the emitter then spells the
column index once rather than once per wave, and deduplicates operand loads by
address. See §Matmul throughput for what it was worth (a lot).

**Staged runs are vector loads when they can be.** A thread's run of four
consecutive columns is one `float4` (or `half4`) device load and four tile writes,
skipping the run's per-element address arithmetic entirely — guarded by a runtime
test that the run is contiguous, aligned, and inside the mask, with the scalar run
emitted beside it for when it is not. Contiguity is a runtime property of the
kernel's stride arguments, but *affinity* — that the address is `p + c * stride`
at all, so that two endpoints characterise the run — is proved statically by
walking the staging subgraph, which is small and closed. The mask is only checked
at the run's last column, which is sound because the same walk proves it monotone
in the column index. Removing the *alignment* half of that test does not produce
wrong answers on an M1 — these parts evidently service an unaligned vector load —
but MSL does not promise that, so it stays; removing the stride or the mask half is
caught by the test suite immediately.

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
step's staging need not wait on the previous step's arithmetic. A third,
`dotVectorStaging`, is **on** by default and exists to be turned off — it is worth
about 20% and the sweep measures it both ways.

### Atomics

`tt.atomic_rmw` and `tt.atomic_cas` lower to Metal's `device` atomics: the
Triton pointer is already a `device T *`, so the lowering is a cast of the
address and a call. What each spelling costs, and what is missing, was
established by compiling every candidate against Metal's own front end on an
**M1 Pro (Metal 3, macOS 26.5)** rather than read off the specification:

| Triton | f32 | i32 |
| --- | --- | --- |
| `add` / `fadd` | `atomic_fetch_add_explicit` on `atomic_float` | `atomic_fetch_add_explicit` on `atomic_int` |
| `max` / `min` | **no such intrinsic** — `tm_atomic_fmax`/`fmin`, a compare-exchange loop | `atomic_fetch_max_explicit` / `min` on `atomic_int` |
| `umax` / `umin` | — (integer only) | the same on `atomic_uint`, operand and result cast |
| `and` / `or` / `xor` | — (integer only) | `atomic_fetch_and/or/xor_explicit` on `atomic_int` |
| `exch` | `atomic_exchange_explicit` | `atomic_exchange_explicit` |
| `tt.atomic_cas` | `tm_atomic_cas` | `tm_atomic_cas` |

Four gaps, each refused by name rather than lowered to something that is not
atomic:

* **No float `atomic_fetch_max_explicit`.** The template exists but
  `_valid_fetch_max_type` is not satisfied for `device float *`, so the call does
  not resolve. Float max and min therefore go through a compare-exchange loop on
  the bit pattern (`tm_atomic_fmax`), which compares *as floats* so that negative
  values order correctly. The cost is honest and bounded: an extra
  `atomic_load` before the loop, and one retry per thread that loses a race on
  the same address — an uncontended max is a load plus one CAS, about twice a
  native fetch-max, and contention degrades linearly in the number of contending
  threads rather than in whatever the hardware does for a native one.
* **No 16-bit or 64-bit atomics.** `atomic<half>` does not exist and
  `atomic_fetch_add_explicit` has no `unsigned long` overload, so `f16` and `i64`
  atomics are refused.
* **One memory order.** MSL declares `memory_order_relaxed` and nothing else —
  `memory_order_acq_rel` is not even a declared identifier, and
  `memory_order_seq_cst` does not resolve. Triton's `sem` (`relaxed`/`acquire`/
  `release`/`acq_rel`) and `scope` (`cta`/`gpu`/`sys`) are parsed and dropped.
  The operation itself is atomic; **nothing else about it is ordered**, so a
  kernel that relies on an atomic's release edge to publish ordinary stores is
  not correctly lowered here.
* **No strong compare-exchange.** Metal has only
  `atomic_compare_exchange_weak_explicit`, which may fail spuriously, so
  `tt.atomic_cas` — which fails only on a value mismatch — is a small loop that
  retries exactly when the observed value still equals the comparand
  (`tm_atomic_cas`).

Two things the *lowering* has to do that a `tt.store` does not.

**A uniform nest performs the atomic once.** Only a nest's innermost loop is
distributed over threads, so a statement whose nest ends on a uniform axis runs
in every thread. For a store that writes the same bytes twice; for an atomic add
it would add the same contribution `threads_per_threadgroup` times. The same
single-writer guard a store gets is therefore a *correctness* requirement here,
and it has a consequence: only one thread performs the access, so only that
thread can be told what was there before. An atomic in such a nest **whose
result is used** is refused by name.

**A masked lane performs no access at all.** `mask ? atomic : nothing`, with the
result reading back zero — not a masked store of the result of an unconditional
atomic, which would corrupt the accumulator.

**Determinism.** Float atomics reorder, and the backend does not pretend
otherwise: the order in which threadgroups reach an address is not defined, and
floating-point addition is not associative, so a `tl.atomic_add` reduction is
run-to-run non-deterministic. The test suite measures the spread rather than
hiding it — five runs over contributions spanning `1e6` and `1e-3` are asserted
to agree within the standard recursive-summation bound
`(k-1) * 2^-24 * sum|x_i|`, and to agree with an in-order CPU sum within the same
bound. Where a test needs equality it uses contributions that are exactly
representable, so that any order gives the same total and a single lost update is
visible.

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
| `tt.trans` | Permutes a tensor's dimensions, with or without the `{order = array<i32: ...>}` attribute (2-D defaults to the reversal). A relabelling of which block axis each dimension indexes — **no code emitted at all**, because the two values are the same per-lane scalar at the same block *point*. What it costs is a nesting constraint, so a `tt.trans` whose result is materialised is refused by `LayoutInference` (§Execution model); a `tt.dot` operand is staged over its own tile's axes and is free. |
| `tt.broadcast` | Expands size-1 dimensions. A relabelling — no code emitted. |
| `tt.addptr` | Pointer + integer offset, scalar or tensor, any rank. |
| `tt.load` | `ptr[, mask[, other]]`, any rank. Masked loads become `mask ? *p : other` (`other` defaults to zero). Cache/eviction attributes are parsed and ignored. |
| `tt.store` | `ptr, value[, mask]` -> `if (mask) { *p = v; }`, any rank. |
| `tt.atomic_rmw` | `<kind>, <sem>, <scope>, %ptr, %val[, %mask]`, any rank, on `f32` and `i32` device pointers. Kinds: `add`, `fadd`, `and`, `or`, `xor`, `max`, `min`, `umax`, `umin`, `exch`. Returns the old value. `sem` and `scope` are parsed and ignored — Metal has one memory order (§Atomics). |
| `tt.atomic_cas` | `<sem>, <scope>, %ptr, %cmp, %val`, `f32` and `i32`. Strong, via a retry loop around Metal's weak compare-exchange (§Atomics). |
| `tt.reduce` | Single-operand, **last axis only** (the distributed one), combiner `add`/`max`/`min`. With an axis outside the reduced one it folds with `simd_shuffle_down` across the row's lane group and broadcasts back with `simd_shuffle`; a rank-1 reduction folds within each simdgroup and then across them through threadgroup memory (§Cross-lane regions). Parsed in MLIR's generic form, which is how Triton prints ops with regions. Allowed inside an `scf.for` (see §Cross-lane regions), not inside an `scf.if`. |
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
| `scf.for` | `iter_args` and results, scalar or tensor, including the multi-result `%r:2 = ... -> (T, U)` / `%r#0` spelling. Results alias the carried variables. A body containing a `tt.dot` or `tt.reduce` hoists the whole loop to a threadgroup-uniform level; a per-lane tensor carried across such a loop is spilled to a threadgroup tile — see §Cross-lane regions. |
| `scf.if` | With or without results; results need an `else` region. |
| `scf.yield`, `tt.reduce.return` | Region terminators. |

Not supported (each fails with the op name): `tt.cat`,
`tt.call`, `tt.histogram`, `tt.scan`, multi-operand `tt.reduce` (argmax/argmin),
`f16`/`i64` atomics and an atomic whose returned old value is read in a
threadgroup-uniform nest (§Atomics),
reductions over a non-innermost axis, cross-lane operations inside an `scf.if`, a
carried tensor spanning more than two block axes, a `tt.trans` whose result is
materialised, a `tl.arange` expanded into both a row and a column index of the
same value, and MLIR's generic op syntax for anything except `tt.reduce` — dump
the pretty form instead.

A `tt.dot` whose operand is another `tt.dot`'s result **is** supported now: the
first dot's result is already a tile, and the second dot's staging loops read it
rather than rebuilding it, as long as the reading nest walks the axes the tile is
indexed by. That is `P = softmax(Q K^T)` feeding `P V`.

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

Two machines, because the ratio depends on both. **studio-b** is a Mac Studio
**M1 Max** (measured peaks 6.2 TFLOP/s f32, 371.5 GB/s) and is the machine to
believe: it has no Xcode, which is why `tmbench` exists at all. The **M1 Pro** is
a laptop and its MPS readings move with its thermal state.

### Apple M1 Max (studio-b)

| size | best configuration | round 1 | round 2 | MPS | round 1 | round 2 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | `64x64x16`, `num_warps=8` | 2.01 TF | **3.21 TF** | ~1.85 TF | 92% | (see below) |
| 1024 | `64x64x16`, `num_warps=8` | 2.75 TF | **4.33 TF** | ~5.70 TF | 50% | **76%** |
| 2048 | `64x64x16`, `num_warps=8` | 3.06 TF | **4.66 TF** | ~6.10 TF | 50% | **76%** |
| 4096 | `64x64x16`, `num_warps=8` | 3.21 TF | **4.64 TF** | ~6.10 TF | 52% | **76%** |

### Apple M1 Pro (laptop)

| size | best configuration | round 1 | round 2 | MPS | round 1 | round 2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | `64x64x16`, `num_warps=8` | 1.57 TF | **2.27 TF** | ~2.77 TF | 50% | **82%** |
| 2048 | `64x64x16`, `num_warps=8` | 1.69 TF | **2.33 TF** | ~2.83 TF | 50% | **82%** |

**~1.5x again on top of the previous round's 1.55x, and ~50% of MPS becomes 76%**
on the machine to believe — 2.3x over the original kernel. That is inside the
62–82% band published Triton reaches against cuBLAS on its native target
(docs/COMPARISON.md), and the 60% this round aimed at is cleared at every size
where the measurement means anything. The best shape moved: `64x64x16` at
`num_warps=8` now wins at every size, where `64x128x16` at `num_warps=16` used to
win the large ones.

Two caveats on the numbers. The **512 rows** are not evidence of anything: that
GEMM is under a millisecond and MPS's own timing swings by 1.5x run to run at that
size (this round it read 1.85 TF, last round 2.29 TF), so no ratio is quoted there.
And the **laptop's** MPS readings move with its thermal state — the same MPS
configuration at 2048 read 3.38 TF earlier in the same session, which would make
that row 69% rather than 82%. The M1 Max is the machine to believe.

One measurement lesson, learned the hard way: the **first configuration measured at
a size runs cold**, and can read 20% low — one calibration dispatch is not always
enough warm-up. Every number here is from a run that measured the configuration at
least twice; run-to-run variation after that is about ±2%, which is worth keeping
in mind against the per-change deltas below.

Reproduce with `.build/release/tmbench --sweep full`, or
`TM_BENCH=1 swift test --filter MatmulBenchmark` where XCTest exists.

### What each change was worth

Measured one at a time on studio-b at 2048, each on top of the one before it, by
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
| **second round, on top of the above** | | |
| + zero accumulator born in registers (no prologue at all) | 3025 | +0.1% |
| + 2-D-distributed epilogue (all 512 threads, not 128) | 3094 | +2% |
| + shared-column wave mapping, so identical operand loads collapse | 3551 | **+15%** |
| + `float4` staging runs behind a runtime contiguity/alignment check | 4271 | **+20%** |
| + two fragment rows per simdgroup where the score ties (at `64x64`) | 4640 | **+8%** |

The last row is at a different shape from the rest, because it is a change to which
shape wins: with two fragments along M the emitter's own choice at `64x64x16` /
`num_warps=8` goes from 4289 to 4640 GFLOP/s, overtaking the `64x128` column this
table is otherwise measured in (4341). Vector staging was re-measured warm and one
axis at a time to be sure of it: 4341 and 4322 GFLOP/s with, 3628 and 3625
without.

The last row of the first round is the odd one out: padding helps `64x64` slightly
and is a net loss at `64x128`, because the slack it adds is what stops that block
shape fitting. It is off by default and swept.

The two rows that paid are both about *not repeating work*, and neither is a CUDA
technique:

**The shared-column wave mapping.** With one output fragment per simdgroup per
wave, a `64x128` accumulator on 16 simdgroups is an 8x16 block grid covered in 8
waves, and the emitter used to hand wave `w` of simdgroup `s` the flat block
`s + 16w`, then recover its grid coordinates with a divide and a modulo. Those
coordinates are not independent: `bN` divides the simdgroup count here, so
`(s + wS) % bN == s % bN` — *every wave of a simdgroup sits in the same column of
the grid*, and therefore wants the same B fragment. Emitting the column index once
instead of once per wave makes that textually visible, at which point deduplicating
loads by address turns a step's 16 `simdgroup_load`s into 9 (8 A fragments and one
B), and the row index of each wave collapses to a compile-time constant. Same
mapping, same arithmetic, 44% fewer threadgroup reads.

**Vector staging.** A run of four consecutive columns is one `float4` device load
rather than four scalar ones — but the win is not mostly the loads. The fast path
skips the *per-element address arithmetic* as well, because it is one branch around
the whole run rather than a select inside it: an early version that kept the
element loop and only replaced the load measured +2%, and the same guard wrapped
around the whole run measured +20%. Getting there needed the guard to be cheap:
one address evaluation per run instead of four, the mask checked only at the run's
last column (sound because the affine walk proves it monotone), and the stride and
alignment tests folded in beside them. A guard that evaluated both endpoints of the
run in full measured **2496 GFLOP/s** — a 30% *loss* — which is the whole lesson in
one number.

### What did not help, and why that is interesting

**Scope: everything in this section is an M1-generation result**, measured on an
M1 Max and an M1 Pro. The reasons given for the inversions are architectural — no
`cp.async` copy engine, occupancy rather than a dedicated copy path as the way
latency is hidden, whole 8x8 fragments moving through threadgroup memory rather
than being distributed over named lanes — and later Apple GPUs may have changed
them; the M3 and M4 memory hierarchy (dynamic caching) is the obvious candidate.
None of these defaults should be assumed correct on M2, M3 or M4 silicon until
`tmbench --sweep full` has been re-run there. The same caveat applies to
§Attention throughput's fused-versus-composite crossover.

**Double buffering is still a wash, re-measured.** With two operand buffers a
contraction step stages into the half the previous step is not reading, so the
trailing barrier can go and the staging need not wait on the arithmetic. On studio-b
at 2048 this measured 2863 GFLOP/s against 2900 without it — inside the noise, and
if anything negative. Re-measured twice in the second round: after the mapping
change, 3633 against 3627, still inside the noise; after vector staging, **3205
against 4328** — a 26% loss. Ping-ponging makes each operand tile's base a
*variable* pointer, and the vector fast path's four tile writes are exactly the
code that wants it to be a constant. What was free to keep is now a real cost, and
it stays off. Metal has no `cp.async`: the prefetch is issued by the same
threads, into the same issue slots, so nothing is actually asynchronous. What CUDA
buys with a dedicated copy engine, Apple's GPU buys with occupancy instead — and
doubling the operand tiles *costs* occupancy, which is roughly what cancels it. It
is kept as `dotDoubleBuffer`, off by default, because it is cheap to re-measure on
a future chip.

**Bank-conflict padding is marginal and not free.** Four elements of slack per tile
row was worth ~2–3% at `64x64`, and the slack comes out of the same 32KB budget
that decides how large a block shape can be — with padding on, a `128x64` f32
accumulator no longer fits at all. Left off by default and swept.

**Register blocking past 1x1 stopped helping, and then started hurting** — and the
second round, which changed the tradeoff, made it worse rather than better. It was
worth 10% when we added it. After the staging work, the full sweep separated 1x1
from 2x2 by under 1% at `64x64` (2711 vs 2691 at 2048) — and at `64x128`, 1x1 runs
**28% faster** than 2x2 (3066 vs 2391) *with the same number of accumulator
fragments per simdgroup*, so it is not register pressure. Deduplicating the operand
loads is the reason to re-measure: it gives 1x1 blocking most of the load traffic a
bigger block was supposed to save, so the full sweep was re-run at the new
operating point twice — once after the mapping change and again after vector
staging. Along **N** the inversion widened: at `64x128` on 16 simdgroups, 1x1 now
measures 4327 GFLOP/s against 2836 for 2x2, 2822 for 4x2 and 1892 for 4x4, a 53%
gap where it used to be 28%. Blocking along **N** breaks the shared-column mapping
that lets a step load one B fragment, and that is now most of what the mapping is
worth.

Along **M** it un-inverted, which is the one place the second round moved a
default's *evidence* rather than the default: `2x1` blocking keeps the shared
column and adds a second A fragment per wave, and at `64x64` on 8 simdgroups it
measures **4635 GFLOP/s against 4262 for 1x1** — +9%, and the fastest configuration
at 2048 on this machine. The emitter still defaults to the smallest blocking that
fits, because the same `2x1` is a small loss at `64x128`; the sweep is what finds
it, which is what the sweep is for. The operand-load traffic
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

Two of the three candidates the previous round named are now done — the prologue
and epilogue (worth 2% between them, so the suspicion that they were the largest
remaining cost was **wrong**) and vector staging (worth 20%, and the largest single
change in either round). The one that paid most was not on that list at all: it was
the wave-to-block mapping, which nobody had looked at because the loads it removes
were the loads register blocking had already been measured not to care about.

And the balance has shifted. At 4.66 TF a 2048-cube GEMM takes 3.7ms, and at the
`64x64` tile that now wins, its 1024 programs read 1 MiB of operands each — ~1.1 GB
through the memory system in that time, **~290 GB/s against a 371 GB/s part**,
where the same calculation gave 145 GB/s at the end of round 1. (Some of that is L2
hits rather than DRAM traffic, so it is an upper bound — but it is the number that
has to fall.) The kernel is at ~75% of the *measured* 6.2 TF compute peak and near
enough to the bandwidth wall that arithmetic intensity, not latency, is what is
left to buy. What is left, in the order it should be tried:

1. **Keeping the accumulator out of threadgroup memory entirely.** A 128x128 tile
   halves the operand traffic of the 64x64 one that now wins, and that traffic is
   now the binding constraint rather than a rounding error. Its accumulator alone
   is 64KB, so the only way there is an epilogue that streams register fragments to
   device memory in panels rather than through one full-size tile — which means
   giving the epilogue its own loop over panels. Untried: the 60% this round aimed
   at was reached without it, and it is the largest change of the three by some way
   — but it is now clearly the right one to try next.
2. **Vector staging for narrow tiles.** Whether a tile is vectorised at all is
   decided by `stagingUnroll`, which needs a run of four columns to leave no
   thread idle. At the winning `64x64x16` on 8 simdgroups both operand tiles clear
   that bar; at `64x128x16` on 16 the `64x16` A tile does not, and stages
   scalar. Forcing runs of four there measures 3069 GFLOP/s against 4223 — the
   occupancy loss is much larger than the vector win — so the fix is not a longer
   run but a staging distribution that lets a *narrow* tile hand out runs of four
   without idling threads, e.g. by giving one thread several rows. That would also
   decouple the choice of block shape from whether staging vectorises, which is
   currently an accident of the arithmetic.
3. **A cheaper mask on the vector path.** The run guard re-evaluates the mask at
   the run's last column, which is a handful of integer ops per run per K step.
   With the block shape known at emission it is often statically true (`BLOCK_N`
   divides `N`), and Triton kernels usually pass the sizes as arguments — a
   specialisation on `N % BLOCK_N == 0` would remove it, along with the masking in
   the scalar path behind it.

## Attention throughput

FlashAttention-2 forward, lowered by this backend, against the **unfused
composite** — `S = Q K^T` and `O = P V` as two `MPSMatrixMultiplication`s with an
`MPSMatrixSoftMax` between them, one head at a time, through a real `S x S` f32
score matrix. That is the comparison a fused kernel exists to win: the composite
runs the same arithmetic through Apple's own GEMM, which is faster than this
backend's, but it has to write the score matrix to device memory and read it back
twice, and removing that traffic is the whole idea.

Both sides are timed the same way as the GEMM sweep — dispatches packed into one
command buffer until each sample is ~25ms of GPU work, median of three after a
warm-up. FLOPs are counted the way attention papers count them, `4 * S^2 * D` per
head, and the softmax is not counted on either side. Best configuration per shape
out of `tmbench --attn`.

### Apple M1 Max (studio-b)

| shape | best config | fused | MPS composite | ratio |
| --- | --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | `16x64`, `num_warps=16` | **1358 GF** | 381 GF | **357%** |
| `b1 h8 s1024 d64` f32 | `32x32`, `num_warps=16` | **1678 GF** | 1575 GF | **107%** |
| `b1 h16 s2048 d64` f32 | `32x32`, `num_warps=16` | 1761 GF | 2586 GF | 68% |
| `b1 h8 s512 d64` f16 | `32x32`, `num_warps=16` | **1680 GF** | 702 GF | **239%** |
| `b1 h8 s1024 d64` f16 | `32x32`, `num_warps=16` | **2089 GF** | 1582 GF | **132%** |

### Apple M1 Pro (laptop)

| shape | best config | fused | MPS composite | ratio |
| --- | --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | `32x32`, `num_warps=16` | **891 GF** | 513 GF | **174%** |
| `b1 h8 s1024 d64` f32 | `32x32`, `num_warps=16` | 948 GF | 1244 GF | 76% |
| `b1 h8 s512 d64` f16 | `16x32`, `num_warps=8` | **772 GF** | 531 GF | **146%** |
| `b1 h8 s1024 d64` f16 | `32x32`, `num_warps=16` | 1195 GF | 1231 GF | 97% |

**The fusion wins where it should and loses where it should.** At `s512` the
composite's GEMMs are small and there are three dispatches per head, so the fused
kernel is 1.7–3.6x faster. By `s2048` the composite's GEMMs are large enough to
run near MPS's peak and the score matrix, however much traffic it is, is no longer
what decides — the fused kernel reaches 68% of it. The crossover on both machines
is around `s1024`, and f16 moves it out: at `s1024` f16 the fused kernel is ahead
on the M1 Max (132%) and level on the M1 Pro (97%).

Two caveats on the composite side. Its readings move a lot run to run at `s512` —
the same M1 Pro configuration read 478, 513 and 588 GF in one session — because
24 small dispatches are mostly submission overhead, so treat the `s512` ratios as
"clearly ahead" rather than as a number. And this composite is not the only one
possible: metalscope's bench measured MPS-composite SDPA at ~744 GF on an M1 Pro
for `b1 h8 s512 d64`, which is faster than the 513–588 GF measured here for the
same shape, so a better-written composite exists and the `s512` margin against it
would be nearer 1.2x than 1.7x.

### What the kernel is spending its time on

The one change measured on its own was worth **3.9x**: reducing across a row's
lane group instead of across the whole threadgroup (§Cross-lane regions). At
`b1 h8 s512 d64` on the M1 Pro the best configuration went from **174 GF to
674 GF** with nothing else altered. The old shape ran `2 * BLOCK_M` threadgroup-wide
folds per key block, serially, each with two barriers and 128 threads cooperating
on 32 values — for `BLOCK_M = 16` that is 32 barrier pairs per iteration against
about 1000 useful MACs per thread.

What is left, in the order it should be tried:

1. **Threadgroup memory is the binding constraint on block shape.** `BLOCK_M`
   above 32 does not fit at `HEAD_DIM = 64`: the f32 accumulator (`BLOCK_M x
   HEAD_DIM`) and the f32 score tile (`BLOCK_M x BLOCK_N`) are both live across the
   whole iteration, and the operand arena has to hold `Q` and `K^T` at once. A
   `64x64` block wants 64KB. That caps arithmetic intensity: at `16x64` each
   iteration stages 10240 elements to do 131072 MACs. Keeping the score tile in
   registers between the two dots, rather than round-tripping it through
   threadgroup memory, is the change that would lift the cap — and it is the same
   change the GEMM wants for its accumulator (below).
2. **`Q` is staged every iteration although it is loop-invariant.** The deferral
   model rebuilds a dot operand inside the dot's own loops, and nothing yet
   notices that this one does not depend on the induction variable. That is 1024
   redundant device loads per iteration at `16x64`.
3. **`P` is computed twice** — once materialised for the row sum, once rebuilt
   inside the second dot's staging. One extra `exp2` per element per iteration.

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
3. ~~**A per-lane tensor carried across a cross-lane loop.**~~ Done — see
   §Cross-lane regions. Carried tensors are spilled to threadgroup tiles and
   updated through a shadow, except the dot accumulator a loop rescales in place,
   which becomes the dot's own accumulator tile and needs no copy at all.

   One thing this note had wrong, and it mattered: FlashAttention-2's `m_i` and
   `l_i` are **not** scalars. They are `BLOCK_M`-wide per-row vectors, rank-1
   tensors in this model, and they needed the same spill machinery as `acc` did.
   Only a rank-1 kernel's online softmax — where the whole program is one row —
   carries genuine scalars, which is why `AdvancedFixtures.onlineSoftmax` worked
   while nothing else did.

   Two things it did not anticipate at all, and both were larger than the spill:
   FA-2 needs **three** block axes rather than two, and a reduction result that a
   later `tt.dot`'s staging loops have to read. See §Execution model and
   §Cross-lane regions.
4. ~~**Atomics coverage**: Metal lacks some of CUDA's atomic dtypes (e.g. fp16
   atomics).~~ Done for `f32` and `i32` — see §Atomics. The note was right about
   the shape of the problem and wrong about which dtype bites: `f16` atomics are
   indeed missing, but so are **64-bit** ones, and the gap that actually cost
   code was float `max`/`min`, which CUDA has had since Kepler and Metal does not
   have at all. Two things it did not anticipate: MSL has exactly one memory
   order, so Triton's `sem`/`scope` are dropped rather than mapped; and an atomic
   in a threadgroup-uniform nest needs the single-writer guard as a *correctness*
   requirement rather than the optimisation it is for a store.
5. **num_warps > actual concurrency**: occupancy model differs; autotune must
   re-learn its search space (feeds metalscope's roofline data back in here).
6. **Occupancy of outer block dimensions**: only the innermost dimension is spread
   across threads, so a `BLOCK_M x BLOCK_N` tile with a small `BLOCK_N` leaves most
   of the threadgroup idle. Half done: a `tt.dot`'s staging loops always spread
   over both tile dimensions, and *its* per-lane block loops now do too, which is
   what makes a dot kernel's epilogue use all 512 threads instead of 128. It is
   still innermost-only for everything else, because only a dot kernel is
   contractually launched at the exact `threads_per_threadgroup` the two-dimensional
   split has to bake in, and because a `tt.reduce` folds across the whole
   threadgroup and so needs every thread in the same row.
7. ~~**`tt.trans`**, which FlashAttention needs for `K^T`.~~ Done, and cheaper
   than even the `transpose_matrix` route this note proposed: a transpose is a
   relabelling of which block axis each dimension indexes, so `K^T`'s tile is
   staged over `(HEAD_DIM, BLOCK_N)` directly and **no code is emitted for the
   `tt.trans` at all**. The `simdgroup_load` flag is still worth having later —
   staging `K^T` this way reads `K` down its columns, so the device loads are not
   coalesced and the vector-staging fast path's runtime stride check correctly
   declines. Staging `K` naturally and transposing at the fragment load would fix
   that; it is a pure throughput change, not a correctness one.

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

10. **FlashAttention-2 forward** — the fused kernel against a CPU reference at
    five shapes (`h2 s127 d64`, `h1 s512 d64`, `h1 s96 d80`, `h3 s33 d64`,
    `h1 s64 d64`, so that neither the sequence length nor the head dimension
    divides the block shape or the 8x8 fragment), f16-in/f32-accumulate at two
    more, `num_warps` 1..8, and a stability case whose raw scores reach ~1100 —
    where a naive `exp` is not a float — checking both that the output is finite
    and that it matches the reference. Plus assertions on what the lowering had to
    do: three iteration axes and no separate contraction axis, `tt.trans` emitting
    no code, `m_i`/`l_i` spilled with a shadow, the accumulator rescaled in place
    inside the second dot's staging pass, the row maximum spilled for that dot to
    read, the two dots' operand tiles sharing one arena, and an over-large block
    shape refused with its byte count.

11. **Atomics** — concurrent `fadd` accumulation across a grid large enough that
    every output slot is contended by hundreds of threads in many threadgroups,
    against a CPU reference, at four shapes and at `num_warps` 1..32; masked
    variants at counts that are not multiples of the block, where one spurious
    add is visible in the total; every integer kind (`add`, `max`, `min`,
    `umax`, `umin`, `and`, `or`, `xor`) against an order-independent reference,
    with a value spread that makes signed and unsigned disagree; `exch`'s
    invariant that the surviving value is one some lane wrote; f32 `max`/`min`
    through the compare-exchange loop, including negatives; the *returned old
    value* as a permutation of `0..<k` (order-independent by construction);
    `tt.atomic_cas`'s one-winner-per-slot property; the reordering bound
    (§Atomics); that a uniform nest performs the atomic once and refuses a used
    result; and the refusals — `f16`, `i64`, integer-only kinds on float
    pointers and `fadd` on integer pointers. Every emitted atomic kernel is also
    put through Metal's own front end.

`MatmulBenchmark` is a further suite that stays off unless `TM_BENCH=1` is set: it
measures a machine, not a contract. The measurement itself lives in the
`TritonMetalBench` library and is shared with the **`tmbench`** executable
(`swift run -c release tmbench`), because XCTest does not exist on a machine with
only the command-line tools installed — which is where the numbers in
§Matmul throughput were taken. `tmbench` sweeps block shapes, `num_warps`,
register blocking, tile padding and double buffering; pins a single configuration
with `--config M,N,K,W[,RM,RN[,U[,P[,DB]]]]`; prints a kernel with `--emit`; and
checks its winners against a CPU reference at `129x257x65` and other awkward
shapes before reporting them.

`tmbench --attn` is the same measurement for attention, against the unfused MPS
composite, and verifies its winners against a CPU reference before reporting them.

Next: port Triton's own backend conformance tests (`test_core.py` subset) against
numpy references, and the FA-2 **backward** pass.

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
   throughput half of this milestone is met twice over: **76% of MPS** at 1024,
   2048 and 4096 on an M1 Max, up from ~33%, via register blocking,
   register-resident accumulators, storage sharing between the accumulator and its
   operand tiles, cheaper staging, a wave mapping that lets every wave of a
   simdgroup share one operand fragment, and vector staging runs. §Matmul
   throughput has the per-change attribution, what did *not* help on Apple silicon,
   and what is left.
   `tt.atomic_*` and `tt.trans` still open.
8. ~~Reductions inside an `scf.for`.~~ Done, for loop-carried scalars *and*
   tensors — the latter spilled to threadgroup tiles (§Cross-lane regions).
9. ~~FlashAttention-2 forward.~~ Done — correct on the GPU against a CPU reference
   at sequence lengths and head dimensions that divide neither the block shape nor
   the 8x8 fragment, in f32 and f16-in/f32-accumulate, and numerically stable on
   scores that overflow a naive `exp`. It needed more than §Hard parts 3 and 7
   between them predicted: a nest per axis-set rather than one shared nest, axis
   *unification* rather than identity seeding, a spill for reduction results a dot
   reads, and a shared operand arena to fit in 32KB. Throughput is **357% of the
   unfused MPS composite** at `b1 h8 s512 d64` and 107% at `s1024` on an M1 Max,
   falling to 68% at `s2048` where MPS's GEMMs reach their own peak
   (§Attention throughput).
10. **Next up**: the FA-2 **backward** pass, `tt.atomic_*`, and pinning a Triton
    release.
