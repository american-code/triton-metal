# triton-metal: A Triton Compiler Backend for Apple Metal

*A Swift core behind a C ABI, emitting readable Metal Shading Language*

---

## Abstract

Triton is how the machine-learning community writes custom fused GPU kernels —
FlashAttention variants, MoE routing, quantization kernels — without hand-writing
CUDA. It targets CUDA and ROCm, and nothing else. That gap is one of the more
durable pieces of CUDA's portable-code moat, and it means a large and growing
body of research code simply does not run on Apple silicon.

This paper describes triton-metal, a compiler backend that lowers Triton IR to
Metal. It reports what works, what does not, and by how much it misses its own
performance target. The backend takes Triton IR text through a recursive-descent
parser, a layout-inference pass, and an emitter that produces textual Metal
Shading Language, then compiles and launches it through Metal at runtime.
Triton's fused-softmax and matmul tutorial kernels both run on the GPU and match
CPU references, including at sizes divisible by neither the block shape nor the
8x8 simdgroup fragment. All compiler logic is Swift, exposed as a C ABI; the
Python package is a ctypes shim with no logic in it, present only because
Triton's backend discovery imports a Python module.

The evaluation is deliberately unflattering where the results are. The test suite
is 127 Swift cases and 14 Python cases, at ~83% region / ~85% function / ~90%
line coverage of the Swift core. The lowered matmul reaches roughly **50% of
`MPSMatrixMultiplication`** at 1024, 2048 and 4096 square on an M1 Max (3.06
TFLOP/s f32 at 2048), and ~50% on an M1 Pro — up from ~33%, which clears the >50%
milestone target but not the 60–80% band the optimisation round aimed at. We give
a per-change attribution, including the two techniques from the CUDA playbook that
did **not** transfer to Apple silicon. Two named blockers stand between
this backend and a FlashAttention-2 forward pass, and the Triton release it
should be pinned to has not been pinned.

---

## 1. Motivation

The interesting property of Triton is not that it generates fast kernels. It is
that a Triton kernel is *portable source*: `tl.load`, `tl.dot`, `tl.reduce` and
a block-programming model, with no mention of warps, shared-memory banks, or
tensor-core fragment layouts. That abstraction is what lets an attention variant
be published as forty lines that anyone can read, modify, and run.

Except that "anyone" means anyone with an NVIDIA GPU, or lately an AMD one.
Triton's in-tree backends target CUDA and ROCm. A researcher on an Apple laptop —
which is to say a large fraction of researchers — cannot run the kernel at all,
and the workaround is to rewrite it. Each rewrite is a fork of a kernel that was
supposed to be portable.

Apple silicon is not a marginal target for this. Unified memory removes the
staging copy that dominates small-kernel latency on discrete GPUs; simdgroups are
32 lanes wide, exactly matching a CUDA warp, so `num_warps` maps across without
reinterpretation; and threadgroup memory plays the role of CUDA shared memory
closely enough that the block-programming model survives intact. The mismatches
are bounded: no cross-threadgroup barrier, different atomic dtype coverage, and
8x8 simdgroup matrices in place of CUDA's mma fragment shapes.

The goal is a backend that makes existing Triton kernels run unmodified on Metal
— not a Metal DSL that resembles Triton, but the same IR, so the same source file
works.

---

## 2. Related work

**Triton's own backends.** The CUDA backend lowers ttir → ttgir → LLVM IR →
PTX, with TritonGPU layout encodings (blocked, sliced, mma) that pin down which
lane holds which tensor element; the ROCm backend follows the same structure with
AMD's matrix instructions. Both invest heavily in the layout system, because on
those targets the mapping from tensor elements to registers *is* the
optimization. A Metal backend cannot reuse those encodings — CUDA's mma layouts
describe a distribution over lanes that Metal's `simdgroup_matrix` type
deliberately hides — but it can and does reuse everything above ttgir, which is
Triton's own MLIR pass pipeline.

**MLX.** Apple's array framework for Apple silicon: a NumPy-shaped API with lazy
evaluation, unified-memory arrays, and hand-written Metal kernels underneath. MLX
is the pragmatic answer today for someone who wants to *train something* on a
Mac, but it is not a solution to the portability problem, because an MLX
implementation of an attention variant is a rewrite of the Triton kernel rather
than an execution of it. The two are complementary: MLX is a framework,
triton-metal is a compiler target.

**MPS / MPSGraph.** Apple's vendor libraries — hand-tuned kernels for the
operations Apple chose to tune. `MPSMatrixMultiplication` is the right yardstick
for a GEMM and is used as such in §6, but a vendor library is the opposite of a
custom-kernel story: it is fast at exactly the operations someone already
implemented, which is why fused custom kernels exist at all.

**Other Triton backend efforts.** Triton's plugin interface has attracted
out-of-tree backends for other accelerators and for CPU. Their common lesson,
taken seriously here, is that the interface churns between releases and that
pinning is not optional (§7).

---

## 3. Design

### 3.1 Emit textual MSL, not AIR

The obvious "serious compiler" choice is to emit AIR (Metal's LLVM bitcode
dialect) and skip a parse. We emit Metal Shading Language *source text* instead,
and compile it with `MTLDevice.makeLibrary(source:)`.

The reason is debuggability, and it has paid for itself repeatedly. Every SSA
value in the IR becomes a named variable in the output: `%13 = arith.addf %9, %12`
is `float v13 = v9 + v12;`. A diff between the input IR and the generated kernel
is a line-for-line diff, which makes a lowering bug something you can *read*
rather than something you bisect.

The costs are accepted: a text round-trip through Metal's front end on every
compile, and no control below the MSL level. AIR emission is a later
optimization, appropriate once correctness is established, and it does not change
any of the analysis above it. A second path — `xcrun metal` producing `.metallib`
bytes offline — exists for callers that want a cacheable artifact. `xcodebuild`
is never used.

### 3.2 Language policy: Swift core, C ABI, no-logic shim

Everything that does work is Swift in `Sources/TritonMetalCore`, exported as
`tm_*` symbols in `libtritonmetal.dylib`. The Python package
`python/triton_metal` is a ctypes binding and nothing else.

This is a policy, not an accident. A backend written half in Python and half in a
compiled core drifts: launch geometry gets computed on the Python side "just for
now", error handling forks, and eventually the compiled core cannot be used from
anything but Python. So the shim marshals arguments, translates a documented
failure sentinel into a `RuntimeError` carrying `tm_last_error`, and computes
nothing.

The load-bearing example is `tm_kernel_info`, which returns launch metadata as
JSON:

```json
{"kernels": [{"name": "add_kernel", "block_size": 1024, "block_shape": [1024],
              "threads_per_threadgroup": 128,
              "args": [{"index": 0, "kind": "pointer", "dtype": "f32"}, …]}]}
```

Block shape, threadgroup size and argument kinds are properties of the *lowered
kernel*. The threadgroup size in particular is not `num_warps * 32`: it is
clamped to the innermost block dimension, rounded **up** to a whole simdgroup,
capped at Metal's 1024, and a `tt.dot` kernel skips the clamp entirely. Computing
that in Python would mean reimplementing part of the emitter there — so the core
computes it, the shim reads JSON, and a Python test asserts exactly that.

The Python package exists for one reason: Triton's backend discovery imports a
Python module, and rebuilding Triton's frontend is a multi-month project that
would defeat the purpose of a *Triton* backend.

### 3.3 One scalar per lane, and the layout redesign

The execution model is: **one Metal threadgroup per Triton program**, and every
tensor in a kernel is a slice of one **block** — an index space of rank R with a
size per dimension — walked by R nested loops. Only the innermost dimension is
distributed over threads; outer dimensions are looped uniformly by every thread.

```metal
for (uint tm_i0 = 0u; tm_i0 < BLOCK_M; ++tm_i0) {                 // uniform
    for (uint tm_i1 = tm_thread_id.x; tm_i1 < BLOCK_N;            // distributed
         tm_i1 += tm_threadgroup_size.x) { ... }
}
```

A value is emitted at the **shallowest depth its own dimensions allow**. So
`tt.get_program_id` lands in the prologue, `offs_m[:, None] * stride` (rank 2,
but constant along columns) is computed once per row, and only column-varying
work reaches the innermost loop. Statements shallower than the distributed loop
run redundantly in every thread, so a store at such a depth is guarded with
`tm_thread_id.x == 0u`.

The subtle part is *which* block dimension a value spans. `tl.arange(0, M)` is
rank 1 even inside a rank-2 kernel, and nothing about it says whether it indexes
rows or columns — only the `tt.expand_dims` that consumes it does. So the mapping
is **inferred**: `Layout.swift` seeds every full-rank tensor with the identity
map and propagates backwards through `tt.expand_dims`, `tt.broadcast` and the
elementwise ops to a fixed point. A rank-deficient value that never reaches a
seed is an error, not a guess.

Adding `tt.dot` forced this design to grow. A `simdgroup_float8x8` is *not* one
scalar per lane: it is a per-simdgroup object whose 64 elements sit in an opaque
distribution across 32 lanes, populated by `simdgroup_load` from memory rather
than assembled from registers. Three things followed.

**A contraction axis.** `tt.dot`'s operands are `MxK` and `KxN` while its result
is `MxN`, so the index space grows a dimension that the elementwise lane
distribution deliberately does *not* iterate. `BlockLayout` keeps iteration axes
(`rank`, `shape`) separate from `contractions`, one per dot, at axis `rank + i`.
Inference seeds the dot's operands *first* — `a` spans `(M, K)`, `b` spans
`(K, N)`, accumulator and result span `(M, N)` — and only then identity-seeds the
ordinary tensors, so that everything feeding an operand inherits the operand's
axes rather than being forced onto the iteration space.

**A second value class.** A *tile* is a tensor value living in a threadgroup
array rather than per-lane registers. Tiles are padded up to whole 8x8 fragments
and the padding is zero-filled during staging — which is what makes `BLOCK_M`,
`BLOCK_N` and `BLOCK_K` that are not multiples of 8 work without a separate
edge-tile path. Back inside the block loops a tile reads as an ordinary per-lane
value: the SSA name is bound to `tile[i0 * padN + i1]`, so every consumer
downstream of a dot — casts, math, masked stores — needs no special case.

**Deferral.** A dot operand is never materialised where it appears; it is rebuilt
inside the dot's own staging loops, over an index space that includes the
contraction axis. A backward liveness walk marks a value *materialised* when some
use other than "operand of a `tt.dot`" needs it, and defers everything else,
replaying the producer subgraph in program order inside the staging nest. A value
that spans the contraction axis *and* is materialised is refused by name.

---

## 4. Implementation

### 4.1 Parsing two MLIR spellings

Triton prints its IR in MLIR's *pretty* op syntax
(`%4 = arith.addi %3, %2 : tensor<1024xi32>`), except for ops carrying regions,
which print in the *generic* form. The parser is recursive descent over a
hand-written lexer, handling the pretty form for everything and the generic form
for `tt.reduce` specifically — including its `^bb0` region block, its
`<{axis = 0 : i32}>` property dictionary, and its `tt.reduce.return` terminator.
Generic syntax for anything else is rejected with a message telling the user to
dump the pretty form.

Everything the parser refuses, it refuses *by name and source position*:
`unsupported op 'tt.trans' at line 3, col 5: …`. The alternative — silently
mis-compiling an op nobody implemented — is the failure mode that makes a
compiler untrustworthy, so 17 of the parser tests are error paths asserting that
the message names the offender and its line.

Real-IR details had to be handled rather than idealised away: both spellings of
`tt.get_program_id`, the multi-result `%r:2 = … -> (T, U)` / `%r#0` form,
`loc(…)` trailers, argument attributes, the older `{cache = …, evict = …}` load
attributes, ttgir layout encodings (parsed, then ignored), and MLIR's raw
bit-pattern spelling of non-finite floats — `dense<0xFF800000>` is how `-inf`
arrives, and softmax does not work if you cannot read it.

### 4.2 The safe-math discovery

**Metal defaults fast math ON.** This is the single most consequential fact we
found, and it is easy to miss because everything still *runs*.

With fast math on, `exp`, division and denormal handling all drift from an IEEE
reference — enough to fail a tolerance-checked comparison against a CPU
implementation, and enough that a numerically careful kernel like an online
softmax stops being numerically careful. The backend sets
`MTLCompileOptions.mathMode = .safe`.

Then the mapping: `math.*` lowers to Metal's `precise::` namespace where one
exists (`exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `sin`, `cos`) and to the
default namespace where it does not (`tanh`, `floor`, `ceil`, `fabs`, `abs`).
`fast::` is never emitted, and a test asserts that. Metal has no `erf` at all, so
`math.erf` lowers to a generated `tm_erf` helper (Abramowitz & Stegun 7.1.26),
accurate to ~1.5e-7 absolute — float precision over erf's range. Every math op is
checked on the GPU against a CPU reference with a per-function tolerance.

The payoff shows up in the numbers: the fused softmax's worst absolute error
against a CPU reference over 12 rows of 100 columns is **7.45e-08**, about one
float ULP at those magnitudes.

### 4.3 Cross-lane ops, uniform-region hoisting, recomputation

A `tt.reduce` lowers to `simd_shuffle_down` within each 32-wide simdgroup plus
threadgroup memory across simdgroups. Because a reduction needs every thread to
reach the same barrier, it **closes** the distributed loop, folds the per-thread
partials, and opens a fresh loop for whatever follows.

What happens to values that were live across the reduction? They are
**recomputed** in the new loop, not spilled. Every lowered op is pure, so
recomputation is always legal; softmax therefore emits three lane loops over the
same row, which is cheaper than `3 x BLOCK` floats of threadgroup memory.
Recomputation is refused with a precise error when the recomputed region would
contain a store or an `scf` region — that is, when purity would be violated.

The harder case is a cross-lane op *inside* an `scf.for`, which is what an online
softmax needs. Such a loop cannot stay inside the block loops, because different
lanes would then execute different trip counts around a barrier. So the whole
loop is **hoisted out of the block loops** and lowered threadgroup-uniformly; the
block loops are opened and closed *inside* its body. Its recompute bookkeeping is
scoped to the region: it inherits the enclosing history, so a lane loop opened
inside the body rebuilds whatever was live when the loops closed around it,
without leaking its own statements back out.

Hoisting costs the loop its ability to carry per-lane tensors. Exactly three
`iter_args` shapes survive:

* **scalars**, which are uniform anyway — a reduction's result is broadcast to
  every thread. This is what an online softmax's running maximum and running sum
  are, and why it works end to end today;
* a **`tt.dot` accumulator**, which becomes the threadgroup tile the dot updates
  in place;
* a **contraction-space pointer** (`a_ptrs += BLOCK_K * stride_ak`), which is not
  carried at all: it is **strength-reduced** back to `init + trip_count * delta`,
  so the dot can rebuild it from scratch inside its staging loops.

Anything else — a per-lane tensor — is refused by name. That refusal is the
FlashAttention-2 blocker (§7).

### 4.4 `tt.dot` lowering, and the staging discovery

Per dot: stage A into a threadgroup tile over `(M, K)`, stage B over `(K, N)`,
barrier, then hand out 8x8 output fragments one per simdgroup:

```metal
for (uint f = tm_simd_group; f < FM * FN; f += tm_simd_count) {
    simdgroup_load(c, tm_dot_c + ...);
    for (uint s = 0u; s < FK; ++s) {
        simdgroup_load(a, ...); simdgroup_load(b, ...);
        simdgroup_multiply_accumulate(c, a, b, c);
    }
    simdgroup_store(c, tm_dot_c + ...);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

f16 operands with an f32 accumulator — the shape that matters for ML — mix
`simdgroup_half8x8` operands with a `simdgroup_float8x8` accumulator, which Metal
supports directly. Tile storage is checked against Metal's 32 KB threadgroup
budget, and an over-large block shape is refused with the byte count.

The performance discovery here was in staging, not in the matrix instructions.
The first implementation strided only the tile's innermost dimension across the
threadgroup, the way an elementwise loop does. But a `BLOCK_M x BLOCK_K` tile is
usually much narrower than the threadgroup, so most threads had nothing to do.
Spreading the threadgroup over **both** tile dimensions — `lanes` consecutive
threads take consecutive columns, which also keeps the device reads coalesced,
and the rest walk the rows — was worth **~4.7x** on the matmul tutorial. By a
distance the single biggest change in the kernel, and it was not in the part of
the code that looks like it does the arithmetic.

---

## 5. Test strategy

127 Swift cases across ten suites (one, the benchmark, is opt-in and skipped by
default), plus 14 Python cases. Everything that can run on the real GPU does.

| Suite | Cases | What it covers |
| --- | --- | --- |
| `ParserTests` | 17 | op/type/attribute shapes, both `tt.get_program_id` spellings, rank-N types, multi-result `%r:2`/`%r#0`, comments and `loc(…)`, and every error path asserting the message names the offender and its line |
| `EmitterTests` | 7 | the exact MSL for vector-add, uniform/row/lane partitioning, threadgroup sizing vs `num_warps`, metadata JSON, that `expand_dims`/`broadcast` emit nothing, that every fixture compiles in Metal's front end |
| `CastMathTests` | 11 | one kernel per `arith` conversion and per `math.*` op against a CPU reference with per-function tolerance; `precise::` used exactly where Metal has it, `fast::` never |
| `ControlFlowTests` | 9 | strided `scf.for` accumulation with tensor `iter_args`, a zero-trip loop, multi-result loops, a tensor-yielding `scf.if` |
| `Rank2Tests` | 9 | tiled add and copy at sizes dividing neither block dimension, a padded-stride guard proving masks protect inter-row gaps, assertions that row-uniform work is hoisted |
| `ReductionTests` | 15 | add/max/min swept across `num_warps` 1..32, row-wise rank-2 reduction, fused softmax at four shapes, a rows-sum-to-one check on inputs large enough to overflow a naive `exp`, and an online softmax with both reductions inside an `scf.for` |
| `DotTests` | 19 | single-tile products at six shapes including `5x3x7` and `12x20x12`; the matmul tutorial at six shapes including `129x257x65`; f16-in/f32-out; `num_warps` 1..32; a sentinel test proving masked stores leave inter-row padding alone; assertions on tile sizes, barrier placement, accumulator residency and pointer strength reduction; and eight error paths |
| `EndToEndTests` | 16 | copy/add/mul/scale-bias/integer kernels vs CPU references at non-multiple-of-BLOCK sizes; guard-region tests; `num_warps` 1..32; the offline `xcrun metal` path and its diagnostics; runtime geometry validation; handle-table type safety |
| `CABITests` | 18 | the whole spine through `tm_*` only, plus NULL-argument handling, handle validity, copy bounds, metallib loading, f32 launch arguments, and leak checks via `tm_live_handle_count` |
| `MatmulBenchmark` | 1 | opt-in (`TM_BENCH=1`); measures a machine, not a contract |

The Python suite's 14 cases assert the shim's *shape* (stage chain, backend
exports, driver delegation) and run vector-add and softmax round trips over the
C ABI — moving bytes and comparing numbers, while every compile, allocation and
launch happens inside the dylib.

Two strategy choices are worth stating. Sizes divide *nothing*: `129x257x65`
divides neither a block dimension nor the 8x8 fragment, softmax rows are 100
columns wide in a 128-wide block, and `5x3x7` is a dot smaller than a single
fragment in every dimension. And error paths are tested as first-class behaviour
— roughly a quarter of the suite asserts that something is refused, with the
right message.

---

## 6. Evaluation

### 6.1 Coverage

Regenerated with `swift test --enable-code-coverage`, then
`xcrun llvm-cov report … -ignore-filename-regex "Tests|\.build"`:

| File | Region | Function | Line |
| --- | --- | --- | --- |
| `Compiler.swift` | 97.58% | 97.22% | 99.01% |
| `IR/Lexer.swift` | 84.13% | 95.35% | 91.86% |
| `IR/Parser.swift` | 80.44% | 87.69% | 85.45% |
| `IR/TritonIR.swift` | 80.28% | 71.43% | 90.76% |
| `Layout.swift` | 84.31% | 70.37% | 90.12% |
| `MSLEmitter.swift` | 83.47% | 84.85% | 91.10% |
| `Runtime.swift` | 85.92% | 100.00% | 92.21% |
| **Total** | **83.43%** | **85.58%** | **90.32%** |

Over ~5,000 lines of Swift core and ~3,600 lines of tests. The 282-line Python
package is the shim. `TritonMetalBench` — the GEMM sweep shared by `tmbench` and
the opt-in XCTest wrapper — is measurement infrastructure rather than core and is
excluded; it is exercised by `tmbench` itself, which verifies every configuration
it reports against a CPU reference.

What the uncovered fraction is made of matters more than the number.
`Runtime.swift`'s remaining gap is entirely *environment-failure* branches: no
Metal device, no command queue, the pre-macOS-15 `fastMathEnabled` fallback, a
failed shared-buffer allocation, a failed command encoder, and a command buffer
that reports a driver-side execution error. On a machine where the suite passes,
none of those can be reached. `Compiler.swift`'s two remaining gaps are the
"could not run `xcrun metal`" branch and a `JSONSerialization` failure on a
dictionary of strings and integers. The genuinely reachable-but-untested code is
in the parser, the layout pass and the emitter — mostly diagnostic paths for
malformed IR that a well-behaved Triton frontend would never produce.

### 6.2 Matmul throughput

The matmul tutorial kernel, lowered by this backend, against
`MPSMatrixMultiplication`, f32, square, best configuration per size out of the
`tmbench` sweep over `BLOCK_M x BLOCK_N x BLOCK_K`, `num_warps`, register
blocking, tile padding and double buffering. Dispatches are packed into one
command buffer until each timed sample is ~25ms of GPU work, so submission
overhead is not most of what either side is measured doing; median of three.

On an **Apple M1 Max** (Mac Studio; measured peaks 6.2 TFLOP/s f32, 371.5 GB/s),
which is the machine to believe — it is not thermally constrained and its MPS
readings are stable:

| size | best configuration | triton-metal | MPS | ratio | was |
| --- | --- | --- | --- | --- | --- |
| 512 | 64x64x16, `num_warps=8` | 2.01 TFLOP/s | ~2.18 TFLOP/s | 92% | 62% |
| 1024 | 64x64x16, `num_warps=8` | 2.75 TFLOP/s | ~5.54 TFLOP/s | **50%** | 33% |
| 2048 | 64x128x16, `num_warps=16` | 3.06 TFLOP/s | ~6.10 TFLOP/s | **50%** | 33% |
| 4096 | 64x128x16, `num_warps=16` | 3.21 TFLOP/s | ~6.13 TFLOP/s | **52%** | 32% |

On an **Apple M1 Pro** (laptop), where MPS readings move with thermal state:

| size | best configuration | triton-metal | MPS | ratio | was |
| --- | --- | --- | --- | --- | --- |
| 1024 | 64x128x16, `num_warps=16` | 1.57 TFLOP/s | ~3.17 TFLOP/s | **50%** | 34% |
| 2048 | 64x128x16, `num_warps=16` | 1.69 TFLOP/s | ~3.37 TFLOP/s | **50%** | 33% |

**About 1.55x throughput on both chips, and ~33% of MPS becomes ~50%.** The >50%
milestone is met at the sizes where the measurement is trustworthy; the 60–80%
band the optimisation round aimed at is not reached.

The 512 row deserves its own sentence, because it would be easy to quote it
misleadingly. At 512 the GEMM is under a millisecond and MPS's own timing swings
by 1.5x run to run — we have measured the same MPS configuration at 1.36 and 2.28
TFLOP/s minutes apart. That row says more about measurement variance than about
either kernel, which is why the honest headline numbers come from 1024 and above.

Reproduce with `.build/release/tmbench --sweep full`, or
`TM_BENCH=1 swift test --filter MatmulBenchmark` where XCTest exists. The
executable exists because the M1 Max above has only the command-line tools
installed, and XCTest ships with Xcode.

#### What each change was worth

Measured one at a time on the M1 Max at 2048, each on top of the one before, by
pinning a configuration with `tmbench --config`:

| change | GFLOP/s | delta |
| --- | --- | --- |
| baseline: one output fragment per simdgroup, tile round trip per K step | 1993 | — |
| + accumulator resident in simdgroup registers across the K loop | 2257 | +13% |
| + register blocking (2x2 output fragments per simdgroup) | 2485 | +10% |
| + staging in runs of consecutive columns | 2726 | +10% |
| + literal (compile-time) staging trip counts | 2902 | +6% |
| + accumulator tile doubling as the operand arena, enabling 64x128 | 3062 | +6% |
| + bank-conflict padding | ~+2%, size-dependent | |

#### Two techniques that did not transfer

This is the part of the result we think is worth publishing.

**Double buffering is a wash.** With two operand buffers a contraction step stages
into the half the previous step is not reading, so the trailing barrier goes and
the staging need not wait on the arithmetic. Measured 2863 GFLOP/s against 2900
without it — inside the noise, if anything negative. Metal has no `cp.async`: the
prefetch is issued by the same threads into the same issue slots, so nothing is
actually asynchronous. What a CUDA GEMM buys with a dedicated copy engine, an
Apple GPU buys with occupancy — and doubling the operand tiles *costs* occupancy,
which is roughly what cancels the gain. Kept as an off-by-default knob.

**Register blocking past 1x1 does not help here, and at some shapes it hurts.** It
was worth 10% when we added it. After the staging work the full sweep separated
1x1 from 2x2 by under 1% at 64x64 — and at 64x128, 1x1 runs **28% faster** than
2x2 (3066 vs 2391 GFLOP/s), with the *same* number of accumulator fragments per
simdgroup, so this is not register pressure. The operand loads a bigger block
saves were never the binding constraint on this chip, and the coarser
fragment-to-simdgroup mapping it produces costs more than the loads it removes. We
changed the emitter's default to the smallest blocking that fits the register
budget — the opposite of the CUDA playbook's advice, and worth 28% at the block
shape that turns out to be the fastest one.

The general lesson from both results is that an instruction-slot account taken
before the fact — ours said multiply-accumulates were an eighth of the slots and
named operand traffic as the top fix — ranks the fixes in the wrong order when the
costs are not independent, and says nothing at all about the techniques whose cost
is occupancy rather than instructions.

### 6.3 Where the gap is now

The pre-optimisation account was counted in instruction slots: for the then-best
64x64x32 / 16-simdgroup shape, one K block cost each simdgroup 16
`simdgroup_multiply_accumulate`s against 32 operand `simdgroup_load`s and 8
accumulator load/stores, plus ~70 scalar ops per thread to stage 8 elements — the
multiply-accumulates about an eighth of the issue slots. That account ranked
operand traffic first and staging third.

**It ranked them wrong**, and how it did is the most transferable thing in this
section. The three costs were not independent: staging was hiding behind the
operand loads, and once staging got cheaper the operand-load fix that had been
worth 10% became worth under 1% (§6.2). An issue-slot count tells you what the
slots are spent on; it does not tell you which spend is on the critical path.

What the measurement says now is that the kernel is short of neither arithmetic
slots nor bandwidth. At 3.06 TFLOP/s a 2048-cube GEMM takes 5.6ms and moves about
0.8 GB, roughly 145 GB/s against a 371 GB/s part. Three candidates remain, in the
order we would try them:

1. **The prologue and epilogue.** Every dot still stages a zero accumulator into
   threadgroup memory and reads its result back per lane through the tile, and the
   per-lane block loops that do it spread over the innermost dimension only. For a
   64x128 tile that is a lot of badly-distributed work either side of a contraction
   loop that is itself only ~128 steps.
2. **Vector (`float4`) staging.** Staging in runs amortised the address
   arithmetic; each element is still its own device load. A real vector load needs
   an affine analysis of the staging subgraph — which is small and closed
   (`tt.make_range`, `tt.splat`, `arith.muli`, `arith.addi`, `tt.addptr`) — plus a
   runtime check that the innermost stride is 1.
3. **Keeping the accumulator out of threadgroup memory entirely.** A 128x128 tile
   would halve staging traffic again; its f32 accumulator alone is 64KB, twice the
   budget. The route is an epilogue that streams register fragments out in panels
   instead of through one full-size tile — much the largest of the three.

None of the three is a layout redesign, and two of them are local to `emitDot`.

---

## 7. Limitations and future work

**FlashAttention-2 needs two things this backend does not have.** It is the
integration milestone kernel, and both blockers are precisely characterised.

*A per-lane tensor carried across a cross-lane loop.* A loop-carried **tensor**
would have to be spilled to a threadgroup tile, and unlike a dot accumulator
(which the dot updates in place inside its own barriers) an arbitrary carried
tensor is read and written by ordinary per-lane code. Making that correct means a
tile write at every `scf.yield`, a barrier discipline around each cross-lane op
that also covers those writes, and a per-value decision between spilling and the
recomputation the emitter uses everywhere else. FA-2's `acc`
(`BLOCK_M x HEAD_DIM`) is exactly this case; its `m_i`/`l_i` are not — they are
the scalars that already work.

*`tt.trans`, for `K^T`.* Cheaper than it looks here: `simdgroup_load` takes a
`transpose_matrix` flag, so a transposed dot *operand* is a staging-time decision
rather than a data movement. A transposed value that is *materialised* instead
has no such shortcut.

**Layout conversions.** The inference in `Layout.swift` is the honest, small
version of what ttgir's layout system does. It maps tensor dimensions to block
dimensions, but says nothing about which lane holds which element — exactly what
an mma layout must pin down. This is also why `tt.dot` stages through threadgroup
memory rather than shuffling fragments between lanes: the emitter does not know
where a fragment's elements live.

**Not pinned to a Triton release.** The plugin surface is written against the
approximate 3.x layout — `triton.backends` discovery, a stage-dict compiler, a
driver with `load_binary`/`launch` — rather than signatures read off a tag. Until
a release is pinned, its signatures verified, and a CI check added, the three
shim modules are an interface sketch that tests green against the Swift core, not
a plugin that drops into a PyTorch install. This is the project's most important
open task, because it gates everything else being *usable* rather than merely
correct.

**Type coverage.** No `f64`, no `bf16` — the latter matters for ML and is the
more urgent. Block pointers (`!tt.ptr<tensor<…>>`) are not lowered. `tt.atomic_*`
is unimplemented, and since Metal lacks some of CUDA's atomic dtypes (fp16 among
them), its eventual coverage will be partial.

**Rank > 2 is untested.** Higher ranks lower through the same nested-loop
machinery and are believed to work — but "believed to work" is not a claim this
paper makes about anything else in it, so it is listed as a limitation.

**Occupancy of outer block dimensions.** Only the innermost dimension is spread
across threads, so a `BLOCK_M x BLOCK_N` tile with a small `BLOCK_N` leaves most
of the threadgroup idle. A `tt.dot`'s staging loops already spread over both tile
dimensions; doing the same for the elementwise nest is the obvious next step.

**No conformance suite yet.** The tests here are ours. Porting a subset of
Triton's own `test_core.py` against numpy references is what would turn
"our tests pass" into "this is a Triton backend".

---

## 8. Conclusion

triton-metal is a working spine with real kernels on it. Triton IR text goes in;
readable MSL, a Metal library, a compute pipeline, unified-memory buffers, a
dispatch, and correct results come out — driven from Swift or from any language
that can call C. The fused-softmax and matmul tutorial kernels both match CPU
references at sizes that divide neither the block shape nor the simdgroup
fragment, and the online softmax demonstrates the control-flow shape a
FlashAttention inner loop needs.

Three design decisions carried most of the weight. Emitting textual MSL made
every lowering bug readable. Keeping all logic in Swift behind a C ABI, with a
shim that computes nothing, kept the core usable from outside Python and stopped
launch geometry from leaking into two places. And inferring the tensor-dimension
to block-dimension mapping to a fixed point, rather than assuming it, allowed
rank-deficient tensors, broadcasting and a `tt.dot` contraction axis to coexist
without kernel-level annotation.

Two results deserve restating plainly. The numerics are good: fast math off,
`precise::` where Metal has it, and a softmax that agrees with a CPU reference to
about one ULP. The performance is now adequate rather than good: ~50% of MPS on a
GEMM, which clears the >50% milestone and falls short of the 60–80% band, with the
remaining gap traced to the epilogue, scalar staging loads and the accumulator's
occupancy of threadgroup memory rather than to the arithmetic.
What stands between this and being *used*, though, is not a research problem — it
is pinning a Triton release, and then the two named blockers on the way to
FlashAttention-2 forward.
