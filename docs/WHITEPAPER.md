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
Metal. Its thesis is competitive, not merely compatible: the same research kernel
runs unmodified on hardware a team already owns, at a fraction of Apple's vendor
library comparable to the fraction Triton reaches against NVIDIA's on its native
target, and the port yields something the CUDA literature does not contain — a
measured account of which GPU optimisation techniques transfer to Apple silicon
and which invert. §2 makes that case and states where CUDA remains ahead. The
backend takes Triton IR text through a recursive-descent
parser, a layout-inference pass, and an emitter that produces textual Metal
Shading Language, then compiles and launches it through Metal at runtime.
Triton's fused-softmax and matmul tutorial kernels both run on the GPU and match
CPU references, including at sizes divisible by neither the block shape nor the
8x8 simdgroup fragment, and so do the **FlashAttention-2 forward and backward
passes** — the kernels this backend was aimed at, and the ones that decide whether
a Triton backend is useful for anything beyond elementwise inference. A real
`@triton.jit` attention layer trains on a Mac GPU through the pinned release, with
its gradients checked against a hand-written numpy autograd and against finite
differences. All compiler logic is Swift, exposed as a C ABI; the Python package
is a ctypes shim with no logic in it, present only because Triton's backend
discovery imports a Python module.

The evaluation is deliberately unflattering where the results are. The test suite
is 184 Swift cases and 22 Python cases (15 of which need no Triton install), at
~83% region / ~85% function / ~90% line coverage of the Swift core. The lowered matmul reaches **76% of
`MPSMatrixMultiplication`** at 1024, 2048 and 4096 square on an M1 Max (4.66
TFLOP/s f32 at 2048), and 69–82% on an M1 Pro depending on its thermal state — up
from ~33% at the first working version, which clears both the >50% milestone and
the 60–80% band the second optimisation round aimed at. Fused attention is
measured against the composite it replaces rather than against a single MPS
kernel, because that is the comparison fusion exists to win: **357% of a
`Q K^T` + `MPSMatrixSoftMax` + `P V` composite** at `b1 h8 s512 d64` on an M1 Max,
107% at `s1024`, and 68% at `s2048` where MPS's own GEMMs reach their peak. We
give a per-change attribution for both kernels, including the two techniques from
the CUDA playbook that did **not** transfer to M1-generation Apple silicon and
the one with no CUDA analogue that was worth more than either of them.

Getting attention to run needed more than the two blockers we had named. Spilling
a loop-carried tensor was one of them, and it was the smaller half: the layout
model also had to stop assuming that a kernel's tensors share one loop nest, and
axis assignment had to become unification rather than seeding, because
FlashAttention's two `tt.dot`s share their axes crosswise and no fixed
(M, N, fresh-K) assignment can describe that.

The backend is now **pinned to Triton v3.7.1 and driven by it**: real
`@triton.jit` source, Triton's own frontend and MLIR passes, this backend, a Mac
GPU (§8.1). Real IR cost the parser one line and the emitter two small
generalisations, which is the most useful thing that section reports — the gap
between IR written from documentation and IR a release actually prints is small
but not empty. The backward pass needed **no** parser or emitter change beyond
what its Swift fixtures had already found, and real IR turned out to be where the
last block-shape restriction lived: Triton's CSE shares one `tl.arange` between a
row index and a column index when two block sizes are equal, which the layout
model correctly read as a diagonal. Un-sharing it before inference removed the
restriction (§8.4).

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

## 2. Value proposition: breaking the portable-kernel moat

CUDA's most durable software advantage is not the CUDA language: it is that the
ecosystem's custom kernels are written in Triton, and Triton emits code for CUDA
and ROCm and nothing else, which makes that work portable in principle and
NVIDIA-only in practice. This section states what this backend changes about that;
its last paragraph, and §8, state what it does not.

**The same source, not a lookalike.** The backend is pinned to Triton **v3.7.1**
and driven by it: real `@triton.jit` Python, Triton's own frontend and MLIR
passes, this backend, a Mac GPU. The vector-add, fused-softmax and matmul
tutorials run as published source and agree with numpy — max absolute error 0,
0 and 1.49e-08 respectively — and **no patch to Triton is required** (§8.1). The
backend attaches through the documented out-of-tree plugin interface; the ten
lines of C++ that Triton's `main.cc` demands of any plugin are the project's entire
non-Swift surface. That is what separates this from a framework port: an MLX or
hand-written Metal version of a kernel forks it, while this *executes* it, so the
upstream file stays the only copy (§3).

**Performance that is credible on arrival.** A portability story running at a
fraction of vendor speed does not get used. The lowered matmul tutorial reaches
**76% of `MPSMatrixMultiplication`** at 1024, 2048 and 4096 square on an M1 Max
(4.66 TFLOP/s f32 at 2048), up from ~33% at the first working version (§7.2). For
calibration, published Triton reaches 62–82% of a CUDA-kernel stack end-to-end on
its native target, and ~80–100% of cuBLAS on GEMM specifically, after years of
NVIDIA-specific pipeline work (COMPARISON.md): two optimisation rounds put a new
backend inside the lower band and not yet in the upper one. Fused attention
decides whether a Triton backend is useful beyond elementwise work.
FlashAttention-2 forward runs, and against the composite it replaces
(`Q K^T` + `MPSMatrixSoftMax` + `P V`) it measures **357%** at `b1 h8 s512 d64`,
**107%** at `s1024` and 68% at `s2048` on an M1 Max — a mapped crossover near
`s1024` that f16 moves outward, rather than a single headline ratio (§7.2b).

**Hardware the team already owns.** The practical form of the argument is that a
lab with Macs gets the Triton ecosystem without a purchase decision: the kernel
file is read, edited and run on the laptop it was found on, and NVIDIA time is
rented for the runs that need it rather than for the ability to run at all. Apple
silicon is a good host for this rather than a merely available one, for the
architectural reasons §1 gives.

**A measured result the CUDA world does not have.** Porting the optimisation
playbook produced a transfer matrix useful independently of this compiler.
Register-resident accumulators, staging locality, literal trip counts and
vectorized loads transfer. Two things **invert**: double buffering is a wash — and
a 26% loss once vector staging lands — because Metal has no `cp.async`, so the
prefetch competes for the same issue slots while larger tiles cost the occupancy
this GPU uses to hide latency; and register blocking past 1x1 inverts and then
widens, with 1x1 beating 2x2 at `64x128` by 28% and, after the second round, 45%,
at identical accumulator-fragment counts. The single largest win has **no CUDA
analogue at all** — a shared-column wave mapping worth 15%, available only because
Metal moves whole 8x8 fragments through threadgroup memory instead of distributing
them over named lanes, so which fragment a simdgroup wants is an emitter decision
rather than a layout's (§7.2, COMPARISON.md).
This is transferable guidance for anyone moving GPU work to Apple silicon, and it
is explicitly **M1-generation-scoped**: measured on an M1 Max and an M1 Pro, with
the M3/M4 memory-hierarchy rework unmeasured.

**Swift core, C ABI, no Python in the hot path.** All compiler logic is Swift
behind `tm_*` C symbols; the 282-line Python package is a ctypes shim that computes
nothing, present only because Triton's backend discovery imports a Python module
(§4.2). Launch geometry — block shape, threadgroup size, argument kinds — is
computed in the core and read out as JSON, so nothing about the lowered kernel is
reimplemented on the Python side. The backend is therefore callable from Swift,
from C, or from any language with an FFI, and the Python dependency is confined to
the frontend.

**Training, not just inference.** The backward pass runs. `tt.atomic_rmw` and
`tt.atomic_cas` lower to Metal's device atomics, and **FlashAttention-2 backward**
is implemented as three recompute-based kernels whose gradients are verified twice
over — against an analytic CPU reference in `Double`, and against central finite
differences of the forward, which know nothing of the backward formulas — in f32
and f16-in/f32-accumulate, at sequence lengths and head dimensions that divide
neither the block shape nor the 8x8 fragment. A real `@triton.jit` attention layer
trains end to end through pinned Triton 3.7.1 on a Mac GPU: gradients agree with a
hand-written numpy autograd to **6.2e-07** relative and with finite differences to
**3.7e-08**, and gradient descent on `0.5*||O - T||^2` takes the loss from 1541.73
to 1482.86 in eight steps with forward and backward both on the GPU (§8.3).
Against the strongest composite MPS can express — five GEMMs with a softmax and a
softmax gradient between them, through two live `S x S` matrices — the fused
backward measures **190%** at `b1 h8 s512 d64` on an M1 Max and 220% in f16,
crossing over near `s1024` (§7.2c).

**Where CUDA remains ahead, plainly.** "Training-capable" here means what it says
and no more: the attention forward and backward, the GEMM, softmax and elementwise
kernels, and atomics on `f32` and `i32`. It does **not** mean a framework. There
is no autograd — a caller writes or generates the backward kernel, as a Triton
user does; no optimizer-state kernels (Adam's moment updates are ordinary
elementwise work and would lower, but none is written or measured here); no
dropout, because there is no `tl.rand` and no RNG lowering at all; and no
distributed anything. There is no `bf16`, which is the type real training uses and
the most urgent gap in this list; no `f64` (Metal has no `double` at all); and no
block pointers. Atomics are 32-bit only — Metal has no 16- or 64-bit atomics and
no float `fetch_max`, which goes through a compare-exchange loop — and expose one
memory order, so a kernel relying on an atomic's release edge to publish ordinary
stores is not correctly lowered. Because Triton publishes no macOS wheel, adoption
costs a **from-source Triton build**: ~9 minutes, once, no CUDA toolchain and no
Xcode, but it is not `pip install triton`. Every performance number and every
inversion above is **single-generation**, M1 Max and M1 Pro. And Triton-on-CUDA is
simply more mature: a ttgir layout system, a conformance suite this project has
not yet ported a subset of, per-architecture pipelining, and a vendor library this
backend still trails by roughly a third on GEMM. The claim is that the
portable-kernel moat is crossable, and that a Triton attention layer now trains on
a Mac — not that the crossing is finished.

---

## 3. Related work

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
for a GEMM and is used as such in §7, but a vendor library is the opposite of a
custom-kernel story: it is fast at exactly the operations someone already
implemented, which is why fused custom kernels exist at all.

**Other Triton backend efforts.** Triton's plugin interface has attracted
out-of-tree backends for other accelerators and for CPU. Their common lesson,
taken seriously here, is that the interface churns between releases and that
pinning is not optional; this backend is pinned to v3.7.1 and vendors that tag's
signatures (§8.1).

---

## 4. Design

### 4.1 Emit textual MSL, not AIR

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

### 4.2 Language policy: Swift core, C ABI, no-logic shim

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

### 4.3 One scalar per lane, and the layout redesign

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

## 5. Implementation

### 5.1 Parsing two MLIR spellings

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

### 5.2 The safe-math discovery

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

### 5.3 Cross-lane ops, uniform-region hoisting, recomputation

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
FlashAttention-2 blocker (§8).

### 5.4 `tt.dot` lowering, and the staging discovery

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

## 6. Test strategy

184 Swift cases across thirteen suites (one, the benchmark, is opt-in and skipped
by default), plus 22 Python cases. Everything that can run on the real GPU does.

| Suite | Cases | What it covers |
| --- | --- | --- |
| `ParserTests` | 17 | op/type/attribute shapes, both `tt.get_program_id` spellings, rank-N types, multi-result `%r:2`/`%r#0`, comments and `loc(…)`, and every error path asserting the message names the offender and its line |
| `EmitterTests` | 7 | the exact MSL for vector-add, uniform/row/lane partitioning, threadgroup sizing vs `num_warps`, metadata JSON, that `expand_dims`/`broadcast` emit nothing, that every fixture compiles in Metal's front end |
| `CastMathTests` | 11 | one kernel per `arith` conversion and per `math.*` op against a CPU reference with per-function tolerance; `precise::` used exactly where Metal has it, `fast::` never |
| `ControlFlowTests` | 9 | strided `scf.for` accumulation with tensor `iter_args`, a zero-trip loop, multi-result loops, a tensor-yielding `scf.if` |
| `Rank2Tests` | 9 | tiled add and copy at sizes dividing neither block dimension, a padded-stride guard proving masks protect inter-row gaps, assertions that row-uniform work is hoisted |
| `ReductionTests` | 15 | add/max/min swept across `num_warps` 1..32, row-wise rank-2 reduction, fused softmax at four shapes, a rows-sum-to-one check on inputs large enough to overflow a naive `exp`, and an online softmax with both reductions inside an `scf.for` |
| `DotTests` | 27 | single-tile products at six shapes including `5x3x7` and `12x20x12`; the matmul tutorial at six shapes including `129x257x65`; f16-in/f32-out; `num_warps` 1..32; a sentinel test proving masked stores leave inter-row padding alone; the vector staging path taken and refused — padded tiles, misaligned rows, a non-unit innermost stride; assertions on tile sizes, barrier placement, accumulator residency, the shared-column mapping and pointer strength reduction; and eight error paths |
| `EndToEndTests` | 16 | copy/add/mul/scale-bias/integer kernels vs CPU references at non-multiple-of-BLOCK sizes; guard-region tests; `num_warps` 1..32; the offline `xcrun metal` path and its diagnostics; runtime geometry validation; handle-table type safety |
| `CABITests` | 20 | the whole spine through `tm_*` only, plus NULL-argument handling, handle validity, copy bounds, metallib loading, f32 and i64 launch arguments (the latter with a value that does not survive 32 bits), `f64` refused rather than narrowed, and leak checks via `tm_live_handle_count` |
| `AtomicTests` | 24 | concurrent `fadd` accumulation across a grid where every slot is contended by hundreds of threads, at four shapes and `num_warps` 1..32; masked tails; every integer kind against an order-independent reference with a spread that makes signed and unsigned disagree; `exch`'s surviving-value invariant; f32 `max`/`min` through the compare-exchange loop; the returned old value as a permutation of `0..<k`; `tt.atomic_cas`'s one-winner-per-slot property; the reordering bound; the single-writer guard and the refusal of a used result under it; and the four type/kind refusals |
| `AttentionTests` | 10 | FA-2 forward at five shapes dividing neither the block nor the fragment, f16-in/f32-accumulate, `num_warps` 1..8, scores that overflow a naive `exp`, and assertions on the three carried tensors, the spilled row maximum, the shared arena and the byte-counted refusal |
| `AttentionBackwardTests` | 9 | the three backward kernels against an analytic `Double` reference at eight shapes including two f16 ones, against central finite differences of the forward, with the real forward's own statistics feeding them, at `num_warps` 1..8; every tolerance derived from a summation bound; plus the arena, the two resident accumulators, the logsumexp recomputation, and the one shape-and-warp combination that does not fit |
| `AxisCloningTests` | 5 | the matmul tutorial as Triton prints it when block sizes collide, on the GPU at four of them; that the shared range is duplicated and the orphan removed; that a kernel which already lowers is untouched; and that a `tt.reduce` result at two dimensions is still refused |
| `MatmulBenchmark` | 1 | opt-in (`TM_BENCH=1`); measures a machine, not a contract |

The Python suite splits in two. Fifteen cases assert the shim's *shape* (the
plugin directory Triton discovers, the registration symbol its `main.cc` calls,
buffer alignment and dtypes) and run vector-add and softmax round trips over the
C ABI — moving bytes and comparing numbers, while every compile, allocation and
launch happens inside the dylib. Seven more drive **real** `@triton.jit` kernels
through the pinned Triton — including an attention layer's forward *and* backward,
with the gradients checked against a hand-written numpy autograd — and skip, with
an actionable message, on any machine that has not built it (§8.1), so the suite
states the dependency without acquiring it.

Two strategy choices are worth stating. Sizes divide *nothing*: `129x257x65`
divides neither a block dimension nor the 8x8 fragment, softmax rows are 100
columns wide in a 128-wide block, and `5x3x7` is a dot smaller than a single
fragment in every dimension. And error paths are tested as first-class behaviour
— roughly a quarter of the suite asserts that something is refused, with the
right message.

---

## 7. Evaluation

### 7.1 Coverage

Regenerated with `swift test --enable-code-coverage`, then
`xcrun llvm-cov report … -ignore-filename-regex "Tests|\.build"`:

| File | Region | Function | Line |
| --- | --- | --- | --- |
| `AxisCloning.swift` | 56.57% | 66.67% | 62.50% |
| `Compiler.swift` | 96.80% | 97.22% | 98.72% |
| `IR/Lexer.swift` | 84.92% | 95.35% | 91.86% |
| `IR/Parser.swift` | 80.79% | 85.71% | 85.24% |
| `IR/TritonIR.swift` | 88.46% | 79.17% | 95.49% |
| `Layout.swift` | 89.49% | 77.00% | 94.40% |
| `MSLEmitter.swift` | 84.53% | 86.33% | 91.97% |
| `Runtime.swift` | 86.11% | 100.00% | 92.31% |
| **Total** | **83.43%** | **85.23%** | **89.50%** |

`AxisCloning` is the least covered file, at 57% region, and the reason is worth
naming rather than rounding away: most of what is uncovered there is two
mechanical operand-remapping switches with one case per operation in the language,
of which only the handful an index-arithmetic cone actually contains is exercised.
The behaviour is covered; the switch arms are not.

Over ~6,900 lines of Swift core and ~3,900 lines of tests. The 282-line Python
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

### 7.2 Matmul throughput

The matmul tutorial kernel, lowered by this backend, against
`MPSMatrixMultiplication`, f32, square, best configuration per size out of the
`tmbench` sweep over `BLOCK_M x BLOCK_N x BLOCK_K`, `num_warps`, register
blocking, tile padding and double buffering. Dispatches are packed into one
command buffer until each timed sample is ~25ms of GPU work, so submission
overhead is not most of what either side is measured doing; median of three.

On an **Apple M1 Max** (Mac Studio; measured peaks 6.2 TFLOP/s f32, 371.5 GB/s),
which is the machine to believe — it is not thermally constrained and its MPS
readings are stable:

| size | best configuration | triton-metal | MPS | ratio | round 1 | baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 64x64x16, `num_warps=8` | 3.21 TFLOP/s | ~1.85 TFLOP/s | — | 92% | 62% |
| 1024 | 64x64x16, `num_warps=8` | 4.33 TFLOP/s | ~5.70 TFLOP/s | **76%** | 50% | 33% |
| 2048 | 64x64x16, `num_warps=8` | 4.66 TFLOP/s | ~6.10 TFLOP/s | **76%** | 50% | 33% |
| 4096 | 64x64x16, `num_warps=8` | 4.64 TFLOP/s | ~6.10 TFLOP/s | **76%** | 52% | 32% |

On an **Apple M1 Pro** (laptop), where MPS readings move with thermal state:

| size | best configuration | triton-metal | MPS | ratio | round 1 | baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 64x64x16, `num_warps=8` | 2.27 TFLOP/s | ~2.77 TFLOP/s | **82%** | 50% | 34% |
| 2048 | 64x64x16, `num_warps=8` | 2.33 TFLOP/s | ~2.83 TFLOP/s | **82%** | 50% | 33% |

**A second optimisation round took ~50% of MPS to 76% on the M1 Max** (1.52x on top
of round 1's 1.55x, so 2.3x over the original kernel), which puts the backend inside
the 62–82% band published Triton reaches against cuBLAS on its native target. The
60% this round aimed at is cleared at every size where the measurement means
anything, and the winning shape moved to `64x64x16` at `num_warps=8` everywhere.
The laptop's 82% is flattered by a warm MPS reading — the same MPS configuration
read 3.38 TFLOP/s earlier in the session, which would make it 69% — which is why
the M1 Max is the machine to believe. Run-to-run variation there is about ±2%, once
a configuration has been measured twice: the *first* measurement at a size runs
cold and can read 20% low, which is a trap this round fell into before noticing.

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
| **round 2** | | |
| + zero accumulator born in registers, no prologue at all | 3025 | +0.1% |
| + 2-D-distributed epilogue (all 512 threads participate, not 128) | 3094 | +2% |
| + shared-column wave mapping, collapsing identical operand loads | 3551 | **+15%** |
| + `float4` staging runs behind a runtime contiguity/alignment check | 4271 | **+20%** |
| + two fragment rows per simdgroup where the score ties (at 64x64) | 4640 | **+8%** |

#### What round 2 found

The round-1 discussion below ends by saying an instruction-slot account ranks
fixes in the wrong order. Round 2 is the same lesson twice more, in both
directions.

**The prologue and epilogue were named as the largest suspected remaining cost.
They were worth 2%.** Both are real inefficiencies — a zero accumulator written to
threadgroup memory and read straight back, an epilogue in which 128 of 512 threads
did all the work — and both are now gone, and together they moved 1993→3094, i.e.
almost nothing. They are once-per-program costs against a contraction loop of 128
steps, which the estimate had not weighed.

**The largest single change was not on the list at all.** With one output fragment
per simdgroup per wave, the emitter used to give wave `w` of simdgroup `s` the flat
block `s + wS` of the output grid and recover its coordinates by division. Those
coordinates are not independent: when the grid's width divides the simdgroup count,
every wave of a simdgroup sits in the *same column*, and so wants the same B
fragment. Making that structural — emitting the column index once, then
deduplicating operand loads by address — turned a contraction step's 16
`simdgroup_load`s into 9 and was worth **15%**, with the mapping, the arithmetic
and the results bit-identical. It went unexamined because the loads it removes are
the same loads register blocking had already been measured not to care about; what
register blocking changes as well, and what evidently costs more, is the mapping.

**Vector staging was worth 20%, but mostly not for the reason it is usually done.**
Replacing four scalar loads with one `float4` inside the element loop measured
**+2%**. Putting the same vector load behind a branch around the *whole run* —
so the fast path skips the per-element address arithmetic too — measured **+20%**.
An intermediate version whose guard re-derived both endpoints of the run in full
measured **-30%**. The load count was never the point; the address arithmetic, and
how cheaply the guard can be computed, were.

#### Two techniques that did not transfer

This is the part of the result we think is worth publishing.

**Scope.** Both results, and the attention crossover in §7.2b below, are
**M1-generation** measurements — an M1 Max and an M1 Pro. We give architectural
reasons for them (Metal has no `cp.async`; an Apple GPU hides latency with
occupancy; simdgroup-matrix fragments move through threadgroup memory rather than
named lanes), and those reasons are properties of a generation, not of Metal. The
M3 and M4 GPUs reworked the memory hierarchy, and we have not measured on them. We
would expect the *direction* of the double-buffering result to survive as long as
there is no copy engine and the register-blocking result to be the more fragile of
the two, but that is a prediction, not a measurement. Re-running it is one command
(`tmbench --sweep full`), and doing so on M2/M3/M4 parts is the first item of
future work.

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
simdgroup, so this is not register pressure. Round 2 gave us a reason to re-measure
— deduplicating operand loads hands 1x1 blocking most of the traffic a bigger block
was supposed to save — and the full sweep was re-run at the new operating point:
1x1 now measures 3617 GFLOP/s against 2494 for 2x2, 2500 for 4x2 and 1747 for 4x4.
The inversion widened from 28% to 45%. The operand loads a bigger block
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

### 7.2b Attention throughput

FlashAttention-2 forward against the unfused composite it replaces — `Q K^T` and
`P V` as two `MPSMatrixMultiplication`s with an `MPSMatrixSoftMax` between them,
one head at a time, through a real `S x S` f32 score matrix. That is the honest
comparison for a fused kernel: the composite runs the same arithmetic through
Apple's own GEMM, which is faster than ours, but it writes the score matrix to
device memory and reads it back twice.

Same methodology as §7.2 — dispatches packed to ~25ms per sample, median of three.
FLOPs counted as `4 * S^2 * D` per head; the softmax is uncounted on both sides.

| shape | M1 Max fused | composite | ratio | M1 Pro fused | composite | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | **1358 GF** | 381 GF | **357%** | **891 GF** | 513 GF | **174%** |
| `b1 h8 s1024 d64` f32 | **1678 GF** | 1575 GF | **107%** | 948 GF | 1244 GF | 76% |
| `b1 h16 s2048 d64` f32 | 1761 GF | 2586 GF | 68% | — | — | — |
| `b1 h8 s512 d64` f16 | **1680 GF** | 702 GF | **239%** | **772 GF** | 531 GF | **146%** |
| `b1 h8 s1024 d64` f16 | **2089 GF** | 1582 GF | **132%** | 1195 GF | 1231 GF | 97% |

The shape of the result is the point. At `s512` the composite's GEMMs are small
and there are three dispatches per head, so fusion wins by 1.5–3.6x. By `s2048`
the composite's GEMMs reach MPS's own peak and the traffic fusion removes stops
being what decides. The crossover sits near `s1024` on both machines and f16 moves
it out.

Two caveats we would rather state than have found. The composite's `s512` readings
move a lot run to run — 478, 513 and 588 GF in one M1 Pro session — because 24
small dispatches are mostly submission overhead, so read those ratios as "clearly
ahead" rather than as a number. And ours is not the only possible composite:
metalscope's bench measured MPS-composite SDPA at ~744 GF on an M1 Pro for the same
shape, faster than the 513–588 GF we measure, so a better-written composite exists
and the `s512` margin against *it* would be nearer 1.2x than 1.7x.

### 7.2c Attention backward throughput

The three backward kernels against the strongest composite MPS can express: five
`MPSMatrixMultiplication`s with an `MPSMatrixSoftMax` and an
`MPSMatrixSoftMaxGradient` between them, through **two** live `S x S` f32
matrices. Both sides are credited with the same arithmetic — `5 * 2 * S^2 * D` per
head, the five GEMMs the mathematics needs — although the fused side runs
*seven*, because `Q K^T` is recomputed in each direction, plus a `Delta` pass over
`O` and `dO`. Counting the mathematics rather than the instructions is the
comparison that flatters us least.

| shape | M1 Max fused | composite | ratio | M1 Pro fused | composite | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | **977 GF** | 515 GF | **190%** | **523 GF** | 178 GF | **294%** |
| `b1 h8 s1024 d64` f32 | 1054 GF | 1519 GF | 69% | 592 GF | 1203 GF | 49% |
| `b1 h16 s2048 d64` f32 | 1172 GF | 2494 GF | 47% | — | — | — |
| `b1 h8 s512 d64` f16 | **1293 GF** | 588 GF | **220%** | **648 GF** | 181 GF | **357%** |
| `b1 h8 s1024 d64` f16 | 1419 GF | 1445 GF | 98% | 725 GF | 1042 GF | 70% |

**The crossover is earlier than the forward's**, and the extra recomputation is
what buys the composite that. The forward is still at 107% by `s1024` on an M1
Max; the backward is at 69% there and 47% at `s2048`. Two reasons, in the order
they matter. The fused side runs seven GEMMs' worth of arithmetic against the
composite's five, so at large `S` — where the composite's GEMMs reach MPS's peak
and score-matrix traffic stops deciding anything — it is paying a 40% arithmetic
premium for traffic that no longer costs. And the `dK`/`dV` kernel is the heaviest
thing this emitter has lowered: four `tt.dot`s and two register-resident
accumulators, at block shapes small enough that two `BLOCK_N x HEAD_DIM` tiles and
an operand arena fit in 32KB.

The honest caveat, again on the composite side: the same f32 composite measured
1519 and 1445 GF at `s1024` minutes apart on the M1 Max, so those ratios are worth
about ±5 points. Where a backward pass is actually used — `s512` and below, and
f16 — the fusion is 1.9x to 3.6x ahead, which is the same shape of answer the
forward gives.

**What one change was worth.** Reducing across a row's lane group rather than
across the whole threadgroup took the best M1 Pro configuration at
`b1 h8 s512 d64` from **174 GF to 674 GF** — 3.9x — with nothing else altered.
This is the same category of result as §7.2's two inversions: the cost that
dominated was not arithmetic and not memory traffic but *serialization*, and no
instruction-slot account would have found it.

### 7.3 Where the gap is now

The pre-optimisation account was counted in instruction slots: for the then-best
64x64x32 / 16-simdgroup shape, one K block cost each simdgroup 16
`simdgroup_multiply_accumulate`s against 32 operand `simdgroup_load`s and 8
accumulator load/stores, plus ~70 scalar ops per thread to stage 8 elements — the
multiply-accumulates about an eighth of the issue slots. That account ranked
operand traffic first and staging third.

**It ranked them wrong**, and how it did is the most transferable thing in this
section. The three costs were not independent: staging was hiding behind the
operand loads, and once staging got cheaper the operand-load fix that had been
worth 10% became worth under 1% (§7.2). An issue-slot count tells you what the
slots are spent on; it does not tell you which spend is on the critical path.

Two of the three candidates that account produced are now done, and the ordering it
gave them was again wrong: the prologue and epilogue it ranked first were worth 2%,
and vector staging it ranked second was worth 20%.

At 4.66 TFLOP/s a 2048-cube GEMM takes 3.7ms, and at the 64x64 tile that now wins
its 1024 programs read 1 MiB of operands each — about 1.1 GB through the memory
system, roughly **290 GB/s against a 371 GB/s part**, where the same calculation
gave 145 GB/s at the end of round 1. (Some of that is L2 rather than DRAM, so it is
an upper bound.) At ~75% of the measured 6.2 TFLOP/s compute peak and that close to
the bandwidth wall, what is left to buy is arithmetic intensity, not latency. What
we would try next:

1. **Keeping the accumulator out of threadgroup memory entirely.** A 128x128 tile
   halves the operand traffic of the 64x64 one that now wins, and that traffic is
   now the binding constraint; its f32 accumulator alone is 64KB, twice the budget.
   The route is an epilogue that streams register fragments out in panels instead
   of through one full-size tile. Untried — the target was reached without it, and
   it remains much the largest of the three.
2. **Vector staging for the A tile as well.** Only B is vectorised at the winning
   shape: a run of four columns of a 64x16 A tile would leave three quarters of the
   threadgroup idle, and forcing it measures 3069 against 4223 GFLOP/s. The fix is a
   staging distribution that lets a narrow tile hand out runs of four without
   idling threads, not a longer run.
3. **Specialising the run mask away.** The vector guard re-checks the mask at the
   run's last column every K step; when `BLOCK_N` divides `N` it is statically true.

None of the three is a layout redesign, and two of them are local to `emitDot`.

---

## 8. Limitations and future work

### 8.1 The Triton pin, and what real IR cost

Until this milestone the backend consumed Triton IR *text* and was tested against
fixtures written from Triton's documentation. It is now pinned to **v3.7.1** — the
release `pytorch/pytorch` release/2.12 pins — and driven by Triton itself: the
vector-add, fused-softmax and matmul tutorials run as Python source and match
numpy on an M1 Max.

Two practical findings, both about the distance between an interface you read and
an interface you run.

*The build is the barrier, and it is smaller than it looks.* Triton publishes no
macOS wheel, so a plugin backend on a Mac means building Triton from source. That
sounds like the expensive part and is not: the prebuilt `macos-arm64` LLVM Triton's
CI publishes exists, and the whole build is ~9 minutes on an M1 Max with **no
patch to Triton**. The one thing that does not work is the obvious economy —
building *only* the plugin backend. 3.7.1's core sources include the in-tree NVIDIA
and AMD dialects' tablegen'd headers (`InstrumentationToLLVM.cpp`, `gluon_ir.cc`),
so a `TRITON_CODEGEN_BACKENDS=""` build does not compile. Skipping their runtime
downloads is fine; skipping the backends is not. Also non-obvious: a plugin must
supply a C++ `init_triton_<name>` symbol, because Triton's `main.cc` expands the
backend-name tuple into declarations and calls. Ten lines of C++ are unavoidable
for a backend that is otherwise entirely Swift.

*Real IR differs from documented IR in small, specific ways.* Triton prints with
debug info enabled, so every operation, the module, the function **and every
function argument** carry a trailing `loc(...)`, and SSA values are named after
the Python variables that produced them (`%offsets_1`, not `%3`). Only the
argument locations were unhandled: one call to `skipTrailingLocation` in the
parser, and the entire fixture corpus was otherwise accurate. The emitter needed
two generalisations, both from the same source-level line —
`ptrs += BLOCK_K * stride` inside a `tt.dot` loop — which the canonicalizer
rewrites into shapes the pointer-advance check had not anticipated: a dense splat
constant, and a scalar product recomputed inside the loop from loop-invariant
operands. Neither is exotic; both were invisible without a real release.

What the tutorials also did was surface a restriction no fixture could have
produced: the matmul tutorial is refused whenever two block sizes are equal, and
the reason is in the IR a release prints rather than in the layout model. §8.4.

### 8.2 FlashAttention-2, and what it took

**FlashAttention-2 forward now runs**, and what it took is the most transferable
result in this paper, because what it took was not what we had written down.

We had named two blockers. *`tt.trans`* was the easy one and turned out easier
than predicted: we had planned to pass `simdgroup_load`'s `transpose_matrix` flag,
but a transpose is a relabelling of which block axis each dimension indexes, so
`K^T`'s tile is staged over `(HEAD_DIM, BLOCK_N)` directly and **no code is
emitted for the op at all**. *Spilling a per-lane tensor carried across a
cross-lane loop* was real and landed roughly as described — a tile, a shadow tile
written where the new value is defined rather than at the `scf.yield`, and a copy
between two barriers at the end of the body — plus one case we had not seen: when
the loop yields a `tt.dot`'s result and the dot's accumulator is a per-lane
function of the carried value (`acc = tt.dot(p, v, acc * alpha[:, None])`, FA-2's
rescale), the carried tile can *be* the dot's accumulator tile and the rescale
becomes the staging pass that fills it — no copy, one tile, at the cost of
register residency.

Three things we had not named were larger than either.

*The loop nest had to stop being one nest.* FA-2's `p` spans
`(BLOCK_M, BLOCK_N)` and its `acc` spans `(BLOCK_M, HEAD_DIM)`. Neither contains
the other, and the old model — one index space of rank R, walked by R nested
loops, innermost distributed — can only serve both by putting all three axes in
one nest, which recomputes each tensor `HEAD_DIM` or `BLOCK_N` times over. Nests
had to become a *tree*: each value is emitted under exactly the axes it varies
along, and sibling nests share their common prefix. That is a change to how every
statement in the emitter is placed.

*Axis assignment had to become unification.* The old pass seeded every full-rank
tensor with the identity map and propagated. FA-2's two dots share their axes
crosswise — the first contracts over the head dimension the accumulator iterates,
the second over the key block the softmax iterates — so no fixed
(M, N, fresh-K) assignment describes it. Every dimension now gets an axis variable
and the ops that relate two tensors merge them. Two rules in that pass were found
by running real Triton IR through it rather than by thinking: a size-1 dimension
must carry no axis identity (the matmul tutorial broadcasts one row mask into both
a `BLOCK_M x BLOCK_K` and a `BLOCK_M x BLOCK_N` tensor, and unifying its second
dimension with both declares the contraction axis and the column axis to be the
same one), and two varying dimensions of one value must not collapse onto one axis
(which is what a single `tl.arange` expanded into both a row and a column index
does, and Triton's CSE hands you that whenever `BLOCK_M == BLOCK_N`).

*A reduction result a later dot has to read.* Dot operands are rebuilt inside the
dot's own staging loops, and a `tt.reduce` is the one thing that cannot be rebuilt
there — the fold has already happened. FA-2's `p = exp2(qk * scale - m_ij)` is the
second dot's left operand and `m_ij` is a row reduction, so the row maximum is
spilled into a `BLOCK_M`-float array where it is computed and read out of it in
the staging nest.

We also had one plain factual error in the earlier version of this section: FA-2's
`m_i` and `l_i` are not scalars. They are `BLOCK_M`-wide per-row vectors, which in
this model are rank-1 tensors and needed the same spill machinery `acc` did. Only
a rank-1 kernel's online softmax carries genuine scalars.

**Attention throughput, and where the fusion stops paying.** Against the unfused
composite (`Q K^T` + `MPSMatrixSoftMax` + `P V`) the fused kernel is 357% at
`b1 h8 s512 d64` on an M1 Max, 107% at `s1024`, and 68% at `s2048`. The crossover
is where MPS's GEMMs get big enough to reach their own peak, at which point the
score-matrix traffic the fusion removes stops being what decides. The single
change that mattered most on our side was worth **3.9x** and had nothing to do
with the dots: a `tt.reduce` used to fold across the whole threadgroup, so an
online softmax over key blocks ran `2 * BLOCK_M` threadgroup-wide folds per
iteration, serially, with 128 threads cooperating on 32 values at a time. Handing
each row a lane group inside one simdgroup makes every row fold at once with no
barrier at all. What now caps the kernel is threadgroup memory: the f32
accumulator and the f32 score tile are both live across the whole iteration, so
`BLOCK_M` above 32 does not fit at `HEAD_DIM = 64`, which caps arithmetic
intensity. Keeping the score tile in registers between the two dots is the same
change the GEMM wants for its accumulator.

### 8.3 The backward pass, and what it cost the compiler

The forward pass made the backend useful. The backward pass is what makes an
attention layer *trainable*, and the result worth reporting is how little new
compiler machinery it needed.

**Shape.** Recompute-based, as Triton's tutorial is. The forward stores one
`BLOCK_M`-wide vector per query block — the per-row logsumexp in log2 units — and
the backward rebuilds `P` from it with the same `exp2(qk_scale * Q.K - M_i)` the
forward evaluated. The `S x S` score matrix never exists on either side. Three
kernels: `Delta = dO.O`, then `dQ` over key blocks, then `dK`/`dV` over query
blocks. `dQ_i` sums over key blocks and `dK_j`/`dV_j` over query blocks, so no
single program owns both ends; we run the two directions as separate programs
rather than accumulating `dQ` with `tt.atomic_rmw fadd`, and the argument that
decides it is determinism, not speed.

**One emitter change.** A loop that puts an accumulator in simdgroup registers
frees that accumulator's tile for the loop's duration, and the emitter let only
*that dot* stage into it, sizing the arena as the **sum** of every dot's operand
tiles. Now every dot in the loop stages there and the arena is the largest single
dot's footprint — sound for the same reason the kernel-wide shared arena is: a dot
ends with a `threadgroup_barrier` after its arithmetic and the next begins by
staging, so two dots' operand tiles are never live at once. Summing overran
Metal's 32KB at every block shape worth having, because the `dQ` loop stages three
dots and the `dK`/`dV` loop four. That is the whole compiler delta. Everything
else the backward needs, the forward had already forced: three block axes,
`tt.trans` as a relabelling, dots reading other dots' result tiles, and
loop-carried dot accumulators. Real Triton IR for the backward then needed no
parser or emitter change at all.

**Verification.** Every gradient is checked twice and independently: against an
analytic CPU reference in `Double`, and against central **finite differences** of
the forward, which know nothing of the backward formulas — the second is what
would catch a reference wrong in the same way the kernel is. Eight shapes,
including `s127`, `s33`, `d80` and `d20`, plus two f16-in/f32-accumulate ones;
`num_warps` 1..8; and the real forward's own logsumexp and output feeding the
backward end to end. The tolerances are **derived, not tuned**: for a gradient
element that is a sum of `n` terms the recursive-summation bound is
`(n-1) * u * sum|term|`, so the per-tensor bound is `n * u * A` with `A` the
cancellation amplification computed from the `Double` reference itself and `u`
either `2^-24` (f32) or `2^-11` (the f16 dot operands). Nothing in the suite is a
tolerance chosen because a smaller one failed.

**End to end.** `python/examples/attention_training.py` writes the forward and the
three backward kernels as `@triton.jit` source, compiles them with Triton 3.7.1's
own frontend and MLIR passes, and trains: gradients agree with a hand-written
numpy autograd to **6.2e-07** relative and with finite differences to **3.7e-08**,
and eight steps of gradient descent on `0.5*||O - T||^2` take the loss from
**1541.73 to 1482.86** with forward and backward both on the GPU.

### 8.4 Where equal block sizes actually collided

`Layout.swift` unifies axis *variables*, one per dimension of every tensor; two
axes of the same size are two axes, and nothing in the inference keys on an
extent. The sharing that made equal block sizes fail is upstream, in Triton's CSE:
with `BLOCK_M == BLOCK_N`, `tl.arange(0, BLOCK_M)` and `tl.arange(0, BLOCK_N)` are
the same expression, so the ttir contains **one** `tt.make_range` with both
`offs_am` and `offs_bn` as `arith.addi`s over it. Expanding one at dimension 1 and
the other at dimension 0 then declares the row axis and the column axis to be the
same, and the accumulator is a diagonal of itself. The refusal is correct about
the IR in front of it; what is wrong is that the IR says something the kernel does
not mean.

Since every operation on that path is pure, the fix is to un-share it: give each
placement its own copy of the arithmetic and re-run the inference. Two phases,
because two equal block sizes and three equal block sizes fail differently — each
`tt.expand_dims` of a conflicted rank-1 class gets its own producer cone, and each
consumer of a multiply-used `tt.broadcast` gets its own copy of the widening,
which is what `(64, 64, 64)` needs on its own (there CSE also shares
`broadcast(1x64 -> 64x64)` between `B`'s column offsets, where the leading
dimension is the contraction, and `C`'s, where it is the row block). A
`tt.broadcast` emits no code, so that copy is free, and neither phase can invent
an axis: a copy that really is the same axis is unified again downstream by the
dot's pinning or by whatever expression reads both.

The pass runs **only** on the one diagnostic it can address — a dedicated
`CoreError.axisCollapse` — because a retry that fired on any layout failure would
let a rewrite silently change what a kernel computes. That narrowness is load
bearing: a mixed-tensor-length kernel and an uninferable-layout kernel both become
lowerable under an unrestricted retry, and both should stay refused. With real
Triton, the matmul tutorial agrees with numpy exactly at `(128, 64, 32)`,
`(64, 64, 32)`, `(64, 64, 64)`, `(32, 32, 32)` and `(64, 32, 32)`.

The transferable part is not about Metal: a layout model can be right about the IR
in front of it and still be reasoning from an artefact of the frontend, and the
only way to see that is to read what a real release prints.

**Layout conversions.** The inference in `Layout.swift` is the honest, small
version of what ttgir's layout system does. It maps tensor dimensions to block
dimensions, but says nothing about which lane holds which element — exactly what
an mma layout must pin down. This is also why `tt.dot` stages through threadgroup
memory rather than shuffling fragments between lanes: the emitter does not know
where a fragment's elements live.

**No CI against the pin.** The plugin surface is now read off v3.7.1's signatures
rather than the approximate 3.x layout (§8.1), but nothing yet re-verifies it when
Triton moves. A check that rebuilds against the pinned tag is what would keep the
adapter from silently drifting out of date.

**Type coverage.** No `bf16`, which is the type real training actually uses and is
now the most urgent gap in this list. No `f64`, and there will not be one: Metal
has no `double` type at all. Block pointers (`!tt.ptr<tensor<…>>`) are not
lowered.

**Atomics are 32-bit, unordered, and partly synthesised.** `tt.atomic_rmw` and
`tt.atomic_cas` lower on `f32` and `i32` device pointers, and the four gaps are
Metal's rather than ours: there are no 16- or 64-bit atomics; there is no float
`atomic_fetch_max_explicit`, so `max`/`min` on `f32` go through a compare-exchange
loop on the bit pattern (a load plus one CAS uncontended, degrading linearly under
contention); there is no strong compare-exchange, so `tt.atomic_cas` retries
around the weak one; and MSL declares exactly one memory order —
`memory_order_acq_rel` is not a declared identifier — so Triton's `sem` and
`scope` are parsed and dropped. The operation itself is atomic; nothing else about
it is ordered, and a kernel relying on an atomic's release edge to publish
ordinary stores is not correctly lowered here. Float atomics also reorder, which
we measure rather than hide: five runs of a contended `fadd` reduction agree
within the recursive-summation bound `(k-1) * 2^-24 * sum|x_i|` and not more
tightly than that.

**The backward's crossover is early, and the recomputation is why.** The split
two-pass formulation recomputes `Q K^T` in both directions — seven GEMMs' worth
where the mathematics needs five — which is a 40% arithmetic premium paid to keep
every gradient deterministic and every accumulator in registers. At `s512` that is
plainly the right trade (190% of the composite); by `s2048` it is plainly the
wrong one (47%). The `tt.atomic_rmw fadd` alternative for `dQ` would remove one of
the two recomputations at the cost of a non-deterministic gradient, and it is now
implementable — the atomics exist. It is not implemented or measured, and the
crossover is the number that would decide it.

**Rank > 2 is untested.** Higher ranks lower through the same nested-loop
machinery and are believed to work — but "believed to work" is not a claim this
paper makes about anything else in it, so it is listed as a limitation.

**Occupancy of outer block dimensions.** Mostly closed: a `tt.dot`'s staging
loops spread over both tile dimensions, a dot kernel's per-lane nest does too, and
a kernel with a reduction now splits the lane index so that several rows reduce at
once. What is left is the elementwise nest of a kernel that has neither — which is
also the only kind of kernel not contractually launched at the reported
threadgroup size, so the split cannot simply be baked in.

**What "training-capable" does not include.** The FA-2 backward pass runs and is
verified (§8.3), which makes gradients possible; it does not make this a training
framework. There is no autograd — a caller writes or generates the backward
kernel, exactly as a Triton user does. There are no optimizer-state kernels:
Adam's moment updates are ordinary elementwise work and would lower through the
existing machinery, but none is written, run or measured here, so we claim
nothing about them. There is no dropout, because there is no `tl.rand` and no RNG
lowering at all. There is nothing distributed. And gradient checkpointing,
mixed-precision loss scaling and the rest of a training loop's furniture live
above this layer entirely.

**Re-measuring on newer Apple silicon.** Every throughput number and every
"this CUDA technique inverts here" claim in this paper is an M1-generation
measurement, on an M1 Max and an M1 Pro. The reasons we give are architectural,
but they are properties of a generation: the M3 and M4 GPUs reworked the memory
hierarchy, and we have not run on them. Re-running is one command
(`tmbench --sweep full`, `tmbench --attn`), and it is the cheapest piece of future
work in this list.

**No conformance suite yet.** The tests here are ours. Porting a subset of
Triton's own `test_core.py` against numpy references is what would turn
"our tests pass" into "this is a Triton backend".

---

## 9. Conclusion

triton-metal is a working spine with real kernels on it. Triton IR text goes in;
readable MSL, a Metal library, a compute pipeline, unified-memory buffers, a
dispatch, and correct results come out — driven from Swift or from any language
that can call C. The fused-softmax and matmul tutorial kernels both match CPU
references at sizes that divide neither the block shape nor the simdgroup
fragment, at every block shape including the ones whose sizes collide, and
FlashAttention-2 runs in **both directions**: an attention layer written as
`@triton.jit` source trains on a Mac GPU, its gradients agreeing with a
hand-written numpy autograd to 6.2e-07 and with finite differences to 3.7e-08.

Three design decisions carried most of the weight. Emitting textual MSL made
every lowering bug readable. Keeping all logic in Swift behind a C ABI, with a
shim that computes nothing, kept the core usable from outside Python and stopped
launch geometry from leaking into two places. And inferring the tensor-dimension
to block-dimension mapping to a fixed point, rather than assuming it, allowed
rank-deficient tensors, broadcasting and a `tt.dot` contraction axis to coexist
without kernel-level annotation.

Two results deserve restating plainly. The numerics are good: fast math off,
`precise::` where Metal has it, a softmax that agrees with a CPU reference to about
one ULP, and gradients whose tolerances are derived from a summation bound rather
than chosen to pass. The performance is good rather than adequate: 76% of MPS on
a GEMM, inside the band published Triton reaches against cuBLAS on its own target;
357% of the MPS composite on attention forward at `s512` and 190% on the backward,
with a measured crossover rather than a headline ratio.

What stands between this and being *used* is no longer a research problem. The
release is pinned and driven, both directions of the kernel that mattered run, and
the remaining distance is `bf16`, a conformance subset, optimizer and RNG kernels
that nothing in the design objects to, and measurements on silicon newer than M1 —
a list of known work rather than of open questions. The honest boundary is that
this compiles and runs the kernels a training loop is made of; it is not itself a
training framework.
