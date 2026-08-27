# triton-metal

A [Triton](https://github.com/triton-lang/triton) compiler backend targeting **Apple Metal**,
with a **Swift core** and a deliberately thin Python shim.

Triton is how the ecosystem writes custom fused kernels (FlashAttention variants, MoE
routing, quantization kernels) without hand-rolling CUDA — and Triton targets only
CUDA/ROCm, which is CUDA's single biggest portable-code moat. This backend runs that
same research code unmodified on hardware a team already owns: real `@triton.jit`
through the pinned Triton 3.7.1, exact results, no patch to Triton — with the matmul
tutorial at **76% of Apple's own `MPSMatrixMultiplication`** and FlashAttention-2
beating the MPS composite it replaces at `s1024` and below in **both directions**.
An attention layer written as `@triton.jit` source now **trains** on a Mac GPU —
forward, backward and a gradient-descent step — with gradients checked against a
hand-written numpy autograd and against finite differences;
[docs/WHITEPAPER.md §2](docs/WHITEPAPER.md#2-value-proposition-breaking-the-portable-kernel-moat)
has the full case and its boundaries, including which CUDA optimisation techniques
transfer to M1-generation Apple silicon and which invert.

## Documentation

| | |
| --- | --- |
| [docs/USAGE.md](docs/USAGE.md) | **Start here.** Runnable Swift, C-ABI and Python examples (every one executed before it was written down), the supported IR subset, exact error behaviour, and concrete starting points for the work that is still missing. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The lowering pipeline, the execution and layout model, the full IR subset and C ABI reference, matmul throughput, and the hard parts ranked. |
| [docs/WHITEPAPER.md](docs/WHITEPAPER.md) | Motivation, the competitive case against CUDA and where CUDA remains ahead (§2), related work, design and implementation rationale, and an evaluation with real coverage and benchmark numbers. |
| [docs/COMPARISON.md](docs/COMPARISON.md) | Measured efficiency vs. published Triton-on-CUDA numbers; the transfer matrix (which CUDA GEMM techniques port to M1-generation Apple silicon and which invert); and fused attention vs. the unfused MPS composite. |

## Structure & language policy

Everything that does real work is Swift, exposed through a C ABI:

```
Sources/TritonMetalCore/   Swift: MSL emission, metallib compilation, Metal runtime,
                           kernel launch — exported as tm_* symbols in libtritonmetal.dylib
python/triton_metal/       ctypes shim only. Exists because Triton's backend discovery
                           imports a Python module; contains no logic and never will.
python/plugin/             What Triton's TRITON_PLUGIN_DIRS points at: the vendored
                           BaseBackend/DriverBase adapters, plus metal.cc — a ~10-line
                           registration stub, the entire C++ surface of the project,
                           required because Triton's main.cc calls init_triton_metal().
```

Rebuilding Triton's Python frontend itself is the one dependency that justifies the shim.

## Pipeline

```
@triton.jit  →  ttir  →  MSL  →  MTLLibrary  →  compute pipeline  →  dispatch
     │           │        └──────────┴──────────────┴───────────────────┴─ Swift core
     │           └─ Triton's own passes (inliner, canonicalizer, CSE, loop unroll)
     └─ Triton's own frontend, from the pinned release
```

Note what is *not* there: no `ttgir`. Triton's TritonGPU passes lower toward LLVM
with a target profile Metal does not have, so the Swift core does its own layout
inference straight from `ttir` — see
[docs/ARCHITECTURE.md §Compatibility](docs/ARCHITECTURE.md#compatibility).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the lowering plan and the known
hard parts (barrier semantics, layout conversions, spilling a loop-carried tensor).

## Status

**Plugged into real Triton.** `import triton`, `@triton.jit`, `kernel[grid](...)`
— Triton 3.7.1's own frontend and MLIR passes, this backend, a Mac GPU. The
[vector-add](python/examples/vector_add.py), [fused-softmax](python/examples/fused_softmax.py)
and [matmul](python/examples/matmul.py) tutorials are run as **Python source**, not
as pre-baked IR, and checked against numpy on an M1 Max:

```
triton 3.7.1 on Apple M1 Max
vector add, n=98432:              max |triton - numpy| = 0
fused softmax, (1823, 781):       max |triton - numpy| = 1.49012e-08
matmul 256x256x256 (tl.dot):      max |triton - numpy| = 0      (every block shape,
                                    including (64,64,32), (64,64,64), (32,32,32))
attention layer, h2 s96 d64:      dQ/dK/dV vs numpy autograd    <= 6.2e-07 relative
                                  vs central finite differences <= 3.7e-08 relative
                                  8 SGD steps: loss 1541.73 -> 1482.86
```

Triton ships no macOS wheel, so this means building Triton from source with
`TRITON_PLUGIN_DIRS` pointing at `python/plugin/` — about **9 minutes**, no CUDA
toolchain, no Xcode, no patch to Triton. Recipe:
[docs/USAGE.md §Building Triton with the Metal backend](docs/USAGE.md#building-triton-with-the-metal-backend).

**Working end-to-end spine, with real kernels on it.** Triton IR text -> MSL ->
`MTLLibrary` -> compute pipeline -> unified-memory buffers -> dispatch -> results,
driven from Swift, from Python over the `tm_*` C ABI, or now from Triton itself.
Triton's fused **softmax**, **matmul** and **FlashAttention-2 forward and backward**
kernels all run on the GPU and match CPU references — at f32 and at
f16-in/f32-accumulate, at sizes that divide neither the block shape nor the 8x8
simdgroup fragment (`129x257x65` and `h2 s127 d64` among them), for attention on
scores large enough that a naive `exp` overflows, and for the gradients against
finite differences of the forward as well as against an analytic reference.

- **Lowering**: a recursive-descent parser for Triton's pretty MLIR syntax (plus
  the generic form for `tt.reduce`, which carries a region) and an MSL emitter.
  Supported subset: 1-D and 2-D tensors with
  `tt.expand_dims`/`tt.broadcast`/`tt.trans`,
  masked `tt.load`/`tt.store` at any rank, the `arith` integer/float/compare ops,
  all the `arith` conversions, `math.*` unary ops, `arith.select`, `scf.for`/
  `scf.if` with `iter_args` and multiple results, `tt.reduce` (add/max/min) over
  the innermost axis, `tt.dot`, and `tt.atomic_rmw`/`tt.atomic_cas` on `f32` and
  `i32` device pointers. Anything else fails with the op name and
  source line. Full list:
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §Supported IR subset.
- **Layout**: every dimension of every tensor gets an axis *variable*, and the ops
  that relate two tensors **unify** them — elementwise dimension-wise,
  `tt.expand_dims`/`tt.reduce` around the inserted or dropped dimension,
  `tt.trans` through its permutation, `tt.dot` onto (M, K)/(K, N)/(M, N). Each
  value is then emitted in a loop nest over exactly the axes it varies along, and
  sibling nests share their common prefix, so row-uniform work is emitted once per
  row and two tensors on different axis pairs (attention's `p` and `acc`) each get
  their own nest. An axis no materialised value spans — a GEMM's K — is walked only
  inside a `tt.dot`'s staging loops.
- **`tt.dot`**: a second SSA value class for tensors that live in a threadgroup
  tile instead of per-lane registers, `simdgroup_load` /
  `simdgroup_multiply_accumulate` / `simdgroup_store` over 8x8 fragments, and
  zero-padded edge tiles so `BLOCK_M`/`BLOCK_N`/`BLOCK_K` need not be multiples of
  8. Everything feeding an operand is rebuilt inside the dot's own staging loops;
  a pointer advanced across the K loop is strength-reduced rather than carried.
- **Reductions**: a reduction with an axis outside the reduced one hands each row a
  *group of lanes* inside one simdgroup and folds with `simd_shuffle_down` plus a
  `simd_shuffle` broadcast — no threadgroup memory, no barrier, every row at once
  (worth 3.9x on attention). A rank-1 reduction has no outer axis, so it still
  folds within each simdgroup and then across them. A reduction closes the
  distributed loop and the values still needed afterwards are recomputed, not
  spilled. A reduction *inside* an `scf.for` hoists the whole loop to a
  threadgroup-uniform level, which is what an online softmax needs.
- **Loop-carried tensors**: a per-lane tensor carried across a loop containing a
  `tt.dot` or `tt.reduce` is spilled to a threadgroup tile and updated through a
  shadow tile, copied back between barriers at the end of the body. A dot
  accumulator the loop rescales in place — FlashAttention's
  `acc = tt.dot(p, v, acc * alpha[:, None])` — instead *becomes* the dot's
  accumulator tile, and the rescale becomes the staging pass that fills it.
- **Atomics**: `tt.atomic_rmw` (`add`, `fadd`, `and`, `or`, `xor`, `max`, `min`,
  `umax`, `umin`, `exch`) and `tt.atomic_cas`, masked, returning the old value.
  What Metal actually provides was established by compiling every candidate rather
  than read off the spec, and the four gaps are refused by name: no 16- or 64-bit
  atomics, no float `atomic_fetch_max_explicit` (so f32 `max`/`min` go through a
  compare-exchange loop on the bit pattern), no strong compare-exchange, and
  exactly one memory order — so Triton's `sem`/`scope` are parsed and dropped, and
  nothing but the operation itself is ordered. Float atomics reorder; the suite
  measures the spread against the recursive-summation bound rather than hiding it.
- **Backward pass**: FlashAttention-2 backward as three recompute-based kernels —
  `Delta = dO.O`, then `dQ` over key blocks, then `dK`/`dV` over query blocks. The
  forward stores one `BLOCK_M`-wide logsumexp vector per query block and the
  backward rebuilds `P` from it, so the `S x S` score matrix never exists. The
  split two-pass shape rather than a `tt.atomic_rmw` accumulation of `dQ`, because
  it keeps every gradient deterministic and every accumulator in registers.
- **Numerics**: fast math is switched off (Metal defaults it on) and `math.*` maps
  to `precise::` wherever Metal has it; `fast::` is never emitted. Every math op is
  checked against a CPU reference on the GPU.
- **Compilation**: `MTLDevice.makeLibrary(source:)` at runtime (primary path, no
  Xcode tools needed); `xcrun metal` for offline `.metallib` bytes. Never xcodebuild.
- **Runtime**: handle-based C ABI — `tm_compile_msl`, `tm_load_kernel`,
  `tm_alloc_buffer`, `tm_launch`, with `tm_last_error` for diagnostics.
- **Performance**: the matmul tutorial reaches **76% of
  `MPSMatrixMultiplication`** at 1024, 2048 and 4096 square — 4.66 TFLOP/s f32 at
  2048 on an **M1 Max**, 2.33 TFLOP/s on an **M1 Pro** — up from ~33% at the first
  working version, which clears the >50% milestone and lands inside the 62–82% band
  published Triton reaches against cuBLAS on its own target.
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §Matmul throughput has the
  per-machine numbers, what each optimisation was worth measured one at a time, the
  two CUDA-playbook techniques (double buffering, and register blocking past 1x1)
  that did **not** transfer to M1-generation Apple silicon (unmeasured on M2/M3/M4,
  which changed the memory hierarchy), and the one with no CUDA analogue —
  a wave-to-fragment mapping that makes every wave of a simdgroup share its B
  fragment — that was worth more than either. Sweep it yourself with
  `swift run -c release tmbench`, which needs no Xcode.
- **Fused attention**: FlashAttention-2 forward against the composite it replaces
  (`Q K^T` + `MPSMatrixSoftMax` + `P V`, three dispatches per head through a real
  `S x S` score matrix) reaches **357%** of it at `b1 h8 s512 d64` and **107%** at
  `s1024` on an M1 Max, falling to 68% at `s2048` where MPS's own GEMMs reach
  their peak. Also M1-generation only.
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §Attention throughput has both
  machines, the run-to-run variance on the composite side, and the one change that
  was worth 3.9x. Sweep it with `swift run -c release tmbench --attn`.
- **Fused attention, backward**: against the strongest composite MPS can express
  (five GEMMs with `MPSMatrixSoftMax` and `MPSMatrixSoftMaxGradient` between them,
  through two live `S x S` matrices) the three backward kernels reach **190%** at
  `b1 h8 s512 d64` and **220%** in f16 on an M1 Max, crossing over near `s1024`.
  Both sides are credited with the five GEMMs the mathematics needs, although the
  fused side runs seven — `Q K^T` is recomputed in each direction — so the ratio
  understates it. `swift run -c release tmbench --attn-bwd`.
- **Tests**: 184 Swift cases (parser, emitter, layout, casts/math, control flow,
  rank-2, reductions, `tt.dot`, atomics, FlashAttention-2 forward *and* backward,
  axis cloning, error paths, and GPU runs verified against CPU references on an
  M1 Pro) + 15 Python cases including vector-add and softmax round trips, plus 7
  more that drive **real** `@triton.jit` kernels — including a forward+backward
  gradient check — and skip with an actionable message when Triton is not
  installed, plus opt-in MPS benchmarks. 83.43% region /
  85.23% function / 89.50% line coverage of the Swift core — see
  [docs/WHITEPAPER.md](docs/WHITEPAPER.md) §Evaluation for the breakdown and for
  what the uncovered fraction is made of.

Not yet: `bf16` (the type real training uses, and the most urgent gap), block
pointers, `f64` (Metal has no `double` at all), and anything above the kernel
layer — no autograd, no optimizer-state kernels, no RNG and so no dropout.
Atomics are 32-bit and unordered (see above). One narrower constraint the Triton
integration exposed: a kernel argument must be backed by Metal memory
(`MetalBuffer`), since a CPU torch tensor's allocation is not an `MTLBuffer` and
cannot be wrapped without a copy.
[docs/USAGE.md §Implementing what's missing](docs/USAGE.md#implementing-whats-missing)
has concrete starting points for each of these.

## Build & test

```
swift build && swift test
cd python && PYTHONPATH=. python3 -m pytest tests/ -q
```

To run the `@triton.jit` tutorials you also need Triton built against this plugin
(≈9 minutes, once) — [docs/USAGE.md §Building Triton with the Metal backend](docs/USAGE.md#building-triton-with-the-metal-backend):

```
python python/examples/vector_add.py
python python/examples/attention_training.py    # forward + backward + a training step
```

Never `xcodebuild`. GEMM throughput against MPS:

```
swift run -c release tmbench                    # block-shape sweep, 512/1024/2048
swift run -c release tmbench --sweep full       # + register blocking, tile padding,
                                                #   double buffering
swift run -c release tmbench --config 64,128,16,16   # pin one configuration
swift run -c release tmbench --emit 64,64,16,16      # print the kernel it lowers
swift run -c release tmbench --attn                  # FlashAttention-2 forward
swift run -c release tmbench --attn-bwd              # ... and backward
```

`tmbench` is an executable rather than an XCTest case because XCTest ships with
Xcode, and the machine the quoted numbers come from has only the command-line
tools. `TM_BENCH=1 swift test --filter MatmulBenchmark` runs the same sweep
through XCTest where it exists. Coverage:

```
swift test --enable-code-coverage
xcrun llvm-cov report .build/debug/triton-metalPackageTests.xctest/Contents/MacOS/triton-metalPackageTests \
  -instr-profile .build/debug/codecov/default.profdata -ignore-filename-regex "Tests|\.build"
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
