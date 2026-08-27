# triton-metal

A [Triton](https://github.com/triton-lang/triton) compiler backend targeting **Apple Metal**,
with a **Swift core** and a deliberately thin Python shim.

Triton is how the ecosystem writes custom fused kernels (FlashAttention variants, MoE
routing, quantization kernels) without hand-rolling CUDA — and Triton targets only
CUDA/ROCm today. That is CUDA's single biggest portable-code moat. A working Metal
backend lets that entire body of research code run on Apple Silicon without rewriting.

## Documentation

| | |
| --- | --- |
| [docs/USAGE.md](docs/USAGE.md) | **Start here.** Runnable Swift, C-ABI and Python examples (every one executed before it was written down), the supported IR subset, exact error behaviour, and concrete starting points for the work that is still missing. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The lowering pipeline, the execution and layout model, the full IR subset and C ABI reference, matmul throughput, and the hard parts ranked. |
| [docs/WHITEPAPER.md](docs/WHITEPAPER.md) | Motivation, related work, design and implementation rationale, and an evaluation with real coverage and benchmark numbers — including the performance target this milestone missed. |

## Structure & language policy

Everything that does real work is Swift, exposed through a C ABI:

```
Sources/TritonMetalCore/   Swift: MSL emission, metallib compilation, Metal runtime,
                           kernel launch — exported as tm_* symbols in libtritonmetal.dylib
python/triton_metal/       ctypes shim only. Exists because Triton's backend discovery
                           imports a Python module; contains no logic and never will.
```

Rebuilding Triton's Python frontend itself is the one dependency that justifies the shim.

## Pipeline

```
@triton.jit  →  ttir  →  ttgir (Triton's own passes)  →  MSL  →  metallib
                                    │                     └────────┴─ Swift core
                                    └─ Metal target profile: 32-wide simdgroups
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the lowering plan and the known
hard parts (barrier semantics, layout conversions, spilling a loop-carried tensor).

## Status

**Working end-to-end spine, with real kernels on it.** Triton IR text -> MSL ->
`MTLLibrary` -> compute pipeline -> unified-memory buffers -> dispatch -> results,
driven either from Swift or from Python over the `tm_*` C ABI. Triton's fused
**softmax** and **matmul** tutorial kernels both run on the GPU and match CPU
references — the matmul at f32 and at f16-in/f32-out, at sizes that divide
neither the block shape nor the 8x8 simdgroup fragment (`129x257x65` among them).

- **Lowering**: a recursive-descent parser for Triton's pretty MLIR syntax (plus
  the generic form for `tt.reduce`, which carries a region) and an MSL emitter.
  Supported subset: 1-D and 2-D tensors with `tt.expand_dims`/`tt.broadcast`,
  masked `tt.load`/`tt.store` at any rank, the `arith` integer/float/compare ops,
  all the `arith` conversions, `math.*` unary ops, `arith.select`, `scf.for`/
  `scf.if` with `iter_args` and multiple results, `tt.reduce` (add/max/min) over
  the innermost axis, and `tt.dot`. Anything else fails with the op name and
  source line. Full list:
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §Supported IR subset.
- **Layout**: tensors live in one per-kernel block index space with a block size
  per dimension. Which block dimension a rank-deficient tensor spans is *inferred*
  from the `tt.expand_dims` that consumes it, so row-uniform work is emitted once
  per row rather than once per element. A `tt.dot` adds a **contraction**
  dimension that the elementwise lane distribution deliberately does not walk.
- **`tt.dot`**: a second SSA value class for tensors that live in a threadgroup
  tile instead of per-lane registers, `simdgroup_load` /
  `simdgroup_multiply_accumulate` / `simdgroup_store` over 8x8 fragments, and
  zero-padded edge tiles so `BLOCK_M`/`BLOCK_N`/`BLOCK_K` need not be multiples of
  8. Everything feeding an operand is rebuilt inside the dot's own staging loops;
  a pointer advanced across the K loop is strength-reduced rather than carried.
- **Reductions**: `simd_shuffle_down` within each 32-wide simdgroup, threadgroup
  memory across simdgroups. A reduction closes the distributed loop and the values
  still needed afterwards are recomputed, not spilled. A reduction *inside* an
  `scf.for` hoists the whole loop to a threadgroup-uniform level, which is what an
  online softmax needs.
- **Numerics**: fast math is switched off (Metal defaults it on) and `math.*` maps
  to `precise::` wherever Metal has it; `fast::` is never emitted. Every math op is
  checked against a CPU reference on the GPU.
- **Compilation**: `MTLDevice.makeLibrary(source:)` at runtime (primary path, no
  Xcode tools needed); `xcrun metal` for offline `.metallib` bytes. Never xcodebuild.
- **Runtime**: handle-based C ABI — `tm_compile_msl`, `tm_load_kernel`,
  `tm_alloc_buffer`, `tm_launch`, with `tm_last_error` for diagnostics.
- **Performance**: the matmul tutorial reaches **~50% of `MPSMatrixMultiplication`**
  at 1024, 2048 and 4096 square — 3.06 TFLOP/s f32 at 2048 on an **M1 Max**, 1.69
  TFLOP/s on an **M1 Pro** — up from ~33%, which clears the >50% milestone and
  falls short of the 60–80% band. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  §Matmul throughput has the per-machine numbers, what each optimisation was
  worth measured one at a time, and the two CUDA-playbook techniques (double
  buffering, and register blocking past 1x1) that did **not** transfer to Apple
  silicon. Sweep it yourself with `swift run -c release tmbench`, which needs no
  Xcode.
- **Tests**: 127 Swift cases (parser, emitter, layout, casts/math, control flow,
  rank-2, reductions, `tt.dot`, error paths, and GPU runs verified against CPU
  references on an M1 Pro) + 14 Python cases including vector-add and softmax
  round trips, plus an opt-in MPS benchmark. 83.43% region / 85.58% function /
  90.32% line coverage of the Swift core — see
  [docs/WHITEPAPER.md](docs/WHITEPAPER.md) §Evaluation for the breakdown and for
  what the uncovered fraction is made of.

Not yet: atomics, `tt.trans`, and a per-lane tensor carried across a loop that
contains a cross-lane op — the last of which is what FlashAttention-2's
accumulator needs (§Hard parts 3). Not yet pinned to a Triton release.
[docs/USAGE.md §Implementing what's missing](docs/USAGE.md#implementing-whats-missing)
has concrete starting points for each of these.

## Build & test

```
swift build && swift test
cd python && PYTHONPATH=. python3 -m pytest tests/ -q
```

Never `xcodebuild`. GEMM throughput against MPS:

```
swift run -c release tmbench                    # block-shape sweep, 512/1024/2048
swift run -c release tmbench --sweep full       # + register blocking, tile padding,
                                                #   double buffering
swift run -c release tmbench --config 64,128,16,16   # pin one configuration
swift run -c release tmbench --emit 64,64,16,16      # print the kernel it lowers
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
