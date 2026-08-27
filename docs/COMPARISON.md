# triton-metal vs. Triton-on-CUDA: a measured comparison

*2026-08-27 (three optimisation rounds plus the backward pass). Local numbers from
tmbench on a Mac Studio M1 Max (measured peaks 6.2 TF fp32 / 371.5 GB/s) and a
MacBook M1 Pro (3.45 TF / 166.4 GB/s), each side timed twice per size with the
faster taken; CUDA numbers cited. The comparison is efficiency fractions against
each platform's vendor library.*

## Published Triton anchors (its native CUDA target)

| comparison | result | source |
|---|---|---|
| Triton end-to-end LLM inference vs. CUDA-kernel stack, A100 | 62–82% | [PyTorch blog](https://pytorch.org/blog/cuda-free-inference-for-llms/) |
| Triton fp16 GEMM vs. cuBLAS, A100 | ≈ parity | [Hexcute, arXiv 2504.16214](https://arxiv.org/pdf/2504.16214) |
| Triton int8 GEMM vs. cuBLAS, A100 | 92–123% by shape | [triton #4876](https://github.com/triton-lang/triton/issues/4876) |

Mature Triton reaches ~80–100% of the vendor library on GEMM after years of
NVIDIA-specific pipeline work; 62–82% end-to-end is the realistic band.

## triton-metal measured (fp32 square GEMM vs. MPSMatrixMultiplication)

| machine | size | baseline | round 1 | round 2 | round 3 | of MPS | of measured peak |
|---|---|---|---|---|---|---|---|
| M1 Max | 1024 | 1.85 TF | 2.75 TF | 4.33 TF | **4.30 TF** | 33% → 50% → **76%** | 69% |
| M1 Max | 2048 | 1.99 TF | 3.06 TF | 4.66 TF | **4.64 TF** | 33% → 50% → **76%** | 75% |
| M1 Max | 4096 | 1.97 TF | 3.21 TF | 4.64 TF | **4.65 TF** | 32% → 52% → **76%** | 75% |
| M1 Pro | 1024 | 1.02 TF | 1.57 TF | 2.27 TF | **2.33 TF** | 34% → 51% → **75%** | 67% |
| M1 Pro | 2048 | 1.11 TF | 1.69 TF | 2.33 TF | **2.43 TF** | 33% → 50% → **75%** | 71% |

One pass of the published CUDA optimization playbook moved 33% → 50–52% (1.55×). A
second pass — of things that are *not* in that playbook — moved 50% → 76% on the
M1 Max (1.52× again, 2.3× over baseline), which lands inside the 62–82% band mature
Triton reaches on its native target and below the 80–100% band it reaches on GEMM
specifically. A third pass did not move it, and produced the mechanism instead
(below). The ratios are 75–76% on both machines, which they were not when MPS was
timed once: MPS is the denominator of every one of them and the first measurement
at a size runs cold, so it now gets a second run too. Correctness is at parity for
the supported subset throughout — every winner is verified against a CPU reference
at non-multiple-of-tile sizes before being reported.

The last column is the one that says how much room is left. MPS reaches 92–99% of
the same machine's *measured* f32 peak; this kernel reaches 75%. The distance is
about four points of hardware utilisation.

## The transfer matrix: what ports from CUDA, what inverts

Measured one change at a time (M1 Max @ 2048).

**Scope: this is an M1-generation result.** Everything below was measured on M1
Max and M1 Pro parts. The inversions are claims about *these* GPUs, and the
reasons given for them — no `cp.async` copy engine, occupancy as the latency-hiding
mechanism, fragments moving through threadgroup memory rather than named lanes —
are architectural properties that later generations may well have changed. The M3
and M4 GPUs reworked the memory hierarchy (dynamic caching) in particular. Nothing
here should be assumed to hold on M2, M3 or M4 silicon until it is re-measured;
`tmbench --sweep full` is the re-measurement, and it is one command.

- **Transferred (M1):** register-resident accumulators (+13%); staging locality + literal
  trip counts (+16%); the accumulator tile doubling as the operand staging arena
  (+6%, and the enabler for 64×128 blocks inside the 32 KB threadgroup budget);
  vectorized (`float4`) global loads (+20%, the largest single change of either
  round — though on this chip most of it is the *address arithmetic* the vector
  path skips, not the loads).
- **Did not transfer (M1):** *bigger block shapes do not pay, and the reason is the
  register file.* Round 3 removed threadgroup memory as a constraint on block shape
  altogether — the epilogue streams the accumulator out of registers a panel of
  rows at a time, so `64x64` costs 8 KB instead of 16, `128x64` 12 KB instead of
  32, and a `128x128` f32 accumulator (64 KB, twice Metal's whole budget) becomes
  emittable. It is worth 0% at the shape that wins, +15% at `128x64`, and the
  `128x128` tile it unlocks — which halves operand traffic per output element —
  runs 21% *slower* than `64x64`. What actually binds is visible in the
  `maxTotalThreadsPerThreadgroup` Metal reports after register allocation: 1024 for
  the winner, 832 for `128x64`, 768 for `128x128`, 448 for `128x128` at 2×2
  blocking, which will not launch its own 512 threads. Occupancy is how this GPU
  hides latency, registers are what occupancy costs, and that one mechanism is
  behind every inversion in this list.
- **Did not transfer (M1):** *double buffering is a wash* — Metal has no `cp.async` copy
  engine, so prefetch competes for the same issue slots and larger tiles cost the
  occupancy Apple silicon uses to hide latency (re-measured after the mapping
  change: 3633 vs 3627 GFLOP/s; after vector staging it is a 26% *loss*, because
  ping-ponging makes each tile's base a variable pointer and the vector path wants
  a constant). *Register blocking past 1×1 along **N** inverts, and inverts harder*
  — at `64x128` 1×1 now beats 2×2 by 53% at equal fragment counts (4327 vs 2836),
  up from 28%, the opposite of CUTLASS guidance. Blocking along **M** is the
  exception that proves the rule: 2×1 keeps the shared column and is worth +8% at
  the shape that wins. And the register account settles it: a `4x2` block at
  `64x64` genuinely does cut a contraction step from 9 operand loads to 6, and
  still measures 4149 GFLOP/s against 2×1's 4642, because it takes the threadgroup
  ceiling from 1024 to 768.
- **Metal-specific, no CUDA analogue (M1):** the *shared-column wave mapping* (+15%).
  Because Metal's simdgroup-matrix ops move whole 8×8 fragments through threadgroup
  memory rather than distributing them over named lanes, which fragment each
  simdgroup wants is an emitter decision — and choosing it so that every wave of a
  simdgroup shares one B fragment cuts a contraction step from 16 threadgroup reads
  to 9. There is nothing to port here: the CUDA equivalent of this decision is made
  by the mma layout, not by the kernel.

## Attention: the fused kernel against the unfused composite

The GEMM comparison above is against the best single kernel Apple ships for the
same job, and this backend does not beat it. FlashAttention-2 is the other kind of
comparison — a *fused* kernel against the *composite* it replaces — and there it
wins, for the reason fusion exists.

Against `Q K^T` + `MPSMatrixSoftMax` + `P V`, three dispatches per head through a
real `S x S` f32 score matrix, on an M1 Max:

| shape | fused | MPS composite | ratio |
| --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | 1358 GF | 381 GF | **357%** |
| `b1 h8 s1024 d64` f32 | 1678 GF | 1575 GF | **107%** |
| `b1 h16 s2048 d64` f32 | 1761 GF | 2586 GF | 68% |
| `b1 h8 s1024 d64` f16 | 2089 GF | 1582 GF | **132%** |

The crossover is around `s1024`: below it the composite's GEMMs are too small to
reach MPS's peak and the dispatch count dominates; above it they do reach it and
the score-matrix traffic the fusion removes stops being what decides. Also an
M1-generation result, and the same caveat applies — a chip with a different cache
hierarchy would move the crossover, plausibly in the fused kernel's favour if
DRAM traffic becomes relatively more expensive, or against it if the composite's
score matrix starts fitting in cache. [ARCHITECTURE.md](ARCHITECTURE.md)
§Attention throughput has the M1 Pro numbers, the run-to-run variance on the
composite side, and a note that a better-written composite (metalscope measured
~744 GF for the same shape on an M1 Pro) would narrow the `s512` margin.

## Attention backward: the same comparison, one step harder

The backward pass is where a Triton backend stops being an inference target. The
composite MPS can express for it is five `MPSMatrixMultiplication`s with an
`MPSMatrixSoftMax` and an `MPSMatrixSoftMaxGradient` between them, through **two**
live `S x S` f32 matrices — six passes over a score matrix the fused version never
materialises. Both sides are credited with the same `5 * 2 * S^2 * D` per head,
the five GEMMs the mathematics needs, although the fused side performs seven:
`Q K^T` is recomputed in the `dQ` direction and again in the `dK`/`dV` one, which
is the price of keeping every gradient deterministic and every accumulator in
simdgroup registers.

| shape | fused | MPS composite | ratio |
| --- | --- | --- | --- |
| `b1 h8 s512 d64` f32 | 977 GF | 515 GF | **190%** |
| `b1 h8 s1024 d64` f32 | 1054 GF | 1519 GF | 69% |
| `b1 h16 s2048 d64` f32 | 1172 GF | 2494 GF | 47% |
| `b1 h8 s512 d64` f16 | 1293 GF | 588 GF | **220%** |
| `b1 h8 s1024 d64` f16 | 1419 GF | 1445 GF | 98% |

The crossover is earlier than the forward's — 69% at `s1024` where the forward is
at 107% — and the 40% arithmetic premium is most of the reason. It is the same
shape of result: fusion wins where the composite's dispatches and traffic dominate
and loses where its GEMMs reach MPS's peak. `tl.atomic_add` for `dQ` would remove
one of the two recomputations at the cost of a non-deterministic gradient; the
atomics exist for it, the trade has not been measured, and the crossover is the
number that would decide it.

Also M1-generation, and the composite's own readings move: the same f32 composite
measured 1519 and 1445 GF at `s1024` minutes apart, so those ratios are worth
about ±5 points. `tmbench --attn-bwd` re-runs the whole thing in one command.

## Where it sits now

Inside the end-to-end band and below the GEMM one: 76% of MPS at 1024, 2048 and
4096 on the M1 Max, 75% at 1024 and 2048 on the M1 Pro. In absolute terms the
2048 GEMM is at 75% of the machine's measured 6.2 TF f32 peak, against MPS's 99%
from the same API surface.

The account of what stood between those two numbers has now been wrong twice, and
both corrections are measurements rather than arguments. The instruction-slot
account ranked the prologue and epilogue first (worth 2%) and vector staging third
(worth 20%). The arithmetic-intensity account that replaced it prescribed a
`128×128` tile to halve operand traffic; that tile now exists and is 21% slower,
and handing the same threadgroup memory back where the accumulator already fitted
is worth nothing. Operand traffic is not what the kernel waits on.

What it waits on is occupancy, and what occupancy costs is registers — the one
mechanism behind every inversion above. So the remaining four points are an
issue-slot problem *at fixed occupancy*: less non-arithmetic work per unit of
matrix arithmetic, bought without registers.
[ARCHITECTURE.md](ARCHITECTURE.md) §Matmul throughput has the candidates, of which
the largest is loading the B operand's 8×8 fragments straight from device memory —
MSL's `simdgroup_load` takes a `device` pointer, each simdgroup reads exactly one
B fragment per step under the shared-column mapping, and staging it therefore buys
only masking, at the cost of an entire staging pass.
