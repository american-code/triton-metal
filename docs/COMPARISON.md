# triton-metal vs. Triton-on-CUDA: a measured comparison

*2026-08-26 (two optimisation rounds, same day). Local numbers from tmbench on a
Mac Studio M1 Max (measured peaks 6.2 TF fp32 / 371.5 GB/s) and a MacBook M1 Pro
(3.45 TF / 166.4 GB/s); CUDA numbers cited. The comparison is efficiency fractions
against each platform's vendor library.*

## Published Triton anchors (its native CUDA target)

| comparison | result | source |
|---|---|---|
| Triton end-to-end LLM inference vs. CUDA-kernel stack, A100 | 62–82% | [PyTorch blog](https://pytorch.org/blog/cuda-free-inference-for-llms/) |
| Triton fp16 GEMM vs. cuBLAS, A100 | ≈ parity | [Hexcute, arXiv 2504.16214](https://arxiv.org/pdf/2504.16214) |
| Triton int8 GEMM vs. cuBLAS, A100 | 92–123% by shape | [triton #4876](https://github.com/triton-lang/triton/issues/4876) |

Mature Triton reaches ~80–100% of the vendor library on GEMM after years of
NVIDIA-specific pipeline work; 62–82% end-to-end is the realistic band.

## triton-metal measured (fp32 square GEMM vs. MPSMatrixMultiplication)

| machine | size | baseline | round 1 | round 2 | of MPS |
|---|---|---|---|---|---|
| M1 Max | 1024 | 1.85 TF | 2.75 TF | **4.33 TF** | 33% → 50% → **76%** |
| M1 Max | 2048 | 1.99 TF | 3.06 TF | **4.66 TF** | 33% → 50% → **76%** |
| M1 Max | 4096 | 1.97 TF | 3.21 TF | **4.64 TF** | 32% → 52% → **76%** |
| M1 Pro | 1024 | 1.02 TF | 1.57 TF | **2.27 TF** | 34% → 51% → **82%** |
| M1 Pro | 2048 | 1.11 TF | 1.69 TF | **2.33 TF** | 33% → 50% → **82%** |

One pass of the published CUDA optimization playbook moved 33% → 50–52% (1.55×). A
second pass — of things that are *not* in that playbook — moved 50% → 76% on the
M1 Max (1.52× again, 2.3× over baseline), which lands inside the 62–82% band mature
Triton reaches on its native target. (The M1 Pro rows are a laptop's, whose MPS
readings move with its thermal state; a cooler MPS reading in the same session
would put its 2048 row at 69%.) Correctness is at parity for the supported
subset throughout — every winner is verified against a CPU reference at
non-multiple-of-tile sizes before being reported.

## The transfer matrix: what ports from CUDA, what inverts

Measured one change at a time (M1 Max @ 2048):

- **Transferred:** register-resident accumulators (+13%); staging locality + literal
  trip counts (+16%); the accumulator tile doubling as the operand staging arena
  (+6%, and the enabler for 64×128 blocks inside the 32 KB threadgroup budget);
  vectorized (`float4`) global loads (+20%, the largest single change of either
  round — though on this chip most of it is the *address arithmetic* the vector
  path skips, not the loads).
- **Did not transfer:** *double buffering is a wash* — Metal has no `cp.async` copy
  engine, so prefetch competes for the same issue slots and larger tiles cost the
  occupancy Apple silicon uses to hide latency (re-measured after the mapping
  change: 3633 vs 3627 GFLOP/s; after vector staging it is a 26% *loss*, because
  ping-ponging makes each tile's base a variable pointer and the vector path wants
  a constant). *Register blocking past 1×1 along **N** inverts, and inverts harder*
  — at `64x128` 1×1 now beats 2×2 by 53% at equal fragment counts (4327 vs 2836),
  up from 28%, the opposite of CUTLASS guidance. Blocking along **M** is the
  exception that proves the rule: 2×1 keeps the shared column and is worth +8% at
  the shape that wins.
- **Metal-specific, no CUDA analogue:** the *shared-column wave mapping* (+15%).
  Because Metal's simdgroup-matrix ops move whole 8×8 fragments through threadgroup
  memory rather than distributing them over named lanes, which fragment each
  simdgroup wants is an emitter decision — and choosing it so that every wave of a
  simdgroup shares one B fragment cuts a contraction step from 16 threadgroup reads
  to 9. There is nothing to port here: the CUDA equivalent of this decision is made
  by the mma layout, not by the kernel.

## Where it sits now

Inside the band, at the sizes where the measurement is trustworthy — 76% of MPS at
1024, 2048 and 4096 on the M1 Max. What is left is
characterized in [ARCHITECTURE.md](ARCHITECTURE.md) §Matmul throughput: register-only
accumulators for 128×128 blocks (untried — the target was reached without it),
vector staging for the narrow A tile, and specialising away the run mask. At 4.66 TF
the 2048 GEMM is at ~75% of the *measured* 6.2 TF device peak and pushes ~290 GB/s
of 371 GB/s through the memory system, so what is left is mostly arithmetic
intensity — which is exactly what the 128×128 tile buys. MPS itself reaches 86–99%
of that measured peak from the same API surface.
