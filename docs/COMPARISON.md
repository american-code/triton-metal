# triton-metal vs. Triton-on-CUDA: a measured comparison

*2026-08-26. Local numbers from tmbench on Mac Studio M1 Max (measured peaks 6.2 TF
fp32 / 371.5 GB/s) and MacBook M1 Pro (3.45 TF / 166.4 GB/s); CUDA numbers cited.
The comparison is efficiency fractions against each platform's vendor library.*

## Published Triton anchors (its native CUDA target)

| comparison | result | source |
|---|---|---|
| Triton end-to-end LLM inference vs. CUDA-kernel stack, A100 | 62–82% | [PyTorch blog](https://pytorch.org/blog/cuda-free-inference-for-llms/) |
| Triton fp16 GEMM vs. cuBLAS, A100 | ≈ parity | [Hexcute, arXiv 2504.16214](https://arxiv.org/pdf/2504.16214) |
| Triton int8 GEMM vs. cuBLAS, A100 | 92–123% by shape | [triton #4876](https://github.com/triton-lang/triton/issues/4876) |

Mature Triton reaches ~80–100% of the vendor library on GEMM after years of
NVIDIA-specific pipeline work; 62–82% end-to-end is the realistic band.

## triton-metal measured (fp32 square GEMM vs. MPSMatrixMultiplication)

| machine | size | baseline | optimized | of MPS |
|---|---|---|---|---|
| M1 Max | 1024 | 1.85 TF | 2.75 TF | 33% → **50%** |
| M1 Max | 2048 | 1.99 TF | 3.06 TF | 33% → **50%** |
| M1 Max | 4096 | 1.97 TF | 3.21 TF | 32% → **52%** |
| M1 Pro | 1024 | 1.02 TF | 1.57 TF | 34% → **51%** |

One pass of the published CUDA optimization playbook moved 33% → 50–52% (1.55×).
Correctness is at parity for the supported subset throughout — every winner is verified
against a CPU reference at non-multiple-of-tile sizes before being reported.

## The transfer matrix: what ports from CUDA, what inverts

Measured one change at a time (M1 Max @ 2048):

- **Transferred:** register-resident accumulators (+13%); staging locality + literal
  trip counts (+16%); the accumulator tile doubling as the operand staging arena
  (+6%, and the enabler for 64×128 blocks inside the 32 KB threadgroup budget).
- **Did not transfer:** *double buffering is a wash* — Metal has no `cp.async` copy
  engine, so prefetch competes for the same issue slots and larger tiles cost the
  occupancy Apple silicon uses to hide latency. *Register blocking past 1×1 inverts* —
  at the fastest shape 1×1 beats 2×2 by 28% at equal fragment counts, the opposite of
  CUTLASS guidance.

## Remaining gap to the 62–82% band

Characterized in [ARCHITECTURE.md](ARCHITECTURE.md) §Matmul throughput: prologue/
epilogue tile traffic, vectorized (`float4`) staging, and register-only accumulators
for 128×128 blocks. Nothing measured indicates a structural ceiling below the band —
MPS itself reaches 86–99% of the *measured* device peak from the same API surface.
