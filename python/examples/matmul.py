"""Triton's matmul tutorial (`tl.dot`) on a Mac GPU.

The kernel is the tutorial's, unmodified, and every block shape works — including
the ones where two or three of `BLOCK_M`, `BLOCK_N` and `BLOCK_K` are equal, which
this backend used to refuse. Triton's CSE makes `tl.arange(0, BLOCK_M)` and
`tl.arange(0, BLOCK_N)` a single value when the two are equal, and the row axis and
the column axis then unify through it; the emitter gives each use its own copy of
that arithmetic instead (docs/ARCHITECTURE.md §Sharing one range between two axes).

    python python/examples/matmul.py
"""

import numpy as np
import triton
import triton.language as tl

from triton_metal.buffer import MetalBuffer


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                  stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


def matmul(a: np.ndarray, b: np.ndarray, block=(128, 64, 32)) -> np.ndarray:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    block_m, block_n, block_k = block
    a_buf = MetalBuffer.from_numpy(a)
    b_buf = MetalBuffer.from_numpy(b)
    c_buf = MetalBuffer((M, N), "float32")
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        a_buf, b_buf, c_buf, M, N, K,
        K, 1,
        N, 1,
        N, 1,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )
    return c_buf.numpy()


def main():
    rng = np.random.default_rng(0)
    M = N = K = 256
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    expected = a @ b
    print(f"triton {triton.__version__} on {triton.runtime.driver.active.get_current_target().arch}")
    # Including the shapes where the block sizes collide, which is where Triton's
    # CSE hands the backend one range serving two (or three) axes.
    for block in ((128, 64, 32), (64, 64, 32), (64, 64, 64), (32, 32, 32)):
        error = float(np.max(np.abs(matmul(a, b, block) - expected)))
        print(f"matmul {M}x{K}x{N} block {block}: max |triton - numpy| = {error:g}")
        assert error < 1e-3, f"matmul mismatch at {block}: {error}"
    print("OK")


if __name__ == "__main__":
    main()
