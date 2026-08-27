"""Triton's fused-softmax tutorial on a Mac GPU.

One threadgroup per row; the max and the sum are cross-lane reductions, which the
Swift emitter lowers to `simd_shuffle_down` folds plus a threadgroup-memory pass
between simdgroups (docs/ARCHITECTURE.md §Cross-lane regions).

    python python/examples/fused_softmax.py
"""

import numpy as np
import triton
import triton.language as tl

from triton_metal.buffer import MetalBuffer


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x: np.ndarray) -> np.ndarray:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    x_buf = MetalBuffer.from_numpy(x)
    out_buf = MetalBuffer(x.shape, "float32")
    softmax_kernel[(n_rows, )](
        out_buf,
        x_buf,
        x.shape[1],
        x.shape[1],
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out_buf.numpy()


def reference(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1823, 781), dtype=np.float32)
    got = softmax(x)
    expected = reference(x)
    error = float(np.max(np.abs(got - expected)))
    print(f"triton {triton.__version__} on {triton.runtime.driver.active.get_current_target().arch}")
    print(f"fused softmax, {x.shape}: max |triton - numpy| = {error:g}")
    assert error < 1e-6, f"softmax mismatch: {error}"
    print("OK")


if __name__ == "__main__":
    main()
