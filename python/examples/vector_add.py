"""Triton's vector-add tutorial, unmodified, on a Mac GPU.

`@triton.jit` source -> Triton's own frontend and MLIR passes -> `tm_emit_msl`
-> `MTLDevice.makeLibrary(source:)` -> dispatch. Everything after the IR text is
Swift; this file is just the kernel and its reference check.

    python python/examples/vector_add.py
"""

import numpy as np
import triton
import triton.language as tl

from triton_metal.buffer import MetalBuffer


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: np.ndarray, y: np.ndarray, block_size: int = 1024) -> np.ndarray:
    x_buf = MetalBuffer.from_numpy(x)
    y_buf = MetalBuffer.from_numpy(y)
    out_buf = MetalBuffer(x.shape, "float32")
    n_elements = x_buf.numel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x_buf, y_buf, out_buf, n_elements, BLOCK_SIZE=block_size)
    return out_buf.numpy()


def main():
    rng = np.random.default_rng(0)
    size = 98_432
    x = rng.standard_normal(size, dtype=np.float32)
    y = rng.standard_normal(size, dtype=np.float32)

    got = add(x, y)
    expected = x + y
    error = float(np.max(np.abs(got - expected)))
    print(f"triton {triton.__version__} on {triton.runtime.driver.active.get_current_target().arch}")
    print(f"vector add, n={size}: max |triton - numpy| = {error:g}")
    assert error == 0.0, "vector add is exact in f32"
    print("OK")


if __name__ == "__main__":
    main()
