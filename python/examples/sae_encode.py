"""The SAE encoder through the Python ctypes path, for a like-for-like timing.

The Swift-native frontend (`Sources/TritonMetalMLX`, driven by `tmsae`) runs the
fused kernel `relu(x @ W_enc + b_enc)` straight on MLX tensors with no Python
anywhere. This is the other half of that comparison: the *same kernel*, lowered
from the *same IR text*, launched through `triton_metal._core` — the ctypes shim
Triton's driver uses.

The IR is not written here. `tmsae --emit-ir` prints it, and this script reads
that, so neither side can drift from the other and the comparison stays a
comparison of hosts rather than of kernels.

    swift build -c release --product tmsae
    python python/examples/sae_encode.py

    python python/examples/sae_encode.py --ir /path/to/sae.ttir --iterations 200

Needs no Triton: the ctypes path is the shim plus the Swift core.
"""

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from triton_metal import _core

#: Must match `SAEEncoder.Blocking`'s defaults, which is what `tmsae --emit-ir`
#: prints. The grid below is the only thing that needs them.
BLOCK_M = 64
BLOCK_F = 64


def read_ir(path: str | None) -> str:
    """The kernel's Triton IR, from `tmsae --emit-ir` or from a file."""
    if path:
        return Path(path).read_text()
    root = Path(__file__).resolve().parents[2]
    candidates = [root / ".build" / c / "tmsae" for c in ("release", "debug")]
    candidates += [Path(p) for p in [shutil.which("tmsae")] if p]
    for candidate in candidates:
        if candidate.exists():
            return subprocess.run(
                [str(candidate), "--emit-ir"], check=True, capture_output=True, text=True).stdout
    raise SystemExit(
        "tmsae not found; build it with `swift build -c release --product tmsae`, or pass the "
        "IR with --ir. The IR lives in Swift (Sources/TritonMetalMLX/SAEEncoder.swift) so that "
        "both sides of this comparison lower identical text.")


def reference(x, w, b):
    return np.maximum(x @ w + b, 0.0)


def synthetic(shape, offset, modulus):
    """The same deterministic inputs `tmsae` uses, so both sides see one problem."""
    count = int(np.prod(shape))
    i = np.arange(count, dtype=np.int64)
    return (((i + offset) * 13 % modulus) / modulus - 0.5).astype(np.float32).reshape(shape)


class Encoder:
    """Compiled once, launched many times — the shape the timing needs."""

    def __init__(self, ttir: str):
        self.msl = _core.emit_msl(ttir)
        info = json.loads(_core.kernel_info(ttir))["kernels"][0]
        self.name = info["name"]
        self.threads = info["threads_per_threadgroup"]
        self.library = _core.compile_msl(self.msl)
        self.kernel = _core.load_kernel(self.library, self.name)

    def buffers(self, x, w, b):
        rows, model = x.shape
        features = w.shape[1]
        handles = []
        for array in (x, w, b):
            handle = _core.alloc_buffer(array.nbytes)
            _core.buffer_write(handle, 0, array.ctypes.data, array.nbytes)
            handles.append(handle)
        out = np.zeros((rows, features), dtype=np.float32)
        handles.append(_core.alloc_buffer(out.nbytes))
        return handles, out

    def launch(self, handles, rows, model, features):
        _core.launch(
            self.kernel,
            (-(-rows // BLOCK_M), -(-features // BLOCK_F), 1),
            self.threads,
            [_core.ARG_BUFFER] * 4 + [_core.ARG_I32] * 3,
            list(handles) + [rows, model, features],
        )

    def release(self, handles):
        for handle in handles:
            _core.free_buffer(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", help="Triton IR file (default: ask tmsae for it)")
    parser.add_argument("--iterations", type=int, default=200)
    options = parser.parse_args()

    if not _core.is_usable():
        print(f"skipping: {_core.unusable_reason()}")
        return 0

    encoder = Encoder(read_ir(options.ir))
    print(f"device: {_core.device_name()}")
    print(f"kernel: {encoder.name}, {encoder.threads} threads/threadgroup")

    print("\n--- relu(x @ W_enc + b_enc): Triton kernel vs numpy ---")
    for rows, model, features in [(8, 64, 128), (64, 512, 2048), (100, 300, 700)]:
        x = synthetic((rows, model), 1, 71)
        w = synthetic((model, features), 7, 53)
        b = synthetic((features, ), 3, 29)
        handles, out = encoder.buffers(x, w, b)
        encoder.launch(handles, rows, model, features)
        _core.buffer_read(handles[3], 0, out.ctypes.data, out.nbytes)
        encoder.release(handles)
        want = reference(x.astype(np.float64), w.astype(np.float64), b.astype(np.float64))
        error = float(np.max(np.abs(out - want)))
        scale = max(float(np.max(np.abs(want))), 1.0)
        print(f"[{rows}, {model}] @ [{model}, {features}]: max |kernel - numpy| = {error:g}, "
              f"relative {error / scale:g}")

    # Two numbers, matching `tmsae`'s:
    #
    #   dispatch   encode + commit + wait with the buffers already filled. This is
    #              `tm_launch`, which is `MetalRuntime.launch` — the same Swift
    #              function the Swift-native path calls. It should come out equal
    #              on both sides, and it is the floor neither path can go under.
    #
    #   per call   what it costs to run the kernel on the arrays a caller holds.
    #              A `numpy` array is host memory, so the inputs have to be
    #              written into Metal buffers and the output read back — the copy
    #              the Swift path does not make, because an `MLXArray` is already
    #              the memory the GPU reads.
    print(f"\n--- host dispatch overhead: compile once, launch {options.iterations} times ---")
    for rows, model, features in [(8, 64, 64), (64, 64, 64), (64, 512, 2048)]:
        x = synthetic((rows, model), 1, 71)
        w = synthetic((model, features), 7, 53)
        b = synthetic((features, ), 3, 29)
        handles, out = encoder.buffers(x, w, b)

        def measure(body):
            for _ in range(20):
                body()
            start = time.perf_counter_ns()
            for _ in range(options.iterations):
                body()
            return (time.perf_counter_ns() - start) / options.iterations / 1000

        def dispatch_only():
            encoder.launch(handles, rows, model, features)

        def per_call():
            for handle, array in zip(handles, (x, w, b)):
                _core.buffer_write(handle, 0, array.ctypes.data, array.nbytes)
            encoder.launch(handles, rows, model, features)
            _core.buffer_read(handles[3], 0, out.ctypes.data, out.nbytes)

        dispatch = measure(dispatch_only)
        call = measure(per_call)
        encoder.release(handles)
        grid = (-(-rows // BLOCK_M), -(-features // BLOCK_F))
        print(f"[{rows}, {model}] @ [{model}, {features}], grid {grid[0]}x{grid[1]}: "
              f"dispatch {dispatch:.1f} us, per call {call:.1f} us "
              f"(copies {call - dispatch:.1f} us)")

    _core.release_kernel(encoder.kernel)
    _core.release_library(encoder.library)
    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
