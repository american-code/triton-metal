"""A real `import triton` + `@triton.jit` round trip, when Triton is installed.

Triton publishes no macOS wheel, so this suite must not acquire a hard dependency
on it: every test here skips with an actionable message when `import triton`
fails. See docs/USAGE.md §Building Triton with the Metal backend for the recipe
that makes them run (~10 minutes of compiling, once).
"""

import pytest

from triton_metal import _core

triton = pytest.importorskip(
    "triton",
    reason="Triton is not installed; build it from source with TRITON_PLUGIN_DIRS "
    "pointing at python/plugin (docs/USAGE.md) to exercise the real round trip",
)

import triton.language as tl  # noqa: E402  (only reachable once triton imported)

from triton_metal.buffer import MetalBuffer  # noqa: E402

PINNED_VERSION = "3.7.1"

requires_metal = pytest.mark.skipif(not _core.is_active(), reason="no Metal device on this machine")


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def test_pinned_triton_version():
    """The vendored `BaseBackend`/`DriverBase` signatures track this release."""
    assert triton.__version__.split("+")[0] == PINNED_VERSION


def test_metal_is_the_discovered_backend():
    from triton.backends import backends
    assert "metal" in backends, f"discovered backends: {sorted(backends)}"
    assert backends["metal"].compiler.__name__ == "MetalBackend"
    assert backends["metal"].driver.__name__ == "MetalDriver"


def test_plugin_is_linked_into_libtriton():
    from triton._C.libtriton import metal
    assert metal.linked()


@requires_metal
def test_metal_is_the_active_driver():
    from triton.runtime import driver
    target = driver.active.get_current_target()
    assert target.backend == "metal"
    assert target.warp_size == 32
    assert target.arch == _core.device_name()


@requires_metal
def test_jit_kernel_compiles_through_the_swift_core():
    from triton.backends.metal.compiler import MetalBackend, MetalOptions
    stages = {}
    MetalBackend(triton.runtime.driver.active.get_current_target()).add_stages(stages, MetalOptions())
    assert list(stages) == ["ttir", "msl"]


@requires_metal
def test_vector_add_runs_on_the_gpu():
    n = 4096
    x = MetalBuffer(n, "float32")
    y = MetalBuffer(n, "float32")
    out = MetalBuffer(n, "float32")
    try:
        for i in range(n):
            x._view()[i] = i * 0.5
            y._view()[i] = float(i % 7)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        kernel = add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
        # The payload Triton cached is the MSL our Swift core emitted.
        assert "kernel void add_kernel(" in kernel.asm["msl"].decode()
        got = out.tolist()
        assert got == [i * 0.5 + (i % 7) for i in range(n)]
    finally:
        for buffer in (x, y, out):
            buffer.free()
