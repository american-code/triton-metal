"""ctypes binding to libtritonmetal.dylib (the Swift core). No logic here.

Every function below is a one-to-one binding for a `tm_*` C export: marshal the
arguments, call, translate the documented failure sentinel (NULL / 0 / -1) into a
RuntimeError carrying `tm_last_error`. Nothing is computed on this side.
"""

import ctypes
import os
import struct
import sysconfig
from pathlib import Path

#: File name of the Swift core's dynamic library, everywhere it might be.
_LIBRARY = "libtritonmetal.dylib"


def _installed_backend_dirs():
    """`triton/backends/metal/` in whichever site directory Triton is installed in.

    Located by path rather than by importing `triton`: this module is imported
    *from* `triton.backends.metal.compiler`, so asking the import system for
    `triton` here would re-enter a package that is only half initialised.

    This is where a wheel built by Tools/bundle-wheel.sh puts the dylib, which is
    what makes the Triton wheel a single self-contained artifact (docs/USAGE.md
    §Building Triton with the Metal backend). It is searched before the
    repo-relative dev builds so that an installed wheel uses its own runtime
    rather than whatever a checkout happens to have lying in `.build`; a
    developer who wants the other one says so with TRITON_METAL_CORE_LIB.
    """
    paths = sysconfig.get_paths()
    roots = [paths.get(key) for key in ("purelib", "platlib")]
    return [
        str(Path(root) / "triton" / "backends" / "metal" / _LIBRARY) for root in roots if root
    ]


_SEARCH = [
    os.environ.get("TRITON_METAL_CORE_LIB"),
    # Bundled into an installed `triton.backends.metal`.
    *_installed_backend_dirs(),
    # Installed alongside this package (a wheel that ships the dylib next to the
    # shim rather than next to the backend).
    str(Path(__file__).resolve().parent / _LIBRARY),
    # Repo-relative dev builds (swift build [-c release] at the repo root).
    str(Path(__file__).resolve().parents[2] / ".build" / "release" / _LIBRARY),
    str(Path(__file__).resolve().parents[2] / ".build" / "debug" / _LIBRARY),
]


def _load() -> ctypes.CDLL:
    for path in _SEARCH:
        if path and os.path.exists(path):
            return ctypes.CDLL(path)
    raise OSError(
        f"{_LIBRARY} not found; run `swift build` at the repo root, install a wheel built "
        "with Tools/bundle-wheel.sh, or set TRITON_METAL_CORE_LIB. Searched:\n  "
        + "\n  ".join(p for p in _SEARCH if p)
    )


_lib = _load()

_i64 = ctypes.c_int64
_i32 = ctypes.c_int32
_ptr = ctypes.c_void_p
_str = ctypes.c_char_p

_SIGNATURES = {
    # name: (restype, argtypes)
    "tm_version": (_ptr, []),
    "tm_last_error": (_ptr, []),
    "tm_free": (None, [_ptr]),
    "tm_is_active": (_i32, []),
    "tm_is_usable": (_i32, []),
    "tm_unusable_reason": (_ptr, []),
    "tm_device_name": (_ptr, []),
    # Compiler
    "tm_emit_msl": (_ptr, [_str, _i32]),
    "tm_kernel_info": (_ptr, [_str, _i32]),
    "tm_compile_msl": (_i64, [_str]),
    "tm_load_metallib": (_i64, [_ptr, _i64]),
    "tm_release_library": (_i32, [_i64]),
    # Runtime
    "tm_load_kernel": (_i64, [_i64, _str]),
    "tm_kernel_max_threads": (_i64, [_i64]),
    "tm_release_kernel": (_i32, [_i64]),
    "tm_alloc_buffer": (_i64, [_i64]),
    "tm_buffer_contents": (_ptr, [_i64]),
    "tm_buffer_length": (_i64, [_i64]),
    "tm_buffer_write": (_i32, [_i64, _i64, _ptr, _i64]),
    "tm_buffer_read": (_i32, [_i64, _i64, _ptr, _i64]),
    "tm_free_buffer": (_i32, [_i64]),
    "tm_launch": (
        _i32,
        [_i64, _i64, _i64, _i64, _i64, ctypes.POINTER(_i32), ctypes.POINTER(_i64), _i32],
    ),
    "tm_live_handle_count": (_i64, []),
}

for _name, (_restype, _argtypes) in _SIGNATURES.items():
    _fn = getattr(_lib, _name)
    _fn.restype = _restype
    _fn.argtypes = _argtypes


def _take_string(ptr):
    if not ptr:
        return None
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    finally:
        _lib.tm_free(ptr)


def last_error() -> str:
    return _take_string(_lib.tm_last_error())


def _check(value, sentinel):
    if value == sentinel:
        raise RuntimeError(last_error())
    return value


def _check_string(ptr):
    text = _take_string(ptr)
    if text is None:
        raise RuntimeError(last_error())
    return text


# --- Device -----------------------------------------------------------------


def version() -> str:
    return _take_string(_lib.tm_version())


def is_active() -> bool:
    return bool(_lib.tm_is_active())


def is_usable() -> bool:
    """Whether a dispatch this backend emits will actually run on this machine.

    Stronger than `is_active`: a virtualised host reports a Metal device that
    answers every query and then fails to compile or run an emitted kernel. The
    probe itself lives in the Swift core.
    """
    return bool(_lib.tm_is_usable())


def unusable_reason():
    """Why `is_usable()` is False, or None when the device is usable."""
    return _take_string(_lib.tm_unusable_reason())


def device_name():
    return _take_string(_lib.tm_device_name())


# --- Compiler ---------------------------------------------------------------


def emit_msl(ttir: str, num_simdgroups: int = 4) -> str:
    return _check_string(_lib.tm_emit_msl(ttir.encode(), num_simdgroups))


def kernel_info(ttir: str, num_simdgroups: int = 4) -> str:
    """Launch metadata as a JSON document, computed entirely in the Swift core."""
    return _check_string(_lib.tm_kernel_info(ttir.encode(), num_simdgroups))


def compile_msl(source: str) -> int:
    return _check(_lib.tm_compile_msl(source.encode()), 0)


def load_metallib(image: bytes) -> int:
    return _check(_lib.tm_load_metallib(image, len(image)), 0)


def release_library(handle: int) -> None:
    _check(_lib.tm_release_library(handle), -1)


# --- Runtime ----------------------------------------------------------------


def load_kernel(library: int, name: str) -> int:
    return _check(_lib.tm_load_kernel(library, name.encode()), 0)


def kernel_max_threads(kernel: int) -> int:
    return _check(_lib.tm_kernel_max_threads(kernel), -1)


def release_kernel(handle: int) -> None:
    _check(_lib.tm_release_kernel(handle), -1)


def alloc_buffer(nbytes: int) -> int:
    return _check(_lib.tm_alloc_buffer(nbytes), 0)


def buffer_contents(handle: int) -> int:
    return _check(_lib.tm_buffer_contents(handle), None)


def buffer_length(handle: int) -> int:
    return _check(_lib.tm_buffer_length(handle), -1)


def buffer_write(handle: int, offset: int, address: int, nbytes: int) -> None:
    _check(_lib.tm_buffer_write(handle, offset, ctypes.c_void_p(address), nbytes), -1)


def buffer_read(handle: int, offset: int, address: int, nbytes: int) -> None:
    _check(_lib.tm_buffer_read(handle, offset, ctypes.c_void_p(address), nbytes), -1)


def free_buffer(handle: int) -> None:
    _check(_lib.tm_free_buffer(handle), -1)


#: `tm_launch` argument kinds (mirrors the C ABI table in docs/ARCHITECTURE.md).
ARG_BUFFER = 0
ARG_I32 = 1
ARG_F32 = 2
#: A 64-bit integer scalar (`constant long &`). There is no f64 kind: Metal has
#: no `double` type at all, so an f64 argument is refused by the Swift emitter.
ARG_I64 = 3


def float_bits(value: float) -> int:
    """`f32` scalars travel in the low 32 bits of a `values[i]` slot (see the
    `tm_launch` table in docs/ARCHITECTURE.md)."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def launch(kernel, grid, threads_per_threadgroup, kinds, values) -> None:
    """Dispatch `kernel` over `grid` (a 3-tuple of threadgroup counts).

    `kinds`/`values` are parallel sequences: see ARG_* above.
    """
    count = len(kinds)
    kind_array = (_i32 * count)(*kinds)
    value_array = (_i64 * count)(*values)
    _check(
        _lib.tm_launch(
            kernel,
            grid[0],
            grid[1],
            grid[2],
            threads_per_threadgroup,
            kind_array,
            value_array,
            count,
        ),
        -1,
    )


def live_handle_count() -> int:
    return _lib.tm_live_handle_count()
