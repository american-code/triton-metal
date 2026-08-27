"""A `.data_ptr()` object over a Swift-owned unified-memory buffer.

Triton identifies a kernel argument as a tensor by duck-typing: anything with a
`data_ptr` attribute is a pointer argument, and its `dtype` names the pointee
(`triton/_utils.py: canonicalize_dtype`). Torch is the usual provider, but the
macOS torch wheel is CPU-only, and a CPU tensor's memory is not a `MTLBuffer` —
so the demo path allocates through `tm_alloc_buffer` and hands Triton one of
these instead.

No logic beyond bookkeeping: allocation, the host address and the copies are all
`tm_*` calls into the Swift core.
"""

import ctypes

from . import _core

#: Element sizes for the dtype names Triton canonicalizes to (`fp32`, `i32`, ...).
_ITEMSIZE = {
    "float16": 2, "float32": 4, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
}

_CTYPE = {
    "float16": ctypes.c_uint16, "float32": ctypes.c_float, "int8": ctypes.c_int8,
    "int16": ctypes.c_int16, "int32": ctypes.c_int32, "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8, "uint16": ctypes.c_uint16, "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
}


class MetalBuffer:
    """Unified memory the GPU can read and the CPU can address."""

    def __init__(self, shape, dtype="float32"):
        if isinstance(shape, int):
            shape = (shape, )
        dtype = str(dtype).split(".")[-1]
        if dtype not in _ITEMSIZE:
            raise TypeError(f"unsupported dtype {dtype!r}; expected one of {sorted(_ITEMSIZE)}")
        self.shape = tuple(shape)
        self.dtype = dtype
        self.numel = 1
        for d in self.shape:
            self.numel *= d
        self.itemsize = _ITEMSIZE[dtype]
        self.nbytes = self.numel * self.itemsize
        self.tm_handle = _core.alloc_buffer(self.nbytes)

    # -- what Triton looks for ------------------------------------------------

    def data_ptr(self):
        return _core.buffer_contents(self.tm_handle)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    # -- host access ----------------------------------------------------------

    def _view(self):
        return (_CTYPE[self.dtype] * self.numel).from_address(self.data_ptr())

    def copy_from(self, address, nbytes=None):
        """Copy `nbytes` from a host address (e.g. `numpy_array.ctypes.data`)."""
        _core.buffer_write(self.tm_handle, 0, address, self.nbytes if nbytes is None else nbytes)

    def copy_to(self, address, nbytes=None):
        _core.buffer_read(self.tm_handle, 0, address, self.nbytes if nbytes is None else nbytes)

    def tolist(self):
        return list(self._view())

    def numpy(self):
        import numpy as np
        out = np.empty(self.shape, dtype=np.dtype(self.dtype))
        self.copy_to(out.ctypes.data, out.nbytes)
        return out

    @classmethod
    def from_numpy(cls, array):
        import numpy as np
        array = np.ascontiguousarray(array)
        buf = cls(array.shape, str(array.dtype))
        buf.copy_from(array.ctypes.data, array.nbytes)
        return buf

    @classmethod
    def from_torch(cls, tensor):
        """Copy a CPU torch tensor into unified memory.

        A copy, not a view: `MTLDevice.makeBuffer(bytesNoCopy:)` needs page-aligned
        memory, which a torch allocation is not. Fine for v1 — see docs/USAGE.md.
        """
        tensor = tensor.contiguous().cpu()
        buf = cls(tuple(tensor.shape), str(tensor.dtype))
        buf.copy_from(tensor.data_ptr(), buf.nbytes)
        return buf

    def to_torch(self):
        import torch
        out = torch.empty(self.shape, dtype=getattr(torch, self.dtype))
        self.copy_to(out.data_ptr(), self.nbytes)
        return out

    def free(self):
        if self.tm_handle:
            _core.free_buffer(self.tm_handle)
            self.tm_handle = 0

    def __repr__(self):
        return f"MetalBuffer(shape={self.shape}, dtype='{self.dtype}', handle={self.tm_handle})"
