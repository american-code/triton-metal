"""Driver shim: device queries and launches delegate to the Swift core."""

from . import _core


class MetalDriver:
    @staticmethod
    def is_active() -> bool:
        return _core.is_active()

    def get_current_target(self):
        return ("metal", _core.device_name())

    def get_benchmarker(self):
        raise NotImplementedError("GPU-timestamp do_bench (Swift core)")

    def load_binary(self, name: str, library: int, shared: int, device):
        return _core.load_kernel(library, name)
