// triton-metal: the entire C++ surface of the plugin.
//
// Triton's python module declares `void init_triton_metal(pybind11::module &&)`
// for every name in TRITON_BACKENDS_TUPLE and calls it while building
// `triton._C.libtriton`. There is nothing for it to do here: MSL emission,
// metallib compilation, allocation and dispatch all live in the Swift core
// (Sources/TritonMetalCore) behind the `tm_*` C ABI, which the Python shim
// reaches through ctypes. This submodule only reports that the backend was
// linked in, which is also how `triton.backends` sanity checks a plugin build.
#include <pybind11/pybind11.h>

void init_triton_metal(pybind11::module &&m) {
  m.doc() = "triton-metal registration stub; all work happens in "
            "libtritonmetal.dylib (Swift core, tm_* C ABI)";
  m.def("linked", []() { return true; },
        "True when this Triton was built with the Metal plugin.");
}
