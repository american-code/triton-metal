# triton-metal: Triton plugin registration.
#
# Triton's `main.cc` expands TRITON_BACKENDS_TUPLE into one
# `init_triton_<name>(pybind11::module &&)` declaration per backend, so every
# plugin must contribute that symbol to libtriton. `metal.cc` is the whole of
# our C++: the compiler and runtime live in Swift, behind the `tm_*` C ABI.
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMETAL ${CMAKE_CURRENT_SOURCE_DIR}/metal.cc)
  target_link_libraries(TritonMETAL PRIVATE Python3::Module pybind11::headers)
endif()
