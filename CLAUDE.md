# triton-metal — notes for Claude

- **Language policy: Swift first, C ABI for interop.** The Python package in `python/`
  is a ctypes shim that exists only because Triton's plugin discovery imports a Python
  module. Never add logic there — new functionality goes in `Sources/TritonMetalCore`
  with a `tm_*` C export.
- Build/test: `swift build && swift test`; then `cd python && PYTHONPATH=. python3 -m
  pytest tests/ -q` (Python tests need the dylib from `swift build`).
- `xcodebuild` is broken machine-wide as of 2026-08-22 — use `swift build` only.
  Offline Metal compiles use `xcrun metal` directly, which still works.
- Do not vendor Triton source; pin a release and code against its published plugin
  interface (docs/ARCHITECTURE.md §Compatibility).
- No cloud AI dependencies. All compilation and execution is local.
- Emit textual MSL (readable) before optimizing to AIR — debuggability first.
