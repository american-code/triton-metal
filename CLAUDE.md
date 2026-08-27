# triton-metal — notes for Claude

- **Language policy: Swift first, C ABI for interop.** The Python package in `python/`
  is a ctypes shim that exists only because Triton's plugin discovery imports a Python
  module. Never add logic there — new functionality goes in `Sources/TritonMetalCore`
  with a `tm_*` C export.
- Build/test: `swift build && swift test`; then `cd python && PYTHONPATH=. python3 -m
  pytest tests/ -q` (Python tests need the dylib from `swift build`).
- `xcodebuild` is broken machine-wide as of 2026-08-22 — use `swift build` only.
  Offline Metal compiles use `xcrun metal` directly, which still works.
- Do not vendor Triton source. **Pinned: v3.7.1.** `python/plugin/backend/` vendors
  that tag's `BaseBackend`/`DriverBase` *signatures* only; build Triton from source
  with `TRITON_PLUGIN_DIRS=python/plugin` (docs/USAGE.md §Building Triton with the
  Metal backend — ~9 min, no patch to Triton, no macOS wheel exists).
- When real Triton IR spells something the backend rejects, fix the **Swift** parser
  or emitter. Never parse or special-case IR in Python.
- No cloud AI dependencies. All compilation and execution is local.
- Emit textual MSL (readable) before optimizing to AIR — debuggability first.
