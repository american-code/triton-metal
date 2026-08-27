"""`triton.backends.metal` — the package Triton's plugin discovery imports.

Triton links this directory into `triton/backends/<name>` (name from
`name.conf`) and then imports `.compiler` and `.driver`, expecting exactly one
concrete `BaseBackend` and one concrete `DriverBase` in them. Both are adapters
over `triton_metal`, whose only job is to call the Swift core's `tm_*` ABI.
"""
