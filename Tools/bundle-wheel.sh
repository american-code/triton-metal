#!/usr/bin/env bash
#
# Stage this repository's runtime into a Triton source tree, so that the wheel
# `python setup.py bdist_wheel` produces is the *only* artifact a stranger with a
# Mac has to install.
#
# Run it after `swift build -c release` and immediately before the wheel build:
#
#   swift build -c release
#   Tools/bundle-wheel.sh ~/tm-build/triton
#   cd ~/tm-build/triton && python setup.py bdist_wheel
#
# Two things get staged, and both are copies into the Triton checkout rather than
# any change to Triton itself:
#
#   libtritonmetal.dylib  ->  <plugin>/backend/libtritonmetal.dylib
#
#       Triton symlinks `TRITON_PLUGIN_DIRS/<plugin>/backend` to
#       `python/triton/backends/metal`, and its MANIFEST.in grafts
#       `python/triton` with `include_package_data=True` — so a file dropped in
#       the backend directory ships as package data of `triton.backends.metal`.
#       (`name.conf` already proved that path works.) `_core.py` looks there
#       first: see `_installed_backend_dirs`.
#
#   python/triton_metal/  ->  <triton>/python/triton_metal/
#
#       The ctypes shim itself. Triton's `get_packages()` calls
#       `find_packages(where="python")`, so a package sitting in that directory
#       is picked up and shipped top-level. Without this the wheel would carry a
#       backend that imports `triton_metal` and no `triton_metal` to import —
#       and `python/examples/*.py` could not get a `MetalBuffer` either.
#
# Nothing here is a patch to Triton: no file of Triton's is edited, and the only
# thing added to its tree is one package directory and one library. Both are
# removed by `Tools/bundle-wheel.sh --clean <triton>`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLEAN=0
if [ "${1:-}" = "--clean" ]; then
    CLEAN=1
    shift
fi

TRITON="${1:-}"
if [ -z "$TRITON" ] || [ ! -f "$TRITON/setup.py" ]; then
    echo "usage: Tools/bundle-wheel.sh [--clean] /path/to/triton-checkout" >&2
    echo "(the directory containing Triton's setup.py)" >&2
    exit 1
fi
TRITON="$(cd "$TRITON" && pwd)"

BACKEND="$ROOT/python/plugin/backend"
SHIM_SOURCE="$ROOT/python/triton_metal"
SHIM_DEST="$TRITON/python/triton_metal"

if [ "$CLEAN" = 1 ]; then
    rm -f "$BACKEND/libtritonmetal.dylib"
    rm -rf "$SHIM_DEST"
    echo "Removed the staged dylib and $SHIM_DEST"
    exit 0
fi

# Prefer a release build; fall back to debug so that a `swift build` with no -c
# still produces a working wheel, loudly labelled.
DYLIB=""
for CONFIG in release debug; do
    CANDIDATE="$ROOT/.build/$CONFIG/libtritonmetal.dylib"
    if [ -f "$CANDIDATE" ]; then
        DYLIB="$CANDIDATE"
        [ "$CONFIG" = debug ] && echo "warning: bundling the DEBUG dylib; run 'swift build -c release' first" >&2
        break
    fi
done
if [ -z "$DYLIB" ]; then
    echo "no libtritonmetal.dylib under $ROOT/.build — run 'swift build -c release' first" >&2
    exit 1
fi

cp "$DYLIB" "$BACKEND/libtritonmetal.dylib"
echo "Staged $(basename "$DYLIB") ($(du -h "$DYLIB" | cut -f1)) -> $BACKEND/"

rm -rf "$SHIM_DEST"
# `__pycache__` would ship compiled bytecode for whatever interpreter built the
# wheel, which is at best noise and at worst wrong for the installing one.
/usr/bin/rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SHIM_SOURCE/" "$SHIM_DEST/"
echo "Staged triton_metal -> $SHIM_DEST"

echo
echo "Now build the wheel:"
echo "  cd $TRITON && python setup.py bdist_wheel"
