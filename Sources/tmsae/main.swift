import Foundation
import MLX
import Metal
import TritonMetalCore
import TritonMetalMLX

// The SAE-encoder demo: correctness against MLX's own ops, the buffer-binding
// finding, and the per-launch host cost.
//
// An executable rather than an XCTest case because the lab machines have the
// command-line tools and no Xcode, so `swift test` cannot run there — and they
// are where the numbers worth quoting get measured (the same reason `tmbench`
// exists). `swift test` covers the same kernel through `SAEEncoderTests`.
//
//   tmsae                 correctness, binding, and the launch timings
//   tmsae --emit-ir       the kernel's Triton IR on stdout, for the Python
//                         comparison in python/examples/sae_encode.py — so both
//                         languages lower byte-identical text and neither can
//                         drift from the other

let arguments = Array(CommandLine.arguments.dropFirst())
let blocking = SAEEncoder.Blocking()

if arguments.contains("--emit-ir") {
    print(SAEEncoder.ir(blockM: blocking.m, blockF: blocking.f, blockD: blocking.d))
    exit(0)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

if let reason = MetalRuntime.unusableReason {
    fail("this machine cannot run an emitted kernel: \(reason)")
}

// MLX's failure to find its shader library is fatal rather than throwable, so
// the file is checked before the first MLX call rather than after it.
let binaryDirectory = Bundle.main.executableURL?.deletingLastPathComponent()
if let binaryDirectory,
    !FileManager.default.fileExists(
        atPath: binaryDirectory.appendingPathComponent("mlx.metallib").path)
{
    fail(
        """
        mlx.metallib is not next to this binary, so MLX cannot run a single GPU op and this
        program would abort rather than report anything. Looked in \(binaryDirectory.path).

        Install it with:    Tools/fetch-metallib.sh

        mlx-swift compiles the MLX C++ core but does not build the Metal shader library — that
        is Xcode's job, and this binary is meant to run under plain `swift build`.
        """)
}

let iterations = 500
print("device: \(MetalRuntime.defaultDeviceName() ?? "unknown")")

/// Deterministic inputs. A seeded MLX RNG would make this a test of MLX's RNG
/// stream, which is not a thing to pin a numerical comparison to.
func synthetic(_ shape: [Int], offset: Int, modulus: Int) -> MLXArray {
    let count = shape.reduce(1, *)
    var values = [Float]()
    values.reserveCapacity(count)
    for i in 0..<count {
        values.append(Float(((i + offset) * 13) % modulus) / Float(modulus) - 0.5)
    }
    return MLXArray(values, shape)
}

// MARK: - Correctness against MLX's own ops

print("\n--- relu(x @ W_enc + b_enc): Triton kernel vs MLX ops ---")
let encoder = try SAEEncoder.compile(blocking: blocking)
print(
    "block \(blocking.m)x\(blocking.f)x\(blocking.d), "
        + "\(encoder.pipeline.threadsPerThreadgroup) threads/threadgroup")

var worst = Float(0)
for (rows, model, features) in [(8, 64, 128), (64, 512, 2048), (100, 300, 700), (1, 512, 512)] {
    let x = synthetic([rows, model], offset: 1, modulus: 71)
    let w = synthetic([model, features], offset: 7, modulus: 53)
    let b = synthetic([features], offset: 3, modulus: 29)
    let got = try encoder(x, w, b)
    let want = SAEEncoder.reference(x, w, b)
    let error = MLX.abs(got - want).max().item(Float.self)
    let scale = MLX.maximum(MLX.abs(want).max(), MLXArray(Float(1))).item(Float.self)
    worst = max(worst, error / scale)
    print(
        "[\(rows), \(model)] @ [\(model), \(features)]: max |kernel - mlx| = \(error), "
            + "relative \(error / scale)")
}
print("worst relative error: \(worst)")

// MARK: - Zero copy, checked rather than claimed

// Proved by writing, not by comparing pointers. Comparing `buffer.contents()`
// against `asData(access: .noCopy)`'s base address answers the question for a
// large array and lies about a small one: Foundation stores payloads of 14 bytes
// or fewer inline, so the `Data` wrapper's address is a temporary rather than
// the array's storage. The binding never goes through `Data` — `asMTLBuffer`
// reaches the backing directly — so the check has to as well.
print("\n--- buffer binding: does a kernel's store land in the array? ---")
let device = MetalRuntime.device!
let fill = try MetalCompiler.compileMSL(
    """
    #include <metal_stdlib>
    using namespace metal;
    kernel void tmsae_fill(device float *o [[buffer(0)]], constant int &n [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
        if (i < uint(n)) { o[i] = float(i) + 1.0f; }
    }
    """, kernelName: "tmsae_fill")

for shape in [[64, 512], [512], [1, 3]] {
    let array = MLX.zeros(shape, dtype: .float32)
    array.eval()
    let buffer = array.asMTLBuffer(device: device, noCopy: true)!
    let threads = min(1024, max(32, array.size))
    try MetalRuntime.launch(
        pipeline: fill,
        threadgroups: MTLSize(width: Grid.covering(array.size, block: threads), height: 1, depth: 1),
        threadsPerThreadgroup: threads,
        arguments: [.buffer(buffer), .int32(Int32(array.size))])
    let values = array.asArray(Float.self)
    let landed = values.enumerated().allSatisfy { $1 == Float($0) + 1 }
    print(
        "contiguous \(shape) (\(array.nbytes) B): aliased = "
            + "\(MLXBinding.isContiguous(array)), kernel's stores visible through MLX = \(landed)")
}
let matrix = MLX.zeros([64, 64], dtype: .float32)
matrix.eval()
let column = matrix[0..., 3]
print(
    "strided view \(column.shape): contiguous = \(MLXBinding.isContiguous(column)) "
        + "-> the binding stages a copy")

// MARK: - Host dispatch overhead

// The claim under test is "no Python in the hot path". Measuring it needs two
// numbers, not one, because the two paths differ in exactly one place and it is
// not the dispatch:
//
//   dispatch      encode + commit + wait, with the arguments already bound — the
//                 hot loop of a caller whose tensors do not change, which is what
//                 `prepare` exists for. Both languages reach the same
//                 `MetalRuntime.launch` here (Python through `tm_launch`), so the
//                 difference between the two numbers is the host path and nothing
//                 else. This is where "no Python in the hot path" is or is not
//                 worth something.
//
//   per call      what it costs to run the kernel on tensors the caller has just
//                 produced, binding included. That is where the paths part for a
//                 different reason: an `MLXArray` is already unified memory, so
//                 this side binds and dispatches, while a Python caller's `numpy`
//                 array is host memory that has to be written into a Metal buffer
//                 and read back out (macOS torch wheels are CPU-only, which is
//                 why the shim has `MetalBuffer` at all).
//
// The other half of the comparison is `python/examples/sae_encode.py`, which
// runs the *same IR text* (`tmsae --emit-ir`) and reports the same two numbers.
print("\n--- host dispatch overhead: compile once, launch \(iterations) times ---")
for (rows, model, features) in [(8, 64, 64), (64, 64, 64), (64, 512, 2048)] {
    let x = synthetic([rows, model], offset: 1, modulus: 71)
    let w = synthetic([model, features], offset: 7, modulus: 53)
    let b = synthetic([features], offset: 3, modulus: 29)
    let out = MLX.zeros([rows, features], dtype: .float32)
    for array in [x, w, b, out] { array.eval() }
    let grid = Grid(
        Grid.covering(rows, block: blocking.m), Grid.covering(features, block: blocking.f))
    let scalars: [LaunchScalar] = [.init(rows), .init(model), .init(features)]

    func time(_ body: () throws -> Void) rethrows -> Double {
        for _ in 0..<20 { try body() }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations { try body() }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / Double(iterations) / 1000
    }

    let prepared = try encoder.pipeline.prepare(arrays: [x, w, b, out], scalars: scalars)
    let dispatch = try time { _ = try encoder.pipeline.launch(prepared, grid: grid) }
    let perCall = try time {
        _ = try encoder.pipeline.launch(arrays: [x, w, b, out], scalars: scalars, grid: grid)
    }
    print(
        "[\(rows), \(model)] @ [\(model), \(features)], grid \(grid.x)x\(grid.y): "
            + String(
                format: "dispatch %.1f us, per call %.1f us (binding %.1f us)",
                dispatch, perCall, perCall - dispatch))
}

// Why `prepare` exists, in two measurements.
//
// Binding one array is cheap and flat in its size: Metal is wrapping pages MLX's
// allocator already got from Metal, not wiring new ones.
print("\n--- cost of binding one array, by size ---")
for bytes in [4096, 65536, 1 << 20, 4 << 20, 16 << 20] {
    let array = MLX.zeros([bytes / 4], dtype: .float32)
    array.eval()
    for _ in 0..<20 { _ = array.asMTLBuffer(device: device, noCopy: true) }
    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iterations { _ = array.asMTLBuffer(device: device, noCopy: true) }
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / Double(iterations) / 1000
    print(String(format: "%9d B: asMTLBuffer %5.1f us", bytes, elapsed))
}

// The cost that actually matters is not the call, it is handing a command buffer
// a resource it has not seen before: same kernel, same grid, same arrays, and the
// only difference is whether the `MTLBuffer` objects are reused or made fresh
// each time. That difference is what `prepare` removes, and it is why the "per
// call" column above is well above the ~4 us per array the binding itself costs.
print("\n--- same dispatch, fresh buffers vs reused ---")
do {
    let big = MLX.zeros([512 * 2048], dtype: .float32)
    let small = MLX.zeros([64 * 64], dtype: .float32)
    for array in [big, small] { array.eval() }
    let sizes: [(String, [MLXArray])] = [
        ("4 x 16 KB", [small, small, small, small]),
        ("3 x 4 MB + 16 KB", [big, big, small, big]),
    ]
    let one = MTLSize(width: 1, height: 1, depth: 1)
    let scalars: [MetalRuntime.LaunchArgument] = [.int32(64), .int32(64), .int32(64)]
    for (label, arrays) in sizes {
        func bound() -> [MetalRuntime.LaunchArgument] {
            arrays.map { .buffer($0.asMTLBuffer(device: device, noCopy: true)!) } + scalars
        }
        let reusedArguments = bound()
        func time(_ body: () throws -> Void) rethrows -> Double {
            for _ in 0..<20 { try body() }
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<iterations { try body() }
            return Double(DispatchTime.now().uptimeNanoseconds - start) / Double(iterations) / 1000
        }
        func dispatch(_ arguments: [MetalRuntime.LaunchArgument]) throws {
            try MetalRuntime.launch(
                pipeline: encoder.pipeline.pipeline, threadgroups: one,
                threadsPerThreadgroup: encoder.pipeline.threadsPerThreadgroup,
                arguments: arguments)
        }
        let reused = try time { try dispatch(reusedArguments) }
        let fresh = try time { try dispatch(bound()) }
        print(
            String(
                format: "%-18s reused %6.1f us, fresh %6.1f us (%+.1f us for 4 buffers)",
                (label as NSString).utf8String!, reused, fresh, fresh - reused))
    }
}

print("\nok")
