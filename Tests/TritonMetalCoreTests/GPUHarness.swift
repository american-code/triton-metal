import Metal
import XCTest

@testable import TritonMetalCore

/// bf16 on the host. Swift has no `BFloat16` and this side only has to move
/// bytes: a bf16 is the top 16 bits of the f32 with the same value, rounded to
/// nearest even, which is exactly why bf16 and f32 interconvert for free on the
/// GPU as well.
enum BFloat16 {
    static func encode(_ value: Float) -> UInt16 {
        if value.isNaN { return 0x7FC0 }
        let bits = value.bitPattern
        let rounding: UInt32 = ((bits >> 16) & 1) &+ 0x7FFF
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }

    static func decode(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
}

/// One host-side kernel argument.
enum HostArg {
    case floats([Float])
    case halves([Float16])
    /// bf16 values given as `Float`; encoded on the way to the buffer.
    case bfloats([Float])
    case ints([Int32])
    case shorts([Int16])
    /// An output buffer of `count` elements of `stride` bytes, zero-filled.
    case output(count: Int, stride: Int = 4)
    case int32(Int32)
    case int64(Int64)
    case float32(Float)
}

struct KernelRun {
    /// Output buffers, in the order they appeared in the argument list.
    var outputs: [MTLBuffer]
    /// Every buffer argument, keyed by its position in the argument list — an
    /// atomic kernel accumulates *into* a buffer it was handed pre-filled, so the
    /// interesting result is not always an `.output`.
    var buffers: [Int: MTLBuffer]
    var kernel: EmittedKernel
    var source: String
}

/// Emit -> compile -> allocate -> launch -> block. The whole spine, on the real
/// GPU, shared by every end-to-end test.
enum GPU {
    static func run(
        ir: String, grid: (Int, Int, Int) = (1, 1, 1), args: [HostArg], numSimdgroups: Int = 4
    ) throws -> KernelRun {
        let emission = try MetalCompiler.emit(ttir: ir, options: .init(numSimdgroups: numSimdgroups))
        guard let kernel = emission.kernels.first else {
            throw CoreError.lowering("no kernel emitted", .unknown)
        }
        guard args.count == kernel.arguments.count else {
            throw CoreError.invalidArgument(
                "kernel '\(kernel.name)' takes \(kernel.arguments.count) arguments, got \(args.count)")
        }
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: kernel.name)

        var launchArgs: [MetalRuntime.LaunchArgument] = []
        var outputs: [MTLBuffer] = []
        var buffers: [Int: MTLBuffer] = [:]
        for (position, arg) in args.enumerated() {
            var buffer: MTLBuffer? = nil
            switch arg {
            case .floats(let values): buffer = try upload(values)
            case .halves(let values): buffer = try upload(values)
            case .bfloats(let values): buffer = try upload(values.map(BFloat16.encode))
            case .ints(let values): buffer = try upload(values)
            case .shorts(let values): buffer = try upload(values)
            case .output(let count, let stride):
                let allocated = try MetalRuntime.makeBuffer(length: max(1, count * stride))
                memset(allocated.contents(), 0, allocated.length)
                outputs.append(allocated)
                buffer = allocated
            case .int32(let value): launchArgs.append(.int32(value))
            case .int64(let value): launchArgs.append(.int64(value))
            case .float32(let value): launchArgs.append(.float32(value))
            }
            if let buffer {
                buffers[position] = buffer
                launchArgs.append(.buffer(buffer))
            }
        }

        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(width: grid.0, height: grid.1, depth: grid.2),
            threadsPerThreadgroup: kernel.threadsPerThreadgroup,
            arguments: launchArgs)

        return KernelRun(
            outputs: outputs, buffers: buffers, kernel: kernel, source: emission.source)
    }

    static func upload<T>(_ values: [T]) throws -> MTLBuffer {
        let buffer = try MetalRuntime.makeBuffer(length: max(1, values.count * MemoryLayout<T>.stride))
        values.withUnsafeBytes {
            if let base = $0.baseAddress { buffer.contents().copyMemory(from: base, byteCount: $0.count) }
        }
        return buffer
    }

    static func read<T>(_ buffer: MTLBuffer, _ type: T.Type, _ count: Int) -> [T] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: T.self), count: count))
    }

    static func cdiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }
}

extension XCTestCase {
    /// Skips unless this machine can actually run an emitted kernel.
    ///
    /// Not the same question as "is there a device". A virtualised host — a
    /// GitHub Actions macOS runner — reports an "Apple Paravirtual device" that
    /// answers every Metal query and then fails to compile or dispatch what the
    /// emitter produces, so the probe in `MetalRuntime.unusableReason` runs a
    /// real kernel (a store and a simdgroup multiply-accumulate) once per
    /// process and this gate reports what it found.
    func skipWithoutMetal() throws {
        if let reason = MetalRuntime.unusableReason { throw XCTSkip(reason) }
    }

    /// Elementwise comparison with an absolute/relative tolerance, reporting the
    /// worst offender rather than dumping thousands of values.
    func assertClose(
        _ actual: [Float], _ expected: [Float], tolerance: Float, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, message, file: file, line: line)
        guard actual.count == expected.count else { return }
        var worstIndex = -1
        var worstError: Float = 0
        for index in actual.indices {
            let a = actual[index]
            let b = expected[index]
            if a.isNaN != b.isNaN {
                XCTFail(
                    "\(message) NaN mismatch at \(index): \(a) vs \(b)", file: file, line: line)
                return
            }
            if a.isNaN { continue }
            let error = abs(a - b) / max(1, abs(b))
            if error > worstError {
                worstError = error
                worstIndex = index
            }
        }
        if worstError > tolerance, worstIndex >= 0 {
            XCTFail(
                "\(message) worst relative error \(worstError) > \(tolerance) at index "
                    + "\(worstIndex): got \(actual[worstIndex]), expected \(expected[worstIndex])",
                file: file, line: line)
        }
    }
}
