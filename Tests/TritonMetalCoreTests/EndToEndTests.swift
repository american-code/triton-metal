import Metal
import XCTest

@testable import TritonMetalCore

/// Full-pipeline tests: Triton IR text -> MSL -> MTLLibrary -> pipeline ->
/// unified-memory buffers -> dispatch -> read back -> compare against a CPU
/// reference. These run on the real GPU.
final class EndToEndTests: XCTestCase {

    /// A host-side kernel argument.
    enum HostArg {
        case floats([Float])
        case ints([Int32])
        /// An output buffer of `count` elements, zero-filled before launch.
        case output(count: Int, elementSize: Int = 4)
        case int32(Int32)
        case float32(Float)
    }

    private struct Run {
        var outputs: [MTLBuffer]
        var kernel: EmittedKernel
    }

    /// Emits, compiles, allocates, launches and blocks — the whole spine.
    private func run(
        ir: String, gridPrograms: Int, args: [HostArg], numSimdgroups: Int = 4
    ) throws -> Run {
        let emission = try MetalCompiler.emit(
            ttir: ir, options: .init(numSimdgroups: numSimdgroups))
        let kernel = try XCTUnwrap(emission.kernels.first)
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: kernel.name)
        XCTAssertEqual(args.count, kernel.arguments.count, "argument count must match the kernel")

        var launchArgs: [MetalRuntime.LaunchArgument] = []
        var outputs: [MTLBuffer] = []
        for arg in args {
            switch arg {
            case .floats(let values):
                let buffer = try MetalRuntime.makeBuffer(length: values.count * 4)
                values.withUnsafeBytes {
                    buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
                }
                launchArgs.append(.buffer(buffer))
            case .ints(let values):
                let buffer = try MetalRuntime.makeBuffer(length: values.count * 4)
                values.withUnsafeBytes {
                    buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
                }
                launchArgs.append(.buffer(buffer))
            case .output(let count, let elementSize):
                let buffer = try MetalRuntime.makeBuffer(length: count * elementSize)
                memset(buffer.contents(), 0, count * elementSize)
                outputs.append(buffer)
                launchArgs.append(.buffer(buffer))
            case .int32(let value):
                launchArgs.append(.int32(value))
            case .float32(let value):
                launchArgs.append(.float32(value))
            }
        }

        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(width: gridPrograms, height: 1, depth: 1),
            threadsPerThreadgroup: kernel.threadsPerThreadgroup,
            arguments: launchArgs)

        return Run(outputs: outputs, kernel: kernel)
    }

    private func floats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    private func ints(_ buffer: MTLBuffer, _ count: Int) -> [Int32] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Int32.self), count: count))
    }

    private func cdiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }

    override func setUpWithError() throws {
        try skipWithoutMetal()
    }

    // MARK: - Kernels

    func testCopyKernelMatchesCPU() throws {
        let n = 1000  // deliberately not a multiple of BLOCK=256
        let input = (0..<n).map { Float($0) * 0.5 - 3 }
        let result = try run(
            ir: IRFixtures.copy,
            gridPrograms: cdiv(n, 256),
            args: [.floats(input), .output(count: n), .int32(Int32(n))])
        XCTAssertEqual(floats(result.outputs[0], n), input)
    }

    func testVectorAddMatchesCPU() throws {
        let n = 98_432
        let a = (0..<n).map { Float($0 % 97) * 0.25 }
        let b = (0..<n).map { Float($0 % 31) * -1.5 }
        let result = try run(
            ir: IRFixtures.vectorAdd,
            gridPrograms: cdiv(n, 1024),
            args: [.floats(a), .floats(b), .output(count: n), .int32(Int32(n))])
        XCTAssertEqual(floats(result.outputs[0], n), zip(a, b).map(+))
    }

    func testVectorMulWithMaskAndOtherMatchesCPU() throws {
        let n = 501  // 3.9 blocks of 128 — the tail block is fully masked
        let a = (0..<n).map { Float($0) * 0.125 }
        let b = (0..<n).map { 2 - Float($0) * 0.0625 }
        let result = try run(
            ir: IRFixtures.vectorMul,
            gridPrograms: cdiv(n, 128),
            args: [.floats(a), .floats(b), .output(count: n), .int32(Int32(n))])
        XCTAssertEqual(floats(result.outputs[0], n), zip(a, b).map(*))
    }

    func testScaleBiasWithFloatScalarsMatchesCPU() throws {
        let n = 300
        let input = (0..<n).map { Float($0) - 150 }
        let scale: Float = 0.75
        let bias: Float = -2.25
        let result = try run(
            ir: IRFixtures.scaleBias,
            gridPrograms: cdiv(n, 64),
            args: [
                .floats(input), .output(count: n), .float32(scale), .float32(bias),
                .int32(Int32(n)),
            ])
        XCTAssertEqual(floats(result.outputs[0], n), input.map { $0 * scale + bias })
    }

    func testIntegerKernelMatchesCPU() throws {
        let n = 77
        let a = (0..<n).map { Int32($0 * 3 - 10) }
        let b = (0..<n).map { Int32($0 % 5) }
        let result = try run(
            ir: IRFixtures.integerAdd,
            gridPrograms: cdiv(n, 32),
            args: [.ints(a), .ints(b), .output(count: n), .int32(Int32(n))])
        XCTAssertEqual(ints(result.outputs[0], n), zip(a, b).map { ($0 + $1) * 3 })
    }

    /// Masking must protect memory, not just produce right answers: allocate the
    /// output exactly `n` long so an out-of-bounds store would corrupt the guard
    /// region right after it.
    func testMaskPreventsOutOfBoundsStores() throws {
        let n = 130  // 1.02 blocks of 128
        let a = [Float](repeating: 1, count: n)
        let b = [Float](repeating: 2, count: n)
        let guardElements = 64
        let emission = try MetalCompiler.emit(ttir: IRFixtures.vectorMul, options: .init())
        let kernel = emission.kernels[0]
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: kernel.name)

        let bufferA = try MetalRuntime.makeBuffer(length: n * 4)
        let bufferB = try MetalRuntime.makeBuffer(length: n * 4)
        let output = try MetalRuntime.makeBuffer(length: (n + guardElements) * 4)
        a.withUnsafeBytes { bufferA.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        b.withUnsafeBytes { bufferB.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let sentinel = Float(-99)
        let outPointer = output.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(n + guardElements) { outPointer[i] = sentinel }

        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(width: cdiv(n, 128), height: 1, depth: 1),
            threadsPerThreadgroup: kernel.threadsPerThreadgroup,
            arguments: [
                .buffer(bufferA), .buffer(bufferB), .buffer(output), .int32(Int32(n)),
            ])

        XCTAssertEqual(floats(output, n + guardElements).prefix(n), ArraySlice(a.map { $0 * 2 }))
        XCTAssertEqual(
            Array(floats(output, n + guardElements).suffix(guardElements)),
            [Float](repeating: sentinel, count: guardElements),
            "a masked tt.store must not write past n_elements")
    }

    /// The same kernel must be correct at every threadgroup size, since threads
    /// stride over the block rather than owning exactly one lane.
    func testResultsAreIndependentOfThreadgroupSize() throws {
        let n = 4096
        let a = (0..<n).map { Float($0) }
        let b = (0..<n).map { Float(n - $0) }
        let expected = zip(a, b).map(+)
        for simdgroups in [1, 2, 4, 8, 16, 32] {
            let result = try run(
                ir: IRFixtures.vectorAdd,
                gridPrograms: cdiv(n, 1024),
                args: [.floats(a), .floats(b), .output(count: n), .int32(Int32(n))],
                numSimdgroups: simdgroups)
            XCTAssertEqual(
                floats(result.outputs[0], n), expected, "wrong result at num_warps=\(simdgroups)")
        }
    }

    // MARK: - Runtime plumbing

    func testUnifiedBuffersAreHostVisible() throws {
        let buffer = try MetalRuntime.makeBuffer(length: 16)
        XCTAssertEqual(buffer.length, 16)
        XCTAssertEqual(buffer.storageMode, .shared)
    }

    func testLoadKernelReportsMissingFunction() throws {
        let library = try MetalCompiler.compileMSL(
            try MetalCompiler.emitMSL(ttir: IRFixtures.copy, options: .init()))
        XCTAssertThrowsError(try MetalRuntime.loadKernel(library: library, kernelName: "nope")) {
            XCTAssertTrue(
                "\($0)".contains("not found in library") && "\($0)".contains("copy_kernel"),
                "\($0)")
        }
    }

    func testInvalidMSLSurfacesTheCompilerDiagnostic() {
        XCTAssertThrowsError(try MetalCompiler.compileMSL("kernel void k() { this is not MSL }")) {
            XCTAssertTrue("\($0)".contains("MSL compilation failed"), "\($0)")
        }
    }

    func testLaunchRejectsOversizedThreadgroups() throws {
        let emission = try MetalCompiler.emit(ttir: IRFixtures.copy, options: .init())
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "copy_kernel")
        let buffer = try MetalRuntime.makeBuffer(length: 64)
        XCTAssertThrowsError(
            try MetalRuntime.launch(
                pipeline: pipeline,
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: 4096,
                arguments: [.buffer(buffer), .buffer(buffer), .int32(16)])
        ) {
            XCTAssertTrue("\($0)".contains("exceeds the pipeline maximum"), "\($0)")
        }
    }

    /// The optional offline path (`xcrun metal`) must produce a metallib the
    /// runtime can load. Skipped if the Metal toolchain isn't installed.
    func testOfflineMetallibPathRoundTrips() throws {
        let source = try MetalCompiler.emitMSL(ttir: IRFixtures.copy, options: .init())
        let metallib: Data
        do {
            metallib = try MetalCompiler.compileMSLOffline(source)
        } catch {
            throw XCTSkip("xcrun metal unavailable: \(error)")
        }
        XCTAssertGreaterThan(metallib.count, 0)
        let pipeline = try MetalRuntime.loadKernel(metallib: metallib, kernelName: "copy_kernel")
        XCTAssertGreaterThan(pipeline.maxTotalThreadsPerThreadgroup, 0)
    }

    /// The offline path must report `xcrun metal`'s own diagnostics rather than
    /// an empty `Data`, and must not leave its scratch directory behind.
    func testOfflineCompileReportsMetalFrontEndDiagnostics() throws {
        do {
            _ = try MetalCompiler.compileMSLOffline("kernel void k() { not msl }")
            XCTFail("expected `xcrun metal` to reject this source")
        } catch let error as CoreError {
            guard case .metal(let message) = error else {
                return XCTFail("expected a metal error, got \(error)")
            }
            try XCTSkipIf(
                message.contains("could not run"), "xcrun metal unavailable: \(message)")
            XCTAssertTrue(message.contains("`xcrun metal` failed"), message)
            XCTAssertTrue(message.contains("error:"), message)
        }
    }

    /// An unknown `-std=` is the cheapest way to prove the standard really is
    /// passed through to the front end rather than hard-coded.
    func testOfflineCompileHonoursTheRequestedLanguageStandard() throws {
        let source = try MetalCompiler.emitMSL(ttir: IRFixtures.copy, options: .init())
        do {
            _ = try MetalCompiler.compileMSLOffline(source, standard: "metal0.1")
            XCTFail("expected `xcrun metal` to reject an unknown -std")
        } catch let error as CoreError {
            guard case .metal(let message) = error else {
                return XCTFail("expected a metal error, got \(error)")
            }
            try XCTSkipIf(message.contains("could not run"), "xcrun metal unavailable")
            XCTAssertTrue(message.contains("metal0.1"), message)
        }
    }

    /// Buffer and dispatch geometry are validated in the Swift core, so the shim
    /// never has to. Metal itself would trap on most of these.
    func testRuntimeRejectsDegenerateGeometry() throws {
        XCTAssertThrowsError(try MetalRuntime.makeBuffer(length: 0)) {
            XCTAssertTrue("\($0)".contains("must be positive"), "\($0)")
        }
        XCTAssertThrowsError(try MetalRuntime.makeBuffer(length: -8)) {
            XCTAssertTrue("\($0)".contains("got -8"), "\($0)")
        }

        let emission = try MetalCompiler.emit(ttir: IRFixtures.copy, options: .init())
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "copy_kernel")
        let buffer = try MetalRuntime.makeBuffer(length: 64)
        let arguments: [MetalRuntime.LaunchArgument] = [
            .buffer(buffer), .buffer(buffer), .int32(16),
        ]

        XCTAssertThrowsError(
            try MetalRuntime.launch(
                pipeline: pipeline,
                threadgroups: MTLSize(width: 1, height: 0, depth: 1),
                threadsPerThreadgroup: 32, arguments: arguments)
        ) {
            XCTAssertTrue("\($0)".contains("grid must be positive"), "\($0)")
        }
        XCTAssertThrowsError(
            try MetalRuntime.launch(
                pipeline: pipeline,
                threadgroups: MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: 0, arguments: arguments)
        ) {
            XCTAssertTrue("\($0)".contains("threadsPerThreadgroup must be positive"), "\($0)")
        }
    }

    /// Handles are untyped `Int64`s across the C ABI, so the table has to catch a
    /// caller that passes a buffer where a library belongs.
    func testHandleTableRejectsAMistypedHandle() throws {
        let table = HandleTable()
        let handle = table.insert(try MetalRuntime.makeBuffer(length: 16))
        XCTAssertNoThrow(try table.lookup(handle, as: MTLBuffer.self, what: "buffer"))
        XCTAssertThrowsError(try table.lookup(handle, as: MTLLibrary.self, what: "library")) {
            XCTAssertTrue("\($0)".contains("is not a library"), "\($0)")
        }
        XCTAssertEqual(table.count, 1)
        XCTAssertTrue(table.remove(handle))
        XCTAssertFalse(table.remove(handle))
        XCTAssertEqual(table.count, 0)
    }
}
