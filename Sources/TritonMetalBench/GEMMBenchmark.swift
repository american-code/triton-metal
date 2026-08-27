import Foundation
import Metal
import MetalPerformanceShaders
import TritonMetalCore

/// Square-GEMM throughput of the lowered matmul tutorial against
/// `MPSMatrixMultiplication`, the fastest thing Apple ships for the same job.
///
/// The measurement lives here rather than in the XCTest bundle so that it can run
/// on a machine that has only the command-line tools (no Xcode, therefore no
/// XCTest). `Tests/.../MatmulBenchmark.swift` and `Sources/tmbench` are both thin
/// wrappers around this.

/// One point of the autotuning search space: the Triton block shape, the
/// threadgroup width, and the per-simdgroup register blocking factors.
public struct GEMMConfig: Sendable, Hashable {
    public var blockM: Int
    public var blockN: Int
    public var blockK: Int
    public var simdgroups: Int
    /// Output fragments per simdgroup, `0` meaning "let the emitter choose".
    public var registerM: Int
    public var registerN: Int
    /// Consecutive columns one thread stages, `0` meaning "let the emitter choose".
    public var stagingUnroll: Int
    /// Elements of slack after each tile row, `-1` meaning "let the emitter choose".
    public var tilePadding: Int
    /// Ping-pong the operand tiles between two arena halves.
    public var doubleBuffer: Bool
    /// Read a staged run of four columns with one vector load. On by default;
    /// the sweep turns it *off* to keep the comparison measurable.
    public var vectorStaging: Bool

    public init(
        blockM: Int, blockN: Int, blockK: Int, simdgroups: Int, registerM: Int = 0,
        registerN: Int = 0, stagingUnroll: Int = 0, tilePadding: Int = -1,
        doubleBuffer: Bool = false, vectorStaging: Bool = true
    ) {
        self.vectorStaging = vectorStaging
        self.blockM = blockM
        self.blockN = blockN
        self.blockK = blockK
        self.simdgroups = simdgroups
        self.registerM = registerM
        self.registerN = registerN
        self.stagingUnroll = stagingUnroll
        self.tilePadding = tilePadding
        self.doubleBuffer = doubleBuffer
    }

    public var options: MetalCompiler.Options {
        var options = MetalCompiler.Options(numSimdgroups: simdgroups)
        options.dotRegisterM = registerM
        options.dotRegisterN = registerN
        options.dotStagingUnroll = stagingUnroll
        options.dotTilePadding = tilePadding
        options.dotDoubleBuffer = doubleBuffer
        options.dotVectorStaging = vectorStaging
        return options
    }

    /// `64x64x32/w16/r2x2/u4` — parses back through `GEMMConfig.parse`.
    public var name: String {
        let blocking = registerM > 0 || registerN > 0
            ? "/r\(max(1, registerM))x\(max(1, registerN))" : ""
        let staging = stagingUnroll > 0 ? "/u\(stagingUnroll)" : ""
        let padding = tilePadding >= 0 ? "/p\(tilePadding)" : ""
        let buffering = (doubleBuffer ? "/db" : "") + (vectorStaging ? "" : "/nov4")
        return "\(blockM)x\(blockN)x\(blockK)/w\(simdgroups)\(blocking)\(staging)\(padding)"
            + buffering
    }

    /// `M,N,K,W[,RM,RN[,U[,P[,DB]]]]`, the spelling `tmbench --config` takes.
    public static func parse(_ text: String) -> GEMMConfig? {
        let fields = text.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard [4, 6, 7, 8, 9, 10].contains(fields.count), !fields.contains(where: { $0 == nil })
        else {
            return nil
        }
        let values = fields.map { $0! }
        return GEMMConfig(
            blockM: values[0], blockN: values[1], blockK: values[2], simdgroups: values[3],
            registerM: values.count >= 6 ? values[4] : 0,
            registerN: values.count >= 6 ? values[5] : 0,
            stagingUnroll: values.count >= 7 ? values[6] : 0,
            tilePadding: values.count >= 8 ? values[7] : -1,
            doubleBuffer: values.count >= 9 && values[8] != 0,
            vectorStaging: values.count < 10 || values[9] != 0)
    }
}

public struct GEMMMeasurement: Sendable {
    public var size: Int
    public var config: GEMMConfig
    public var gflops: Double
    public var mpsGflops: Double
    public var ratio: Double { mpsGflops > 0 ? gflops / mpsGflops : 0 }
}

public enum GEMMBenchmark {

    // MARK: - Search space

    /// Block shapes worth trying. The larger ones only fit once the accumulator
    /// stops sharing the threadgroup budget with the operand tiles, so several of
    /// these are rejected by the emitter on the narrow configurations — the sweep
    /// simply skips whatever fails to lower.
    public static let blockShapes = [
        (32, 32), (32, 64), (64, 32), (64, 64), (64, 128), (128, 64), (128, 128),
    ]
    public static let blockKs = [16, 32, 64]
    public static let simdgroupCounts = [4, 8, 16, 32]
    /// Per-simdgroup output-fragment blocking. `(0, 0)` asks the emitter to pick.
    public static let registerBlockings = [
        (1, 1), (2, 1), (1, 2), (2, 2), (2, 4), (4, 2), (4, 4),
    ]
    /// Elements of slack after each tile row. `-1` asks the emitter to pick.
    public static let tilePaddings = [0, 4, 8]

    public enum Sweep: String, Sendable {
        /// Block shapes and threadgroup widths, everything else left to the
        /// emitter. 84 configurations.
        case quick
        /// The same, plus every one-axis deviation from the emitter's choice:
        /// each explicit register blocking, each tile padding, double buffering,
        /// and scalar (non-vector) staging. One axis at a time rather than the full cross product —
        /// that is both tractable and how the axes were attributed in the first
        /// place (docs/ARCHITECTURE.md §Matmul throughput).
        case full
    }

    public static func configurations(_ sweep: Sweep) -> [GEMMConfig] {
        var configurations: [GEMMConfig] = []
        for (blockM, blockN) in blockShapes {
            for blockK in blockKs {
                for simdgroups in simdgroupCounts {
                    let base = GEMMConfig(
                        blockM: blockM, blockN: blockN, blockK: blockK, simdgroups: simdgroups)
                    configurations.append(base)
                    guard sweep == .full else { continue }
                    for (registerM, registerN) in registerBlockings {
                        var variant = base
                        variant.registerM = registerM
                        variant.registerN = registerN
                        configurations.append(variant)
                    }
                    for padding in tilePaddings {
                        var variant = base
                        variant.tilePadding = padding
                        configurations.append(variant)
                    }
                    var buffered = base
                    buffered.doubleBuffer = true
                    configurations.append(buffered)
                    var scalarStaging = base
                    scalarStaging.vectorStaging = false
                    configurations.append(scalarStaging)
                }
            }
        }
        return configurations
    }

    // MARK: - Harness

    /// Everything a run needs that is worth building exactly once.
    public struct Harness {
        public let device: MTLDevice
        public let queue: MTLCommandQueue

        public init() throws {
            guard let device = MetalRuntime.device else {
                throw BenchError("no Metal device on this machine")
            }
            guard let queue = device.makeCommandQueue() else {
                throw BenchError("could not create a command queue")
            }
            self.device = device
            self.queue = queue
        }
    }

    public struct BenchError: Error, CustomStringConvertible {
        public let description: String
        public init(_ description: String) { self.description = description }
    }

    /// A compiled kernel plus the launch geometry for one config at one size.
    private struct Plan {
        let pipeline: MTLComputePipelineState
        let grid: MTLSize
        let threads: MTLSize
    }

    private static func plan(_ config: GEMMConfig, size: Int) throws -> Plan {
        let emission = try MetalCompiler.emit(
            ttir: GEMMKernel.tutorial(
                blockM: config.blockM, blockN: config.blockN, blockK: config.blockK),
            options: config.options)
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "matmul_kernel")
        guard let kernel = emission.kernels.first else { throw BenchError("no kernel emitted") }
        guard kernel.threadsPerThreadgroup <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw BenchError(
                "\(kernel.threadsPerThreadgroup) threads exceeds the pipeline maximum "
                    + "\(pipeline.maxTotalThreadsPerThreadgroup)")
        }
        return Plan(
            pipeline: pipeline,
            grid: MTLSize(
                width: cdiv(size, config.blockM), height: cdiv(size, config.blockN), depth: 1),
            threads: MTLSize(width: kernel.threadsPerThreadgroup, height: 1, depth: 1))
    }

    /// Median seconds per dispatch.
    ///
    /// The number of dispatches packed into one command buffer is calibrated so
    /// that every sample is roughly `targetSeconds` of GPU work, whatever the
    /// matrix size: one 512-cube GEMM is well under a millisecond, and submission
    /// overhead would otherwise be most of what either side is measured doing.
    private static func time(
        harness: Harness, targetSeconds: Double = 0.025, samples: Int = 3,
        _ encode: (MTLCommandBuffer) -> Void
    ) throws -> Double {
        func once(_ repeats: Int) throws -> Double {
            guard let commandBuffer = harness.queue.makeCommandBuffer() else {
                throw BenchError("could not create a command buffer")
            }
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<repeats { encode(commandBuffer) }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw BenchError("\(error)") }
            return (CFAbsoluteTimeGetCurrent() - start) / Double(repeats)
        }

        let calibration = try once(1)  // also the warm-up
        let repeats = max(1, min(200, Int((targetSeconds / max(calibration, 1e-6)).rounded())))
        var timings: [Double] = []
        for _ in 0..<samples { timings.append(try once(repeats)) }
        return timings.sorted()[timings.count / 2]
    }

    private static func timeKernel(
        _ config: GEMMConfig, size: Int, harness: Harness, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer
    ) throws -> Double {
        let plan = try plan(config, size: size)
        let scalars: [Int32] = [
            Int32(size), Int32(size), Int32(size), Int32(size), 1, Int32(size), 1, Int32(size), 1,
        ]
        return try time(harness: harness) { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(plan.pipeline)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(b, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            for (offset, scalar) in scalars.enumerated() {
                var value = scalar
                encoder.setBytes(&value, length: 4, index: 3 + offset)
            }
            encoder.dispatchThreadgroups(plan.grid, threadsPerThreadgroup: plan.threads)
            encoder.endEncoding()
        }
    }

    private static func timeMPS(
        size: Int, harness: Harness, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer
    ) throws -> Double {
        let descriptor = MPSMatrixDescriptor(
            rows: size, columns: size, rowBytes: size * 4, dataType: .float32)
        let left = MPSMatrix(buffer: a, descriptor: descriptor)
        let right = MPSMatrix(buffer: b, descriptor: descriptor)
        let result = MPSMatrix(buffer: c, descriptor: descriptor)
        let multiply = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: false,
            resultRows: size, resultColumns: size, interiorColumns: size, alpha: 1, beta: 0)
        return try time(harness: harness) { commandBuffer in
            multiply.encode(
                commandBuffer: commandBuffer, leftMatrix: left, rightMatrix: right,
                resultMatrix: result)
        }
    }

    // MARK: - Correctness

    /// A CPU-reference check at sizes that divide neither the block shape nor the
    /// 8x8 fragment, run before the timings are believed. `swift test` covers this
    /// far more thoroughly; this exists so the executable is not the one place
    /// where a wrong kernel could look fast on a machine without XCTest.
    public static func verify(_ config: GEMMConfig, harness: Harness) throws -> String? {
        for (m, n, k) in [(129, 257, 65), (64, 64, 64), (37, 41, 43)] {
            let a = (0..<(m * k)).map { Float(($0 % 101)) * 0.01 - 0.5 }
            let b = (0..<(k * n)).map { Float(($0 % 97)) * 0.01 - 0.5 }
            var expected = [Float](repeating: 0, count: m * n)
            for row in 0..<m {
                for step in 0..<k {
                    let left = a[row * k + step]
                    guard left != 0 else { continue }
                    for column in 0..<n {
                        expected[row * n + column] += left * b[step * n + column]
                    }
                }
            }

            let emission = try MetalCompiler.emit(
                ttir: GEMMKernel.tutorial(
                    blockM: config.blockM, blockN: config.blockN, blockK: config.blockK),
                options: config.options)
            let pipeline = try MetalCompiler.compileMSL(
                emission.source, kernelName: "matmul_kernel")
            let bufferA = try upload(a)
            let bufferB = try upload(b)
            let bufferC = try MetalRuntime.makeBuffer(length: m * n * 4)
            memset(bufferC.contents(), 0, bufferC.length)

            try MetalRuntime.launch(
                pipeline: pipeline,
                threadgroups: MTLSize(
                    width: cdiv(m, config.blockM), height: cdiv(n, config.blockN), depth: 1),
                threadsPerThreadgroup: emission.kernels[0].threadsPerThreadgroup,
                arguments: [
                    .buffer(bufferA), .buffer(bufferB), .buffer(bufferC),
                    .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                    .int32(Int32(k)), .int32(1), .int32(Int32(n)), .int32(1),
                    .int32(Int32(n)), .int32(1),
                ])

            let actual = Array(
                UnsafeBufferPointer(
                    start: bufferC.contents().assumingMemoryBound(to: Float.self), count: m * n))
            for index in actual.indices {
                let error = abs(actual[index] - expected[index]) / max(1, abs(expected[index]))
                if error > 1e-4 || actual[index].isNaN {
                    return "\(config.name) is wrong at \(m)x\(n)x\(k) index \(index): "
                        + "got \(actual[index]), expected \(expected[index])"
                }
            }
        }
        return nil
    }

    // MARK: - Driver

    /// Sweeps `configurations` at every size, keeping the best per size.
    public static func run(
        sizes: [Int], configurations: [GEMMConfig], harness: Harness, verbose: Bool = false,
        log: (String) -> Void = { print($0) }
    ) throws -> [GEMMMeasurement] {
        var results: [GEMMMeasurement] = []
        for size in sizes {
            let a = try upload((0..<(size * size)).map { Float(($0 % 101)) * 0.01 - 0.5 })
            let b = try upload((0..<(size * size)).map { Float(($0 % 97)) * 0.01 - 0.5 })
            let c = try MetalRuntime.makeBuffer(length: size * size * 4)

            var best: (seconds: Double, config: GEMMConfig)?
            var failures = 0
            for configuration in configurations {
                do {
                    let seconds = try timeKernel(
                        configuration, size: size, harness: harness, a: a, b: b, c: c)
                    if verbose {
                        log(
                            "  " + pad(configuration.name, 24)
                                + String(format: "%9.1f GFLOP/s", flops(size) / seconds / 1e9))
                    }
                    if best == nil || seconds < best!.seconds {
                        best = (seconds, configuration)
                    }
                } catch {
                    failures += 1
                    if verbose { log("  \(configuration.name) skipped: \(error)") }
                }
            }
            guard let best else {
                throw BenchError("every configuration failed to lower or run at \(size)")
            }
            let mps = try timeMPS(size: size, harness: harness, a: a, b: b, c: c)
            results.append(
                GEMMMeasurement(
                    size: size, config: best.config, gflops: flops(size) / best.seconds / 1e9,
                    mpsGflops: flops(size) / mps / 1e9))
            if verbose { log("  (\(failures) configurations skipped)") }
        }
        return results
    }

    /// `String(format:)` does not pad `%@`, and every row of this report is a
    /// config name in a column, so padding is done by hand.
    public static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
    }

    /// One row of the report, shared by `tmbench` and the XCTest wrapper.
    public static func row(_ measurement: GEMMMeasurement) -> String {
        pad("\(measurement.size)", 8) + pad(measurement.config.name, 25)
            + pad(String(format: "%.1f", measurement.gflops), 18)
            + pad(String(format: "%.1f", measurement.mpsGflops), 15)
            + String(format: "%.0f%%", measurement.ratio * 100)
    }

    public static let header =
        pad("size", 8) + pad("best config", 25) + pad("triton GFLOP/s", 18)
        + pad("MPS GFLOP/s", 15) + "ratio"

    public static func flops(_ size: Int) -> Double {
        2.0 * Double(size) * Double(size) * Double(size)
    }

    static func cdiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }

    static func upload(_ values: [Float]) throws -> MTLBuffer {
        let buffer = try MetalRuntime.makeBuffer(length: max(1, values.count * 4))
        values.withUnsafeBytes {
            if let base = $0.baseAddress {
                buffer.contents().copyMemory(from: base, byteCount: $0.count)
            }
        }
        return buffer
    }
}
