import Foundation
import Metal
import MetalPerformanceShaders
import TritonMetalCore

/// FlashAttention-2 forward throughput, against the **unfused composite** the
/// fusion exists to beat: `S = Q K^T` and `O = P V` as two
/// `MPSMatrixMultiplication`s with an `MPSMatrixSoftMax` between them.
///
/// That is the honest comparison for a fused kernel. The composite runs the same
/// arithmetic through Apple's own GEMM, which is faster than this backend's, but
/// it has to write the whole `S x S` score matrix to device memory and read it
/// back twice — which is the traffic FlashAttention removes. Whether removing it
/// wins depends on how far ahead the GEMM is, so the number is worth measuring
/// rather than assuming (docs/ARCHITECTURE.md §Attention throughput).
///
/// Lives here rather than in the XCTest bundle for the same reason `GEMMBenchmark`
/// does: the machines whose numbers are worth quoting have only the command-line
/// tools.

/// One point of the attention search space.
public struct AttentionConfig: Sendable, Hashable {
    public var blockM: Int
    public var blockN: Int
    public var simdgroups: Int

    public init(blockM: Int, blockN: Int, simdgroups: Int) {
        self.blockM = blockM
        self.blockN = blockN
        self.simdgroups = simdgroups
    }

    public var options: MetalCompiler.Options {
        MetalCompiler.Options(numSimdgroups: simdgroups)
    }

    public var name: String { "\(blockM)x\(blockN)/w\(simdgroups)" }

    /// `M,N,W`, the spelling `tmbench --attn-config` takes.
    public static func parse(_ text: String) -> AttentionConfig? {
        let fields = text.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard fields.count == 3, !fields.contains(where: { $0 == nil }) else { return nil }
        return AttentionConfig(blockM: fields[0]!, blockN: fields[1]!, simdgroups: fields[2]!)
    }
}

/// A shape in the usual attention spelling.
public struct AttentionShape: Sendable, Hashable {
    public var batch: Int
    public var heads: Int
    public var seq: Int
    public var dim: Int

    public init(batch: Int, heads: Int, seq: Int, dim: Int) {
        self.batch = batch
        self.heads = heads
        self.seq = seq
        self.dim = dim
    }

    public var name: String { "b\(batch) h\(heads) s\(seq) d\(dim)" }
    public var batchedHeads: Int { batch * heads }

    /// The two matmuls, counted the way attention papers do: `2 * 2 * S^2 * D`
    /// per head. The softmax is not counted, on either side.
    public var flops: Double {
        4.0 * Double(seq) * Double(seq) * Double(dim) * Double(batchedHeads)
    }

    /// Bytes of device traffic the *unfused* composite adds over the fused one:
    /// the score matrix written once and read twice, in f32.
    public var scoreTraffic: Double {
        3.0 * Double(seq) * Double(seq) * 4 * Double(batchedHeads)
    }
}

public struct AttentionMeasurement: Sendable {
    public var shape: AttentionShape
    public var config: AttentionConfig
    public var element: String
    public var gflops: Double
    public var composeGflops: Double
    public var ratio: Double { composeGflops > 0 ? gflops / composeGflops : 0 }
}

public enum AttentionBenchmark {

    public static let blockShapes = [
        (16, 32), (16, 64), (16, 128), (32, 32), (32, 64), (32, 128), (64, 32), (64, 64),
    ]
    public static let simdgroupCounts = [1, 2, 4, 8, 16]

    public static func configurations() -> [AttentionConfig] {
        var configurations: [AttentionConfig] = []
        for (blockM, blockN) in blockShapes {
            for simdgroups in simdgroupCounts {
                configurations.append(
                    AttentionConfig(blockM: blockM, blockN: blockN, simdgroups: simdgroups))
            }
        }
        return configurations
    }

    // MARK: - The fused kernel

    private static func timeFused(
        _ config: AttentionConfig, shape: AttentionShape, element: String,
        harness: GEMMBenchmark.Harness, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer
    ) throws -> Double {
        let emission = try MetalCompiler.emit(
            ttir: AttentionKernel.forward(
                blockM: config.blockM, blockN: config.blockN, headDim: shape.dim,
                element: element),
            options: config.options)
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "attn_fwd")
        guard let kernel = emission.kernels.first else {
            throw GEMMBenchmark.BenchError("no kernel emitted")
        }
        guard kernel.threadsPerThreadgroup <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw GEMMBenchmark.BenchError(
                "\(kernel.threadsPerThreadgroup) threads exceeds the pipeline maximum")
        }
        let grid = MTLSize(
            width: GEMMBenchmark.cdiv(shape.seq, config.blockM), height: shape.batchedHeads,
            depth: 1)
        let threads = MTLSize(width: kernel.threadsPerThreadgroup, height: 1, depth: 1)
        var scale = 1 / Float(shape.dim).squareRoot()
        var strideHead = Int32(shape.seq * shape.dim)
        var strideSeq = Int32(shape.dim)
        var context = Int32(shape.seq)

        return try time(harness: harness) { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(q, offset: 0, index: 0)
            encoder.setBuffer(k, offset: 0, index: 1)
            encoder.setBuffer(v, offset: 0, index: 2)
            encoder.setBuffer(out, offset: 0, index: 3)
            encoder.setBytes(&scale, length: 4, index: 4)
            encoder.setBytes(&strideHead, length: 4, index: 5)
            encoder.setBytes(&strideSeq, length: 4, index: 6)
            encoder.setBytes(&context, length: 4, index: 7)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
            encoder.endEncoding()
        }
    }

    // MARK: - The unfused composite

    /// `S = Q K^T`, `P = softmax(S)`, `O = P V`, one head at a time, all through
    /// MPS. The scores are a real `S x S` f32 buffer, which is the whole point.
    private static func timeComposite(
        shape: AttentionShape, harness: GEMMBenchmark.Harness, q: MTLBuffer, k: MTLBuffer,
        v: MTLBuffer, out: MTLBuffer, scores: MTLBuffer
    ) throws -> Double {
        let (seq, dim) = (shape.seq, shape.dim)
        let rowsDescriptor = MPSMatrixDescriptor(
            rows: seq, columns: dim, rowBytes: dim * 4, dataType: .float32)
        let scoreDescriptor = MPSMatrixDescriptor(
            rows: seq, columns: seq, rowBytes: seq * 4, dataType: .float32)
        let scoreMatrix = MPSMatrix(buffer: scores, descriptor: scoreDescriptor)

        let qk = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: true,
            resultRows: seq, resultColumns: seq, interiorColumns: dim,
            alpha: Double(1 / Float(dim).squareRoot()), beta: 0)
        let softmax = MPSMatrixSoftMax(device: harness.device)
        let pv = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: false,
            resultRows: seq, resultColumns: dim, interiorColumns: seq, alpha: 1, beta: 0)

        let headBytes = seq * dim * 4
        return try time(harness: harness) { commandBuffer in
            for head in 0..<shape.batchedHeads {
                let offset = head * headBytes
                let qm = MPSMatrix(buffer: q, offset: offset, descriptor: rowsDescriptor)
                let km = MPSMatrix(buffer: k, offset: offset, descriptor: rowsDescriptor)
                let vm = MPSMatrix(buffer: v, offset: offset, descriptor: rowsDescriptor)
                let om = MPSMatrix(buffer: out, offset: offset, descriptor: rowsDescriptor)
                qk.encode(
                    commandBuffer: commandBuffer, leftMatrix: qm, rightMatrix: km,
                    resultMatrix: scoreMatrix)
                softmax.encode(
                    commandBuffer: commandBuffer, inputMatrix: scoreMatrix,
                    resultMatrix: scoreMatrix)
                pv.encode(
                    commandBuffer: commandBuffer, leftMatrix: scoreMatrix, rightMatrix: vm,
                    resultMatrix: om)
            }
        }
    }

    /// Median seconds per dispatch, calibrated so each sample is ~25ms of GPU
    /// work — the same methodology as the GEMM sweep, so the two are comparable.
    private static func time(
        harness: GEMMBenchmark.Harness, targetSeconds: Double = 0.025, samples: Int = 3,
        _ encode: (MTLCommandBuffer) -> Void
    ) throws -> Double {
        func once(_ repeats: Int) throws -> Double {
            guard let commandBuffer = harness.queue.makeCommandBuffer() else {
                throw GEMMBenchmark.BenchError("could not create a command buffer")
            }
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<repeats { encode(commandBuffer) }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw GEMMBenchmark.BenchError("\(error)") }
            return (CFAbsoluteTimeGetCurrent() - start) / Double(repeats)
        }
        let calibration = try once(1)
        let repeats = max(1, min(200, Int((targetSeconds / max(calibration, 1e-6)).rounded())))
        var timings: [Double] = []
        for _ in 0..<samples { timings.append(try once(repeats)) }
        return timings.sorted()[timings.count / 2]
    }

    // MARK: - Correctness

    /// The CPU check the executable runs before believing a timing, at a sequence
    /// length that divides neither block dimension. `swift test`'s `AttentionTests`
    /// is the thorough version; this exists so that a fast-but-wrong kernel cannot
    /// be quoted from a machine without XCTest.
    public static func verify(_ config: AttentionConfig, harness: GEMMBenchmark.Harness) throws
        -> String?
    {
        let shape = AttentionShape(batch: 1, heads: 2, seq: 67, dim: 64)
        let count = shape.batchedHeads * shape.seq * shape.dim
        let q = (0..<count).map { Float(($0 % 23)) / 23 - 0.5 }
        let k = (0..<count).map { Float(($0 % 29)) / 29 - 0.5 }
        let v = (0..<count).map { Float(($0 % 31)) / 31 - 0.5 }
        let scale = 1 / Float(shape.dim).squareRoot()

        let emission = try MetalCompiler.emit(
            ttir: AttentionKernel.forward(
                blockM: config.blockM, blockN: config.blockN, headDim: shape.dim),
            options: config.options)
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "attn_fwd")
        let output = try MetalRuntime.makeBuffer(length: count * 4)
        memset(output.contents(), 0, output.length)
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(
                width: GEMMBenchmark.cdiv(shape.seq, config.blockM), height: shape.batchedHeads,
                depth: 1),
            threadsPerThreadgroup: emission.kernels[0].threadsPerThreadgroup,
            arguments: [
                .buffer(try GEMMBenchmark.upload(q)), .buffer(try GEMMBenchmark.upload(k)),
                .buffer(try GEMMBenchmark.upload(v)), .buffer(output), .float32(scale),
                .int32(Int32(shape.seq * shape.dim)), .int32(Int32(shape.dim)),
                .int32(Int32(shape.seq)),
            ])
        let actual = Array(
            UnsafeBufferPointer(
                start: output.contents().assumingMemoryBound(to: Float.self), count: count))

        for head in 0..<shape.batchedHeads {
            let base = head * shape.seq * shape.dim
            for row in 0..<shape.seq {
                var scores = [Double](repeating: 0, count: shape.seq)
                for column in 0..<shape.seq {
                    var total = 0.0
                    for element in 0..<shape.dim {
                        total +=
                            Double(q[base + row * shape.dim + element])
                            * Double(k[base + column * shape.dim + element])
                    }
                    scores[column] = total * Double(scale)
                }
                let peak = scores.max() ?? 0
                var sum = 0.0
                for column in 0..<shape.seq {
                    scores[column] = exp(scores[column] - peak)
                    sum += scores[column]
                }
                for element in 0..<shape.dim {
                    var total = 0.0
                    for column in 0..<shape.seq {
                        total += scores[column] * Double(v[base + column * shape.dim + element])
                    }
                    let expected = Float(total / sum)
                    let index = base + row * shape.dim + element
                    let error = abs(actual[index] - expected) / max(1, abs(expected))
                    if error > 1e-4 || actual[index].isNaN {
                        return "\(config.name) is wrong at head \(head) row \(row) element "
                            + "\(element): got \(actual[index]), expected \(expected)"
                    }
                }
            }
        }
        return nil
    }

    // MARK: - Driver

    public static func run(
        shapes: [AttentionShape], configurations: [AttentionConfig], element: String,
        harness: GEMMBenchmark.Harness, verbose: Bool = false, log: (String) -> Void = { print($0) }
    ) throws -> [AttentionMeasurement] {
        var results: [AttentionMeasurement] = []
        for shape in shapes {
            let count = shape.batchedHeads * shape.seq * shape.dim
            let width = element == "f16" ? 2 : 4
            let q = try buffer(count: count, width: width, seed: 23)
            let k = try buffer(count: count, width: width, seed: 29)
            let v = try buffer(count: count, width: width, seed: 31)
            let out = try MetalRuntime.makeBuffer(length: count * width)

            var best: (seconds: Double, config: AttentionConfig)?
            var failures = 0
            for configuration in configurations {
                do {
                    let seconds = try timeFused(
                        configuration, shape: shape, element: element, harness: harness,
                        q: q, k: k, v: v, out: out)
                    if verbose {
                        log(
                            "  " + GEMMBenchmark.pad(configuration.name, 18)
                                + String(format: "%9.1f GFLOP/s", shape.flops / seconds / 1e9))
                    }
                    if best == nil || seconds < best!.seconds { best = (seconds, configuration) }
                } catch {
                    failures += 1
                    if verbose { log("  \(configuration.name) skipped: \(error)") }
                }
            }
            guard let best else {
                throw GEMMBenchmark.BenchError(
                    "every configuration failed to lower or run at \(shape.name)")
            }
            if verbose && failures > 0 { log("  (\(failures) configurations skipped)") }

            // The composite is always f32: MPSMatrixSoftMax has no half path, and
            // an f32 composite is the strongest version of the thing to beat.
            let f32Count = count * 4
            let composite = try timeComposite(
                shape: shape, harness: harness,
                q: element == "f16" ? try buffer(count: count, width: 4, seed: 23) : q,
                k: element == "f16" ? try buffer(count: count, width: 4, seed: 29) : k,
                v: element == "f16" ? try buffer(count: count, width: 4, seed: 31) : v,
                out: element == "f16" ? try MetalRuntime.makeBuffer(length: f32Count) : out,
                scores: try MetalRuntime.makeBuffer(
                    length: shape.seq * shape.seq * 4))

            results.append(
                AttentionMeasurement(
                    shape: shape, config: best.config, element: element,
                    gflops: shape.flops / best.seconds / 1e9,
                    composeGflops: shape.flops / composite / 1e9))
        }
        return results
    }

    private static func buffer(count: Int, width: Int, seed: Int) throws -> MTLBuffer {
        let values = (0..<count).map { Float(($0 % seed)) / Float(seed) - 0.5 }
        guard width == 2 else { return try GEMMBenchmark.upload(values) }
        let halves = values.map { Float16($0) }
        let buffer = try MetalRuntime.makeBuffer(length: max(1, count * 2))
        halves.withUnsafeBytes {
            if let base = $0.baseAddress {
                buffer.contents().copyMemory(from: base, byteCount: $0.count)
            }
        }
        return buffer
    }

    public static let header =
        GEMMBenchmark.pad("shape", 20) + GEMMBenchmark.pad("best config", 16)
        + GEMMBenchmark.pad("in", 6) + GEMMBenchmark.pad("fused GFLOP/s", 16)
        + GEMMBenchmark.pad("MPS composite", 16) + "ratio"

    public static func row(_ measurement: AttentionMeasurement) -> String {
        GEMMBenchmark.pad(measurement.shape.name, 20)
            + GEMMBenchmark.pad(measurement.config.name, 16)
            + GEMMBenchmark.pad(measurement.element, 6)
            + GEMMBenchmark.pad(String(format: "%.1f", measurement.gflops), 16)
            + GEMMBenchmark.pad(String(format: "%.1f", measurement.composeGflops), 16)
            + String(format: "%.0f%%", measurement.ratio * 100)
    }
}
