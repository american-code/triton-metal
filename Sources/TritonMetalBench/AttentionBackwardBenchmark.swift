import Foundation
import Metal
import MetalPerformanceShaders
import TritonMetalCore

/// FlashAttention-2 **backward** throughput, against the unfused composite that
/// does the same job out of Apple's own kernels.
///
/// The composite is the strongest one MPS can express, per head:
///
/// ```
/// S  = Q K^T * scale        MPSMatrixMultiplication (transposeRight)
/// P  = softmax(S)           MPSMatrixSoftMax
/// dV = P^T dO               MPSMatrixMultiplication (transposeLeft)
/// dP = dO V^T               MPSMatrixMultiplication (transposeRight)
/// dS = softmax'(dP, P)      MPSMatrixSoftMaxGradient
/// dQ = dS K * scale         MPSMatrixMultiplication
/// dK = dS^T Q * scale       MPSMatrixMultiplication (transposeLeft)
/// ```
///
/// Five GEMMs, a softmax and a softmax gradient, through **two** live `S x S`
/// f32 matrices — six passes over the score matrix where the fused version has
/// none. That traffic is what the fusion removes, and removing it is the only
/// reason a fused backward exists.
///
/// **Both sides are credited with the same arithmetic**: `5 * 2 * S^2 * D` per
/// head, the five GEMMs the mathematics needs, exactly as the forward benchmark
/// counts two and ignores the softmax. The fused side actually performs *seven*
/// GEMMs' worth, because `Q K^T` is recomputed in the `dQ` kernel and again in
/// the `dK`/`dV` one, plus a `Delta` pass over `O` and `dO`. Counting the
/// mathematics rather than the instructions is what makes the ratio a wall-clock
/// comparison at equal useful work, and it is the comparison that flatters the
/// fused side least.
public struct AttentionBackwardMeasurement: Sendable {
    public var shape: AttentionShape
    public var config: AttentionConfig
    public var element: String
    public var gflops: Double
    public var composeGflops: Double
    public var ratio: Double { composeGflops > 0 ? gflops / composeGflops : 0 }
}

public enum AttentionBackwardBenchmark {

    /// `5 * 2 * S^2 * D` per head — the five GEMMs the backward pass needs.
    public static func flops(_ shape: AttentionShape) -> Double {
        10.0 * Double(shape.seq) * Double(shape.seq) * Double(shape.dim)
            * Double(shape.batchedHeads)
    }

    /// The same search space as the forward, minus the combinations the `dK`/`dV`
    /// kernel's two `BLOCK_N x HEAD_DIM` accumulators cannot fit in 32KB — those
    /// are skipped by the driver when they fail to lower, not filtered here, so
    /// that a shape that starts fitting is picked up automatically.
    public static func configurations() -> [AttentionConfig] {
        AttentionBenchmark.configurations()
    }

    // MARK: - The fused kernels

    private struct FusedPipelines {
        var preprocess: MTLComputePipelineState
        var preprocessThreads: Int
        var dq: MTLComputePipelineState
        var dqThreads: Int
        var dkdv: MTLComputePipelineState
        var dkdvThreads: Int
    }

    private static func build(
        _ config: AttentionConfig, dim: Int, element: String
    ) throws -> FusedPipelines {
        func compile(_ ir: String, _ name: String) throws -> (MTLComputePipelineState, Int) {
            let emission = try MetalCompiler.emit(ttir: ir, options: config.options)
            let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: name)
            guard let kernel = emission.kernels.first else {
                throw GEMMBenchmark.BenchError("no kernel emitted for \(name)")
            }
            guard kernel.threadsPerThreadgroup <= pipeline.maxTotalThreadsPerThreadgroup else {
                throw GEMMBenchmark.BenchError(
                    "\(name): \(kernel.threadsPerThreadgroup) threads exceeds the pipeline maximum")
            }
            return (pipeline, kernel.threadsPerThreadgroup)
        }
        let pre = try compile(
            AttentionBackwardKernel.preprocess(
                blockM: config.blockM, headDim: dim, element: element),
            "attn_bwd_preprocess")
        let dq = try compile(
            AttentionBackwardKernel.dq(
                blockM: config.blockM, blockN: config.blockN, headDim: dim, element: element),
            "attn_bwd_dq")
        let dkdv = try compile(
            AttentionBackwardKernel.dkdv(
                blockM: config.blockM, blockN: config.blockN, headDim: dim, element: element),
            "attn_bwd_dkdv")
        return FusedPipelines(
            preprocess: pre.0, preprocessThreads: pre.1, dq: dq.0, dqThreads: dq.1,
            dkdv: dkdv.0, dkdvThreads: dkdv.1)
    }

    private struct Buffers {
        var q: MTLBuffer
        var k: MTLBuffer
        var v: MTLBuffer
        var o: MTLBuffer
        var dOut: MTLBuffer
        var lse: MTLBuffer
        var delta: MTLBuffer
        var dq: MTLBuffer
        var dk: MTLBuffer
        var dv: MTLBuffer
    }

    private static func timeFused(
        _ config: AttentionConfig, shape: AttentionShape, element: String,
        harness: GEMMBenchmark.Harness, buffers: Buffers
    ) throws -> Double {
        let pipelines = try build(config, dim: shape.dim, element: element)
        var scale = 1 / Float(shape.dim).squareRoot()
        var strideHead = Int32(shape.seq * shape.dim)
        var strideSeq = Int32(shape.dim)
        var strideLSE = Int32(shape.seq)
        var context = Int32(shape.seq)
        let heads = shape.batchedHeads
        let queryGrid = MTLSize(
            width: GEMMBenchmark.cdiv(shape.seq, config.blockM), height: heads, depth: 1)
        let keyGrid = MTLSize(
            width: GEMMBenchmark.cdiv(shape.seq, config.blockN), height: heads, depth: 1)

        return try AttentionBenchmark.timeCommandBuffer(harness: harness) { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(pipelines.preprocess)
            encoder.setBuffer(buffers.o, offset: 0, index: 0)
            encoder.setBuffer(buffers.dOut, offset: 0, index: 1)
            encoder.setBuffer(buffers.delta, offset: 0, index: 2)
            encoder.setBytes(&strideHead, length: 4, index: 3)
            encoder.setBytes(&strideSeq, length: 4, index: 4)
            encoder.setBytes(&strideLSE, length: 4, index: 5)
            encoder.setBytes(&context, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                queryGrid,
                threadsPerThreadgroup: MTLSize(
                    width: pipelines.preprocessThreads, height: 1, depth: 1))

            encoder.setComputePipelineState(pipelines.dq)
            encoder.setBuffer(buffers.q, offset: 0, index: 0)
            encoder.setBuffer(buffers.k, offset: 0, index: 1)
            encoder.setBuffer(buffers.v, offset: 0, index: 2)
            encoder.setBuffer(buffers.dOut, offset: 0, index: 3)
            encoder.setBuffer(buffers.dq, offset: 0, index: 4)
            encoder.setBuffer(buffers.lse, offset: 0, index: 5)
            encoder.setBuffer(buffers.delta, offset: 0, index: 6)
            encoder.setBytes(&scale, length: 4, index: 7)
            encoder.setBytes(&strideHead, length: 4, index: 8)
            encoder.setBytes(&strideSeq, length: 4, index: 9)
            encoder.setBytes(&strideLSE, length: 4, index: 10)
            encoder.setBytes(&context, length: 4, index: 11)
            encoder.dispatchThreadgroups(
                queryGrid,
                threadsPerThreadgroup: MTLSize(width: pipelines.dqThreads, height: 1, depth: 1))

            encoder.setComputePipelineState(pipelines.dkdv)
            encoder.setBuffer(buffers.q, offset: 0, index: 0)
            encoder.setBuffer(buffers.k, offset: 0, index: 1)
            encoder.setBuffer(buffers.v, offset: 0, index: 2)
            encoder.setBuffer(buffers.dOut, offset: 0, index: 3)
            encoder.setBuffer(buffers.dk, offset: 0, index: 4)
            encoder.setBuffer(buffers.dv, offset: 0, index: 5)
            encoder.setBuffer(buffers.lse, offset: 0, index: 6)
            encoder.setBuffer(buffers.delta, offset: 0, index: 7)
            encoder.setBytes(&scale, length: 4, index: 8)
            encoder.setBytes(&strideHead, length: 4, index: 9)
            encoder.setBytes(&strideSeq, length: 4, index: 10)
            encoder.setBytes(&strideLSE, length: 4, index: 11)
            encoder.setBytes(&context, length: 4, index: 12)
            encoder.dispatchThreadgroups(
                keyGrid,
                threadsPerThreadgroup: MTLSize(width: pipelines.dkdvThreads, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    // MARK: - The unfused composite

    private static func timeComposite(
        shape: AttentionShape, harness: GEMMBenchmark.Harness, buffers: Buffers,
        scores: MTLBuffer, gradient: MTLBuffer
    ) throws -> Double {
        let (seq, dim) = (shape.seq, shape.dim)
        let scale = Double(1 / Float(dim).squareRoot())
        let rows = MPSMatrixDescriptor(
            rows: seq, columns: dim, rowBytes: dim * 4, dataType: .float32)
        let square = MPSMatrixDescriptor(
            rows: seq, columns: seq, rowBytes: seq * 4, dataType: .float32)
        let scoreMatrix = MPSMatrix(buffer: scores, descriptor: square)
        let gradientMatrix = MPSMatrix(buffer: gradient, descriptor: square)

        let qk = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: true,
            resultRows: seq, resultColumns: seq, interiorColumns: dim, alpha: scale, beta: 0)
        let softmax = MPSMatrixSoftMax(device: harness.device)
        let softmaxGradient = MPSMatrixSoftMaxGradient(device: harness.device)
        // dV = P^T dO
        let pv = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: true, transposeRight: false,
            resultRows: seq, resultColumns: dim, interiorColumns: seq, alpha: 1, beta: 0)
        // dP = dO V^T
        let dov = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: true,
            resultRows: seq, resultColumns: seq, interiorColumns: dim, alpha: 1, beta: 0)
        // dQ = dS K, dK = dS^T Q
        let dsk = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: false, transposeRight: false,
            resultRows: seq, resultColumns: dim, interiorColumns: seq, alpha: scale, beta: 0)
        let dsq = MPSMatrixMultiplication(
            device: harness.device, transposeLeft: true, transposeRight: false,
            resultRows: seq, resultColumns: dim, interiorColumns: seq, alpha: scale, beta: 0)

        let headBytes = seq * dim * 4
        return try AttentionBenchmark.timeCommandBuffer(harness: harness) { commandBuffer in
            for head in 0..<shape.batchedHeads {
                let offset = head * headBytes
                let qm = MPSMatrix(buffer: buffers.q, offset: offset, descriptor: rows)
                let km = MPSMatrix(buffer: buffers.k, offset: offset, descriptor: rows)
                let vm = MPSMatrix(buffer: buffers.v, offset: offset, descriptor: rows)
                let dom = MPSMatrix(buffer: buffers.dOut, offset: offset, descriptor: rows)
                let dqm = MPSMatrix(buffer: buffers.dq, offset: offset, descriptor: rows)
                let dkm = MPSMatrix(buffer: buffers.dk, offset: offset, descriptor: rows)
                let dvm = MPSMatrix(buffer: buffers.dv, offset: offset, descriptor: rows)
                qk.encode(
                    commandBuffer: commandBuffer, leftMatrix: qm, rightMatrix: km,
                    resultMatrix: scoreMatrix)
                softmax.encode(
                    commandBuffer: commandBuffer, inputMatrix: scoreMatrix,
                    resultMatrix: scoreMatrix)
                pv.encode(
                    commandBuffer: commandBuffer, leftMatrix: scoreMatrix, rightMatrix: dom,
                    resultMatrix: dvm)
                dov.encode(
                    commandBuffer: commandBuffer, leftMatrix: dom, rightMatrix: vm,
                    resultMatrix: gradientMatrix)
                softmaxGradient.encode(
                    to: commandBuffer, gradientMatrix: gradientMatrix,
                    forwardOutputMatrix: scoreMatrix, resultMatrix: gradientMatrix)
                dsk.encode(
                    commandBuffer: commandBuffer, leftMatrix: gradientMatrix, rightMatrix: km,
                    resultMatrix: dqm)
                dsq.encode(
                    commandBuffer: commandBuffer, leftMatrix: gradientMatrix, rightMatrix: qm,
                    resultMatrix: dkm)
            }
        }
    }

    // MARK: - Correctness

    /// The CPU check the executable runs before believing a timing, at a sequence
    /// length that divides neither block dimension. `swift test`'s
    /// `AttentionBackwardTests` is the thorough version (analytic reference *and*
    /// finite differences); this exists so a fast-but-wrong kernel cannot be
    /// quoted from a machine without XCTest.
    public static func verify(_ config: AttentionConfig, harness: GEMMBenchmark.Harness) throws
        -> String?
    {
        let shape = AttentionShape(batch: 1, heads: 2, seq: 67, dim: 64)
        let heads = shape.batchedHeads
        let (seq, dim) = (shape.seq, shape.dim)
        let count = heads * seq * dim
        let q = (0..<count).map { Float($0 % 23) / 23 - 0.5 }
        let k = (0..<count).map { Float($0 % 29) / 29 - 0.5 }
        let v = (0..<count).map { Float($0 % 31) / 31 - 0.5 }
        let dOut = (0..<count).map { Float($0 % 37) / 37 - 0.5 }
        let scale = Double(1 / Float(dim).squareRoot())
        let log2e = 1.4426950408889634

        // Reference: probabilities, output, logsumexp, and the three gradients.
        var output = [Float](repeating: 0, count: count)
        var lse = [Float](repeating: 0, count: heads * seq)
        var expectedQ = [Double](repeating: 0, count: count)
        var expectedK = [Double](repeating: 0, count: count)
        var expectedV = [Double](repeating: 0, count: count)
        for head in 0..<heads {
            let base = head * seq * dim
            var probability = [[Double]](repeating: [Double](repeating: 0, count: seq), count: seq)
            var delta = [Double](repeating: 0, count: seq)
            for row in 0..<seq {
                var scores = [Double](repeating: 0, count: seq)
                for column in 0..<seq {
                    var total = 0.0
                    for element in 0..<dim {
                        total +=
                            Double(q[base + row * dim + element])
                            * Double(k[base + column * dim + element])
                    }
                    scores[column] = total * scale
                }
                let peak = scores.max() ?? 0
                var sum = 0.0
                for column in 0..<seq {
                    probability[row][column] = exp(scores[column] - peak)
                    sum += probability[row][column]
                }
                for column in 0..<seq { probability[row][column] /= sum }
                lse[head * seq + row] = Float(peak * log2e + log2(sum))
                for element in 0..<dim {
                    var total = 0.0
                    for column in 0..<seq {
                        total += probability[row][column] * Double(v[base + column * dim + element])
                    }
                    output[base + row * dim + element] = Float(total)
                    delta[row] += total * Double(dOut[base + row * dim + element])
                }
            }
            var ds = [[Double]](repeating: [Double](repeating: 0, count: seq), count: seq)
            for row in 0..<seq {
                for column in 0..<seq {
                    var dp = 0.0
                    for element in 0..<dim {
                        dp +=
                            Double(dOut[base + row * dim + element])
                            * Double(v[base + column * dim + element])
                    }
                    ds[row][column] = probability[row][column] * (dp - delta[row])
                }
            }
            for row in 0..<seq {
                for element in 0..<dim {
                    var total = 0.0
                    for column in 0..<seq {
                        total += ds[row][column] * Double(k[base + column * dim + element]) * scale
                    }
                    expectedQ[base + row * dim + element] = total
                }
            }
            for column in 0..<seq {
                for element in 0..<dim {
                    var kTotal = 0.0
                    var vTotal = 0.0
                    for row in 0..<seq {
                        kTotal += ds[row][column] * Double(q[base + row * dim + element]) * scale
                        vTotal += probability[row][column] * Double(dOut[base + row * dim + element])
                    }
                    expectedK[base + column * dim + element] = kTotal
                    expectedV[base + column * dim + element] = vTotal
                }
            }
        }

        let buffers = Buffers(
            q: try GEMMBenchmark.upload(q), k: try GEMMBenchmark.upload(k),
            v: try GEMMBenchmark.upload(v), o: try GEMMBenchmark.upload(output),
            dOut: try GEMMBenchmark.upload(dOut), lse: try GEMMBenchmark.upload(lse),
            delta: try MetalRuntime.makeBuffer(length: heads * seq * 4),
            dq: try MetalRuntime.makeBuffer(length: count * 4),
            dk: try MetalRuntime.makeBuffer(length: count * 4),
            dv: try MetalRuntime.makeBuffer(length: count * 4))
        for buffer in [buffers.delta, buffers.dq, buffers.dk, buffers.dv] {
            memset(buffer.contents(), 0, buffer.length)
        }

        let pipelines = try build(config, dim: dim, element: "f32")
        try MetalRuntime.launch(
            pipeline: pipelines.preprocess,
            threadgroups: MTLSize(
                width: GEMMBenchmark.cdiv(seq, config.blockM), height: heads, depth: 1),
            threadsPerThreadgroup: pipelines.preprocessThreads,
            arguments: [
                .buffer(buffers.o), .buffer(buffers.dOut), .buffer(buffers.delta),
                .int32(Int32(seq * dim)), .int32(Int32(dim)), .int32(Int32(seq)),
                .int32(Int32(seq)),
            ])
        let scalar = Float(scale)
        try MetalRuntime.launch(
            pipeline: pipelines.dq,
            threadgroups: MTLSize(
                width: GEMMBenchmark.cdiv(seq, config.blockM), height: heads, depth: 1),
            threadsPerThreadgroup: pipelines.dqThreads,
            arguments: [
                .buffer(buffers.q), .buffer(buffers.k), .buffer(buffers.v), .buffer(buffers.dOut),
                .buffer(buffers.dq), .buffer(buffers.lse), .buffer(buffers.delta),
                .float32(scalar), .int32(Int32(seq * dim)), .int32(Int32(dim)),
                .int32(Int32(seq)), .int32(Int32(seq)),
            ])
        try MetalRuntime.launch(
            pipeline: pipelines.dkdv,
            threadgroups: MTLSize(
                width: GEMMBenchmark.cdiv(seq, config.blockN), height: heads, depth: 1),
            threadsPerThreadgroup: pipelines.dkdvThreads,
            arguments: [
                .buffer(buffers.q), .buffer(buffers.k), .buffer(buffers.v), .buffer(buffers.dOut),
                .buffer(buffers.dk), .buffer(buffers.dv), .buffer(buffers.lse),
                .buffer(buffers.delta), .float32(scalar), .int32(Int32(seq * dim)),
                .int32(Int32(dim)), .int32(Int32(seq)), .int32(Int32(seq)),
            ])

        for (name, buffer, expected) in [
            ("dQ", buffers.dq, expectedQ), ("dK", buffers.dk, expectedK),
            ("dV", buffers.dv, expectedV),
        ] {
            let actual = Array(
                UnsafeBufferPointer(
                    start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
            let peak = max(expected.map { abs($0) }.max() ?? 0, 1e-30)
            for index in 0..<count {
                let error = abs(Double(actual[index]) - expected[index])
                if actual[index].isNaN || error > 1e-4 * peak {
                    return "\(config.name) \(name) is wrong at \(index): got \(actual[index]), "
                        + "expected \(expected[index])"
                }
            }
        }
        return nil
    }

    // MARK: - Driver

    public static func run(
        shapes: [AttentionShape], configurations: [AttentionConfig], element: String,
        harness: GEMMBenchmark.Harness, verbose: Bool = false, log: (String) -> Void = { print($0) }
    ) throws -> [AttentionBackwardMeasurement] {
        var results: [AttentionBackwardMeasurement] = []
        for shape in shapes {
            let heads = shape.batchedHeads
            let count = heads * shape.seq * shape.dim
            let width = element == "f16" ? 2 : 4
            let buffers = Buffers(
                q: try GEMMBenchmark.upload(values(count, 23)),
                k: try GEMMBenchmark.upload(values(count, 29)),
                v: try GEMMBenchmark.upload(values(count, 31)),
                o: try GEMMBenchmark.upload(values(count, 17)),
                dOut: try GEMMBenchmark.upload(values(count, 37)),
                lse: try GEMMBenchmark.upload([Float](repeating: 0, count: heads * shape.seq)),
                delta: try MetalRuntime.makeBuffer(length: heads * shape.seq * 4),
                dq: try MetalRuntime.makeBuffer(length: count * 4),
                dk: try MetalRuntime.makeBuffer(length: count * 4),
                dv: try MetalRuntime.makeBuffer(length: count * 4))
            // The fused kernels read and write in `element`; the composite is
            // always f32, because MPSMatrixSoftMax has no half path and an f32
            // composite is the strongest version of the thing to beat.
            let fused = width == 2
                ? Buffers(
                    q: try half(count, 23), k: try half(count, 29), v: try half(count, 31),
                    o: try half(count, 17), dOut: try half(count, 37), lse: buffers.lse,
                    delta: buffers.delta,
                    dq: try MetalRuntime.makeBuffer(length: count * 2),
                    dk: try MetalRuntime.makeBuffer(length: count * 2),
                    dv: try MetalRuntime.makeBuffer(length: count * 2))
                : buffers

            var best: (seconds: Double, config: AttentionConfig)?
            var failures = 0
            for configuration in configurations {
                do {
                    let seconds = try timeFused(
                        configuration, shape: shape, element: element, harness: harness,
                        buffers: fused)
                    if verbose {
                        log(
                            "  " + GEMMBenchmark.pad(configuration.name, 18)
                                + String(format: "%9.1f GFLOP/s", flops(shape) / seconds / 1e9))
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

            let composite = try timeComposite(
                shape: shape, harness: harness, buffers: buffers,
                scores: try MetalRuntime.makeBuffer(length: shape.seq * shape.seq * 4),
                gradient: try MetalRuntime.makeBuffer(length: shape.seq * shape.seq * 4))

            results.append(
                AttentionBackwardMeasurement(
                    shape: shape, config: best.config, element: element,
                    gflops: flops(shape) / best.seconds / 1e9,
                    composeGflops: flops(shape) / composite / 1e9))
        }
        return results
    }

    private static func values(_ count: Int, _ seed: Int) -> [Float] {
        (0..<count).map { Float($0 % seed) / Float(seed) - 0.5 }
    }

    private static func half(_ count: Int, _ seed: Int) throws -> MTLBuffer {
        let halves = values(count, seed).map { Float16($0) }
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

    public static func row(_ measurement: AttentionBackwardMeasurement) -> String {
        GEMMBenchmark.pad(measurement.shape.name, 20)
            + GEMMBenchmark.pad(measurement.config.name, 16)
            + GEMMBenchmark.pad(measurement.element, 6)
            + GEMMBenchmark.pad(String(format: "%.1f", measurement.gflops), 16)
            + GEMMBenchmark.pad(String(format: "%.1f", measurement.composeGflops), 16)
            + String(format: "%.0f%%", measurement.ratio * 100)
    }
}
