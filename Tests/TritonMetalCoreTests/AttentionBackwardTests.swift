import Foundation
import Metal
import TritonMetalBench
import XCTest

@testable import TritonMetalCore

/// Milestone: the FlashAttention-2 **backward** pass — what turns an
/// inference-only backend into one that can train.
///
/// Every gradient is checked twice and independently: against an analytic CPU
/// reference computed in `Double`, and against **finite differences** of the
/// forward pass, which knows nothing about the backward formulas at all. The
/// second is what would catch a reference that is wrong in the same way the
/// kernel is.
///
/// The tolerances are derived rather than tuned. For a gradient element that is
/// a sum of `n` floating-point terms, the standard recursive-summation bound is
/// `|computed - exact| <= (n-1) * u * sum|term|`, so the tolerance for a whole
/// tensor is `n * u * A` where `A = max_i sum_j |term_ij| / max|output|` is the
/// cancellation amplification, computed here from the `Double` reference itself
/// rather than guessed. `u` is `2^-24` for the f32 path and `2^-11` for the
/// f16-in/f32-accumulate one, where the dot operands are rounded to half.
final class AttentionBackwardTests: XCTestCase {

    // MARK: - CPU reference

    /// Everything the backward pass needs, in `Double`, from the definitions.
    private struct Reference {
        /// `[heads, seq, dim]`
        var output: [Double]
        /// `[heads, seq]` — log2-scale logsumexp, exactly what the forward writes.
        var logsumexp: [Double]
        /// `[heads, seq]` — `dO_i . O_i`.
        var delta: [Double]
        var dq: [Double]
        var dk: [Double]
        var dv: [Double]
        /// `max_i sum_j |term|` over the summed dimension, per gradient, divided
        /// by the largest magnitude in that gradient: how much cancellation the
        /// sum goes through, which is what turns a per-term rounding error into
        /// an error in the answer.
        var amplification: (dq: Double, dk: Double, dv: Double)
    }

    private func reference(
        q: [Float], k: [Float], v: [Float], dOut: [Float],
        heads: Int, seq: Int, dim: Int, scale: Double
    ) -> Reference {
        let log2e = 1.4426950408889634
        var output = [Double](repeating: 0, count: heads * seq * dim)
        var logsumexp = [Double](repeating: 0, count: heads * seq)
        var delta = [Double](repeating: 0, count: heads * seq)
        var dq = [Double](repeating: 0, count: heads * seq * dim)
        var dk = [Double](repeating: 0, count: heads * seq * dim)
        var dv = [Double](repeating: 0, count: heads * seq * dim)
        var dqTerms = 0.0
        var dkTerms = 0.0
        var dvTerms = 0.0

        for head in 0..<heads {
            let base = head * seq * dim
            var probability = [[Double]](
                repeating: [Double](repeating: 0, count: seq), count: seq)

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
                // The forward writes m_i + log2(l_i) with everything already in
                // log2 units, which is log2 of the (unshifted) softmax denominator.
                logsumexp[head * seq + row] = (peak * log2e) + log2(sum)

                for element in 0..<dim {
                    var total = 0.0
                    for column in 0..<seq {
                        total += probability[row][column] * Double(v[base + column * dim + element])
                    }
                    output[base + row * dim + element] = total
                }
                var d = 0.0
                for element in 0..<dim {
                    d +=
                        Double(dOut[base + row * dim + element])
                        * output[base + row * dim + element]
                }
                delta[head * seq + row] = d
            }

            // dS, then the three gradients.
            var ds = [[Double]](repeating: [Double](repeating: 0, count: seq), count: seq)
            for row in 0..<seq {
                for column in 0..<seq {
                    var dp = 0.0
                    for element in 0..<dim {
                        dp +=
                            Double(dOut[base + row * dim + element])
                            * Double(v[base + column * dim + element])
                    }
                    ds[row][column] = probability[row][column] * (dp - delta[head * seq + row])
                }
            }
            for row in 0..<seq {
                for element in 0..<dim {
                    var total = 0.0
                    var absolute = 0.0
                    for column in 0..<seq {
                        let term = ds[row][column] * Double(k[base + column * dim + element]) * scale
                        total += term
                        absolute += abs(term)
                    }
                    dq[base + row * dim + element] = total
                    dqTerms = max(dqTerms, absolute)
                }
            }
            for column in 0..<seq {
                for element in 0..<dim {
                    var kTotal = 0.0
                    var kAbsolute = 0.0
                    var vTotal = 0.0
                    var vAbsolute = 0.0
                    for row in 0..<seq {
                        let kTerm = ds[row][column] * Double(q[base + row * dim + element]) * scale
                        kTotal += kTerm
                        kAbsolute += abs(kTerm)
                        let vTerm =
                            probability[row][column] * Double(dOut[base + row * dim + element])
                        vTotal += vTerm
                        vAbsolute += abs(vTerm)
                    }
                    dk[base + column * dim + element] = kTotal
                    dkTerms = max(dkTerms, kAbsolute)
                    dv[base + column * dim + element] = vTotal
                    dvTerms = max(dvTerms, vAbsolute)
                }
            }
        }

        func peak(_ values: [Double]) -> Double {
            max(values.map { abs($0) }.max() ?? 0, 1e-30)
        }
        return Reference(
            output: output, logsumexp: logsumexp, delta: delta, dq: dq, dk: dk, dv: dv,
            amplification: (
                dq: dqTerms / peak(dq), dk: dkTerms / peak(dk), dv: dvTerms / peak(dv)))
    }

    /// The scalar the finite-difference check differentiates: `sum(O * dO)` with
    /// `dO` held fixed, whose gradient with respect to `Q`, `K` and `V` is
    /// exactly `dQ`, `dK`, `dV`.
    private func loss(
        q: [Double], k: [Double], v: [Double], dOut: [Float], heads: Int, seq: Int, dim: Int,
        scale: Double
    ) -> Double {
        var total = 0.0
        for head in 0..<heads {
            let base = head * seq * dim
            for row in 0..<seq {
                var scores = [Double](repeating: 0, count: seq)
                for column in 0..<seq {
                    var accumulator = 0.0
                    for element in 0..<dim {
                        accumulator += q[base + row * dim + element] * k[base + column * dim + element]
                    }
                    scores[column] = accumulator * scale
                }
                let peak = scores.max() ?? 0
                var sum = 0.0
                for column in 0..<seq {
                    scores[column] = exp(scores[column] - peak)
                    sum += scores[column]
                }
                for element in 0..<dim {
                    var accumulator = 0.0
                    for column in 0..<seq {
                        accumulator += scores[column] * v[base + column * dim + element]
                    }
                    total += (accumulator / sum) * Double(dOut[base + row * dim + element])
                }
            }
        }
        return total
    }

    private func tensor(_ count: Int, seed: Int, spread: Float = 1) -> [Float] {
        (0..<count).map { index -> Float in
            let bucket: Int = (index &* 31 &+ seed &* 17) % 23
            return (Float(bucket) / 23 * 2 - 1) * spread
        }
    }

    // MARK: - Harness

    private struct Shape {
        var heads: Int
        var seq: Int
        var dim: Int
        var blockM: Int
        var blockN: Int
        var element: String = "f32"

        var label: String {
            "\(element) h=\(heads) s=\(seq) d=\(dim) BLOCK=\(blockM)x\(blockN)"
        }
    }

    private struct Gradients {
        var delta: [Float]
        var dq: [Float]
        var dk: [Float]
        var dv: [Float]
        var sources: [String]
    }

    /// Runs the whole backward pass on the GPU: `Delta` from the preprocess
    /// kernel, then `dQ`, then `dK`/`dV`. `logsumexp` comes from the caller so
    /// that a failure here is a failure of the backward pass and not of the
    /// forward's statistics.
    private func runBackward(
        _ shape: Shape, q: [Float], k: [Float], v: [Float], dOut: [Float], output: [Float],
        logsumexp: [Float], scale: Float, numSimdgroups: Int = 4
    ) throws -> Gradients {
        let count = shape.heads * shape.seq * shape.dim
        let statistics = shape.heads * shape.seq
        let half = shape.element == "f16"
        func input(_ values: [Float]) -> HostArg {
            half ? .halves(values.map(Float16.init)) : .floats(values)
        }
        let strideHead = Int32(shape.seq * shape.dim)
        let strideSeq = Int32(shape.dim)
        let strideLSE = Int32(shape.seq)
        var sources: [String] = []

        let pre = try GPU.run(
            ir: AttentionBackwardKernel.preprocess(
                blockM: shape.blockM, headDim: shape.dim, element: shape.element),
            grid: (GPU.cdiv(shape.seq, shape.blockM), shape.heads, 1),
            args: [
                input(output), input(dOut), .output(count: statistics),
                .int32(strideHead), .int32(strideSeq), .int32(strideLSE),
                .int32(Int32(shape.seq)),
            ], numSimdgroups: numSimdgroups)
        let delta = GPU.read(pre.outputs[0], Float.self, statistics)
        sources.append(pre.source)

        let dqRun = try GPU.run(
            ir: AttentionBackwardKernel.dq(
                blockM: shape.blockM, blockN: shape.blockN, headDim: shape.dim,
                element: shape.element),
            grid: (GPU.cdiv(shape.seq, shape.blockM), shape.heads, 1),
            args: [
                input(q), input(k), input(v), input(dOut),
                .output(count: count, stride: half ? 2 : 4),
                .floats(logsumexp), .floats(delta), .float32(scale),
                .int32(strideHead), .int32(strideSeq), .int32(strideLSE),
                .int32(Int32(shape.seq)),
            ], numSimdgroups: numSimdgroups)
        sources.append(dqRun.source)

        let dkdvRun = try GPU.run(
            ir: AttentionBackwardKernel.dkdv(
                blockM: shape.blockM, blockN: shape.blockN, headDim: shape.dim,
                element: shape.element),
            grid: (GPU.cdiv(shape.seq, shape.blockN), shape.heads, 1),
            args: [
                input(q), input(k), input(v), input(dOut),
                .output(count: count, stride: half ? 2 : 4),
                .output(count: count, stride: half ? 2 : 4),
                .floats(logsumexp), .floats(delta), .float32(scale),
                .int32(strideHead), .int32(strideSeq), .int32(strideLSE),
                .int32(Int32(shape.seq)),
            ], numSimdgroups: numSimdgroups)
        sources.append(dkdvRun.source)

        func read(_ buffer: MTLBuffer) -> [Float] {
            half
                ? GPU.read(buffer, Float16.self, count).map(Float.init)
                : GPU.read(buffer, Float.self, count)
        }
        return Gradients(
            delta: delta, dq: read(dqRun.outputs[0]), dk: read(dkdvRun.outputs[0]),
            dv: read(dkdvRun.outputs[0 + 1]), sources: sources)
    }

    /// `max|a - b|` against a bound derived from the summation error, reporting
    /// both so a run that is merely inside the bound still shows how close it is.
    private func assertWithinSummationBound(
        _ actual: [Float], _ expected: [Double], terms: Int, unitRoundoff: Double,
        amplification: Double, _ label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
        guard actual.count == expected.count else { return }
        let peak = max(expected.map { abs($0) }.max() ?? 0, 1e-30)
        var worst = 0.0
        var worstIndex = 0
        for index in actual.indices {
            let error = abs(Double(actual[index]) - expected[index])
            if error > worst {
                worst = error
                worstIndex = index
            }
        }
        let bound = Double(terms) * unitRoundoff * amplification * peak
        XCTAssertLessThanOrEqual(
            worst, bound,
            "\(label): worst absolute error \(worst) at index \(worstIndex) "
                + "(got \(actual[worstIndex]), expected \(expected[worstIndex])) exceeds the "
                + "summation bound \(bound) = \(terms) * \(unitRoundoff) * \(amplification) "
                + "* \(peak)", file: file, line: line)
    }

    // MARK: - Gradients against the analytic reference

    /// The matrix the task calls for: f32 and f16, at sequence lengths and head
    /// dimensions that divide neither the block shape nor the 8x8 fragment.
    func testGradientsMatchTheAnalyticReference() throws {
        try skipWithoutMetal()
        for shape in [
            Shape(heads: 1, seq: 64, dim: 64, blockM: 16, blockN: 16),
            Shape(heads: 2, seq: 127, dim: 64, blockM: 16, blockN: 32),
            Shape(heads: 1, seq: 96, dim: 80, blockM: 16, blockN: 32),
            Shape(heads: 3, seq: 33, dim: 64, blockM: 32, blockN: 32),
            Shape(heads: 1, seq: 50, dim: 20, blockM: 16, blockN: 16),
            Shape(heads: 2, seq: 128, dim: 64, blockM: 32, blockN: 32),
            Shape(heads: 2, seq: 127, dim: 64, blockM: 16, blockN: 32, element: "f16"),
            Shape(heads: 1, seq: 128, dim: 64, blockM: 32, blockN: 32, element: "f16"),
        ] {
            try checkOneShape(shape)
        }
    }

    private func checkOneShape(_ shape: Shape) throws {
        let count = shape.heads * shape.seq * shape.dim
        let half = shape.element == "f16"
        func round(_ values: [Float]) -> [Float] {
            half ? values.map { Float(Float16($0)) } : values
        }
        let q = round(tensor(count, seed: 1))
        let k = round(tensor(count, seed: 5))
        let v = round(tensor(count, seed: 9))
        let dOut = round(tensor(count, seed: 13))
        let scale = 1 / Float(shape.dim).squareRoot()

        let expected = reference(
            q: q, k: k, v: v, dOut: dOut, heads: shape.heads, seq: shape.seq, dim: shape.dim,
            scale: Double(scale))
        let actual = try runBackward(
            shape, q: q, k: k, v: v, dOut: dOut,
            output: expected.output.map { Float($0) },
            logsumexp: expected.logsumexp.map { Float($0) }, scale: scale)

        // The dot operands are rounded to half in the f16 path; everything else,
        // including both accumulators, stays f32.
        let unitRoundoff = half ? pow(2.0, -11.0) : pow(2.0, -24.0)
        // Each gradient element is a sum over the sequence, and every term of it
        // came out of a dot summed over the head dimension.
        let terms = shape.seq + shape.dim + 2

        assertWithinSummationBound(
            actual.delta, expected.delta, terms: shape.dim + 1, unitRoundoff: unitRoundoff,
            amplification: 1, "\(shape.label) Delta")
        assertWithinSummationBound(
            actual.dq, expected.dq, terms: terms, unitRoundoff: unitRoundoff,
            amplification: expected.amplification.dq, "\(shape.label) dQ")
        assertWithinSummationBound(
            actual.dk, expected.dk, terms: terms, unitRoundoff: unitRoundoff,
            amplification: expected.amplification.dk, "\(shape.label) dK")
        assertWithinSummationBound(
            actual.dv, expected.dv, terms: terms, unitRoundoff: unitRoundoff,
            amplification: expected.amplification.dv, "\(shape.label) dV")
    }

    // MARK: - Gradients against finite differences

    /// The check that owes nothing to the backward formulas: perturb one input
    /// element, evaluate `sum(O * dO)` twice through the forward definition in
    /// `Double`, and compare the central difference with what the GPU produced.
    ///
    /// A central difference has truncation error `O(h^2 |f'''|)` and roundoff
    /// `O(eps_f64 |f| / h)`; at `h = 1e-5` on values of order 1 both are around
    /// `1e-10`, which is four orders below the f32 kernel's own error, so the
    /// comparison is entirely a test of the kernel.
    func testGradientsMatchFiniteDifferences() throws {
        try skipWithoutMetal()
        let shape = Shape(heads: 1, seq: 48, dim: 24, blockM: 16, blockN: 16)
        let count = shape.heads * shape.seq * shape.dim
        let q = tensor(count, seed: 1)
        let k = tensor(count, seed: 5)
        let v = tensor(count, seed: 9)
        let dOut = tensor(count, seed: 13)
        let scale = 1 / Float(shape.dim).squareRoot()

        let expected = reference(
            q: q, k: k, v: v, dOut: dOut, heads: shape.heads, seq: shape.seq, dim: shape.dim,
            scale: Double(scale))
        let actual = try runBackward(
            shape, q: q, k: k, v: v, dOut: dOut, output: expected.output.map { Float($0) },
            logsumexp: expected.logsumexp.map { Float($0) }, scale: scale)

        let qd = q.map(Double.init)
        let kd = k.map(Double.init)
        let vd = v.map(Double.init)
        let h = 1e-5
        // A spread of indices rather than all of them: each one costs two full
        // O(S^2 D) forward passes in Double.
        let probes = stride(from: 0, to: count, by: max(1, count / 12))

        let unitRoundoff = pow(2.0, -24.0)
        let terms = Double(shape.seq + shape.dim + 2)
        let bounds = (
            dq: terms * unitRoundoff * expected.amplification.dq
                * max(expected.dq.map { abs($0) }.max() ?? 0, 1e-30),
            dk: terms * unitRoundoff * expected.amplification.dk
                * max(expected.dk.map { abs($0) }.max() ?? 0, 1e-30),
            dv: terms * unitRoundoff * expected.amplification.dv
                * max(expected.dv.map { abs($0) }.max() ?? 0, 1e-30)
        )

        for index in probes {
            for (name, base, gpu, analytic, bound) in [
                ("dQ", qd, actual.dq, expected.dq, bounds.dq),
                ("dK", kd, actual.dk, expected.dk, bounds.dk),
                ("dV", vd, actual.dv, expected.dv, bounds.dv),
            ] {
                var plus = base
                var minus = base
                plus[index] += h
                minus[index] -= h
                let up: Double
                let down: Double
                switch name {
                case "dQ":
                    up = loss(
                        q: plus, k: kd, v: vd, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                    down = loss(
                        q: minus, k: kd, v: vd, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                case "dK":
                    up = loss(
                        q: qd, k: plus, v: vd, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                    down = loss(
                        q: qd, k: minus, v: vd, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                default:
                    up = loss(
                        q: qd, k: kd, v: plus, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                    down = loss(
                        q: qd, k: kd, v: minus, dOut: dOut, heads: shape.heads, seq: shape.seq,
                        dim: shape.dim, scale: Double(scale))
                }
                let numeric = (up - down) / (2 * h)
                // First: the analytic reference itself is the derivative. This is
                // what would catch a reference that is wrong in the same way the
                // kernel is.
                XCTAssertLessThanOrEqual(
                    abs(numeric - analytic[index]), 1e-6 * max(1, abs(numeric)),
                    "\(name)[\(index)]: analytic \(analytic[index]) vs finite difference \(numeric)")
                // Then the GPU, at the f32 kernel's own error bound.
                XCTAssertLessThanOrEqual(
                    abs(numeric - Double(gpu[index])), bound,
                    "\(name)[\(index)]: GPU \(gpu[index]) vs finite difference \(numeric), "
                        + "bound \(bound)")
            }
        }
    }

    // MARK: - The forward's statistics close the loop

    /// The backward reads one `BLOCK_M`-wide vector per query block, and the
    /// forward is what writes it. This runs the real forward with `emitStats`,
    /// feeds *its* logsumexp and *its* output into the backward, and checks the
    /// gradients — so nothing in the chain comes from the CPU except the check.
    func testForwardStatisticsFeedTheBackward() throws {
        try skipWithoutMetal()
        let shape = Shape(heads: 2, seq: 100, dim: 64, blockM: 16, blockN: 32)
        let count = shape.heads * shape.seq * shape.dim
        let statistics = shape.heads * shape.seq
        let q = tensor(count, seed: 1)
        let k = tensor(count, seed: 5)
        let v = tensor(count, seed: 9)
        let dOut = tensor(count, seed: 13)
        let scale = 1 / Float(shape.dim).squareRoot()

        let forward = try GPU.run(
            ir: AttentionKernel.forward(
                blockM: shape.blockM, blockN: shape.blockN, headDim: shape.dim, emitStats: true),
            grid: (GPU.cdiv(shape.seq, shape.blockM), shape.heads, 1),
            args: [
                .floats(q), .floats(k), .floats(v), .output(count: count), .float32(scale),
                .int32(Int32(shape.seq * shape.dim)), .int32(Int32(shape.dim)),
                .int32(Int32(shape.seq)),
                .output(count: statistics), .int32(Int32(shape.seq)),
            ])
        let output = GPU.read(forward.outputs[0], Float.self, count)
        let logsumexp = GPU.read(forward.outputs[1], Float.self, statistics)

        let expected = reference(
            q: q, k: k, v: v, dOut: dOut, heads: shape.heads, seq: shape.seq, dim: shape.dim,
            scale: Double(scale))
        // The statistics themselves first: the logsumexp is a log, so its error
        // is the relative error of the sum it takes the log of.
        assertWithinSummationBound(
            logsumexp, expected.logsumexp, terms: shape.seq + shape.dim,
            unitRoundoff: pow(2.0, -24.0), amplification: 1, "forward logsumexp")

        let actual = try runBackward(
            shape, q: q, k: k, v: v, dOut: dOut, output: output, logsumexp: logsumexp, scale: scale)
        let terms = shape.seq + shape.dim + 2
        assertWithinSummationBound(
            actual.dq, expected.dq, terms: terms, unitRoundoff: pow(2.0, -24.0),
            amplification: expected.amplification.dq, "end-to-end dQ")
        assertWithinSummationBound(
            actual.dk, expected.dk, terms: terms, unitRoundoff: pow(2.0, -24.0),
            amplification: expected.amplification.dk, "end-to-end dK")
        assertWithinSummationBound(
            actual.dv, expected.dv, terms: terms, unitRoundoff: pow(2.0, -24.0),
            amplification: expected.amplification.dv, "end-to-end dV")
    }

    /// The gradients do not depend on the threadgroup size: the three kernels are
    /// launched at whatever `threads_per_threadgroup` the backend reports, and
    /// every dot, staging pass and barrier has to be right at each of them.
    ///
    /// `BLOCK_N = 16` rather than 32, because the `dK`/`dV` kernel at
    /// `BLOCK_N = 32` does not fit in Metal's 32KB at `num_warps = 1`: with one
    /// simdgroup the register blocking is chosen large so that the single
    /// simdgroup holds many fragments, and every tile is then padded up to whole
    /// blocks of fragments (`testTheDKDVKernelNeedsTwoSimdgroupsAtBLOCKN32`).
    func testGradientsAtEveryNumWarps() throws {
        try skipWithoutMetal()
        let shape = Shape(heads: 1, seq: 65, dim: 64, blockM: 16, blockN: 16)
        let count = shape.heads * shape.seq * shape.dim
        let q = tensor(count, seed: 1)
        let k = tensor(count, seed: 5)
        let v = tensor(count, seed: 9)
        let dOut = tensor(count, seed: 13)
        let scale = 1 / Float(shape.dim).squareRoot()
        let expected = reference(
            q: q, k: k, v: v, dOut: dOut, heads: shape.heads, seq: shape.seq, dim: shape.dim,
            scale: Double(scale))
        let terms = shape.seq + shape.dim + 2

        for simdgroups in [1, 2, 4, 8] {
            let actual = try runBackward(
                shape, q: q, k: k, v: v, dOut: dOut, output: expected.output.map { Float($0) },
                logsumexp: expected.logsumexp.map { Float($0) }, scale: scale,
                numSimdgroups: simdgroups)
            assertWithinSummationBound(
                actual.dq, expected.dq, terms: terms, unitRoundoff: pow(2.0, -24.0),
                amplification: expected.amplification.dq, "num_warps=\(simdgroups) dQ")
            assertWithinSummationBound(
                actual.dk, expected.dk, terms: terms, unitRoundoff: pow(2.0, -24.0),
                amplification: expected.amplification.dk, "num_warps=\(simdgroups) dK")
            assertWithinSummationBound(
                actual.dv, expected.dv, terms: terms, unitRoundoff: pow(2.0, -24.0),
                amplification: expected.amplification.dv, "num_warps=\(simdgroups) dV")
        }
    }

    // MARK: - What the lowering had to do

    /// The `dQ` loop stages three dots and the `dK`/`dV` loop four, all into the
    /// storage of a register-resident accumulator whose own tile is free for the
    /// duration. Summing their footprints — which is what the emitter did when
    /// only one dot could use the arena — overruns Metal's 32KB at every block
    /// shape worth having.
    func testEveryDotInTheLoopStagesIntoTheAccumulatorsArena() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionBackwardKernel.dq(blockM: 16, blockN: 32, headDim: 64),
            options: .init())
        // One accumulator tile, and every operand tile is a pointer into it.
        let accumulators = source.components(separatedBy: "threadgroup float tm_dot_c").count - 1
        XCTAssertEqual(accumulators, 3, source)  // dQ, plus the two score tiles
        for tile in ["tm_dot_a", "tm_dot_b"] {
            XCTAssertTrue(
                source.contains("threadgroup float *\(tile)"),
                "\(tile) should be a pointer into the arena\n\(source)")
        }
        XCTAssertFalse(source.contains("threadgroup float tm_dot_a"), source)
        XCTAssertFalse(source.contains("threadgroup float tm_dot_b"), source)
    }

    /// Two loop-carried `tt.dot` accumulators, both yielded directly by their
    /// dot, so both live in simdgroup registers for the whole query loop — and
    /// only the first of them lends its storage to the operand arena.
    func testTheDKDVLoopCarriesTwoRegisterResidentAccumulators() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionBackwardKernel.dkdv(blockM: 16, blockN: 32, headDim: 64),
            options: .init())
        XCTAssertTrue(source.contains("also the arena its operand tiles stage into"), source)
        XCTAssertEqual(
            source.components(separatedBy: "also the arena its operand tiles stage into").count - 1,
            1, "only one accumulator should carry the arena\n\(source)")
        // Both accumulators' fragments are named registers, not array slots.
        XCTAssertTrue(source.contains("simdgroup_float8x8 tm_dot_c"), source)
    }

    /// The one block-shape-and-warp combination the backward does *not* fit, by
    /// name and with its byte count, so that the constraint is a checked fact
    /// rather than a remark in a document.
    ///
    /// At one simdgroup the register blocking is chosen to hand that simdgroup
    /// many fragments, and every tile is padded up to whole blocks of them, which
    /// inflates the two `BLOCK_N x HEAD_DIM` accumulators and the operand arena
    /// together past 32KB. Two simdgroups is enough.
    func testTheDKDVKernelNeedsTwoSimdgroupsAtBLOCKN32() throws {
        let ir = AttentionBackwardKernel.dkdv(blockM: 16, blockN: 32, headDim: 64)
        do {
            _ = try MetalCompiler.emitMSL(ttir: ir, options: .init(numSimdgroups: 1))
            XCTFail("expected a threadgroup-memory error")
        } catch {
            XCTAssertTrue("\(error)".contains("32768-byte limit"), "\(error)")
        }
        XCTAssertNoThrow(try MetalCompiler.emitMSL(ttir: ir, options: .init(numSimdgroups: 2)))
    }

    /// `P` is never stored: the backward recomputes it from the one
    /// `BLOCK_M`-wide logsumexp vector the forward wrote, which is what makes
    /// this a recompute-based backward rather than one that needs the `S x S`
    /// score matrix in memory.
    func testTheBackwardRecomputesTheProbabilitiesFromTheLogsumexp() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionBackwardKernel.dq(blockM: 16, blockN: 32, headDim: 64),
            options: .init())
        XCTAssertTrue(source.contains("precise::exp2"), source)
        // ...and the forward is what writes it.
        let forward = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(
                blockM: 16, blockN: 32, headDim: 64, emitStats: true), options: .init())
        XCTAssertTrue(forward.contains("precise::log2"), forward)
    }

    /// Every backward kernel is accepted by Metal's own front end at every block
    /// shape the tests use.
    func testEveryBackwardKernelCompilesInMetal() throws {
        try skipWithoutMetal()
        for element in ["f32", "f16"] {
            for (m, n, d) in [(16, 16, 64), (16, 32, 64), (32, 32, 64), (16, 32, 80), (16, 16, 20)] {
                for ir in [
                    AttentionBackwardKernel.preprocess(blockM: m, headDim: d, element: element),
                    AttentionBackwardKernel.dq(
                        blockM: m, blockN: n, headDim: d, element: element),
                    AttentionBackwardKernel.dkdv(
                        blockM: m, blockN: n, headDim: d, element: element),
                ] {
                    let source = try MetalCompiler.emitMSL(ttir: ir, options: .init())
                    XCTAssertNoThrow(try MetalCompiler.compileMSL(source), source)
                }
            }
        }
    }
}
