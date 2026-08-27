import Foundation
import Metal
import TritonMetalBench
import XCTest

@testable import TritonMetalCore

/// Milestone: FlashAttention-2 forward, fused, against a CPU reference.
///
/// This is the kernel the backend was aimed at. It needs, all at once: three
/// block axes rather than two (`p` spans `BLOCK_M x BLOCK_N` and `acc` spans
/// `BLOCK_M x HEAD_DIM`, and neither contains the other), a `tt.trans` for `K^T`,
/// two `tt.dot`s whose axes cross, a `tt.reduce` whose result a later dot's
/// staging loops have to read, and three loop-carried tensors — two per-row
/// vectors and the accumulator, which is rescaled between the dots.
final class AttentionTests: XCTestCase {

    // MARK: - CPU reference

    /// Plain (unfused) attention, in f64 accumulation, over `[heads, seq, dim]`.
    private func reference(
        q: [Float], k: [Float], v: [Float], heads: Int, seq: Int, dim: Int, scale: Float
    ) -> [Float] {
        var out = [Float](repeating: 0, count: heads * seq * dim)
        for head in 0..<heads {
            let base = head * seq * dim
            for row in 0..<seq {
                var scores = [Double](repeating: 0, count: seq)
                for column in 0..<seq {
                    var accumulator = 0.0
                    for element in 0..<dim {
                        accumulator +=
                            Double(q[base + row * dim + element])
                            * Double(k[base + column * dim + element])
                    }
                    scores[column] = accumulator * Double(scale)
                }
                let peak = scores.max() ?? 0
                var total = 0.0
                for column in 0..<seq {
                    scores[column] = exp(scores[column] - peak)
                    total += scores[column]
                }
                for element in 0..<dim {
                    var accumulator = 0.0
                    for column in 0..<seq {
                        accumulator += scores[column] * Double(v[base + column * dim + element])
                    }
                    out[base + row * dim + element] = Float(accumulator / total)
                }
            }
        }
        return out
    }

    private func tensor(_ count: Int, seed: Int, spread: Float = 1) -> [Float] {
        (0..<count).map { index -> Float in
            let bucket: Int = (index &* 31 &+ seed &* 17) % 23
            let unit: Float = Float(bucket) / Float(23)
            return (unit * 2 - 1) * spread
        }
    }

    // MARK: - Harness

    private struct Shape {
        var heads: Int
        var seq: Int
        var dim: Int
        var blockM: Int
        var blockN: Int
    }

    @discardableResult
    private func runAttention(
        _ shape: Shape, element: String = "f32", spread: Float = 1, scale: Float? = nil,
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> (actual: [Float], expected: [Float]) {
        let count = shape.heads * shape.seq * shape.dim
        var q = tensor(count, seed: 1, spread: spread)
        var k = tensor(count, seed: 5, spread: spread)
        var v = tensor(count, seed: 9, spread: spread)
        if element == "f16" {
            // Round the inputs to the precision the kernel will actually see, so
            // the reference measures the fusion rather than the rounding.
            q = q.map { Float(Float16($0)) }
            k = k.map { Float(Float16($0)) }
            v = v.map { Float(Float16($0)) }
        }
        let softmaxScale = scale ?? 1 / Float(shape.dim).squareRoot()
        let ir = AttentionKernel.forward(
            blockM: shape.blockM, blockN: shape.blockN, headDim: shape.dim, element: element)
        let half = element == "f16"
        let run = try GPU.run(
            ir: ir,
            grid: (GPU.cdiv(shape.seq, shape.blockM), shape.heads, 1),
            args: [
                half ? .halves(q.map(Float16.init)) : .floats(q),
                half ? .halves(k.map(Float16.init)) : .floats(k),
                half ? .halves(v.map(Float16.init)) : .floats(v),
                .output(count: count, stride: half ? 2 : 4),
                .float32(softmaxScale),
                .int32(Int32(shape.seq * shape.dim)), .int32(Int32(shape.dim)),
                .int32(Int32(shape.seq)),
            ])
        let actual = half
            ? GPU.read(run.outputs[0], Float16.self, count).map(Float.init)
            : GPU.read(run.outputs[0], Float.self, count)
        let expected = reference(
            q: q, k: k, v: v, heads: shape.heads, seq: shape.seq, dim: shape.dim,
            scale: softmaxScale)
        return (actual, expected)
    }

    // MARK: - Correctness

    /// f32, at sequence lengths and head dimensions that divide neither the block
    /// shape nor the 8x8 fragment.
    func testForwardMatchesCPUAcrossShapes() throws {
        try skipWithoutMetal()
        let shapes = [
            Shape(heads: 2, seq: 127, dim: 64, blockM: 16, blockN: 32),
            Shape(heads: 1, seq: 512, dim: 64, blockM: 32, blockN: 32),
            Shape(heads: 1, seq: 96, dim: 80, blockM: 16, blockN: 32),
            Shape(heads: 3, seq: 33, dim: 64, blockM: 32, blockN: 32),
            Shape(heads: 1, seq: 64, dim: 64, blockM: 16, blockN: 16),
        ]
        for shape in shapes {
            let (actual, expected) = try runAttention(shape)
            assertClose(
                actual, expected, tolerance: 2e-5,
                "h=\(shape.heads) s=\(shape.seq) d=\(shape.dim) "
                    + "BLOCK=\(shape.blockM)x\(shape.blockN)")
        }
    }

    /// f16 in, f32 accumulate — the shape that matters for ML. The dots run on
    /// `simdgroup_half8x8` operands with a `simdgroup_float8x8` accumulator, and
    /// the softmax statistics stay in f32 throughout.
    func testForwardInHalfPrecisionAccumulatesInFloat() throws {
        try skipWithoutMetal()
        for shape in [
            Shape(heads: 2, seq: 127, dim: 64, blockM: 16, blockN: 32),
            Shape(heads: 1, seq: 128, dim: 64, blockM: 32, blockN: 32),
        ] {
            let (actual, expected) = try runAttention(shape, element: "f16")
            assertClose(
                actual, expected, tolerance: 4e-3,
                "f16 h=\(shape.heads) s=\(shape.seq) d=\(shape.dim)")
        }
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64, element: "f16"),
            options: .init())
        XCTAssertTrue(source.contains("simdgroup_half8x8"), source)
        XCTAssertTrue(source.contains("simdgroup_float8x8"), source)
    }

    /// The whole point of an online softmax: scores large enough that a naive
    /// `exp` overflows still produce finite, correct output, and every row's
    /// attention weights still sum to one.
    func testForwardIsStableOnScoresThatOverflowExp() throws {
        try skipWithoutMetal()
        let shape = Shape(heads: 1, seq: 100, dim: 64, blockM: 16, blockN: 32)
        // Raw scores reach ~dim * spread^2 * scale; at spread 12 that is ~1100,
        // and exp(1100) is not a float.
        let (actual, expected) = try runAttention(shape, spread: 12, scale: 1)
        for value in actual { XCTAssertTrue(value.isFinite, "output is not finite: \(value)") }
        assertClose(actual, expected, tolerance: 1e-4, "overflowing scores")

        // With V a partition of unity the output of each row is a weighted mean,
        // so every element must land inside the range of V.
        let bounds = (min: Float(-12), max: Float(12))
        for value in actual {
            XCTAssertLessThanOrEqual(value, bounds.max + 1e-3)
            XCTAssertGreaterThanOrEqual(value, bounds.min - 1e-3)
        }
    }

    /// A `tt.dot` kernel is launched at the threadgroup size the backend reports;
    /// the reductions inside this one fold across whatever that turns out to be.
    func testForwardAtEveryNumWarps() throws {
        try skipWithoutMetal()
        let shape = Shape(heads: 1, seq: 65, dim: 64, blockM: 16, blockN: 32)
        for simdgroups in [1, 2, 4, 8] {
            let count = shape.heads * shape.seq * shape.dim
            let q = tensor(count, seed: 1)
            let k = tensor(count, seed: 5)
            let v = tensor(count, seed: 9)
            let scale = 1 / Float(shape.dim).squareRoot()
            let run = try GPU.run(
                ir: AttentionKernel.forward(
                    blockM: shape.blockM, blockN: shape.blockN, headDim: shape.dim),
                grid: (GPU.cdiv(shape.seq, shape.blockM), shape.heads, 1),
                args: [
                    .floats(q), .floats(k), .floats(v), .output(count: count), .float32(scale),
                    .int32(Int32(shape.seq * shape.dim)), .int32(Int32(shape.dim)),
                    .int32(Int32(shape.seq)),
                ], numSimdgroups: simdgroups)
            assertClose(
                GPU.read(run.outputs[0], Float.self, count),
                reference(
                    q: q, k: k, v: v, heads: shape.heads, seq: shape.seq, dim: shape.dim,
                    scale: scale),
                tolerance: 2e-5, "num_warps=\(simdgroups)")
        }
    }

    // MARK: - What the lowering had to do

    /// Three iteration axes, not two, and no separate contraction axis at all:
    /// the first dot contracts over the head dimension its accumulator iterates
    /// and the second over the key block its softmax iterates.
    func testAttentionNeedsThreeIterationAxes() throws {
        let ir = AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64)
        let layout = try LayoutInference.compute(for: TritonIRParser.parse(ir).functions[0])
        XCTAssertEqual(layout.rank, 3)
        XCTAssertEqual(layout.contractions, [])
        XCTAssertEqual(layout.shape, [16, 64, 32])
        // The rows are the shared outer loop; the head dimension and the key
        // block are each the innermost axis of a nest of their own.
        XCTAssertEqual(layout.uniformAxes, [0])
        XCTAssertEqual(layout.path(of: "p"), [0, 2])
        XCTAssertEqual(layout.path(of: "acc"), [0, 1])
        XCTAssertEqual(layout.path(of: "m_i"), [0])
        // K is loaded (BLOCK_N, HEAD_DIM) and transposed; the transpose is a
        // relabelling, so K^T's rows are the head-dimension axis.
        XCTAssertEqual(layout.axes["k"], [2, 1])
        XCTAssertEqual(layout.axes["kt"], [1, 2])
    }

    /// `tt.trans` emits no code: the transposed operand is staged over the axes
    /// its own tile wants, in the order the tile wants them.
    func testTransposeIsAStagingTimeRelabelling() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64), options: .init())
        // The K^T tile is HEAD_DIM x BLOCK_N, walked by the head-dimension loop
        // outside and the key loop inside.
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_b7"), source)
        let stageB = try XCTUnwrap(source.range(of: "// tt.dot: stage B (64x32)"))
        let body = String(source[stageB.upperBound...].prefix(400))
        XCTAssertTrue(body.contains("for (uint tm_i1 ="), body)
        XCTAssertTrue(body.contains("for (uint tm_i2"), body)
    }

    /// The three carried tensors, and how each is lowered: `m_i` and `l_i` are
    /// spilled with a shadow, `acc` is the accumulator tile the second dot
    /// updates in place, and the per-row rescale becomes that dot's staging pass.
    func testCarriedTensorsAreSpilledAndTheAccumulatorIsRescaledInPlace() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64), options: .init())
        XCTAssertTrue(source.contains("carries '%m_i' across a cross-lane operation"), source)
        XCTAssertTrue(source.contains("carries '%l_i' across a cross-lane operation"), source)
        XCTAssertTrue(
            source.contains("carries a 16x64 tt.dot accumulator that is rescaled per iteration"),
            source)
        // The rescale reads and writes the same element of the accumulator tile,
        // which is why it needs no ordering of its own.
        let rescale = try XCTUnwrap(
            source.range(of: "// tt.dot: the carried accumulator, rescaled in place"))
        let body = String(source[rescale.upperBound...].prefix(600))
        XCTAssertTrue(body.contains("= tm_dot_c4[tm_i0 * 64u + tm_i1] * valpha;"), body)
        XCTAssertTrue(body.contains("tm_dot_c4[tm_i0 * 64u + tm_i1] = "), body)
        // And the shadow copy happens once, at the end of the body, fenced.
        let yield = try XCTUnwrap(
            source.range(of: "// scf.yield: the carried tensors take their new values"))
        let tail = String(source[yield.upperBound...].prefix(400))
        XCTAssertTrue(tail.contains("tm_carry0[tm_i0] = tm_carry_next1[tm_i0];"), tail)
    }

    /// A row reduction the second dot's staging loops need cannot be rebuilt
    /// there — the fold has already happened — so it is written into a small
    /// threadgroup array where it is computed.
    func testTheRowMaximumIsSpilledForTheSecondDot() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64), options: .init())
        XCTAssertTrue(source.contains("threadgroup float tm_reduce8[16];"), source)
        XCTAssertTrue(source.contains("tm_reduce8[tm_i0] = vrow_max;"), source)
        // ...and read back inside the staging loops of the second dot.
        let stageA = try XCTUnwrap(source.range(of: "// tt.dot: stage A (16x32)"))
        let body = String(source[stageA.upperBound...].prefix(800))
        XCTAssertTrue(body.contains("tm_reduce8[tm_i0]"), body)
    }

    /// Two dots' operand tiles are never live at the same time, so they share one
    /// arena — without which the kernel does not fit in Metal's 32KB.
    func testOperandTilesShareOneArena() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AttentionKernel.forward(blockM: 16, blockN: 32, headDim: 64), options: .init())
        XCTAssertTrue(source.contains("shared tt.dot operand arena"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_a6 = tm_dot_arena;"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_a9 = tm_dot_arena;"), source)
    }

    /// A block shape whose tiles overrun the threadgroup budget is refused with
    /// the byte count, as the GEMM's is.
    func testOversizedAttentionBlockIsRefusedWithItsByteCount() {
        do {
            _ = try MetalCompiler.emitMSL(
                ttir: AttentionKernel.forward(blockM: 128, blockN: 128, headDim: 128),
                options: .init())
            XCTFail("expected a threadgroup-memory error")
        } catch {
            XCTAssertTrue("\(error)".contains("32768-byte limit"), "\(error)")
        }
    }
}
