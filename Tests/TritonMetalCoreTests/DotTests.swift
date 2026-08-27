import Foundation
import Metal
import XCTest

@testable import TritonMetalCore

/// Milestone: `tt.dot` on Metal's simdgroup matrices, and the blocked GEMM built
/// on top of it.
///
/// Every GPU case is checked against a CPU reference at sizes that divide neither
/// the block shape nor the 8x8 fragment.
final class DotTests: XCTestCase {

    // MARK: - CPU references

    private func reference(
        _ a: [Float], _ b: [Float], m: Int, n: Int, k: Int, lda: Int, ldb: Int
    ) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        for row in 0..<m {
            for step in 0..<k {
                let left = a[row * lda + step]
                if left == 0 { continue }
                for column in 0..<n {
                    c[row * n + column] += left * b[step * ldb + column]
                }
            }
        }
        return c
    }

    private func matrix(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float(($0 &* 7 &+ seed &* 13) % 17) * 0.25 - 2 }
    }

    // MARK: - One tile, no control flow

    /// `C = A @ B` in a single `tt.dot`. The sizes that are not multiples of 8
    /// are the point: they exercise the zero-padded edge fragments.
    func testSingleTileProduct() throws {
        try skipWithoutMetal()
        for (m, n, k) in [(16, 16, 16), (8, 8, 8), (32, 16, 24), (12, 20, 12), (5, 3, 7), (1, 1, 1)]
        {
            let a = matrix(m * k, seed: 1)
            let b = matrix(k * n, seed: 5)
            let run = try GPU.run(
                ir: DotFixtures.singleTile(m: m, n: n, k: k),
                args: [.floats(a), .floats(b), .output(count: m * n)])
            assertClose(
                GPU.read(run.outputs[0], Float.self, m * n),
                reference(a, b, m: m, n: n, k: k, lda: k, ldb: n),
                tolerance: 1e-5, "\(m)x\(n)x\(k)")
        }
    }

    /// The block index space grows a third, non-iterated dimension for K; the
    /// launch metadata still describes the MxN space the threads walk.
    func testContractionDimensionIsNotPartOfTheIterationSpace() throws {
        let emission = try MetalCompiler.emit(
            ttir: DotFixtures.singleTile(m: 32, n: 16, k: 24), options: .init(numSimdgroups: 4))
        XCTAssertEqual(emission.kernels[0].blockShape, [32, 16])
        XCTAssertEqual(emission.kernels[0].blockSize, 32 * 16)
        // A dot kernel is sized for simdgroup-matrix work, not for BLOCK_N.
        XCTAssertEqual(emission.kernels[0].threadsPerThreadgroup, 128)

        let module = try TritonIRParser.parse(DotFixtures.singleTile(m: 32, n: 16, k: 24))
        let layout = try LayoutInference.compute(for: module.functions[0])
        XCTAssertEqual(layout.rank, 2)
        XCTAssertEqual(layout.shape, [32, 16])
        XCTAssertEqual(layout.contractions, [24])
        // %14 is the left operand: rows on the M axis, columns on the K axis.
        XCTAssertEqual(layout.axes["14"], [0, 2])
        XCTAssertEqual(layout.axes["22"], [2, 1])
        XCTAssertEqual(layout.axes["23"], [0, 1])
    }

    func testTilesArePaddedToWholeFragmentsAndZeroFilled() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.singleTile(m: 12, n: 20, k: 12), options: .init())
        // 12 -> 16, 20 -> 24, 12 -> 16.
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[384];  // 16x24"), source)
        XCTAssertTrue(source.contains("threadgroup float tm_dot_a1[256];  // 16x16"), source)
        XCTAssertTrue(source.contains("threadgroup float tm_dot_b2[384];  // 16x24"), source)
        // The padding lanes take a zero without evaluating any pointer arithmetic.
        XCTAssertTrue(source.contains("float tm_staged = 0.0f;  // 8x8 fragment padding"), source)
        XCTAssertTrue(source.contains("if (tm_i0 < 12u && tm_i2 < 12u) {"), source)
        // 2x3 fragments over 4 simdgroups: two rows of fragments per simdgroup and
        // one column, which is the blocking that measures fastest on Apple silicon
        // — the shared column is what lets the step's three loads (two A, one B)
        // feed two multiply-accumulates.
        XCTAssertTrue(
            source.contains(
                "// tt.dot accumulator: 1x3 blocks of 2x1 fragments over 4 simdgroups"),
            source)
        XCTAssertTrue(source.contains("// tt.dot: 2 contraction steps, each 2 "
            + "multiply-accumulates off 3 operand loads"), source)
    }

    func testStagingIsFencedByBarriersOnBothSides() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16), options: .init())
        let stageA = try XCTUnwrap(source.range(of: "// tt.dot: stage A"))
        let fragments = try XCTUnwrap(source.range(of: "simdgroup_multiply_accumulate"))
        let between = String(source[stageA.upperBound..<fragments.lowerBound])
        XCTAssertTrue(between.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), between)
        // And one after, so the next iteration may overwrite the operand tiles.
        let after = String(source[fragments.upperBound...])
        XCTAssertTrue(after.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), after)
    }

    /// The accumulator is the one tensor that survives the K loop, and it lives in
    /// threadgroup memory rather than in per-lane registers — but a `dense<0.0>`
    /// seed never passes *through* that memory: the fragments are born zero in
    /// registers, so the prologue is empty and the tile is only written at the end.
    func testAccumulatorLivesInThreadgroupMemoryAcrossTheLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16), options: .init())
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[1024];"), source)
        let loop = try XCTUnwrap(source.range(of: "for (int varg12 ="))
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        XCTAssertFalse(prologue.contains("tm_dot_c0[tm_i0 * 32u + tm_i1] = vcst;"), prologue)
        XCTAssertTrue(
            prologue.contains(
                "simdgroup_float8x8 tm_dot_c0_r_0_0_0 = "
                    + "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"), prologue)
        // Read back per lane after the loop, by indexing the tile — a panel of
        // it, since the epilogue streams the fragments out (see
        // `testTheEpilogueStreamsTheAccumulatorOutInPanels`).
        let epilogue = String(source[loop.upperBound...])
        XCTAssertTrue(
            epilogue.contains("tm_dot_c0[(tm_i0 - tm_panel0) * 32u + tm_i1] * vcone;"), epilogue)
    }

    /// A non-zero seed still has to be staged in and loaded, because the value has
    /// to reach the fragments somehow.
    func testNonZeroAccumulatorSeedIsStillStagedThroughTheTile() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16, seed: "1.000000e+00"),
            options: .init())
        let loop = try XCTUnwrap(source.range(of: "for (int varg12 ="))
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        XCTAssertTrue(prologue.contains("tm_dot_c0[tm_i0 * 32u + tm_i1] = vcst;"), prologue)
        XCTAssertTrue(prologue.contains("simdgroup_load(tm_dot_c0_r_"), prologue)
        XCTAssertFalse(prologue.contains("make_filled_simdgroup_matrix"), prologue)
    }

    /// ...but it does not make a round trip through that memory on every K step:
    /// the output fragments are loaded into simdgroup registers once, before the
    /// loop, and stored back once, after it.
    func testAccumulatorFragmentsStayInRegistersAcrossTheLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 64, blockK: 32),
            options: .init(numSimdgroups: 16))
        let loop = try XCTUnwrap(source.range(of: "for (int varg12 ="))
        let close = try XCTUnwrap(source.range(of: "// tt.dot accumulator: 2 panels"))
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        let body = String(source[loop.upperBound..<close.lowerBound])
        let epilogue = String(source[close.lowerBound...])

        // 8x8 fragments over 16 simdgroups in 2x1 blocks: two waves of two
        // accumulator registers, zero-filled before the loop and stored after it.
        XCTAssertTrue(
            prologue.contains(
                "// tt.dot accumulator: 4x8 blocks of 2x1 fragments over 16 simdgroups, 2 waves"),
            prologue)
        XCTAssertEqual(occurrences(of: "make_filled_simdgroup_matrix", in: prologue), 4)
        XCTAssertEqual(occurrences(of: "simdgroup_load(tm_dot_c0_r_", in: prologue), 0)
        XCTAssertEqual(occurrences(of: "simdgroup_store(tm_dot_c0_r_", in: epilogue), 4)

        // Nothing touches the accumulator tile inside the loop.
        XCTAssertEqual(occurrences(of: "simdgroup_load(tm_dot_c0_r_", in: body), 0)
        XCTAssertEqual(occurrences(of: "simdgroup_store(", in: body), 0)
        XCTAssertEqual(occurrences(of: "simdgroup_multiply_accumulate(", in: body), 4)
        // Four A fragments and *one* B fragment feed those four: both waves of a
        // simdgroup sit in the same column of the block grid, and a 2x1 block is
        // one column wide, so the B fragment is loaded once rather than per wave.
        XCTAssertEqual(occurrences(of: "simdgroup_load(", in: body), 5)
        XCTAssertEqual(occurrences(of: "simdgroup_load(tm_b0_", in: body), 1)
    }

    /// A run of four consecutive columns is one vector load when the addresses
    /// turn out to be contiguous, aligned and inside the mask — and the scalar
    /// run is still there for when they are not, because all three facts are
    /// about the kernel's arguments rather than its shape.
    func testStagingRunsAreReadWithOneVectorLoadWhenTheyCan() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 128, blockK: 16),
            options: .init(numSimdgroups: 16))
        // The stride check names the kernel's own `stride_bn` argument, which the
        // affine walk over the staging subgraph identified as the column stride.
        XCTAssertTrue(
            source.contains(
                "tm_dot_b2_vok = tm_dot_b2_vok && (1 * v26) == 1 && "
                    + "(((uintptr_t)tm_dot_b2_vbase & 15u) == 0u);"), source)
        XCTAssertTrue(
            source.contains(
                "float4 tm_dot_b2_vec = *(device const float4 *)(tm_dot_b2_vbase);"), source)
        XCTAssertTrue(
            source.contains("tm_dot_b2[tm_i2 * 128u + tm_i1_run + 3u] = tm_dot_b2_vec[3u];"),
            source)
        // Both paths are emitted, and the scalar one still does its own masking.
        XCTAssertTrue(source.contains("float v73 = v72 ? *varg15 : vzeroB;"), source)

        // The mask is checked at the run's *last* column only, which is sound
        // because it is monotone in the column index.
        XCTAssertTrue(source.contains("uint tm_i1 = tm_i1_run + 3u;"), source)

        // Off by request, nothing of it survives.
        let scalar = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 128, blockK: 16),
            options: .init(numSimdgroups: 16, dotVectorStaging: false))
        XCTAssertFalse(scalar.contains("float4 tm_dot_b2_vec"), scalar)
        XCTAssertFalse(scalar.contains("uintptr_t"), scalar)
    }

    /// The register blocking is a knob the autotuner sweeps, so an explicit
    /// request has to override the emitter's own choice — which, on Apple
    /// silicon, is the *smallest* blocking rather than the largest one that fits.
    func testRegisterBlockingCanBeRequested() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 64, blockK: 32),
            options: .init(numSimdgroups: 4, dotRegisterM: 4, dotRegisterN: 2))
        XCTAssertTrue(
            source.contains("// tt.dot accumulator: 2x4 blocks of 4x2 fragments over 4 simdgroups"),
            source)
        // 12 operand loads spelled out, 10 issued: the two waves share a column of
        // the block grid and therefore their two B fragments.
        XCTAssertTrue(source.contains("// tt.dot: 4 contraction steps, each 16 "
            + "multiply-accumulates off 10 operand loads"), source)
    }

    /// A register-resident accumulator's tile is dead for exactly as long as the
    /// operand tiles exist, so the two share storage. That is what lets a block
    /// shape whose accumulator alone fills Metal's 32KB budget still lower.
    func testAccumulatorTileDoublesAsTheStagingArena() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 128, blockN: 64, blockK: 32),
            options: .init(numSimdgroups: 16))
        // A 128x64 f32 accumulator is 8192 floats — Metal's entire 32KB budget —
        // but nothing ever holds all of it: the operand tiles alias into it
        // during the contraction, and the epilogue streams the fragments out a
        // panel at a time, so what is actually declared is the larger of the two,
        // which is the 6144-slot operand footprint (128x32 A plus 32x64 B).
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[6144];"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_a1 = tm_dot_c0;"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_b2 = tm_dot_c0 + 4096;"), source)
        XCTAssertEqual(occurrences(of: "threadgroup float tm_dot", in: source), 1)

        // The sharing is only safe because a barrier separates the operand tiles'
        // last read from the accumulator's write back over the top of them. (In
        // the other direction there is nothing to order: a zero accumulator is
        // never in the tile in the first place.)
        let arithmetic = try XCTUnwrap(source.range(of: "simdgroup_multiply_accumulate"))
        let store = try XCTUnwrap(source.range(of: "simdgroup_store(tm_dot_c0_r_"))
        let between = String(source[arithmetic.upperBound..<store.lowerBound])
        XCTAssertTrue(between.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), between)
    }

    /// The accumulator lives in registers for the whole contraction; the only
    /// thing that wants it in threadgroup memory is the epilogue, which reads it
    /// back per lane. So the epilogue takes it a panel at a time: store the
    /// fragments whose rows are in this panel, run the per-lane tail over those
    /// rows, repeat. The tile then costs one panel instead of `BLOCK_M x BLOCK_N`.
    func testTheEpilogueStreamsTheAccumulatorOutInPanels() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 64, blockK: 16),
            options: .init(numSimdgroups: 8))
        // 64x64 is 4096 floats; what is declared is the 2048-slot operand
        // footprint, which the 32-row panel fits inside for free.
        XCTAssertTrue(
            source.contains(
                "threadgroup float tm_dot_c0[2048];  // 64x64 accumulator, streamed out 32 rows"),
            source)
        XCTAssertTrue(
            source.contains("for (uint tm_panel0 = 0u; tm_panel0 < 64u; tm_panel0 += 32u) {"),
            source)

        let panel = try XCTUnwrap(source.range(of: "for (uint tm_panel0 ="))
        let body = String(source[panel.upperBound...])
        // Every fragment is stored under a test against the panel's row range,
        // so each is written once across the whole epilogue rather than once per
        // panel; the ones not in this panel simply stay in registers.
        XCTAssertEqual(occurrences(of: "simdgroup_store(tm_dot_c0_r_", in: body), 8)
        XCTAssertEqual(occurrences(of: ">= tm_panel0 &&", in: body), 8)
        // The per-lane walk is restricted to the panel, and reads the tile
        // relative to it.
        XCTAssertTrue(
            body.contains(
                "for (uint tm_i0 = tm_panel0 + tm_thread_id.x / 64u; tm_i0 < tm_panel0 + 32u;"),
            body)
        XCTAssertTrue(body.contains("tm_dot_c0[(tm_i0 - tm_panel0) * 64u + tm_i1]"), body)
        // Both barriers matter: the leading one orders the previous panel's
        // reads (and, on the first pass, the operand tiles' last reads) before
        // this panel's stores.
        let store = try XCTUnwrap(body.range(of: "simdgroup_store(tm_dot_c0_r_"))
        let before = String(body[body.startIndex..<store.lowerBound])
        XCTAssertTrue(before.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), before)
    }

    /// The shape the panelled epilogue exists for. A `128x128` f32 accumulator is
    /// 64KB — twice Metal's whole threadgroup budget — so before the fragments
    /// could be streamed out this kernel could not be lowered at all.
    func testLargeBlockShapesLowerOnlyBecauseTheEpilogueStreams() throws {
        let ir = DotFixtures.tutorial(blockM: 128, blockN: 128, blockK: 16)
        let options = MetalCompiler.Options(numSimdgroups: 16)
        let source = try MetalCompiler.emitMSL(ttir: ir, options: options)
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[4096];"), source)

        var withoutPanels = options
        withoutPanels.dotEpiloguePanel = 0
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: withoutPanels)) {
            XCTAssertTrue("\($0)".contains("threadgroup memory"), "\($0)")
        }
    }

    /// ...and it computes the same thing, at sizes that divide neither the block
    /// shape nor the 8x8 fragment, with and without panels where both lower.
    func testPanelledEpilogueMatchesTheCPUReference() throws {
        try skipWithoutMetal()
        for (m, n, k, blockM, blockN, blockK, warps) in [
            (129, 257, 65, 128, 128, 16, 16),  // only lowers with panels
            (37, 41, 43, 128, 128, 16, 16),
            (129, 257, 65, 64, 64, 16, 8),
            (100, 100, 33, 128, 64, 16, 8),
        ] {
            let a = matrix(m * k, seed: 3)
            let b = matrix(k * n, seed: 11)
            let run = try GPU.run(
                ir: DotFixtures.tutorial(blockM: blockM, blockN: blockN, blockK: blockK),
                grid: (GPU.cdiv(m, blockM), GPU.cdiv(n, blockN), 1), args: [
                    .floats(a), .floats(b), .output(count: m * n),
                    .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                    .int32(Int32(k)), .int32(1),
                    .int32(Int32(n)), .int32(1),
                    .int32(Int32(n)), .int32(1),
                ], numSimdgroups: warps)
            assertClose(
                GPU.read(run.outputs[0], Float.self, m * n),
                reference(a, b, m: m, n: n, k: k, lda: k, ldb: n),
                tolerance: 1e-5, "\(m)x\(n)x\(k) BLOCK=\(blockM)x\(blockN)x\(blockK)/w\(warps)")
        }
    }

    /// Without residency there is no dead window to share, so a stand-alone dot
    /// keeps its accumulator in a tile of its own.
    func testStandaloneDotDoesNotShareItsAccumulatorTile() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.singleTile(m: 32, n: 32, k: 32), options: .init())
        XCTAssertEqual(occurrences(of: "threadgroup float tm_dot", in: source), 3)
        XCTAssertFalse(source.contains("threadgroup float *tm_dot"), source)
    }

    private func occurrences(of needle: String, in haystack: String) -> Int {
        haystack.components(separatedBy: needle).count - 1
    }

    /// A carried pointer is not carried: it is rebuilt from the induction
    /// variable, because the dot's staging loops recompute their operands.
    func testCarriedPointersAreStrengthReduced() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16), options: .init())
        XCTAssertTrue(
            source.contains("int tm_advance0 = ((varg12 - vc0_i32) / vc1_i32) * v39;"), source)
        XCTAssertTrue(source.contains("device float *varg14 = v23 + tm_advance0;"), source)
    }

    // MARK: - The matmul tutorial, end to end

    func testTutorialMatmulAgainstCPUReference() throws {
        try skipWithoutMetal()
        for (m, n, k, blockM, blockN, blockK) in [
            (64, 64, 64, 32, 32, 32),
            (129, 257, 65, 32, 32, 16),  // ragged in all three dimensions
            (1, 1, 1, 16, 16, 16),
            (37, 8, 96, 16, 32, 32),
            (100, 100, 3, 32, 64, 16),
            (65, 33, 129, 64, 32, 16),
        ] {
            let a = matrix(m * k, seed: 3)
            let b = matrix(k * n, seed: 11)
            let run = try GPU.run(
                ir: DotFixtures.tutorial(blockM: blockM, blockN: blockN, blockK: blockK),
                grid: (GPU.cdiv(m, blockM), GPU.cdiv(n, blockN), 1),
                args: [
                    .floats(a), .floats(b), .output(count: m * n),
                    .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                    .int32(Int32(k)), .int32(1),  // stride_am, stride_ak
                    .int32(Int32(n)), .int32(1),  // stride_bk, stride_bn
                    .int32(Int32(n)), .int32(1),  // stride_cm, stride_cn
                ])
            assertClose(
                GPU.read(run.outputs[0], Float.self, m * n),
                reference(a, b, m: m, n: n, k: k, lda: k, ldb: n),
                tolerance: 1e-5, "\(m)x\(n)x\(k) BLOCK=\(blockM)x\(blockN)x\(blockK)")
        }
    }

    /// The vector staging path decides *at runtime* whether a run of four columns
    /// is contiguous, aligned and unmasked, so the cases that matter are the ones
    /// where it is not: row strides that put every other row out of alignment,
    /// and block shapes whose tiles are padded out to whole fragments (which is
    /// where the fast path must not so much as form an address).
    func testVectorStagingFallsBackOnAwkwardStridesAndPaddedTiles() throws {
        try skipWithoutMetal()
        for (m, n, k, blockM, blockN, blockK, lda, ldb) in [
            (70, 36, 40, 128, 36, 20, 40, 36),  // padded tiles: 36 -> 40, 20 -> 24
            (33, 40, 27, 32, 40, 20, 30, 43),  // rows misaligned for a float4 read
            (64, 64, 64, 32, 64, 32, 65, 67),  // odd strides, unpadded tiles
        ] {
            let a = matrix(m * lda, seed: 5)
            let b = matrix(k * ldb, seed: 7)
            let run = try GPU.run(
                ir: DotFixtures.tutorial(blockM: blockM, blockN: blockN, blockK: blockK),
                grid: (GPU.cdiv(m, blockM), GPU.cdiv(n, blockN), 1),
                args: [
                    .floats(a), .floats(b), .output(count: m * n),
                    .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                    .int32(Int32(lda)), .int32(1),  // stride_am, stride_ak
                    .int32(Int32(ldb)), .int32(1),  // stride_bk, stride_bn
                    .int32(Int32(n)), .int32(1),  // stride_cm, stride_cn
                ])
            assertClose(
                GPU.read(run.outputs[0], Float.self, m * n),
                reference(a, b, m: m, n: n, k: k, lda: lda, ldb: ldb),
                tolerance: 1e-5,
                "\(m)x\(n)x\(k) BLOCK=\(blockM)x\(blockN)x\(blockK) lda=\(lda) ldb=\(ldb)")
        }
    }

    /// ...and the case the runtime check exists for: a *non-unit* innermost
    /// stride, where a run of four columns is four strided elements and the
    /// vector path must not be taken at all.
    func testVectorStagingRefusesANonUnitInnermostStride() throws {
        try skipWithoutMetal()
        let (m, n, k) = (40, 40, 40)
        let (columnStride, lda, ldb) = (3, 40 * 3, 40 * 3)
        let a = matrix(m * lda, seed: 5)
        let b = matrix(k * ldb, seed: 7)
        var expected = [Float](repeating: 0, count: m * n)
        for row in 0..<m {
            for step in 0..<k {
                let left = a[row * lda + step * columnStride]
                for column in 0..<n {
                    expected[row * n + column] += left * b[step * ldb + column * columnStride]
                }
            }
        }
        let run = try GPU.run(
            ir: DotFixtures.tutorial(blockM: 32, blockN: 40, blockK: 20),
            grid: (GPU.cdiv(m, 32), GPU.cdiv(n, 40), 1),
            args: [
                .floats(a), .floats(b), .output(count: m * n),
                .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                .int32(Int32(lda)), .int32(Int32(columnStride)),  // stride_am, stride_ak
                .int32(Int32(ldb)), .int32(Int32(columnStride)),  // stride_bk, stride_bn
                .int32(Int32(n)), .int32(1),  // stride_cm, stride_cn
            ])
        assertClose(
            GPU.read(run.outputs[0], Float.self, m * n), expected, tolerance: 1e-5,
            "\(m)x\(n)x\(k) with an innermost stride of \(columnStride)")
    }

    /// f16 operands accumulated in f32 — the shape that matters for ML, and the
    /// one that makes the emitter mix `simdgroup_half8x8` with a float
    /// accumulator fragment.
    func testHalfOperandsAccumulateInFloat() throws {
        try skipWithoutMetal()
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 32, element: "f16"),
            options: .init())
        // The f32 accumulator's tile doubles as the arena the half operand tiles
        // stage into — 1024 floats holds either the accumulator or both operands.
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[1024];"), source)
        XCTAssertTrue(
            source.contains("threadgroup half *tm_dot_a1 = (threadgroup half *)(tm_dot_c0);"),
            source)
        XCTAssertTrue(
            source.contains(
                "threadgroup half *tm_dot_b2 = (threadgroup half *)(tm_dot_c0 + 512);"), source)
        // Half operand fragments feeding a float accumulator: a 2x1 block, so two
        // A fragments and the one B fragment they share.
        XCTAssertTrue(
            source.contains("simdgroup_half8x8 tm_a0_0_0, tm_a0_0_1, tm_b0_0_0;"), source)
        XCTAssertTrue(
            source.contains(
                "simdgroup_float8x8 tm_dot_c0_r_0_0_0 = "
                    + "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"), source)

        let (m, n, k) = (70, 66, 34)
        let a = (0..<(m * k)).map { Float16(Float($0 % 11) * 0.125 - 0.5) }
        let b = (0..<(k * n)).map { Float16(Float($0 % 7) * 0.25 - 0.75) }
        let run = try GPU.run(
            ir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 32, element: "f16"),
            grid: (GPU.cdiv(m, 32), GPU.cdiv(n, 32), 1),
            args: [
                .halves(a), .halves(b), .output(count: m * n, stride: 2),
                .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                .int32(Int32(k)), .int32(1), .int32(Int32(n)), .int32(1),
                .int32(Int32(n)), .int32(1),
            ])
        let expected = reference(
            a.map(Float.init), b.map(Float.init), m: m, n: n, k: k, lda: k, ldb: n)
        assertClose(
            GPU.read(run.outputs[0], Float16.self, m * n).map(Float.init), expected,
            tolerance: 5e-3, "f16 \(m)x\(n)x\(k)")
    }

    /// bf16 operands accumulating in f32 — the dtype ML training actually uses,
    /// and the one Triton treats as table stakes on CUDA.
    ///
    /// Metal has `bfloat` (MSL 3.1) and `simdgroup_matrix<bfloat, 8, 8>` in
    /// hardware on Apple silicon, so this is a real simdgroup matrix multiply
    /// rather than a widen-to-float-and-multiply.
    func testBFloat16OperandsAccumulateInFloat() throws {
        try skipWithoutMetal()
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 32, element: "bf16"),
            options: .init())
        XCTAssertTrue(source.contains("device bfloat *varg0"), source)
        XCTAssertTrue(
            source.contains("threadgroup bfloat *tm_dot_a1 = (threadgroup bfloat *)(tm_dot_c0);"),
            source)
        XCTAssertTrue(
            source.contains(
                "simdgroup_matrix<bfloat, 8, 8> tm_a0_0_0, tm_a0_0_1, tm_b0_0_0;"), source)
        // The accumulator is still f32: bf16 in, f32 across the contraction,
        // bf16 out.
        XCTAssertTrue(
            source.contains(
                "simdgroup_float8x8 tm_dot_c0_r_0_0_0 = "
                    + "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"), source)

        // Against an f32 CPU reference computed from the *rounded* inputs, so
        // the only error being measured is the accumulation, not the input
        // rounding. bf16 keeps 8 mantissa bits, hence the 2^-8 tolerance.
        let (m, n, k) = (70, 66, 34)
        let a = (0..<(m * k)).map { Float($0 % 11) * 0.125 - 0.5 }
        let b = (0..<(k * n)).map { Float($0 % 7) * 0.25 - 0.75 }
        let run = try GPU.run(
            ir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 32, element: "bf16"),
            grid: (GPU.cdiv(m, 32), GPU.cdiv(n, 32), 1),
            args: [
                .bfloats(a), .bfloats(b), .output(count: m * n, stride: 2),
                .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                .int32(Int32(k)), .int32(1), .int32(Int32(n)), .int32(1),
                .int32(Int32(n)), .int32(1),
            ])
        let rounded: ([Float]) -> [Float] = { $0.map { BFloat16.decode(BFloat16.encode($0)) } }
        let expected = reference(
            rounded(a), rounded(b), m: m, n: n, k: k, lda: k, ldb: n)
        assertClose(
            GPU.read(run.outputs[0], UInt16.self, m * n).map(BFloat16.decode), expected,
            tolerance: 1.0 / 256, "bf16 \(m)x\(n)x\(k)")
    }

    /// bf16 is not f16, and the emitter must not let one stand in for the other:
    /// they are the same width, so neither `truncf` nor `extf` connects them and
    /// a dot cannot accumulate one into the other.
    func testBFloat16IsNotInterchangeableWithFloat16() throws {
        XCTAssertEqual(try MSLEmitter.metalTypeName(.bfloat), "bfloat")
        XCTAssertEqual(try MSLEmitter.metalTypeName(.float(width: 16)), "half")
        XCTAssertNotEqual(TMType.bfloat, TMType.float(width: 16))
        XCTAssertEqual(TMType.bfloat.scalarByteWidth, 2)
        XCTAssertEqual("\(TMType.bfloat)", "bf16")

        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<f16>) {
                %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
                %1 = tt.load %0 : tensor<8xbf16>
                %2 = arith.extf %1 : tensor<8xbf16> to tensor<8xf16>
                %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>>
                tt.store %3, %2 : tensor<8x!tt.ptr<f16>>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("must widen"), "\($0)")
        }
    }

    /// Reductions fold within a simdgroup, so a dot kernel must still be correct
    /// at every threadgroup size the backend will report.
    func testMatmulAtEveryNumWarps() throws {
        try skipWithoutMetal()
        let (m, n, k) = (48, 40, 40)
        let a = matrix(m * k, seed: 2)
        let b = matrix(k * n, seed: 9)
        let expected = reference(a, b, m: m, n: n, k: k, lda: k, ldb: n)
        for simdgroups in [1, 2, 4, 8, 16, 32] {
            let run = try GPU.run(
                ir: DotFixtures.tutorial(blockM: 16, blockN: 16, blockK: 16),
                grid: (GPU.cdiv(m, 16), GPU.cdiv(n, 16), 1),
                args: [
                    .floats(a), .floats(b), .output(count: m * n),
                    .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                    .int32(Int32(k)), .int32(1), .int32(Int32(n)), .int32(1),
                    .int32(Int32(n)), .int32(1),
                ], numSimdgroups: simdgroups)
            assertClose(
                GPU.read(run.outputs[0], Float.self, m * n), expected, tolerance: 1e-5,
                "num_warps=\(simdgroups)")
        }
    }

    /// Masked stores must leave the padding between rows alone, exactly as they
    /// do for elementwise kernels.
    func testMatmulDoesNotWriteOutsideTheLogicalMatrix() throws {
        try skipWithoutMetal()
        let (m, n, k, stride) = (17, 19, 23, 24)
        let a = matrix(m * k, seed: 4)
        let b = matrix(k * n, seed: 6)
        let sentinel = Float(-1234)
        let output = try MetalRuntime.makeBuffer(length: m * stride * 4)
        let pointer = output.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<(m * stride) { pointer[index] = sentinel }

        let emission = try MetalCompiler.emit(
            ttir: DotFixtures.tutorial(blockM: 16, blockN: 16, blockK: 16), options: .init())
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "matmul_kernel")
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSizeMake(GPU.cdiv(m, 16), GPU.cdiv(n, 16), 1),
            threadsPerThreadgroup: emission.kernels[0].threadsPerThreadgroup,
            arguments: [
                .buffer(try GPU.upload(a)), .buffer(try GPU.upload(b)), .buffer(output),
                .int32(Int32(m)), .int32(Int32(n)), .int32(Int32(k)),
                .int32(Int32(k)), .int32(1), .int32(Int32(n)), .int32(1),
                .int32(Int32(stride)), .int32(1),
            ])

        let actual = GPU.read(output, Float.self, m * stride)
        let expected = reference(a, b, m: m, n: n, k: k, lda: k, ldb: n)
        for row in 0..<m {
            for column in 0..<stride {
                if column < n {
                    XCTAssertEqual(
                        actual[row * stride + column], expected[row * n + column], accuracy: 1e-4,
                        "at [\(row), \(column)]")
                } else {
                    XCTAssertEqual(actual[row * stride + column], sentinel, "at [\(row), \(column)]")
                }
            }
        }
    }

    func testDotFixturesCompileOnDevice() throws {
        try skipWithoutMetal()
        let fixtures: [(String, String)] = [
            ("single_tile", DotFixtures.singleTile(m: 16, n: 16, k: 16)),
            ("single_tile_ragged", DotFixtures.singleTile(m: 12, n: 20, k: 12)),
            ("tutorial_f32", DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16)),
            ("tutorial_f16", DotFixtures.tutorial(blockM: 64, blockN: 64, blockK: 32, element: "f16")),
        ]
        for (name, ir) in fixtures {
            let result = try MetalCompiler.emit(ttir: ir, options: .init())
            let library = try MetalCompiler.compileMSL(result.source)
            XCTAssertTrue(
                library.functionNames.contains(result.kernels[0].name),
                "\(name): library exposes \(library.functionNames)")
        }
    }

    // MARK: - Parsing

    func testParsesBothAttributeSpellings() throws {
        for suffix in [
            "", ", inputPrecision = tf32", ", inputPrecision = ieee, maxNumImpreciseAcc = 0",
            " {allowTF32 = true}",
        ] {
            let ir = """
                module {
                  tt.func public @k(%arg0: !tt.ptr<f32>) {
                    %a = arith.constant dense<1.000000e+00> : tensor<16x8xf32>
                    %b = arith.constant dense<2.000000e+00> : tensor<8x16xf32>
                    %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                    %d = tt.dot %a, %b, %c\(suffix) : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
                    tt.return
                  }
                }
                """
            let function = try TritonIRParser.parse(ir).functions[0]
            guard case .dot(let dot) = function.body[3].kind else {
                return XCTFail("expected tt.dot for suffix '\(suffix)'")
            }
            XCTAssertEqual(dot.lhs, "a")
            XCTAssertEqual(dot.rhs, "b")
            XCTAssertEqual(dot.accumulator, "c")
            XCTAssertEqual(dot.lhsType, .tensor(shape: [16, 8], element: .float(width: 32)))
            XCTAssertEqual(dot.resultType, .tensor(shape: [16, 16], element: .float(width: 32)))
        }
    }

    // MARK: - Error paths

    private func expectError(_ ir: String, contains needle: String, _ line: UInt = #line) {
        do {
            _ = try MetalCompiler.emitMSL(ttir: ir, options: .init())
            XCTFail("expected an error containing '\(needle)'", line: line)
        } catch {
            let description = (error as? CoreError)?.description ?? "\(error)"
            XCTAssertTrue(
                description.contains(needle),
                "error '\(description)' does not mention '\(needle)'", line: line)
        }
    }

    private func dotKernel(_ body: String) -> String {
        "module {\n  tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: i32) {\n\(body)\n    tt.return\n  }\n}"
    }

    func testTwoOperandDotIsRejected() {
        expectError(
            dotKernel(
                """
                    %a = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
                    %d = tt.dot %a, %a : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
                """), contains: "tt.dot takes three operands")
    }

    func testMismatchedContractionIsReported() {
        expectError(
            dotKernel(
                """
                    %a = arith.constant dense<1.000000e+00> : tensor<16x8xf32>
                    %b = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
                    %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                    %d = tt.dot %a, %b, %c : tensor<16x8xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
                """), contains: "contracted dimensions differ")
    }

    func testIntegerDotIsNamedAsUnsupported() {
        expectError(
            dotKernel(
                """
                    %a = arith.constant dense<1> : tensor<16x8xi32>
                    %b = arith.constant dense<1> : tensor<8x16xi32>
                    %c = arith.constant dense<0> : tensor<16x16xi32>
                    %d = tt.dot %a, %b, %c : tensor<16x8xi32> * tensor<8x16xi32> -> tensor<16x16xi32>
                """), contains: "Metal's simdgroup matrices are half, bfloat or float")
    }

    func testDotInsideAnIfIsRejectedWithItsReason() {
        expectError(
            dotKernel(
                """
                    %a = arith.constant dense<1.000000e+00> : tensor<16x8xf32>
                    %b = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
                    %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                    %t = arith.constant true
                    %r = scf.if %t -> (tensor<16x16xf32>) {
                      %d = tt.dot %a, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
                      scf.yield %d : tensor<16x16xf32>
                    } else {
                      scf.yield %c : tensor<16x16xf32>
                    }
                """), contains: "tt.dot inside an scf.if is not lowered")
    }

    /// A value that spans a `tt.dot`'s contracted dimension *and* is materialised
    /// used to be refused. It no longer is: the axis simply becomes an ordinary
    /// iteration axis, walked by a nest of its own. FlashAttention needs exactly
    /// this — its `p` is both the second dot's left operand and the tensor the
    /// row sum reduces.
    func testAContractionSpaceValueMayAlsoBeMaterialised() throws {
        try skipWithoutMetal()
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                %a = arith.constant dense<1.500000e+00> : tensor<16x8xf32>
                %b = arith.constant dense<2.000000e+00> : tensor<8x16xf32>
                %d = tt.dot %a, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
                %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %k = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %mr = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
                %kc = tt.expand_dims %k {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
                %c8 = arith.constant dense<8> : tensor<16x1xi32>
                %row = arith.muli %mr, %c8 : tensor<16x1xi32>
                %rowb = tt.broadcast %row : tensor<16x1xi32> -> tensor<16x8xi32>
                %colb = tt.broadcast %kc : tensor<1x8xi32> -> tensor<16x8xi32>
                %off = arith.addi %rowb, %colb : tensor<16x8xi32>
                %p = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
                %q = tt.addptr %p, %off : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
                tt.store %q, %a : tensor<16x8x!tt.ptr<f32>>
                %cm = arith.constant dense<16> : tensor<16x1xi32>
                %nr = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %nc = tt.expand_dims %nr {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
                %crow = arith.muli %mr, %cm : tensor<16x1xi32>
                %crowb = tt.broadcast %crow : tensor<16x1xi32> -> tensor<16x16xi32>
                %ccolb = tt.broadcast %nc : tensor<1x16xi32> -> tensor<16x16xi32>
                %coff = arith.addi %crowb, %ccolb : tensor<16x16xi32>
                %cp = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
                %cq = tt.addptr %cp, %coff : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
                tt.store %cq, %d : tensor<16x16x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        // Three iteration axes now: M, N and the one the dot contracts over.
        let layout = try LayoutInference.compute(for: TritonIRParser.parse(ir).functions[0])
        XCTAssertEqual(layout.rank, 3)
        XCTAssertEqual(layout.contractions, [])

        let run = try GPU.run(ir: ir, args: [.output(count: 256), .output(count: 128)])
        assertClose(
            GPU.read(run.outputs[0], Float.self, 256),
            [Float](repeating: 1.5 * 2.0 * 8, count: 256), tolerance: 1e-5, "the dot's result")
        assertClose(
            GPU.read(run.outputs[1], Float.self, 128),
            [Float](repeating: 1.5, count: 128), tolerance: 1e-5, "the left operand, stored")
    }

    /// A tensor carried across a dot loop that is not the accumulator used to have
    /// nowhere to live. It is now spilled to a threadgroup tile and updated
    /// through a shadow (docs/ARCHITECTURE.md §Cross-lane regions).
    func testAnUnrelatedTensorCarriedAcrossADotLoopIsSpilled() throws {
        try skipWithoutMetal()
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: i32) {
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                %a = arith.constant dense<1.000000e+00> : tensor<16x8xf32>
                %b = arith.constant dense<2.000000e+00> : tensor<8x16xf32>
                %e = arith.constant dense<3.000000e+00> : tensor<16x16xf32>
                %r:2 = scf.for %i = %c0 to %arg1 step %c1 iter_args(%acc = %c, %other = %e)
                    -> (tensor<16x16xf32>, tensor<16x16xf32>) : i32 {
                  %d = tt.dot %a, %b, %acc : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
                  %n = arith.addf %other, %e : tensor<16x16xf32>
                  scf.yield %d, %n : tensor<16x16xf32>, tensor<16x16xf32>
                }
                %m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %n = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %mr = tt.expand_dims %m {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
                %nc = tt.expand_dims %n {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
                %c16 = arith.constant dense<16> : tensor<16x1xi32>
                %row = arith.muli %mr, %c16 : tensor<16x1xi32>
                %rowb = tt.broadcast %row : tensor<16x1xi32> -> tensor<16x16xi32>
                %colb = tt.broadcast %nc : tensor<1x16xi32> -> tensor<16x16xi32>
                %off = arith.addi %rowb, %colb : tensor<16x16xi32>
                %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
                %q = tt.addptr %p, %off : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
                tt.store %q, %r#1 : tensor<16x16x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        for trips in [0, 1, 5] {
            let run = try GPU.run(ir: ir, args: [.output(count: 256), .int32(Int32(trips))])
            assertClose(
                GPU.read(run.outputs[0], Float.self, 256),
                [Float](repeating: Float(3 * (trips + 1)), count: 256), tolerance: 1e-5,
                "\(trips) trips")
        }
    }

    func testThreadgroupMemoryOverrunIsReported() {
        expectError(
            DotFixtures.tutorial(blockM: 128, blockN: 128, blockK: 64),
            contains: "over Metal's 32768-byte limit")
    }
}
