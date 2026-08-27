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
        // 2x3 fragments over 4 simdgroups, one fragment per simdgroup per wave —
        // the smallest blocking, which is what measures fastest on Apple silicon.
        XCTAssertTrue(
            source.contains(
                "// tt.dot accumulator: 2x3 blocks of 1x1 fragments over 4 simdgroups, 2 waves"),
            source)
        XCTAssertTrue(source.contains("// tt.dot: 2 contraction steps, each 2 "
            + "multiply-accumulates off 4 operand loads"), source)
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
    /// threadgroup memory rather than in per-lane registers.
    func testAccumulatorLivesInThreadgroupMemoryAcrossTheLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16), options: .init())
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[1024];  // 32x32"), source)
        let loop = try XCTUnwrap(source.range(of: "for (int varg12 ="))
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        XCTAssertTrue(prologue.contains("tm_dot_c0[tm_i0 * 32u + tm_i1] = vcst;"), prologue)
        // Read back per lane after the loop, by indexing the tile.
        let epilogue = String(source[loop.upperBound...])
        XCTAssertTrue(epilogue.contains("float v51 = tm_dot_c0[tm_i0 * 32u + tm_i1] * vcone;"), epilogue)
    }

    /// ...but it does not make a round trip through that memory on every K step:
    /// the output fragments are loaded into simdgroup registers once, before the
    /// loop, and stored back once, after it.
    func testAccumulatorFragmentsStayInRegistersAcrossTheLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 64, blockN: 64, blockK: 32),
            options: .init(numSimdgroups: 16))
        let loop = try XCTUnwrap(source.range(of: "for (int varg12 ="))
        let close = try XCTUnwrap(source.range(of: "// tt.dot accumulator: back to threadgroup"))
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        let body = String(source[loop.upperBound..<close.lowerBound])
        let epilogue = String(source[close.lowerBound...])

        // 8x8 fragments over 16 simdgroups, one each per wave: four accumulator
        // registers, loaded before the loop and stored after it.
        XCTAssertTrue(
            prologue.contains(
                "// tt.dot accumulator: 8x8 blocks of 1x1 fragments over 16 simdgroups, 4 waves"),
            prologue)
        XCTAssertEqual(occurrences(of: "simdgroup_load(tm_dot_c0_r_", in: prologue), 4)
        XCTAssertEqual(occurrences(of: "simdgroup_store(tm_dot_c0_r_", in: epilogue), 4)

        // Nothing touches the accumulator tile inside the loop.
        XCTAssertEqual(occurrences(of: "simdgroup_load(tm_dot_c0_r_", in: body), 0)
        XCTAssertEqual(occurrences(of: "simdgroup_store(", in: body), 0)
        XCTAssertEqual(occurrences(of: "simdgroup_multiply_accumulate(", in: body), 4)
        XCTAssertEqual(occurrences(of: "simdgroup_load(", in: body), 8)
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
        XCTAssertTrue(source.contains("// tt.dot: 4 contraction steps, each 16 "
            + "multiply-accumulates off 12 operand loads"), source)
    }

    /// A register-resident accumulator's tile is dead for exactly as long as the
    /// operand tiles exist, so the two share storage. That is what lets a block
    /// shape whose accumulator alone fills Metal's 32KB budget still lower.
    func testAccumulatorTileDoublesAsTheStagingArena() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: DotFixtures.tutorial(blockM: 128, blockN: 64, blockK: 32),
            options: .init(numSimdgroups: 16))
        // 128x64 floats is the whole 32KB budget; the operands alias into it.
        XCTAssertTrue(source.contains("threadgroup float tm_dot_c0[8192];"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_a1 = tm_dot_c0;"), source)
        XCTAssertTrue(source.contains("threadgroup float *tm_dot_b2 = tm_dot_c0 + 4096;"), source)
        XCTAssertEqual(occurrences(of: "threadgroup float tm_dot", in: source), 1)

        // The sharing is only safe because a barrier separates the last read of
        // the accumulator tile (into registers) from the first write of the
        // operand tiles over the top of it.
        let load = try XCTUnwrap(source.range(of: "simdgroup_load(tm_dot_c0_r_"))
        let stage = try XCTUnwrap(source.range(of: "// tt.dot: stage A"))
        let between = String(source[load.upperBound..<stage.lowerBound])
        XCTAssertTrue(between.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), between)
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
        // Half operand fragments feeding a float accumulator, 2x2 per simdgroup.
        XCTAssertTrue(source.contains("simdgroup_half8x8 tm_a0_0_0, tm_b0_0_0;"), source)
        XCTAssertTrue(source.contains("simdgroup_float8x8 tm_dot_c0_r_0_0_0;"), source)

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
                """), contains: "Metal's simdgroup matrices are half or float")
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

    /// A contraction-space value that something other than the dot wants is the
    /// one thing the deferral model cannot serve.
    func testUsingAContractionSpaceValueOutsideTheDotIsReported() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                %a = arith.constant dense<1.000000e+00> : tensor<16x8xf32>
                %b = arith.constant dense<2.000000e+00> : tensor<8x16xf32>
                %d = tt.dot %a, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
                %p = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
                tt.store %p, %a : tensor<16x8x!tt.ptr<f32>>
                tt.return
              }
            }
            """, contains: "the contraction dimension of a tt.dot")
    }

    /// A tensor carried across a dot loop that is not the accumulator has nowhere
    /// to live: it is neither a tile nor rebuildable from the induction variable.
    func testCarryingAnUnrelatedTensorAcrossADotLoopIsReported() {
        expectError(
            """
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
                %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
                tt.store %p, %r#1 : tensor<16x16x!tt.ptr<f32>>
                tt.return
              }
            }
            """, contains: "the only tensor such a loop can carry is the accumulator")
    }

    func testThreadgroupMemoryOverrunIsReported() {
        expectError(
            DotFixtures.tutorial(blockM: 128, blockN: 128, blockK: 64),
            contains: "over Metal's 32768-byte limit")
    }
}
