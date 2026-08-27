import Foundation
import XCTest

@testable import TritonMetalCore

/// Milestone: `tt.reduce` lowered to `simd_shuffle_down` within a simdgroup plus
/// threadgroup memory across simdgroups, and the fused-softmax integration kernel.
final class ReductionTests: XCTestCase {
    override func setUpWithError() throws { try skipWithoutMetal() }

    // MARK: - 1-D reductions

    private static let combiners: [(name: String, mnemonic: String, identity: String)] = [
        ("add", "arith.addf", "0.000000e+00"),
        ("max", "arith.maxnumf", "0xFF800000"),
        ("min", "arith.minnumf", "0x7F800000"),
    ]

    func testOneDimensionalReductionsMatchCPU() throws {
        let block = 128
        let n = 1_000  // 7.8 blocks: the last program is mostly masked off
        let input = (0..<n).map { Float(($0 * 37) % 211) - 100 }
        let programs = GPU.cdiv(n, block)

        for combiner in ReductionTests.combiners {
            let result = try GPU.run(
                ir: AdvancedFixtures.reduce1D(
                    combiner: combiner.mnemonic, identity: combiner.identity),
                grid: (programs, 1, 1),
                args: [.floats(input), .output(count: programs), .int32(Int32(n))])

            let expected = (0..<programs).map { program -> Float in
                let slice = input[(program * block)..<min(n, (program + 1) * block)]
                switch combiner.name {
                case "add": return slice.reduce(0, +)
                case "max": return slice.max() ?? -.infinity
                default: return slice.min() ?? .infinity
                }
            }
            assertClose(
                GPU.read(result.outputs[0], Float.self, programs), expected, tolerance: 1e-6,
                combiner.name)
        }
    }

    /// The cross-simdgroup half of the reduction only runs when the threadgroup
    /// holds more than one simdgroup, so sweep `num_warps`.
    func testReductionIsIndependentOfSimdgroupCount() throws {
        let block = 128
        let n = 512
        let input = (0..<n).map { Float($0) * 0.5 }
        let programs = GPU.cdiv(n, block)
        let expected = (0..<programs).map { program in
            input[(program * block)..<((program + 1) * block)].reduce(0, +)
        }
        for simdgroups in [1, 2, 4, 8, 16, 32] {
            let result = try GPU.run(
                ir: AdvancedFixtures.reduce1D(combiner: "arith.addf", identity: "0.000000e+00"),
                grid: (programs, 1, 1),
                args: [.floats(input), .output(count: programs), .int32(Int32(n))],
                numSimdgroups: simdgroups)
            assertClose(
                GPU.read(result.outputs[0], Float.self, programs), expected, tolerance: 1e-6,
                "num_warps=\(simdgroups)")
        }
    }

    // MARK: - Row-wise reductions over rank-2 tiles

    func testRowWiseSumOverARankTwoTile() throws {
        let (rows, columns, stride) = (37, 50, 64)
        let input = (0..<(rows * stride)).map { Float($0 % 71) * 0.25 - 5 }
        let result = try GPU.run(
            ir: AdvancedFixtures.rowSum,
            grid: (GPU.cdiv(rows, 8), 1, 1),
            args: [
                .floats(input), .output(count: rows), .int32(Int32(rows)), .int32(Int32(columns)),
                .int32(Int32(stride)),
            ])
        let expected = (0..<rows).map { row in
            (0..<columns).reduce(Float(0)) { $0 + input[row * stride + $1] }
        }
        assertClose(GPU.read(result.outputs[0], Float.self, rows), expected, tolerance: 1e-6)
    }

    /// A row-wise reduction produces a value indexed by the *outer* dimension, so
    /// it must be computed once per row, between the two loops.
    func testRowWiseReductionSitsBetweenTheLoops() throws {
        let source = try MetalCompiler.emitMSL(ttir: AdvancedFixtures.rowSum, options: .init())
        let outer = try XCTUnwrap(source.range(of: "for (uint tm_i0 = 0u;"))
        let reduce = try XCTUnwrap(source.range(of: "// tt.reduce (add)"))
        let inner = try XCTUnwrap(source.range(of: "for (uint tm_i1 = tm_thread_id.x;"))
        XCTAssertLessThan(outer.lowerBound, inner.lowerBound)
        XCTAssertLessThan(inner.lowerBound, reduce.lowerBound)
        XCTAssertTrue(source.contains("simd_shuffle_down"), source)
        XCTAssertTrue(source.contains("threadgroup float tm_scratch0[32];"), source)
        XCTAssertTrue(
            source.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), source)
    }

    // MARK: - Softmax (the integration kernel)

    func testFusedSoftmaxMatchesCPU() throws {
        for (rows, columns, block) in [(1, 128, 128), (17, 100, 128), (64, 1, 1024), (5, 781, 1024)] {
            let stride = columns + 3  // padded rows, as torch tensors often are
            let input = (0..<(rows * stride)).map { Float(($0 * 31) % 97) * 0.35 - 15 }
            let result = try GPU.run(
                ir: AdvancedFixtures.softmax(block: block),
                grid: (rows, 1, 1),
                args: [
                    .floats(input), .output(count: rows * stride), .int32(Int32(stride)),
                    .int32(Int32(stride)), .int32(Int32(columns)),
                ])

            var expected = [Float](repeating: 0, count: rows * stride)
            for row in 0..<rows {
                let slice = (0..<columns).map { input[row * stride + $0] }
                let maximum = slice.max() ?? 0
                let exponentials = slice.map { exp($0 - maximum) }
                let total = exponentials.reduce(0, +)
                for column in 0..<columns {
                    expected[row * stride + column] = exponentials[column] / total
                }
            }
            assertClose(
                GPU.read(result.outputs[0], Float.self, rows * stride), expected, tolerance: 1e-6,
                "\(rows)x\(columns), BLOCK=\(block)")
        }
    }

    /// Every row must sum to 1 — the property the numerically-stable form exists
    /// to preserve, checked on inputs large enough to overflow a naive `exp`.
    func testSoftmaxIsNumericallyStableOnLargeInputs() throws {
        let (rows, columns, block) = (8, 200, 256)
        let input = (0..<(rows * columns)).map { Float(($0 % 200)) * 500 }  // up to 1e5
        let result = try GPU.run(
            ir: AdvancedFixtures.softmax(block: block),
            grid: (rows, 1, 1),
            args: [
                .floats(input), .output(count: rows * columns), .int32(Int32(columns)),
                .int32(Int32(columns)), .int32(Int32(columns)),
            ])
        let output = GPU.read(result.outputs[0], Float.self, rows * columns)
        for row in 0..<rows {
            let total = (0..<columns).reduce(Float(0)) { $0 + output[row * columns + $1] }
            XCTAssertEqual(total, 1, accuracy: 1e-5, "row \(row) sums to \(total)")
        }
    }

    /// Softmax recomputes the loaded row in each of its three lane loops, since a
    /// reduction closes the loop that produced it. Three loops, one load each.
    func testSoftmaxRematerialisesItsRowInEachLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AdvancedFixtures.softmax(block: 128), options: .init())
        XCTAssertEqual(
            source.components(separatedBy: "for (uint tm_i0 = tm_thread_id.x;").count - 1, 3,
            source)
        XCTAssertEqual(
            source.components(separatedBy: "// tt.reduce (").count - 1, 2,
            "two reductions, each closing and reopening the lane loop")
        XCTAssertEqual(
            source.components(separatedBy: "threadgroup float tm_scratch").count - 1, 2, source)
        XCTAssertTrue(source.contains("float vcst = -INFINITY;"), source)
        XCTAssertTrue(source.contains("precise::exp("), source)
    }

    // MARK: - Reductions inside a loop

    /// The online-softmax shape: the row is streamed in `BLOCK`-sized chunks and
    /// both reductions happen *inside* the `scf.for`, which is therefore lowered
    /// threadgroup-uniformly. Checked against the same reference as the
    /// single-pass kernel, at row lengths that are not multiples of the block.
    func testOnlineSoftmaxStreamsTheRowThroughAnScfFor() throws {
        try skipWithoutMetal()
        for (block, rows, columns, stride) in [
            (64, 4, 200, 208), (32, 3, 32, 32), (128, 2, 5, 8), (16, 5, 129, 129),
        ] {
            let input = (0..<(rows * stride)).map { Float($0 % 37) * 0.5 - 9 }
            let run = try GPU.run(
                ir: AdvancedFixtures.onlineSoftmax(block: block), grid: (rows, 1, 1),
                args: [
                    .floats(input), .output(count: rows * stride), .int32(Int32(stride)),
                    .int32(Int32(stride)), .int32(Int32(columns)),
                ])
            let actual = GPU.read(run.outputs[0], Float.self, rows * stride)
            for row in 0..<rows {
                let slice = Array(input[(row * stride)..<(row * stride + columns)])
                let maximum = slice.max() ?? 0
                let exponentials = slice.map { expf($0 - maximum) }
                let total = exponentials.reduce(0, +)
                assertClose(
                    Array(actual[(row * stride)..<(row * stride + columns)]),
                    exponentials.map { $0 / total }, tolerance: 1e-6,
                    "BLOCK=\(block), row \(row) of \(columns)")
            }
        }
    }

    /// The reduction closes and reopens the lane loop *inside* the loop body, and
    /// the running maximum and sum are carried as plain scalars.
    func testOnlineSoftmaxLowersItsReductionsInsideTheLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AdvancedFixtures.onlineSoftmax(block: 64), options: .init())
        let loop = try XCTUnwrap(source.range(of: "for (int vj ="))
        let body = String(source[loop.upperBound...])
        XCTAssertEqual(
            body.components(separatedBy: "// tt.reduce (").count - 1, 2,
            "both reductions belong inside the loop body")
        XCTAssertTrue(body.contains("vm = v29;"), body)
        XCTAssertTrue(body.contains("vl = v37;"), body)
        // The loop itself is threadgroup-uniform: it sits outside the lane loops.
        let prologue = String(source[source.startIndex..<loop.lowerBound])
        XCTAssertFalse(prologue.contains("for (uint tm_i0 = tm_thread_id.x"), prologue)
    }

    /// A per-lane tensor cannot be carried across a loop that has to be
    /// threadgroup-uniform — the precise obstacle, reported by name.
    func testCarryingATensorAcrossAnInLoopReductionIsRefused() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: i32) {
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
                %r = scf.for %i = %c0 to %arg1 step %c1 iter_args(%acc = %cst)
                    -> (tensor<64xf32>) : i32 {
                  %20 = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
                  ^bb0(%a: f32, %b: f32):
                    %21 = arith.addf %a, %b : f32
                    tt.reduce.return %21 : f32
                  }) : (tensor<64xf32>) -> f32
                  %22 = tt.splat %20 : f32 -> tensor<64xf32>
                  %23 = arith.addf %acc, %22 : tensor<64xf32>
                  scf.yield %23 : tensor<64xf32>
                }
                %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %q = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %s = tt.addptr %p, %q : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %s, %r : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue(
                "\($0)".contains("carries the tensor '%acc' across a cross-lane operation"), "\($0)")
        }
    }

    // MARK: - Error paths

    func testReduceOverANonInnermostAxisIsRejected() {
        let ir = AdvancedFixtures.rowSum.replacingOccurrences(
            of: "<{axis = 1 : i32}>", with: "<{axis = 0 : i32}>"
        ).replacingOccurrences(of: "-> tensor<8xf32>", with: "-> tensor<64xf32>")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("only the last axis"), "\($0)")
        }
    }

    func testUnsupportedCombinerNamesTheOp() {
        let ir = AdvancedFixtures.reduce1D(combiner: "arith.mulf", identity: "1.000000e+00")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("tt.reduce combining with arith.mulf"), "\($0)")
        }
    }

    func testPrettyReduceSpellingIsRejectedWithGuidance() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.reduce %0 {axis = 0 : i32} : tensor<16xi32> -> i32
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("carries a combine region"), "\($0)")
        }
    }

    /// Values live across a reduction are recomputed, which is only sound while the
    /// recomputed statements are pure. A store in the way must be refused, not
    /// silently duplicated.
    func testReduceAfterAStoreIsRejectedRatherThanDuplicatingIt() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
                %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                %3 = tt.load %2 : tensor<64xf32>
                tt.store %2, %3 : tensor<64x!tt.ptr<f32>>
                %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
                ^bb0(%a: f32, %b: f32):
                  %5 = arith.addf %a, %b : f32
                  tt.reduce.return %5 : f32
                }) : (tensor<64xf32>) -> f32
                %6 = tt.splat %4 : f32 -> tensor<64xf32>
                %7 = arith.addf %3, %6 : tensor<64xf32>
                tt.store %2, %7 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("a tt.store cannot be recomputed"), "\($0)")
        }
    }

    func testMultiOperandReduceIsRejected() {
        let ir = AdvancedFixtures.reduce1D(combiner: "arith.addf", identity: "0.000000e+00")
            .replacingOccurrences(of: "\"tt.reduce\"(%9)", with: "\"tt.reduce\"(%9, %9)")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("argmax"), "\($0)")
        }
    }
}
