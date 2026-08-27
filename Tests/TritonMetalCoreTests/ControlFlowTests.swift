import Foundation
import XCTest

@testable import TritonMetalCore

/// Milestone: `scf.for` / `scf.if` with `iter_args` and results.
///
/// Both regions lower *inside* the block loops: an `scf.for` whose carried values
/// are tensors becomes a per-lane loop, so each lane carries its own accumulator.
/// That is what makes tensor `iter_args` work without spilling to threadgroup
/// memory; the cost is that cross-lane ops inside a region are rejected.
final class ControlFlowTests: XCTestCase {
    override func setUpWithError() throws { try skipWithoutMetal() }

    /// Each program accumulates a BLOCK-wide vector over a strided walk of the
    /// whole array, then stores its partial. Verified against the same reduction
    /// done on the CPU.
    func testStridedSumAccumulatesAcrossLoopIterations() throws {
        let block = 64
        let programs = 5
        let n = 1_337  // not a multiple of block * programs
        let chunks = GPU.cdiv(n, block * programs)
        let input = (0..<n).map { Float($0 % 61) * 0.5 - 7 }

        let result = try GPU.run(
            ir: AdvancedFixtures.stridedSum,
            grid: (programs, 1, 1),
            args: [
                .floats(input), .output(count: block * programs), .int32(Int32(n)),
                .int32(Int32(chunks)),
            ])

        var expected = [Float](repeating: 0, count: block * programs)
        for program in 0..<programs {
            for lane in 0..<block {
                var total: Float = 0
                for chunk in 0..<chunks {
                    let index = program * block + lane + chunk * block * programs
                    if index < n { total += input[index] }
                }
                expected[program * block + lane] = total
            }
        }
        assertClose(
            GPU.read(result.outputs[0], Float.self, block * programs), expected, tolerance: 1e-6)
    }

    /// The loop's trip count is a runtime scalar, so zero iterations must leave the
    /// accumulator at its initial value.
    func testLoopWithZeroTripCountYieldsTheInitialValue() throws {
        let block = 64
        let result = try GPU.run(
            ir: AdvancedFixtures.stridedSum,
            grid: (1, 1, 1),
            args: [.floats([1, 2, 3, 4]), .output(count: block), .int32(4), .int32(0)])
        XCTAssertEqual(
            GPU.read(result.outputs[0], Float.self, block), [Float](repeating: 0, count: block))
    }

    /// `%r:2 = scf.for ... -> (i32, i32)` with `%r#0` / `%r#1` result selectors,
    /// in a kernel with no tensors at all.
    func testMultipleLoopResults() throws {
        let iterations = 10
        let result = try GPU.run(
            ir: AdvancedFixtures.loopTwoResults,
            grid: (1, 1, 1),
            args: [.output(count: 2), .int32(Int32(iterations))])
        XCTAssertEqual(
            GPU.read(result.outputs[0], Int32.self, 2),
            [Int32((0..<iterations).reduce(0, +)), Int32(1 << iterations)])
        XCTAssertEqual(result.kernel.blockShape, [], "a scalar-only kernel has no block")
        XCTAssertEqual(result.kernel.threadsPerThreadgroup, 1)
    }

    /// A program-uniform `scf.if` yielding a tensor.
    func testConditionalKernelPicksTheRightBranchPerProgram() throws {
        let block = 64
        let n = 300
        let input = (0..<n).map { Float($0) * 0.25 - 10 }
        let result = try GPU.run(
            ir: AdvancedFixtures.conditional,
            grid: (GPU.cdiv(n, block), 1, 1),
            args: [.floats(input), .output(count: n), .int32(Int32(n))])
        let expected = (0..<n).map { index -> Float in
            (index / block) % 2 == 0 ? input[index] * 2 : -input[index]
        }
        XCTAssertEqual(GPU.read(result.outputs[0], Float.self, n), expected)
    }

    // MARK: - Emitted shape

    func testLoopBodyIsEmittedInsideThePerLaneLoop() throws {
        let source = try MetalCompiler.emitMSL(ttir: AdvancedFixtures.stridedSum, options: .init())
        let laneLoop = try XCTUnwrap(source.range(of: "for (uint tm_i0"))
        let scfLoop = try XCTUnwrap(source.range(of: "for (int varg5"))
        XCTAssertLessThan(
            laneLoop.lowerBound, scfLoop.lowerBound,
            "the scf.for must nest inside the lane loop so each lane carries its own value")
        XCTAssertTrue(source.contains("varg6 = v27;"), source)
    }

    func testIfElseIsEmittedAsAConditionalAssignment() throws {
        let source = try MetalCompiler.emitMSL(ttir: AdvancedFixtures.conditional, options: .init())
        XCTAssertTrue(source.contains("float v12;"), source)
        XCTAssertTrue(source.contains("if (v11) {"), source)
        XCTAssertTrue(source.contains("else {"), source)
        XCTAssertTrue(source.contains("v12 = v20;"), source)
        XCTAssertTrue(source.contains("v12 = v21;"), source)
    }

    // MARK: - Error paths

    /// An `scf.for` gets hoisted to a threadgroup-uniform level when it contains a
    /// `tt.reduce` (see `ReductionTests.testOnlineSoftmax...`), but an `scf.if`
    /// cannot: it is per-lane by construction, and splitting its region's lane
    /// loop is exactly what a reduction would need.
    func testReduceInsideAnIfIsRejectedPrecisely() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
                %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
                %c0_i32 = arith.constant 0 : i32
                %0 = arith.cmpi slt, %arg2, %c0_i32 : i32
                %1 = scf.if %0 -> (tensor<64xf32>) {
                  %10 = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
                  ^bb0(%a: f32, %b: f32):
                    %11 = arith.addf %a, %b : f32
                    tt.reduce.return %11 : f32
                  }) : (tensor<64xf32>) -> f32
                  %12 = tt.splat %10 : f32 -> tensor<64xf32>
                  scf.yield %12 : tensor<64xf32>
                } else {
                  scf.yield %cst : tensor<64xf32>
                }
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("inside an scf region"), "\($0)")
        }
    }

    func testYieldArityIsChecked() {
        let ir = AdvancedFixtures.loopTwoResults.replacingOccurrences(
            of: "scf.yield %10, %11 : i32, i32", with: "scf.yield %10 : i32")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("yields 1 values for 2"), "\($0)")
        }
    }

    func testResultCountMustMatchIterArgs() {
        let ir = AdvancedFixtures.loopTwoResults.replacingOccurrences(
            of: "%0:2 = scf.for", with: "%0:3 = scf.for")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("binds 3 results for 2 iter_args"), "\($0)")
        }
    }
}
