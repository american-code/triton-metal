import Foundation
import XCTest

@testable import TritonMetalCore

/// The pairwise-distinct block-size restriction, and its removal.
///
/// The emitter never identified an axis by its extent — `Layout.swift` unifies
/// axis *variables* — but Triton's CSE makes `tl.arange(0, BLOCK_M)` and
/// `tl.arange(0, BLOCK_N)` the same value when the two blocks are equal, and the
/// row and column axes then unify through it. `AxisCloning` gives each
/// `tt.expand_dims` its own copy of that arithmetic.
final class AxisCloningTests: XCTestCase {

    private func reference(_ a: [Float], _ b: [Float], _ n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: n * n)
        for row in 0..<n {
            for column in 0..<n {
                var total = 0.0
                for k in 0..<n { total += Double(a[row * n + k]) * Double(b[k * n + column]) }
                c[row * n + column] = Float(total)
            }
        }
        return c
    }

    /// The kernel real Triton emits at `BLOCK_M == BLOCK_N`, on the GPU, against
    /// a CPU reference. This is the case docs used to record as refused.
    func testEqualBlockSizesRunAndMatchTheCPU() throws {
        try skipWithoutMetal()
        let n = 128
        let a = (0..<(n * n)).map { Float(($0 * 37) % 23) / 23 - 0.5 }
        let b = (0..<(n * n)).map { Float(($0 * 17) % 29) / 29 - 0.5 }
        for (block, blockK) in [(64, 32), (32, 16), (64, 64), (32, 32)] {
            let run = try GPU.run(
                ir: DotFixtures.tutorialWithSharedRange(block: block, blockK: blockK),
                grid: (n / block, n / block, 1),
                args: [
                    .floats(a), .floats(b), .output(count: n * n),
                    .int32(Int32(n)), .int32(Int32(n)), .int32(Int32(n)),
                    .int32(Int32(n)), .int32(Int32(n)), .int32(Int32(n)),
                ], numSimdgroups: 8)
            assertClose(
                GPU.read(run.outputs[0], Float.self, n * n), reference(a, b, n),
                tolerance: 1e-5, "block \(block)x\(block)x\(blockK)")
        }
    }

    /// What the pass did: one `tt.make_range` becomes two, and the two offset
    /// chains stop sharing an axis variable.
    func testTheSharedRangeIsClonedPerExpansion() throws {
        let ir = DotFixtures.tutorialWithSharedRange(block: 64, blockK: 32)
        let function = try TritonIRParser.parse(ir).functions[0]
        // Two: the shared M/N range, and the contraction range, which every
        // matmul expands at both dimensions. Only the first is a real collapse;
        // cloning the second is harmless and the pass does not try to tell them
        // apart, because it only ever runs on a kernel that already failed.
        XCTAssertEqual(try LayoutInference.expansionConflicts(in: function).count, 2)

        let rewritten = AxisCloning.rewrite(function)
        var ranges = 0
        for instruction in rewritten.body {
            if case .makeRange(_, let type, _, _) = instruction.kind, type.shape == [64] {
                ranges += 1
            }
        }
        XCTAssertEqual(ranges, 2, "the shared tt.make_range should be duplicated")

        // And the layout now has three distinct axes with the right extents.
        let layout = try LayoutInference.compute(for: rewritten)
        XCTAssertEqual(layout.rank, 2)
        XCTAssertEqual(layout.shape, [64, 64])
        XCTAssertEqual(layout.contractions, [32])
    }

    /// The orphaned chain is removed rather than left to carry axis identity
    /// into the inference.
    func testTheOrphanedChainIsRemoved() throws {
        let function = try TritonIRParser.parse(
            DotFixtures.tutorialWithSharedRange(block: 64, blockK: 32)).functions[0]
        let rewritten = AxisCloning.rewrite(function)
        let names = Set(rewritten.body.flatMap { $0.kind.resultNames })
        XCTAssertFalse(names.contains("offs_bn_4"), "the rewritten expansion's old operand")
        XCTAssertFalse(names.contains("offs_bn_3"), "and the splat only it read")
    }

    /// A kernel that already lowers is never rewritten: the pass runs only after
    /// `LayoutInference` has refused one, so nothing that worked changes shape.
    func testKernelsThatAlreadyLowerAreNotRewritten() throws {
        for ir in [
            DotFixtures.tutorial(blockM: 128, blockN: 64, blockK: 32),
            AdvancedFixtures.tile2D(blockM: 16, blockN: 24, add: true),
            AdvancedFixtures.rowSum,
        ] {
            let source = try MetalCompiler.emitMSL(ttir: ir, options: .init())
            XCTAssertFalse(source.contains("_tmaxis"), "rewritten a kernel that already lowered")
        }
    }

    /// The collapse the pass cannot fix is still refused by name: a `tt.reduce`
    /// result placed at two dimensions cannot be rebuilt, because the fold has
    /// already happened.
    func testAnUnclonableCollapseIsStillRefused() throws {
        let ir = """
            module {
              tt.func public @collapse_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
                %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %rr = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
                %rc = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
                %rrb = tt.broadcast %rr : tensor<16x1xi32> -> tensor<16x16xi32>
                %rcb = tt.broadcast %rc : tensor<1x16xi32> -> tensor<16x16xi32>
                %off = arith.addi %rrb, %rcb : tensor<16x16xi32>
                %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
                %ptrs = tt.addptr %p, %off : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
                %v = tt.load %ptrs : tensor<16x16xf32>
                %s = "tt.reduce"(%v) <{axis = 1 : i32}> ({
                ^bb0(%a: f32, %b: f32):
                  %t = arith.addf %a, %b : f32
                  tt.reduce.return %t : f32
                }) : (tensor<16x16xf32>) -> tensor<16xf32>
                %sr = tt.expand_dims %s {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
                %sc = tt.expand_dims %s {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
                %srb = tt.broadcast %sr : tensor<16x1xf32> -> tensor<16x16xf32>
                %scb = tt.broadcast %sc : tensor<1x16xf32> -> tensor<16x16xf32>
                %sum = arith.addf %srb, %scb : tensor<16x16xf32>
                %q = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
                %qptrs = tt.addptr %q, %off : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
                tt.store %qptrs, %sum : tensor<16x16x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        do {
            _ = try MetalCompiler.emitMSL(ttir: ir, options: .init())
            XCTFail("expected the diagonal to be refused")
        } catch {
            XCTAssertTrue("\(error)".contains("describes a diagonal"), "\(error)")
        }
    }
}
