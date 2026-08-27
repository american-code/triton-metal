import Foundation
import XCTest

@testable import TritonMetalCore

/// Milestone: rank-2 tensors, `tt.expand_dims` / `tt.broadcast`, per-dimension
/// block sizes, and 2-D masked load/store with strides built from `tt.addptr`.
final class Rank2Tests: XCTestCase {
    override func setUpWithError() throws { try skipWithoutMetal() }

    /// `out = a + b` over a 2-D tile at sizes that divide neither block dimension.
    func testTiledAddAtAwkwardSizes() throws {
        for (blockM, blockN, rows, columns, stride) in [
            (16, 32, 37, 71, 71),   // ragged in both dimensions
            (8, 64, 8, 64, 64),     // exactly one tile
            (32, 8, 100, 5, 9),     // more rows than columns, padded stride
            (4, 128, 3, 300, 301),  // one partial row block
        ] {
            let a = (0..<(rows * stride)).map { Float($0 % 197) * 0.125 - 3 }
            let b = (0..<(rows * stride)).map { Float($0 % 89) * -0.25 }
            let result = try GPU.run(
                ir: AdvancedFixtures.tile2D(blockM: blockM, blockN: blockN, add: true),
                grid: (GPU.cdiv(rows, blockM), GPU.cdiv(columns, blockN), 1),
                args: [
                    .floats(a), .floats(b), .output(count: rows * stride), .int32(Int32(rows)),
                    .int32(Int32(columns)), .int32(Int32(stride)),
                ])

            var expected = [Float](repeating: 0, count: rows * stride)
            for row in 0..<rows {
                for column in 0..<columns {
                    expected[row * stride + column] = a[row * stride + column] + b[row * stride + column]
                }
            }
            XCTAssertEqual(
                GPU.read(result.outputs[0], Float.self, rows * stride), expected,
                "BLOCK=\(blockM)x\(blockN), \(rows)x\(columns) stride \(stride)")
            XCTAssertEqual(result.kernel.blockShape, [blockM, blockN])
            XCTAssertEqual(result.kernel.blockSize, blockM * blockN)
        }
    }

    /// A 2-D copy (`out = a * a`) with the same masks — proves the mask really
    /// protects the padding columns between rows.
    func testTiledCopyDoesNotWriteOutsideTheLogicalMatrix() throws {
        let (blockM, blockN) = (16, 32)
        let (rows, columns, stride) = (20, 40, 48)
        let a = (0..<(rows * stride)).map { Float($0 % 13) + 1 }
        let sentinel = Float(-999)
        let output = try MetalRuntime.makeBuffer(length: rows * stride * 4)
        let pointer = output.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<(rows * stride) { pointer[index] = sentinel }

        let emission = try MetalCompiler.emit(
            ttir: AdvancedFixtures.tile2D(blockM: blockM, blockN: blockN, add: false),
            options: .init())
        let pipeline = try MetalCompiler.compileMSL(emission.source, kernelName: "tile2d_kernel")
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSizeMake(GPU.cdiv(rows, blockM), GPU.cdiv(columns, blockN), 1),
            threadsPerThreadgroup: emission.kernels[0].threadsPerThreadgroup,
            arguments: [
                .buffer(try GPU.upload(a)), .buffer(try GPU.upload(a)), .buffer(output),
                .int32(Int32(rows)), .int32(Int32(columns)), .int32(Int32(stride)),
            ])

        let actual = GPU.read(output, Float.self, rows * stride)
        for row in 0..<rows {
            for column in 0..<stride {
                let index = row * stride + column
                let expected = column < columns ? a[index] * a[index] : sentinel
                XCTAssertEqual(actual[index], expected, "at [\(row), \(column)]")
            }
        }
    }

    // MARK: - Layout model

    /// Row-uniform values must be computed once per row, not once per element:
    /// that is the whole point of tracking which block dimension a value spans.
    func testRowUniformWorkIsHoistedOutOfTheInnermostLoop() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AdvancedFixtures.tile2D(blockM: 16, blockN: 32, add: true), options: .init())
        let outer = try XCTUnwrap(source.range(of: "for (uint tm_i0 = 0u;"))
        let inner = try XCTUnwrap(source.range(of: "for (uint tm_i1 = tm_thread_id.x;"))
        XCTAssertLessThan(outer.lowerBound, inner.lowerBound)

        let rowScope = String(source[outer.upperBound..<inner.lowerBound])
        // offs_m and the row-stride multiply live between the two loops.
        XCTAssertTrue(rowScope.contains("int v4 = int(tm_i0);"), rowScope)
        XCTAssertTrue(rowScope.contains("int v13 = v7 * v12;"), rowScope)
        // offs_n only exists inside the distributed loop.
        XCTAssertFalse(rowScope.contains("tm_i1"), rowScope)
        XCTAssertTrue(source.contains("int v5 = int(tm_i1);"), source)
    }

    func testExpandDimsAndBroadcastAreFreeRelabellings() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AdvancedFixtures.tile2D(blockM: 16, blockN: 32, add: true), options: .init())
        // %10/%11 (expand_dims) and %14/%15 (broadcast) must not become copies.
        for dead in ["int v10 =", "int v11 =", "int v14 =", "int v15 =", "bool v21 =", "bool v22 ="] {
            XCTAssertFalse(source.contains(dead), "\(dead) should be a relabelling, not a copy")
        }
        XCTAssertTrue(source.contains("int v16 = v13 + v9;"), source)
    }

    func testThreadsCoverTheInnermostDimension() throws {
        for (blockN, simdgroups, expected) in [(32, 4, 32), (256, 4, 128), (256, 1, 32), (8, 8, 32)] {
            let kernels = try MetalCompiler.emit(
                ttir: AdvancedFixtures.tile2D(blockM: 16, blockN: blockN, add: true),
                options: .init(numSimdgroups: simdgroups)
            ).kernels
            XCTAssertEqual(
                kernels[0].threadsPerThreadgroup, expected, "BLOCK_N=\(blockN), warps=\(simdgroups)")
        }
    }

    func testRankTwoTypesParse() throws {
        var parser = try TritonIRParser("tensor<16x32xf32>")
        XCTAssertEqual(try parser.parseType(), .tensor(shape: [16, 32], element: .float(width: 32)))
        var pointers = try TritonIRParser("tensor<8x1x!tt.ptr<f16>>")
        XCTAssertEqual(
            try pointers.parseType(),
            .tensor(shape: [8, 1], element: .pointer(.float(width: 16))))
        var cube = try TritonIRParser("tensor<2x4x8xi8>")
        XCTAssertEqual(try cube.parseType(), .tensor(shape: [2, 4, 8], element: .integer(width: 8)))
    }

    // MARK: - Error paths

    /// A rank-1 tensor in a rank-2 kernel is ambiguous until an expand_dims says
    /// which dimension it indexes.
    func testUninferableLayoutIsReported() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
                %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x4x!tt.ptr<f32>>
                %4 = tt.addptr %3, %2 : tensor<1x4x!tt.ptr<f32>>, tensor<1x4xi32>
                %5 = tt.load %4 : tensor<1x4xf32>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("cannot infer which block dimension '%0'"), "\($0)")
        }
    }

    func testConflictingBlockSizesAlongOneDimensionAreReported() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
                %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
                %4 = tt.broadcast %1 : tensor<16x1xi32> -> tensor<16x4xi32>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("mixes tensor lengths"), "\($0)")
        }
    }

    func testExpandDimsShapeIsChecked() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x4xi32>
                tt.return
              }
            }
            """
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("tt.expand_dims axis 1 cannot turn"), "\($0)")
        }
    }
}
