import XCTest

@testable import TritonMetalCore

final class EmitterTests: XCTestCase {
    func testVectorAddEmitsExpectedMSL() throws {
        let result = try MetalCompiler.emit(ttir: IRFixtures.vectorAdd, options: .init(numSimdgroups: 4))
        let source = result.source

        XCTAssertTrue(source.contains("#include <metal_stdlib>"), source)
        XCTAssertTrue(source.contains("kernel void add_kernel("), source)
        XCTAssertTrue(source.contains("device float *varg0 [[buffer(0)]]"), source)
        XCTAssertTrue(source.contains("device float *varg2 [[buffer(2)]]"), source)
        XCTAssertTrue(source.contains("constant int &varg3 [[buffer(3)]]"), source)
        XCTAssertTrue(source.contains("uint3 tm_program_id [[threadgroup_position_in_grid]]"), source)

        // Program-uniform values are hoisted above the per-lane loop.
        let loopStart = try XCTUnwrap(source.range(of: "for (uint tm_i0"))
        let prologue = String(source[source.startIndex..<loopStart.lowerBound])
        XCTAssertTrue(prologue.contains("int v0 = int(tm_program_id.x);"), prologue)
        XCTAssertTrue(prologue.contains("int v1 = v0 * vc1024_i32;"), prologue)
        XCTAssertFalse(prologue.contains("tm_i0"), prologue)

        // Tensor values become per-lane scalars inside it.
        let body = String(source[loopStart.lowerBound...])
        XCTAssertTrue(body.contains("int v2 = int(tm_i0);"), body)
        XCTAssertTrue(body.contains("bool v6 = v4 < v5;"), body)
        XCTAssertTrue(body.contains("device float *v8 = v7 + v4;"), body)
        XCTAssertTrue(body.contains("float v9 = v6 ? *v8 : 0.0f;"), body)
        XCTAssertTrue(body.contains("float v13 = v9 + v12;"), body)
        XCTAssertTrue(body.contains("if (v6) { *v15 = v13; }"), body)
    }

    func testKernelMetadata() throws {
        let result = try MetalCompiler.emit(ttir: IRFixtures.vectorAdd, options: .init(numSimdgroups: 4))
        XCTAssertEqual(result.kernels.count, 1)
        let kernel = result.kernels[0]
        XCTAssertEqual(kernel.name, "add_kernel")
        XCTAssertEqual(kernel.blockSize, 1024)
        // num_warps=4 -> 128 threads; each thread strides over the 1024-wide block.
        XCTAssertEqual(kernel.threadsPerThreadgroup, 128)
        XCTAssertEqual(
            kernel.arguments,
            [
                KernelArgument(index: 0, kind: .pointer, dtype: "f32"),
                KernelArgument(index: 1, kind: .pointer, dtype: "f32"),
                KernelArgument(index: 2, kind: .pointer, dtype: "f32"),
                KernelArgument(index: 3, kind: .scalar, dtype: "i32"),
            ])
    }

    func testThreadgroupSizeTracksNumSimdgroupsAndBlockSize() throws {
        func threads(numSimdgroups: Int, ir: String) throws -> Int {
            try MetalCompiler.emit(ttir: ir, options: .init(numSimdgroups: numSimdgroups))
                .kernels[0].threadsPerThreadgroup
        }
        XCTAssertEqual(try threads(numSimdgroups: 1, ir: IRFixtures.vectorAdd), 32)
        XCTAssertEqual(try threads(numSimdgroups: 8, ir: IRFixtures.vectorAdd), 256)
        // Never more threads than the block has lanes.
        XCTAssertEqual(try threads(numSimdgroups: 8, ir: IRFixtures.integerAdd), 32)
        // Never more than Metal's 1024-thread threadgroup limit.
        XCTAssertEqual(try threads(numSimdgroups: 64, ir: IRFixtures.vectorAdd), 1024)
    }

    func testIntegerKernelUsesIntTypes() throws {
        let source = try MetalCompiler.emitMSL(ttir: IRFixtures.integerAdd, options: .init())
        XCTAssertTrue(source.contains("device int *varg0 [[buffer(0)]]"), source)
        XCTAssertTrue(source.contains("int v9 = v6 ? *v8 : 0;"), source)
        XCTAssertTrue(source.contains("int v15 = v13 * v14;"), source)
    }

    func testScalarFloatArgumentsAndOtherOperand() throws {
        let scale = try MetalCompiler.emit(ttir: IRFixtures.scaleBias, options: .init())
        XCTAssertTrue(scale.source.contains("constant float &varg2 [[buffer(2)]]"), scale.source)
        XCTAssertTrue(
            scale.source.contains("uint3 tm_grid_size [[threadgroups_per_grid]]"), scale.source)
        XCTAssertEqual(
            scale.kernels[0].arguments.map(\.kind), [.pointer, .pointer, .scalar, .scalar, .scalar])

        let mul = try MetalCompiler.emitMSL(ttir: IRFixtures.vectorMul, options: .init())
        // `other` becomes the else-branch of the masked load.
        XCTAssertTrue(mul.contains("float v9 = v6 ? *v8 : vcst;"), mul)
        XCTAssertTrue(mul.contains("float vcst = 1.0f;"), mul)
    }

    func testKernelInfoJSONShape() throws {
        let json = try MetalCompiler.emit(ttir: IRFixtures.copy, options: .init(numSimdgroups: 2))
            .kernelInfoJSON()
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        let kernels = try XCTUnwrap(object?["kernels"] as? [[String: Any]])
        XCTAssertEqual(kernels.count, 1)
        XCTAssertEqual(kernels[0]["name"] as? String, "copy_kernel")
        XCTAssertEqual(kernels[0]["block_size"] as? Int, 256)
        XCTAssertEqual(kernels[0]["block_shape"] as? [Int], [256])
        XCTAssertEqual(kernels[0]["threads_per_threadgroup"] as? Int, 64)
        let args = try XCTUnwrap(kernels[0]["args"] as? [[String: Any]])
        XCTAssertEqual(args.map { $0["kind"] as? String }, ["pointer", "pointer", "scalar"])
        XCTAssertEqual(args.map { $0["dtype"] as? String }, ["f32", "f32", "i32"])
    }

    /// Every fixture must survive the real Metal front-end, not just our emitter.
    func testAllFixturesCompileOnDevice() throws {
        try skipWithoutMetal()
        var fixtures: [(String, String)] = [
            ("copy", IRFixtures.copy), ("add", IRFixtures.vectorAdd),
            ("mul", IRFixtures.vectorMul), ("scale_bias", IRFixtures.scaleBias),
            ("iadd", IRFixtures.integerAdd),
            ("select", AdvancedFixtures.select),
            ("strided_sum", AdvancedFixtures.stridedSum),
            ("two_results", AdvancedFixtures.loopTwoResults),
            ("conditional", AdvancedFixtures.conditional),
            ("tile2d", AdvancedFixtures.tile2D(blockM: 16, blockN: 32, add: true)),
            ("row_sum", AdvancedFixtures.rowSum),
            ("softmax", AdvancedFixtures.softmax(block: 128)),
            ("online_softmax", AdvancedFixtures.onlineSoftmax(block: 64)),
            ("dot_single_tile", DotFixtures.singleTile(m: 16, n: 16, k: 16)),
            ("matmul", DotFixtures.tutorial(blockM: 32, blockN: 32, blockK: 16)),
            (
                "reduce_max",
                AdvancedFixtures.reduce1D(combiner: "arith.maxnumf", identity: "0xFF800000")
            ),
        ]
        for mnemonic in AdvancedFixtures.everyMathMnemonic {
            fixtures.append((mnemonic, AdvancedFixtures.mathKernel(mnemonic)))
        }
        for cast in AdvancedFixtures.everyCast {
            fixtures.append(
                (cast.mnemonic, AdvancedFixtures.castKernel(cast.mnemonic, from: cast.from, to: cast.to)))
        }

        for (name, ir) in fixtures {
            let result = try MetalCompiler.emit(ttir: ir, options: .init())
            let library = try MetalCompiler.compileMSL(result.source)
            XCTAssertTrue(
                library.functionNames.contains(result.kernels[0].name),
                "\(name): library exposes \(library.functionNames)")
        }
    }
}
