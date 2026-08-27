import Foundation
import MLX
import TritonMetalCore
import XCTest

@testable import TritonMetalMLX

/// `MetalPipeline`: compiling Triton IR to something callable, and the argument
/// checking that stands between an `MLXArray` and a `[[buffer(N)]]` slot.
final class MetalPipelineTests: XCTestCase {

    /// `out = x + y` over one block, with an element count and a float scale, so
    /// there is one of every argument kind to bind.
    private static func scaledAdd(block: Int = 128) -> String {
        """
        module {
          tt.func public @scaled_add(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                                     %n: i32, %scale: f32) {
            %cb = arith.constant \(block) : i32
            %pid = tt.get_program_id x : i32
            %start = arith.muli %pid, %cb : i32
            %range = tt.make_range {end = \(block) : i32, start = 0 : i32} : tensor<\(block)xi32>
            %splat = tt.splat %start : i32 -> tensor<\(block)xi32>
            %offs = arith.addi %splat, %range : tensor<\(block)xi32>
            %sn = tt.splat %n : i32 -> tensor<\(block)xi32>
            %mask = arith.cmpi slt, %offs, %sn : tensor<\(block)xi32>
            %xp = tt.splat %x : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
            %xptrs = tt.addptr %xp, %offs : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
            %xv = tt.load %xptrs, %mask : tensor<\(block)xf32>
            %yp = tt.splat %y : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
            %yptrs = tt.addptr %yp, %offs : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
            %yv = tt.load %yptrs, %mask : tensor<\(block)xf32>
            %sum = arith.addf %xv, %yv : tensor<\(block)xf32>
            %ss = tt.splat %scale : f32 -> tensor<\(block)xf32>
            %scaled = arith.mulf %sum, %ss : tensor<\(block)xf32>
            %op = tt.splat %out : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
            %optrs = tt.addptr %op, %offs : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
            tt.store %optrs, %scaled, %mask : tensor<\(block)x!tt.ptr<f32>>
            tt.return
          }
        }
        """
    }

    private func compiled() throws -> MetalPipeline {
        try MetalPipeline.compile(ttir: Self.scaledAdd())
    }

    // MARK: - Metadata

    func testMetadataDescribesTheKernelsArguments() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        XCTAssertEqual(pipeline.metadata.name, "scaled_add")
        XCTAssertEqual(pipeline.pointerArguments.map(\.dtype), ["f32", "f32", "f32"])
        XCTAssertEqual(pipeline.scalarArguments.map(\.dtype), ["i32", "f32"])
        XCTAssertEqual(pipeline.metadata.arguments.map(\.index), [0, 1, 2, 3, 4])
        XCTAssertTrue(pipeline.threadsPerThreadgroup % 32 == 0)
        XCTAssertTrue(pipeline.source.contains("kernel void scaled_add"))
    }

    // MARK: - Launching

    func testLaunchComputesTheRightAnswer() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let n = 1000
        let x = synthetic([n], offset: 1)
        let y = synthetic([n], offset: 5, modulus: 53)
        let out = MLX.zeros([n], dtype: .float32)

        let results = try pipeline.launch(
            arrays: [x, y, out], scalars: [.init(n), .init(Float(0.5))],
            grid: Grid(Grid.covering(n, block: 128)))

        XCTAssertEqual(results.count, 3)
        XCTAssertLessThan(maxDifference(results[2], (x + y) * 0.5), 1e-6)
        // The output array was bound by aliasing, so it is the same array.
        XCTAssertLessThan(maxDifference(out, (x + y) * 0.5), 1e-6)
    }

    /// A strided input is staged, and the kernel still gets the right values —
    /// this is the case the alias path cannot serve.
    func testStridedInputIsStagedAndStillCorrect() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let matrix = synthetic([16, 8])
        let column = matrix[0..., 3]   // 16 values, stride 8
        let y = synthetic([16], offset: 2)
        let out = MLX.zeros([16], dtype: .float32)

        let results = try pipeline.launch(
            arrays: [column, y, out], scalars: [.init(16), .init(Float(1))], grid: Grid(1))
        XCTAssertLessThan(maxDifference(results[2], column + y), 1e-6)
    }

    /// Unevaluated inputs are evaluated before their address is taken. Without
    /// the `eval()` in the binding this reads whatever the graph node had before.
    func testLazyInputsAreEvaluatedBeforeBinding() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let x = synthetic([256], offset: 1) * 2 + 1   // a graph, not a buffer
        let y = MLX.zeros([256], dtype: .float32)
        let out = MLX.zeros([256], dtype: .float32)

        let results = try pipeline.launch(
            arrays: [x, y, out], scalars: [.init(256), .init(Float(1))], grid: Grid(2))
        XCTAssertLessThan(maxDifference(results[2], x), 1e-6)
    }

    /// `prepare` binds once and `launch(_:grid:)` reuses the binding, which is
    /// what a hot loop with fixed weights wants. The results have to track the
    /// buffers across launches, not be frozen at prepare time.
    func testPreparedArgumentsCanBeLaunchedRepeatedly() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let x = synthetic([256], offset: 1)
        let y = MLX.zeros([256], dtype: .float32)
        let out = MLX.zeros([256], dtype: .float32)
        let prepared = try pipeline.prepare(
            arrays: [x, y, out], scalars: [.init(256), .init(Float(1))])
        XCTAssertEqual(prepared.modes, [.aliased, .aliased, .aliased])

        _ = try pipeline.launch(prepared, grid: Grid(2))
        XCTAssertLessThan(maxDifference(prepared.results[2], x), 1e-6)

        // The kernel reads `y` through the same buffer, so changing it changes
        // the next launch's answer without rebinding.
        let mutated = try pipeline.prepare(
            arrays: [x, x, out], scalars: [.init(256), .init(Float(1))])
        _ = try pipeline.launch(mutated, grid: Grid(2))
        XCTAssertLessThan(maxDifference(out, x + x), 1e-6)
    }

    func testPrepareChecksArgumentsWithoutLaunching() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.prepare(arrays: [a], scalars: []), contains: "3 pointer and 2 scalar")
        // A bad grid is a launch-time question, not a binding-time one.
        let prepared = try pipeline.prepare(
            arrays: [a, a, a], scalars: [.init(128), .init(Float(1))])
        assertCoreError(
            try pipeline.launch(prepared, grid: Grid(0)), contains: "positive in every dimension")
    }

    func testOneCallConvenienceCompilesAndLaunches() throws {
        try MLXRuntime.require()
        let x = synthetic([128], offset: 1)
        let y = synthetic([128], offset: 9)
        let out = MLX.zeros([128], dtype: .float32)
        let results = try MetalPipeline.run(
            ttir: Self.scaledAdd(), arrays: [x, y, out],
            scalars: [.init(128), .init(Float(2))], grid: Grid(1))
        XCTAssertLessThan(maxDifference(results[2], (x + y) * 2), 1e-6)
    }

    // MARK: - Argument checking

    func testWrongNumberOfArgumentsIsRefused() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.launch(arrays: [a, a], scalars: [.init(128), .init(Float(1))],
                                grid: Grid(1)),
            contains: "3 pointer and 2 scalar arguments")
        assertCoreError(
            try pipeline.launch(arrays: [a, a, a], scalars: [.init(128)], grid: Grid(1)),
            contains: "3 pointer and 2 scalar arguments")
    }

    func testDtypeMismatchNamesBothSides() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let float = synthetic([128])
        let ints = MLX.zeros([128], dtype: .int32)
        assertCoreError(
            try pipeline.launch(
                arrays: [ints, float, float], scalars: [.init(128), .init(Float(1))],
                grid: Grid(1)),
            contains: "!tt.ptr<f32>")
    }

    /// `uint32` rather than `float64`, because MLX refuses to allocate a
    /// `float64` array on the GPU at all (`MLX/ErrorHandler.swift`: "float64 is
    /// not supported on the GPU", and it aborts rather than throws). An unsigned
    /// integer is a real MLX array that this backend still has no Triton spelling
    /// for, which is the case worth checking.
    func testUnsupportedDtypeIsRefusedAtLaunch() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let float = synthetic([128])
        let unsigned = MLX.zeros([128], dtype: .uint32)
        assertCoreError(
            try pipeline.launch(
                arrays: [unsigned, float, float], scalars: [.init(128), .init(Float(1))],
                grid: Grid(1)),
            contains: "uint32")
    }

    /// An integer where the kernel declared `f32` is refused rather than
    /// converted: `1` and `1.0` mean different things, and guessing which one the
    /// caller meant is how a scale factor silently becomes an index.
    func testIntegerForAFloatParameterIsRefused() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.launch(arrays: [a, a, a], scalars: [.init(128), 1], grid: Grid(1)),
            contains: "pass a float")
    }

    func testFloatForAnIntegerParameterIsRefused() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.launch(arrays: [a, a, a], scalars: [1.0, .init(Float(1))], grid: Grid(1)),
            contains: "a float was passed")
    }

    func testOutOfRangeScalarIsRefusedRatherThanTruncated() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.launch(
                arrays: [a, a, a], scalars: [.init(Int(Int32.max) + 1), .init(Float(1))],
                grid: Grid(1)),
            contains: "does not fit")
    }

    func testNonPositiveGridIsRefused() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        assertCoreError(
            try pipeline.launch(
                arrays: [a, a, a], scalars: [.init(128), .init(Float(1))], grid: Grid(1, 0)),
            contains: "positive in every dimension")
    }

    func testEmptyArrayArgumentIsRefused() throws {
        try MLXRuntime.require()
        let pipeline = try compiled()
        let a = synthetic([128])
        let empty = MLX.zeros([0], dtype: .float32)
        assertCoreError(
            try pipeline.launch(
                arrays: [empty, a, a], scalars: [.init(128), .init(Float(1))], grid: Grid(1)),
            contains: "empty")
    }

    // MARK: - Kernel selection

    func testAmbiguousModuleNeedsAKernelName() throws {
        try MLXRuntime.require()
        let two = """
            module {
            \(Self.scaledAdd().dropFirst("module {".count).dropLast(1))
            \(Self.scaledAdd()
                .replacingOccurrences(of: "scaled_add", with: "scaled_add_2")
                .dropFirst("module {".count).dropLast(1))
            }
            """
        assertCoreError(try MetalPipeline.compile(ttir: two), contains: "name the one you want")
        XCTAssertEqual(
            try MetalPipeline.compile(ttir: two, kernel: "scaled_add_2").metadata.name,
            "scaled_add_2")
        assertCoreError(
            try MetalPipeline.compile(ttir: two, kernel: "nope"), contains: "no kernel named")
    }

    func testCompileAllReturnsEveryKernelInTheModule() throws {
        try MLXRuntime.require()
        XCTAssertEqual(try MetalPipeline.compileAll(ttir: Self.scaledAdd()).count, 1)
    }
}
