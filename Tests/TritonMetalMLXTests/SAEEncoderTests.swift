import Foundation
import MLX
import TritonMetalCore
import XCTest

@testable import TritonMetalMLX

/// The fused SAE encoder — `relu(x @ W_enc + b_enc)` in one Triton kernel —
/// against MLX's own ops computing the same thing.
///
/// MLX is the reference rather than a hand-rolled CPU loop on purpose: the claim
/// being made is that a Swift caller can run a Triton kernel *instead of* the MLX
/// ops it would otherwise call, so the thing worth checking is that the two agree.
final class SAEEncoderTests: XCTestCase {

    private static var encoder: SAEEncoder.Compiled?

    /// Compiled once for the whole suite. Lowering and building the pipeline is
    /// the expensive part and it does not vary per case.
    private func encoder() throws -> SAEEncoder.Compiled {
        if let existing = Self.encoder { return existing }
        let built = try SAEEncoder.compile()
        Self.encoder = built
        return built
    }

    private func check(
        rows: Int, model: Int, features: Int, tolerance: Float = 2e-6,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let x = synthetic([rows, model], offset: 1, modulus: 71)
        let w = synthetic([model, features], offset: 7, modulus: 53)
        let b = synthetic([features], offset: 3, modulus: 29)
        let got = try encoder()(x, w, b)
        let want = SAEEncoder.reference(x, w, b)
        XCTAssertEqual(got.shape, [rows, features], file: file, line: line)
        let scale = max(MLX.abs(want).max().item(Float.self), 1)
        XCTAssertLessThan(
            maxDifference(got, want) / scale, tolerance,
            "[\(rows), \(model)] @ [\(model), \(features)]", file: file, line: line)
    }

    /// Shapes that are whole multiples of the 64x64x32 block, so no mask fires.
    func testExactMultiplesOfTheBlock() throws {
        try MLXRuntime.require()
        try check(rows: 64, model: 32, features: 64)
        try check(rows: 128, model: 128, features: 256)
        try check(rows: 64, model: 512, features: 2048)
    }

    /// Ragged in all three dimensions at once, which is what the masks are for.
    func testRaggedShapes() throws {
        try MLXRuntime.require()
        try check(rows: 100, model: 300, features: 700)
        try check(rows: 1, model: 1, features: 1)
        try check(rows: 7, model: 13, features: 5)
        try check(rows: 65, model: 33, features: 65)
    }

    /// A single token through a realistically-shaped dictionary — the shape an
    /// interpretability tool actually runs when it probes one activation.
    func testSingleRowAgainstAWideDictionary() throws {
        try MLXRuntime.require()
        try check(rows: 1, model: 512, features: 4096)
    }

    /// The ReLU is doing something: with a bias low enough to push most
    /// pre-activations negative, most of the output is exactly zero and the rest
    /// still matches.
    func testReLUZeroesTheNegativeHalf() throws {
        try MLXRuntime.require()
        let x = synthetic([32, 64], offset: 1)
        let w = synthetic([64, 128], offset: 7, modulus: 53)
        let b = MLX.full([128], values: MLXArray(Float(-5)))
        let got = try encoder()(x, w, b)
        let want = SAEEncoder.reference(x, w, b)
        XCTAssertLessThan(maxDifference(got, want), 1e-6)
        XCTAssertGreaterThan(
            (got .== 0).sum().item(Int32.self), 128,
            "a -5 bias should have driven most units to exactly zero")
        XCTAssertEqual(got.min().item(Float.self), 0, "relu produced a negative value")
    }

    /// The output array the wrapper allocates is contiguous, so the kernel writes
    /// it in place — the result is that array, not a copy of it.
    func testTheResultIsTheKernelsOwnOutputBuffer() throws {
        try MLXRuntime.require()
        let pipeline = try encoder().pipeline
        let x = synthetic([64, 64], offset: 1)
        let w = synthetic([64, 64], offset: 7, modulus: 53)
        let b = synthetic([64], offset: 3, modulus: 29)
        let out = MLX.zeros([64, 64], dtype: .float32)

        let results = try pipeline.launch(
            arrays: [x, w, b, out], scalars: [.init(64), .init(64), .init(64)], grid: Grid(1, 1))
        XCTAssertEqual(maxDifference(results[3], out), 0, "the result is not `out` itself")
        XCTAssertLessThan(maxDifference(out, SAEEncoder.reference(x, w, b)), 1e-6)
    }

    /// Same inputs, same answer, twice — a kernel that reads uninitialised
    /// threadgroup memory usually passes once.
    func testRepeatedLaunchesAgree() throws {
        try MLXRuntime.require()
        let x = synthetic([70, 130], offset: 1)
        let w = synthetic([130, 90], offset: 7, modulus: 53)
        let b = synthetic([90], offset: 3, modulus: 29)
        let first = try encoder()(x, w, b)
        for _ in 0..<4 {
            XCTAssertEqual(maxDifference(try encoder()(x, w, b), first), 0)
        }
    }

    // MARK: - Shape checking

    /// `MetalPipeline.launch` cannot check extents — a Triton kernel's arguments
    /// are bare pointers by the time they reach this backend — but a wrapper that
    /// knows its own op can, and this one does.
    func testMismatchedShapesAreRefused() throws {
        try MLXRuntime.require()
        let encoder = try encoder()
        let x = synthetic([8, 64])
        assertCoreError(
            try encoder(x, synthetic([32, 128]), synthetic([128])),
            contains: "contraction dimensions differ")
        assertCoreError(
            try encoder(x, synthetic([64, 128]), synthetic([64])),
            contains: "128 features but b_enc has 64")
        assertCoreError(
            try encoder(synthetic([8]), synthetic([64, 128]), synthetic([128])),
            contains: "expects x [M, D]")
        assertCoreError(
            try encoder(x, synthetic([64, 128]), synthetic([1, 128])),
            contains: "expects x [M, D]")
    }

    // MARK: - The IR itself

    func testTheKernelDeclaresTheArgumentsTheWrapperBinds() throws {
        try MLXRuntime.require()
        let pipeline = try encoder().pipeline
        XCTAssertEqual(pipeline.metadata.name, "sae_encode_kernel")
        XCTAssertEqual(pipeline.pointerArguments.map(\.dtype), ["f32", "f32", "f32", "f32"])
        XCTAssertEqual(pipeline.scalarArguments.map(\.dtype), ["i32", "i32", "i32"])
        XCTAssertEqual(pipeline.metadata.blockShape, [64, 64])
    }

    /// The bias add and the ReLU are in the kernel, not in a second pass. If they
    /// ever stop being fused this is the test that notices.
    func testTheEpilogueIsFusedIntoTheKernel() throws {
        try MLXRuntime.require()
        let source = try encoder().pipeline.source
        XCTAssertEqual(
            source.components(separatedBy: "kernel void").count - 1, 1,
            "the encoder lowered to more than one kernel")
        XCTAssertTrue(source.contains("= max("), "no ReLU in the emitted MSL")
        XCTAssertTrue(source.contains("simdgroup_multiply_accumulate"), "the dot did not lower")
    }

    /// Other block shapes lower and agree too — the IR is parameterised and the
    /// default is a choice, not a constraint.
    func testAlternativeBlockShapes() throws {
        try MLXRuntime.require()
        for blocking in [
            SAEEncoder.Blocking(m: 32, f: 32, d: 16), SAEEncoder.Blocking(m: 16, f: 64, d: 64),
        ] {
            let encoder = try SAEEncoder.compile(blocking: blocking)
            let x = synthetic([50, 40], offset: 1)
            let w = synthetic([40, 90], offset: 7, modulus: 53)
            let b = synthetic([90], offset: 3, modulus: 29)
            XCTAssertLessThan(
                maxDifference(try encoder(x, w, b), SAEEncoder.reference(x, w, b)), 1e-5,
                "block \(blocking)")
        }
    }
}
