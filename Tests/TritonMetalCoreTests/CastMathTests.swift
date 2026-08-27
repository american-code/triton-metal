import Foundation
import XCTest

@testable import TritonMetalCore

/// Milestone: `arith` conversions and `math.*` unary ops, each launched on the
/// GPU and compared against a CPU reference.
///
/// Tolerances are per-function: Metal's `precise::` namespace is IEEE-ish but not
/// bit-identical to libm, and the runtime compiles with fast math off precisely so
/// these stay tight (see `MetalRuntime.makeLibrary`).
final class CastMathTests: XCTestCase {
    override func setUpWithError() throws { try skipWithoutMetal() }

    private let count = 200  // 3.125 blocks of 64 — the tail block is masked

    // MARK: - math.*

    /// One kernel per op, so a failure names the function that drifted.
    func testMathOpsMatchCPUReferences() throws {
        let cases: [(mnemonic: String, reference: (Float) -> Float, domain: ClosedRange<Float>, tolerance: Float)] = [
            ("math.exp", { exp($0) }, -8...8, 1e-6),
            ("math.exp2", { exp2($0) }, -8...8, 1e-6),
            ("math.log", { log($0) }, 0.01...50, 1e-6),
            ("math.log2", { log2($0) }, 0.01...50, 1e-6),
            ("math.sqrt", { sqrt($0) }, 0...50, 1e-6),
            ("math.rsqrt", { 1 / sqrt($0) }, 0.05...50, 1e-5),
            ("math.sin", { sin($0) }, -6...6, 1e-5),
            ("math.cos", { cos($0) }, -6...6, 1e-5),
            ("math.tanh", { tanh($0) }, -6...6, 1e-6),
            ("math.erf", { erf($0) }, -3...3, 1e-6),
            ("math.absf", { abs($0) }, -5...5, 0),
            ("math.floor", { $0.rounded(.down) }, -5...5, 0),
            ("math.ceil", { $0.rounded(.up) }, -5...5, 0),
        ]

        for testCase in cases {
            let input = ramp(count, in: testCase.domain)
            let result = try GPU.run(
                ir: AdvancedFixtures.mathKernel(testCase.mnemonic),
                grid: (GPU.cdiv(count, 64), 1, 1),
                args: [.floats(input), .output(count: count), .int32(Int32(count))])
            assertClose(
                GPU.read(result.outputs[0], Float.self, count), input.map(testCase.reference),
                tolerance: testCase.tolerance, testCase.mnemonic)
        }
    }

    func testIntegerAbsoluteValue() throws {
        let input = (0..<count).map { Int32($0 - count / 2) }
        let result = try GPU.run(
            ir: AdvancedFixtures.mathKernel("math.absi", type: "i32"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.ints(input), .output(count: count), .int32(Int32(count))])
        XCTAssertEqual(GPU.read(result.outputs[0], Int32.self, count), input.map { abs($0) })
    }

    func testPreciseNamespaceIsUsedWhereMetalHasOne() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AdvancedFixtures.mathKernel("math.exp"), options: .init())
        XCTAssertTrue(source.contains("precise::exp("), source)
        XCTAssertFalse(source.contains("fast::"), source)
        // tanh/erf/floor/ceil/fabs have no precise:: variant; they must not be
        // spelled with one or the Metal front end rejects the kernel.
        for (mnemonic, expected) in [
            ("math.tanh", "tanh("), ("math.erf", "tm_erf("), ("math.floor", "floor("),
            ("math.ceil", "ceil("), ("math.absf", "fabs("),
        ] {
            let source = try MetalCompiler.emitMSL(
                ttir: AdvancedFixtures.mathKernel(mnemonic), options: .init())
            XCTAssertTrue(source.contains(expected), "\(mnemonic): \(source)")
            XCTAssertFalse(source.contains("precise::\(expected)"), "\(mnemonic): \(source)")
        }
    }

    // MARK: - arith conversions

    func testSignedAndUnsignedIntegerToFloat() throws {
        let input = (0..<count).map { Int32($0 * 7 - 700) }
        for (mnemonic, reference) in [
            ("arith.sitofp", { (v: Int32) in Float(v) }),
            ("arith.uitofp", { (v: Int32) in Float(UInt32(bitPattern: v)) }),
        ] {
            let result = try GPU.run(
                ir: AdvancedFixtures.castKernel(mnemonic, from: "i32", to: "f32"),
                grid: (GPU.cdiv(count, 64), 1, 1),
                args: [.ints(input), .output(count: count), .int32(Int32(count))])
            XCTAssertEqual(
                GPU.read(result.outputs[0], Float.self, count), input.map(reference), mnemonic)
        }
    }

    func testFloatToIntegerTruncatesTowardZero() throws {
        let input = ramp(count, in: -60...60)
        let result = try GPU.run(
            ir: AdvancedFixtures.castKernel("arith.fptosi", from: "f32", to: "i32"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.floats(input), .output(count: count), .int32(Int32(count))])
        XCTAssertEqual(
            GPU.read(result.outputs[0], Int32.self, count), input.map { Int32($0.rounded(.towardZero)) })

        let positive = ramp(count, in: 0...60)
        let unsigned = try GPU.run(
            ir: AdvancedFixtures.castKernel("arith.fptoui", from: "f32", to: "i32"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.floats(positive), .output(count: count), .int32(Int32(count))])
        XCTAssertEqual(
            GPU.read(unsigned.outputs[0], Int32.self, count),
            positive.map { Int32(UInt32($0.rounded(.towardZero))) })
    }

    func testIntegerWidening() throws {
        let input = (0..<count).map { Int16(truncatingIfNeeded: $0 * 331 - 30_000) }
        for (mnemonic, reference) in [
            ("arith.extsi", { (v: Int16) in Int32(v) }),
            ("arith.extui", { (v: Int16) in Int32(UInt16(bitPattern: v)) }),
        ] {
            let result = try GPU.run(
                ir: AdvancedFixtures.castKernel(mnemonic, from: "i16", to: "i32"),
                grid: (GPU.cdiv(count, 64), 1, 1),
                args: [.shorts(input), .output(count: count), .int32(Int32(count))])
            XCTAssertEqual(
                GPU.read(result.outputs[0], Int32.self, count), input.map(reference), mnemonic)
        }
    }

    func testIntegerNarrowing() throws {
        let input = (0..<count).map { Int32($0 * 9_001 - 900_000) }
        let result = try GPU.run(
            ir: AdvancedFixtures.castKernel("arith.trunci", from: "i32", to: "i16"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.ints(input), .output(count: count, stride: 2), .int32(Int32(count))])
        XCTAssertEqual(
            GPU.read(result.outputs[0], Int16.self, count),
            input.map { Int16(truncatingIfNeeded: $0) })
    }

    func testFloatWidthConversions() throws {
        let input = ramp(count, in: -100...100)
        let narrowed = try GPU.run(
            ir: AdvancedFixtures.castKernel("arith.truncf", from: "f32", to: "f16"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.floats(input), .output(count: count, stride: 2), .int32(Int32(count))])
        let halves = GPU.read(narrowed.outputs[0], Float16.self, count)
        XCTAssertEqual(halves, input.map { Float16($0) })

        let widened = try GPU.run(
            ir: AdvancedFixtures.castKernel("arith.extf", from: "f16", to: "f32"),
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.halves(halves), .output(count: count), .int32(Int32(count))])
        XCTAssertEqual(GPU.read(widened.outputs[0], Float.self, count), halves.map { Float($0) })
    }

    func testSelect() throws {
        let input = ramp(count, in: -20...20)
        let result = try GPU.run(
            ir: AdvancedFixtures.select,
            grid: (GPU.cdiv(count, 64), 1, 1),
            args: [.floats(input), .output(count: count), .int32(Int32(count))])
        XCTAssertEqual(
            GPU.read(result.outputs[0], Float.self, count), input.map { $0 > 0 ? $0 * 2 : $0 - 1 })
    }

    // MARK: - Error paths

    func testCastsCheckTheirOperandKinds() {
        for (ir, needle) in [
            (AdvancedFixtures.castKernel("arith.sitofp", from: "f32", to: "f32"),
                "expects an integer source"),
            (AdvancedFixtures.castKernel("arith.extsi", from: "i32", to: "i16"), "must widen"),
            (AdvancedFixtures.castKernel("arith.truncf", from: "f16", to: "f32"), "must narrow"),
        ] {
            XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init()), needle) {
                XCTAssertTrue("\($0)".contains(needle), "\($0)")
            }
        }
    }

    func testMathOpsRejectIntegerOperands() {
        let ir = AdvancedFixtures.mathKernel("math.exp", type: "i32")
        XCTAssertThrowsError(try MetalCompiler.emitMSL(ttir: ir, options: .init())) {
            XCTAssertTrue("\($0)".contains("math.exp on tensor<64xi32>"), "\($0)")
        }
    }

    // MARK: - Helpers

    private func ramp(_ count: Int, in range: ClosedRange<Float>) -> [Float] {
        (0..<count).map {
            range.lowerBound
                + (range.upperBound - range.lowerBound) * Float($0) / Float(max(1, count - 1))
        }
    }
}
