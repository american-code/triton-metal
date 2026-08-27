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

    // MARK: - bf16

    /// bf16 arithmetic, `math.*`, a reduction and both conversions to and from
    /// f32, in one kernel.
    ///
    /// Metal has `bfloat` as a storage and arithmetic type but *no* `bfloat`
    /// overloads in its math library and no unambiguous `max(bfloat, bfloat)`,
    /// so the emitter widens into those calls and narrows back out — which is
    /// what the hardware does anyway, there being no bf16 transcendental unit.
    /// This checks that the widening is where it has to be and nowhere else.
    func testBFloat16ElementwiseAndReduction() throws {
        let ir = """
            module {
              tt.func public @bf(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>,
                                 %arg2: !tt.ptr<f32>, %arg3: i32) {
                %c64 = arith.constant 64 : i32
                %half = arith.constant dense<5.000000e-01> : tensor<64xbf16>
                %0 = tt.get_program_id x : i32
                %1 = arith.muli %0, %c64 : i32
                %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
                %3 = tt.splat %1 : i32 -> tensor<64xi32>
                %4 = arith.addi %3, %2 : tensor<64xi32>
                %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
                %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
                %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
                %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
                %9 = tt.load %8, %6 : tensor<64xbf16>
                %10 = arith.mulf %9, %9 : tensor<64xbf16>
                %11 = arith.maximumf %10, %half : tensor<64xbf16>
                %12 = math.sqrt %11 : tensor<64xbf16>
                %13 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
                %14 = tt.addptr %13, %4 : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
                tt.store %14, %12, %6 : tensor<64x!tt.ptr<bf16>>
                %15 = arith.extf %12 : tensor<64xbf16> to tensor<64xf32>
                %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
                %17 = tt.addptr %16, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
                tt.store %17, %15, %6 : tensor<64x!tt.ptr<f32>>
                tt.return
              }
            }
            """
        let source = try MetalCompiler.emitMSL(ttir: ir, options: .init())
        XCTAssertTrue(source.contains("device bfloat *varg0"), source)
        // Plain arithmetic stays in bfloat; only the calls widen.
        XCTAssertTrue(source.contains("bfloat v10 = v9 * v9;"), source)
        XCTAssertTrue(
            source.contains("bfloat v11 = bfloat(max(float(v10), float(vhalf)));"), source)
        XCTAssertTrue(
            source.contains("bfloat v12 = bfloat(precise::sqrt(float(v11)));"), source)
        XCTAssertTrue(source.contains("float v15 = float(v12);"), source)

        let input = ramp(count, in: -4...4)
        let result = try GPU.run(
            ir: ir, grid: (GPU.cdiv(count, 64), 1, 1),
            args: [
                .bfloats(input), .output(count: count, stride: 2), .output(count: count),
                .int32(Int32(count)),
            ])
        // The reference rounds at every step the GPU does: bf16 in, bf16 after
        // the square, bf16 after the max, bf16 after the sqrt.
        func bf(_ value: Float) -> Float { BFloat16.decode(BFloat16.encode(value)) }
        let expected = input.map { value -> Float in
            let x = bf(value)
            return bf(sqrt(bf(max(bf(x * x), bf(0.5)))))
        }
        assertClose(
            GPU.read(result.outputs[0], UInt16.self, count).map(BFloat16.decode), expected,
            tolerance: 0, "bf16 round trip")
        // ...and `extf` to f32 is exact, because every bf16 is an f32.
        assertClose(
            GPU.read(result.outputs[1], Float.self, count), expected, tolerance: 0, "bf16 -> f32")
    }

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
