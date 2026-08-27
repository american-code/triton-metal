import Foundation
import Metal
import XCTest

@testable import TritonMetalCore

/// `tt.atomic_rmw` and `tt.atomic_cas`: the operation the backward pass of every
/// fused kernel needs, and the last of the CUDA-gap blockers.
///
/// The correctness question an atomic poses is different from every other op's.
/// A lowering that loses updates and one that does not produce the *same* answer
/// on a single thread, so the tests here all run a grid large enough that every
/// output slot is contended by hundreds of threads across many threadgroups, and
/// compare against a CPU reference computed in an order the GPU is free not to
/// use. Where the reference is order-dependent (float addition is not
/// associative) the value is chosen to be exactly representable, so the assertion
/// is equality rather than a tolerance that would hide a lost update — and the
/// reordering is measured separately, in
/// `testFloatAtomicsReorderWithinTheSummationBound`.
final class AtomicTests: XCTestCase {

    private let block = 64

    // MARK: - Harness

    /// Runs the scatter kernel and returns the accumulator buffer's contents.
    @discardableResult
    private func scatter(
        kind: String, values: [Float], buckets: Int, seed: [Float], masked: Bool = true,
        numSimdgroups: Int = 4
    ) throws -> (result: [Float], run: KernelRun) {
        let run = try GPU.run(
            ir: AtomicFixtures.scatter(kind: kind, element: "f32", block: block, masked: masked),
            grid: (GPU.cdiv(values.count, block), 1, 1),
            args: [
                .floats(values), .floats(seed), .int32(Int32(values.count)), .int32(Int32(buckets)),
            ], numSimdgroups: numSimdgroups)
        return (GPU.read(run.buffers[1]!, Float.self, buckets), run)
    }

    private func scatterInt(
        kind: String, values: [Int32], buckets: Int, seed: [Int32], masked: Bool = true
    ) throws -> [Int32] {
        let run = try GPU.run(
            ir: AtomicFixtures.scatter(kind: kind, element: "i32", block: block, masked: masked),
            grid: (GPU.cdiv(values.count, block), 1, 1),
            args: [
                .ints(values), .ints(seed), .int32(Int32(values.count)), .int32(Int32(buckets)),
            ])
        return GPU.read(run.buffers[1]!, Int32.self, buckets)
    }

    // MARK: - f32 accumulation

    /// Concurrent `fadd` across many threadgroups against a CPU reference.
    ///
    /// The contributions are small integers, which f32 adds exactly at these
    /// magnitudes, so *any* order gives the same total and the assertion can be
    /// equality: a single lost update changes the answer.
    func testFloatAddAccumulatesExactlyAcrossThreadgroups() throws {
        try skipWithoutMetal()
        for (count, buckets) in [(4096, 8), (4096, 1), (1000, 7), (65536, 32)] {
            let values = (0..<count).map { Float($0 % 8) }
            var expected = [Float](repeating: 0, count: buckets)
            for (index, value) in values.enumerated() { expected[index % buckets] += value }

            let (actual, _) = try scatter(
                kind: "fadd", values: values, buckets: buckets,
                seed: [Float](repeating: 0, count: buckets))
            XCTAssertEqual(
                actual, expected, "n=\(count) buckets=\(buckets): a lost update or a double count")
        }
    }

    /// The same kernel at every threadgroup size, because the single-writer guard
    /// and the lane distribution both depend on it.
    func testFloatAddAtEveryNumWarps() throws {
        try skipWithoutMetal()
        let count = 4096
        let buckets = 8
        let values = (0..<count).map { Float($0 % 8) }
        var expected = [Float](repeating: 0, count: buckets)
        for (index, value) in values.enumerated() { expected[index % buckets] += value }
        for simdgroups in [1, 2, 4, 8, 16, 32] {
            let (actual, _) = try scatter(
                kind: "fadd", values: values, buckets: buckets,
                seed: [Float](repeating: 0, count: buckets), numSimdgroups: simdgroups)
            XCTAssertEqual(actual, expected, "num_warps=\(simdgroups)")
        }
    }

    /// The accumulator is read-modify-written, not overwritten: a non-zero seed
    /// survives.
    func testFloatAddAccumulatesOntoAnExistingValue() throws {
        try skipWithoutMetal()
        let count = 2048
        let buckets = 4
        let values = (0..<count).map { Float($0 % 4) }
        let seed: [Float] = [10, 20, 30, 40]
        var expected = seed
        for (index, value) in values.enumerated() { expected[index % buckets] += value }
        let (actual, _) = try scatter(
            kind: "fadd", values: values, buckets: buckets, seed: seed)
        XCTAssertEqual(actual, expected)
    }

    /// The masked form: a count that is not a multiple of the block leaves a tail
    /// of lanes that must perform **no** access at all — one spurious add is
    /// visible in the total.
    func testMaskedAtomicSkipsTheTailEntirely() throws {
        try skipWithoutMetal()
        for count in [1, 63, 65, 127, 1000, 4095] {
            let buckets = 8
            let values = (0..<count).map { Float($0 % 8 + 1) }
            var expected = [Float](repeating: 0, count: buckets)
            for (index, value) in values.enumerated() { expected[index % buckets] += value }
            let (actual, _) = try scatter(
                kind: "fadd", values: values, buckets: buckets,
                seed: [Float](repeating: 0, count: buckets))
            XCTAssertEqual(actual, expected, "n=\(count)")
        }
    }

    /// Without a mask the same kernel over a whole number of blocks agrees, which
    /// is what proves the mask is the only difference.
    func testUnmaskedAtomicMatchesTheMaskedOneOnAWholeNumberOfBlocks() throws {
        try skipWithoutMetal()
        let count = 4096
        let buckets = 8
        let values = (0..<count).map { Float($0 % 8 + 1) }
        let seed = [Float](repeating: 0, count: buckets)
        let (masked, _) = try scatter(kind: "fadd", values: values, buckets: buckets, seed: seed)
        let (unmasked, _) = try scatter(
            kind: "fadd", values: values, buckets: buckets, seed: seed, masked: false)
        XCTAssertEqual(masked, unmasked)
    }

    // MARK: - i32

    func testIntegerAddAccumulatesExactly() throws {
        try skipWithoutMetal()
        for (count, buckets) in [(4096, 8), (1000, 7), (65536, 16)] {
            let values = (0..<count).map { Int32($0 % 13) }
            var expected = [Int32](repeating: 0, count: buckets)
            for (index, value) in values.enumerated() { expected[index % buckets] += value }
            XCTAssertEqual(
                try scatterInt(
                    kind: "add", values: values, buckets: buckets,
                    seed: [Int32](repeating: 0, count: buckets)),
                expected, "n=\(count) buckets=\(buckets)")
        }
    }

    /// `max`, `min`, `umax`, `umin`, `and`, `or`, `xor` on i32 — every one of them
    /// is order-independent, so the CPU reference is unambiguous.
    func testEveryIntegerReadModifyWriteMatchesTheCPU() throws {
        try skipWithoutMetal()
        let count = 4096
        let buckets = 8
        // A spread that includes negatives, so `max`/`min` (signed) and
        // `umax`/`umin` (unsigned) disagree and the cast is exercised.
        let values = (0..<count).map { Int32(($0 % 251) - 125) }
        let cases: [(String, Int32, (Int32, Int32) -> Int32)] = [
            ("max", Int32.min, { max($0, $1) }),
            ("min", Int32.max, { min($0, $1) }),
            ("umax", 0, { Int32(bitPattern: max(UInt32(bitPattern: $0), UInt32(bitPattern: $1))) }),
            ("umin", -1, { Int32(bitPattern: min(UInt32(bitPattern: $0), UInt32(bitPattern: $1))) }),
            ("and", -1, { $0 & $1 }),
            ("or", 0, { $0 | $1 }),
            ("xor", 0, { $0 ^ $1 }),
        ]
        for (kind, identity, combine) in cases {
            var expected = [Int32](repeating: identity, count: buckets)
            for (index, value) in values.enumerated() {
                expected[index % buckets] = combine(expected[index % buckets], value)
            }
            XCTAssertEqual(
                try scatterInt(
                    kind: kind, values: values, buckets: buckets,
                    seed: [Int32](repeating: identity, count: buckets)),
                expected, "tt.atomic_rmw \(kind)")
        }
    }

    /// `exch` has no order-independent total, but it does have an invariant: the
    /// value left behind is one of the values some lane wrote.
    func testExchangeLeavesOneOfTheWrittenValues() throws {
        try skipWithoutMetal()
        let count = 4096
        let buckets = 8
        let values = (0..<count).map { Int32($0 + 1) }
        let result = try scatterInt(
            kind: "exch", values: values, buckets: buckets,
            seed: [Int32](repeating: 0, count: buckets))
        for slot in 0..<buckets {
            let candidates = Set(
                stride(from: slot, to: count, by: buckets).map { values[$0] })
            XCTAssertTrue(
                candidates.contains(result[slot]),
                "slot \(slot) holds \(result[slot]), which no lane wrote")
        }
    }

    // MARK: - f32 max/min through the compare-exchange loop

    /// Metal has no float `atomic_fetch_max_explicit`, so these go through
    /// `tm_atomic_fmax`. Max and min are order-independent and exact, so the
    /// assertion is equality.
    func testFloatMaxAndMinGoThroughTheCompareExchangeLoop() throws {
        try skipWithoutMetal()
        let count = 8192
        let buckets = 8
        // Includes negatives on purpose: a bit-pattern comparison would get the
        // sign wrong, and `tm_atomic_fmax` compares as floats.
        let values = (0..<count).map { Float(($0 % 997) - 500) * 0.25 }

        for (kind, identity, combine) in [
            ("max", -Float.greatestFiniteMagnitude, { (a: Float, b: Float) in max(a, b) }),
            ("min", Float.greatestFiniteMagnitude, { (a: Float, b: Float) in min(a, b) }),
        ] {
            var expected = [Float](repeating: identity, count: buckets)
            for (index, value) in values.enumerated() {
                expected[index % buckets] = combine(expected[index % buckets], value)
            }
            let (actual, run) = try scatter(
                kind: kind, values: values, buckets: buckets,
                seed: [Float](repeating: identity, count: buckets))
            XCTAssertEqual(actual, expected, "tt.atomic_rmw \(kind) on f32")
            XCTAssertTrue(run.source.contains("tm_atomic_f\(kind)"), run.source)
        }
    }

    // MARK: - The old value

    /// The value an atomic returns. With every contribution equal to 1 the set of
    /// old values a bucket hands out is exactly `{0, 1, ..., k-1}` whatever the
    /// order, so this is an order-independent test of a read-modify-write's read
    /// half — and of a side-effecting op that defines a value.
    func testTheReturnedOldValueIsThePartialSumBeforeThisLanesContribution() throws {
        try skipWithoutMetal()
        let count = 2048
        let buckets = 4
        let values = [Int32](repeating: 1, count: count)
        let run = try GPU.run(
            ir: AtomicFixtures.scatter(kind: "add", element: "i32", block: block, keepOld: true),
            grid: (GPU.cdiv(count, block), 1, 1),
            args: [
                .ints(values), .ints([Int32](repeating: 0, count: buckets)),
                .int32(Int32(count)), .int32(Int32(buckets)), .output(count: count),
            ])
        let totals = GPU.read(run.buffers[1]!, Int32.self, buckets)
        XCTAssertEqual(totals, [Int32](repeating: Int32(count / buckets), count: buckets))

        let olds = GPU.read(run.outputs[0], Int32.self, count)
        for slot in 0..<buckets {
            let seen = stride(from: slot, to: count, by: buckets).map { olds[$0] }.sorted()
            XCTAssertEqual(
                seen, (0..<Int32(count / buckets)).map { $0 },
                "slot \(slot) handed out \(seen.count) old values that are not a permutation of "
                    + "0..<\(count / buckets)")
        }
    }

    // MARK: - Compare-and-swap

    /// Exactly one lane per slot can replace the sentinel, and the lane that wins
    /// is the one whose index is left behind.
    func testCompareAndSwapHasExactlyOneWinnerPerSlot() throws {
        try skipWithoutMetal()
        let count = 4096
        let buckets = 8
        let run = try GPU.run(
            ir: AtomicFixtures.compareAndSwap(element: "i32", block: block),
            grid: (GPU.cdiv(count, block), 1, 1),
            args: [
                .ints([Int32](repeating: -1, count: buckets)), .output(count: count),
                .int32(Int32(count)), .int32(Int32(buckets)),
            ])
        let slots = GPU.read(run.buffers[0]!, Int32.self, buckets)
        let seen = GPU.read(run.outputs[0], Int32.self, count)
        for slot in 0..<buckets {
            let winner = slots[slot]
            XCTAssertNotEqual(winner, -1, "slot \(slot) was never claimed")
            XCTAssertEqual(Int(winner) % buckets, slot, "slot \(slot) claimed by \(winner)")
            // Exactly one lane saw the sentinel: everyone else observed a winner.
            let sentinelSightings = stride(from: slot, to: count, by: buckets)
                .filter { seen[$0] == -1 }
            XCTAssertEqual(
                sentinelSightings.count, 1,
                "slot \(slot) reported \(sentinelSightings.count) lanes seeing the sentinel")
        }
    }

    // MARK: - Determinism

    /// Float atomics **reorder**, and this measures by how much rather than
    /// hiding it. The same kernel is run repeatedly over contributions with a
    /// wide dynamic range, where reassociation is visible; the results are
    /// compared with each other and with a CPU reference summed in index order.
    ///
    /// The bound is the standard one for recursive summation and is *derived*,
    /// not tuned: |computed - exact| <= (k-1) * u * sum|x_i|, with u = 2^-24 for
    /// binary32. It is asserted as an upper bound on the observed spread, and the
    /// observed spread is printed so a regression that made it worse would show.
    func testFloatAtomicsReorderWithinTheSummationBound() throws {
        try skipWithoutMetal()
        let count = 16384
        let buckets = 4
        let perBucket = count / buckets
        // One large value per bucket and a long tail of small ones: the classic
        // shape where summation order matters.
        let values = (0..<count).map { index -> Float in
            index < buckets ? 1e6 : 1e-3 * Float((index % 7) + 1)
        }
        var absoluteSums = [Float](repeating: 0, count: buckets)
        var reference = [Double](repeating: 0, count: buckets)
        for (index, value) in values.enumerated() {
            absoluteSums[index % buckets] += abs(value)
            reference[index % buckets] += Double(value)
        }
        let unitRoundoff = Float(pow(2.0, -24.0))
        let bound = (0..<buckets).map {
            Float(perBucket - 1) * unitRoundoff * absoluteSums[$0]
        }

        var runs: [[Float]] = []
        for _ in 0..<5 {
            let (actual, _) = try scatter(
                kind: "fadd", values: values, buckets: buckets,
                seed: [Float](repeating: 0, count: buckets))
            runs.append(actual)
        }
        for slot in 0..<buckets {
            let observed = runs.map { $0[slot] }
            let spread = (observed.max() ?? 0) - (observed.min() ?? 0)
            let error = observed.map { abs($0 - Float(reference[slot])) }.max() ?? 0
            XCTAssertLessThanOrEqual(
                spread, bound[slot],
                "slot \(slot): run-to-run spread \(spread) exceeds the summation bound "
                    + "\(bound[slot])")
            XCTAssertLessThanOrEqual(
                error, bound[slot],
                "slot \(slot): error against an in-order CPU sum \(error) exceeds the summation "
                    + "bound \(bound[slot])")
        }
    }

    // MARK: - The single-writer guard

    /// An atomic whose loop nest ends on a block dimension every thread walks
    /// would otherwise be performed once per thread, which for an add is
    /// `threads_per_threadgroup` times the intended contribution.
    func testAUniformNestPerformsTheAtomicOnce() throws {
        try skipWithoutMetal()
        let rows = 8
        let columns = 16
        let stride = 20  // > columns, so the rows are padded
        let tile = (0..<(rows * stride)).map { Float($0 % 9) }
        var expected = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            for column in 0..<columns { expected[row] += tile[row * stride + column] }
        }
        let run = try GPU.run(
            ir: AtomicFixtures.rowUniform(keepOld: false),
            args: [.floats(tile), .output(count: rows), .int32(Int32(stride))])
        assertClose(
            GPU.read(run.outputs[0], Float.self, rows), expected, tolerance: 1e-6,
            "a uniform nest must perform the atomic once, not once per thread")
        XCTAssertTrue(
            run.source.contains("tm_thread_id.x % ") || run.source.contains("tm_thread_id.x == 0u"),
            run.source)
    }

    /// ...and because only one thread performs it, only that thread could be told
    /// what was there before. Using the result is refused by name.
    func testUsingTheOldValueFromAUniformNestIsRefused() {
        do {
            _ = try MetalCompiler.emitMSL(
                ttir: AtomicFixtures.rowUniform(keepOld: true), options: .init())
            XCTFail("expected a lowering error")
        } catch {
            XCTAssertTrue("\(error)".contains("exists only there"), "\(error)")
        }
    }

    // MARK: - What the lowering emits

    func testAtomicsLowerToRelaxedDeviceAtomics() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AtomicFixtures.scatter(kind: "fadd"), options: .init())
        XCTAssertTrue(
            source.contains("atomic_fetch_add_explicit((device atomic_float *)"), source)
        XCTAssertTrue(source.contains("memory_order_relaxed"), source)
        // The helpers are not dragged in by a kernel that does not need them.
        XCTAssertFalse(source.contains("tm_atomic_fmax"), source)
        XCTAssertFalse(source.contains("tm_atomic_cas"), source)
    }

    func testIntegerAtomicsPickTheSignedOrUnsignedAtomicType() throws {
        let signed = try MetalCompiler.emitMSL(
            ttir: AtomicFixtures.scatter(kind: "max", element: "i32"), options: .init())
        XCTAssertTrue(
            signed.contains("atomic_fetch_max_explicit((device atomic_int *)"), signed)
        let unsigned = try MetalCompiler.emitMSL(
            ttir: AtomicFixtures.scatter(kind: "umax", element: "i32"), options: .init())
        XCTAssertTrue(
            unsigned.contains("atomic_fetch_max_explicit((device atomic_uint *)"), unsigned)
        XCTAssertTrue(unsigned.contains("uint("), unsigned)
    }

    func testCompareAndSwapPullsInTheStrongLoopHelper() throws {
        let source = try MetalCompiler.emitMSL(
            ttir: AtomicFixtures.compareAndSwap(), options: .init())
        XCTAssertTrue(source.contains("inline int tm_atomic_cas"), source)
        XCTAssertTrue(
            source.contains("atomic_compare_exchange_weak_explicit"), source)
    }

    /// Every emitted atomic kernel is accepted by Metal's own front end.
    func testEveryAtomicKernelCompilesInMetal() throws {
        try skipWithoutMetal()
        var sources: [String] = []
        for kind in ["add", "max", "min", "umax", "umin", "and", "or", "xor", "exch"] {
            sources.append(
                try MetalCompiler.emitMSL(
                    ttir: AtomicFixtures.scatter(kind: kind, element: "i32"), options: .init()))
        }
        for kind in ["fadd", "max", "min", "exch"] {
            sources.append(
                try MetalCompiler.emitMSL(
                    ttir: AtomicFixtures.scatter(kind: kind, element: "f32"), options: .init()))
        }
        // The CAS helper carries both overloads, so this also puts the f32 one
        // through Metal's front end.
        sources.append(
            try MetalCompiler.emitMSL(ttir: AtomicFixtures.compareAndSwap(), options: .init()))
        for source in sources {
            XCTAssertNoThrow(try MetalCompiler.compileMSL(source), source)
        }
    }

    // MARK: - Refusals

    /// Metal's atomics are 32-bit. Both gaps are refused by name rather than
    /// lowered to something that is not atomic.
    func testUnsupportedAtomicTypesAreRefusedByName() {
        for (element, needle) in [
            ("f16", "no 16-bit atomics"),
            ("i64", "no 64-bit atomics"),
        ] {
            do {
                _ = try MetalCompiler.emitMSL(
                    ttir: AtomicFixtures.scatter(kind: element == "f16" ? "fadd" : "add",
                                                 element: element), options: .init())
                XCTFail("expected \(element) to be refused")
            } catch {
                XCTAssertTrue("\(error)".contains(needle), "\(element): \(error)")
            }
        }
    }

    func testIntegerOnlyOperationsAreRefusedOnFloatPointers() {
        do {
            _ = try MetalCompiler.emitMSL(
                ttir: AtomicFixtures.scatter(kind: "and", element: "f32"), options: .init())
            XCTFail("expected an error")
        } catch {
            XCTAssertTrue("\(error)".contains("integer-only"), "\(error)")
        }
    }

    func testFloatAddIsRefusedOnIntegerPointers() {
        do {
            _ = try MetalCompiler.emitMSL(
                ttir: AtomicFixtures.scatter(kind: "fadd", element: "i32"), options: .init())
            XCTFail("expected an error")
        } catch {
            XCTAssertTrue("\(error)".contains("floating-point operation"), "\(error)")
        }
    }

    // MARK: - Parsing

    func testParserReadsTheKindSemanticsAndScope() throws {
        let module = try TritonIRParser.parse(AtomicFixtures.scatter(kind: "fadd"))
        let atomic = module.functions[0].body.compactMap { instruction -> AtomicRMW? in
            if case .atomicRMW(let atomic) = instruction.kind { return atomic }
            return nil
        }
        XCTAssertEqual(atomic.count, 1)
        XCTAssertEqual(atomic[0].op, .fadd)
        XCTAssertEqual(atomic[0].semantic, .acq_rel)
        XCTAssertEqual(atomic[0].scope, .gpu)
        XCTAssertNotNil(atomic[0].mask)

        let unmasked = try TritonIRParser.parse(
            AtomicFixtures.scatter(kind: "fadd", masked: false))
        for instruction in unmasked.functions[0].body {
            if case .atomicRMW(let atomic) = instruction.kind { XCTAssertNil(atomic.mask) }
        }
    }

    func testUnknownAtomicKindIsAParseError() {
        do {
            _ = try TritonIRParser.parse(AtomicFixtures.scatter(kind: "fmul"))
            XCTFail("expected a parse error")
        } catch {
            XCTAssertTrue("\(error)".contains("unknown tt.atomic_rmw kind 'fmul'"), "\(error)")
        }
    }

    func testUnknownMemorySemanticsIsAParseError() {
        let ir = AtomicFixtures.scatter(kind: "fadd")
            .replacingOccurrences(of: "fadd, acq_rel, gpu", with: "fadd, weird, gpu")
        do {
            _ = try TritonIRParser.parse(ir)
            XCTFail("expected a parse error")
        } catch {
            XCTAssertTrue("\(error)".contains("unknown atomic memory semantics"), "\(error)")
        }
    }
}
