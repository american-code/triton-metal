import Foundation
import TritonMetalBench
import XCTest

@testable import TritonMetalCore

/// Square-GEMM throughput of the lowered matmul tutorial against
/// `MPSMatrixMultiplication`, the fastest thing Apple ships for the same job.
///
/// Off by default — it takes seconds and it measures a machine, not a contract.
/// Run it with `TM_BENCH=1 swift test --filter MatmulBenchmark`.
///
/// The measurement itself lives in `TritonMetalBench` and is shared with the
/// `tmbench` executable, which is how the same sweep runs on a machine that has
/// only the command-line tools installed (no Xcode, therefore no XCTest). This
/// test is a wrapper so that `swift test` still exercises the path.
final class MatmulBenchmark: XCTestCase {

    func testMatmulThroughputAgainstMPS() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["TM_BENCH"] == nil,
            "set TM_BENCH=1 to run the MPS comparison")
        try skipWithoutMetal()

        let harness = try GEMMBenchmark.Harness()
        print("device: \(harness.device.name)")
        let sweep: GEMMBenchmark.Sweep =
            ProcessInfo.processInfo.environment["TM_BENCH_FULL"] == nil ? .quick : .full
        let results = try GEMMBenchmark.run(
            sizes: [512, 1024, 2048], configurations: GEMMBenchmark.configurations(sweep),
            harness: harness)

        print(GEMMBenchmark.header)
        for result in results {
            print(GEMMBenchmark.row(result))
            XCTAssertGreaterThan(result.gflops, 0)
            XCTAssertNil(
                try GEMMBenchmark.verify(result.config, harness: harness),
                "the fastest configuration at \(result.size) computes the wrong answer")
        }
    }

    /// The autotuner's config names have to survive a round trip, because that is
    /// how a measured winner gets pinned with `tmbench --config`.
    func testConfigNamesRoundTrip() throws {
        for text in ["64,64,32,16", "128,64,16,8,2,4"] {
            let config = try XCTUnwrap(GEMMConfig.parse(text))
            let fields = text.split(separator: ",").compactMap { Int($0) }
            XCTAssertEqual(config.blockM, fields[0])
            XCTAssertEqual(config.blockN, fields[1])
            XCTAssertEqual(config.blockK, fields[2])
            XCTAssertEqual(config.simdgroups, fields[3])
            if fields.count == 6 {
                XCTAssertEqual(config.registerM, fields[4])
                XCTAssertEqual(config.registerN, fields[5])
                XCTAssertTrue(config.name.hasSuffix("/r\(fields[4])x\(fields[5])"))
            }
        }
        XCTAssertNil(GEMMConfig.parse("64,64,32"))
        XCTAssertNil(GEMMConfig.parse("64,64,32,16,2"))
        XCTAssertNil(GEMMConfig.parse("sixty-four"))
    }
}
