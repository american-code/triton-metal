import XCTest
import TritonMetalBench
@testable import TritonMetalCore

final class ProbeBlockTests: XCTestCase {
    func testBlockCollision() throws {
        for (m,n,k) in [(64,64,32),(64,32,64),(32,64,64),(64,64,64),(128,64,32)] {
            do { _ = try MetalCompiler.emitMSL(ttir: GEMMKernel.tutorial(blockM: m, blockN: n, blockK: k), options: .init())
                 print("[\(m)x\(n)x\(k)] OK") }
            catch { print("[\(m)x\(n)x\(k)] \(error)") }
        }
    }
}
