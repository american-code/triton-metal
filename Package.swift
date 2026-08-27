// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "triton-metal",
    platforms: [.macOS(.v14)],
    products: [
        // Dynamic library with a C ABI (`tm_*` symbols). The Python shim in
        // python/ binds to this via ctypes; all real work happens here.
        .library(name: "tritonmetal", type: .dynamic, targets: ["TritonMetalCore"]),
        // The GEMM sweep as an executable, so it can run on a machine that has
        // the command-line tools but no Xcode (and therefore no XCTest).
        .executable(name: "tmbench", targets: ["tmbench"]),
    ],
    targets: [
        .target(name: "TritonMetalCore"),
        .target(name: "TritonMetalBench", dependencies: ["TritonMetalCore"]),
        .executableTarget(name: "tmbench", dependencies: ["TritonMetalBench"]),
        .testTarget(
            name: "TritonMetalCoreTests",
            dependencies: ["TritonMetalCore", "TritonMetalBench"]),
    ]
)
