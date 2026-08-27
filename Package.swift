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
        // The MLX frontend. A separate product over a separate target, because
        // it is the only thing in the package with an external dependency:
        // linking `tritonmetal` must never drag mlx-swift in behind it.
        .library(name: "TritonMetalMLX", targets: ["TritonMetalMLX"]),
        // The SAE-encoder demo and the dispatch-overhead measurement, as an
        // executable — the lab nodes have the command-line tools but no XCTest.
        .executable(name: "tmsae", targets: ["tmsae"]),
    ],
    dependencies: [
        // Reached exclusively from `TritonMetalMLX` and `tmsae`. `TritonMetalCore`
        // is not allowed to depend on it; the arrow only points one way.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0")
    ],
    targets: [
        .target(name: "TritonMetalCore"),
        .target(name: "TritonMetalBench", dependencies: ["TritonMetalCore"]),
        .executableTarget(name: "tmbench", dependencies: ["TritonMetalBench"]),
        // The MLX seam.
        .target(
            name: "TritonMetalMLX",
            dependencies: [
                "TritonMetalCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "tmsae",
            dependencies: [
                "TritonMetalMLX", "TritonMetalCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "TritonMetalCoreTests",
            dependencies: ["TritonMetalCore", "TritonMetalBench"]),
        // The frontend's tests live apart from `TritonMetalCoreTests` on purpose:
        // the core suite must keep running on a machine with no mlx.metallib, and
        // must not acquire an MLX dependency by association.
        .testTarget(
            name: "TritonMetalMLXTests",
            dependencies: [
                "TritonMetalMLX", "TritonMetalCore",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
