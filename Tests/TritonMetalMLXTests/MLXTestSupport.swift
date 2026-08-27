import Foundation
import MLX
import TritonMetalCore
import XCTest

// Support for the MLX frontend's tests.
//
// Two gates, and every test in this bundle passes through both.
//
// **The Metal device.** `TritonMetalCoreTests` already gates its GPU cases on
// `MetalRuntime.unusableReason`, because a device object existing is not the same
// as a device that runs what this backend emits — a virtualised host (a GitHub
// Actions macOS runner is the case that forced it) reports an "Apple Paravirtual
// device" that answers every query and then produces nothing.
//
// **MLX's shader library.** mlx-swift compiles the MLX C++ core but does not
// build `mlx.metallib`: that is Xcode's job, and `swift build` never runs it. A
// binary without it links and starts, then *aborts the process* the first time it
// touches the GPU — the failure is fatal rather than throwable, so it cannot be
// caught and turned into a test failure. The file's presence is therefore checked
// before any MLX call is made, and a missing one is a skip carrying the command
// that fixes it.
//
// The check pattern is adapted from mccl's `MCCLMLXTests/MLXTestSupport.swift`,
// which solved this first for the same reason.

enum MLXRuntime {

    /// Directories MLX will search for `mlx.metallib`, in the order it searches.
    ///
    /// MLX resolves the library relative to the binary containing the MLX code
    /// (`dladdr` on one of its own symbols). Under SwiftPM's static linking that
    /// is the test bundle's executable, so its directory is the one that matters
    /// — `.build/<config>/triton-metalPackageTests.xctest/Contents/MacOS`.
    static var searchDirectories: [URL] {
        var directories: [URL] = []
        func add(_ url: URL?) {
            guard let url, !directories.contains(url) else { return }
            directories.append(url)
        }
        if let override = ProcessInfo.processInfo.environment["TM_MLX_METALLIB_DIR"] {
            add(URL(fileURLWithPath: override))
        }
        add(Bundle.main.executableURL?.deletingLastPathComponent())
        for bundle in Bundle.allBundles {
            add(bundle.executableURL?.deletingLastPathComponent())
            add(bundle.resourceURL)
        }
        return directories
    }

    static var metallibPath: URL? {
        searchDirectories
            .map { $0.appendingPathComponent("mlx.metallib") }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Call at the top of every test in this bundle, before touching MLX.
    static func require() throws {
        if let reason = MetalRuntime.unusableReason {
            throw XCTSkip("this machine cannot run an emitted kernel: \(reason)")
        }
        guard metallibPath == nil else { return }
        throw XCTSkip(
            """
            mlx.metallib is not installed, so MLX cannot run a single GPU op and this test \
            would abort the process rather than fail.

            Install it with:    Tools/fetch-metallib.sh

            mlx-swift compiles the MLX C++ core but does not build the Metal shader library — \
            that is Xcode's job, and these tests run under plain `swift test`. The script \
            fetches the version-matched library out of the mlx-metal pip wheel. Searched:
            \(searchDirectories.map { "  " + $0.path }.joined(separator: "\n"))
            """)
    }
}

/// Deterministic test data. A seeded MLX RNG would make these tests depend on
/// MLX's RNG stream, which changes without warning and has nothing to do with
/// what is being checked.
func synthetic(_ shape: [Int], offset: Int = 0, modulus: Int = 71) -> MLXArray {
    let count = shape.reduce(1, *)
    var values = [Float]()
    values.reserveCapacity(count)
    for i in 0..<count {
        values.append(Float(((i + offset) * 13) % modulus) / Float(modulus) - 0.5)
    }
    return MLXArray(values, shape)
}

/// Largest absolute difference between two arrays, as a `Float`.
func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a - b).max().item(Float.self)
}

/// Asserts that `error` is a `CoreError` whose message mentions `fragment`.
func assertCoreError(
    _ expression: @autoclosure () throws -> Any, contains fragment: String,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertThrowsError(try expression(), file: file, line: line) { error in
        guard let core = error as? CoreError else {
            return XCTFail("expected a CoreError, got \(error)", file: file, line: line)
        }
        XCTAssertTrue(
            core.description.contains(fragment),
            "expected the message to mention '\(fragment)': \(core.description)",
            file: file, line: line)
    }
}
