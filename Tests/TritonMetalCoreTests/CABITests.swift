import XCTest

@testable import TritonMetalCore

/// Exercises the `tm_*` entry points exactly as the ctypes shim does: opaque
/// handles, malloc'd strings, sentinel returns plus `tm_last_error`.
final class CABITests: XCTestCase {

    private func takeString(_ ptr: UnsafeMutablePointer<CChar>?) -> String? {
        guard let ptr else { return nil }
        defer { tm_free(ptr) }
        return String(cString: ptr)
    }

    private func lastError() -> String {
        takeString(tm_last_error()) ?? ""
    }

    func testVersionRoundTrip() {
        XCTAssertEqual(takeString(tm_version()), "0.0.1")
    }

    func testIsActiveMatchesDevicePresence() {
        XCTAssertEqual(tm_is_active(), MetalRuntime.defaultDeviceName() != nil ? 1 : 0)
    }

    func testDeviceNameIsReported() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        XCTAssertEqual(takeString(tm_device_name()), MetalRuntime.defaultDeviceName())
    }

    func testEmitMSLThroughTheABI() throws {
        let msl = try XCTUnwrap(IRFixtures.vectorAdd.withCString { tm_emit_msl($0, 4) })
        let source = try XCTUnwrap(takeString(msl))
        XCTAssertTrue(source.contains("kernel void add_kernel("), source)
    }

    func testKernelInfoThroughTheABI() throws {
        let json = try XCTUnwrap(
            takeString(IRFixtures.vectorAdd.withCString { tm_kernel_info($0, 4) }))
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        let kernels = try XCTUnwrap(object?["kernels"] as? [[String: Any]])
        XCTAssertEqual(kernels[0]["name"] as? String, "add_kernel")
        XCTAssertEqual(kernels[0]["threads_per_threadgroup"] as? Int, 128)
    }

    func testEmitMSLReportsParseErrorsViaLastError() {
        let result = "module {}".withCString { tm_emit_msl($0, 4) }
        XCTAssertNil(result)
        XCTAssertTrue(lastError().contains("no tt.func"), lastError())
    }

    func testUnsupportedOpReachesLastError() {
        let ir = """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.histogram %arg0 : !tt.ptr<f32>
                tt.return
              }
            }
            """
        XCTAssertNil(ir.withCString { tm_emit_msl($0, 4) })
        XCTAssertTrue(lastError().contains("unsupported op 'tt.histogram'"), lastError())
    }

    func testInvalidHandlesAreRejected() {
        XCTAssertEqual(tm_load_kernel(999_999, "k"), 0)
        XCTAssertTrue(lastError().contains("library handle 999999 is not live"), lastError())

        XCTAssertEqual(tm_buffer_length(999_999), -1)
        XCTAssertTrue(lastError().contains("is not live"), lastError())

        XCTAssertEqual(tm_free_buffer(999_999), -1)
        XCTAssertEqual(tm_launch(999_999, 1, 1, 1, 32, nil, nil, 0), -1)
        XCTAssertTrue(lastError().contains("kernel handle"), lastError())
    }

    func testCompileErrorsSurfaceAsZeroHandle() {
        XCTAssertEqual("not msl at all".withCString { tm_compile_msl($0) }, 0)
        XCTAssertTrue(lastError().contains("MSL compilation failed"), lastError())
    }

    /// The whole spine through the C ABI only: emit -> compile -> load -> alloc ->
    /// write -> launch -> read -> release.
    func testVectorAddEndToEndThroughTheABI() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        let liveBefore = tm_live_handle_count()

        let source = try XCTUnwrap(takeString(IRFixtures.vectorAdd.withCString { tm_emit_msl($0, 4) }))
        let library = source.withCString { tm_compile_msl($0) }
        XCTAssertNotEqual(library, 0, lastError())
        let kernel = "add_kernel".withCString { tm_load_kernel(library, $0) }
        XCTAssertNotEqual(kernel, 0, lastError())
        XCTAssertGreaterThanOrEqual(tm_kernel_max_threads(kernel), 128)

        let n = 5000
        let a = (0..<n).map { Float($0) * 0.5 }
        let b = (0..<n).map { Float($0 % 13) }
        let bytes = Int64(n * 4)

        let bufferA = tm_alloc_buffer(bytes)
        let bufferB = tm_alloc_buffer(bytes)
        let bufferOut = tm_alloc_buffer(bytes)
        XCTAssertTrue(bufferA != 0 && bufferB != 0 && bufferOut != 0, lastError())
        XCTAssertEqual(tm_buffer_length(bufferA), bytes)

        a.withUnsafeBytes { XCTAssertEqual(tm_buffer_write(bufferA, 0, $0.baseAddress, bytes), 0) }
        b.withUnsafeBytes { XCTAssertEqual(tm_buffer_write(bufferB, 0, $0.baseAddress, bytes), 0) }

        let kinds: [Int32] = [0, 0, 0, 1]
        let values: [Int64] = [bufferA, bufferB, bufferOut, Int64(n)]
        let status = kinds.withUnsafeBufferPointer { kindPtr in
            values.withUnsafeBufferPointer { valuePtr in
                tm_launch(
                    kernel, Int64((n + 1023) / 1024), 1, 1, 128,
                    kindPtr.baseAddress, valuePtr.baseAddress, 4)
            }
        }
        XCTAssertEqual(status, 0, lastError())

        var out = [Float](repeating: .nan, count: n)
        out.withUnsafeMutableBytes {
            XCTAssertEqual(tm_buffer_read(bufferOut, 0, $0.baseAddress, bytes), 0)
        }
        XCTAssertEqual(out, zip(a, b).map(+))

        // Reading past the end is refused rather than corrupting the host.
        out.withUnsafeMutableBytes {
            XCTAssertEqual(tm_buffer_read(bufferOut, 4, $0.baseAddress, bytes), -1)
        }
        XCTAssertTrue(lastError().contains("exceeds the"), lastError())

        XCTAssertEqual(tm_free_buffer(bufferA), 0)
        XCTAssertEqual(tm_free_buffer(bufferB), 0)
        XCTAssertEqual(tm_free_buffer(bufferOut), 0)
        XCTAssertEqual(tm_release_kernel(kernel), 0)
        XCTAssertEqual(tm_release_library(library), 0)
        XCTAssertEqual(tm_live_handle_count(), liveBefore, "handles leaked")
    }

    func testUnknownLaunchArgumentKindIsRejected() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        let source = try XCTUnwrap(takeString(IRFixtures.copy.withCString { tm_emit_msl($0, 4) }))
        let library = source.withCString { tm_compile_msl($0) }
        let kernel = "copy_kernel".withCString { tm_load_kernel(library, $0) }
        defer {
            _ = tm_release_kernel(kernel)
            _ = tm_release_library(library)
        }
        let kinds: [Int32] = [7]
        let values: [Int64] = [0]
        let status = kinds.withUnsafeBufferPointer { kindPtr in
            values.withUnsafeBufferPointer { valuePtr in
                tm_launch(kernel, 1, 1, 1, 32, kindPtr.baseAddress, valuePtr.baseAddress, 1)
            }
        }
        XCTAssertEqual(status, -1)
        XCTAssertTrue(lastError().contains("unknown launch argument kind 7"), lastError())
    }

    /// A ctypes caller that passes `None` where a string is expected must get the
    /// documented sentinel and a message, not a crash. Every entry point taking a
    /// `const char *` checks it.
    func testNullStringArgumentsAreRejected() {
        XCTAssertNil(tm_emit_msl(nil, 4))
        XCTAssertTrue(lastError().contains("tm_emit_msl received a NULL module"), lastError())

        XCTAssertNil(tm_kernel_info(nil, 4))
        XCTAssertTrue(lastError().contains("tm_kernel_info received a NULL module"), lastError())

        XCTAssertEqual(tm_compile_msl(nil), 0)
        XCTAssertTrue(lastError().contains("tm_compile_msl received NULL source"), lastError())

        XCTAssertEqual(tm_load_kernel(1, nil), 0)
        XCTAssertTrue(lastError().contains("NULL kernel name"), lastError())
    }

    /// `tm_kernel_info` shares `tm_emit_msl`'s lowering, so it must report the
    /// same failures rather than returning an empty document.
    func testKernelInfoReportsLoweringErrors() {
        XCTAssertNil("module {}".withCString { tm_kernel_info($0, 4) })
        XCTAssertTrue(lastError().contains("no tt.func"), lastError())
    }

    func testInvalidAllocationsAndQueriesUseSentinels() {
        XCTAssertEqual(tm_alloc_buffer(0), 0)
        XCTAssertTrue(lastError().contains("buffer length must be positive"), lastError())

        XCTAssertEqual(tm_kernel_max_threads(999_999), -1)
        XCTAssertTrue(lastError().contains("kernel handle 999999 is not live"), lastError())

        XCTAssertNil(tm_buffer_contents(999_999))
        XCTAssertTrue(lastError().contains("is not live"), lastError())

        XCTAssertEqual(tm_release_kernel(999_999), -1)
        XCTAssertEqual(tm_release_library(999_999), -1)
    }

    /// `tm_buffer_contents` is the zero-copy path the shim uses to write unified
    /// memory in place, and the bounds guard on the copying path is what keeps a
    /// bad offset or length from scribbling on the host.
    func testBufferContentsAndCopyBounds() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        let buffer = tm_alloc_buffer(16)
        XCTAssertNotEqual(buffer, 0, lastError())
        defer { XCTAssertEqual(tm_free_buffer(buffer), 0) }

        let contents = try XCTUnwrap(tm_buffer_contents(buffer))
        contents.assumingMemoryBound(to: Float.self)[0] = 1.25
        var readBack: Float = 0
        XCTAssertEqual(tm_buffer_read(buffer, 0, &readBack, 4), 0)
        XCTAssertEqual(readBack, 1.25)

        // NULL host pointer, negative length and negative offset are all refused
        // before the handle is even looked up.
        XCTAssertEqual(tm_buffer_write(buffer, 0, nil, 4), -1)
        XCTAssertTrue(lastError().contains("invalid bounds"), lastError())
        XCTAssertEqual(tm_buffer_write(buffer, 0, &readBack, -1), -1)
        XCTAssertEqual(tm_buffer_read(buffer, -4, &readBack, 4), -1)
        XCTAssertTrue(lastError().contains("invalid bounds"), lastError())
    }

    /// The offline `xcrun metal` path feeds `tm_load_metallib`, which is how a
    /// cached `.metallib` re-enters the runtime without recompiling MSL.
    func testMetallibLoadingThroughTheABI() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        let source = try MetalCompiler.emitMSL(ttir: IRFixtures.copy, options: .init())
        let image: Data
        do {
            image = try MetalCompiler.compileMSLOffline(source)
        } catch {
            throw XCTSkip("xcrun metal unavailable: \(error)")
        }

        let library = image.withUnsafeBytes { tm_load_metallib($0.baseAddress, Int64(image.count)) }
        XCTAssertNotEqual(library, 0, lastError())
        let kernel = "copy_kernel".withCString { tm_load_kernel(library, $0) }
        XCTAssertNotEqual(kernel, 0, lastError())
        XCTAssertEqual(tm_release_kernel(kernel), 0)
        XCTAssertEqual(tm_release_library(library), 0)

        // An empty image, and bytes that are not a metallib at all.
        XCTAssertEqual(tm_load_metallib(nil, 0), 0)
        XCTAssertTrue(lastError().contains("empty image"), lastError())
        var garbage: [UInt8] = Array(repeating: 0xAB, count: 64)
        XCTAssertEqual(tm_load_metallib(&garbage, 64), 0)
        XCTAssertTrue(lastError().contains("could not load metallib"), lastError())
    }

    /// f32 scalars reach the kernel as a bit pattern in the low 32 bits of the
    /// `values` array — the one launch argument kind the vector-add path misses.
    func testFloatScalarLaunchArguments() throws {
        try XCTSkipIf(MetalRuntime.device == nil, "no Metal device")
        let source = try XCTUnwrap(
            takeString(IRFixtures.scaleBias.withCString { tm_emit_msl($0, 4) }))
        let library = source.withCString { tm_compile_msl($0) }
        let kernel = "scale_bias_kernel".withCString { tm_load_kernel(library, $0) }
        XCTAssertNotEqual(kernel, 0, lastError())
        defer {
            _ = tm_release_kernel(kernel)
            _ = tm_release_library(library)
        }

        let n = 64
        let input = (0..<n).map { Float($0) }
        let bytes = Int64(n * 4)
        let bufferIn = tm_alloc_buffer(bytes)
        let bufferOut = tm_alloc_buffer(bytes)
        defer {
            _ = tm_free_buffer(bufferIn)
            _ = tm_free_buffer(bufferOut)
        }
        input.withUnsafeBytes { XCTAssertEqual(tm_buffer_write(bufferIn, 0, $0.baseAddress, bytes), 0) }

        let scale: Float = 2.5
        let bias: Float = -1.75
        let kinds: [Int32] = [0, 0, 2, 2, 1]
        let values: [Int64] = [
            bufferIn, bufferOut, Int64(scale.bitPattern), Int64(bias.bitPattern), Int64(n),
        ]
        let status = kinds.withUnsafeBufferPointer { kindPtr in
            values.withUnsafeBufferPointer { valuePtr in
                tm_launch(kernel, 1, 1, 1, 128, kindPtr.baseAddress, valuePtr.baseAddress, 5)
            }
        }
        XCTAssertEqual(status, 0, lastError())

        var out = [Float](repeating: .nan, count: n)
        out.withUnsafeMutableBytes {
            XCTAssertEqual(tm_buffer_read(bufferOut, 0, $0.baseAddress, bytes), 0)
        }
        XCTAssertEqual(out, input.map { $0 * scale + bias })

        // A non-zero argument count with NULL arrays is a caller bug, not a crash.
        XCTAssertEqual(tm_launch(kernel, 1, 1, 1, 128, nil, nil, 5), -1)
        XCTAssertTrue(lastError().contains("NULL argument arrays"), lastError())
    }

    /// `tm_last_error` reports non-`CoreError` failures too — anything thrown out
    /// of Foundation or Metal still has to reach the ctypes caller as a string.
    func testLastErrorCarriesNonCoreErrors() {
        struct Odd: Error, CustomStringConvertible { var description: String { "odd failure" } }
        LastError.store(Odd())
        XCTAssertTrue(lastError().contains("odd failure"), lastError())
        // Taking it clears it.
        XCTAssertEqual(lastError(), "no error")
    }
}
