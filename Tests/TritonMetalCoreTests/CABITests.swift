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

    /// `tm_is_usable` is the stronger question the CI runners forced: a
    /// virtualised host reports a device and then cannot run what the emitter
    /// produces. The two answers must be consistent in the one direction that is
    /// guaranteed — an unusable machine is never a usable one — and the reason
    /// string must be present exactly when the verdict is negative.
    func testIsUsableAgreesWithItsReason() {
        XCTAssertEqual(tm_is_usable(), MetalRuntime.unusableReason == nil ? 1 : 0)
        XCTAssertEqual(takeString(tm_unusable_reason()), MetalRuntime.unusableReason)
        if tm_is_usable() == 1 { XCTAssertEqual(tm_is_active(), 1) }
    }

    func testDeviceNameIsReported() throws {
        try skipWithoutMetal()
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
        try skipWithoutMetal()
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
        try skipWithoutMetal()
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
        try skipWithoutMetal()
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
        try skipWithoutMetal()
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
        try skipWithoutMetal()
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

/// 64-bit scalar kernel arguments, and the one Metal simply does not have.
extension CABITests {

    /// `out[i] = i64_arg + i` through the whole ABI: emitted as `constant long &`,
    /// bound with launch kind 3, and read back exactly — a value that does not
    /// survive a round trip through 32 bits.
    func testI64ScalarArgumentsSurviveTheLaunchABI() throws {
        try skipWithoutMetal()
        let ir = """
            module {
              tt.func public @i64_kernel(%out: !tt.ptr<i64>, %base: i64, %n: i32) {
                %c8_i32 = arith.constant 8 : i32
                %pid = tt.get_program_id x : i32
                %off = arith.muli %pid, %c8_i32 : i32
                %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %off_b = tt.splat %off : i32 -> tensor<8xi32>
                %idx = arith.addi %off_b, %range : tensor<8xi32>
                %n_b = tt.splat %n : i32 -> tensor<8xi32>
                %mask = arith.cmpi slt, %idx, %n_b : tensor<8xi32>
                %idx64 = arith.extsi %idx : tensor<8xi32> to tensor<8xi64>
                %base_b = tt.splat %base : i64 -> tensor<8xi64>
                %value = arith.addi %base_b, %idx64 : tensor<8xi64>
                %p = tt.splat %out : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
                %ptrs = tt.addptr %p, %idx : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
                tt.store %ptrs, %value, %mask : tensor<8x!tt.ptr<i64>>
                tt.return
              }
            }
            """
        let source = try MetalCompiler.emitMSL(ttir: ir, options: .init())
        XCTAssertTrue(source.contains("constant long &"), source)

        // A value that needs all 64 bits: truncating it to 32 gives 0.
        let base: Int64 = 1 << 40
        let n = 24
        let run = try GPU.run(
            ir: ir, grid: (3, 1, 1),
            args: [.output(count: n, stride: 8), .int64(base), .int32(Int32(n))])
        XCTAssertEqual(
            GPU.read(run.outputs[0], Int64.self, n), (0..<n).map { base + Int64($0) })
    }

    /// Metal has no `double` — its front end says so outright ("'double' is not
    /// supported in Metal") — so there is no f64 launch-argument kind and an
    /// `f64` argument is refused by name rather than quietly narrowed to a float.
    /// The parser refuses the *type*; `MSLEmitter.scalarTypeName` refuses it
    /// again, so a width that reached the emitter another way cannot be widened
    /// away either.
    func testF64IsRefusedRatherThanNarrowed() throws {
        let ir = """
            module {
              tt.func public @f64_kernel(%out: !tt.ptr<f32>, %scale: f64) {
                %c0_i32 = arith.constant 0 : i32
                %p = tt.addptr %out, %c0_i32 : !tt.ptr<f32>, i32
                %v = arith.truncf %scale : f64 to f32
                tt.store %p, %v : !tt.ptr<f32>
                tt.return
              }
            }
            """
        do {
            _ = try MetalCompiler.emitMSL(ttir: ir, options: .init())
            XCTFail("expected f64 to be refused")
        } catch {
            XCTAssertTrue("\(error)".contains("f64"), "\(error)")
        }
        XCTAssertThrowsError(
            try MSLEmitter.metalTypeName(.float(width: 64))
        ) { error in
            XCTAssertTrue("\(error)".contains("Metal has no double"), "\(error)")
        }
    }
}
