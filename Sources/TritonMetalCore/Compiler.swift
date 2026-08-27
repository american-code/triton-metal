import Foundation
import Metal

/// The compiler core: lowers Triton IR (received as serialized MLIR text from the
/// Python shim) to Metal Shading Language, then to a loaded Metal library.
///
/// Pipeline (docs/ARCHITECTURE.md §Lowering):
///     ttir/ttgir (text, from shim) -> MSL source -> MTLLibrary (or .metallib bytes)
///
/// The ttir->ttgir stages stay on the Triton side (they're Triton's own MLIR
/// passes); this core owns everything Metal-specific.
public enum MetalCompiler {
    public struct Options: Sendable {
        /// Triton's `num_warps`. A Metal simdgroup is 32 lanes wide, so the
        /// threadgroup gets `numSimdgroups * 32` threads (capped by the block size
        /// and Metal's 1024-thread limit).
        public var numSimdgroups: Int

        /// `tt.dot` register blocking: how many 8x8 output fragments each
        /// simdgroup accumulates along M and along N. Zero means "choose", which
        /// is what every caller except the autotuner wants — see
        /// `MSLEmitter`'s `registerBlocking`.
        public var dotRegisterM: Int
        public var dotRegisterN: Int

        /// How many consecutive columns one thread stages into a `tt.dot` tile.
        /// Zero means "choose" — see `MSLEmitter`'s `stagingUnroll`.
        public var dotStagingUnroll: Int

        /// Elements of slack after each row of a `tt.dot` tile, to skew its rows
        /// across threadgroup-memory banks. Negative means "choose" — see
        /// `MSLEmitter`'s `rowPadding`.
        public var dotTilePadding: Int

        /// Stage each contraction step into the other of two threadgroup buffers,
        /// so a step's staging does not have to wait for the previous step's
        /// arithmetic to finish reading. Metal has no `cp.async`, so this is plain
        /// manual double buffering; it costs twice the operand tile memory.
        public var dotDoubleBuffer: Bool

        public init(
            numSimdgroups: Int = 4, dotRegisterM: Int = 0, dotRegisterN: Int = 0,
            dotStagingUnroll: Int = 0, dotTilePadding: Int = -1, dotDoubleBuffer: Bool = false
        ) {
            self.numSimdgroups = numSimdgroups
            self.dotRegisterM = dotRegisterM
            self.dotRegisterN = dotRegisterN
            self.dotStagingUnroll = dotStagingUnroll
            self.dotTilePadding = dotTilePadding
            self.dotDoubleBuffer = dotDoubleBuffer
        }
    }

    /// Parse + lower an IR module, returning MSL source and per-kernel launch
    /// metadata (block size, threadgroup size, argument kinds).
    public static func emit(ttir: String, options: Options) throws -> EmissionResult {
        let module = try TritonIRParser.parse(ttir)
        return try MSLEmitter.emit(module: module, options: options)
    }

    /// Emit MSL source for a Triton IR module.
    public static func emitMSL(ttir: String, options: Options) throws -> String {
        try emit(ttir: ttir, options: options).source
    }

    /// Compile MSL source into a library the runtime can build pipelines from.
    /// Runtime compilation (`MTLDevice.makeLibrary(source:)`) is the primary path:
    /// it needs no Xcode command-line tools and no temporary files.
    public static func compileMSL(_ source: String) throws -> MTLLibrary {
        try MetalRuntime.makeLibrary(source: source)
    }

    /// Compile MSL and immediately resolve one kernel's pipeline.
    public static func compileMSL(_ source: String, kernelName: String) throws
        -> MTLComputePipelineState
    {
        try MetalRuntime.loadKernel(library: try compileMSL(source), kernelName: kernelName)
    }

    /// Optional offline path: shell out to `xcrun metal` to produce real
    /// `.metallib` bytes (cacheable on disk, loadable via `makeLibrary(metallib:)`).
    /// Never uses xcodebuild. Throws if the Metal toolchain is unavailable.
    public static func compileMSLOffline(_ source: String, standard: String = "metal3.1") throws
        -> Data
    {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("triton-metal-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let sourceURL = directory.appendingPathComponent("kernel.metal")
        let outputURL = directory.appendingPathComponent("kernel.metallib")
        try source.write(to: sourceURL, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = [
            "metal", "-std=\(standard)", "-o", outputURL.path, sourceURL.path,
        ]
        let stderr = Pipe()
        process.standardError = stderr
        process.standardOutput = Pipe()

        do {
            try process.run()
        } catch {
            throw CoreError.metal("could not run `xcrun metal`: \(error.localizedDescription)")
        }
        let diagnostics = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let text = String(data: diagnostics, encoding: .utf8) ?? ""
            throw CoreError.metal(
                "`xcrun metal` failed (status \(process.terminationStatus)): \(text)")
        }
        return try Data(contentsOf: outputURL)
    }
}

// MARK: - Kernel metadata (JSON)

extension EmissionResult {
    /// Launch metadata for the Python shim, which must not compute it itself.
    func kernelInfoJSON() throws -> String {
        let payload: [String: Any] = [
            "kernels": kernels.map { kernel in
                [
                    "name": kernel.name,
                    "block_size": kernel.blockSize,
                    "block_shape": kernel.blockShape,
                    "threads_per_threadgroup": kernel.threadsPerThreadgroup,
                    "args": kernel.arguments.map {
                        ["index": $0.index, "kind": $0.kind.rawValue, "dtype": $0.dtype]
                            as [String: Any]
                    },
                ] as [String: Any]
            }
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.sortedKeys, .prettyPrinted])
        guard let text = String(data: data, encoding: .utf8) else {
            throw CoreError.lowering("could not serialize kernel metadata", .unknown)
        }
        return text
    }
}

// MARK: - C ABI (what python/triton_metal/_core.py binds via ctypes)
//
// Conventions:
//   * `char *` results are malloc'd; the caller frees them with `tm_free`.
//   * Handle-returning calls return 0 on failure; `int32` calls return -1.
//   * On any failure the reason is available from `tm_last_error` (which clears it).

/// Returns a malloc'd C string the caller must free with `tm_free`.
@_cdecl("tm_version")
public func tm_version() -> UnsafeMutablePointer<CChar> {
    strdup("0.0.1")
}

@_cdecl("tm_free")
public func tm_free(_ ptr: UnsafeMutableRawPointer?) {
    free(ptr)
}

/// Lower Triton IR text to MSL. Returns a malloc'd C string (caller frees via
/// `tm_free`), or NULL on failure with the error in `tm_last_error`.
@_cdecl("tm_emit_msl")
public func tm_emit_msl(_ ttir: UnsafePointer<CChar>?, _ numSimdgroups: Int32)
    -> UnsafeMutablePointer<CChar>?
{
    guard let ttir else {
        LastError.store(CoreError.invalidArgument("tm_emit_msl received a NULL module"))
        return nil
    }
    do {
        let msl = try MetalCompiler.emitMSL(
            ttir: String(cString: ttir),
            options: .init(numSimdgroups: Int(numSimdgroups))
        )
        return strdup(msl)
    } catch {
        LastError.store(error)
        return nil
    }
}

/// Lower Triton IR text and return launch metadata as JSON:
/// `{"kernels":[{"name":..,"block_size":..,"threads_per_threadgroup":..,"args":[..]}]}`.
@_cdecl("tm_kernel_info")
public func tm_kernel_info(_ ttir: UnsafePointer<CChar>?, _ numSimdgroups: Int32)
    -> UnsafeMutablePointer<CChar>?
{
    guard let ttir else {
        LastError.store(CoreError.invalidArgument("tm_kernel_info received a NULL module"))
        return nil
    }
    do {
        let result = try MetalCompiler.emit(
            ttir: String(cString: ttir),
            options: .init(numSimdgroups: Int(numSimdgroups))
        )
        return strdup(try result.kernelInfoJSON())
    } catch {
        LastError.store(error)
        return nil
    }
}

/// Compile MSL source at runtime. Returns a library handle, or 0 on failure.
@_cdecl("tm_compile_msl")
public func tm_compile_msl(_ source: UnsafePointer<CChar>?) -> Int64 {
    guard let source else {
        LastError.store(CoreError.invalidArgument("tm_compile_msl received NULL source"))
        return 0
    }
    do {
        return tmHandles.insert(try MetalCompiler.compileMSL(String(cString: source)))
    } catch {
        LastError.store(error)
        return 0
    }
}

/// Load a precompiled `.metallib` image. Returns a library handle, or 0.
@_cdecl("tm_load_metallib")
public func tm_load_metallib(_ bytes: UnsafeRawPointer?, _ length: Int64) -> Int64 {
    guard let bytes, length > 0 else {
        LastError.store(CoreError.invalidArgument("tm_load_metallib received an empty image"))
        return 0
    }
    do {
        let data = Data(bytes: bytes, count: Int(length))
        return tmHandles.insert(try MetalRuntime.makeLibrary(metallib: data))
    } catch {
        LastError.store(error)
        return 0
    }
}

/// Build a compute pipeline for `name` in `library`. Returns a kernel handle, or 0.
@_cdecl("tm_load_kernel")
public func tm_load_kernel(_ library: Int64, _ name: UnsafePointer<CChar>?) -> Int64 {
    guard let name else {
        LastError.store(CoreError.invalidArgument("tm_load_kernel received a NULL kernel name"))
        return 0
    }
    do {
        let library = try tmHandles.lookup(library, as: MTLLibrary.self, what: "library")
        let pipeline = try MetalRuntime.loadKernel(
            library: library, kernelName: String(cString: name))
        return tmHandles.insert(pipeline)
    } catch {
        LastError.store(error)
        return 0
    }
}

/// Hardware limit for a loaded kernel's threadgroup size, or -1 on failure.
@_cdecl("tm_kernel_max_threads")
public func tm_kernel_max_threads(_ kernel: Int64) -> Int64 {
    do {
        let pipeline = try tmHandles.lookup(kernel, as: MTLComputePipelineState.self, what: "kernel")
        return Int64(pipeline.maxTotalThreadsPerThreadgroup)
    } catch {
        LastError.store(error)
        return -1
    }
}

/// Allocate a unified-memory (`.storageModeShared`) buffer. Returns a handle, or 0.
@_cdecl("tm_alloc_buffer")
public func tm_alloc_buffer(_ length: Int64) -> Int64 {
    do {
        return tmHandles.insert(try MetalRuntime.makeBuffer(length: Int(length)))
    } catch {
        LastError.store(error)
        return 0
    }
}

/// Host pointer to a shared buffer's contents (valid until the buffer is freed).
@_cdecl("tm_buffer_contents")
public func tm_buffer_contents(_ buffer: Int64) -> UnsafeMutableRawPointer? {
    do {
        return try tmHandles.lookup(buffer, as: MTLBuffer.self, what: "buffer").contents()
    } catch {
        LastError.store(error)
        return nil
    }
}

@_cdecl("tm_buffer_length")
public func tm_buffer_length(_ buffer: Int64) -> Int64 {
    do {
        return Int64(try tmHandles.lookup(buffer, as: MTLBuffer.self, what: "buffer").length)
    } catch {
        LastError.store(error)
        return -1
    }
}

/// Copy `length` bytes from `source` into the buffer at `offset`. 0 on success, -1 on failure.
@_cdecl("tm_buffer_write")
public func tm_buffer_write(
    _ buffer: Int64, _ offset: Int64, _ source: UnsafeRawPointer?, _ length: Int64
) -> Int32 {
    tmCopy(buffer: buffer, offset: offset, length: length, host: source) { device, host, count in
        device.copyMemory(from: host, byteCount: count)
    }
}

/// Copy `length` bytes out of the buffer at `offset` into `destination`.
@_cdecl("tm_buffer_read")
public func tm_buffer_read(
    _ buffer: Int64, _ offset: Int64, _ destination: UnsafeMutableRawPointer?, _ length: Int64
) -> Int32 {
    tmCopy(
        buffer: buffer, offset: offset, length: length,
        host: destination.map(UnsafeRawPointer.init)
    ) { device, host, count in
        UnsafeMutableRawPointer(mutating: host).copyMemory(from: device, byteCount: count)
    }
}

private func tmCopy(
    buffer: Int64, offset: Int64, length: Int64, host: UnsafeRawPointer?,
    _ body: (UnsafeMutableRawPointer, UnsafeRawPointer, Int) -> Void
) -> Int32 {
    guard let host, length >= 0, offset >= 0 else {
        LastError.store(CoreError.invalidArgument("buffer copy received invalid bounds"))
        return -1
    }
    do {
        let mtlBuffer = try tmHandles.lookup(buffer, as: MTLBuffer.self, what: "buffer")
        guard Int(offset) + Int(length) <= mtlBuffer.length else {
            throw CoreError.invalidArgument(
                "copy of \(length) bytes at offset \(offset) exceeds the "
                    + "\(mtlBuffer.length)-byte buffer")
        }
        body(mtlBuffer.contents().advanced(by: Int(offset)), host, Int(length))
        return 0
    } catch {
        LastError.store(error)
        return -1
    }
}

@_cdecl("tm_free_buffer")
public func tm_free_buffer(_ buffer: Int64) -> Int32 {
    tmHandles.remove(buffer) ? 0 : -1
}

@_cdecl("tm_release_kernel")
public func tm_release_kernel(_ kernel: Int64) -> Int32 {
    tmHandles.remove(kernel) ? 0 : -1
}

@_cdecl("tm_release_library")
public func tm_release_library(_ library: Int64) -> Int32 {
    tmHandles.remove(library) ? 0 : -1
}

/// Dispatch `kernel` over a `gridX * gridY * gridZ` grid of threadgroups, each of
/// `threadsPerThreadgroup` threads, then wait for completion.
///
/// Arguments are described by two parallel arrays of length `argCount`, bound at
/// buffer index `i` (== the `tt.func` argument position):
///   kind 0: `values[i]` is a buffer handle from `tm_alloc_buffer`
///   kind 1: `values[i]` is an i32 scalar (sign-extended into the low 32 bits)
///   kind 2: `values[i]` is the bit pattern of an f32 scalar in the low 32 bits
///
/// Returns 0 on success, -1 on failure (see `tm_last_error`).
@_cdecl("tm_launch")
public func tm_launch(
    _ kernel: Int64,
    _ gridX: Int64, _ gridY: Int64, _ gridZ: Int64,
    _ threadsPerThreadgroup: Int64,
    _ kinds: UnsafePointer<Int32>?,
    _ values: UnsafePointer<Int64>?,
    _ argCount: Int32
) -> Int32 {
    do {
        let pipeline = try tmHandles.lookup(kernel, as: MTLComputePipelineState.self, what: "kernel")
        var arguments: [MetalRuntime.LaunchArgument] = []
        if argCount > 0 {
            guard let kinds, let values else {
                throw CoreError.invalidArgument("tm_launch received NULL argument arrays")
            }
            for i in 0..<Int(argCount) {
                let value = values[i]
                switch kinds[i] {
                case 0:
                    arguments.append(
                        .buffer(try tmHandles.lookup(value, as: MTLBuffer.self, what: "buffer")))
                case 1:
                    arguments.append(.int32(Int32(truncatingIfNeeded: value)))
                case 2:
                    arguments.append(
                        .float32(Float(bitPattern: UInt32(truncatingIfNeeded: value))))
                default:
                    throw CoreError.invalidArgument(
                        "unknown launch argument kind \(kinds[i]) at position \(i)")
                }
            }
        }
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(width: Int(gridX), height: Int(gridY), depth: Int(gridZ)),
            threadsPerThreadgroup: Int(threadsPerThreadgroup),
            arguments: arguments)
        return 0
    } catch {
        LastError.store(error)
        return -1
    }
}

/// Number of live handles — a leak check for tests.
@_cdecl("tm_live_handle_count")
public func tm_live_handle_count() -> Int64 {
    Int64(tmHandles.count)
}

@_cdecl("tm_last_error")
public func tm_last_error() -> UnsafeMutablePointer<CChar> {
    strdup(LastError.take() ?? "no error")
}

enum LastError {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var message: String?

    static func store(_ error: Error) {
        lock.lock(); defer { lock.unlock() }
        if let core = error as? CoreError {
            message = core.description
        } else {
            message = String(describing: error)
        }
    }

    static func take() -> String? {
        lock.lock(); defer { lock.unlock() }
        defer { message = nil }
        return message
    }
}
