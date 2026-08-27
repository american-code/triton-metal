import Foundation
import Metal

/// The runtime core: device queries, unified-memory buffers, library loading,
/// pipeline creation and kernel launch. The Python driver shim calls into this
/// through the `tm_*` C ABI; no compute or data movement happens in Python.
public enum MetalRuntime {
    /// One process-wide device + queue. Metal objects are internally synchronized;
    /// the handle table below serializes the bookkeeping around them.
    public static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let queue: MTLCommandQueue? = device?.makeCommandQueue()

    public static func defaultDeviceName() -> String? {
        device?.name
    }

    static func requireDevice() throws -> MTLDevice {
        guard let device else {
            throw CoreError.metal("no Metal device available on this system")
        }
        return device
    }

    static func requireQueue() throws -> MTLCommandQueue {
        guard let queue else {
            throw CoreError.metal("could not create a Metal command queue")
        }
        return queue
    }

    // MARK: - Libraries and pipelines

    /// Compiles MSL source at runtime (`MTLDevice.makeLibrary(source:options:)`).
    ///
    /// Fast math is switched **off**: Metal defaults it on, which would make
    /// `precise::exp`, division and denormal handling diverge from the CPU
    /// references the backend is validated against. Correctness first; a
    /// fast-math option belongs next to Triton's own `num_stages`/`num_warps`
    /// knobs, not in the default path.
    public static func makeLibrary(source: String) throws -> MTLLibrary {
        let device = try requireDevice()
        let options = MTLCompileOptions()
        if #available(macOS 15.0, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            throw CoreError.metal("MSL compilation failed: \(error.localizedDescription)")
        }
    }

    /// Loads a precompiled `.metallib` image (the `xcrun metal` offline path).
    public static func makeLibrary(metallib: Data) throws -> MTLLibrary {
        let device = try requireDevice()
        do {
            return try metallib.withUnsafeBytes { raw -> MTLLibrary in
                let dispatchData = DispatchData(bytes: raw)
                return try device.makeLibrary(data: dispatchData as __DispatchData)
            }
        } catch {
            throw CoreError.metal("could not load metallib: \(error.localizedDescription)")
        }
    }

    /// Builds a compute pipeline for `kernelName` in an already-loaded library.
    public static func loadKernel(library: MTLLibrary, kernelName: String) throws
        -> MTLComputePipelineState
    {
        guard let function = library.makeFunction(name: kernelName) else {
            let available = library.functionNames.sorted().joined(separator: ", ")
            throw CoreError.metal(
                "kernel '\(kernelName)' not found in library (available: \(available.isEmpty ? "none" : available))"
            )
        }
        do {
            return try (try requireDevice()).makeComputePipelineState(function: function)
        } catch let error as CoreError {
            throw error
        } catch {
            throw CoreError.metal(
                "could not build a pipeline for '\(kernelName)': \(error.localizedDescription)")
        }
    }

    /// Load a metallib and prepare a compute pipeline for `kernelName`.
    public static func loadKernel(metallib: Data, kernelName: String) throws
        -> MTLComputePipelineState
    {
        try loadKernel(library: try makeLibrary(metallib: metallib), kernelName: kernelName)
    }

    // MARK: - Buffers

    /// Unified memory: `.storageModeShared` buffers are visible to both the CPU
    /// and the GPU on Apple silicon, so there is no staging copy.
    public static func makeBuffer(length: Int) throws -> MTLBuffer {
        guard length > 0 else {
            throw CoreError.invalidArgument("buffer length must be positive, got \(length)")
        }
        guard let buffer = try requireDevice().makeBuffer(length: length, options: .storageModeShared)
        else {
            throw CoreError.metal("could not allocate a \(length)-byte shared buffer")
        }
        return buffer
    }

    // MARK: - Launch

    public enum LaunchArgument {
        case buffer(MTLBuffer)
        case int32(Int32)
        /// A 64-bit integer scalar, bound as `constant long &`. Metal has `long`;
        /// it has no `double`, so there is deliberately no `float64` case here
        /// (see `MSLEmitter`'s refusal of `f64`).
        case int64(Int64)
        case float32(Float)
    }

    /// Encodes and runs one dispatch, then blocks until the GPU is done. Arguments
    /// are bound at their list position, matching the emitted `[[buffer(N)]]`
    /// indices (which are the `tt.func` argument positions).
    public static func launch(
        pipeline: MTLComputePipelineState,
        threadgroups: MTLSize,
        threadsPerThreadgroup: Int,
        arguments: [LaunchArgument]
    ) throws {
        guard threadgroups.width > 0, threadgroups.height > 0, threadgroups.depth > 0 else {
            throw CoreError.invalidArgument("threadgroup grid must be positive in every dimension")
        }
        guard threadsPerThreadgroup > 0 else {
            throw CoreError.invalidArgument("threadsPerThreadgroup must be positive")
        }
        guard threadsPerThreadgroup <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw CoreError.invalidArgument(
                "threadsPerThreadgroup \(threadsPerThreadgroup) exceeds the pipeline maximum "
                    + "\(pipeline.maxTotalThreadsPerThreadgroup)")
        }

        let queue = try requireQueue()
        guard let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw CoreError.metal("could not create a compute command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        for (index, argument) in arguments.enumerated() {
            switch argument {
            case .buffer(let buffer):
                encoder.setBuffer(buffer, offset: 0, index: index)
            case .int32(var value):
                encoder.setBytes(&value, length: MemoryLayout<Int32>.size, index: index)
            case .int64(var value):
                encoder.setBytes(&value, length: MemoryLayout<Int64>.size, index: index)
            case .float32(var value):
                encoder.setBytes(&value, length: MemoryLayout<Float>.size, index: index)
            }
        }
        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw CoreError.metal("kernel execution failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Handle table

/// Opaque `Int64` handles for the C ABI. Handle 0 is always invalid, so callers
/// can treat 0 as "failure, see tm_last_error".
final class HandleTable: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Int64: Any] = [:]
    private var nextHandle: Int64 = 1

    func insert(_ value: Any) -> Int64 {
        lock.lock()
        defer { lock.unlock() }
        let handle = nextHandle
        nextHandle += 1
        storage[handle] = value
        return handle
    }

    func lookup<T>(_ handle: Int64, as type: T.Type, what: String) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        guard let value = storage[handle] else {
            throw CoreError.invalidHandle("\(what) handle \(handle) is not live")
        }
        guard let typed = value as? T else {
            throw CoreError.invalidHandle("handle \(handle) is not a \(what)")
        }
        return typed
    }

    @discardableResult
    func remove(_ handle: Int64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return storage.removeValue(forKey: handle) != nil
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage.count
    }
}

let tmHandles = HandleTable()

// MARK: - C ABI (device queries)

/// Returns 1 when a Metal device is present (the shim's `is_active` check).
@_cdecl("tm_is_active")
public func tm_is_active() -> Int32 {
    MetalRuntime.defaultDeviceName() != nil ? 1 : 0
}

/// Returns a malloc'd device-name string (caller frees via `tm_free`), or NULL.
@_cdecl("tm_device_name")
public func tm_device_name() -> UnsafeMutablePointer<CChar>? {
    MetalRuntime.defaultDeviceName().map { strdup($0) }
}
