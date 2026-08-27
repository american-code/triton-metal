import Foundation
import MLX
import Metal
import TritonMetalCore

/// A scalar kernel argument.
///
/// Integer and float literals work directly (`scalars: [512, 2048, 0.5]`); the
/// width is reconciled with what the kernel declared when the launch is checked,
/// so an `i64` parameter accepts an integer literal without ceremony. There is
/// no `f64` case: Metal has no `double`, and the emitter refuses an `f64` kernel
/// argument rather than silently narrowing it.
public enum LaunchScalar: Sendable, Equatable, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral
{
    case integer(Int64)
    case float32(Float)

    public init(integerLiteral value: Int) { self = .integer(Int64(value)) }
    public init(floatLiteral value: Double) { self = .float32(Float(value)) }
    public init(_ value: Int) { self = .integer(Int64(value)) }
    public init(_ value: Float) { self = .float32(value) }
}

/// The Triton program grid: how many programs to launch along each axis.
///
/// One program is one Metal threadgroup, so this is the `threadgroups` argument
/// of the dispatch. The threads *inside* a threadgroup are not a choice — the
/// emitter fixed that when it lowered the kernel, and `MetalPipeline` reads it
/// out of the kernel's metadata.
public struct Grid: Sendable, Equatable {
    public var x: Int
    public var y: Int
    public var z: Int

    public init(_ x: Int, _ y: Int = 1, _ z: Int = 1) {
        self.x = x
        self.y = y
        self.z = z
    }

    /// The grid that covers `total` elements in `block`-sized steps, rounded up.
    public static func covering(_ total: Int, block: Int) -> Int {
        (total + block - 1) / block
    }
}

/// One compiled Triton kernel, launchable on `MLXArray`s.
///
/// This is the whole Swift-native path: Triton IR text goes in, a callable
/// object comes out, and MLX tensors go straight to the GPU with no Python
/// anywhere. The compile side is `TritonMetalCore` (parser, emitter, Metal
/// runtime); what this type adds is the MLX seam — dtype checking against the
/// kernel's own argument metadata, forced evaluation, and the buffer binding
/// described in `MLXBuffers.swift`.
///
/// ```swift
/// let pipeline = try MetalPipeline.compile(ttir: SAEEncoder.ir(blockM: 64, blockF: 64, blockD: 32))
/// let out = MLX.zeros([rows, features], dtype: .float32)
/// let results = try pipeline.launch(
///     arrays: [x, wEnc, bEnc, out],
///     scalars: [rows, model, features],
///     grid: Grid(Grid.covering(rows, block: 64), Grid.covering(features, block: 64)))
/// let activations = results[3]        // read the returned array, not `out`
/// ```
public final class MetalPipeline {

    /// The kernel's launch metadata, straight from `EmissionResult`: name, block
    /// shape, threadgroup size, and the argument list this type validates
    /// against.
    public let metadata: EmittedKernel

    /// The MSL the kernel lowered to. Kept because reading it is the fastest way
    /// to answer "what did my kernel actually become", which is the reason the
    /// backend emits textual MSL at all.
    public let source: String

    public let pipeline: MTLComputePipelineState
    let device: MTLDevice

    /// Threads per threadgroup, as the emitter chose it. Not a launch parameter:
    /// the emitted MSL strides its block over exactly this many threads.
    public var threadsPerThreadgroup: Int { metadata.threadsPerThreadgroup }

    /// Pointer arguments, in `tt.func` order — the slots `arrays:` fills.
    ///
    /// Split once at construction rather than filtered per launch: at the small
    /// sizes where dispatch overhead is the thing being measured, a few hundred
    /// nanoseconds of bookkeeping per launch is a visible fraction of the answer.
    public let pointerArguments: [KernelArgument]

    /// Scalar arguments, in `tt.func` order — the slots `scalars:` fills.
    public let scalarArguments: [KernelArgument]

    init(metadata: EmittedKernel, source: String, pipeline: MTLComputePipelineState,
         device: MTLDevice) {
        self.metadata = metadata
        self.source = source
        self.pipeline = pipeline
        self.device = device
        self.pointerArguments = metadata.arguments.filter { $0.kind == .pointer }
        self.scalarArguments = metadata.arguments.filter { $0.kind == .scalar }
    }

    // MARK: - Compiling

    /// Lowers `ttir` and builds a pipeline for one of its kernels.
    ///
    /// `kernel` names which `tt.func` to take when the module has more than one;
    /// a single-kernel module does not need it.
    public static func compile(
        ttir: String, kernel name: String? = nil,
        options: MetalCompiler.Options = .init()
    ) throws -> MetalPipeline {
        let all = try compileAll(ttir: ttir, options: options)
        guard let name else {
            guard all.count == 1 else {
                throw CoreError.invalidArgument(
                    "this module declares \(all.count) kernels "
                        + "(\(all.map(\.metadata.name).joined(separator: ", "))); name the one you "
                        + "want with `kernel:`")
            }
            return all[0]
        }
        guard let match = all.first(where: { $0.metadata.name == name }) else {
            throw CoreError.invalidArgument(
                "no kernel named '\(name)' in this module (it declares "
                    + "\(all.map(\.metadata.name).joined(separator: ", ")))")
        }
        return match
    }

    /// Lowers `ttir` and builds a pipeline for every kernel in it. The module is
    /// compiled once; the pipelines share the resulting `MTLLibrary`.
    public static func compileAll(
        ttir: String, options: MetalCompiler.Options = .init()
    ) throws -> [MetalPipeline] {
        if let reason = MetalRuntime.unusableReason {
            throw CoreError.metal(reason)
        }
        guard let device = MetalRuntime.device else {
            throw CoreError.metal("no Metal device available on this system")
        }
        let emission = try MetalCompiler.emit(ttir: ttir, options: options)
        let library = try MetalCompiler.compileMSL(emission.source)
        return try emission.kernels.map { kernel in
            MetalPipeline(
                metadata: kernel, source: emission.source,
                pipeline: try MetalRuntime.loadKernel(library: library, kernelName: kernel.name),
                device: device)
        }
    }

    // MARK: - Launching

    /// Runs the kernel over `arrays` and `scalars`, and returns one array per
    /// pointer argument holding what the kernel left in that buffer.
    ///
    /// `arrays` fills the kernel's pointer arguments in `tt.func` order and
    /// `scalars` its scalar arguments in `tt.func` order; the two interleave back
    /// into the emitted `[[buffer(N)]]` indices from `metadata.arguments`, so a
    /// caller never has to know where the pointers sit among the sizes.
    ///
    /// **Read the returned arrays, not the ones you passed.** For a contiguous
    /// array the returned element *is* the array you passed, mutated in place by
    /// the kernel; for a strided one it is a new array holding the result. See
    /// `MLXBuffers.swift` for why the rule is written that way.
    ///
    /// Every array is `eval()`ed before its address is taken. An unevaluated
    /// `MLXArray` is a node in a graph, and its backing is not a buffer of
    /// results.
    ///
    /// ### What is checked, and what cannot be
    ///
    /// Checked against `metadata.arguments`: the number of arrays and of scalars,
    /// each array's dtype against the pointee type the kernel declared, each
    /// array's non-emptiness, and each scalar's width against the parameter's
    /// declared integer or float type.
    ///
    /// Not checked, because it is not in the metadata: per-argument **extents**.
    /// Triton lowers a tensor's shape into pointer arithmetic over scalar size
    /// and stride parameters, so by the time a kernel reaches this backend its
    /// arguments are bare `!tt.ptr<T>` with no rank and no extent. What the
    /// metadata does carry is the *block* shape (`metadata.blockShape`) — the
    /// tile one program walks, not the size of anything it is pointed at. A
    /// kernel's expectation of `[M, D]` versus `[D, M]` is therefore between the
    /// caller and the kernel; the wrapper in `SAEEncoder` shows the shape of the
    /// checking a specific op can do, because it knows its own arguments.
    @discardableResult
    public func launch(
        arrays: [MLXArray], scalars: [LaunchScalar] = [], grid: Grid
    ) throws -> [MLXArray] {
        try launch(prepare(arrays: arrays, scalars: scalars), grid: grid)
    }

    /// An argument list already bound to buffers, reusable across launches.
    ///
    /// Worth hoisting out of a loop whenever the tensors do not change between
    /// launches — an SAE's dictionary and bias are fixed for every batch it ever
    /// encodes. `MTLDevice.makeBuffer(bytesNoCopy:length:)` has to wire the pages
    /// it is handed, so its cost grows with the buffer, and re-binding a 4 MB
    /// weight matrix on every launch is measurably worse than binding it once
    /// (docs/USAGE.md §Dispatch overhead has the numbers).
    ///
    /// Holding one keeps its arrays alive, which is the point: a `bytesNoCopy`
    /// buffer does not own its memory.
    public final class Arguments {
        let bindings: [MLXBinding]
        let launchArguments: [MetalRuntime.LaunchArgument]

        init(bindings: [MLXBinding], launchArguments: [MetalRuntime.LaunchArgument]) {
            self.bindings = bindings
            self.launchArguments = launchArguments
        }

        /// One array per pointer argument, holding what the last launch left in
        /// that buffer. Same rule as `launch`: read these, not what you passed.
        public var results: [MLXArray] { bindings.map { $0.result() } }

        /// Whether each pointer argument aliases its array's storage or a
        /// contiguous copy of it.
        public var modes: [MLXBinding.Mode] { bindings.map(\.mode) }
    }

    /// Checks and binds an argument list without launching anything.
    public func prepare(arrays: [MLXArray], scalars: [LaunchScalar] = []) throws -> Arguments {
        let pointers = pointerArguments
        let scalarArgs = scalarArguments

        guard arrays.count == pointers.count, scalars.count == scalarArgs.count else {
            throw CoreError.invalidArgument(
                "kernel '\(metadata.name)' takes \(pointers.count) pointer and "
                    + "\(scalarArgs.count) scalar arguments (\(signature())); "
                    + "got \(arrays.count) arrays and \(scalars.count) scalars")
        }
        var bindings: [MLXBinding] = []
        bindings.reserveCapacity(pointers.count)
        for (array, argument) in zip(arrays, pointers) {
            let spelling = try MLXDataType.triton(array.dtype)
            guard spelling == argument.dtype else {
                throw CoreError.invalidArgument(
                    "argument \(argument.index) of '\(metadata.name)' is "
                        + "!tt.ptr<\(argument.dtype)>, but the array passed for it is "
                        + "\(array.dtype) (\(spelling)). "
                        + (MLXDataType.mlx(argument.dtype).map { "Pass a \($0) array" }
                            ?? "This backend has no MLX dtype for \(argument.dtype)")
                        + ", or change the kernel.")
            }
            bindings.append(
                try MLXBinding.bind(
                    array, device: device,
                    what: "argument \(argument.index) of '\(metadata.name)'"))
        }

        // Both lists are in `tt.func` order and their `index` fields are the
        // emitted `[[buffer(N)]]` slots, so one walk over `metadata.arguments`
        // interleaves them back without a dictionary.
        var launchArguments: [MetalRuntime.LaunchArgument] = []
        launchArguments.reserveCapacity(metadata.arguments.count)
        var nextPointer = 0
        var nextScalar = 0
        for argument in metadata.arguments {
            switch argument.kind {
            case .pointer:
                launchArguments.append(.buffer(bindings[nextPointer].buffer))
                nextPointer += 1
            case .scalar:
                launchArguments.append(try bind(scalar: scalars[nextScalar], to: argument))
                nextScalar += 1
            }
        }

        return Arguments(bindings: bindings, launchArguments: launchArguments)
    }

    /// Runs the kernel over an already-bound argument list.
    @discardableResult
    public func launch(_ arguments: Arguments, grid: Grid) throws -> [MLXArray] {
        guard grid.x > 0, grid.y > 0, grid.z > 0 else {
            throw CoreError.invalidArgument(
                "grid must be positive in every dimension, got (\(grid.x), \(grid.y), \(grid.z))")
        }
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(width: grid.x, height: grid.y, depth: grid.z),
            threadsPerThreadgroup: threadsPerThreadgroup,
            arguments: arguments.launchArguments)
        return arguments.results
    }

    /// Compile and launch in one call, for a caller that runs a kernel once.
    ///
    /// Compilation is not cached, so this is the wrong entry point for a loop —
    /// hold a `MetalPipeline` and call `launch` for that. It exists because the
    /// one-shot case is otherwise three lines of ceremony around one idea.
    @discardableResult
    public static func run(
        ttir: String, kernel name: String? = nil, options: MetalCompiler.Options = .init(),
        arrays: [MLXArray], scalars: [LaunchScalar] = [], grid: Grid
    ) throws -> [MLXArray] {
        try compile(ttir: ttir, kernel: name, options: options)
            .launch(arrays: arrays, scalars: scalars, grid: grid)
    }

    // MARK: - Argument binding

    private func bind(scalar: LaunchScalar, to argument: KernelArgument) throws
        -> MetalRuntime.LaunchArgument
    {
        switch (scalar, argument.dtype) {
        case (.float32(let value), "f32"):
            return .float32(value)
        case (.integer(let value), "i64"):
            return .int64(value)
        case (.integer(let value), "i1"), (.integer(let value), "i8"),
            (.integer(let value), "i16"), (.integer(let value), "i32"):
            // Narrower parameters are still bound as i32: `tm_launch`'s kind 1
            // sign-extends into the low 32 bits and the emitted MSL declares the
            // parameter at its own width. Out of range is an error, never a
            // truncation.
            guard let narrowed = Int32(exactly: value) else {
                throw CoreError.invalidArgument(
                    "argument \(argument.index) of '\(metadata.name)' is \(argument.dtype), and "
                        + "\(value) does not fit in it")
            }
            return .int32(narrowed)
        case (.integer, "f32"):
            throw CoreError.invalidArgument(
                "argument \(argument.index) of '\(metadata.name)' is f32; pass a float "
                    + "(`1.0`, not `1`) so the conversion is yours and not this wrapper's")
        case (.float32, _):
            throw CoreError.invalidArgument(
                "argument \(argument.index) of '\(metadata.name)' is \(argument.dtype); a float "
                    + "was passed for it")
        default:
            throw CoreError.invalidArgument(
                "argument \(argument.index) of '\(metadata.name)' has type \(argument.dtype), "
                    + "which this backend cannot bind as a scalar (Metal has no double, so f64 "
                    + "is refused at emission)")
        }
    }

    /// The kernel's argument list, the way an error message should say it.
    func signature() -> String {
        metadata.arguments
            .map { $0.kind == .pointer ? "!tt.ptr<\($0.dtype)>" : $0.dtype }
            .joined(separator: ", ")
    }
}
