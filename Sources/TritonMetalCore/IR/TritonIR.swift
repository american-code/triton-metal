import Foundation

// MARK: - Types

/// The type subset the Metal backend understands today.
public indirect enum TMType: Equatable, Sendable, CustomStringConvertible {
    /// `i1`, `i8`, `i16`, `i32`, `i64`
    case integer(width: Int)
    /// `f16`, `f32`
    case float(width: Int)
    /// `bf16` — the same 8-bit exponent as `f32` with 7 mantissa bits, which is
    /// why ML training uses it and why it is not `f16`. Metal spells it
    /// `bfloat` (MSL 3.1 and later) and has `simdgroup_matrix<bfloat, 8, 8>`.
    case bfloat
    /// `!tt.ptr<T>` — always a device-address-space pointer in Metal.
    case pointer(TMType)
    /// `tensor<D0x...xDNxT>` — rank >= 1, one entry per dimension.
    case tensor(shape: [Int], element: TMType)

    public var description: String {
        switch self {
        case .integer(let w): return "i\(w)"
        case .float(let w): return "f\(w)"
        case .bfloat: return "bf16"
        case .pointer(let p): return "!tt.ptr<\(p)>"
        case .tensor(let s, let e): return "tensor<\(s.map(String.init).joined(separator: "x"))x\(e)>"
        }
    }

    public var isTensor: Bool {
        if case .tensor = self { return true }
        return false
    }

    /// Element type of a tensor, or the type itself for scalars.
    public var scalarized: TMType {
        if case .tensor(_, let e) = self { return e }
        return self
    }

    /// `nil` for scalars; the dimension list for tensors.
    public var shape: [Int]? {
        if case .tensor(let s, _) = self { return s }
        return nil
    }

    /// 0 for scalars, otherwise the number of dimensions.
    public var rank: Int { shape?.count ?? 0 }

    /// Total element count of a tensor (`nil` for scalars).
    public var elementCount: Int? { shape.map { $0.reduce(1, *) } }

    /// Rebuilds `self`'s shape around a new element type.
    public func withElement(_ element: TMType) -> TMType {
        if case .tensor(let s, _) = self { return .tensor(shape: s, element: element) }
        return element
    }

    public var isIntegerLike: Bool {
        if case .integer = scalarized { return true }
        return false
    }

    public var isFloatLike: Bool {
        switch scalarized {
        case .float, .bfloat: return true
        default: return false
        }
    }

    /// True for the two 16-bit float types, which share every rule that is about
    /// width rather than about precision.
    public var isNarrowFloat: Bool {
        switch scalarized {
        case .float(let w): return w == 16
        case .bfloat: return true
        default: return false
        }
    }

    /// Size in bytes of the scalar element (used for buffer math and `setBytes`).
    public var scalarByteWidth: Int? {
        switch scalarized {
        case .integer(let w): return w == 1 ? 1 : w / 8
        case .float(let w): return w / 8
        case .bfloat: return 2
        case .pointer: return 8
        case .tensor: return nil
        }
    }
}

// MARK: - Instructions

public enum IntBinaryOp: String, Sendable {
    case add, sub, mul, divs, divu, rems, remu, and, or, xor, shl, shrs, shru, maxs, mins, maxu, minu
}

public enum FloatBinaryOp: String, Sendable {
    case add, sub, mul, div, maximum, minimum
}

/// `arith` conversion ops. The result type comes from the op's `to` clause.
public enum CastOp: String, Sendable {
    /// signed int -> float
    case sitofp
    /// unsigned int -> float
    case uitofp
    /// float -> signed int
    case fptosi
    /// float -> unsigned int
    case fptoui
    /// sign-extend int
    case extsi
    /// zero-extend int
    case extui
    /// narrow int
    case trunci
    /// narrow float (f32 -> f16)
    case truncf
    /// widen float (f16 -> f32)
    case extf
    /// reinterpret an int as a pointer offset-free bitcast
    case bitcast
}

/// `math.*` unary ops, plus `math.absi` for integers.
public enum MathOp: String, Sendable {
    case exp, exp2, log, log2, sqrt, rsqrt, sin, cos, tanh, erf, absf, absi, floor, ceil
}

/// The combiner recognised inside a `tt.reduce` region.
public enum ReduceOp: String, Sendable {
    case add, max, min
}

/// `tt.atomic_rmw`'s kind attribute, spelled as Triton prints it
/// (`TT_AtomicRMWAttr` in TritonAttrDefs.td — note `XCHG` prints as `exch`).
public enum AtomicRMWOp: String, Sendable, CaseIterable {
    case and, or, xor, add, fadd, max, min, umax, umin, exch

    static var allSpellings: String {
        allCases.map(\.rawValue).joined(separator: ", ")
    }
}

/// `tt.atomic_rmw`'s memory-semantics and sync-scope attributes.
///
/// Both are parsed and **ignored**: MSL exposes exactly one memory order for
/// device atomics (`memory_order_relaxed`; `memory_order_acq_rel` is not even a
/// declared identifier in Metal 3), and one scope. They are kept on the op so
/// diagnostics can name what was asked for.
public enum AtomicSemantic: String, Sendable {
    case relaxed, acquire, release, acq_rel
}

public enum AtomicScope: String, Sendable {
    case cta, gpu, sys
}

public enum ConstantValue: Equatable, Sendable {
    case integer(Int64)
    case float(Double)
}

/// One lowered Triton operation. Result types are inferred during emission from
/// the operand types, so only the syntactic operands are recorded here — except
/// where the op's own result type carries information the operands do not
/// (`tt.make_range`, `tt.splat`, casts, `tt.expand_dims`, `scf` results).
public enum OpKind: Sendable {
    case constant(result: String, type: TMType, value: ConstantValue)
    case programID(result: String, axis: Int)
    case numPrograms(result: String, axis: Int)
    case makeRange(result: String, type: TMType, start: Int, end: Int)
    case splat(result: String, type: TMType, source: String)
    case addPtr(result: String, pointer: String, offset: String)
    case intBinary(result: String, op: IntBinaryOp, lhs: String, rhs: String)
    case floatBinary(result: String, op: FloatBinaryOp, lhs: String, rhs: String)
    case intCompare(result: String, predicate: String, lhs: String, rhs: String)
    case floatCompare(result: String, predicate: String, lhs: String, rhs: String)
    case cast(result: String, op: CastOp, source: String, type: TMType)
    case unary(result: String, op: MathOp, source: String)
    case select(result: String, condition: String, whenTrue: String, whenFalse: String)
    case expandDims(result: String, type: TMType, source: String, axis: Int)
    /// `tt.trans %x {order = array<i32: 1, 0>}` — result dimension `i` is
    /// source dimension `order[i]`. A relabelling of the block axes, no data
    /// movement (see `BlockLayout` and `MSLEmitter.emit`).
    case trans(result: String, type: TMType, source: String, order: [Int])
    case broadcast(result: String, type: TMType, source: String)
    case reduce(result: String, source: String, axis: Int, op: ReduceOp)
    case dot(DotOp)
    case load(result: String, pointer: String, mask: String?, other: String?)
    case store(pointer: String, value: String, mask: String?)
    /// `tt.atomic_rmw <op>, <sem>, <scope>, %ptr, %val[, %mask]` — reads,
    /// modifies and writes back one device location, returning the **old** value.
    case atomicRMW(AtomicRMW)
    /// `tt.atomic_cas <sem>, <scope>, %ptr, %cmp, %val` — returns the old value.
    case atomicCAS(AtomicCAS)
    case forLoop(ForLoop)
    case ifOp(IfOp)
    case yield(values: [String])
    case ret
}

/// `tt.dot %a, %b, %acc : tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxU>`
///
/// The one operation whose operands do **not** live in the elementwise iteration
/// space: `K` is a contraction dimension (see `BlockLayout`), so the operand types
/// are kept here rather than being re-derived from the operands' own layouts.
public struct DotOp: Sendable {
    public var result: String
    public var lhs: String
    public var rhs: String
    /// The `C` of `D = A * B + C`; Triton always passes one, zero-filled if unused.
    public var accumulator: String
    public var lhsType: TMType
    public var rhsType: TMType
    public var resultType: TMType
}

/// `%old = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val, %mask : (...) -> ...`
///
/// One device read-modify-write per block point, returning the value that was
/// there before. Every operand has the pointer's shape; `mask` is the `i1` tensor
/// Triton builds from the same bounds test a `tt.store` would use, and a lane
/// whose mask is false performs no access at all and reads back zero.
public struct AtomicRMW: Sendable {
    public var result: String
    public var op: AtomicRMWOp
    public var pointer: String
    public var value: String
    public var mask: String?
    /// Parsed and ignored — see `AtomicSemantic`.
    public var semantic: AtomicSemantic
    public var scope: AtomicScope
}

/// `%old = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val : (...) -> ...`
///
/// Unmasked in Triton's own op definition: `tt.atomic_cas` takes no mask.
public struct AtomicCAS: Sendable {
    public var result: String
    public var pointer: String
    public var compare: String
    public var value: String
    public var semantic: AtomicSemantic
    public var scope: AtomicScope
}

/// `scf.for %iv = %lb to %ub step %st iter_args(%a = %init) -> (T) { ... }`
public struct ForLoop: Sendable {
    public var results: [String]
    public var inductionVariable: String
    public var inductionType: TMType
    public var lowerBound: String
    public var upperBound: String
    public var step: String
    /// Region block arguments and the values they are initialised from.
    public var iterArguments: [(name: String, initial: String, type: TMType)]
    public var body: [Instruction]
}

/// `scf.if %cond -> (T) { ... } else { ... }`
public struct IfOp: Sendable {
    public var results: [String]
    public var resultTypes: [TMType]
    public var condition: String
    public var thenBody: [Instruction]
    public var elseBody: [Instruction]
}

extension OpKind {
    /// Every SSA value this op reads. Regions report only the values crossing
    /// their boundary (bounds, initialisers, condition); their bodies are walked
    /// separately.
    var operandNames: [String] {
        switch self {
        case .constant, .programID, .numPrograms, .makeRange, .ret:
            return []
        case .splat(_, _, let source), .expandDims(_, _, let source, _),
            .trans(_, _, let source, _),
            .broadcast(_, _, let source), .cast(_, _, let source, _),
            .unary(_, _, let source), .reduce(_, let source, _, _):
            return [source]
        case .addPtr(_, let a, let b), .intBinary(_, _, let a, let b),
            .floatBinary(_, _, let a, let b), .intCompare(_, _, let a, let b),
            .floatCompare(_, _, let a, let b):
            return [a, b]
        case .select(_, let a, let b, let c):
            return [a, b, c]
        case .dot(let dot):
            return [dot.lhs, dot.rhs, dot.accumulator]
        case .load(_, let pointer, let mask, let other):
            return [pointer, mask, other].compactMap { $0 }
        case .store(let pointer, let value, let mask):
            return [pointer, value, mask].compactMap { $0 }
        case .atomicRMW(let atomic):
            return [atomic.pointer, atomic.value, atomic.mask].compactMap { $0 }
        case .atomicCAS(let atomic):
            return [atomic.pointer, atomic.compare, atomic.value]
        case .yield(let values):
            return values
        case .forLoop(let loop):
            return [loop.lowerBound, loop.upperBound, loop.step] + loop.iterArguments.map(\.initial)
        case .ifOp(let branch):
            return [branch.condition]
        }
    }

    /// Every SSA value this op defines (region block arguments included).
    var resultNames: [String] {
        switch self {
        case .constant(let r, _, _), .programID(let r, _), .numPrograms(let r, _),
            .makeRange(let r, _, _, _), .splat(let r, _, _), .addPtr(let r, _, _),
            .intBinary(let r, _, _, _), .floatBinary(let r, _, _, _),
            .intCompare(let r, _, _, _), .floatCompare(let r, _, _, _),
            .cast(let r, _, _, _), .unary(let r, _, _), .select(let r, _, _, _),
            .expandDims(let r, _, _, _), .trans(let r, _, _, _), .broadcast(let r, _, _),
            .reduce(let r, _, _, _), .load(let r, _, _, _):
            return [r]
        case .dot(let dot):
            return [dot.result]
        case .atomicRMW(let atomic):
            return [atomic.result]
        case .atomicCAS(let atomic):
            return [atomic.result]
        case .forLoop(let loop):
            return loop.results + [loop.inductionVariable] + loop.iterArguments.map(\.name)
        case .ifOp(let branch):
            return branch.results
        case .store, .yield, .ret:
            return []
        }
    }
}

public struct Instruction: Sendable {
    public var kind: OpKind
    /// The op mnemonic as written, kept for diagnostics.
    public var mnemonic: String
    public var loc: SourceLoc
}

public struct FunctionArgument: Sendable {
    public var name: String
    public var type: TMType
    /// Position in the argument list == Metal buffer index.
    public var index: Int
    public var loc: SourceLoc
}

public struct TritonFunction: Sendable {
    public var name: String
    public var arguments: [FunctionArgument]
    public var body: [Instruction]
    public var loc: SourceLoc
}

public struct TritonModule: Sendable {
    public var functions: [TritonFunction]
}

// MARK: - Errors

public enum CoreError: Error, CustomStringConvertible {
    case unimplemented(String)
    case parse(String, SourceLoc)
    case unsupportedOp(String, SourceLoc)
    case unsupportedType(String, SourceLoc)
    case lowering(String, SourceLoc)
    /// One value indexing two block dimensions with two of its own — reported
    /// exactly like a lowering error, but distinguished so that `MSLEmitter` can
    /// tell the one failure `AxisCloning` might be able to rewrite away from
    /// every other one. Widening the retry to *any* layout failure would let it
    /// silently change what a kernel means.
    case axisCollapse(String, SourceLoc)
    case metal(String)
    case invalidHandle(String)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .unimplemented(let what):
            return "unimplemented: \(what)"
        case .parse(let message, let loc):
            return "parse error at \(loc): \(message)"
        case .unsupportedOp(let name, let loc):
            return "unsupported op '\(name)' at \(loc): the Metal backend does not lower this "
                + "operation yet (see docs/ARCHITECTURE.md §Supported IR subset)"
        case .unsupportedType(let name, let loc):
            return "unsupported type '\(name)' at \(loc)"
        case .lowering(let message, let loc), .axisCollapse(let message, let loc):
            return "lowering error at \(loc): \(message)"
        case .metal(let message):
            return "metal error: \(message)"
        case .invalidHandle(let message):
            return "invalid handle: \(message)"
        case .invalidArgument(let message):
            return "invalid argument: \(message)"
        }
    }
}
