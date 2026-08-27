import Foundation

/// How one kernel's tensors map onto the threadgroup's iteration space.
///
/// Every tensor in a Triton program is a slice of one *block*: a rank-`rank`
/// index space with `shape[d]` elements along dimension `d`. The emitter walks
/// that space with one nested loop per dimension (see `MSLEmitter`), so each
/// tensor value needs to know which block dimension each of *its* dimensions
/// corresponds to — that is what `axes` records.
///
/// Ranks below the block rank appear all the time: `tl.arange(0, M)` is rank 1
/// even in a rank-2 kernel, and only `tt.expand_dims` says whether it indexes
/// rows or columns. So `axes` is inferred rather than assumed.
struct BlockLayout {
    /// Number of *iteration* block dimensions — the ones the emitter walks with
    /// nested loops (0 when the kernel is entirely scalar).
    let rank: Int
    /// Iteration block size per dimension. Broadcast-only dimensions are 1.
    let shape: [Int]
    /// SSA name -> block dimension for each of the value's own dimensions.
    let axes: [String: [Int]]
    /// Extents of the **contraction** dimensions introduced by `tt.dot`, one per
    /// dot. The axis index of `contractions[i]` is `rank + i`.
    ///
    /// A contraction axis is deliberately *not* part of the iteration space: a
    /// `tt.dot`'s operands are `MxK` and `KxN` while its result is `MxN`, so K is
    /// walked only inside the dot's own staging loops and never by the
    /// elementwise lane distribution.
    let contractions: [Int]

    static let scalar = BlockLayout(rank: 0, shape: [], axes: [:], contractions: [])

    /// Total number of block dimensions, iteration and contraction together.
    var axisCount: Int { rank + contractions.count }

    func extent(ofAxis axis: Int) -> Int {
        axis < rank ? shape[axis] : contractions[axis - rank]
    }

    /// Block dimensions this value actually varies along (size-1 dimensions are
    /// broadcast, so the value does not depend on their index).
    func varyingAxes(of ssa: String, shape valueShape: [Int]) -> [Int] {
        guard let map = axes[ssa], map.count == valueShape.count else { return [] }
        return zip(map, valueShape).filter { $0.1 > 1 }.map(\.0)
    }

    /// True when the value varies along a contraction axis, which means it can
    /// only be materialised inside a `tt.dot`'s staging loops.
    func spansContraction(_ ssa: String, shape valueShape: [Int]) -> Bool {
        varyingAxes(of: ssa, shape: valueShape).contains { $0 >= rank }
    }
}

/// Decides which values a kernel has to *materialise* in the elementwise
/// iteration nest and which exist only to feed a `tt.dot`.
///
/// A `tt.dot` operand is never a per-lane value: it is staged into a threadgroup
/// tile over an index space that includes the contraction dimension. Everything
/// that feeds one — the pointer arithmetic, the masks, the loads — therefore has
/// to be emitted inside the dot's own staging loops rather than where it appears.
/// This is a backward liveness walk: a value is materialised when some use other
/// than "an operand of a tt.dot" needs it, and deferred otherwise.
enum DotDeferral {
    /// Values reachable backwards from a real use. Everything else is either dead
    /// or exists only for a `tt.dot`.
    static func materialised(in function: TritonFunction) -> Set<String> {
        var producers: [String: [String]] = [:]
        var roots: [String] = function.arguments.map(\.name)
        /// Values that must all be materialised together, or not at all: an
        /// `scf.for`'s initialiser, block argument, yielded value and result.
        var chains: [[String]] = []

        func walk(_ body: [Instruction]) {
            for instruction in body {
                switch instruction.kind {
                case .store(let pointer, let value, let mask):
                    roots.append(contentsOf: [pointer, value, mask].compactMap { $0 })

                case .dot(let dot):
                    // The dot itself is materialised (as a tile), but reaching it
                    // through `lhs`/`rhs`/`accumulator` does not materialise them.
                    roots.append(dot.result)
                    producers[dot.result] = []

                case .forLoop(let loop):
                    roots.append(contentsOf: [loop.lowerBound, loop.upperBound, loop.step])
                    let yielded = terminator(of: loop.body) ?? []
                    for (index, argument) in loop.iterArguments.enumerated() {
                        var chain = [argument.initial, argument.name]
                        if index < loop.results.count { chain.append(loop.results[index]) }
                        if index < yielded.count { chain.append(yielded[index]) }
                        chains.append(chain)
                    }
                    walk(loop.body)

                case .ifOp(let branch):
                    roots.append(branch.condition)
                    for body in [branch.thenBody, branch.elseBody] {
                        let yielded = terminator(of: body) ?? []
                        for (index, value) in yielded.enumerated()
                        where index < branch.results.count {
                            chains.append([value, branch.results[index]])
                        }
                        walk(body)
                    }

                case .yield, .ret:
                    continue

                default:
                    let operands = instruction.kind.operandNames
                    for result in instruction.kind.resultNames {
                        producers[result, default: []].append(contentsOf: operands)
                    }
                }
            }
        }
        walk(function.body)

        var live = Set(roots)
        var work = roots
        var settled = false
        while !settled {
            while let value = work.popLast() {
                for producer in producers[value] ?? [] where live.insert(producer).inserted {
                    work.append(producer)
                }
            }
            settled = true
            for chain in chains where chain.contains(where: live.contains) {
                for member in chain where live.insert(member).inserted {
                    work.append(member)
                    settled = false
                }
            }
        }
        return live
    }

    private static func terminator(of body: [Instruction]) -> [String]? {
        guard case .yield(let values) = body.last?.kind else { return nil }
        return values
    }
}

/// Infers the block layout of one `tt.func`.
///
/// Two passes: shapes first (a light-weight rank/shape propagation that mirrors
/// what the emitter later does for full types), then the axis assignment, by
/// seeding every full-rank tensor with the identity map and propagating that
/// through `tt.expand_dims`, `tt.broadcast`, `tt.reduce` and the elementwise ops.
enum LayoutInference {
    static func compute(for function: TritonFunction) throws -> BlockLayout {
        var solver = Solver()
        for argument in function.arguments {
            solver.shapes[argument.name] = argument.type.shape ?? []
        }
        try solver.walk(function.body)

        let rank = solver.shapes.values.map(\.count).max() ?? 0
        guard rank > 0 else { return .scalar }
        return try solver.solve(rank: rank, function: function)
    }

    private struct Solver {
        /// `[]` for scalars, the dimension list for tensors.
        var shapes: [String: [Int]] = [:]
        /// Values that must share one axis map (same rank, elementwise-related).
        var groups: [[String]] = []
        /// `tt.expand_dims`: the result gains a size-1 dimension at `axis`.
        var expansions: [(source: String, result: String, axis: Int, loc: SourceLoc)] = []
        /// `tt.reduce`: the result drops the dimension at `axis`.
        var reductions: [(source: String, result: String, axis: Int, loc: SourceLoc)] = []
        /// `tt.dot`: pins its operands onto (M, K) / (K, N) / (M, N).
        var dots: [(op: DotOp, loc: SourceLoc)] = []
        var locations: [String: SourceLoc] = [:]

        private func shape(_ ssa: String) -> [Int] { shapes[ssa] ?? [] }

        /// Records that every tensor among `values` shares one axis map.
        private mutating func relate(_ values: [String]) {
            let tensors = values.filter { !shape($0).isEmpty }
            if tensors.count > 1 { groups.append(tensors) }
        }

        private mutating func define(_ ssa: String, _ shape: [Int], _ loc: SourceLoc) {
            shapes[ssa] = shape
            locations[ssa] = loc
        }

        mutating func walk(_ body: [Instruction]) throws {
            for instruction in body {
                let loc = instruction.loc
                switch instruction.kind {
                case .ret, .yield:
                    continue

                case .constant(let result, let type, _),
                    .makeRange(let result, let type, _, _),
                    .splat(let result, let type, _):
                    define(result, type.shape ?? [], loc)

                case .programID(let result, _), .numPrograms(let result, _):
                    define(result, [], loc)

                case .expandDims(let result, let type, let source, let axis):
                    define(result, type.shape ?? [], loc)
                    expansions.append((source: source, result: result, axis: axis, loc: loc))

                case .broadcast(let result, let type, let source):
                    define(result, type.shape ?? [], loc)
                    relate([source, result])

                case .cast(let result, _, let source, let type):
                    define(result, type.shape ?? [], loc)
                    relate([source, result])

                case .reduce(let result, let source, let axis, _):
                    var reduced = shape(source)
                    guard axis < reduced.count else {
                        throw CoreError.lowering(
                            "tt.reduce axis \(axis) is out of range for \(reduced.count)-D operand",
                            loc)
                    }
                    reduced.remove(at: axis)
                    define(result, reduced, loc)
                    reductions.append((source: source, result: result, axis: axis, loc: loc))

                case .dot(let dot):
                    define(dot.result, dot.resultType.shape ?? [], loc)
                    // The accumulator and the result occupy the same MxN space.
                    relate([dot.accumulator, dot.result])
                    dots.append((op: dot, loc: loc))

                case .addPtr(let result, let pointer, let offset):
                    define(result, shape(offset).isEmpty ? shape(pointer) : shape(offset), loc)
                    relate([pointer, offset, result])

                case .intBinary(let result, _, let lhs, let rhs),
                    .floatBinary(let result, _, let lhs, let rhs),
                    .intCompare(let result, _, let lhs, let rhs),
                    .floatCompare(let result, _, let lhs, let rhs):
                    define(result, shape(lhs).isEmpty ? shape(rhs) : shape(lhs), loc)
                    relate([lhs, rhs, result])

                case .unary(let result, _, let source):
                    define(result, shape(source), loc)
                    relate([source, result])

                case .select(let result, let condition, let whenTrue, let whenFalse):
                    define(result, shape(whenTrue), loc)
                    relate([condition, whenTrue, whenFalse, result])

                case .load(let result, let pointer, let mask, let other):
                    define(result, shape(pointer), loc)
                    relate([pointer, mask, other, result].compactMap { $0 })

                case .store(let pointer, let value, let mask):
                    relate([pointer, value, mask].compactMap { $0 })

                case .forLoop(let loop):
                    define(loop.inductionVariable, [], loc)
                    for (index, argument) in loop.iterArguments.enumerated() {
                        let shape = argument.type.shape ?? []
                        define(argument.name, shape, loc)
                        define(loop.results[index], shape, loc)
                        relate([argument.initial, argument.name, loop.results[index]])
                    }
                    try walk(loop.body)
                    if let yielded = terminatorValues(of: loop.body) {
                        for (index, value) in yielded.enumerated()
                        where index < loop.iterArguments.count {
                            relate([value, loop.iterArguments[index].name])
                        }
                    }

                case .ifOp(let branch):
                    for (index, type) in branch.resultTypes.enumerated() {
                        define(branch.results[index], type.shape ?? [], loc)
                    }
                    try walk(branch.thenBody)
                    try walk(branch.elseBody)
                    for body in [branch.thenBody, branch.elseBody] {
                        guard let yielded = terminatorValues(of: body) else { continue }
                        for (index, value) in yielded.enumerated() where index < branch.results.count {
                            relate([value, branch.results[index]])
                        }
                    }
                }
            }
        }

        private func terminatorValues(of body: [Instruction]) -> [String]? {
            guard case .yield(let values) = body.last?.kind else { return nil }
            return values
        }

        /// One round of constraint propagation, to a fixed point. Chains are short
        /// (rank 1 -> rank 2), so this converges in a couple of passes; the bound
        /// just keeps a malformed module from spinning.
        private func propagate(_ axes: inout [String: [Int]], rank: Int) throws {
            for _ in 0..<(rank + 4) {
                var changed = false
                for group in groups {
                    guard let known = group.first(where: { axes[$0] != nil }) else { continue }
                    let map = axes[known]!
                    for member in group where axes[member] == nil {
                        guard shapes[member]?.count == map.count else { continue }
                        axes[member] = map
                        changed = true
                    }
                }
                for expansion in expansions {
                    guard axes[expansion.source] == nil, var map = axes[expansion.result] else {
                        continue
                    }
                    guard expansion.axis < map.count else {
                        throw CoreError.lowering(
                            "tt.expand_dims axis \(expansion.axis) is out of range for its "
                                + "\(map.count)-D result", expansion.loc)
                    }
                    map.remove(at: expansion.axis)
                    axes[expansion.source] = map
                    changed = true
                }
                for reduction in reductions {
                    guard axes[reduction.result] == nil, var map = axes[reduction.source] else {
                        continue
                    }
                    guard reduction.axis < map.count else { continue }
                    map.remove(at: reduction.axis)
                    axes[reduction.result] = map
                    changed = true
                }
                if !changed { break }
            }
        }

        /// Pins one value's axis map, reporting a clash rather than overwriting.
        private func pin(
            _ axes: inout [String: [Int]], _ ssa: String, _ map: [Int], _ what: String,
            _ loc: SourceLoc
        ) throws {
            guard let shape = shapes[ssa], shape.count == map.count else {
                throw CoreError.lowering(
                    "tt.dot's \(what) '%\(ssa)' must be a rank-\(map.count) tensor, found "
                        + "rank \(shapes[ssa]?.count ?? 0)", loc)
            }
            if let existing = axes[ssa], existing != map {
                throw CoreError.lowering(
                    "'%\(ssa)' is used both as tt.dot's \(what) and along block dimensions "
                        + "\(existing); one value cannot span two different index spaces", loc)
            }
            axes[ssa] = map
        }

        mutating func solve(rank: Int, function: TritonFunction) throws -> BlockLayout {
            var axes: [String: [Int]] = [:]
            let identity = Array(0..<rank)

            // tt.dot pins its own operands first: `a` spans (M, K), `b` spans
            // (K, N) and the accumulator/result span (M, N). Only then does the
            // identity seeding below fill in the ordinary elementwise tensors, so
            // that everything feeding an operand inherits the operand's axes
            // rather than being forced onto the iteration space.
            if !dots.isEmpty {
                guard rank == 2 else {
                    throw CoreError.lowering(
                        "tt.dot needs a rank-2 block index space; the widest tensor in kernel "
                            + "'\(function.name)' is rank \(rank)", dots[0].loc)
                }
            }
            for (index, entry) in dots.enumerated() {
                let contraction = rank + index
                try pin(&axes, entry.op.result, identity, "result", entry.loc)
                try pin(&axes, entry.op.accumulator, identity, "accumulator", entry.loc)
                try pin(&axes, entry.op.lhs, [0, contraction], "left operand", entry.loc)
                try pin(&axes, entry.op.rhs, [contraction, 1], "right operand", entry.loc)
            }
            try propagate(&axes, rank: rank)

            for (ssa, shape) in shapes where shape.count == rank && axes[ssa] == nil {
                axes[ssa] = identity
            }
            try propagate(&axes, rank: rank)

            for (ssa, shape) in shapes where !shape.isEmpty {
                guard let map = axes[ssa] else {
                    throw CoreError.lowering(
                        "cannot infer which block dimension '%\(ssa)' spans: it is "
                            + "\(shape.count)-D in a \(rank)-D kernel and never reaches a "
                            + "tt.expand_dims or a full-rank operation",
                        locations[ssa] ?? function.loc)
                }
                guard map.count == shape.count else {
                    throw CoreError.lowering(
                        "'%\(ssa)' is \(shape.count)-D but is combined with \(map.count)-D values",
                        locations[ssa] ?? function.loc)
                }
            }

            // Block size per dimension: every value must agree, ignoring broadcasts.
            let axisCount = rank + dots.count
            var block = [Int](repeating: 1, count: axisCount)
            var pinned = [Bool](repeating: false, count: axisCount)
            for (ssa, shape) in shapes where !shape.isEmpty {
                let map = axes[ssa]!
                for (index, dimension) in shape.enumerated() {
                    guard dimension != 1 else { continue }
                    guard dimension > 0 else {
                        throw CoreError.lowering(
                            "tensor dimension must be positive, found \(dimension) in '%\(ssa)'",
                            locations[ssa] ?? function.loc)
                    }
                    let axis = map[index]
                    guard axis < axisCount else {
                        throw CoreError.lowering(
                            "'%\(ssa)' refers to block dimension \(axis), which does not exist",
                            locations[ssa] ?? function.loc)
                    }
                    if pinned[axis], block[axis] != dimension {
                        throw CoreError.lowering(
                            "kernel '\(function.name)' mixes tensor lengths \(block[axis]) and "
                                + "\(dimension) along block dimension \(axis); one block size per "
                                + "dimension is lowered",
                            locations[ssa] ?? function.loc)
                    }
                    block[axis] = dimension
                    pinned[axis] = true
                }
            }
            return BlockLayout(
                rank: rank, shape: Array(block[0..<rank]), axes: axes,
                contractions: Array(block[rank...]))
        }
    }
}
