import Foundation

/// How one kernel's tensors map onto the threadgroup's iteration space.
///
/// Every tensor in a Triton program is a slice of one *block*: an index space
/// whose dimensions this type calls **axes**. The emitter walks a value's axes
/// with one nested loop each (see `MSLEmitter`), so each tensor value needs to
/// know which block axis each of *its* dimensions corresponds to — that is what
/// `axes` records — and in which nest it is emitted, which is `paths`.
///
/// Ranks below the block rank appear all the time: `tl.arange(0, M)` is rank 1
/// even in a rank-2 kernel, and only `tt.expand_dims` says whether it indexes
/// rows or columns. So `axes` is inferred rather than assumed.
///
/// Axes are *not* one nest shared by every value. FlashAttention-2 is the kernel
/// that forces this: its `p` spans (M, N) and its `acc` spans (M, HEAD_DIM), and
/// neither is a slice of the other. Each value is therefore emitted in the nest
/// over exactly the axes it varies along (`paths`), the nests sharing whatever
/// prefix they have in common. See §Execution model.
struct BlockLayout {
    /// Number of *iteration* block axes — the ones the emitter walks with nested
    /// loops (0 when the kernel is entirely scalar).
    let rank: Int
    /// Iteration block size per axis. Broadcast-only axes are 1.
    let shape: [Int]
    /// SSA name -> block axis for each of the value's own dimensions.
    let axes: [String: [Int]]
    /// Extents of axes that no materialised value spans, so they are walked only
    /// inside a `tt.dot`'s staging loops. The axis index of `contractions[i]` is
    /// `rank + i`.
    ///
    /// A `tt.dot`'s contracted axis lands here *when nothing else needs it*. It
    /// need not: FA-2's second dot contracts over the same axis its softmax
    /// iterates, and that axis is an ordinary iteration axis.
    let contractions: [Int]
    /// Iteration axes whose loop every thread walks in full, because some value
    /// nests another axis inside it. The last axis of any nest is the one strided
    /// across the threadgroup.
    let uniformAxes: Set<Int>
    /// SSA name -> the loop nest the value is emitted in, outermost axis first.
    ///
    /// A value's own varying axes, plus every axis a *consumer* nests outside
    /// them — so that a value is always emitted in a prefix of its consumers'
    /// nests and its variable is still in scope where it is read.
    let paths: [String: [Int]]
    /// True when the kernel contains a `tt.dot` at all.
    let hasDot: Bool

    static let scalar = BlockLayout(
        rank: 0, shape: [], axes: [:], contractions: [], uniformAxes: [], paths: [:],
        hasDot: false)

    /// Total number of block axes, iteration and contraction together.
    var axisCount: Int { rank + contractions.count }

    func extent(ofAxis axis: Int) -> Int {
        axis < rank ? shape[axis] : contractions[axis - rank]
    }

    /// Block axes this value actually varies along (size-1 dimensions are
    /// broadcast, so the value does not depend on their index).
    func varyingAxes(of ssa: String, shape valueShape: [Int]) -> [Int] {
        guard let map = axes[ssa], map.count == valueShape.count else { return [] }
        return zip(map, valueShape).filter { $0.1 > 1 }.map(\.0)
    }

    /// True when the value varies along an axis no materialised value spans,
    /// which means it can only be built inside a `tt.dot`'s staging loops.
    func spansContraction(_ ssa: String, shape valueShape: [Int]) -> Bool {
        varyingAxes(of: ssa, shape: valueShape).contains { $0 >= rank }
    }

    /// The loop nest `ssa` is emitted in, outermost first.
    func path(of ssa: String) -> [Int] { paths[ssa] ?? [] }

    /// True when every thread walks this axis in full (see `uniformAxes`).
    func isUniform(_ axis: Int) -> Bool { uniformAxes.contains(axis) }
}

/// Decides which values a kernel has to *materialise* in an elementwise
/// iteration nest and which exist only to feed a `tt.dot`.
///
/// A deferred `tt.dot` operand is never a per-lane value: it is staged into a
/// threadgroup tile over an index space that includes the contraction dimension.
/// Everything that feeds one — the pointer arithmetic, the masks, the loads —
/// therefore has to be emitted inside the dot's own staging loops rather than
/// where it appears. This is a backward liveness walk: a value is materialised
/// when some use other than "an operand of a tt.dot" needs it, and deferred
/// otherwise. FA-2's `p` is materialised *and* a dot operand — it is reduced to
/// build `l_ij` — which is exactly why the two questions are separate.
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

                case .atomicRMW, .atomicCAS:
                    // Side-effecting like a store, and its result is a real value:
                    // both the operands and the old value it returns are live.
                    roots.append(contentsOf: instruction.kind.operandNames)
                    roots.append(contentsOf: instruction.kind.resultNames)

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

/// Union-find over axis *variables*: one per dimension of every tensor value.
private struct DisjointSet {
    private var parent: [Int] = []

    mutating func make() -> Int {
        parent.append(parent.count)
        return parent.count - 1
    }

    mutating func find(_ x: Int) -> Int {
        var root = x
        while parent[root] != root { root = parent[root] }
        var walk = x
        while parent[walk] != root {
            let next = parent[walk]
            parent[walk] = root
            walk = next
        }
        return root
    }

    mutating func union(_ a: Int, _ b: Int) {
        let (ra, rb) = (find(a), find(b))
        if ra != rb { parent[rb] = ra }
    }
}

/// Infers the block layout of one `tt.func`.
///
/// Two passes: shapes first (a light-weight rank/shape propagation that mirrors
/// what the emitter later does for full types), then the axis assignment.
///
/// The axis assignment is *unification*, not seeding: every dimension of every
/// tensor gets its own axis variable, and the ops that relate two tensors merge
/// them — elementwise ops dimension-wise, `tt.expand_dims`/`tt.reduce` around the
/// inserted or dropped dimension, `tt.trans` through its permutation, and
/// `tt.dot` by pinning `a` to (M, K), `b` to (K, N) and the accumulator/result to
/// (M, N). Two full-rank tensors that no chain relates are then merged onto one
/// index space, which is what the old identity seeding did for every tensor and
/// is still what an ordinary elementwise kernel wants.
///
/// Unification rather than seeding is what makes FlashAttention-2 expressible:
/// its two dots share axes crosswise (the first contracts over the head
/// dimension its accumulator iterates, the second over the key block its softmax
/// iterates), which no fixed (M, N, fresh-K) assignment can describe.
enum LayoutInference {
    static func compute(for function: TritonFunction) throws -> BlockLayout {
        var built = try solver(for: function)
        return try built.solve(function: function)
    }

    private static func solver(for function: TritonFunction) throws -> Solver {
        var solver = Solver()
        for argument in function.arguments {
            solver.define(argument.name, argument.type.shape ?? [], argument.loc)
        }
        try solver.walk(function.body)
        return solver
    }

    /// One `tt.expand_dims` whose source has to be rebuilt from its own copy of
    /// the index arithmetic, because some *other* `tt.expand_dims` places the
    /// same rank-1 value at a different dimension. See `AxisCloning`.
    struct ExpansionConflict {
        /// The `tt.expand_dims` result whose operand is rewritten.
        var expansion: String
        /// That operand: the root of the producer cone to clone.
        var source: String
        /// The dimension this expansion inserts, for diagnostics.
        var axis: Int
    }

    /// Rank-1 values whose elementwise class reaches `tt.expand_dims` at two
    /// different dimensions.
    ///
    /// This is the shape Triton's CSE hands you whenever two block sizes are
    /// equal: `tl.arange(0, BLOCK_M)` and `tl.arange(0, BLOCK_N)` become **one**
    /// `tt.make_range`, `offs_am` and `offs_bn` are then two `arith.addi`s over
    /// it, and the elementwise relation unifies all four onto one axis variable —
    /// after which expanding one at dimension 0 and the other at dimension 1
    /// declares the row axis and the column axis to be the same one.
    ///
    /// The conflict is returned rather than refused: the arithmetic is pure, so
    /// each expansion can have its own copy of it (`AxisCloning`).
    static func expansionConflicts(in function: TritonFunction) throws -> [ExpansionConflict] {
        var built = try self.solver(for: function)
        return built.expansionConflicts()
    }

    private struct Solver {
        /// `[]` for scalars, the dimension list for tensors.
        var shapes: [String: [Int]] = [:]
        /// Definition order, so every decision below is deterministic.
        var order: [String] = []
        /// Values that must share one axis map (same rank, elementwise-related).
        var groups: [[String]] = []
        /// `tt.expand_dims`: the result gains a size-1 dimension at `axis`.
        var expansions: [(source: String, result: String, axis: Int, loc: SourceLoc)] = []
        /// `tt.reduce`: the result drops the dimension at `axis`.
        var reductions: [(source: String, result: String, axis: Int, loc: SourceLoc)] = []
        /// `tt.trans`: `result`'s dimension `i` is `source`'s dimension `order[i]`.
        var transposes: [(source: String, result: String, order: [Int], loc: SourceLoc)] = []
        /// `tt.dot`: pins its operands onto (M, K) / (K, N) / (M, N).
        var dots: [(op: DotOp, loc: SourceLoc)] = []
        var locations: [String: SourceLoc] = [:]

        private func shape(_ ssa: String) -> [Int] { shapes[ssa] ?? [] }

        /// Records that every tensor among `values` shares one axis map.
        private mutating func relate(_ values: [String]) {
            let tensors = values.filter { !shape($0).isEmpty }
            if tensors.count > 1 { groups.append(tensors) }
        }

        mutating func define(_ ssa: String, _ shape: [Int], _ loc: SourceLoc) {
            if shapes[ssa] == nil { order.append(ssa) }
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

                case .trans(let result, let type, let source, let permutation):
                    define(result, type.shape ?? [], loc)
                    transposes.append(
                        (source: source, result: result, order: permutation, loc: loc))

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
                    guard axis == reduced.count - 1 else {
                        throw CoreError.unsupportedOp(
                            "tt.reduce over axis \(axis) of a \(reduced.count)-D tensor "
                                + "(only the last axis, which is the axis distributed across "
                                + "threads, is lowered)", loc)
                    }
                    reduced.remove(at: axis)
                    define(result, reduced, loc)
                    reductions.append((source: source, result: result, axis: axis, loc: loc))

                case .dot(let dot):
                    define(dot.result, dot.resultType.shape ?? [], loc)
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

                case .atomicRMW(let atomic):
                    define(atomic.result, shape(atomic.pointer), loc)
                    relate(
                        [atomic.pointer, atomic.value, atomic.mask, atomic.result]
                            .compactMap { $0 })

                case .atomicCAS(let atomic):
                    define(atomic.result, shape(atomic.pointer), loc)
                    relate([atomic.pointer, atomic.compare, atomic.value, atomic.result])

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
                        for (index, value) in yielded.enumerated()
                        where index < branch.results.count {
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

        // MARK: - Expansion conflicts

        /// The elementwise classes of rank-1 values, and which `tt.expand_dims`
        /// dimensions each class reaches. A class that reaches two of them is a
        /// collapse waiting to happen; every expansion after the first
        /// dimension is reported so that its producer cone can be cloned.
        mutating func expansionConflicts() -> [ExpansionConflict] {
            var sets = DisjointSet()
            var variable: [String: Int] = [:]
            for ssa in order where shape(ssa).count == 1 && shape(ssa)[0] > 1 {
                variable[ssa] = sets.make()
            }
            // The same relation `solve` unifies rank-for-rank, restricted to the
            // rank-1 members: two rank-1 values in one elementwise group share an
            // axis variable.
            for group in groups {
                var representative: String? = nil
                for member in group where variable[member] != nil {
                    if let first = representative {
                        sets.union(variable[first]!, variable[member]!)
                    } else {
                        representative = member
                    }
                }
            }

            var axesByClass: [Int: [Int]] = [:]
            var expansionsByClass: [Int: [ExpansionConflict]] = [:]
            for expansion in expansions {
                guard let root = variable[expansion.source].map({ sets.find($0) }) else { continue }
                if !(axesByClass[root] ?? []).contains(expansion.axis) {
                    axesByClass[root, default: []].append(expansion.axis)
                }
                expansionsByClass[root, default: []].append(
                    ExpansionConflict(
                        expansion: expansion.result, source: expansion.source,
                        axis: expansion.axis))
            }

            var conflicts: [ExpansionConflict] = []
            for (root, axes) in axesByClass where axes.count > 1 {
                // *Every* expansion of a conflicted class gets its own copy of
                // the arithmetic, bar one arbitrary representative — not just the
                // ones at the second dimension. With all three block sizes equal
                // the matmul tutorial has one `tt.make_range` serving as the row
                // index, the column index *and* the contraction index, and two
                // expansions at the same dimension still have to end up on two
                // different axes. Anything that really is one axis is unified
                // again downstream, by the `tt.dot`'s pinning or by whatever
                // elementwise expression reads both.
                for entry in (expansionsByClass[root] ?? []).dropFirst() {
                    conflicts.append(entry)
                }
            }
            // Deterministic: definition order of the expansion results.
            let position = Dictionary(
                uniqueKeysWithValues: order.enumerated().map { ($0.element, $0.offset) })
            return conflicts.sorted {
                (position[$0.expansion] ?? 0) < (position[$1.expansion] ?? 0)
            }
        }

        // MARK: - Solving

        mutating func solve(function: TritonFunction) throws -> BlockLayout {
            let maxRank = shapes.values.map(\.count).max() ?? 0
            guard maxRank > 0 else { return .scalar }

            var sets = DisjointSet()
            var variables: [String: [Int]] = [:]
            for ssa in order where !shape(ssa).isEmpty {
                variables[ssa] = shape(ssa).map { _ in sets.make() }
            }

            /// Unifies two axis maps dimension-wise, **skipping dimensions of
            /// extent 1**.
            ///
            /// A size-1 dimension is a broadcast placeholder, not an identity: the
            /// matmul tutorial broadcasts one `tensor<BLOCK_M x 1 x i1>` row mask
            /// into both a `BLOCK_M x BLOCK_K` and a `BLOCK_M x BLOCK_N` tensor,
            /// and unifying its second dimension with both would declare the
            /// contraction axis and the column axis to be the same one.
            func unify(_ a: String, _ b: String, dropping axis: Int? = nil) {
                guard var left = variables[a], let right = variables[b],
                    var leftShape = shapes[a], let rightShape = shapes[b]
                else { return }
                if let axis, axis < left.count, axis < leftShape.count {
                    left.remove(at: axis)
                    leftShape.remove(at: axis)
                }
                guard left.count == right.count else { return }
                for index in left.indices where leftShape[index] > 1 && rightShape[index] > 1 {
                    sets.union(left[index], right[index])
                }
            }

            // Elementwise relations: every member of a group with the same rank
            // shares one axis map.
            for group in groups {
                var byRank: [Int: String] = [:]
                for member in group {
                    guard let map = variables[member] else { continue }
                    if let representative = byRank[map.count] {
                        unify(representative, member)
                    } else {
                        byRank[map.count] = member
                    }
                }
            }
            for expansion in expansions {
                guard let map = variables[expansion.result] else { continue }
                guard expansion.axis < map.count else {
                    throw CoreError.lowering(
                        "tt.expand_dims axis \(expansion.axis) is out of range for its "
                            + "\(map.count)-D result", expansion.loc)
                }
                unify(expansion.result, expansion.source, dropping: expansion.axis)
            }
            for reduction in reductions {
                unify(reduction.source, reduction.result, dropping: reduction.axis)
            }
            for transpose in transposes {
                guard let result = variables[transpose.result],
                    let source = variables[transpose.source],
                    let sourceShape = shapes[transpose.source],
                    transpose.order.count == result.count, result.count == source.count
                else { continue }
                for (index, from) in transpose.order.enumerated() {
                    guard from >= 0, from < source.count else {
                        throw CoreError.lowering(
                            "tt.trans order \(transpose.order) is not a permutation of its "
                                + "\(source.count) dimensions", transpose.loc)
                    }
                    if sourceShape[from] > 1 { sets.union(result[index], source[from]) }
                }
            }
            for entry in dots {
                let dot = entry.op
                guard let lhs = variables[dot.lhs], lhs.count == 2,
                    let rhs = variables[dot.rhs], rhs.count == 2,
                    let result = variables[dot.result], result.count == 2,
                    let lhsShape = shapes[dot.lhs], let rhsShape = shapes[dot.rhs]
                else {
                    throw CoreError.lowering(
                        "tt.dot's operands must be rank-2 tensors, found \(dot.lhsType) * "
                            + "\(dot.rhsType) -> \(dot.resultType)", entry.loc)
                }
                if lhsShape[0] > 1 { sets.union(lhs[0], result[0]) }
                if lhsShape[1] > 1 { sets.union(lhs[1], rhs[0]) }
                if rhsShape[1] > 1 { sets.union(rhs[1], result[1]) }
                unify(dot.accumulator, dot.result)
            }

            // Two full-rank tensors that nothing above relates share the kernel's
            // one index space — the old identity seeding, kept for exactly the
            // case it was written for. Tensors a `tt.dot` already placed are
            // excluded: FA-2's `p` and `acc` are both rank 2 and must *not* be
            // merged.
            var dotVariables: Set<Int> = []
            for entry in dots {
                for value in [entry.op.lhs, entry.op.rhs, entry.op.accumulator, entry.op.result] {
                    for variable in variables[value] ?? [] { dotVariables.insert(sets.find(variable)) }
                }
            }
            // A transpose places its two values relative to each other and must not
            // be overridden by the fallback either, or the contradiction it creates
            // is reported as a size clash instead of by name.
            for transpose in transposes {
                for value in [transpose.source, transpose.result] {
                    for variable in variables[value] ?? [] { dotVariables.insert(sets.find(variable)) }
                }
            }
            let live = dots.isEmpty ? nil : DotDeferral.materialised(in: function)
            func isMaterialised(_ ssa: String) -> Bool { live?.contains(ssa) ?? true }

            var identity: String? = nil
            // Only values the kernel actually uses: a dead `tensor<BLOCK_M x
            // BLOCK_N>` constant left over from another spelling of the same
            // kernel would otherwise be merged onto the live index space and
            // declare two unrelated block dimensions to be one.
            for ssa in order where shape(ssa).count == maxRank && isMaterialised(ssa) {
                guard let map = variables[ssa] else { continue }
                guard !map.contains(where: { dotVariables.contains(sets.find($0)) }) else { continue }
                if let representative = identity {
                    unify(representative, ssa)
                } else {
                    identity = ssa
                }
            }

            // Extents. A dimension of 1 is a broadcast and says nothing.
            var extents: [Int: Int] = [:]
            for ssa in order {
                guard let map = variables[ssa] else { continue }
                for (index, dimension) in shape(ssa).enumerated() where dimension != 1 {
                    guard dimension > 0 else {
                        throw CoreError.lowering(
                            "tensor dimension must be positive, found \(dimension) in '%\(ssa)'",
                            locations[ssa] ?? function.loc)
                    }
                    let root = sets.find(map[index])
                    if let existing = extents[root], existing != dimension {
                        throw CoreError.lowering(
                            "kernel '\(function.name)' mixes tensor lengths \(existing) and "
                                + "\(dimension) along block dimension \(index); one block size "
                                + "per dimension is lowered",
                            locations[ssa] ?? function.loc)
                    }
                    extents[root] = dimension
                }
            }

            // A rank-deficient value that never reaches a full-rank one is a
            // guess the inference refuses to make.
            var fullRank: Set<Int> = []
            for ssa in order where shape(ssa).count == maxRank {
                guard let map = variables[ssa] else { continue }
                for (index, size) in shape(ssa).enumerated() where size > 1 {
                    fullRank.insert(sets.find(map[index]))
                }
            }
            for ssa in order
            where !shape(ssa).isEmpty && shape(ssa).count < maxRank && shape(ssa).contains(where: { $0 > 1 })
            {
                guard let map = variables[ssa],
                    zip(map, shape(ssa)).contains(where: { $0.1 > 1 && fullRank.contains(sets.find($0.0)) })
                else {
                    throw CoreError.lowering(
                        "cannot infer which block dimension '%\(ssa)' spans: it is "
                            + "\(shape(ssa).count)-D in a \(maxRank)-D kernel and never reaches a "
                            + "tt.expand_dims or a full-rank operation",
                        locations[ssa] ?? function.loc)
                }
            }

            // Iteration axes are the ones a materialised value spans; the rest are
            // walked only inside a tt.dot's staging loops.
            var iteration: [Int] = []
            var seenIteration: Set<Int> = []
            for ssa in order where !shape(ssa).isEmpty && isMaterialised(ssa) {
                // Only dimensions that actually vary: a size-1 dimension is a
                // broadcast placeholder and never gets a loop.
                for (index, size) in shape(ssa).enumerated() where size > 1 {
                    let root = sets.find(variables[ssa]![index])
                    if seenIteration.insert(root).inserted { iteration.append(root) }
                }
            }

            // Nesting order: a materialised value's own dimension order says which
            // of its axes is outside which. Deferred operands do not constrain it
            // — they are staged over their own two dimensions, in whichever order
            // they are written (`K^T` and `V` disagree about the head dimension,
            // and both are right).
            var successors: [Int: Set<Int>] = [:]
            var incoming: [Int: Int] = [:]
            for root in iteration {
                successors[root] = []
                incoming[root] = 0
            }
            for ssa in order where isMaterialised(ssa) {
                guard let map = variables[ssa] else { continue }
                let roots = zip(map, shape(ssa)).filter { $0.1 > 1 }.map { sets.find($0.0) }
                for index in 1..<max(1, roots.count) {
                    let (before, after) = (roots[index - 1], roots[index])
                    guard before != after, successors[before] != nil, successors[after] != nil else {
                        continue
                    }
                    if successors[before]!.insert(after).inserted {
                        incoming[after] = (incoming[after] ?? 0) + 1
                    }
                }
            }
            var ordered: [Int] = []
            var ready = iteration.filter { incoming[$0] == 0 }
            while !ready.isEmpty {
                let root = ready.removeFirst()
                ordered.append(root)
                for next in iteration where successors[root]!.contains(next) {
                    incoming[next]! -= 1
                    if incoming[next] == 0 { ready.append(next) }
                }
            }
            guard ordered.count == iteration.count else {
                throw CoreError.lowering(
                    "kernel '\(function.name)' nests its block dimensions in two incompatible "
                        + "orders; a value cannot be iterated both ways (a materialised tt.trans "
                        + "of a value that is also used untransposed does this)",
                    function.loc)
            }

            var index: [Int: Int] = [:]
            for (position, root) in ordered.enumerated() { index[root] = position }
            let rank = ordered.count
            var contractionRoots: [Int] = []
            for ssa in order {
                guard let map = variables[ssa] else { continue }
                for (position, size) in shape(ssa).enumerated() where size > 1 {
                    let root = sets.find(map[position])
                    if index[root] == nil {
                        index[root] = rank + contractionRoots.count
                        contractionRoots.append(root)
                    }
                }
            }
            // Broadcast placeholders: every size-1 dimension keeps an axis index of
            // its own so that a tile's two dimensions never collide, but it has no
            // extent and no loop.
            var placeholder = rank + contractionRoots.count
            for ssa in order {
                for variable in variables[ssa] ?? [] where index[sets.find(variable)] == nil {
                    index[sets.find(variable)] = placeholder
                    placeholder += 1
                }
            }

            var axes: [String: [Int]] = [:]
            for ssa in order {
                guard let map = variables[ssa] else { continue }
                let resolved = map.map { index[sets.find($0)]! }
                // Two varying dimensions of one value must index two different
                // block dimensions. They collapse when a single `tl.arange` is
                // expanded once into a row index and once into a column index —
                // Triton's CSE will hand us that if BLOCK_M == BLOCK_N — and the
                // value would then be a diagonal of itself.
                var seen: Set<Int> = []
                for (dimension, size) in shape(ssa).enumerated() where size > 1 {
                    guard seen.insert(resolved[dimension]).inserted else {
                        throw CoreError.axisCollapse(
                            "'%\(ssa)' indexes one block dimension with two of its own "
                                + "dimensions; a value reached both a row and a column position "
                                + "(a tl.arange expanded along axis 0 in one place and axis 1 in "
                                + "another), which describes a diagonal rather than a tile",
                            locations[ssa] ?? function.loc)
                    }
                }
                axes[ssa] = resolved
            }
            let shapeByAxis = ordered.map { extents[$0] ?? 1 }
            let contractionExtents = contractionRoots.map { extents[$0] ?? 1 }

            // Nests. A value is emitted over its own varying axes, extended with
            // every axis a consumer nests outside them so that its variable is
            // still in scope where it is read (Triton's `offs_n < N` mask spans
            // only the column axis but is consumed inside the row loop).
            var pathSets: [String: Set<Int>] = [:]
            for ssa in order where isMaterialised(ssa) && !shape(ssa).isEmpty {
                let map = axes[ssa]!
                var own: Set<Int> = []
                for (dimension, size) in shape(ssa).enumerated() where size > 1 {
                    if map[dimension] < rank { own.insert(map[dimension]) }
                }
                pathSets[ssa] = own
            }
            var related: [[String]] = groups
            for expansion in expansions { related.append([expansion.source, expansion.result]) }
            for reduction in reductions { related.append([reduction.source, reduction.result]) }
            for _ in 0..<(rank + 2) {
                var changed = false
                for group in related {
                    let members = group.filter { pathSets[$0] != nil }
                    guard !members.isEmpty else { continue }
                    var union: Set<Int> = []
                    for member in members { union.formUnion(pathSets[member]!) }
                    for member in members {
                        guard let deepest = pathSets[member]!.max() else { continue }
                        let filled = union.filter { $0 < deepest }
                        if !filled.isSubset(of: pathSets[member]!) {
                            pathSets[member]!.formUnion(filled)
                            changed = true
                        }
                    }
                }
                if !changed { break }
            }
            var paths: [String: [Int]] = [:]
            for (ssa, set) in pathSets { paths[ssa] = set.sorted() }

            var uniform: Set<Int> = []
            for (_, path) in paths where path.count > 1 {
                uniform.formUnion(path.dropLast())
            }

            return BlockLayout(
                rank: rank, shape: shapeByAxis, axes: axes, contractions: contractionExtents,
                uniformAxes: uniform, paths: paths, hasDot: !dots.isEmpty)
        }
    }
}
