import Foundation

/// Gives each `tt.expand_dims` its own copy of the index arithmetic behind it,
/// when the same rank-1 value would otherwise be placed at two different block
/// dimensions.
///
/// This is what lifts the **pairwise-distinct block sizes** restriction. The
/// emitter never identified an axis by its extent — `Layout.swift` unifies axis
/// *variables* — but Triton's CSE does something equivalent upstream: with
/// `BLOCK_M == BLOCK_N`, `tl.arange(0, BLOCK_M)` and `tl.arange(0, BLOCK_N)` are
/// one `tt.make_range`, so the matmul tutorial's
///
/// ```mlir
/// %range   = tt.make_range {start = 0, end = 64}      : tensor<64xi32>
/// %offs_am = arith.addi %pid_m_block, %range          : tensor<64xi32>
/// %offs_bn = arith.addi %pid_n_block, %range          : tensor<64xi32>
/// %a_rows  = tt.expand_dims %offs_am {axis = 1}       : ... -> tensor<64x1xi32>
/// %b_cols  = tt.expand_dims %offs_bn {axis = 0}       : ... -> tensor<1x64xi32>
/// ```
///
/// unifies `%range`, `%offs_am` and `%offs_bn` onto one axis variable, and then
/// the two expansions unify the row axis with the column axis. The accumulator
/// is refused, correctly, as a diagonal of itself.
///
/// The fix is not to weaken the unification — it is right, and the diagonal it
/// refuses is a real one — but to notice that the *sharing* is an artefact.
/// Every operation on the path is pure, so rebuilding it once per dimension
/// costs a few integer instructions per program and nothing else. This pass
/// finds the rank-1 classes that reach two `tt.expand_dims` dimensions
/// (`LayoutInference.expansionConflicts`) and, for every dimension after the
/// first, clones the producer cone under fresh names and points that expansion
/// at the copy.
///
/// It refuses to rewrite — leaving the original diagnostic to fire — when the
/// cone contains anything it cannot honestly duplicate: a `tt.reduce` (the fold
/// has happened), a `tt.dot`, a region result or block argument, a store, or an
/// atomic. Those are collapses the pass genuinely cannot fix.
enum AxisCloning {

    static func rewrite(_ function: TritonFunction) -> TritonFunction {
        let conflicts = (try? LayoutInference.expansionConflicts(in: function)) ?? []

        var definitions: [String: Instruction] = [:]
        var taken: Set<String> = Set(function.arguments.map(\.name))
        var order: [String: Int] = [:]
        var next = 0
        collect(function.body, into: &definitions, names: &taken, order: &order, next: &next)

        /// The clones one conflicted expansion needs, in program order, and the
        /// name its operand becomes.
        var plans: [String: (instructions: [Instruction], source: String)] = [:]
        var counter = 0

        for conflict in conflicts {
            guard
                let plan = clone(
                    cone: conflict.source, definitions: definitions, order: order, taken: &taken,
                    counter: &counter)
            else {
                // One unclonable cone is enough to leave the whole kernel alone:
                // a partial rewrite would report a different, less informative
                // collapse than the one the layout model already explains.
                return function
            }
            plans[conflict.expansion] = plan
        }

        var rewritten = function
        rewritten.body = splice(function.body, plans: plans)
        rewritten.body = splittingSharedBroadcasts(rewritten.body, taken: &taken)
        rewritten.body = removingDeadPureValues(rewritten.body)
        return rewritten
    }

    // MARK: - The cone

    /// Every value the source is built from that is itself rank-1 and therefore
    /// carries the axis identity being duplicated. Rank-0 operands (a program id,
    /// a stride, a splat's scalar) carry none and are shared with the original.
    private static func clone(
        cone root: String, definitions: [String: Instruction], order: [String: Int],
        taken: inout Set<String>, counter: inout Int
    ) -> (instructions: [Instruction], source: String)? {
        var members: [String] = []
        var seen: Set<String> = []
        var work = [root]
        while let value = work.popLast() {
            guard seen.insert(value).inserted else { continue }
            guard let instruction = definitions[value], isClonable(instruction.kind) else {
                return nil
            }
            members.append(value)
            for operand in instruction.kind.operandNames
            where definitions[operand] != nil && carriesAxisIdentity(definitions[operand]!.kind) {
                work.append(operand)
            }
        }

        // Program order, so the clones are still in SSA order among themselves.
        let sorted = members.sorted { (order[$0] ?? 0) < (order[$1] ?? 0) }

        counter += 1
        var renames: [String: String] = [:]
        for value in sorted {
            var candidate = "\(value)_tmaxis\(counter)"
            var bump = 0
            while taken.contains(candidate) {
                bump += 1
                candidate = "\(value)_tmaxis\(counter)_\(bump)"
            }
            taken.insert(candidate)
            renames[value] = candidate
        }

        var instructions: [Instruction] = []
        for value in sorted {
            guard let original = definitions[value],
                let kind = rename(original.kind, renames: renames)
            else { return nil }
            instructions.append(
                Instruction(kind: kind, mnemonic: original.mnemonic, loc: original.loc))
        }
        return (instructions: instructions, source: renames[root]!)
    }

    /// True for a value whose *own* axis identity is what the clone is separating:
    /// a rank-1 tensor. Everything the pass can clone is single-result, so this is
    /// a property of the defining op's result type where the op carries one, and
    /// otherwise of whether the op can produce a tensor at all.
    private static func carriesAxisIdentity(_ kind: OpKind) -> Bool {
        switch kind {
        case .programID, .numPrograms:
            return false
        case .constant(_, let type, _), .makeRange(_, let type, _, _),
            .splat(_, let type, _), .cast(_, _, _, let type), .broadcast(_, let type, _):
            return type.isTensor
        case .expandDims, .trans:
            // Rank >= 2 by construction: never part of a rank-1 cone.
            return false
        default:
            // addptr / arithmetic / load: rank follows the operands, and the only
            // way one of these reaches a rank-1 expansion source is by being
            // rank-1 itself.
            return isClonable(kind)
        }
    }

    /// Pure, single-result, and rebuildable from its operands.
    private static func isClonable(_ kind: OpKind) -> Bool {
        switch kind {
        case .constant, .makeRange, .splat, .addPtr, .intBinary, .floatBinary, .intCompare,
            .floatCompare, .cast, .unary, .select, .broadcast, .load, .programID, .numPrograms:
            return true
        // A reduce's fold has already happened; a dot's result is a tile; a
        // region result is not a value this pass can re-derive; and nothing
        // side-effecting may be duplicated.
        case .reduce, .dot, .store, .atomicRMW, .atomicCAS, .forLoop, .ifOp, .yield, .ret,
            .expandDims, .trans:
            return false
        }
    }

    // MARK: - Renaming

    private static func rename(_ kind: OpKind, renames: [String: String]) -> OpKind? {
        func map(_ value: String) -> String { renames[value] ?? value }
        switch kind {
        case .constant(let r, let type, let value):
            return .constant(result: map(r), type: type, value: value)
        case .programID(let r, let axis):
            return .programID(result: map(r), axis: axis)
        case .numPrograms(let r, let axis):
            return .numPrograms(result: map(r), axis: axis)
        case .makeRange(let r, let type, let start, let end):
            return .makeRange(result: map(r), type: type, start: start, end: end)
        case .splat(let r, let type, let source):
            return .splat(result: map(r), type: type, source: map(source))
        case .broadcast(let r, let type, let source):
            return .broadcast(result: map(r), type: type, source: map(source))
        case .addPtr(let r, let pointer, let offset):
            return .addPtr(result: map(r), pointer: map(pointer), offset: map(offset))
        case .intBinary(let r, let op, let lhs, let rhs):
            return .intBinary(result: map(r), op: op, lhs: map(lhs), rhs: map(rhs))
        case .floatBinary(let r, let op, let lhs, let rhs):
            return .floatBinary(result: map(r), op: op, lhs: map(lhs), rhs: map(rhs))
        case .intCompare(let r, let predicate, let lhs, let rhs):
            return .intCompare(result: map(r), predicate: predicate, lhs: map(lhs), rhs: map(rhs))
        case .floatCompare(let r, let predicate, let lhs, let rhs):
            return .floatCompare(
                result: map(r), predicate: predicate, lhs: map(lhs), rhs: map(rhs))
        case .cast(let r, let op, let source, let type):
            return .cast(result: map(r), op: op, source: map(source), type: type)
        case .unary(let r, let op, let source):
            return .unary(result: map(r), op: op, source: map(source))
        case .select(let r, let condition, let whenTrue, let whenFalse):
            return .select(
                result: map(r), condition: map(condition), whenTrue: map(whenTrue),
                whenFalse: map(whenFalse))
        case .load(let r, let pointer, let mask, let other):
            return .load(
                result: map(r), pointer: map(pointer), mask: mask.map(map), other: other.map(map))
        default:
            return nil
        }
    }

    // MARK: - Splicing

    private static func collect(
        _ body: [Instruction], into definitions: inout [String: Instruction],
        names: inout Set<String>, order: inout [String: Int], next: inout Int
    ) {
        for instruction in body {
            for result in instruction.kind.resultNames {
                names.insert(result)
                if definitions[result] == nil {
                    definitions[result] = instruction
                    order[result] = next
                    next += 1
                }
            }
            switch instruction.kind {
            case .forLoop(let loop):
                collect(loop.body, into: &definitions, names: &names, order: &order, next: &next)
            case .ifOp(let branch):
                collect(
                    branch.thenBody, into: &definitions, names: &names, order: &order, next: &next)
                collect(
                    branch.elseBody, into: &definitions, names: &names, order: &order, next: &next)
            default:
                continue
            }
        }
    }

    /// Inserts each plan's clones immediately in front of the `tt.expand_dims`
    /// they belong to, wherever in the region tree that is. Every value a clone
    /// reads was in scope where its original was defined, which is at or outside
    /// this point, so the insertion is always well-formed.
    private static func splice(
        _ body: [Instruction], plans: [String: (instructions: [Instruction], source: String)]
    ) -> [Instruction] {
        var result: [Instruction] = []
        for instruction in body {
            switch instruction.kind {
            case .expandDims(let value, let type, _, let axis) where plans[value] != nil:
                let plan = plans[value]!
                result.append(contentsOf: plan.instructions)
                result.append(
                    Instruction(
                        kind: .expandDims(
                            result: value, type: type, source: plan.source, axis: axis),
                        mnemonic: instruction.mnemonic, loc: instruction.loc))
            case .forLoop(var loop):
                loop.body = splice(loop.body, plans: plans)
                result.append(
                    Instruction(
                        kind: .forLoop(loop), mnemonic: instruction.mnemonic, loc: instruction.loc))
            case .ifOp(var branch):
                branch.thenBody = splice(branch.thenBody, plans: plans)
                branch.elseBody = splice(branch.elseBody, plans: plans)
                result.append(
                    Instruction(
                        kind: .ifOp(branch), mnemonic: instruction.mnemonic, loc: instruction.loc))
            default:
                result.append(instruction)
            }
        }
        return result
    }

    /// Drops pure single-result instructions nothing reads any more.
    ///
    /// Rewriting an expansion's operand usually orphans the chain it used to
    /// read — the matmul tutorial's `offs_bn` has exactly one use. An orphan is
    /// harmless (it would be emitted and never read) but it still carries axis
    /// identity into the inference, so it is cleaner to remove it than to reason
    /// about what it merges with.
    private static func removingDeadPureValues(_ body: [Instruction]) -> [Instruction] {
        var body = body
        var changed = true
        while changed {
            changed = false
            var used: Set<String> = []
            func mark(_ body: [Instruction]) {
                for instruction in body {
                    used.formUnion(instruction.kind.operandNames)
                    switch instruction.kind {
                    case .forLoop(let loop): mark(loop.body)
                    case .ifOp(let branch):
                        mark(branch.thenBody)
                        mark(branch.elseBody)
                    default: continue
                    }
                }
            }
            mark(body)
            func prune(_ body: [Instruction]) -> [Instruction] {
                var result: [Instruction] = []
                for instruction in body {
                    switch instruction.kind {
                    case .forLoop(var loop):
                        loop.body = prune(loop.body)
                        result.append(
                            Instruction(
                                kind: .forLoop(loop), mnemonic: instruction.mnemonic,
                                loc: instruction.loc))
                    case .ifOp(var branch):
                        branch.thenBody = prune(branch.thenBody)
                        branch.elseBody = prune(branch.elseBody)
                        result.append(
                            Instruction(
                                kind: .ifOp(branch), mnemonic: instruction.mnemonic,
                                loc: instruction.loc))
                    default:
                        let results = instruction.kind.resultNames
                        if isClonable(instruction.kind), !results.isEmpty,
                            results.allSatisfy({ !used.contains($0) })
                        {
                            changed = true
                            continue
                        }
                        result.append(instruction)
                    }
                }
                return result
            }
            body = prune(body)
        }
        return body
    }
}

// MARK: - Splitting a shared broadcast

extension AxisCloning {

    /// Gives each consumer of a multiply-used `tt.broadcast` its own copy of it.
    ///
    /// The second half of the same problem, one rank up. With *all three* block
    /// sizes equal, Triton's CSE also shares the widening itself: the matmul
    /// tutorial's `tt.broadcast %b_cols : tensor<1x64xi32> -> tensor<64x64xi32>`
    /// is both `B`'s column offsets — where the leading dimension is the
    /// contraction — and `C`'s — where it is the row block. One value, two
    /// meanings for its leading dimension, and the two axes unify through it.
    ///
    /// A `tt.broadcast` emits no code at all (it is a relabelling), so a copy per
    /// consumer is free. And the copy cannot *invent* an axis: a broadcast
    /// unifies only the dimensions that vary in its source, so the widened
    /// dimension's identity comes entirely from whatever the consumer relates it
    /// to — which is exactly the thing that differs.
    static func splittingSharedBroadcasts(
        _ body: [Instruction], taken: inout Set<String>
    ) -> [Instruction] {
        var uses: [String: Int] = [:]
        count(body, into: &uses)
        var shared: [String: Instruction] = [:]
        collectBroadcasts(body, into: &shared, uses: uses)
        guard !shared.isEmpty else { return body }

        var consumed: [String: Int] = [:]
        var counter = 0
        return split(body, shared: shared, consumed: &consumed, taken: &taken, counter: &counter)
    }

    private static func count(_ body: [Instruction], into uses: inout [String: Int]) {
        for instruction in body {
            for operand in instruction.kind.operandNames { uses[operand, default: 0] += 1 }
            switch instruction.kind {
            case .forLoop(let loop): count(loop.body, into: &uses)
            case .ifOp(let branch):
                count(branch.thenBody, into: &uses)
                count(branch.elseBody, into: &uses)
            default: continue
            }
        }
    }

    private static func collectBroadcasts(
        _ body: [Instruction], into shared: inout [String: Instruction], uses: [String: Int]
    ) {
        for instruction in body {
            if case .broadcast(let result, _, _) = instruction.kind, (uses[result] ?? 0) > 1 {
                shared[result] = instruction
            }
            switch instruction.kind {
            case .forLoop(let loop): collectBroadcasts(loop.body, into: &shared, uses: uses)
            case .ifOp(let branch):
                collectBroadcasts(branch.thenBody, into: &shared, uses: uses)
                collectBroadcasts(branch.elseBody, into: &shared, uses: uses)
            default: continue
            }
        }
    }

    private static func split(
        _ body: [Instruction], shared: [String: Instruction], consumed: inout [String: Int],
        taken: inout Set<String>, counter: inout Int
    ) -> [Instruction] {
        var result: [Instruction] = []
        for instruction in body {
            var kind = instruction.kind
            var prefix: [Instruction] = []
            for operand in Set(instruction.kind.operandNames) {
                guard let original = shared[operand] else { continue }
                let index = consumed[operand, default: 0]
                consumed[operand] = index + 1
                guard index > 0 else { continue }
                counter += 1
                var name = "\(operand)_tmwide\(counter)"
                while taken.contains(name) {
                    counter += 1
                    name = "\(operand)_tmwide\(counter)"
                }
                taken.insert(name)
                guard case .broadcast(_, let type, let source) = original.kind,
                    let substituted = substitutingOperands(kind, [operand: name])
                else { continue }
                prefix.append(
                    Instruction(
                        kind: .broadcast(result: name, type: type, source: source),
                        mnemonic: original.mnemonic, loc: original.loc))
                kind = substituted
            }
            switch kind {
            case .forLoop(var loop):
                loop.body = split(
                    loop.body, shared: shared, consumed: &consumed, taken: &taken,
                    counter: &counter)
                kind = .forLoop(loop)
            case .ifOp(var branch):
                branch.thenBody = split(
                    branch.thenBody, shared: shared, consumed: &consumed, taken: &taken,
                    counter: &counter)
                branch.elseBody = split(
                    branch.elseBody, shared: shared, consumed: &consumed, taken: &taken,
                    counter: &counter)
                kind = .ifOp(branch)
            default:
                break
            }
            result.append(contentsOf: prefix)
            result.append(
                Instruction(kind: kind, mnemonic: instruction.mnemonic, loc: instruction.loc))
        }
        return result
    }

    /// Every op's operands, remapped. Unlike `rename` this leaves results alone
    /// and covers the whole language, because a shared broadcast can be consumed
    /// by anything.
    private static func substitutingOperands(
        _ kind: OpKind, _ map: [String: String]
    ) -> OpKind? {
        func m(_ value: String) -> String { map[value] ?? value }
        switch kind {
        case .constant, .programID, .numPrograms, .makeRange, .ret:
            return kind
        case .splat(let r, let type, let source):
            return .splat(result: r, type: type, source: m(source))
        case .expandDims(let r, let type, let source, let axis):
            return .expandDims(result: r, type: type, source: m(source), axis: axis)
        case .trans(let r, let type, let source, let order):
            return .trans(result: r, type: type, source: m(source), order: order)
        case .broadcast(let r, let type, let source):
            return .broadcast(result: r, type: type, source: m(source))
        case .cast(let r, let op, let source, let type):
            return .cast(result: r, op: op, source: m(source), type: type)
        case .unary(let r, let op, let source):
            return .unary(result: r, op: op, source: m(source))
        case .reduce(let r, let source, let axis, let op):
            return .reduce(result: r, source: m(source), axis: axis, op: op)
        case .addPtr(let r, let pointer, let offset):
            return .addPtr(result: r, pointer: m(pointer), offset: m(offset))
        case .intBinary(let r, let op, let lhs, let rhs):
            return .intBinary(result: r, op: op, lhs: m(lhs), rhs: m(rhs))
        case .floatBinary(let r, let op, let lhs, let rhs):
            return .floatBinary(result: r, op: op, lhs: m(lhs), rhs: m(rhs))
        case .intCompare(let r, let predicate, let lhs, let rhs):
            return .intCompare(result: r, predicate: predicate, lhs: m(lhs), rhs: m(rhs))
        case .floatCompare(let r, let predicate, let lhs, let rhs):
            return .floatCompare(result: r, predicate: predicate, lhs: m(lhs), rhs: m(rhs))
        case .select(let r, let condition, let whenTrue, let whenFalse):
            return .select(
                result: r, condition: m(condition), whenTrue: m(whenTrue),
                whenFalse: m(whenFalse))
        case .dot(var dot):
            dot.lhs = m(dot.lhs)
            dot.rhs = m(dot.rhs)
            dot.accumulator = m(dot.accumulator)
            return .dot(dot)
        case .load(let r, let pointer, let mask, let other):
            return .load(result: r, pointer: m(pointer), mask: mask.map(m), other: other.map(m))
        case .store(let pointer, let value, let mask):
            return .store(pointer: m(pointer), value: m(value), mask: mask.map(m))
        case .atomicRMW(var atomic):
            atomic.pointer = m(atomic.pointer)
            atomic.value = m(atomic.value)
            atomic.mask = atomic.mask.map(m)
            return .atomicRMW(atomic)
        case .atomicCAS(var atomic):
            atomic.pointer = m(atomic.pointer)
            atomic.compare = m(atomic.compare)
            atomic.value = m(atomic.value)
            return .atomicCAS(atomic)
        case .yield(let values):
            return .yield(values: values.map(m))
        case .forLoop(var loop):
            loop.lowerBound = m(loop.lowerBound)
            loop.upperBound = m(loop.upperBound)
            loop.step = m(loop.step)
            loop.iterArguments = loop.iterArguments.map {
                (name: $0.name, initial: m($0.initial), type: $0.type)
            }
            return .forLoop(loop)
        case .ifOp(var branch):
            branch.condition = m(branch.condition)
            return .ifOp(branch)
        }
    }
}
