import Foundation

/// Attribute values parsed out of `{...}` dictionaries. Most are ignored by the
/// emitter; `tt.make_range`, `tt.expand_dims`, `tt.reduce` and `arith.constant`
/// are the ops that actually read them back.
indirect enum Attr {
    case integer(Int64)
    case float(Double)
    case boolean(Bool)
    case string(String)
    case ident(String)
    case type(TMType)
    case list([Attr])
    case dict([String: Attr])

    var intValue: Int64? {
        if case .integer(let v) = self { return v }
        return nil
    }
}

/// Recursive-descent parser for the subset of Triton IR (`ttir`, and the
/// structurally identical `ttgir` for elementwise ops) that this backend lowers.
///
/// It parses MLIR's *pretty* op syntax. The generic form (`"tt.load"(%0) : ...`)
/// is rejected with a clear message rather than silently mis-parsed — with one
/// exception: `tt.reduce` carries a combine region and Triton always prints it
/// generically, so that one spelling is accepted.
public struct TritonIRParser {
    private var tokens: [Token]
    private var pos = 0

    public init(_ text: String) throws {
        var lexer = Lexer(text)
        tokens = try lexer.tokenize()
    }

    public static func parse(_ text: String) throws -> TritonModule {
        var parser = try TritonIRParser(text)
        return try parser.parseModule()
    }

    // MARK: - Token helpers

    private func peek(_ offset: Int = 0) -> Token {
        let i = pos + offset
        return i < tokens.count ? tokens[i] : tokens[tokens.count - 1]
    }

    private var currentLoc: SourceLoc { peek().loc }

    @discardableResult
    private mutating func advance() -> Token {
        let t = peek()
        if pos < tokens.count - 1 { pos += 1 }
        return t
    }

    private mutating func consume(punct: String) -> Bool {
        if peek().isPunct(punct) {
            advance()
            return true
        }
        return false
    }

    private mutating func consume(ident: String) -> Bool {
        if peek().isIdent(ident) {
            advance()
            return true
        }
        return false
    }

    private mutating func expect(punct: String) throws {
        guard consume(punct: punct) else {
            throw CoreError.parse("expected '\(punct)', found '\(peek().text)'", currentLoc)
        }
    }

    private mutating func expectIdent() throws -> Token {
        guard peek().kind == .ident else {
            throw CoreError.parse("expected an identifier, found '\(peek().text)'", currentLoc)
        }
        return advance()
    }

    private mutating func expectSSA() throws -> String {
        guard peek().kind == .ssa else {
            throw CoreError.parse("expected an SSA value (%name), found '\(peek().text)'", currentLoc)
        }
        return advance().text
    }

    private mutating func expectInteger() throws -> Int64 {
        guard peek().kind == .integer, let v = Int64(peek().text) else {
            throw CoreError.parse("expected an integer literal, found '\(peek().text)'", currentLoc)
        }
        advance()
        return v
    }

    // MARK: - Module

    public mutating func parseModule() throws -> TritonModule {
        var functions: [TritonFunction] = []

        loop: while true {
            let token = peek()
            switch token.kind {
            case .eof:
                break loop
            case .attrAlias:
                try skipAliasDefinition()
            case .punct where token.text == "}":
                advance()
                try skipTrailingLocation()
            case .ident where token.text == "loc":
                try skipTrailingLocation()
            case .ident where token.text == "module":
                advance()
                if consume(ident: "attributes") {
                    _ = try parseAttrDict()
                }
                try expect(punct: "{")
            case .ident where token.text == "tt.func" || token.text == "func.func":
                functions.append(try parseFunction())
            case .string:
                throw CoreError.parse(
                    "generic MLIR op syntax (\"\(token.text)\"(...)) is not supported; "
                        + "emit the pretty form (mlir-print-op-generic=false)",
                    token.loc)
            default:
                throw CoreError.parse(
                    "expected a module or a tt.func at top level, found '\(token.text)'", token.loc)
            }
        }

        guard !functions.isEmpty else {
            throw CoreError.parse("module contains no tt.func to lower", SourceLoc(line: 1, column: 1))
        }
        return TritonModule(functions: functions)
    }

    /// `#loc1 = loc(...)` / `#blocked = #ttg.blocked<...>` — recorded nowhere.
    private mutating func skipAliasDefinition() throws {
        advance() // the alias name
        try expect(punct: "=")
        _ = try parseAttrValue()
    }

    // MARK: - Functions

    private mutating func parseFunction() throws -> TritonFunction {
        let loc = currentLoc
        advance() // tt.func

        // Optional visibility.
        _ = consume(ident: "public") || consume(ident: "private") || consume(ident: "nested")

        guard peek().kind == .symbol else {
            throw CoreError.parse("expected @name after tt.func, found '\(peek().text)'", currentLoc)
        }
        let name = advance().text

        try expect(punct: "(")
        var arguments: [FunctionArgument] = []
        if !consume(punct: ")") {
            repeat {
                let argLoc = currentLoc
                let argName = try expectSSA()
                try expect(punct: ":")
                let type = try parseType()
                if peek().isPunct("{") {
                    _ = try parseAttrDict()
                }
                // Real Triton output names its arguments:
                // `%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x_ptr"(#loc))`.
                try skipTrailingLocation()
                arguments.append(
                    FunctionArgument(name: argName, type: type, index: arguments.count, loc: argLoc))
            } while consume(punct: ",")
            try expect(punct: ")")
        }

        if consume(ident: "attributes") {
            _ = try parseAttrDict()
        }
        if peek().kind == .arrow {
            advance()
            if consume(punct: "(") {
                try expect(punct: ")")
            } else {
                let resultLoc = currentLoc
                let type = try parseType()
                throw CoreError.lowering(
                    "kernel '\(name)' returns \(type); Triton kernels lowered to Metal must return void",
                    resultLoc)
            }
        }
        if peek().kind == .attrAlias || peek().isIdent("loc") {
            try skipTrailingLocation()
        }

        let body = try parseRegionBody(what: "tt.func @\(name)")
        try skipTrailingLocation()

        return TritonFunction(name: name, arguments: arguments, body: body, loc: loc)
    }

    /// `{ <instruction>* }` — a region body with no block arguments.
    private mutating func parseRegionBody(what: String) throws -> [Instruction] {
        try expect(punct: "{")
        var body: [Instruction] = []
        while !peek().isPunct("}") {
            if peek().kind == .eof {
                throw CoreError.parse("unterminated body for \(what)", currentLoc)
            }
            body.append(try parseInstruction())
        }
        try expect(punct: "}")
        return body
    }

    // MARK: - Instructions

    private mutating func parseInstruction() throws -> Instruction {
        let loc = currentLoc

        var results: [String] = []
        if peek().kind == .ssa {
            let name = peek().text
            if peek(1).isPunct(":") {
                // `%5:2 = ...` — a multi-result op; results are `%5#0`, `%5#1`.
                advance()
                advance()
                let count = try expectInteger()
                guard count > 0, count <= 32 else {
                    throw CoreError.parse("implausible result count \(count)", loc)
                }
                results = (0..<Int(count)).map { "\(name)#\($0)" }
                guard consume(punct: "=") else {
                    throw CoreError.parse("expected '=' after '%\(name):\(count)'", currentLoc)
                }
            } else {
                guard peek(1).isPunct("=") else {
                    throw CoreError.parse("expected '=' after '%\(name)'", peek(1).loc)
                }
                advance()
                advance() // '='
                results = [name]
            }
        }

        if peek().kind == .string {
            // Generic syntax. `tt.reduce` is the one op Triton always prints this
            // way (it carries a combine region), so it is accepted here.
            guard peek().text == "tt.reduce" else {
                throw CoreError.parse(
                    "generic MLIR op syntax (\"\(peek().text)\"(...)) is not supported; "
                        + "emit the pretty form (mlir-print-op-generic=false)",
                    currentLoc)
            }
            advance()
            let kind = try parseGenericReduce(result: try single(results, "tt.reduce", loc), loc: loc)
            try skipTrailingLocation()
            return Instruction(kind: kind, mnemonic: "tt.reduce", loc: loc)
        }

        let mnemonic = try expectIdent().text
        let kind = try parseOpKind(mnemonic: mnemonic, results: results, loc: loc)
        try skipTrailingLocation()
        return Instruction(kind: kind, mnemonic: mnemonic, loc: loc)
    }

    private func single(_ results: [String], _ mnemonic: String, _ loc: SourceLoc) throws -> String {
        guard results.count == 1 else {
            throw CoreError.parse(
                results.isEmpty
                    ? "'\(mnemonic)' must produce a result"
                    : "'\(mnemonic)' produces one result, found \(results.count)", loc)
        }
        return results[0]
    }

    private mutating func parseOpKind(mnemonic: String, results: [String], loc: SourceLoc) throws
        -> OpKind
    {
        switch mnemonic {
        case "arith.constant":
            return try parseConstant(result: try single(results, mnemonic, loc), loc: loc)

        case "tt.get_program_id", "tt.get_num_programs":
            let result = try single(results, mnemonic, loc)
            let axis = try parseAxis(loc: loc)
            _ = try parseOptionalTrailingTypes()
            return mnemonic == "tt.get_program_id"
                ? .programID(result: result, axis: axis)
                : .numPrograms(result: result, axis: axis)

        case "tt.make_range":
            let result = try single(results, mnemonic, loc)
            let attrs = try parseAttrDict()
            guard let start = attrs["start"]?.intValue, let end = attrs["end"]?.intValue else {
                throw CoreError.parse("tt.make_range requires integer 'start' and 'end' attributes", loc)
            }
            let types = try parseOptionalTrailingTypes()
            guard let type = types.last else {
                throw CoreError.parse("tt.make_range requires a result type", loc)
            }
            return .makeRange(result: result, type: type, start: Int(start), end: Int(end))

        case "tt.splat":
            let result = try single(results, mnemonic, loc)
            let source = try expectSSA()
            skipAttrDictIfPresent()
            let types = try parseOptionalTrailingTypes()
            guard let type = types.last else {
                throw CoreError.parse("tt.splat requires a result type", loc)
            }
            return .splat(result: result, type: type, source: source)

        case "tt.expand_dims":
            let result = try single(results, mnemonic, loc)
            let source = try expectSSA()
            let attrs = try parseAttrDict()
            guard let axis = attrs["axis"]?.intValue, axis >= 0 else {
                throw CoreError.parse("tt.expand_dims requires a non-negative 'axis' attribute", loc)
            }
            let types = try parseOptionalTrailingTypes()
            guard let type = types.last, type.isTensor else {
                throw CoreError.parse("tt.expand_dims requires a tensor result type", loc)
            }
            return .expandDims(result: result, type: type, source: source, axis: Int(axis))

        case "tt.trans":
            let result = try single(results, mnemonic, loc)
            let source = try expectSSA()
            var permutation: [Int]? = nil
            if peek().isPunct("{") {
                let attrs = try parseAttrDict()
                if case .list(let items)? = attrs["order"] {
                    permutation = items.compactMap { $0.intValue.map(Int.init) }
                    guard permutation?.count == items.count else {
                        throw CoreError.parse("tt.trans 'order' must be a list of integers", loc)
                    }
                }
            }
            let types = try parseOptionalTrailingTypes()
            guard let type = types.last, let shape = type.shape else {
                throw CoreError.parse("tt.trans requires a tensor result type", loc)
            }
            // Triton omits `order` for the 2-D case it emits most, where the only
            // permutation is the reversal.
            let order = permutation ?? Array((0..<shape.count).reversed())
            guard order.count == shape.count, Set(order) == Set(0..<shape.count) else {
                throw CoreError.parse(
                    "tt.trans order \(order) is not a permutation of \(shape.count) dimensions", loc)
            }
            return .trans(result: result, type: type, source: source, order: order)

        case "tt.broadcast":
            let result = try single(results, mnemonic, loc)
            let source = try expectSSA()
            skipAttrDictIfPresent()
            let types = try parseOptionalTrailingTypes()
            guard let type = types.last, type.isTensor else {
                throw CoreError.parse("tt.broadcast requires a tensor result type", loc)
            }
            return .broadcast(result: result, type: type, source: source)

        case "tt.addptr":
            let result = try single(results, mnemonic, loc)
            let pointer = try expectSSA()
            try expect(punct: ",")
            let offset = try expectSSA()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            return .addPtr(result: result, pointer: pointer, offset: offset)

        case "tt.load":
            let result = try single(results, mnemonic, loc)
            let operands = try parseOperandList()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            guard (1...3).contains(operands.count) else {
                throw CoreError.parse(
                    "tt.load takes 1-3 operands (ptr[, mask[, other]]), found \(operands.count)", loc)
            }
            return .load(
                result: result,
                pointer: operands[0],
                mask: operands.count > 1 ? operands[1] : nil,
                other: operands.count > 2 ? operands[2] : nil)

        case "tt.store":
            let operands = try parseOperandList()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            guard (2...3).contains(operands.count) else {
                throw CoreError.parse(
                    "tt.store takes 2-3 operands (ptr, value[, mask]), found \(operands.count)", loc)
            }
            return .store(
                pointer: operands[0], value: operands[1], mask: operands.count > 2 ? operands[2] : nil)

        case "tt.atomic_rmw":
            let result = try single(results, mnemonic, loc)
            let kindText = try expectIdent().text
            guard let kind = AtomicRMWOp(rawValue: kindText) else {
                throw CoreError.parse(
                    "unknown tt.atomic_rmw kind '\(kindText)'; Triton spells them "
                        + AtomicRMWOp.allSpellings, loc)
            }
            try expect(punct: ",")
            let semantic = try parseAtomicSemantic(loc)
            try expect(punct: ",")
            let scope = try parseAtomicScope(loc)
            try expect(punct: ",")
            let operands = try parseOperandList()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            guard (2...3).contains(operands.count) else {
                throw CoreError.parse(
                    "tt.atomic_rmw takes 2-3 operands (ptr, val[, mask]), found \(operands.count)",
                    loc)
            }
            return .atomicRMW(
                AtomicRMW(
                    result: result, op: kind, pointer: operands[0], value: operands[1],
                    mask: operands.count > 2 ? operands[2] : nil, semantic: semantic, scope: scope))

        case "tt.atomic_cas":
            let result = try single(results, mnemonic, loc)
            let semantic = try parseAtomicSemantic(loc)
            try expect(punct: ",")
            let scope = try parseAtomicScope(loc)
            try expect(punct: ",")
            let operands = try parseOperandList()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            guard operands.count == 3 else {
                throw CoreError.parse(
                    "tt.atomic_cas takes three operands (ptr, cmp, val), found \(operands.count)",
                    loc)
            }
            return .atomicCAS(
                AtomicCAS(
                    result: result, pointer: operands[0], compare: operands[1], value: operands[2],
                    semantic: semantic, scope: scope))

        case "tt.dot":
            return .dot(try parseDot(result: try single(results, mnemonic, loc), loc: loc))

        case "tt.reduce":
            throw CoreError.parse(
                "tt.reduce carries a combine region, which Triton prints in generic form "
                    + "(\"tt.reduce\"(%x) <{axis = N}> ({ ... })); the pretty spelling is not valid IR",
                loc)

        case "tt.return", "func.return":
            if peek().kind == .ssa {
                throw CoreError.lowering(
                    "returning a value from a Triton kernel is not supported on Metal", loc)
            }
            _ = try parseOptionalTrailingTypes()
            return .ret

        case "arith.select":
            let result = try single(results, mnemonic, loc)
            let condition = try expectSSA()
            try expect(punct: ",")
            let whenTrue = try expectSSA()
            try expect(punct: ",")
            let whenFalse = try expectSSA()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            return .select(
                result: result, condition: condition, whenTrue: whenTrue, whenFalse: whenFalse)

        case "arith.cmpi", "arith.cmpf":
            let result = try single(results, mnemonic, loc)
            let predicate = try expectIdent().text
            try expect(punct: ",")
            let lhs = try expectSSA()
            try expect(punct: ",")
            let rhs = try expectSSA()
            skipAttrDictIfPresent()
            _ = try parseOptionalTrailingTypes()
            return mnemonic == "arith.cmpi"
                ? .intCompare(result: result, predicate: predicate, lhs: lhs, rhs: rhs)
                : .floatCompare(result: result, predicate: predicate, lhs: lhs, rhs: rhs)

        case "scf.for":
            return .forLoop(try parseForLoop(results: results, loc: loc))

        case "scf.if":
            return .ifOp(try parseIfOp(results: results, loc: loc))

        case "scf.yield", "tt.reduce.return":
            let values = try parseOperandList()
            _ = try parseOptionalTrailingTypes()
            return .yield(values: values)

        default:
            if let op = TritonIRParser.castOps[mnemonic] {
                let result = try single(results, mnemonic, loc)
                let source = try expectSSA()
                skipAttrDictIfPresent()
                let type = try parseCastResultType(mnemonic: mnemonic, loc: loc)
                return .cast(result: result, op: op, source: source, type: type)
            }
            if let op = TritonIRParser.mathOps[mnemonic] {
                let result = try single(results, mnemonic, loc)
                let source = try expectSSA()
                skipAttrDictIfPresent()
                _ = try parseOptionalTrailingTypes()
                return .unary(result: result, op: op, source: source)
            }
            if let op = TritonIRParser.integerBinaryOps[mnemonic] {
                let result = try single(results, mnemonic, loc)
                let (lhs, rhs) = try parseBinaryOperands()
                return .intBinary(result: result, op: op, lhs: lhs, rhs: rhs)
            }
            if let op = TritonIRParser.floatBinaryOps[mnemonic] {
                let result = try single(results, mnemonic, loc)
                let (lhs, rhs) = try parseBinaryOperands()
                return .floatBinary(result: result, op: op, lhs: lhs, rhs: rhs)
            }
            throw CoreError.unsupportedOp(mnemonic, loc)
        }
    }

    static let integerBinaryOps: [String: IntBinaryOp] = [
        "arith.addi": .add, "arith.subi": .sub, "arith.muli": .mul,
        "arith.divsi": .divs, "arith.divui": .divu,
        "arith.remsi": .rems, "arith.remui": .remu,
        "arith.andi": .and, "arith.ori": .or, "arith.xori": .xor,
        "arith.shli": .shl, "arith.shrsi": .shrs, "arith.shrui": .shru,
        "arith.maxsi": .maxs, "arith.minsi": .mins,
        "arith.maxui": .maxu, "arith.minui": .minu,
    ]

    static let floatBinaryOps: [String: FloatBinaryOp] = [
        "arith.addf": .add, "arith.subf": .sub, "arith.mulf": .mul, "arith.divf": .div,
        "arith.maximumf": .maximum, "arith.minimumf": .minimum,
        "arith.maxnumf": .maximum, "arith.minnumf": .minimum,
    ]

    static let castOps: [String: CastOp] = [
        "arith.sitofp": .sitofp, "arith.uitofp": .uitofp,
        "arith.fptosi": .fptosi, "arith.fptoui": .fptoui,
        "arith.extsi": .extsi, "arith.extui": .extui, "arith.trunci": .trunci,
        "arith.truncf": .truncf, "arith.extf": .extf, "arith.bitcast": .bitcast,
    ]

    static let mathOps: [String: MathOp] = [
        "math.exp": .exp, "math.exp2": .exp2, "math.log": .log, "math.log2": .log2,
        "math.sqrt": .sqrt, "math.rsqrt": .rsqrt, "math.sin": .sin, "math.cos": .cos,
        "math.tanh": .tanh, "math.erf": .erf,
        "math.absf": .absf, "math.absi": .absi, "math.abs": .absf,
        "math.floor": .floor, "math.ceil": .ceil,
        // Triton emits these through its own dialect for the same operations.
        "tt.exp": .exp, "tt.log": .log, "tt.sqrt": .sqrt, "tt.sin": .sin, "tt.cos": .cos,
    ]

    private mutating func parseBinaryOperands() throws -> (String, String) {
        let lhs = try expectSSA()
        try expect(punct: ",")
        let rhs = try expectSSA()
        skipAttrDictIfPresent()
        _ = try parseOptionalTrailingTypes()
        return (lhs, rhs)
    }

    /// `: <source type> to <result type>` — MLIR's conversion-op signature.
    private mutating func parseCastResultType(mnemonic: String, loc: SourceLoc) throws -> TMType {
        guard consume(punct: ":") else {
            throw CoreError.parse("\(mnemonic) requires a ': T to U' type signature", loc)
        }
        _ = try parseType()
        guard consume(ident: "to") else {
            throw CoreError.parse(
                "expected 'to' in \(mnemonic)'s type signature, found '\(peek().text)'", currentLoc)
        }
        return try parseType()
    }

    /// The `sem` and `scope` attributes Triton prints as bare keywords in front of
    /// an atomic's operands. Both are accepted and carried; the emitter ignores
    /// them, because Metal has exactly one memory order for device atomics.
    private mutating func parseAtomicSemantic(_ loc: SourceLoc) throws -> AtomicSemantic {
        let text = try expectIdent().text
        guard let semantic = AtomicSemantic(rawValue: text) else {
            throw CoreError.parse(
                "unknown atomic memory semantics '\(text)'; expected one of relaxed, acquire, "
                    + "release, acq_rel", loc)
        }
        return semantic
    }

    private mutating func parseAtomicScope(_ loc: SourceLoc) throws -> AtomicScope {
        let text = try expectIdent().text
        guard let scope = AtomicScope(rawValue: text) else {
            throw CoreError.parse(
                "unknown atomic sync scope '\(text)'; expected one of cta, gpu, sys", loc)
        }
        return scope
    }

    private mutating func parseOperandList() throws -> [String] {
        var operands: [String] = []
        guard peek().kind == .ssa else { return operands }
        repeat {
            operands.append(try expectSSA())
        } while peek().isPunct(",") && peek(1).kind == .ssa && consume(punct: ",")
        return operands
    }

    // MARK: - tt.dot

    /// `tt.dot %a, %b, %acc[, inputPrecision = tf32][, maxNumImpreciseAcc = 0]
    ///  [{allowTF32 = true}] : tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxU>`
    ///
    /// Both the current `inputPrecision = ...` spelling and the older
    /// `{allowTF32 = ...}` attribute dictionary are accepted and ignored: Metal's
    /// simdgroup matrices have one precision per element type.
    private mutating func parseDot(result: String, loc: SourceLoc) throws -> DotOp {
        let lhs = try expectSSA()
        try expect(punct: ",")
        let rhs = try expectSSA()
        guard consume(punct: ",") else {
            throw CoreError.parse(
                "tt.dot takes three operands (%a, %b, %acc); Triton always passes an "
                    + "accumulator, zero-filled when the kernel does not supply one", loc)
        }
        let accumulator = try expectSSA()

        // Trailing `, name = value` pairs (`inputPrecision = tf32`,
        // `maxNumImpreciseAcc = 0`). The value is a single token: consuming it
        // with the general attribute parser would swallow the `:` that starts the
        // type signature as if it were an attribute's type suffix.
        while peek().isPunct(",") && peek(1).kind == .ident && peek(2).isPunct("=") {
            advance()
            advance()
            advance()
            advance()
        }
        skipAttrDictIfPresent()

        guard consume(punct: ":") else {
            throw CoreError.parse(
                "tt.dot requires a ': tensor<MxKxT> * tensor<KxNxT> -> tensor<MxNxU>' signature",
                loc)
        }
        let lhsType = try parseType()
        guard consume(punct: "*") else {
            throw CoreError.parse(
                "expected '*' between tt.dot's operand types, found '\(peek().text)'", currentLoc)
        }
        let rhsType = try parseType()
        guard peek().kind == .arrow else {
            throw CoreError.parse(
                "expected '->' before tt.dot's result type, found '\(peek().text)'", currentLoc)
        }
        advance()
        let resultType = try parseType()

        // The signature is self-describing, so check it here rather than after
        // layout inference — which would otherwise report a mismatched contraction
        // as a block-size clash.
        guard case .tensor(let lhsShape, let lhsElement) = lhsType, lhsShape.count == 2,
            case .tensor(let rhsShape, let rhsElement) = rhsType, rhsShape.count == 2,
            case .tensor(let resultShape, let resultElement) = resultType, resultShape.count == 2
        else {
            throw CoreError.lowering(
                "tt.dot operates on rank-2 tensors, found \(lhsType) * \(rhsType) -> \(resultType)",
                loc)
        }
        guard lhsShape[1] == rhsShape[0] else {
            throw CoreError.lowering(
                "tt.dot's contracted dimensions differ: \(lhsType) * \(rhsType)", loc)
        }
        guard resultShape == [lhsShape[0], rhsShape[1]] else {
            throw CoreError.lowering(
                "tt.dot result \(resultType) does not match \(lhsShape[0])x\(rhsShape[1])", loc)
        }
        guard lhsElement == rhsElement else {
            throw CoreError.lowering(
                "tt.dot operands have different element types: \(lhsElement) and \(rhsElement)",
                loc)
        }
        for element in [lhsElement, resultElement] {
            guard element == .float(width: 16) || element == .float(width: 32) else {
                throw CoreError.unsupportedOp(
                    "tt.dot on \(element) (Metal's simdgroup matrices are half or float)", loc)
            }
        }
        guard resultElement != .float(width: 16) || lhsElement == .float(width: 16) else {
            throw CoreError.lowering("tt.dot cannot accumulate f32 operands into an f16 result", loc)
        }

        return DotOp(
            result: result, lhs: lhs, rhs: rhs, accumulator: accumulator,
            lhsType: lhsType, rhsType: rhsType, resultType: resultType)
    }

    // MARK: - Control flow

    private mutating func parseForLoop(results: [String], loc: SourceLoc) throws -> ForLoop {
        let inductionVariable = try expectSSA()
        try expect(punct: "=")
        let lowerBound = try expectSSA()
        guard consume(ident: "to") else {
            throw CoreError.parse("expected 'to' in scf.for, found '\(peek().text)'", currentLoc)
        }
        let upperBound = try expectSSA()
        guard consume(ident: "step") else {
            throw CoreError.parse("expected 'step' in scf.for, found '\(peek().text)'", currentLoc)
        }
        let step = try expectSSA()

        var iterNames: [(String, String)] = []
        if consume(ident: "iter_args") {
            try expect(punct: "(")
            if !consume(punct: ")") {
                repeat {
                    let name = try expectSSA()
                    try expect(punct: "=")
                    let initial = try expectSSA()
                    iterNames.append((name, initial))
                } while consume(punct: ",")
                try expect(punct: ")")
            }
        }

        var resultTypes: [TMType] = []
        if peek().kind == .arrow {
            advance()
            resultTypes = try parseTypeListOrSingle()
        }
        var inductionType: TMType = .integer(width: 32)
        if consume(punct: ":") {
            inductionType = try parseType()
        }
        if peek().kind == .attrAlias || peek().isIdent("loc") {
            try skipTrailingLocation()
        }

        guard resultTypes.count == iterNames.count else {
            throw CoreError.parse(
                "scf.for declares \(resultTypes.count) result types for \(iterNames.count) "
                    + "iter_args", loc)
        }
        guard results.count == iterNames.count else {
            throw CoreError.parse(
                "scf.for binds \(results.count) results for \(iterNames.count) iter_args", loc)
        }

        let body = try parseRegionBody(what: "scf.for")
        return ForLoop(
            results: results,
            inductionVariable: inductionVariable,
            inductionType: inductionType,
            lowerBound: lowerBound,
            upperBound: upperBound,
            step: step,
            iterArguments: zip(iterNames, resultTypes).map {
                (name: $0.0.0, initial: $0.0.1, type: $0.1)
            },
            body: body)
    }

    private mutating func parseIfOp(results: [String], loc: SourceLoc) throws -> IfOp {
        let condition = try expectSSA()
        var resultTypes: [TMType] = []
        if peek().kind == .arrow {
            advance()
            resultTypes = try parseTypeListOrSingle()
        }
        if peek().kind == .attrAlias || peek().isIdent("loc") {
            try skipTrailingLocation()
        }
        guard results.count == resultTypes.count else {
            throw CoreError.parse(
                "scf.if binds \(results.count) results for \(resultTypes.count) result types", loc)
        }
        let thenBody = try parseRegionBody(what: "scf.if")
        var elseBody: [Instruction] = []
        if consume(ident: "else") {
            elseBody = try parseRegionBody(what: "scf.if else")
        }
        return IfOp(
            results: results, resultTypes: resultTypes, condition: condition,
            thenBody: thenBody, elseBody: elseBody)
    }

    /// `(T, U)` or a bare `T`.
    private mutating func parseTypeListOrSingle() throws -> [TMType] {
        var types: [TMType] = []
        if consume(punct: "(") {
            if !consume(punct: ")") {
                repeat {
                    types.append(try parseType())
                } while consume(punct: ",")
                try expect(punct: ")")
            }
        } else {
            types.append(try parseType())
        }
        return types
    }

    // MARK: - tt.reduce (generic form, with a combine region)

    /// `"tt.reduce"(%src) <{axis = N : i32}> ({ ^bb0(%a: T, %b: T): <combine> }) : (...) -> T`
    private mutating func parseGenericReduce(result: String, loc: SourceLoc) throws -> OpKind {
        try expect(punct: "(")
        let operands = try parseOperandList()
        try expect(punct: ")")
        guard operands.count == 1 else {
            throw CoreError.unsupportedOp(
                "tt.reduce with \(operands.count) operands (argmax/argmin-style reductions)", loc)
        }

        var attrs: [String: Attr] = [:]
        if consume(punct: "<") {
            attrs = try parseAttrDict()
            try expect(punct: ">")
        } else if peek().isPunct("{") {
            attrs = try parseAttrDict()
        }
        guard let axis = attrs["axis"]?.intValue, axis >= 0 else {
            throw CoreError.parse("tt.reduce requires a non-negative 'axis' attribute", loc)
        }

        try expect(punct: "(")
        let arguments = try parseBlockLabel()
        var region: [Instruction] = []
        while !peek().isPunct("}") {
            if peek().kind == .eof {
                throw CoreError.parse("unterminated tt.reduce combine region", currentLoc)
            }
            region.append(try parseInstruction())
        }
        try expect(punct: "}")
        try expect(punct: ")")
        _ = try parseOptionalTrailingTypes()

        let op = try TritonIRParser.combiner(region: region, arguments: arguments.map(\.name), loc: loc)
        return .reduce(result: result, source: operands[0], axis: Int(axis), op: op)
    }

    /// `{ ^bb0(%a: f32, %b: f32):` — the opening of a region with block arguments.
    private mutating func parseBlockLabel() throws -> [(name: String, type: TMType)] {
        try expect(punct: "{")
        guard consume(punct: "^") else { return [] }
        _ = try expectIdent()  // bb0
        var arguments: [(name: String, type: TMType)] = []
        if consume(punct: "(") {
            if !consume(punct: ")") {
                repeat {
                    let name = try expectSSA()
                    try expect(punct: ":")
                    arguments.append((name: name, type: try parseType()))
                    try skipTrailingLocation()
                } while consume(punct: ",")
                try expect(punct: ")")
            }
        }
        try skipTrailingLocation()
        try expect(punct: ":")
        return arguments
    }

    /// Recognises the combine region shape Triton emits: one binary op over the two
    /// block arguments, then `tt.reduce.return` of its result.
    static func combiner(region: [Instruction], arguments: [String], loc: SourceLoc) throws
        -> ReduceOp
    {
        let supported = "supported combiners are arith.addf/addi, arith.maxnumf/maximumf/maxsi "
            + "and arith.minnumf/minimumf/minsi"
        guard region.count == 2, case .yield(let yielded) = region[1].kind else {
            throw CoreError.unsupportedOp(
                "tt.reduce with a \(region.count)-operation combine region (\(supported))", loc)
        }
        let combine = region[0]
        var op: ReduceOp
        var result: String
        var operands: [String]
        switch combine.kind {
        case .intBinary(let r, let binary, let lhs, let rhs):
            switch binary {
            case .add: op = .add
            case .maxs: op = .max
            case .mins: op = .min
            default:
                throw CoreError.unsupportedOp("tt.reduce combining with \(combine.mnemonic)", loc)
            }
            (result, operands) = (r, [lhs, rhs])
        case .floatBinary(let r, let binary, let lhs, let rhs):
            switch binary {
            case .add: op = .add
            case .maximum: op = .max
            case .minimum: op = .min
            default:
                throw CoreError.unsupportedOp("tt.reduce combining with \(combine.mnemonic)", loc)
            }
            (result, operands) = (r, [lhs, rhs])
        default:
            throw CoreError.unsupportedOp(
                "tt.reduce combining with \(combine.mnemonic) (\(supported))", loc)
        }
        guard yielded == [result], Set(operands) == Set(arguments), arguments.count == 2 else {
            throw CoreError.lowering(
                "tt.reduce's combine region must reduce its two block arguments and return the "
                    + "result", loc)
        }
        return op
    }

    // MARK: - Attributes on simple ops

    /// `x` / `y` / `z`, or `{axis = N : i32}`.
    private mutating func parseAxis(loc: SourceLoc) throws -> Int {
        if peek().kind == .ident {
            let text = advance().text
            switch text {
            case "x": return 0
            case "y": return 1
            case "z": return 2
            default: throw CoreError.parse("expected axis x, y or z, found '\(text)'", loc)
            }
        }
        if peek().isPunct("{") {
            let attrs = try parseAttrDict()
            guard let axis = attrs["axis"]?.intValue, (0...2).contains(axis) else {
                throw CoreError.parse("expected an 'axis' attribute in 0...2", loc)
            }
            return Int(axis)
        }
        throw CoreError.parse("expected a program axis", loc)
    }

    private mutating func parseConstant(result: String, loc: SourceLoc) throws -> OpKind {
        var isDense = false
        if peek().isIdent("dense") {
            advance()
            try expect(punct: "<")
            isDense = true
        }

        var value: ConstantValue
        var bitPattern: UInt64? = nil
        switch peek().kind {
        case .integer:
            value = .integer(try expectInteger())
        case .hex:
            bitPattern = UInt64(advance().text, radix: 16)
            value = .integer(Int64(bitPattern: bitPattern ?? 0))
        case .float:
            guard let v = Double(peek().text) else {
                throw CoreError.parse("malformed float literal '\(peek().text)'", currentLoc)
            }
            advance()
            value = .float(v)
        case .ident where peek().text == "true" || peek().text == "false":
            value = .integer(advance().text == "true" ? 1 : 0)
        default:
            throw CoreError.parse(
                "unsupported constant value '\(peek().text)'; only scalar and splat "
                    + "integer/float constants are supported",
                currentLoc)
        }

        if isDense {
            try expect(punct: ">")
        }

        guard consume(punct: ":") else {
            // `arith.constant true` with no explicit type.
            return .constant(result: result, type: .integer(width: 1), value: value)
        }
        let type = try parseType()
        if case .float(let width) = type.scalarized {
            if let bits = bitPattern {
                // MLIR spells non-finite floats as raw bit patterns.
                value = .float(
                    width == 16
                        ? Double(Float16(bitPattern: UInt16(truncatingIfNeeded: bits)))
                        : Double(Float(bitPattern: UInt32(truncatingIfNeeded: bits))))
            } else if case .integer(let i) = value {
                value = .float(Double(i))
            }
        }
        return .constant(result: result, type: type, value: value)
    }

    // MARK: - Trailing type signatures

    /// Parses `: T` / `: T, U` / `: (T, U) -> V` / `: T -> V`, returning every type
    /// mentioned with the *result* type last.
    private mutating func parseOptionalTrailingTypes() throws -> [TMType] {
        guard consume(punct: ":") else { return [] }
        var types: [TMType] = []
        if consume(punct: "(") {
            if !consume(punct: ")") {
                repeat {
                    types.append(try parseType())
                } while consume(punct: ",")
                try expect(punct: ")")
            }
        } else {
            repeat {
                types.append(try parseType())
            } while peek().isPunct(",") && !peek(1).isPunct("}") && consume(punct: ",")
        }
        if peek().kind == .arrow {
            advance()
            if consume(punct: "(") {
                if !consume(punct: ")") {
                    repeat {
                        types.append(try parseType())
                    } while consume(punct: ",")
                    try expect(punct: ")")
                }
            } else {
                types.append(try parseType())
            }
        }
        return types
    }

    private mutating func skipTrailingLocation() throws {
        while peek().isIdent("loc") {
            advance()
            try expect(punct: "(")
            var depth = 1
            while depth > 0 {
                if peek().kind == .eof {
                    throw CoreError.parse("unterminated loc(...)", currentLoc)
                }
                if peek().isPunct("(") { depth += 1 }
                if peek().isPunct(")") { depth -= 1 }
                advance()
            }
        }
    }

    // MARK: - Types

    mutating func parseType() throws -> TMType {
        let token = peek()
        switch token.kind {
        case .dialectType:
            advance()
            guard token.text == "!tt.ptr" else {
                throw CoreError.unsupportedType(token.text, token.loc)
            }
            try expect(punct: "<")
            let pointee = try parseType()
            if consume(punct: ",") {
                _ = try? expectInteger()  // address space, ignored (always `device`)
            }
            try expect(punct: ">")
            return .pointer(pointee)

        case .ident where token.text == "tensor":
            advance()
            try expect(punct: "<")
            let type = try parseTensorBody(loc: token.loc)
            // Optional layout encoding (`, #blocked`) — parsed, then ignored.
            if consume(punct: ",") {
                _ = try parseAttrValue()
            }
            try expect(punct: ">")
            return type

        case .ident:
            advance()
            return try TritonIRParser.scalarType(named: token.text, loc: token.loc)

        default:
            throw CoreError.parse("expected a type, found '\(token.text)'", token.loc)
        }
    }

    /// Parses the inside of `tensor<...>`: `D0 x D1 x ... x element`.
    ///
    /// The lexer glues digits and letters together (`1024xi32` is one identifier,
    /// and `16x1x` precedes a `!tt.ptr` element), so the dimensions have to be
    /// recovered from the identifier's text.
    private mutating func parseTensorBody(loc: SourceLoc) throws -> TMType {
        var shape = [Int(try expectInteger())]

        guard peek().kind == .ident, peek().text.hasPrefix("x") else {
            throw CoreError.parse("malformed tensor shape", currentLoc)
        }
        let parts = advance().text.dropFirst().components(separatedBy: "x")
        for (index, part) in parts.enumerated() {
            if part.isEmpty {
                guard index == parts.count - 1 else {
                    throw CoreError.parse("malformed tensor shape", loc)
                }
                // The element type is a dialect type and follows as its own token.
                let element = try parseType()
                return .tensor(shape: shape, element: element)
            }
            if let dimension = Int(part) {
                shape.append(dimension)
                continue
            }
            guard index == parts.count - 1 else {
                throw CoreError.parse("malformed tensor shape (unexpected '\(part)')", loc)
            }
            return .tensor(shape: shape, element: try TritonIRParser.scalarType(named: part, loc: loc))
        }
        throw CoreError.parse("tensor type is missing its element type", loc)
    }

    static func scalarType(named name: String, loc: SourceLoc) throws -> TMType {
        if name.hasPrefix("i"), let width = Int(name.dropFirst()) {
            guard [1, 8, 16, 32, 64].contains(width) else {
                throw CoreError.unsupportedType(name, loc)
            }
            return .integer(width: width)
        }
        if name.hasPrefix("f"), let width = Int(name.dropFirst()) {
            guard [16, 32].contains(width) else {
                throw CoreError.unsupportedType(
                    "\(name) (Metal supports half and float; f64 has no GPU equivalent)", loc)
            }
            return .float(width: width)
        }
        throw CoreError.unsupportedType(name, loc)
    }

    // MARK: - Attributes

    private mutating func skipAttrDictIfPresent() {
        if peek().isPunct("{") {
            _ = try? parseAttrDict()
        }
    }

    mutating func parseAttrDict() throws -> [String: Attr] {
        try expect(punct: "{")
        var attrs: [String: Attr] = [:]
        if consume(punct: "}") { return attrs }
        repeat {
            let keyToken = peek()
            guard keyToken.kind == .ident || keyToken.kind == .string else {
                throw CoreError.parse("expected an attribute name, found '\(keyToken.text)'", keyToken.loc)
            }
            advance()
            guard consume(punct: "=") else {
                // Unit attributes (`{noinline}`).
                attrs[keyToken.text] = .boolean(true)
                continue
            }
            attrs[keyToken.text] = try parseAttrValue()
        } while consume(punct: ",")
        try expect(punct: "}")
        return attrs
    }

    private mutating func parseAttrValue() throws -> Attr {
        let token = peek()
        switch token.kind {
        case .punct where token.text == "{":
            return .dict(try parseAttrDict())

        case .punct where token.text == "[":
            advance()
            var items: [Attr] = []
            if !consume(punct: "]") {
                repeat {
                    items.append(try parseAttrValue())
                } while consume(punct: ",")
                try expect(punct: "]")
            }
            return .list(items)

        case .string:
            advance()
            return .string(token.text)

        case .integer:
            let value = try expectInteger()
            try skipAttributeTypeSuffix()
            return .integer(value)

        case .hex:
            advance()
            try skipAttributeTypeSuffix()
            return .integer(Int64(bitPattern: UInt64(token.text, radix: 16) ?? 0))

        case .float:
            advance()
            try skipAttributeTypeSuffix()
            return .float(Double(token.text) ?? 0)

        case .ident where token.text == "true" || token.text == "false":
            advance()
            return .boolean(token.text == "true")

        case .ident, .attrAlias:
            advance()
            if peek().isPunct("<") {
                try skipBalanced(open: "<", close: ">")
            }
            if peek().isPunct("(") {
                try skipBalanced(open: "(", close: ")")
            }
            try skipAttributeTypeSuffix()
            return .ident(token.text)

        case .dialectType:
            return .type(try parseType())

        default:
            throw CoreError.parse("unsupported attribute value '\(token.text)'", token.loc)
        }
    }

    /// `1024 : i32` — the `: type` suffix carries no information we need.
    private mutating func skipAttributeTypeSuffix() throws {
        guard peek().isPunct(":") else { return }
        advance()
        _ = try parseType()
    }

    private mutating func skipBalanced(open: String, close: String) throws {
        try expect(punct: open)
        var depth = 1
        while depth > 0 {
            if peek().kind == .eof {
                throw CoreError.parse("unterminated '\(open)'", currentLoc)
            }
            if peek().isPunct(open) { depth += 1 }
            if peek().isPunct(close) { depth -= 1 }
            advance()
        }
    }
}
