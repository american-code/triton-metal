import Foundation

/// A position in the input IR text, used for diagnostics.
public struct SourceLoc: Equatable, Sendable, CustomStringConvertible {
    public let line: Int
    public let column: Int

    public init(line: Int, column: Int) {
        self.line = line
        self.column = column
    }

    public var description: String { "line \(line), col \(column)" }

    static let unknown = SourceLoc(line: 0, column: 0)
}

enum TokenKind: Equatable {
    /// `%0`, `%arg0`, `%c1024_i32`, `%5#1` (one result of a multi-result op)
    case ssa
    /// `@add_kernel`
    case symbol
    /// `tt.load`, `arith.addf`, `slt`, `x`, `dense`
    case ident
    /// `!tt.ptr` — a dialect type prefix; the `<...>` payload follows as tokens.
    case dialectType
    /// `#loc1`, `#blocked` — attribute alias.
    case attrAlias
    case integer
    /// `0xFF800000` — MLIR's bit-pattern spelling of a float literal.
    case hex
    case float
    case string
    /// One of `(){}<>[]:,=*?^`
    case punct
    /// `->`
    case arrow
    case eof
}

struct Token {
    let kind: TokenKind
    let text: String
    let loc: SourceLoc

    func isPunct(_ c: String) -> Bool { kind == .punct && text == c }
    func isIdent(_ s: String) -> Bool { kind == .ident && text == s }
}

/// Tokenizer for the subset of MLIR generic/pretty syntax that Triton emits.
///
/// This is deliberately small: it recognizes SSA names, symbol names, dialect
/// type prefixes, attribute aliases, numeric/string literals and punctuation.
/// Anything structural (types, ops, regions) is the parser's job.
struct Lexer {
    private let src: [Character]
    private var pos = 0
    private var line = 1
    private var column = 1

    init(_ text: String) {
        src = Array(text)
    }

    private var current: Character? { pos < src.count ? src[pos] : nil }

    private func peek(_ offset: Int) -> Character? {
        let i = pos + offset
        return i < src.count ? src[i] : nil
    }

    private var loc: SourceLoc { SourceLoc(line: line, column: column) }

    @discardableResult
    private mutating func advance() -> Character {
        let c = src[pos]
        pos += 1
        if c == "\n" {
            line += 1
            column = 1
        } else {
            column += 1
        }
        return c
    }

    private static func isIdentStart(_ c: Character) -> Bool {
        c.isLetter || c == "_"
    }

    private static func isIdentBody(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_" || c == "." || c == "$"
    }

    private static func isSSABody(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_" || c == "."
    }

    private mutating func skipTrivia() {
        while let c = current {
            if c.isWhitespace {
                advance()
            } else if c == "/", peek(1) == "/" {
                while let c = current, c != "\n" { advance() }
            } else {
                break
            }
        }
    }

    private mutating func take(while predicate: (Character) -> Bool) -> String {
        var out = ""
        while let c = current, predicate(c) {
            out.append(c)
            advance()
        }
        return out
    }

    mutating func tokenize() throws -> [Token] {
        var tokens: [Token] = []
        while true {
            skipTrivia()
            let start = loc
            guard let c = current else {
                tokens.append(Token(kind: .eof, text: "", loc: start))
                return tokens
            }

            switch c {
            case "%":
                advance()
                var name = take(while: Lexer.isSSABody)
                guard !name.isEmpty else {
                    throw CoreError.parse("expected a name after '%'", start)
                }
                // `%5#1` selects one result of a multi-result op; keep it in the name
                // so the symbol table can key on it directly.
                if current == "#", let next = peek(1), next.isNumber {
                    advance()
                    name += "#" + take(while: { $0.isNumber })
                }
                tokens.append(Token(kind: .ssa, text: name, loc: start))

            case "@":
                advance()
                let name = take(while: Lexer.isIdentBody)
                guard !name.isEmpty else {
                    throw CoreError.parse("expected a name after '@'", start)
                }
                tokens.append(Token(kind: .symbol, text: name, loc: start))

            case "!":
                advance()
                let name = take(while: Lexer.isIdentBody)
                guard !name.isEmpty else {
                    throw CoreError.parse("expected a type name after '!'", start)
                }
                tokens.append(Token(kind: .dialectType, text: "!" + name, loc: start))

            case "#":
                advance()
                let name = take(while: Lexer.isIdentBody)
                tokens.append(Token(kind: .attrAlias, text: "#" + name, loc: start))

            case "\"":
                advance()
                var text = ""
                while let c = current, c != "\"" {
                    if c == "\\", let next = peek(1) {
                        advance()
                        advance()
                        switch next {
                        case "n": text.append("\n")
                        case "t": text.append("\t")
                        default: text.append(next)
                        }
                    } else {
                        text.append(advance())
                    }
                }
                guard current == "\"" else {
                    throw CoreError.parse("unterminated string literal", start)
                }
                advance()
                tokens.append(Token(kind: .string, text: text, loc: start))

            case "-":
                if let next = peek(1), next == ">" {
                    advance()
                    advance()
                    tokens.append(Token(kind: .arrow, text: "->", loc: start))
                } else if let next = peek(1), next.isNumber {
                    advance()
                    let (text, isFloat) = lexNumberBody()
                    tokens.append(Token(kind: isFloat ? .float : .integer, text: "-" + text, loc: start))
                } else {
                    throw CoreError.parse("unexpected '-'", start)
                }

            case _ where c.isNumber:
                if let hex = lexHexBitPattern() {
                    tokens.append(Token(kind: .hex, text: hex, loc: start))
                    break
                }
                let (text, isFloat) = lexNumberBody()
                tokens.append(Token(kind: isFloat ? .float : .integer, text: text, loc: start))

            case _ where Lexer.isIdentStart(c):
                let text = take(while: Lexer.isIdentBody)
                tokens.append(Token(kind: .ident, text: text, loc: start))

            case "(", ")", "{", "}", "<", ">", "[", "]", ":", ",", "=", "*", "?", "^":
                advance()
                tokens.append(Token(kind: .punct, text: String(c), loc: start))

            default:
                throw CoreError.parse("unexpected character '\(c)'", start)
            }
        }
    }

    /// MLIR prints non-finite and round-trip-exact float attributes as raw bit
    /// patterns (`dense<0xFF800000>` is -inf). Recognized only for exactly 8 or 16
    /// hex digits, so that tensor shapes like `tensor<0xf32>` keep lexing as a
    /// dimension followed by an element type.
    private mutating func lexHexBitPattern() -> String? {
        guard current == "0", let x = peek(1), x == "x" || x == "X" else { return nil }
        var digits = ""
        var offset = 2
        while let c = peek(offset), c.isHexDigit {
            digits.append(c)
            offset += 1
        }
        guard digits.count == 8 || digits.count == 16 else { return nil }
        // Reject `0xFFFFFFFFthing` — the run must end at a non-identifier character.
        if let next = peek(offset), Lexer.isIdentBody(next) { return nil }
        for _ in 0..<offset { advance() }
        return digits
    }

    /// Lexes digits plus an optional fraction/exponent. Returns the raw text and
    /// whether it looked like a float. Stops before a trailing `x` so that
    /// tensor shapes (`1024xi32`) split into a dimension and an element type.
    private mutating func lexNumberBody() -> (String, Bool) {
        var text = take(while: { $0.isNumber })
        var isFloat = false
        if current == ".", let next = peek(1), next.isNumber {
            text.append(advance())
            text += take(while: { $0.isNumber })
            isFloat = true
        }
        if let c = current, c == "e" || c == "E" {
            let save = (pos, line, column)
            var exponent = String(advance())
            if let sign = current, sign == "+" || sign == "-" {
                exponent.append(advance())
            }
            let digits = take(while: { $0.isNumber })
            if digits.isEmpty {
                (pos, line, column) = save
            } else {
                text += exponent + digits
                isFloat = true
            }
        }
        return (text, isFloat)
    }
}
