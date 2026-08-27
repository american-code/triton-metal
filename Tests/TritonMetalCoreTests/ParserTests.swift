import XCTest

@testable import TritonMetalCore

final class ParserTests: XCTestCase {

    // MARK: - Structure

    func testParsesVectorAddSignatureAndBody() throws {
        let module = try TritonIRParser.parse(IRFixtures.vectorAdd)
        XCTAssertEqual(module.functions.count, 1)

        let function = module.functions[0]
        XCTAssertEqual(function.name, "add_kernel")
        XCTAssertEqual(function.arguments.map(\.type), [
            .pointer(.float(width: 32)),
            .pointer(.float(width: 32)),
            .pointer(.float(width: 32)),
            .integer(width: 32),
        ])
        XCTAssertEqual(function.arguments.map(\.index), [0, 1, 2, 3])
        XCTAssertEqual(function.body.count, 19)
        XCTAssertEqual(function.body.last?.mnemonic, "tt.return")
    }

    func testParsesOperandsOfEachOpShape() throws {
        let function = try TritonIRParser.parse(IRFixtures.vectorMul).functions[0]

        guard case .constant(_, let constType, let constValue) = function.body[0].kind else {
            return XCTFail("expected a dense constant first")
        }
        XCTAssertEqual(constType, .tensor(shape: [128], element: .float(width: 32)))
        XCTAssertEqual(constValue, .float(1.0))

        guard case .makeRange(_, let rangeType, let start, let end) = function.body[4].kind else {
            return XCTFail("expected tt.make_range")
        }
        XCTAssertEqual(rangeType, .tensor(shape: [128], element: .integer(width: 32)))
        XCTAssertEqual(start, 0)
        XCTAssertEqual(end, 128)

        // tt.load with mask + other and the legacy cache/evict attribute dict.
        let load = function.body.first { $0.mnemonic == "tt.load" }
        guard case .load(_, let pointer, let mask, let other) = load?.kind else {
            return XCTFail("expected tt.load")
        }
        XCTAssertEqual(pointer, "8")
        XCTAssertEqual(mask, "6")
        XCTAssertEqual(other, "cst")

        guard case .store(_, _, let storeMask) = function.body.last(where: { $0.mnemonic == "tt.store" })?.kind
        else {
            return XCTFail("expected tt.store")
        }
        XCTAssertEqual(storeMask, "6")
    }

    func testParsesBothProgramIDSpellings() throws {
        let attributeForm = try TritonIRParser.parse(IRFixtures.scaleBias).functions[0]
        guard case .programID(_, let axis) = attributeForm.body[1].kind else {
            return XCTFail("expected tt.get_program_id from the {axis = 0} spelling")
        }
        XCTAssertEqual(axis, 0)
        guard case .numPrograms(_, let gridAxis) = attributeForm.body[2].kind else {
            return XCTFail("expected tt.get_num_programs")
        }
        XCTAssertEqual(gridAxis, 0)

        let keywordForm = try TritonIRParser.parse(IRFixtures.vectorAdd).functions[0]
        guard case .programID(_, let keywordAxis) = keywordForm.body[1].kind else {
            return XCTFail("expected tt.get_program_id from the keyword spelling")
        }
        XCTAssertEqual(keywordAxis, 0)
    }

    // MARK: - Types

    func testParsesTypes() throws {
        func type(_ text: String) throws -> TMType {
            var parser = try TritonIRParser("\(text)")
            return try parser.parseType()
        }
        XCTAssertEqual(try type("i32"), .integer(width: 32))
        XCTAssertEqual(try type("i1"), .integer(width: 1))
        XCTAssertEqual(try type("f16"), .float(width: 16))
        XCTAssertEqual(try type("!tt.ptr<f32>"), .pointer(.float(width: 32)))
        XCTAssertEqual(try type("!tt.ptr<f32, 1>"), .pointer(.float(width: 32)))
        XCTAssertEqual(
            try type("tensor<1024xi32>"), .tensor(shape: [1024], element: .integer(width: 32)))
        XCTAssertEqual(
            try type("tensor<64x!tt.ptr<f16>>"),
            .tensor(shape: [64], element: .pointer(.float(width: 16))))
        // ttgir layout encodings parse and are ignored for the 1-D subset.
        XCTAssertEqual(
            try type("tensor<32xf32, #blocked>"),
            .tensor(shape: [32], element: .float(width: 32)))
    }

    func testCommentsAndLocationsAreIgnored() throws {
        let ir = """
            #loc = loc(unknown)
            module {
              // a comment
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32 loc(#loc)
                tt.return loc(#loc)
              } loc(#loc)
            }
            """
        let function = try TritonIRParser.parse(ir).functions[0]
        XCTAssertEqual(function.name, "k")
        XCTAssertEqual(function.body.count, 2)
    }

    // MARK: - Error paths

    private func expectError(_ ir: String, contains needle: String, _ line: UInt = #line) {
        do {
            _ = try MetalCompiler.emitMSL(ttir: ir, options: .init())
            XCTFail("expected an error containing '\(needle)'", line: line)
        } catch {
            let description = (error as? CoreError)?.description ?? "\(error)"
            XCTAssertTrue(
                description.contains(needle),
                "error '\(description)' does not mention '\(needle)'", line: line)
        }
    }

    func testUnsupportedOpIsNamedPrecisely() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.cat %0, %0 : tensor<16xi32> -> tensor<32xi32>
                tt.return
              }
            }
            """, contains: "unsupported op 'tt.cat'")
    }

    /// `tt.trans` records a permutation and produces no code; both the explicit
    /// `order` attribute and the bare 2-D form Triton prints are accepted.
    func testTransposeParsesBothSpellings() throws {
        for spelling in [
            "%1 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>",
            "%1 = tt.trans %0 : tensor<8x16xf32> -> tensor<16x8xf32>",
        ] {
            let ir = """
                module {
                  tt.func public @k(%arg0: !tt.ptr<f32>) {
                    %0 = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
                \(spelling)
                    tt.return
                  }
                }
                """
            let function = try TritonIRParser.parse(ir).functions[0]
            guard case .trans(let result, let type, let source, let order) = function.body[1].kind
            else { return XCTFail("expected tt.trans for '\(spelling)'") }
            XCTAssertEqual(result, "1")
            XCTAssertEqual(source, "0")
            XCTAssertEqual(order, [1, 0])
            XCTAssertEqual(type, .tensor(shape: [16, 8], element: .float(width: 32)))
        }
    }

    /// A transposed value that has to be *materialised* asks for two contradictory
    /// nestings of the same two block dimensions, and is refused by name.
    func testMaterialisedTransposeIsRejected() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
                %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
                %4 = tt.broadcast %2 : tensor<8x1xi32> -> tensor<8x16xi32>
                %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<8x16xi32>
                %6 = arith.addi %4, %5 : tensor<8x16xi32>
                %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
                %8 = tt.addptr %7, %6 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
                %9 = tt.load %8 : tensor<8x16xf32>
                %10 = tt.trans %9 : tensor<8x16xf32> -> tensor<16x8xf32>
                %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
                %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
                %14 = tt.broadcast %13 : tensor<16x1xi32> -> tensor<16x8xi32>
                %15 = tt.addptr %11, %14 : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
                tt.store %15, %10 : tensor<16x8x!tt.ptr<f32>>
                tt.return
              }
            }
            """, contains: "nests its block dimensions in two incompatible orders")
    }

    func testUnsupportedAtomicsAreNamed() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: f32) {
                %0 = tt.atomic_add %arg0, %arg1 : f32
                tt.return
              }
            }
            """, contains: "unsupported op 'tt.atomic_add'")
    }

    func testMakeRangeMustStayOneDimensional() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<16x16xi32>
                tt.return
              }
            }
            """, contains: "must produce a 1-D tensor")
    }

    func testEmptyModuleReportsNoKernel() {
        expectError("module {}", contains: "no tt.func")
    }

    func testGenericOpSyntaxIsRejected() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = "tt.get_program_id"() : () -> i32
                tt.return
              }
            }
            """, contains: "generic MLIR op syntax")
    }

    func testUnknownTypeIsRejected() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f64>) {
                tt.return
              }
            }
            """, contains: "unsupported type")
    }

    func testMismatchedBlockSizesAreRejected() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
                tt.return
              }
            }
            """, contains: "mixes tensor lengths")
    }

    func testMakeRangeLengthMustMatchItsBounds() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<16xi32>
                tt.return
              }
            }
            """, contains: "does not match its result length")
    }

    func testUndefinedValueIsReported() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>) {
                %0 = arith.addi %missing, %missing : i32
                tt.return
              }
            }
            """, contains: "undefined value")
    }

    func testTypeMismatchIsReported() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: i32) {
                %0 = arith.constant 1.000000e+00 : f32
                %1 = arith.addf %0, %arg1 : f32
                tt.return
              }
            }
            """, contains: "operand types differ")
    }

    func testMaskShapeIsChecked() {
        expectError(
            """
            module {
              tt.func public @k(%arg0: !tt.ptr<f32>, %arg1: i32) {
                %0 = arith.constant 1 : i32
                %1 = arith.cmpi slt, %0, %arg1 : i32
                %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
                %4 = tt.addptr %3, %2 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
                %5 = tt.load %4, %1 : tensor<8x!tt.ptr<f32>>
                tt.return
              }
            }
            """, contains: "does not match pointer shape")
    }

    func testErrorsCarryLineNumbers() {
        do {
            _ = try MetalCompiler.emitMSL(
                ttir: """
                    module {
                      tt.func public @k(%arg0: !tt.ptr<f32>) {
                        %0 = tt.cat %arg0, %arg0 : !tt.ptr<f32>
                        tt.return
                      }
                    }
                    """, options: .init())
            XCTFail("expected an error")
        } catch let error as CoreError {
            XCTAssertTrue(error.description.contains("line 3"), error.description)
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }
}
