import Foundation
import TritonMetalBench
import TritonMetalCore

/// `tmbench` — the GEMM sweep against MPS, as an executable.
///
/// `swift test` needs XCTest, which only exists where Xcode is installed. The
/// machines whose throughput numbers are worth quoting do not all have it, so the
/// measurement is packaged as a plain executable with hand-rolled argument
/// parsing and no dependencies.
///
///     tmbench [--sizes 512,1024,2048] [--sweep quick|full]
///             [--config M,N,K,W[,RM,RN[,U[,P[,DB[,V4]]]]]] [--verbose] [--no-verify]
///             [--emit M,N,K,W[,...]]
///     tmbench --attn [--attn-shapes b,h,s,d;...] [--attn-config M,N,W]
///             [--attn-element f32|f16] [--emit-attn M,N,W,D]
///
/// `--attn` measures the FlashAttention-2 forward kernel against the unfused
/// composite (two MPS GEMMs with an MPS softmax between them), which is the
/// comparison a fused kernel exists to win.

struct Arguments {
    var sizes = [512, 1024, 2048]
    var sweep = GEMMBenchmark.Sweep.quick
    var pinned: [GEMMConfig] = []
    var verbose = false
    var verify = true
    var emit: GEMMConfig?
    var attention = false
    var attentionShapes = [
        AttentionShape(batch: 1, heads: 8, seq: 512, dim: 64),
        AttentionShape(batch: 1, heads: 8, seq: 1024, dim: 64),
        AttentionShape(batch: 1, heads: 16, seq: 2048, dim: 64),
    ]
    var attentionPinned: [AttentionConfig] = []
    var attentionElement = "f32"
    var emitAttention: (config: AttentionConfig, dim: Int)?
}

/// `b,h,s,d;b,h,s,d` — the spelling `--attn-shapes` takes.
func parseShapes(_ text: String) -> [AttentionShape]? {
    var shapes: [AttentionShape] = []
    for group in text.split(separator: ";") {
        let fields = group.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard fields.count == 4, !fields.contains(where: { $0 == nil }) else { return nil }
        shapes.append(
            AttentionShape(batch: fields[0]!, heads: fields[1]!, seq: fields[2]!, dim: fields[3]!))
    }
    return shapes.isEmpty ? nil : shapes
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("tmbench: " + message + "\n").utf8))
    exit(1)
}

func parseArguments() -> Arguments {
    var arguments = Arguments()
    var rest = Array(CommandLine.arguments.dropFirst())
    while let flag = rest.first {
        rest.removeFirst()
        func value(_ name: String) -> String {
            guard let next = rest.first else { fail("\(name) needs a value") }
            rest.removeFirst()
            return next
        }
        switch flag {
        case "--sizes":
            arguments.sizes = value(flag).split(separator: ",").compactMap { Int($0) }
            if arguments.sizes.isEmpty { fail("--sizes needs a comma-separated list of integers") }
        case "--sweep":
            guard let sweep = GEMMBenchmark.Sweep(rawValue: value(flag)) else {
                fail("--sweep takes 'quick' or 'full'")
            }
            arguments.sweep = sweep
        case "--config":
            guard let config = GEMMConfig.parse(value(flag)) else {
                fail("--config takes M,N,K,W[,RM,RN[,U[,P[,DB[,V4]]]]]")
            }
            arguments.pinned.append(config)
        case "--emit":
            guard let config = GEMMConfig.parse(value(flag)) else {
                fail("--emit takes M,N,K,W[,RM,RN[,U[,P[,DB[,V4]]]]]")
            }
            arguments.emit = config
        case "--attn": arguments.attention = true
        case "--attn-shapes":
            guard let shapes = parseShapes(value(flag)) else {
                fail("--attn-shapes takes b,h,s,d[;b,h,s,d...]")
            }
            arguments.attentionShapes = shapes
            arguments.attention = true
        case "--attn-config":
            guard let config = AttentionConfig.parse(value(flag)) else {
                fail("--attn-config takes BLOCK_M,BLOCK_N,W")
            }
            arguments.attentionPinned.append(config)
            arguments.attention = true
        case "--attn-element":
            let element = value(flag)
            guard element == "f32" || element == "f16" else {
                fail("--attn-element takes f32 or f16")
            }
            arguments.attentionElement = element
            arguments.attention = true
        case "--emit-attn":
            let text = value(flag)
            let fields = text.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard fields.count == 4, !fields.contains(where: { $0 == nil }),
                let config = AttentionConfig.parse(
                    "\(fields[0]!),\(fields[1]!),\(fields[2]!)")
            else { fail("--emit-attn takes BLOCK_M,BLOCK_N,W,HEAD_DIM") }
            arguments.emitAttention = (config: config, dim: fields[3]!)
        case "--verbose", "-v": arguments.verbose = true
        case "--no-verify": arguments.verify = false
        case "--help", "-h":
            print(
                """
                tmbench — square-GEMM throughput of the triton-metal tt.dot lowering
                          against MPSMatrixMultiplication.

                  --sizes N,N,...      matrix sizes to measure (default 512,1024,2048)
                  --sweep quick|full   block shapes only, or crossed with every explicit
                                       register blocking (default quick)
                  --config M,N,K,W[,RM,RN[,U[,P[,DB[,V4]]]]]
                                       measure exactly this configuration; repeatable.
                                       Overrides --sweep. RM/RN register blocking,
                                       U staging run length, P tile row padding,
                                       DB double buffering, V4 vector staging (on
                                       unless given as 0).
                  --emit M,N,K,W[,RM,RN[,U[,P[,DB[,V4]]]]]
                                       print the emitted MSL for one configuration and exit
                  --verbose            print every configuration's throughput
                  --no-verify          skip the CPU-reference correctness check

                FlashAttention-2 forward, against the unfused MPS composite
                (Q K^T, softmax, P V — three dispatches through a real S x S
                score matrix):

                  --attn               measure attention instead of GEMM
                  --attn-shapes b,h,s,d[;...]
                                       shapes to measure (default
                                       1,8,512,64;1,8,1024,64;1,16,2048,64)
                  --attn-config M,N,W  measure exactly this block shape and
                                       threadgroup width; repeatable
                  --attn-element f32|f16
                                       kernel input/output type (the accumulator
                                       and the softmax are always f32)
                  --emit-attn M,N,W,D  print the emitted MSL and exit
                """)
            exit(0)
        default:
            fail("unknown argument '\(flag)' (try --help)")
        }
    }
    return arguments
}

let arguments = parseArguments()

if let config = arguments.emit {
    do {
        print(
            try MetalCompiler.emitMSL(
                ttir: GEMMKernel.tutorial(
                    blockM: config.blockM, blockN: config.blockN, blockK: config.blockK),
                options: config.options))
        exit(0)
    } catch {
        fail("\(error)")
    }
}

if let request = arguments.emitAttention {
    do {
        print(
            try MetalCompiler.emitMSL(
                ttir: AttentionKernel.forward(
                    blockM: request.config.blockM, blockN: request.config.blockN,
                    headDim: request.dim, element: arguments.attentionElement),
                options: request.config.options))
        exit(0)
    } catch {
        fail("\(error)")
    }
}

if arguments.attention {
    do {
        let harness = try GEMMBenchmark.Harness()
        print("device: \(harness.device.name)")
        let configurations =
            arguments.attentionPinned.isEmpty
            ? AttentionBenchmark.configurations() : arguments.attentionPinned
        print(
            "attention sweep: \(configurations.count) configuration"
                + "\(configurations.count == 1 ? "" : "s"), shapes "
                + arguments.attentionShapes.map(\.name).joined(separator: " / "))

        let results = try AttentionBenchmark.run(
            shapes: arguments.attentionShapes, configurations: configurations,
            element: arguments.attentionElement, harness: harness, verbose: arguments.verbose)

        if arguments.verify {
            var checked: Set<AttentionConfig> = []
            for result in results where !checked.contains(result.config) {
                checked.insert(result.config)
                if let complaint = try AttentionBenchmark.verify(result.config, harness: harness) {
                    fail(complaint)
                }
            }
            print(
                "verified \(checked.count) winning configuration"
                    + "\(checked.count == 1 ? "" : "s") against a CPU reference")
        }

        print("")
        print(AttentionBenchmark.header)
        for result in results { print(AttentionBenchmark.row(result)) }
        exit(0)
    } catch {
        fail("\(error)")
    }
}

do {
    let harness = try GEMMBenchmark.Harness()
    print("device: \(harness.device.name)")

    let configurations =
        arguments.pinned.isEmpty
        ? GEMMBenchmark.configurations(arguments.sweep) : arguments.pinned
    print(
        "sweep: \(configurations.count) configuration"
            + "\(configurations.count == 1 ? "" : "s"), sizes "
            + arguments.sizes.map(String.init).joined(separator: "/"))

    let results = try GEMMBenchmark.run(
        sizes: arguments.sizes, configurations: configurations, harness: harness,
        verbose: arguments.verbose)

    if arguments.verify {
        // Only the winners need checking, and they are the only numbers reported:
        // a fast-but-wrong kernel must not be quotable on a machine without
        // XCTest. `swift test` is where the real correctness coverage lives.
        var checked: Set<GEMMConfig> = []
        for result in results where !checked.contains(result.config) {
            checked.insert(result.config)
            if let complaint = try GEMMBenchmark.verify(result.config, harness: harness) {
                fail(complaint)
            }
        }
        print("verified \(checked.count) winning configuration"
            + "\(checked.count == 1 ? "" : "s") against a CPU reference")
    }

    print("")
    print(GEMMBenchmark.header)
    for result in results { print(GEMMBenchmark.row(result)) }
} catch {
    fail("\(error)")
}
