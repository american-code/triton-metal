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

struct Arguments {
    var sizes = [512, 1024, 2048]
    var sweep = GEMMBenchmark.Sweep.quick
    var pinned: [GEMMConfig] = []
    var verbose = false
    var verify = true
    var emit: GEMMConfig?
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
