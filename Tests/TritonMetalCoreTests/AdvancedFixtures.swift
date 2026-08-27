import Foundation

/// Triton IR fixtures for the lowerings beyond 1-D elementwise: casts and
/// `math.*`, `scf` control flow, rank-2 tensors with broadcasting, and
/// `tt.reduce`. Written the way Triton's own `ttir` dump prints them — including
/// the generic spelling of `tt.reduce`, which carries a combine region.
enum AdvancedFixtures {

    // MARK: - Shared skeleton

    /// `out[i] = <op>(in[i])` over a BLOCK of 64, masked to `n`.
    ///
    /// `body` receives the loaded value (`%9`) and must define `%10`.
    private static func unaryKernel(
        name: String, inType: String, outType: String, body: String
    ) -> String {
        """
        module {
          tt.func public @\(name)(%arg0: !tt.ptr<\(inType)>, %arg1: !tt.ptr<\(outType)>,
                                  %arg2: i32) {
            %c64_i32 = arith.constant 64 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c64_i32 : i32
            %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %3 = tt.splat %1 : i32 -> tensor<64xi32>
            %4 = arith.addi %3, %2 : tensor<64xi32>
            %5 = tt.splat %arg2 : i32 -> tensor<64xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
            %7 = tt.splat %arg0 : !tt.ptr<\(inType)> -> tensor<64x!tt.ptr<\(inType)>>
            %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<\(inType)>>, tensor<64xi32>
            %9 = tt.load %8, %6 : tensor<64x!tt.ptr<\(inType)>>
        \(body)
            %11 = tt.splat %arg1 : !tt.ptr<\(outType)> -> tensor<64x!tt.ptr<\(outType)>>
            %12 = tt.addptr %11, %4 : tensor<64x!tt.ptr<\(outType)>>, tensor<64xi32>
            tt.store %12, %10, %6 : tensor<64x!tt.ptr<\(outType)>>
            tt.return
          }
        }
        """
    }

    /// `out[i] = <math op>(in[i])`, same element type in and out.
    static func mathKernel(_ mnemonic: String, type: String = "f32") -> String {
        unaryKernel(
            name: "math_kernel", inType: type, outType: type,
            body: "    %10 = \(mnemonic) %9 : tensor<64x\(type)>")
    }

    /// Every `math.*` mnemonic the emitter claims to lower (f32 forms).
    static let everyMathMnemonic = [
        "math.exp", "math.exp2", "math.log", "math.log2", "math.sqrt", "math.rsqrt",
        "math.sin", "math.cos", "math.tanh", "math.erf", "math.absf", "math.floor", "math.ceil",
    ]

    /// Every `arith` conversion the emitter claims to lower, with a valid pair of
    /// types for each.
    static let everyCast: [(mnemonic: String, from: String, to: String)] = [
        ("arith.sitofp", "i32", "f32"), ("arith.uitofp", "i32", "f32"),
        ("arith.fptosi", "f32", "i32"), ("arith.fptoui", "f32", "i32"),
        ("arith.extsi", "i16", "i32"), ("arith.extui", "i16", "i32"),
        ("arith.trunci", "i32", "i16"), ("arith.truncf", "f32", "f16"),
        ("arith.extf", "f16", "f32"), ("arith.bitcast", "i32", "f32"),
    ]

    /// `out[i] = <cast>(in[i])` between two scalar types.
    static func castKernel(_ mnemonic: String, from: String, to: String) -> String {
        unaryKernel(
            name: "cast_kernel", inType: from, outType: to,
            body: "    %10 = \(mnemonic) %9 : tensor<64x\(from)> to tensor<64x\(to)>")
    }

    /// `out[i] = in[i] > 0 ? in[i] * 2 : in[i] - 1` via `arith.select`.
    static let select = unaryKernel(
        name: "select_kernel", inType: "f32", outType: "f32",
        body: """
                %cst0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
                %cst2 = arith.constant dense<2.000000e+00> : tensor<64xf32>
                %cst1 = arith.constant dense<1.000000e+00> : tensor<64xf32>
                %a = arith.cmpf ogt, %9, %cst0 : tensor<64xf32>
                %b = arith.mulf %9, %cst2 : tensor<64xf32>
                %c = arith.subf %9, %cst1 : tensor<64xf32>
                %10 = arith.select %a, %b, %c : tensor<64xi1>, tensor<64xf32>
            """)

    // MARK: - scf control flow

    /// Per-program partial sums: program `p` walks the array in strides of
    /// `num_programs * BLOCK`, accumulating a BLOCK-wide vector across an
    /// `scf.for` with a tensor `iter_args`, then stores its partial.
    static let stridedSum = """
        module {
          tt.func public @strided_sum_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                             %arg2: i32, %arg3: i32) {
            %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %c64_i32 = arith.constant 64 : i32
            %0 = tt.get_program_id x : i32
            %1 = tt.get_num_programs x : i32
            %2 = arith.muli %1, %c64_i32 : i32
            %3 = arith.muli %0, %c64_i32 : i32
            %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %5 = tt.splat %3 : i32 -> tensor<64xi32>
            %6 = arith.addi %5, %4 : tensor<64xi32>
            %7 = tt.splat %arg2 : i32 -> tensor<64xi32>
            %8 = scf.for %arg5 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg6 = %cst)
                -> (tensor<64xf32>) : i32 {
              %20 = arith.muli %arg5, %2 : i32
              %21 = tt.splat %20 : i32 -> tensor<64xi32>
              %22 = arith.addi %6, %21 : tensor<64xi32>
              %23 = arith.cmpi slt, %22, %7 : tensor<64xi32>
              %24 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
              %25 = tt.addptr %24, %22 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
              %26 = tt.load %25, %23, %cst : tensor<64xf32>
              %27 = arith.addf %arg6, %26 : tensor<64xf32>
              scf.yield %27 : tensor<64xf32>
            }
            %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
            %10 = tt.addptr %9, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            tt.store %10, %8 : tensor<64x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// Two `scf.for` results — a running sum and a running power of two —
    /// exercising the `%r:2 = scf.for ... -> (T, U)` / `%r#0` multi-result
    /// spelling. The kernel has no tensors at all, so it also covers the
    /// scalar-only (rank-0) path.
    static let loopTwoResults = """
        module {
          tt.func public @two_results_kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2_i32 = arith.constant 2 : i32
            %0:2 = scf.for %i = %c0_i32 to %arg1 step %c1_i32
                iter_args(%sum = %c0_i32, %pow = %c1_i32) -> (i32, i32) : i32 {
              %10 = arith.addi %sum, %i : i32
              %11 = arith.muli %pow, %c2_i32 : i32
              scf.yield %10, %11 : i32, i32
            }
            %1 = tt.addptr %arg0, %c0_i32 : !tt.ptr<i32>, i32
            tt.store %1, %0#0 : !tt.ptr<i32>
            %2 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i32>, i32
            tt.store %2, %0#1 : !tt.ptr<i32>
            tt.return
          }
        }
        """

    /// A uniform `scf.if` yielding a tensor: even programs double their slice,
    /// odd programs negate it.
    static let conditional = """
        module {
          tt.func public @conditional_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                             %arg2: i32) {
            %cst2 = arith.constant dense<2.000000e+00> : tensor<64xf32>
            %cstm1 = arith.constant dense<-1.000000e+00> : tensor<64xf32>
            %c1_i32 = arith.constant 1 : i32
            %c0_i32 = arith.constant 0 : i32
            %c64_i32 = arith.constant 64 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c64_i32 : i32
            %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %3 = tt.splat %1 : i32 -> tensor<64xi32>
            %4 = arith.addi %3, %2 : tensor<64xi32>
            %5 = tt.splat %arg2 : i32 -> tensor<64xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>>
            %10 = arith.andi %0, %c1_i32 : i32
            %11 = arith.cmpi eq, %10, %c0_i32 : i32
            %12 = scf.if %11 -> (tensor<64xf32>) {
              %20 = arith.mulf %9, %cst2 : tensor<64xf32>
              scf.yield %20 : tensor<64xf32>
            } else {
              %21 = arith.mulf %9, %cstm1 : tensor<64xf32>
              scf.yield %21 : tensor<64xf32>
            }
            %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
            %14 = tt.addptr %13, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            tt.store %14, %12, %6 : tensor<64x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    // MARK: - Rank-2 tensors and broadcasting

    /// `out[m, n] = a[m, n] (+ b[m, n])` over a `BLOCK_M x BLOCK_N` tile, with the
    /// row/column masks Triton builds from `offs_m[:, None]` and `offs_n[None, :]`.
    /// `blockM`/`blockN` are deliberately awkward, and the strides come from
    /// `tt.addptr` arithmetic rather than a block pointer.
    static func tile2D(blockM: Int, blockN: Int, add: Bool) -> String {
        let secondInput = add
            ? """
                  %27 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>
                  %28 = tt.addptr %27, %16 : tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockN)xi32>
                  %29 = tt.load %28, %23 : tensor<\(blockM)x\(blockN)xf32>
                  %30 = arith.addf %26, %29 : tensor<\(blockM)x\(blockN)xf32>
            """
            : "      %30 = arith.mulf %26, %26 : tensor<\(blockM)x\(blockN)xf32>"
        return """
            module {
              tt.func public @tile2d_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                            %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32,
                                            %arg5: i32) {
                %cm = arith.constant \(blockM) : i32
                %cn = arith.constant \(blockN) : i32
                %0 = tt.get_program_id x : i32
                %1 = tt.get_program_id y : i32
                %2 = arith.muli %0, %cm : i32
                %3 = arith.muli %1, %cn : i32
                %4 = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : tensor<\(blockM)xi32>
                %5 = tt.make_range {end = \(blockN) : i32, start = 0 : i32} : tensor<\(blockN)xi32>
                %6 = tt.splat %2 : i32 -> tensor<\(blockM)xi32>
                %7 = arith.addi %6, %4 : tensor<\(blockM)xi32>
                %8 = tt.splat %3 : i32 -> tensor<\(blockN)xi32>
                %9 = arith.addi %8, %5 : tensor<\(blockN)xi32>
                %10 = tt.expand_dims %7 {axis = 1 : i32} : tensor<\(blockM)xi32> -> tensor<\(blockM)x1xi32>
                %11 = tt.expand_dims %9 {axis = 0 : i32} : tensor<\(blockN)xi32> -> tensor<1x\(blockN)xi32>
                %12 = tt.splat %arg5 : i32 -> tensor<\(blockM)x1xi32>
                %13 = arith.muli %10, %12 : tensor<\(blockM)x1xi32>
                %14 = tt.broadcast %13 : tensor<\(blockM)x1xi32> -> tensor<\(blockM)x\(blockN)xi32>
                %15 = tt.broadcast %11 : tensor<1x\(blockN)xi32> -> tensor<\(blockM)x\(blockN)xi32>
                %16 = arith.addi %14, %15 : tensor<\(blockM)x\(blockN)xi32>
                %17 = tt.splat %arg3 : i32 -> tensor<\(blockM)x1xi32>
                %18 = arith.cmpi slt, %10, %17 : tensor<\(blockM)x1xi32>
                %19 = tt.splat %arg4 : i32 -> tensor<1x\(blockN)xi32>
                %20 = arith.cmpi slt, %11, %19 : tensor<1x\(blockN)xi32>
                %21 = tt.broadcast %18 : tensor<\(blockM)x1xi1> -> tensor<\(blockM)x\(blockN)xi1>
                %22 = tt.broadcast %20 : tensor<1x\(blockN)xi1> -> tensor<\(blockM)x\(blockN)xi1>
                %23 = arith.andi %21, %22 : tensor<\(blockM)x\(blockN)xi1>
                %24 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>
                %25 = tt.addptr %24, %16 : tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockN)xi32>
                %26 = tt.load %25, %23 : tensor<\(blockM)x\(blockN)xf32>
            \(secondInput)
                %31 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>
                %32 = tt.addptr %31, %16 : tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockN)xi32>
                tt.store %32, %30, %23 : tensor<\(blockM)x\(blockN)x!tt.ptr<f32>>
                tt.return
              }
            }
            """
    }

    // MARK: - Reductions

    /// One `tt.reduce` per program over a BLOCK of 128, storing a scalar per
    /// program. `combiner` is the region body's op; `identity` the `other` value
    /// masked-off lanes contribute.
    static func reduce1D(combiner: String, identity: String) -> String {
        """
        module {
          tt.func public @reduce_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
            %cst = arith.constant dense<\(identity)> : tensor<128xf32>
            %c128_i32 = arith.constant 128 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c128_i32 : i32
            %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
            %3 = tt.splat %1 : i32 -> tensor<128xi32>
            %4 = arith.addi %3, %2 : tensor<128xi32>
            %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %9 = tt.load %8, %6, %cst : tensor<128xf32>
            %10 = "tt.reduce"(%9) <{axis = 0 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %20 = \(combiner) %a, %b : f32
              tt.reduce.return %20 : f32
            }) : (tensor<128xf32>) -> f32
            %11 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
            tt.store %11, %10 : !tt.ptr<f32>
            tt.return
          }
        }
        """
    }

    /// Row-wise `tt.reduce` over a rank-2 tile: `out[m] = sum(a[m, :])`, one
    /// program per group of 8 rows. The result is rank-1 over the *outer* block
    /// dimension, so it is computed once per row, outside the distributed loop.
    static let rowSum = """
        module {
          tt.func public @row_sum_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                         %arg2: i32, %arg3: i32, %arg4: i32) {
            %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32>
            %c8_i32 = arith.constant 8 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c8_i32 : i32
            %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
            %3 = tt.splat %1 : i32 -> tensor<8xi32>
            %4 = arith.addi %3, %2 : tensor<8xi32>
            %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
            %7 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
            %8 = tt.splat %arg4 : i32 -> tensor<8x1xi32>
            %9 = arith.muli %6, %8 : tensor<8x1xi32>
            %10 = tt.broadcast %9 : tensor<8x1xi32> -> tensor<8x64xi32>
            %11 = tt.broadcast %7 : tensor<1x64xi32> -> tensor<8x64xi32>
            %12 = arith.addi %10, %11 : tensor<8x64xi32>
            %13 = tt.splat %arg2 : i32 -> tensor<8x1xi32>
            %14 = arith.cmpi slt, %6, %13 : tensor<8x1xi32>
            %15 = tt.splat %arg3 : i32 -> tensor<1x64xi32>
            %16 = arith.cmpi slt, %7, %15 : tensor<1x64xi32>
            %17 = tt.broadcast %14 : tensor<8x1xi1> -> tensor<8x64xi1>
            %18 = tt.broadcast %16 : tensor<1x64xi1> -> tensor<8x64xi1>
            %19 = arith.andi %17, %18 : tensor<8x64xi1>
            %20 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x64x!tt.ptr<f32>>
            %21 = tt.addptr %20, %12 : tensor<8x64x!tt.ptr<f32>>, tensor<8x64xi32>
            %22 = tt.load %21, %19, %cst : tensor<8x64xf32>
            %23 = "tt.reduce"(%22) <{axis = 1 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %30 = arith.addf %a, %b : f32
              tt.reduce.return %30 : f32
            }) : (tensor<8x64xf32>) -> tensor<8xf32>
            %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
            %25 = tt.addptr %24, %4 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
            %26 = tt.splat %arg2 : i32 -> tensor<8xi32>
            %27 = arith.cmpi slt, %4, %26 : tensor<8xi32>
            tt.store %25, %23, %27 : tensor<8x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// Online (streaming) softmax: one program per row, but the row is walked in
    /// `BLOCK`-sized chunks by an `scf.for` that carries the running maximum and
    /// the running sum as **scalars**, rescaling the sum whenever the maximum
    /// moves. Both `tt.reduce`s therefore happen *inside* the loop.
    ///
    /// This is the control-flow shape a FlashAttention inner loop needs, minus
    /// the `tt.dot`s: the loop has to be lowered threadgroup-uniformly so that
    /// every thread reaches the reduction's barriers.
    static func onlineSoftmax(block: Int) -> String {
        """
        module {
          tt.func public @online_softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                                %arg2: i32, %arg3: i32, %arg4: i32) {
            %ninf = arith.constant 0xFF800000 : f32
            %cstninf = arith.constant dense<0xFF800000> : tensor<\(block)xf32>
            %zero = arith.constant 0.000000e+00 : f32
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %cb_i32 = arith.constant \(block) : i32
            %cbm1_i32 = arith.constant \(block - 1) : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %arg2 : i32
            %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
            %3 = tt.make_range {end = \(block) : i32, start = 0 : i32} : tensor<\(block)xi32>
            %4 = arith.addi %arg4, %cbm1_i32 : i32
            %5 = arith.divsi %4, %cb_i32 : i32

            %6:2 = scf.for %j = %c0_i32 to %5 step %c1_i32 iter_args(%m = %ninf, %l = %zero)
                -> (f32, f32) : i32 {
              %20 = arith.muli %j, %cb_i32 : i32
              %21 = tt.splat %20 : i32 -> tensor<\(block)xi32>
              %22 = arith.addi %21, %3 : tensor<\(block)xi32>
              %23 = tt.splat %arg4 : i32 -> tensor<\(block)xi32>
              %24 = arith.cmpi slt, %22, %23 : tensor<\(block)xi32>
              %25 = tt.splat %2 : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
              %26 = tt.addptr %25, %22 : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
              %27 = tt.load %26, %24, %cstninf : tensor<\(block)xf32>
              %28 = "tt.reduce"(%27) <{axis = 0 : i32}> ({
              ^bb0(%a: f32, %b: f32):
                %40 = arith.maxnumf %a, %b : f32
                tt.reduce.return %40 : f32
              }) : (tensor<\(block)xf32>) -> f32
              %29 = arith.maxnumf %m, %28 : f32
              %30 = tt.splat %29 : f32 -> tensor<\(block)xf32>
              %31 = arith.subf %27, %30 : tensor<\(block)xf32>
              %32 = math.exp %31 : tensor<\(block)xf32>
              %33 = "tt.reduce"(%32) <{axis = 0 : i32}> ({
              ^bb0(%c: f32, %d: f32):
                %41 = arith.addf %c, %d : f32
                tt.reduce.return %41 : f32
              }) : (tensor<\(block)xf32>) -> f32
              %34 = arith.subf %m, %29 : f32
              %35 = math.exp %34 : f32
              %36 = arith.mulf %l, %35 : f32
              %37 = arith.addf %36, %33 : f32
              scf.yield %29, %37 : f32, f32
            }

            %7 = arith.muli %0, %arg3 : i32
            %8 = tt.addptr %arg1, %7 : !tt.ptr<f32>, i32
            %9:1 = scf.for %k = %c0_i32 to %5 step %c1_i32 iter_args(%unused = %zero) -> (f32) : i32 {
              %50 = arith.muli %k, %cb_i32 : i32
              %51 = tt.splat %50 : i32 -> tensor<\(block)xi32>
              %52 = arith.addi %51, %3 : tensor<\(block)xi32>
              %53 = tt.splat %arg4 : i32 -> tensor<\(block)xi32>
              %54 = arith.cmpi slt, %52, %53 : tensor<\(block)xi32>
              %55 = tt.splat %2 : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
              %56 = tt.addptr %55, %52 : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
              %57 = tt.load %56, %54, %cstninf : tensor<\(block)xf32>
              %58 = tt.splat %6#0 : f32 -> tensor<\(block)xf32>
              %59 = arith.subf %57, %58 : tensor<\(block)xf32>
              %60 = math.exp %59 : tensor<\(block)xf32>
              %61 = tt.splat %6#1 : f32 -> tensor<\(block)xf32>
              %62 = arith.divf %60, %61 : tensor<\(block)xf32>
              %63 = tt.splat %8 : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
              %64 = tt.addptr %63, %52 : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
              tt.store %64, %62, %54 : tensor<\(block)x!tt.ptr<f32>>
              scf.yield %unused : f32
            }
            tt.return
          }
        }
        """
    }

    /// Triton's fused-softmax tutorial kernel: one program per row, a numerically
    /// stable `max -> sub -> exp -> sum -> div` over a BLOCK-wide row. `-inf` is
    /// spelled the way MLIR prints it, as a raw bit pattern.
    static func softmax(block: Int) -> String {
        """
        module {
          tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                         %arg2: i32, %arg3: i32, %arg4: i32) {
            %cst = arith.constant dense<0xFF800000> : tensor<\(block)xf32>
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %arg2 : i32
            %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
            %3 = tt.make_range {end = \(block) : i32, start = 0 : i32} : tensor<\(block)xi32>
            %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
            %5 = tt.addptr %4, %3 : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
            %6 = tt.splat %arg4 : i32 -> tensor<\(block)xi32>
            %7 = arith.cmpi slt, %3, %6 : tensor<\(block)xi32>
            %8 = tt.load %5, %7, %cst : tensor<\(block)xf32>
            %9 = "tt.reduce"(%8) <{axis = 0 : i32}> ({
            ^bb0(%arg5: f32, %arg6: f32):
              %30 = arith.maxnumf %arg5, %arg6 : f32
              tt.reduce.return %30 : f32
            }) : (tensor<\(block)xf32>) -> f32
            %10 = tt.splat %9 : f32 -> tensor<\(block)xf32>
            %11 = arith.subf %8, %10 : tensor<\(block)xf32>
            %12 = math.exp %11 : tensor<\(block)xf32>
            %13 = "tt.reduce"(%12) <{axis = 0 : i32}> ({
            ^bb0(%arg7: f32, %arg8: f32):
              %31 = arith.addf %arg7, %arg8 : f32
              tt.reduce.return %31 : f32
            }) : (tensor<\(block)xf32>) -> f32
            %14 = tt.splat %13 : f32 -> tensor<\(block)xf32>
            %15 = arith.divf %12, %14 : tensor<\(block)xf32>
            %16 = arith.muli %0, %arg3 : i32
            %17 = tt.addptr %arg1, %16 : !tt.ptr<f32>, i32
            %18 = tt.splat %17 : !tt.ptr<f32> -> tensor<\(block)x!tt.ptr<f32>>
            %19 = tt.addptr %18, %3 : tensor<\(block)x!tt.ptr<f32>>, tensor<\(block)xi32>
            tt.store %19, %15, %7 : tensor<\(block)x!tt.ptr<f32>>
            tt.return
          }
        }
        """
    }
}
