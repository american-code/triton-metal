import Foundation
import MLX
import TritonMetalCore

/// A sparse-autoencoder encoder, fused into one Triton kernel.
///
/// `h = relu(x @ W_enc + b_enc)`, with `x` an `[M, D]` batch of residual-stream
/// activations, `W_enc` a `[D, F]` dictionary and `b_enc` an `[F]` bias. It is
/// the op mechanistic-interpretability code spends most of its time in: an SAE
/// over a model's residual stream runs this on every token of every batch, and
/// `F` is typically 8-64x `D`, so the encoder is the largest single matmul in
/// the pipeline.
///
/// Fusing matters here for the ordinary reason: unfused, the bias add and the
/// ReLU are two extra full passes over an `[M, F]` tensor that is already the
/// biggest thing in memory. Written as one Triton kernel they are an add and a
/// `max` on an accumulator that is still in registers, and the `[M, F]` result
/// is written exactly once.
///
/// The IR below is a blocked GEMM in the shape Triton's own `ttir` dump prints —
/// an `scf.for` over `D` carrying the accumulator and both operand pointers,
/// masks clipping all three ragged edges — with the epilogue Triton would emit
/// for `tl.maximum(acc + bias[None, :], 0.0)`.
public enum SAEEncoder {

    /// Triton IR for the fused encoder.
    ///
    /// One program per `blockM x blockF` output tile: `grid = (cdiv(M, blockM),
    /// cdiv(F, blockF))`. Arguments, in order:
    ///
    ///     x: !tt.ptr<f32>      [M, D] row-major
    ///     w: !tt.ptr<f32>      [D, F] row-major
    ///     b: !tt.ptr<f32>      [F]
    ///     out: !tt.ptr<f32>    [M, F] row-major
    ///     M, D, F: i32
    ///
    /// Row-major contiguous layout is baked in rather than passed as six stride
    /// parameters, because that is what an MLX array of these shapes always is
    /// and the demo is clearer without them. `SAEEncoder.encode` enforces it.
    public static func ir(blockM: Int, blockF: Int, blockD: Int) -> String {
        """
        module {
          tt.func public @sae_encode_kernel(
              %x: !tt.ptr<f32>, %w: !tt.ptr<f32>, %bias: !tt.ptr<f32>, %out: !tt.ptr<f32>,
              %M: i32, %D: i32, %F: i32) {
            %acc0 = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockF)xf32>
            %relu0 = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockF)xf32>
            %zeroX = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockD)xf32>
            %zeroW = arith.constant dense<0.000000e+00> : tensor<\(blockD)x\(blockF)xf32>
            %zeroB = arith.constant dense<0.000000e+00> : tensor<1x\(blockF)xf32>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32 = arith.constant 1 : i32
            %cm_i32 = arith.constant \(blockM) : i32
            %cf_i32 = arith.constant \(blockF) : i32
            %cd_i32 = arith.constant \(blockD) : i32
            %cdm1_i32 = arith.constant \(blockD - 1) : i32

            %pid_m = tt.get_program_id x : i32
            %pid_f = tt.get_program_id y : i32
            %m_base = arith.muli %pid_m, %cm_i32 : i32
            %f_base = arith.muli %pid_f, %cf_i32 : i32
            %range_m = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : \
        tensor<\(blockM)xi32>
            %range_f = tt.make_range {end = \(blockF) : i32, start = 0 : i32} : \
        tensor<\(blockF)xi32>
            %range_d = tt.make_range {end = \(blockD) : i32, start = 0 : i32} : \
        tensor<\(blockD)xi32>
            %splat_m = tt.splat %m_base : i32 -> tensor<\(blockM)xi32>
            %offs_m = arith.addi %splat_m, %range_m : tensor<\(blockM)xi32>
            %splat_f = tt.splat %f_base : i32 -> tensor<\(blockF)xi32>
            %offs_f = arith.addi %splat_f, %range_f : tensor<\(blockF)xi32>

            %rows = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<\(blockM)xi32> -> \
        tensor<\(blockM)x1xi32>
            %cols = tt.expand_dims %offs_f {axis = 0 : i32} : tensor<\(blockF)xi32> -> \
        tensor<1x\(blockF)xi32>
            %dcols = tt.expand_dims %range_d {axis = 0 : i32} : tensor<\(blockD)xi32> -> \
        tensor<1x\(blockD)xi32>
            %drows = tt.expand_dims %range_d {axis = 1 : i32} : tensor<\(blockD)xi32> -> \
        tensor<\(blockD)x1xi32>

            // x_ptrs = x + offs_m[:, None] * D + offs_d[None, :]
            %splat_D_m = tt.splat %D : i32 -> tensor<\(blockM)x1xi32>
            %x_row = arith.muli %rows, %splat_D_m : tensor<\(blockM)x1xi32>
            %x_rowb = tt.broadcast %x_row : tensor<\(blockM)x1xi32> -> \
        tensor<\(blockM)x\(blockD)xi32>
            %x_colb = tt.broadcast %dcols : tensor<1x\(blockD)xi32> -> \
        tensor<\(blockM)x\(blockD)xi32>
            %x_off = arith.addi %x_rowb, %x_colb : tensor<\(blockM)x\(blockD)xi32>
            %x_splat = tt.splat %x : !tt.ptr<f32> -> \
        tensor<\(blockM)x\(blockD)x!tt.ptr<f32>>
            %x_ptrs = tt.addptr %x_splat, %x_off : \
        tensor<\(blockM)x\(blockD)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockD)xi32>

            // w_ptrs = w + offs_d[:, None] * F + offs_f[None, :]
            %splat_F_d = tt.splat %F : i32 -> tensor<\(blockD)x1xi32>
            %w_row = arith.muli %drows, %splat_F_d : tensor<\(blockD)x1xi32>
            %w_rowb = tt.broadcast %w_row : tensor<\(blockD)x1xi32> -> \
        tensor<\(blockD)x\(blockF)xi32>
            %w_colb = tt.broadcast %cols : tensor<1x\(blockF)xi32> -> \
        tensor<\(blockD)x\(blockF)xi32>
            %w_off = arith.addi %w_rowb, %w_colb : tensor<\(blockD)x\(blockF)xi32>
            %w_splat = tt.splat %w : !tt.ptr<f32> -> \
        tensor<\(blockD)x\(blockF)x!tt.ptr<f32>>
            %w_ptrs = tt.addptr %w_splat, %w_off : \
        tensor<\(blockD)x\(blockF)x!tt.ptr<f32>>, tensor<\(blockD)x\(blockF)xi32>

            // Row and column masks: loop invariant, reused by the epilogue.
            %splat_M = tt.splat %M : i32 -> tensor<\(blockM)x1xi32>
            %mask_m = arith.cmpi slt, %rows, %splat_M : tensor<\(blockM)x1xi32>
            %splat_F = tt.splat %F : i32 -> tensor<1x\(blockF)xi32>
            %mask_f = arith.cmpi slt, %cols, %splat_F : tensor<1x\(blockF)xi32>

            %trip0 = arith.addi %D, %cdm1_i32 : i32
            %trips = arith.divsi %trip0, %cd_i32 : i32
            %w_step = arith.muli %F, %cd_i32 : i32

            %loop:3 = scf.for %k = %c0_i32 to %trips step %c1_i32
                iter_args(%acc = %acc0, %xp = %x_ptrs, %wp = %w_ptrs)
                -> (tensor<\(blockM)x\(blockF)xf32>,
                    tensor<\(blockM)x\(blockD)x!tt.ptr<f32>>,
                    tensor<\(blockD)x\(blockF)x!tt.ptr<f32>>) : i32 {
              %consumed = arith.muli %k, %cd_i32 : i32
              %left = arith.subi %D, %consumed : i32

              %left_cols = tt.splat %left : i32 -> tensor<1x\(blockD)xi32>
              %mask_dc = arith.cmpi slt, %dcols, %left_cols : tensor<1x\(blockD)xi32>
              %mask_dcb = tt.broadcast %mask_dc : tensor<1x\(blockD)xi1> -> \
        tensor<\(blockM)x\(blockD)xi1>
              %mask_mb = tt.broadcast %mask_m : tensor<\(blockM)x1xi1> -> \
        tensor<\(blockM)x\(blockD)xi1>
              %mask_x = arith.andi %mask_mb, %mask_dcb : tensor<\(blockM)x\(blockD)xi1>
              %xv = tt.load %xp, %mask_x, %zeroX : tensor<\(blockM)x\(blockD)xf32>

              %left_rows = tt.splat %left : i32 -> tensor<\(blockD)x1xi32>
              %mask_dr = arith.cmpi slt, %drows, %left_rows : tensor<\(blockD)x1xi32>
              %mask_drb = tt.broadcast %mask_dr : tensor<\(blockD)x1xi1> -> \
        tensor<\(blockD)x\(blockF)xi1>
              %mask_fb = tt.broadcast %mask_f : tensor<1x\(blockF)xi1> -> \
        tensor<\(blockD)x\(blockF)xi1>
              %mask_w = arith.andi %mask_drb, %mask_fb : tensor<\(blockD)x\(blockF)xi1>
              %wv = tt.load %wp, %mask_w, %zeroW : tensor<\(blockD)x\(blockF)xf32>

              %acc1 = tt.dot %xv, %wv, %acc, inputPrecision = tf32 : \
        tensor<\(blockM)x\(blockD)xf32> * tensor<\(blockD)x\(blockF)xf32> -> \
        tensor<\(blockM)x\(blockF)xf32>

              %x_adv = tt.splat %cd_i32 : i32 -> tensor<\(blockM)x\(blockD)xi32>
              %xp1 = tt.addptr %xp, %x_adv : \
        tensor<\(blockM)x\(blockD)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockD)xi32>
              %w_adv = tt.splat %w_step : i32 -> tensor<\(blockD)x\(blockF)xi32>
              %wp1 = tt.addptr %wp, %w_adv : \
        tensor<\(blockD)x\(blockF)x!tt.ptr<f32>>, tensor<\(blockD)x\(blockF)xi32>
              scf.yield %acc1, %xp1, %wp1 : tensor<\(blockM)x\(blockF)xf32>, \
        tensor<\(blockM)x\(blockD)x!tt.ptr<f32>>, tensor<\(blockD)x\(blockF)x!tt.ptr<f32>>
            }

            // Epilogue: + b_enc[None, :], then relu, on the accumulator.
            %b_splat = tt.splat %bias : !tt.ptr<f32> -> tensor<1x\(blockF)x!tt.ptr<f32>>
            %b_ptrs = tt.addptr %b_splat, %cols : \
        tensor<1x\(blockF)x!tt.ptr<f32>>, tensor<1x\(blockF)xi32>
            %b_val = tt.load %b_ptrs, %mask_f, %zeroB : tensor<1x\(blockF)xf32>
            %b_bcast = tt.broadcast %b_val : tensor<1x\(blockF)xf32> -> \
        tensor<\(blockM)x\(blockF)xf32>
            %biased = arith.addf %loop#0, %b_bcast : tensor<\(blockM)x\(blockF)xf32>
            %activated = arith.maximumf %biased, %relu0 : tensor<\(blockM)x\(blockF)xf32>

            // out_ptrs = out + offs_m[:, None] * F + offs_f[None, :]
            %splat_F_m = tt.splat %F : i32 -> tensor<\(blockM)x1xi32>
            %o_row = arith.muli %rows, %splat_F_m : tensor<\(blockM)x1xi32>
            %o_rowb = tt.broadcast %o_row : tensor<\(blockM)x1xi32> -> \
        tensor<\(blockM)x\(blockF)xi32>
            %o_colb = tt.broadcast %cols : tensor<1x\(blockF)xi32> -> \
        tensor<\(blockM)x\(blockF)xi32>
            %o_off = arith.addi %o_rowb, %o_colb : tensor<\(blockM)x\(blockF)xi32>
            %o_splat = tt.splat %out : !tt.ptr<f32> -> \
        tensor<\(blockM)x\(blockF)x!tt.ptr<f32>>
            %o_ptrs = tt.addptr %o_splat, %o_off : \
        tensor<\(blockM)x\(blockF)x!tt.ptr<f32>>, tensor<\(blockM)x\(blockF)xi32>
            %o_mask_m = tt.broadcast %mask_m : tensor<\(blockM)x1xi1> -> \
        tensor<\(blockM)x\(blockF)xi1>
            %o_mask_f = tt.broadcast %mask_f : tensor<1x\(blockF)xi1> -> \
        tensor<\(blockM)x\(blockF)xi1>
            %o_mask = arith.andi %o_mask_m, %o_mask_f : tensor<\(blockM)x\(blockF)xi1>
            tt.store %o_ptrs, %activated, %o_mask : tensor<\(blockM)x\(blockF)x!tt.ptr<f32>>
            tt.return
          }
        }
        """
    }

    /// Block shape, defaulted to what the M1 Pro measurements in
    /// docs/ARCHITECTURE.md §Matmul throughput settled on for f32.
    public struct Blocking: Sendable, Equatable {
        public var m: Int
        public var f: Int
        public var d: Int

        public init(m: Int = 64, f: Int = 64, d: Int = 32) {
            self.m = m
            self.f = f
            self.d = d
        }
    }

    /// A compiled encoder, ready to run on MLX tensors.
    ///
    /// Holding one of these is the point: compilation happens once, and each
    /// `callAsFunction` is a dispatch and nothing else.
    public final class Compiled {
        public let pipeline: MetalPipeline
        public let blocking: Blocking

        init(pipeline: MetalPipeline, blocking: Blocking) {
            self.pipeline = pipeline
            self.blocking = blocking
        }

        /// `relu(x @ wEnc + bEnc)`.
        ///
        /// Shapes are checked here rather than in `MetalPipeline.launch`, which
        /// cannot: a Triton kernel's arguments carry no extents by the time they
        /// reach this backend (see `MetalPipeline.launch`). This wrapper knows
        /// what its own op means, so it can.
        public func callAsFunction(
            _ x: MLXArray, _ wEnc: MLXArray, _ bEnc: MLXArray
        ) throws -> MLXArray {
            guard x.ndim == 2, wEnc.ndim == 2, bEnc.ndim == 1 else {
                throw CoreError.invalidArgument(
                    "sae_encode expects x [M, D], W_enc [D, F], b_enc [F]; got \(x.shape), "
                        + "\(wEnc.shape), \(bEnc.shape)")
            }
            let (rows, model) = (x.shape[0], x.shape[1])
            let features = wEnc.shape[1]
            guard wEnc.shape[0] == model else {
                throw CoreError.invalidArgument(
                    "x is [\(rows), \(model)] and W_enc is \(wEnc.shape); their contraction "
                        + "dimensions differ")
            }
            guard bEnc.shape[0] == features else {
                throw CoreError.invalidArgument(
                    "W_enc has \(features) features but b_enc has \(bEnc.shape[0])")
            }

            let out = MLX.zeros([rows, features], dtype: .float32)
            let results = try pipeline.launch(
                arrays: [x, wEnc, bEnc, out],
                scalars: [.init(rows), .init(model), .init(features)],
                grid: Grid(
                    Grid.covering(rows, block: blocking.m),
                    Grid.covering(features, block: blocking.f)))
            return results[3]
        }
    }

    /// Compiles the encoder for a block shape.
    public static func compile(
        blocking: Blocking = .init(), options: MetalCompiler.Options = .init()
    ) throws -> Compiled {
        Compiled(
            pipeline: try MetalPipeline.compile(
                ttir: ir(blockM: blocking.m, blockF: blocking.f, blockD: blocking.d),
                options: options),
            blocking: blocking)
    }

    /// What MLX's own ops compute for the same thing — the reference the kernel
    /// is checked against.
    public static func reference(_ x: MLXArray, _ wEnc: MLXArray, _ bEnc: MLXArray) -> MLXArray {
        MLX.maximum(MLX.matmul(x, wEnc) + bEnc, MLXArray(Float(0)))
    }
}
