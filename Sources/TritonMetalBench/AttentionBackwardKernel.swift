import Foundation

/// Triton IR for the FlashAttention-2 **backward** pass, in the shape of
/// Triton's own tutorial: recompute-based, storing only the per-row logsumexp
/// from the forward and rebuilding `P` in the backward, and split so that no
/// gradient needs a cross-program reduction.
///
/// The maths, once, because every kernel below is a projection of it. With
/// `S = sm_scale * Q K^T`, `P = softmax(S)` row-wise and `O = P V`:
///
/// ```
/// dP_ij    = dO_i . V_j
/// Delta_i  = dO_i . O_i            = sum_j P_ij dP_ij
/// dS_ij    = P_ij (dP_ij - Delta_i)
/// dV_j     = sum_i P_ij dO_i
/// dQ_i     = sm_scale * sum_j dS_ij K_j
/// dK_j     = sm_scale * sum_i dS_ij Q_i
/// ```
///
/// `P` is never stored. It is rebuilt from `M_i`, the forward's per-row
/// logsumexp in **log2 units** (`AttentionKernel.forward(emitStats: true)`
/// writes `m_i + log2(l_i)`), as `P_ij = exp2(qk_scale * (Q_i . K_j) - M_i)`
/// with `qk_scale = sm_scale * log2(e)` — the same expression the forward
/// evaluates, so the recomputation is bit-identical to what the forward did up
/// to the accumulation order of the dot.
///
/// **Why three kernels rather than one.** `dQ_i` sums over key blocks and
/// `dK_j`/`dV_j` sum over query blocks; a single program cannot own both ends of
/// that. Two ways out: accumulate `dQ` with `tt.atomic_rmw fadd` from the
/// key-block program, or run the two directions as separate programs. This takes
/// the second — the split two-pass formulation Triton's tutorial uses — because
/// it needs nothing the forward did not already need, keeps every gradient
/// deterministic, and each program's accumulator stays a register-resident dot
/// accumulator instead of a stream of read-modify-writes to device memory. The
/// atomic path is measured against it in `tmbench --attn-bwd`.
public enum AttentionBackwardKernel {

    /// `Delta_i = dO_i . O_i`, one row per query.
    ///
    /// A rank-2 load and a row reduction — the cheapest of the three kernels and
    /// the reason the other two never need `O` at all. One program per `BLOCK_M`
    /// block of queries (`program_id x`) per batch-and-head (`program_id y`).
    ///
    /// Arguments: `O`, `dO`, `Delta`, `stride_head`, `stride_seq`, `stride_lse`,
    /// `n_ctx`. `Delta` is `[B*H, S]` f32, so `stride_lse` is the distance
    /// between one head's row statistics and the next.
    public static func preprocess(blockM: Int, headDim: Int, element: String = "f32") -> String {
        let isHalf = element == "f16"
        let widen = isHalf
            ? """
                  %o_f = arith.extf %o : tensor<\(blockM)x\(headDim)xf16> to \
            tensor<\(blockM)x\(headDim)xf32>
                  %do_f = arith.extf %do : tensor<\(blockM)x\(headDim)xf16> to \
            tensor<\(blockM)x\(headDim)xf32>
            """
            : """
                  %o_f = arith.mulf %o, %one_md : tensor<\(blockM)x\(headDim)xf32>
                  %do_f = arith.mulf %do, %one_md : tensor<\(blockM)x\(headDim)xf32>
            """
        return """
            module {
              tt.func public @attn_bwd_preprocess(
                  %O: !tt.ptr<\(element)>, %DO: !tt.ptr<\(element)>, %Delta: !tt.ptr<f32>,
                  %stride_head: i32, %stride_seq: i32, %stride_lse: i32, %n_ctx: i32) {
                %cbm_i32 = arith.constant \(blockM) : i32
                %one_md = arith.constant dense<1.000000e+00> : \
            tensor<\(blockM)x\(headDim)xf32>
                %zero_md = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)x\(element)>

                %pid_m = tt.get_program_id x : i32
                %pid_h = tt.get_program_id y : i32
                %m_start = arith.muli %pid_m, %cbm_i32 : i32
                %h_off = arith.muli %pid_h, %stride_head : i32
                %lse_off = arith.muli %pid_h, %stride_lse : i32
                %o_base = tt.addptr %O, %h_off : !tt.ptr<\(element)>, i32
                %do_base = tt.addptr %DO, %h_off : !tt.ptr<\(element)>, i32
                %d_base = tt.addptr %Delta, %lse_off : !tt.ptr<f32>, i32

                %range_m = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : \
            tensor<\(blockM)xi32>
                %range_d = tt.make_range {end = \(headDim) : i32, start = 0 : i32} : \
            tensor<\(headDim)xi32>
                %m_splat = tt.splat %m_start : i32 -> tensor<\(blockM)xi32>
                %offs_m = arith.addi %m_splat, %range_m : tensor<\(blockM)xi32>
                %offs_m_r = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<\(blockM)xi32> -> \
            tensor<\(blockM)x1xi32>
                %offs_d_c = tt.expand_dims %range_d {axis = 0 : i32} : tensor<\(headDim)xi32> -> \
            tensor<1x\(headDim)xi32>
                %ss_m = tt.splat %stride_seq : i32 -> tensor<\(blockM)x1xi32>
                %row = arith.muli %offs_m_r, %ss_m : tensor<\(blockM)x1xi32>
                %row_b = tt.broadcast %row : tensor<\(blockM)x1xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %col_b = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %off = arith.addi %row_b, %col_b : tensor<\(blockM)x\(headDim)xi32>
                %nctx_r = tt.splat %n_ctx : i32 -> tensor<\(blockM)x1xi32>
                %row_ok = arith.cmpi slt, %offs_m_r, %nctx_r : tensor<\(blockM)x1xi32>
                %mask = tt.broadcast %row_ok : tensor<\(blockM)x1xi1> -> \
            tensor<\(blockM)x\(headDim)xi1>

                %o_p = tt.splat %o_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %o_ptrs = tt.addptr %o_p, %off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                %o = tt.load %o_ptrs, %mask, %zero_md : tensor<\(blockM)x\(headDim)x\(element)>
                %do_p = tt.splat %do_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %do_ptrs = tt.addptr %do_p, %off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                %do = tt.load %do_ptrs, %mask, %zero_md : tensor<\(blockM)x\(headDim)x\(element)>
            \(widen)
                %prod = arith.mulf %o_f, %do_f : tensor<\(blockM)x\(headDim)xf32>
                %delta = "tt.reduce"(%prod) <{axis = 1 : i32}> ({
                ^bb0(%da: f32, %db: f32):
                  %ds = arith.addf %da, %db : f32
                  tt.reduce.return %ds : f32
                }) : (tensor<\(blockM)x\(headDim)xf32>) -> tensor<\(blockM)xf32>

                %d_p = tt.splat %d_base : !tt.ptr<f32> -> tensor<\(blockM)x!tt.ptr<f32>>
                %d_ptrs = tt.addptr %d_p, %offs_m : tensor<\(blockM)x!tt.ptr<f32>>, \
            tensor<\(blockM)xi32>
                %nctx_1 = tt.splat %n_ctx : i32 -> tensor<\(blockM)xi32>
                %ok = arith.cmpi slt, %offs_m, %nctx_1 : tensor<\(blockM)xi32>
                tt.store %d_ptrs, %delta, %ok : tensor<\(blockM)x!tt.ptr<f32>>
                tt.return
              }
            }
            """
    }

    /// `dQ_i = sm_scale * sum_j dS_ij K_j`, one program per `BLOCK_M` block of
    /// queries, looping over key blocks — the mirror image of the forward.
    ///
    /// Three `tt.dot`s per iteration (`Q K^T`, `dO V^T`, `dS K`) and one
    /// loop-carried accumulator, which the loop yields directly, so `dQ` stays in
    /// simdgroup registers for the whole key loop. Everything between the first
    /// two dots and the third — the rescale, the exponential, the `dP - Delta`
    /// correction — is a `tt.dot` operand and nothing else, so none of it is
    /// materialised: it is rebuilt inside the third dot's staging loops, reading
    /// the first two dots' result tiles where they sit.
    ///
    /// Arguments: `Q`, `K`, `V`, `dO`, `dQ`, `M` (log2 logsumexp), `Delta`,
    /// `sm_scale`, `stride_head`, `stride_seq`, `stride_lse`, `n_ctx`.
    public static func dq(
        blockM: Int, blockN: Int, headDim: Int, element: String = "f32"
    ) -> String {
        let isHalf = element == "f16"
        let dsOperand = isHalf
            ? """
                      %ds_op = arith.truncf %ds : tensor<\(blockM)x\(blockN)xf32> to \
                tensor<\(blockM)x\(blockN)xf16>
                """
            : "          %ds_op = arith.mulf %ds, %one_mn : tensor<\(blockM)x\(blockN)xf32>"
        let output = isHalf
            ? """
                  %dq_e = arith.truncf %dq_scaled : tensor<\(blockM)x\(headDim)xf32> to \
                tensor<\(blockM)x\(headDim)xf16>
                """
            : "      %dq_e = arith.mulf %dq_scaled, %one_md : tensor<\(blockM)x\(headDim)xf32>"

        return """
            module {
              tt.func public @attn_bwd_dq(
                  %Q: !tt.ptr<\(element)>, %K: !tt.ptr<\(element)>, %V: !tt.ptr<\(element)>,
                  %DO: !tt.ptr<\(element)>, %DQ: !tt.ptr<\(element)>,
                  %L: !tt.ptr<f32>, %D: !tt.ptr<f32>, %sm_scale: f32,
                  %stride_head: i32, %stride_seq: i32, %stride_lse: i32, %n_ctx: i32) {
                %log2e = arith.constant 1.44269504088896340736 : f32
                %qk_scale = arith.mulf %sm_scale, %log2e : f32
                %c0_i32 = arith.constant 0 : i32
                %c1_i32 = arith.constant 1 : i32
                %cbm_i32 = arith.constant \(blockM) : i32
                %cbn_i32 = arith.constant \(blockN) : i32
                %cbnm1_i32 = arith.constant \(blockN - 1) : i32
                %zero_f32 = arith.constant 0.000000e+00 : f32
                %zero_m = arith.constant dense<0.000000e+00> : tensor<\(blockM)xf32>
                %zero_mn = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockN)xf32>
                %ninf_mn = arith.constant dense<0xFF800000> : tensor<\(blockM)x\(blockN)xf32>
                %one_mn = arith.constant dense<1.000000e+00> : tensor<\(blockM)x\(blockN)xf32>
                %one_md = arith.constant dense<1.000000e+00> : \
            tensor<\(blockM)x\(headDim)xf32>
                %dq_init = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)xf32>
                %zero_md = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)x\(element)>
                %zero_nd = arith.constant dense<0.000000e+00> : \
            tensor<\(blockN)x\(headDim)x\(element)>

                %pid_m = tt.get_program_id x : i32
                %pid_h = tt.get_program_id y : i32
                %m_start = arith.muli %pid_m, %cbm_i32 : i32
                %h_off = arith.muli %pid_h, %stride_head : i32
                %lse_off = arith.muli %pid_h, %stride_lse : i32
                %q_base = tt.addptr %Q, %h_off : !tt.ptr<\(element)>, i32
                %k_base = tt.addptr %K, %h_off : !tt.ptr<\(element)>, i32
                %v_base = tt.addptr %V, %h_off : !tt.ptr<\(element)>, i32
                %do_base = tt.addptr %DO, %h_off : !tt.ptr<\(element)>, i32
                %dq_base = tt.addptr %DQ, %h_off : !tt.ptr<\(element)>, i32
                %l_base = tt.addptr %L, %lse_off : !tt.ptr<f32>, i32
                %d_base = tt.addptr %D, %lse_off : !tt.ptr<f32>, i32

                %range_m = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : \
            tensor<\(blockM)xi32>
                %range_n = tt.make_range {end = \(blockN) : i32, start = 0 : i32} : \
            tensor<\(blockN)xi32>
                %range_d = tt.make_range {end = \(headDim) : i32, start = 0 : i32} : \
            tensor<\(headDim)xi32>
                %m_splat = tt.splat %m_start : i32 -> tensor<\(blockM)xi32>
                %offs_m = arith.addi %m_splat, %range_m : tensor<\(blockM)xi32>
                %offs_m_r = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<\(blockM)xi32> -> \
            tensor<\(blockM)x1xi32>
                %offs_d_c = tt.expand_dims %range_d {axis = 0 : i32} : tensor<\(headDim)xi32> -> \
            tensor<1x\(headDim)xi32>

                %ss_m = tt.splat %stride_seq : i32 -> tensor<\(blockM)x1xi32>
                %q_row = arith.muli %offs_m_r, %ss_m : tensor<\(blockM)x1xi32>
                %q_row_b = tt.broadcast %q_row : tensor<\(blockM)x1xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %d_col_m = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %q_off = arith.addi %q_row_b, %d_col_m : tensor<\(blockM)x\(headDim)xi32>
                %nctx_r = tt.splat %n_ctx : i32 -> tensor<\(blockM)x1xi32>
                %row_ok = arith.cmpi slt, %offs_m_r, %nctx_r : tensor<\(blockM)x1xi32>
                %q_mask = tt.broadcast %row_ok : tensor<\(blockM)x1xi1> -> \
            tensor<\(blockM)x\(headDim)xi1>
                %q_p = tt.splat %q_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %q_ptrs = tt.addptr %q_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                %q = tt.load %q_ptrs, %q_mask, %zero_md : tensor<\(blockM)x\(headDim)x\(element)>
                %do_p = tt.splat %do_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %do_ptrs = tt.addptr %do_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                %do = tt.load %do_ptrs, %q_mask, %zero_md : \
            tensor<\(blockM)x\(headDim)x\(element)>

                %nctx_m = tt.splat %n_ctx : i32 -> tensor<\(blockM)xi32>
                %m_ok = arith.cmpi slt, %offs_m, %nctx_m : tensor<\(blockM)xi32>
                %l_p = tt.splat %l_base : !tt.ptr<f32> -> tensor<\(blockM)x!tt.ptr<f32>>
                %l_ptrs = tt.addptr %l_p, %offs_m : tensor<\(blockM)x!tt.ptr<f32>>, \
            tensor<\(blockM)xi32>
                %lse = tt.load %l_ptrs, %m_ok, %zero_m : tensor<\(blockM)xf32>
                %dd_p = tt.splat %d_base : !tt.ptr<f32> -> tensor<\(blockM)x!tt.ptr<f32>>
                %dd_ptrs = tt.addptr %dd_p, %offs_m : tensor<\(blockM)x!tt.ptr<f32>>, \
            tensor<\(blockM)xi32>
                %delta = tt.load %dd_ptrs, %m_ok, %zero_m : tensor<\(blockM)xf32>
                %lse_r = tt.expand_dims %lse {axis = 1 : i32} : tensor<\(blockM)xf32> -> \
            tensor<\(blockM)x1xf32>
                %lse_b = tt.broadcast %lse_r : tensor<\(blockM)x1xf32> -> \
            tensor<\(blockM)x\(blockN)xf32>
                %delta_r = tt.expand_dims %delta {axis = 1 : i32} : tensor<\(blockM)xf32> -> \
            tensor<\(blockM)x1xf32>
                %delta_b = tt.broadcast %delta_r : tensor<\(blockM)x1xf32> -> \
            tensor<\(blockM)x\(blockN)xf32>

                %n_round = arith.addi %n_ctx, %cbnm1_i32 : i32
                %trips = arith.divsi %n_round, %cbn_i32 : i32

                %dq_res = scf.for %j = %c0_i32 to %trips step %c1_i32
                    iter_args(%dq = %dq_init) -> (tensor<\(blockM)x\(headDim)xf32>) : i32 {
                  %start_n = arith.muli %j, %cbn_i32 : i32
                  %n_splat = tt.splat %start_n : i32 -> tensor<\(blockN)xi32>
                  %offs_n = arith.addi %n_splat, %range_n : tensor<\(blockN)xi32>
                  %offs_n_r = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<\(blockN)xi32> -> \
            tensor<\(blockN)x1xi32>
                  %offs_n_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<\(blockN)xi32> -> \
            tensor<1x\(blockN)xi32>

                  %ss_n = tt.splat %stride_seq : i32 -> tensor<\(blockN)x1xi32>
                  %kv_row = arith.muli %offs_n_r, %ss_n : tensor<\(blockN)x1xi32>
                  %kv_row_b = tt.broadcast %kv_row : tensor<\(blockN)x1xi32> -> \
            tensor<\(blockN)x\(headDim)xi32>
                  %d_col_n = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockN)x\(headDim)xi32>
                  %kv_off = arith.addi %kv_row_b, %d_col_n : tensor<\(blockN)x\(headDim)xi32>
                  %nctx_n = tt.splat %n_ctx : i32 -> tensor<\(blockN)x1xi32>
                  %key_ok = arith.cmpi slt, %offs_n_r, %nctx_n : tensor<\(blockN)x1xi32>
                  %kv_mask = tt.broadcast %key_ok : tensor<\(blockN)x1xi1> -> \
            tensor<\(blockN)x\(headDim)xi1>
                  %k_p = tt.splat %k_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                  %k_ptrs = tt.addptr %k_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                  %k = tt.load %k_ptrs, %kv_mask, %zero_nd : \
            tensor<\(blockN)x\(headDim)x\(element)>
                  %kt = tt.trans %k {order = array<i32: 1, 0>} : \
            tensor<\(blockN)x\(headDim)x\(element)> -> tensor<\(headDim)x\(blockN)x\(element)>
                  %v_p = tt.splat %v_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                  %v_ptrs = tt.addptr %v_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                  %v = tt.load %v_ptrs, %kv_mask, %zero_nd : \
            tensor<\(blockN)x\(headDim)x\(element)>
                  %vt = tt.trans %v {order = array<i32: 1, 0>} : \
            tensor<\(blockN)x\(headDim)x\(element)> -> tensor<\(headDim)x\(blockN)x\(element)>

                  %qk_raw = tt.dot %q, %kt, %zero_mn, inputPrecision = ieee : \
            tensor<\(blockM)x\(headDim)x\(element)> * tensor<\(headDim)x\(blockN)x\(element)> -> \
            tensor<\(blockM)x\(blockN)xf32>
                  %dp_raw = tt.dot %do, %vt, %zero_mn, inputPrecision = ieee : \
            tensor<\(blockM)x\(headDim)x\(element)> * tensor<\(headDim)x\(blockN)x\(element)> -> \
            tensor<\(blockM)x\(blockN)xf32>

                  %scale_b = tt.splat %qk_scale : f32 -> tensor<\(blockM)x\(blockN)xf32>
                  %qk_scaled = arith.mulf %qk_raw, %scale_b : tensor<\(blockM)x\(blockN)xf32>
                  %nctx_c = tt.splat %n_ctx : i32 -> tensor<1x\(blockN)xi32>
                  %col_ok = arith.cmpi slt, %offs_n_c, %nctx_c : tensor<1x\(blockN)xi32>
                  %col_mask = tt.broadcast %col_ok : tensor<1x\(blockN)xi1> -> \
            tensor<\(blockM)x\(blockN)xi1>
                  %qk = arith.select %col_mask, %qk_scaled, %ninf_mn : \
            tensor<\(blockM)x\(blockN)xi1>, tensor<\(blockM)x\(blockN)xf32>
                  %qk_centred = arith.subf %qk, %lse_b : tensor<\(blockM)x\(blockN)xf32>
                  %p = math.exp2 %qk_centred : tensor<\(blockM)x\(blockN)xf32>
                  %dp_centred = arith.subf %dp_raw, %delta_b : tensor<\(blockM)x\(blockN)xf32>
                  %ds = arith.mulf %p, %dp_centred : tensor<\(blockM)x\(blockN)xf32>
            \(dsOperand)
                  %dq_new = tt.dot %ds_op, %k, %dq, inputPrecision = ieee : \
            tensor<\(blockM)x\(blockN)x\(element)> * tensor<\(blockN)x\(headDim)x\(element)> -> \
            tensor<\(blockM)x\(headDim)xf32>
                  scf.yield %dq_new : tensor<\(blockM)x\(headDim)xf32>
                }

                %sm_b = tt.splat %sm_scale : f32 -> tensor<\(blockM)x\(headDim)xf32>
                %dq_scaled = arith.mulf %dq_res, %sm_b : tensor<\(blockM)x\(headDim)xf32>
            \(output)
                %dq_p = tt.splat %dq_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %dq_ptrs = tt.addptr %dq_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                tt.store %dq_ptrs, %dq_e, %q_mask : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                tt.return
              }
            }
            """
    }

    /// `dV_j = sum_i P_ij dO_i` and `dK_j = sm_scale * sum_i dS_ij Q_i`, one
    /// program per `BLOCK_N` block of keys, looping over query blocks.
    ///
    /// Everything is transposed relative to `dq`: the program owns a block of
    /// keys, so the score tile is `P^T` (`BLOCK_N x BLOCK_M`) and the two
    /// accumulators are `BLOCK_N x HEAD_DIM`. Four `tt.dot`s per iteration
    /// (`K Q^T`, `P^T dO`, `V dO^T`, `dS^T Q`) and **two** loop-carried
    /// accumulators, both yielded directly by their dot.
    public static func dkdv(
        blockM: Int, blockN: Int, headDim: Int, element: String = "f32"
    ) -> String {
        let isHalf = element == "f16"
        let ptOperand = isHalf
            ? """
                      %pt_op = arith.truncf %pt : tensor<\(blockN)x\(blockM)xf32> to \
                tensor<\(blockN)x\(blockM)xf16>
                """
            : "          %pt_op = arith.mulf %pt, %one_nm : tensor<\(blockN)x\(blockM)xf32>"
        let dstOperand = isHalf
            ? """
                      %dst_op = arith.truncf %dst : tensor<\(blockN)x\(blockM)xf32> to \
                tensor<\(blockN)x\(blockM)xf16>
                """
            : "          %dst_op = arith.mulf %dst, %one_nm : tensor<\(blockN)x\(blockM)xf32>"
        let outputs = isHalf
            ? """
                  %dk_e = arith.truncf %dk_scaled : tensor<\(blockN)x\(headDim)xf32> to \
                tensor<\(blockN)x\(headDim)xf16>
                      %dv_e = arith.truncf %dv_res : tensor<\(blockN)x\(headDim)xf32> to \
                tensor<\(blockN)x\(headDim)xf16>
                """
            : """
                  %dk_e = arith.mulf %dk_scaled, %one_nd : tensor<\(blockN)x\(headDim)xf32>
                      %dv_e = arith.mulf %dv_res, %one_nd : tensor<\(blockN)x\(headDim)xf32>
                """

        return """
            module {
              tt.func public @attn_bwd_dkdv(
                  %Q: !tt.ptr<\(element)>, %K: !tt.ptr<\(element)>, %V: !tt.ptr<\(element)>,
                  %DO: !tt.ptr<\(element)>, %DK: !tt.ptr<\(element)>, %DV: !tt.ptr<\(element)>,
                  %L: !tt.ptr<f32>, %D: !tt.ptr<f32>, %sm_scale: f32,
                  %stride_head: i32, %stride_seq: i32, %stride_lse: i32, %n_ctx: i32) {
                %log2e = arith.constant 1.44269504088896340736 : f32
                %qk_scale = arith.mulf %sm_scale, %log2e : f32
                %c0_i32 = arith.constant 0 : i32
                %c1_i32 = arith.constant 1 : i32
                %cbm_i32 = arith.constant \(blockM) : i32
                %cbn_i32 = arith.constant \(blockN) : i32
                %cbmm1_i32 = arith.constant \(blockM - 1) : i32
                %zero_m = arith.constant dense<0.000000e+00> : tensor<\(blockM)xf32>
                %zero_nm = arith.constant dense<0.000000e+00> : tensor<\(blockN)x\(blockM)xf32>
                %ninf_nm = arith.constant dense<0xFF800000> : tensor<\(blockN)x\(blockM)xf32>
                %one_nm = arith.constant dense<1.000000e+00> : tensor<\(blockN)x\(blockM)xf32>
                %one_nd = arith.constant dense<1.000000e+00> : \
            tensor<\(blockN)x\(headDim)xf32>
                %dk_init = arith.constant dense<0.000000e+00> : \
            tensor<\(blockN)x\(headDim)xf32>
                %dv_init = arith.constant dense<0.000000e+00> : \
            tensor<\(blockN)x\(headDim)xf32>
                %zero_md = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)x\(element)>
                %zero_nd = arith.constant dense<0.000000e+00> : \
            tensor<\(blockN)x\(headDim)x\(element)>

                %pid_n = tt.get_program_id x : i32
                %pid_h = tt.get_program_id y : i32
                %n_start = arith.muli %pid_n, %cbn_i32 : i32
                %h_off = arith.muli %pid_h, %stride_head : i32
                %lse_off = arith.muli %pid_h, %stride_lse : i32
                %q_base = tt.addptr %Q, %h_off : !tt.ptr<\(element)>, i32
                %k_base = tt.addptr %K, %h_off : !tt.ptr<\(element)>, i32
                %v_base = tt.addptr %V, %h_off : !tt.ptr<\(element)>, i32
                %do_base = tt.addptr %DO, %h_off : !tt.ptr<\(element)>, i32
                %dk_base = tt.addptr %DK, %h_off : !tt.ptr<\(element)>, i32
                %dv_base = tt.addptr %DV, %h_off : !tt.ptr<\(element)>, i32
                %l_base = tt.addptr %L, %lse_off : !tt.ptr<f32>, i32
                %d_base = tt.addptr %D, %lse_off : !tt.ptr<f32>, i32

                %range_m = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : \
            tensor<\(blockM)xi32>
                %range_n = tt.make_range {end = \(blockN) : i32, start = 0 : i32} : \
            tensor<\(blockN)xi32>
                %range_d = tt.make_range {end = \(headDim) : i32, start = 0 : i32} : \
            tensor<\(headDim)xi32>
                %n_splat = tt.splat %n_start : i32 -> tensor<\(blockN)xi32>
                %offs_n = arith.addi %n_splat, %range_n : tensor<\(blockN)xi32>
                %offs_n_r = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<\(blockN)xi32> -> \
            tensor<\(blockN)x1xi32>
                %offs_d_c = tt.expand_dims %range_d {axis = 0 : i32} : tensor<\(headDim)xi32> -> \
            tensor<1x\(headDim)xi32>

                %ss_n = tt.splat %stride_seq : i32 -> tensor<\(blockN)x1xi32>
                %kv_row = arith.muli %offs_n_r, %ss_n : tensor<\(blockN)x1xi32>
                %kv_row_b = tt.broadcast %kv_row : tensor<\(blockN)x1xi32> -> \
            tensor<\(blockN)x\(headDim)xi32>
                %d_col_n = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockN)x\(headDim)xi32>
                %kv_off = arith.addi %kv_row_b, %d_col_n : tensor<\(blockN)x\(headDim)xi32>
                %nctx_r = tt.splat %n_ctx : i32 -> tensor<\(blockN)x1xi32>
                %key_ok = arith.cmpi slt, %offs_n_r, %nctx_r : tensor<\(blockN)x1xi32>
                %kv_mask = tt.broadcast %key_ok : tensor<\(blockN)x1xi1> -> \
            tensor<\(blockN)x\(headDim)xi1>
                %k_p = tt.splat %k_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                %k_ptrs = tt.addptr %k_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                %k = tt.load %k_ptrs, %kv_mask, %zero_nd : \
            tensor<\(blockN)x\(headDim)x\(element)>
                %v_p = tt.splat %v_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                %v_ptrs = tt.addptr %v_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                %v = tt.load %v_ptrs, %kv_mask, %zero_nd : \
            tensor<\(blockN)x\(headDim)x\(element)>

                %m_round = arith.addi %n_ctx, %cbmm1_i32 : i32
                %trips = arith.divsi %m_round, %cbm_i32 : i32

                %res:2 = scf.for %i = %c0_i32 to %trips step %c1_i32
                    iter_args(%dk = %dk_init, %dv = %dv_init)
                    -> (tensor<\(blockN)x\(headDim)xf32>, tensor<\(blockN)x\(headDim)xf32>) : i32 {
                  %start_m = arith.muli %i, %cbm_i32 : i32
                  %m_splat = tt.splat %start_m : i32 -> tensor<\(blockM)xi32>
                  %offs_m = arith.addi %m_splat, %range_m : tensor<\(blockM)xi32>
                  %offs_m_r = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<\(blockM)xi32> -> \
            tensor<\(blockM)x1xi32>
                  %offs_m_c = tt.expand_dims %offs_m {axis = 0 : i32} : tensor<\(blockM)xi32> -> \
            tensor<1x\(blockM)xi32>

                  %ss_m = tt.splat %stride_seq : i32 -> tensor<\(blockM)x1xi32>
                  %q_row = arith.muli %offs_m_r, %ss_m : tensor<\(blockM)x1xi32>
                  %q_row_b = tt.broadcast %q_row : tensor<\(blockM)x1xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                  %d_col_m = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                  %q_off = arith.addi %q_row_b, %d_col_m : tensor<\(blockM)x\(headDim)xi32>
                  %nctx_m_r = tt.splat %n_ctx : i32 -> tensor<\(blockM)x1xi32>
                  %row_ok = arith.cmpi slt, %offs_m_r, %nctx_m_r : tensor<\(blockM)x1xi32>
                  %q_mask = tt.broadcast %row_ok : tensor<\(blockM)x1xi1> -> \
            tensor<\(blockM)x\(headDim)xi1>
                  %q_p = tt.splat %q_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                  %q_ptrs = tt.addptr %q_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                  %q = tt.load %q_ptrs, %q_mask, %zero_md : \
            tensor<\(blockM)x\(headDim)x\(element)>
                  %qt = tt.trans %q {order = array<i32: 1, 0>} : \
            tensor<\(blockM)x\(headDim)x\(element)> -> tensor<\(headDim)x\(blockM)x\(element)>
                  %do_p = tt.splat %do_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                  %do_ptrs = tt.addptr %do_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                  %do = tt.load %do_ptrs, %q_mask, %zero_md : \
            tensor<\(blockM)x\(headDim)x\(element)>
                  %dot_t = tt.trans %do {order = array<i32: 1, 0>} : \
            tensor<\(blockM)x\(headDim)x\(element)> -> tensor<\(headDim)x\(blockM)x\(element)>

                  %nctx_m = tt.splat %n_ctx : i32 -> tensor<\(blockM)xi32>
                  %m_ok = arith.cmpi slt, %offs_m, %nctx_m : tensor<\(blockM)xi32>
                  %l_p = tt.splat %l_base : !tt.ptr<f32> -> tensor<\(blockM)x!tt.ptr<f32>>
                  %l_ptrs = tt.addptr %l_p, %offs_m : tensor<\(blockM)x!tt.ptr<f32>>, \
            tensor<\(blockM)xi32>
                  %lse = tt.load %l_ptrs, %m_ok, %zero_m : tensor<\(blockM)xf32>
                  %dd_p = tt.splat %d_base : !tt.ptr<f32> -> tensor<\(blockM)x!tt.ptr<f32>>
                  %dd_ptrs = tt.addptr %dd_p, %offs_m : tensor<\(blockM)x!tt.ptr<f32>>, \
            tensor<\(blockM)xi32>
                  %delta = tt.load %dd_ptrs, %m_ok, %zero_m : tensor<\(blockM)xf32>
                  %lse_c = tt.expand_dims %lse {axis = 0 : i32} : tensor<\(blockM)xf32> -> \
            tensor<1x\(blockM)xf32>
                  %lse_b = tt.broadcast %lse_c : tensor<1x\(blockM)xf32> -> \
            tensor<\(blockN)x\(blockM)xf32>
                  %delta_c = tt.expand_dims %delta {axis = 0 : i32} : tensor<\(blockM)xf32> -> \
            tensor<1x\(blockM)xf32>
                  %delta_b = tt.broadcast %delta_c : tensor<1x\(blockM)xf32> -> \
            tensor<\(blockN)x\(blockM)xf32>

                  %qkt_raw = tt.dot %k, %qt, %zero_nm, inputPrecision = ieee : \
            tensor<\(blockN)x\(headDim)x\(element)> * tensor<\(headDim)x\(blockM)x\(element)> -> \
            tensor<\(blockN)x\(blockM)xf32>
                  %dpt_raw = tt.dot %v, %dot_t, %zero_nm, inputPrecision = ieee : \
            tensor<\(blockN)x\(headDim)x\(element)> * tensor<\(headDim)x\(blockM)x\(element)> -> \
            tensor<\(blockN)x\(blockM)xf32>

                  %scale_b = tt.splat %qk_scale : f32 -> tensor<\(blockN)x\(blockM)xf32>
                  %qkt_scaled = arith.mulf %qkt_raw, %scale_b : tensor<\(blockN)x\(blockM)xf32>
                  %nctx_c = tt.splat %n_ctx : i32 -> tensor<1x\(blockM)xi32>
                  %col_ok = arith.cmpi slt, %offs_m_c, %nctx_c : tensor<1x\(blockM)xi32>
                  %col_mask = tt.broadcast %col_ok : tensor<1x\(blockM)xi1> -> \
            tensor<\(blockN)x\(blockM)xi1>
                  %qkt = arith.select %col_mask, %qkt_scaled, %ninf_nm : \
            tensor<\(blockN)x\(blockM)xi1>, tensor<\(blockN)x\(blockM)xf32>
                  %qkt_centred = arith.subf %qkt, %lse_b : tensor<\(blockN)x\(blockM)xf32>
                  %pt = math.exp2 %qkt_centred : tensor<\(blockN)x\(blockM)xf32>
            \(ptOperand)
                  %dv_new = tt.dot %pt_op, %do, %dv, inputPrecision = ieee : \
            tensor<\(blockN)x\(blockM)x\(element)> * tensor<\(blockM)x\(headDim)x\(element)> -> \
            tensor<\(blockN)x\(headDim)xf32>
                  %dpt_centred = arith.subf %dpt_raw, %delta_b : tensor<\(blockN)x\(blockM)xf32>
                  %dst = arith.mulf %pt, %dpt_centred : tensor<\(blockN)x\(blockM)xf32>
            \(dstOperand)
                  %dk_new = tt.dot %dst_op, %q, %dk, inputPrecision = ieee : \
            tensor<\(blockN)x\(blockM)x\(element)> * tensor<\(blockM)x\(headDim)x\(element)> -> \
            tensor<\(blockN)x\(headDim)xf32>
                  scf.yield %dk_new, %dv_new : tensor<\(blockN)x\(headDim)xf32>, \
            tensor<\(blockN)x\(headDim)xf32>
                }

                %sm_b = tt.splat %sm_scale : f32 -> tensor<\(blockN)x\(headDim)xf32>
                %dk_res = arith.mulf %res#0, %sm_b : tensor<\(blockN)x\(headDim)xf32>
                %dk_scaled = arith.mulf %dk_res, %one_nd : tensor<\(blockN)x\(headDim)xf32>
                %dv_res = arith.mulf %res#1, %one_nd : tensor<\(blockN)x\(headDim)xf32>
            \(outputs)
                %dk_p = tt.splat %dk_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                %dk_ptrs = tt.addptr %dk_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                tt.store %dk_ptrs, %dk_e, %kv_mask : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                %dv_p = tt.splat %dv_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                %dv_ptrs = tt.addptr %dv_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                tt.store %dv_ptrs, %dv_e, %kv_mask : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                tt.return
              }
            }
            """
    }
}
