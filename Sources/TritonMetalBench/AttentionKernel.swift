import Foundation

/// Triton IR for the FlashAttention-2 forward pass, in the shape of Triton's own
/// tutorial: an online softmax over blocks of keys, `Q @ K^T` and `P @ V` fused
/// into one kernel so the `BLOCK_M x N_CTX` score matrix never reaches memory.
///
/// Like `GEMMKernel`, the text lives in a library target so that `swift test` and
/// the `tmbench` executable lower byte-identical IR — the executable is what runs
/// on a machine with only the command-line tools, which is where the quotable
/// numbers get measured.
///
/// Written the way Triton prints `ttir`: pointer arithmetic rather than block
/// pointers, `tt.trans` for `K^T`, `math.exp2` with the scale folded into
/// `log2(e)`, and the generic spelling of `tt.reduce`.
///
/// The tensors it carries across the key loop are what made this the milestone
/// kernel: `m_i` and `l_i` are per-row vectors and `acc` is a `BLOCK_M x HEAD_DIM`
/// tile, none of which is a scalar, and `acc` is rescaled by a per-row correction
/// factor *between* the two dots (docs/ARCHITECTURE.md §Cross-lane regions).
public enum AttentionKernel {

    /// One program per `BLOCK_M` block of queries (`program_id x`) per
    /// batch-and-head (`program_id y`).
    ///
    /// Arguments, in buffer order: `Q`, `K`, `V`, `Out`, `sm_scale`,
    /// `stride_head` (elements between one head's `Q` and the next),
    /// `stride_seq` (elements between one row and the next) and `N_CTX`. The head
    /// dimension is contiguous, which is what a `[B, H, S, D]` tensor looks like.
    ///
    /// `element` is the input/output type; the accumulator, the softmax
    /// statistics and the score matrix are always f32, which is the
    /// f16-in/f32-accumulate shape that matters for ML.
    public static func forward(
        blockM: Int, blockN: Int, headDim: Int, element: String = "f32"
    ) -> String {
        let isHalf = element == "f16"
        // P feeds the second dot, whose two operands must share an element type.
        let probability = isHalf
            ? """
                      %p_op = arith.truncf %p : tensor<\(blockM)x\(blockN)xf32> to \
                tensor<\(blockM)x\(blockN)xf16>
            """
            : "          %p_op = arith.mulf %p, %one_mn : tensor<\(blockM)x\(blockN)xf32>"
        let output = isHalf
            ? """
                  %out_e = arith.truncf %out : tensor<\(blockM)x\(headDim)xf32> to \
                tensor<\(blockM)x\(headDim)xf16>
            """
            : "      %out_e = arith.mulf %out, %one_md : tensor<\(blockM)x\(headDim)xf32>"

        return """
            module {
              tt.func public @attn_fwd(
                  %Q: !tt.ptr<\(element)>, %K: !tt.ptr<\(element)>, %V: !tt.ptr<\(element)>,
                  %O: !tt.ptr<\(element)>, %sm_scale: f32,
                  %stride_head: i32, %stride_seq: i32, %n_ctx: i32) {
                %log2e = arith.constant 1.44269504088896340736 : f32
                %qk_scale = arith.mulf %sm_scale, %log2e : f32
                %c0_i32 = arith.constant 0 : i32
                %c1_i32 = arith.constant 1 : i32
                %cbm_i32 = arith.constant \(blockM) : i32
                %cbn_i32 = arith.constant \(blockN) : i32
                %cbnm1_i32 = arith.constant \(blockN - 1) : i32
                %m_init = arith.constant dense<0xFF800000> : tensor<\(blockM)xf32>
                %l_init = arith.constant dense<0.000000e+00> : tensor<\(blockM)xf32>
                %acc_init = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)xf32>
                %zero_mn = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockN)xf32>
                %ninf_mn = arith.constant dense<0xFF800000> : tensor<\(blockM)x\(blockN)xf32>
                %one_mn = arith.constant dense<1.000000e+00> : tensor<\(blockM)x\(blockN)xf32>
                %one_md = arith.constant dense<1.000000e+00> : tensor<\(blockM)x\(headDim)xf32>
                %zero_md = arith.constant dense<0.000000e+00> : \
            tensor<\(blockM)x\(headDim)x\(element)>
                %zero_nd = arith.constant dense<0.000000e+00> : \
            tensor<\(blockN)x\(headDim)x\(element)>

                %pid_m = tt.get_program_id x : i32
                %pid_h = tt.get_program_id y : i32
                %m_start = arith.muli %pid_m, %cbm_i32 : i32
                %h_off = arith.muli %pid_h, %stride_head : i32
                %q_base = tt.addptr %Q, %h_off : !tt.ptr<\(element)>, i32
                %k_base = tt.addptr %K, %h_off : !tt.ptr<\(element)>, i32
                %v_base = tt.addptr %V, %h_off : !tt.ptr<\(element)>, i32
                %o_base = tt.addptr %O, %h_off : !tt.ptr<\(element)>, i32

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

                // q_ptrs = Q + off_h + offs_m[:, None] * stride_seq + offs_d[None, :]
                %ss_m = tt.splat %stride_seq : i32 -> tensor<\(blockM)x1xi32>
                %q_row = arith.muli %offs_m_r, %ss_m : tensor<\(blockM)x1xi32>
                %q_row_b = tt.broadcast %q_row : tensor<\(blockM)x1xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %d_col_m = tt.broadcast %offs_d_c : tensor<1x\(headDim)xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %q_off = arith.addi %q_row_b, %d_col_m : tensor<\(blockM)x\(headDim)xi32>
                %q_p = tt.splat %q_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %q_ptrs = tt.addptr %q_p, %q_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                %nctx_m = tt.splat %n_ctx : i32 -> tensor<\(blockM)x1xi32>
                %row_ok = arith.cmpi slt, %offs_m_r, %nctx_m : tensor<\(blockM)x1xi32>
                %q_mask = tt.broadcast %row_ok : tensor<\(blockM)x1xi1> -> \
            tensor<\(blockM)x\(headDim)xi1>
                %q = tt.load %q_ptrs, %q_mask, %zero_md : tensor<\(blockM)x\(headDim)x\(element)>

                %n_round = arith.addi %n_ctx, %cbnm1_i32 : i32
                %trips = arith.divsi %n_round, %cbn_i32 : i32

                %res:3 = scf.for %j = %c0_i32 to %trips step %c1_i32
                    iter_args(%m_i = %m_init, %l_i = %l_init, %acc = %acc_init)
                    -> (tensor<\(blockM)xf32>, tensor<\(blockM)xf32>,
                        tensor<\(blockM)x\(headDim)xf32>) : i32 {
                  %start_n = arith.muli %j, %cbn_i32 : i32
                  %n_splat = tt.splat %start_n : i32 -> tensor<\(blockN)xi32>
                  %offs_n = arith.addi %n_splat, %range_n : tensor<\(blockN)xi32>
                  %offs_n_r = tt.expand_dims %offs_n {axis = 1 : i32} : tensor<\(blockN)xi32> -> \
            tensor<\(blockN)x1xi32>
                  %offs_n_c = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<\(blockN)xi32> -> \
            tensor<1x\(blockN)xi32>

                  // K and V rows: (BLOCK_N, HEAD_DIM), sharing one mask.
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

                  %qk_raw = tt.dot %q, %kt, %zero_mn, inputPrecision = ieee : \
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

                  %row_max = "tt.reduce"(%qk) <{axis = 1 : i32}> ({
                  ^bb0(%ma: f32, %mb: f32):
                    %mm = arith.maxnumf %ma, %mb : f32
                    tt.reduce.return %mm : f32
                  }) : (tensor<\(blockM)x\(blockN)xf32>) -> tensor<\(blockM)xf32>
                  %m_ij = arith.maxnumf %m_i, %row_max : tensor<\(blockM)xf32>
                  %m_ij_r = tt.expand_dims %m_ij {axis = 1 : i32} : tensor<\(blockM)xf32> -> \
            tensor<\(blockM)x1xf32>
                  %m_ij_b = tt.broadcast %m_ij_r : tensor<\(blockM)x1xf32> -> \
            tensor<\(blockM)x\(blockN)xf32>
                  %qk_centred = arith.subf %qk, %m_ij_b : tensor<\(blockM)x\(blockN)xf32>
                  %p = math.exp2 %qk_centred : tensor<\(blockM)x\(blockN)xf32>
                  %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
                  ^bb0(%sa: f32, %sb: f32):
                    %ss = arith.addf %sa, %sb : f32
                    tt.reduce.return %ss : f32
                  }) : (tensor<\(blockM)x\(blockN)xf32>) -> tensor<\(blockM)xf32>

                  %m_delta = arith.subf %m_i, %m_ij : tensor<\(blockM)xf32>
                  %alpha = math.exp2 %m_delta : tensor<\(blockM)xf32>
                  %l_scaled = arith.mulf %l_i, %alpha : tensor<\(blockM)xf32>
                  %l_new = arith.addf %l_scaled, %l_ij : tensor<\(blockM)xf32>
                  %alpha_r = tt.expand_dims %alpha {axis = 1 : i32} : tensor<\(blockM)xf32> -> \
            tensor<\(blockM)x1xf32>
                  %alpha_b = tt.broadcast %alpha_r : tensor<\(blockM)x1xf32> -> \
            tensor<\(blockM)x\(headDim)xf32>
                  %acc_scaled = arith.mulf %acc, %alpha_b : tensor<\(blockM)x\(headDim)xf32>

                  %v_p = tt.splat %v_base : !tt.ptr<\(element)> -> \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>
                  %v_ptrs = tt.addptr %v_p, %kv_off : \
            tensor<\(blockN)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockN)x\(headDim)xi32>
                  %v = tt.load %v_ptrs, %kv_mask, %zero_nd : \
            tensor<\(blockN)x\(headDim)x\(element)>
            \(probability)
                  %acc_new = tt.dot %p_op, %v, %acc_scaled, inputPrecision = ieee : \
            tensor<\(blockM)x\(blockN)x\(element)> * tensor<\(blockN)x\(headDim)x\(element)> -> \
            tensor<\(blockM)x\(headDim)xf32>
                  scf.yield %m_ij, %l_new, %acc_new : tensor<\(blockM)xf32>, \
            tensor<\(blockM)xf32>, tensor<\(blockM)x\(headDim)xf32>
                }

                %l_r = tt.expand_dims %res#1 {axis = 1 : i32} : tensor<\(blockM)xf32> -> \
            tensor<\(blockM)x1xf32>
                %l_b = tt.broadcast %l_r : tensor<\(blockM)x1xf32> -> \
            tensor<\(blockM)x\(headDim)xf32>
                %out = arith.divf %res#2, %l_b : tensor<\(blockM)x\(headDim)xf32>
            \(output)
                %o_row = arith.muli %offs_m_r, %ss_m : tensor<\(blockM)x1xi32>
                %o_row_b = tt.broadcast %o_row : tensor<\(blockM)x1xi32> -> \
            tensor<\(blockM)x\(headDim)xi32>
                %o_off = arith.addi %o_row_b, %d_col_m : tensor<\(blockM)x\(headDim)xi32>
                %o_p = tt.splat %o_base : !tt.ptr<\(element)> -> \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                %o_ptrs = tt.addptr %o_p, %o_off : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>, tensor<\(blockM)x\(headDim)xi32>
                tt.store %o_ptrs, %out_e, %q_mask : \
            tensor<\(blockM)x\(headDim)x!tt.ptr<\(element)>>
                tt.return
              }
            }
            """
    }
}
