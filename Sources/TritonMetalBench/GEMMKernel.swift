import Foundation

/// Triton IR for the blocked GEMM from Triton's matmul tutorial.
///
/// This lives in a library target rather than in the test bundle so that both
/// `swift test` and the `tmbench` executable lower exactly the same IR. The
/// executable matters because XCTest does not exist on a machine with only the
/// command-line tools installed, which is where the throughput numbers that are
/// worth quoting get measured (docs/ARCHITECTURE.md §Matmul throughput).
///
/// Written the way Triton's own `ttir` dump prints it, so the parser sees the
/// spellings it will actually be handed.
public enum GEMMKernel {

    /// One program per `BLOCK_M x BLOCK_N` output tile; an `scf.for` walks K in
    /// `BLOCK_K` steps carrying the accumulator and both operand pointers; masks
    /// clip the ragged edges in all three dimensions. `inputPrecision` is spelled
    /// the way Triton 3.x prints it.
    ///
    /// The f16 form loads half operands and accumulates in f32 — the shape that
    /// matters for ML — and truncates on the way out.
    /// `seed` is the accumulator's initial value. Triton always writes `0.0`, and
    /// the emitter has a fast path for exactly that (the fragments start zero in
    /// registers instead of being staged through the tile); the parameter exists
    /// so the slow path stays covered.
    public static func tutorial(
        blockM: Int, blockN: Int, blockK: Int, element: String = "f32",
        seed: String = "0.000000e+00"
    ) -> String {
        let isHalf = element == "f16"
        let store = isHalf
            ? """
                  %52 = arith.truncf %51 : tensor<\(blockM)x\(blockN)xf32> to \
                tensor<\(blockM)x\(blockN)xf16>
            """
            : "      %52 = arith.mulf %51, %cone : tensor<\(blockM)x\(blockN)xf32>"

        return """
            module {
              tt.func public @matmul_kernel(
                  %arg0: !tt.ptr<\(element)>, %arg1: !tt.ptr<\(element)>, %arg2: !tt.ptr<\(element)>,
                  %arg3: i32, %arg4: i32, %arg5: i32,
                  %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
                %cst = arith.constant dense<\(seed)> : tensor<\(blockM)x\(blockN)xf32>
                %cone = arith.constant dense<1.000000e+00> : tensor<\(blockM)x\(blockN)xf32>
                %zeroA = arith.constant dense<0.000000e+00> : tensor<\(blockM)x\(blockK)x\(element)>
                %zeroB = arith.constant dense<0.000000e+00> : tensor<\(blockK)x\(blockN)x\(element)>
                %c0_i32 = arith.constant 0 : i32
                %c1_i32 = arith.constant 1 : i32
                %cm_i32 = arith.constant \(blockM) : i32
                %cn_i32 = arith.constant \(blockN) : i32
                %ck_i32 = arith.constant \(blockK) : i32
                %ckm1_i32 = arith.constant \(blockK - 1) : i32

                %0 = tt.get_program_id x : i32
                %1 = tt.get_program_id y : i32
                %2 = arith.muli %0, %cm_i32 : i32
                %3 = arith.muli %1, %cn_i32 : i32
                %4 = tt.make_range {end = \(blockM) : i32, start = 0 : i32} : tensor<\(blockM)xi32>
                %5 = tt.make_range {end = \(blockN) : i32, start = 0 : i32} : tensor<\(blockN)xi32>
                %6 = tt.make_range {end = \(blockK) : i32, start = 0 : i32} : tensor<\(blockK)xi32>
                %7 = tt.splat %2 : i32 -> tensor<\(blockM)xi32>
                %8 = arith.addi %7, %4 : tensor<\(blockM)xi32>
                %9 = tt.splat %3 : i32 -> tensor<\(blockN)xi32>
                %10 = arith.addi %9, %5 : tensor<\(blockN)xi32>

                %11 = tt.expand_dims %8 {axis = 1 : i32} : tensor<\(blockM)xi32> -> \
                tensor<\(blockM)x1xi32>
                %12 = tt.expand_dims %10 {axis = 0 : i32} : tensor<\(blockN)xi32> -> \
                tensor<1x\(blockN)xi32>
                %13 = tt.expand_dims %6 {axis = 0 : i32} : tensor<\(blockK)xi32> -> \
                tensor<1x\(blockK)xi32>
                %14 = tt.expand_dims %6 {axis = 1 : i32} : tensor<\(blockK)xi32> -> \
                tensor<\(blockK)x1xi32>

                // a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                %15 = tt.splat %arg6 : i32 -> tensor<\(blockM)x1xi32>
                %16 = arith.muli %11, %15 : tensor<\(blockM)x1xi32>
                %17 = tt.splat %arg7 : i32 -> tensor<1x\(blockK)xi32>
                %18 = arith.muli %13, %17 : tensor<1x\(blockK)xi32>
                %19 = tt.broadcast %16 : tensor<\(blockM)x1xi32> -> tensor<\(blockM)x\(blockK)xi32>
                %20 = tt.broadcast %18 : tensor<1x\(blockK)xi32> -> tensor<\(blockM)x\(blockK)xi32>
                %21 = arith.addi %19, %20 : tensor<\(blockM)x\(blockK)xi32>
                %22 = tt.splat %arg0 : !tt.ptr<\(element)> -> \
                tensor<\(blockM)x\(blockK)x!tt.ptr<\(element)>>
                %23 = tt.addptr %22, %21 : tensor<\(blockM)x\(blockK)x!tt.ptr<\(element)>>, \
                tensor<\(blockM)x\(blockK)xi32>

                // b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                %24 = tt.splat %arg8 : i32 -> tensor<\(blockK)x1xi32>
                %25 = arith.muli %14, %24 : tensor<\(blockK)x1xi32>
                %26 = tt.splat %arg9 : i32 -> tensor<1x\(blockN)xi32>
                %27 = arith.muli %12, %26 : tensor<1x\(blockN)xi32>
                %28 = tt.broadcast %25 : tensor<\(blockK)x1xi32> -> tensor<\(blockK)x\(blockN)xi32>
                %29 = tt.broadcast %27 : tensor<1x\(blockN)xi32> -> tensor<\(blockK)x\(blockN)xi32>
                %30 = arith.addi %28, %29 : tensor<\(blockK)x\(blockN)xi32>
                %31 = tt.splat %arg1 : !tt.ptr<\(element)> -> \
                tensor<\(blockK)x\(blockN)x!tt.ptr<\(element)>>
                %32 = tt.addptr %31, %30 : tensor<\(blockK)x\(blockN)x!tt.ptr<\(element)>>, \
                tensor<\(blockK)x\(blockN)xi32>

                // Row/column masks, loop invariant.
                %33 = tt.splat %arg3 : i32 -> tensor<\(blockM)x1xi32>
                %34 = arith.cmpi slt, %11, %33 : tensor<\(blockM)x1xi32>
                %35 = tt.splat %arg4 : i32 -> tensor<1x\(blockN)xi32>
                %36 = arith.cmpi slt, %12, %35 : tensor<1x\(blockN)xi32>

                // Loop trip count and the per-step pointer advances.
                %37 = arith.addi %arg5, %ckm1_i32 : i32
                %38 = arith.divsi %37, %ck_i32 : i32
                %39 = arith.muli %arg7, %ck_i32 : i32
                %40 = arith.muli %arg8, %ck_i32 : i32

                %41:3 = scf.for %arg12 = %c0_i32 to %38 step %c1_i32
                    iter_args(%arg13 = %cst, %arg14 = %23, %arg15 = %32)
                    -> (tensor<\(blockM)x\(blockN)xf32>,
                        tensor<\(blockM)x\(blockK)x!tt.ptr<\(element)>>,
                        tensor<\(blockK)x\(blockN)x!tt.ptr<\(element)>>) : i32 {
                  %60 = arith.muli %arg12, %ck_i32 : i32
                  %61 = arith.subi %arg5, %60 : i32
                  %62 = tt.splat %61 : i32 -> tensor<1x\(blockK)xi32>
                  %63 = arith.cmpi slt, %13, %62 : tensor<1x\(blockK)xi32>
                  %64 = tt.broadcast %63 : tensor<1x\(blockK)xi1> -> tensor<\(blockM)x\(blockK)xi1>
                  %65 = tt.broadcast %34 : tensor<\(blockM)x1xi1> -> tensor<\(blockM)x\(blockK)xi1>
                  %66 = arith.andi %65, %64 : tensor<\(blockM)x\(blockK)xi1>
                  %67 = tt.load %arg14, %66, %zeroA : tensor<\(blockM)x\(blockK)x\(element)>

                  %68 = tt.splat %61 : i32 -> tensor<\(blockK)x1xi32>
                  %69 = arith.cmpi slt, %14, %68 : tensor<\(blockK)x1xi32>
                  %70 = tt.broadcast %69 : tensor<\(blockK)x1xi1> -> tensor<\(blockK)x\(blockN)xi1>
                  %71 = tt.broadcast %36 : tensor<1x\(blockN)xi1> -> tensor<\(blockK)x\(blockN)xi1>
                  %72 = arith.andi %70, %71 : tensor<\(blockK)x\(blockN)xi1>
                  %73 = tt.load %arg15, %72, %zeroB : tensor<\(blockK)x\(blockN)x\(element)>

                  %74 = tt.dot %67, %73, %arg13, inputPrecision = tf32 : \
                tensor<\(blockM)x\(blockK)x\(element)> * tensor<\(blockK)x\(blockN)x\(element)> -> \
                tensor<\(blockM)x\(blockN)xf32>

                  %75 = tt.splat %39 : i32 -> tensor<\(blockM)x\(blockK)xi32>
                  %76 = tt.addptr %arg14, %75 : tensor<\(blockM)x\(blockK)x!tt.ptr<\(element)>>, \
                tensor<\(blockM)x\(blockK)xi32>
                  %77 = tt.splat %40 : i32 -> tensor<\(blockK)x\(blockN)xi32>
                  %78 = tt.addptr %arg15, %77 : tensor<\(blockK)x\(blockN)x!tt.ptr<\(element)>>, \
                tensor<\(blockK)x\(blockN)xi32>
                  scf.yield %74, %76, %78 : tensor<\(blockM)x\(blockN)xf32>, \
                tensor<\(blockM)x\(blockK)x!tt.ptr<\(element)>>, \
                tensor<\(blockK)x\(blockN)x!tt.ptr<\(element)>>
                }
                %51 = arith.mulf %41#0, %cone : tensor<\(blockM)x\(blockN)xf32>
            \(store)

                // c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                %53 = tt.splat %arg10 : i32 -> tensor<\(blockM)x1xi32>
                %54 = arith.muli %11, %53 : tensor<\(blockM)x1xi32>
                %55 = tt.splat %arg11 : i32 -> tensor<1x\(blockN)xi32>
                %56 = arith.muli %12, %55 : tensor<1x\(blockN)xi32>
                %57 = tt.broadcast %54 : tensor<\(blockM)x1xi32> -> tensor<\(blockM)x\(blockN)xi32>
                %58 = tt.broadcast %56 : tensor<1x\(blockN)xi32> -> tensor<\(blockM)x\(blockN)xi32>
                %59 = arith.addi %57, %58 : tensor<\(blockM)x\(blockN)xi32>
                %80 = tt.splat %arg2 : !tt.ptr<\(element)> -> \
                tensor<\(blockM)x\(blockN)x!tt.ptr<\(element)>>
                %81 = tt.addptr %80, %59 : tensor<\(blockM)x\(blockN)x!tt.ptr<\(element)>>, \
                tensor<\(blockM)x\(blockN)xi32>
                %82 = tt.broadcast %34 : tensor<\(blockM)x1xi1> -> tensor<\(blockM)x\(blockN)xi1>
                %83 = tt.broadcast %36 : tensor<1x\(blockN)xi1> -> tensor<\(blockM)x\(blockN)xi1>
                %84 = arith.andi %82, %83 : tensor<\(blockM)x\(blockN)xi1>
                tt.store %81, %52, %84 : tensor<\(blockM)x\(blockN)x!tt.ptr<\(element)>>
                tt.return
              }
            }
            """
    }
}
