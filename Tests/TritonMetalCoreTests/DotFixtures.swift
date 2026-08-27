import Foundation
import TritonMetalBench

/// Triton IR fixtures for `tt.dot`: a single-tile product with no control flow,
/// and the blocked GEMM from Triton's matmul tutorial (masked loads, an `scf.for`
/// over K, pointers advanced inside the loop).
///
/// Written the way Triton's own `ttir` dump prints them, so the parser sees the
/// spellings it will actually be handed.
enum DotFixtures {

    /// `C = A @ B` for one tile, no masks and no K loop: A is `MxK`, B is `KxN`,
    /// both row-major and exactly the block size, so the whole product is one
    /// `tt.dot`. Element type `element` in, `accumulate` out.
    static func singleTile(m: Int, n: Int, k: Int, element: String = "f32") -> String {
        """
        module {
          tt.func public @dot_kernel(%arg0: !tt.ptr<\(element)>, %arg1: !tt.ptr<\(element)>,
                                     %arg2: !tt.ptr<f32>) {
            %cst = arith.constant dense<0.000000e+00> : tensor<\(m)x\(n)xf32>
            %ck = arith.constant \(k) : i32
            %cn = arith.constant \(n) : i32
            %0 = tt.make_range {end = \(m) : i32, start = 0 : i32} : tensor<\(m)xi32>
            %1 = tt.make_range {end = \(n) : i32, start = 0 : i32} : tensor<\(n)xi32>
            %2 = tt.make_range {end = \(k) : i32, start = 0 : i32} : tensor<\(k)xi32>
            %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<\(m)xi32> -> tensor<\(m)x1xi32>
            %4 = tt.expand_dims %2 {axis = 0 : i32} : tensor<\(k)xi32> -> tensor<1x\(k)xi32>
            %5 = tt.expand_dims %2 {axis = 1 : i32} : tensor<\(k)xi32> -> tensor<\(k)x1xi32>
            %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<\(n)xi32> -> tensor<1x\(n)xi32>

            // A pointers: row m, column k of an MxK row-major matrix.
            %7 = tt.splat %ck : i32 -> tensor<\(m)x1xi32>
            %8 = arith.muli %3, %7 : tensor<\(m)x1xi32>
            %9 = tt.broadcast %8 : tensor<\(m)x1xi32> -> tensor<\(m)x\(k)xi32>
            %10 = tt.broadcast %4 : tensor<1x\(k)xi32> -> tensor<\(m)x\(k)xi32>
            %11 = arith.addi %9, %10 : tensor<\(m)x\(k)xi32>
            %12 = tt.splat %arg0 : !tt.ptr<\(element)> -> tensor<\(m)x\(k)x!tt.ptr<\(element)>>
            %13 = tt.addptr %12, %11 : tensor<\(m)x\(k)x!tt.ptr<\(element)>>, tensor<\(m)x\(k)xi32>
            %14 = tt.load %13 : tensor<\(m)x\(k)x\(element)>

            // B pointers: row k, column n of a KxN row-major matrix.
            %15 = tt.splat %cn : i32 -> tensor<\(k)x1xi32>
            %16 = arith.muli %5, %15 : tensor<\(k)x1xi32>
            %17 = tt.broadcast %16 : tensor<\(k)x1xi32> -> tensor<\(k)x\(n)xi32>
            %18 = tt.broadcast %6 : tensor<1x\(n)xi32> -> tensor<\(k)x\(n)xi32>
            %19 = arith.addi %17, %18 : tensor<\(k)x\(n)xi32>
            %20 = tt.splat %arg1 : !tt.ptr<\(element)> -> tensor<\(k)x\(n)x!tt.ptr<\(element)>>
            %21 = tt.addptr %20, %19 : tensor<\(k)x\(n)x!tt.ptr<\(element)>>, tensor<\(k)x\(n)xi32>
            %22 = tt.load %21 : tensor<\(k)x\(n)x\(element)>

            %23 = tt.dot %14, %22, %cst : tensor<\(m)x\(k)x\(element)> * \
                tensor<\(k)x\(n)x\(element)> -> tensor<\(m)x\(n)xf32>

            // C pointers: row m, column n of an MxN row-major matrix.
            %24 = tt.splat %cn : i32 -> tensor<\(m)x1xi32>
            %25 = arith.muli %3, %24 : tensor<\(m)x1xi32>
            %26 = tt.broadcast %25 : tensor<\(m)x1xi32> -> tensor<\(m)x\(n)xi32>
            %27 = tt.broadcast %6 : tensor<1x\(n)xi32> -> tensor<\(m)x\(n)xi32>
            %28 = arith.addi %26, %27 : tensor<\(m)x\(n)xi32>
            %29 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<\(m)x\(n)x!tt.ptr<f32>>
            %30 = tt.addptr %29, %28 : tensor<\(m)x\(n)x!tt.ptr<f32>>, tensor<\(m)x\(n)xi32>
            tt.store %30, %23 : tensor<\(m)x\(n)x!tt.ptr<f32>>
            tt.return
          }
        }
        """
    }

    /// Triton's matmul tutorial, as `ttir`.
    ///
    /// The text itself lives in `TritonMetalBench` so that `swift test` and the
    /// `tmbench` executable lower byte-identical IR — the executable exists
    /// because XCTest is absent on a command-line-tools-only machine, which is
    /// where the quotable throughput numbers get measured.
    static func tutorial(
        blockM: Int, blockN: Int, blockK: Int, element: String = "f32",
        seed: String = "0.000000e+00"
    ) -> String {
        GEMMKernel.tutorial(
            blockM: blockM, blockN: blockN, blockK: blockK, element: element, seed: seed)
    }

    /// The matmul tutorial exactly as **Triton 3.7.1 prints it when `BLOCK_M ==
    /// BLOCK_N`** — copied from a `ttir` dump of `python/examples/matmul.py` at
    /// `(64, 64, 32)` on the build machine, and parameterised.
    ///
    /// The one thing that matters here and nowhere else: `tl.arange(0, BLOCK_M)`
    /// and `tl.arange(0, BLOCK_N)` are the same expression, so CSE emits **one**
    /// `tt.make_range` and both `offs_am` and `offs_bn` are `arith.addi`s over
    /// it. That single shared rank-1 value is then expanded at dimension 1 in one
    /// place and dimension 0 in the other, which is what used to unify the row
    /// axis with the column axis and get the kernel refused as a diagonal
    /// (`AxisCloning`).
    ///
    /// Specialisation has folded the unit strides to constants and dropped the
    /// masks, so `M`, `N` and `K` must be whole multiples of the block — which is
    /// exactly the shape the dump had.
    ///
    /// When `blockK == block` the contraction range is CSE'd into the *same*
    /// value as well, so one `tt.make_range` serves as the row index, the column
    /// index and the contraction index at once — which is what `(64, 64, 64)`
    /// looks like and why every expansion of a conflicted class needs its own
    /// copy, not just the ones at the second dimension.
    static func tutorialWithSharedRange(block: Int, blockK: Int) -> String {
        let kRange = blockK == block ? "offs_am_0" : "offs_k"
        return """
        module {
          tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>,
                                        %c_ptr: !tt.ptr<f32>, %M: i32, %N: i32, %K: i32,
                                        %stride_am: i32, %stride_bk: i32, %stride_cm: i32) {
            %ckm1_i32 = arith.constant \(blockK - 1) : i32
            %accumulator = arith.constant dense<0.000000e+00> : tensor<\(block)x\(block)xf32>
            %c1_i32 = arith.constant 1 : i32
            %c0_i32 = arith.constant 0 : i32
            %cst = arith.constant dense<\(blockK)> : tensor<\(block)x\(blockK)xi32>
            %ck_i32 = arith.constant \(blockK) : i32
            %cb_i32 = arith.constant \(block) : i32
            %pid_m = tt.get_program_id x : i32
            %pid_n = tt.get_program_id y : i32
            %offs_am = arith.muli %pid_m, %cb_i32 : i32
            %offs_am_0 = tt.make_range {end = \(block) : i32, start = 0 : i32} : \
        tensor<\(block)xi32>
            %offs_am_1 = tt.splat %offs_am : i32 -> tensor<\(block)xi32>
            %offs_am_2 = arith.addi %offs_am_1, %offs_am_0 : tensor<\(block)xi32>
            %offs_bn = arith.muli %pid_n, %cb_i32 : i32
            %offs_bn_3 = tt.splat %offs_bn : i32 -> tensor<\(block)xi32>
            %offs_bn_4 = arith.addi %offs_bn_3, %offs_am_0 : tensor<\(block)xi32>
        \(blockK == block ? "" : "    %offs_k = tt.make_range {end = \(blockK) : i32, start = 0 : i32} : tensor<\(blockK)xi32>")
            %a_ptrs = tt.expand_dims %offs_am_2 {axis = 1 : i32} : tensor<\(block)xi32> -> \
        tensor<\(block)x1xi32>
            %a_ptrs_5 = tt.splat %stride_am : i32 -> tensor<\(block)x1xi32>
            %a_ptrs_6 = arith.muli %a_ptrs, %a_ptrs_5 : tensor<\(block)x1xi32>
            %a_ptrs_7 = tt.expand_dims %\(kRange) {axis = 0 : i32} : tensor<\(blockK)xi32> -> \
        tensor<1x\(blockK)xi32>
            %a_ptrs_8 = tt.broadcast %a_ptrs_6 : tensor<\(block)x1xi32> -> \
        tensor<\(block)x\(blockK)xi32>
            %a_ptrs_9 = tt.broadcast %a_ptrs_7 : tensor<1x\(blockK)xi32> -> \
        tensor<\(block)x\(blockK)xi32>
            %a_ptrs_10 = arith.addi %a_ptrs_8, %a_ptrs_9 : tensor<\(block)x\(blockK)xi32>
            %a_ptrs_11 = tt.splat %a_ptr : !tt.ptr<f32> -> \
        tensor<\(block)x\(blockK)x!tt.ptr<f32>>
            %a_ptrs_12 = tt.addptr %a_ptrs_11, %a_ptrs_10 : \
        tensor<\(block)x\(blockK)x!tt.ptr<f32>>, tensor<\(block)x\(blockK)xi32>
            %b_ptrs = tt.expand_dims %\(kRange) {axis = 1 : i32} : tensor<\(blockK)xi32> -> \
        tensor<\(blockK)x1xi32>
            %b_ptrs_13 = tt.splat %stride_bk : i32 -> tensor<\(blockK)x1xi32>
            %b_ptrs_14 = arith.muli %b_ptrs, %b_ptrs_13 : tensor<\(blockK)x1xi32>
            %b_ptrs_15 = tt.expand_dims %offs_bn_4 {axis = 0 : i32} : tensor<\(block)xi32> -> \
        tensor<1x\(block)xi32>
            %b_ptrs_16 = tt.broadcast %b_ptrs_14 : tensor<\(blockK)x1xi32> -> \
        tensor<\(blockK)x\(block)xi32>
            %b_ptrs_17 = tt.broadcast %b_ptrs_15 : tensor<1x\(block)xi32> -> \
        tensor<\(blockK)x\(block)xi32>
            %b_ptrs_18 = arith.addi %b_ptrs_16, %b_ptrs_17 : tensor<\(blockK)x\(block)xi32>
            %b_ptrs_19 = tt.splat %b_ptr : !tt.ptr<f32> -> \
        tensor<\(blockK)x\(block)x!tt.ptr<f32>>
            %b_ptrs_20 = tt.addptr %b_ptrs_19, %b_ptrs_18 : \
        tensor<\(blockK)x\(block)x!tt.ptr<f32>>, tensor<\(blockK)x\(block)xi32>
            %0 = arith.addi %K, %ckm1_i32 : i32
            %1 = arith.divsi %0, %ck_i32 : i32
            %accumulator_21:3 = scf.for %k = %c0_i32 to %1 step %c1_i32 \
        iter_args(%a_ptrs_28 = %a_ptrs_12, %b_ptrs_29 = %b_ptrs_20, \
        %accumulator_30 = %accumulator) -> (tensor<\(block)x\(blockK)x!tt.ptr<f32>>, \
        tensor<\(blockK)x\(block)x!tt.ptr<f32>>, tensor<\(block)x\(block)xf32>)  : i32 {
              %a = tt.load %a_ptrs_28 : tensor<\(block)x\(blockK)x!tt.ptr<f32>>
              %b = tt.load %b_ptrs_29 : tensor<\(blockK)x\(block)x!tt.ptr<f32>>
              %accumulator_31 = tt.dot %a, %b, %accumulator_30 : \
        tensor<\(block)x\(blockK)xf32> * tensor<\(blockK)x\(block)xf32> -> \
        tensor<\(block)x\(block)xf32>
              %a_ptrs_32 = tt.addptr %a_ptrs_28, %cst : \
        tensor<\(block)x\(blockK)x!tt.ptr<f32>>, tensor<\(block)x\(blockK)xi32>
              %b_ptrs_33 = arith.muli %stride_bk, %ck_i32 : i32
              %b_ptrs_34 = tt.splat %b_ptrs_33 : i32 -> tensor<\(blockK)x\(block)xi32>
              %b_ptrs_35 = tt.addptr %b_ptrs_29, %b_ptrs_34 : \
        tensor<\(blockK)x\(block)x!tt.ptr<f32>>, tensor<\(blockK)x\(block)xi32>
              scf.yield %a_ptrs_32, %b_ptrs_35, %accumulator_31 : \
        tensor<\(block)x\(blockK)x!tt.ptr<f32>>, tensor<\(blockK)x\(block)x!tt.ptr<f32>>, \
        tensor<\(block)x\(block)xf32>
            }
            %c_ptrs = tt.splat %stride_cm : i32 -> tensor<\(block)x1xi32>
            %c_ptrs_22 = arith.muli %c_ptrs, %a_ptrs : tensor<\(block)x1xi32>
            %c_ptrs_23 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<\(block)x1x!tt.ptr<f32>>
            %c_ptrs_24 = tt.addptr %c_ptrs_23, %c_ptrs_22 : \
        tensor<\(block)x1x!tt.ptr<f32>>, tensor<\(block)x1xi32>
            %c_ptrs_25 = tt.broadcast %c_ptrs_24 : tensor<\(block)x1x!tt.ptr<f32>> -> \
        tensor<\(block)x\(block)x!tt.ptr<f32>>
            %c_ptrs_26 = tt.broadcast %b_ptrs_15 : tensor<1x\(block)xi32> -> \
        tensor<\(block)x\(block)xi32>
            %c_ptrs_27 = tt.addptr %c_ptrs_25, %c_ptrs_26 : \
        tensor<\(block)x\(block)x!tt.ptr<f32>>, tensor<\(block)x\(block)xi32>
            tt.store %c_ptrs_27, %accumulator_21#2 : tensor<\(block)x\(block)x!tt.ptr<f32>>
            tt.return
          }
        }
        """
    }
}
