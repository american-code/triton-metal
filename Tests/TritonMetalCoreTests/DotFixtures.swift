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
}
