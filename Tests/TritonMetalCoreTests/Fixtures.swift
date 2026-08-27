import Foundation

/// Triton IR fixtures, written the way Triton's own `ttir` dump prints them
/// (pretty op syntax, argument attributes, explicit trailing types).
enum IRFixtures {
    /// `out[i] = in[i]` with a bounds mask — the shape of `tl.load`/`tl.store`
    /// in Triton's tutorial kernels.
    static let copy = """
        module {
          tt.func public @copy_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg2: i32) attributes {noinline = false} {
            %c256_i32 = arith.constant 256 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c256_i32 : i32
            %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
            %3 = tt.splat %1 : i32 -> tensor<256xi32>
            %4 = arith.addi %3, %2 : tensor<256xi32>
            %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
            %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
            %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
            %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
            tt.store %11, %9, %6 : tensor<256x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// Triton's vector-add tutorial kernel, verbatim in shape.
    static let vectorAdd = """
        module {
          tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                     %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                     %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                     %arg3: i32 {tt.divisibility = 16 : i32})
              attributes {noinline = false} {
            %c1024_i32 = arith.constant 1024 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c1024_i32 : i32
            %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
            %3 = tt.splat %1 : i32 -> tensor<1024xi32>
            %4 = arith.addi %3, %2 : tensor<1024xi32>
            %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
            %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
            %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
            %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
            %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
            %13 = arith.addf %9, %12 : tensor<1024xf32>
            %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
            %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
            tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// Elementwise multiply, with `other` on the masked loads and the older
    /// `{cache = ..., evict = ...}` attribute spelling on `tt.load`.
    static let vectorMul = """
        module {
          tt.func public @mul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                     %arg2: !tt.ptr<f32>, %arg3: i32) {
            %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
            %c128_i32 = arith.constant 128 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c128_i32 : i32
            %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
            %3 = tt.splat %1 : i32 -> tensor<128xi32>
            %4 = arith.addi %3, %2 : tensor<128xi32>
            %5 = tt.splat %arg3 : i32 -> tensor<128xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %9 = tt.load %8, %6, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32>
            %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
            %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            %12 = tt.load %11, %6, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32>
            %13 = arith.mulf %9, %12 : tensor<128xf32>
            %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
            %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
            tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// `out[i] = in[i] * scale + bias` — exercises f32 scalar kernel arguments,
    /// the `{axis = 0}` spelling of `tt.get_program_id`, and `tt.get_num_programs`.
    static let scaleBias = """
        module attributes {"triton_gpu.num-warps" = 4 : i32} {
          tt.func public @scale_bias_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                            %arg2: f32, %arg3: f32, %arg4: i32) {
            %c64_i32 = arith.constant 64 : i32
            %0 = tt.get_program_id {axis = 0 : i32} : i32
            %np = tt.get_num_programs {axis = 0 : i32} : i32
            %unused = arith.muli %np, %c64_i32 : i32
            %1 = arith.muli %0, %c64_i32 : i32
            %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %3 = tt.splat %1 : i32 -> tensor<64xi32>
            %4 = arith.addi %3, %2 : tensor<64xi32>
            %5 = tt.splat %arg4 : i32 -> tensor<64xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
            %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
            %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            %9 = tt.load %8, %6 : tensor<64x!tt.ptr<f32>>
            %10 = tt.splat %arg2 : f32 -> tensor<64xf32>
            %11 = arith.mulf %9, %10 : tensor<64xf32>
            %12 = tt.splat %arg3 : f32 -> tensor<64xf32>
            %13 = arith.addf %11, %12 : tensor<64xf32>
            %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
            %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
            tt.store %15, %13, %6 : tensor<64x!tt.ptr<f32>>
            tt.return
          }
        }
        """

    /// Integer kernel: `out[i] = (a[i] + b[i]) * 3`, i32 element type throughout.
    static let integerAdd = """
        module {
          tt.func public @iadd_kernel(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>,
                                      %arg2: !tt.ptr<i32>, %arg3: i32) {
            %c32_i32 = arith.constant 32 : i32
            %c3 = arith.constant 3 : i32
            %0 = tt.get_program_id x : i32
            %1 = arith.muli %0, %c32_i32 : i32
            %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
            %3 = tt.splat %1 : i32 -> tensor<32xi32>
            %4 = arith.addi %3, %2 : tensor<32xi32>
            %5 = tt.splat %arg3 : i32 -> tensor<32xi32>
            %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
            %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
            %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
            %9 = tt.load %8, %6 : tensor<32x!tt.ptr<i32>>
            %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
            %11 = tt.addptr %10, %4 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
            %12 = tt.load %11, %6 : tensor<32x!tt.ptr<i32>>
            %13 = arith.addi %9, %12 : tensor<32xi32>
            %14 = tt.splat %c3 : i32 -> tensor<32xi32>
            %15 = arith.muli %13, %14 : tensor<32xi32>
            %16 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
            %17 = tt.addptr %16, %4 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
            tt.store %17, %15, %6 : tensor<32x!tt.ptr<i32>>
            tt.return
          }
        }
        """
}
