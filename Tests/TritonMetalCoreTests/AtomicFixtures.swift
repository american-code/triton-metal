import Foundation

/// Triton IR fixtures for `tt.atomic_rmw` and `tt.atomic_cas`, spelled the way
/// Triton 3.7.1 prints them: the RMW kind, the memory semantics and the sync
/// scope as bare keywords in front of the operands, then a functional type
/// (`TritonOps.td`'s `$atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val
/// (`,` $mask^)? attr-dict `:` functional-type(operands, $result)`).
enum AtomicFixtures {

    /// Scatter-accumulate: every element of `in` is folded into
    /// `out[i % buckets]` with one atomic, over a grid of programs that all
    /// contend for the same small output. The `mask` is the usual tail guard.
    ///
    /// `kind` is the `tt.atomic_rmw` op spelling; `element` the pointee type.
    /// When `keepOld` is set the value the atomic returns is stored to a third
    /// buffer, which is what exercises the result of a side-effecting op.
    static func scatter(
        kind: String, element: String = "f32", block: Int = 64, keepOld: Bool = false,
        masked: Bool = true
    ) -> String {
        let maskOperand = masked ? ", %mask" : ""
        let maskType = masked ? ", tensor<\(block)xi1>" : ""
        let loadMask = masked ? ", %mask" : ""
        let oldStore =
            keepOld
            ? """
                  %old_p = tt.splat %arg4 : !tt.ptr<\(element)> -> \
            tensor<\(block)x!tt.ptr<\(element)>>
                  %old_ptr = tt.addptr %old_p, %idx : tensor<\(block)x!tt.ptr<\(element)>>, \
            tensor<\(block)xi32>
                  tt.store %old_ptr, %old\(loadMask) : tensor<\(block)x!tt.ptr<\(element)>>
            """
            : ""
        let oldArgument = keepOld ? ", %arg4: !tt.ptr<\(element)>" : ""
        return """
            module {
              tt.func public @atomic_scatter_kernel(%arg0: !tt.ptr<\(element)>,
                                                    %arg1: !tt.ptr<\(element)>,
                                                    %arg2: i32, %arg3: i32\(oldArgument)) {
                %cb_i32 = arith.constant \(block) : i32
                %pid = tt.get_program_id x : i32
                %base = arith.muli %pid, %cb_i32 : i32
                %range = tt.make_range {end = \(block) : i32, start = 0 : i32} : \
            tensor<\(block)xi32>
                %base_b = tt.splat %base : i32 -> tensor<\(block)xi32>
                %idx = arith.addi %base_b, %range : tensor<\(block)xi32>
                %n_b = tt.splat %arg2 : i32 -> tensor<\(block)xi32>
                %mask = arith.cmpi slt, %idx, %n_b : tensor<\(block)xi32>
                %in_p = tt.splat %arg0 : !tt.ptr<\(element)> -> \
            tensor<\(block)x!tt.ptr<\(element)>>
                %in_ptr = tt.addptr %in_p, %idx : tensor<\(block)x!tt.ptr<\(element)>>, \
            tensor<\(block)xi32>
                %val = tt.load %in_ptr\(loadMask) : tensor<\(block)x\(element)>
                %buckets = tt.splat %arg3 : i32 -> tensor<\(block)xi32>
                %slot = arith.remsi %idx, %buckets : tensor<\(block)xi32>
                %out_p = tt.splat %arg1 : !tt.ptr<\(element)> -> \
            tensor<\(block)x!tt.ptr<\(element)>>
                %out_ptr = tt.addptr %out_p, %slot : tensor<\(block)x!tt.ptr<\(element)>>, \
            tensor<\(block)xi32>
                %old = tt.atomic_rmw \(kind), acq_rel, gpu, %out_ptr, %val\(maskOperand) : \
            (tensor<\(block)x!tt.ptr<\(element)>>, tensor<\(block)x\(element)>\(maskType)) -> \
            tensor<\(block)x\(element)>
            \(oldStore)
                tt.return
              }
            }
            """
    }

    /// A `tt.atomic_cas` spin: every lane tries to publish its own index into
    /// `out[i % buckets]`, but only if the slot still holds the sentinel. Exactly
    /// one lane per slot can win, so the result is a valid winner rather than a
    /// value that depends on the order.
    static func compareAndSwap(element: String = "i32", block: Int = 64) -> String {
        """
        module {
          tt.func public @atomic_cas_kernel(%arg0: !tt.ptr<\(element)>,
                                            %arg1: !tt.ptr<\(element)>,
                                            %arg2: i32, %arg3: i32) {
            %cb_i32 = arith.constant \(block) : i32
            %pid = tt.get_program_id x : i32
            %base = arith.muli %pid, %cb_i32 : i32
            %range = tt.make_range {end = \(block) : i32, start = 0 : i32} : tensor<\(block)xi32>
            %base_b = tt.splat %base : i32 -> tensor<\(block)xi32>
            %idx = arith.addi %base_b, %range : tensor<\(block)xi32>
            %n_b = tt.splat %arg2 : i32 -> tensor<\(block)xi32>
            %mask = arith.cmpi slt, %idx, %n_b : tensor<\(block)xi32>
            %sentinel = arith.constant dense<-1> : tensor<\(block)x\(element)>
            %buckets = tt.splat %arg3 : i32 -> tensor<\(block)xi32>
            %slot = arith.remsi %idx, %buckets : tensor<\(block)xi32>
            %out_p = tt.splat %arg0 : !tt.ptr<\(element)> -> tensor<\(block)x!tt.ptr<\(element)>>
            %out_ptr = tt.addptr %out_p, %slot : tensor<\(block)x!tt.ptr<\(element)>>, \
        tensor<\(block)xi32>
            %old = tt.atomic_cas acq_rel, gpu, %out_ptr, %sentinel, %idx : \
        (tensor<\(block)x!tt.ptr<\(element)>>, tensor<\(block)x\(element)>, \
        tensor<\(block)x\(element)>) -> tensor<\(block)x\(element)>
            %seen_p = tt.splat %arg1 : !tt.ptr<\(element)> -> tensor<\(block)x!tt.ptr<\(element)>>
            %seen_ptr = tt.addptr %seen_p, %idx : tensor<\(block)x!tt.ptr<\(element)>>, \
        tensor<\(block)xi32>
            tt.store %seen_ptr, %old, %mask : tensor<\(block)x!tt.ptr<\(element)>>
            tt.return
          }
        }
        """
    }

    /// A rank-2 atomic whose nest ends on the **row** axis: `out[m] += a[m, n]`
    /// is not expressible that way, so this is the shape that reaches the
    /// single-writer guard — a rank-1 atomic in a rank-2 kernel, where the row
    /// loop is uniform because the column loop is nested inside it.
    static func rowUniform(keepOld: Bool) -> String {
        let oldStore =
            keepOld
            ? """
                  %old_ptr = tt.addptr %col_p, %rows : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
                  tt.store %old_ptr, %old : tensor<8x!tt.ptr<f32>>
            """
            : ""
        return """
            module {
              tt.func public @row_uniform_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                                                 %arg2: i32) {
                %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
                %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
                %rows_r = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
                %cols_c = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
                %stride = tt.splat %arg2 : i32 -> tensor<8x1xi32>
                %row_off = arith.muli %rows_r, %stride : tensor<8x1xi32>
                %row_b = tt.broadcast %row_off : tensor<8x1xi32> -> tensor<8x16xi32>
                %col_b = tt.broadcast %cols_c : tensor<1x16xi32> -> tensor<8x16xi32>
                %off = arith.addi %row_b, %col_b : tensor<8x16xi32>
                %in_p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
                %in_ptr = tt.addptr %in_p, %off : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
                %tile = tt.load %in_ptr : tensor<8x16xf32>
                %sum = "tt.reduce"(%tile) <{axis = 1 : i32}> ({
                ^bb0(%a: f32, %b: f32):
                  %s = arith.addf %a, %b : f32
                  tt.reduce.return %s : f32
                }) : (tensor<8x16xf32>) -> tensor<8xf32>
                %col_p = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
                %out_ptr = tt.addptr %col_p, %rows : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
                %old = tt.atomic_rmw fadd, acq_rel, gpu, %out_ptr, %sum : \
            (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) -> tensor<8xf32>
            \(oldStore)
                tt.return
              }
            }
            """
    }
}
