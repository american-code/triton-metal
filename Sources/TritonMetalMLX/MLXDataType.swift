import MLX
import TritonMetalCore

/// Translation between MLX's `DType` and the Triton element-type spellings that
/// `EmittedKernel.arguments` carries (`f32`, `bf16`, `i32`, ...).
///
/// The mapping is deliberately partial. A dtype the backend does not lower is an
/// error rather than a cast: handing a `float64` tensor to an `f32` kernel by
/// quietly narrowing it produces a plausible, wrong answer, and `f64` is refused
/// at emission anyway because Metal has no `double` at all. MLX's unsigned
/// integers have no Triton spelling in `TMType` either — it prints every integer
/// as `iN` — so they are refused rather than reinterpreted as signed.
public enum MLXDataType {

    /// The dtypes an `MLXArray` may be bound to a kernel argument in.
    public static let supported: [DType] = [
        .float32, .float16, .bfloat16, .int8, .int16, .int32, .int64, .bool,
    ]

    /// The Triton spelling of `dtype`, as `KernelArgument.dtype` writes it.
    public static func triton(_ dtype: DType) throws -> String {
        switch dtype {
        case .float32: return "f32"
        case .float16: return "f16"
        case .bfloat16: return "bf16"
        case .int8: return "i8"
        case .int16: return "i16"
        case .int32: return "i32"
        case .int64: return "i64"
        case .bool: return "i1"
        default:
            throw CoreError.invalidArgument(
                "MLX dtype \(dtype) has no Triton spelling this backend lowers; it carries "
                    + supported.map { (try? triton($0)) ?? "?" }.joined(separator: ", ")
                    + ". Cast the array yourself if a narrowing conversion is what you want.")
        }
    }

    /// The `DType` a kernel argument spelled `triton` expects, or `nil` when the
    /// spelling is one no `MLXArray` can hold.
    public static func mlx(_ triton: String) -> DType? {
        switch triton {
        case "f32": return .float32
        case "f16": return .float16
        case "bf16": return .bfloat16
        case "i8": return .int8
        case "i16": return .int16
        case "i32": return .int32
        case "i64": return .int64
        case "i1": return .bool
        default: return nil
        }
    }
}
