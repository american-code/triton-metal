import Foundation
import MLX
import Metal
import TritonMetalCore

// Handing an `MLXArray` to an emitted Triton kernel.
//
// ## What is copied, measured
//
// Nothing, for any contiguous array. An `MLXArray`'s storage comes out of MLX's
// own Metal allocator, so the bytes are already GPU-resident unified memory;
// `MTLDevice.makeBuffer(bytesNoCopy:length:)` over them returns a buffer whose
// `contents()` is the array's own address, and a kernel dispatched against that
// buffer reads and writes the array's storage in place. mlx-swift exposes
// exactly this as `MLXArray.asMTLBuffer(device:noCopy:)`, which is what
// `MLXBinding` calls; the work here is deciding when it is safe and reporting
// which of its two paths ran.
//
// Verified end to end rather than assumed — `MLXBufferTests` writes through an
// aliased buffer with a Metal kernel and reads the result back through MLX, and
// `SAEEncoderTests` checks the same property for the real kernel:
//
//  * `buffer.contents()` is the array's own base address;
//  * values a kernel stores are visible to `asArray`, to `sum(stream: .cpu)` and
//    to `sum()` on the GPU stream — MLX is not holding a stale copy;
//  * an array packed into the *same page* is untouched.
//
// Two expectations from the CUDA world do not apply and are worth stating, since
// both would otherwise look like something to work around:
//
// **Page alignment is not required.** Metal's documentation for `bytesNoCopy`
// asks for a page-aligned pointer, and MLX's allocator only guarantees that for
// allocations of at least one page. Measured on an M1 Pro (macOS 26.5, 16 KB
// pages), MLX page-aligns every allocation of ≥ 16384 B and packs smaller ones
// into a shared page — 32 fresh 2048-byte arrays land page-aligned 4 times, 32
// fresh 8192-byte ones 16 times. `makeBuffer(bytesNoCopy:)` nevertheless accepts
// the unaligned pointers, returns the same address, and dispatches against them
// correctly, so there is no size floor here and no alias/copy lottery.
//
// **There is no staging copy in either direction.** A CUDA equivalent of this
// file would `cudaMemcpy` every input to the device and every output back. The
// only copy that ever happens is the one below, and it is about layout, not
// about buses.
//
// ## The one case that copies
//
// A strided or broadcast array — a column of a matrix, an unsqueezed vector —
// has no single contiguous run of bytes for a kernel to address. `asMTLBuffer`
// materialises a contiguous copy for those, which is the correct behaviour and
// the reason `MLXBinding` reports its mode instead of promising zero-copy
// unconditionally. `MLXBinding.mode` is `.copied` exactly then.
//
// ## Reading results
//
// `MetalPipeline.launch` returns one array per pointer argument and the rule is
// *read the returned arrays, never the ones you passed*. On the aliased path the
// returned array is the very array you passed, mutated in place by the kernel;
// on the copied path it is a new array carrying what the kernel left in the
// staging buffer, and the array you passed is untouched. One rule covers both,
// and it never silently drops a result.
//
// The alternative — always writing back into the caller's array — was rejected
// because mlx-swift's only public route to an array's bytes is `asData(access:)`,
// which returns a Foundation `Data`, and `Data(bytesNoCopy:count:deallocator:)`
// is not no-copy for payloads of 14 bytes or fewer: Foundation stores those
// inline, so a write through the wrapper lands in a temporary and is lost. That
// boundary was measured in mccl's MLX adapter (`MCCLMLX/MLXBridge.swift`, whose
// structure this target follows); nothing here goes near it, because
// `asMTLBuffer` reaches the backing through `mlx_array_data_uint8` instead.

/// One kernel argument's `MTLBuffer`, and whether it aliases the array's storage
/// or a contiguous copy of it.
public struct MLXBinding {

    /// Whether the buffer *is* the array's storage or a staging copy of it.
    public enum Mode: String, Sendable, Equatable {
        /// The buffer is the array's own unified memory. The kernel's stores land
        /// in the array.
        case aliased
        /// The array's layout is not contiguous, so the buffer holds a contiguous
        /// materialisation of it. The kernel's stores land in the copy.
        case copied
    }

    public let buffer: MTLBuffer
    public let mode: Mode

    /// The array the buffer came from. Retained because `bytesNoCopy` buffers do
    /// not own their memory: releasing the array while the buffer is live would
    /// leave the GPU reading freed pages.
    let array: MLXArray
    let shape: [Int]
    let dtype: DType

    /// True when the array's backing is a single contiguous run of bytes, which
    /// is the condition `asMTLBuffer(noCopy:)` aliases under. Replicates
    /// mlx-swift's own internal `contiguousToDimension() == 0` predicate, whose
    /// result it has no other way to report.
    ///
    /// The strides come from `asData(access: .noCopy)` — the one accessor that
    /// hands back the *backing's* strides rather than a contiguous array's — and
    /// its `Data` is discarded unread, so nothing here depends on Foundation's
    /// inline-storage behaviour for small payloads.
    public static func isContiguous(_ array: MLXArray) -> Bool {
        var expected = 1
        for (extent, stride) in zip(array.shape, array.asData(access: .noCopy).strides).reversed() {
            if stride != expected { return false }
            expected *= extent
        }
        return true
    }

    /// Binds `array` for a kernel argument. Forces evaluation first: an
    /// `MLXArray` is a node in a graph until something makes it a buffer, and
    /// the backing of an unevaluated array is not a buffer of results.
    static func bind(_ array: MLXArray, device: MTLDevice, what: String) throws -> MLXBinding {
        guard array.size > 0 else {
            throw CoreError.invalidArgument(
                "\(what) is empty (shape \(array.shape)); a kernel argument needs at least one "
                    + "element to have an address")
        }
        array.eval()
        let contiguous = isContiguous(array)
        guard let buffer = array.asMTLBuffer(device: device, noCopy: true) else {
            throw CoreError.metal(
                "could not make an MTLBuffer over \(what) (\(array.nbytes) bytes, shape "
                    + "\(array.shape))")
        }
        return MLXBinding(
            buffer: buffer, mode: contiguous ? .aliased : .copied, array: array,
            shape: array.shape, dtype: array.dtype)
    }

    /// The array holding what the kernel left in this buffer.
    ///
    /// `.aliased` returns the array that was passed in — the kernel wrote its
    /// storage. `.copied` builds a fresh array from the staging buffer, because
    /// the passed-in array's strided layout is not what the kernel addressed.
    func result() -> MLXArray {
        switch mode {
        case .aliased:
            return array
        case .copied:
            let data = Data(bytes: buffer.contents(), count: buffer.length)
            return MLXArray(data, shape, dtype: dtype)
        }
    }
}
