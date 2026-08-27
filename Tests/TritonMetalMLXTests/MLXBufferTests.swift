import Foundation
import MLX
import Metal
import TritonMetalCore
import XCTest

@testable import TritonMetalMLX

/// The claims `MLXBuffers.swift` makes about copying, checked rather than
/// asserted in a comment.
final class MLXBufferTests: XCTestCase {

    // MARK: - Data types

    func testEverySupportedDataTypeRoundTrips() throws {
        for dtype in MLXDataType.supported {
            let spelling = try MLXDataType.triton(dtype)
            XCTAssertEqual(MLXDataType.mlx(spelling), dtype, "\(dtype) -> \(spelling) -> ?")
        }
    }

    /// A dtype the backend does not lower has to be an error. The alternative —
    /// a silent narrowing to the nearest supported type — produces a plausible
    /// wrong answer, which is the worst kind. `f64` in particular is refused at
    /// emission because Metal has no `double` at all.
    func testUnsupportedDataTypeIsRejectedRatherThanCast() {
        for dtype in [DType.float64, .uint8, .uint16, .uint32, .uint64, .complex64] {
            // The mapping is a pure function of the dtype, so this needs no
            // device and no allocation — `float64` is in the list even though MLX
            // will not put one on a GPU.
            assertCoreError(try MLXDataType.triton(dtype), contains: "\(dtype)")
        }
    }

    func testUnknownTritonSpellingHasNoMLXType() {
        for spelling in ["f64", "f8", "!tt.ptr<f32>", ""] {
            XCTAssertNil(MLXDataType.mlx(spelling))
        }
    }

    // MARK: - Contiguity

    func testContiguityPredicateMatchesLayout() throws {
        try MLXRuntime.require()
        let matrix = synthetic([8, 16])
        matrix.eval()
        XCTAssertTrue(MLXBinding.isContiguous(matrix))
        XCTAssertTrue(MLXBinding.isContiguous(matrix.reshaped([16, 8])))
        // A row of a row-major matrix is a contiguous run; a column is not.
        XCTAssertTrue(MLXBinding.isContiguous(matrix[3]))
        XCTAssertFalse(MLXBinding.isContiguous(matrix[0..., 3]))
        XCTAssertFalse(MLXBinding.isContiguous(matrix.transposed()))
    }

    // MARK: - Aliasing

    /// A `device float *` fill kernel, used to prove that what a kernel stores
    /// through a bound buffer is what MLX subsequently reads.
    private func fillPipeline() throws -> MTLComputePipelineState {
        try MetalCompiler.compileMSL(
            """
            #include <metal_stdlib>
            using namespace metal;
            kernel void tm_test_fill(device float *o [[buffer(0)]], constant int &n [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
                if (i < uint(n)) { o[i] = float(i) + 1.0f; }
            }
            """, kernelName: "tm_test_fill")
    }

    private func fill(_ array: MLXArray, with pipeline: MTLComputePipelineState) throws {
        let device = try XCTUnwrap(MetalRuntime.device)
        let binding = try MLXBinding.bind(array, device: device, what: "test array")
        XCTAssertEqual(binding.mode, .aliased)
        let threads = min(1024, max(32, array.size))
        try MetalRuntime.launch(
            pipeline: pipeline,
            threadgroups: MTLSize(
                width: Grid.covering(array.size, block: threads), height: 1, depth: 1),
            threadsPerThreadgroup: threads,
            arguments: [.buffer(binding.buffer), .int32(Int32(array.size))])
    }

    /// The central zero-copy claim: a kernel dispatched against the buffer this
    /// target binds writes the `MLXArray`'s own storage.
    ///
    /// Sizes chosen to straddle the two boundaries that could plausibly break it:
    /// 16 KB is the page size on Apple silicon, and MLX only guarantees page
    /// alignment at or above one page; 12 bytes is below Foundation's 14-byte
    /// inline-`Data` threshold, which is where mccl's adapter found writes being
    /// silently lost through `Data(bytesNoCopy:)`. Neither matters here, and this
    /// test is what says so.
    func testKernelStoresLandInTheArrayAtEverySize() throws {
        try MLXRuntime.require()
        let pipeline = try fillPipeline()
        for shape in [[3], [1, 3], [512], [4096], [64, 512], [7, 13]] {
            let array = MLX.zeros(shape, dtype: .float32)
            try fill(array, with: pipeline)
            let values = array.asArray(Float.self)
            XCTAssertEqual(values.count, shape.reduce(1, *))
            for (index, value) in values.enumerated() {
                XCTAssertEqual(
                    value, Float(index) + 1,
                    "shape \(shape) (\(array.nbytes) B) lost the store at \(index)")
            }
        }
    }

    /// And MLX's own ops see the stores, on both streams — the array is not
    /// holding a cached evaluation from before the kernel ran.
    func testMLXOperationsSeeWhatTheKernelWrote() throws {
        try MLXRuntime.require()
        let array = MLX.zeros([256], dtype: .float32)
        try fill(array, with: try fillPipeline())
        let expected = Float((256 * 257) / 2)
        XCTAssertEqual(array.sum(stream: .cpu).item(Float.self), expected)
        XCTAssertEqual(array.sum(stream: .gpu).item(Float.self), expected)
    }

    /// MLX packs sub-page allocations into a shared page, so a bound buffer can
    /// overlap the page of an array nobody asked to touch. Metal maps pages, but
    /// the kernel addresses only the length it was given — this is the test that
    /// says a neighbour survives.
    func testANeighbourInTheSamePageIsUntouched() throws {
        try MLXRuntime.require()
        let page = Int(getpagesize())
        var arrays: [MLXArray] = []
        for _ in 0..<64 {
            let array = MLX.zeros([512], dtype: .float32)  // 2048 B: eight to a page
            array.eval()
            arrays.append(array)
        }
        func address(_ array: MLXArray) -> Int {
            array.asData(access: .noCopy).data.withUnsafeBytes {
                Int(bitPattern: $0.baseAddress)
            }
        }
        // Two arrays 2048 bytes apart are in one page by construction.
        var pair: (MLXArray, MLXArray)?
        for (index, array) in arrays.enumerated().dropFirst()
        where abs(address(array) - address(arrays[index - 1])) < page {
            pair = (array, arrays[index - 1])
            break
        }
        let (target, neighbour) = try XCTUnwrap(pair, "MLX did not pack two arrays into a page")

        try fill(target, with: try fillPipeline())
        XCTAssertEqual(target.asArray(Float.self).first, 1)
        XCTAssertEqual(
            neighbour.sum(stream: .cpu).item(Float.self), 0,
            "the kernel wrote outside the buffer it was given")
    }

    // MARK: - The copied path

    /// A strided array has no contiguous run of bytes to hand over, so the
    /// binding stages a copy and says so.
    func testStridedArrayIsStagedRatherThanAliased() throws {
        try MLXRuntime.require()
        let device = try XCTUnwrap(MetalRuntime.device)
        let matrix = synthetic([8, 16])
        matrix.eval()
        let column = matrix[0..., 5]
        let binding = try MLXBinding.bind(column, device: device, what: "column")
        XCTAssertEqual(binding.mode, .copied)
        XCTAssertEqual(binding.buffer.length, column.nbytes)

        // The staged bytes are the column's values, contiguously.
        let staged = UnsafeBufferPointer(
            start: binding.buffer.contents().assumingMemoryBound(to: Float.self), count: 8)
        let expected = column.asArray(Float.self)
        for index in 0..<8 { XCTAssertEqual(staged[index], expected[index]) }
    }

    /// `result()` on a copied binding returns a *new* array; the caller's strided
    /// array is left alone. That is the whole reason the launch API's rule is
    /// "read the returned arrays".
    func testCopiedBindingResultIsANewArray() throws {
        try MLXRuntime.require()
        let device = try XCTUnwrap(MetalRuntime.device)
        let matrix = MLX.zeros([8, 16], dtype: .float32)
        matrix.eval()
        let column = matrix[0..., 5]
        let binding = try MLXBinding.bind(column, device: device, what: "column")
        binding.buffer.contents().assumingMemoryBound(to: Float.self)[0] = 42

        XCTAssertEqual(binding.result().asArray(Float.self)[0], 42)
        XCTAssertEqual(column.asArray(Float.self)[0], 0, "the caller's array was mutated")
    }

    func testEmptyArrayIsRefused() throws {
        try MLXRuntime.require()
        let device = try XCTUnwrap(MetalRuntime.device)
        assertCoreError(
            try MLXBinding.bind(MLX.zeros([0], dtype: .float32), device: device, what: "empty"),
            contains: "empty")
    }
}
