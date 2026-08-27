"""A real training step for an attention layer, on a Mac GPU, through Triton.

Forward *and* backward as `@triton.jit` kernels, compiled by Triton's own
frontend and MLIR passes, lowered to Metal Shading Language by the Swift core,
and run on the GPU. The gradients are checked against a hand-written numpy
autograd — the same maths written twice, independently — and then used for a few
steps of gradient descent, which is the point: this is the pass the backend did
not have.

    python python/examples/attention_training.py

Two constraints worth knowing before reading the kernels, both documented in
docs/ARCHITECTURE.md:

* **The block sizes must be pairwise distinct.** The emitter identifies a block
  axis by its extent, so `BLOCK_M`, `BLOCK_N` and `HEAD_DIM` may not collide.
  `(16, 32, 64)` is fine; `(32, 32, 64)` is refused by name.
* **`dQ` and `dK`/`dV` are separate kernels.** `dQ_i` sums over key blocks and
  `dK_j`/`dV_j` over query blocks, and no single program owns both ends. The
  alternative is a `tl.atomic_add` accumulation of `dQ`, which the backend now
  supports but which makes the gradient depend on threadgroup arrival order.
"""

import math

import numpy as np
import triton
import triton.language as tl

from triton_metal.buffer import MetalBuffer

LOG2E = 1.44269504088896340736


# --------------------------------------------------------------------------- #
# Forward: FlashAttention-2, storing the per-row logsumexp the backward needs.
# --------------------------------------------------------------------------- #


@triton.jit
def attn_fwd(Q, K, V, O, L, sm_scale, stride_head, stride_seq, stride_lse, N_CTX,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    qk_scale = sm_scale * 1.44269504088896340736
    base = pid_h * stride_head
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m[:, None] < N_CTX
    q = tl.load(Q + base + offs_m[:, None] * stride_seq + offs_d[None, :], mask=row_ok, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for start_n in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_ok = offs_n[:, None] < N_CTX
        kv_off = base + offs_n[:, None] * stride_seq + offs_d[None, :]
        k = tl.load(K + kv_off, mask=key_ok, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        v = tl.load(V + kv_off, mask=key_ok, other=0.0)
        acc = tl.dot(p, v, acc * alpha[:, None])
        m_i = m_ij

    o = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_seq + offs_d[None, :], o, mask=row_ok)
    tl.store(L + pid_h * stride_lse + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX)


# --------------------------------------------------------------------------- #
# Backward.
# --------------------------------------------------------------------------- #


@triton.jit
def attn_bwd_preprocess(O, DO, Delta, stride_head, stride_seq, stride_lse, N_CTX,
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m[:, None] < N_CTX
    off = pid_h * stride_head + offs_m[:, None] * stride_seq + offs_d[None, :]
    o = tl.load(O + off, mask=row_ok, other=0.0)
    do = tl.load(DO + off, mask=row_ok, other=0.0)
    tl.store(Delta + pid_h * stride_lse + offs_m, tl.sum(o * do, 1), mask=offs_m < N_CTX)


@triton.jit
def attn_bwd_dq(Q, K, V, DO, DQ, L, D, sm_scale, stride_head, stride_seq, stride_lse, N_CTX,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    qk_scale = sm_scale * 1.44269504088896340736
    base = pid_h * stride_head
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m[:, None] < N_CTX
    q_off = base + offs_m[:, None] * stride_seq + offs_d[None, :]
    q = tl.load(Q + q_off, mask=row_ok, other=0.0)
    do = tl.load(DO + q_off, mask=row_ok, other=0.0)
    m_ok = offs_m < N_CTX
    lse = tl.load(L + pid_h * stride_lse + offs_m, mask=m_ok, other=0.0)
    delta = tl.load(D + pid_h * stride_lse + offs_m, mask=m_ok, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_ok = offs_n[:, None] < N_CTX
        kv_off = base + offs_n[:, None] * stride_seq + offs_d[None, :]
        k = tl.load(K + kv_off, mask=key_ok, other=0.0)
        v = tl.load(V + kv_off, mask=key_ok, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(offs_n[None, :] < N_CTX, qk, float("-inf"))
        p = tl.math.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds, k, dq)

    tl.store(DQ + q_off, dq * sm_scale, mask=row_ok)


@triton.jit
def attn_bwd_dkdv(Q, K, V, DO, DK, DV, L, D, sm_scale, stride_head, stride_seq, stride_lse, N_CTX,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    qk_scale = sm_scale * 1.44269504088896340736
    base = pid_h * stride_head
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    key_ok = offs_n[:, None] < N_CTX
    kv_off = base + offs_n[:, None] * stride_seq + offs_d[None, :]
    k = tl.load(K + kv_off, mask=key_ok, other=0.0)
    v = tl.load(V + kv_off, mask=key_ok, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for start_m in range(0, tl.cdiv(N_CTX, BLOCK_M)):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = offs_m[:, None] < N_CTX
        q_off = base + offs_m[:, None] * stride_seq + offs_d[None, :]
        q = tl.load(Q + q_off, mask=row_ok, other=0.0)
        do = tl.load(DO + q_off, mask=row_ok, other=0.0)
        m_ok = offs_m < N_CTX
        lse = tl.load(L + pid_h * stride_lse + offs_m, mask=m_ok, other=0.0)
        delta = tl.load(D + pid_h * stride_lse + offs_m, mask=m_ok, other=0.0)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale
        qkT = tl.where(offs_m[None, :] < N_CTX, qkT, float("-inf"))
        pT = tl.math.exp2(qkT - lse[None, :])
        dv = tl.dot(pT, do, dv)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - delta[None, :])
        dk = tl.dot(dsT, q, dk)

    tl.store(DK + kv_off, dk * sm_scale, mask=key_ok)
    tl.store(DV + kv_off, dv, mask=key_ok)


# --------------------------------------------------------------------------- #
# The numpy side: the same maths, written independently.
# --------------------------------------------------------------------------- #


def numpy_attention(q, k, v, scale):
    """[H, S, D] -> output, probabilities. Plain, stable softmax attention."""
    scores = np.einsum("hid,hjd->hij", q, k) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("hij,hjd->hid", weights, v), weights


def numpy_attention_backward(q, k, v, weights, d_out, scale):
    """Hand-written autograd for `numpy_attention`, from the definitions."""
    dv = np.einsum("hij,hid->hjd", weights, d_out)
    dp = np.einsum("hid,hjd->hij", d_out, v)
    delta = (dp * weights).sum(axis=-1, keepdims=True)
    ds = weights * (dp - delta)
    dq = np.einsum("hij,hjd->hid", ds, k) * scale
    dk = np.einsum("hij,hid->hjd", ds, q) * scale
    return dq, dk, dv


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #


class Layer:
    """The three projections, on the GPU, plus the kernels that move them."""

    def __init__(self, heads, seq, dim, block_m=16, block_n=32):
        assert len({block_m, block_n, dim}) == 3, \
            "BLOCK_M, BLOCK_N and HEAD_DIM must be pairwise distinct (see the docstring)"
        self.heads, self.seq, self.dim = heads, seq, dim
        self.block_m, self.block_n = block_m, block_n
        self.scale = 1.0 / math.sqrt(dim)
        shape = (heads, seq, dim)
        self.buffers = {name: MetalBuffer(shape, "float32") for name in
                        ("q", "k", "v", "o", "do", "dq", "dk", "dv")}
        self.buffers["lse"] = MetalBuffer((heads, seq), "float32")
        self.buffers["delta"] = MetalBuffer((heads, seq), "float32")

    def free(self):
        for buffer in self.buffers.values():
            buffer.free()

    def _upload(self, name, array):
        array = np.ascontiguousarray(array, dtype=np.float32)
        self.buffers[name].copy_from(array.ctypes.data, array.nbytes)

    def forward(self, q, k, v):
        self._upload("q", q)
        self._upload("k", k)
        self._upload("v", v)
        grid = (triton.cdiv(self.seq, self.block_m), self.heads)
        attn_fwd[grid](
            self.buffers["q"], self.buffers["k"], self.buffers["v"], self.buffers["o"],
            self.buffers["lse"], self.scale,
            self.seq * self.dim, self.dim, self.seq, self.seq,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, HEAD_DIM=self.dim)
        return self.buffers["o"].numpy()

    def backward(self, d_out):
        self._upload("do", d_out)
        query_grid = (triton.cdiv(self.seq, self.block_m), self.heads)
        key_grid = (triton.cdiv(self.seq, self.block_n), self.heads)
        attn_bwd_preprocess[query_grid](
            self.buffers["o"], self.buffers["do"], self.buffers["delta"],
            self.seq * self.dim, self.dim, self.seq, self.seq,
            BLOCK_M=self.block_m, HEAD_DIM=self.dim)
        attn_bwd_dq[query_grid](
            self.buffers["q"], self.buffers["k"], self.buffers["v"], self.buffers["do"],
            self.buffers["dq"], self.buffers["lse"], self.buffers["delta"], self.scale,
            self.seq * self.dim, self.dim, self.seq, self.seq,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, HEAD_DIM=self.dim)
        attn_bwd_dkdv[key_grid](
            self.buffers["q"], self.buffers["k"], self.buffers["v"], self.buffers["do"],
            self.buffers["dk"], self.buffers["dv"], self.buffers["lse"], self.buffers["delta"],
            self.scale, self.seq * self.dim, self.dim, self.seq, self.seq,
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, HEAD_DIM=self.dim)
        return (self.buffers["dq"].numpy(), self.buffers["dk"].numpy(),
                self.buffers["dv"].numpy())


def relative_error(got, want):
    scale = max(float(np.max(np.abs(want))), 1e-30)
    return float(np.max(np.abs(got - want))) / scale


def main():
    rng = np.random.default_rng(0)
    heads, seq, dim = 2, 96, 64
    q = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
    k = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
    v = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5
    target = rng.standard_normal((heads, seq, dim), dtype=np.float32) * 0.5

    target_arch = triton.runtime.driver.active.get_current_target()
    print(f"triton {triton.__version__} on {target_arch.backend}:{target_arch.arch}")
    print(f"attention layer: h={heads} s={seq} d={dim}, BLOCK_M=16 BLOCK_N=32\n")

    layer = Layer(heads, seq, dim)
    try:
        # ---- one forward + backward, checked against numpy ------------------
        out = layer.forward(q, k, v)
        reference, weights = numpy_attention(q, k, v, layer.scale)
        d_out = (out - target).astype(np.float32)          # d/dO of 0.5 * ||O - T||^2
        dq, dk, dv = layer.backward(d_out)
        ref_dq, ref_dk, ref_dv = numpy_attention_backward(
            q, k, v, weights, (reference - target).astype(np.float32), layer.scale)

        print("forward and gradients against a hand-written numpy autograd")
        print(f"  O   max |triton - numpy| = {np.max(np.abs(out - reference)):.3e}"
              f"   relative {relative_error(out, reference):.3e}")
        for name, got, want in (("dQ", dq, ref_dq), ("dK", dk, ref_dk), ("dV", dv, ref_dv)):
            print(f"  {name}  max |triton - numpy| = {np.max(np.abs(got - want)):.3e}"
                  f"   relative {relative_error(got, want):.3e}")
            assert relative_error(got, want) < 2e-5, f"{name} disagrees with numpy"
        assert relative_error(out, reference) < 2e-5

        # ---- and against finite differences, which know no formulas ---------
        print("\ngradients against central finite differences of the loss")

        def loss_of(qq, kk, vv):
            o, _ = numpy_attention(qq.astype(np.float64), kk.astype(np.float64),
                                   vv.astype(np.float64), layer.scale)
            return 0.5 * float(np.sum((o - target.astype(np.float64))**2))

        h = 1e-5
        worst = 0.0
        for name, base, gradient in (("dQ", q, dq), ("dK", k, dk), ("dV", v, dv)):
            for _ in range(4):
                index = tuple(rng.integers(0, s) for s in base.shape)
                plus = base.astype(np.float64).copy()
                minus = base.astype(np.float64).copy()
                plus[index] += h
                minus[index] -= h
                args = {"dQ": (plus, k, v), "dK": (q, plus, v), "dV": (q, k, plus)}[name]
                down = {"dQ": (minus, k, v), "dK": (q, minus, v), "dV": (q, k, minus)}[name]
                numeric = (loss_of(*args) - loss_of(*down)) / (2 * h)
                error = abs(numeric - float(gradient[index])) / max(1.0, abs(numeric))
                worst = max(worst, error)
                print(f"  {name}{list(index)}  triton {float(gradient[index]):+.6f}"
                      f"   finite difference {numeric:+.6f}   relative {error:.2e}")
        assert worst < 1e-4, f"finite-difference check failed at {worst}"

        # ---- a few steps of gradient descent -------------------------------
        print("\ngradient descent on 0.5 * ||O - T||^2, forward and backward on the GPU")
        rate = 0.5
        for step in range(8):
            out = layer.forward(q, k, v)
            loss = 0.5 * float(np.sum((out - target)**2))
            d_out = (out - target).astype(np.float32)
            dq, dk, dv = layer.backward(d_out)
            print(f"  step {step}: loss {loss:.6f}   |dQ| {np.linalg.norm(dq):.4f}"
                  f"   |dK| {np.linalg.norm(dk):.4f}   |dV| {np.linalg.norm(dv):.4f}")
            if step == 0:
                first = loss
            q = q - rate * dq
            k = k - rate * dk
            v = v - rate * dv
        final = 0.5 * float(np.sum((layer.forward(q, k, v) - target)**2))
        print(f"  step 8: loss {final:.6f}")
        assert final < first, f"loss did not decrease: {first} -> {final}"
        print("\nOK")
    finally:
        layer.free()


if __name__ == "__main__":
    main()
