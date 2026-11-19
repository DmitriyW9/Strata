#!/usr/bin/env python3
# tools/naive_real_layer_fixture.py - a reference forward of TWO real layers, from the model's own
# quantised weights, for tests/naive/test_naive_layer.cpp.
#
# The loader test (tools/naive_real_fixture.py) proved the loader reads the small F32 tensors exactly
# and runs the router over them.  This one answers the next step in docs/NAIVE.md: does a GEMV over
# the *quantised* Q8_0/Q6_K weights, feeding the tested kernels (rmsnorm, split-half RoPE, sink
# attention, the dense SwiGLU), reproduce a reference forward of the layer?  Two layers, both read
# from shard 1 of the model:
#
#   gemv_probe     one row against a real Q8_0 weight (blk.1.attn_v) - the quantised GEMV on its own.
#   layer0_dense   layer 0 (dense, no experts): rmsnorm -> SwiGLU(gate, up) -> down, Q8_0 throughout.
#   swa_small /    layer 1 (sliding-window): rmsnorm -> Q/K/V projections (Q6_K, Q8_0), split-half
#   swa_window     RoPE per position, GQA (64 q heads / 8 kv heads, contiguous groups), a per-head
#                  sink logit, a 128 window, then the O projection (Q6_K).  swa_window caches more
#                  rows than the window, so the window truncation is exercised too.
#
# WHY THESE DECODERS ARE AN INDEPENDENT REFERENCE: the C++ dequantises with ggml's compiled
# `to_float`.  Here the Q8_0 and Q6_K bytes are decoded from the struct layout in ggml-common.h with
# numpy alone, over an element-index mapping derived from the format spec - a different control flow
# from ggml's nested half/sub-block loop, and no ggml call - so when the C++ (ggml decode) and this
# (numpy decode) agree on the same fixture, that agreement is a real cross-check of ggml's decoder on
# the model's bytes, not a wrapper around it.  A built-in self-test first checks the vectorised
# mapping against a literal line-by-line transcription of ggml's loop (ggml-quants.c:1939) on random
# bytes, so the mapping itself is pinned before any fixture is trusted.  IQ2 is NOT decoded here (it
# needs ggml's 1024-entry codebook, which the port deliberately borrows rather than transcribes);
# the MoE/expert path is a later step.
#
# The reference math transcribes NaiveRT: run_attn_swa.py (rmsnorm / project / hf_rope /
# reference_attention, the GQA q[T,H,D]-over-k[S,kv,D] view, the window visible = min(pos+1, WINDOW)),
# prefill.py:scale_values (value_scale folded onto V after the projection, in bf16), and
# training_moe's SwiGLU for the dense FFN.  Intermediates are rounded to bf16 where the reference
# rounds (the projections, the activation, the value scale), so the C++ rounds the same way and the
# two compare like with like.
#
#   python terminal/tools/naive_real_layer_fixture.py [model-shard-1.gguf] [outdir]
#     default model:  /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/...00001-of-00004.gguf
#     default outdir: engine/tests/naive/fixtures

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402
from naive_iq2 import dequant_iq2_xs, dequant_iq2_xxs, self_test as self_test_iq2  # noqa: E402
from naive_kernels_oracle import bf16_round, rmsnorm, rope_head, swa_attention, write_arr  # noqa: E402

DEFAULT_MODEL = ("/home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/"
                 "Naive-N0.5-Flash-MQ87-00001-of-00004.gguf")

# The model's own geometry (asserted against shard 1's metadata in main, so a shape change is caught
# rather than silently absorbed).
N_EMBD, N_HEAD, K_DIM, V_DIM = 4096, 64, 192, 128
N_KV_SWA, WINDOW, N_ROT = 8, 128, 64
VALUE_SCALE, THETA_SWA, RMS_EPS = 0.707, 1e4, 1e-5
DENSE_FF = 16384


def silu(g: np.ndarray) -> np.ndarray:
    """g / (1 + e^-g), the SwiGLU gate (training_moe: gate*sigmoid(gate)*up)."""
    return g / (1.0 + np.exp(-g))


# ------------------------------------------------------------------ independent quant decoders
# Byte layouts from ggml-common.h: block_q8_0 = {half d; int8 qs[32]} (34 B / 32 elts); block_q6_K =
# {u8 ql[128]; u8 qh[64]; i8 scales[16]; half d} (210 B / 256 elts).  Vectorised over an element-index
# mapping derived from the format, chunked to keep the temporaries small.


def _f16(u16: np.ndarray) -> np.ndarray:
    """uint16 bit patterns read as float16, widened exactly to float32 (a reinterpretation, not a
    numeric cast: 0x3C00 must become 1.0, not 15360.0).  The input must be contiguous."""
    return np.asarray(u16, np.uint16).view(np.float16).astype(np.float32)


_Q8_BB = BLOCK_GEOMETRY["Q8_0"][1]      # 34
_Q8_CHUNK = 65536                        # blocks per chunk (2M elements)


def dequant_q8_0(raw: np.ndarray, n: int) -> np.ndarray:
    """y[i] = qs[i] * d, one fp16 scale per 32 int8 codes (ggml-common.h block_q8_0)."""
    assert n % 32 == 0 and raw.size == n // 32 * _Q8_BB
    nb = n // 32
    out = np.empty(n, np.float32)
    for a in range(0, nb, _Q8_CHUNK):                 # over blocks; bytes = blocks * 34
        b = min(a + _Q8_CHUNK, nb)
        blk = np.frombuffer(raw[a * _Q8_BB : b * _Q8_BB].tobytes(), np.uint8).reshape(b - a, _Q8_BB)
        d = _f16(blk[:, 0:2].copy().view(np.uint16)[:, 0])
        qs = blk[:, 2:34].astype(np.int8).astype(np.float32)
        out[a * 32 : b * 32] = (qs * d[:, None]).reshape(-1)
    return out


# The Q6_K element mapping (derived, not copied from the loop): over 256 elements a block is two
# 128-halves; within a half, element r sits at sub-byte l = r % 32, column c = r // 32 selects which
# nibble of ql and which 2-bit slice of qh carries the code's top two bits, and which sub-scale
# applies (is = l // 16, scale = sc[is + 2c] within the half's eight).  code = (nib | bits<<4) - 32.
_Q6_E = np.arange(256)
_Q6_H = _Q6_E // 128                                 # 0/1: which 128-elt half
_Q6_R = _Q6_E % 128
_Q6_L = _Q6_R % 32
_Q6_C = _Q6_R // 32
_Q6_QL = _Q6_H * 64 + _Q6_L + (_Q6_C % 2) * 32       # byte of the 128-byte ql
_Q6_QH = _Q6_H * 32 + _Q6_L                          # byte of the 64-byte qh
_Q6_SHIFT = (_Q6_C * 2).astype(np.uint8)             # 0/2/4/6: which 2 bits of qh[l]
_Q6_HI = _Q6_C >= 2                                  # c in {2,3} take the high nibble
_Q6_SC = _Q6_H * 8 + _Q6_L // 16 + _Q6_C * 2         # int8 scale, of the block's 16
_Q6_BB = BLOCK_GEOMETRY["Q6_K"][1]                   # 210
_Q6_CHUNK = 16384                                    # blocks per chunk (4M elements)


def dequant_q6_K(raw: np.ndarray, n: int) -> np.ndarray:
    assert n % 256 == 0 and raw.size == n // 256 * _Q6_BB, (raw.size, n)
    nb = n // 256
    out = np.empty(n, np.float32)
    for a in range(0, nb, _Q6_CHUNK):                 # over blocks; bytes = blocks * 210
        b = min(a + _Q6_CHUNK, nb)
        blk = np.frombuffer(raw[a * _Q6_BB : b * _Q6_BB].tobytes(), np.uint8).reshape(b - a, _Q6_BB)
        ql, qh = blk[:, 0:128], blk[:, 128:192]
        sc = blk[:, 192:208].astype(np.int8).astype(np.float32)
        d = _f16(blk[:, 208:210].copy().view(np.uint16)[:, 0])
        nib = np.where(_Q6_HI[None, :], ql[:, _Q6_QL] >> 4, ql[:, _Q6_QL] & 0xF)
        two = (qh[:, _Q6_QH] >> _Q6_SHIFT[None, :]) & 3
        code = ((nib | (two << 4)).astype(np.int16) - 32).astype(np.float32)   # -32..31
        out[a * 256 : b * 256] = (d[:, None] * sc[:, _Q6_SC] * code).reshape(-1)
    return out


def _q6_K_literal(raw: np.ndarray, n: int) -> np.ndarray:
    """Line-by-line transcription of ggml-quants.c:1939 (dequantize_row_q6_K), float64.

    Only used by the self-test below, to pin the vectorised mapping above to the loop ggml runs."""
    out = np.zeros(n, np.float64)
    for i in range(n // 256):
        bs, eo = i * 210, i * 256                      # byte stride 210, element stride 256
        d = float(np.frombuffer(raw[bs + 208 : bs + 210].tobytes(), np.float16)[0])
        ql = raw[bs : bs + 128]
        qh = raw[bs + 128 : bs + 192]
        sc = np.frombuffer(raw[bs + 192 : bs + 208].tobytes(), np.int8)
        y = out[eo : eo + 256]
        for hn in (0, 128):                           # the loop's y += 128, ql += 64, qh += 32, sc += 8
            h = hn // 128
            for l in range(32):
                is_ = l // 16                         # the C `int is = l/16`
                c0 = int(ql[l + 64 * h] & 0xF | ((qh[l + 32 * h] >> 0) & 3) << 4) - 32
                c1 = int(ql[l + 32 + 64 * h] & 0xF | ((qh[l + 32 * h] >> 2) & 3) << 4) - 32
                c2 = int(ql[l + 64 * h] >> 4 | ((qh[l + 32 * h] >> 4) & 3) << 4) - 32
                c3 = int(ql[l + 32 + 64 * h] >> 4 | ((qh[l + 32 * h] >> 6) & 3) << 4) - 32
                y[hn + l + 0] = d * sc[h * 8 + is_ + 0] * c0
                y[hn + l + 32] = d * sc[h * 8 + is_ + 2] * c1
                y[hn + l + 64] = d * sc[h * 8 + is_ + 4] * c2
                y[hn + l + 96] = d * sc[h * 8 + is_ + 6] * c3
    return out


def self_test_decoders() -> None:
    """The vectorised Q6_K mapping must equal the literal ggml loop, and Q8_0 must match bit-exactly.

    Bytes are random, but each block's fp16 scale field is forced finite (the loop reads d from raw
    bits, so random bytes put NaN/Inf there; both decoders agree on NaN, but a real weight never
    carries it, and NaN poisons a relative-error metric)."""
    rng = np.random.default_rng(6)
    # Shrink the chunk sizes for the self-test so the chunk boundaries are exercised: the decoders
    # index `raw` in BYTES (blocks * 34 / * 210), which is not the element stride the output uses.
    global _Q8_CHUNK, _Q6_CHUNK
    _Q8_CHUNK, _Q6_CHUNK = 17, 13
    for trial in range(3):
        nb = 64 + trial * 7
        raw6 = rng.integers(0, 256, nb * 210, dtype=np.uint8).copy()
        raw6.reshape(nb, 210)[:, 208:210] = np.uint8([0x00, 0x3C])   # d = 1.0f, as finite bytes
        got = dequant_q6_K(raw6, nb * 256).astype(np.float64)
        ref = _q6_K_literal(raw6, nb * 256)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)
        if not (rel.max() <= 1e-6):
            raise SystemExit(f"q6_K self-test mismatch (trial {trial}): max rel {rel.max():.3g}")
        raw8 = rng.integers(0, 256, nb * 34, dtype=np.uint8).copy()
        raw8.reshape(nb, 34)[:, 0:2] = np.uint8([0x00, 0x3C])        # d = 1.0f
        g8 = dequant_q8_0(raw8, nb * 32)                              # the decoder's own float32
        blk = np.frombuffer(raw8.tobytes(), np.uint8).reshape(-1, 34)
        r8 = (blk[:, 2:34].astype(np.int8).astype(np.float64)
              * blk[:, 0:2].copy().view(np.float16)[:, 0].astype(np.float64)[:, None]
              ).astype(np.float32).reshape(-1)                    # exact product, then rounded
        # exact on purpose: int8 x fp16-as-float is representable in float32's 24-bit mantissa, so
        # the C float multiply and this fp64-then-round product are the same number.  A byte-offset
        # bug cannot hide behind the tolerance.
        if not np.array_equal(g8, r8):
            raise SystemExit("q8_0 self-test mismatch")
    print("decoders self-tested: q6_K vs the literal ggml loop, q8_0 bit-exact vs the layout")


def dequant_bf16(raw: np.ndarray, n: int) -> np.ndarray:
    """BF16 (ggml block size 1): each value is a bf16 -> widen to fp32 by a 16-bit left shift.  The
    indexer's weights are BF16; this is the byte layout in ggml-common.h, not a ggml call."""
    assert raw.size == n * 2, (raw.size, n)
    u = np.frombuffer(raw[: n * 2].tobytes(), np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


# The IQ2 experts are decoded by tools/naive_iq2.py: numpy decoders written from the struct layout
# and self-tested bit-for-bit against a literal transcription of ggml's decode loop (see that file
# for where the independence is and is not - the codebook tables are shared format data).
DECODERS = {"Q8_0": dequant_q8_0, "Q6_K": dequant_q6_K,
            "IQ2_XXS": dequant_iq2_xxs, "IQ2_XS": dequant_iq2_xs, "BF16": dequant_bf16}


# ------------------------------------------------------------------ multi-shard weight reader
def _sibling_shards(shard1: pathlib.Path):
    """The other shards of a split model, by the `-of-N` naming (read-only, like everything here)."""
    name = shard1.name
    i = name.rfind("-of-")
    if i < 0:
        return []
    total = int(name[i + 4 : name.index(".", i + 4)])
    stem = name[: name.rfind("-of-")]                 # "<stem>-<nnnnn>"
    stem = stem[: stem.rfind("-") + 1]                # "<stem>-"
    out = []
    for no in range(2, total + 1):
        p = shard1.with_name(f"{stem}{no:05d}-of-{total:05d}.gguf")
        if p.exists():
            out.append(p)
    return out


class Model:
    """Opens the shards around one path and reads a named tensor's bytes from the shard holding it."""

    def __init__(self, shard1: pathlib.Path):
        self.paths = [shard1] + _sibling_shards(shard1)
        self.files = [GGUFFile(p) for p in self.paths]
        self.index: dict[str, tuple[int, object]] = {}
        for si, gf in enumerate(self.files):
            for t in gf.tensors:
                self.index.setdefault(t.name, (si, t))   # first match wins, like C++ GgufModel::find

    def _info(self, name: str):
        if name not in self.index:
            raise SystemExit(f"no tensor {name} among {len(self.index)} in {len(self.paths)} shards")
        return self.index[name]

    def raw(self, name: str) -> np.ndarray:
        si, t = self._info(name)
        nbytes = t.expected_bytes()
        with self.paths[si].open("rb") as fh:
            fh.seek(self.files[si].data_start + t.offset)
            raw = fh.read(nbytes)
        if len(raw) != nbytes:
            raise SystemExit(f"{name} runs past the end of {self.paths[si].name}")
        return np.frombuffer(raw, np.uint8)

    def raw_range(self, name: str, elem_start: int, n_elems: int, block_elems: int,
                  block_bytes: int) -> np.ndarray:
        """A byte range of a quantised tensor, given in ELEMENTS - whole blocks only.

        A 3-D expert weight is 6.4 G elements (12 GiB of IQ2); a fixture reads ONE expert's slice
        (8.4 M elements, ~2.2 MB) straight by block offset, exactly the window the C++ side takes.
        Offsets: elements [a, b) of a block-quantised tensor live at bytes [a/be*bb, b/be*bb)."""
        assert elem_start % block_elems == 0 and n_elems % block_elems == 0
        si, t = self._info(name)
        a = elem_start // block_elems * block_bytes
        nbytes = n_elems // block_elems * block_bytes
        with self.paths[si].open("rb") as fh:
            fh.seek(self.files[si].data_start + t.offset + a)
            raw = fh.read(nbytes)
        if len(raw) != nbytes:
            raise SystemExit(f"{name}[{elem_start}:{elem_start + n_elems}] runs past {self.paths[si].name}")
        return np.frombuffer(raw, np.uint8)

    def weight_f32(self, name: str) -> np.ndarray:
        """The tensor's values in file order (dim 0 fastest): F32 raw, else decoded via DECODERS."""
        _, t = self._info(name)
        raw, n = self.raw(name), t.elements
        if t.type_name == "F32":
            return np.frombuffer(raw[: n * 4].tobytes(), "<f4").copy()
        if t.type_name == "BF16":
            # bf16 -> f32 is a 16-bit left shift: view as u16<<16, then f32 (exact, no rounding).
            u = np.frombuffer(raw[: n * 2].tobytes(), "<u2").astype(np.uint32) << 16
            return u.view(np.float32).copy()
        dec = DECODERS.get(t.type_name)
        if dec is None:
            raise SystemExit(f"{name} is {t.type_name}; this generator decodes F32/BF16/Q8_0/Q6_K only")
        return dec(raw, n)


# ------------------------------------------------------------------ reference forward pieces
def project(xn: np.ndarray, w_file: np.ndarray, n_embd: int, n_out: int, chunk: int = 4096) -> np.ndarray:
    """run_attn_swa.project: xn @ W.T in fp64 (rounded bf16).  xn is [..., n_embd]; result [..., n_out].

    The weight arrives in GGUF element order (dim 0 = n_embd fastest), so row m of the logical
    {n_embd, n_out} matrix is the contiguous slice w[m*n_embd : (m+1)*n_embd] - a reshape to
    (n_out, n_embd) reads them as rows.  Same element order C++ gemv_dequant walks.  Chunked along
    n_out so the fp64 copy of a 268 MB weight never materialises beside the running engine."""
    W = w_file.reshape(n_out, n_embd)
    xn64 = np.asarray(xn, np.float64)
    y = np.empty(xn64.shape[:-1] + (n_out,), np.float64)
    for a in range(0, n_out, chunk):
        b = min(a + chunk, n_out)
        y[..., a:b] = xn64 @ np.ascontiguousarray(W[a:b].T.astype(np.float64))
    return bf16_round(y).astype(np.float32)


def rope_rows(rows: np.ndarray, pos: int) -> np.ndarray:
    """Split-half RoPE on the first N_ROT dims of every head row, at one position. rows [H, K]."""
    return rope_head(rows[np.newaxis].astype(np.float64), np.array([pos]), N_ROT, THETA_SWA)[0]


def build_gemv_probe(m: Model, x: np.ndarray, outdir: str) -> None:
    """One row against a real Q8_0 weight, decoded independently - the quantised GEMV alone."""
    w = m.weight_f32("blk.1.attn_v.weight")            # {4096, 1024} Q8_0, file order
    y = np.asarray(x, np.float64) @ np.ascontiguousarray(w.reshape(1024, 4096).T.astype(np.float64))
    write_arr(outdir, "gemv_probe", "x", x)
    write_arr(outdir, "gemv_probe", "y", y.astype(np.float32))


def build_dense(m: Model, x: np.ndarray, outdir: str) -> None:
    """Layer 0's dense FFN: rmsnorm -> SwiGLU(gate, up) -> down, all Q8_0."""
    gamma = m.weight_f32("blk.0.ffn_norm.weight")
    xn = bf16_round(rmsnorm(x, gamma, RMS_EPS)).astype(np.float32)   # norms round to bf16 before the GEMV
    gate_w = m.weight_f32("blk.0.ffn_gate.weight")
    gate = project(xn, gate_w, N_EMBD, DENSE_FF)
    del gate_w
    up_w = m.weight_f32("blk.0.ffn_up.weight")
    up = project(xn, up_w, N_EMBD, DENSE_FF)
    del up_w
    act = bf16_round(silu(gate) * up)                  # silu(g)*u, rounded bf16 (training_moe)
    down_w = m.weight_f32("blk.0.ffn_down.weight")
    ffn = project(act, down_w, DENSE_FF, N_EMBD)       # down: {16384, 4096} -> 4096
    del down_w
    for name, arr in (("x", x), ("xn", xn), ("gate", gate), ("up", up), ("act", act), ("ffn_out", ffn)):
        write_arr(outdir, "layer0_dense", name, arr)


def gqa_sink_attention(q, kcache, vcache, sink, T, window):
    """GQA + sink + window over a cache, contiguous head groups (run_attn_swa's [T,H,D]/[S,kv,D] view).

    q [H, K], kcache [P, kv, K], vcache [P, kv, V], sink [H].  The query at row T attends rows
    w_first..T where w_first = T + 1 - min(T+1, window); head h reads kv head h // (H // kv) and that
    head's sink[h].  Returns the [H, V] attention output and the window's first row."""
    H, KV = q.shape[0], kcache.shape[1]
    group = H // KV
    visible = min(T + 1, window)
    w_first = T + 1 - visible
    out = np.empty((H, V_DIM), np.float64)
    for g in range(KV):
        sl = slice(g * group, (g + 1) * group)
        out[sl] = swa_attention(q[sl], kcache[w_first : T + 1, g], vcache[w_first : T + 1, g],
                                sink[sl], 0, T - w_first)
    return out, w_first


def build_swa(m: Model, x_rows: np.ndarray, outdir: str, case: str) -> None:
    """A sliding-window layer's attention over a cache of hidden states, query at the last row."""
    gamma = m.weight_f32("blk.1.attn_norm.weight")
    P, T = x_rows.shape[0], x_rows.shape[0] - 1
    xns = np.stack([bf16_round(rmsnorm(x_rows[p], gamma, RMS_EPS)) for p in range(P)]).astype(np.float32)

    wk = m.weight_f32("blk.1.attn_k.weight")           # {4096, 1536}  Q8_0 (8 kv x 192)
    k_all = project(xns, wk, N_EMBD, N_KV_SWA * K_DIM)
    del wk
    wv = m.weight_f32("blk.1.attn_v.weight")           # {4096, 1024}  Q8_0 (8 kv x 128)
    v_all = project(xns, wv, N_EMBD, N_KV_SWA * V_DIM)
    del wv
    v_all = bf16_round(v_all.astype(np.float64) * VALUE_SCALE).astype(np.float32)
    # (The reference folds value_scale into the weight when it loads it; scaling the projected V
    # instead differs from that fold by at most one bf16 ulp of the product.  The fixture and the
    # C++ both take the post-projection choice, which is what is under test here.)
    kcache = np.empty((P, N_KV_SWA, K_DIM), np.float64)
    for p in range(P):
        kcache[p] = rope_rows(k_all[p].reshape(N_KV_SWA, K_DIM), p)
    vcache = v_all.reshape(P, N_KV_SWA, V_DIM).astype(np.float64)

    wq = m.weight_f32("blk.1.attn_q.weight")           # {4096, 12288} Q6_K
    q = rope_rows(project(xns[T], wq, N_EMBD, N_HEAD * K_DIM).reshape(N_HEAD, K_DIM), T)
    del wq

    sink = m.weight_f32("blk.1.attn_sinks.weight")     # {64} F32
    attn_heads, w_first = gqa_sink_attention(q, kcache, vcache, sink, T, WINDOW)
    attn_flat = bf16_round(attn_heads.reshape(-1)).astype(np.float32)   # [64*128], the O-proj input
    wo = m.weight_f32("blk.1.attn_output.weight")      # {8192, 4096}  Q6_K
    attn_out = project(attn_flat, wo, N_HEAD * V_DIM, N_EMBD)           # O projection back to 4096
    del wo

    write_arr(outdir, case, "x_rows", x_rows)
    write_arr(outdir, case, "q_rope", q.astype(np.float32))
    write_arr(outdir, case, "k_cache", kcache.reshape(P, N_KV_SWA * K_DIM).astype(np.float32))
    write_arr(outdir, case, "v_cache", vcache.reshape(P, N_KV_SWA * V_DIM).astype(np.float32))
    write_arr(outdir, case, "first", np.array([w_first], np.int32))
    write_arr(outdir, case, "attn_heads", attn_heads.astype(np.float32))
    write_arr(outdir, case, "attn_out", attn_out)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root, two levels above terminal/tools/
    model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "engine", "tests", "naive", "fixtures")
    os.makedirs(outdir, exist_ok=True)

    self_test_decoders()
    m = Model(model)
    md, pre = m.files[0].metadata, "naive_n05_flash."
    assert md[pre + "attention.value_scale"] == VALUE_SCALE, md[pre + "attention.value_scale"]
    assert md[pre + "attention.sliding_window"] == WINDOW
    assert md[pre + "rope.freq_base_swa"] == THETA_SWA
    assert md[pre + "attention.head_count_kv"][1] == N_KV_SWA   # layer 1 is a SWA layer
    print(f"opened {len(m.paths)} shards, {len(m.index)} tensors")

    rng = np.random.default_rng(2026)                  # fixed, like the kernel oracle's seeds
    x = (rng.standard_normal(N_EMBD) * 1.4).astype(np.float32)

    build_gemv_probe(m, x, outdir)
    print("gemv_probe:  x @ Q8_0(blk.1.attn_v)")

    build_dense(m, x, outdir)
    print("layer0_dense: rmsnorm -> SwiGLU -> down (Q8_0)")

    # swa_small fits the window (T=8); swa_window exceeds it (T=140, first=13) to exercise truncation.
    for case, P in (("swa_small", 9), ("swa_window", 141)):
        rows = (rng.standard_normal((P, N_EMBD)) * 1.4).astype(np.float32)
        build_swa(m, rows, outdir, case)
        print(f"{case}: P={P}, window truncates={P > WINDOW}")


if __name__ == "__main__":
    main()
