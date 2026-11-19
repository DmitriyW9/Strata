#!/usr/bin/env python3
# tools/naive_real_moe_dsa_fixture.py - a reference forward of the three pieces docs/NAIVE.md still
# leaves open - the IQ2 MoE FFN, the DSA (indexer) layer, and the two-layer trunk - from the model's
# own quantised weights, for tests/naive/test_naive_moe_dsa.cpp.
#
#   moe_layer1  layer 1's MoE FFN for one token: rmsnorm (training rounding) -> router over the REAL
#               F32 gate/bias -> top-8 -> the FIRST selected expert's SwiGLU over IQ2_XXS gate/up and
#               IQ2_XS down slices read by byte-window -> combine (bf16) with the router weights.
#   dsa_layer0  layer 0 for one query row over P cache rows: k/v/indexer-key projections, the indexer
#               key LayerNorm + rope + E4M3 row quantisation (the selection cache), the index-query
#               projection + rope + E4M3, the score weight w = (xn @ w_proj) * 16^-0.5, the selection
#               logits, the top-k pick (a small k; the arange rule when visible <= index_top_k), and
#               the grouped softmax attention over the visible rows.
#   trunk       a two-token sequence end to end: two hidden-state rows -> layer 0 (DSA attention +
#               dense SwiGLU FFN, both residuals) -> layer 1 (SWA attention over two rows + IQ2 MoE
#               FFN over all 8 routed experts, both residuals) -> output_norm -> logits against the
#               output head, compared by top-5 ids and values.  The token embedding row is read from
#               the 2.4 GB Q8_0 table as one 4096-element window and written as a fixture: it proves
#               that read, and it is the natural content for the sequence's second row.
#
# WHY THIS IS AN INDEPENDENT REFERENCE: the C++ candidate decodes every quantised weight with ggml's
# compiled `to_float` (loader.hpp) and runs kernels.hpp.  Here IQ2_XXS/IQ2_XS are decoded by the
# vectorised numpy transcriptions in tools/naive_iq2.py (self-tested bit-exactly against a literal
# line-by-line transcription of ggml's own decode loops), and Q8_0/Q6_K/BF16/F32 by the decoders in
# tools/naive_real_layer_fixture.py (likewise self-tested).  The element mapping, the sign tables
# and the scale nibbles are computed differently on the two sides, so agreement on real model bytes
# is a cross-check of ggml's decoder and the port's formulas, not a wrapper around either.
#
# The rounding graph transcribes NaiveRT, boundary by boundary:
#   * rmsnorm: models/naive_n0p5/training_moe.py:rmsnorm - the normalized vector rounds to bf16
#     BEFORE gamma multiplies it (training_moe is the model's own training spec, the only place
#     training rounding is written down; run_attn_swa's round-after-gamma differs by <=1 bf16 ulp).
#   * router: training_moe.py:route - fp32 sigmoid, top-k selects on (p + bias), weights are the
#     unbiased p renormalised by (sum + 1e-20).  The GGUF's noaux_tc layout keeps the gate weights
#     and the routing bias apart, exactly as the reference does.
#   * MoE combine: training_moe.py:combine - completed bf16 expert outputs, fp32 weighted sum,
#     rounded to bf16 once; the residual is added after.
#   * DSA: tests/gpu/attn_dsa/run_attn_dsa.py - rmsnorm, project -> bf16, V scaled by value_scale
#     folded into the bf16 round (reference: ((xn @ wv.T) * scale).bfloat16(); this GGUF ships DSA
#     attn_v at plain Q8_0, so the fold happens post-projection here, the choice the SWA fixtures
#     already make), indexer keys = project -> bf16 -> LayerNorm (F32 weight/bias) -> bf16 ->
#     hf_rope(theta 1e7) -> bf16 -> E4M3 row quant (scale = amax/448, min 1e-30) for the SELECTION
#     cache; logits = sum_h relu(q_h . k_j) * w[h] over the fp8-dequantised rows; selection = top-k,
#     ascending indices (stable_topk's tie rule); visible <= index_top_k selects every row
#     (reference_selection's arange rule); attention = softmax(q k^T / sqrt(192)) over the visible
#     rows, grouped-query 64/4, no sink (reference_attention).  The reference quantises its INDEXER
#     WEIGHTS to fp8 from its checkpoint; this GGUF ships them as BF16, so both sides read bf16
#     values - the deviation is recorded in docs/NAIVE.md.
#   * dense FFN / attention / output head: the same boundaries test_naive_layer.cpp already pins
#     (projections -> bf16, silu*up -> bf16, residual after; logits = bf16(xn @ W^T), chunked).
#
# No expert stack is ever fully decoded: layer 1's IQ2 gate/up/down tensors are 6.4 G elements
# (~12 GiB), and every read here is ONE expert's byte window (8.4 M elements, ~2 MB) through
# Model.raw_range - the same window shape the C++ reader (gemv_dequant_3d) opens.  The head reads
# the output table in 8192-row chunks for the same reason.
#
#   python terminal/tools/naive_real_moe_dsa_fixture.py [model-shard-1.gguf] [outdir]
#     default model:  /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/...00001-of-00004.gguf
#     default outdir: engine/tests/naive/fixtures

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import BLOCK_GEOMETRY  # noqa: E402
from naive_iq2 import dequant_iq2_xxs, dequant_iq2_xs  # noqa: E402
from naive_kernels_oracle import (  # noqa: E402
    bf16_round, fp8_rows_dequant, index_key_layernorm, rope_head, stable_topk, swa_attention,
    write_arr)
from naive_real_layer_fixture import (  # noqa: E402
    DEFAULT_MODEL, DENSE_FF, K_DIM, N_EMBD, N_HEAD, N_KV_SWA, N_ROT, RMS_EPS, THETA_SWA, V_DIM,
    VALUE_SCALE, Model, dequant_q8_0, silu)

# Layer 0 / indexer / MoE geometry, all asserted against the shard metadata in main().
N_KV_DSA, THETA_DSA = 4, 1e7
INDEX_HEADS, INDEX_DIM, INDEX_TOPK = 16, 128, 2048
N_EXPERT, TOPK, EXPERT_FF = 256, 8, 2048
VOCAB = 152576
IQ2XXS_BB = BLOCK_GEOMETRY["IQ2_XXS"][1]   # 66 bytes / 256 elements
IQ2XS_BB = BLOCK_GEOMETRY["IQ2_XS"][1]     # 74 bytes / 256 elements
Q8_0_BB = BLOCK_GEOMETRY["Q8_0"][1]        # 34 bytes / 32 elements


def rmsnorm_training_np(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """training_moe.py:rmsnorm - (x * rstd) rounds to bf16, then bf16(gamma) multiplies, then rounds.

    The training path, not run_attn_swa's round-after-gamma: training_moe.py is the model's own
    training spec, and the only place its rounding is written down.  The two agree to 1 bf16 ulp.
    """
    x = np.asarray(x, np.float64)
    rstd = 1.0 / np.sqrt((x * x).mean() + RMS_EPS)
    normalized = bf16_round(x * rstd).astype(np.float32)
    return bf16_round(normalized.astype(np.float64)
                      * bf16_round(gamma).astype(np.float32).astype(np.float64)).astype(np.float32)


def project_f64(xn: np.ndarray, w: np.ndarray, n_embd: int, n_out: int, chunk: int = 4096) -> np.ndarray:
    """xn @ W.T in fp64, NO rounding.  project() in naive_real_layer_fixture.py is this with the
    bf16 round folded in; keeping the round outside lets a caller place it exactly where ITS
    reference puts it (the DSA V fold rounds once, after the scale)."""
    W = w.reshape(n_out, n_embd)
    y = np.empty(np.asarray(xn).shape[:-1] + (n_out,), np.float64)
    xn64 = np.asarray(xn, np.float64)
    for a in range(0, n_out, chunk):
        b = min(a + chunk, n_out)
        y[..., a:b] = xn64 @ np.ascontiguousarray(W[a:b].T.astype(np.float64))
    return y


def project_bf16(xn: np.ndarray, w: np.ndarray, n_embd: int, n_out: int) -> np.ndarray:
    """run_attn_swa.project: the fp64 GEMV rounded to bf16, returned as float32."""
    return bf16_round(project_f64(xn, w, n_embd, n_out)).astype(np.float32)


def rope_t(rows: np.ndarray, pos: int, theta: float) -> np.ndarray:
    """rope_head over every [head_dim] row of `rows` at one position, fp64 (theta per layer type)."""
    return rope_head(np.asarray(rows, np.float64), np.array([pos]), N_ROT, theta)


def expert_slice(m: Model, name: str, expert: int, type_name: str, n_in: int, n_out: int) -> np.ndarray:
    """One expert's slice of a {n_in, n_out, 256} IQ2 stack, decoded: [n_in * n_out] fp32, file order.

    The GGUF's own key says it ("C-order [expert, output, input]; GGUF dimensions [input, output,
    expert]" - quantization.tensor_layout metadata), so expert e is the element range
    [e*n_in*n_out, +n_in*n_out) and a row m inside it is elements [m*n_in, +n_in) - the same window
    the C++ gemv_dequant_3d opens."""
    be, bb = 256, (IQ2XXS_BB if type_name == "IQ2_XXS" else IQ2XS_BB)
    raw = m.raw_range(name, expert * n_in * n_out, n_in * n_out, be, bb)
    dec = dequant_iq2_xxs if type_name == "IQ2_XXS" else dequant_iq2_xs
    return dec(raw, n_in * n_out)


# ------------------------------------------------------------------ the DSA layer's caches
def dsa_caches(m: Model, xns: np.ndarray, outdir: str, case: str) -> tuple:
    """Layer 0's per-row caches for every (already normed) row of `xns`, plus the last row's query,
    selection and attention.  Returns (k_all, v_all, attn_flat); writes the case's fixtures.

    The weights are read by name here rather than passed in: every weight is used exactly once per
    case and the shard's page cache keeps the second case cheap, which matters because a full BF16
    indexer weight and a Q6_K Q weight together are ~130 MB of live temporaries."""
    P = xns.shape[0]
    wk, wv = m.weight_f32("blk.0.attn_k.weight"), m.weight_f32("blk.0.attn_v.weight")
    wik = m.weight_f32("blk.0.indexer.k_proj.weight")
    ln_w, ln_b = m.weight_f32("blk.0.indexer.k_norm.weight"), m.weight_f32("blk.0.indexer.k_norm.bias")
    k_all = np.empty((P, N_KV_DSA * K_DIM), np.float64)
    v_all = np.empty((P, N_KV_DSA * V_DIM), np.float64)
    ik_ln_all = np.empty((P, INDEX_DIM), np.float64)
    for p in range(P):
        # k: bf16 after the projection, rope(theta 1e7) per 4 kv heads of 192 (reference_caches:
        # k = project(...); k[:, :ROPE] = hf_rope(...)).
        k = bf16_round(project_f64(xns[p], wk, N_EMBD, N_KV_DSA * K_DIM))
        k_all[p] = bf16_round(rope_t(k.reshape(N_KV_DSA, K_DIM), p, THETA_DSA)).reshape(-1)
        # v: projection * value_scale, rounded once (reference: ((xn @ wv.T) * scale).bfloat16()).
        v_all[p] = bf16_round(project_f64(xns[p], wv, N_EMBD, N_KV_DSA * V_DIM) * VALUE_SCALE)
        # indexer keys: bf16, LayerNorm(F32 affine), bf16, rope, bf16 - all four boundaries are in
        # reference_caches: project->bf16, layer_norm(...).to(bf16), hf_rope, the return .to(bf16).
        ik = bf16_round(project_f64(xns[p], wik, N_EMBD, INDEX_DIM)).astype(np.float32)
        ik = bf16_round(index_key_layernorm(ik, ln_w, ln_b)).astype(np.float32)
        ik_ln_all[p] = bf16_round(rope_t(ik.reshape(1, INDEX_DIM), p, THETA_DSA)[0])
    del wk, wv, wik
    ik_bf16 = bf16_round(ik_ln_all).astype(np.float32)     # cache rows are bf16 before fp8
    ki_deq, ki_scales, _ = fp8_rows_dequant(ik_bf16)       # E4M3 rows: amax/448, min 1e-30

    T = P - 1
    wq = m.weight_f32("blk.0.attn_q.weight")
    q = bf16_round(project_f64(xns[T], wq, N_EMBD, N_HEAD * K_DIM)).astype(np.float32)
    q_rope = bf16_round(rope_t(q.reshape(N_HEAD, K_DIM), T, THETA_DSA)).astype(np.float32)
    del wq
    wiq = m.weight_f32("blk.0.indexer.q_proj.weight")
    iq = bf16_round(project_f64(xns[T], wiq, N_EMBD, INDEX_HEADS * INDEX_DIM)).astype(np.float32)
    iq_deq, iq_scales, _ = fp8_rows_dequant(
        bf16_round(rope_t(iq.reshape(INDEX_HEADS, INDEX_DIM), T, THETA_DSA)).astype(np.float32))
    wis = m.weight_f32("blk.0.indexer.proj.weight")
    w = project_f64(xns[T], wis, N_EMBD, INDEX_HEADS) * INDEX_HEADS ** -0.5   # fold, run_attn_dsa
    del wiq, wis

    # Selection logits over the E4M3-dequantised rows (reference_selection's logits line).
    scores = ki_deq @ iq_deq.T                             # [visible, 16]
    logits = (np.maximum(scores, 0.0) * w[None, :]).sum(-1)
    sel_small = stable_topk(logits.astype(np.float32), 10)             # exercises the selection rule
    sel_all = np.arange(P, dtype=np.int32)                 # visible (P) <= index_top_k: every row

    # Grouped softmax attention over the visible rows, 64 q heads / 4 kv, no sink (reference_attention;
    # swa_attention with sink=None is the same formula).
    out = np.empty((N_HEAD, V_DIM), np.float64)
    group = N_HEAD // N_KV_DSA
    for g in range(N_KV_DSA):
        sl = slice(g * group, (g + 1) * group)
        out[sl] = swa_attention(q_rope[sl].astype(np.float64), k_all[:, g * K_DIM:(g + 1) * K_DIM],
                                v_all[:, g * V_DIM:(g + 1) * V_DIM], None, 0, len(sel_all) - 1)
    attn_flat = bf16_round(out.reshape(-1)).astype(np.float32)

    write_arr(outdir, case, "k_cache", k_all.astype(np.float32))
    write_arr(outdir, case, "v_cache", v_all.astype(np.float32))
    write_arr(outdir, case, "ki_deq", ki_deq.astype(np.float32))
    write_arr(outdir, case, "ki_scales", ki_scales.astype(np.float32))
    write_arr(outdir, case, "q_rope", q_rope)
    write_arr(outdir, case, "iq_deq", iq_deq.astype(np.float32))
    write_arr(outdir, case, "iq_scales", iq_scales.astype(np.float32))
    write_arr(outdir, case, "w", w.astype(np.float32))
    write_arr(outdir, case, "logits", logits.astype(np.float32))
    write_arr(outdir, case, "sel_small", sel_small.astype(np.int32))
    write_arr(outdir, case, "attn_out", attn_flat)
    return k_all, v_all, attn_flat


# ------------------------------------------------------------------ the three cases
def moe_layer1(m: Model, x: np.ndarray, outdir: str) -> int:
    """One token's MoE FFN: training-rounding rmsnorm, the real router, one expert's IQ2 SwiGLU, combine."""
    gamma = m.weight_f32("blk.1.ffn_norm.weight")
    xn = rmsnorm_training_np(x, gamma)
    gate_w = m.weight_f32("blk.1.ffn_gate_inp.weight")     # F32 {4096, 256}: expert e = row e
    bias = m.weight_f32("blk.1.exp_probs_b.bias")          # F32 {256}
    scores = xn.astype(np.float64) @ np.ascontiguousarray(
        gate_w.reshape(N_EXPERT, N_EMBD).T.astype(np.float64))
    p = 1.0 / (1.0 + np.exp(-scores))
    idx = stable_topk(p + bias, TOPK)                      # (p + bias) decides WHICH experts
    sel = p[idx]
    wts = (sel / (sel.sum() + 1e-20)).astype(np.float32)   # the unbiased p, renormalised (training_moe)

    e0 = int(idx[0])
    gate = project_bf16(xn, expert_slice(m, "blk.1.ffn_gate_exps.weight", e0, "IQ2_XXS",
                                         N_EMBD, EXPERT_FF), N_EMBD, EXPERT_FF)
    up = project_bf16(xn, expert_slice(m, "blk.1.ffn_up_exps.weight", e0, "IQ2_XXS",
                                       N_EMBD, EXPERT_FF), N_EMBD, EXPERT_FF)
    act = bf16_round(silu(gate) * up)                      # silu(g)*u rounded bf16 (training_moe)
    down = project_bf16(act, expert_slice(m, "blk.1.ffn_down_exps.weight", e0, "IQ2_XS",
                                          EXPERT_FF, N_EMBD), EXPERT_FF, N_EMBD)
    expert_out = down                                      # the "completed expert": bf16-rounded
    moe_out = bf16_round(expert_out.astype(np.float64) * float(wts[0])).astype(np.float32)

    write_arr(outdir, "moe_layer1", "x", x)
    write_arr(outdir, "moe_layer1", "xn", xn)
    write_arr(outdir, "moe_layer1", "route_idx", idx.astype(np.int32))
    write_arr(outdir, "moe_layer1", "route_wts", wts)
    write_arr(outdir, "moe_layer1", "expert_id", np.array([e0], np.int32))
    write_arr(outdir, "moe_layer1", "expert_gate", gate)
    write_arr(outdir, "moe_layer1", "expert_up", up)
    write_arr(outdir, "moe_layer1", "expert_act", act)
    write_arr(outdir, "moe_layer1", "expert_out", expert_out)
    write_arr(outdir, "moe_layer1", "moe_out", moe_out)
    return e0


def dsa_layer0(m: Model, x_rows: np.ndarray, outdir: str) -> None:
    """Standalone DSA case: norm the seeded rows, run the caches + selection + attention, write all."""
    gamma = m.weight_f32("blk.0.attn_norm.weight")
    xns = np.stack([rmsnorm_training_np(x_rows[p], gamma) for p in range(x_rows.shape[0])])
    write_arr(outdir, "dsa_layer0", "x_rows", x_rows)     # the C++ side rebuilds every row's xn from these
    write_arr(outdir, "dsa_layer0", "xn", xns[-1])         # the query row's normed input
    dsa_caches(m, xns, outdir, "dsa_layer0")


def moe_ffn(m: Model, xn2: np.ndarray) -> tuple:
    """All 8 routed experts of layer 1's MoE FFN (training_moe rounding): (idx, wts, moe_out)."""
    scores = xn2.astype(np.float64) @ np.ascontiguousarray(
        m.weight_f32("blk.1.ffn_gate_inp.weight").reshape(N_EXPERT, N_EMBD).T.astype(np.float64))
    p = 1.0 / (1.0 + np.exp(-scores))
    idx = stable_topk(p + m.weight_f32("blk.1.exp_probs_b.bias"), TOPK)
    sel_p = p[idx]
    wts = (sel_p / (sel_p.sum() + 1e-20)).astype(np.float32)
    expert_outs = np.empty((TOPK, N_EMBD), np.float32)
    for t, e in enumerate(idx):
        gate = project_bf16(xn2, expert_slice(m, "blk.1.ffn_gate_exps.weight", int(e), "IQ2_XXS",
                                              N_EMBD, EXPERT_FF), N_EMBD, EXPERT_FF)
        up = project_bf16(xn2, expert_slice(m, "blk.1.ffn_up_exps.weight", int(e), "IQ2_XXS",
                                            N_EMBD, EXPERT_FF), N_EMBD, EXPERT_FF)
        act = bf16_round(silu(gate) * up)
        expert_outs[t] = project_bf16(act, expert_slice(m, "blk.1.ffn_down_exps.weight", int(e),
                                                        "IQ2_XS", EXPERT_FF, N_EMBD), EXPERT_FF, N_EMBD)
    moe_out = bf16_round((expert_outs.astype(np.float64) * wts[:, None]).sum(0)).astype(np.float32)
    return idx, wts, moe_out


def trunk_case(m: Model, token: int, outdir: str) -> None:
    """A two-token trunk (layer 0 DSA+dense, layer 1 SWA+MoE) from two hidden rows to logits.

    The trunk starts where a layer input starts - two hidden-state rows (a tiny prefill; a longer
    one changes row counts, not any formula under test) - so every norm reads the SAME row it must:
    the layer-0 attention norm reads the input row, the layer-0 FFN norm reads the post-attention
    row, and so on.  The embedding row is the sequence's second row; the first is a seeded hidden
    state standing in for token 0's (an embedding-to-embedding pass would add no formula)."""
    emb0 = dequant_q8_0(m.raw_range("token_embd.weight", token * N_EMBD, N_EMBD, 32, Q8_0_BB),
                        N_EMBD)                            # one vocab row, no full decode
    rng = np.random.default_rng(4242)
    row0 = (rng.standard_normal(N_EMBD) * 1.2).astype(np.float32)
    rows = np.stack([row0, emb0])

    # ---- layer 0: DSA attention (query at pos 1) + dense SwiGLU FFN, residuals after each
    g0 = m.weight_f32("blk.0.attn_norm.weight")
    xns0 = np.stack([rmsnorm_training_np(rows[p], g0) for p in range(2)])
    _, _, attn_out = dsa_caches(m, xns0, outdir, "trunk")
    attn_proj = project_bf16(attn_out, m.weight_f32("blk.0.attn_output.weight"), N_HEAD * V_DIM, N_EMBD)
    h1 = bf16_round(rows[1].astype(np.float64) + attn_proj.astype(np.float64)).astype(np.float32)

    xn1 = rmsnorm_training_np(h1, m.weight_f32("blk.0.ffn_norm.weight"))
    gate = project_bf16(xn1, m.weight_f32("blk.0.ffn_gate.weight"), N_EMBD, DENSE_FF)
    up = project_bf16(xn1, m.weight_f32("blk.0.ffn_up.weight"), N_EMBD, DENSE_FF)
    act = bf16_round(silu(gate) * up)
    ffn_out = project_bf16(act, m.weight_f32("blk.0.ffn_down.weight"), DENSE_FF, N_EMBD)
    h2 = bf16_round(h1.astype(np.float64) + ffn_out.astype(np.float64)).astype(np.float32)

    # ---- layer 1: SWA attention (window 128, per-head sinks, theta 1e4) over the two rows, then
    # the IQ2 MoE FFN over all 8 routed experts, residuals after each.
    g1 = m.weight_f32("blk.1.attn_norm.weight")
    xn_swa = np.stack([rmsnorm_training_np(h1, g1), rmsnorm_training_np(h2, g1)])
    k1 = bf16_round(project_f64(xn_swa, m.weight_f32("blk.1.attn_k.weight"),
                                N_EMBD, N_KV_SWA * K_DIM)).astype(np.float32)
    v1 = bf16_round(project_f64(xn_swa, m.weight_f32("blk.1.attn_v.weight"),
                                N_EMBD, N_KV_SWA * V_DIM) * VALUE_SCALE).astype(np.float32)
    k1_rope = np.stack([bf16_round(rope_t(k1[p].reshape(N_KV_SWA, K_DIM), p, THETA_SWA)).reshape(-1)
                        for p in range(2)])
    q1 = bf16_round(project_f64(xn_swa[1], m.weight_f32("blk.1.attn_q.weight"),
                                N_EMBD, N_HEAD * K_DIM)).astype(np.float32)
    q1_rope = bf16_round(rope_t(q1.reshape(N_HEAD, K_DIM), 1, THETA_SWA))
    sink = m.weight_f32("blk.1.attn_sinks.weight")
    out1 = np.empty((N_HEAD, V_DIM), np.float64)
    g8 = N_HEAD // N_KV_SWA
    for g in range(N_KV_SWA):
        sl = slice(g * g8, (g + 1) * g8)
        out1[sl] = swa_attention(q1_rope[sl], k1_rope[:, g * K_DIM:(g + 1) * K_DIM],
                                 v1[:, g * V_DIM:(g + 1) * V_DIM], sink[sl], 0, 1)
    attn1_flat = bf16_round(out1.reshape(-1)).astype(np.float32)
    attn1 = project_bf16(attn1_flat, m.weight_f32("blk.1.attn_output.weight"), N_HEAD * V_DIM, N_EMBD)
    h3 = bf16_round(h2.astype(np.float64) + attn1.astype(np.float64)).astype(np.float32)

    xn2 = rmsnorm_training_np(h3, m.weight_f32("blk.1.ffn_norm.weight"))
    idx, wts, moe_out = moe_ffn(m, xn2)
    h4 = bf16_round(h3.astype(np.float64) + moe_out.astype(np.float64)).astype(np.float32)

    # ---- head: output_norm (training rounding) then bf16 logits, chunked so the 2.4 GB dequantised
    # head and its fp64 copy never sit in RAM beside the running engine.
    xnl = rmsnorm_training_np(h4, m.weight_f32("output_norm.weight"))
    logits = np.empty(VOCAB, np.float32)
    for a in range(0, VOCAB, 8192):
        b = min(a + 8192, VOCAB)
        w_rows = dequant_q8_0(m.raw_range("output.weight", a * N_EMBD, (b - a) * N_EMBD, 32, Q8_0_BB),
                              (b - a) * N_EMBD)
        logits[a:b] = bf16_round(xnl.astype(np.float64) @ np.ascontiguousarray(
            w_rows.reshape(b - a, N_EMBD).T.astype(np.float64)))
    top5 = np.argsort(-logits.astype(np.float64), kind="stable")[:5]

    write_arr(outdir, "trunk", "row0", row0)
    write_arr(outdir, "trunk", "emb", emb0)
    write_arr(outdir, "trunk", "xn0", xns0[1])
    write_arr(outdir, "trunk", "attn0_out", attn_out)
    write_arr(outdir, "trunk", "h1", h1)
    write_arr(outdir, "trunk", "ffn0_out", ffn_out)
    write_arr(outdir, "trunk", "h2", h2)
    write_arr(outdir, "trunk", "attn1_out", attn1_flat)
    write_arr(outdir, "trunk", "h3", h3)
    write_arr(outdir, "trunk", "route_idx", idx.astype(np.int32))
    write_arr(outdir, "trunk", "route_wts", wts)
    write_arr(outdir, "trunk", "moe_out", moe_out)
    write_arr(outdir, "trunk", "h4", h4)
    write_arr(outdir, "trunk", "xnl", xnl)
    write_arr(outdir, "trunk", "top_ids", top5.astype(np.int32))
    write_arr(outdir, "trunk", "top_vals", logits[top5])
    print(f"   trunk: top-1 id {int(top5[0])}, logits {np.round(logits[top5], 2).tolist()}")


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root, two levels above terminal/tools/
    model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "engine", "tests", "naive", "fixtures")
    os.makedirs(outdir, exist_ok=True)

    m = Model(model)
    md, pre = m.files[0].metadata, "naive_n05_flash."
    assert md[pre + "index_top_k"] == INDEX_TOPK, md[pre + "index_top_k"]
    assert md[pre + "index_n_heads"] == INDEX_HEADS and md[pre + "index_head_dim"] == INDEX_DIM
    assert md[pre + "rope.freq_base"] == THETA_DSA
    assert md[pre + "attention.head_count_kv"][0] == N_KV_DSA     # layer 0 is a DSA layer
    assert md[pre + "expert_count"] == N_EXPERT and md[pre + "expert_used_count"] == TOPK
    assert md[pre + "expert_feed_forward_length"] == EXPERT_FF
    assert md[pre + "quantization.tensor_layout"] == \
        "C-order [expert, output, input]; GGUF dimensions [input, output, expert]"
    print(f"opened {len(m.paths)} shards, {len(m.index)} tensors")

    rng = np.random.default_rng(777)
    x = (rng.standard_normal(N_EMBD) * 1.4).astype(np.float32)
    e0 = moe_layer1(m, x, outdir)
    print(f"moe_layer1: first expert {e0}: IQ2_XXS gate/up + IQ2_XS down, router top-{TOPK}")

    P = 24              # visible < index_top_k: attention covers every row; sel_small=10 tests selection
    x_rows = (rng.standard_normal((P, N_EMBD)) * 1.3).astype(np.float32)
    dsa_layer0(m, x_rows, outdir)
    print(f"dsa_layer0: {P} cache rows, selection k=10 over fp8 rows, attention over all visible")

    trunk_case(m, 5000, outdir)
    print("trunk: two rows -> DSA+dense -> SWA+MoE -> head")


if __name__ == "__main__":
    main()
