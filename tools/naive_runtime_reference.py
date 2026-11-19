#!/usr/bin/env python3
# tools/naive_runtime_reference.py - an INDEPENDENT numpy reference for the whole naive trunk loop,
# run over the synthetic model of tests/naive/test_naive_runtime.cpp.  The runtime
# (include/strata/naive/runtime.hpp) is the STATE between rows - the DSA/window caches, the session
# rewind, the selection, the sampler - and that state is what no per-layer fixture covers.  This
# tool answers the same questions the C++ runtime answers (per-row DSA selections, the prompt
# logits, the next greedy token) by walking the same GGUF bytes with the vectorised numpy decoders,
# sharing no code with the runtime beyond the byte-window conventions every fixture tool already
# follows (tools/naive_real_moe_dsa_fixture.py's expert windows, tools/naive_real_layer_fixture.py's
# column/byte views).
#
# ACCUMULATION ORDER.  Random synthetic weights make the logits ill-conditioned sums of 4096
# cancelling products, where a last-bit difference in the dot - BLAS blocks its reductions, the
# C++ sums each row serially - flips a bf16 rounding and moves a logit by tens of percent.  So
# every sum in this file mirrors the C++ order exactly: dots accumulate one k at a time, the k-loop
# OUTERMOST and vectorised across output rows (each element is then its own independent IEEE
# addition, in ascending k, bit-for-bit what the scalar C++ loop does).  Values the comparison
# reads as discrete decisions (selections, router picks) and the logits themselves all match to
# that standard; what a random-weight head leaves between the two is then only the codec's own
# last bit, far inside the tolerance.
#
# The rounding graph mirrors the tested references, same as the C++ does: rmsnorm_training (round
# normalised, then gamma), bf16 after every projection, K/V cached post-rope/post-fold bf16, the
# index key through LayerNorm->bf16->rope->bf16->E4M3->dequant, attention fp64 (the fp reference
# form the kernel tests accept), residuals bf16(f64 add), the MoE combine fp64->bf16, logits bf16.
#
# Cases written (Arr format, the same files fixture_io.hpp reads):
#   p1__logits / p1__next            the 4-token prompt's logits and its greedy token
#   p1__sel_l{L}_r{R} / p1__moe_l{L}_r{R}    per-row DSA selections and router picks, per layer
#   p1__logits_step2 / p1__next_step2  continuing the session with p1__next (the runtime decodes
#                                    this from its caches; this tool just feeds another row)
#   p3__sel_l{L}_r4 / p3__moe_l{L}_r4  the decisions of that continued row
#   p2__logits / p2__sel_* / p2__moe_*   a fresh session for the branching prompt [10, 200, 999]
# `... <gguf> <outdir> xdump` prints the per-layer hidden-state fingerprints of the branch prompt,
# matching the test's xdump case (debug aid for a divergence, not part of the ctest path).
#
#   python3 tools/naive_runtime_reference.py <synthetic.gguf> <outdir> [xdump]

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile                       # noqa: E402
from naive_kernels_oracle import bf16_round, fp8_decode, fp8_encode, write_arr   # noqa: E402
from naive_real_layer_fixture import dequant_q6_K, dequant_q8_0        # noqa: E402
from naive_iq2 import dequant_iq2_xxs, dequant_iq2_xs                  # noqa: E402

PREFIX = "naive_n05_flash."
_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
_LIBM.expf.restype = ctypes.c_float
_LIBM.expf.argtypes = [ctypes.c_float]


def expf(g: np.ndarray) -> np.ndarray:
    """libm's expf, element for element - the same call C++'s silu makes (numpy's f32 exp is its
    own vectorised polynomial and can differ by an ulp, which is exactly what this file rules out)."""
    return np.array([_LIBM.expf(ctypes.c_float(float(v))) for v in np.asarray(g).reshape(-1)],
                    np.float32).reshape(np.shape(g))


def bf16(x) -> np.ndarray:
    """The C++'s bf16_round(float(x)): narrow to f32 first (the argument conversion), then round."""
    return bf16_round(x)


class File:
    """One GGUF: the metadata, the tensor directory, and element/byte-window reads off the handle."""

    def __init__(self, path: pathlib.Path):
        self.gf = GGUFFile(path)
        self.path = path
        self.index = {t.name: t for t in self.gf.tensors}
        self.fh = path.open("rb")

    def meta(self, key: str, default=None):
        v = self.gf.metadata.get(PREFIX + key, self.gf.metadata.get(key, default))
        if v is None:
            raise SystemExit(f"the file carries no {PREFIX + key}")
        return v

    def raw_elems(self, name: str, elem_start: int, n_elems: int) -> np.ndarray:
        t = self.index[name]
        be, bb = BLOCK_GEOMETRY[t.type_name]
        assert elem_start % be == 0 and n_elems % be == 0, f"{name}: window not whole blocks"
        at = self.gf.data_start + t.offset + elem_start // be * bb
        self.fh.seek(at)
        raw = self.fh.read(n_elems // be * bb)
        if len(raw) != n_elems // be * bb:
            raise SystemExit(f"{name}[{elem_start}:{n_elems}] runs past the file")
        return np.frombuffer(raw, np.uint8)

    def decoded(self, name: str, elem_start: int = 0, n_elems: int | None = None) -> np.ndarray:
        """A byte window decoded to fp32 in FILE order (dim 0 fastest)."""
        t = self.index[name]
        n = t.elements if n_elems is None else n_elems
        raw = self.raw_elems(name, elem_start, n)
        if t.type_name == "F32":
            return np.frombuffer(raw[: n * 4].tobytes(), "<f4").copy()
        if t.type_name == "BF16":
            u = np.frombuffer(raw[: n * 2].tobytes(), "<u2").astype(np.uint32) << 16
            return u.view(np.float32).copy()
        if t.type_name == "Q8_0":
            return dequant_q8_0(raw, n)
        if t.type_name == "Q6_K":
            return dequant_q6_K(raw, n)
        if t.type_name == "IQ2_XXS":
            return dequant_iq2_xxs(raw, n)
        if t.type_name == "IQ2_XS":
            return dequant_iq2_xs(raw, n)
        raise SystemExit(f"{name}: no decoder for {t.type_name}")


def project(xn: np.ndarray, w: np.ndarray, n_out: int, round_bf16: bool) -> np.ndarray:
    """The C++ gemv's row loop mirrored: acc[m] += f64(x[k]) * f64(W[m, k]), k ascending, vectorised
    over m (each acc element is its own IEEE add chain - bit-identical to the scalar C++ order).
    The optional single bf16 round is the C++'s gemv_bf16 rounding the trunk wiring applies."""
    n_in = xn.shape[-1]
    W = np.ascontiguousarray(w.reshape(n_out, n_in))
    xn = np.asarray(xn, np.float32)
    acc = np.zeros(n_out, np.float64)
    for k in range(n_in):
        acc += np.float64(xn[k]) * W[:, k]
    y = acc.astype(np.float32)
    return bf16(y).astype(np.float32) if round_bf16 else y


def rmsnorm_training(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    """kernels.hpp::rmsnorm_training, line for line: serial f64 sum of squares, float rstd, then
    bf16(bf16(x*rstd) * bf16(gamma)) - the rounding the naive moe/dsa/trunk fixtures pin."""
    x = np.asarray(x, np.float32)
    x64 = x.astype(np.float64)
    sumsq = 0.0
    for v in x64:
        sumsq += float(v) * float(v)
    rstd = np.float32(1.0 / math.sqrt(sumsq / len(x) + eps))
    return bf16(bf16(x * rstd).astype(np.float32) * bf16(gamma)).astype(np.float32)


def silu_f32(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, np.float32)
    return (g / (np.float32(1.0) + expf(-g))).astype(np.float32)


def rope_rows(x: np.ndarray, n_head: int, head_dim: int, n_rot: int, pos: int, theta: float) -> np.ndarray:
    """kernels.hpp::rope_rows: split-half pairs of the first n_rot dims of each head row, float c/s
    from double angles, the rest copied."""
    x = np.asarray(x, np.float32).reshape(n_head, head_dim)
    out = x.copy()
    half = n_rot // 2
    for j in range(half):
        inv = 1.0 / math.pow(theta, float(2 * j) / n_rot)
        ang = pos * inv
        c = np.float32(math.cos(ang))
        s = np.float32(math.sin(ang))
        x1, x2 = x[:, j], x[:, j + half]
        out[:, j] = x1 * c - x2 * s
        out[:, j + half] = x2 * c + x1 * s
    return out.reshape(-1)


def fp8_rows_dequant(rows: np.ndarray, n: int):
    """kernels.hpp::fp8_rows_dequant per row: f32 amax (clamped 1e-30), f32 scale amax/448, each
    element encoded through the oracle's table search (independent of the C++'s frexp split) and
    decoded, rescaled in f32. Returns the dequantised rows."""
    rows = np.asarray(rows, np.float32).reshape(-1, n)
    out = np.empty_like(rows)
    for r in range(rows.shape[0]):
        row = rows[r]
        amax = float(np.max(np.abs(row))) if row.size else 0.0
        if not (amax > 1e-30):
            amax = 1e-30
        scale = np.float32(amax) / np.float32(448.0)
        codes = [fp8_encode(float(v) / float(scale)) for v in row]
        out[r] = (np.array([fp8_decode(c) for c in codes], np.float32) * scale).astype(np.float32)
    return out


def dsa_index_logits(iq: np.ndarray, w: np.ndarray, ki: np.ndarray, H: int, D: int, vis: int) -> np.ndarray:
    """kernels.hpp::dsa_index_logits: per visible row, a FLOAT32 dot per head (ascending d), relu,
    accumulated per head in double with w[h]."""
    out = np.empty(vis, np.float32)
    for j in range(vis):
        kj = np.asarray(ki[j], np.float32)
        acc = 0.0
        for h in range(H):
            qh = np.asarray(iq[h * D:(h + 1) * D], np.float32)
            dot = np.float32(0.0)
            for d in range(D):
                dot = np.float32(dot + qh[d] * kj[d])
            if dot > 0:
                acc += float(dot) * float(w[h])
        out[j] = np.float32(acc)
    return out


def stable_topk(v: np.ndarray, k: int) -> np.ndarray:
    """kernels.hpp::stable_topk's exact rule as one sort: (value desc, index asc), winners ascending,
    NaN never wins (the C++'s guard)."""
    v = np.asarray(v, np.float64)
    n = len(v)
    order = [i for i in range(n) if v[i] == v[i]]
    order.sort(key=lambda i: (-v[i], i))
    return np.array(sorted(order[:min(k, len(order))]), np.int32)


def moe_route(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, topk: int):
    """kernels.hpp::moe_route mirrored: serial f64 dot per expert (ascending k, vectorised over
    experts), sigmoid in double narrowed to f32, the pick by f32 (p+bias) with the stable rule,
    the renormalisation in double over the picked order, weights narrowed to f32."""
    x = np.asarray(x, np.float32)
    W = np.ascontiguousarray(np.asarray(weight, np.float32))
    n_expert = W.shape[0]
    acc = np.zeros(n_expert, np.float64)
    for k in range(x.shape[-1]):
        acc += np.float64(x[k]) * W[:, k]
    p = np.array([np.float32(1.0 / (1.0 + math.exp(-a))) for a in acc], np.float32)
    key = (p + np.asarray(bias, np.float32)).astype(np.float32)
    idx = stable_topk(key, topk)
    s = 0.0
    for t in idx:
        s += float(p[t])
    inv = 1.0 / (s + 1e-20)
    return idx, np.array([np.float32(float(p[t]) * inv) for t in idx], np.float32)


def index_key_layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """kernels.hpp::index_key_layernorm: serial f64 mean/var, float rstd, float affine."""
    x = np.asarray(x, np.float32)
    n = len(x)
    s = 0.0
    for v in x:
        s += float(v)
    mean = np.float32(s / n)
    vs = 0.0
    for v in x:
        d = float(np.float32(v - mean))
        vs += d * d
    rstd = np.float32(1.0 / math.sqrt(vs / n + eps))
    return (np.asarray(bias, np.float32) + (x - mean) * rstd * np.asarray(weight, np.float32)).astype(np.float32)


def swa_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, sink, K: int, V: int, n_head: int,
                  n_rows: int) -> np.ndarray:
    """kernels.hpp::swa_attention line for line: three serial f64 passes over the keys (running max
    with the sink included, then the denominator starting from the sink's exp, then the weighted V
    in a double accumulator), narrowed to f32 per head."""
    q = np.asarray(q, np.float32).reshape(n_head, K)
    k = np.asarray(k, np.float32).reshape(n_rows, K)
    v = np.asarray(v, np.float32).reshape(n_rows, V)
    scale = 1.0 / math.sqrt(K)
    out = np.empty(n_head * V, np.float32)
    for h in range(n_head):
        qh = q[h]
        scores = []
        for j in range(n_rows):
            kj = k[j]
            acc = 0.0
            for d in range(K):
                acc += float(qh[d]) * float(kj[d])
            scores.append(acc * scale)
        maxlog = float(sink[h]) if sink is not None else -math.inf
        for s in scores:
            if s > maxlog:
                maxlog = s
        denom = math.exp(float(sink[h]) - maxlog) if sink is not None else 0.0
        for s in scores:
            denom += math.exp(s - maxlog)
        inv = 1.0 / denom
        acc_v = [0.0] * V
        for j, s in enumerate(scores):
            wgt = math.exp(s - maxlog) * inv
            vj = v[j]
            for d in range(V):
                acc_v[d] += wgt * float(vj[d])
        out[h * V:(h + 1) * V] = np.array(acc_v, np.float64).astype(np.float32)
    return out


def head_logits(m: File, xn: np.ndarray, n_embd: int, n_vocab: int, chunk: int = 8192) -> np.ndarray:
    """The output head with the same serial-k order as gemv_raw, vectorised over output rows: the
    vocabulary in chunks (one double accumulator per vocab row, stepped k ascending), rounded bf16
    like gemv_bf16.  A vocab row v is the element range [v*n_embd, +n_embd) of the GGUF order."""
    xn = np.asarray(xn, np.float32)
    out = np.empty(n_vocab, np.float32)
    for a in range(0, n_vocab, chunk):
        b = min(a + chunk, n_vocab)
        W = np.ascontiguousarray(m.decoded("output.weight", a * n_embd, (b - a) * n_embd)
                                 .reshape(b - a, n_embd))
        acc = np.zeros(b - a, np.float64)
        for k in range(n_embd):
            acc += np.float64(xn[k]) * W[:, k]
        out[a:b] = bf16(acc).astype(np.float32)
    return out


class Session:
    """One running session: caches plus the row-by-row forward of the whole model."""

    def __init__(self, m: File, geo: dict):
        self.m, self.G = m, geo
        self.pos = 0
        self.dbg = False                       # record x after every layer (the xdump mode);
        self.dbg_x: list[np.ndarray] = []      # dbg == 2 additionally records the named probes
        self.dbg_pts: list[tuple[str, np.ndarray]] = []
        self.dsa = {l: {"k": [], "v": [], "ki": []} for l in geo["dsa_layers"]}
        self.swa = {l: {"k": [], "v": []} for l in range(geo["n_layers"]) if l not in geo["dsa_layers"]}
        self.wcache: dict[str, np.ndarray] = {}
        self.expcache: dict[tuple[str, int], np.ndarray] = {}
        self.last_sel: dict[int, np.ndarray] = {}
        self.last_moe: dict[int, np.ndarray] = {}

    def w(self, name: str) -> np.ndarray:
        if name not in self.wcache:
            self.wcache[name] = self.m.decoded(name)
        return self.wcache[name]

    def expert(self, name: str, e: int, n_in: int, n_out: int) -> np.ndarray:
        key = (name, e)
        if key not in self.expcache:
            self.expcache[key] = self.m.decoded(name, e * n_in * n_out, n_in * n_out)
        return self.expcache[key]

    def probe(self, name: str, v: np.ndarray):
        if self.dbg == 2:
            self.dbg_pts.append((name, np.asarray(v, np.float32)))

    def row(self, token: int, want_logits: bool):
        G = self.G
        x = self.m.decoded("token_embd.weight", token * G["n_embd"], G["n_embd"]).astype(np.float32)
        self.probe("emb", x)
        for l in range(G["n_layers"]):
            x = self.layer(l, x)
            if self.dbg:
                self.dbg_x.append(np.asarray(x, np.float32))
        if want_logits:
            xn = rmsnorm_training(x, self.w("output_norm.weight"), G["eps"])
            self.logits = head_logits(self.m, xn, G["n_embd"], G["n_vocab"])
        self.pos += 1

    def layer(self, l: int, x: np.ndarray) -> np.ndarray:
        G, dsa = self.G, l in self.G["dsa_layers"]
        nkv, group = G["kv"][l], G["n_head"] // G["kv"][l]
        theta = G["theta_full"] if dsa else G["theta_swa"]
        pos = self.pos
        xn = rmsnorm_training(x, self.w(f"blk.{l}.attn_norm.weight"), G["eps"])
        self.probe("xn", xn)

        k = project(xn, self.w(f"blk.{l}.attn_k.weight"), nkv * G["kd"], True)
        k = bf16(rope_rows(k, nkv, G["kd"], G["n_rot"], pos, theta)).astype(np.float32)
        self.probe("k", k)
        vraw = project(xn, self.w(f"blk.{l}.attn_v.weight"), nkv * G["vd"], False)
        v = bf16(vraw.astype(np.float64) * G["value_scale"]).astype(np.float32)
        self.probe("v", v)
        if dsa:
            self.dsa[l]["k"].append(k)
            self.dsa[l]["v"].append(v)
        else:
            self.swa[l]["k"].append(k)
            self.swa[l]["v"].append(v)

        if dsa:                                                    # the index key, cached dequantised
            D = G["idx_d"]
            ik = project(xn, self.w(f"blk.{l}.indexer.k_proj.weight"), D, True)
            ln = index_key_layernorm(ik, self.w(f"blk.{l}.indexer.k_norm.weight"),
                                     self.w(f"blk.{l}.indexer.k_norm.bias"), G["eps"])
            ln = bf16(ln).astype(np.float32)
            lnr = bf16(rope_rows(ln, 1, D, min(G["n_rot"], D), pos, theta)).astype(np.float32)
            self.probe("ik", lnr)
            ki = fp8_rows_dequant(lnr, D)[0]
            self.probe("ikdeq", ki)
            self.dsa[l]["ki"].append(ki)

        q = project(xn, self.w(f"blk.{l}.attn_q.weight"), G["n_head"] * G["kd"], True)
        q = bf16(rope_rows(q, G["n_head"], G["kd"], G["n_rot"], pos, theta)).astype(np.float32)
        self.probe("q", q)

        # Which rows this query reads (the runtime's rules): DSA = stable top-k over [0..pos], the
        # identity at or below top_k (NaiveRT's rule, published without computing the index query);
        # window = [pos-window+1, pos] (adapters.py "a query at p reads [p - window + 1, p]").
        if dsa:
            vis = pos + 1
            if vis > G["top_k"]:
                IH = G["idx_h"]
                iq = project(xn, self.w(f"blk.{l}.indexer.q_proj.weight"), IH * D, True)
                iqr = bf16(rope_rows(iq, IH, D, min(G["n_rot"], D), pos, theta)).astype(np.float32)
                iqdeq = fp8_rows_dequant(iqr.reshape(IH * D), D)
                wraw = project(xn, self.w(f"blk.{l}.indexer.proj.weight"), IH, False)
                wf = (wraw.astype(np.float64) * (1.0 / math.sqrt(IH))).astype(np.float32)
                sel = stable_topk(dsa_index_logits(np.asarray(iqdeq).reshape(IH * D), wf,
                                                   np.stack(self.dsa[l]["ki"]), IH, D, vis), G["top_k"])
            else:
                sel = np.arange(vis, dtype=np.int32)
            self.last_sel[l] = sel.astype(np.int32)
            krows = [self.dsa[l]["k"][i] for i in sel]
            vrows = [self.dsa[l]["v"][i] for i in sel]
            sink = None
        else:
            first = max(0, pos - G["window"] + 1)
            rows = list(range(first, pos + 1))
            krows = [self.swa[l]["k"][i] for i in rows]
            vrows = [self.swa[l]["v"][i] for i in rows]
            sink = self.w(f"blk.{l}.attn_sinks.weight") if G["add_swa_sink"] else None

        attn = np.empty(G["n_head"] * G["vd"], np.float32)
        for g in range(nkv):
            kg = np.stack([r.reshape(nkv, G["kd"])[g] for r in krows])
            vg = np.stack([r.reshape(nkv, G["vd"])[g] for r in vrows])
            sg = None if sink is None else np.asarray(sink[g * group:(g + 1) * group], np.float64)
            attn[g * group * G["vd"]:(g + 1) * group * G["vd"]] = swa_attention(
                q[g * group * G["kd"]:(g + 1) * group * G["kd"]], kg, vg, sg,
                G["kd"], G["vd"], group, len(krows))
        attn = bf16(attn.astype(np.float64)).astype(np.float32)
        self.probe("attn", attn)
        d = project(attn, self.w(f"blk.{l}.attn_output.weight"), G["n_embd"], True)
        self.probe("o", d)
        x = bf16(x.astype(np.float64) + d.astype(np.float64)).astype(np.float32)
        self.probe("xa", x)

        # ---- FFN: the dense layer 0, or router + IQ2 experts.
        h = rmsnorm_training(x, self.w(f"blk.{l}.ffn_norm.weight"), G["eps"])
        self.probe("h", h)
        if not G["moe"][l]:
            n_ff = G["ffn_len"][l]
            gate = project(h, self.w(f"blk.{l}.ffn_gate.weight"), n_ff, True)
            up = project(h, self.w(f"blk.{l}.ffn_up.weight"), n_ff, True)
            act = bf16(silu_f32(gate) * up).astype(np.float32)
            self.probe("act", act)
            d = project(act, self.w(f"blk.{l}.ffn_down.weight"), G["n_embd"], True)
            self.probe("dffn", d)
        else:
            w_r = self.w(f"blk.{l}.ffn_gate_inp.weight").reshape(G["n_expert"], G["n_embd"])
            idx, wts = moe_route(h, w_r, self.w(f"blk.{l}.exp_probs_b.bias"), G["n_expert_used"])
            self.last_moe[l] = idx.astype(np.int32)
            FE = G["ff_expert"]
            outs = np.empty((G["n_expert_used"], G["n_embd"]), np.float32)
            for t, e in enumerate(idx):
                gate = project(h, self.expert(f"blk.{l}.ffn_gate_exps.weight", int(e), G["n_embd"], FE), FE, True)
                up = project(h, self.expert(f"blk.{l}.ffn_up_exps.weight", int(e), G["n_embd"], FE), FE, True)
                act = bf16(silu_f32(gate) * up).astype(np.float32)
                outs[t] = project(act, self.expert(f"blk.{l}.ffn_down_exps.weight", int(e), FE, G["n_embd"]),
                                  G["n_embd"], True)
            mixed = np.empty(G["n_embd"], np.float32)
            for i in range(G["n_embd"]):
                a = 0.0
                for t in range(G["n_expert_used"]):
                    a += float(wts[t]) * float(outs[t][i])
                mixed[i] = a
            d = bf16(mixed.astype(np.float64)).astype(np.float32)
        return bf16(x.astype(np.float64) + d.astype(np.float64)).astype(np.float32)


def read_geo(m: File) -> dict:
    geo = {
        "n_layers": int(m.meta("block_count")),
        "n_embd": int(m.meta("embedding_length")),
        "eps": float(m.meta("attention.layer_norm_rms_epsilon")),
        "n_head": int(m.meta("attention.head_count")),
        "kd": int(m.meta("attention.key_length")),
        "vd": int(m.meta("attention.value_length")),
        "window": int(m.meta("attention.sliding_window")),
        "value_scale": float(m.meta("attention.value_scale")),
        "add_swa_sink": bool(m.meta("attention.add_swa_sink")),
        "n_rot": int(m.meta("rope.dimension_count")),
        "theta_full": float(m.meta("rope.freq_base")),
        "theta_swa": float(m.meta("rope.freq_base_swa")),
        "top_k": int(m.meta("index_top_k")),
        "idx_d": int(m.meta("index_head_dim")),
        "idx_h": int(m.meta("index_n_heads")),
        "n_expert": int(m.meta("expert_count")),
        "n_expert_used": int(m.meta("expert_used_count")),
        "ff_expert": int(m.meta("expert_feed_forward_length")),
        "kv": [int(v) for v in m.meta("attention.head_count_kv")],
        "hybrid": [int(v) for v in m.meta("attention.hybrid_layer_pattern")],
        "moe": [bool(v) for v in m.meta("expert.layer_pattern")],
        "ffn_len": [int(v) for v in m.meta("feed_forward_length")],
        "n_vocab": int(m.index["output.weight"].shape[1]),
    }
    geo["dsa_layers"] = [l for l in range(geo["n_layers"]) if geo["hybrid"][l] == 0]
    return geo


def xdump(m: File, pts: bool = False) -> int:
    """Debug mode: run the branch prompt and fingerprint x after every layer (xor is bit-exact
    between the two implementations; the sum is printed with slack).  Matches the test's xdump."""
    s = Session(m, read_geo(m))
    s.dbg = 2 if pts else True
    for tok in (10, 200, 999):
        s.dbg_x.clear()
        s.dbg_pts.clear()
        s.row(tok, want_logits=False)
        if pts:
            for name, v in s.dbg_pts:
                vv = np.asarray(v, np.float32).reshape(-1)
                xor = int(np.bitwise_xor.reduce(vv.view(np.uint32)))
                print("%s %08x %.6f" % (name, xor, float(np.sum(vv.astype(np.float64)))))
            continue
        for x in s.dbg_x:
            xor = int(np.bitwise_xor.reduce(np.asarray(x, np.float32).view(np.uint32)))
            print("xor %08x sum %.6f" % (xor, float(np.sum(np.asarray(x, np.float64)))))
    return 0


def main() -> int:
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(__doc__)
        return 2
    m = File(pathlib.Path(sys.argv[1]))
    if len(sys.argv) == 4 and sys.argv[3] in ("xdump", "xdbg2"):
        return xdump(m, pts=sys.argv[3] == "xdbg2")
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    geo = read_geo(m)
    moe_layers = [l for l in range(geo["n_layers"]) if geo["moe"][l]]

    t0 = time.time()
    p1 = [10, 200, 3000, 40000]
    s = Session(m, geo)
    for i, tok in enumerate(p1):
        s.row(tok, want_logits=(i + 1 == len(p1)))
        for l in geo["dsa_layers"]:
            write_arr(outdir, "p1", f"sel_l{l}_r{i}", s.last_sel[l])
        for l in moe_layers:
            write_arr(outdir, "p1", f"moe_l{l}_r{i}", s.last_moe[l])
    write_arr(outdir, "p1", "logits", s.logits.reshape(1, -1))
    next1 = int(np.argmax(s.logits))
    write_arr(outdir, "p1", "next", np.array([next1], np.int32))
    # continued row: the prompt + its greedy token, one more row (the runtime does this from its
    # caches; from scratch here it is just the next position).
    s.row(next1, want_logits=True)
    write_arr(outdir, "p1", "logits_step2", s.logits.reshape(1, -1))
    write_arr(outdir, "p1", "next_step2", np.array([int(np.argmax(s.logits))], np.int32))
    for l in geo["dsa_layers"]:
        write_arr(outdir, "p3", f"sel_l{l}_r4", s.last_sel[l])
    for l in moe_layers:
        write_arr(outdir, "p3", f"moe_l{l}_r4", s.last_moe[l])
    print(f"p1 done in {time.time() - t0:.1f}s, next={next1}")

    t1 = time.time()
    p2 = [10, 200, 999]
    s2 = Session(m, geo)
    for i, tok in enumerate(p2):
        s2.row(tok, want_logits=(i + 1 == len(p2)))
        for l in geo["dsa_layers"]:
            write_arr(outdir, "p2", f"sel_l{l}_r{i}", s2.last_sel[l])
        for l in moe_layers:
            write_arr(outdir, "p2", f"moe_l{l}_r{i}", s2.last_moe[l])
    write_arr(outdir, "p2", "logits", s2.logits.reshape(1, -1))
    print(f"p2 done in {time.time() - t1:.1f}s, next={int(np.argmax(s2.logits))}")
    print(f"wrote the runtime reference fixtures to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
