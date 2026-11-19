#!/usr/bin/env python3
# tools/naive_kernels_oracle.py - a numpy oracle for the Naive-N0.5-Flash trunk arithmetic, and a
# fixture generator that the C++ test (tests/naive/test_naive_kernels.cpp) reads.
#
# Why this exists: the model authors' runtime (NaiveRT) expresses the trunk in torch + sm_103a CUDA,
# which will not run on this machine (sm_86/sm_70, and torch is not installed). So the reference we
# can actually run here is a numpy transcription of NaiveRT's *own* torch references, kept as close to
# those functions as numpy allows. Each block below names the NaiveRT file and symbol it transcribes.
#
# It is independent of the C++ (include/strata/naive/kernels.hpp) on purpose: the oracle reaches the
# same numbers by different routes (a table search for FP8 where C++ uses frexp, pairwise reductions
# where C++ accumulates in double), so when the C++ test agrees with these fixtures, that is a real
# cross-check of two independent implementations, not one definition copied twice. The remaining gap
# - that this oracle matches the true torch kernel - is closed by reading (each block cites its
# source) and, on a box that can build NaiveRT, by running its own tests/gpu harnesses.
#
# Writes binary fixtures (an ASCII "dtype rows cols" header + raw little-endian values; see write_arr)
# that the C++ test reads at exact float32 bits, with `--print CASE` for a text view of any case.
#
#   python tools/naive_kernels_oracle.py [outdir]      (default: tests/naive/fixtures)

from __future__ import annotations

import os
import sys

import numpy as np

EPS = 1e-5
BF16_MIN = 1e-30
FP8_MAX = 448.0

# --------------------------------------------------------------------- io
# One array per file: an ASCII header line "<dtype> <rows> <cols>" (dtype 'f' float32, 'i' int32;
# flat arrays use rows=count, cols=1) followed by the raw little-endian values. The header stays
# human-scannable (head -1 shows the shape); the body is exact float32/int32, which is what lets a
# 1M-element router matrix round-trip to the C++ at its true bits without 17 MB of text. Regenerated
# deterministically (fixed seed), and `--print CASE` shows any case's arrays as text for review.
def write_arr(outdir: str, case: str, name: str, arr: np.ndarray) -> None:
    dtype, arr = ("f", np.asarray(arr, np.float32)) if arr.dtype.kind != "i" else ("i", np.asarray(arr, np.int32))
    a = arr.reshape(-1) if arr.ndim == 1 else arr
    rows, cols = (a.shape[0], a.shape[1] if a.ndim > 1 else 1)
    path = os.path.join(outdir, f"{case}__{name}.txt")
    raw = a.reshape(-1).astype("<f4" if dtype == "f" else "<i4").tobytes()
    with open(path, "wb") as fh:
        fh.write(f"{dtype} {rows} {cols}\n".encode())
        fh.write(raw)


def dump_case(outdir: str, case: str) -> None:
    """Print a case's arrays as text (index and value) for a reviewer, without committing bulk text."""
    prefix = f"{case}__"
    for path in sorted(p for p in os.listdir(outdir) if p.startswith(prefix)):
        with open(os.path.join(outdir, path), "rb") as fh:
            header = fh.readline().decode().split()
            dt, rows, cols = header[0], int(header[1]), int(header[2])
            n = rows * cols
            raw = fh.read()
        arr = np.frombuffer(raw, dtype="<f4" if dt == "f" else "<i4")
        print(f"{path}  {dt} {rows}x{cols}")
        show = arr[: min(n, 8)]
        print("   " + " ".join(f"{v:.6g}" if dt == "f" else str(v) for v in show) + (" ..." if n > 8 else ""))



def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even to bfloat16 and back to float32 (matches kernels.hpp::bf16_round)."""
    x = np.asarray(x, np.float32)
    b = x.view(np.uint32).copy()
    lsb = (b >> np.uint32(16)) & np.uint32(1)
    b = b + np.uint32(0x7FFF) + lsb
    b &= np.uint32(0xFFFF0000)
    return b.view(np.float32)


# --------------------------------------------------------------------- fp8 e4m3 (independent: table search)
# Build the 255 finite magnitudes of e4m3fn and their codes; encode(a>=0) is the nearest table value,
# ties to an even mantissa, saturating to 448 above the top midpoint. Deliberately a different
# algorithm than the C++ (which splits with frexp), so agreement validates both.
def _fp8_decode(c: int) -> float:
    sgn = -1.0 if (c & 0x80) else 1.0
    E, m = (c >> 3) & 15, c & 7
    if E == 15 and m == 7:
        return float("nan")
    if E == 0:
        return sgn * m * 2.0**-9
    return sgn * (1.0 + m / 8.0) * 2.0 ** (E - 7)


_TABLE = np.array([_fp8_decode(c) for c in range(127)])          # 0x00..0x7e: the finite non-negative values


def _table_search_encode(a: float) -> int:
    """fp8_encode for a finite a (>=0 path only; sign stripped by the caller)."""
    if a <= _TABLE[0]:                                            # zero (and below the first step)
        return 0
    if a >= _TABLE[-1]:
        return 0x7E
    hi = int(np.searchsorted(_TABLE, a, side="left"))
    lo = hi - 1
    dlo, dhi = a - _TABLE[lo], _TABLE[hi] - a
    if dlo < dhi:
        return lo
    if dhi < dlo:
        return hi
    return lo if (lo & 7) % 2 == 0 else hi                        # midpoint: the even mantissa


def fp8_encode(a) -> int:
    if isinstance(a, np.ndarray):
        return np.array([fp8_encode(v) for v in a], np.int32)
    a = float(a)
    if np.isnan(a):
        return 0x7F
    sgn = 0x80 if (a < 0 or np.signbit(a)) else 0
    return sgn | _table_search_encode(abs(a))


def fp8_decode(c: int) -> float:
    return _fp8_decode(int(c))


# --------------------------------------------------------------------- the transcribed ops
def rmsnorm(x, gamma, eps=EPS):
    xf = np.asarray(x, np.float64)
    rstd = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + eps)
    return xf * rstd * gamma                                       # pre-round fp32 (matches kernels.hpp)


def rope_head(x, pos, n_rot, theta):
    """Split-half rotate on the first n_rot dims of each row; dims >= n_rot copied. x [P, head_dim]."""
    x = np.asarray(x, np.float64)
    head_dim = x.shape[-1]
    out = x.copy()
    half = n_rot // 2
    inv = 1.0 / (theta ** (np.arange(0, n_rot, 2, dtype=np.float64) / n_rot))
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]       # [P, half]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:n_rot]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:n_rot] = x2 * cos + x1 * sin
    return out


def swa_attention(q, k, v, sink, first, last):
    """One head-set q [H,K] over k/v rows first..last, with a per-head sink logit if `sink` is not None.
    Softmax scale 1/sqrt(K); the sink column is dropped from the weights. Transcribes attn_swa:reference_attention."""
    q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
    K = q.shape[1]
    ks, vs = k[first : last + 1], v[first : last + 1]
    scores = (q @ ks.T) / np.sqrt(K)                                # [H, n]
    if sink is None:
        logits = scores
    else:
        logits = np.concatenate([scores, np.asarray(sink, np.float64)[:, None]], axis=-1)
    z = logits.max(-1, keepdims=True)
    e = np.exp(logits - z)
    probs = e / e.sum(-1, keepdims=True)
    if sink is not None:
        probs = probs[:, :-1]                                       # drop the sink column
    return probs @ vs                                               # [H, V]


def index_key_layernorm(x, weight, bias, eps=EPS):
    """Plain LayerNorm (mean, var, affine). Transcribes ops.py:index_key_norm / attn_dsa:reference_caches."""
    x = np.asarray(x, np.float64)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    return bias + (x - mean) * rstd * weight


def fp8_rows_dequant(rows):
    """Per-row e4m3 (scale = amax/448, min 1e-30), dequantised back. Transcribes attn_dsa:quantize_e4m3_rows."""
    rows = np.asarray(rows, np.float64)
    amax = np.maximum(np.abs(rows).max(1), BF16_MIN)
    scale = amax / FP8_MAX
    codes = np.array([[fp8_encode(v / s) for v in row] for row, s in zip(rows, scale)], np.int32)
    deq = np.array([[fp8_decode(c) for c in rc] for rc in codes], np.float64) * scale[:, None]
    return deq, scale, codes


def dsa_index_logits(iq_deq, w, ki_deq):
    """logit[j] = sum_h relu(q_h . k_j) * w[h]. Transcribes attn_dsa:reference_selection (logits line)."""
    iq_deq, ki_deq, w = (np.asarray(t, np.float64) for t in (iq_deq, ki_deq, w))
    scores = ki_deq @ iq_deq.T                                      # [visible, H]
    return (np.maximum(scores, 0.0) * w[None, :]).sum(-1)           # [visible]


def stable_topk(v, k):
    """Top-k by value, ties to the lower index, winners in ascending index order. Transcribes
    prefill/native/stable_dsa_topk.py. Implemented by an explicit (value desc, index asc) sort."""
    v = np.asarray(v, np.float64)
    order = np.lexsort((np.arange(len(v)), -v))                    # primary -v, secondary index asc
    return np.sort(order[:k]).astype(np.int32)


def moe_route(x, weight, bias, topk):
    """sigmoid scores, select top-k by (p+bias), renormalise the unbiased p over the k by (sum+1e-20).
    Transcribes training_moe.py:route (fp32). Returns (idx ascending, weights) + the raw scores."""
    x, weight, bias = (np.asarray(t, np.float64) for t in (x, weight, bias))
    scores = x @ weight.T
    p = 1.0 / (1.0 + np.exp(-scores))
    idx = stable_topk(p + bias, topk)                               # bias only decides the pick
    sel = p[idx]
    return idx, (sel / (sel.sum() + 1e-20)).astype(np.float32), scores.astype(np.float32)


def moe_combine(expert_out, w):
    """out = sum_t w[t] * expert_out[t]. Transcribes training_moe.py combine (fp32, before bf16 round)."""
    return (np.asarray(expert_out, np.float64) * np.asarray(w, np.float64)[:, None]).sum(0)


# --------------------------------------------------------------------- fixtures
def gen(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 0

    def emit(case, **arrays):
        nonlocal n
        for name, arr in arrays.items():
            write_arr(outdir, case, name, arr)
            n += 1

    # rmsnorm: one row of the real hidden width, gamma near 1 as the model's norms are.
    x = (rng.standard_normal(4096) * 1.5).astype(np.float32)
    gamma = (1.0 + 0.1 * rng.standard_normal(4096)).astype(np.float32)
    emit("rmsnorm", x=x, gamma=gamma, eps=np.array([EPS], np.float32), out=rmsnorm(x, gamma).astype(np.float32))

    # rope: real partial rotation (64 of 192), both thetas, and the indexer's 64 of 128 case.
    for case, theta, hd in (("rope_swa", 1e4, 192), ("rope_dsa", 1e7, 192), ("rope_idx", 1e7, 128)):
        pos = np.array([0, 1, 2, 5, 127, 128, 129, 1000, 4095, 65535], np.int32)
        xin = (rng.standard_normal((len(pos), hd)) * 1.2).astype(np.float32)
        out = rope_head(xin, pos, 64, theta)
        emit(case, x=xin, pos=pos, n_rot=np.array([64], np.int32), theta=np.array([theta], np.float32),
             out=out.astype(np.float32))

    # swa attention with a sink, and a sinkless (DSA-style) variant: both over a window of a full cache.
    P, Kd, Vd, H = 136, 192, 128, 8
    k = (rng.standard_normal((P, Kd)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((P, Vd)) * 0.5).astype(np.float32)
    q = (rng.standard_normal((H, Kd)) * 0.5).astype(np.float32)
    first, last = 8, 135
    sink = (0.5 * rng.standard_normal(H)).astype(np.float32)
    emit("swa_attn", q=q, k=k, v=v, sink=sink, first=np.array([first], np.int32),
         last=np.array([last], np.int32), out=swa_attention(q, k, v, sink, first, last).astype(np.float32))
    emit("dsa_attn", q=q, k=k, v=v, first=np.array([first], np.int32), last=np.array([last], np.int32),
         out=swa_attention(q, k, v, None, first, last).astype(np.float32))

    # index-key layernorm: one 128-dim index key with affine weight/bias.
    ik = (rng.standard_normal(128) * 0.8).astype(np.float32)
    ikw = (1.0 + 0.1 * rng.standard_normal(128)).astype(np.float32)
    ikb = (0.05 * rng.standard_normal(128)).astype(np.float32)
    emit("index_ln", x=ik, weight=ikw, bias=ikb, eps=np.array([EPS], np.float32),
         out=index_key_layernorm(ik, ikw, ikb).astype(np.float32))

    # fp8 codec: every finite code round-trips, midpoints go to the even mantissa, subnormals and
    # saturation included. The C++ must reproduce the same code and the same decoded value.
    codes = np.arange(255, dtype=np.int32)                        # 0x00..0xfe: every code but one NaN
    vals = np.array([fp8_decode(c) for c in codes], np.float32)
    # midpoints between adjacent representable magnitudes (positive side; _TABLE is non-negative).
    mids = ((_TABLE[1:] + _TABLE[:-1]) / 2.0).astype(np.float32)
    grid = np.concatenate([vals, mids, mids * -1,
                           np.array([0.0, -0.0, 0.001, 1e-9, 448.0, 448.1, 500.0, 447.9], np.float32)]).astype(np.float32)
    gcodes = np.array([fp8_encode(v) for v in grid], np.int32)
    gdec = np.array([fp8_decode(c) for c in gcodes], np.float32)
    emit("fp8_codec", in_=grid, code=gcodes, decoded=gdec, code_table=codes, val_table=vals)

    # fp8 rows end-to-end: bf16-rounded rows -> per-row scale -> dequantised (what the scores consume).
    # The /scale and *scale happen in float32 here, as they do in C++, so the rounding matches element-wise.
    r = bf16_round(rng.standard_normal((64, 128)) * 0.6).astype(np.float32)
    amax = np.maximum(np.abs(r.astype(np.float32)).max(1), np.float32(BF16_MIN)).astype(np.float32)
    scale = (amax / np.float32(FP8_MAX)).astype(np.float32)
    scaled = (r / scale[:, None]).astype(np.float32)
    codes = fp8_encode(scaled)
    deq = (np.array([[fp8_decode(c) for c in rc] for rc in codes], np.float32) * scale[:, None]).astype(np.float32)
    emit("fp8_rows", rows=r, deq=deq, scale=scale)

    # dsa index logits from already-dequantised iq/ki (isolates the relu+weighted-sum).
    IH, IHD, VIS = 16, 128, 40
    iq = bf16_round(rng.standard_normal((IH, IHD)) * 0.4).astype(np.float32)
    ki = bf16_round(rng.standard_normal((VIS, IHD)) * 0.4).astype(np.float32)
    w = ((0.4 * rng.standard_normal(IH)) * (IHD ** -0.5)).astype(np.float32)
    emit("dsa_logits", iq=iq, ki=ki, w=w, out=dsa_index_logits(iq, w, ki).astype(np.float32))

    # stable top-k with deliberate ties: half the values duplicated, so the tie rule is exercised.
    vt = np.round(rng.standard_normal(64), 2).astype(np.float32)
    vt[8:16] = vt[0:8]
    vt[0] = vt[4] = vt[30] = vt[31] = vt.max()                    # a big tie block at the top
    emit("stable_topk", v=vt, k=np.array([8], np.int32), out=stable_topk(vt, 8))

    # moe router at the real shapes: 256 experts, top-8, 4096-wide, a small correction bias.
    embd, ne = 4096, 256
    xr = (rng.standard_normal(embd) * 1.0).astype(np.float32)
    wr = (rng.standard_normal((ne, embd)) * 0.02).astype(np.float32)
    br = (0.1 * rng.standard_normal(ne)).astype(np.float32)
    ridx, rwts, rscores = moe_route(xr, wr, br, 8)
    emit("moe_route", x=xr, weight=wr, bias=br, topk=np.array([8], np.int32),
         idx=ridx, wts=rwts, scores=rscores)

    # moe combine over the real width.
    eo = (rng.standard_normal((8, embd)) * 0.3).astype(np.float32)
    cw = (np.sort(rng.random(8))[::-1]).astype(np.float32)
    emit("moe_combine", expert_out=eo, w=cw, out=moe_combine(eo, cw).astype(np.float32))

    # silu over a sign-crossing grid.
    g = np.linspace(-12.0, 12.0, 97).astype(np.float32)
    emit("silu", gate=g, out=(g / (1.0 + np.exp(-g))).astype(np.float32))

    return n


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(here, "tests", "naive", "fixtures")
    args = sys.argv[1:]
    if args and args[0] == "--print":
        dump_case(default, args[1])
        return
    outdir = args[0] if args else default
    count = gen(outdir)
    print(f"wrote {count} fixture arrays to {outdir}  (review any case with: "
          f"python {os.path.relpath(__file__, here)} --print CASE)")


if __name__ == "__main__":
    main()
