#!/usr/bin/env python3
# tools/naive_iq2.py - numpy IQ2_XXS / IQ2_XS decoders for the MoE expert weights.
#
# WHY THIS IS A VALID REFERENCE, AND WHERE ITS INDEPENDENCE STOPS.  ggml's IQ2 decode is fixed
# arithmetic over three lookup tables - two codebooks of eight 2-bit magnitudes packed into a u64
# (iq2xxs_grid[256], iq2xs_grid[512]) and a sign byte table (ksigns_iq2xs[128]).  Those tables are
# FORMAT DATA, not a formula; the port reads the very same tables through ggml's compiled `to_float`
# (docs/NAIVE.md, loader.hpp) rather than re-deriving a 1024-entry codebook.  The table CONTENT is
# therefore shared by construction and is NOT what this checks (tools/extract_iq2_tables.py copies
# them from ggml-common.h so a version bump is picked up, not diverged from).  What IS independent,
# and what a fixture cross-check exercises, is everything around the tables:
#   * the element / sub-block mapping (which byte of the packed word indexes which codebook row for
#     which of the 256 output elements) - written here from the struct layout and vectorised over a
#     precomputed element-index map, as opposed to ggml's nested ib32/l/j loop;
#   * the sign extraction (which bits pick the 128-row sign byte, and which of the eight kmask bits
#     flips which magnitude);
#   * the per-sub-block scale fold (XXS: the packed word's top nibble; XS: a nibble of the scale byte).
# self_test() pins all three: the vectorised decoder is compared, value-for-value, against a literal
# line-by-line transcription of ggml's decode loop (ggml-quants.c dequantize_row_iq2_xxs / _iq2_xs) on
# random bytes.  Different control flow, same tables, same bytes -> agreement means the mapping and
# sign/scale application match the loop ggml runs.

from __future__ import annotations

import numpy as np

from iq2_tables import IQ2_XXS_GRID, IQ2_XS_GRID, KSIGNS_IQ2XS, KMASK_IQ2XS

_XXS_GRID = np.array(IQ2_XXS_GRID, np.uint64)     # [256]
_XS_GRID = np.array(IQ2_XS_GRID, np.uint64)       # [512]
_KSIGNS = np.array(KSIGNS_IQ2XS, np.uint8)        # [128]
_KMASK = np.array(KMASK_IQ2XS, np.uint8)          # [1,2,4,8,16,32,64,128]


def _grid_magnitudes(grid: np.ndarray) -> np.ndarray:
    """Unpack a u64 codebook to [rows, 8] magnitude bytes; byte j (the j-th least-significant) is the
    magnitude of sub-element j.  Each magnitude is one of {8, 19, 25} (never 0)."""
    shift = (np.arange(8, dtype=np.uint64) * 8)[None, :]
    return ((grid[:, None] >> shift) & 0xFF).astype(np.uint8)


_XXS_MAG = _grid_magnitudes(_XXS_GRID)            # [256, 8]
_XS_MAG = _grid_magnitudes(_XS_GRID)              # [512, 8]


def _f16(u16: np.ndarray) -> np.ndarray:
    """uint16 bit patterns read as float16, widened exactly to float32 (a reinterpretation)."""
    return np.asarray(u16, np.uint16).view(np.float16).astype(np.float32)


# Within a 256-element block, ggml writes element e sequentially: sub-block ib32 = e // 32 (8 of
# them), then one of four codebook bytes l = (e % 32) // 8 (each covering 8 elements), then the
# magnitude / sign bit j = e % 8.  The decoders below fill e in that order.
_IBB_XXS, _IBB_XS = 66, 74     # block bytes
_XXS_CHUNK, _XS_CHUNK = 8192, 8192


def dequant_iq2_xxs(raw: np.ndarray, n: int) -> np.ndarray:
    """IQ2_XXS: block = {half d; uint16 qs[32]} (66 B / 256 elts).  Four codebook bytes come from a
    32-bit word (aux8[0..3]); the scale nibble and the four sign rows come from the next 32-bit word."""
    assert n % 256 == 0 and raw.size == n // 256 * _IBB_XXS, (raw.size, n)
    nb = n // 256
    out = np.empty(n, np.float32)
    for a in range(0, nb, _XXS_CHUNK):
        b = min(a + _XXS_CHUNK, nb)
        blk = np.frombuffer(raw[a * _IBB_XXS : b * _IBB_XXS].tobytes(), np.uint8).reshape(b - a, _IBB_XXS)
        d = _f16(blk[:, 0:2].copy().view(np.uint16)[:, 0])                    # [cb]
        # The eight 8-byte groups start at byte 2; group ib32 spans bytes 2+8*ib32 .. +8.  ggml reads
        # each group as two uint32 (aux32); the high one (aux32[1], the scale nibble + four sign rows)
        # is at bytes 6+8*ib32.  The four codebook bytes (aux8[0..3]) are the low group's bytes 2..5.
        hi = np.stack([blk[:, 6 + 8 * ib : 10 + 8 * ib].copy().view(np.uint32)[:, 0] for ib in range(8)], 1)  # [cb,8]
        db = d[:, None] * (0.5 + ((hi >> 28) & 0xF).astype(np.float32)) * 0.25      # [cb,8] per sub-block
        # A block's 256 elements are 32 sub-bytes of 8 elements each: sub-byte m = 4*ib32 + l covers
        # elements [8m, 8m+8) and carries ONE codebook row and ONE sign row.  Maps below are [cb, 32].
        gidx = np.empty((b - a, 32), np.int64)
        srow = np.empty((b - a, 32), np.int64)
        for ib32 in range(8):
            for l in range(4):
                gidx[:, 4 * ib32 + l] = blk[:, 2 + 8 * ib32 + l]               # codebook byte (aux8[l])
                srow[:, 4 * ib32 + l] = (hi[:, ib32] >> (7 * l)) & 0x7F        # ksigns row
        mag = _XXS_MAG[gidx].astype(np.float32).reshape(b - a, 256)            # [cb,32,8] -> element order
        flip = np.where((_KSIGNS[srow][..., None] & _KMASK[None, None, :]) != 0,
                        -1.0, 1.0).reshape(b - a, 256)
        scale32 = db.repeat(32, axis=1)                                        # [cb,256] one scale per 32
        out[a * 256 : b * 256] = (scale32 * mag * flip).reshape(-1)
    return out


def dequant_iq2_xs(raw: np.ndarray, n: int) -> np.ndarray:
    """IQ2_XS: block = {half d; uint16 qs[32]; uint8 scales[8]} (74 B / 256 elts).  Each uint16 qs is
    one 8-elt sub-block: low 9 bits index the 512-row codebook, top 7 bits are the sign row; the
    sub-block scale is a nibble of scales[ib32] (low nibble for l 0,1, high for l 2,3)."""
    assert n % 256 == 0 and raw.size == n // 256 * _IBB_XS, (raw.size, n)
    nb = n // 256
    out = np.empty(n, np.float32)
    for a in range(0, nb, _XS_CHUNK):
        b = min(a + _XS_CHUNK, nb)
        blk = np.frombuffer(raw[a * _IBB_XS : b * _IBB_XS].tobytes(), np.uint8).reshape(b - a, _IBB_XS)
        d = _f16(blk[:, 0:2].copy().view(np.uint16)[:, 0])                    # [cb]
        qs = blk[:, 2:66].copy().view(np.uint16)                              # [cb, 32]
        sc = blk[:, 66:74]                                                    # [cb, 8]
        qidx = np.empty((b - a, 32), np.int64)        # one uint16 per 8 elements: 32 per block
        nib = np.empty((b - a, 32), np.float32)
        for ib32 in range(8):
            for l in range(4):
                qidx[:, 4 * ib32 + l] = qs[:, 4 * ib32 + l].astype(np.int64)
                nib[:, 4 * ib32 + l] = (((sc[:, ib32] >> (0 if l < 2 else 4)) & 0xF).astype(np.float32))
        grid_idx = qidx & 0x1FF
        sign_row = (qidx >> 9) & 0x7F
        mag = _XS_MAG[grid_idx].astype(np.float32).reshape(b - a, 256)     # [cb,32,8] -> element order
        flip = np.where((_KSIGNS[sign_row][..., None] & _KMASK[None, None, :]) != 0,
                        -1.0, 1.0).reshape(b - a, 256)
        db = (d[:, None] * (0.5 + nib) * 0.25).repeat(8, axis=1)           # [cb,256] one scale per 8
        out[a * 256 : b * 256] = (db * mag * flip).reshape(-1)
    return out


# ------------------------------------------------------------------ literal ggml-loop transcriptions
def _iq2_xxs_literal(raw: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, np.float64)
    for i in range(n // 256):
        bs, eo = i * 66, i * 256
        d = float(np.frombuffer(raw[bs : bs + 2].tobytes(), np.float16)[0])
        for ib32 in range(8):
            g = bs + 2 + 8 * ib32
            aux8 = [int(raw[g + k]) for k in range(4)]                       # the four grid-index bytes
            aux1 = int.from_bytes(raw[g + 4 : g + 8], "little")              # aux32[1]
            db = d * (0.5 + (aux1 >> 28)) * 0.25
            for l in range(4):
                grid = int(_XXS_GRID[aux8[l]])
                signs = int(_KSIGNS[(aux1 >> (7 * l)) & 127])
                for j in range(8):
                    mag = (grid >> (8 * j)) & 0xFF
                    out[eo + ib32 * 32 + l * 8 + j] = db * mag * (-1.0 if (signs & (1 << j)) else 1.0)
    return out


def _iq2_xs_literal(raw: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, np.float64)
    for i in range(n // 256):
        bs, eo = i * 74, i * 256
        d = float(np.frombuffer(raw[bs : bs + 2].tobytes(), np.float16)[0])
        for ib32 in range(8):
            db0 = d * (0.5 + (raw[bs + 66 + ib32] & 0xF)) * 0.25
            db1 = d * (0.5 + (raw[bs + 66 + ib32] >> 4)) * 0.25
            for l in range(4):
                o = bs + 2 + 2 * (4 * ib32 + l)
                q = int.from_bytes(raw[o : o + 2], "little")
                grid = int(_XS_GRID[q & 511])
                signs = int(_KSIGNS[q >> 9])
                db = db0 if l < 2 else db1
                for j in range(8):
                    mag = (grid >> (8 * j)) & 0xFF
                    out[eo + ib32 * 32 + l * 8 + j] = db * mag * (-1.0 if (signs & (1 << j)) else 1.0)
    return out


def self_test(trials: int = 3) -> None:
    """The vectorised IQ2 decoders must equal a literal ggml loop, value-for-value, on random bytes."""
    global _XXS_CHUNK, _XS_CHUNK
    _XXS_CHUNK, _XS_CHUNK = 13, 17                            # exercise the chunk boundaries
    rng = np.random.default_rng(9)
    for t in range(trials):
        nb = 48 + t * 5
        rx = rng.integers(0, 256, nb * 66, dtype=np.uint8).copy()
        rx.reshape(nb, 66)[:, 0:2] = np.uint8([0x00, 0x3C])   # d = 1.0f (finite fp16 scale)
        got = dequant_iq2_xxs(rx, nb * 256).astype(np.float64)
        ref = _iq2_xxs_literal(rx, nb * 256)
        if not np.array_equal(got, ref):
            raise SystemExit(f"iq2_xxs self-test mismatch (trial {t}): "
                             f"{int((got != ref).sum())}/{nb * 256} differ, max abs {np.abs(got - ref).max():.3g}")
        rs = rng.integers(0, 256, nb * 74, dtype=np.uint8).copy()
        rs.reshape(nb, 74)[:, 0:2] = np.uint8([0x00, 0x3C])
        got2 = dequant_iq2_xs(rs, nb * 256).astype(np.float64)
        ref2 = _iq2_xs_literal(rs, nb * 256)
        if not np.array_equal(got2, ref2):
            raise SystemExit(f"iq2_xs self-test mismatch (trial {t}): {int((got2 != ref2).sum())} differ")
    print("iq2 decoders self-tested: iq2_xxs and iq2_xs bit-exact vs the literal ggml loops")


if __name__ == "__main__":
    self_test()
