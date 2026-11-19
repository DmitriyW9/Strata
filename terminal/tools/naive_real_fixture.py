#!/usr/bin/env python3
# tools/naive_real_fixture.py - oracle fixtures taken from the *real* model weights, for
# tests/naive/test_naive_loader.cpp.
#
# The kernels test (tools/naive_kernels_oracle.py) pins the trunk's arithmetic on synthetic inputs.
# This one answers the next question: does the loader's own layer (raw bytes, dequant into fp32, the
# element order a GGUF carries) feed the tested router the same numbers a reference reads straight
# from the file's bytes? So the fixture inputs here are the real tensors, not random ones:
#
#   real_router  blk.1.ffn_gate_inp.weight {4096,256} + blk.1.exp_probs_b.bias (both F32) read from
#                the shard that holds them; a fixed-seed row x; the expected idx/weights from the
#                oracle's moe_route (a numpy transcription of NaiveRT training_moe.py:route).
#                F32 => the file's bytes ARE the values, so this checks the loader's byte window,
#                the {dim0 fastest} element order, and the router formula over the real 4 MiB matrix.
#   dequant_f32  the first block of blk.1.attn_norm.weight (F32): dequant must be the identity.
#
# The file itself is never copied: only the two small F32 tensors (4 MiB + 1 KiB) are extracted, so
# the fixtures stay small and commit-able, and the check runs with the model absent (the C++ test
# skips that part then).  The quantised expert stacks are NOT here - dequantising them in numpy
# would mean transcribing ggml's IQ2 codebook, which is exactly what the port refuses to do; their
# raw-byte identity and the sign property are checked in C++ against ggml itself instead.
#
#   python terminal/tools/naive_real_fixture.py [model-shard-1.gguf] [outdir]
#     default model:  /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/
#                     Naive-N0.5-Flash-MQ87-00001-of-00004.gguf
#     default outdir: engine/tests/naive/fixtures

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402
from naive_kernels_oracle import moe_route, write_arr  # noqa: E402

DEFAULT_MODEL = ("/home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/"
                 "Naive-N0.5-Flash-MQ87-00001-of-00004.gguf")
ROUTER = "blk.1.ffn_gate_inp.weight"
ROUTER_BIAS = "blk.1.exp_probs_b.bias"
NORM = "blk.1.attn_norm.weight"


def read_tensor(gguf: GGUFFile, fh, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The values of one F32 tensor, read straight from the file's bytes at its data offset."""
    info = next((t for t in gguf.tensors if t.name == name), None)
    if info is None:
        raise SystemExit(f"{gguf.path.name} holds no {name}")
    if info.type_name != "F32":
        raise SystemExit(f"{name} is {info.type_name}, this generator reads only F32")
    nbytes = info.expected_bytes()
    at = gguf.data_start + info.offset
    fh.seek(at)
    raw = fh.read(nbytes)                       # a plain read: an mmap offset must be page aligned
    if len(raw) != nbytes:
        raise SystemExit(f"{name} lies past the end of {gguf.path.name}")
    arr = np.frombuffer(raw, dtype="<f4", count=info.elements)
    return arr.reshape(list(reversed(info.shape))).copy(), arr.copy()   # (logical, file order)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root, two levels above terminal/tools/
    model = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "engine", "tests", "naive", "fixtures")
    os.makedirs(outdir, exist_ok=True)

    gguf = GGUFFile(model)
    with model.open("rb") as fh:
        weight_logical, weight_file = read_tensor(gguf, fh, ROUTER)
        bias_logical, bias_file = read_tensor(gguf, fh, ROUTER_BIAS)
        norm_logical, norm_file = read_tensor(gguf, fh, NORM)

    n_expert, n_embd = weight_logical.shape                  # torch order: rows are experts
    if (n_expert, n_embd) != (256, 4096):
        raise SystemExit(f"{ROUTER} is {weight_logical.shape}, expected (256, 4096)")

    # A fixed-seed row, the same one the C++ test reads. In float32, as the model's activations are.
    rng = np.random.default_rng(11)
    x = (rng.standard_normal(n_embd) * 1.2).astype(np.float32)
    idx, wts, scores = moe_route(x, weight_logical, bias_logical, 8)

    write_arr(outdir, "real_router", "x", x)
    write_arr(outdir, "real_router", "weight", weight_file.reshape(n_expert, n_embd))
    write_arr(outdir, "real_router", "bias", bias_file)
    write_arr(outdir, "real_router", "topk", np.array([8], np.int32))
    write_arr(outdir, "real_router", "idx", idx)
    write_arr(outdir, "real_router", "wts", wts)
    # The window the loader must return for the router tensor: (name, type, elements, FNV-1a of bytes).
    # The C++ reads the tensor from the real shards through loader.hpp and hashes the same window.
    rb = weight_file.reshape(-1).astype("<f4").tobytes()
    h = fnv1a(rb)
    # int32 carries the low 32 bits; the C++ test compares the same 32 bits as uint32. struct.pack
    # so a hash at or above 2**31 survives (np.array would raise on an out-of-range Python int).
    write_arr(outdir, "raw_router", "bytes_hash", np.frombuffer(struct.pack("<I", h), np.int32))
    write_arr(outdir, "raw_router", "bytes_len", np.array([len(rb)], np.int32))
    write_arr(outdir, "raw_router", "type_id", np.array([0], np.int32))       # F32
    write_arr(outdir, "raw_router", "elements", np.array([n_expert * n_embd], np.int32))
    write_arr(outdir, "raw_router", "shape", np.array([n_embd, n_expert], np.int32))  # GGUF order
    write_arr(outdir, "raw_router", "first4", weight_file.reshape(-1)[:4])

    # F32 dequant is the identity: the first block of a real norm tensor.
    write_arr(outdir, "dequant_f32", "raw", norm_file[:256])
    write_arr(outdir, "dequant_f32", "deq", norm_file[:256])

    # What the C++ test checks by name, without the model: every inventory tensor's ggml size, and
    # the model's F32 router/bias/norm raw bytes.  Keep the counts so a regenerated fixture set is
    # self-describing (printed, not asserted - the C++ test reads only the arrays it names).
    print(f"real_router: weight {weight_logical.shape} x bias {bias_logical.shape}")
    print(f"  top-8 experts = {idx.tolist()}")
    print(f"  weights = {[round(float(w), 6) for w in wts]}")
    print(f"  raw window: {len(rb):,} bytes, FNV-1a = {h:#010x}")
    print(f"wrote fixtures for real_router / raw_router / dequant_f32 to {outdir}")


def fnv1a(data: bytes) -> int:
    """FNV-1a (32-bit): cheap, order-sensitive, and independent of the C++ test's own hash line."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


if __name__ == "__main__":
    main()
