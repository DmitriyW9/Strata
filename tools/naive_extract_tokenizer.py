#!/usr/bin/env python3
"""tools/naive_extract_tokenizer.py - write a pack-style tokenizer folder from a Naive-N0.5-Flash GGUF.

The server (serve/server.py) reads a tokenizer folder of vocab.json + merges.txt + token_type.json +
tokenizer.json (+ chat_template.jinja): the same five files a pack carries.  The GGUF holds all of them
as metadata (tokenizer.ggml.* and tokenizer.chat_template); this copies them out unchanged and records
the GGUF's own pre-tokenizer name (tokenizer.ggml.pre = "qwen2" on MQ87) in tokenizer.json, the way
tools/dsv41/pack_gguf.py's write_tokenizer does for a DeepSeek pack.

  python tools/naive_extract_tokenizer.py --model <shard1.gguf> --out <dir>

Nothing is invented: a GGUF without a tokenizer is refused, like pack_gguf's version.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

# The qwen2 split rule, as tools/strata_tokenizer.py compiles it (third_party/llama.cpp L383).  Stored
# with the tokenizer so the folder says outright which rule its ids were cut with.
QWEN2_PATTERN = ("(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
                 "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                 "|\\p{N}"
                 "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                 "|\\s*[\\r\\n]+"
                 "|\\s+(?!\\S)"
                 "|\\s+")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the model's GGUF (any shard: the metadata is in shard 1)")
    ap.add_argument("--out", required=True, help="the tokenizer folder to write")
    a = ap.parse_args()

    md = GGUFFile(pathlib.Path(a.model)).metadata
    need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
    missing = [k for k in need if k not in md]
    if missing:
        raise SystemExit(f"{a.model}: no tokenizer in it (missing {', '.join(missing)})")
    model = md.get("tokenizer.ggml.model")
    if model != "gpt2":
        raise SystemExit(f"{a.model}: tokenizer model is {model!r}, this tool writes byte-level BPE (gpt2) folders only")
    tokens = list(md["tokenizer.ggml.tokens"])
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    merges = [str(m) for m in md["tokenizer.ggml.merges"]]
    types = [int(t) for t in (md.get("tokenizer.ggml.token_type") or [1] * len(tokens))]
    pre = str(md.get("tokenizer.ggml.pre", "qwen35"))
    special = {"bos": int(md.get("tokenizer.ggml.bos_token_id", 0)),
               "eos": int(md.get("tokenizer.ggml.eos_token_id", 1)),
               "pad": int(md.get("tokenizer.ggml.padding_token_id", 2))}

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types))
    cfg = {"model": "gpt2", "pre": pre, "vocab_size": len(tokens), "n_merges": len(merges),
           "special_ids": special, "add_bos_token": bool(md.get("tokenizer.ggml.add_bos_token", False))}
    if pre == "qwen2":
        cfg["pre_pattern"] = QWEN2_PATTERN
        cfg["pre_pattern_source"] = "third_party/llama.cpp src/llama-vocab.cpp (LLAMA_VOCAB_PRE_TYPE_QWEN2)"
    (out / "tokenizer.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(str(tpl), encoding="utf-8")
    print(f"wrote {out}: {len(tokens)} tokens, {len(merges)} merges, pre={pre}, "
          f"template={'yes' if tpl else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
