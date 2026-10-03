#!/usr/bin/env python3
"""Measure this PC's engine under a few configurations and write one text file for the AI to interpret.

The engine already adapts on its own (the PCIe probe, `--expert-cache auto`).  What it cannot decide for
itself are the choices a sweep has to measure on the machine it runs on, and this tool measures them, one
engine start per arm:

  thp / 4kb        the expert arena's page backing.  Since 0.1.37 an anonymous arena without a hugepage pool
                   is aligned to 2 MB and madvised MADV_HUGEPAGE (transparent hugepages); STRATA_NO_LARGEPAGES=1
                   is the old 4 KB arm.  The two differ in TLB reach over the 75 GiB arena.
  per-layer        --expert-cache-per-layer: each layer keeps its own hottest experts instead of one shared
                   counter (the help text's 2.97% vs 21.4%/70.4% at 8/64 slots per layer was measured elsewhere).
  threads          STRATA_ARENA_THREADS=N: the arena's load threads (default 6, measured on a 6-core PC; this
                   one has 16).  Shows up in the startup line "loaded X GiB at Y GiB/s", which is captured too.

Each arm starts its own engine (the server must be stopped first: the arms need the GPU to themselves),
warms up once, then measures decode tok/s RUNS times on the same three prompts, and the run is appended to
--out as it goes (a killed run leaves the arms already measured).  The file is JSON: per arm the args, the
env, the per-run rates, the startup facts the engine printed (backing, slots, load rate, workers, PCIe).

    python tools/measure_configs.py strata-uncensored-q4_k_m.json
    python tools/measure_configs.py strata-uncensored-q4_k_m.json --arms 4kb,per-layer --runs 2
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from calibrate import PROMPTS, chat_ids, close, engine_error   # noqa: E402

# arm name -> (extra args as {flag: value}, extra env, what the arm is)
ARMS = {
    "default": ({}, {}, "the config as it is (since 0.1.37: transparent 2 MB pages, 6 load threads)"),
    "4kb": ({}, {"STRATA_NO_LARGEPAGES": "1"}, "the arena on 4 KB pages (the old fallback)"),
    "per-layer": ({"--expert-cache-per-layer": None}, {}, "each layer keeps its own hottest experts"),
    "threads": ({}, {"STRATA_ARENA_THREADS": "12"}, "the arena loads on 12 threads instead of 6"),
}

# the startup lines worth keeping from the engine's log, verbatim
FACTS = [re.compile(p, re.I) for p in (
    r"expert arena:", r"loaded .*GiB at .*GiB/s", r"expert cache \d+ slots", r"policy is",
    r"expert-pool workers", r"PCIe probe", r"pre-filled \d+ of")]


def measure_arm(name: str, cfg: dict, runs: int, max_new: int, ids_list, say) -> dict:
    from serve.server import StrataEngine, child_env, engine_args

    extra_args, extra_env, why = ARMS[name]
    args = list(engine_args(cfg))
    for flag in extra_args:                      # all of these arms are boolean flags
        if flag not in args:
            args.append(flag)
    env = {**child_env(cfg), **extra_env}
    log = Path(cfg.get("log") or "measure-engine.log").with_suffix(f".{name}.log")
    log_start = log.stat().st_size if log.exists() else 0
    say(f"  [{name}] starting the engine ({why}) ...")
    try:
        eng = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=str(log), env=env)
    except Exception as e:
        # an arm whose engine refuses to start is a RESULT, not a crash: record the engine's own reason
        # and measure the next arm (the 0.1.31 binary refuses --expert-cache-per-layer on a native pack)
        reason = engine_error(str(log), log_start) or str(e)
        say(f"  [{name}] the engine refused to start: {reason}")
        return {"arm": name, "why": why, "env": extra_env, "extra_args": dict(extra_args),
                "error": reason, "decode_median": None}
    rates, prompt_rates, facts = [], [], []
    try:
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if any(p.search(line) for p in FACTS):
                facts.append(line.strip())
        for _ in range(2):                      # warm-up: the first prompt of a boot is never representative
            eng.generate(ids_list[0], 16, {"temperature": 0}, threading.Event())
        for r in range(runs):
            n = sum(1 for t in eng.generate(ids_list[0], max_new, {"temperature": 0}, threading.Event())
                    if t is not None)
            last = eng.last or {}
            if n > 8 and last.get("decode_ms"):
                rates.append(n / (last["decode_ms"] / 1000.0))
            if last.get("prompt_tokens") and last.get("prompt_ms"):
                prompt_rates.append(last["prompt_tokens"] / (last["prompt_ms"] / 1000.0))
            say(f"  [{name}] run {r + 1}/{runs}: decode {rates[-1] if rates else 0:.1f} tok/s")
    finally:
        close(eng)
    return {"arm": name, "why": why, "env": extra_env,
            "extra_args": {k: v for k, v in extra_args.items()},
            "engine": eng.info.get("engine") or eng.info.get("version"),
            "decode_tok_s": [round(x, 2) for x in rates],
            "decode_median": round(statistics.median(rates), 2) if rates else None,
            "prompt_tok_s": [round(x, 1) for x in prompt_rates],
            "startup_facts": facts}


def run(cfg_path: Path, arms: list[str], runs: int, max_new: int, out: Path, say=print) -> None:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    results = []
    for name in arms:
        r = measure_arm(name, cfg, runs, max_new, ids_list, say)
        results.append(r)
        med = r["decode_median"]
        say(f"  [{name}] decode median {med if med is not None else 'no samples'} tok/s -> writing {out}")
        out.write_text(json.dumps({"config": str(cfg_path), "runs": runs, "max_new": max_new,
                                   "arms": results}, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of " + ",".join(ARMS))
    ap.add_argument("--runs", type=int, default=2, help="measured runs per arm (after one warm-up)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("strata-measure.txt"))
    a = ap.parse_args()
    chosen = [x.strip() for x in a.arms.split(",") if x.strip()]
    if bad := [x for x in chosen if x not in ARMS]:
        sys.exit(f"unknown arm(s) {bad}; known: {', '.join(ARMS)}")
    run(a.config, chosen, a.runs, a.max_new, a.out)
