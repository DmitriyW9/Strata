#!/usr/bin/env python3
"""naive_memory_plan.py - the strata-naive start-up memory plan, without starting the engine.

Answers the same questions the engine's INFO plan answers (engine/include/strata/naive/pool.hpp,
printed by engine/src/naive/naive_main.cpp): where every weight would live at start-up, which tier
the box falls to, and why a smaller box refuses residency. The numbers come from the file's own
tensor table (terminal/tools/gguf_reader.py, header only - no weight is read), the RAM from
/proc/meminfo's MemAvailable, the cards from nvidia-smi. The same role rule as pool.hpp: the three
.ffn_*_exps.weight stacks are experts (raw resident, a card may hold them), every other named
tensor is dense (fp32 resident in RAM).

Same keys as the engine's INFO lines, so a log line and this tool's answer are comparable:

    python3 terminal/tools/naive_memory_plan.py <shard1.gguf>            # human, INFO-style lines
    python3 terminal/tools/naive_memory_plan.py <shard1.gguf> --json     # machine
    python3 terminal/tools/naive_memory_plan.py <shard1.gguf> --weights cache --no-gpu

Reserve (pool.hpp's constants, kept in sync by hand - pool.hpp is the source of truth):
RAM keeps 8 GiB free for the runtime's caches, each card 700 MiB for the desktop and the context.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile, BLOCK_GEOMETRY  # noqa: E402

GIB = 1024.0 ** 3
RAM_RESERVE_BYTES = 8 * 1024**3      # pool.hpp kPlanRamReserveBytes
CARD_RESERVE_BYTES = 700 * 1024**2   # pool.hpp kPlanCardReserveBytes
EXPERT_SUFFIXES = (".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight")
GLOBALS = ("token_embd.weight", "output.weight", "output_norm.weight")


def role_of(name: str) -> str:
    """pool.hpp role_of: expert stacks by suffix, dense by the inventory's names, else transient."""
    if name.endswith(EXPERT_SUFFIXES):
        return "expert"
    if name in GLOBALS or name.startswith("blk."):
        return "dense"
    return "transient"


def shard_paths(any_shard: pathlib.Path) -> list[pathlib.Path]:
    """The sibling shards of a "...-NNNNN-of-MMMMM.gguf" name (gguf_split.hpp's contract)."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", any_shard.name)
    if not m:
        return [any_shard]
    total = int(m.group(2))
    out = []
    for i in range(1, total + 1):
        name = any_shard.name[: m.start(1)] + f"{i:05d}" + any_shard.name[m.end(1):]
        p = any_shard.with_name(name)
        if not p.exists():
            raise SystemExit(f"missing model shard {p}")
        out.append(p)
    return out


def payload_bytes(t) -> int | None:
    geom = BLOCK_GEOMETRY.get(t.type_name)
    if geom is None:
        return None
    block_elems, block_bytes = geom
    if t.elements % block_elems:
        return None
    return t.elements // block_elems * block_bytes


def ram_available_bytes() -> int:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def probe_cards(want: list[int] | None) -> list[dict]:
    """nvidia-smi's index/total/free per card (MiB -> bytes), like the engine's cudaMemGetInfo probe.
    nvidia-smi answers before any CUDA context exists, so its free is a little larger than what the
    engine sees: the engine probes after init(), when its own context (~0.3 GiB a card) is already on
    each card. The plan is honest about the box it runs on, not about the engine's later room."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, total, free = int(parts[0]), int(parts[1]), int(parts[2])
        if want is not None and idx not in want:
            continue
        cards.append({"index": idx, "total_bytes": total * 1024**2, "free_bytes": free * 1024**2})
    return cards


def plan(paths, ram_available, cards, expert_only=False):
    """pool.hpp plan_pool over the file's tensor table. Returns the sums, fits, per-card bytes, note."""
    expert = dense_fp32 = dense_raw = 0
    stacks = []  # (name, raw bytes) in file order - what the card tier places, never split
    for p in paths:
        f = GGUFFile(p)
        for t in f.tensors:
            role = role_of(t.name)
            raw = payload_bytes(t)
            if raw is None:
                raise SystemExit(f"{t.name}: unknown ggml type {t.type_name}")
            if role == "transient":
                raise SystemExit(f"{t.name}: not a tensor the naive inventory names")
            if role == "expert":
                expert += raw
                stacks.append((t.name, raw))
            else:
                dense_raw += raw
                if not expert_only:
                    dense_fp32 += t.elements * 4
    ram_bytes = dense_fp32 + expert
    room = max(ram_available - RAM_RESERVE_BYTES, 0)
    fits_raw = expert != 0 and expert <= room
    fits_all = fits_raw and (expert if expert_only else ram_bytes) <= room
    note = ""
    card_bytes = [0] * len(cards)
    if not fits_raw:
        note = (f"the experts ({expert / GIB:.2f} GiB) do not fit in the free RAM "
                f"({room / GIB:.1f} GiB after the reserve)")
    elif not fits_all:
        note = ("the experts fit in RAM; the dense weights stay on disk at this tier" if expert_only
                else f"the full fp32 dense set ({dense_fp32 / GIB:.2f} GiB) plus the experts do not fit "
                     f"in the free RAM; the experts stay resident and the dense weights read the mmap windows")
    else:
        # The card tier: whole stacks, file order, one card's room at a time (naive_main's upload loop).
        room_cards = [max(c["free_bytes"] - CARD_RESERVE_BYTES, 0) for c in cards]
        ci = 0
        for _name, nbytes in stacks:
            while ci < len(room_cards) and room_cards[ci] < nbytes:
                ci += 1
            if ci == len(room_cards):
                break
            room_cards[ci] -= nbytes
            card_bytes[ci] += nbytes
    return {"expert_bytes": expert, "dense_fp32_bytes": dense_fp32, "dense_raw_bytes": dense_raw,
            "ram_bytes": ram_bytes, "ram_available_bytes": ram_available,
            "fits_all": fits_all, "fits_experts": fits_raw,
            "card_bytes": card_bytes, "note": note, "stacks": len(stacks), "stack_list": stacks}


def gib4(x: float) -> str:
    """pool.hpp's std::to_string(GiB).substr(0, 4): six decimals, cut to four characters."""
    return f"{x / GIB:.6f}"[:4]


def split_plan(stacks, dense_fp32, ram_available, cards):
    """pool.hpp plan_split over the stacks plan() collected. The exclusive tier (--weights
    vram-split): a card holds a whole stack and the RAM keeps no copy of it, so the cards are
    filled richest-first in file order, the RAM takes the fp32 dense set first and the stacks no
    card took, and a stack that fits nowhere reads the mmap windows (counted skipped, named).
    Same numbers as the engine's split_ INFO lines."""
    room = [max(c["free_bytes"] - CARD_RESERVE_BYTES, 0) for c in cards]
    order = sorted(range(len(cards)), key=lambda i: -cards[i]["free_bytes"])  # richest first, stable
    if order and room[order[0]] >= CARD_RESERVE_BYTES:
        room[order[0]] -= CARD_RESERVE_BYTES  # the context + scratch rides on one card, once
    expert = sum(nbytes for _name, nbytes in stacks)
    card_bytes = [0] * len(cards)
    vram = 0
    on_card = [False] * len(stacks)
    for ci in order:
        for i, (_name, nbytes) in enumerate(stacks):
            if not on_card[i] and 0 < nbytes <= room[ci]:
                on_card[i] = True
                room[ci] -= nbytes
                card_bytes[ci] += nbytes
                vram += nbytes
    ram_room = max(ram_available - RAM_RESERVE_BYTES, 0)
    dense_fits = dense_fp32 <= ram_room
    ram_left = max(ram_room - dense_fp32, 0) if dense_fits else 0
    ram_expert = skipped = 0
    first_skipped = ""
    for i, (name, nbytes) in enumerate(stacks):
        if on_card[i]:
            continue
        if nbytes <= ram_left:
            ram_left -= nbytes
            ram_expert += nbytes
        else:
            skipped += nbytes
            first_skipped = first_skipped or name
    note = ""
    fits_all = skipped == 0 and (not stacks or vram + ram_expert > 0) and dense_fits
    if not fits_all:
        if not dense_fits:
            note = (f"the fp32 dense set ({gib4(dense_fp32)} GiB) does not fit in the free RAM "
                    f"({ram_room // 1024**3} GiB after the reserve)")
        elif skipped:
            note = (f"no card and no RAM room for {first_skipped} and possibly later stacks "
                    f"({gib4(skipped)} GiB in all): those read the mmap windows")
    return {"expert_bytes": expert, "vram_expert_bytes": vram, "ram_expert_bytes": ram_expert,
            "skipped_expert_bytes": skipped, "dense_fp32_bytes": dense_fp32,
            "card_expert_bytes": card_bytes, "dense_fits": dense_fits, "fits_all": fits_all,
            "ram_available_bytes": ram_available, "note": note}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="the first shard of the model (...-00001-of-NNNNN.gguf)")
    ap.add_argument("--weights", choices=["auto", "resident", "cache", "vram-split"], default="auto",
                    help="the tier to report (what strata-naive --weights would be asked for)")
    ap.add_argument("--gpus", help="cards to plan for, as nvidia-smi numbers (e.g. 0,1)")
    ap.add_argument("--no-gpu", action="store_true", help="plan for RAM only, probe no card")
    ap.add_argument("--expert-only", action="store_true",
                    help="plan the raw-dense tier (dense weights stay on disk)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    paths = shard_paths(pathlib.Path(a.model).expanduser())
    cards = [] if a.no_gpu else probe_cards(
        [int(x) for x in a.gpus.split(",")] if a.gpus else None)
    if a.weights == "cache":
        plan_out = plan(paths, 0, [], a.expert_only)  # the cache tier reserves nothing: report as refusal
        plan_out["note"] = "weights=cache: nothing is made resident, every read goes through the mmap"
        plan_out["fits_all"] = plan_out["fits_experts"] = False
        cards = []
    else:
        plan_out = plan(paths, ram_available_bytes(), cards, a.expert_only)

    # The exclusive plan, like the engine's split_ INFO lines: printed whenever cards were probed
    # or the tier asks for it (naive_main computes it the same way).
    split = None
    if a.weights != "cache" and (cards or a.weights == "vram-split"):
        split = split_plan(plan_out["stack_list"], plan_out["dense_fp32_bytes"],
                           ram_available_bytes(), cards)

    if a.json:
        print(json.dumps({"model": str(paths[0]), "shards": len(paths), "weights": a.weights,
                          **{k: v for k, v in plan_out.items() if k not in ("stacks", "stack_list")},
                          **({"split": {k: v for k, v in split.items()}} if split else {}),
                          "cards": [{"index": c["index"], "free_gib": round(c["free_bytes"] / GIB, 2)}
                                    for c in (cards if not a.no_gpu and a.weights != "cache" else [])]},
                         indent=1))
        return 0

    print(f"model={paths[0].name} shards={len(paths)}")
    print(f"INFO plan_weights={a.weights} plan_expert_gib={plan_out['expert_bytes'] / GIB:.2f} "
          f"plan_dense_fp32_gib={plan_out['dense_fp32_bytes'] / GIB:.2f} "
          f"plan_dense_raw_gib={plan_out['dense_raw_bytes'] / GIB:.2f} "
          f"plan_ram_gib={plan_out['ram_bytes'] / GIB:.2f} "
          f"plan_ram_available_gib={plan_out['ram_available_bytes'] / GIB:.2f} "
          f"plan_fits_all={int(plan_out['fits_all'])} plan_fits_experts={int(plan_out['fits_experts'])}")
    for c, nbytes in zip(cards if not a.no_gpu and a.weights != "cache" else [], plan_out["card_bytes"]):
        print(f"INFO plan_card{c['index']}_expert_gib={nbytes / GIB:.2f} "
              f"(of {c['free_bytes'] / GIB:.2f} free GiB)")
    if plan_out["note"]:
        print(f"INFO plan_note={plan_out['note']}")
    if split:
        print(f"INFO split_vram_expert_gib={split['vram_expert_bytes'] / GIB:.2f} "
              f"split_ram_expert_gib={split['ram_expert_bytes'] / GIB:.2f} "
              f"split_skipped_gib={split['skipped_expert_bytes'] / GIB:.2f} "
              f"split_fits_all={int(split['fits_all'])} split_dense_fits={int(split['dense_fits'])}")
        for c, nbytes in zip(cards, split["card_expert_bytes"]):
            print(f"INFO split_card{c['index']}_expert_gib={nbytes / GIB:.2f}")
        if split["note"]:
            print(f"INFO split_note={split['note']}")
    print("plan: experts %.2f GiB + dense fp32 %.2f GiB = %.2f GiB RAM; %s" % (
        plan_out["expert_bytes"] / GIB, plan_out["dense_fp32_bytes"] / GIB, plan_out["ram_bytes"] / GIB,
        "everything fits resident" if plan_out["fits_all"] else
        ("experts resident, dense on disk" if plan_out["fits_experts"] else "too little RAM: all on disk")))
    if split:
        print("split: %.2f GiB of experts on cards + %.2f GiB in RAM + %.2f GiB on disk windows; "
              "dense fp32 %.2f GiB in RAM; %s" % (
                  split["vram_expert_bytes"] / GIB, split["ram_expert_bytes"] / GIB,
                  split["skipped_expert_bytes"] / GIB, split["dense_fp32_bytes"] / GIB,
                  "every named weight lives on a card or in RAM" if split["fits_all"] else
                  ("some experts would read the mmap windows" if split["dense_fits"]
                   else "the dense set does not fit: the tier would refuse to start")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
