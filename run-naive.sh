#!/bin/sh
# run-naive.sh - serve the Naive-N0.5-Flash model (strata-naive) on its own port.
#
# It starts the web server (terminal/serve/server.py) with the strata-naive config, separate from the running
# Flash-Next engine (that one stays on port 8080 and is not touched here).  The strata-naive engine
# validates the model on start and answers chat requests with the CPU trunk forward
# (engine/include/strata/naive/runtime.hpp, docs/NAIVE.md); the DSpark draft stays opt-in and is off here.
# By default the weight plan (engine/include/strata/naive/pool.hpp, docs/NAIVE.md "Where every byte
# lives at start-up") brings what fits into RAM and VRAM at start-up; --weights cache returns to the
# old on-demand read from the shards, where the first token of a cold start waits for the page cache.
#
# Usage:
#   ./run-naive.sh                          # weights=auto: everything resident when it fits, else
#                                           # the experts; the rest reads the mmap windows
#   ./run-naive.sh --weights resident       # refuse to start unless the whole file fits resident
#   ./run-naive.sh --weights cache          # nothing resident (the path before the pool existed)
#   ./run-naive.sh --weights vram-split     # whole expert stacks live ON a card (no RAM copy) or in
#                                           # RAM, never both; needs a CUDA build and room for the
#                                           # dense set in RAM
#   ./run-naive.sh --weights auto --gpus 0  # show the engine card 0 only (nvidia-smi number)
#   ./run-naive.sh --gpus 0,1               # show it both cards: the server limits what the engine
#                                           # sees (CUDA_VISIBLE_DEVICES), so it numbers them 0, 1
#   ./run-naive.sh --no-gpu                 # plan for RAM only, touch no card
#   ./run-naive.sh --port 8090              # any other option goes to the server unchanged
#
# To only validate the model without starting the server (no GPU, no weights loaded):
#   engine-cuda12/strata-naive --native <shard1.gguf> --draft <draft.gguf> --check
# and to see this box's memory plan without starting anything:
#   .venv/bin/python terminal/tools/naive_memory_plan.py <shard1.gguf>
cd "/home/dmitriy/projects/ai/LLM/Strata" || exit 2
VENV_PY="/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python"
CONFIG="/home/dmitriy/projects/ai/LLM/Strata/strata-naive-naive.json"

# The three strata-naive weight-placement options (engine/src/naive/naive_main.cpp): collected here
# and written into the config's "user_args" (terminal/strata_control/user_args.py - the person's
# options, which the server applies over the generated ones at every start), in a temp copy of the
# config so the checked-in one is never edited.
EXTRA=""
SERVER_ARGS=""
GPUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --weights) [ $# -ge 2 ] || { echo "run-naive.sh: --weights wants auto, resident, cache or vram-split" >&2; exit 2; }
                   EXTRA="$EXTRA --weights $2"; shift 2 ;;
        --gpus)    [ $# -ge 2 ] || { echo "run-naive.sh: --gpus wants a card list, as nvidia-smi numbers them (0,1)" >&2; exit 2; }
                   GPUS="$2"; shift 2 ;;
        --no-gpu)  EXTRA="$EXTRA --no-gpu"; shift ;;
        *)         SERVER_ARGS="$SERVER_ARGS $1"; shift ;;
    esac
done

if [ -n "$EXTRA" ] || [ -n "$GPUS" ]; then
    TMP_CONFIG="${XDG_RUNTIME_DIR:-/tmp}/strata-naive-run.$$.json"
    # shellcheck disable=SC2086
    NAIVE_GPUS="$GPUS" "$VENV_PY" - "$CONFIG" "$TMP_CONFIG" $EXTRA <<'EOF' || exit 2
import json, os, sys
cfg_path, out_path, *extra = sys.argv[1:]
cfg = json.load(open(cfg_path))
cfg["user_args"] = list(cfg.get("user_args") or []) + extra
# --gpus here names the cards as nvidia-smi numbers them, and the server's answer to that is the
# config's "gpu": it exports CUDA_VISIBLE_DEVICES with exactly those cards (server.py child_env),
# so the engine sees them and numbers them from 0. Passing the engine --gpus 1 beside a one-card
# CUDA_VISIBLE_DEVICES=1 would be a second numbering over the first: "card 1 is out of range
# (1 visible)". With the list in the config, the engine's own --gpus stays empty and takes every
# card it can see.
gpus = os.environ.get("NAIVE_GPUS", "").strip()
if gpus:
    cfg["gpu"] = [int(x) for x in gpus.split(",") if x.strip() != ""]
json.dump(cfg, open(out_path, "w"), indent=1)
EOF
    trap 'rm -f "$TMP_CONFIG"' EXIT INT TERM
    CONFIG="$TMP_CONFIG"
fi

# shellcheck disable=SC2086
exec "$VENV_PY" "/home/dmitriy/projects/ai/LLM/Strata/terminal/serve/server.py" --engine strata --config "$CONFIG" --port 8081 $SERVER_ARGS
