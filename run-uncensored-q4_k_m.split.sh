#!/bin/sh
# The all-cards split engine (strata-split-2, sm_70 + sm_89) as a separate opt-in experiment (#5119 adapted to this mixed pair).
# The config was written by strata_control/split_engine.py's split_config: the UD-Q4_K_XL set (the engine reads Q4_K/Q5_K/Q8_0
# experts), the model's draft layer (--mtp) instead of DFlash2, and the experts the two cards cannot hold kept in page-locked
# RAM (FNX_ARENA=1, FNX_EXPERT_RAM_GIB=66).  NOTE: the RAM rule refuses this PC (about 108 GiB needed, 94.2 GiB in it), so this
# is an experiment: with pinned RAM short, the engine may fail at start and the server falls back to the normal layer split.
# Pinned RAM wants ulimit -l unlimited (systemd: LimitMEMLOCK=infinity); the shell default (8 MiB) is not enough.
cd "/home/dmitriy/projects/ai/LLM/Strata"
exec "/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python" "/home/dmitriy/projects/ai/LLM/Strata/serve/server.py" "--engine" "strata" "--config" "/home/dmitriy/projects/ai/LLM/Strata/strata-uncensored-q4_k_m.split.json" "--port" "8080"
