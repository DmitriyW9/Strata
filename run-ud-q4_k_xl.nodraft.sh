#!/bin/sh
# The drafting-off arm of strata-ud-q4_k_xl.pcie-auto.json: the same engine, the same pack, the same cards, and no
# drafter at all, so the verify window that faulted on 0.1.66 is never built. Port 8092 beside the served 8080, and on
# localhost only. Why: docs/LOCAL-BOX.md, "2026-11-24: the first start on 0.1.66 faulted in the verify window".
cd "/home/dmitriy/projects/ai/LLM/Strata"
exec "/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python" "/home/dmitriy/projects/ai/LLM/Strata/terminal/serve/server.py" "--engine" "strata" "--config" "/home/dmitriy/projects/ai/LLM/Strata/strata-ud-q4_k_xl.nodraft.json" "--port" "8092"
