#!/bin/sh
cd "/home/dmitriy/projects/ai/LLM/Strata"
exec "/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python" "/home/dmitriy/projects/ai/LLM/Strata/serve/server.py" "--engine" "strata" "--config" "/home/dmitriy/projects/ai/LLM/Strata/strata-ud-q4_k_xl.pcie-auto.json" "--port" "8080"
