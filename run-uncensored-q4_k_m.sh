#!/bin/sh
cd "/home/dmitriy/разное/projects/ai/LLM/Strata"
exec "/home/dmitriy/разное/projects/ai/LLM/Strata/.venv/bin/python" "/home/dmitriy/разное/projects/ai/LLM/Strata/serve/server.py" "--engine" "strata" "--config" "/home/dmitriy/разное/projects/ai/LLM/Strata/strata-uncensored-q4_k_m.json" "--port" "8080"
