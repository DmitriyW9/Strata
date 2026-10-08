cd "/home/dmitriy/projects/ai/LLM/Strata" &&
exec "/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python" \
  "/home/dmitriy/projects/ai/LLM/Strata/serve/server.py" \
  --engine strata \
  --config "/home/dmitriy/projects/ai/LLM/Strata/strata-uncensored-q5_k_m.json" \
  --port 8080 \
  --open
