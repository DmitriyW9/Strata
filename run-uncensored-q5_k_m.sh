cd "/home/dmitriy/ai_models/Strata" &&
exec "/home/dmitriy/ai_models/Strata/.venv/bin/python" \
  "/home/dmitriy/ai_models/Strata/serve/server.py" \
  --engine strata \
  --config "/home/dmitriy/ai_models/Strata/strata-uncensored-q5_k_m.json" \
  --port 8080 \
  --open
