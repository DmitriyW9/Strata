#!/bin/sh
# run-naive.sh - serve the Naive-N0.5-Flash model (strata-naive) on its own port.
#
# It starts the web server (serve/server.py) with the strata-naive config, separate from the running
# Flash-Next engine (that one stays on port 8080 and is not touched here).  The strata-naive engine
# validates the model on start and, until the trunk forward lands (docs/NAIVE.md), refuses a request
# with the reason; the server and the model facts still come up, so the setup is checkable now.
#
# To only validate the model without starting the server (no GPU, no weights loaded):
#   engine-cuda12/strata-naive --native <shard1.gguf> --draft <draft.gguf> --check
cd "/home/dmitriy/projects/ai/LLM/Strata"
exec "/home/dmitriy/projects/ai/LLM/Strata/.venv/bin/python" "/home/dmitriy/projects/ai/LLM/Strata/serve/server.py" "--engine" "strata" "--config" "/home/dmitriy/projects/ai/LLM/Strata/strata-naive-naive.json" "--port" "8081"
