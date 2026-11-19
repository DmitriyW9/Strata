# This box: V100-PCIE 32 GB + RTX 4070 SUPER 12 GB

Machine-specific notes for the checkout at `/home/dmitriy/projects/ai/LLM/Strata` (the symlink
`/home/dmitriy/разное/projects/ai/LLM/Strata` is the same tree). Nothing here is upstream policy: it is what this
computer does, and what is still unmeasured on it.

## Hardware and software

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads |
| RAM | 94 GiB (a loaded 110 GB model leaves about 4–7 GiB free) |
| GPU 0 | NVIDIA GeForce RTX 4070 SUPER 12 GB, `sm_89` |
| GPU 1 | Tesla V100-PCIE 32 GB, `sm_70`, on a x16 link (the 4070 SITS ON A ~x4 LINK: the engine measured 2.9 GB/s against the V100's 13.1 GB/s) |
| Driver / CUDA | 580.178.04, CUDA 12.9 in `/usr/local/cuda-12.9` (the system `nvcc` is 13.4 — build with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`) |
| Python | 3.14.10 in `.venv` |
| Memory guard | `systemd-oomd` is **inactive** here and `user@1000.service` has `ManagedOOMMemoryPressure=auto` |

Build for both cards: `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;89"`, and for a Volta card also
`-DSTRATA_EXPERIMENTAL_SM60=ON` (the CMake gate for `arch < 75`). The Naive engine needs `-DSTRATA_BUILD_NAIVE=ON`.
`tools/packaging/build_split_cuda.sh` builds the two-card engine, with `STRATA_SPLIT_ARCH2=89` beside `70`.

## The served model: `strata-ud-q4_k_xl.pcie-auto.json`

`"gpu": [1, 0]` (V100 first, so the 4070 SUPER is the last stage and carries the head and the drafter),
`"layer_split": "auto"`, `"gpu_order": "as_given"`, `--expert-cache auto`, `--prefill auto`, `--kv int8` with
`--kv-resident 32768` at `--max-context 262144`, `--vram-reserve-mib 700`, `--vram-reserve-later-mib 2048`,
`--ple-io direct`, vision on the CPU reader.

### What the engine itself chose on this box (from `strata-ud-q4_k_xl.pcie-auto.log`, engine 0.1.61, 2026-11-19 00:08)

These lines are this computer's own numbers, so they are the baseline to compare any change against:

- cost model: `CUDA0 80 SMs at 1.38 GHz -> 0.66 ms per layer`, `CUDA1 56 SMs at 2.64 GHz -> 0.49 ms`;
- link term: `CUDA0 link 13.1 GB/s x1.00`, `CUDA1 link 2.9 GB/s x4.44`;
- split: `K=44`, `NEAR TIE with K=41`, caches holding 10 985 of 24 576 profiled pairs (~97.5 % of routed mass);
- expert cache: 8 822 slots (25.63 GiB) on the V100, 1 847 slots (5.72 GiB) on the 4070 SUPER; the run ended with
  1 987 MiB free on the 4070 SUPER, which is why `--vram-reserve-later-mib 2048` is a guess worth A/B-ing;
- KV: `32768 of 262144 cells per QSA layer in VRAM, the K/V in 2.84 GiB of pinned RAM`, 95–96 % of KV block reads hit VRAM;
- drafter: DFlash2 on CUDA1, `--spec 6 --draft-conf 0.35`; the engine adds `on a 12 GB NVIDIA card the MTP drafter
  was faster in most measurements (code prompts at a window of 6 gained 3%)`.

Older speed ladder for this model on this box (`benchmark-output/2026-11-06/`): 8k 770–1143 prompt tok/s,
32k 1209–1277, 64k 1253–1283, 262k 976; decode 60–76 tok/s; TTFT 264.7 s at 243 333 prompt tokens.

### Why four switches left the config on 2026-11-19 (engine 0.1.62)

Read from the code, not measured; each is now either a default or inert here:

- `STRATA_STAGE_TRIM=1` — the trim is the default with a split since PR #4522 (`engine/src/program/generate.cpp:5589`),
  and setting it marks "the user asked", which skips the early spill-walk choice of the split
  (`engine/src/program/generate.cpp:5605`).
- `STRATA_HC_Q8=1` — `NativeDense::apply_hc_q8_default` sets it itself for a GGUF that carries Q8_0 hyper-connection
  rows, which the UD files do (`engine/src/core/native_dense.cpp:659-667`).
- `--remote-expert-opt` — needs a remote expert cache (`engine/src/program/generate.cpp:9761`), which this config has
  none of, and it turns off the self-chosen asynchronous tier (`engine/include/strata/core/adapt_async_default.hpp:54`).
- `--spec 4 --spec-min-p 0.5 --mtp DIR` — the drafting policy removes the MTP-only flags when it picks DFlash2
  (`terminal/strata_control/drafting.py:1597`); the real start line in the log carries `--draft dflash2 … --spec 6
  --draft-conf 0.35`. `--mtp-draft-vocab` stays: DFlash2 uses it.

Kept: `STRATA_SM70_TABLE=1` and `STRATA_ARGMAX_ROWS_SM70=1` (community numbers for a Volta: +10–15 % decode and
+0.7–0.8 %, measured on a V100-**SXM2** under Windows — `docs/bench-results/2026-10-08-community-v100-sm70-table/`,
`docs/NVIDIA_V100.md`; **not yet measured on this PCIe V100 under Linux**), `STRATA_MTP_KEEP=1` (keeps the drafter on
the 12 GB last stage, `engine/src/program/generate.cpp:7973,7996`), `STRATA_KV_HOST_DMA=1` (inert on CUDA by itself,
required for `STRATA_DP_PREFILL`).

## What the v0.2.16 / 0.1.62 merge changed for this box

- Two paths in the local configs moved with the tree: `data/expert-profile.bin` and `data/draft_vocab_cyrillic.bin`
  are now `terminal/data/…`. With an unreadable profile a split is refused (`engine/src/program/generate.cpp:4718-4721`).
- The Terminal now starts the engine with `MALLOC_ARENA_MAX=2` (`terminal/strata_control/childenv.py:20-36`,
  `terminal/serve/server.py:3684`). Measured upstream for the server's RSS growth only, on another machine; what it
  costs a 32-thread box whose experts also run on the CPU is **not measured here**. The config's own `"env"` is
  applied after it (`terminal/serve/server.py:3693`), so `"MALLOC_ARENA_MAX": "16"` in the config is the way to undo it.
- The new memory guard does nothing here: `make_room` and the SIGSTOP pacer only work under a user manager that
  `systemd-oomd` watches with a kill limit (`terminal/serve/oomd_pacer.py:71,94`), and on this machine oomd is
  inactive with `ManagedOOMMemoryPressure=auto` (checked with `systemctl`). No page-cache drop, no pause at load.
- Volta prompt path: default-on int8 prompt tiles and a record-write kernel (`#6053`), bit-identical, measured on a
  V100-SXM2; `STRATA_I8_SM70=0` is the switch back. Whether the PCIe V100 gains the same is **unmeasured**.
- `strata-vision` no longer uses llama.cpp's `mtmd`; this box runs the CPU reader, and its speed after the rebuild is
  **unmeasured**.

## To measure when the GPU is free (nothing here was measured on 2026-11-19)

`terminal/tools/measure_configs.py` says it outright: the server must be stopped first, the arms need the GPU to
themselves — here both cards are full while a model is served. One knob per arm, a config of its own with its own
`port` and `log`, `--max-context` and `--kv-resident` unchanged, and read the engine's own lines: `room for experts`,
`expert cache N slots`, `MiB of VRAM free with everything loaded`, `PCIe share:`, `decode expert cache hit rate`,
`KV streaming`. In the order of expected size:

1. `--vram-reserve-later-mib` 2048 against 1024 and 1536 (about 1 GiB of cache on the 12 GB card);
2. `"drafting": "mtp"` against the DFlash2 default (the engine's own note says +3 % on code prompts at a window of 6);
3. `--layer-split 41` against `auto` (the log calls 41 and 44 a near tie);
4. `STRATA_SM70_TABLE` and `STRATA_I8_SM70` on, then off, on this PCIe V100;
5. `MALLOC_ARENA_MAX` 2 against 16 (decode tok/s and RSS over an hour);
6. `STRATA_DP_PREFILL=1` (it wants `STRATA_STAGE_TRIM=0` and every expert in RAM, `docs/MULTI_GPU.md:155`);
7. `STRATA_ROUTE_TAIL_SKIP` 7 against 0 — a quality trade, UD-Q4_K_XL KL 0.056 → 0.086 (`docs/QUANTS.md:155-160`).

## What is deployed in `engine-cuda12/` right now

- `strata-naive` — **0.1.62**, built here on 2026-11-19 from the merged tree with `-DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_BUILD_NAIVE=ON -DSTRATA_EXPERIMENTAL_SM60=ON`; the previous build is kept as
  `strata-naive.old-0.1.61.pre-0.1.62`.
- `strata` and `strata-vision` — still **0.1.61** (built 2026-11-17). The merged tree is 0.1.62, so the served model
  still runs the older engine until these two are rebuilt: `tools/packaging/build_split_cuda.sh` (it passes
  `STRATA_SPLIT_ARCH2=89` beside `70` and `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`). Keep the running
  server's binary in place: the build goes to a build directory first, and the deployed file is renamed aside
  (`strata.old-<old>.pre-<new>`) before the new one is copied in.

## Two hazards on this checkout

**Running upstream's setup or tests from this folder moves the local files away.** `terminal/setup.py:7940-7961`
(`move_over_old_layout`) treats `strata-*.json`, `run-*.sh` and `engine-cuda12/` in the root of a checkout as
leftovers of the pre-0.2.16 layout and **moves them into `terminal/`**, rewriting the paths inside the configs. On
2026-11-19 a run of `python -m pytest terminal/tests/terminal` from this root did exactly that: the five configs, the
four run scripts and the 6 GB `engine-cuda12/` ended up under `terminal/`, where `.gitignore` hides them
(`/terminal/strata-*.json`, `/terminal/run-*.sh`), so `git status` said nothing at all. Everything was moved back the
same day, and the configs were committed. Until this fork's local files move into `terminal/` on purpose, run
`terminal/setup.py` and `terminal/tests/terminal` from a separate clone, not from this root.

**Two tests fail here for the box's own reasons, and neither came from the merge:**

- `terminal/tests/terminal/test_connections.py::…test_deeply_nested_telemetry_returns_one_redacted_error` — it
  assumes a 20 000-deep JSON breaks the parser; Python 3.14.10 parses it. Both files are byte-identical to
  `strata/main`, and the test fails the same way at `pre-merge-2026-11-19`.
- `terminal/serve/test_server.py::ThinkingBudget::test_tokens_drained_after_the_stop_join_the_thinking` — fails the
  same way at `pre-merge-2026-11-19`, and the test does not exist on `strata/main`.

## Naive (local-only) model

`Naive-N0.5-Flash` runs on the CPU path through its own binary. Build it with `-DSTRATA_BUILD_NAIVE=ON`
(`STRATA_EXPERIMENTAL_SM60=ON` for this Volta card), serve it with `run-naive.sh` and
`strata-naive-naive.json` (port 8081, so it can run beside the model on 8080). The engine's own check on the real
shards, engine 0.1.62, this box, 2026-11-19:

```
$ ./build-naive2/strata-naive --native …/Naive-N0.5-Flash-MQ87-00001-of-00004.gguf \
      --draft …/Naive-N0.5-Flash-DSpark-Draft-Q8_0.gguf --check
ok: naive_n05_flash, 48 layers (9 DSA + 39 SWA), 256 experts top-8 sigmoid/noaux_tc, 4 shards
ok: draft, 5 layers, block 7, 8 taps, markov 256 drafting=off
```

After the tree move the Naive sources are in `engine/src/naive` and `engine/include/strata/naive`, the fixtures and
the numpy oracle in `engine/tests/naive` and `terminal/tools/naive_*.py`; `docs/NAIVE.md` carries the details.
