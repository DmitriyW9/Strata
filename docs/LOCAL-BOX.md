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
`--ple-io direct`, vision on the CPU reader. Since 2026-11-23 also `--adapt-every 4 --adapt-swaps 96
--adapt-decay 0.7`, which keeps the adaptive-cache curve this box has been running (see the 0.2.20 section), and the
environment `STRATA_SM70_TABLE`, `STRATA_ARGMAX_ROWS_SM70`, `STRATA_KV_HOST_DMA`, `STRATA_MTP_KEEP`,
`STRATA_ROUTE_TAIL_SKIP=7`, `STRATA_SPLIT_CPU_SHARE=0`.

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
- The Terminal starts the engine with `MALLOC_ARENA_MAX=2` (`terminal/strata_control/childenv.py:20-36`; since 0.2.17
  the config's own `"env"` is applied at `terminal/serve/server.py:3737` and Strata's start defaults are filled only
  for the names the config does not carry, `terminal/strata_control/speed_switches.py:183-185`). Measured upstream
  for the server's RSS growth only, on another machine; what it costs a 32-thread box whose experts also run on the
  CPU is **not measured here**. `"MALLOC_ARENA_MAX": "16"` in the config is the way to undo it.
- The new memory guard does nothing here: `make_room` and the SIGSTOP pacer only work under a user manager that
  `systemd-oomd` watches with a kill limit (`terminal/serve/oomd_pacer.py:71,94`), and on this machine oomd is
  inactive with `ManagedOOMMemoryPressure=auto` (checked with `systemctl`). No page-cache drop, no pause at load.
- Volta prompt path: default-on int8 prompt tiles and a record-write kernel (`#6053`), bit-identical, measured on a
  V100-SXM2; `STRATA_I8_SM70=0` is the switch back. Whether the PCIe V100 gains the same is **unmeasured**.
- `strata-vision` no longer uses llama.cpp's `mtmd`; this box runs the CPU reader, and its speed after the rebuild is
  **unmeasured**.

## 2026-11-21: what the v0.2.17 / 0.1.63 merge changed for this box

Nothing in this range touched a Volta or Ada default: `engine/include/strata/core/arch_defaults.hpp`,
`engine/include/strata/kernels/i8_expert_rule.hpp` and `docs/NVIDIA_V100.md` are byte-identical between `58ee32277`
(0.1.62) and `3142b66b9` (0.1.63) — checked with `git diff --quiet`. The new defaults are cc-12.0/12.1, HIP, SYCL,
Metal and Windows rules. What does reach this box is three reversions of what it ran on 2026-11-19:

- **The route tail skip is Off now unless you name it.** The engine's rule for an unset value became 0 on every
  backend (`engine/include/strata/core/tail_skip_default.hpp:2-3,12`, owner decision 2026-11-19; 0.1.42–0.1.62 set
  rank 7 at a CUDA serve start), and the server writes `STRATA_ROUTE_TAIL_SKIP=0` into every engine start it builds
  (`terminal/serve/server.py:3739-3741`, `terminal/strata_control/speed_switches.py:177-186`). This box ran **with**
  it: `strata serve: route tail skip: 5560236 missed experts skipped (5852608 entries) since the start`
  (`strata-ud-q4_k_xl.pcie-auto.log:3968`). Upstream's own numbers for this model family are +29 % decode for rank 7
  on an RTX 5090 with DFlash2 `--spec 6` (KL 0.0126, top-1 97.3 %), while `docs/DETAILS.md` also records 1–2 % slower
  on a Tesla V100 at a 94.8 % expert-cache hit. Both arms are open here, so the config keeps the previous behaviour:
  `"STRATA_ROUTE_TAIL_SKIP": "7"`. It wins because a name the config carries is never rewritten
  (`terminal/strata_control/speed_switches.py:183-185`). Measure 0 against 7 before trusting either.
- **A two-card split arms the CPU prompt share by default** (`engine/include/strata/prefill/share_rules.hpp:105-117`,
  `engine/src/program/generate.cpp:11966-11997`, `STRATA_SPLIT_CPU_SHARE=0` turns it back). The share needs 3 GiB of
  free RAM beside its own buffers (`share_rules.hpp:124`), and its own header says that on tight RAM "chats failed
  and the engine restarted" until it was turned off. This box serves a 110 GB model from 94 GiB of RAM and is down to
  1–7 GiB, so the config keeps the pre-merge path: `"STRATA_SPLIT_CPU_SHARE": "0"`.
- **The CPU expert pool runs its one-batch layer call by default** (`engine/src/kernels/cpu/pool.cpp:853-854`,
  `STRATA_POOL_DEP=0` is the way back), measured bit-exact and 2.9 % off another machine's CPU. Left at the new
  default here and put on the A/B list instead: this CPU is different hardware, and guessing for it is what the last
  paragraph is about.

Not affected: the parser's flag list is the same 198 flags at both refs (nothing our config passes is obsolete), no
directory moved again, `MALLOC_ARENA_MAX=2` and the (here inactive) memory guard are unchanged, and none of the
local configs carries `--resident-budget-gib`, so the one-time RAM-budget rewrite cannot touch them.

## 2026-11-23: what the v0.2.20 / 0.1.66 merge changed for this box

Three releases at once (0.1.63 to 0.1.66). Most of the diff is two new model families with their own binaries
(`engine/src/hy4flash`, `engine/src/inkling2`) and nothing of them touches this qwen path.

**Neither of the two new default switch-tables arms here.** 0.1.65 added a 21-switch table for Volta
(`engine/src/core/nv_defaults.cpp:137-167`, upstream's own +24.32 % decode measured on a dual **V100-SXM2-16GB
NVLink** with locked clocks) and a 16-switch table for Ada (`:74-105`, measured on an RTX 4090). Both are armed only
when every visible card has the same compute capability (`engine/src/program/generate.cpp:4421-4446`); this split is
sm_70 plus sm_89, so the engine skips them and the box keeps running on its two own pins. Whether setting those
switches by hand on a PCIe V100 with an Ada head beats the two pins is open — the +24 % number is NVLink SXM2, a
different card and a different link.

**One default did reach this box, and it is pinned back.** Issue #4870 (`engine/src/program/generate.cpp:15038-15060`)
reads the expert cache's coverage and, for 15-65 %, replaces the adaptive curve with a faster one: every 4 rounds to
2, swaps 96 to 192, decay 0.7 to 0.92, saying so with `strata serve: #4870 mid-coverage expert cache (…): adaptive
defaults every 4 -> 2 …`. This box sits at 43 % coverage (8 822 + 1 847 slots of 24 576 pairs), so the rule would
have fired at the next start. Upstream measured +8 % hit rate and +10 % decode; the code names no hardware, and a
doubled exchange is exactly the pressure a box with 1-7 GiB of free RAM and swap already in use should not take on
on trust. The config therefore names the old numbers itself (`--adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7`;
the rule leaves a value the command line gave, `:15050-15052`), and the new curve is the first thing to measure.

New in 0.1.64-0.1.66 that this box does **not** get by default, and what to think about it:

- low-VRAM track 6A (`engine/include/strata/core/low_vram_rules.hpp`, new file): the auto VRAM reserve drops to
  480/550 MiB for 8.5/12.5 GiB cards — we pass `--vram-reserve-mib 700` and `--vram-reserve-later-mib 2048` by hand,
  so it does not apply; the tiny-cache dampening needs coverage under 15 %, we are at 43 %; `STRATA_L3_PF` may arm
  itself on a card under 13 GiB (`engine/src/program/generate.cpp:10408-10424` reads the current device), so the
  start log of the next start should be read for that line before believing any number.
- `STRATA_ROUTE_TAIL_RENORM` (off by default, `engine/src/core/device_hits_plan.hpp:29`) renormalises the router sum
  when the tail skip drops experts — worth one arm beside the tail-skip A/B.
- `--ssd-experts` with a new three-tier NVMe scheduler (`engine/include/strata/core/ssd_stream_scheduler.hpp`) and
  `--ram-experts-gb` (an alias of `--resident-budget-gib`, which no local config carries). Both opt-in; the SSD one
  forces `--mmap-experts`, so it is a different machine shape than the one this box runs today.
- `STRATA_VHEAD_BITPLANE` (the LM head, cc >= 80) — that is the 4070 SUPER's head, off by default, opt-in.
- The parser's flags went from 198 to 202 and none was removed; all twelve flags this config passes are still read.
- The server's environment order is unchanged where it matters: the config's `env` is applied, and the start defaults
  fill only the names it does not carry (`terminal/serve/server.py:3893-3899`,
  `terminal/strata_control/speed_switches.py:183-185`), so all six pins survive; the new `OMP_NUM_THREADS` injection
  waits for a config that names threads or pool-workers, which this one does not.
- Both pins from 2026-11-21 are still needed and still work: the tail skip rule
  (`engine/include/strata/core/tail_skip_default.hpp`) is byte-identical, and the split CPU-share rule
  (`engine/include/strata/prefill/share_rules.hpp:108-117`) still defaults to armed on a two-card split.

## To measure when the GPU is free (nothing here was measured on 2026-11-19, 2026-11-21 or 2026-11-23)

`terminal/tools/measure_configs.py` says it outright: the server must be stopped first, the arms need the GPU to
themselves — here both cards are full while a model is served. One knob per arm, a config of its own with its own
`port` and `log`, `--max-context` and `--kv-resident` unchanged, and read the engine's own lines: `room for experts`,
`expert cache N slots`, `MiB of VRAM free with everything loaded`, `PCIe share:`, `decode expert cache hit rate`,
`KV streaming`. In the order of expected size:

1. the #4870 curve the config now pins away: drop the three `--adapt-*` args (the engine then prints `#4870
   mid-coverage expert cache (…): every 4 -> 2, swaps 96 -> 192, decay 0.70 -> 0.92`) against keeping them — decode
   tok/s, `decode expert cache hit rate`, and RSS plus swap over an hour, since a doubled exchange is exactly what
   this box has no free RAM for;
2. `STRATA_ROUTE_TAIL_SKIP` 7 against 0 — the biggest single swap on this model (`docs/QUANTS.md:155-160` gives the
   quality side: UD-Q4_K_XL KL 0.056 → 0.086), and the reason this box's evidence is 0.1.62-era; with it,
   `STRATA_ROUTE_TAIL_RENORM=1` as a third arm;
3. `STRATA_SPLIT_CPU_SHARE` 0 against 1 (watch the start-up line and `prompt tok/s`, and whether the engine says only
   so much RAM is free and refuses to arm);
4. `STRATA_POOL_DEP` 1 against 0 — 0.1.64-0.1.66 rewrote the pool's one-batch internals, so even the old A/B is stale;
5. the 21 Volta switches of `engine/src/core/nv_defaults.cpp:137-167` set by hand against the two pins this box uses
   (upstream's +24.32 % is a V100-SXM2 on NVLink with locked clocks, this is a PCIe card at x16 with an Ada head);
6. `--vram-reserve-later-mib` 2048 against 1024 and 1536 (about 1 GiB of cache on the 12 GB card);
7. `"drafting": "mtp"` against the DFlash2 default (the engine's own note says +3 % on code prompts at a window of 6);
8. `--layer-split 41` against `auto` (the log calls 41 and 44 a near tie; 0.1.64-0.1.66 changed the per-stage slot trim
   (`engine/include/strata/core/residency_trim.hpp:16-40`), so re-read the printed split and slot count before comparing);
7. `STRATA_SM70_TABLE` and `STRATA_I8_SM70` on, then off, on this PCIe V100;
8. `MALLOC_ARENA_MAX` 2 against 16 (decode tok/s and RSS over an hour);
9. `STRATA_DP_PREFILL=1` (it wants `STRATA_STAGE_TRIM=0`, no `--draft dflash2` and every expert in RAM,
   `docs/MULTI_GPU.md:155` — so it cannot be tried beside the Terminal's drafting policy).

## What is deployed in `engine-cuda12/` right now

- `strata-naive` — **0.1.66**, rebuilt here on 2026-11-23 from the merged tree with `-DSTRATA_ENABLE_CUDA=OFF
  -DSTRATA_BUILD_NAIVE=ON -DSTRATA_EXPERIMENTAL_SM60=ON`; the build it replaced is kept beside it as
  `strata-naive.old-0.1.63.pre-0.1.66` (and the older ones further back). Nothing was listening on 8081 when it was
  replaced.
- `strata` and `strata-vision` — still **0.1.61** (built 2026-11-17), and the model served on 8080 is served **from
  this checkout** (`/home/dmitriy/projects/ai/LLM/Strata/engine-cuda12/strata`, started 2026-11-23 18:18), so these two
  are not rebuilt, relinked or renamed while it runs. The tree is 0.1.66, which means everything in the two sections
  above lands only at the next rebuild: `tools/packaging/build_split_cuda.sh` (it passes `STRATA_SPLIT_ARCH2=89`
  beside `70` and `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`), a build directory first, and the deployed
  file renamed aside (`strata.old-<old>.pre-<new>`) before the new one goes in.
- What was compiled on 2026-11-23 to prove the merge is sound: the Naive engine and its tests (CPU, `-j2`, and its
  `--check` on the real shards), and `s2_qpn8.cu`, `dequant_bf16.cu`, `layer.cpp`, `expert_cache.cpp`,
  `native_dense.cpp`, `state_codec.cpp`, `prefill.cpp`, `generate.cpp` for `70;89`, plus `kv_host_dma_test` and
  `split_rules_test` — all with `-j1` and a memory cap, because the served model owns the rest. **No link of `strata`
  and no run of the merged CUDA engine**, same reason as before.

## Two hazards on this checkout

**Running upstream's setup or tests from this folder moves the local files away.** `terminal/setup.py:8382-8407`
(`move_over_old_layout`) treats `strata-*.json`, `run-*.sh` and `engine-cuda12/` in the root of a checkout as leftovers
of the pre-0.2.16 layout and **moves them into `terminal/`**, rewriting the paths inside them; the patterns are
`terminal/setup.py:8373-8374` (the directory list still names `engine-cuda12`), the skip-guard is `:8390-8392`, and in
this tree the function has exactly one caller, `setup.main()` at `terminal/setup.py:11418`. `.gitignore` hides what
lands there (`/terminal/strata-*.json`, `/terminal/run-*.sh`), so `git status` only says the root copies were deleted.
It happened three times on this fork: in this folder on 2026-11-19, in a throwaway `git worktree` on 2026-11-21, and
again on 2026-11-23 in a fresh worktree of the merged tree — each time during a plain `python -m pytest` over
`terminal/serve` and `terminal/tests/terminal`. What the 2026-11-21 note blamed was wrong: `setup.update_install`,
which `terminal/tests/terminal/test_fixapp_0217_update_build.py` and `test_update_launch_rules.py` call, does not
reach the move (12 tests, run on 2026-11-23 in a clean worktree, left all five root configs where they are), and
nothing else in `setup.py` renames a root file. Which test inside that run hands the checkout to `setup.main()` is
being pinned down with a plugin that stops the run the moment a root config disappears. Until then the rule is the
one that works: **run that suite from a separate clone or a worktree, never from this root** — and since 2026-11-23
this is not only about tidiness, because the served model runs from `engine-cuda12/strata` in this very folder, and a
move would leave its binary and its configs somewhere else for the next restart.

**Nine tests fail here for the box's own reasons, and none of them came from a merge** (all nine were re-run in
worktrees of `strata/main` and of `pre-merge-2026-11-23` on 2026-11-23 and fail identically at both refs):

- `terminal/tests/terminal/test_connections.py::…test_deeply_nested_telemetry_returns_one_redacted_error` — it
  assumes a 20 000-deep JSON breaks the parser; Python 3.14.10 parses it. The file is byte-identical to
  `strata/main`.
- `terminal/serve/test_server.py::ThinkingBudget::test_tokens_drained_after_the_stop_join_the_thinking` — does not
  exist on `strata/main`; fails the same way at every `pre-merge-*` ref.
- Eight TUI tests: `test_fix2_tui_0216.py::S2Nothing80ColumnsCut` ×2, `test_fix_tui_0216.py::SmallWindow`,
  `test_launch_tui.py` ×2, `test_tuiqa_monitor.py::LightTheme` ×3 — window-layout assertions this terminal fails.

## Naive (local-only) model

`Naive-N0.5-Flash` runs on the CPU path through its own binary. Build it with `-DSTRATA_BUILD_NAIVE=ON`
(`STRATA_EXPERIMENTAL_SM60=ON` for this Volta card), serve it with `run-naive.sh` and
`strata-naive-naive.json` (port 8081, so it can run beside the model on 8080). The engine's own check on the real
shards, engine 0.1.66, this box, 2026-11-23:

```
$ ./build-naive2/strata-naive \
      --native /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/Naive-N0.5-Flash-MQ87-00001-of-00004.gguf \
      --draft  /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/draft/Naive-N0.5-Flash-DSpark-Draft-Q8_0.gguf \
      --check
ok: naive_n05_flash, 48 layers (9 DSA + 39 SWA), 256 experts top-8 sigmoid/noaux_tc, 4 shards
ok: draft, 5 layers, block 7, 8 taps, markov 256
```

The model lives in `models/Naive-N0.5-Flash-MQ87/` (the shard names carry `MQ87` too) and the draft in its `draft/`
subfolder; naming the folder `Naive-N0.5-Flash` gets `missing model shard` from the engine.

`ctest --test-dir build-naive2 -R "naive_(kernels|runtime)"` passes both (2/2). Run by hand, `naive_kernels_test`
wants the fixtures directory as its first argument, or a working directory of `engine/` — from the repo root it
reports 14 failed checks and "cannot open fixture tests/naive/fixtures/…", which is the missing argument, not a
regression (`engine/CMakeLists.txt:302-303` passes that argument for `ctest`).

After the tree move the Naive sources are in `engine/src/naive` and `engine/include/strata/naive`, the fixtures and
the numpy oracle in `engine/tests/naive` and `terminal/tools/naive_*.py`; `docs/NAIVE.md` carries the details.
