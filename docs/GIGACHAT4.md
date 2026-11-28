# GigaChat4 (37B-A4B) on the llama.cpp engine path

GigaChat4-37B-A4B is a hybrid model (GatedDeltaNet layers, sparse attention with an indexer,
192 experts with 8+1 active): its shape is not the Flash-Next one, so the resident `strata` engine
cannot run it. Strata runs it through llama.cpp instead - through its own engine program,
`strata-llama`, which speaks the same serve protocol as `strata` and `strata-naive`, so the web
server supervises it the same way.

What was added:

- `third_party/llama.cpp` carries the GigaChat4 model (`general.architecture = "gigachat4"`),
  ported from the llama.cpp fork at `github.com/ajpqs/test_llama.cpp` (branch `gigachat4`,
  commit `0541e50 "model : add GigaChat4"`) onto the vendored tree.
- `terminal/tools/llama/` - the `strata-llama` engine: `--native <shard1.gguf>`, the serve
  protocol (INFO/READY, GEN in, T/DONE out, STOP, QUIT), the session's prefix reuse through
  llama.cpp's own memory, and the sampler copied from `strata-naive` (same request, same tokens).
- `terminal/tools/strata_tokenizer.py` knows the `gigachat4` pre-tokenizer name (the o200k rule,
  which llama.cpp maps `tokenizer.ggml.pre = "gigachat4"` to).
- `packs/gigachat4-bf16/tokenizer/` - the tokenizer folder written from the model's own GGUF by
  `terminal/tools/naive_extract_tokenizer.py`.
- `strata-gigachat4-bf16.pcie-auto.json` + `run-gigachat4.pcie-auto.sh` - the config and the start
  command, the same shape as the `ud-q4_k_xl` pair.

## Building the engine

On this machine (gcc 16 as the default compiler, CUDA 12.9, glibc whose headers CUDA 12.9 does
not know) the CUDA build needs the same two settings the Strata engine itself uses - the CUDA
host compiler `g++-14` and the `tools/cuda_glibc_compat.h` force-include:

```sh
cmake -S terminal/tools/llama -B build-llama-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_LLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;89" \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
      "-DCMAKE_CUDA_FLAGS=-Xcompiler=-include,$PWD/tools/cuda_glibc_compat.h"
cmake --build build-llama-cuda --target strata-llama -j
cp -a build-llama-cuda/strata-llama build-llama-cuda/bin/libllama.so* \
      build-llama-cuda/bin/libggml*.so* engine-cuda12/    # where the config looks for it
```

The `$ORIGIN` rpath is what lets the deployed program find the libraries copied beside it.

The CUDA backend is off without `-DSTRATA_LLAMA_CUDA=ON`; nothing in the `strata` binary changes -
this is its own program, like `strata-naive` and `strata-vision`.

## Running it

```sh
./run-gigachat4.pcie-auto.sh        # serves on :8080 with the config beside it
```

or by hand:

```sh
engine-cuda12/strata-llama --check \
  --native /home/dmitriy/projects/ai/LLM/models/GigaChat4-37B-A4B-bf16-00001-of-00002.gguf
```

The model file is a split set: pass any shard, llama.cpp opens the rest through its `.parts`
naming. The GGUF declares a trained context of 1,048,576 tokens (YaRN factor 32 is carried in the
file, so the engine applies it by itself); the config keeps 131072, which is already a generous
session and leaves room in RAM. The bf16 file is 76.6 GB; the V100's 32 GB keeps 13 layers
(`--n-gpu-layers 13`) and llama.cpp leaves the rest in RAM - see the next section before putting
a second card into `"gpu"`. `--stop-token 128001` makes the engine end a turn at the model's
`<|message_sep|>` closer before it reaches the text; the engine also stops at the model's own
`eos` (2).

## When the prompt reads slowly (measured 2026-11-27, the first real run)

The first 106,927-token prompt read at ~14 tokens/s, and the run log said why:

- `sched_reserve: graph splits = 602 (with bs=512), 31 (with bs=1)` - with `--n-gpu-layers 16`
  across both cards the prefill batch ping-ponged between four backend devices (CUDA0, CUDA1,
  the two CPU tiers); every split is a copy plus a stream sync. The log counted ~400 CUDA graph
  computes and 44 CUDA-graph capture resets per 64-token chunk: the syncs, not the maths, ate
  the time.
- 160k DEBUG lines from the library ("CUDA Graph id reused", "create_tensor: loading tensor")
  went through the engine's stderr into the run log. Without `--verbose` the engine now filters
  llama.cpp's DEBUG/TRACE lines itself (the `INFO` lines the server reads are untouched).
- The V100 (sm_70) has no fast bf16 path: CUDA converts the model's bf16 to f32 there
  (`fast_bf16_hardware_available` in `ggml-cuda.cu`), so its layers cost about twice the
  bandwidth.

The config therefore serves on one card (`"gpu": [0]`, 13 layers - what fits the V100's 32 GB
beside the KV cache), where llama.cpp keeps ~31 splits instead of 602. To try the bf16-native
card instead: `"gpu": [1]` with `--n-gpu-layers 3` (the 4070 SUPER holds ~12 GB). Which one is
faster is not yet measured - watch the server's "reading the prompt: N tokens, T s" line: the
rate should climb from ~14 tok/s toward the dozens. If prefill still crawls, the wall is the
~50 GB of expert weights that stay in RAM by necessity (the model is bf16); converting the file
to q8_0 halves that traffic and the config keeps the same shape either way.

## GigaChat4 on one V100: the Q4_K_M file (the tuned pairing)

The bf16 file cannot fit any single card of this box, so its layers sit partly in RAM and every
prefill chunk crosses the CPU↔GPU border (the section above). The Q4_K_M file changes that:
`GigaChat4-37B-A4B-Q4_K_M.gguf` is 21.65 GiB of weights (4.86 BPW), and at a 131 072-token
context the caches measured from the file itself are 1152 MiB (the 8 MLA layers) + 256 MiB (the
DSA indexer) + 50 MiB (recurrent states) + 518 MiB of compute buffer - together ~24.7 GiB
with the weights, inside the V100-PCIE's 31.4 GiB. So `strata-gigachat4-q4_k_m.pcie-auto.json`
+ `run-gigachat4-q4.pcie-auto.sh` ask llama.cpp for everything: `--n-gpu-layers -1` on
`"gpu": [0]`. No layer stays in RAM, the graph keeps ~1 split instead of hundreds, and the card
serves 900 GB/s of HBM instead of the PCIe's ~12.

The file carries the same tokenizer (`tokenizer.ggml.pre = "gigachat4"`, 128 256 tokens), so the
pack `packs/gigachat4-q4_k_m/tokenizer/` is a copy of the bf16 one. The quantized file keeps its
expert weights in Q4_K and the dense parts in Q6_K - all types the CUDA backend computes on
Volta without bf16 (which the V100 does not have). What this buys is not yet measured; the log
lines to compare after a start are `graph splits` (about 1, not 602), the CUDA0 buffer size, and
the server's "reading the prompt" rate.

## What this path does and does not do (measured and read 2026-11-27)

- One request at a time (no batch slots), no draft, no pictures, no tool-call parsing: GigaChat4
  answers tools in its `｜GCML｜` XML form, which this build passes through as text.
- Thinking works like the qwen35 family: the model's template ends the prompt with an open
  `<think>` and the server reads the reply as reasoning, then text.
- The web server tokenizes with its own o200k rule (`gigachat4` joins `O200K_PRES`); the engine
  takes ids in and gives ids out, so llama.cpp's own approximation of the GigaChat4
  pre-tokenizer never touches a served request.
