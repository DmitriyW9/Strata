# Naive-N0.5-Flash (`naive_n05_flash`): the port's state

This page tracks the engine path for **Naive-N0.5-Flash** models, the `strata-naive` executable
(`src/naive`, built with `-DSTRATA_BUILD_NAIVE=ON`).  It is a separate binary on purpose: Strata's
`strata` engine compiles its CUDA kernels against ONE shape (`include/strata/core/geometry_checks.hpp`
static_asserts the Flash-Next geometry), and this architecture shares none of it - like the
`strata-dsv41` path, a second shape means a second binary.

**Status today: the model is read, checked and served; the trunk's arithmetic, the weight loader,
the first two real layers (the dense layer 0 and one window layer), the IQ2 expert FFN, the DSA
(indexer) layer and a two-layer trunk slice (layer 0 + layer 1 + the output head) are built and
unit-tested against an independent numpy reference; a prompt is still refused.**  `strata-naive`
starts under the web server, answers `/v1/models` and the chat page, and refuses every request with
one honest line: `the strata-naive trunk forward is not implemented yet`.  Nothing is guessed and
nothing answers wrong.

## What the model is (measured on Naive-N0.5-Flash-MQ87, not copied)

| | |
| --- | --- |
| Size | 308.9 B parameters, 80.94 GiB of weights in 4 GGUF shards (613 tensors) |
| Layers | 48: 9 full-history DSA + 39 sliding-window (window 128); layer 0 dense, layers 1-47 MoE |
| MoE | 256 experts per layer, top-8, sigmoid scores, `noaux_tc` selection with a correction bias (`exp_probs_b.bias`) |
| Attention | GQA, 64 query heads; 4 kv heads on the DSA layers, 8 on the window layers; K 192 / V 128, value scale 0.707; a sink scalar per query head on the window layers |
| Indexer | DSA layers only: 16 heads x 128, one kv head, top-2048, FP8 E4M3 activations; BF16 matrices |
| RoPE | split-half (GPT-NeoX), two thetas (1e7 full / 1e4 window), 64 dims, partial factor 0.334 |
| Quant | experts gate/up IQ2_XXS, down IQ2_XS; Q/O Q6_K; K/V Q8_0; indexer BF16; router, norms and sinks F32 |
| Draft | external DSpark GGUF (`draft/`): 5 layers, 7-token blocks, 8 hidden taps, Markov rank 256, confidence head; shares the target's embedding and output |
| Context | 1 048 576 tokens; vocab 152 576 (byte-level BPE, pre-tokenizer `qwen2`) |

The KV cache at the model's own context is about 22.6 GiB (DSA layers keep every position, window
layers the window plus a chunk); `strata-naive` prints it as an INFO line so the plan is checkable.

The model's own provenance keys say the same thing: its `runtime` row names an external Rust runtime
and notes that a Naive-specific GQA/DSA integration is required.  That integration is the work below.

## Why `strata` cannot run it (verified this session)

- `strata --families` answers `qwen4exp qwen35moe`; the family gate refuses an unknown architecture before any load.
- The compiled-geometry asserts pin head_dim 256, hyper-connections, GDN on 36 of 48 layers, a 4-head
  block-4 QSA indexer and one RoPE theta.  This model has head_dim 192/128, no hyper-connections, no
  GDN, a 16-head FP8 indexer with top-2048, and two thetas.
- No code path reads a routing correction bias, and no router scores with sigmoid + `noaux_tc`.

So the port is its own binary, and `strata`'s default path stays byte-identical.

## What exists today (validated on the real 4-shard file)

| Piece | File | State |
| --- | --- | --- |
| Geometry (every metadata key, per-layer lists) | `include/strata/naive/geometry.hpp` | reads the model; cross-key rules in `check_geometry` |
| Reader + tensor inventory (types, shapes, shard bounds) | `include/strata/naive/reader.hpp` | `--check` passes on MQ87 and its draft |
| `strata-naive` binary (`--check`, serve INFO/READY, honest refusal) | `src/naive/naive_main.cpp` | builds CPU-only; serve protocol verified against `serve/server.py` |
| CMake target, OFF by default | `CMakeLists.txt` (`STRATA_BUILD_NAIVE`) | does not change any existing target |
| `qwen2` pre-tokenizer (llama.cpp's QWEN2 rule, QWEN35 minus `\p{M}`) | `tools/strata_tokenizer.py` | unit tests pass; encodes/round-trips the model's own vocabulary |
| Tokenizer folder from the GGUF (vocab, merges, types, chat template) | `tools/naive_extract_tokenizer.py` | wrote the pack folder; the server loads it |
| Run config and script (port 8081, next to the running 8080 engine) | `strata-naive-naive.json`, `run-naive.sh` | the server starts, lists the model, refuses requests in 0.3 s |
| Trunk arithmetic (RMSNorm, split-half RoPE, sink attention, FP8 E4M3, index LayerNorm, stable top-k, router, combine, SiLU) | `include/strata/naive/kernels.hpp` | CPU-only header; every function cites its NaiveRT reference |
| A numpy oracle for that arithmetic + fixtures | `tools/naive_kernels_oracle.py`, `tests/naive/fixtures/` | independent transcription of the same references; 64 fixture arrays |
| Parity test (this header vs the oracle's fixtures) | `tests/naive/test_naive_kernels.cpp` (`ctest -R naive_kernels_test`) | 18 checks pass: FP8 codes exact, floats within accumulation tolerance |
| Weight loader (mmap window, ggml's own dequant, F32 copy) | `include/strata/naive/loader.hpp` | CPU-only header; dequant is `ggml_get_type_traits(t).to_float`, not a re-derived codebook |
| Real-weight fixtures (router, bias, norm, raw bytes - straight from the model file) | `tools/naive_real_fixture.py`, `tests/naive/fixtures/real_*`, `raw_router/`, `dequant_f32/` | reads shard 1 with plain `seek`/`read`; stores the file's own FNV-1a so C++ compares against the bytes, not against python |
| Loader test (ggml tables, a synthetic GGUF, IQ2 sign property, the real model) | `tests/naive/test_naive_loader.cpp` (`ctest -R naive_loader_test`) | passes with the model (shards, 613-tensor sizes, byte windows, the router over real weights); prints a skip line and exits 77 without it |
| Quantised GEMV (`gemv_dequant`: ggml's decoder one output row at a time, no resident weight) + `rope_rows` | `include/strata/naive/loader.hpp`, `include/strata/naive/kernels.hpp` | CPU-only header; the row, not the whole weight, must be whole blocks |
| Real-layer fixtures (layer-0 MLP, window attention, numpy decoders written from the format, never calling ggml) | `tools/naive_real_layer_fixture.py`, `tests/naive/fixtures/layer0_dense__*`, `swa_small__*`, `swa_window__*`, `gemv_probe__*` | 22 arrays (14 MB); the generator self-tests its Q6_K decoder against a literal transcription of ggml's loop and its Q8_0 decoder bit-exact |
| Layer test (GEMV + norms + RoPE + sink attention + SwiGLU over the real Q8_0/Q6_K weights) | `tests/naive/test_naive_layer.cpp` (`ctest -R naive_layer_test`) | passes with the model in 1.6 s (every tensor within one bf16 ulp of the numpy reference); skip line and exit 77 without it |

`--check` needs no GPU and loads no weights:

```sh
engine-cuda12/strata-naive --native <shard1.gguf> --draft <dspark.gguf> --check
# ok: naive_n05_flash, 48 layers (9 DSA + 39 SWA), 256 experts top-8 sigmoid/noaux_tc, 4 shards
```

## The arithmetic core is built and tested without the model

The forward's numbers live in `include/strata/naive/kernels.hpp`, one function per piece, each a direct
transcription of a named reference in the model authors' runtime (NaiveRT): RMSNorm; split-half RoPE on
the first 64 dims (theta 1e7 on the DSA layers, 1e4 on the window layers, the rest copied); softmax
attention at scale 1/sqrt(192) with a per-head sink logit added to the denominator (window layers) or
without it (DSA); the index-key LayerNorm; FP8 E4M3 row quantisation and the index logits
`sum_h relu(q_h . k_j) * w[h]`; stable top-2048 selection; the sigmoid + correction-bias router with
renormalised weights; the weighted expert combine; and the SwiGLU activation.

Because NaiveRT's kernels target Blackwell and torch is absent here, the check is a **numpy oracle**
(`tools/naive_kernels_oracle.py`) - the same references, transcribed by hand into numpy with different
algorithms where they can differ (a table search for FP8 where the C++ splits with `frexp`, pairwise
reductions where the C++ sums in double). It writes binary fixtures (`tests/naive/fixtures/`: an ASCII
`dtype rows cols` header plus raw float32/int32, exact at the bit, ~5 MB total, regenerated from a fixed
seed; `--print CASE` shows any case as text) that `tests/naive/test_naive_kernels.cpp` reads and checks
the header against: integers (expert ids, FP8 codes, top-k indices with their tie rule) must match
bit-for-bit, floats within a tolerance that admits the fp32-vs-double spread but no formula bug.
`ctest -R naive_kernels_test` runs it anywhere - no GPU, no model, no torch.

What this does **not** yet prove: that the oracle equals the true Blackwell kernel. That gap is closed
by reading (every function cites its `run_attn_swa.py` / `run_attn_dsa.py` / `training_moe.py` source)
and, on a box that can build NaiveRT, by running that project's own `tests/gpu` harnesses against them.
No generation number has been measured yet: the router answers over real weights (next section), but no
full forward runs until the layer wiring below lands.

## The weight loader is built and tested (real model included)

`include/strata/naive/loader.hpp` reads weights out of the shards on top of the reader. The model is
81 GiB and this machine keeps a running engine resident beside it, so the loader never slurps a file:
it hands back a pointer to a tensor's bytes in the mmap (no copy) and dequantises into fp32 on demand.
The dequant is ggml's own per-type converter (`ggml_get_type_traits(t).to_float`) - the same reference
the model authors' runtime loads through - not a re-derived IQ2 codebook. GGUF stores dim 0 fastest, so
a loaded tensor comes back in that element order with its shape carried; nothing silently transposes it.

`tests/naive/test_naive_loader.cpp` (`ctest -R naive_loader_test`) checks the port's own layer, four ways:

- **Byte counts**: three independent tables (ggml's size functions, the reader's block geometry, the
  test's hard-coded rows) agree for all six types the model uses - for every one of the real file's
  613 tensors, each of which also has to lie inside its shard.
- **The file's own bytes**: `raw_tensor` returns a window whose FNV-1a hash equals a hash stored from
  the file by `tools/naive_real_fixture.py`, so C++ compares against the bytes, not against python.
  The same fixture holds the F32 router weight and bias, and `load_tensor_f32` returns them byte-exact
  (ggml carries no fp32 decoder - upstream is the same, fp32 needs none - so the loader copies those).
- **A synthetic GGUF** written by the test itself: five quantised types plus ragged and unknown-type
  tensors. A type id outside ggml's table (a u32 from a file ggml never saw) gets a refusal, not the
  abort ggml's size switch would raise on it.
- **IQ2 without the codebook**: the test flips the 7 sign bits of one sub-block inside a real-length
  IQ2_XXS and IQ2_XS block and checks that exactly those 8 values negate and the other 248 stay
  byte-exact. That pins each block's stride and element order without transcribing 1024 weights.

On top of the loader, the tested router runs over the model's real weights: the file's own
`blk.1.ffn_gate_inp.weight` and `blk.1.exp_probs_b.bias` load through `load_tensor_f32`, and
`kernels.hpp::moe_route` over a fixed test vector returns the same 8 experts and renormalised weights
as the independent python oracle. Without the model file the test prints a skip line and exits 77, so
`ctest` stays green anywhere; the model-free part is 16 checks on its own.

## The first real layers run and are tested against numpy

The next step is a quantised GEMV over the model's own weights feeding the tested kernels, checked
against a reference that never calls ggml. Two pieces:

`gemv_dequant` (`loader.hpp`) computes `y = x @ W` for a named 2-D weight by decoding ONE output row
at a time through ggml's `to_float` and dotting it in double. The fp32 buffer is one row (16 KiB for
this model), reused per output row, so a GEMV never materialises a weight - the whole 81 GiB model
stays on disk. It refuses (naming the tensor) a wrong shape, a type with no decoder, F32 (no decoder
by design), and rows that are not whole blocks - the ROW is checked, not only the tensor total, since
a {48, 4} Q8_0 tensor is whole blocks while its rows are not.

`tools/naive_real_layer_fixture.py` is the reference. It decodes the same bytes with numpy decoders
written from the block layout in `ggml-common.h` - the Q6_K element map derived element by element,
then validated against a literal line-by-line transcription of ggml's decode loop, and the Q8_0
decoder bit-exact against the layout - so the fixtures do not inherit a ggml misread, and the C++
side does not inherit a python one. Each projection rounds to bf16 where the reference does (norms,
projections, the activation, the value scale), per `run_attn_swa.py`; the value scale is folded onto V
after the projection as bf16, the reference's weight-folding differs by at most one bf16 ulp, and both
sides here take the same post-projection choice.

`tests/naive/test_naive_layer.cpp` (`ctest -R naive_layer_test`) runs the port's GEMV plus the tested
kernels and compares against those fixtures, on real weights:

- the layer-0 MLP: RMSNorm -> gate/up -> `silu(gate) * up` -> down (five Q8_0 tensors, 6 arrays);
- one sliding-window layer twice (9 rows and 141 rows, so the 128 window truncates to `first=13` in
  the second case): norm -> Q/K/V (Q6_K, Q8_0), split-half RoPE at each key's own position, GQA over
  contiguous head groups (64 q / 8 kv), the per-head sink as an extra logit, the O projection;
- a raw one-row GEMV over the real `blk.1.attn_v` with no bf16 round, at 1e-6 relative;
- `gemv_dequant` itself on a synthetic GGUF this test writes, against the same weight read whole -
  bit-exact, which pins the row-offset arithmetic - plus four named refusals.

Every checked tensor is bf16-rounded on both sides, so the bound is one bf16 ulp (0.4%); the test
accepts 0.8%, which leaves the fp32-vs-fp64 pre-round flip room and rejects anything larger. A wrong
row offset, head mapping or window bound moves values by hundreds of percent, not 0.8%. The GQA head
mapping (head h reads kv head h/8) is the convention NaiveRT's attention call uses; GGUF bytes cannot
confirm a head mapping, so both sides transcribe the reference - the one part of this step that is
read, not measured. The whole real pass decodes about 2.4 G elements single-threaded and runs in 1.6 s.

## The IQ2 experts, the DSA layer and a trunk slice run and are tested

The next pieces are the model's two hardest paths and the wiring between them: layer 1's expert FFN
(IQ2_XXS / IQ2_XS, the narrowest quant the model carries), the DSA indexer layer (BF16 projections,
an E4M3 selection cache), and a two-layer trunk that ends in real logits.

Two read helpers carry them. `gemv_dequant_3d` (`loader.hpp`) reads ONE expert's slice of a
`{n_in, n_out, 256}` stack: the file's own `quantization.tensor_layout` key names the order
(`[expert, output, input]`), so expert *e* is the element range `[e*n_in*n_out, +n_in*n_out)` and
`ggml_payload_bytes` turns that into a byte window; one expert of layer 1 is 8.4 M elements (~2 MB),
while the full stack is 6.4 G elements (12 GiB) and is never touched. `load_tensor_cols` reads one
column of a 2-D table - a token's embedding row is column *t* of `token_embd`, 4096 elements out of
a 2.4 GB tensor.

`tools/naive_real_moe_dsa_fixture.py` is the reference. It decodes the same IQ2 bytes with the
vectorised numpy transcriptions in `tools/naive_iq2.py`, themselves tested bit-for-bit against a
literal line-by-line transcription of ggml's decode loops (the IQ2 codebook tables are format data
shared by construction, not re-typed; the element mapping, the sign tables and the scale nibbles are
computed differently on the two sides). It reads the experts and the embedding row by one-window byte
ranges - the same windows `gemv_dequant_3d` and `load_tensor_cols` open - and follows the rounding
graph boundary by boundary: the rmsnorm here is the *training* path (`training_moe.py:rmsnorm` - the
normalized vector rounds to bf16 before gamma multiplies; the layer-0/SWA fixtures above use the
round-after-gamma variant their reference uses, and the two differ by at most one bf16 ulp), the
router keeps `training_moe.py:route`'s split (p + bias decides which experts, the unbiased p is
renormalised by `sum + 1e-20`), and the DSA chain keeps `run_attn_dsa.py`'s four rounds on the
indexer key (bf16 -> LayerNorm -> bf16 -> rope -> bf16) before the E4M3 row quant. One deviation is
recorded rather than hidden: the reference quantises its *indexer weights* to fp8 from its own
checkpoint, while this GGUF ships them as BF16, so both sides read bf16 here -
`include/strata/naive/trunk.hpp` carries the block-fp8 round-trip (`packing.py:quantize_fp8_block`)
for a checkpoint that ships it, built and compiled by the test, unused on this file.

`tests/naive/test_naive_moe_dsa.cpp` (`ctest -R naive_moe_dsa_test`, cases `moe | dsa | trunk`) runs
the port against those fixtures on real weights:

- **moe** - the layer-1 FFN for one token: the training rmsnorm over the real F32 gamma, the router
  over the real gate and bias (8 ids exact), one expert's SwiGLU over the IQ2 windows, the bf16
  combine.  9 checks, 0.19 s.
- **dsa** - layer 0 for one query row over 24 cache rows: the k/v/indexer-key projections, the
  LayerNorm + rope + E4M3 selection cache, the index-query rope + E4M3, the score weight with the
  1/sqrt(16) fold, the selection logits over the fp8 rows, the stable top-10 pick (exact ints), and
  the grouped 64/4 softmax attention with no sink.  12 checks, 0.27 s.
- **trunk** - a two-token sequence: the embedding row read as one column and compared exact, layer 0
  (DSA attention + dense SwiGLU, residuals bf16(x+d)), layer 1 (SWA attention with sinks and theta
  1e4 + the MoE FFN over all 8 routed experts), the output norm, and the logits against the output
  head - the top-5 ids exact and their values within one bf16 ulp.  16 checks, 1.89 s; the head
  decodes 625 M elements in 8192-row chunks, never dequantised whole.

`ctest -R naive` is 4/4 in 3.97 s.  Regenerating the fixtures is byte-reproducible: the 50 arrays
(moe 10, dsa 13, trunk 27) hash the same after a fresh run of the generator, so the seeds and the
reference are stable, and a formula change cannot slip in as a silent fixture edit.  The same
tolerance rules as the layer test apply (a bf16-rounded array one bf16 ulp wide; fp32-stage arrays -
the router weights, the fp8-dequantised rows - 1e-4, both sides summing in double; ids exact).  The
GQA head mapping and the BF16-indexer deviation are read, not measured, as the section above says.

What the trunk slice does **not** close: the running cache (every layer here starts from its own
fixture rows), the 48-layer loop, sampling, and the draft. A prompt is still refused; the numbers
above pin the forward of a two-layer slice, not a served generation.

## What is left before it answers a prompt

The loader, the router, the first two layers, the expert FFN, the DSA layer and a two-layer trunk
slice now work over the real file (tested above); steps 1-5 below are closed. What remains is the
48-layer runtime around them and the draft.

1. **Weight loader**: landed as `include/strata/naive/loader.hpp` - see the section above. The BF16
   indexer and the F32 router repacks ride on the same `load_tensor_f32` path as layer work needs them.
2. **Router**: closed end to end - `kernels.hpp::moe_route` over the model's own layer-1 router weight
   and bias equals the python oracle's top-8 ids and renormalised weights, in the loader test.
   Left for later: per-token batching and the recorded-prompt check once a full forward exists.
3. **Dense layer 0 + one window layer end to end**: landed - `gemv_dequant` over the real Q8_0/Q6_K
   weights drives the layer-0 MLP and the window layer's attention, within one bf16 ulp of the numpy
   reference (the section above). Left for later: the residual adds and per-token batching, which need
   the surrounding layers.
4. **IQ2 experts** (IQ2_XXS gate/up, IQ2_XS down): landed - `gemv_dequant_3d` reads one expert's byte
   window and decodes it through ggml's `to_float`; the layer-1 FFN for a token (one expert, and all
   eight in the trunk case) equals the numpy IQ2 reference within one bf16 ulp (the moe and trunk
   cases above). The combine and activation were already tested.
5. **DSA layers**: landed - the projections over the real BF16 weights, the E4M3 selection cache, the
   index logits and the stable top-k run in `dsa_chain` and equal `run_attn_dsa.py` on real weights
   (the dsa case above). Left for later: the full-history KV across all 9 DSA layers and the running
   cache, which need the 48-layer loop.
6. **Draft**: the DSpark block drafter (5 layers, taps, Markov head, confidence) and verification.
   Opt-in: the port serves correctly without it.  No numpy reference exists for it yet, so it is not built.
7. **Long context**: the window/index caches at 262k and 1M, admission checked before start.

The model's family reference (its recipe names it) is the upstream Rust runtime; the read-state table
above is what Strata's own path replaces.  Every number here was measured on this machine. The loader,
the router, the first two layers, the expert FFN, the DSA layer and a two-layer trunk slice over real
weights have been run (the four `ctest -R naive` tests, model included); the 48-layer loop, sampling
and the draft have no measurement yet, because none of them exists yet - this page says so rather than
implying a run.

## Files named by this port

- `src/naive/naive_main.cpp`, `include/strata/naive/geometry.hpp`, `include/strata/naive/reader.hpp`
- `include/strata/naive/kernels.hpp` (the trunk arithmetic), `tools/naive_kernels_oracle.py`,
  `tests/naive/test_naive_kernels.cpp`, `tests/naive/fixtures/` (64 arrays)
- `include/strata/naive/loader.hpp` (mmap window + ggml dequant + `gemv_dequant`),
  `tests/naive/fixture_io.hpp` (the fixture reader both tests share), `tools/naive_real_fixture.py`
  (real-weight fixtures), `tests/naive/test_naive_loader.cpp` (`ctest -R naive_loader_test`)
- `tools/naive_real_layer_fixture.py` (the numpy-decoder layer reference),
  `tests/naive/test_naive_layer.cpp` (`ctest -R naive_layer_test`), the 22 `*__*` arrays under
  `tests/naive/fixtures/` (14 MB)
- `include/strata/naive/trunk.hpp` (the block-fp8 weight round-trip, built and compiled by the trunk
  test; unused on this file's BF16 indexer), `tools/naive_iq2.py` (the independent IQ2 numpy decoders)
  and `tools/extract_iq2_tables.py` -> `tools/iq2_tables.py` (the codebook tables parsed from
  `ggml-common.h`, with count and spot checks), `tools/naive_real_moe_dsa_fixture.py` (the MoE / DSA /
  trunk reference), `tests/naive/test_naive_moe_dsa.cpp` (`ctest -R naive_moe_dsa_test`), and the
  50 `moe_layer1__` / `dsa_layer0__` / `trunk__` fixture arrays (15 MB in the whole directory)
- `tools/naive_extract_tokenizer.py`, `tools/strata_tokenizer.py` (the `qwen2` rule), `tools/test_strata_tokenizer.py`
- `strata-naive-naive.json`, `run-naive.sh`
- `Strata-data/packs/naive-n05-flash/tokenizer/` (five files extracted from the model's GGUF)
