// tests/naive/test_naive_runtime.cpp - the runtime layer of the strata-naive path
// (include/strata/naive/runtime.hpp): the state between rows.  The per-layer ARITHMETIC is pinned
// by the other three naive suites against fixtures from the real weights; this one pins what those
// suites do not cover - the running caches (DSA full-history K/V + E4M3 index keys, the window
// ring), the session rewind, the selection fast path, the sampler, and the GEN line - by running a
// whole small naive model end to end and comparing against an INDEPENDENT numpy reference
// (terminal/tools/naive_runtime_reference.py) that walks the same GGUF bytes with the vectorised decoders.
//
//   test_naive_runtime [repo-fixtures-dir] [scratch-dir] [case]
//     argv[1] is the CTest fixtures dir (its grandgrandparent is the repo root, where tools/ is);
//     scratch default: $TMPDIR/strata_naive_runtime;  case: unit | model | all (default all).
//     python3+numpy absent (or the reference failing) -> skip line + exit 77.
//
// The model is a synthetic GGUF the test writes itself: the shapes the layout gate hard-codes
// (n_embd 4096, vocab 152576) with everything else shrunk to what a CPU test can run in minutes -
// 3 layers (layer 0 DSA+dense, layer 1 window+MoE, layer 2 DSA+MoE), 8 heads, 16 experts top-8,
// window 8, index_top_k 2 so the real top-k path runs from the third token on.  Geometry, layout
// and metadata go through the SAME gates the engine runs (read_geometry, check_gguf_layout) - the
// test model is refused by those gates exactly when a runtime would be.
//
// What is compared, and how strictly:
//   * logits per case, elementwise, at the suites' tolerance (rel 0.008 - the fp32-vs-fp64
//     pre-round dot may flip a bf16 rounding boundary - or abs 2e-3);
//   * the next greedy token and the DSA selections per row per layer, exact ints;
//   * cache discipline, bit-exact: logits after a rewind to the branch point must equal the
//     logits of a fresh full prefill of the same rows (bitwise), and logits after continuing the
//     session with a generated token must equal the reference running the continued prompt.

#include "strata/naive/geometry.hpp"
#include "strata/naive/reader.hpp"
#include "strata/naive/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "fixture_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

// ------------------------------------------------------------------ part 1: the pure pieces
void unit_checks() {
    // GEN line: keys in the middle, ids start at the first token without an '=' (the main parser's
    // rule), comma-split ids, unknown keys skipped, temperature/top_k/seed parsed.
    {
        strata::naive::GenReq q;
        std::string err;
        const std::string line =
            "GEN 64 temperature=0.7 top_p=0.95 top_k=40 seed=12345 logprobs=5 10,200,3000,40000";
        if (!strata::naive::parse_gen_line(line, q, err)) check(false, "GEN parse accepts a full line");
        else check(q.max_new == 64 && q.temperature == 0.7f && q.top_p == 0.95f && q.top_k == 40 &&
                       q.seed == 12345ull && q.ids.size() == 4 && q.ids[3] == 40000,
                   "GEN parse: keys, ignored keys, comma ids");
        q.ids.clear();
        check(!strata::naive::parse_gen_line("GEN 8 temperature=1", q, err) && !err.empty(),
              "GEN without ids is refused");
        strata::naive::GenReq p;
        check(strata::naive::parse_gen_line("GEN 1 7,8,", p, err) && p.ids.size() == 2 && p.temperature == 0.f,
              "GEN greedy default and trailing comma");
    }
    // The fast stable top-k against the O(n^2) reference, tie-heavy vectors included.
    {
        std::mt19937 rng(7);
        bool same = true;
        for (int trial = 0; trial < 60 && same; ++trial) {
            const int n = 1 + (int) (rng() % 17), k = 1 + (int) (rng() % 6);
            std::vector<float> v((size_t) n);
            for (int i = 0; i < n; ++i) v[(size_t) i] = (float) (rng() % 4);   // many exact ties
            std::vector<int> a((size_t) std::min(n, k));
            strata::naive::stable_topk(v.data(), n, k, a.data());
            std::vector<int32_t> b;
            strata::naive::stable_topk_fast(v.data(), n, k, b);
            if (a.size() != b.size() || !std::equal(a.begin(), a.end(), b.begin())) {
                same = false;
                std::printf("  trial n=%d k=%d: %zu vs %zu\n", n, k, a.size(), b.size());
            }
        }
        check(same, "stable_topk_fast == stable_topk on tie-heavy vectors (incl. k >= n)");
    }
    // The sampler: greedy is the first argmax; a temperature draw replays bit-identically on seed.
    {
        std::vector<float> lg(64, -3.f);
        lg[9] = 1.f; lg[40] = 1.f; lg[63] = 0.5f;
        strata::naive::Sampler s;
        strata::naive::GenReq q;
        check(s.sample(lg.data(), (int) lg.size(), q, {}) == 9, "greedy takes the first argmax");
        q.temperature = 2.0f; q.top_k = 64; q.seed = 777;
        s.reseed(777);
        const int32_t a = s.sample(lg.data(), (int) lg.size(), q, {});
        s.reseed(777);
        const int32_t b = s.sample(lg.data(), (int) lg.size(), q, {});
        check(a == b && a != 63, "a sampled step replays bit-identically under its seed");
    }
}

// ------------------------------------------------------------------ part 2: the synthetic model
// The GGUF writer follows tests/naive/test_naive_loader.cpp's synthetic file (same v3 layout, same
// 32-byte alignment arithmetic) with the naive geometry's metadata keys added.
template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// Metadata writer that COUNTS its pairs, so the n_kv in the header is exact by construction (the
// reader refuses a file whose declared count and actual pairs differ).
struct Meta {
    std::vector<uint8_t> b;
    size_t n = 0;
    void str(const std::string& k, const std::string& v) {
        put_str(b, k); put<uint32_t>(b, 8); put_str(b, v); ++n;
    }
    void u32(const std::string& k, uint32_t v) {
        put_str(b, k); put<uint32_t>(b, 4); put<uint32_t>(b, v); ++n;
    }
    void i32(const std::string& k, int32_t v) {
        put_str(b, k); put<uint32_t>(b, 5); put<int32_t>(b, v); ++n;
    }
    void f64(const std::string& k, double v) {
        put_str(b, k); put<uint32_t>(b, 12); put<double>(b, v); ++n;
    }
    void boolean(const std::string& k, bool v) {
        put_str(b, k); put<uint32_t>(b, 7); put<uint8_t>(b, v ? 1 : 0); ++n;
    }
    void arr(const std::string& k, const std::vector<int32_t>& v) {
        put_str(b, k); put<uint32_t>(b, 9); put<uint32_t>(b, 5); put<uint64_t>(b, v.size());
        for (int32_t x : v) put<int32_t>(b, x);
        ++n;
    }
};

struct Syn {
    std::string name;
    uint32_t type;                       // ggml/GGUF id: F32 0, Q8_0 8, Q6_K 14, IQ2_XXS 16, IQ2_XS 17, BF16 30
    std::vector<uint64_t> shape;         // GGUF order (dim 0 fastest)
    std::vector<uint8_t> bytes;
};

constexpr int64_t N_EMBD = 4096, N_VOCAB = 152576;   // the shapes the layout gate hard-codes
constexpr int64_t N_LAYERS = 3, N_HEAD = 8, KV_DSA = 4, KV_SWA = 8;
constexpr int64_t K_DIM = 192, V_DIM = 128, N_ROT = 64, IDX_H = 16, IDX_D = 128, TOPK_SEL = 2;
constexpr int64_t N_EXPERT = 16, N_EXPERT_USED = 8, EXPERT_FF = 256, DENSE_FF = 512;
constexpr int64_t WINDOW = 8, CHUNK = 8, CTX = 4096;
constexpr int64_t HISTORY = 16;          // the runtime's --session-history for this test
constexpr int64_t RING_CAP = WINDOW + std::max(CHUNK, HISTORY);   // >= 2*window by the runtime's rule

// Random bytes with a finite fp16 scale window at every block start (the loader test's trick, moved
// down to d ~ 2^-7*(1+e) or ~0.05 so a 4096-dot lands around O(1..40) and nothing nears overflow).
std::vector<uint8_t> fill_blocks(std::mt19937& rng, size_t n, size_t block_bytes, size_t scale_bytes,
                                 uint16_t base, uint16_t span) {
    std::vector<uint8_t> b(n);
    for (uint8_t& v : b) v = (uint8_t) (rng() & 0xff);
    for (size_t off = 0; off + block_bytes <= n; off += block_bytes)
        for (size_t s = 0; s + 2 <= scale_bytes && off + s + 2 <= n; s += 2) {
            const uint16_t d = (uint16_t) (base | (rng() % span));
            std::memcpy(b.data() + off + s, &d, 2);
        }
    return b;
}
uint16_t f32_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
std::vector<uint8_t> fill_bf16(std::mt19937& rng, size_t elems, double sigma) {
    std::vector<uint8_t> b(elems * 2);
    std::normal_distribution<double> nd(0.0, sigma);
    for (size_t i = 0; i < elems; ++i) {
        const uint16_t u = f32_to_bf16((float) nd(rng));
        std::memcpy(b.data() + i * 2, &u, 2);
    }
    return b;
}
std::vector<uint8_t> fill_f32(std::mt19937& rng, size_t elems, double lo, double hi) {
    std::vector<uint8_t> b(elems * 4);
    std::uniform_real_distribution<double> ud(lo, hi);
    for (size_t i = 0; i < elems; ++i) {
        const float f = (float) ud(rng);
        std::memcpy(b.data() + i * 4, &f, 4);
    }
    return b;
}
std::vector<uint8_t> fill_f32n(std::mt19937& rng, size_t elems, double mu, double sigma) {
    std::vector<uint8_t> b(elems * 4);
    std::normal_distribution<double> nd(mu, sigma);
    for (size_t i = 0; i < elems; ++i) {
        const float f = (float) nd(rng);
        std::memcpy(b.data() + i * 4, &f, 4);
    }
    return b;
}

std::string fname(const char* fmt, int l) {
    char buf[96];
    std::snprintf(buf, sizeof buf, fmt, l);
    return std::string(buf);
}
// Q6_K's fp16 d is the block's LAST two bytes (ql 128 + qh 64 + scales 16 + d 2 = 210); random d
// bytes decode to huge or NaN values, so pin them and leave the code bytes themselves random.
std::vector<uint8_t> fill_q6k(std::mt19937& rng, size_t elems) {
    std::vector<uint8_t> b(elems / 256 * 210);
    for (uint8_t& v : b) v = (uint8_t) (rng() & 0xff);
    for (size_t off = 0; off + 210 <= b.size(); off += 210) {
        const uint16_t d = (uint16_t) (0x2C00 | (rng() % 0x60));   // ~2^-7 (1 + e/96)
        std::memcpy(b.data() + off + 208, &d, 2);
    }
    return b;
}
std::vector<uint8_t> q8(std::mt19937& rng, size_t elems) {
    return fill_blocks(rng, elems / 32 * 34, 34, 2, 0x2C00, 0x60);
}
std::vector<uint8_t> q6k(std::mt19937& rng, size_t elems) {
    return fill_q6k(rng, elems);
}

std::vector<Syn> build_model(std::mt19937& rng) {
    std::vector<Syn> ts;
    // Globals: the layout gate fixes these three exactly (main_globals in reader.hpp).
    ts.push_back({"token_embd.weight", 8, {(uint64_t) N_EMBD, (uint64_t) N_VOCAB},
                  q8(rng, (size_t) N_EMBD * N_VOCAB)});
    ts.push_back({"output.weight", 8, {(uint64_t) N_EMBD, (uint64_t) N_VOCAB},
                  q8(rng, (size_t) N_EMBD * N_VOCAB)});
    ts.push_back({"output_norm.weight", 0, {(uint64_t) N_EMBD}, fill_f32(rng, N_EMBD, 0.5, 1.5)});
    for (int64_t l = 0; l < N_LAYERS; ++l) {
        const bool dsa = (l == 0 || l == 2);
        const int64_t nkv = dsa ? KV_DSA : KV_SWA;
        ts.push_back({fname("blk.%lld.attn_norm.weight", (int) l), 0, {(uint64_t) N_EMBD},
                      fill_f32(rng, N_EMBD, 0.5, 1.5)});
        ts.push_back({fname("blk.%lld.attn_q.weight", (int) l), 14,
                      {(uint64_t) N_EMBD, (uint64_t) (N_HEAD * K_DIM)},
                      q6k(rng, (size_t) N_EMBD * N_HEAD * K_DIM)});
        ts.push_back({fname("blk.%lld.attn_k.weight", (int) l), 8,
                      {(uint64_t) N_EMBD, (uint64_t) (nkv * K_DIM)},
                      q8(rng, (size_t) N_EMBD * nkv * K_DIM)});
        ts.push_back({fname("blk.%lld.attn_v.weight", (int) l), 8,
                      {(uint64_t) N_EMBD, (uint64_t) (nkv * V_DIM)},
                      q8(rng, (size_t) N_EMBD * nkv * V_DIM)});
        ts.push_back({fname("blk.%lld.attn_output.weight", (int) l), 14,
                      {(uint64_t) (N_HEAD * V_DIM), (uint64_t) N_EMBD},
                      q6k(rng, (size_t) N_HEAD * V_DIM * N_EMBD)});
        if (dsa) {
            ts.push_back({fname("blk.%lld.indexer.q_proj.weight", (int) l), 30,
                          {(uint64_t) N_EMBD, (uint64_t) (IDX_H * IDX_D)},
                          fill_bf16(rng, N_EMBD * IDX_H * IDX_D, 0.02)});
            ts.push_back({fname("blk.%lld.indexer.k_proj.weight", (int) l), 30,
                          {(uint64_t) N_EMBD, (uint64_t) IDX_D},
                          fill_bf16(rng, N_EMBD * IDX_D, 0.02)});
            ts.push_back({fname("blk.%lld.indexer.proj.weight", (int) l), 30,
                          {(uint64_t) N_EMBD, (uint64_t) IDX_H},
                          fill_bf16(rng, N_EMBD * IDX_H, 0.5)});
            ts.push_back({fname("blk.%lld.indexer.k_norm.weight", (int) l), 0, {(uint64_t) IDX_D},
                          fill_f32(rng, IDX_D, 0.9, 1.1)});
            ts.push_back({fname("blk.%lld.indexer.k_norm.bias", (int) l), 0, {(uint64_t) IDX_D},
                          fill_f32(rng, IDX_D, -0.1, 0.1)});
        } else {
            ts.push_back({fname("blk.%lld.attn_sinks.weight", (int) l), 0, {(uint64_t) N_HEAD},
                          fill_f32n(rng, N_HEAD, 0.0, 1.5)});
        }
        ts.push_back({fname("blk.%lld.ffn_norm.weight", (int) l), 0, {(uint64_t) N_EMBD},
                      fill_f32(rng, N_EMBD, 0.5, 1.5)});
        if (l == 0) {   // the one dense layer (expert.layer_pattern[0] == 0, check_geometry's rule)
            ts.push_back({fname("blk.%lld.ffn_gate.weight", (int) l), 8,
                          {(uint64_t) N_EMBD, (uint64_t) DENSE_FF}, q8(rng, (size_t) N_EMBD * DENSE_FF)});
            ts.push_back({fname("blk.%lld.ffn_up.weight", (int) l), 8,
                          {(uint64_t) N_EMBD, (uint64_t) DENSE_FF}, q8(rng, (size_t) N_EMBD * DENSE_FF)});
            ts.push_back({fname("blk.%lld.ffn_down.weight", (int) l), 8,
                          {(uint64_t) DENSE_FF, (uint64_t) N_EMBD}, q8(rng, (size_t) DENSE_FF * N_EMBD)});
        } else {
            ts.push_back({fname("blk.%lld.ffn_gate_inp.weight", (int) l), 0,
                          {(uint64_t) N_EMBD, (uint64_t) N_EXPERT},
                          fill_f32n(rng, N_EMBD * N_EXPERT, 0.0, 0.05)});
            ts.push_back({fname("blk.%lld.exp_probs_b.bias", (int) l), 0, {(uint64_t) N_EXPERT},
                          fill_f32(rng, N_EXPERT, -0.5, 0.5)});
            ts.push_back({fname("blk.%lld.ffn_gate_exps.weight", (int) l), 16,
                          {(uint64_t) N_EMBD, (uint64_t) EXPERT_FF, (uint64_t) N_EXPERT},
                          fill_blocks(rng, (size_t) N_EMBD * EXPERT_FF * N_EXPERT / 256 * 66, 66, 8, 0x34CD, 0x40)});
            ts.push_back({fname("blk.%lld.ffn_up_exps.weight", (int) l), 16,
                          {(uint64_t) N_EMBD, (uint64_t) EXPERT_FF, (uint64_t) N_EXPERT},
                          fill_blocks(rng, (size_t) N_EMBD * EXPERT_FF * N_EXPERT / 256 * 66, 66, 8, 0x34CD, 0x40)});
            ts.push_back({fname("blk.%lld.ffn_down_exps.weight", (int) l), 17,
                          {(uint64_t) EXPERT_FF, (uint64_t) N_EMBD, (uint64_t) N_EXPERT},
                          fill_blocks(rng, (size_t) EXPERT_FF * N_EMBD * N_EXPERT / 256 * 74, 74, 8, 0x34CD, 0x40)});
        }
    }
    return ts;
}

Meta build_metadata() {
    const std::string p = "naive_n05_flash.";
    auto k = [&](const char* suffix) { return p + suffix; };
    Meta m;
    m.str("general.architecture", strata::naive::kArchMain);
    m.u32("general.alignment", 32);
    m.i32(k("block_count"), (int32_t) N_LAYERS);
    m.i32(k("embedding_length"), (int32_t) N_EMBD);
    m.i32(k("context_length"), (int32_t) CTX);
    m.f64(k("attention.layer_norm_rms_epsilon"), 1e-5);
    m.i32(k("attention.head_count"), (int32_t) N_HEAD);
    m.i32(k("attention.key_length"), (int32_t) K_DIM);
    m.i32(k("attention.value_length"), (int32_t) V_DIM);
    m.i32(k("attention.sliding_window"), (int32_t) WINDOW);
    m.i32(k("attention_chunk_size"), (int32_t) CHUNK);
    m.f64(k("attention.value_scale"), 0.707);
    m.str(k("attention.projection_layout"), "split");
    m.boolean(k("attention.add_swa_sink"), true);
    m.boolean(k("attention.add_full_sink"), false);
    m.i32(k("rope.dimension_count"), (int32_t) N_ROT);
    m.i32(k("rope.dimension_count_swa"), (int32_t) N_ROT);
    m.f64(k("rope.freq_base"), 1e7);
    m.f64(k("rope.freq_base_swa"), 1e4);
    m.f64(k("rope.partial_rotary_factor"), 0.334);
    m.str(k("rope.pairing"), "split-half / GPT-NeoX");
    m.i32(k("index_top_k"), (int32_t) TOPK_SEL);
    m.i32(k("index_head_dim"), (int32_t) IDX_D);
    m.i32(k("index_n_heads"), (int32_t) IDX_H);
    m.i32(k("index_n_kv_heads"), 1);
    m.str(k("indexer_activation_dtype"), "fp8_e4m3");
    m.i32(k("expert_count"), (int32_t) N_EXPERT);
    m.i32(k("expert_used_count"), (int32_t) N_EXPERT_USED);
    m.i32(k("expert_feed_forward_length"), (int32_t) EXPERT_FF);
    m.boolean(k("expert_weights_norm"), true);
    m.i32(k("expert_gating_func"), 2);
    m.str(k("moe.topk_method"), "noaux_tc");
    m.str(k("moe.scoring_func"), "sigmoid");
    m.boolean(k("attention.dsa_enabled"), true);
    m.arr(k("attention.hybrid_layer_pattern"), {0, 1, 0});
    m.arr(k("expert.layer_pattern"), {0, 1, 1});
    m.arr(k("attention.head_count_kv"), {(int32_t) KV_DSA, (int32_t) KV_SWA, (int32_t) KV_DSA});
    m.arr(k("feed_forward_length"), {(int32_t) DENSE_FF, (int32_t) EXPERT_FF, (int32_t) EXPERT_FF});
    m.arr(k("attention.dsa_layers"), {0, 2});
    return m;
}

bool write_gguf(const std::filesystem::path& path, const std::vector<Syn>& ts, const Meta& meta) {
    std::vector<uint64_t> off(ts.size()), nbytes(ts.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        uint64_t n = 1;
        for (uint64_t d : ts[i].shape) n *= d;
        nbytes[i] = strata::naive::ggml_payload_bytes(ts[i].type, n);
        if (nbytes[i] == 0 || nbytes[i] != ts[i].bytes.size()) {
            std::fprintf(stderr, "test model: %s byte count disagrees (%llu vs %zu)\n", ts[i].name.c_str(),
                         (unsigned long long) nbytes[i], ts[i].bytes.size());
            return false;
        }
        off[i] = cur;
        cur += (nbytes[i] + 31) / 32 * 32;
    }
    std::vector<uint8_t> file;
    put<uint32_t>(file, 0x46554747u);
    put<uint32_t>(file, 3);
    put<uint64_t>(file, ts.size());
    put<uint64_t>(file, (uint64_t) meta.n);
    file.insert(file.end(), meta.b.begin(), meta.b.end());
    for (size_t i = 0; i < ts.size(); ++i) {
        put_str(file, ts[i].name);
        put<uint32_t>(file, (uint32_t) ts[i].shape.size());
        for (uint64_t d : ts[i].shape) put<uint64_t>(file, d);
        put<uint32_t>(file, ts[i].type);
        put<uint64_t>(file, off[i]);
    }
    file.resize((file.size() + 31) / 32 * 32, 0);
    const size_t data_start = file.size();
    file.resize(data_start + (size_t) cur, 0);
    for (size_t i = 0; i < ts.size(); ++i)
        std::memcpy(file.data() + data_start + off[i], ts[i].bytes.data(), ts[i].bytes.size());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(file.data()), (std::streamsize) file.size());
    return (bool) out;
}

// ------------------------------------------------------------------ part 3: the runtime vs numpy
std::string g_refdir;  // set by model_checks before any load

naive_test::Arr load_ref(const std::string& case_, const std::string& name) {
    naive_test::Arr a;
    int before = g_failures;
    if (!naive_test::load(g_refdir, case_, name, a, g_failures) || g_failures > before) {
        std::fprintf(stderr, "reference fixture %s__%s missing or unreadable\n", case_.c_str(), name.c_str());
        std::exit(3);
    }
    return a;
}

void compare_case_logits(const char* case_, const char* what, const std::vector<float>& got) {
    const naive_test::Arr ref = load_ref(case_, "logits");
    bool finite = true;
    for (float f : got) finite = finite && (f == f);            // check_floats lets NaN==NaN pass;
    check(finite, (std::string(what) + ": the logits are all finite").c_str());   // finiteness is a
    const int before = g_failures;                                              // a claim of its own
    naive_test::check_floats(case_, what, got, ref, 2e-3, 0.008, g_failures);
    if (g_failures > before) {                          // which value moved the most, for the log
        double worst = 0.0; size_t wi = 0;
        for (size_t i = 0; i < got.size() && i < ref.f.size(); ++i) {
            const double tol = 2e-3 + 0.008 * std::fabs((double) ref.f[i]);
            const double d = std::fabs((double) got[i] - (double) ref.f[i]);
            if (d > tol && d / tol > worst) { worst = d / tol; wi = i; }
        }
        std::printf("  worst: id %zu got %.6g ref %.6g (%.1fx over tolerance)\n", wi, (double) got[wi],
                    (double) ref.f[wi], worst);
    }
}

void compare_case_ints(const char* case_, const std::string& name, const char* what,
                       const std::vector<int>& got) {
    const naive_test::Arr ref = load_ref(case_, name);
    naive_test::check_ints(case_, what, got, ref, g_failures);
}

std::vector<int> sel_as_ints(const std::vector<int32_t>& s) {
    return std::vector<int>(s.begin(), s.end());
}

int model_checks(const std::filesystem::path& scratch, const std::filesystem::path& repo_root) {
    const auto gguf_path = scratch / "strata_naive_runtime.gguf";
    std::mt19937 rng(20261118);
    const auto t0 = std::chrono::steady_clock::now();
    // STRATA_NAIVE_TEST_REUSE=1 keeps an already-written model (the generator is deterministic;
    // handy when iterating on the python side).  The normal path always rewrites it.
    const bool reuse = std::getenv("STRATA_NAIVE_TEST_REUSE") != nullptr &&
                       std::filesystem::exists(gguf_path);
    if (!reuse && !write_gguf(gguf_path, build_model(rng), build_metadata())) return 3;
    std::printf("model: %s %s in %.1f s\n", reuse ? "reused" : "wrote", gguf_path.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Open it through the engine's own gates: geometry read, layout checked, runtime initialised.
    strata::GgufModel model(strata::gguf_split_paths(gguf_path.string()));
    strata::naive::NaiveGeometry g;
    if (auto e = strata::naive::read_geometry(model.meta(), g); !e.empty()) {
        std::fprintf(stderr, "read_geometry refused the test model: %s\n", e.c_str());
        return 3;
    }
    if (auto e = strata::naive::check_gguf_layout(model, g); !e.empty()) {
        std::fprintf(stderr, "check_gguf_layout refused the test model: %s\n", e.c_str());
        return 3;
    }
    std::string rerr;
    strata::naive::NaiveRuntime rt(model, g, 4, HISTORY, rerr);
    if (!rt.ok()) { std::fprintf(stderr, "runtime refused the test model: %s\n", rerr.c_str()); return 3; }
    check(true, "the test model passes read_geometry, check_gguf_layout and the runtime gate");

    // The reference: python3 + the vectorised numpy decoders.  Absent or failing -> CTest skip.
    const auto refdir = scratch / "ref";
    std::filesystem::create_directories(refdir);
    const std::string cmd = "python3 '" + (repo_root / "terminal/tools/naive_runtime_reference.py").string() +
                            "' '" + gguf_path.string() + "' '" + refdir.string() + "'";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::printf("SKIP: the numpy reference failed (rc %d: absent python3/numpy?)\n", rc);
        return 77;
    }
    g_refdir = refdir.string();

    const std::vector<int32_t> p1{10, 200, 3000, 40000};
    // ---- p1: prefill 4 rows (logits on the last), comparing the DSA selections row by row.
    for (size_t i = 0; i < p1.size(); ++i) {
        if (auto e = rt.forward_row(p1[i], i + 1 == p1.size()); !e.empty()) {
            std::fprintf(stderr, "forward_row: %s\n", e.c_str());
            return 3;
        }
        compare_case_ints("p1", "sel_l0_r" + std::to_string(i), "p1 row selection (DSA layer 0)",
                          sel_as_ints(rt.selection_at(0)));
        compare_case_ints("p1", "sel_l2_r" + std::to_string(i), "p1 row selection (DSA layer 2)",
                          sel_as_ints(rt.selection_at(2)));
        compare_case_ints("p1", "moe_l1_r" + std::to_string(i), "p1 row router (layer 1)",
                          sel_as_ints(rt.expert_at(1)));
        compare_case_ints("p1", "moe_l2_r" + std::to_string(i), "p1 row router (layer 2)",
                          sel_as_ints(rt.expert_at(2)));
    }
    compare_case_logits("p1", "p1 logits", rt.logits());
    const std::vector<int> next_ref = load_ref("p1", "next").i;
    strata::naive::GenReq greedy;
    strata::naive::Sampler sampler;
    const int32_t t1 = sampler.sample(rt.logits().data(), (int) rt.logits().size(), greedy, rt.session());
    check(t1 == next_ref[0], "p1 next greedy token matches the reference");

    // Continue decoding: feeding t1 from the caches must equal the reference running the CONTINUED
    // prompt (the caches hold rows 0..4 exactly like a fresh prefill of those rows).
    if (auto e = rt.forward_row(t1, true); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 3; }
    {
        const naive_test::Arr ref = load_ref("p1", "logits_step2");
        naive_test::check_floats("p1", "p1 step 2 (decoded from cache) logits", rt.logits(), ref, 2e-3, 0.008,
                                 g_failures);
        compare_case_ints("p1", "next_step2", "p1 step 2 greedy token (cache-fed)",
                          {(int) sampler.sample(rt.logits().data(), (int) rt.logits().size(), greedy,
                                                rt.session())});
    }
    compare_case_ints("p3", "sel_l0_r4", "decoded row selection (DSA layer 0)", sel_as_ints(rt.selection_at(0)));
    compare_case_ints("p3", "sel_l2_r4", "decoded row selection (DSA layer 2)", sel_as_ints(rt.selection_at(2)));
    compare_case_ints("p3", "moe_l1_r4", "decoded row router (layer 1)", sel_as_ints(rt.expert_at(1)));
    compare_case_ints("p3", "moe_l2_r4", "decoded row router (layer 2)", sel_as_ints(rt.expert_at(2)));

    // ---- p2: branch at token 2 (the session rewinds; the rings still reach).  Then the same p2
    // run fresh: bit-identical logits prove the caches hold exactly the branch's rows, no residue.
    const std::vector<int32_t> p2{10, 200, 999};
    const int64_t same = rt.common_prefix(p2);
    check(same == 2, "common_prefix finds the branch point");
    if (!rt.rewind(same > 0 ? same - 1 : 0)) { std::fprintf(stderr, "rewind refused\n"); return 3; }
    check(rt.session_len() == 1, "rewind cut the session to the branch");
    for (size_t i = 1; i < p2.size(); ++i)
        if (auto e = rt.forward_row(p2[i], i + 1 == p2.size()); !e.empty()) {
            std::fprintf(stderr, "p2 forward: %s\n", e.c_str());
            return 3;
        }
    compare_case_ints("p2", "sel_l0_r2", "p2 last-row selection (DSA layer 0)", sel_as_ints(rt.selection_at(0)));
    compare_case_ints("p2", "sel_l2_r2", "p2 last-row selection (DSA layer 2)", sel_as_ints(rt.selection_at(2)));
    compare_case_ints("p2", "moe_l1_r2", "p2 last-row router (layer 1)", sel_as_ints(rt.expert_at(1)));
    compare_case_ints("p2", "moe_l2_r2", "p2 last-row router (layer 2)", sel_as_ints(rt.expert_at(2)));
    const std::vector<float> rewound = rt.logits();
    compare_case_logits("p2", "p2 (rewound) logits", rewound);
    rt.reset();
    for (size_t i = 0; i < p2.size(); ++i)
        if (auto e = rt.forward_row(p2[i], i + 1 == p2.size()); !e.empty()) {
            std::fprintf(stderr, "p2 fresh: %s\n", e.c_str());
            return 3;
        }
    check(rewound == rt.logits(), "a rewound session's logits are BIT-identical to a fresh prefill");
    if (rewound != rt.logits()) {                                // which bits moved, for the log
        size_t bad = 0; double worst = 0.0; size_t wi = 0;
        for (size_t i = 0; i < rewound.size(); ++i) {
            if (rewound[i] != rt.logits()[i]) ++bad;
            const double d = std::fabs((double) rewound[i] - (double) rt.logits()[i]);
            if (d > worst) { worst = d; wi = i; }
        }
        std::printf("  diff: %zu of %zu values, worst %.3g at %zu (got %.6g want %.6g)\n", bad,
                    rewound.size(), worst, wi, (double) rewound[wi], (double) rt.logits()[wi]);
    }

    // ---- the ring's refusal boundary: past `len - cap + window - 1` the rings cannot reach, and
    // rewind() must say so (the caller then resets; a silent wrong-window read would be the bug).
    rt.reset();
    std::vector<int32_t> long_prompt;
    for (int i = 0; i < RING_CAP + 3; ++i) {
        long_prompt.push_back((int32_t) ((i * 7919) % 151000 + 1));
        if (auto e = rt.forward_row(long_prompt.back(), false); !e.empty()) {
            std::fprintf(stderr, "ring test forward: %s\n", e.c_str());
            return 3;
        }
    }
    const int64_t len = rt.session_len();
    check(!rt.rewind(len - RING_CAP + WINDOW - 2), "rewind past the ring's reach is REFUSED");
    check(rt.rewind(len - RING_CAP + WINDOW - 1), "rewind just inside the ring's reach succeeds");
    std::printf("(ring cap %lld, session %lld)\n", (long long) RING_CAP, (long long) len);
    return 0;
}

// Debug case: fingerprint x after every layer of the branch prompt, the same lines the reference's
// `xdump` mode prints (xor of the fp32 bits is bit-exact across implementations; the sum is slack).
int xdump_checks(const std::filesystem::path& scratch, bool pts) {
    const auto gguf_path = scratch / "strata_naive_runtime.gguf";
    strata::GgufModel model(strata::gguf_split_paths(gguf_path.string()));
    strata::naive::NaiveGeometry g;
    if (auto e = strata::naive::read_geometry(model.meta(), g); !e.empty()) return 3;
    std::string rerr;
    strata::naive::NaiveRuntime rt(model, g, 4, HISTORY, rerr);
    if (!rt.ok()) return 3;
    rt.dbg = pts ? 2 : 1;
    auto fp = [](const std::vector<float>& x) {
        uint32_t xo = 0; double s = 0.0;
        for (float f : x) { uint32_t b; std::memcpy(&b, &f, 4); xo ^= b; s += (double) f; }
        return std::make_pair(xo, s);
    };
    for (int32_t t : {10, 200, 999}) {
        rt.dbg_x.clear(); rt.dbg_pts.clear();
        if (auto e = rt.forward_row(t, false); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 3; }
        if (pts) {
            for (const auto& [name, v] : rt.dbg_pts) {
                auto [x, s] = fp(v);
                std::printf("%s %08x %.6f\n", name.c_str(), x, s);
                if (t == 10 && name == "emb") {           // element-level dump of the first row
                    std::ofstream out("logs/emb_cpp.bin", std::ios::binary);
                    out.write((const char*) v.data(), (std::streamsize) v.size() * 4);
                }
            }
        }
        else
            for (const auto& x : rt.dbg_x) { auto [v, s] = fp(x); std::printf("xor %08x sum %.6f\n", v, s); }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path scratch =
        argc > 2 ? std::filesystem::path(argv[2])
                 : std::filesystem::temp_directory_path() / "strata_naive_runtime";
    std::error_code ec;
    std::filesystem::create_directories(scratch, ec);
    // The repo root, found from the fixtures dir CTest passes (…/tests/naive/fixtures) or from the
<<<<<<< HEAD
    // cwd: the first parent that holds terminal/tools/naive_runtime_reference.py, the numpy reference below.
    std::filesystem::path root = argc > 1 ? std::filesystem::path(argv[1]) : "tests/naive/fixtures";
    root = std::filesystem::weakly_canonical(root.parent_path().parent_path().parent_path(), ec);
    for (int up = 0; up < 5 && !std::filesystem::exists(root / "terminal/tools/naive_runtime_reference.py"); ++up)
=======
    // cwd: the first parent that holds tools/naive_runtime_reference.py, the numpy reference below.
    std::filesystem::path root = argc > 1 ? std::filesystem::path(argv[1]) : "tests/naive/fixtures";
    root = std::filesystem::weakly_canonical(root.parent_path().parent_path().parent_path(), ec);
    for (int up = 0; up < 5 && !std::filesystem::exists(root / "tools/naive_runtime_reference.py"); ++up)
>>>>>>> 26e9c713c7a23f885f059ef0d990daf59fbb3d84
        root = root.parent_path();
    const std::string only = argc > 3 ? argv[3] : "all";

    if (only == "xdump") return xdump_checks(scratch, false);
    if (only == "xdbg2") return xdump_checks(scratch, true);
    if (only == "all" || only == "unit") unit_checks();
    if (only == "all" || only == "model") {
        const int rc = model_checks(scratch, root);
        if (rc) return rc;
    }
    if (g_failures) { std::printf("%d check(s) FAILED\n", g_failures); return 1; }
    std::printf("all naive runtime checks passed\n");
    return 0;
}
