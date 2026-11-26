// tests/naive/synthetic_model.hpp - the small synthetic naive GGUF the naive suites write and
// read: the shapes the layout gate hard-codes with everything else shrunk to what a CPU test runs
// in minutes (3 layers, 16 experts, 8 heads).  Extracted from test_naive_runtime.cpp unchanged, so
// the runtime suite and the pool suite (test_naive_pool.cpp) drive the SAME file through their
// gates - header-only like fixture_io.hpp: two executables including one definition cannot drift.
#pragma once

#include "strata/naive/loader.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace naive_synth {

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

}  // namespace naive_synth
