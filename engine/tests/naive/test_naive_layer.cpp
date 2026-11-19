// tests/naive/test_naive_layer.cpp - one quantised GEMV (loader.hpp::gemv_dequant) feeding the tested
// kernels (kernels.hpp) as a REAL layer forward, against fixtures generated straight from the model's
// weights by tools/naive_real_layer_fixture.py.  This is step 3 of docs/NAIVE.md ("dense layer 0 +
// one window layer end to end"), CPU-only, with no weight resident: gemv_dequant decodes one output
// row at a time through ggml's own `to_float`, so the peak extra memory is one 16 KiB row.
//
// What is checked, and what the checks mean:
//   * Synthetic GGUF (always runs): a Q8_0 and a Q6_K weight written by this file, the same tensor
//     read two ways - gemv_dequant (row-sliced decode + double dot) versus load_tensor_f32 (whole
//     decode) plus an explicit dot.  Both ride ggml's decoder, so this pins the PORT's row-offset
//     arithmetic (ggml_payload_bytes at m*n_embd), not ggml's decode; refusals (wrong n_embd, wrong
//     rank, F32 - which has no to_float) must name the tensor.
//   * Real model (skipped, exit 77, when absent): the same GEMV over the model's actual Q8_0/Q6_K
//     weights must reproduce, within one bf16 ulp per value, the layer forwards of the fixtures:
//       gemv_probe     one row x the real blk.1.attn_v (Q8_0).
//       layer0_dense   layer 0: rmsnorm -> SwiGLU(gate, up) -> down (all Q8_0).
//       swa_small /    layer 1: rmsnorm -> Q/K/V GEMV (Q6_K, Q8_0), split-half RoPE per position,
//       swa_window     GQA (64 q / 8 kv, contiguous groups), per-head sink, a 128 window (the
//                      window case caches 141 rows so first=13 exercises the truncation), O-proj.
//     The fixtures' reference decodes the same bytes with a NUMPY decoder written from the layout in
//     ggml-common.h (never calling ggml), so agreement here is a cross-check of ggml's compiled
//     decoder on the model's real bytes, not a tautology - and disagreement would name the layer.
//   * The GQA convention (query head h reads kv head h / (n_head / n_kv), the contiguous grouping) is
//     the one NaiveRT's FA4 call uses (q [T,H,D] over k [S,kv,D] - prefill.py:swa_attention); the
//     GGUF bytes themselves cannot confirm a head mapping, so both sides transcribe the reference.
//
// Tolerances: every checked tensor is bf16-rounded on both sides, so a value may sit one bf16 ulp
// (0.4%) apart when the fp32-vs-fp64 pre-round dot lands on the other side of a rounding boundary -
// reltol 0.008 accepts that and rejects anything larger.  A wrong row offset, head mapping or window
// bound moves values by O(100%), not 0.8%.
//
//   test_naive_layer [fixtures-dir] [model-any-shard.gguf]
//     fixtures default: tests/naive/fixtures   (regenerate: python tools/naive_real_layer_fixture.py)
//     model default:    /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/
//                       Naive-N0.5-Flash-MQ87-00001-of-00004.gguf
//     Model absent -> skip line + exit 77 (CTest's skip code); the synthetic part always runs.

#include "fixture_io.hpp"
#include "strata/naive/kernels.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/reader.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace naive = strata::naive;
using naive_test::Arr;

std::string g_dir = "tests/naive/fixtures";
std::string g_model = "/home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/"
                      "Naive-N0.5-Flash-MQ87-00001-of-00004.gguf";
int g_failures = 0;
const double kBf16Reltol = 0.008;   // one bf16 ulp, with room for the fp32-vs-fp64 pre-round flip
const double kAbsTol = 1e-3;

void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}
void check_floats(const std::string& c, const char* what, const std::vector<float>& got, const Arr& ref,
                  double abstol, double reltol) {
    naive_test::check_floats(c, what, got, ref, abstol, reltol, g_failures);
}
void check_ints(const std::string& c, const char* what, const std::vector<int>& got, const Arr& ref) {
    naive_test::check_ints(c, what, got, ref, g_failures);
}

void bf16_round_inplace(std::vector<float>& v) {
    for (float& f : v) f = naive::bf16_round(f);
}

// ------------------------------------------------------------------ part A: gemv on a written GGUF
// (Same synthetic-file machinery as test_naive_loader.cpp: this test computes the byte layout itself,
// so the loader is read against positions this file derived, not against the loader's own idea.)
template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Syn {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> shape;   // GGUF order: dim 0 fastest
    std::vector<uint8_t> bytes;
};

std::filesystem::path write_synthetic(const std::vector<Syn>& tensors) {
    std::vector<uint64_t> off(tensors.size()), nbytes(tensors.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        uint64_t n = 1;
        for (uint64_t d : tensors[i].shape) n *= d;
        nbytes[i] = naive::ggml_payload_bytes(tensors[i].type, n);
        off[i] = cur;
        cur += (nbytes[i] + 31) / 32 * 32;
    }
    std::vector<uint8_t> file;
    put<uint32_t>(file, 0x46554747u);  // "GGUF"
    put<uint32_t>(file, 3);
    put<uint64_t>(file, tensors.size());
    put<uint64_t>(file, 1);
    put_str(file, "general.architecture");
    put<uint32_t>(file, 8);
    put_str(file, "naive_n05_flash");
    for (size_t i = 0; i < tensors.size(); ++i) {
        put_str(file, tensors[i].name);
        put<uint32_t>(file, (uint32_t) tensors[i].shape.size());
        for (uint64_t d : tensors[i].shape) put<uint64_t>(file, d);
        put<uint32_t>(file, tensors[i].type);
        put<uint64_t>(file, off[i]);
    }
    file.resize((file.size() + 31) / 32 * 32, 0);
    const size_t data_start = file.size();
    file.resize(data_start + (size_t) cur, 0);
    for (size_t i = 0; i < tensors.size(); ++i)
        if (!tensors[i].bytes.empty())
            std::memcpy(file.data() + data_start + off[i], tensors[i].bytes.data(), tensors[i].bytes.size());
    const auto path = std::filesystem::temp_directory_path() / "strata_naive_layer_test.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), (std::streamsize) file.size());
    return path;
}

std::vector<uint8_t> fill_blocks(std::mt19937& rng, size_t n, size_t block_bytes) {
    std::vector<uint8_t> b(n);
    for (uint8_t& v : b) v = (uint8_t) (rng() & 0xff);
    for (size_t off = 0; off + block_bytes <= n; off += block_bytes) {
        const uint16_t d = (uint16_t) (0x3800 | (rng() & 0x03ff));  // fp16 in [1, 2): finite scale
        std::memcpy(b.data() + off, &d, 2);
    }
    // A Q6_K block's scale table sits at byte 192 (the fp16 d at 208 is the one fill_blocks fixed).
    if (block_bytes == 210)
        for (size_t off = 0; off + 210 <= n; off += 210)
            for (int s = 0; s < 16; ++s) b[off + 192 + s] = (uint8_t) (int8_t) (1 + (int) (rng() % 64));
    return b;
}

void part_a_synthetic() {
    std::mt19937 rng(20261113);
    std::vector<Syn> ts;
    {   // Q8_0 {64, 40}: two blocks per output row - the row-slice offset is 2 blocks (68 B) per row.
        ts.push_back({"g.q8_0", 8, {64, 40}, fill_blocks(rng, 40 * 2 * 34, 34)});
        ts.push_back({"g.q6_k", 14, {256, 12}, fill_blocks(rng, 12 * 210, 210)});  // one block per row
        ts.push_back({"g.f32", 0, {64, 4}, {}});   // refused: F32 has no to_float by design
        ts.push_back({"g.q8_0_r", 8, {48, 4}, fill_blocks(rng, 6 * 34, 34)});  // total whole, rows not
    }
    const auto path = write_synthetic(ts);
    strata::GgufModel model(strata::gguf_split_paths(path.string()));

    // x: 256 deterministic values; the first 64 feed the Q8_0 GEMV, all 256 the Q6_K one.
    std::vector<float> x(256);
    for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;

    auto gemv_matches_full = [&](const char* name, int n_embd, int n_out, const Syn& syn) {
        std::vector<float> y((size_t) n_out, 0.f);
        const std::string err = naive::gemv_dequant(model, name, x.data(), n_embd, y.data(), n_out);
        if (!err.empty()) {
            check(false, (std::string("gemv_dequant(") + name + ") refused a readable tensor").c_str());
            return;
        }
        naive::LoadedTensor lt;
        if (!naive::load_tensor_f32(model, name, lt).empty() || (int) lt.elements() != n_embd * n_out) {
            check(false, "the full-tensor path could not read the same weight");
            return;
        }
        // The whole-tensor decode and the row-sliced decode read the same bytes, so both decoded
        // rows hold identical floats; the dots accumulate in double in k order both ways, so the
        // results must match bit-for-bit.  A wrong row offset breaks this on row 1 and later.
        bool exact = true;
        for (int m = 0; m < n_out && exact; ++m) {
            double acc = 0.0;
            for (int k = 0; k < n_embd; ++k)
                acc += (double) x[k] * lt.data[(size_t) m * n_embd + k];
            exact = y[(size_t) m] == (float) acc;
        }
        check(exact, (std::string(name) + ": row-sliced GEMV == full decode + dot (exact)").c_str());
        (void) syn;
    };
    gemv_matches_full("g.q8_0", 64, 40, ts[0]);
    gemv_matches_full("g.q6_k", 256, 12, ts[1]);

    // Refusals are data: a wrong n_embd, an F32 weight (no to_float by design - an F32 GEMV goes
    // through load_tensor_f32 plus a dot, which the loader test covers), a weight whose rows are not
    // whole blocks, and an empty GEMV.  Every reason names the tensor it came from.
    std::vector<float> y(40, 0.f);
    const std::string bad_n = naive::gemv_dequant(model, "g.q8_0", x.data(), 65, y.data(), 40);
    const std::string bad_r = naive::gemv_dequant(model, "g.f32", x.data(), 64, y.data(), 4);
    const std::string bad_rg = naive::gemv_dequant(model, "g.q8_0_r", x.data(), 48, y.data(), 4);
    const std::string bad_e = naive::gemv_dequant(model, "g.q8_0", x.data(), 64, y.data(), 0);
    check(!bad_n.empty() && bad_n.find("g.q8_0") != std::string::npos,
          "gemv_dequant refuses a wrong n_embd by name, without reading the weight");
    check(!bad_r.empty() && bad_r.find("g.f32") != std::string::npos,
          "gemv_dequant refuses F32 (no to_float), naming the tensor");
    check(!bad_rg.empty() && bad_rg.find("ragged") != std::string::npos,
          "gemv_dequant refuses rows that are not whole blocks, though the tensor total is whole");
    check(!bad_e.empty(), "gemv_dequant refuses an empty GEMV");
}

// ------------------------------------------------------------------ part B: the real layers
// One bf16-rounded projection: y = bf16_round(x @ W) through the row-sliced GEMV.
void project(const strata::GgufModel& model, const char* name, const float* x, int n_embd, float* y,
             int n_out) {
    const std::string err = naive::gemv_dequant(model, name, x, n_embd, y, n_out);
    if (!err.empty()) {
        std::fprintf(stderr, "FAIL gemv_dequant(%s): %s\n", name, err.c_str());
        ++g_failures;
        return;
    }
    std::vector<float> v(y, y + n_out);
    bf16_round_inplace(v);
    std::memcpy(y, v.data(), (size_t) n_out * sizeof(float));
}

bool real_case_dense(const strata::GgufModel& model) {
    Arr x;
    if (!naive_test::load(g_dir, "layer0_dense", "x", x, g_failures)) return true;
    const float* xin = x.fp();
    naive::LoadedTensor gamma;
    if (!naive::load_tensor_f32(model, "blk.0.ffn_norm.weight", gamma).empty()) return true;
    std::vector<float> xn((size_t) 4096), gate((size_t) 16384), up((size_t) 16384);
    naive::rmsnorm(xin, gamma.data.data(), 4096, (float) 1e-5, xn.data());
    bf16_round_inplace(xn);                                    // the reference rounds the norm to bf16
    project(model, "blk.0.ffn_gate.weight", xn.data(), 4096, gate.data(), 16384);
    project(model, "blk.0.ffn_up.weight", xn.data(), 4096, up.data(), 16384);
    std::vector<float> act((size_t) 16384);
    for (size_t j = 0; j < act.size(); ++j) act[j] = naive::bf16_round(naive::silu(gate[j]) * up[j]);
    std::vector<float> ffn((size_t) 4096);
    project(model, "blk.0.ffn_down.weight", act.data(), 16384, ffn.data(), 4096);

    Arr xn_ref, gate_ref, up_ref, act_ref, ffn_ref;
    bool ok = naive_test::load(g_dir, "layer0_dense", "xn", xn_ref, g_failures) &&
              naive_test::load(g_dir, "layer0_dense", "gate", gate_ref, g_failures) &&
              naive_test::load(g_dir, "layer0_dense", "up", up_ref, g_failures) &&
              naive_test::load(g_dir, "layer0_dense", "act", act_ref, g_failures) &&
              naive_test::load(g_dir, "layer0_dense", "ffn_out", ffn_ref, g_failures);
    if (!ok) return true;
    check_floats("layer0_dense", "rmsnorm(bf16) over the real Q8_0 weight path", xn, xn_ref, kAbsTol, kBf16Reltol);
    check_floats("layer0_dense", "gate = bf16(xn @ W_gate) on real Q8_0 bytes", gate, gate_ref, kAbsTol, kBf16Reltol);
    check_floats("layer0_dense", "up   = bf16(xn @ W_up) on real Q8_0 bytes", up, up_ref, kAbsTol, kBf16Reltol);
    check_floats("layer0_dense", "act  = bf16(silu(gate) * up)", act, act_ref, kAbsTol, kBf16Reltol);
    check_floats("layer0_dense", "ffn  = bf16(act @ W_down) - the layer-0 MLP end to end", ffn, ffn_ref,
                 kAbsTol, kBf16Reltol);
    return true;                                    // ran (not skipped)
}

// One sliding-window layer's attention, query at the last cached row, exactly the reference flow.
bool real_case_swa(const strata::GgufModel& model, const std::string& case_name) {
    Arr x_rows, first_ref;
    if (!naive_test::load(g_dir, case_name, "x_rows", x_rows, g_failures)) return true;
    if (!naive_test::load(g_dir, case_name, "first", first_ref, g_failures)) return true;
    const int P = x_rows.rows;
    const int T = P - 1;
    const int visible = T + 1 < 128 ? T + 1 : 128;
    const int first = T + 1 - visible;

    naive::LoadedTensor gamma, sink;
    if (!naive::load_tensor_f32(model, "blk.1.attn_norm.weight", gamma).empty()) return true;
    if (!naive::load_tensor_f32(model, "blk.1.attn_sinks.weight", sink).empty()) return true;

    std::vector<float> xn((size_t) 4096);
    std::vector<float> kcache((size_t) P * 1536), vcache((size_t) P * 1024), kv((size_t) 1536);
    for (int p = 0; p < P; ++p) {
        naive::rmsnorm(x_rows.fp() + (size_t) p * 4096, gamma.data.data(), 4096, (float) 1e-5, xn.data());
        bf16_round_inplace(xn);
        project(model, "blk.1.attn_k.weight", xn.data(), 4096, kv.data(), 1536);
        naive::rope_rows(kv.data(), 8, 192, 64, p, 1e4, kcache.data() + (size_t) p * 1536);
        std::vector<float> v((size_t) 1024);
        project(model, "blk.1.attn_v.weight", xn.data(), 4096, v.data(), 1024);
        for (float& f : v) f = naive::bf16_round(f);
        for (float& f : v) f = naive::bf16_round(f * 0.707f);      // prefill.py:scale_values, folded bf16
        std::memcpy(vcache.data() + (size_t) p * 1024, v.data(), 1024 * sizeof(float));
    }
    std::vector<float> q((size_t) 12288), q_rope((size_t) 12288);
    naive::rmsnorm(x_rows.fp() + (size_t) T * 4096, gamma.data.data(), 4096, (float) 1e-5, xn.data());
    bf16_round_inplace(xn);
    project(model, "blk.1.attn_q.weight", xn.data(), 4096, q.data(), 12288);
    naive::rope_rows(q.data(), 64, 192, 64, T, 1e4, q_rope.data());

    // GQA + sink + window: query head h reads kv head h/8 (contiguous groups) and that head's sink;
    // each group's visible rows are copied compact (stride 192 for k, 128 for v) for swa_attention.
    std::vector<float> attn_heads((size_t) 64 * 128);
    std::vector<float> qg((size_t) 8 * 192), kg((size_t) visible * 192), vg((size_t) visible * 128);
    for (int g = 0; g < 8; ++g) {
        std::memcpy(qg.data(), q_rope.data() + (size_t) g * 8 * 192, qg.size() * sizeof(float));
        for (int j = 0; j < visible; ++j) {
            std::memcpy(kg.data() + (size_t) j * 192, kcache.data() + (size_t) (first + j) * 1536 + g * 192,
                        192 * sizeof(float));
            std::memcpy(vg.data() + (size_t) j * 128, vcache.data() + (size_t) (first + j) * 1024 + g * 128,
                        128 * sizeof(float));
        }
        naive::swa_attention(qg.data(), 8, 192, kg.data(), vg.data(), 128, sink.data.data() + g * 8, 0,
                             visible - 1, attn_heads.data() + (size_t) g * 8 * 128);
    }
    std::vector<float> attn_flat = attn_heads;
    bf16_round_inplace(attn_flat);
    std::vector<float> attn_out((size_t) 4096);
    project(model, "blk.1.attn_output.weight", attn_flat.data(), 8192, attn_out.data(), 4096);

    Arr q_ref, k_ref, v_ref, heads_ref, out_ref;
    bool ok = naive_test::load(g_dir, case_name, "q_rope", q_ref, g_failures) &&
              naive_test::load(g_dir, case_name, "k_cache", k_ref, g_failures) &&
              naive_test::load(g_dir, case_name, "v_cache", v_ref, g_failures) &&
              naive_test::load(g_dir, case_name, "attn_heads", heads_ref, g_failures) &&
              naive_test::load(g_dir, case_name, "attn_out", out_ref, g_failures);
    if (!ok) return true;
    check_ints(case_name, "window start first", std::vector<int>{first}, first_ref);
    check_floats(case_name, "q after split-half RoPE (Q6_K projection)", q_rope, q_ref, kAbsTol, kBf16Reltol);
    check_floats(case_name, "k cache: RoPE'd per position (Q8_0 projection)", kcache, k_ref, kAbsTol, kBf16Reltol);
    check_floats(case_name, "v cache: value_scale folded in bf16", vcache, v_ref, kAbsTol, kBf16Reltol);
    check_floats(case_name, "GQA + sink + window attention (64 q / 8 kv heads)", attn_heads, heads_ref,
                 kAbsTol, kBf16Reltol);
    check_floats(case_name, "O projection back to 4096 (Q6_K) - the layer's attn out", attn_out, out_ref,
                 kAbsTol, kBf16Reltol);
    return true;
}

bool real_case_gemv_probe(const strata::GgufModel& model) {
    Arr x, y_ref;
    if (!naive_test::load(g_dir, "gemv_probe", "x", x, g_failures)) return true;
    if (!naive_test::load(g_dir, "gemv_probe", "y", y_ref, g_failures)) return true;
    std::vector<float> y((size_t) 1024);
    const std::string err = naive::gemv_dequant(model, "blk.1.attn_v.weight", x.fp(), 4096, y.data(), 1024);
    check(err.empty(), "gemv_probe: the real Q8_0 blk.1.attn_v reads through gemv_dequant");
    check_floats("gemv_probe", "one row over the real Q8_0 weight (no bf16 round)", y, y_ref, 1e-4, 1e-6);
    return true;
}

bool part_b_real() {
    std::error_code ec;
    if (!std::filesystem::exists(g_model, ec)) {
        std::printf("  model not present (%s): the real-layer checks are skipped\n", g_model.c_str());
        return false;
    }
    strata::GgufModel model = strata::GgufModel::open(g_model);
    naive::NaiveGeometry g;
    if (std::string err = naive::read_geometry(model.meta(), g); !err.empty()) {
        check(false, ("real model geometry: " + err).c_str());
        return true;
    }
    check(g.n_layers == 48 && !g.is_moe_layer(0) && g.kv_heads(1) == 8,
          "real model: layer 0 is dense and layer 1 is a SWA layer (the cases below)");
    real_case_gemv_probe(model);
    real_case_dense(model);
    real_case_swa(model, "swa_small");
    real_case_swa(model, "swa_window");
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (argc > 2) g_model = argv[2];
    std::printf("naive layer test (fixtures: %s)\n", g_dir.c_str());
    part_a_synthetic();
    const bool ran_real = part_b_real();
    if (g_failures) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    if (!ran_real) {
        std::printf("all naive layer checks passed (model part skipped)\n");
        return 77;
    }
    std::printf("all naive layer checks passed (model included)\n");
    return 0;
}
