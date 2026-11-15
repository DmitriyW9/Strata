// tests/naive/test_naive_moe_dsa.cpp - the IQ2 MoE FFN, the DSA (indexer) layer and the two-layer
// trunk, run through the port (loader.hpp + kernels.hpp, ggml's compiled `to_float` for every
// quantised weight), against fixtures that tools/naive_real_moe_dsa_fixture.py computes from the
// model's real weights with independent numpy decoders.  This covers steps 4-5 (and the trunk half
// of 7) of docs/NAIVE.md; test_naive_layer.cpp covers its steps 1-3.
//
//   test_naive_moe_dsa [fixtures-dir] [model-any-shard.gguf] [case]
//     fixtures default: tests/naive/fixtures  (regenerate: python tools/naive_real_moe_dsa_fixture.py)
//     model default:    /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/
//                       Naive-N0.5-Flash-MQ87-00001-of-00004.gguf
//     case:             moe | dsa | trunk | all (default all)
//     Model absent -> skip line + exit 77 (CTest's skip code).
//
// What each case checks:
//   * moe    layer 1's MoE FFN for one token: the training-path rmsnorm (kernels.hpp
//            rmsnorm_training - normalized first to bf16, then gamma, training_moe.py), the router
//            over the REAL F32 gate/bias (exact ids; the (p+bias)-selects / unbiased-p-weights
//            split of training_moe.py:route), one expert's SwiGLU over IQ2_XXS gate/up and IQ2_XS
//            down slices read by one-expert window (loader.hpp gemv_dequant_3d - the element base
//            e*n_in*n_out and the per-row decode the fixture's byte windows use), and the bf16
//            combine with that expert's router weight (training_moe.py:combine, one expert).
//   * dsa    layer 0 for one query row over P cache rows: the k/v/indexer-key projections with
//            their rounding boundaries (run_attn_dsa.py:reference_caches - bf16 after each
//            projection, the value_scale folded into the bf16 round, the indexer key's LayerNorm +
//            rope + bf16 before the E4M3 row quant), the E4M3 row scales amax/448
//            (fp8_rows_dequant), the index-query rope + E4M3, the score weight w with the
//            1/sqrt(16) fold, the selection logits over the fp8-dequantised rows
//            (dsa_index_logits), the stable top-10 pick (exact ints - where the fp8 stage is
//            actually exercised), and the grouped softmax attention over the visible rows with no
//            sink.
//   * trunk  a two-token sequence end to end: the embedding row read as one column of the 2.4 GB
//            Q8_0 table (load_tensor_cols - compared exact), layer 0 (the dsa chain over two rows
//            + the dense SwiGLU FFN, residuals bf16(x+d)), layer 1 (the SWA chain with sinks and
//            theta 1e4 + the MoE FFN over ALL 8 routed experts), the output norm, and the logits
//            against the output head - compared by the top-5 ids (exact ints) and their values.
//
// INDEPENDENCE: the fixtures decode IQ2_XXS/IQ2_XS with the vectorised numpy transcriptions of
// tools/naive_iq2.py (self-tested bit-exactly against a literal transcription of ggml's decode
// loops) and Q8_0/Q6_K with the layout decoders of tools/naive_real_layer_fixture.py; this file
// decodes the same bytes through ggml's compiled `to_float` and runs kernels.hpp.  A wrong expert
// window, a swapped grid half, or a mis-folded scale nibble moves values by O(100%), not 0.8%.
//
// Tolerances follow the layer test: a bf16-rounded array may sit one bf16 ulp (0.4%) apart when
// the fp32-vs-fp64 pre-round dot lands on the other side of a rounding boundary - reltol 0.008
// accepts that and rejects anything larger.  Router weights and fp8-dequantised rows are fp32 all
// the way (both sides accumulate in double), so their checks are tight; the embedding row and the
// top-5 ids are exact.  A bf16-rounded residual add is one bf16 round wide: a wrong boundary moves
// values, it does not shift them by 0.4%.
//
// The GQA convention (query head h reads kv head h / (n_head / n_kv), contiguous groups -
// prefill.py:swa_attention's q[T,H,D]-over-k[S,kv,D] view) and the indexer weight dtype (this GGUF
// ships BF16; the reference's fp8 packing belongs to its checkpoint) are the same read-not-measured
// conventions test_naive_layer.cpp documents; the deviation is recorded in docs/NAIVE.md.

#include "fixture_io.hpp"
#include "strata/naive/kernels.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/reader.hpp"
// trunk.hpp's block-FP8 weight round-trip is the reference's packing for the DSA index-query
// weight.  This GGUF ships that weight as BF16 (see docs/NAIVE.md), so the fixtures and the
// checks below read bf16 values and the round-trip stays unused here; the header is included so
// the build compiles it and an fp8-packaged checkpoint finds it ready.
#include "strata/naive/trunk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
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
const double kFp32Reltol = 1e-4;    // fp32-stage arrays: both sides sum in double, only order may differ
const double kFp32AbsTol = 1e-6;
// The E4M3 stage and everything read through it (the ki cache rows, the logits summed over them):
// one bf16 ulp of the fp32-vs-fp64 pre-round dot can flip one e4m3 code by one tick, and that tick
// is 1/16 of the row amax - up to ~6% of a small element, ~1% of a row's logit.  0.14 accepts one
// tick and rejects two; the exact-int checks (selection ids) sit on top of these same values.
const double kFp8Reltol = 0.14;

// The model's geometry, asserted against the shard's metadata below; if the file ever disagrees,
// the fixtures generated from it would describe a different model and every check would mean
// nothing, so the test refuses before it computes.
constexpr int64_t N_EMBD = 4096, N_HEAD = 64, K_DIM = 192, V_DIM = 128, N_ROT = 64;
constexpr int64_t N_KV_SWA = 8, N_KV_DSA = 4, WINDOW = 128, DENSE_FF = 16384;
constexpr int64_t INDEX_HEADS = 16, INDEX_DIM = 128, N_EXPERT = 256, TOPK = 8, EXPERT_FF = 2048;
constexpr double VALUE_SCALE = 0.707, THETA_SWA = 1e4, THETA_DSA = 1e7, RMS_EPS = 1e-5;
constexpr int64_t VOCAB = 152576;

void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}
void check_floats(const std::string& c, const char* what, const std::vector<float>& got, const Arr& ref,
                  double abstol = kAbsTol, double reltol = kBf16Reltol) {
    naive_test::check_floats(c, what, got, ref, abstol, reltol, g_failures);
}
void check_ints(const std::string& c, const char* what, const std::vector<int>& got, const Arr& ref) {
    naive_test::check_ints(c, what, got, ref, g_failures);
}
void bf16_round_inplace(std::vector<float>& v) {
    for (float& f : v) f = naive::bf16_round(f);
}

/// bf16(residual + delta): the references add in fp32/fp64 and round once, after the projection.
std::vector<float> add_residual(const std::vector<float>& residual, const std::vector<float>& delta) {
    std::vector<float> out(residual.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = naive::bf16_round((double) residual[i] + (double) delta[i]);
    return out;
}

/// A projection through the port: gemv_dequant (row-by-row ggml decode, fp32 accumulate in double)
/// rounded to bf16, exactly where run_attn_swa.py:project and run_attn_dsa.py:project round.
std::vector<float> project(const strata::GgufModel& m, const char* tensor, const float* x,
                           int64_t n_in, int64_t n_out) {
    std::vector<float> y((size_t) n_out);
    if (const std::string err = naive::gemv_dequant(m, tensor, x, n_in, y.data(), n_out); !err.empty()) {
        std::printf("  FAIL %s: %s\n", tensor, err.c_str());
        ++g_failures;
    }
    bf16_round_inplace(y);
    return y;
}

/// The MoE FFN's rmsnorm for one row: the training path (training_moe.py:rmsnorm).
void rmsnorm_row(const naive::LoadedTensor& gamma, const float* x, float* out) {
    naive::rmsnorm_training(x, gamma.data.data(), (int) N_EMBD, (float) RMS_EPS, out);
}

// ------------------------------------------------------------------ the MoE FFN
/// Router over the real F32 gate + bias (training_moe.py:route via kernels.hpp moe_route):
/// idx ascending (stable), the unbiased p renormalised.  moe_route sums in double, fp32 result.
void route(const naive::LoadedTensor& gate_w, const naive::LoadedTensor& bias, const float* xn,
           std::vector<int>& idx, std::vector<float>& wts) {
    idx.assign((size_t) TOPK, 0);
    wts.assign((size_t) TOPK, 0.f);
    naive::moe_route(xn, gate_w.data.data(), bias.data.data(), (int) N_EMBD, (int) N_EXPERT,
                     (int) TOPK, idx.data(), wts.data());
}

/// One routed expert's SwiGLU through gemv_dequant_3d (expert-window byte base + row decode):
/// gate/up bf16, act = bf16(silu(g)*u), down bf16 [4096].
bool expert_ffn(const strata::GgufModel& m, int64_t expert, const float* xn, std::vector<float>& out) {
    std::vector<float> gate((size_t) EXPERT_FF), up((size_t) EXPERT_FF), act((size_t) EXPERT_FF);
    out.assign((size_t) N_EMBD, 0.f);
    if (const std::string err = naive::gemv_dequant_3d(m, "blk.1.ffn_gate_exps.weight", (uint64_t) expert,
                                                       xn, N_EMBD, gate.data(), EXPERT_FF); !err.empty()) {
        std::printf("  FAIL blk.1.ffn_gate_exps[%lld]: %s\n", (long long) expert, err.c_str());
        ++g_failures;
        return false;
    }
    bf16_round_inplace(gate);
    if (const std::string err = naive::gemv_dequant_3d(m, "blk.1.ffn_up_exps.weight", (uint64_t) expert,
                                                       xn, N_EMBD, up.data(), EXPERT_FF); !err.empty()) {
        std::printf("  FAIL blk.1.ffn_up_exps[%lld]: %s\n", (long long) expert, err.c_str());
        ++g_failures;
        return false;
    }
    bf16_round_inplace(up);
    for (int64_t i = 0; i < EXPERT_FF; ++i) act[i] = naive::bf16_round(naive::silu(gate[i]) * up[i]);
    if (const std::string err = naive::gemv_dequant_3d(m, "blk.1.ffn_down_exps.weight", (uint64_t) expert,
                                                       act.data(), EXPERT_FF, out.data(), N_EMBD); !err.empty()) {
        std::printf("  FAIL blk.1.ffn_down_exps[%lld]: %s\n", (long long) expert, err.c_str());
        ++g_failures;
        return false;
    }
    bf16_round_inplace(out);
    return true;
}

/// The trunk's MoE FFN: all 8 routed experts, combine bf16(sum w*e) (training_moe.py:combine).
/// Experts are decoded one window at a time - the stack is 12 GiB and is never materialised.
void moe_ffn(const strata::GgufModel& m, const float* xn2, const std::vector<int>& idx,
             const std::vector<float>& wts, std::vector<float>& moe_out) {
    std::vector<float> expert_out((size_t) TOPK * N_EMBD);
    for (int t = 0; t < TOPK; ++t) {
        std::vector<float> e;
        if (!expert_ffn(m, idx[t], xn2, e)) return;
        std::memcpy(&expert_out[(size_t) t * N_EMBD], e.data(), N_EMBD * sizeof(float));
    }
    std::vector<float> raw((size_t) N_EMBD);
    naive::moe_combine(expert_out.data(), wts.data(), (int) TOPK, (int) N_EMBD, raw.data());
    moe_out.assign((size_t) N_EMBD, 0.f);
    for (int64_t i = 0; i < N_EMBD; ++i) moe_out[i] = naive::bf16_round(raw[i]);
}

// ------------------------------------------------------------------ the DSA layer's chain
/// Layer 0's per-row caches, selection and attention over P rows of normed inputs (`xns`,
/// P*N_EMBD), mirroring dsa_caches in the fixture generator row by row.  Writes nothing; the
/// caller compares what the fixtures carry.
struct DsaOut {
    std::vector<float> xn_query;    // the last row's normed input (echoed for the checks)
    std::vector<float> k_cache;     // P * (N_KV_DSA*K_DIM), bf16-rounded values
    std::vector<float> v_cache;     // P * (N_KV_DSA*V_DIM)
    std::vector<float> ki_deq;      // P * (N_KV_DSA*INDEX_DIM) fp8-dequantised + ki_scales
    std::vector<float> ki_scales;   // P
    std::vector<float> q_rope;      // (the last row) N_HEAD*K_DIM
    std::vector<float> iq_deq;      // INDEX_HEADS*INDEX_DIM
    std::vector<float> w;           // INDEX_HEADS
    std::vector<float> logits;      // P
    std::vector<int> sel_small;     // stable top-10 of the logits
    std::vector<float> attn_out;    // N_HEAD*V_DIM, bf16-rounded
};

DsaOut dsa_chain(const strata::GgufModel& m, int64_t P, const std::vector<float>& xns) {
    DsaOut o;
    naive::LoadedTensor ln_w, ln_b;
    if (!naive::load_tensor_f32(m, "blk.0.indexer.k_norm.weight", ln_w).empty() ||
        !naive::load_tensor_f32(m, "blk.0.indexer.k_norm.bias", ln_b).empty()) {
        check(false, "the indexer key LayerNorm weights load");
        return o;
    }
    o.k_cache.assign((size_t) P * N_KV_DSA * K_DIM, 0.f);
    o.v_cache.assign((size_t) P * N_KV_DSA * V_DIM, 0.f);
    std::vector<float> ik_bf16((size_t) P * INDEX_DIM);
    std::vector<float> row((size_t) INDEX_DIM);
    for (int64_t p = 0; p < P; ++p) {
        const float* xn = xns.data() + (size_t) p * N_EMBD;
        // k: bf16 after the projection, rope(theta 1e7) per 4 kv heads of 192, rounded bf16 again
        // before the cache (reference_caches: k = project(...).bf16(); k[:, :ROPE] = hf_rope(...);
        // the cache row is stored bf16 - the same double round the fixture applies).
        std::vector<float> k = project(m, "blk.0.attn_k.weight", xn, N_EMBD, N_KV_DSA * K_DIM);
        naive::rope_rows(k.data(), (int) N_KV_DSA, (int) K_DIM, (int) N_ROT, p, THETA_DSA,
                         o.k_cache.data() + (size_t) p * N_KV_DSA * K_DIM);
        for (float* f = o.k_cache.data() + (size_t) p * N_KV_DSA * K_DIM;
             f != o.k_cache.data() + (size_t) (p + 1) * N_KV_DSA * K_DIM; ++f)
            *f = naive::bf16_round(*f);
        // v: projection * value_scale, rounded once - the DSA reference folds scale before the round.
        std::vector<float> v = project(m, "blk.0.attn_v.weight", xn, N_EMBD, N_KV_DSA * V_DIM);
        // project() rounds before the fold (the fixture's fold-after-round differs by <=1 ulp):
        // the DSA fold needs the pre-round value, so redo it pre-round here.  project() already
        // reported any load error; the values were left rounded, so recompute without the round.
        std::vector<float> v_raw((size_t) N_KV_DSA * V_DIM);
        if (const std::string err = naive::gemv_dequant(m, "blk.0.attn_v.weight", xn, N_EMBD,
                                                        v_raw.data(), N_KV_DSA * V_DIM); !err.empty()) {
            std::printf("  FAIL blk.0.attn_v: %s\n", err.c_str());
            ++g_failures;
            return o;
        }
        for (float& f : v_raw) f = naive::bf16_round(f * VALUE_SCALE);
        std::memcpy(o.v_cache.data() + (size_t) p * N_KV_DSA * V_DIM, v_raw.data(),
                    v_raw.size() * sizeof(float));
        // indexer keys: bf16 -> LayerNorm -> bf16 -> rope -> bf16 (reference_caches' four rounds).
        std::vector<float> ik = project(m, "blk.0.indexer.k_proj.weight", xn, N_EMBD, INDEX_DIM);
        naive::index_key_layernorm(ik.data(), ln_w.data.data(), ln_b.data.data(), (int) INDEX_DIM,
                                   (float) RMS_EPS, row.data());
        for (float& f : row) f = naive::bf16_round(f);
        std::vector<float> roped((size_t) INDEX_DIM);
        naive::rope_rows(row.data(), 1, (int) INDEX_DIM, (int) N_ROT, p, THETA_DSA, roped.data());
        for (float& f : roped) f = naive::bf16_round(f);
        std::memcpy(ik_bf16.data() + (size_t) p * INDEX_DIM, roped.data(), INDEX_DIM * sizeof(float));
    }
    // The selection cache: E4M3 rows (amax/448, min 1e-30), decoded back for the logits.
    o.ki_deq.assign((size_t) P * INDEX_DIM, 0.f);
    o.ki_scales.assign((size_t) P, 0.f);
    naive::fp8_rows_dequant(ik_bf16.data(), (int) P, (int) INDEX_DIM, o.ki_deq.data(),
                            o.ki_scales.data());

    const int64_t T = P - 1;
    const float* xn = xns.data() + (size_t) T * N_EMBD;
    o.xn_query.assign(xn, xn + N_EMBD);
    std::vector<float> q = project(m, "blk.0.attn_q.weight", xn, N_EMBD, N_HEAD * K_DIM);
    o.q_rope.assign((size_t) N_HEAD * K_DIM, 0.f);
    naive::rope_rows(q.data(), (int) N_HEAD, (int) K_DIM, (int) N_ROT, T, THETA_DSA, o.q_rope.data());
    // reference_query stores the rope'd query bf16 (k_proj's round-trip); the fixture keeps
    // q_rope in that form, so the rounded value is what attention reads.
    bf16_round_inplace(o.q_rope);
    std::vector<float> iq = project(m, "blk.0.indexer.q_proj.weight", xn, N_EMBD, INDEX_HEADS * INDEX_DIM);
    std::vector<float> iq_rope((size_t) INDEX_HEADS * INDEX_DIM);
    naive::rope_rows(iq.data(), (int) INDEX_HEADS, (int) INDEX_DIM, (int) N_ROT, T, THETA_DSA,
                     iq_rope.data());
    for (float& f : iq_rope) f = naive::bf16_round(f);
    std::vector<float> iq_scales((size_t) INDEX_HEADS);
    o.iq_deq.assign((size_t) INDEX_HEADS * INDEX_DIM, 0.f);
    naive::fp8_rows_dequant(iq_rope.data(), (int) INDEX_HEADS, (int) INDEX_DIM, o.iq_deq.data(),
                            iq_scales.data());
    o.w.assign((size_t) INDEX_HEADS, 0.f);
    std::vector<float> w_raw((size_t) INDEX_HEADS);
    if (const std::string err = naive::gemv_dequant(m, "blk.0.indexer.proj.weight", xn, N_EMBD,
                                                    w_raw.data(), INDEX_HEADS); !err.empty()) {
        std::printf("  FAIL blk.0.indexer.proj: %s\n", err.c_str());
        ++g_failures;
        return o;
    }
    const double fold = 1.0 / std::sqrt((double) INDEX_HEADS);
    for (int64_t h = 0; h < INDEX_HEADS; ++h) o.w[h] = (float) ((double) w_raw[h] * fold);

    // Selection logits over the fp8-dequantised rows (dsa_index_logits); visible = all P rows.
    o.logits.assign((size_t) P, 0.f);
    naive::dsa_index_logits(o.iq_deq.data(), (int) INDEX_HEADS, (int) INDEX_DIM, o.w.data(),
                            o.ki_deq.data(), (int) P, o.logits.data());
    // The fixture's case uses top-k 10 (P < index_top_k=2048 -> attention sees every row; the
    // arange rule itself is asserted from the metadata, not from data).
    o.sel_small.assign(10, 0);
    naive::stable_topk(o.logits.data(), (int) P, 10, o.sel_small.data());

    // Grouped softmax attention over the visible rows, 64 q / 4 kv, no sink (run_attn_dsa
    // reference_attention; swa_attention with sink=nullptr is that formula).
    o.attn_out.assign((size_t) N_HEAD * V_DIM, 0.f);
    const int64_t group = N_HEAD / N_KV_DSA;
    std::vector<float> qg((size_t) group * K_DIM), kg((size_t) P * K_DIM), vg((size_t) P * V_DIM);
    for (int64_t g = 0; g < N_KV_DSA; ++g) {
        std::memcpy(qg.data(), o.q_rope.data() + (size_t) g * group * K_DIM, qg.size() * sizeof(float));
        for (int64_t j = 0; j < P; ++j) {
            std::memcpy(kg.data() + (size_t) j * K_DIM,
                        o.k_cache.data() + (size_t) j * N_KV_DSA * K_DIM + g * K_DIM, K_DIM * sizeof(float));
            std::memcpy(vg.data() + (size_t) j * V_DIM,
                        o.v_cache.data() + (size_t) j * N_KV_DSA * V_DIM + g * V_DIM, V_DIM * sizeof(float));
        }
        naive::swa_attention(qg.data(), (int) group, (int) K_DIM, kg.data(), vg.data(), (int) V_DIM,
                             nullptr, 0, (int) P - 1, o.attn_out.data() + (size_t) g * group * V_DIM);
    }
    bf16_round_inplace(o.attn_out);
    return o;
}

// ------------------------------------------------------------------ the cases
/// moe_layer1: the fixture's x -> xn, the real router, expert idx[0]'s IQ2 SwiGLU, combine by wts[0].
void case_moe(const strata::GgufModel& model) {
    const std::string c = "moe_layer1";
    Arr x_ref;
    if (!naive_test::load(g_dir, c, "x", x_ref, g_failures)) return;
    naive::LoadedTensor gamma, gate_w, bias;
    if (!naive::load_tensor_f32(model, "blk.1.ffn_norm.weight", gamma).empty() ||
        !naive::load_tensor_f32(model, "blk.1.ffn_gate_inp.weight", gate_w).empty() ||
        !naive::load_tensor_f32(model, "blk.1.exp_probs_b.bias", bias).empty()) {
        check(false, "moe_layer1: the norm / router gate / routing bias load");
        return;
    }
    std::vector<float> xn((size_t) N_EMBD);
    rmsnorm_row(gamma, x_ref.fp(), xn.data());

    std::vector<int> idx;
    std::vector<float> wts;
    route(gate_w, bias, xn.data(), idx, wts);
    Arr idr, wr, eid;
    if (!naive_test::load(g_dir, c, "route_idx", idr, g_failures) ||
        !naive_test::load(g_dir, c, "route_wts", wr, g_failures) ||
        !naive_test::load(g_dir, c, "expert_id", eid, g_failures)) return;
    Arr xn_ref;
    if (naive_test::load(g_dir, c, "xn", xn_ref, g_failures))
        check_floats(c, "rmsnorm (training rounding) over the real F32 gamma", xn, xn_ref);
    check_ints(c, "router: the 8 expert ids (p+bias selects, stable ascending)", idx, idr);
    check_floats(c, "router: the unbiased p, renormalised by (sum + 1e-20)", wts, wr,
                 kFp32AbsTol, kFp32Reltol);

    const int64_t e0 = idx[0];
    check(e0 == eid.i[0], "router: the fixture's first expert id matches");
    std::vector<float> expert;
    if (expert_ffn(model, e0, xn.data(), expert)) {
        Arr g_r, u_r, a_r, o_r;
        if (!naive_test::load(g_dir, c, "expert_gate", g_r, g_failures) ||
            !naive_test::load(g_dir, c, "expert_up", u_r, g_failures) ||
            !naive_test::load(g_dir, c, "expert_act", a_r, g_failures) ||
            !naive_test::load(g_dir, c, "expert_out", o_r, g_failures)) return;
        // expert_ffn returns the down output; re-run gate/up for their own comparisons (each
        // expert window decode is ~0.1 s; the whole case stays far under the timeout).
        std::vector<float> gate((size_t) EXPERT_FF), up((size_t) EXPERT_FF), act((size_t) EXPERT_FF);
        if (naive::gemv_dequant_3d(model, "blk.1.ffn_gate_exps.weight", (uint64_t) e0, xn.data(),
                                   N_EMBD, gate.data(), EXPERT_FF).empty()) {
            bf16_round_inplace(gate);
            check_floats(c, "IQ2_XXS gate GEMV over one expert window", gate, g_r);
        }
        if (naive::gemv_dequant_3d(model, "blk.1.ffn_up_exps.weight", (uint64_t) e0, xn.data(),
                                   N_EMBD, up.data(), EXPERT_FF).empty()) {
            bf16_round_inplace(up);
            check_floats(c, "IQ2_XXS up GEMV over one expert window", up, u_r);
            for (int64_t i = 0; i < EXPERT_FF; ++i)
                act[i] = naive::bf16_round(naive::silu(gate[i]) * up[i]);
            check_floats(c, "act = bf16(silu(gate) * up)", act, a_r);
        }
        check_floats(c, "IQ2_XS down GEMV - the completed expert output", expert, o_r);
        std::vector<float> moe_out((size_t) N_EMBD);
        const std::vector<float> w0{wts[0]};
        naive::moe_combine(expert.data(), w0.data(), 1, (int) N_EMBD, moe_out.data());
        for (float& f : moe_out) f = naive::bf16_round(f);
        Arr mo_r;
        if (naive_test::load(g_dir, c, "moe_out", mo_r, g_failures))
            check_floats(c, "combine (one expert, bf16) - the MoE layer's out", moe_out, mo_r);
    }
}

/// dsa_layer0: the fixture's P seeded rows -> xn per row -> the chain -> every intermediate.
void case_dsa(const strata::GgufModel& model) {
    const std::string c = "dsa_layer0";
    Arr x_rows;
    if (!naive_test::load(g_dir, c, "x_rows", x_rows, g_failures)) return;
    naive::LoadedTensor gamma;
    if (!naive::load_tensor_f32(model, "blk.0.attn_norm.weight", gamma).empty()) {
        check(false, "dsa_layer0: the attention norm loads");
        return;
    }
    const int64_t P = x_rows.rows;
    std::vector<float> xns((size_t) P * N_EMBD);
    for (int64_t p = 0; p < P; ++p)
        rmsnorm_row(gamma, x_rows.fp() + (size_t) p * N_EMBD, xns.data() + (size_t) p * N_EMBD);

    const DsaOut got = dsa_chain(model, P, xns);
    Arr xn_ref, k_ref, v_ref, ki_ref, kis_ref, q_ref, iq_ref, iqs_ref, w_ref, lg_ref, sel_ref, a_ref;
    if (!naive_test::load(g_dir, c, "xn", xn_ref, g_failures) ||
        !naive_test::load(g_dir, c, "k_cache", k_ref, g_failures) ||
        !naive_test::load(g_dir, c, "v_cache", v_ref, g_failures) ||
        !naive_test::load(g_dir, c, "ki_deq", ki_ref, g_failures) ||
        !naive_test::load(g_dir, c, "ki_scales", kis_ref, g_failures) ||
        !naive_test::load(g_dir, c, "q_rope", q_ref, g_failures) ||
        !naive_test::load(g_dir, c, "iq_deq", iq_ref, g_failures) ||
        !naive_test::load(g_dir, c, "iq_scales", iqs_ref, g_failures) ||
        !naive_test::load(g_dir, c, "w", w_ref, g_failures) ||
        !naive_test::load(g_dir, c, "logits", lg_ref, g_failures) ||
        !naive_test::load(g_dir, c, "sel_small", sel_ref, g_failures) ||
        !naive_test::load(g_dir, c, "attn_out", a_ref, g_failures)) return;

    check_floats(c, "xn (query row): rmsnorm training rounding", got.xn_query, xn_ref);
    check_floats(c, "k cache: BF16 GEMV + split-half rope(theta 1e7)", got.k_cache, k_ref);
    check_floats(c, "v cache: value_scale folded into the bf16 round", got.v_cache, v_ref);
    check_floats(c, "indexer-key cache: LayerNorm + rope, E4M3-dequantised", got.ki_deq, ki_ref,
                 kAbsTol, kFp8Reltol);
    check_floats(c, "E4M3 row scales: amax/448 (min 1e-30)", got.ki_scales, kis_ref,
                 kFp32AbsTol, kFp32Reltol);
    check_floats(c, "q after split-half RoPE (Q6_K, theta 1e7)", got.q_rope, q_ref);
    check_floats(c, "index-query rows: BF16 GEMV + rope, E4M3-dequantised", got.iq_deq, iq_ref,
                 kAbsTol, kFp8Reltol);
    check_floats(c, "index score weights w with the 1/sqrt(16) fold", got.w, w_ref,
                 kFp32AbsTol, kFp32Reltol);
    check_floats(c, "selection logits: sum relu(q.k)*w over fp8 rows", got.logits, lg_ref,
                 kAbsTol, kFp8Reltol);
    check_ints(c, "stable top-10 selection (exact - the fp8 stage pins it)", got.sel_small, sel_ref);
    check_floats(c, "grouped attention over all visible rows, no sink", got.attn_out, a_ref);
}

/// trunk: two rows -> layer 0 (DSA + dense) -> layer 1 (SWA + MoE) -> head -> top-5.
void case_trunk(const strata::GgufModel& model) {
    const std::string c = "trunk";
    Arr row0_ref, emb_ref;
    if (!naive_test::load(g_dir, c, "row0", row0_ref, g_failures) ||
        !naive_test::load(g_dir, c, "emb", emb_ref, g_failures)) return;

    // The embedding row, read as one column of the 2.4 GB Q8_0 table.  Compared exact: the int8 x
    // fp16 product fits float32's mantissa, so both sides hold the same number.
    naive::LoadedTensor emb;
    if (const std::string err = naive::load_tensor_cols(model, "token_embd.weight", 5000, 1, emb); !err.empty()) {
        std::printf("  FAIL token_embd column: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    {
        bool exact = emb.elements() == (uint64_t) N_EMBD;
        for (int64_t i = 0; exact && i < N_EMBD; ++i) exact = emb.data[i] == emb_ref.f[i];
        check(exact, "one vocab row (token 5000) out of the Q8_0 emb table, exact");
    }

    // ---- layer 0: DSA attention (query at pos 1 over the two rows) + dense FFN.
    naive::LoadedTensor g0;
    if (!naive::load_tensor_f32(model, "blk.0.attn_norm.weight", g0).empty()) {
        check(false, "trunk: layer 0 attention norm loads");
        return;
    }
    std::vector<float> rows((size_t) 2 * N_EMBD);
    std::memcpy(rows.data(), row0_ref.fp(), N_EMBD * sizeof(float));
    std::memcpy(rows.data() + N_EMBD, emb_ref.fp(), N_EMBD * sizeof(float));
    std::vector<float> xns((size_t) 2 * N_EMBD);
    rmsnorm_row(g0, rows.data(), xns.data());
    rmsnorm_row(g0, rows.data() + N_EMBD, xns.data() + N_EMBD);
    const DsaOut dsa = dsa_chain(model, 2, xns);
    Arr xn0_ref, a0_ref;
    if (!naive_test::load(g_dir, c, "xn0", xn0_ref, g_failures) ||
        !naive_test::load(g_dir, c, "attn0_out", a0_ref, g_failures)) return;
    check_floats(c, "trunk L0: xn (query row) training rounding", dsa.xn_query, xn0_ref);
    check_floats(c, "trunk L0: DSA attention out (2 rows, grouped 64/4)", dsa.attn_out, a0_ref);

    std::vector<float> attn_proj = project(model, "blk.0.attn_output.weight", dsa.attn_out.data(),
                                           N_HEAD * V_DIM, N_EMBD);
    const std::vector<float> q_row(rows.begin() + N_EMBD, rows.end());
    std::vector<float> h1 = add_residual(q_row, attn_proj);
    Arr h1_ref;
    if (naive_test::load(g_dir, c, "h1", h1_ref, g_failures))
        check_floats(c, "h1 = bf16(x + attn_out) - the first residual", h1, h1_ref);

    naive::LoadedTensor gf;
    if (!naive::load_tensor_f32(model, "blk.0.ffn_norm.weight", gf).empty()) {
        check(false, "trunk: layer 0 FFN norm loads");
        return;
    }
    std::vector<float> xn1((size_t) N_EMBD);
    rmsnorm_row(gf, h1.data(), xn1.data());
    std::vector<float> gate = project(model, "blk.0.ffn_gate.weight", xn1.data(), N_EMBD, DENSE_FF);
    std::vector<float> up = project(model, "blk.0.ffn_up.weight", xn1.data(), N_EMBD, DENSE_FF);
    std::vector<float> act((size_t) DENSE_FF);
    for (int64_t i = 0; i < DENSE_FF; ++i) act[i] = naive::bf16_round(naive::silu(gate[i]) * up[i]);
    std::vector<float> ffn0 = project(model, "blk.0.ffn_down.weight", act.data(), DENSE_FF, N_EMBD);
    std::vector<float> h2 = add_residual(h1, ffn0);
    Arr f0_ref, h2_ref;
    if (!naive_test::load(g_dir, c, "ffn0_out", f0_ref, g_failures) ||
        !naive_test::load(g_dir, c, "h2", h2_ref, g_failures)) return;
    check_floats(c, "trunk L0: dense SwiGLU FFN (16384) over Q8_0", ffn0, f0_ref);
    check_floats(c, "h2 = bf16(h1 + ffn_out) - the second residual", h2, h2_ref);

    // ---- layer 1: SWA attention over the two rows (sinks, theta 1e4) + the IQ2 MoE FFN.
    naive::LoadedTensor g1, sink;
    if (!naive::load_tensor_f32(model, "blk.1.attn_norm.weight", g1).empty() ||
        !naive::load_tensor_f32(model, "blk.1.attn_sinks.weight", sink).empty()) {
        check(false, "trunk: layer 1 norm / sinks load");
        return;
    }
    std::vector<float> xn_swa((size_t) 2 * N_EMBD);
    rmsnorm_row(g1, h1.data(), xn_swa.data());
    rmsnorm_row(g1, h2.data(), xn_swa.data() + N_EMBD);
    std::vector<float> k1 = project(model, "blk.1.attn_k.weight", xn_swa.data(), N_EMBD, N_KV_SWA * K_DIM);
    std::vector<float> k1_rope((size_t) 2 * N_KV_SWA * K_DIM);
    naive::rope_rows(k1.data(), (int) N_KV_SWA, (int) K_DIM, (int) N_ROT, 0, THETA_SWA, k1_rope.data());
    std::vector<float> k1r = project(model, "blk.1.attn_k.weight", xn_swa.data() + N_EMBD, N_EMBD,
                                     N_KV_SWA * K_DIM);
    naive::rope_rows(k1r.data(), (int) N_KV_SWA, (int) K_DIM, (int) N_ROT, 1, THETA_SWA,
                     k1_rope.data() + (size_t) N_KV_SWA * K_DIM);
    for (float& f : k1_rope) f = naive::bf16_round(f);   // the fixture stores the rope'd k bf16
    // v rows with the scale folded (pre-round value, one bf16 round after the fold).
    std::vector<float> v1((size_t) 2 * N_KV_SWA * V_DIM);
    for (int64_t p = 0; p < 2; ++p) {
        std::vector<float> vraw((size_t) N_KV_SWA * V_DIM);
        if (const std::string err = naive::gemv_dequant(model, "blk.1.attn_v.weight",
                                                        xn_swa.data() + (size_t) p * N_EMBD, N_EMBD,
                                                        vraw.data(), N_KV_SWA * V_DIM); !err.empty()) {
            std::printf("  FAIL blk.1.attn_v: %s\n", err.c_str());
            ++g_failures;
            return;
        }
        for (size_t i = 0; i < vraw.size(); ++i)
            v1[(size_t) p * N_KV_SWA * V_DIM + i] = naive::bf16_round(vraw[i] * VALUE_SCALE);
    }
    std::vector<float> q1 = project(model, "blk.1.attn_q.weight", xn_swa.data() + N_EMBD, N_EMBD,
                                    N_HEAD * K_DIM);
    std::vector<float> q1_rope((size_t) N_HEAD * K_DIM);
    naive::rope_rows(q1.data(), (int) N_HEAD, (int) K_DIM, (int) N_ROT, 1, THETA_SWA, q1_rope.data());
    bf16_round_inplace(q1_rope);   // the fixture stores the rope'd query bf16, as the reference does
    std::vector<float> attn1_heads((size_t) N_HEAD * V_DIM);
    const int64_t group = N_HEAD / N_KV_SWA;
    std::vector<float> qg((size_t) group * K_DIM), kg(2 * K_DIM), vg(2 * V_DIM);
    for (int64_t g = 0; g < N_KV_SWA; ++g) {
        std::memcpy(qg.data(), q1_rope.data() + (size_t) g * group * K_DIM, qg.size() * sizeof(float));
        for (int64_t j = 0; j < 2; ++j) {   // window 128 >= 2: both rows visible (first = 0)
            std::memcpy(kg.data() + (size_t) j * K_DIM,
                        k1_rope.data() + (size_t) j * N_KV_SWA * K_DIM + g * K_DIM, K_DIM * sizeof(float));
            std::memcpy(vg.data() + (size_t) j * V_DIM,
                        v1.data() + (size_t) j * N_KV_SWA * V_DIM + g * V_DIM, V_DIM * sizeof(float));
        }
        naive::swa_attention(qg.data(), (int) group, (int) K_DIM, kg.data(), vg.data(), (int) V_DIM,
                             sink.data.data() + g * group, 0, 1,
                             attn1_heads.data() + (size_t) g * group * V_DIM);
    }
    std::vector<float> attn1_flat = attn1_heads;
    bf16_round_inplace(attn1_flat);
    Arr a1_ref;
    if (naive_test::load(g_dir, c, "attn1_out", a1_ref, g_failures))
        check_floats(c, "trunk L1: SWA attention with sinks (window 128, 2 rows)", attn1_flat, a1_ref);
    std::vector<float> attn1 = project(model, "blk.1.attn_output.weight", attn1_flat.data(),
                                       N_HEAD * V_DIM, N_EMBD);
    std::vector<float> h3 = add_residual(h2, attn1);
    Arr h3_ref;
    if (naive_test::load(g_dir, c, "h3", h3_ref, g_failures))
        check_floats(c, "h3 = bf16(h2 + attn1_out) - the third residual", h3, h3_ref);

    naive::LoadedTensor gm, gw, gb;
    if (!naive::load_tensor_f32(model, "blk.1.ffn_norm.weight", gm).empty() ||
        !naive::load_tensor_f32(model, "blk.1.ffn_gate_inp.weight", gw).empty() ||
        !naive::load_tensor_f32(model, "blk.1.exp_probs_b.bias", gb).empty()) {
        check(false, "trunk: MoE norm / gate / bias load");
        return;
    }
    std::vector<float> xn2((size_t) N_EMBD);
    rmsnorm_row(gm, h3.data(), xn2.data());
    std::vector<int> idx;
    std::vector<float> wts;
    route(gw, gb, xn2.data(), idx, wts);
    Arr idr, wr;
    if (!naive_test::load(g_dir, c, "route_idx", idr, g_failures) ||
        !naive_test::load(g_dir, c, "route_wts", wr, g_failures)) return;
    check_ints(c, "trunk L1: all 8 routed expert ids (exact)", idx, idr);
    check_floats(c, "trunk L1: router weights (unbiased p / (sum + 1e-20))", wts, wr,
                 kFp32AbsTol, kFp32Reltol);
    std::vector<float> moe_out;
    moe_ffn(model, xn2.data(), idx, wts, moe_out);
    Arr mo_ref, h4_ref;
    if (!naive_test::load(g_dir, c, "moe_out", mo_ref, g_failures) ||
        !naive_test::load(g_dir, c, "h4", h4_ref, g_failures)) return;
    check_floats(c, "trunk L1: MoE FFN, 8 experts combined (bf16)", moe_out, mo_ref);
    std::vector<float> h4 = add_residual(h3, moe_out);
    check_floats(c, "h4 = bf16(h3 + moe_out) - the last residual", h4, h4_ref);

    // ---- head: output_norm (training rounding) + bf16 logits, chunked so the head's 625 M
    // elements never sit dequantised in RAM (8192 rows of 4096 at a time, 128 MB).
    naive::LoadedTensor gout;
    if (!naive::load_tensor_f32(model, "output_norm.weight", gout).empty()) {
        check(false, "trunk: the output norm loads");
        return;
    }
    std::vector<float> xnl((size_t) N_EMBD);
    rmsnorm_row(gout, h4.data(), xnl.data());
    Arr xnl_ref;
    if (naive_test::load(g_dir, c, "xnl", xnl_ref, g_failures))
        check_floats(c, "xnl = output_norm(h4), training rounding", xnl, xnl_ref);

    std::vector<float> top_vals(5, 0.f);
    std::vector<int> top_ids(5, -1);
    {   // One fp32 pass over the logits, keeping the running top-5 - no 600 KB array to sort.
        // The head is {n_embd, vocab}: a vocab row is a COLUMN of the GGUF table, so the chunk is
        // a column slice (load_tensor_cols), not a row range - 8192 columns (128 MB) at a time.
        struct Top { float v; int id; };
        std::vector<Top> top{{-1e30f, -1}, {-1e30f, -1}, {-1e30f, -1}, {-1e30f, -1}, {-1e30f, -1}};
        constexpr int64_t kChunk = 8192;
        naive::LoadedTensor chunk;
        for (int64_t a = 0; a < VOCAB; a += kChunk) {
            const int64_t n = std::min(kChunk, VOCAB - a);
            if (const std::string err = naive::load_tensor_cols(model, "output.weight",
                                                                (uint64_t) a, (uint64_t) n, chunk);
                !err.empty()) {
                std::printf("  FAIL output.weight chunk %lld: %s\n", (long long) a, err.c_str());
                ++g_failures;
                return;
            }
            const float* w = chunk.data.data();
            for (int64_t j = 0; j < n; ++j) {
                const float* wrow = w + (size_t) j * N_EMBD;   // column j: N_EMBD contiguous
                double acc = 0.0;
                for (int64_t k = 0; k < N_EMBD; ++k) acc += (double) xnl[k] * wrow[k];
                const float v = naive::bf16_round((float) acc);
                if (v > top[4].v) {
                    top[4] = {v, (int) (a + j)};
                    std::sort(top.begin(), top.end(), [](const Top& x, const Top& y2) { return x.v > y2.v; });
                }
            }
        }
        for (int i = 0; i < 5; ++i) { top_ids[i] = top[i].id; top_vals[i] = top[i].v; }
    }
    Arr tid, tval;
    if (!naive_test::load(g_dir, c, "top_ids", tid, g_failures) ||
        !naive_test::load(g_dir, c, "top_vals", tval, g_failures)) return;
    check_ints(c, "the trunk's logits: top-5 token ids (exact ints)", top_ids, tid);
    check_floats(c, "the trunk's logits: top-5 values (bf16-rounded)", top_vals, tval);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = g_dir, model = g_model, only = "all";
    if (argc > 1) dir = g_dir = argv[1];
    if (argc > 2) model = g_model = argv[2];
    if (argc > 3) only = argv[3];
    std::printf("naive moe/dsa/trunk test (fixtures: %s, case: %s)\n", dir.c_str(), only.c_str());

    std::error_code ec;
    if (!std::filesystem::exists(model, ec)) {
        std::printf("  model not present (%s): everything this test checks needs it - skipped\n",
                    model.c_str());
        return 77;
    }
    strata::GgufModel m = strata::GgufModel::open(model);

    // The fixtures were generated from this model's metadata; a file that moved under them would
    // turn every number below into a coincidence.  Assert the geometry, refuse on mismatch.
    naive::NaiveGeometry g;
    if (std::string err = naive::read_geometry(m.meta(), g); !err.empty()) {
        check(false, ("geometry: " + err).c_str());
        return 1;
    }
    {
        const strata::MetaValue* top = m.meta().get("naive_n05_flash.index_top_k");
        const strata::MetaValue* ih = m.meta().get("naive_n05_flash.index_n_heads");
        const strata::MetaValue* idim = m.meta().get("naive_n05_flash.index_head_dim");
        const strata::MetaValue* kv0 = m.meta().get("naive_n05_flash.attention.head_count_kv");
        bool ok = top && ih && idim && kv0 && top->u == 2048 && ih->u == (uint64_t) INDEX_HEADS &&
                  idim->u == (uint64_t) INDEX_DIM && kv0->count >= 2 &&
                  kv0->items[0].u == (uint64_t) N_KV_DSA && kv0->items[1].u == (uint64_t) N_KV_SWA;
        check(ok, "metadata: index_top_k 2048, 16x128 indexer, kv heads [4 (L0 DSA), 8 (L1 SWA)]");
    }
    check(g.n_layers == 48 && g.n_expert == N_EXPERT && g.n_expert_used == TOPK &&
              g.n_ff_expert == EXPERT_FF && g.index_top_k == 2048,
          "geometry: 48 layers, 256 experts / top-8 / ff 2048, index top-k 2048");

    if (only == "all" || only == "moe") case_moe(m);
    if (only == "all" || only == "dsa") case_dsa(m);
    if (only == "all" || only == "trunk") case_trunk(m);

    if (g_failures) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("all naive moe/dsa/trunk checks passed\n");
    return 0;
}
