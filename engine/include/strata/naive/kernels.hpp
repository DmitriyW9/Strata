// include/strata/naive/kernels.hpp - the Naive-N0.5-Forward's numeric building blocks, one small
// function per piece of the arithmetic, each a direct transcription of the model authors' runtime
// (NaiveRT) so the two can be checked against each other on the same numbers.
//
// This is the trunk's arithmetic WITHOUT the GPU kernels.  NaiveRT runs the same math through Triton
// and sm_103a CUDA kernels that will not compile on this box (sm_86 / sm_70), so what is borrowed
// here is the *formula*, not the code: every function below mirrors a named reference in NaiveRT's
// own torch tests and model code, and `terminal/tools/naive_kernels_oracle.py` (a numpy transcription of the
// same references) plus `tests/naive/test_naive_kernels.cpp` (this header against those fixtures)
// hold the C++ to that reference.  Nothing here loads a weight or touches a GPU; a later loader
// feeds these functions real tensors, and a later GPU backend can replace them with kernels.
//
// Every reference the transcription is taken from is cited above its function, with the file in
// NaiveRT and the symbol, so a reviewer can open both and compare.  The dtypes match the model: the
// trunk runs in BF16 with an FP32 accumulate, the DSA indexer quantises its keys and queries to FP8
// E4M3, and the router scores in FP32.  Where a reference rounds an intermediate to BF16 (the
// projections, the norms, the expert combine), that rounding is `bf16_round`, applied in the caller's
// data flow rather than inside these pure functions, so a fixture can feed the same rounded values
// to the C++ and to the oracle and compare like with like.
//
// Dependency-free on purpose, like naive/geometry.hpp: <cmath>/<cstring>/<cstdint> only.  The oracle
// reimplements each of these independently in numpy, so agreement is a real cross-check, not a tautology.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::naive {

// ---------------------------------------------------------------- bf16 helpers
// The trunk's storage dtype.  `bf16_round` is round-to-nearest-even on the 16 low mantissa bits,
// which is what a cast to bfloat16 does; NaN keeps its NaN-ness.  Kept here (and mirrored in the
// oracle) because so many references round an fp32 intermediate to bf16 at a boundary, and the test
// must round the same way on both sides.
inline float bf16_round(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    if (((b >> 23) & 0xff) == 0xff) return x;              // NaN / Inf: pass through unchanged
    const uint32_t lsb = (b >> 16) & 1u;
    b += 0x7fffu + lsb;                                    // ties away-to-even, then truncate low 16
    b &= 0xffff0000u;
    float out;
    std::memcpy(&out, &b, 4);
    return out;
}

// ---------------------------------------------------------------- fp8 e4m3fn
// The DSA indexer's activation dtype: 4-bit exponent, 3-bit mantissa, bias 7, no infinities, one NaN
// pattern (exponent all-ones, mantissa 7), max finite 448.  `encode` is round-to-nearest-even with
// saturation to +/-448; `decode` is the exact inverse of an encoded code.  Transcribed from the
// float8_e4m3fn cast the references use (tests/gpu/attn_dsa/run_attn_dsa.py:quantize_e4m3_rows)
// and checked against the oracle over a grid that includes every rounding boundary.
inline uint8_t fp8_e4m3_encode(float x) {
    if (std::isnan(x)) return 0x7f;
    const uint8_t sign = std::signbit(x) ? 0x80u : 0u;
    const float a = std::fabs(x);
    if (a >= 464.0f) return (uint8_t) (sign | 0x7eu);       // >= the 448/512 midpoint: saturate to 448
    int e;
    const float fr = std::frexp(a, &e);                     // a = fr * 2^e, fr in [0.5, 1)
    int p = e - 1;                                          // a = (fr*2) * 2^p, fr*2 in [1, 2)
    if (a == 0.0f || p < -6) {                              // below the smallest normal: subnormal, step 2^-9
        const int m = (int) std::nearbyintf(a * 512.0f);    // a / 2^-9, round-to-nearest-even
        if (m >= 8) return (uint8_t) (sign | 0x08u);        // rounded up into the smallest normal (E=1, M=0)
        return (uint8_t) (sign | (uint8_t) m);
    }
    int m3 = (int) std::nearbyintf((fr * 2.0f - 1.0f) * 8.0f);
    if (m3 == 8) {                                          // mantissa carried: bump the exponent
        m3 = 0;
        ++p;
    }
    int E = p + 7;                                          // biased exponent
    if (E >= 15 && m3 > 6) return (uint8_t) (sign | 0x7eu); // above 448's code: saturate
    if (E > 15) { E = 15; m3 = 6; }
    return (uint8_t) (sign | (uint8_t) (E << 3) | (uint8_t) m3);
}

inline float fp8_e4m3_decode(uint8_t b) {
    const int E = (b >> 3) & 0xf;
    const int m = b & 7;
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    if (E == 15 && m == 7) return std::nanf("");            // the e4m3fn NaN pattern
    if (E == 0) return sign * (float) m * 0.001953125f;     // subnormal: m * 2^-9
    return sign * (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, E - 7);
}

// ---------------------------------------------------------------- rmsnorm
// x * rsqrt(mean(x^2) + eps) * gamma, in fp32, returned *before* the bf16 rounding the reference
// applies (so a fixture can check the fp32 value, and bf16_round gives the stored one).
// NaiveRT: tests/gpu/attn_swa/run_attn_swa.py:rmsnorm (identical in attn_dsa), eps 1e-5.
inline void rmsnorm(const float* x, const float* gamma, int n, float eps, float* out) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) sumsq += (double) x[i] * x[i];
    const float rstd = (float) (1.0 / std::sqrt(sumsq / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * rstd * gamma[i];
}

// The same norm with the TRAINING path's rounding: the normalized vector rounds to bf16 first, then
// gamma multiplies it (`normalized.to(torch.bfloat16) * gamma`).  NaiveRT:
// naivert/models/naive_n0p5/training_moe.py:rmsnorm - the model's own training spec, the only place
// training rounding is written down, and the path the MoE fixtures follow.  The two rmsnorm variants
// differ by at most one bf16 ulp per element (gamma is bf16 on both sides); the MoE/DSA/trunk
// fixtures and the trunk tests pick this one, the layer-0/SWA fixtures of test_naive_layer.cpp pick
// the round-after-gamma variant their reference (run_attn_swa.py) uses.  The gamma entries are
// rounded here as the oracle does, so the caller's gamma may be the file's f32 values.
inline void rmsnorm_training(const float* x, const float* gamma, int n, float eps, float* out) {
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) sumsq += (double) x[i] * x[i];
    const float rstd = (float) (1.0 / std::sqrt(sumsq / n + eps));
    for (int i = 0; i < n; ++i)
        out[i] = bf16_round(bf16_round(x[i] * rstd) * bf16_round(gamma[i]));
}

// ---------------------------------------------------------------- split-half RoPE
// rotate_half (GPT-NeoX / HF) on the first n_rot dims of a head, the remaining dims copied.  With
// inv[j] = 1 / theta^(2j/n_rot) and ang = pos * inv: for i in the first half out[i] = x[i]cos -
// x[i+n_rot/2]sin, and for the second half out[i] = x[i]cos + x[i-n_rot/2]sin, both at ang[i mod
// n_rot/2].  theta is 1e7 on the DSA layers and 1e4 on the SWA layers.
// NaiveRT: tests/gpu/attn_swa/run_attn_swa.py:hf_rope (dims past ROPE copied in _rope_fma, ops.py).
inline void rope_head(const float* x, int head_dim, int n_rot, int64_t pos, double theta, float* out) {
    const int half = n_rot / 2;
    for (int j = 0; j < half; ++j) {
        const double inv = 1.0 / std::pow(theta, (double) (2 * j) / n_rot);
        const double ang = (double) pos * inv;
        const float c = (float) std::cos(ang), s = (float) std::sin(ang);
        const float x1 = x[j], x2 = x[j + half];
        out[j] = x1 * c - x2 * s;
        out[j + half] = x2 * c + x1 * s;
    }
    for (int i = n_rot; i < head_dim; ++i) out[i] = x[i];    // not rotated
}

// rope_head applied to every head of a head-major block at one position: x and out are
// [n_head, head_dim].  The references rotate exactly this way: run_attn_swa.py:reference_query /
// reference_caches take `q.view(HEADS, HEAD_DIM)` / k rows and assign `[:, :ROPE] = hf_rope(...)` -
// the first 64 dims of each head - and naivert's prefill rope op (prefill.py:apply_rope) takes the
// same [T, H, 192] view with a [max_pos, 64] table.
inline void rope_rows(const float* x, int n_head, int head_dim, int n_rot, int64_t pos, double theta,
                      float* out) {
    for (int h = 0; h < n_head; ++h)
        rope_head(x + (size_t) h * head_dim, head_dim, n_rot, pos, theta, out + (size_t) h * head_dim);
}

// ---------------------------------------------------------------- SWA attention (with a sink)
// One query head set against a contiguous slice of keys/values, sliding-window causal, with one
// learnable sink logit per head appended to each row's logits (so the softmax denominator includes
// exp(sink[h])) and that appended column dropped from the weights.  softmax scale 1/sqrt(head_dim_k).
// No sink on the DSA layers - pass sink=nullptr there and this reduces to plain softmax attention.
// NaiveRT: tests/gpu/attn_swa/run_attn_swa.py:reference_attention (fp32 path, out_fp32).
inline void swa_attention(const float* q, int n_head, int head_dim_k, const float* k, const float* v,
                          int head_dim_v, const float* sink, int first, int last, float* out) {
    const double scale = 1.0 / std::sqrt((double) head_dim_k);
    const int n = last - first + 1;
    std::vector<double> acc_v((size_t) head_dim_v);               // V accumulator, widened to double
    for (int h = 0; h < n_head; ++h) {
        const float* qh = q + (size_t) h * head_dim_k;
        // logits over the visible keys (plus the sink in a running max/sum, no buffer needed).
        double maxlog = sink ? sink[h] : -INFINITY;
        for (int j = 0; j < n; ++j) {
            const float* kj = k + (size_t) (first + j) * head_dim_k;
            double acc = 0.0;
            for (int d = 0; d < head_dim_k; ++d) acc += (double) qh[d] * kj[d];
            const double s = acc * scale;
            if (s > maxlog) maxlog = s;
        }
        double denom = sink ? std::exp(sink[h] - maxlog) : 0.0;    // the sink's share of Z
        for (int j = 0; j < n; ++j) {
            const float* kj = k + (size_t) (first + j) * head_dim_k;
            double acc = 0.0;
            for (int d = 0; d < head_dim_k; ++d) acc += (double) qh[d] * kj[d];
            denom += std::exp(acc * scale - maxlog);
        }
        const double inv = 1.0 / denom;
        for (int d = 0; d < head_dim_v; ++d) acc_v[d] = 0.0;
        for (int j = 0; j < n; ++j) {
            const float* kj = k + (size_t) (first + j) * head_dim_k;
            double acc = 0.0;
            for (int d = 0; d < head_dim_k; ++d) acc += (double) qh[d] * kj[d];
            const double w = std::exp(acc * scale - maxlog) * inv;
            const float* vj = v + (size_t) (first + j) * head_dim_v;
            for (int d = 0; d < head_dim_v; ++d) acc_v[d] += w * vj[d];
        }
        float* oh = out + (size_t) h * head_dim_v;
        for (int d = 0; d < head_dim_v; ++d) oh[d] = (float) acc_v[d];
    }
}

// ---------------------------------------------------------------- DSA index-key LayerNorm
// Plain LayerNorm (mean, variance, then affine out = bias + (x-mean)*rstd*weight) - NOT rmsnorm.
// NaiveRT: naivert/prefill/ops.py:index_key_norm; tests/gpu/attn_dsa/run_attn_dsa.py:reference_caches
// (torch.nn.functional.layer_norm), eps 1e-5.
inline void index_key_layernorm(const float* x, const float* weight, const float* bias, int n,
                                float eps, float* out) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += x[i];
    const float mean = (float) (sum / n);
    double vs = 0.0;
    for (int i = 0; i < n; ++i) { const double d = (double) (x[i] - mean); vs += d * d; }
    const float rstd = (float) (1.0 / std::sqrt(vs / n + eps));
    for (int i = 0; i < n; ++i) out[i] = bias[i] + (x[i] - mean) * rstd * weight[i];
}

// ---------------------------------------------------------------- fp8 row quantisation (dequantised back)
// One scale per row (row amax clamped to 1e-30, divided by 448), each element encoded to e4m3 and
// immediately decoded and rescaled, which is exactly the value the attention scores use.  The rows
// are expected already rounded to bf16, as the reference casts them (ki.to(bfloat16)) before quantising.
// NaiveRT: tests/gpu/attn_dsa/run_attn_dsa.py:quantize_e4m3_rows.
inline void fp8_rows_dequant(const float* rows, int m, int n, float* deq, float* scales) {
    for (int r = 0; r < m; ++r) {
        const float* row = rows + (size_t) r * n;
        float amax = 0.0f;
        for (int i = 0; i < n; ++i) amax = amax > std::fabs(row[i]) ? amax : std::fabs(row[i]);
        if (!(amax > 1e-30f)) amax = 1e-30f;
        const float scale = amax / 448.0f;
        if (scales) scales[r] = scale;
        float* o = deq + (size_t) r * n;
        for (int i = 0; i < n; ++i) o[i] = fp8_e4m3_decode(fp8_e4m3_encode(row[i] / scale)) * scale;
    }
}

// ---------------------------------------------------------------- DSA index logits
// For each visible key j: logit[j] = sum over the 16 index-query heads of relu(q_h . k_j) * w[h].
// q and k are the e4m3-dequantised rows; w already carries the 1/sqrt(n_heads) fold.  No softmax scale.
// NaiveRT: tests/gpu/attn_dsa/run_attn_dsa.py:reference_selection (logits line); fold in
// naivert/kernels/naive_n0p5/prefill.py:fold_index_weights.
inline void dsa_index_logits(const float* iq_deq, int n_heads, int head_dim, const float* w,
                             const float* ki_deq, int visible, float* logits) {
    for (int j = 0; j < visible; ++j) {
        const float* kj = ki_deq + (size_t) j * head_dim;
        double acc = 0.0;
        for (int h = 0; h < n_heads; ++h) {
            const float* qh = iq_deq + (size_t) h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
            if (dot > 0.0f) acc += (double) dot * w[h];      // relu, then the per-head weight
        }
        logits[j] = (float) acc;
    }
}

// ---------------------------------------------------------------- stable top-k (DSA selection)
// Pick the k largest of v[0..n).  A position wins on strictly-greater value; ties go to the lower
// index; the winners are returned in ascending index order.  (For distinct values this is the plain
// top-k sorted; the rule only decides ties, matching the model's stable DSA selection.)
// NaiveRT: naivert/prefill/native/stable_dsa_topk.py.
inline void stable_topk(const float* v, int n, int k, int* out) {
    if (k > n) k = n;
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (!(v[i] == v[i])) continue;                            // NaN ranks nowhere: it never wins,
        int rank = 0;                                             // and it cannot break the k-write bound
        for (int j = 0; j < n; ++j) {
            if (v[j] > v[i]) ++rank;
            else if (j < i && v[j] == v[i]) ++rank;          // lower index wins a tie
        }
        if (rank < k) out[cnt++] = i;
    }
}

// ---------------------------------------------------------------- MoE router (sigmoid + bias)
// scores[e] = x . W[e] (fp32); p = sigmoid(scores); pick the top-k by (p + bias), the bias only
// breaking ties in *which* experts; the selected weights are the *unbiased* p, renormalised over the
// k by (sum + 1e-20).  Experts are chosen stably (strictly greater, lower index on tie) and returned
// in ascending id order; combine is order-independent, so the ids order is only for a stable test.
// NaiveRT: naivert/models/naive_n0p5/training_moe.py:route (torch.sigmoid, topk(p+bias), gather p,
// renorm with 1e-20).
inline void moe_route(const float* x, const float* weight, const float* bias, int n_embd, int n_expert,
                      int topk, int* idx_out, float* wts_out) {
    std::vector<float> score((size_t) n_expert), key((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        const float* we = weight + (size_t) e * n_embd;
        double acc = 0.0;
        for (int d = 0; d < n_embd; ++d) acc += (double) x[d] * we[d];
        const float p = (float) (1.0 / (1.0 + std::exp(-acc)));
        score[e] = p;
        key[e] = p + bias[e];
    }
    stable_topk(key.data(), n_expert, topk, idx_out);
    double sum = 0.0;
    for (int t = 0; t < topk; ++t) sum += score[idx_out[t]];
    const double inv = 1.0 / (sum + 1e-20);
    for (int t = 0; t < topk; ++t) wts_out[t] = (float) (score[idx_out[t]] * inv);
}

// ---------------------------------------------------------------- MoE combine
// out = sum over the selected experts of w[t] * expert_out[t], fp32 (the reference adds the residual
// after and rounds the sum to bf16).  NaiveRT: training_moe.py combine ((out.float()*w).sum(1).to(bf16)).
inline void moe_combine(const float* expert_out, const float* w, int topk, int n, float* out) {
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int t = 0; t < topk; ++t) acc += (double) w[t] * expert_out[(size_t) t * n + i];
        out[i] = (float) acc;
    }
}

// ---------------------------------------------------------------- dense SwiGLU activation
// The layer-0 dense FFN (and each expert) gate and up with silu: act = silu(gate) * up, silu(g) =
// g / (1 + e^-g).  NaiveRT: training_moe.py expert_activation_from_float (gate*sigmoid(gate)*up).
inline float silu(float g) { return g / (1.0f + std::exp(-g)); }

}  // namespace strata::naive
