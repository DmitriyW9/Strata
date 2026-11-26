// include/strata/naive/runtime.hpp - the Naive-N0.5-Flash forward RUNTIME: the running caches (the
// DSA layers' full-history K/V + the E4M3 index-key cache, the window layers' ring), the 48-layer
// loop over rows, the output head, the sampler, and the GEN line's parser.  On top of the four
// tested pieces - kernels.hpp's arithmetic (each function still names its NaiveRT reference),
// loader.hpp's byte windows and ggml decoders, reader.hpp's geometry gate, trunk.hpp's fp8 packing -
// it adds only the state between rows, the order between layers, and the top-k fast path.
//
// Every per-row, per-layer computation mirrors the wiring tests/naive/test_naive_moe_dsa.cpp pins
// against fixtures (rmsnorm_training -> projections bf16 -> split-half rope per key's own position
// -> the DSA indexer's LayerNorm/rope/E4M3 cache -> selection -> grouped attention (the sink on the
// window layers only) -> bf16 residual -> ffn norm -> dense SwiGLU or the sigmoid/noaux_tc router +
// IQ2 experts -> bf16 residual), so the runtime is the same arithmetic the four `ctest -R naive`
// suites already hold to the numpy reference, now running over a session.  The visible-range
// arithmetic is NaiveRT's: a window query at p reads [p-window+1, p] (naivert/prefill/adapters.py,
// "a query at p reads [p - window + 1, p], so once the chunk is over, every earlier row of it is
// already out of every future window"), and a DSA query selects its keys over the layer's whole
// history (naivert/kernels/naive_n0p5/prefill.py:indexer_select, width max(max_seqlen, topk): with
// top-k visible every row is visible; otherwise the stable top-k rows, ascending).
//
// Caches (per session, rewound with the conversation, never requantised):
//   * DSA layer l:  K  bf16 codes, n_kv*192 per position, every position ever written;
//                   V  bf16 codes, n_kv*128 per position (the 0.707 folded at write time, as the
//                      tested dsa case folds it);
//                   ki the E4M3 row's DEQUANTISED values, index_head_dim per position - what the
//                      selection logits read (fp8_rows_dequant's output; the code itself carries no
//                      extra information the logits need).
//   * window layer l: a ring of `window + history` rows of the same K/V shape (the model card's
//                     "window plus a chunk"; `history` - the DSA rewind headroom, see rewind() -
//                     widens the chunk past the model's attention_chunk so a branch inside it
//                     rewinds instead of re-prefilling).
// bf16 codes, not floats: every cached value is a bf16-rounded float already (the tested references
// round K/V to bf16 before caching), so a bf1616 code stores exactly what the float did, at half the
// bytes: at the model's own 262k plan that is the difference between ~13 GiB and ~6.6 GiB of cache.
//
// Determinism: every GEMV still sums one output row in double in k order (loader.hpp's rule), so
// OpenMP parallelism splits ROWS, never a row's sum, and the same input gives byte-identical
// numbers single- or multi-threaded.  The selection's fast path picks the same winners as
// kernels.hpp::stable_topk (value descending, lower index wins a tie, winners ascending).
//
// CPU-only by design, like the rest of the strata-naive path (docs/NAIVE.md).
#pragma once

#include "strata/naive/gpu.hpp"
#include "strata/naive/kernels.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/pool.hpp"
#include "strata/naive/reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace strata::naive {

// ---------------------------------------------------------------- bf16 storage helpers
// A cached value is a bf16_round()'ed float, so its high 16 bits ARE the value (the low 16 are 0).
inline uint16_t bf16_bits(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return (uint16_t) (b >> 16);
}
inline float bf16_val(uint16_t b) {
    const uint32_t x = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------- the fast stable top-k
// The same rule as kernels.hpp::stable_topk - a position wins on strictly-greater value, ties go to
// the lower index, the winners come back in ascending index order - in O(n) average instead of
// O(n^2): nth_element on that order, then sort the first k.  A test checks it against the O(n^2)
// reference, including tie-heavy vectors, where the two orderings must agree elementwise.
inline void stable_topk_fast(const float* v, int n, int k, std::vector<int32_t>& out) {
    out.clear();
    if (n <= 0 || k <= 0) return;
    const int kk = std::min(k, n);
    std::vector<int32_t> idx((size_t) n);
    if (kk == n) {                       // everything wins: ascending, which IS the sorted order
        for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
        out.swap(idx);
        return;
    }
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    auto better = [v](int32_t a, int32_t b) {
        const float va = v[a], vb = v[b];
        const bool na = !(va == va), nb = !(vb == vb);
        if (na != nb) return !na;                              // NaN loses to everything (and never
        return va != vb ? va > vb : a < b;                     // to itself: a strict order, also for NaN)
    };
    std::nth_element(idx.begin(), idx.begin() + kk, idx.end(), better);
    idx.resize((size_t) kk);
    std::sort(idx.begin(), idx.end());                       // winners ascending, as the reference does
    out.swap(idx);
}

// ---------------------------------------------------------------- the GEN line's request
// `GEN <max_new> [key=value ...] <id,id,...>` - the shape of the serve protocol's request line
// (src/program/generate.cpp parses the same keys for the Flash-Next engine; keys this engine has no
// meaning for are accepted and ignored, exactly as that parser skips unknown keys, and the ids start
// at the first token without an '=').
struct GenReq {
    int64_t max_new = 0;
    std::vector<int32_t> ids;
    // Sampling.  Absent keys keep the engine greedy: temperature 0, like the main engine's default.
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 40;
    unsigned long long seed = 0;                             // 0: the engine's own entropy at first use
    // Repetition penalties, llama's convention (the main engine's key set): the last penalty_last_n
    // tokens of the session starting at pen_from get logit>0 ? logit/penalty : logit*penalty, then
    // the frequency and presence subtractions.
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_presence = 0.0f;
    int64_t pen_from = -1;
};

inline bool parse_gen_line(const std::string& line, GenReq& q, std::string& err) {
    const char* p = line.c_str() + 4;                        // past "GEN "
    char* endp = nullptr;
    q = GenReq{};
    q.max_new = std::strtoll(p, &endp, 10);
    if (endp == p) { err = "GEN needs a max_new"; return false; }
    for (;;) {
        while (*endp == ' ') ++endp;
        const char* start = endp;
        while (*endp != '\0' && *endp != ' ') ++endp;
        if (endp == start) break;
        const std::string tok(start, (size_t) (endp - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {                       // the ids: the first token without an '='
            size_t pos = 0;
            while (pos < tok.size()) {
                const size_t c = tok.find(',', pos);
                const std::string one = tok.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                if (!one.empty()) q.ids.push_back((int32_t) std::strtol(one.c_str(), nullptr, 10));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
            break;
        }
        const std::string key = tok.substr(0, eq);
        const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
        if (key == "temperature") q.temperature = fv;
        else if (key == "top_p") q.top_p = fv;
        else if (key == "top_k") q.top_k = std::atoi(tok.c_str() + eq + 1);
        else if (key == "min_p") q.min_p = fv;
        else if (key == "seed") q.seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
        else if (key == "penalty_last_n") q.penalty_last_n = std::atoi(tok.c_str() + eq + 1);
        else if (key == "penalty_repeat") q.penalty_repeat = fv;
        else if (key == "penalty_freq") q.penalty_freq = fv;
        else if (key == "penalty_present") q.penalty_presence = fv;
        else if (key == "pen_from") q.pen_from = std::atoll(tok.c_str() + eq + 1);
        // every other key (logprobs, ckpt, hold, draft, ...) is this engine's to ignore, like the
        // main parser skips unknown keys; the ids start at the first token without an '='.
    }
    if (q.ids.empty()) { err = "GEN carries no prompt ids"; return false; }
    return true;
}

// ---------------------------------------------------------------- the sampler
// Greedy at temperature 0 (first argmax wins - the same id for the same logits, on any thread count).
// Otherwise the main engine's shape: penalties, then divide by temperature, top-k, min-p (a fraction
// of the best), top-p, then a draw from the survivors with a seeded mt19937_64.  The generator is
// the engine's own (seed 0: seeded from entropy once, then carried), so one seed replays one reply.
class Sampler {
public:
    int32_t sample(const float* logits, int n, const GenReq& q, const std::vector<int32_t>& history) {
        work_.assign(logits, logits + n);
        apply_penalties(history, q);
        if (q.temperature <= 0.0f) {
            int best = 0;
            for (int i = 1; i < n; ++i)
                if (work_[i] > work_[best]) best = i;        // first max wins
            return best;
        }
        const float inv = 1.0f / q.temperature;
        for (int i = 0; i < n; ++i) work_[i] = work_[i] * inv;
        // top-k over a candidate list, then the p-filters over the sorted front.
        const int k = std::max(1, std::min(q.top_k <= 0 ? n : q.top_k, n));
        cand_.resize((size_t) n);
        for (int i = 0; i < n; ++i) cand_[(size_t) i] = {work_[i], i};
        std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(),
                          [](const Cand& a, const Cand& b) { return a.v > b.v; });
        const float top = cand_[0].v;
        int m = k;
        if (q.min_p > 0.0f) {                                // min-p over the top-k set
            const float cut = top + std::log(std::max(q.min_p, 1e-20f));  // p_i >= min_p * p_max
            while (m > 1 && cand_[(size_t) m - 1].v < cut) --m;
        }
        if (q.top_p < 1.0f && m > 1) {                       // top-p over the same order
            std::vector<float> pr((size_t) m);
            float mx = cand_[0].v, sum = 0.0f;
            for (int i = 0; i < m; ++i) { pr[(size_t) i] = std::exp(cand_[(size_t) i].v - mx); sum += pr[(size_t) i]; }
            float acc = 0.0f;
            int keep = m;
            for (int i = 0; i < m; ++i) {
                acc += pr[(size_t) i] / sum;
                if (acc >= q.top_p) { keep = i + 1; break; }
            }
            m = std::max(1, keep);
        }
        std::vector<double> w((size_t) m);
        double mx = cand_[0].v, tot = 0.0;
        for (int i = 0; i < m; ++i) { w[(size_t) i] = std::exp((double) cand_[(size_t) i].v - mx); tot += w[(size_t) i]; }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double pick = u(rng) * tot, acc = 0.0;
        for (int i = 0; i < m; ++i) {
            acc += w[(size_t) i];
            if (pick < acc) return cand_[(size_t) i].id;
        }
        return cand_[(size_t) m - 1].id;
    }

    void reseed(unsigned long long seed) {                   // seed 0: entropy once, then carried
        if (seed != 0) rng.seed(seed);
        else if (!seeded_) {
            std::random_device rd;
            rng.seed(((unsigned long long) rd() << 32) ^ rd());
            seeded_ = true;
        }
    }

private:
    struct Cand { float v; int32_t id; };
    std::mt19937_64 rng{1};
    bool seeded_ = false;
    std::vector<float> work_;
    std::vector<Cand> cand_;

    void apply_penalties(const std::vector<int32_t>& history, const GenReq& q) {
        if (q.penalty_last_n <= 0 || history.empty()) return;
        if (q.penalty_repeat == 1.0f && q.penalty_freq == 0.0f && q.penalty_presence == 0.0f) return;
        const int64_t from = q.pen_from >= 0 ? q.pen_from : 0;
        const int64_t lo = std::max<int64_t>(from, (int64_t) history.size() - q.penalty_last_n);
        std::vector<int> counts(work_.size(), 0);
        for (int64_t i = lo; i < (int64_t) history.size(); ++i) {
            const int32_t t = history[(size_t) i];
            if (t >= 0 && t < (int32_t) work_.size()) ++counts[(size_t) t];
        }
        for (size_t t = 0; t < counts.size(); ++t) {
            if (!counts[t]) continue;
            if (q.penalty_repeat != 1.0f)
                work_[t] = work_[t] > 0.0f ? work_[t] / q.penalty_repeat : work_[t] * q.penalty_repeat;
            if (q.penalty_freq > 0.0f) work_[t] -= q.penalty_freq * counts[t];
            if (q.penalty_presence > 0.0f) work_[t] -= q.penalty_presence;
        }
    }
};

// ---------------------------------------------------------------- the runtime
class NaiveRuntime {
public:
    /// `pool` / `gpu` are optional (nullptr: read through the mmap windows, the pre-pool path). When
    /// a pool is given, every weight read below first asks it: a resident dense tensor reads its fp32
    /// copy, a resident expert stack its raw copy (bit-identical to the mmap, pool.hpp), and a stack
    /// the plan placed on a card (gpu->handle_of) runs its GEMV there. A name the pool does not hold
    /// falls back to the mmap windows - the same bytes, the same arithmetic, just the disk behind them.
    NaiveRuntime(const GgufModel& model, const NaiveGeometry& g, int threads, int64_t history_rows,
                 std::string& err, const NaivePool* pool = nullptr, NaiveGpu* gpu = nullptr)
        : m_(model), g_(g), threads_(threads > 0 ? threads : 1), pool_(pool), gpu_(gpu) {
        if (g_.add_full_sink) { err = "DSA-layer sinks (add_full_sink) are not implemented by this build"; return; }
        if (g_.n_dsa_layers() > 0 && !g_.dsa_enabled) { err = "dsa_enabled is false but the pattern has DSA layers"; return; }
        if (!g_.expert_weights_norm) { err = "expert_weights_norm false: this router always renormalises"; return; }
        if (g_.n_rot > g_.head_dim_k || g_.n_rot_swa > g_.head_dim_k) { err = "rope dimensions exceed the head width"; return; }
        if (g_.index_n_kv_heads != 1) { err = "index_n_kv_heads other than 1 is not implemented"; return; }
        if (g_.index_n_heads * g_.index_head_dim > (int64_t) 1 << 20) { err = "indexer too wide"; return; }
        if (g_.n_vocab <= 0) { err = "no tokenizer vocabulary in the metadata"; return; }
        const int64_t ring = std::max<int64_t>(g_.swa_window + std::max<int64_t>(g_.attn_chunk, history_rows),
                                               2 * g_.swa_window);
        const size_t nl = (size_t) g_.n_layers;
        ring_cap_.assign(nl, 0);
        for (size_t l = 0; l < nl; ++l)
            if (!g_.is_dsa_layer((int64_t) l)) ring_cap_[l] = (uint64_t) ring;
        dsa_k_.resize(nl); dsa_v_.resize(nl); dsa_ki_.resize(nl);
        swa_k_.resize(nl); swa_v_.resize(nl);
        lw_.resize(nl);
        sel_.resize(nl);
        moe_.resize(nl);
        // Scratch sized for the widest rows of the model.
        x_.assign((size_t) g_.n_embd, 0.f);
        xn_.assign((size_t) g_.n_embd, 0.f);
        qrow_.assign((size_t) (g_.n_head * g_.head_dim_k), 0.f);
        qro_.assign(qrow_.size(), 0.f);
        krow_.assign((size_t) (8 * g_.head_dim_k), 0.f);
        vrow_.assign((size_t) (8 * g_.head_dim_v), 0.f);
        logits_.assign((size_t) g_.n_vocab, 0.f);
        sel_logits_.assign(1024, 0.f);
        err.clear();
        ok_ = true;
    }

    bool ok() const { return ok_; }
    const NaiveGeometry& geom() const { return g_; }
    int threads() const { return threads_; }
    int64_t session_len() const { return (int64_t) tok_.size(); }
    const std::vector<int32_t>& session() const { return tok_; }

    /// The cache bytes the session may grow to at `ctx` positions (the plan the engine prints as INFO
    /// so the number stays checkable, like the metadata-only build's kv_cache_gib line).
    double cache_gib_at(int64_t ctx) const {
        double bytes = 0;
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            const int64_t nkv = g_.kv_heads(l);
            if (g_.is_dsa_layer(l))
                bytes += (double) ctx * (nkv * (g_.head_dim_k + g_.head_dim_v) * 2.0 + g_.index_head_dim * 4.0);
            else
                bytes += (double) ring_cap_[(size_t) l] * nkv * (g_.head_dim_k + g_.head_dim_v) * 2.0;
        }
        return bytes / (1024.0 * 1024.0 * 1024.0);
    }

    void reset() {
        tok_.clear();
        for (size_t l = 0; l < dsa_k_.size(); ++l) {
            dsa_k_[l].clear(); dsa_v_[l].clear(); dsa_ki_[l].clear();
            swa_k_[l].clear(); swa_v_[l].clear();
        }
    }

    /// The longest common prefix of `ids` with what the caches already hold.
    int64_t common_prefix(const std::vector<int32_t>& ids) const {
        const int64_t n = std::min<int64_t>(ids.size(), tok_.size());
        int64_t i = 0;
        while (i < n && tok_[(size_t) i] == ids[(size_t) i]) ++i;
        return i;
    }

    /// Rewind every cache to `d` positions.  Possible while the window rings still hold the rows
    /// the query at d-1 saw (a row written at position r is later overwritten at r+ring cap, so a
    /// rewind to d is safe while d >= len - cap + window); DSA rows are never overwritten, so only
    /// the rings decide.  false: the rings cannot reach that far back, the caller resets.
    bool rewind(int64_t d) {
        if (d < 0 || d > (int64_t) tok_.size()) return false;
        // Check every ring first: a false answer must leave the session usable as it was, the
        // caller resets only when this returns false and it must not half-shrink the DSA caches.
        for (int64_t l = 0; l < g_.n_layers; ++l)
            if (!g_.is_dsa_layer(l) &&
                d - g_.swa_window + 1 < (int64_t) tok_.size() - (int64_t) ring_cap_[(size_t) l])
                return false;
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            if (g_.is_dsa_layer(l)) {
                const size_t rowk = (size_t) (g_.kv_heads(l) * g_.head_dim_k);
                const size_t rowv = (size_t) (g_.kv_heads(l) * g_.head_dim_v);
                dsa_k_[l].resize((size_t) d * rowk);
                dsa_v_[l].resize((size_t) d * rowv);
                dsa_ki_[l].resize((size_t) d * (size_t) g_.index_head_dim);
            }
        }
        tok_.resize((size_t) d);
        return true;
    }

    /// Feed one token at position tok_.size() and write its caches.  With want_logits the output
    /// head also runs and logits() holds the bf16-rounded row.  "" or a reason naming the tensor.
    std::string forward_row(int32_t token, bool want_logits) {
        const int64_t pos = (int64_t) tok_.size();
        // The embedding row: one column of the {n_embd, vocab} table (loader.hpp's column window).
        LoadedTensor emb;
        if (const std::string e = load_cols("token_embd.weight", (uint64_t) token, 1, emb); !e.empty())
            return e;
        std::memcpy(x_.data(), emb.data.data(), (size_t) g_.n_embd * sizeof(float));
        if (dbg == 2) dbg_pts.emplace_back("emb", x_);   // the embedding row, before layer 0
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            if (const std::string e = layer_row(l, pos); !e.empty()) return e;
            if (dbg) dbg_x.push_back(x_);                       // test hook: x after every layer
        }
        if (want_logits) {
            LayerWeights& w = lw_[(size_t) (g_.n_layers - 1)];
            if (!w.head_ok) {
                if (const std::string e = load_f32("output_norm.weight", w.head); !e.empty()) return e;
                w.head_ok = true;
            }
            rmsnorm_training(x_.data(), w.head.data.data(), (int) g_.n_embd, (float) g_.rms_eps, xn_.data());
            if (const std::string e = head_logits(xn_.data()); !e.empty()) return e;
        }
        tok_.push_back(token);
        return {};
    }

    const std::vector<float>& logits() const { return logits_; }
    /// The last DSA row's selected key positions, per layer (empty for a window layer; ascending;
    /// when the layer saw <= index_top_k rows the selection is the arange NaiveRT's width rule makes
    /// it).  Exposed for the runtime test, which compares it against fixtures.
    const std::vector<int32_t>& selection_at(int64_t l) const { return sel_[(size_t) l]; }
    int dbg = 0;                                                // record x after every layer (test
    std::vector<std::vector<float>> dbg_x;                      // hook; off; 2 adds the probes
    std::vector<std::pair<std::string, std::vector<float>>>     // named probes inside layer_row,
        dbg_pts;                                                // recorded only when dbg == 2
    /// The last row's router pick, per MoE layer (empty for dense/attention layers): a test hook
    /// in the same spirit as selection_at - it changes nothing in the forward.
    const std::vector<int32_t>& expert_at(int64_t l) const { return moe_[(size_t) l]; }

private:
    struct LayerWeights {
        LoadedTensor attn_norm, ffn_norm, sink, ikn_w, ikn_b, gate, bias, head;
        bool attn_ok = false, ffn_ok = false, dsa_ok = false, moe_ok = false, swa_ok = false, head_ok = false;
    };

    // One layer, one row at `pos`: attention with its cache write, then the FFN.  x_ holds the
    // residual stream, exactly as the tested trunk wires them.
    std::string layer_row(int64_t l, int64_t pos) {
        const bool dsa = g_.is_dsa_layer(l);
        const int64_t nkv = g_.kv_heads(l);
        const int64_t group = g_.n_head / nkv;
        const double theta = dsa ? g_.rope_theta_full : g_.rope_theta_swa;
        const int64_t n_rot = dsa ? g_.n_rot : g_.n_rot_swa;
        char nm[96];
        LayerWeights& w = lw_[(size_t) l];
        if (!w.attn_ok) {
            std::snprintf(nm, sizeof nm, "blk.%lld.attn_norm.weight", (long long) l);
            if (auto e = load_f32(nm, w.attn_norm); !e.empty()) return e;
            if (dsa) {
                std::snprintf(nm, sizeof nm, "blk.%lld.indexer.k_norm.weight", (long long) l);
                if (auto e = load_f32(nm, w.ikn_w); !e.empty()) return e;
                std::snprintf(nm, sizeof nm, "blk.%lld.indexer.k_norm.bias", (long long) l);
                if (auto e = load_f32(nm, w.ikn_b); !e.empty()) return e;
            } else if (g_.add_swa_sink) {
                std::snprintf(nm, sizeof nm, "blk.%lld.attn_sinks.weight", (long long) l);
                if (auto e = load_f32(nm, w.sink); !e.empty()) return e;
            }
            w.attn_ok = true;
        }
        rmsnorm_training(x_.data(), w.attn_norm.data.data(), (int) g_.n_embd, (float) g_.rms_eps, xn_.data());
        auto probe = [this](const char* name, const float* p, size_t n) {   // debug-only, dbg == 2
            if (dbg == 2) dbg_pts.emplace_back(name, std::vector<float>(p, p + n));
        };
        probe("xn", xn_.data(), (size_t) g_.n_embd);

        // K/V: k bf16 after the projection, rope at THIS row's position, bf16 again into the cache;
        // v with the value scale folded before the single bf16 round (the tested dsa/trunk wiring).
        std::snprintf(nm, sizeof nm, "blk.%lld.attn_k.weight", (long long) l);
        if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, nkv * g_.head_dim_k, krow_.data()); !e.empty()) return e;
        {   // rope in place on the bf16-rounded row, then round again (the reference's double round)
            std::vector<float> tmp((size_t) nkv * g_.head_dim_k);
            rope_rows(krow_.data(), (int) nkv, (int) g_.head_dim_k, (int) n_rot, pos, theta, tmp.data());
            for (float& f : tmp) f = bf16_round(f);
            probe("k", tmp.data(), tmp.size());
            if (dsa) {
                uint16_t* dst = grow_u16(dsa_k_[(size_t) l], (size_t) nkv * (size_t) g_.head_dim_k);
                for (size_t i = 0; i < tmp.size(); ++i) dst[i] = bf16_bits(tmp[i]);
            } else {
                put_ring_u16(swa_k_[(size_t) l], ring_cap_[(size_t) l], pos, tmp.data(), (size_t) nkv * (size_t) g_.head_dim_k);
            }
        }
        std::snprintf(nm, sizeof nm, "blk.%lld.attn_v.weight", (long long) l);
        if (auto e = gemv_raw(nm, xn_.data(), g_.n_embd, nkv * g_.head_dim_v, vrow_.data()); !e.empty()) return e;
        {
            for (int64_t i = 0; i < nkv * g_.head_dim_v; ++i)
                vrow_[(size_t) i] = bf16_round((double) vrow_[(size_t) i] * g_.value_scale);
            probe("v", vrow_.data(), (size_t) nkv * (size_t) g_.head_dim_v);
            if (dsa) {
                uint16_t* dst = grow_u16(dsa_v_[(size_t) l], (size_t) nkv * (size_t) g_.head_dim_v);
                for (size_t i = 0; i < (size_t) nkv * (size_t) g_.head_dim_v; ++i)
                    dst[i] = bf16_bits(vrow_[(size_t) i]);
            } else {
                put_ring_u16(swa_v_[(size_t) l], ring_cap_[(size_t) l], pos, vrow_.data(),
                             (size_t) nkv * (size_t) g_.head_dim_v);
            }
        }

        // The DSA indexer's key: bf16 -> LayerNorm -> bf16 -> rope -> bf16 -> the E4M3 row, stored
        // dequantised (run_attn_dsa's four rounds before the row quant).
        if (dsa) {
            const int64_t D = g_.index_head_dim;
            std::snprintf(nm, sizeof nm, "blk.%lld.indexer.k_proj.weight", (long long) l);
            std::vector<float> ik((size_t) D);
            if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, D, ik.data()); !e.empty()) return e;
            std::vector<float> ln((size_t) D), rope((size_t) D);
            index_key_layernorm(ik.data(), w.ikn_w.data.data(), w.ikn_b.data.data(), (int) D,
                                (float) g_.rms_eps, ln.data());
            for (float& f : ln) f = bf16_round(f);
            rope_rows(ln.data(), 1, (int) D, (int) std::min<int64_t>(n_rot, D), pos, theta, rope.data());
            for (float& f : rope) f = bf16_round(f);
            probe("ik", rope.data(), (size_t) D);
            std::vector<float> deq((size_t) D);
            fp8_rows_dequant(rope.data(), 1, (int) D, deq.data(), nullptr);
            probe("ikdeq", deq.data(), (size_t) D);
            float* dst = grow_f32(dsa_ki_[(size_t) l], (size_t) D);
            std::memcpy(dst, deq.data(), (size_t) D * sizeof(float));
        }

        // Q: projection bf16, rope at this position, bf16 into attention.
        std::snprintf(nm, sizeof nm, "blk.%lld.attn_q.weight", (long long) l);
        if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, g_.n_head * g_.head_dim_k, qrow_.data()); !e.empty())
            return e;
        rope_rows(qrow_.data(), (int) g_.n_head, (int) g_.head_dim_k, (int) n_rot, pos, theta, qro_.data());
        for (float& f : qro_) f = bf16_round(f);
        probe("q", qro_.data(), qro_.size());

        // Which keys this row reads, and the attention over them.
        std::vector<float> attn((size_t) (g_.n_head * g_.head_dim_v), 0.f);
        if (dsa) {
            // The selection: index logits over every stored row of this layer, then the stable
            // top-k (width max(visible, top_k): at or below top_k every row is visible, ascending -
            // the arange rule the tested dsa case documents, and the rule NaiveRT's identity
            // selection implements in hardware: below 2048 visible it skips the index-query
            // projection and the weights projection ENTIRELY and publishes selections[q][t] = t
            // (attn_dsa_body.cuh:359-362,990-1005; run_attn_dsa.py:310-313 arange).  Same winners
            // either way; skipping only saves the work.
            const int vis = (int) pos + 1;
            if (vis > (int) g_.index_top_k) {
                sel_logits_.assign((size_t) vis, 0.f);
                std::snprintf(nm, sizeof nm, "blk.%lld.indexer.proj.weight", (long long) l);
                std::vector<float> wraw((size_t) g_.index_n_heads);
                if (auto e = gemv_raw(nm, xn_.data(), g_.n_embd, g_.index_n_heads, wraw.data()); !e.empty())
                    return e;
                const double fold = 1.0 / std::sqrt((double) g_.index_n_heads);
                std::vector<float> wf((size_t) g_.index_n_heads);
                for (int64_t h = 0; h < g_.index_n_heads; ++h)
                    wf[(size_t) h] = (float) ((double) wraw[(size_t) h] * fold);
                // The query rows: bf16 after the projection, rope, bf16, then the E4M3 row quant
                // (the tested dsa_chain's iq path).
                std::snprintf(nm, sizeof nm, "blk.%lld.indexer.q_proj.weight", (long long) l);
                const int64_t IH = g_.index_n_heads, D = g_.index_head_dim;
                std::vector<float> iq((size_t) (IH * D)), iqr((size_t) (IH * D)), iqdeq((size_t) (IH * D));
                if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, IH * D, iq.data()); !e.empty()) return e;
                rope_rows(iq.data(), (int) IH, (int) D, (int) std::min<int64_t>(n_rot, D), pos, theta, iqr.data());
                for (float& f : iqr) f = bf16_round(f);
                fp8_rows_dequant(iqr.data(), (int) IH, (int) D, iqdeq.data(), nullptr);
                dsa_logits_par(iqdeq.data(), (int) IH, (int) D, wf.data(), dsa_ki_[(size_t) l].data(), vis,
                               sel_logits_.data());
                stable_topk_fast(sel_logits_.data(), vis, (int) g_.index_top_k, sel_[(size_t) l]);
            } else {
                sel_[(size_t) l].resize((size_t) vis);
                for (int i = 0; i < vis; ++i) sel_[(size_t) l][(size_t) i] = i;
            }
            attn_over_rows(l, sel_[(size_t) l], qro_.data(), attn.data(), nullptr);
        } else {
            // The window: [p-window+1, p], every row still in the ring (adapters.py's rule).
            const int64_t first = std::max<int64_t>(0, pos - g_.swa_window + 1);
            std::vector<int32_t> rows((size_t) (pos - first + 1));
            for (int64_t j = first; j <= pos; ++j) rows[(size_t) (j - first)] = (int32_t) j;
            sel_[(size_t) l].clear();                       // a window layer selects nothing
            attn_over_rows(l, rows, qro_.data(), attn.data(),
                           g_.add_swa_sink ? w.sink.data.data() : nullptr);
        }
        for (float& f : attn) f = bf16_round(f);
        probe("attn", attn.data(), attn.size());
        std::snprintf(nm, sizeof nm, "blk.%lld.attn_output.weight", (long long) l);
        std::vector<float> delta((size_t) g_.n_embd);
        if (auto e = gemv_bf16(nm, attn.data(), g_.n_head * g_.head_dim_v, g_.n_embd, delta.data()); !e.empty())
            return e;
        probe("o", delta.data(), delta.size());
        for (int64_t i = 0; i < g_.n_embd; ++i) x_[(size_t) i] = bf16_round((double) x_[(size_t) i] + (double) delta[(size_t) i]);
        probe("xa", x_.data(), (size_t) g_.n_embd);

        // ---- the FFN: the dense layer's SwiGLU, or the router + IQ2 experts.
        if (!w.ffn_ok) {
            std::snprintf(nm, sizeof nm, "blk.%lld.ffn_norm.weight", (long long) l);
            if (auto e = load_f32(nm, w.ffn_norm); !e.empty()) return e;
            if (g_.is_moe_layer(l)) {
                std::snprintf(nm, sizeof nm, "blk.%lld.ffn_gate_inp.weight", (long long) l);
                if (auto e = load_f32(nm, w.gate); !e.empty()) return e;
                std::snprintf(nm, sizeof nm, "blk.%lld.exp_probs_b.bias", (long long) l);
                if (auto e = load_f32(nm, w.bias); !e.empty()) return e;
            }
            w.ffn_ok = true;
        }
        rmsnorm_training(x_.data(), w.ffn_norm.data.data(), (int) g_.n_embd, (float) g_.rms_eps, xn_.data());
        probe("h", xn_.data(), (size_t) g_.n_embd);
        if (!g_.is_moe_layer(l)) {
            const int64_t n_ff = g_.ffn_length[(size_t) l];
            std::vector<float> gate((size_t) n_ff), up((size_t) n_ff), act((size_t) n_ff);
            std::snprintf(nm, sizeof nm, "blk.%lld.ffn_gate.weight", (long long) l);
            if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, n_ff, gate.data()); !e.empty()) return e;
            std::snprintf(nm, sizeof nm, "blk.%lld.ffn_up.weight", (long long) l);
            if (auto e = gemv_bf16(nm, xn_.data(), g_.n_embd, n_ff, up.data()); !e.empty()) return e;
            for (int64_t i = 0; i < n_ff; ++i) act[(size_t) i] = bf16_round(silu(gate[(size_t) i]) * up[(size_t) i]);
            probe("act", act.data(), act.size());
            std::snprintf(nm, sizeof nm, "blk.%lld.ffn_down.weight", (long long) l);
            if (auto e = gemv_bf16(nm, act.data(), n_ff, g_.n_embd, delta.data()); !e.empty()) return e;
            probe("dffn", delta.data(), delta.size());
        } else {
            std::vector<int> idx((size_t) g_.n_expert_used);
            std::vector<float> wts((size_t) g_.n_expert_used);
            moe_route(xn_.data(), w.gate.data.data(), w.bias.data.data(), (int) g_.n_embd,
                      (int) g_.n_expert, (int) g_.n_expert_used, idx.data(), wts.data());
            moe_[l].assign(idx.begin(), idx.end());             // the row's expert ids, for the test
            const int64_t FE = g_.n_ff_expert;
            std::vector<float> ex_out((size_t) g_.n_expert_used * g_.n_embd);
            for (int t = 0; t < (int) g_.n_expert_used; ++t) {
                float* eo = ex_out.data() + (size_t) t * g_.n_embd;
                std::vector<float> gate((size_t) FE), up((size_t) FE), act((size_t) FE);
                std::snprintf(nm, sizeof nm, "blk.%lld.ffn_gate_exps.weight", (long long) l);
                if (auto e = expert_gemv_bf16(nm, idx[(size_t) t], xn_.data(), g_.n_embd, FE, gate.data()); !e.empty())
                    return e;
                std::snprintf(nm, sizeof nm, "blk.%lld.ffn_up_exps.weight", (long long) l);
                if (auto e = expert_gemv_bf16(nm, idx[(size_t) t], xn_.data(), g_.n_embd, FE, up.data()); !e.empty())
                    return e;
                for (int64_t i = 0; i < FE; ++i) act[(size_t) i] = bf16_round(silu(gate[(size_t) i]) * up[(size_t) i]);
                std::snprintf(nm, sizeof nm, "blk.%lld.ffn_down_exps.weight", (long long) l);
                if (auto e = expert_gemv_bf16(nm, idx[(size_t) t], act.data(), FE, g_.n_embd, eo); !e.empty())
                    return e;
            }
            std::vector<float> raw((size_t) g_.n_embd);
            moe_combine(ex_out.data(), wts.data(), (int) g_.n_expert_used, (int) g_.n_embd, raw.data());
            for (int64_t i = 0; i < g_.n_embd; ++i) delta[(size_t) i] = bf16_round(raw[(size_t) i]);
        }
        for (int64_t i = 0; i < g_.n_embd; ++i) x_[(size_t) i] = bf16_round((double) x_[(size_t) i] + (double) delta[(size_t) i]);
        return {};
    }

    /// Attention over `rows` (ascending cache positions) for one row: gather the kv group, run
    //  kernels.hpp::swa_attention (the sink only when given), 64/group heads per kv head - the same
    ///  grouped view the tested dsa/trunk cases build.
    void attn_over_rows(int64_t l, const std::vector<int32_t>& rows, const float* q, float* out,
                        const float* sink) {
        const int64_t nkv = g_.kv_heads(l);
        const int64_t group = g_.n_head / nkv;
        const int n = (int) rows.size();
        const int32_t* rowp = rows.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::min(threads_, (int) nkv))
#endif
        for (int64_t g = 0; g < nkv; ++g) {
            std::vector<float> kg((size_t) n * g_.head_dim_k), vg((size_t) n * g_.head_dim_v);
            for (int j = 0; j < n; ++j) {
                gather_kv(l, rowp[j], g, kg.data() + (size_t) j * g_.head_dim_k,
                          vg.data() + (size_t) j * g_.head_dim_v);
            }
            swa_attention(q + (size_t) g * group * g_.head_dim_k, (int) group, (int) g_.head_dim_k,
                          kg.data(), vg.data(), (int) g_.head_dim_v,
                          sink ? sink + g * group : nullptr, 0, n - 1,
                          out + (size_t) g * group * g_.head_dim_v);
        }
    }

    /// One cache row of one kv head, de-coded to float for the attention's scratch.
    void gather_kv(int64_t l, int64_t pos, int64_t kv, float* k, float* v) const {
        const int64_t K = g_.head_dim_k, V = g_.head_dim_v;
        if (g_.is_dsa_layer(l)) {
            const uint16_t* kr = dsa_k_[(size_t) l].data() + (size_t) pos * (size_t) g_.kv_heads(l) * K + (size_t) kv * K;
            const uint16_t* vr = dsa_v_[(size_t) l].data() + (size_t) pos * (size_t) g_.kv_heads(l) * V + (size_t) kv * V;
            for (int64_t i = 0; i < K; ++i) k[i] = bf16_val(kr[i]);
            for (int64_t i = 0; i < V; ++i) v[i] = bf16_val(vr[i]);
        } else {
            const uint64_t cap = ring_cap_[(size_t) l];
            const uint64_t slot = (uint64_t) pos % cap;
            const uint16_t* kr = swa_k_[(size_t) l].data() + slot * (size_t) g_.kv_heads(l) * K + (size_t) kv * K;
            const uint16_t* vr = swa_v_[(size_t) l].data() + slot * (size_t) g_.kv_heads(l) * V + (size_t) kv * V;
            for (int64_t i = 0; i < K; ++i) k[i] = bf16_val(kr[i]);
            for (int64_t i = 0; i < V; ++i) v[i] = bf16_val(vr[i]);
        }
    }

    /// The DSA selection logits, the serial kernels.hpp formula per position, rows split over
    /// threads (every output element independent: the same bits single- or multi-threaded).
    void dsa_logits_par(const float* iq, int H, int D, const float* w, const float* ki_deq, int vis,
                        float* out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
        for (int j = 0; j < vis; ++j) {
            const float* kj = ki_deq + (size_t) j * D;
            double acc = 0.0;
            for (int h = 0; h < H; ++h) {
                const float* qh = iq + (size_t) h * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qh[d] * kj[d];
                if (dot > 0.0f) acc += (double) dot * w[h];
            }
            out[j] = (float) acc;
        }
    }

    // ---- the GEMV trio: the same per-row decode and double dot as loader.hpp, rows split over
    // threads.  A row's sum is never split, so the numbers do not move with the thread count.
    // With a pool (runtime's optional ctor argument) a resident weight is read from the pool's copy
    // first - fp32 for a dense tensor, native bytes for a raw one - and the mmap windows only answer
    // for what the pool does not hold; the arithmetic is the same loop either way (pool.hpp).
    std::string gemv_bf16(const char* name, const float* x, int64_t n_in, int64_t n_out, float* y) {
        std::string e = gemv_raw(name, x, n_in, n_out, y);
        if (e.empty()) for (int64_t i = 0; i < n_out; ++i) y[i] = bf16_round(y[i]);
        return e;
    }
    std::string gemv_raw(const char* name, const float* x, int64_t n_in, int64_t n_out, float* y) {
        if (pool_) {
            if (pool_->gemv_f32(name, x, n_in, n_out, y, threads_)) return {};
            if (pool_->gemv_raw(name, x, n_in, n_out, y, threads_)) return {};
        }
        size_t shard = 0;
        const TensorInfo* t = m_.find(name, &shard);
        if (!t) return std::string("the model has no ") + name;
        if (t->shape.size() != 2 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != n_out)
            return std::string(name) + " is not {" + std::to_string(n_in) + "," + std::to_string(n_out) + "}";
        const uint32_t ty = t->type;
        if (ggml_payload_bytes(ty, t->elements()) == 0)
            return std::string(name) + ": unknown or ragged ggml type " + t->type_name();
        if (n_in <= 0 || n_in % (int64_t) ggml_blck_size(ggml_of(ty)) != 0)
            return std::string(name) + ": a row of " + std::to_string(n_in) + " elements is ragged for " + t->type_name();
        if (!m_.in_bounds(*t, shard)) return std::string(name) + ": payload lies outside its shard";
        const uint8_t* base = m_.shard(shard).tensor_data(*t);
        int rc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(| : rc)
#endif
        for (int64_t mi = 0; mi < n_out; ++mi) {
#ifdef _OPENMP
            thread_local
#endif
            std::vector<float> row;
            if ((int64_t) row.size() != n_in) row.resize((size_t) n_in);
            const uint8_t* src = base + ggml_payload_bytes(ty, (uint64_t) mi * (uint64_t) n_in);
            if (dequant_to_float(ty, src, row.data(), n_in)) { rc = 1; continue; }
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
            y[mi] = (float) acc;
        }
        if (rc) return std::string(name) + ": no CPU dequantizer for " + t->type_name();
        return {};
    }
    std::string expert_gemv_bf16(const char* name, int expert, const float* x, int64_t n_in,
                                 int64_t n_out, float* y) {
        std::string e = expert_gemv_raw(name, expert, x, n_in, n_out, y);
        if (e.empty()) for (int64_t i = 0; i < n_out; ++i) y[i] = bf16_round(y[i]);
        return e;
    }
    std::string expert_gemv_raw(const char* name, int expert, const float* x, int64_t n_in,
                                int64_t n_out, float* y) {
        // The GPU tier first for a stack the plan placed on a card: a card that holds the stack
        // OWNS its rows for this engine, so a GPU-side failure is the request's error (the same
        // "never answer wrong" rule as the serve loop), never a silent re-read of the CPU copy.
        if (gpu_ && gpu_->ok()) {
            const int h = gpu_->handle_of(name);
            if (h >= 0) return gpu_->gemv_expert(h, (uint64_t) expert, x, n_in, y, n_out);
        }
        if (pool_ && pool_->gemv_expert_raw(name, (uint64_t) expert, x, n_in, n_out, y, threads_))
            return {};
        size_t shard = 0;
        const TensorInfo* t = m_.find(name, &shard);
        if (!t) return std::string("the model has no ") + name;
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != n_out)
            return std::string(name) + " is not an expert stack with n_in " + std::to_string(n_in);
        if (expert < 0 || (uint64_t) expert >= t->shape[2])
            return std::string(name) + ": expert " + std::to_string(expert) + " outside the stack";
        const uint32_t ty = t->type;
        if (ggml_payload_bytes(ty, t->elements()) == 0)
            return std::string(name) + ": unknown ggml type " + t->type_name();
        if (n_in <= 0 || n_in % (int64_t) ggml_blck_size(ggml_of(ty)) != 0)
            return std::string(name) + ": a row of " + std::to_string(n_in) + " is ragged for " + t->type_name();
        if (!m_.in_bounds(*t, shard)) return std::string(name) + ": payload lies outside its shard";
        const uint64_t slice = (uint64_t) n_in * (uint64_t) n_out;
        const uint8_t* slice0 = m_.shard(shard).tensor_data(*t) + ggml_payload_bytes(ty, (uint64_t) expert * slice);
        int rc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_) reduction(| : rc)
#endif
        for (int64_t mi = 0; mi < n_out; ++mi) {
#ifdef _OPENMP
            thread_local
#endif
            std::vector<float> row;
            if ((int64_t) row.size() != n_in) row.resize((size_t) n_in);
            const uint8_t* src = slice0 + ggml_payload_bytes(ty, (uint64_t) mi * (uint64_t) n_in);
            if (dequant_to_float(ty, src, row.data(), n_in)) { rc = 1; continue; }
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
            y[mi] = (float) acc;
        }
        if (rc) return std::string(name) + ": no CPU dequantizer for " + t->type_name();
        return {};
    }

    /// The output head: y over the {n_embd, vocab} table - a vocab row is one COLUMN of the GGUF
    /// order, so this is gemv_raw's row loop straight over the table with bf16 rounding, the same
    /// column windows the tested trunk's chunked scan reads.
    std::string head_logits(const float* x) {
        return gemv_bf16("output.weight", x, g_.n_embd, g_.n_vocab, logits_.data());
    }

    // ---- the two loader reads, pool-first.  load_f32 answers from the pool's fp32 copy (the maps
    // are stable after load, pool.hpp); load_cols from the pool's copy of the table; both fall back
    // to loader.hpp's mmap windows for a name the pool does not hold.
    std::string load_f32(const char* name, LoadedTensor& out) {
        if (pool_)
            if (const LoadedTensor* p = pool_->fp32(name)) {
                out = *p;
                return {};
            }
        return load_tensor_f32(m_, name, out);
    }
    std::string load_cols(const char* name, uint64_t col_start, uint64_t cols, LoadedTensor& out) {
        if (pool_ && pool_->resident(name)) {
            size_t shard = 0;
            const TensorInfo* t = m_.find(name, &shard);
            if (!t) return std::string("the model has no ") + name;
            out.shape = {t->shape[0], cols};
            out.type = t->type;
            out.data.resize((size_t) (t->shape[0] * cols));
            if (pool_->cols(name, col_start, cols, out.data.data())) return {};
            return std::string(name) + ": the resident copy could not answer this column window";
        }
        return load_tensor_cols(m_, name, col_start, cols, out);
    }

    // ---- cache row helpers
    uint16_t* grow_u16(std::vector<uint16_t>& v, size_t row) {
        v.resize(v.size() + row);
        return v.data() + v.size() - row;
    }
    float* grow_f32(std::vector<float>& v, size_t n) {
        v.resize(v.size() + n);
        return v.data() + v.size() - n;
    }
    void put_ring_u16(std::vector<uint16_t>& v, uint64_t cap, int64_t pos, const float* src, size_t n) {
        if (v.size() < (size_t) cap * n) v.assign((size_t) cap * n, 0);
        uint16_t* dst = v.data() + ((uint64_t) pos % cap) * n;
        for (size_t i = 0; i < n; ++i) dst[i] = bf16_bits(src[i]);
    }

    const GgufModel& m_;
    NaiveGeometry g_;
    int threads_;
    const NaivePool* pool_ = nullptr;  // optional: the resident weights (pool.hpp)
    NaiveGpu* gpu_ = nullptr;          // optional: the card tier over the pool's expert stacks
    bool ok_ = false;
    std::vector<int32_t> tok_;                              // the session, position = index
    std::vector<uint64_t> ring_cap_;
    std::vector<std::vector<uint16_t>> dsa_k_, dsa_v_;      // DSA: every position, bf16 codes
    std::vector<std::vector<float>> dsa_ki_;                // DSA: E4M3 rows, dequantised
    std::vector<std::vector<uint16_t>> swa_k_, swa_v_;      // window: the rings
    std::vector<LayerWeights> lw_;
    std::vector<std::vector<int32_t>> sel_;                 // the last row's selection, per layer
    std::vector<std::vector<int32_t>> moe_;                 // the last row's expert ids, per layer
    // Row scratch, reused across layers.
    std::vector<float> x_, xn_, qrow_, qro_, krow_, vrow_, logits_, sel_logits_;
};

}  // namespace strata::naive
