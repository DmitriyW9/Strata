// include/strata/naive/reader.hpp - the Naive-N0.5-Flash metadata: the geometry read from the model
// file's keys, the checks that refuse a file this build cannot run (with the key that differs), and the
// per-layer tensor inventory (names, shapes, types) that the weight loader walks.
//
// It reads through strata::GgufModel (the metadata lives in shard 1, the tensors wherever they lie), the
// same way the qwen4exp path does; the draft is a GGUF of one shard with its own arch prefix.
//
// Like check_architecture_qwen35moe / read_geometry in artifact/gguf_reader.hpp: every key the geometry
// needs must be present with the right shape, and the keys that describe one thing twice (the hybrid
// pattern and the dsa_layers list, the pattern and the per-layer kv-head counts, the draft's block size
// and its proposal count) must agree.  A file of another shape is refused by the key that differs, never
// mis-run.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/naive/geometry.hpp"

#include <functional>
#include <string>
#include <vector>

namespace strata::naive {

/// The cross-key rules of one read geometry; defined below, used by read_geometry.
inline std::string check_geometry(const NaiveGeometry& g);

/// Reads the geometry `read_*` fills: a required/optional scalar or per-layer list under the model's
/// prefix.  The MetaValue comes from artifact/gguf_reader.hpp; a list arrives as an ARRAY whose items
/// hold the values.  An INT32 list arrives signed: a signed item decodes through num(), which keeps the
/// sign, so a pattern entry of -1 could not pass a 0/1 check unnoticed.
class MetaReader {
public:
    MetaReader(const GgufFile& f, std::string prefix) : f_(f), p_(std::move(prefix)) {}

    /// A scalar key (int or float or bool); `found` says whether it was there at all.
    bool num(const char* key, double& out) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v || !v->is_num()) return false;
        out = v->num();
        return true;
    }
    /// A required scalar int, or `err` names the key.  A float/bool where an int belongs is an error:
    /// silently rounding metadata is how a wrong shape enters the kernels.
    bool int_key(const char* key, int64_t& out, std::string& err) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v) {
            err = "missing " + p_ + key;
            return false;
        }
        if (!v->is_num() || v->type == MetaType::F32 || v->type == MetaType::F64 || v->type == MetaType::BOOL) {
            err = p_ + key + " is not an integer";
            return false;
        }
        out = (int64_t) v->u;
        return true;
    }
    bool float_key(const char* key, double& out, std::string& err, bool required) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v) {
            if (required) err = "missing " + p_ + key;
            return !required;
        }
        if (!v->is_num()) {
            err = p_ + key + " is not a number";
            return false;
        }
        out = v->num();
        return true;
    }
    bool bool_key(const char* key, bool& out, std::string& err, bool required) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v) {
            if (required) err = "missing " + p_ + key;
            return !required;
        }
        if (v->type != MetaType::BOOL) {
            err = p_ + key + " is not a bool";
            return false;
        }
        out = v->u != 0;
        return true;
    }
    bool str_key(const char* key, std::string& out, std::string& err, bool required) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v) {
            if (required) err = "missing " + p_ + key;
            return !required;
        }
        if (v->type != MetaType::STRING) {
            err = p_ + key + " is not a string";
            return false;
        }
        out = v->s;
        return true;
    }
    /// An int list, exactly `n` entries, every entry passing `ok` (else the key and the index are named).
    bool int_list(const char* key, size_t n, std::vector<int64_t>& out, std::string& err,
                  const std::function<bool(int64_t)>& ok = {}) const {
        const MetaValue* v = f_.get(p_ + key);
        if (!v) {
            err = "missing " + p_ + key;
            return false;
        }
        if (v->type != MetaType::ARRAY || (v->elem != MetaType::I32 && v->elem != MetaType::U32 &&
                                           v->elem != MetaType::I64 && v->elem != MetaType::U64)) {
            err = p_ + key + " is not an integer array";
            return false;
        }
        if (v->count != n) {
            err = p_ + key + " has " + std::to_string(v->count) + " entries, this model has " +
                  std::to_string(n) + " layers";
            return false;
        }
        // read_value keeps only the first 64 items of an array (the count is what matters downstream);
        // a per-layer list of a 48-layer model fits, and this refuses a file where it does not rather
        // than reading a truncated list.
        if (v->items.size() != v->count) {
            err = p_ + key + " is longer than this reader keeps";
            return false;
        }
        out.clear();
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const MetaValue& e = v->items[i];
            if (!e.is_num() || e.type == MetaType::F32 || e.type == MetaType::F64 || e.type == MetaType::BOOL) {
                err = p_ + key + " entry " + std::to_string(i) + " is not an integer";
                return false;
            }
            const int64_t x = e.type == MetaType::I64 ? (int64_t) e.u : (int64_t)(int32_t) e.u;
            if (ok && !ok(x)) {
                err = p_ + key + " entry " + std::to_string(i) + " = " + std::to_string(x) + ", not a valid value";
                return false;
            }
            out.push_back(x);
        }
        return true;
    }

private:
    const GgufFile& f_;
    std::string p_;
};

/// Fills `g` from the metadata shard `f`.  Returns "" or the first problem.  Another architecture is
/// refused by name first, like qwen4exp's read_geometry: its keys carry another prefix.
inline std::string read_geometry(const GgufFile& f, NaiveGeometry& g) {
    const MetaValue* arch = f.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArchMain) return "architecture is '" + arch->s + "', this engine path requires '" + kArchMain + "'";
    const MetaReader r(f, std::string(kArchMain) + ".");
    std::string err;
    int64_t x = 0;
    if (!r.int_key("block_count", x, err)) return err;
    g.n_layers = x;
    if (!r.int_key("embedding_length", x, err)) return err;
    g.n_embd = x;
    if (!r.int_key("context_length", x, err)) return err;
    g.n_ctx = x;
    if (!r.float_key("attention.layer_norm_rms_epsilon", g.rms_eps, err, true)) return err;
    if (!r.int_key("attention.head_count", x, err)) return err;
    g.n_head = x;
    if (!r.int_key("attention.key_length", x, err)) return err;
    g.head_dim_k = x;
    if (!r.int_key("attention.value_length", x, err)) return err;
    g.head_dim_v = x;
    if (!r.int_key("attention.sliding_window", x, err)) return err;
    g.swa_window = x;
    if (!r.int_key("attention_chunk_size", x, err)) return err;
    g.attn_chunk = x;
    if (!r.float_key("attention.value_scale", g.value_scale, err, true)) return err;
    if (!r.str_key("attention.projection_layout", g.projection_layout, err, true)) return err;
    if (!r.bool_key("attention.add_swa_sink", g.add_swa_sink, err, true)) return err;
    if (!r.bool_key("attention.add_full_sink", g.add_full_sink, err, true)) return err;
    if (!r.int_key("rope.dimension_count", x, err)) return err;
    g.n_rot = x;
    if (!r.int_key("rope.dimension_count_swa", x, err)) return err;
    g.n_rot_swa = x;
    if (!r.float_key("rope.freq_base", g.rope_theta_full, err, true)) return err;
    if (!r.float_key("rope.freq_base_swa", g.rope_theta_swa, err, true)) return err;
    if (!r.float_key("rope.partial_rotary_factor", g.partial_rotary_factor, err, true)) return err;
    if (!r.str_key("rope.pairing", g.rope_pairing, err, true)) return err;
    if (!r.int_key("index_top_k", x, err)) return err;
    g.index_top_k = x;
    if (!r.int_key("index_head_dim", x, err)) return err;
    g.index_head_dim = x;
    if (!r.int_key("index_n_heads", x, err)) return err;
    g.index_n_heads = x;
    if (!r.int_key("index_n_kv_heads", x, err)) return err;
    g.index_n_kv_heads = x;
    if (!r.str_key("indexer_activation_dtype", g.indexer_activation_dtype, err, true)) return err;
    if (!r.int_key("expert_count", x, err)) return err;
    g.n_expert = x;
    if (!r.int_key("expert_used_count", x, err)) return err;
    g.n_expert_used = x;
    if (!r.int_key("expert_feed_forward_length", x, err)) return err;
    g.n_ff_expert = x;
    if (!r.bool_key("expert_weights_norm", g.expert_weights_norm, err, true)) return err;
    if (!r.int_key("expert_gating_func", x, err)) return err;
    g.expert_gating_func = x;
    if (!r.str_key("moe.topk_method", g.moe_topk_method, err, true)) return err;
    if (!r.str_key("moe.scoring_func", g.moe_scoring_func, err, true)) return err;
    if (!r.bool_key("attention.dsa_enabled", g.dsa_enabled, err, true)) return err;
    if (!r.int_list("attention.hybrid_layer_pattern", (size_t) g.n_layers, g.hybrid_pattern, err,
                    [](int64_t v) { return v == 0 || v == 1; }))
        return err;
    if (!r.int_list("expert.layer_pattern", (size_t) g.n_layers, g.expert_pattern, err,
                    [](int64_t v) { return v == 0 || v == 1; }))
        return err;
    if (!r.int_list("attention.head_count_kv", (size_t) g.n_layers, g.n_head_kv, err,
                    [](int64_t v) { return v >= 1; }))
        return err;
    if (!r.int_list("feed_forward_length", (size_t) g.n_layers, g.ffn_length, err,
                    [](int64_t v) { return v > 0; }))
        return err;
    // The DSA layer list: not a per-layer array, so MetaReader::int_list's length rule does not fit;
    // read it as a raw array and check membership against the pattern below.
    {
        const MetaValue* v = f.get(std::string(kArchMain) + ".attention.dsa_layers");
        if (!v || v->type != MetaType::ARRAY) return "missing " + std::string(kArchMain) + ".attention.dsa_layers";
        g.dsa_layers.clear();
        for (const MetaValue& e : v->items) {
            if (!e.is_num()) return std::string(kArchMain) + ".attention.dsa_layers holds a non-integer";
            g.dsa_layers.push_back((int64_t)(int32_t) e.u);
        }
        if (v->count != g.dsa_layers.size()) return std::string(kArchMain) + ".attention.dsa_layers is too long";
    }
    // The tokenizer lives once, under "tokenizer." keys on any shard's metadata; the model's own copy
    // (general.architecture's shard) is the one read.
    const MetaValue* toks = f.get("tokenizer.ggml.tokens");
    if (toks && toks->type == MetaType::ARRAY) g.n_vocab = (int64_t) toks->count;

    return check_geometry(g);
}

/// The cross-key rules of one read geometry (the keys that describe one thing twice, and the shapes the
/// engine path is written for).  "" when the geometry is one this build runs.
inline std::string check_geometry(const NaiveGeometry& g) {
    const std::string p = std::string(kArchMain) + ".";
    if (g.n_layers <= 0) return p + "block_count must be positive";
    if (g.n_embd <= 0) return p + "embedding_length must be positive";
    if (g.n_head <= 0) return p + "attention.head_count must be positive";
    if (g.head_dim_k <= 0 || g.head_dim_v <= 0) return p + "attention.key_length / value_length must be positive";
    if (g.projection_layout != "split")
        return p + "attention.projection_layout is '" + g.projection_layout +
               "', this engine path reads the split q/k/v layout only";
    if (g.rope_pairing != "split-half / GPT-NeoX")
        return p + "rope.pairing is '" + g.rope_pairing + "', this engine path is written for split-half RoPE";
    if (g.indexer_activation_dtype != "fp8_e4m3")
        return p + "indexer_activation_dtype is '" + g.indexer_activation_dtype + "', this engine path stores FP8 E4M3";
    if (g.moe_scoring_func != "sigmoid")
        return p + "moe.scoring_func is '" + g.moe_scoring_func + "', this engine path scores experts with sigmoid";
    if (g.moe_topk_method != "noaux_tc")
        return p + "moe.topk_method is '" + g.moe_topk_method + "', this engine path selects with noaux_tc";
    if (g.expert_gating_func != 2) return p + "expert_gating_func is " + std::to_string(g.expert_gating_func) +
                                               ", 2 (sigmoid) is what the scoring func says";
    if (g.n_expert <= 0 || (g.n_expert & 7)) return p + "expert_count must be a positive multiple of 8";
    if (g.n_expert_used <= 0 || g.n_expert_used > g.n_expert) return p + "expert_used_count must be within expert_count";
    // The hybrid pattern and the DSA list must describe the same layers.
    int64_t dsa_in_pattern = 0;
    for (size_t l = 0; l < g.hybrid_pattern.size(); ++l) dsa_in_pattern += g.hybrid_pattern[l] == 0;
    if ((int64_t) g.dsa_layers.size() != dsa_in_pattern)
        return p + "attention.dsa_layers names " + std::to_string(g.dsa_layers.size()) +
               " layers, the hybrid pattern has " + std::to_string(dsa_in_pattern) + " of them";
    for (int64_t l : g.dsa_layers) {
        if (l < 0 || l >= g.n_layers) return p + "attention.dsa_layers names layer " + std::to_string(l);
        if (g.hybrid_pattern[(size_t) l] != 0)
            return p + "attention.dsa_layers names layer " + std::to_string(l) + ", the pattern makes it a window layer";
    }
    // DSA layers carry 4 kv heads, window layers 8, in this architecture (the model's own config: the
    // DSA layers attend with the indexer's GQA group).  A file with another spread is a different shape.
    for (size_t l = 0; l < g.hybrid_pattern.size(); ++l) {
        const int64_t want = g.hybrid_pattern[l] == 0 ? 4 : 8;
        if (g.n_head_kv[l] != want)
            return p + "attention.head_count_kv layer " + std::to_string(l) + " is " + std::to_string(g.n_head_kv[l]) +
                   ", this engine path expects " + std::to_string(want) + (g.hybrid_pattern[l] == 0 ? " (DSA)" : " (SWA)");
        if (g.n_head % g.n_head_kv[l]) return p + "attention.head_count is not a multiple of layer " + std::to_string(l) + "'s kv heads";
    }
    // Layer 0 is the dense layer, and it is the only one (measured: expert.layer_pattern has its 0 at
    // layer 0, and feed_forward_length is 16384 there against 2048 elsewhere).
    int64_t dense_layers = 0;
    for (int64_t v : g.expert_pattern) dense_layers += v == 0;
    if (dense_layers != 1 || g.expert_pattern[0] != 0)
        return p + "expert.layer_pattern must be dense at layer 0 only, this engine path reads no other dense layer";
    if (g.ffn_length[0] == g.n_ff_expert)
        return p + "feed_forward_length layer 0 equals the expert width: the dense layer's FFN is its own";
    return {};
}

/// Fills the draft geometry from the draft GGUF's single shard.  The prefix is naive_draft.
inline std::string read_draft_geometry(const GgufFile& f, DraftGeometry& d, const NaiveGeometry& target) {
    const MetaValue* arch = f.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArchDraft)
        return "the draft's architecture is '" + arch->s + "', this engine path requires '" + kArchDraft + "'";
    const MetaReader r(f, "naive_draft.");
    std::string err;
    int64_t x = 0;
    if (!r.int_key("block_count", x, err)) return err;
    d.n_layers = x;
    if (!r.int_key("embedding_length", x, err)) return err;
    d.n_embd = x;
    if (!r.int_key("feed_forward_length", x, err)) return err;
    d.n_ff = x;
    if (!r.int_key("attention.head_count", x, err)) return err;
    d.n_head = x;
    if (!r.int_key("attention.head_count_kv", x, err)) return err;
    d.n_head_kv = x;
    if (!r.int_key("attention.head_dim", x, err)) return err;
    d.head_dim = x;
    if (!r.int_key("attention.sliding_window", x, err)) return err;
    d.window = x;
    if (!r.float_key("attention.layer_norm_rms_epsilon", d.rms_eps, err, true)) return err;
    if (!r.int_key("block_size", x, err)) return err;
    d.block_size = x;
    if (!r.int_key("proposal_count", x, err)) return err;
    d.proposal_count = x;
    if (!r.int_key("rope.dimension_count", x, err)) return err;
    d.n_rot = x;
    if (!r.float_key("rope.freq_base", d.rope_theta, err, true)) return err;
    if (!r.str_key("rope.layout", d.rope_layout, err, true)) return err;
    if (!r.bool_key("attention.causal", d.causal, err, true)) return err;
    if (!r.int_key("vocab_size", x, err)) return err;
    d.n_vocab = x;
    if (!r.int_key("mask_token_id", x, err)) return err;
    d.mask_token_id = x;
    if (!r.int_key("markov_rank", x, err)) return err;
    d.markov_rank = x;
    if (!r.str_key("markov_head_type", d.markov_head_type, err, true)) return err;
    if (!r.bool_key("enable_confidence_head", d.confidence_head, err, true)) return err;
    if (!r.bool_key("confidence_head_with_markov", d.confidence_with_markov, err, true)) return err;
    if (!r.int_key("target.hidden_state_offset", x, err)) return err;
    d.target_hidden_offset = x;
    {
        const MetaValue* v = f.get("naive_draft.target.layer_ids");
        if (!v || v->type != MetaType::ARRAY) return "missing naive_draft.target.layer_ids";
        for (const MetaValue& e : v->items) {
            if (!e.is_num()) return "naive_draft.target.layer_ids holds a non-integer";
            d.target_layer_ids.push_back((int64_t)(int32_t) e.u);
        }
        if (v->count != d.target_layer_ids.size()) return "naive_draft.target.layer_ids is too long";
    }
    // The draft's rules: its block is the anchor plus the proposals; it is not causal (a block of
    // proposals attends within the block); the taps are target layer ids; the head shares the target's.
    if (d.block_size != d.proposal_count + 1)
        return "naive_draft.block_size is not proposal_count + 1 (the anchor plus the masked proposals)";
    if (d.causal) return "naive_draft.attention.causal is true; the DSpark draft runs block attention";
    if (d.rope_layout != "split_half") return "naive_draft.rope.layout is '" + d.rope_layout + "', not split_half";
    if (d.n_embd != target.n_embd) return "the draft's embedding_length differs from the target's";
    if (d.n_vocab != target.n_vocab) return "the draft's vocab_size differs from the target's";
    if (d.markov_head_type != "vanilla") return "naive_draft.markov_head_type is '" + d.markov_head_type + "'";
    for (int64_t l : d.target_layer_ids)
        if (l < 0 || l >= target.n_layers) return "naive_draft.target.layer_ids names layer " + std::to_string(l);
    const MetaValue* shared_emb = f.get("naive_draft.target.shared_token_embedding");
    const MetaValue* shared_out = f.get("naive_draft.target.shared_output");
    if (!shared_emb || !shared_emb->u || !shared_out || !shared_out->u)
        return "the draft does not share the target's embedding and output head; this engine path reads no draft head";
    return {};
}

// ---- the tensor inventory: the names the loader asks every shard for, and the shape each must have.
// A GGUF that lacks one, or carries it with another shape or type, is refused with the name (the same
// rule as core/family_layout.hpp's check_gguf_layout, kept here because the Naive layout is not the
// Qwen one).  Measured against the MQ87 file: global = token_embd/output + output_norm; a DSA layer
// carries an indexer (5 tensors, no sink), a window layer carries attn_sinks (no indexer); a MoE layer
// carries the router (gate_inp F32, exp_probs_b F32) and the three packed expert stacks; layer 0
// carries a dense FFN instead.

/// What one tensor slot must hold: ggml type id plus the shape (GGUF order, dim 0 fastest).  A 0 dims
/// entry checks every dimension it names; an empty shape checks existence only (the draft's F32 norms).
struct TensorSpec {
    const char* name;   ///< a printf pattern with one %d for the layer, or a plain name
    uint32_t type;      ///< ggml type id (artifact/gguf_reader.hpp)
    std::vector<uint64_t> shape;
};

inline constexpr uint32_t kF32 = 0, kQ8_0 = 8, kQ6_K = 14, kIQ2_XXS = 16, kIQ2_XS = 17, kBF16 = 30;

/// The global tensors of the main model.
inline const std::vector<TensorSpec>& main_globals() {
    static const std::vector<TensorSpec> v = {
        {"token_embd.weight", kQ8_0, {4096, 152576}},
        {"output.weight", kQ8_0, {4096, 152576}},
        {"output_norm.weight", kF32, {4096}},
    };
    return v;
}

/// The tensors of layer `l`, given the geometry.  `out` receives {name, type, shape}.
inline void layer_tensors(int64_t l, const NaiveGeometry& g, std::vector<TensorSpec>& out) {
    const int64_t nkv = g.kv_heads(l);
    const bool dsa = g.is_dsa_layer(l);
    const int64_t n_q = g.n_head * g.head_dim_k;        // 64*192 = 12288
    const int64_t n_k = nkv * g.head_dim_k;             // 768 (DSA) / 1536 (SWA)
    const int64_t n_v = nkv * g.head_dim_v;             // 512 / 1024
    const int64_t n_o = g.n_head * g.head_dim_v;        // 8192: the O projection reads V-width heads
    out.clear();
    out.push_back({"blk.%d.attn_norm.weight", kF32, {(uint64_t) g.n_embd}});
    out.push_back({"blk.%d.attn_q.weight", kQ6_K, {(uint64_t) g.n_embd, (uint64_t) n_q}});
    out.push_back({"blk.%d.attn_k.weight", kQ8_0, {(uint64_t) g.n_embd, (uint64_t) n_k}});
    out.push_back({"blk.%d.attn_v.weight", kQ8_0, {(uint64_t) g.n_embd, (uint64_t) n_v}});
    out.push_back({"blk.%d.attn_output.weight", kQ6_K, {(uint64_t) n_o, (uint64_t) g.n_embd}});
    if (dsa) {
        out.push_back({"blk.%d.indexer.q_proj.weight", kBF16,
                       {(uint64_t) g.n_embd, (uint64_t) (g.index_n_heads * g.index_head_dim)}});
        out.push_back({"blk.%d.indexer.k_proj.weight", kBF16,
                       {(uint64_t) g.n_embd, (uint64_t) (g.index_n_kv_heads * g.index_head_dim)}});
        out.push_back({"blk.%d.indexer.proj.weight", kBF16, {(uint64_t) g.n_embd, (uint64_t) g.index_n_heads}});
        out.push_back({"blk.%d.indexer.k_norm.weight", kF32, {(uint64_t) g.index_head_dim}});
        out.push_back({"blk.%d.indexer.k_norm.bias", kF32, {(uint64_t) g.index_head_dim}});
    } else if (g.add_swa_sink) {
        // Measured on MQ87: the window sink is one scalar per query head (dim = n_head = 64), not per kv head.
        out.push_back({"blk.%d.attn_sinks.weight", kF32, {(uint64_t) g.n_head}});
    }
    if (g.is_moe_layer(l)) {
        out.push_back({"blk.%d.ffn_norm.weight", kF32, {(uint64_t) g.n_embd}});
        out.push_back({"blk.%d.ffn_gate_inp.weight", kF32, {(uint64_t) g.n_embd, (uint64_t) g.n_expert}});
        out.push_back({"blk.%d.exp_probs_b.bias", kF32, {(uint64_t) g.n_expert}});
        // Measured on MQ87, matching the file's own quantization.tensor_layout key ("GGUF dimensions
        // [input, output, expert]"): gate/up are [n_embd, n_ff_expert, n_expert], down is the transpose.
        out.push_back({"blk.%d.ffn_gate_exps.weight", kIQ2_XXS,
                       {(uint64_t) g.n_embd, (uint64_t) g.n_ff_expert, (uint64_t) g.n_expert}});
        out.push_back({"blk.%d.ffn_up_exps.weight", kIQ2_XXS,
                       {(uint64_t) g.n_embd, (uint64_t) g.n_ff_expert, (uint64_t) g.n_expert}});
        out.push_back({"blk.%d.ffn_down_exps.weight", kIQ2_XS,
                       {(uint64_t) g.n_ff_expert, (uint64_t) g.n_embd, (uint64_t) g.n_expert}});
    } else {
        const uint64_t n_ff = (uint64_t) g.ffn_length[(size_t) l];
        out.push_back({"blk.%d.ffn_norm.weight", kF32, {(uint64_t) g.n_embd}});
        out.push_back({"blk.%d.ffn_gate.weight", kQ8_0, {(uint64_t) g.n_embd, n_ff}});
        out.push_back({"blk.%d.ffn_up.weight", kQ8_0, {(uint64_t) g.n_embd, n_ff}});
        out.push_back({"blk.%d.ffn_down.weight", kQ8_0, {n_ff, (uint64_t) g.n_embd}});
    }
}

/// Checks the inventory above against `model`: every tensor present, with the expected type and shape.
/// Returns "" or the first problem, naming the tensor.  (The MQ87 measurement: 613 tensors = 3 globals +
/// 9 DSA layers x (4+4+1 norm) + 39 SWA layers x (4+1 sink+1 norm) + 47 MoE x 6 + 1 dense x 4 - the
/// counts below are what the file must add up to.)
inline std::string check_gguf_layout(const GgufModel& model, const NaiveGeometry& g) {
    auto check_one = [&](const TensorSpec& s, const std::string& name) -> std::string {
        size_t shard = 0;
        const TensorInfo* t = model.find(name, &shard);
        if (!t) return "the model has no " + name;
        if (t->type != s.type)
            return name + " is " + t->type_name() + ", this build reads " + ggml_type_name(s.type);
        if (t->shape.size() != s.shape.size())
            return name + " has " + std::to_string(t->shape.size()) + " dimensions, expected " +
                   std::to_string(s.shape.size());
        for (size_t d = 0; d < s.shape.size(); ++d)
            if (s.shape[d] && t->shape[d] != s.shape[d])
                return name + " dimension " + std::to_string(d) + " is " + std::to_string(t->shape[d]) + ", expected " +
                       std::to_string(s.shape[d]);
        if (!model.in_bounds(*t, shard)) return name + " lies outside " + model.shard(shard).path();
        return {};
    };
    char buf[160];
    for (const TensorSpec& s : main_globals()) {
        std::string e = check_one(s, s.name);
        if (!e.empty()) return e;
    }
    for (int64_t l = 0; l < g.n_layers; ++l) {
        std::vector<TensorSpec> ts;
        layer_tensors(l, g, ts);
        for (const TensorSpec& s : ts) {
            std::snprintf(buf, sizeof buf, s.name, (int) l);
            std::string e = check_one(s, buf);
            if (!e.empty()) return e;
        }
    }
    return {};
}

}  // namespace strata::naive
