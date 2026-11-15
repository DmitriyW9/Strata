// include/strata/naive/geometry.hpp - the Naive-N0.5-Flash geometry (`naive_n05_flash`), read from the model
// file's own metadata, and the DSpark draft's (`naive_n05_flash_dspark`).
//
// This is a NEW engine path (src/naive, the strata-naive executable), separate from core/geometry.hpp:
// the Flash-Next geometry and the kernels compiled against it (core/geometry_checks.hpp) fix one shape
// (GDN on 36/48 layers, hyper-connections, head_dim 256, a 4-head QSA indexer), and this architecture
// shares none of it: 48 GQA-attention layers (9 full-history DSA with a 16-head FP8 indexer, 39 sliding
// window of 128), K=192/V=128 heads, split-half RoPE with two thetas, and a sigmoid/noaux_tc router with
// a correction bias.  Like the dsv41 path (cmake/strata_gpu_engine.cmake's STRATA_BUILD_DSV41), it is its
// own target; nothing here is compiled into `strata`.
//
// The keys and their meanings are from the model's own GGUF (measured on Naive-N0.5-Flash-MQ87,
// gguf-dump): see docs/NAIVE.md for the tensor inventory and the read-state table.
//
// Dependency-free on purpose, like core/geometry.hpp: the reader (naive/reader.hpp) includes it, and a
// later kernel file can too.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::naive {

inline constexpr const char* kArchMain = "naive_n05_flash";
inline constexpr const char* kArchDraft = "naive_n05_flash_dspark";

/// The main model's shape, every field read from a "<arch>." metadata key (named beside it).  The
/// defaults are the published MQ87 conversion's values (NaiveAI/Naive-N0.5-Flash revision
/// 0235b3b5ff27422b1f57cdc2acddfaf643e08356); `read_geometry` refuses a file whose required keys differ
/// in TYPE or a per-layer list whose length is not the block count, and checks the cross-key rules
/// (`check_geometry`) - it never mis-runs a file by trusting a default for a key the file lacks.
struct NaiveGeometry {
    int64_t n_vocab = 152576;       ///< the length of tokenizer.ggml.tokens
    int64_t n_embd = 4096;          ///< embedding_length
    int64_t n_layers = 48;          ///< block_count
    int64_t n_ctx = 1048576;        ///< context_length
    double rms_eps = 1e-5;          ///< attention.layer_norm_rms_epsilon

    // GQA attention, every layer; no GDN, no hyper-connections (unlike Flash-Next).
    int64_t n_head = 64;            ///< attention.head_count
    int64_t head_dim_k = 192;       ///< attention.key_length
    int64_t head_dim_v = 128;       ///< attention.value_length
    int64_t swa_window = 128;       ///< attention.sliding_window (the SWA layers' window)
    int64_t attn_chunk = 128;       ///< attention_chunk_size
    double value_scale = 0.707;     ///< attention.value_scale
    std::string projection_layout = "split";  ///< attention.projection_layout: "split" (q/k/v apart, no latent)
    bool add_swa_sink = true;       ///< attention.add_swa_sink: an attn_sinks.weight per SWA layer
    bool add_full_sink = false;     ///< attention.add_full_sink: the DSA layers' sinks (this model: none)

    /// attention.head_count_kv, one entry per layer: 4 on the DSA layers, 8 on the SWA layers (checked
    /// against the layer pattern by `check_geometry`).
    std::vector<int64_t> n_head_kv;
    /// attention.hybrid_layer_pattern, one entry per layer: 0 = full-history DSA, 1 = sliding window.
    std::vector<int64_t> hybrid_pattern;
    /// attention.dsa_layers: the layers with the indexer and the full K/V history (the pattern's 0s).
    std::vector<int64_t> dsa_layers;
    bool dsa_enabled = true;        ///< attention.dsa_enabled

    // Split-half (GPT-NeoX) RoPE, two thetas; `partial` of each head's K is rotated.
    int64_t n_rot = 64;             ///< rope.dimension_count (the DSA layers)
    int64_t n_rot_swa = 64;         ///< rope.dimension_count_swa (the SWA layers)
    double rope_theta_full = 1e7;   ///< rope.freq_base
    double rope_theta_swa = 1e4;    ///< rope.freq_base_swa
    double partial_rotary_factor = 0.334;  ///< rope.partial_rotary_factor (n_rot / head_dim_k, rounded)
    std::string rope_pairing = "split-half / GPT-NeoX";  ///< rope.pairing

    // The DSA indexer, on the DSA layers only: 16 query heads x 128 against 1 key head, FP8 E4M3
    // activations, top-2048 selected positions of the layer's full history.
    int64_t index_top_k = 2048;     ///< index_top_k
    int64_t index_head_dim = 128;   ///< index_head_dim
    int64_t index_n_heads = 16;     ///< index_n_heads
    int64_t index_n_kv_heads = 1;   ///< index_n_kv_heads
    std::string indexer_activation_dtype = "fp8_e4m3";  ///< indexer_activation_dtype

    // MoE: 256 experts, top-8, sigmoid scores and noaux_tc selection with a correction bias
    // (exp_probs_b.bias) - NOT the Qwen paths' softmax/renorm router.  Layer 0 is dense (no experts):
    // expert.layer_pattern says which layers are MoE, and feed_forward_length gives each layer's dense
    // FFN width (16384 on layer 0, 2048 elsewhere).
    int64_t n_expert = 256;         ///< expert_count
    int64_t n_expert_used = 8;      ///< expert_used_count
    int64_t n_ff_expert = 2048;     ///< expert_feed_forward_length
    bool expert_weights_norm = true; ///< expert_weights_norm: the top-8 weights are renormalized
    int64_t expert_gating_func = 2; ///< expert_gating_func (llama.cpp's enum: 2 = sigmoid)
    std::string moe_topk_method = "noaux_tc";   ///< moe.topk_method
    std::string moe_scoring_func = "sigmoid";   ///< moe.scoring_func
    std::vector<int64_t> ffn_length;            ///< feed_forward_length, one entry per layer
    std::vector<int64_t> expert_pattern;        ///< expert.layer_pattern: 1 = MoE layer, 0 = dense

    constexpr bool is_dsa_layer(int64_t l) const { return hybrid_pattern[(size_t) l] == 0; }
    constexpr bool is_moe_layer(int64_t l) const { return expert_pattern[(size_t) l] != 0; }
    /// The DSA layers keep the whole K/V history; the SWA layers keep a window.  Derived, not a list.
    constexpr int64_t n_dsa_layers() const { return (int64_t) dsa_layers.size(); }
    constexpr int64_t n_swa_layers() const { return n_layers - n_dsa_layers(); }
    constexpr int64_t kv_heads(int64_t l) const { return n_head_kv[(size_t) l]; }
};

/// The DSpark draft's shape (draft/README.md, the draft GGUF's own keys).  Five SWA(1024) layers, a
/// 7-position block (1 anchor + 6 proposals from a learned mask), eight target hidden taps, a vanilla
/// rank-256 Markov head and a confidence head.  The target's embedding and output head are shared, not
/// duplicated; the draft brings its own `fc` over the concatenated taps (8 x 4096 = 32768 rows).
struct DraftGeometry {
    int64_t n_layers = 5;           ///< naive_draft.block_count
    int64_t n_embd = 4096;          ///< naive_draft.embedding_length
    int64_t n_ff = 4096;            ///< naive_draft.feed_forward_length
    int64_t n_head = 32;            ///< naive_draft.attention.head_count
    int64_t n_head_kv = 4;          ///< naive_draft.attention.head_count_kv
    int64_t head_dim = 128;         ///< naive_draft.attention.head_dim
    int64_t window = 1024;          ///< naive_draft.attention.sliding_window
    double rms_eps = 1e-5;          ///< naive_draft.attention.layer_norm_rms_epsilon
    int64_t block_size = 7;         ///< naive_draft.block_size (1 anchor + proposal_count masks)
    int64_t proposal_count = 6;     ///< naive_draft.proposal_count
    int64_t n_rot = 128;            ///< naive_draft.rope.dimension_count
    double rope_theta = 1e4;        ///< naive_draft.rope.freq_base
    std::string rope_layout = "split_half";  ///< naive_draft.rope.layout
    bool causal = false;            ///< naive_draft.attention.causal (block attention, not causal)
    int64_t n_vocab = 152576;       ///< naive_draft.vocab_size
    int64_t mask_token_id = 151675; ///< naive_draft.mask_token_id
    int64_t markov_rank = 256;      ///< naive_draft.markov_rank
    std::string markov_head_type = "vanilla";  ///< naive_draft.markov_head_type
    bool confidence_head = true;    ///< naive_draft.enable_confidence_head
    bool confidence_with_markov = true;        ///< naive_draft.confidence_head_with_markov
    std::vector<int64_t> target_layer_ids;     ///< naive_draft.target.layer_ids (the 8 hidden taps)
    int64_t target_hidden_offset = 1;          ///< naive_draft.target.hidden_state_offset
};

}  // namespace strata::naive
