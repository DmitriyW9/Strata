#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class PeerExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
//
// 0.1.52: a tier can now be a second card's expert cache for the layers the PRIMARY stage already
// runs (--peer-device on a layer split), not only a helper cache for layers no stage runs. Same weighted
// partial-sum reduce: the tier computes its share of a layer's routed experts into compact rows, this class
// folds them with the router weights into one per-token sum, and the verifier adds that sum and skips the
// rows the tier returned. That is the real expert parallelism on the native/Qwen-pack decode path.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    void attach(PeerExperts& peer);     // before PeerExperts::open (a layer split's second card as an expert tier)
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    /// True while a verify window brackets the dispatch with begin()/end(): the reduce path is live.
    bool active() const { return weights_ != nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    friend class PeerExperts;
    static size_t metadata_bytes();
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);
    // the peer tier on a layer split (real EP): same reduce, on the peer card's compact rows
    void prepare(const PeerExperts& peer, void* metadata) const;
    bool reduce(PeerExperts& peer, const void* metadata, std::string& err);
    void accumulate(const PeerExperts& peer);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    struct PeerTier { PeerExperts* peer; float* sum = nullptr; };
    std::vector<Peer> peers_;
    std::vector<PeerTier> ppeers_;   // --peer-device tiers on a layer split
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
