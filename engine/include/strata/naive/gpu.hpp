// include/strata/naive/gpu.hpp - the Naive-N0.5-Flash GPU tier: the cards' memory and the IQ2 expert
// GEMVs that run on them. CPU-only header on purpose: it names no CUDA type, so runtime.hpp and the
// CPU tests include it freely. The methods are DEFINED twice, by whichever half of the CMake
// STRATA_BUILD_NAIVE block the build enabled: src/naive/gpu.cu (the real kernels, built only with
// -DSTRATA_NAIVE_CUDA=ON into strata_naive_gpu) or src/naive/pool.cpp (the stub: ok() false, every
// call returns "this build has no CUDA kernels", so a CPU-only binary - CI, this engine's default -
// answers the runtime's GPU probes without a CUDA dependency and stays byte-identical to the path
// the naive suites already tested).
//
// WHAT IT DOES. The plan (pool.hpp) says which IQ2 expert stacks fit on which card; NaiveGpu copies
// those stacks (their raw bytes, unchanged) to the card once, at start-up, and answers the runtime's
// per-row expert GEMV (y = x . W_e for the eight routers picks) on the card instead of the CPU. The
// arithmetic is ggml's own IQ2_XXS / IQ2_XS decode (the same codebook tables loader.hpp's dequant
// reads) transcribed into the kernel, so a GPU expert row lands within one bf16 ulp of the CPU row -
// the runtime rounds it to bf16 before the combine, exactly as the CPU path does.
//
// WHY A SEPARATE TIER AND NOT a rewrite of the forward. The rest of the model (attention, the DSA
// indexer, the norms/router, the embedding and output head) stays on the CPU forward that the four
// `ctest -R naive` suites pin against numpy; only the 61 GiB of experts - the part the router touches
// eight ways per row and that dominates the bytes - moves to the cards. A row's numbers therefore do
// not depend on whether a card holds its experts: the runtime reads the CPU pool copy when no GPU is
// present and the GPU when one is, and the test checks the two agree to one bf16 ulp.
#pragma once

#include "strata/naive/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::naive {

/// One card the GPU tier can use: its ordinal and the free/total bytes it reported at probe time.
/// (Distinct from pool.hpp's CardMem, which the CPU-side plan reads from whatever prober the caller
/// chose; NaiveGpu fills these from cudaMemGetInfo itself.)
struct GpuCard {
    int index = 0;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
};

/// The GPU tier over the cards this process can see. Lifetime: construct once after the pool is
/// loaded, call upload() once per expert stack to place it, then gemv_expert() per row. Copying an
/// expert stack copies the pool's resident bytes (a cudaMemcpyAsync of bytes the pool guarantees are
/// stable for the process), so upload never reads the disk. All methods are safe on an absent GPU:
/// ok() is false and the calls report why.
class NaiveGpu {
public:
    NaiveGpu();
    ~NaiveGpu();
    NaiveGpu(const NaiveGpu&) = delete;
    NaiveGpu& operator=(const NaiveGpu&) = delete;

    /// Probe the cards (cudaGetDeviceCount / cudaMemGetInfo). "" or a reason; ok() reflects it. An
    /// empty `want` takes every visible card; a non-empty one restricts to those ordinals.
    std::string init(const std::vector<int>& want);
    bool ok() const;
    std::string reason() const { return reason_; }
    const std::vector<GpuCard>& cards() const { return cards_; }
    /// Whether this binary was built with the CUDA kernels compiled in. False in a CPU-only build,
    /// where init() leaves ok() false with "this build has no CUDA kernels".
    static bool built_with_cuda();

    /// Copy one expert stack (the pool's bytes) to `card`, keeping it resident. Returns a handle
    /// >= 0 to pass to gemv_expert, or -1 with `err` naming the reason (no GPU, the card is unknown,
    /// or the stack does not fit the card's room the plan left). Idempotent by name: re-uploading a
    /// name already placed returns its handle without a second copy.
    int upload(const NaivePool::ResidentExpert& stack, int card, std::string& err);

    /// The exclusive tier's upload (--weights vram-split, pool.hpp's plan_split): copy the stack's
    /// bytes to `card` STRAIGHT from `src` - the caller passes the mmap pointer the loader reads
    /// (GgufFile::tensor_data), so the stack never gets a RAM copy the engine holds: the pages the
    /// copy touches are the file's own page-cache pages, clean and reclaimable, which is exactly
    /// the RAM the OS may take back under pressure (a pool vector never gives its bytes back).
    /// Same checks and idempotence as upload(); `st` carries the name, shape, type and size.
    /// A CUDA/DMA failure frees the card room and returns -1 with the reason: the caller then
    /// places the stack on the RAM pool instead (the exclusive plan's fallback).
    int upload_from_mmap(const ExpertStack& st, const uint8_t* src, int card, std::string& err);

    /// Free every placed stack and release the cards. After this ok() is false.
    void release();

    /// The handle an uploaded stack got, or -1 when the tier does not hold that name. The runtime
    /// asks this per row to pick the card path for a stack the plan placed on one.
    int handle_of(const std::string& name) const;

    /// The per-row expert GEMV on the card: y[m] = sum_k x[k] * W_e[k, m] over expert `expert` of the
    /// stack behind `handle`, with x on the host and y read back to `y` (n_out floats). "" or a reason.
    /// The kernel decodes the IQ2 block for each output row through the same tables as loader.hpp; the
    /// result is what the CPU pool's gemv_expert_raw would return, within one bf16 ulp.
    std::string gemv_expert(int handle, uint64_t expert, const float* x, int64_t n_in, float* y,
                            int64_t n_out);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::vector<GpuCard> cards_;
    std::string reason_;
};

}  // namespace strata::naive
