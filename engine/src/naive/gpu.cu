// src/naive/gpu.cu - the CUDA kernels of the Naive-N0.5-Flash GPU tier (include/strata/naive/gpu.hpp).
//
// Built only with -DSTRATA_NAIVE_CUDA=ON, into its own small static library (strata_naive_gpu), and
// linked into strata-naive INSTEAD of the stub in src/naive/pool.cpp - so exactly one translation
// unit defines the NaiveGpu methods in every build, and a CPU-only build (CI, the default) still
// links and still runs the CPU pool path unchanged. This file is the only CUDA translation unit on
// this path: the Flash-Next kernels under src/prefill are compiled against the qwen4exp geometry and
// share nothing with this shape (docs/NAIVE.md, "Why strata cannot run it").
//
// WHAT RUNS HERE: the IQ2 expert GEMV of one routed expert of one stack - y[m] = sum_k x[k] *
// W_e[k, m] - for the two quants the model's expert stacks carry (IQ2_XXS for gate/up, IQ2_XS for
// down). The stack's bytes are the pool's resident copy, uploaded unchanged (pool.hpp owns them);
// the kernel decodes each block exactly as ggml's CPU decoder does (dequantize_row_iq2_xxs /
// _iq2_xs in ggml-quants.c, line for line: the same kmask_iq2xs / ksigns_iq2xs / iq2xxs_grid /
// iq2xs_grid tables, included from the same ggml-common.h with the CUDA DECL/IMPL macros), and the
// products widen to double and sum over k as loader.hpp's GEMVs do. The k order across a row's
// lanes is the tree, not the line, so the GPU row can leave the CPU row by one accumulation ulp -
// invisible after the caller's bf16 round, which is the tolerance tests/naive/test_naive_pool.cpp
// pins. Everything else - attention, the indexer, the head - stays on the CPU forward the naive
// suites test; this file changes no other kernel and nothing in strata's own targets.
//
// One warp computes one output row: lane l decodes blocks l, l+32, l+64, ... of that row (a row is
// n_in/256 blocks; n_in=4096 gives 16 blocks per lane), accumulates its partial in double, and the
// warp folds the partials with a shuffle tree. x lives in shared memory; y is one float per row.

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"  // __device__ IQ2 tables + block_iq2_xxs / block_iq2_xs + QK_K

#include "strata/naive/gpu.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string cuda_reason(cudaError_t e, const char* what) {
    return std::string(what) + ": " + cudaGetErrorString(e);
}

__constant__ float kZeroDummy[1];  // keeps the TU CUDA-bearing when a toolchain prunes unused data

/// A byte-wise unaligned u32 load - what the CPU decoder's memcpy(aux32, x[i].qs + 4*ib32, 8) does on
/// little-endian, in device code without memcpy: a row's blocks start at multiples of 66 (or 74) bytes,
/// so qs + 4*ib32 is only 2-byte aligned and a u32 load there is illegal; four byte loads are not.
__device__ inline uint32_t ld_u32(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

/// One IQ2_XXS expert row: y[m] = sum_k x[k] * W[m][k], one warp per row. The decode is
/// dequantize_row_iq2_xxs (ggml-quants.c:2488) transcribed: aux32 is the 8 bytes at qs + 4*ib32
/// (two u32 halves, aux32[1] starting at qs + 4*ib32 + 4, as the CPU decoder's memcpy of
/// 2*sizeof(uint32_t) reads them), db = d*(0.5+(aux32[1]>>28))*0.25, grid = iq2xxs_grid + aux8[l],
/// signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127], value = db * grid[j] * (sign & kmask ? -1 : 1).
__global__ void gemv_iq2_xxs_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w,
                                    float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ float sx[];  // x's row, n_in floats
    const int tid = threadIdx.x, nt = blockDim.x;
    const int row = blockIdx.x * (nt / 32) + (tid >> 5);
    for (int k = tid; k < n_in; k += nt) sx[k] = x[k];
    __syncthreads();
    if (row >= n_out) return;
    const int lane = tid & 31;
    const int nb = n_in / QK_K;
    const block_iq2_xxs* blk =
        (const block_iq2_xxs*)(w + (size_t) row * (size_t) nb * sizeof(block_iq2_xxs));
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*) aux32;
    double acc = 0.0;
    for (int i = lane; i < nb; i += 32) {
        const float d = __half2float(blk[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            aux32[0] = ld_u32((const uint8_t*) (blk[i].qs + 4 * ib32));
            aux32[1] = ld_u32((const uint8_t*) (blk[i].qs + 4 * ib32 + 2));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            const int j0 = ib32 * 32;
            const int b0 = i * QK_K;  // this block's first element: sx[] is indexed by the global k
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int e = b0 + j0 + l * 8 + j;
                    acc += (double) sx[e] *
                           (double) (db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
                }
            }
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = (float) acc;
}

/// IQ2_XS: dequantize_row_iq2_xs (ggml-quants.c:2517) transcribed: the two per-32 scales ride in
/// scales[ib32] as nibbles, db[l/2] = d*(0.5+lo|hi)*0.25; the grid is iq2xs_grid + (qs & 511) and the
/// signs ksigns_iq2xs[qs >> 9].
__global__ void gemv_iq2_xs_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w,
                                   float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ float sx[];
    const int tid = threadIdx.x, nt = blockDim.x;
    const int row = blockIdx.x * (nt / 32) + (tid >> 5);
    for (int k = tid; k < n_in; k += nt) sx[k] = x[k];
    __syncthreads();
    if (row >= n_out) return;
    const int lane = tid & 31;
    const int nb = n_in / QK_K;
    const block_iq2_xs* blk =
        (const block_iq2_xs*)(w + (size_t) row * (size_t) nb * sizeof(block_iq2_xs));
    double acc = 0.0;
    for (int i = lane; i < nb; i += 32) {
        const float d = __half2float(blk[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0 = d * (0.5f + (blk[i].scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (blk[i].scales[ib32] >> 4)) * 0.25f;
            const int j0 = ib32 * 32;
            const int b0 = i * QK_K;  // this block's first element: sx[] is indexed by the global k
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = blk[i].qs[4 * ib32 + l];
                const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q & 511));
                const uint8_t signs = ksigns_iq2xs[q >> 9];
                const float db = (l / 2) ? db1 : db0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int e = b0 + j0 + l * 8 + j;
                    acc += (double) sx[e] *
                           (double) (db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
                }
            }
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = (float) acc;
}

}  // namespace

namespace strata::naive {

struct NaiveGpu::Impl {
    struct Stack {
        std::string name;
        void* d = nullptr;  // the stack's bytes on `card`
        int card = 0;
        std::vector<uint64_t> shape;  // {n_in, n_out, n_expert}
        uint32_t type = 0;            // IQ2_XXS 16 / IQ2_XS 17
    };
    bool ok = false;
    std::vector<Stack> stacks;
    std::vector<void*> scratch_x, scratch_y;  // per card, sized on first use
    // x's and y's sizes differ per stack kind - a gate row reads n_in=N_EMBD and writes
    // n_out=EXPERT_FF, a down row the other way - so each buffer carries its own size. Sizing them
    // by their sum is how a down row once wrote N_EMBD floats into an EXPERT_FF-sized y.
    std::vector<size_t> scratch_x_bytes, scratch_y_bytes;
};

namespace {

cudaError_t pick_card(int card) { return cudaSetDevice(card); }

}  // namespace

NaiveGpu::NaiveGpu() : impl_(new Impl()) {}
NaiveGpu::~NaiveGpu() {
    if (!impl_) return;
    release();
    delete impl_;
    impl_ = nullptr;
}

bool NaiveGpu::built_with_cuda() { return true; }  // this TU only exists in a CUDA build

bool NaiveGpu::ok() const { return impl_ && impl_->ok; }

std::string NaiveGpu::init(const std::vector<int>& want) {
    int count = 0;
    if (cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess) {
        reason_ = cuda_reason(e, "cudaGetDeviceCount");
        return reason_;
    }
    if (count == 0) {
        reason_ = "no CUDA device is present";
        return reason_;
    }
    std::vector<int> ordinals = want;
    if (ordinals.empty())
        for (int i = 0; i < count; ++i) ordinals.push_back(i);
    for (int ord : ordinals) {
        if (ord < 0 || ord >= count) {
            reason_ = "card " + std::to_string(ord) + " is out of range (" +
                      std::to_string(count) + " visible)";
            return reason_;
        }
        // cudaMemGetInfo answers for the CURRENT device only, and selecting a device brings up its
        // primary context - the ~300 MiB the plan's per-card reserve pays for.
        if (cudaError_t e = cudaSetDevice(ord); e != cudaSuccess) {
            reason_ = cuda_reason(e, "cudaSetDevice");
            return reason_;
        }
        GpuCard c;
        c.index = ord;
        size_t free_b = 0, total_b = 0;
        if (cudaError_t e = cudaMemGetInfo(&free_b, &total_b); e != cudaSuccess) {
            reason_ = cuda_reason(e, "cudaMemGetInfo");
            return reason_;
        }
        c.free_bytes = free_b;
        c.total_bytes = total_b;
        cards_.push_back(c);
    }
    impl_->scratch_x.assign(count, nullptr);
    impl_->scratch_y.assign(count, nullptr);
    impl_->scratch_x_bytes.assign(count, 0);
    impl_->scratch_y_bytes.assign(count, 0);
    impl_->ok = true;
    reason_.clear();
    cudaGetLastError();  // probe calls can leave a sticky none; start clean
    return {};
}

void NaiveGpu::release() {
    if (!impl_) return;
    for (auto& s : impl_->stacks) {
        if (!s.d) continue;
        if (pick_card(s.card) == cudaSuccess) cudaFree(s.d);
    }
    impl_->stacks.clear();
    for (size_t c = 0; c < impl_->scratch_x.size(); ++c) {
        if (pick_card((int) c) != cudaSuccess) continue;
        if (impl_->scratch_x[c]) cudaFree(impl_->scratch_x[c]);
        if (impl_->scratch_y[c]) cudaFree(impl_->scratch_y[c]);
        impl_->scratch_x[c] = impl_->scratch_y[c] = nullptr;
        impl_->scratch_x_bytes[c] = impl_->scratch_y_bytes[c] = 0;
    }
    impl_->ok = false;
}

int NaiveGpu::upload(const NaivePool::ResidentExpert& stack, int card, std::string& err) {
    if (!impl_ || !impl_->ok) {
        err = "no CUDA device in use: " + (impl_ ? reason_ : std::string("no state"));
        return -1;
    }
    for (size_t i = 0; i < impl_->stacks.size(); ++i)
        if (impl_->stacks[i].name == stack.name) return (int) i;  // idempotent by name
    if (card < 0 || (size_t) card >= impl_->scratch_x.size()) {
        err = "card " + std::to_string(card) + " is not a card this engine probed";
        return -1;
    }
    if (stack.shape.size() != 3) {
        err = stack.name + ": an expert stack is 3-D";
        return -1;
    }
    if (stack.type != kIQ2_XXS && stack.type != kIQ2_XS) {
        err = stack.name + ": the GPU tier reads IQ2_XXS and IQ2_XS, this is " +
              std::string(ggml_type_name(stack.type));
        return -1;
    }
    if (pick_card(card) != cudaSuccess) {
        err = "cudaSetDevice(" + std::to_string(card) + ") failed";
        return -1;
    }
    void* d = nullptr;
    if (cudaError_t e = cudaMalloc(&d, (size_t) stack.bytes_len); e != cudaSuccess) {
        err = stack.name + ": " + cuda_reason(e, "cudaMalloc");
        return -1;
    }
    if (cudaError_t e = cudaMemcpy(d, stack.bytes, (size_t) stack.bytes_len, cudaMemcpyHostToDevice);
        e != cudaSuccess) {
        cudaFree(d);
        err = stack.name + ": " + cuda_reason(e, "cudaMemcpy");
        return -1;
    }
    Impl::Stack s;
    s.name = stack.name;
    s.d = d;
    s.card = card;
    s.shape = stack.shape;
    s.type = stack.type;
    impl_->stacks.push_back(std::move(s));
    return (int) impl_->stacks.size() - 1;
}

int NaiveGpu::upload_from_mmap(const ExpertStack& st, const uint8_t* src, int card, std::string& err) {
    if (!impl_ || !impl_->ok) {
        err = "no CUDA device in use: " + (impl_ ? reason_ : std::string("no state"));
        return -1;
    }
    for (size_t i = 0; i < impl_->stacks.size(); ++i)
        if (impl_->stacks[i].name == st.name) return (int) i;  // idempotent by name
    if (card < 0 || (size_t) card >= impl_->scratch_x.size()) {
        err = "card " + std::to_string(card) + " is not a card this engine probed";
        return -1;
    }
    if (st.shape.size() != 3 || st.bytes == 0 || src == nullptr) {
        err = st.name + ": an expert stack is a non-empty 3-D tensor with bytes to read";
        return -1;
    }
    if (st.type != kIQ2_XXS && st.type != kIQ2_XS) {
        err = st.name + ": the GPU tier reads IQ2_XXS and IQ2_XS, this is " +
              std::string(ggml_type_name(st.type));
        return -1;
    }
    if (pick_card(card) != cudaSuccess) {
        err = "cudaSetDevice(" + std::to_string(card) + ") failed";
        return -1;
    }
    // The exclusive tier's one host touch: cudaMemcpy reads the mmap pages directly (the driver
    // pages them in as needed); no engine-owned buffer ever holds the stack.
    void* d = nullptr;
    if (cudaError_t e = cudaMalloc(&d, (size_t) st.bytes); e != cudaSuccess) {
        err = st.name + ": " + cuda_reason(e, "cudaMalloc");
        return -1;
    }
    if (cudaError_t e = cudaMemcpy(d, src, (size_t) st.bytes, cudaMemcpyHostToDevice); e != cudaSuccess) {
        cudaFree(d);
        err = st.name + ": " + cuda_reason(e, "cudaMemcpy");
        return -1;
    }
    Impl::Stack s;
    s.name = st.name;
    s.d = d;
    s.card = card;
    s.shape = st.shape;
    s.type = st.type;
    impl_->stacks.push_back(std::move(s));
    return (int) impl_->stacks.size() - 1;
}

int NaiveGpu::handle_of(const std::string& name) const {
    if (!impl_) return -1;
    for (size_t i = 0; i < impl_->stacks.size(); ++i)
        if (impl_->stacks[i].name == name) return (int) i;
    return -1;
}

std::string NaiveGpu::gemv_expert(int handle, uint64_t expert, const float* x, int64_t n_in, float* y,
                                  int64_t n_out) {
    if (!impl_ || !impl_->ok) return "no CUDA device in use";
    if (handle < 0 || (size_t) handle >= impl_->stacks.size()) return "bad expert-stack handle";
    Impl::Stack& s = impl_->stacks[(size_t) handle];
    if (expert >= s.shape[2])
        return s.name + ": expert " + std::to_string(expert) + " outside " +
               std::to_string(s.shape[2]);
    if ((int64_t) s.shape[0] != n_in || (int64_t) s.shape[1] != n_out)
        return s.name + " is {" + std::to_string(s.shape[0]) + "," + std::to_string(s.shape[1]) +
               "," + std::to_string(s.shape[2]) + "}, this call reads {" + std::to_string(n_in) +
               "," + std::to_string(n_out) + "}";
    if (pick_card(s.card) != cudaSuccess) return "cudaSetDevice failed";
    const size_t need_x = (size_t) n_in * 4, need_y = (size_t) n_out * 4;
    // Each buffer grows on its own need: gate/up and down want x and y the other way around, and
    // the two wants add up to the same sum - sizing by the sum kept the old (wrong) size alive.
    if (impl_->scratch_x_bytes[(size_t) s.card] < need_x) {
        if (impl_->scratch_x[(size_t) s.card]) cudaFree(impl_->scratch_x[(size_t) s.card]);
        void* px = nullptr;
        if (cudaError_t e = cudaMalloc(&px, need_x); e != cudaSuccess) {
            impl_->scratch_x[(size_t) s.card] = nullptr;
            impl_->scratch_x_bytes[(size_t) s.card] = 0;
            return cuda_reason(e, "cudaMalloc(x)");
        }
        impl_->scratch_x[(size_t) s.card] = px;
        impl_->scratch_x_bytes[(size_t) s.card] = need_x;
    }
    if (impl_->scratch_y_bytes[(size_t) s.card] < need_y) {
        if (impl_->scratch_y[(size_t) s.card]) cudaFree(impl_->scratch_y[(size_t) s.card]);
        void* py = nullptr;
        if (cudaError_t e = cudaMalloc(&py, need_y); e != cudaSuccess) {
            impl_->scratch_y[(size_t) s.card] = nullptr;
            impl_->scratch_y_bytes[(size_t) s.card] = 0;
            return cuda_reason(e, "cudaMalloc(y)");
        }
        impl_->scratch_y[(size_t) s.card] = py;
        impl_->scratch_y_bytes[(size_t) s.card] = need_y;
    }
    void* dx = impl_->scratch_x[(size_t) s.card];
    void* dy = impl_->scratch_y[(size_t) s.card];
    if (cudaError_t e = cudaMemcpy(dx, x, need_x, cudaMemcpyHostToDevice); e != cudaSuccess)
        return cuda_reason(e, "cudaMemcpy(x)");
    // The expert slice's byte offset: ggml_payload_bytes for n_in*n_out elements, computed the same
    // way pool.hpp / loader.hpp compute it (elements / 256 blocks x 66 or 74 bytes a block).
    const uint64_t slice_bytes =
        ((uint64_t) n_in * (uint64_t) n_out / 256) * (uint64_t) (s.type == kIQ2_XXS ? 66 : 74);
    const uint8_t* dW = (const uint8_t*) s.d + (size_t) expert * (size_t) slice_bytes;
    const int threads = 128;  // 4 warps = 4 rows per block
    const int blocks = (int) ((n_out + 3) / 4);
    const size_t smem = (size_t) n_in * 4;
    if (s.type == kIQ2_XXS)
        gemv_iq2_xxs_kernel<<<blocks, threads, smem>>>(const_cast<float*>((const float*) dx), dW,
                                                       (float*) dy, (int) n_in, (int) n_out);
    else
        gemv_iq2_xs_kernel<<<blocks, threads, smem>>>(const_cast<float*>((const float*) dx), dW,
                                                      (float*) dy, (int) n_in, (int) n_out);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return cuda_reason(e, "kernel launch");
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
        return cuda_reason(e, "kernel run");
    if (cudaError_t e = cudaMemcpy(y, dy, need_y, cudaMemcpyDeviceToHost); e != cudaSuccess)
        return cuda_reason(e, "cudaMemcpy(y)");
    return {};
}

}  // namespace strata::naive
