// src/naive/gpu_stub.cpp - the CPU-only build's half of the GPU tier (include/strata/naive/gpu.hpp).
//
// CMake links this file into strata-naive when -DSTRATA_NAIVE_CUDA=OFF (the default, and every CI
// box): the NaiveGpu methods exist so the runtime can probe the tier unconditionally, and every call
// refuses with the reason the header promises. With the flag ON, src/naive/gpu.cu defines the same
// methods and this file is not compiled, so exactly one translation unit ever holds them.
#include "strata/naive/gpu.hpp"

namespace strata::naive {

struct NaiveGpu::Impl {};

NaiveGpu::NaiveGpu() : impl_(new Impl()) {}
NaiveGpu::~NaiveGpu() {
    delete impl_;
    impl_ = nullptr;
}

bool NaiveGpu::built_with_cuda() { return false; }
bool NaiveGpu::ok() const { return false; }

std::string NaiveGpu::init(const std::vector<int>&) {
    reason_ = "this build has no CUDA kernels (strata-naive built with -DSTRATA_NAIVE_CUDA=OFF)";
    return reason_;
}

void NaiveGpu::release() {}

int NaiveGpu::upload(const NaivePool::ResidentExpert&, int, std::string& err) {
    err = "this build has no CUDA kernels";
    return -1;
}

int NaiveGpu::upload_from_mmap(const ExpertStack&, const uint8_t*, int, std::string& err) {
    err = "this build has no CUDA kernels";
    return -1;
}

int NaiveGpu::handle_of(const std::string&) const { return -1; }

std::string NaiveGpu::gemv_expert(int, uint64_t, const float*, int64_t, float*, int64_t) {
    return "this build has no CUDA kernels";
}

}  // namespace strata::naive
