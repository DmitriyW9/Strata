// include/strata/naive/trunk.hpp - the one piece the single-token forward (the "trunk") needs on top
// of kernels.hpp and loader.hpp: the block-FP8 weight dequant the DSA index-query projection runs on
// its BF16 wiq weight.  Every function names its NaiveRT reference; docs/NAIVE.md tracks what is
// measured.  (bf16 rounding lives in kernels.hpp::bf16_round; this header adds only what is new.)
//
// The reference flow is run_naive_n0p5_decode.py (the authors' own torch oracle) and its DSA layer
// run_attn_dsa.py.  The DSA index-query projection multiplies the normalised hidden state by a BF16
// weight (blk.L.indexer.q_proj.weight) that the reference round-trips through BLOCK FP8 first
// (packing.py:quantize_fp8_block): one e4m3 code per element with one fp32 scale per 128x128 block
// (amax/448, clamped at 1e-12), the e4m3 code multiplied back by the scale and the product rounded to
// bf16 - the pre-computed `wiq_deq` (run_attn_dsa.py:Layer.__init__ -> dequant_block).  The
// activations are the fp32 (bf16-rounded) rmsnorm output, cast to e4m3 with no scale (scale 1), as the
// reference's `xn.to(torch.float8_e4m3fn).float()` does.
#pragma once

#include "strata/naive/kernels.hpp"

#include <cmath>
#include <cstddef>

namespace strata::naive {

/// A {rows, cols} weight (already read to fp32 from its BF16 bits) round-tripped through block-e4m3:
/// for each 128x128 block, amax (clamped to 1e-12), scale = amax/448, each element encoded to e4m3 and
/// decoded and rescaled, the product rounded to bf16.  `deq` must not alias `w` (the amax pass and the
/// write pass read the same input).  rows and cols must be multiples of 128 (the model's wiq is
/// 2048x4096).  `scales` may be null; otherwise it takes [rows/128, cols/128] row-major.
/// NaiveRT: naivert/kernels/packing.py:quantize_fp8_block (clamp_min 1e-12, scale amax/448) +
/// checkpoint_fp8.dequant_block (q*scale, .to(bf16)); the encode saturates at 448 like torch's clamp.
inline void fp8_block_rows_dequant(const float* w, int rows, int cols, float* deq, float* scales) {
    for (int br = 0; br < rows; br += 128) {
        for (int bc = 0; bc < cols; bc += 128) {
            float amax = 0.0f;
            for (int i = br; i < br + 128; ++i)
                for (int j = bc; j < bc + 128; ++j)
                    amax = amax > std::fabs(w[(size_t) i * cols + j]) ? amax : std::fabs(w[(size_t) i * cols + j]);
            if (!(amax > 1e-12f)) amax = 1e-12f;
            const float scale = amax / 448.0f;
            if (scales) scales[(size_t) (br / 128) * (cols / 128) + bc / 128] = scale;
            for (int i = br; i < br + 128; ++i)
                for (int j = bc; j < bc + 128; ++j) {
                    const size_t idx = (size_t) i * cols + j;
                    deq[idx] = bf16_round(fp8_e4m3_decode(fp8_e4m3_encode(w[idx] / scale)) * scale);
                }
        }
    }
}

}  // namespace strata::naive
