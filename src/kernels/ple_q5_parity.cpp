// Real-artifact Q5_0 and Q5_1 PLE rows against ggml's reference dequantizer.
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_q5_parity <gguf-containing-ple>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* tensor = gguf.find("per_layer_token_embd.weight");
    if (!tensor || (tensor->type != 6 && tensor->type != 7) || tensor->shape.size() != 2 ||
        tensor->shape[0] != 160) {
        std::fprintf(stderr, "expected Q5_0 or Q5_1 PLE [160, N]\n");
        return 2;
    }
    const auto type = tensor->type == 7 ? GGML_TYPE_Q5_1 : GGML_TYPE_Q5_0;
    const size_t row_bytes = tensor->type == 7 ? 120 : 110;
    const char* type_name = tensor->type == 7 ? "Q5_1" : "Q5_0";
    const auto* traits = ggml_get_type_traits(type);
    const uint8_t* bytes = gguf.tensor_data(*tensor);
    const uint32_t probes[] = {0, 1, 12345, 20000003, (uint32_t) (tensor->shape[1] - 1)};
    bool pass = true;
    for (const auto mode : {strata::kernels::PleIo::Mmap, strata::kernels::PleIo::Direct}) {
        strata::kernels::PleIoOptions options;
        options.mode = mode;
        strata::kernels::PleTable table;
        std::string err;
        if (!table.open(argv[1], err, options)) {
            std::fprintf(stderr, "%s: %s\n", mode == strata::kernels::PleIo::Direct ? "Direct" : "Mmap", err.c_str());
            return 1;
        }
        double max_abs = 0.0;
        for (uint32_t row : probes) {
            float got[160], want[160];
            table.read_row(row, got);
            traits->to_float(bytes + (size_t) row * row_bytes, want, 160);
            for (int i = 0; i < 160; ++i)
                max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
        }
        uint32_t rows[16];
        for (int i = 0; i < 16; ++i) rows[i] = probes[i % 5];
        float batch[16 * 160];
        if (!table.issue(rows) || !table.collect(batch, err)) {
            std::fprintf(stderr, "%s collect: %s\n",
                         mode == strata::kernels::PleIo::Direct ? "Direct" : "Mmap", err.c_str());
            return 1;
        }
        for (int h = 0; h < 16; ++h) {
            float want[160];
            traits->to_float(bytes + (size_t) rows[h] * row_bytes, want, 160);
            for (int i = 0; i < 160; ++i)
                max_abs = std::max(max_abs, (double) std::fabs(batch[h * 160 + i] - want[i]));
        }
        const bool mode_pass = max_abs <= 1e-6;
        std::printf("%s PLE %s: %zu rows, max_abs %.3e %s\n", type_name,
                    mode == strata::kernels::PleIo::Direct ? "Direct" : "Mmap",
                    (size_t) table.rows(), max_abs, mode_pass ? "PASS" : "FAIL");
        pass = pass && mode_pass;
    }
    return pass ? 0 : 1;
}
