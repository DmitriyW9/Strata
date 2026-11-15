// src/naive/naive_main.cpp - the Naive-N0.5-Flash engine path (the strata-naive executable).
//
// What this build does: it opens the model's GGUF shards, reads the geometry from the model's own
// metadata (naive/geometry.hpp), checks every tensor of the layout (naive/reader.hpp's inventory),
// reads the draft's geometry the same way, prints the facts (INFO lines, the serve protocol's shape),
// and refuses, with the reason, anything it cannot run.  It never mmaps less and trusts more: a model
// of another shape is refused by the key or tensor that differs.
//
// What it does NOT do yet: run the trunk.  The forward (SWA + DSA attention with the FP8 indexer, the
// sigmoid/noaux_tc router, the IQ2 experts, the DSpark drafter) is the work tracked in docs/NAIVE.md;
// until it lands, a request is refused here, in the open, rather than answered wrong.  `--check` is the
// validation the CI and the run script use; without it the binary still validates first, then refuses.
//
// Like the dsv41 path, this is its own executable: the Flash-Next kernels are compiled against one
// geometry (core/geometry_checks.hpp), and this shape shares none of it.
//
//   strata-naive --native <shard1.gguf> [--draft <draft.gguf>] [--check]
//   strata-naive --serve ...            (the serve protocol; refuses GEN until the trunk exists)

#include "strata/naive/geometry.hpp"
#include "strata/naive/reader.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0.0.0-naive"
#endif

namespace {

void usage() {
    std::fprintf(stderr,
                 "strata-naive --native <model.gguf (any shard)> [--draft <dspark-draft.gguf>]\n"
                 "               [--check] [--max-context N] [--verbose] | --serve\n"
                 "\n"
                 "  --native PATH     the model's GGUF; a split model is opened by any of its shards\n"
                 "  --draft PATH      the DSpark draft GGUF (optional: without it the engine drafts nothing)\n"
                 "  --check           validate the model (geometry, layout, draft) and exit 0; nonzero + the\n"
                 "                    reason otherwise.  No GPU, no weights loaded: CI can run it anywhere.\n"
                 "  --max-context N   cap the context for the plan report (default: the model's own)\n"
                 "  --serve           the serve protocol's INFO/READY lines, then refuse every GEN with the\n"
                 "                    reason (the trunk forward is the work in progress; docs/NAIVE.md)\n");
}

/// One INFO fact, the shape the server's Monitor reads (`INFO key=value ...`).
void info(const char* fmt, ...) {
    std::fputs("INFO ", stderr);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

struct Args {
    std::string native, draft;
    bool check = false, serve = false, verbose = false;
    long long max_context = 0;
    std::string err;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto value = [&](std::string& out) {
            if (i + 1 >= argc) {
                a.err = s + " needs a value";
                return false;
            }
            out = argv[++i];
            return true;
        };
        if (s == "--native") { if (!value(a.native)) return false; }
        else if (s == "--draft") { if (!value(a.draft)) return false; }
        else if (s == "--check") a.check = true;
        else if (s == "--serve") a.serve = true;
        else if (s == "--verbose") a.verbose = true;
        else if (s == "--max-context") {
            std::string v;
            if (!value(v)) return false;
            a.max_context = std::strtoll(v.c_str(), nullptr, 10);
        }
        else if (s == "--help" || s == "-h") { usage(); std::exit(0); }
        else if (s == "--version") { std::printf("strata-naive %s\n", STRATA_VERSION); std::exit(0); }
        else { a.err = "unknown argument: " + s; usage(); return false; }
    }
    if (a.native.empty()) { a.err = "--native is required"; usage(); return false; }
    return true;
}

/// The per-layer KV bytes the cache plan needs: the DSA layers keep every position of the whole
/// history (K 192 + V 128 per kv head, F16), the window layers keep `swa_window` + one chunk.  This is
/// the same arithmetic the model's own memory table projects (its README's "K/V histories, BF16" row);
/// printed as INFO so the launcher and the Monitor can plan, and so the number is checkable, not magic.
double kv_cache_gib(const strata::naive::NaiveGeometry& g, long long ctx) {
    double bytes = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t nkv = g.kv_heads(l);
        const double per_pos = (double) nkv * (double) (g.head_dim_k + g.head_dim_v) * 2.0;   // F16
        const double cells = g.is_dsa_layer(l) ? (double) ctx : (double) (g.swa_window + g.attn_chunk);
        bytes += per_pos * cells;
    }
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    info("engine=strata-naive version=%s", STRATA_VERSION);

    // ---- the model: metadata + tensor directory of every shard (the data pages map too; nothing is
    // read from them here beyond the headers, and --check stops at the inventory).
    strata::GgufModel* model = nullptr;
    try {
        model = new strata::GgufModel(strata::gguf_split_paths(a.native));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-naive: %s\n", e.what());
        return 2;
    }
    if (model->size() > 1) info("shards=%zu", model->size());

    strata::naive::NaiveGeometry g;
    if (std::string err = strata::naive::read_geometry(model->meta(), g); !err.empty()) {
        std::fprintf(stderr, "strata-naive: %s\n", err.c_str());
        return 2;
    }
    if (std::string err = strata::naive::check_gguf_layout(*model, g); !err.empty()) {
        std::fprintf(stderr, "strata-naive: %s\n", err.c_str());
        return 2;
    }
    info("arch=%s layers=%lld dsa=%lld swa=%lld experts=%lld top%d %s/%s embd=%lld kv_cache_gib_at_model_ctx=%.2f",
         strata::naive::kArchMain, (long long) g.n_layers, (long long) g.n_dsa_layers(),
         (long long) g.n_swa_layers(), (long long) g.n_expert, (long long) g.n_expert_used,
         g.moe_scoring_func.c_str(), g.moe_topk_method.c_str(), (long long) g.n_embd,
         kv_cache_gib(g, (long long) g.n_ctx));

    // ---- the draft, when given.  Without it the engine serves without speculation, like the MTP-less
    // Qwen configs.
    strata::naive::DraftGeometry d;
    if (!a.draft.empty()) {
        try {
            strata::GgufFile df(a.draft);
            if (std::string err = strata::naive::read_draft_geometry(df, d, g); !err.empty()) {
                std::fprintf(stderr, "strata-naive: %s\n", err.c_str());
                return 2;
            }
            info("draft=naive_n05_flash_dspark layers=%lld block=%lld taps=%zu markov=%lld",
                 (long long) d.n_layers, (long long) d.block_size, d.target_layer_ids.size(),
                 (long long) d.markov_rank);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-naive: the draft: %s\n", e.what());
            return 2;
        }
    }

    if (a.check) {
        std::printf("ok: %s, %lld layers (%lld DSA + %lld SWA), %lld experts top-%lld %s/%s, %zu shards\n",
                    strata::naive::kArchMain, (long long) g.n_layers, (long long) g.n_dsa_layers(),
                    (long long) g.n_swa_layers(), (long long) g.n_expert, (long long) g.n_expert_used,
                    g.moe_scoring_func.c_str(), g.moe_topk_method.c_str(), model->size());
        if (!a.draft.empty())
            std::printf("ok: draft, %lld layers, block %lld, %zu taps, markov %lld\n", (long long) d.n_layers,
                        (long long) d.block_size, d.target_layer_ids.size(), (long long) d.markov_rank);
        delete model;
        return 0;
    }

    // ---- not --check: the serve path.  The protocol's shape, so a launcher sees the model it carries,
    // and an honest refusal at the point the missing work actually is (the trunk forward).
    const long long ctx = a.max_context > 0 ? a.max_context : (long long) g.n_ctx;
    std::printf("READY %lld\n", ctx);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || !std::cin) break;
        if (line.rfind("GEN", 0) == 0 || line.rfind("GENI", 0) == 0) {
            // The serve protocol's refusal line is `ERR <reason>` (src/program/generate.cpp prints the
            // same prefix); the server shows the text after the four characters "ERR ", so a word like
            // "ERROR" would reach the caller as "OR ...".  One ERR, then the reason: the engine stays
            // idle, exactly like a refusal in the qwen path, and the next request is read normally.
            std::fprintf(stdout, "ERR the strata-naive trunk forward is not implemented yet "
                                 "(docs/NAIVE.md tracks it); this build validates the model only\n");
            std::fflush(stdout);
        }
    }
    delete model;
    return 0;
}
