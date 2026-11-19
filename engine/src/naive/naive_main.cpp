// src/naive/naive_main.cpp - the Naive-N0.5-Flash engine path (the strata-naive executable).
//
// What this build does: it opens the model's GGUF shards, reads the geometry from the model's own
// metadata (naive/geometry.hpp), checks every tensor of the layout (naive/reader.hpp's inventory),
// reads the draft's geometry the same way, prints the facts (INFO lines, the serve protocol's shape),
// and serves: under --serve it runs the trunk (naive/runtime.hpp) over the real weights - the
// 48-layer loop with the DSA / window caches, the IQ2 experts, the sampled output - and speaks the
// serve protocol the web server supervises (GEN lines in, T / DONE lines out, STOP between them).
// It never mmaps less and trusts more: a model of another shape is refused by the key or tensor
// that differs, and a request that asks for what this build does not do (the draft, a picture) is
// refused with the reason instead of answered wrong.
//
// What it does NOT do: the DSpark draft still only validates.  Drafting stays off, like the
// MTP-less Qwen configs - docs/NAIVE.md tracks the drafter as its own next step.  `--check` is the
// validation the CI and the run script use; without it the binary validates first, then serves.
//
// Like the dsv41 path, this is its own executable: the Flash-Next kernels are compiled against one
// geometry (core/geometry_checks.hpp), and this shape shares none of it.
//
//   strata-naive --native <shard1.gguf> [--draft <draft.gguf>] [--check]
//   strata-naive --serve ...            (the serve protocol; INFO/READY, then GEN / STOP / QUIT)

#include "strata/naive/geometry.hpp"
#include "strata/naive/reader.hpp"
#include "strata/naive/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef STRATA_VERSION
#define STRATA_VERSION "0.0.0-naive"
#endif

namespace {

void usage() {
    std::fprintf(stderr,
                 "strata-naive --native <model.gguf (any shard)> [--draft <dspark-draft.gguf>]\n"
                 "               [--check] [--max-context N] [--threads N] [--session-history N]\n"
                 "               [--verbose] | --serve\n"
                 "\n"
                 "  --native PATH     the model's GGUF; a split model is opened by any of its shards\n"
                 "  --draft PATH      the DSpark draft GGUF (optional: without it the engine drafts nothing;\n"
                 "                    with it, it validates it - drafting itself is not built yet)\n"
                 "  --check           validate the model (geometry, layout, draft) and exit 0; nonzero + the\n"
                 "                    reason otherwise.  No GPU, no weights loaded: CI can run it anywhere.\n"
                 "  --max-context N   cap the context for the plan report (default: the model's own)\n"
                 "  --threads N       the GEMV row pool (default: every core this box reports)\n"
                 "  --session-history N\n"
                 "                    window-ring rewind headroom in rows (default 4096): a branch older\n"
                 "                    than the ring re-prefills the conversation instead\n"
                 "  --serve           the serve protocol's INFO/READY lines, then GEN / STOP / QUIT\n");
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
    int threads = 0;
    long long session_history = 4096;
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
        else if (s == "--threads") {
            std::string v;
            if (!value(v)) return false;
            a.threads = (int) std::strtol(v.c_str(), nullptr, 10);
        }
        else if (s == "--session-history") {
            std::string v;
            if (!value(v)) return false;
            a.session_history = std::strtoll(v.c_str(), nullptr, 10);
        }
        else if (s == "--help" || s == "-h") { usage(); std::exit(0); }
        else if (s == "--version") { std::printf("strata-naive %s\n", STRATA_VERSION); std::exit(0); }
        else { a.err = "unknown argument: " + s; usage(); return false; }
    }
    if (a.native.empty()) { a.err = "--native is required"; usage(); return false; }
    return true;
}

/// The per-layer KV bytes the cache plan needs: the DSA layers keep every position of the whole
/// history (K 192 + V 128 per kv head, BF16 - the cached values are bf16-rounded, the runtime
/// stores them as bf16 codes), the window layers keep `swa_window` + one chunk.  This is
/// the same arithmetic the model's own memory table projects (its README's "K/V histories, BF16"
/// row); printed as INFO so the launcher and the Monitor can plan, and so the number is checkable,
/// not magic.
double kv_cache_gib(const strata::naive::NaiveGeometry& g, long long ctx) {
    double bytes = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t nkv = g.kv_heads(l);
        const double per_pos = (double) nkv * (double) (g.head_dim_k + g.head_dim_v) * 2.0;   // BF16
        const double cells = g.is_dsa_layer(l) ? (double) ctx : (double) (g.swa_window + g.attn_chunk);
        bytes += per_pos * cells;
    }
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

using Clock = std::chrono::steady_clock;
double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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
    // Qwen configs; with it it validates it the same way - the DSpark forward is docs/NAIVE.md step 6.
    strata::naive::DraftGeometry d;
    if (!a.draft.empty()) {
        try {
            strata::GgufFile df(a.draft);
            if (std::string err = strata::naive::read_draft_geometry(df, d, g); !err.empty()) {
                std::fprintf(stderr, "strata-naive: %s\n", err.c_str());
                return 2;
            }
            info("draft=naive_n05_flash_dspark layers=%lld block=%lld taps=%zu markov=%lld drafting=off",
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

    // ---- not --check: the serve path.  The runtime first: its gate (runtime.hpp) refuses a shape
    // its caches and kernels were not written for, naming the key - before READY, so the launcher
    // sees a failed start, never a live engine that cannot answer.
    int threads = a.threads > 0 ? a.threads
                                : (int) std::thread::hardware_concurrency();
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads > 1) info("openmp=no threads=1 (this binary has no OpenMP: the rows run one at a time)");
#endif
    std::string rerr;
    strata::naive::NaiveRuntime rt(*model, g, threads, a.session_history, rerr);
    if (!rt.ok()) {
        std::fprintf(stderr, "strata-naive: %s\n", rerr.c_str());
        delete model;
        return 2;
    }

    // The end-of-turn ids, from the model's own metadata (measured on MQ87: eos_token_id 151645,
    // the chat template's <|im_end|>).  The server also stops on the decoded stop strings; the
    // engine stops the session exactly at the token, so nothing past the turn is ever produced.
    std::vector<int64_t> eos_ids;
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
        const strata::MetaValue* v = model->meta().get(key);
        if (v && v->is_num()) eos_ids.push_back((int64_t) v->num());
    }
    {
        std::string s;
        for (size_t i = 0; i < eos_ids.size(); ++i) s += (i ? "," : "") + std::to_string(eos_ids[i]);
        info("eos=%s", s.empty() ? "none" : s.c_str());
    }

    const long long ctx = a.max_context > 0 ? a.max_context : (long long) g.n_ctx;
    info("threads=%d session_history=%lld runtime_cache_gib_at_ctx=%.2f", threads,
         (long long) a.session_history, rt.cache_gib_at(ctx));
    std::printf("READY %lld stop\n", ctx);
    std::fflush(stdout);

    std::atomic<bool> stop_req{false};
    strata::naive::Sampler sampler;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || !std::cin) break;
        if (line == "STOP") { stop_req.store(true); continue; }
        if (line.rfind("GENI ", 0) == 0) {
            std::printf("ERR this build serves text only (no --vision path): send GEN with token ids\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            if (!line.empty() && line[0] != '#')
                std::printf("ERR expected: GEN <max_new> <id,id,...> or QUIT\n");
            std::fflush(stdout);
            continue;
        }
        strata::naive::GenReq q;
        if (std::string err; !strata::naive::parse_gen_line(line, q, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) q.ids.size() > ctx) {
            std::printf("ERR the prompt is %zu tokens, this context is %lld\n", q.ids.size(), ctx);
            std::fflush(stdout);
            continue;
        }
        stop_req.store(false);
        sampler.reseed(q.seed);

        // ---- resume the session where this prompt continues it; re-prefill from the first token
        // that differs (or from 0 when the rings cannot reach back that far).  One row before the
        // branch point is always re-fed: it is rewritten with the same bits, and it is what makes
        // this request's logits fresh even when the prompt already sits fully in the cache (a
        // retry): the head has to run for the last prompt row before the first sample.
        const int64_t same = rt.common_prefix(q.ids);
        const int64_t d = same > 0 ? same - 1 : 0;
        if (!rt.rewind(d)) {
            rt.reset();
            if (a.verbose) info("session rewind %zu/%lld refused: re-prefilling from token 0",
                                q.ids.size(), (long long) d);
        }
        const auto t0 = Clock::now();
        bool failed = false;
        const int64_t from = rt.session_len();
        for (size_t i = (size_t) from; i < q.ids.size(); ++i) {
            const bool last = i + 1 == q.ids.size();
            if (const std::string e = rt.forward_row(q.ids[i], last && q.max_new > 0); !e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::fflush(stdout);
                failed = true;
                break;
            }
            const bool last_row = (i + 1 == q.ids.size()) ||
                                  (((int64_t) i - from + 1) % 16 == 0) || stop_req.load();
            if (last_row) {
                std::printf("PP %lld %lld %.1f\n", (long long) rt.session_len(), (long long) q.ids.size(),
                            ms_since(t0));
                std::fflush(stdout);
            }
            if (stop_req.load()) break;                      // the client left mid-prompt: end the request
        }
        if (failed) { rt.reset(); continue; }
        const double prompt_ms = ms_since(t0);
        if ((int64_t) rt.session_len() != (int64_t) q.ids.size()) {   // stopped during the prompt read
            std::printf("DONE 0 %zu %.1f 0.0 cancel\n", q.ids.size(), prompt_ms);
            std::fflush(stdout);
            continue;
        }
        if (q.max_new <= 0) {                                // prefill only (the split path's prefill=1): no token
            std::printf("DONE 0 %zu %.1f 0.0 length\n", q.ids.size(), prompt_ms);
            std::fflush(stdout);
            continue;
        }

        // ---- decode.  One T line per token, flushed as it is made; the session holds prompt plus
        // output, so a later request that continues this turn resumes at this exact point.
        const auto t1 = Clock::now();
        int64_t produced = 0;
        const char* fin = "length";
        while (produced < q.max_new) {
            const auto& lg = rt.logits();
            bool nonfinite = false;
            for (float f : lg)
                if (!(f == f)) { nonfinite = true; break; }
            if (nonfinite) { fin = "nonfinite"; break; }     // #879's rule: that window's tokens are not emitted
            const int32_t y = sampler.sample(lg.data(), (int) lg.size(), q, rt.session());
            std::printf("T %d\n", (int) y);
            std::fflush(stdout);
            ++produced;
            if (a.verbose) info("tok %lld/%lld %d %.0f tok/min", (long long) produced, q.max_new, (int) y,
                                produced * 60000.0 / std::max(1.0, ms_since(t1)));
            if (std::find(eos_ids.begin(), eos_ids.end(), (int64_t) y) != eos_ids.end()) { fin = "stop"; break; }
            if (stop_req.load()) { fin = "cancel"; break; }
            if (rt.session_len() >= ctx) { fin = "length"; break; }
            if (const std::string e = rt.forward_row(y, true); !e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::fflush(stdout);
                failed = true;
                break;
            }
        }
        if (failed) {                                         // the ERR line above already ended this request:
            rt.reset();                                       // a trailing DONE would be a second ending
            continue;
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %lld\n", (long long) produced, q.ids.size(), prompt_ms,
                    ms_since(t1), fin, (long long) from);   // the DONE's fields, positional, as _parse_done reads them
        std::fflush(stdout);
    }
    delete model;
    return 0;
}
