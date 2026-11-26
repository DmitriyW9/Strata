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
#include <memory>
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
                 "  --weights MODE    auto (default): load every weight resident when the RAM fits it, else\n"
                 "                    keep the experts resident and read the rest through the page cache;\n"
                 "                    resident: refuse to start unless everything fits; cache: the old\n"
                 "                    mmap path, nothing resident; vram-split: place whole expert stacks\n"
                 "                    on the cards EXCLUSIVELY (a stack a card holds gets no RAM copy),\n"
                 "                    the rest of the experts and the dense set in RAM; needs a CUDA build\n"
                 "  --gpus LIST       comma-separated CUDA card ordinals the experts may be placed on\n"
                 "                    (default: every card the probe sees; needs a -DSTRATA_NAIVE_CUDA=ON build)\n"
                 "  --layer-split auto  accepted for the launcher's multi-card default: this engine places\n"
                 "                    its layers and expert stacks itself (pool.hpp's plan); any other\n"
                 "                    value is refused with the reason\n"
                 "  --no-gpu          keep the experts on the CPU even when a card is there\n"
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
    std::string weights = "auto";  // auto | resident | cache | vram-split (pool.hpp's tiers)
    std::string gpus;               // --gpus "0,1"; empty = every card the probe sees
    bool no_gpu = false;            // keep the CPU path even when a card is there
    bool layer_split_auto = false;  // --layer-split auto: the launcher's multi-card default, a no-op here
    std::string err;
};

/// What the OS says it can still give this process without swapping (/proc/meminfo's MemAvailable,
/// in bytes; 0 when the file cannot be read, which makes the plan refuse rather than guess).  The
/// plan (pool.hpp) never reads /proc itself - the caller probes and passes the number, so a test
/// can drive the plan with a made-up box.
uint64_t ram_available_bytes() {
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        uint64_t kb = 0;
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, "MemAvailable:", 13) == 0) {
                std::sscanf(line + 13, "%llu", (unsigned long long*) &kb);
                break;
            }
        std::fclose(f);
        return kb * 1024ull;
    }
#ifdef __APPLE__
    return 8ull << 30;  // no /proc on the Mac CI box: a small figure, so the plan stays conservative
#else
    return 0;
#endif
}

std::vector<int> parse_gpus(const std::string& s) {
    std::vector<int> out;
    size_t p = 0;
    while (p < s.size()) {
        const size_t c = s.find(',', p);
        const std::string tok = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
        if (!tok.empty()) out.push_back((int) std::strtol(tok.c_str(), nullptr, 10));
        if (c == std::string::npos) break;
        p = c + 1;
    }
    return out;
}

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
        else if (s == "--weights") {
            if (!value(a.weights)) return false;
            if (a.weights != "auto" && a.weights != "resident" && a.weights != "cache" &&
                a.weights != "vram-split") {
                a.err = "--weights wants auto, resident, cache or vram-split";
                return false;
            }
        }
        else if (s == "--gpus") { if (!value(a.gpus)) return false; }
        else if (s == "--layer-split") {
            // The web server adds this for every multi-card config (serve/server.py:
            // layer_split_of). "auto" is the launcher's default and means "the engine decides":
            // that is exactly what this engine does (pool.hpp's plan places whole expert stacks
            // on cards or in RAM), so it is accepted and said so in the log. A hand-written
            // split ("24") names layers for the staged qwen4exp engine; this shape has no
            // staged layers, so it is refused with the reason, never ignored.
            std::string v;
            if (!value(v)) return false;
            if (v != "auto") {
                a.err = "--layer-split " + v + ": strata-naive places its layers and expert stacks"
                        " itself (pool.hpp's plan); only auto is meaningful here";
                return false;
            }
            a.layer_split_auto = true;
        }
        else if (s == "--no-gpu") a.no_gpu = true;
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
    if (!parse(argc, argv, a)) {
        // The reason first, then the usage parse() already printed for the flags that show it:
        // a wrong value should say what it was, the way every refusal in this binary does.
        if (!a.err.empty()) std::fprintf(stderr, "strata-naive: %s\n", a.err.c_str());
        return 2;
    }

    info("engine=strata-naive version=%s", STRATA_VERSION);
    if (a.layer_split_auto)
        info("layer_split=auto placement=this engine's own plan (pool.hpp: whole expert stacks on"
             " cards or in RAM, dense in RAM)");

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

    // ---- the memory plan (pool.hpp): where every weight would live at start-up, from the file's own
    // tensor table and the memory this box reports.  Nothing is loaded to make it - the cards are
    // probed (cudaMemGetInfo, when this binary was built with CUDA) and RAM read from /proc/meminfo -
    // and the answer prints as INFO so both `--check` and a real start report the same numbers.  This
    // is the "how is the memory planned" answer the user asked for, in the log the launcher reads.
    strata::naive::NaiveGpu gpu;
    std::vector<strata::naive::CardMem> cards;
    if (a.check) {
        // --check keeps its documented contract: no GPU touched, no weights read. The plan prints
        // its RAM half here; what the cards would hold depends on their free memory, which only a
        // real start probes.
        info("plan_cards=not_probed_under_check");
    } else if (!a.no_gpu && strata::naive::NaiveGpu::built_with_cuda()) {
        if (const std::string gerr = gpu.init(parse_gpus(a.gpus)); !gerr.empty()) {
            info("gpu=off reason=%s", gerr.c_str());
        } else {
            for (const strata::naive::GpuCard& c : gpu.cards())
                cards.push_back({c.index, c.total_bytes, c.free_bytes});
            info("gpu=on cards=%zu", cards.size());
        }
    } else if (!a.no_gpu) {
        info("gpu=off reason=this build has no CUDA kernels (built with -DSTRATA_NAIVE_CUDA=OFF)");
    }
    const uint64_t ram_available = ram_available_bytes();
    const bool expert_only = false;  // the fp32-dense tier is the only one this build serves
    const strata::naive::PoolPlan plan =
        strata::naive::plan_pool(*model, g, ram_available, cards, expert_only);
    info("plan_weights=%s plan_expert_gib=%.2f plan_dense_fp32_gib=%.2f plan_dense_raw_gib=%.2f "
         "plan_ram_gib=%.2f plan_ram_available_gib=%.2f plan_fits_all=%d plan_fits_experts=%d",
         a.weights.c_str(), plan.expert_gib(), plan.dense_fp32_gib(), plan.dense_raw_gib(),
         plan.ram_gib(), (double) ram_available / (1024.0 * 1024.0 * 1024.0), plan.fits_all ? 1 : 0,
         plan.fits_raw ? 1 : 0);
    for (size_t c = 0; c < cards.size() && c < plan.card_expert_bytes.size(); ++c)
        info("plan_card%d_expert_gib=%.2f (of %.2f free GiB)", cards[c].index,
             (double) plan.card_expert_bytes[c] / (1024.0 * 1024.0 * 1024.0),
             (double) cards[c].free_bytes / (1024.0 * 1024.0 * 1024.0));
    if (!plan.note.empty()) info("plan_note=%s", plan.note.c_str());

    // The exclusive plan (--weights vram-split, pool.hpp's plan_split): where each whole expert
    // stack would live when a card holds it INSTEAD of RAM. Computed whenever cards were probed or
    // the tier asks for it; --check (no cards) reports the RAM half, like plan_pool above.
    std::vector<strata::naive::ExpertStack> stacks;
    strata::naive::SplitPlan split;
    bool have_split = false;
    if (a.weights == "vram-split" || !cards.empty()) {
        std::string serr;
        stacks = strata::naive::expert_stacks(*model, g, serr);
        if (serr.empty()) {
            split = strata::naive::plan_split(stacks, plan.dense_fp32_bytes, ram_available, cards);
            have_split = true;
            info("split_vram_expert_gib=%.2f split_ram_expert_gib=%.2f split_skipped_gib=%.2f "
                 "split_fits_all=%d split_dense_fits=%d",
                 split.vram_expert_gib(), split.ram_expert_gib(), split.skipped_expert_gib(),
                 split.fits_all ? 1 : 0, split.dense_fits ? 1 : 0);
            for (size_t c = 0; c < cards.size() && c < split.card_expert_bytes.size(); ++c)
                info("split_card%d_expert_gib=%.2f", cards[c].index,
                     (double) split.card_expert_bytes[c] / (1024.0 * 1024.0 * 1024.0));
            if (!split.note.empty()) info("split_note=%s", split.note.c_str());
        } else {
            info("split_note=%s", serr.c_str());
        }
    }

    if (a.check) {
        std::printf("ok: %s, %lld layers (%lld DSA + %lld SWA), %lld experts top-%lld %s/%s, %zu shards\n",
                    strata::naive::kArchMain, (long long) g.n_layers, (long long) g.n_dsa_layers(),
                    (long long) g.n_swa_layers(), (long long) g.n_expert, (long long) g.n_expert_used,
                    g.moe_scoring_func.c_str(), g.moe_topk_method.c_str(), model->size());
        std::printf("plan: experts %.2f GiB + dense fp32 %.2f GiB = %.2f GiB RAM; %s; %zu card(s)%s\n",
                    plan.expert_gib(), plan.dense_fp32_gib(), plan.ram_gib(),
                    plan.fits_all ? "everything fits resident"
                                   : (plan.fits_raw ? "experts resident, dense on disk" : "too little RAM: all on disk"),
                    cards.size(), plan.note.empty() ? "" : (" - " + plan.note).c_str());
        if (!a.draft.empty())
            std::printf("ok: draft, %lld layers, block %lld, %zu taps, markov %lld\n", (long long) d.n_layers,
                        (long long) d.block_size, d.target_layer_ids.size(), (long long) d.markov_rank);
        delete model;
        return 0;
    }

    // ---- not --check: the serve path.  The weights first, per --weights (pool.hpp's tiers):
    //   cache       - nothing resident, every read goes through loader.hpp's mmap windows (the old path);
    //   resident    - refuse to start unless the whole file fits the room the plan counted;
    //   auto        - everything when it fits, else the largest set that does (always the experts: they
    //                 are the bytes a cold session pages), the rest through the mmap windows.
    //   vram-split  - the exclusive tier (plan_split): whole expert stacks copied to the cards straight
    //                 from the mmap (a stack a card holds gets NO RAM copy), the rest of the experts and
    //                 the dense set in the RAM pool.  Refuses to start without a CUDA build or without
    //                 room for the dense set - it never silently becomes the copy tier.
    // The pool prints what it actually holds, so the INFO plan above is checkable against it.
    std::unique_ptr<strata::naive::NaivePool> pool;
    if (a.weights == "vram-split") {
        if (!gpu.ok() || cards.empty()) {
            std::fprintf(stderr, "strata-naive: --weights vram-split needs a card: %s\n",
                         gpu.reason().empty() ? "no card was probed (--no-gpu?)" : gpu.reason().c_str());
            delete model;
            return 2;
        }
        if (!have_split) {
            std::fprintf(stderr, "strata-naive: --weights vram-split: the split plan refused\n");
            delete model;
            return 2;
        }
        if (!split.dense_fits) {
            std::fprintf(stderr, "strata-naive: --weights vram-split refuses to start: %s\n",
                         split.note.empty() ? "the dense set does not fit the RAM room" : split.note.c_str());
            delete model;
            return 2;
        }
        // The card half first: upload_from_mmap copies each placed stack from the loader's mmap
        // pointer to its card, never through a buffer this engine owns.  A stack whose upload fails
        // (a card that changed under the plan) is named in vram_note and reads the mmap windows -
        // the runtime's answer does not move with the placement (gpu.hpp), so this is honest.
        uint64_t vram_placed = 0;
        int vram_stacks = 0, vram_failed = 0;
        for (const strata::naive::SplitPlacement& pl : split.placements) {
            if (pl.card < 0) continue;
            for (const strata::naive::ExpertStack& st : stacks) {
                if (st.name != pl.name) continue;
                const strata::TensorInfo* t = model->find(st.name);
                std::string uerr;
                if (gpu.upload_from_mmap(st, model->shard(st.shard).tensor_data(*t),
                                         cards[pl.card].index, uerr) < 0) {
                    info("vram_note=%s (reads the mmap windows: %s)", st.name.c_str(), uerr.c_str());
                    ++vram_failed;
                } else {
                    vram_placed += st.bytes;
                    ++vram_stacks;
                }
                break;
            }
        }
        // The RAM half: the whole dense set (file order: globals, then per layer) plus the expert
        // stacks plan_split placed on RAM.  The stacks plan_split skipped read the mmap windows.
        std::vector<std::string> ram_names;
        {
            for (const strata::naive::TensorSpec& s : strata::naive::main_globals())
                if (strata::naive::role_of(s.name) == strata::naive::Role::Dense) ram_names.push_back(s.name);
            std::vector<strata::naive::TensorSpec> ts;
            char buf[160];
            for (int64_t l = 0; l < g.n_layers; ++l) {
                strata::naive::layer_tensors(l, g, ts);
                for (const strata::naive::TensorSpec& s : ts) {
                    std::snprintf(buf, sizeof buf, s.name, (int) l);
                    if (strata::naive::role_of(buf) == strata::naive::Role::Dense) ram_names.push_back(buf);
                }
            }
        }
        for (const strata::naive::SplitPlacement& pl : split.placements)
            if (pl.card < 0) ram_names.push_back(pl.name);
        pool.reset(new strata::naive::NaivePool(*model, g));
        if (const std::string perr = pool->load(ram_names); !perr.empty()) {
            std::fprintf(stderr, "strata-naive: the split pool: %s\n", perr.c_str());
            delete model;
            return 2;
        }
        info("weights=vram-split vram_expert_gib=%.2f vram_expert_stacks=%d vram_failed=%d "
             "ram_expert_gib=%.2f dense_gib=%.2f skipped_gib=%.2f",
             (double) vram_placed / (1024.0 * 1024.0 * 1024.0), vram_stacks, vram_failed,
             (double) pool->resident_expert_bytes() / (1024.0 * 1024.0 * 1024.0),
             (double) pool->resident_dense_bytes() / (1024.0 * 1024.0 * 1024.0),
             (double) split.skipped_expert_bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (a.weights != "cache") {
        if (a.weights == "resident" && !plan.fits_all) {
            std::fprintf(stderr, "strata-naive: --weights resident refuses to start: %s\n",
                         plan.note.empty() ? "the model does not fit this box" : plan.note.c_str());
            delete model;
            return 2;
        }
        pool.reset(new strata::naive::NaivePool(*model, g));
        std::string perr;
        if (plan.fits_all) {
            perr = pool->load_all();
        } else if (plan.fits_raw) {
            const strata::naive::ResidentSet set =
                strata::naive::select(*model, g, plan.expert_bytes, /*include_dense=*/false);
            info("plan_resident_gib=%.2f plan_resident_tensors=%zu plan_skipped_gib=%.2f",
                 (double) set.resident_bytes / (1024.0 * 1024.0 * 1024.0), set.resident.size(),
                 (double) set.skipped_bytes / (1024.0 * 1024.0 * 1024.0));
            if (!set.note.empty()) info("plan_select_note=%s", set.note.c_str());
            perr = pool->load(set.resident);
        } else {
            info("weights=cache reason=%s",
                 plan.note.empty() ? "the plan sees no room" : plan.note.c_str());
            pool.reset();
        }
        if (!perr.empty()) {
            std::fprintf(stderr, "strata-naive: the resident pool: %s\n", perr.c_str());
            delete model;
            return 2;
        }
        if (pool)
            info("weights=resident resident=%zu expert_gib=%.2f dense_gib=%.2f",
                 pool->resident_count(), (double) pool->resident_expert_bytes() / (1024.0 * 1024.0 * 1024.0),
                 (double) pool->resident_dense_bytes() / (1024.0 * 1024.0 * 1024.0));
    }

    // ---- the card tier: the expert stacks the pool holds, placed on the cards the plan made room
    // for - file order, one card's room at a time, a stack never split.  A stack that will not fit
    // or a card that refuses stays on the CPU pool copy: the runtime answers those rows there, and
    // the numbers do not move with the placement (gpu.hpp).  Not under vram-split: that tier placed
    // its cards above, straight from the mmap, and must not copy the pool's RAM residents a second
    // time - the split's whole point is that no stack lives in both memories.
    if (pool && a.weights != "vram-split" && gpu.ok() && !cards.empty()) {
        std::vector<uint64_t> room(cards.size());
        for (size_t c = 0; c < cards.size(); ++c)
            room[c] = strata::naive::card_room_bytes(cards[c], strata::naive::kPlanCardReserveBytes);
        size_t card = 0;
        uint64_t placed = 0;
        int placed_stacks = 0;
        std::vector<bool> touched(room.size(), false);
        std::string first_err;
        for (const strata::naive::NaivePool::ResidentExpert& st : pool->resident_experts()) {
            while (card < room.size() && room[card] < st.bytes_len) ++card;
            if (card == room.size()) break;  // no card has room for this or any later stack
            std::string uerr;
            if (gpu.upload(st, cards[card].index, uerr) < 0) {
                if (first_err.empty()) first_err = uerr;
                continue;  // the CPU copy answers this stack; the plan's INFO stays honest
            }
            touched[card] = true;
            room[card] -= st.bytes_len;
            placed += st.bytes_len;
            ++placed_stacks;
        }
        int placed_cards = 0;
        for (bool t : touched) placed_cards += t;
        info("vram_expert_gib=%.2f vram_expert_stacks=%d cards_used=%d",
             (double) placed / (1024.0 * 1024.0 * 1024.0), placed_stacks, placed_cards);
        if (!first_err.empty()) info("vram_note=%s (the stacks it names stay on the CPU copy)",
                                     first_err.c_str());
    }

    // The runtime next: its gate (runtime.hpp) refuses a shape its caches and kernels were not
    // written for, naming the key - before READY, so the launcher sees a failed start, never a live
    // engine that cannot answer.
    int threads = a.threads > 0 ? a.threads
                                : (int) std::thread::hardware_concurrency();
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads > 1) info("openmp=no threads=1 (this binary has no OpenMP: the rows run one at a time)");
#endif
    std::string rerr;
    strata::naive::NaiveRuntime rt(*model, g, threads, a.session_history, rerr, pool.get(),
                                   pool ? &gpu : nullptr);
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
