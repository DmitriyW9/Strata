// terminal/tools/llama/strata_llama.cpp - the strata-llama engine path (the strata-llama executable).
//
// What this build does: it runs, through llama.cpp's own library (third_party/llama.cpp, with the
// GigaChat4 model added), any model llama.cpp itself knows - the shapes the Flash-Next kernels were
// not written for, first of them GigaChat4. Like strata-naive it is its own executable: it opens the
// model's GGUF (a split set by any of its shards), prints the facts (INFO lines, stderr), says READY
// (stdout), and serves the serve protocol the web server supervises: GEN lines in, T / DONE lines
// out, STOP between them, QUIT ends. The prompt arrives as token ids (the server tokenizes); this
// engine forwards them, keeps one session, and streams sampled tokens back.
//
// The session is llama.cpp's own memory: a request whose prompt continues the session keeps the
// shared prefix (llama_memory_seq_keep/rm, the same rule strata-naive's rings follow) and re-feeds
// only the new tail; a STOP drops what the request produced; an error clears it.
//
// The GEN line's shape and the sampler are copied from strata/naive/runtime.hpp (GenReq,
// parse_gen_line, Sampler) so the two engines answer the same request with the same tokens; that
// file is the source of truth and this copy follows it.
//
//   strata-llama --native <shard1.gguf> [--max-context N] [--threads N] [--n-gpu-layers N]
//                [--main-gpu K] [--tensor-split a,b,...] [--cache-type-k f16] [--cache-type-v f16]
//                [--check] [--verbose]
//
// Flags the launcher adds for the resident Strata engine (--layer-split, --vram-reserve-mib,
// --vision, ...) are accepted and answered with a line when they mean nothing here, like the
// naive engine's --layer-split auto - recommend, never crash.

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0.0.0-llama"
#endif

namespace {

void usage() {
    std::fprintf(stderr,
                 "strata-llama --native <model.gguf (any shard)> [--max-context N] [--threads N]\n"
                 "                 [--n-gpu-layers N | -1 for all] [--main-gpu K] [--tensor-split a,b,...]\n"
                 "                 [--cache-type-k T] [--cache-type-v T] [--check] [--verbose]\n"
                 "\n"
                 "  --native PATH      the model's GGUF; a split model is opened by any of its shards\n"
                 "  --max-context N    the context to run (0: the model's own trained size)\n"
                 "  --n-gpu-layers N   layers to keep on the cards; -1 (default): all of them\n"
                 "  --check            load the model, print the INFO facts, and exit 0\n"
                 "  --serve            the serve protocol's INFO/READY lines, then GEN / STOP / QUIT\n");
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

/// llama.cpp's library logs its every step at DEBUG level ("create_tensor: loading tensor", "CUDA Graph id reused") -
/// a 106k-token prompt made 160k such lines in the run log of 2026-11-27, the server collecting them one by one.
/// Without --serve --verbose the callback keeps INFO and above, which is what the server and the run log are for.
void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_INFO) std::fputs(text, stderr);
}

using Clock = std::chrono::steady_clock;
double ms_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ---------------------------------------------------------------- the GEN line's request
// Copied from strata/naive/runtime.hpp (the source of truth) so both engines read one line the
// same way: `GEN <max_new> [key=value ...] <id,id,...>`.
struct GenReq {
    int64_t max_new = 0;
    std::vector<int32_t> ids;
    // Sampling.  Absent keys keep the engine greedy: temperature 0, like the main engine's default.
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 40;
    unsigned long long seed = 0;                             // 0: the engine's own entropy at first use
    // Repetition penalties, llama's convention (the main engine's key set).
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_presence = 0.0f;
    int64_t pen_from = -1;
};

bool parse_gen_line(const std::string& line, GenReq& q, std::string& err) {
    const char* p = line.c_str() + 4;                        // past "GEN "
    char* endp = nullptr;
    q = GenReq{};
    q.max_new = std::strtoll(p, &endp, 10);
    if (endp == p) { err = "GEN needs a max_new"; return false; }
    for (;;) {
        while (*endp == ' ') ++endp;
        const char* start = endp;
        while (*endp != '\0' && *endp != ' ') ++endp;
        if (endp == start) break;
        const std::string tok(start, (size_t) (endp - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {                       // the ids: the first token without an '='
            size_t pos = 0;
            while (pos < tok.size()) {
                const size_t c = tok.find(',', pos);
                const std::string one = tok.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                if (!one.empty()) q.ids.push_back((int32_t) std::strtol(one.c_str(), nullptr, 10));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
            break;
        }
        const std::string key = tok.substr(0, eq);
        const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
        if (key == "temperature") q.temperature = fv;
        else if (key == "top_p") q.top_p = fv;
        else if (key == "top_k") q.top_k = std::atoi(tok.c_str() + eq + 1);
        else if (key == "min_p") q.min_p = fv;
        else if (key == "seed") q.seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
        else if (key == "penalty_last_n") q.penalty_last_n = std::atoi(tok.c_str() + eq + 1);
        else if (key == "penalty_repeat") q.penalty_repeat = fv;
        else if (key == "penalty_freq") q.penalty_freq = fv;
        else if (key == "penalty_present") q.penalty_presence = fv;
        else if (key == "pen_from") q.pen_from = std::atoll(tok.c_str() + eq + 1);
        // every other key is this engine's to ignore, like the main parser skips unknown keys.
    }
    if (q.ids.empty()) { err = "GEN carries no prompt ids"; return false; }
    return true;
}

// ---------------------------------------------------------------- the sampler
// The naive engine's (runtime.hpp): greedy at temperature 0; otherwise penalties, temperature,
// top-k, min-p, top-p, a draw with a seeded mt19937_64. One seed replays one reply.
class Sampler {
public:
    int32_t sample(const float* logits, int n, const GenReq& q, const std::vector<int32_t>& history) {
        work_.assign(logits, logits + n);
        apply_penalties(history, q);
        if (q.temperature <= 0.0f) {
            int best = 0;
            for (int i = 1; i < n; ++i)
                if (work_[i] > work_[best]) best = i;        // first max wins
            return best;
        }
        const float inv = 1.0f / q.temperature;
        for (int i = 0; i < n; ++i) work_[i] = work_[i] * inv;
        const int k = std::max(1, std::min(q.top_k <= 0 ? n : q.top_k, n));
        cand_.resize((size_t) n);
        for (int i = 0; i < n; ++i) cand_[(size_t) i] = {work_[i], i};
        std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(),
                          [](const Cand& a, const Cand& b) { return a.v > b.v; });
        const float top = cand_[0].v;
        int m = k;
        if (q.min_p > 0.0f) {                                // min-p over the top-k set
            const float cut = top + std::log(std::max(q.min_p, 1e-20f));  // p_i >= min_p * p_max
            while (m > 1 && cand_[(size_t) m - 1].v < cut) --m;
        }
        if (q.top_p < 1.0f && m > 1) {                       // top-p over the same order
            std::vector<float> pr((size_t) m);
            float mx = cand_[0].v, sum = 0.0f;
            for (int i = 0; i < m; ++i) { pr[(size_t) i] = std::exp(cand_[(size_t) i].v - mx); sum += pr[(size_t) i]; }
            float acc = 0.0f;
            int keep = m;
            for (int i = 0; i < m; ++i) {
                acc += pr[(size_t) i] / sum;
                if (acc >= q.top_p) { keep = i + 1; break; }
            }
            m = std::max(1, keep);
        }
        std::vector<double> w((size_t) m);
        double mx = cand_[0].v, tot = 0.0;
        for (int i = 0; i < m; ++i) { w[(size_t) i] = std::exp((double) cand_[(size_t) i].v - mx); tot += w[(size_t) i]; }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double pick = u(rng) * tot, acc = 0.0;
        for (int i = 0; i < m; ++i) {
            acc += w[(size_t) i];
            if (pick < acc) return cand_[(size_t) i].id;
        }
        return cand_[(size_t) m - 1].id;
    }

    void reseed(unsigned long long seed) {                   // seed 0: entropy once, then carried
        if (seed != 0) rng.seed(seed);
        else if (!seeded_) {
            std::random_device rd;
            rng.seed(((unsigned long long) rd() << 32) ^ rd());
            seeded_ = true;
        }
    }

private:
    struct Cand { float v; int32_t id; };
    std::mt19937_64 rng{1};
    bool seeded_ = false;
    std::vector<float> work_;
    std::vector<Cand> cand_;

    void apply_penalties(const std::vector<int32_t>& history, const GenReq& q) {
        if (q.penalty_last_n <= 0 || history.empty()) return;
        if (q.penalty_repeat == 1.0f && q.penalty_freq == 0.0f && q.penalty_presence == 0.0f) return;
        const int64_t from = q.pen_from >= 0 ? q.pen_from : 0;
        const int64_t lo = std::max<int64_t>(from, (int64_t) history.size() - q.penalty_last_n);
        std::vector<int> counts(work_.size(), 0);
        for (int64_t i = lo; i < (int64_t) history.size(); ++i) {
            const int32_t t = history[(size_t) i];
            if (t >= 0 && t < (int32_t) work_.size()) ++counts[(size_t) t];
        }
        for (size_t t = 0; t < counts.size(); ++t) {
            if (!counts[t]) continue;
            if (q.penalty_repeat != 1.0f)
                work_[t] = work_[t] > 0.0f ? work_[t] / q.penalty_repeat : work_[t] * q.penalty_repeat;
            work_[t] -= q.penalty_freq * counts[t] + q.penalty_presence;
        }
    }
};

// ---------------------------------------------------------------- the engine's options
struct Args {
    std::string model;
    int64_t max_context = 0;
    int threads = 0;                                        // 0: llama.cpp's own default
    int n_gpu_layers = -1;                                  // -1: all of them, llama's rule
    int main_gpu = 0;
    std::vector<float> tensor_split;                        // empty: llama.cpp spreads them itself
    std::vector<int32_t> stop_tokens;                       // extra ids that end a turn, kept out of the text
    std::string cache_k = "f16", cache_v = "f16";
    int n_batch = 2048, n_ubatch = 512;
    bool check = false, verbose = false;
};

enum ggml_type parse_type(const std::string& s) {
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "iq4_nl") return GGML_TYPE_IQ4_NL;
    return GGML_TYPE_COUNT;
}

/// The engine's own options, then the launcher's additions it must swallow without complaint
/// (strata-naive answers `--layer-split auto` the same way).  An unknown option is a usage error.
bool parse_args(int argc, char** argv, Args& a, std::string& err) {
    auto need = [&](int& i, const char* what) -> const char* {
        if (i + 1 >= argc) { err = std::string(what) + " needs a value"; return nullptr; }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const char* v;
        if (s == "--native" || s == "-m") {
            if (!(v = need(i, "--native"))) return false;
            a.model = v;
        } else if (s == "--max-context" || s == "-c") {
            if (!(v = need(i, "--max-context"))) return false;
            a.max_context = std::atoll(v);
        } else if (s == "--threads" || s == "-t") {
            if (!(v = need(i, "--threads"))) return false;
            a.threads = std::atoi(v);
        } else if (s == "--threads-batch") {
            if (!(v = need(i, "--threads-batch"))) return false;
            (void) v;                                        // llama.cpp's default uses them all
        } else if (s == "--batch") {
            if (!(v = need(i, "--batch"))) return false;
            if (std::atoi(v) > 1) info("batch=%d ignored: strata-llama runs one request at a time", std::atoi(v));
        } else if (s == "--ubatch") {
            if (!(v = need(i, "--ubatch"))) return false;
            a.n_ubatch = std::atoi(v);
        } else if (s == "--n-gpu-layers" || s == "-ngl") {
            if (!(v = need(i, "--n-gpu-layers"))) return false;
            a.n_gpu_layers = std::atoi(v);
        } else if (s == "--main-gpu") {
            if (!(v = need(i, "--main-gpu"))) return false;
            a.main_gpu = std::atoi(v);
        } else if (s == "--tensor-split") {
            if (!(v = need(i, "--tensor-split"))) return false;
            size_t pos = 0;
            while (pos < std::strlen(v)) {
                const char* end = v + pos;
                while (*end && *end != ',') ++end;
                a.tensor_split.push_back((float) std::strtod(v + pos, (char**) &end));
                pos = (size_t) (end - v) + (*end ? 1 : 0);
            }
        } else if (s == "--stop-token") {
            if (!(v = need(i, "--stop-token"))) return false;
            a.stop_tokens.push_back((int32_t) std::atoi(v));
        } else if (s == "--cache-type-k") {
            if (!(v = need(i, "--cache-type-k"))) return false;
            a.cache_k = v;
        } else if (s == "--cache-type-v") {
            if (!(v = need(i, "--cache-type-v"))) return false;
            a.cache_v = v;
        } else if (s == "--check") {
            a.check = true;
        } else if (s == "--verbose") {
            a.verbose = true;
        } else if (s == "--serve" || s == "--split-skip-if-fits" || s == "--vram-elastic" || s == "--vision") {
            // the protocol's mode (serving is this build's only mode) and the launcher's additions
        } else if (s == "--layer-split" || s == "--vram-reserve-mib" || s == "--vram-reserve-later-mib" ||
                   s == "--vram-segment-mib" || s == "--pool-cores" || s == "--gpus" || s == "--session-history" ||
                   s == "--weights" || s == "--expert-cache" || s == "--prefill") {
            if (!(v = need(i, s.c_str()))) return false;
            info("%s=%s has no meaning for strata-llama (llama.cpp plans its own split): kept, ignored",
                 s.c_str(), v);
        } else if (s == "--parallel") {
            if (!(v = need(i, "--parallel"))) return false;
            if (std::atoi(v) > 1) info("parallel=%s ignored: strata-llama runs one request at a time", v);
        } else if (s == "--draft" || s == "--mtp-draft-vocab" || s == "--ple-gguf" || s == "--expert-profile") {
            if (!(v = need(i, s.c_str()))) return false;
            err = s + " names a drafter or a pack this engine does not read: " + v;
            return false;
        } else {
            err = "unknown option: " + s;
            return false;
        }
    }
    if (a.model.empty()) { err = "--native names the model's GGUF"; return false; }
    if (parse_type(a.cache_k) == GGML_TYPE_COUNT) { err = "--cache-type-k: unknown type " + a.cache_k; return false; }
    if (parse_type(a.cache_v) == GGML_TYPE_COUNT) { err = "--cache-type-v: unknown type " + a.cache_v; return false; }
    return true;
}

/// The session: llama.cpp's memory under one sequence, and the ids it holds.
struct Session {
    llama_memory_t mem = nullptr;
    std::vector<int32_t> ids;                               // what the memory holds, from token 0
    int64_t len = 0;                                        // positions 0..len-1 are filled

    void truncate_to(int64_t keep) {                        // drop the tail above `keep`
        if (keep < len) {
            llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1);
            ids.resize((size_t) keep);
            len = keep;
        }
    }
    void clear() {
        llama_memory_seq_rm(mem, 0, 0, -1);
        ids.clear();
        len = 0;
    }
};

struct Engine {
    Args a;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    Session sess;
    std::atomic<bool> stop_req{false};
    std::vector<int32_t> eos_ids;                           // emitted, then the turn ends (llama's eos/eot)
    std::vector<int32_t> stop_ids;                          // the turn ends BEFORE them: never in the text
    int64_t ctx_n = 0;
    int n_vocab = 0;

    ~Engine() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool load(std::string& err) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = a.n_gpu_layers;                    // negative: all layers, llama's rule
        mp.main_gpu = a.main_gpu;
        std::vector<float> split(llama_max_devices(), 0.0f);
        if (!a.tensor_split.empty()) {
            for (size_t i = 0; i < a.tensor_split.size() && i < split.size(); ++i) split[i] = a.tensor_split[i];
            mp.tensor_split = split.data();
        }
        model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) { err = "llama.cpp could not load " + a.model + " (its log is above)"; return false; }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = a.max_context > 0 ? (uint32_t) a.max_context : 0;   // 0: the model's trained size
        cp.n_batch = (uint32_t) a.n_batch;
        cp.n_ubatch = (uint32_t) std::min(a.n_batch, a.n_ubatch);
        cp.n_seq_max = 1;
        if (a.threads > 0) cp.n_threads = a.threads, cp.n_threads_batch = a.threads;
        cp.type_k = parse_type(a.cache_k);
        cp.type_v = parse_type(a.cache_v);
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.kv_unified = false;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { err = "llama.cpp could not build the context (not enough memory for it)"; return false; }
        sess.mem = llama_get_memory(ctx);
        ctx_n = llama_n_ctx(ctx);

        const llama_vocab* vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        const llama_token e = llama_vocab_eos(vocab), t = llama_vocab_eot(vocab);
        if (e >= 0) eos_ids.push_back(e);
        if (t >= 0 && std::find(eos_ids.begin(), eos_ids.end(), t) == eos_ids.end()) eos_ids.push_back(t);
        stop_ids = a.stop_tokens;                             // GigaChat4: `<|message_sep|>\n\n` closes a turn
        return true;
    }

    void report() {
        const char* name = strrchr(a.model.c_str(), '/');
        name = name ? name + 1 : a.model.c_str();
        info("engine=strata-llama version=%s", STRATA_VERSION);
        info("name=%s n_ctx=%lld n_ctx_train=%d vocab=%d params=%.2fB size_gib=%.2f",
             name, (long long) ctx_n, llama_model_n_ctx_train(model), n_vocab,
             llama_model_n_params(model) / 1e9, llama_model_size(model) / (1024.0 * 1024.0 * 1024.0));
        info("devices=%zu n_gpu_layers=%d main_gpu=%d threads=%d batch=%d kv=%s,%s",
             llama_max_devices(), a.n_gpu_layers, a.main_gpu,
             a.threads > 0 ? a.threads : (int) std::thread::hardware_concurrency(), a.n_batch,
             a.cache_k.c_str(), a.cache_v.c_str());
        std::string s;
        for (size_t i = 0; i < eos_ids.size(); ++i) s += (i ? "," : "") + std::to_string(eos_ids[i]);
        info("eos=%s", s.empty() ? "none" : s.c_str());
        std::string st;
        for (size_t i = 0; i < stop_ids.size(); ++i) st += (i ? "," : "") + std::to_string(stop_ids[i]);
        if (!st.empty()) info("stop_tokens=%s", st.c_str());
    }

    /// Feed rows [from, n) of the prompt, one ubatch at a time, positions from their index; PP
    /// lines as the prompt goes through, like strata-naive's.  "" or a message; a non-empty one
    /// means the rows did not go through.
    std::string feed(const std::vector<int32_t>& tokens, size_t from, bool want_logits,
                     const Clock::time_point& t0) {
        llama_batch b = llama_batch_init(a.n_batch, 0, 1);
        std::string err;
        size_t next_pp = 16;
        const size_t chunk = (size_t) std::min(a.n_batch, 64);   // small rows: PP lines keep the server alive
        for (size_t i = from; i < tokens.size() && err.empty(); i += chunk) {
            const size_t n = std::min(chunk, tokens.size() - i);
            const bool last_chunk = i + n == tokens.size();
            for (size_t j = 0; j < n; ++j) {
                b.token[j] = tokens[i + j];
                b.pos[j] = (llama_pos) (i + j);
                b.n_seq_id[j] = 1;
                b.seq_id[j][0] = 0;
                b.logits[j] = (want_logits && last_chunk && j + 1 == n) ? 1 : 0;
            }
            b.n_tokens = (int32_t) n;
            if (llama_decode(ctx, b) != 0) err = "llama_decode refused the prompt at token " + std::to_string(i);
            if (i + n >= next_pp || i + n == tokens.size()) {
                std::printf("PP %zu %zu %.1f\n", i + n, tokens.size(), ms_since(t0));
                std::fflush(stdout);
                while (next_pp <= i + n) next_pp += 16;
            }
            if (stop_req.load()) break;
        }
        llama_batch_free(b);
        if (!err.empty()) sess.clear();
        return err;
    }

    /// Decode one sampled token at position `pos`; "" means it went through.
    std::string feed_one(int32_t tok, int64_t pos) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = tok;
        b.pos[0] = (llama_pos) pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { sess.clear(); return "llama_decode refused the new token"; }
        return {};
    }
};

void serve(Engine& en, bool verbose) {
    std::printf("READY %lld stop\n", (long long) en.ctx_n);
    std::fflush(stdout);

    Sampler sampler;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT" || !std::cin) break;
        if (line == "STOP") { en.stop_req.store(true); continue; }
        if (line.rfind("GENI ", 0) == 0) {
            std::printf("ERR this build serves text only (no picture path): send GEN with token ids\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            if (!line.empty() && line[0] != '#')
                std::printf("ERR expected: GEN <max_new> <id,id,...> or QUIT\n");
            std::fflush(stdout);
            continue;
        }
        GenReq q;
        if (std::string err; !parse_gen_line(line, q, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) q.ids.size() > en.ctx_n) {
            std::printf("ERR the prompt is %zu tokens, this context is %lld\n", q.ids.size(), (long long) en.ctx_n);
            std::fflush(stdout);
            continue;
        }
        {                                                   // llama.cpp aborts on an id outside its vocab: answered here
            const auto bad = std::find_if(q.ids.begin(), q.ids.end(), [&](int32_t t) { return t < 0 || t >= en.n_vocab; });
            if (bad != q.ids.end()) {
                std::printf("ERR the prompt carries a token id outside the vocabulary: %d\n", (int) *bad);
                std::fflush(stdout);
                continue;                                    // the session is untouched: a bad id never reaches the model
            }
        }
        en.stop_req.store(false);
        sampler.reseed(q.seed);

        // ---- resume the session where this prompt continues it; the shared prefix stays in the
        // memory, the tail is re-fed (strata-naive's rule, through llama.cpp's own memory api).
        int64_t same = 0;
        const int64_t hold = std::min<int64_t>(en.sess.len, (int64_t) q.ids.size());
        while (same < hold && en.sess.ids[(size_t) same] == q.ids[(size_t) same]) ++same;
        en.sess.truncate_to(same);

        const auto t0 = Clock::now();
        const int64_t from = en.sess.len;
        en.sess.ids = q.ids;                                  // what the memory will hold after the feed
        en.sess.len = (int64_t) q.ids.size();
        if (const std::string e = en.feed(q.ids, (size_t) same, q.max_new > 0, t0); !e.empty()) {
            std::printf("ERR %s\n", e.c_str());
            std::fflush(stdout);
            en.sess.clear();
            continue;
        }
        if (en.stop_req.load()) {                             // the client left mid-prompt: the tail is not held
            en.sess.truncate_to(from);                        // only what the earlier turns proved is kept
            std::printf("DONE 0 %zu %.1f 0.0 cancel\n", q.ids.size(), ms_since(t0));
            std::fflush(stdout);
            continue;
        }
        const double prompt_ms = ms_since(t0);
        if (q.max_new <= 0) {                                 // prefill only (the split path's prefill=1): no token
            std::printf("DONE 0 %zu %.1f 0.0 length\n", q.ids.size(), prompt_ms);
            std::fflush(stdout);
            continue;
        }

        // ---- decode. One T line per token, flushed as it is made; the session holds prompt plus
        // output, so a later request that continues this turn resumes at this exact point.
        const float* lg = llama_get_logits(en.ctx);            // the one output row: the prompt's last row
        if (!lg) { std::printf("ERR llama.cpp returned no logits for the prompt\n"); std::fflush(stdout); en.sess.clear(); continue; }
        const int n_vocab = en.n_vocab;
        bool nonfinite = false;
        for (int i = 0; i < n_vocab; ++i) if (!(lg[i] == lg[i])) { nonfinite = true; break; }
        if (nonfinite) { std::printf("DONE 0 %zu %.1f 0.0 nonfinite\n", q.ids.size(), prompt_ms); std::fflush(stdout); en.sess.clear(); continue; }

        const auto t1 = Clock::now();
        int64_t produced = 0;
        const char* fin = "length";
        std::vector<int32_t> history = en.sess.ids;            // the penalty window reads prompt + output
        while (produced < q.max_new) {
            const int32_t y = sampler.sample(lg, n_vocab, q, history);
            if (std::find(en.stop_ids.begin(), en.stop_ids.end(), y) != en.stop_ids.end()) {
                fin = "stop";                                 // the turn's closer: never written, never kept
                break;                                        // (the session does not hold what was never fed)
            }
            std::printf("T %d\n", (int) y);
            std::fflush(stdout);
            ++produced;
            history.push_back(y);
            if (std::find(en.eos_ids.begin(), en.eos_ids.end(), (int64_t) y) != en.eos_ids.end()) { fin = "stop"; break; }
            if (en.stop_req.load()) { fin = "cancel"; break; }   // the token the client did not see: never fed
            if (en.sess.len >= en.ctx_n) { fin = "length"; break; }
            const int64_t pos = en.sess.len;                     // the position this token takes
            en.sess.ids.push_back(y);
            en.sess.len++;
            if (const std::string e = en.feed_one(y, pos); !e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::fflush(stdout);
                fin = nullptr;
                break;
            }
            const float* nx = llama_get_logits(en.ctx);
            if (!nx) { std::printf("ERR llama.cpp returned no logits while decoding\n"); std::fflush(stdout); en.sess.clear(); fin = nullptr; break; }
            bool nf = false;
            for (int i = 0; i < n_vocab; ++i) if (!(nx[i] == nx[i])) { nf = true; break; }
            if (nf) { fin = "nonfinite"; break; }              // #879's rule: that window's tokens are not emitted
            lg = nx;
            if (verbose) info("tok %lld/%lld %d %.0f tok/min", (long long) produced, q.max_new, (int) y,
                              produced * 60000.0 / std::max(1.0, ms_since(t1)));
        }
        if (fin == nullptr) {                                 // the ERR line above already ended this request
            continue;
        }
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %lld\n", (long long) produced, q.ids.size(), prompt_ms,
                    ms_since(t1), fin, (long long) from);      // the DONE's fields, positional, as _parse_done reads them
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    std::string err;
    if (argc < 2 || (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    if (!parse_args(argc, argv, a, err)) {
        std::fprintf(stderr, "strata-llama: %s\n", err.c_str());
        usage();
        return 2;
    }
    if (!a.verbose) {                                        // the DEBUG flood of the library stays out of the run log
        llama_log_set(quiet_log, nullptr);
        ggml_log_set(quiet_log, nullptr);
    }
    Engine en;
    en.a = a;
    if (!en.load(err)) {
        std::fprintf(stderr, "strata-llama: %s\n", err.c_str());
        return 2;
    }
    en.report();
    if (a.check) {
        std::fprintf(stderr, "strata-llama: the model loaded; it answers the serve protocol with --serve\n");
        return 0;
    }
    serve(en, a.verbose);
    return 0;
}
