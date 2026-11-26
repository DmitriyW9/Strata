// tests/naive/test_naive_pool.cpp - the residency pool and the memory plan (include/strata/naive/
// pool.hpp) plus the GPU tier's CPU-visible half (gpu.hpp).  The pool promises three things, and
// this test pins all three on the shared synthetic model (synthetic_model.hpp, the same file the
// runtime suite writes):
//   1. the PLAN is arithmetic, not a guess: role_of classifies every name the inventory emits,
//      plan_pool's byte sums match the file's own tensor table, and its fits/refuses follow the
//      RAM and card room the caller probes (made-up cards here, so CI needs no GPU);
//   2. the resident COPIES answer exactly what the mmap answers: a resident expert's bytes are the
//      file's bytes bitwise, so gemv_expert_raw / gemv_raw / cols are bitwise-equal to loader.hpp's
//      gemv_dequant_3d / gemv_dequant / load_tensor_cols, and the fp32 dense copies bitwise-equal to
//      load_tensor_f32 (same ggml decoder, same element order); a runtime run through the pool is
//      therefore bitwise-equal to a runtime run through the mmap - the pool moves bytes, never
//      numbers;
//   3. the GPU tier lands within one bf16 ulp of the CPU row (the kernel folds its warp in a shuffle
//      tree, not a line).  Skipped with a SKIP line when the build has no CUDA or no card answers.
//
//   test_naive_pool [scratch-dir]     (CTest passes ${CMAKE_CURRENT_BINARY_DIR}/naive_pool_scratch)
//   Exits 0 with ok lines, 1 on the first FAIL, 3 on a setup problem (model write / open refused).

#include "strata/naive/gpu.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/pool.hpp"
#include "strata/naive/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "synthetic_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace naive_synth;

namespace {

int g_failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failures;
}

uint64_t fnv1a(const uint8_t* p, size_t n) {  // the loader test's digest, for the byte-copies check
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// ---------------------------------------------------------------- the plan, on made-up memory

void plan_checks(const strata::GgufModel& m, const strata::naive::NaiveGeometry& g,
                 const std::vector<Syn>& ts, uint64_t expert_bytes, uint64_t dense_fp32_bytes) {
    using namespace strata::naive;
    // role_of: every name the synthetic inventory emits is Expert (the three stacks) or Dense.
    size_t n_expert = 0, n_dense = 0, n_transient = 0;
    for (const Syn& s : ts) {
        switch (role_of(s.name)) {
            case Role::Expert: ++n_expert; break;
            case Role::Dense: ++n_dense; break;
            default: ++n_transient; break;
        }
    }
    check(n_transient == 0 && n_expert == 6 && n_dense == ts.size() - 6,
          "role_of names every synthetic tensor (6 expert stacks, the rest dense, none transient)");

    const uint64_t gib = 1ull << 30;
    // A box with plenty of RAM and one 40-GiB card: everything fits, the card takes all experts.
    {
        const std::vector<CardMem> cards{{0, 48 * gib, 40 * gib}};
        const PoolPlan p = plan_pool(m, g, 256 * gib, cards);
        check(p.note.empty() && p.fits_all && p.fits_raw, "plan_pool: a big box fits everything");
        check(p.expert_bytes == expert_bytes && p.dense_fp32_bytes == dense_fp32_bytes,
              "plan_pool: the byte sums come from the file's tensor table");
        check(p.card_expert_bytes.size() == 1 && p.card_expert_bytes[0] == expert_bytes,
              "plan_pool: one card with room takes every expert stack");
        const ResidentSet s = select(m, g, p.ram_bytes, true);
        check(s.all && s.note.empty() && s.resident.size() == ts.size(),
              "select: the full box returns every name, no skips");
    }
    // Plenty of RAM but a small card: experts still all fit RAM, the card takes only its room.
    {
        const std::vector<CardMem> cards{{0, 12 * gib, (kPlanCardReserveBytes >> 20) + 1024 + gib}};
        const PoolPlan p = plan_pool(m, g, 256 * gib, cards);
        const uint64_t room = card_room_bytes(cards[0], kPlanCardReserveBytes);
        check(p.fits_all && p.card_expert_bytes[0] == std::min<uint64_t>(room, expert_bytes),
              "plan_pool: a small card takes its room, never the whole stack budget");
    }
    // RAM for the experts only: fits_all falls to fits_raw and the note says what stayed on disk.
    {
        const std::vector<CardMem> cards;
        const PoolPlan p = plan_pool(m, g, kPlanRamReserveBytes + expert_bytes + 1, cards);
        check(p.fits_raw && !p.fits_all && !p.note.empty(),
              "plan_pool: expert-only RAM sets fits_raw, clears fits_all, and says why");
        const ResidentSet s = select(m, g, expert_bytes, false);
        check(s.all && s.resident.size() == 6, "select: the expert-only set is the six stacks exactly");
    }
    // Too little RAM for even the experts: nothing resident, the note names the number.
    {
        const std::vector<CardMem> cards;
        const PoolPlan p = plan_pool(m, g, 1 * gib, cards);
        check(!p.fits_raw && !p.fits_all && p.note.find("experts") != std::string::npos,
              "plan_pool: a 1-GiB box refuses residency with a reason");
    }

    // ---- the exclusive plan (plan_split): a stack lives on a card OR in RAM, never both
    std::string eerr;
    const std::vector<ExpertStack> estacks = expert_stacks(m, g, eerr);
    check(eerr.empty() && estacks.size() == 6, "expert_stacks: the six synthetic stacks, no error");
    // The stacks' real sizes differ (IQ2_XXS gate/up vs IQ2_XS down), so the cases below size the
    // cards from the file's own numbers: gsz = a gate/up stack, dsz = a down stack.
    uint64_t gsz = 0, dsz = 0;
    for (const ExpertStack& st : estacks) {
        if (st.name == "blk.1.ffn_gate_exps.weight") gsz = st.bytes;
        if (st.name == "blk.1.ffn_down_exps.weight") dsz = st.bytes;
    }
    check(gsz != 0 && dsz != 0 && expert_bytes == 4 * gsz + 2 * dsz,
          "expert_stacks: the sums match the file (4 gate/up + 2 down stacks)");
    // A card big enough for every stack: every stack exclusive on the card, nothing in RAM. The
    // first card's room pays the one context charge (kPlanCardReserveBytes), so 40 GiB covers all.
    {
        const std::vector<CardMem> cards{{0, 48 * gib, 40 * gib}};
        const SplitPlan p = plan_split(estacks, dense_fp32_bytes, kPlanRamReserveBytes + dense_fp32_bytes,
                                       cards);
        check(p.vram_expert_bytes == expert_bytes && p.ram_expert_bytes == 0 &&
                  p.skipped_expert_bytes == 0 && p.fits_all && p.note.empty(),
              "plan_split: a card with room holds every stack, RAM holds only the dense set");
        check(p.card_expert_bytes.size() == 1 && p.card_expert_bytes[0] == expert_bytes,
              "plan_split: the card's bytes are the whole expert set");
        size_t on_card = 0;
        for (const SplitPlacement& pl : p.placements)
            if (pl.card >= 0) ++on_card;
        check(on_card == 6, "plan_split: every placement names the card, none names RAM");
    }
    // Two cards sized to the file's own stacks: card 0's room is exactly gate1 + up1 (the file's
    // first two stacks, both gate/up-sized), card 1's exactly down1. File order, whole stacks.
    // The three stacks no card took fall to RAM (2*gsz + dsz of it).
    {
        const uint64_t r = kPlanCardReserveBytes;
        const std::vector<CardMem> cards{{0, 24 * gib, r + r + 2 * gsz},   // pays the context charge too
                                         {1, 24 * gib, r + dsz}};
        const SplitPlan p = plan_split(estacks, dense_fp32_bytes,
                                       kPlanRamReserveBytes + dense_fp32_bytes + 3 * gsz + dsz + 1,
                                       cards);
        check(p.card_expert_bytes.size() == 2 && p.card_expert_bytes[0] == 2 * gsz &&
                  p.card_expert_bytes[1] == dsz,
              "plan_split: two cards take exactly their room (2 gate/up + 1 down), richest first");
        check(p.vram_expert_bytes == 2 * gsz + dsz && p.ram_expert_bytes == 2 * gsz + dsz &&
                  p.skipped_expert_bytes == 0,
              "plan_split: the rest goes to RAM, none skipped");
        check(p.fits_all && p.note.empty(), "plan_split: every stack placed, the plan is silent");
    }
    // No card at all: the exclusive plan degrades to the RAM pool exactly like the copy tier -
    // experts in RAM, note honest about the skipped stacks when RAM is short.
    {
        const std::vector<CardMem> cards;
        const SplitPlan p = plan_split(estacks, dense_fp32_bytes,
                                       kPlanRamReserveBytes + dense_fp32_bytes + expert_bytes + 1,
                                       cards);
        check(p.vram_expert_bytes == 0 && p.ram_expert_bytes == expert_bytes && p.fits_all,
              "plan_split: with no card every stack lives in RAM (the copy tier's shape, no copies)");
        const SplitPlan tight = plan_split(estacks, dense_fp32_bytes,
                                           kPlanRamReserveBytes + dense_fp32_bytes + 2 * gsz,
                                           cards);
        check(!tight.fits_all && tight.skipped_expert_bytes == expert_bytes - 2 * gsz &&
                  tight.note.find("mmap") != std::string::npos,
              "plan_split: RAM for two stacks skips the rest and says they read the mmap windows");
    }
    // The tier's premise: without room for the dense set, vram-split refuses (main.cpp exits 2) -
    // the part every row reads never falls to the mmap silently.
    {
        const std::vector<CardMem> cards{{0, 48 * gib, 40 * gib}};
        const SplitPlan p = plan_split(estacks, dense_fp32_bytes, kPlanRamReserveBytes, cards);
        check(!p.dense_fits && !p.fits_all && p.note.find("dense") != std::string::npos,
              "plan_split: no RAM room for the dense set is named in the note, dense_fits false");
    }
    // The plan's promise: every stack exactly once, the card bytes + RAM bytes + skipped add up.
    {
        const std::vector<CardMem> cards{{0, 48 * gib, 2 * gsz + kPlanCardReserveBytes * 2}};
        const SplitPlan p = plan_split(estacks, dense_fp32_bytes,
                                       kPlanRamReserveBytes + dense_fp32_bytes + expert_bytes + 1,
                                       cards);
        uint64_t placed = 0;
        size_t seen = 0;
        for (const SplitPlacement& pl : p.placements) {
            placed += pl.bytes;
            ++seen;
        }
        check(seen == 6 && placed + p.skipped_expert_bytes == expert_bytes &&
                  p.vram_expert_bytes + p.ram_expert_bytes == placed,
              "plan_split: each stack is placed once, the bytes add up");
    }
}

// ---------------------------------------------------------------- the pool's copies vs the mmap

void pool_checks(const strata::GgufModel& m, const strata::naive::NaiveGeometry& g,
                 uint64_t expert_bytes) {
    using namespace strata::naive;
    NaivePool pool(m, g);
    // The names the checks below read - not load_all: the synthetic file's fp32 dense copies
    // (token_embd / output at 4096 x 152576) are ~5.5 GiB, more than a small CI box may hold.
    // load_all's completeness is the plan check's job (select's full-box set above).
    const std::vector<std::string> wanted{
        "blk.1.ffn_gate_exps.weight", "blk.2.ffn_gate_exps.weight",
        "blk.1.ffn_up_exps.weight",   "blk.2.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight", "blk.2.ffn_down_exps.weight",
        "token_embd.weight",          "blk.1.attn_k.weight",        "output_norm.weight",
        "blk.0.attn_norm.weight",     "blk.1.ffn_gate_inp.weight"};
    if (const std::string e = pool.load(wanted); !e.empty()) {
        std::printf("FAIL pool.load: %s\n", e.c_str());
        ++g_failures;
        return;
    }
    check(pool.resident_count() == wanted.size(),
          "load brought exactly the requested names resident (" + std::to_string(wanted.size()) + ")");
    check(pool.resident_expert_bytes() == expert_bytes,
          "the pool's raw tier holds exactly the plan's expert bytes");
    // Idempotence: a second load of the same names is a no-op (no second copy, no error).
    check(pool.load({"blk.1.ffn_gate_exps.weight"}).empty() &&
              pool.resident_expert_bytes() == expert_bytes,
          "load is idempotent per name (the expert bytes did not double)");

    // 1. The raw expert bytes are the file's bytes, byte for byte (the loader test's digest, on the
    //    pool's copy instead of the mmap).
    bool bytes_equal = true;
    for (const NaivePool::ResidentExpert& st : pool.resident_experts()) {
        strata::TensorInfo* t = nullptr;
        {
            size_t shard = 0;
            const strata::TensorInfo* found = m.find(st.name, &shard);
            if (!found) { bytes_equal = false; break; }
            t = const_cast<strata::TensorInfo*>(found);
            const uint8_t* mmap = m.shard(shard).tensor_data(*t);
            if (fnv1a(mmap, (size_t) st.bytes_len) != fnv1a(st.bytes, (size_t) st.bytes_len) ||
                st.bytes_len != ggml_payload_bytes(t->type, t->elements())) {
                bytes_equal = false;
            }
        }
        (void) t;
    }
    check(bytes_equal, "every resident expert stack is the file's bytes bitwise (FNV-1a + length)");

    std::mt19937 rng(4242);
    std::normal_distribution<double> nd(0.0, 0.5);
    auto rand_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (float& f : v) f = (float) nd(rng);
        return v;
    };

    // 2. A resident expert's GEMV equals the mmap's bitwise (same bytes, same decoder, same order).
    {
        const std::string name = "blk.1.ffn_gate_exps.weight";
        const int64_t n_in = N_EMBD, n_out = EXPERT_FF;
        const std::vector<float> x = rand_vec((size_t) n_in);
        std::vector<float> a((size_t) n_out), b((size_t) n_out);
        if (const std::string e = gemv_dequant_3d(m, name, 3, x.data(), n_in, a.data(), n_out); !e.empty()) {
            std::printf("FAIL gemv_dequant_3d: %s\n", e.c_str());
            ++g_failures;
        } else {
            check(pool.gemv_expert_raw(name.c_str(), 3, x.data(), n_in, n_out, b.data(), 4) &&
                      std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
                  "gemv_expert_raw == gemv_dequant_3d bitwise (expert 3 of the gate stack)");
        }
    }
    // 3. A dense weight the pool holds as an fp32 copy answers the same GEMV the mmap's per-row
    //    decode answers, bitwise: load_tensor_f32 and gemv_dequant run the same ggml decoder in
    //    the same element order, so the bytes in the dot are the same bytes (pool.hpp's promise).
    {
        const std::string name = "blk.1.attn_k.weight";  // Q8_0: Dense by role, held fp32 by the pool
        const int64_t n_in = N_EMBD, n_out = KV_SWA * K_DIM;  // the window layer's K table, {4096,1536}
        const std::vector<float> x = rand_vec((size_t) n_in);
        std::vector<float> a((size_t) n_out), b((size_t) n_out, 0.f);
        if (const std::string e = gemv_dequant(m, name, x.data(), n_in, a.data(), n_out); !e.empty()) {
            std::printf("FAIL gemv_dequant: %s\n", e.c_str());
            ++g_failures;
        } else {
            check(pool.gemv_f32(name.c_str(), x.data(), n_in, n_out, b.data(), 4) &&
                      std::memcmp(a.data(), b.data(), a.size() * 4) == 0,
                  "pool gemv_f32 == gemv_dequant bitwise (fp32 copy, same decoder, same order)");
        }
    }
    // 4. A column window of a resident table equals the loader's (the embedding row the trunk reads).
    {
        std::vector<float> b((size_t) N_EMBD * 3);
        LoadedTensor lt;
        if (const std::string e = load_tensor_cols(m, "token_embd.weight", 12345, 3, lt); !e.empty()) {
            std::printf("FAIL load_tensor_cols: %s\n", e.c_str());
            ++g_failures;
        } else {
            check(pool.cols("token_embd.weight", 12345, 3, b.data()) &&
                  std::memcmp(lt.data.data(), b.data(), b.size() * 4) == 0,
                  "pool cols == load_tensor_cols bitwise (token_embd, 3 columns)");
        }
    }
    // 5. The fp32 dense copies equal load_tensor_f32 elementwise bitwise (same ggml decoder).
    {
        bool same = true;
        for (const char* name : {"output_norm.weight", "blk.0.attn_norm.weight",
                                 "blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_exps.weight"}) {
            if (role_of(name) == Role::Expert) continue;  // experts stay raw; not in the fp32 map
            LoadedTensor lt;
            if (const std::string e = load_tensor_f32(m, name, lt); !e.empty()) { same = false; break; }
            const LoadedTensor* p = pool.fp32(name);
            if (!p || p->data.size() != lt.data.size() ||
                std::memcmp(p->data.data(), lt.data.data(), lt.data.size() * 4) != 0) {
                same = false;
            }
        }
        check(same, "fp32 dense copies == load_tensor_f32 bitwise (norms, router; experts excluded)");
    }
}

// ------------------------------------------------------- the runtime through the pool, end to end

void runtime_checks(const std::filesystem::path& gguf_path, const strata::GgufModel& m,
                    const strata::naive::NaiveGeometry& g) {
    using namespace strata::naive;
    std::mt19937 rng(99);
    std::vector<int64_t> prompt;
    for (int i = 0; i < 3; ++i) prompt.push_back((int64_t) (rng() % (uint32_t) N_VOCAB));
    auto run = [&](NaivePool* pool) -> std::vector<float> {
        std::string err;
        NaiveRuntime rt(m, g, 4, HISTORY, err, pool, nullptr);
        if (!rt.ok()) {
            std::printf("FAIL the runtime refused the model: %s\n", err.c_str());
            ++g_failures;
            return {};
        }
        for (size_t i = 0; i < prompt.size(); ++i)
            if (const std::string e = rt.forward_row(prompt[i], i + 1 == prompt.size()); !e.empty()) {
                std::printf("FAIL forward_row: %s\n", e.c_str());
                ++g_failures;
                return {};
            }
        return rt.logits();
    };
    const std::vector<float> mmap_logits = run(nullptr);

    // The whole synthetic file resident: on a box without the RAM for it, say so and skip only
    // this part (the plan/pool checks above need only the expert bytes, which always fit).
    const uint64_t full = plan_pool(m, g, 1ull << 40, {}).ram_bytes;
    uint64_t mem_avail = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        uint64_t kb = 0;
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, "MemAvailable:", 13) == 0) {
                std::sscanf(line + 13, "%llu", (unsigned long long*) &kb);
                break;
            }
        std::fclose(f);
        mem_avail = kb * 1024ull;
    }
    // The 2 GiB headroom is the process itself plus a margin so a small CI box skips this part
    // instead of being OOM-killed by a 5 GiB resident copy of the two vocab tables.
    if (full + (2ull << 30) > mem_avail) {
        std::printf("SKIP: the full-pool run needs %.1f GiB and this box reports %.1f GiB free\n",
                    (double) full / (1024.0 * 1024.0 * 1024.0),
                    (double) mem_avail / (1024.0 * 1024.0 * 1024.0));
        (void) gguf_path;
        return;
    }
    NaivePool pool(m, g);
    if (const std::string e = pool.load_all(); !e.empty()) {
        std::printf("FAIL load_all: %s\n", e.c_str());
        ++g_failures;
        return;
    }
    const std::vector<float> pool_logits = run(&pool);
    if (mmap_logits.empty() || pool_logits.empty()) return;
    check(mmap_logits.size() == pool_logits.size() &&
              std::memcmp(mmap_logits.data(), pool_logits.data(), mmap_logits.size() * 4) == 0,
          "the whole model through the pool answers the same logits bitwise (3-row prefill)");
}

// ------------------------------------------------------------------ the GPU tier, if one is there

void gpu_checks(const strata::GgufModel& m, const strata::naive::NaiveGeometry& g) {
    using namespace strata::naive;
    if (!NaiveGpu::built_with_cuda()) {
        std::printf("SKIP: this build has no CUDA kernels (-DSTRATA_NAIVE_CUDA=OFF)\n");
        return;
    }
    NaiveGpu gpu;
    if (const std::string e = gpu.init({}); !e.empty()) {
        std::printf("SKIP: no CUDA card answers (%s)\n", e.c_str());
        return;
    }
    NaivePool pool(m, g);
    if (const std::string e = pool.load({"blk.1.ffn_gate_exps.weight"}); !e.empty()) {
        std::printf("FAIL pool.load: %s\n", e.c_str());
        ++g_failures;
        return;
    }
    const std::vector<NaivePool::ResidentExpert> stacks = pool.resident_experts();
    if (stacks.size() != 1) {
        std::printf("FAIL the pool holds the one stack this check loads\n");
        ++g_failures;
        return;
    }
    for (const GpuCard& c : gpu.cards()) {
        std::string uerr;
        const int h = gpu.upload(stacks[0], c.index, uerr);
        if (h < 0) {
            std::printf("SKIP: card %d cannot hold the stack (%s)\n", c.index, uerr.c_str());
            continue;
        }
        std::mt19937 rng(5);
        std::normal_distribution<double> nd(0.0, 0.5);
        std::vector<float> x((size_t) N_EMBD), a((size_t) EXPERT_FF), b((size_t) EXPERT_FF);
        for (float& f : x) f = (float) nd(rng);
        if (const std::string e = gpu.gemv_expert(h, 3, x.data(), N_EMBD, a.data(), EXPERT_FF); !e.empty()) {
            if (e.find("no kernel image") != std::string::npos)
                std::printf("SKIP: card %d has no kernel image in this build (%s)\n", c.index, e.c_str());
            else {
                std::printf("FAIL gemv_expert on card %d: %s\n", c.index, e.c_str());
                ++g_failures;
            }
            continue;
        }
        if (!pool.gemv_expert_raw("blk.1.ffn_gate_exps.weight", 3, x.data(), N_EMBD, EXPERT_FF,
                                  b.data(), 4)) {
            std::printf("FAIL the CPU pool refused its own stack\n");
            ++g_failures;
            continue;
        }
        int bad = 0;
        double worst = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double tol = 2e-3 + 0.008 * std::fabs((double) b[i]);   // the runtime suites' tolerance
            const double d = std::fabs((double) a[i] - (double) b[i]);
            if (d > tol) ++bad;
            worst = std::max(worst, d / std::max(1e-30, std::fabs((double) b[i])));
        }
        check(bad == 0, "the card's IQ2 GEMV lands inside one bf16 ulp of the CPU row (card " +
                            std::to_string(c.index) + ", worst rel " + std::to_string(worst).substr(0, 8) +
                            ")");
        break;  // one card answered: the rest are the same kernels
    }
    // The exclusive tier's upload (--weights vram-split): the stack goes to the card straight from
    // the loader's mmap pointer - no pool copy exists for it - and its rows must answer what the
    // CPU path answers. The CPU reference here loads the same stack into its own pool on purpose:
    // it is the answer the engine would give without a card, and the check is that the card gives
    // the same one from bytes that never passed through a RAM copy the engine owns.
    {
        std::string eerr;
        const std::vector<ExpertStack> est = expert_stacks(m, g, eerr);
        const ExpertStack* down = nullptr;
        for (const ExpertStack& st : est)
            if (st.name == "blk.2.ffn_down_exps.weight") down = &st;
        bool checked = false;
        for (const GpuCard& c : gpu.cards()) {
            size_t shard = 0;
            const strata::TensorInfo* t = m.find("blk.2.ffn_down_exps.weight", &shard);
            if (!down || !t) {
                std::printf("FAIL expert_stacks/find lost blk.2.ffn_down_exps.weight\n");
                ++g_failures;
                break;
            }
            std::string uerr;
            const int h = gpu.upload_from_mmap(*down, m.shard(shard).tensor_data(*t), c.index, uerr);
            if (h < 0) {
                std::printf("SKIP: card %d cannot hold the exclusive upload (%s)\n", c.index,
                            uerr.c_str());
                continue;
            }
            // Past the upload there is no next card: uploads are idempotent by name, so a second
            // card's call would answer with this card's handle. One card proves the path.
            std::mt19937 rng(6);
            std::normal_distribution<double> nd(0.0, 0.5);
            std::vector<float> x((size_t) EXPERT_FF), a((size_t) N_EMBD), b((size_t) N_EMBD);
            for (float& f : x) f = (float) nd(rng);
            if (const std::string e =
                    gpu.gemv_expert(h, 3, x.data(), EXPERT_FF, a.data(), N_EMBD);
                !e.empty()) {
                if (e.find("no kernel image") != std::string::npos)
                    std::printf("SKIP: card %d has no kernel image (%s)\n", c.index, e.c_str());
                else {
                    std::printf("FAIL the exclusive gemv_expert on card %d: %s\n", c.index, e.c_str());
                    ++g_failures;
                }
                break;
            }
            NaivePool ref(m, g);
            if (!ref.load({"blk.2.ffn_down_exps.weight"}).empty() ||
                !ref.gemv_expert_raw("blk.2.ffn_down_exps.weight", 3, x.data(), EXPERT_FF, N_EMBD,
                                     b.data(), 4)) {
                std::printf("FAIL the CPU reference refused blk.2.ffn_down_exps.weight\n");
                ++g_failures;
                break;  // the stack is on this card now; a second card's upload would be refused
            }
            int bad = 0;
            double worst = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double tol = 2e-3 + 0.008 * std::fabs((double) b[i]);
                const double dd = std::fabs((double) a[i] - (double) b[i]);
                if (dd > tol) ++bad;
                worst = std::max(worst, dd / std::max(1e-30, std::fabs((double) b[i])));
            }
            check(bad == 0, "the exclusive upload's IQ2 GEMV lands inside one bf16 ulp of the CPU row"
                            " (card " + std::to_string(c.index) + ", worst rel " +
                                std::to_string(worst).substr(0, 8) + ")");
            checked = true;
            break;  // one card answered: the rest are the same kernels
        }
        (void) checked;  // every SKIP above already named itself
    }
    gpu.release();
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path scratch = argc > 1 ? argv[1] : std::filesystem::temp_directory_path();
    std::filesystem::create_directories(scratch);
    const auto gguf_path = scratch / "strata_naive_pool.gguf";
    std::mt19937 rng(20261118);  // the runtime suite's seed: byte-identical model
    const std::vector<Syn> ts = build_model(rng);
    if (!write_gguf(gguf_path, ts, build_metadata())) {
        std::fprintf(stderr, "cannot write %s\n", gguf_path.c_str());
        return 3;
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model.reset(new strata::GgufModel(strata::gguf_split_paths(gguf_path.string())));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open refused: %s\n", e.what());
        return 3;
    }
    strata::naive::NaiveGeometry g;
    if (const std::string e = strata::naive::read_geometry(model->meta(), g); !e.empty()) {
        std::fprintf(stderr, "read_geometry: %s\n", e.c_str());
        return 3;
    }
    if (const std::string e = strata::naive::check_gguf_layout(*model, g); !e.empty()) {
        std::fprintf(stderr, "check_gguf_layout: %s\n", e.c_str());
        return 3;
    }
    // The plan's two byte sums, computed straight from the tensor table the test wrote (the plan
    // reads the same table through the inventory - the check is that it adds it up the same way).
    uint64_t expert_bytes = 0, dense_fp32_bytes = 0;
    for (const Syn& s : ts) {
        uint64_t n = 1;
        for (uint64_t d : s.shape) n *= d;
        if (strata::naive::role_of(s.name) == strata::naive::Role::Expert) expert_bytes += s.bytes.size();
        else dense_fp32_bytes += n * 4;
    }
    plan_checks(*model, g, ts, expert_bytes, dense_fp32_bytes);
    pool_checks(*model, g, expert_bytes);
    runtime_checks(gguf_path, *model, g);
    gpu_checks(*model, g);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "ok", g_failures);
    return g_failures ? 1 : 0;
}
