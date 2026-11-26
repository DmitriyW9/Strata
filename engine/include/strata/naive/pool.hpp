// include/strata/naive/pool.hpp - the Naive-N0.5-Flash weight RESIDENCY POOL: the plan that says where
// every weight lives at start-up, and the resident copies it puts there.
//
// WHY THIS EXISTS, IN ONE PARAGRAPH. The strata-naive forward (runtime.hpp) reads every weight through
// loader.hpp's byte windows straight out of the mmap, so a cold session pays the 81 GiB of shards
// through the page cache before its first token (docs/NAIVE.md measured 12.7 s/token on a cold prompt
// while a second engine held the cache). This pool brings the weights into memory ONCE, at start-up,
// and hands the runtime stable pointers it reads for every row after: the dense tensors (the attention
// Q6_K/Q8_0, the BF16 indexer, the F32 norms/router/sinks, the Q8_0 embedding and output head) become
// fp32 copies in RAM, and the IQ2 expert stacks - 61 of the 81 GiB - become raw byte-for-byte copies,
// in RAM always and, on the cards the plan fits them onto, in VRAM too. The first token then reads
// RAM/VRAM, never the disk.
//
// WHAT LIVES WHERE, AND WHY. The two roles are the model's own: the expert stacks
// (blk.N.ffn_gate_exps / ffn_up_exps IQ2_XXS, ffn_down_exps IQ2_XS) are touched only through the router's
// eight picks per row, so they are the part worth putting in VRAM and worth keeping in their native
// bytes (dequantised to fp32 they would be 4x larger and would not fit); everything else is the DENSE
// part - read in full every row - and is stored dequantised to fp32 so the runtime's GEMVs read it with
// no per-row decode. The two roles are exactly what the reader's inventory (reader.hpp) already names,
// so the split is a lookup, not a guess.
//
// THE PLAN IS A PLAN, NOT A PROMISE. plan_pool() answers, from the model's tensor table and the memory
// the caller read (free VRAM per card from cudaMemGetInfo, free RAM from /proc/meminfo), three
// questions: does the whole file fit resident (fits_all), what budget of RAM and VRAM that needs
// (ram_bytes, vram_per_card_bytes), and what a box with LESS room should keep resident (select()).
// main.cpp prints the answer as INFO lines and, when the whole file does not fit, either refuses to
// start (--weights resident) or loads the subset and reads the rest through the old mmap windows
// (--weights auto, the default). Nothing here reads a card or /proc/meminfo itself: the caller probes
// and passes the numbers, so the plan is a pure function a test can drive with made-up cards.
//
// THE COPIES ARE THE LOADER'S OWN BYTES. A resident expert's raw bytes are byte-for-byte the mmap's
// (the loader test's FNV-1a property, applied to the copy), so a GEMV over the pool (gemv_raw,
// gemv_expert_raw) is bit-identical to the same GEMV over the mmap - the pool changes where the bytes
// come from, never what the arithmetic sees. The fp32 dense copies are dequantised with ggml's own
// per-type converter (loader.hpp's dequant_to_float), tensor by tensor, so the runtime's dense GEMVs
// read fp32 with no decoder; that differs from the per-row mmap path only in the order a dot accumulates
// (whole-tensor decode, then the same double k-ascending dot), which the runtime test pins to one bf16 ulp.
//
// TWO WAYS TO USE A CARD. The auto/resident tiers above keep every expert in RAM and let a card hold a
// COPY of some of them - simple, and right when RAM is plentiful. --weights vram-split instead makes the
// placement EXCLUSIVE (plan_split): a stack a card holds never gets a RAM copy at all (the bytes go to the
// card straight from the mmap, and mmap pages are the one RAM the OS can take back under pressure), so a
// box with more total VRAM than RAM can still keep the whole 61 GiB of experts off the disk. The runtime
// needs no second copy for this: its GPU path already owns the rows of every stack a card holds
// (runtime.hpp's expert_gemv_raw).
//
// CPU-only by design, like loader.hpp: this header holds the bytes and plans where they go; the CUDA
// tier that copies the VRAM-resident experts onto a card lives in gpu.hpp and is reached only when the
// caller built it (naive_gpu_available()). The runtime reads the CPU copy whether or not a card also
// holds one, so the numbers do not move with the GPU tier.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace strata::naive {

/// The role a tensor plays in the pool, decided by name against the reader's own inventory.
enum class Role {
    Expert,  // an IQ2 expert stack: raw resident, the part a card may hold in VRAM
    Dense,   // everything else: fp32 resident in RAM, read by every row
    Transient,  // a tensor the inventory does not name: never resident (the plan refuses a file whose
                // inventory does not cover it, so this only appears for a foreign file the caller fed it)
};

/// The role of one tensor, from its name alone (never a geometry-built name list: the same rule has
/// to classify the real 48-layer file and the runtime test's 3-layer synthetic model). The three
/// expert stacks are the only Role::Expert names reader.hpp's inventory emits (layer_tensors); every
/// other name the inventory emits is a global (token_embd / output / output_norm) or a per-layer
/// tensor ("blk.<l>.<rest>"), so Dense is "a global or a blk tensor that is not an expert stack", and
/// anything else is a foreign name the inventory never emits (Transient: never resident).
inline Role role_of(const std::string& name) {
    static const char* kExpert[] = {".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"};
    for (const char* suf : kExpert) {
        const size_t n = std::strlen(suf);
        if (name.size() >= n && name.compare(name.size() - n, n, suf) == 0) return Role::Expert;
    }
    static const char* kGlobals[] = {"token_embd.weight", "output.weight", "output_norm.weight"};
    for (const char* gname : kGlobals)
        if (name == gname) return Role::Dense;
    if (name.size() > 4 && name.compare(0, 4, "blk.") == 0) return Role::Dense;
    return Role::Transient;
}

// ---------------------------------------------------------------- the plan's inputs

/// One card's memory as the caller probed it (cudaMemGetInfo / nvidia-smi), in bytes. `free_bytes` is
/// what nobody holds right now - the number that decides what fits - and a card the caller could not
/// probe passes 0 so the plan refuses to place anything on it rather than trusting its size.
struct CardMem {
    int index = 0;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
};

/// Bytes the caller wants kept free ON EACH card after the experts land (the desktop / other engine's
/// room). Mirrors split_engine.py's FREE_RESERVE: the plan counts what a card has FREE minus what must
/// stay free, never its full size.
inline constexpr uint64_t kPlanCardReserveBytes = 700ull << 20;  // 700 MiB, the engine's --vram-reserve-mib
/// Bytes the caller wants kept free in RAM after the weights land: the runtime's KV/index caches plus a
/// little. The plan reports room against (available - this); the caches themselves are sized at ctx by
/// runtime.hpp::cache_gib_at and printed beside the plan so the two numbers are checkable together.
inline constexpr uint64_t kPlanRamReserveBytes = 8ull << 30;  // 8 GiB

/// Bytes one card can give the experts: what is free now, minus the reserve. Never below zero.
inline uint64_t card_room_bytes(const CardMem& c, uint64_t reserve) {
    return c.free_bytes > reserve ? c.free_bytes - reserve : 0;
}

// ---------------------------------------------------------------- the plan's output

/// Where the plan says every byte goes, and what it would cost. A pure value: plan_pool() fills it, a
/// test compares it, main.cpp prints it. Bytes, not GiB - the printer converts.
struct PoolPlan {
    uint64_t expert_bytes = 0;    // the IQ2 stacks, summed over every MoE layer (the raw resident set)
    uint64_t dense_fp32_bytes = 0;  // every dense tensor dequantised to fp32 (the RAM fp32 set)
    uint64_t dense_raw_bytes = 0;   // the same dense tensors in their file bytes (what --expert-only keeps)
    uint64_t ram_bytes = 0;         // the full-residency RAM figure (dense fp32 + expert raw)
    uint64_t ram_raw_bytes = 0;     // --expert-only's RAM figure (expert raw only)
    bool full_fp32 = true;          // the tier plan_pool was asked for (false: --expert-only)
    bool fits_all = false;          // every weight resident on this box, at the tiers below
    bool fits_raw = false;          // experts resident in RAM (the least that removes the cold read)
    bool expert_only = false;       // the plan chose the raw-dense tier (full_fp32 false)
    uint64_t ram_available_bytes = 0;  // what the caller probed (echoed back for the INFO line)
    std::vector<uint64_t> card_expert_bytes;  // expert bytes plan_pool placed on each input card
    std::string note;               // why it does not fit, or the empty string; never a silent shrink

    double expert_gib() const { return (double) expert_bytes / (1024.0 * 1024.0 * 1024.0); }
    double dense_fp32_gib() const { return (double) dense_fp32_bytes / (1024.0 * 1024.0 * 1024.0); }
    double dense_raw_gib() const { return (double) dense_raw_bytes / (1024.0 * 1024.0 * 1024.0); }
    double ram_gib() const { return (double) ram_bytes / (1024.0 * 1024.0 * 1024.0); }
    double ram_raw_gib() const { return (double) ram_raw_bytes / (1024.0 * 1024.0 * 1024.0); }
};

/// The weights a box can keep resident, as a set of tensor names. select() returns the largest prefix
/// that fits, main.cpp hands it to NaivePool::load, and the runtime reads a name not in the set through
/// the old mmap windows - so a box that cannot hold all 81 GiB still stops paying the disk for what it
/// can hold, and the arithmetic is the same either way.
struct ResidentSet {
    std::vector<std::string> resident;   // the names to load (in file order, experts interleaved by layer)
    uint64_t resident_bytes = 0;         // the bytes select() counted for them
    uint64_t skipped_bytes = 0;          // the bytes left on disk
    bool all = true;                     // every tensor fit: no mmap fallback will be read
    std::string note;                    // what was dropped and why (empty when all)
};

// ---------------------------------------------------------------- the exclusive (vram-split) plan

/// One expert stack as the file declares it: where its bytes live and how big they are. The exclusive
/// tier (plan_split) plans over these, and NaiveGpu::upload_from_mmap copies a VRAM stack straight from
/// the loader's mmap pointer into the card, so its bytes never enter the RAM pool at all.
struct ExpertStack {
    std::string name;
    size_t shard = 0;        // the GgufModel shard that holds it
    uint64_t file_offset = 0;  // TensorInfo::offset: bytes from the shard's data start
    uint64_t bytes = 0;        // its payload size (plan_pool's expert_bytes is the sum of these)
    std::vector<uint64_t> shape;  // {n_in, n_out, n_expert}, GGUF order - upload_from_mmap checks it
    uint32_t type = 0;         // IQ2_XXS / IQ2_XS
};

/// The model's expert stacks in file order (globals first, then per layer - the order select() uses),
/// straight from the tensor table. Like plan_pool, a tensor the inventory names but the file lacks
/// fails with its name in `err`; an expert name that is not a 3-D IQ2 stack (a foreign file) does too.
inline std::vector<ExpertStack> expert_stacks(const GgufModel& model, const NaiveGeometry& g,
                                              std::string& err) {
    std::vector<ExpertStack> out;
    std::vector<std::string> names;
    for (const TensorSpec& s : main_globals()) names.push_back(s.name);
    std::vector<TensorSpec> ts;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        layer_tensors(l, g, ts);
        char buf[160];
        for (const TensorSpec& s : ts) {
            std::snprintf(buf, sizeof buf, s.name, (int) l);
            names.push_back(buf);
        }
    }
    for (const std::string& name : names) {
        if (role_of(name) != Role::Expert) continue;
        size_t shard = 0;
        const TensorInfo* t = model.find(name, &shard);
        if (!t) { err = "the plan found no " + name + " in the model"; return {}; }
        if (t->shape.size() != 3) { err = name + " is not a 3-D expert stack"; return {}; }
        if (t->type != kIQ2_XXS && t->type != kIQ2_XS) {
            err = name + " is " + std::string(t->type_name()) + ", not an IQ2 stack";
            return {};
        }
        out.push_back({name, shard, t->offset, tensor_payload_bytes(*t), t->shape, t->type});
    }
    return out;
}

/// One expert stack's place in the exclusive plan: the card that holds it, or RAM when card < 0.
struct SplitPlacement {
    std::string name;
    int card = -1;      // the CardMem index the plan placed it on; -1: the RAM pool holds it
    uint64_t bytes = 0;
};

/// The exclusive plan (--weights vram-split): every expert stack lives EITHER on a card OR in the RAM
/// pool, never both. Same style as plan_pool - a pure function over probed numbers (the caller reads
/// MemAvailable and cudaMemGetInfo), so a test drives it with made-up cards. `dense_fp32_bytes` is the
/// fp32 dense set plan_pool counted (it always lives in RAM); `ram_reserve` / `card_reserve` are the
/// room the OS and the desktop keep.
///
/// Order and rules, stated so a review can check them against the code:
///   * Cards first: file order, a whole stack on one card (never split), only stacks that fit the
///     card's room; the card with the most room goes first, equal rooms keep the caller's order.
///   * Then RAM: dense fp32 first (every row reads it), then the stacks no card took; a stack that
///     fits neither is skipped and reads the mmap windows (the skipped_* INFO lines name it).
///   * The first card's room pays for every card's context+scratch up front (the same ~700 MiB the
///     per-card reserve already keeps on each card - charged once, not per card, so a two-card box
///     is not double-charged).
///   * dense_fits is the tier's premise, reported apart: the fp32 dense set always lives in RAM, and
///     a box without room for it gets a refusal, not a silent mmap fallback for the part every row reads.
///   * fits_all says every NAMED tensor is resident somewhere (VRAM or RAM) - the plan's promise;
///     fits_raw says at least one expert left the disk. Neither promises a cudaMalloc that later
///     fails: main.cpp reports what landed, as the copy tier does.
/// The note is honest about why: every expert in VRAM + the dense set fitting RAM is fits_all with an
/// empty note even when zero experts fit in RAM - the split's whole point.
struct SplitPlan {
    std::vector<SplitPlacement> placements;
    std::vector<uint64_t> card_expert_bytes;  // per input card, like PoolPlan's
    uint64_t vram_expert_bytes = 0;           // expert bytes on cards (not in RAM)
    uint64_t ram_expert_bytes = 0;            // expert bytes the RAM pool must hold
    uint64_t dense_fp32_bytes = 0;            // echo: the fp32 dense set, always RAM
    uint64_t expert_bytes = 0;                // the whole expert set (vram + ram + skipped)
    uint64_t skipped_expert_bytes = 0;        // experts neither a card nor RAM could take
    uint64_t ram_available_bytes = 0;         // echoed for the INFO line
    bool fits_all = false;   // every named tensor resident (cards + RAM), nothing reads the disk
    bool fits_raw = false;   // at least one expert left the disk
    bool dense_fits = false;  // the fp32 dense set fits the RAM room: the tier's premise, and the
                              // reason vram-split refuses to start when it does not
    std::string note;       // why it does not fit, or the empty string; never a silent shrink

    double vram_expert_gib() const { return (double) vram_expert_bytes / (1024.0 * 1024.0 * 1024.0); }
    double ram_expert_gib() const { return (double) ram_expert_bytes / (1024.0 * 1024.0 * 1024.0); }
    double dense_fp32_gib() const { return (double) dense_fp32_bytes / (1024.0 * 1024.0 * 1024.0); }
    double skipped_expert_gib() const { return (double) skipped_expert_bytes / (1024.0 * 1024.0 * 1024.0); }
};

inline SplitPlan plan_split(const std::vector<ExpertStack>& stacks, uint64_t dense_fp32_bytes,
                            uint64_t ram_available_bytes, const std::vector<CardMem>& cards,
                            uint64_t ram_reserve = kPlanRamReserveBytes,
                            uint64_t card_reserve = kPlanCardReserveBytes) {
    SplitPlan p;
    p.dense_fp32_bytes = dense_fp32_bytes;
    p.ram_available_bytes = ram_available_bytes;
    p.card_expert_bytes.assign(cards.size(), 0);
    std::vector<uint64_t> room(cards.size());
    for (const ExpertStack& st : stacks) p.expert_bytes += st.bytes;

    // The cards, richest first (stable: equal rooms keep the caller's order). The context+scratch
    // that no per-card reserve covers is charged to the first card's room once.
    std::vector<size_t> order(cards.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return cards[a].free_bytes > cards[b].free_bytes; });
    for (size_t i = 0; i < order.size(); ++i) room[order[i]] = card_room_bytes(cards[order[i]], card_reserve);
    if (!order.empty() && room[order[0]] >= kPlanCardReserveBytes) room[order[0]] -= kPlanCardReserveBytes;
    std::vector<bool> on_card(stacks.size(), false);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const size_t c = order[oi];
        for (size_t i = 0; i < stacks.size(); ++i) {
            const ExpertStack& st = stacks[i];
            if (on_card[i] || st.bytes == 0 || st.bytes > room[c]) continue;
            on_card[i] = true;
            room[c] -= st.bytes;
            p.card_expert_bytes[c] += st.bytes;
            p.vram_expert_bytes += st.bytes;
            p.placements.push_back({st.name, (int) c, st.bytes});
        }
    }

    // RAM: the fp32 dense set first, then the stacks no card took. A stack that fits neither lives on
    // the disk windows and is counted skipped, never silently dropped from the numbers.
    uint64_t ram_room = ram_available_bytes > ram_reserve ? ram_available_bytes - ram_reserve : 0;
    uint64_t ram_left = ram_room > dense_fp32_bytes ? ram_room - dense_fp32_bytes : 0;
    const bool dense_fits = dense_fp32_bytes <= ram_room;
    if (!dense_fits) ram_left = 0;
    std::string first_skipped;
    for (size_t i = 0; i < stacks.size(); ++i) {
        const ExpertStack& st = stacks[i];
        if (on_card[i]) continue;
        if (st.bytes <= ram_left) {
            ram_left -= st.bytes;
            p.ram_expert_bytes += st.bytes;
            p.placements.push_back({st.name, -1, st.bytes});
        } else {
            p.skipped_expert_bytes += st.bytes;
            if (first_skipped.empty()) first_skipped = st.name;
        }
    }

    p.fits_raw = p.vram_expert_bytes != 0 || p.ram_expert_bytes != 0;
    p.fits_all = p.skipped_expert_bytes == 0 && (stacks.empty() || p.fits_raw) && dense_fits;
    p.dense_fits = dense_fits;
    if (!p.fits_all) {
        if (!dense_fits) {
            p.note = "the fp32 dense set (" + std::to_string(p.dense_fp32_gib()).substr(0, 4) +
                     " GiB) does not fit in the free RAM (" + std::to_string(ram_room >> 30) +
                     " GiB after the reserve)";
        } else if (p.skipped_expert_bytes) {
            p.note = "no card and no RAM room for " + first_skipped + " and possibly later stacks (" +
                     std::to_string(p.skipped_expert_gib()).substr(0, 4) +
                     " GiB in all): those read the mmap windows";
        }
    }
    return p;
}

// ---------------------------------------------------------------- plan_pool

/// The plan, from the model's tensor table and the memory the caller probed. `model`/`g` give the
/// inventory (reader.hpp) so the roles and byte counts come from the file, not from constants; a tensor
/// the inventory does not name fails the plan with its name (a foreign file is refused, not half-run).
/// `ram_available_bytes` is MemAvailable (or a container's limit); `cards` the probed cards, which may
/// be empty (no GPU: the plan still reports what RAM would hold). `expert_only` asks the raw-dense tier.
inline PoolPlan plan_pool(const GgufModel& model, const NaiveGeometry& g, uint64_t ram_available_bytes,
                          const std::vector<CardMem>& cards, bool expert_only = false,
                          uint64_t ram_reserve = kPlanRamReserveBytes,
                          uint64_t card_reserve = kPlanCardReserveBytes) {
    PoolPlan p;
    p.expert_only = expert_only;
    p.full_fp32 = !expert_only;
    p.ram_available_bytes = ram_available_bytes;

    // Walk the inventory once; a tensor the file carries with another size than the inventory says is
    // refused by check_gguf_layout upstream, so here every tensor is expected to be found.
    auto add = [&](const std::string& name, std::string& err) -> bool {
        size_t shard = 0;
        const TensorInfo* t = model.find(name, &shard);
        if (!t) { err = "the plan found no " + name + " in the model"; return false; }
        const uint64_t raw = tensor_payload_bytes(*t);
        if (raw == 0) { err = name + " has an unknown type " + std::string(t->type_name()); return false; }
        const Role r = role_of(name);
        if (r == Role::Expert) {
            p.expert_bytes += raw;
        } else if (r == Role::Dense) {
            p.dense_raw_bytes += raw;
            // fp32 copies only when the tier asked for them; the raw tier keeps dense on disk.
            if (!expert_only) p.dense_fp32_bytes += t->elements() * 4;
        } else {
            err = name + " is not a tensor this build's inventory names"; return false;
        }
        return true;
    };
    std::string err;
    for (const TensorSpec& s : main_globals())
        if (!add(s.name, err)) { p.note = err; return p; }
    std::vector<TensorSpec> ts;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        layer_tensors(l, g, ts);
        char buf[160];
        for (const TensorSpec& s : ts) {
            std::snprintf(buf, sizeof buf, s.name, (int) l);
            if (!add(buf, err)) { p.note = err; return p; }
        }
    }

    p.ram_bytes = p.dense_fp32_bytes + p.expert_bytes;
    p.ram_raw_bytes = p.expert_bytes;
    const uint64_t ram_room = ram_available_bytes > ram_reserve ? ram_available_bytes - ram_reserve : 0;

    // RAM decides the tier. Experts always live in RAM (the cards hold a subset of that copy), so the
    // smaller bar - experts alone - is the floor below which nothing resident is possible.
    p.fits_raw = p.expert_bytes != 0 && p.expert_bytes <= ram_room;
    p.fits_all = p.fits_raw && (expert_only ? p.expert_bytes <= ram_room : p.ram_bytes <= ram_room);
    if (!p.fits_raw) {
        p.note = "the experts (" + std::to_string(p.expert_gib()).substr(0, 4) + " GiB) do not fit in the "
                 "free RAM (" + std::to_string(ram_room >> 30) + " GiB after the reserve)";
        return p;
    }
    if (!p.fits_all) {
        p.note = expert_only
                     ? "the experts fit in RAM; the dense weights stay on disk at this tier"
                     : "the full fp32 dense set (" + std::to_string(p.dense_fp32_gib()).substr(0, 4) +
                           " GiB) plus the experts do not fit in the free RAM; the experts stay resident "
                           "and the dense weights read the mmap windows";
    }

    // VRAM spreads the experts across the cards, file order, one card's room at a time (a stack never
    // splits across cards - the CUDA tier copies a whole stack, the runtime picks one card per row).
    p.card_expert_bytes.assign(cards.size(), 0);
    std::vector<uint64_t> room(cards.size());
    for (size_t c = 0; c < cards.size(); ++c) room[c] = card_room_bytes(cards[c], card_reserve);
    uint64_t unplaced = p.expert_bytes;
    size_t c = 0;
    for (; c < cards.size() && unplaced; ++c) {
        const uint64_t take = std::min(room[c], unplaced);
        p.card_expert_bytes[c] = take;
        unplaced -= take;
    }
    return p;
}

/// The largest set of weights that fits the RAM the caller has room for. `resident_limit` is the RAM the
/// pool may hold (plan_pool's ram_bytes when all fits, else its expert_bytes for the experts-only set);
/// `prefer_vram` says which card a skipped expert stack could have used, so the note can name the card
/// whose memory the skip is about. File order, experts interleaved by layer: a partial run loads whole
/// layers' worth of experts before it starts skipping, so an early token still finds its layers resident.
inline ResidentSet select(const GgufModel& model, const NaiveGeometry& g, uint64_t resident_limit,
                         bool include_dense, uint64_t vram_room_total = 0) {
    ResidentSet s;
    auto take = [&](const std::string& name, uint64_t cost) -> bool {
        if (cost && s.resident_bytes + cost > resident_limit) return false;
        s.resident.push_back(name);
        s.resident_bytes += cost;
        return true;
    };
    std::vector<std::string> names;
    for (const TensorSpec& sp : main_globals()) names.push_back(sp.name);
    std::vector<TensorSpec> ts;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        layer_tensors(l, g, ts);
        char buf[160];
        for (const TensorSpec& sp : ts) {
            std::snprintf(buf, sizeof buf, sp.name, (int) l);
            names.push_back(buf);
        }
    }
    for (const std::string& name : names) {
        size_t shard = 0;
        const TensorInfo* t = model.find(name, &shard);
        if (!t) { s.all = false; s.note = "the plan found no " + name; return s; }
        const uint64_t raw = tensor_payload_bytes(*t);
        const Role r = role_of(name);
        const uint64_t cost = (r == Role::Dense && include_dense) ? t->elements() * 4 : raw;
        const bool wanted = (r == Role::Expert) || include_dense;
        if (!wanted) continue;  // raw tier: dense stays on disk, not skipped-as-over-budget
        if (!take(name, cost)) {
            s.all = false;
            s.skipped_bytes += raw;
            if (s.note.empty())
                s.note = "resident RAM is full at " + name + "; the rest reads the mmap windows" +
                         (vram_room_total ? " (VRAM holds a subset of the resident experts)" : "");
        }
    }
    return s;
}

// ---------------------------------------------------------------- the resident pool itself

/// The resident bytes, in the two tiers plan_pool named: expert stacks kept in their file bytes (the
/// pool owns a copy, byte-for-byte the mmap's), dense tensors kept dequantised to fp32. A name absent
/// from the maps is not resident and the caller reads it through the mmap windows exactly as before.
class NaivePool {
public:
    NaivePool(const GgufModel& model, const NaiveGeometry& g) : m_(model), g_(g) {}

    /// Load `names` into the pool, fp32 for the dense role and raw for the experts (a name whose role
    /// is Transient, missing, or whose bytes do not fit is a refusal that names it; the pool is left
    /// empty). Idempotent per name: a second call loads only the names not already held.
    std::string load(const std::vector<std::string>& names) {
        for (const std::string& name : names) {
            if (raw_.count(name) || fp32_.count(name)) continue;  // already resident
            size_t shard = 0;
            const TensorInfo* t = m_.find(name, &shard);
            if (!t) return "the model has no " + name;
            const uint64_t raw = tensor_payload_bytes(*t);
            if (raw == 0) return name + ": unknown or ragged ggml type " + std::string(t->type_name());
            if (!m_.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
            const uint8_t* src = m_.shard(shard).tensor_data(*t);
            const Role r = role_of(name);
            if (r == Role::Expert) {
                std::vector<uint8_t> copy(raw);
                std::memcpy(copy.data(), src, (size_t) raw);
                RawTensor rt;
                rt.bytes = std::move(copy);
                rt.shape = t->shape;
                rt.type = t->type;
                raw_[name] = std::move(rt);
            } else if (r == Role::Dense) {
                LoadedTensor lt;
                if (const std::string e = load_tensor_f32(m_, name, lt); !e.empty()) return e;
                fp32_[name] = std::move(lt);
            } else {
                return name + " is not a tensor this build reads resident";
            }
        }
        return {};
    }

    /// Load everything the inventory names - the whole model resident. The caller plans first
    /// (plan_pool/select) so this is the "it fits, take it all" path.
    std::string load_all() {
        std::vector<std::string> names;
        for (const TensorSpec& s : main_globals()) names.push_back(s.name);
        std::vector<TensorSpec> ts;
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            layer_tensors(l, g_, ts);
            char buf[160];
            for (const TensorSpec& s : ts) {
                std::snprintf(buf, sizeof buf, s.name, (int) l);
                names.push_back(buf);
            }
        }
        return load(names);
    }

    bool resident(const std::string& name) const {
        return raw_.count(name) || fp32_.count(name);
    }
    /// The pool's fp32 copy of a resident dense tensor, or nullptr when it holds no such copy. The
    /// maps never re-hash after load, so the LoadedTensor (and its data) is stable for the pool's
    /// lifetime - a caller may read it in place. runtime.hpp's norm/router loads use this.
    const LoadedTensor* fp32(const std::string& name) const {
        const auto it = fp32_.find(name);
        return it == fp32_.end() ? nullptr : &it->second;
    }
    size_t resident_count() const { return raw_.size() + fp32_.size(); }
    /// The resident tiers, for the INFO line: expert raw bytes and dense fp32 bytes the pool holds.
    uint64_t resident_expert_bytes() const {
        uint64_t b = 0;
        for (const auto& kv : raw_) b += kv.second.bytes.size();
        return b;
    }
    uint64_t resident_dense_bytes() const {
        uint64_t b = 0;
        for (const auto& kv : fp32_) b += kv.second.data.size() * 4;
        return b;
    }
    /// The expert stacks held in RAM, in file order - what NaiveGpu copies to a card (gpu.hpp). The
    /// bytes are stable for the pool's lifetime (a vector never re-allocated after load), which is
    /// what lets a cudaMemcpyAsync read them while the runtime serves rows.
    struct ResidentExpert {
        std::string name;
        const uint8_t* bytes = nullptr;
        uint64_t bytes_len = 0;
        std::vector<uint64_t> shape;  // {n_in, n_out, n_expert}, GGUF order
        uint32_t type = 0;            // IQ2_XXS / IQ2_XS
    };
    std::vector<ResidentExpert> resident_experts() const {
        std::vector<ResidentExpert> v;
        for (const auto& kv : raw_)
            v.push_back({kv.first, kv.second.bytes.data(), kv.second.bytes.size(), kv.second.shape,
                         kv.second.type});
        return v;
    }

    // ---- the runtime's read path over the resident bytes (bit-identical to loader.hpp's mmap path)

    /// y[m] = sum_k x[k] * W[k, m] over a resident 2-D dense weight, read from its fp32 copy in k order
    /// and summed in double - the same accumulation order runtime.hpp's gemv_raw uses, so the only
    /// difference from the mmap path is that the fp32 was written at load rather than per row. Rows
    /// split over threads the way runtime.hpp splits them (a row's sum is never split, so the numbers
    /// do not move with the thread count). `threads` <= 1 keeps one row at a time.
    bool gemv_f32(const char* name, const float* x, int64_t n_in, int64_t n_out, float* y,
                  int threads = 1) const {
        const auto it = fp32_.find(name);
        if (it == fp32_.end() || it->second.data.size() != (size_t) (n_in * n_out)) return false;
        const float* w = it->second.data.data();
        int rc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads)) reduction(|: rc)
#endif
        for (int64_t mi = 0; mi < n_out; ++mi) {
            const float* row = w + (size_t) mi * (size_t) n_in;
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
            y[mi] = (float) acc;
        }
        (void) rc;
        return true;
    }

    /// y[m] = sum_k x[k] * W[k, m] over a resident weight in its native bytes, dequantised one output
    /// row at a time through ggml's decoder - loader.hpp::gemv_dequant with the source swapped from the
    /// mmap to the pool's copy. Bit-identical to it, because the bytes are byte-for-byte the same.
    bool gemv_raw(const char* name, const float* x, int64_t n_in, int64_t n_out, float* y,
                  int threads = 1) const {
        const auto it = raw_.find(name);
        if (it == raw_.end()) return false;
        const RawTensor& rt = it->second;
        if (rt.shape.size() != 2 || (int64_t) rt.shape[0] != n_in || (int64_t) rt.shape[1] != n_out)
            return false;
        const int64_t bs = ggml_blck_size(ggml_of(rt.type));
        if (n_in <= 0 || n_in % bs != 0) return false;
        const int64_t row_bytes = (int64_t) ggml_payload_bytes(rt.type, (uint64_t) n_in);
        int rc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads)) reduction(|: rc)
#endif
        for (int64_t mi = 0; mi < n_out; ++mi) {
#ifdef _OPENMP
            thread_local
#endif
            std::vector<float> row;
            if ((int64_t) row.size() != n_in) row.resize((size_t) n_in);
            const uint8_t* src = rt.bytes.data() + (size_t) mi * (size_t) row_bytes;
            if (dequant_to_float(rt.type, src, row.data(), n_in)) {
                rc = 1;
                continue;
            }
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
            y[mi] = (float) acc;
        }
        return rc == 0;
    }

    /// One expert slice of a resident 3-D stack, the raw GEMV row by row - expert_gemv_raw with the
    /// source swapped. The stack's GGUF order {n_in, n_out, n_expert} is loader.hpp's.
    bool gemv_expert_raw(const char* name, uint64_t expert, const float* x, int64_t n_in, int64_t n_out,
                         float* y, int threads = 1) const {
        const auto it = raw_.find(name);
        if (it == raw_.end()) return false;
        const RawTensor& rt = it->second;
        if (rt.shape.size() != 3 || (int64_t) rt.shape[0] != n_in || (int64_t) rt.shape[1] != n_out ||
            expert >= rt.shape[2])
            return false;
        const int64_t bs = ggml_blck_size(ggml_of(rt.type));
        if (n_in <= 0 || n_in % bs != 0) return false;
        const uint64_t slice_elems = (uint64_t) n_in * (uint64_t) n_out;
        const uint64_t slice_bytes = ggml_payload_bytes(rt.type, slice_elems);
        const uint64_t row_bytes = ggml_payload_bytes(rt.type, (uint64_t) n_in);
        const uint8_t* slice = rt.bytes.data() + (size_t) expert * (size_t) slice_bytes;
        int rc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(std::max(1, threads)) reduction(|: rc)
#endif
        for (int64_t mi = 0; mi < n_out; ++mi) {
#ifdef _OPENMP
            thread_local
#endif
            std::vector<float> row;
            if ((int64_t) row.size() != n_in) row.resize((size_t) n_in);
            const uint8_t* src = slice + (size_t) mi * (size_t) row_bytes;
            if (dequant_to_float(rt.type, src, row.data(), n_in)) {
                rc = 1;
                continue;
            }
            double acc = 0.0;
            for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
            y[mi] = (float) acc;
        }
        return rc == 0;
    }

    /// One column of a resident 2-D table (a token's embedding row, an output-head column) into fp32.
    /// load_tensor_cols with the source swapped; whole columns only, the caller's n_in must be whole
    /// blocks for a quantised copy.
    bool cols(const char* name, uint64_t col_start, uint64_t cols_n, float* out) const {
        const auto it = raw_.find(name);
        if (it != raw_.end()) {
            const RawTensor& rt = it->second;
            if (rt.shape.size() != 2 || col_start + cols_n > rt.shape[1]) return false;
            const int64_t rows = (int64_t) rt.shape[0];
            const int64_t bs = ggml_blck_size(ggml_of(rt.type));
            if (rows % bs != 0) return false;
            const int64_t row_bytes = (int64_t) ggml_payload_bytes(rt.type, (uint64_t) rows);
            for (uint64_t c = 0; c < cols_n; ++c) {
                const uint8_t* src = rt.bytes.data() + (size_t) (col_start + c) * (size_t) row_bytes;
                if (rt.type == 0 /* F32 */) {
                    std::memcpy(out + (size_t) c * rows, src, (size_t) rows * 4);
                    continue;
                }
                if (dequant_to_float(rt.type, src, out + (size_t) c * rows, rows)) return false;
            }
            return true;
        }
        const auto f = fp32_.find(name);  // a resident fp32 table: a column is a contiguous slice
        if (f == fp32_.end()) return false;
        if (f->second.shape.size() != 2 || col_start + cols_n > f->second.shape[1]) return false;
        const uint64_t rows = f->second.shape[0];
        for (uint64_t c = 0; c < cols_n; ++c)
            std::memcpy(out + (size_t) c * rows,
                        f->second.data.data() + (size_t) (col_start + c) * rows, (size_t) rows * 4);
        return true;
    }

    /// True when this name is resident and a GEMV could read it as a 2-D raw stack (a token-embd/output
    /// copy kept raw at --expert-only): the runtime uses it to pick raw-vs-fp32 without a copy lookup.
    bool resident_raw2d(const std::string& name) const {
        const auto it = raw_.find(name);
        return it != raw_.end() && it->second.shape.size() == 2;
    }

private:
    struct RawTensor {
        std::vector<uint8_t> bytes;
        std::vector<uint64_t> shape;
        uint32_t type = 0;
    };
    const GgufModel& m_;
    NaiveGeometry g_;
    std::map<std::string, RawTensor> raw_;
    std::map<std::string, LoadedTensor> fp32_;
};

}  // namespace strata::naive
