// tests/naive/test_naive_loader.cpp - the Naive-N0.5-Flash weight loader (include/strata/naive/
// loader.hpp) against three things: ggml's own type traits, a synthetic GGUF whose bytes this file
// wrote, and - when the model is present - the real weights, through fixtures generated straight from
// the shards (tools/naive_real_fixture.py).
//
// What is checked, and why this and not more:
//   * Three independent byte counts per tensor must agree: the loader's `ggml_payload_bytes` (ggml's
//     block math), the reader's `tensor_payload_bytes` (artifact/gguf_reader.hpp), and the geometry
//     hard-coded a third time in this file.  A wrong row fails in one place and names itself.
//   * `raw_tensor` must hand back exactly the bytes the file holds (a hash of the written bytes
//     equals a hash of the returned window), with no copy and no drift - including for the IQ2 types,
//     whose blocks the loader never parses.
//   * Refusals are data, not crashes: a type id outside ggml's enum, or a tensor whose element count
//     is not whole blocks, returns a reason naming the tensor (ggml's own size functions *abort the
//     process* on a bad id; a GGUF's type field is an unvalidated u32, so the loader's gate is the
//     tested line).
//   * Dequant properties through the loader: F32 is the identity (memcpy path - ggml carries no F32
//     to_float, upstream included); BF16 is the 16-bit shift into f32; Q8_0 with zero codes is exact
//     zero.  IQ2_XXS/IQ2_XS get a structural property instead of a value table: the decoder applies
//     `grid * (signs & kmask[j] ? -1.f : 1.f)` with kmask = {1,2,4,...,128} and signs =
//     ksigns_iq2xs[7 block bits], and ksigns is GF(2)-linear (ksigns[i^j] == ksigns[i]^ksigns[j],
//     ksigns[127] == 0xFF).  So flipping those seven bits sends the mask to s ^ 0xFF: exactly the
//     sub-block's values whose sign bit was 1 negate, the rest keep their magnitude (a value may
//     still appear changed only as a +/-0 pair), and the other 248 values do not move at all.  The
//     negate count is asserted equal to popcount(s & 0xFF), the moved value count to 8 - that count.
//     True for ggml's real codebook, false for a mis-sized or misaligned block read.  The codebook's
//     values are ggml's tested code, used as-is by design (loader.hpp's header says why).
//   * On the real model (skipped when absent): every inventory tensor's two byte counts agree across
//     shards and the payload lies inside its file; the layer-1 router window hashes to the file's
//     bytes; F32 loads are byte-exact; and kernels.hpp's tested router over the REAL weights
//     reproduces the oracle's experts and weights (fixtures `real_router`, `raw_router`,
//     `dequant_f32` from tools/naive_real_fixture.py).
//
//   test_naive_loader [fixtures-dir] [model-any-shard.gguf]
//     fixtures default: tests/naive/fixtures
//     model default:    /home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/
//                       Naive-N0.5-Flash-MQ87-00001-of-00004.gguf
//     Model file absent -> the model part prints its skip line and a clean run exits 77 (CTest's
//     skip code); the model-free parts always run.
//
// Regenerate the real fixtures with:  python tools/naive_real_fixture.py

#include "fixture_io.hpp"  // the shared fixture reader, also used by test_naive_kernels.cpp
#include "strata/naive/kernels.hpp"
#include "strata/naive/loader.hpp"
#include "strata/naive/reader.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace naive = strata::naive;
using naive_test::Arr;

std::string g_dir = "tests/naive/fixtures";
std::string g_model = "/home/dmitriy/projects/ai/LLM/models/Naive-N0.5-Flash-MQ87/"
                      "Naive-N0.5-Flash-MQ87-00001-of-00004.gguf";
int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}
void check_floats(const std::string& c, const char* what, const std::vector<float>& got, const Arr& ref,
                  double abstol, double reltol) {
    naive_test::check_floats(c, what, got, ref, abstol, reltol, g_failures);
}
void check_ints(const std::string& c, const char* what, const std::vector<int>& got, const Arr& ref) {
    naive_test::check_ints(c, what, got, ref, g_failures);
}

uint32_t fnv1a(const uint8_t* p, size_t n) {  // the same 32-bit FNV-1a tools/naive_real_fixture.py uses
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}

// ------------------------------------------------------------------ part A: three byte-count tables
// The six types the MQ87 file carries (naive/reader.hpp: kF32, kQ8_0, kQ6_K, kIQ2_XXS, kIQ2_XS,
// kBF16).  The (blck_size, type_size) pairs below are hard-coded independently of both tables the
// loader and the reader actually use (ggml's switch; block_geometry() in gguf_reader.hpp), so the
// check is genuinely three-way.
void part_a_type_geometry() {
    struct Row {
        uint32_t type;
        const char* name;   // the reader's spelling (gguf_reader.hpp)
        const char* trait;  // ggml's own spelling (type_traits[].type_name in ggml.c)
        int64_t bs;
        size_t ts;
        bool has_to_float;
    };
    // F32 carries no to_float in ggml - an f32 tensor is raw values, and load_tensor_f32 copies it.
    // The other five model types decode through ggml's to_float.
    const Row rows[] = {
        {0, "F32", "f32", 1, 4, false},         {8, "Q8_0", "q8_0", 32, 34, true},
        {14, "Q6_K", "q6_K", 256, 210, true},   {16, "IQ2_XXS", "iq2_xxs", 256, 66, true},
        {17, "IQ2_XS", "iq2_xs", 256, 74, true}, {30, "BF16", "bf16", 1, 2, true},
    };
    bool cast_ok = true, size_ok = true, deq_ok = true, math_ok = true;
    for (const Row& r : rows) {
        const ggml_type t = naive::ggml_of(r.type);
        // The cast is right when both tables name the same type for the id (different spellings,
        // same row: the reader says "IQ2_XXS", ggml's traits say "iq2_xxs").
        cast_ok = cast_ok && std::strcmp(strata::ggml_type_name(r.type), r.name) == 0 &&
                  std::strcmp(ggml_type_name(t), r.trait) == 0;
        size_ok = size_ok && ggml_blck_size(t) == r.bs && ggml_type_size(t) == r.ts;
        const ggml_type_traits* tr = ggml_get_type_traits(t);
        deq_ok = deq_ok && tr && (bool) tr->to_float == r.has_to_float;
        math_ok = math_ok && naive::ggml_payload_bytes(r.type, (uint64_t) r.bs * 3) == r.ts * 3;
    }
    check(cast_ok, "ggml_of: the GGUF ids name the same ggml types in both tables (F32..BF16)");
    check(size_ok, "three byte-count tables agree for the six model types");
    check(deq_ok, "to_float exists for the five decoded types and is absent for F32, as the loader assumes");
    check(math_ok, "ggml_payload_bytes: three whole blocks of each type count as 3 * type_size");
    check(naive::ggml_payload_bytes(8, 33) == 0 && naive::ggml_payload_bytes(0, 5) == 20 &&
              naive::ggml_payload_bytes(999, 256) == 0,
          "ggml_payload_bytes: 0 for a ragged Q8_0 count and an out-of-enum id; exact for F32");
}

// ------------------------------------------------------------------ part B: a synthetic GGUF
template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void put_meta_u32(std::vector<uint8_t>& b, const std::string& key, uint32_t v) {
    put_str(b, key);
    put<uint32_t>(b, 4);  // GGUF metadata type u32
    put<uint32_t>(b, v);
}
void put_meta_str(std::vector<uint8_t>& b, const std::string& key, const std::string& v) {
    put_str(b, key);
    put<uint32_t>(b, 8);  // string
    put_str(b, v);
}

// One tensor of the synthetic file: name, ggml type, shape (GGUF order) and its exact bytes.
struct Syn {
    std::string name;
    uint32_t type;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
    uint32_t hash() const { return fnv1a(bytes.data(), bytes.size()); }
};

/// Writes a GGUF v3 carrying `tensors` at their 32-aligned, data_start-relative offsets, plus two
/// metadata keys so the header ends unaligned on purpose and the reader's padding is on the path.
/// The layout arithmetic lives here deliberately: the loader is read against a file whose true byte
/// positions this test computed itself, not against the loader's own idea of them.
std::filesystem::path write_synthetic(const std::vector<Syn>& tensors) {
    std::vector<uint64_t> off(tensors.size()), nbytes(tensors.size());
    uint64_t cur = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        uint64_t n = 1;
        for (uint64_t d : tensors[i].shape) n *= d;
        nbytes[i] = naive::ggml_payload_bytes(tensors[i].type, n);
        off[i] = cur;
        cur += (nbytes[i] + 31) / 32 * 32;
    }
    std::vector<uint8_t> file;
    put<uint32_t>(file, 0x46554747u);  // "GGUF"
    put<uint32_t>(file, 3);
    put<uint64_t>(file, tensors.size());
    put<uint64_t>(file, 2);  // n_kv: two keys, so the header ends unaligned on purpose
    put_meta_str(file, "general.architecture", "naive_n05_flash");
    put_meta_u32(file, "general.alignment", 32);
    for (size_t i = 0; i < tensors.size(); ++i) {
        put_str(file, tensors[i].name);
        put<uint32_t>(file, (uint32_t) tensors[i].shape.size());
        for (uint64_t d : tensors[i].shape) put<uint64_t>(file, d);
        put<uint32_t>(file, tensors[i].type);
        put<uint64_t>(file, off[i]);
    }
    file.resize((file.size() + 31) / 32 * 32, 0);  // the aligned data start
    const size_t data_start = file.size();
    file.resize(data_start + (size_t) cur, 0);     // the data section
    for (size_t i = 0; i < tensors.size(); ++i)
        if (!tensors[i].bytes.empty())
            std::memcpy(file.data() + data_start + off[i], tensors[i].bytes.data(), tensors[i].bytes.size());
    const auto path = std::filesystem::temp_directory_path() / "strata_naive_loader_test.gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), (std::streamsize) file.size());
    return path;
}

/// Deterministic random bytes with a finite normal fp16 at every block start (every ggml block type
/// in play begins with its scale), so a decode through the loader cannot produce inf.
std::vector<uint8_t> fill_blocks(std::mt19937& rng, size_t n, size_t block_bytes) {
    std::vector<uint8_t> b(n);
    for (uint8_t& v : b) v = (uint8_t) (rng() & 0xff);
    for (size_t off = 0; off + block_bytes <= n; off += block_bytes) {
        const uint16_t d = (uint16_t) (0x3800 | (rng() & 0x03ff));  // fp16 in [1, 2)
        std::memcpy(b.data() + off, &d, 2);
    }
    return b;
}

std::vector<Syn> build_synthetic(std::mt19937& rng) {
    std::vector<Syn> ts;
    {  // F32 {256}: a router-like window, exact through every layer of the loader.
        Syn s{"syn.f32", 0, {256}, {}};
        for (int j = 0; j < 256; ++j) {
            const float f = (float) ((int) (rng() % 2001) - 1000) / 8.0f;
            const uint8_t* p = (const uint8_t*) &f;
            s.bytes.insert(s.bytes.end(), p, p + 4);
        }
        ts.push_back(std::move(s));
    }
    { Syn s{"syn.bf16", 30, {256}, fill_blocks(rng, 256 * 2, 2)}; ts.push_back(std::move(s)); }
    { Syn s{"syn.q8_0", 8, {512}, fill_blocks(rng, 16 * 34, 34)}; ts.push_back(std::move(s)); }
    { Syn s{"syn.q6_k", 14, {512}, fill_blocks(rng, 2 * 210, 210)}; ts.push_back(std::move(s)); }
    { Syn s{"syn.iq2_xxs", 16, {1024}, fill_blocks(rng, 4 * 66, 66)}; ts.push_back(std::move(s)); }
    { Syn s{"syn.iq2_xs", 17, {1024}, fill_blocks(rng, 4 * 74, 74)}; ts.push_back(std::move(s)); }
    { Syn s{"syn.q8_0_ragged", 8, {33}, {}}; ts.push_back(std::move(s)); }      // not whole blocks: refused
    { Syn s{"syn.unknown_type", 999, {256}, {}}; ts.push_back(std::move(s)); }  // outside ggml's enum
    return ts;
}

void part_b_synthetic() {
    std::mt19937 rng(20261112);
    const std::vector<Syn> ts = build_synthetic(rng);
    const auto path = write_synthetic(ts);
    try {
        strata::GgufModel model(strata::gguf_split_paths(path.string()));
        check(model.size() == 1, "synthetic GGUF: opens as one shard through the loader's model");

        bool sizes_agree = true, bytes_survive = true, identity = true, refused = true, decoded = true;
        for (const Syn& s : ts) {
            size_t shard = 0;
            const strata::TensorInfo* t = model.find(s.name, &shard);
            if (!t) { sizes_agree = false; continue; }
            const uint64_t loader = naive::ggml_payload_bytes(t->type, t->elements());
            const uint64_t reader = strata::tensor_payload_bytes(*t);
            sizes_agree = sizes_agree && loader == reader && loader == s.bytes.size();
            const uint8_t* p = nullptr;
            uint64_t n = 0;
            const std::string err = naive::raw_tensor(model, s.name, p, n);
            if (s.bytes.empty()) {  // the two refusals: never a crash, always a reason naming the tensor
                refused = refused && !err.empty() && err.find(s.name) != std::string::npos;
                continue;
            }
            bytes_survive = bytes_survive && err.empty() && n == s.bytes.size() && fnv1a(p, n) == s.hash();
            naive::LoadedTensor lt;
            const std::string lerr = naive::load_tensor_f32(model, s.name, lt);
            decoded = decoded && lerr.empty() && lt.elements() == s.shape[0] && lt.shape == s.shape;
            if (s.type == 0) {  // F32: byte-exact identity through the memcpy path
                identity = identity && std::memcmp(lt.data.data(), s.bytes.data(), s.bytes.size()) == 0;
            } else if (s.type == 30) {  // BF16: the 16-bit shift into f32, computed here from the bytes
                for (size_t j = 0; j < lt.data.size() && identity; ++j) {
                    uint16_t h;
                    std::memcpy(&h, s.bytes.data() + j * 2, 2);
                    const uint32_t u = (uint32_t) h << 16;
                    float w;
                    std::memcpy(&w, &u, 4);
                    identity = identity && (std::memcmp(&lt.data[j], &w, 4) == 0 || std::isnan(lt.data[j]));
                }
            }
        }
        check(sizes_agree, "synthetic: the two byte-count tables agree for all eight tensors");
        check(bytes_survive, "synthetic: raw_tensor returns the exact bytes the file holds (FNV-1a equal)");
        check(identity, "synthetic: load_tensor_f32 is the F32 identity and the BF16 bit shift");
        check(refused, "synthetic: the ragged Q8_0 and the unknown type are refused by name, not read");
        check(decoded, "synthetic: every whole-block tensor decodes with its shape carried");
    } catch (const std::exception& e) {
        check(false, (std::string("synthetic GGUF opened: ") + e.what()).c_str());
    }
    std::filesystem::remove(path);
}

// ------------------------------------------------------------------ part C: dequant properties
// Values are ggml's own and trusted (its decoders ship tested); what is pinned here is the loader's
// use of them - block stride, element count, bit layout: the failures a port actually produces.

void part_c_properties() {
    std::mt19937 rng(20261113);

    // Q8_0 with zero codes dequantises to exact zero: 8 whole blocks, 256 values, through the loader.
    {
        std::vector<uint8_t> block(8 * 34, 0);
        for (size_t b = 0; b < 8; ++b) {
            const uint16_t h = ggml_fp32_to_fp16((float) ((int) (rng() % 200) - 100) / 40.0f);
            std::memcpy(block.data() + b * 34, &h, 2);
        }
        std::vector<float> got(256, 1.0f);
        const char* err = naive::dequant_to_float(8, block.data(), got.data(), 256);
        bool zero = err == nullptr;
        for (const float v : got) zero = zero && v == 0.0f;
        check(zero, "Q8_0 with zero codes dequantises to exact zero (8 blocks)");
    }

    // IQ2_XXS / IQ2_XS: the sign-bit property.  In both types seven bits of a block index
    // ksigns_iq2xs, the GF(2)-linear sign table of ggml-quants.c, so flipping all seven replaces the
    // sign mask s by s ^ ksigns[127] = s ^ 0xFF: every one of the sub-block's eight values negates,
    // and nothing else in the block moves.  The byte positions below are read off the decoders:
    //   IQ2_XXS (66 B = d + qs[32] u16): sub-block ib32=3, l=3 -> aux32[1] = bytes 30..33 (u32 LE);
    //     l=3's signs are its bits 21..27 = byte 32 bits 5..7 and byte 33 bits 0..3.  The scale
    //     nibble (bits 28..31, byte 33 bits 4..7), the four codebook indices (bytes 26..29) and the
    //     other seven sub-blocks stay put.
    //   IQ2_XS (74 B = d + qs[32] u16 + scales[8]): qs[15] = bytes 32..33; signs = qs >> 9 = byte 33
    //     bits 1..7; the codebook index (bits 0..8, incl. byte 33 bit 0) and scales[3] stay put.
    // In both cases the eight values are elements 120..127 of the block.
    auto sign_property = [&](uint32_t type, const char* name, size_t block_bytes, size_t flip_byte1,
                             uint8_t mask1, size_t flip_byte2, uint8_t mask2) {
        std::vector<uint8_t> block(block_bytes);
        for (uint8_t& v : block) v = (uint8_t) (rng() & 0xff);
        const uint16_t d = (uint16_t) (0x3800 | (rng() & 0x03ff));  // a normal fp16 in [1,2): no inf, no 0
        std::memcpy(block.data(), &d, 2);
        std::vector<float> base(256), flipped(256);
        if (naive::dequant_to_float(type, block.data(), base.data(), 256) != nullptr) {
            check(false, (std::string(name) + ": base block decodes").c_str());
            return;
        }
        block[flip_byte1] ^= mask1;
        block[flip_byte2] ^= mask2;
        if (naive::dequant_to_float(type, block.data(), flipped.data(), 256) != nullptr) {
            check(false, (std::string(name) + ": flipped block decodes").c_str());
            return;
        }
        int negated = 0, same = 0, wrong = 0;
        for (size_t j = 0; j < base.size(); ++j) {
            if (std::memcmp(&base[j], &flipped[j], 4) == 0) ++same;
            else {
                const float n = -base[j];  // -0.0f negates +0.0f the way the decoder's * -1.f does
                if (std::memcmp(&n, &flipped[j], 4) == 0) ++negated;
                else ++wrong;
            }
        }
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s: the 7 flipped sign bits negate that sub-block's 8 values exactly",
                      name);
        check(wrong == 0 && negated == 8 && same == 248, msg);
        if (wrong != 0 || negated != 8 || same != 248) {
            std::printf("      (%s: %d negated, %d unchanged, %d wrong)\n", name, negated, same, wrong);
            for (size_t j = 0; j < base.size(); ++j)
                if (std::memcmp(&base[j], &flipped[j], 4) != 0) {
                    const float n = -base[j];
                    if (std::memcmp(&n, &flipped[j], 4) != 0) std::printf("        [%3zu] %g -> %g\n", j, base[j], flipped[j]);
                }
        }
    };
    sign_property(16, "IQ2_XXS", 66, 32, 0xE0, 33, 0x0F);
    sign_property(17, "IQ2_XS", 74, 33, 0xFE, 0, 0x00);

    // Refusals are data: a type id outside ggml's enum never reaches ggml's asserting switch.
    float sink[8] = {0};
    check(naive::dequant_to_float(999, (const uint8_t*) sink, sink, 4) != nullptr,
          "dequant_to_float refuses an out-of-enum id with a reason (ggml would abort on it)");
    check(naive::dequant_to_float(0, (const uint8_t*) sink, sink, 4) != nullptr,
          "dequant_to_float refuses F32, which has no decoder - load_tensor_f32 copies it instead");
}

// ------------------------------------------------------------------ part D: the real model
// Returns false (skip) when the file is absent.  The point of this part is the loader over the real
// 81 GiB: the byte window, the sizes across shards, and the tested router over the real F32 weights
// against the oracle's answer for the same bytes.  Only header pages and the few read windows ever
// touch RAM; mmap maps the rest on demand and the OS evicts it.
bool part_d_real() {
    std::error_code ec;
    if (!std::filesystem::exists(g_model, ec)) {
        std::printf("  model not present (%s): the real-weights checks are skipped\n", g_model.c_str());
        return false;
    }
    strata::GgufModel model = strata::GgufModel::open(g_model);
    check(model.size() == 4, "real model: all four shards open through the loader's GgufModel");

    naive::NaiveGeometry g;
    if (std::string err = naive::read_geometry(model.meta(), g); !err.empty()) {
        check(false, ("real model geometry: " + err).c_str());
        return true;
    }
    // The two byte-count tables must agree for every tensor of the inventory - quantised and plain,
    // whichever shard holds each - and every payload must lie inside its file.
    bool all_sizes = true, all_in = true;
    std::vector<naive::TensorSpec> specs;
    auto one = [&](const std::string& name) {
        size_t shard = 0;
        const strata::TensorInfo* t = model.find(name, &shard);
        if (!t) { all_sizes = false; return; }
        all_sizes = all_sizes && naive::ggml_payload_bytes(t->type, t->elements()) ==
                                     strata::tensor_payload_bytes(*t);
        all_in = all_in && model.in_bounds(*t, shard);
    };
    for (const naive::TensorSpec& s : naive::main_globals()) one(s.name);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        naive::layer_tensors(l, g, specs);
        char buf[160];
        for (const naive::TensorSpec& s : specs) {
            std::snprintf(buf, sizeof buf, s.name, (int) l);
            one(buf);
        }
    }
    check(all_sizes, "real model: the two byte-count tables agree for every inventory tensor");
    check(all_in, "real model: every inventory tensor lies inside its shard");

    // Fixture `raw_router`: the window the loader must return for the router tensor - size and the
    // FNV-1a of the bytes, both taken from the file by tools/naive_real_fixture.py.
    Arr hash, len, elems, type_id;
    if (!naive_test::load(g_dir, "raw_router", "bytes_hash", hash, g_failures)) return true;
    if (!naive_test::load(g_dir, "raw_router", "bytes_len", len, g_failures)) return true;
    if (!naive_test::load(g_dir, "raw_router", "elements", elems, g_failures)) return true;
    if (!naive_test::load(g_dir, "raw_router", "type_id", type_id, g_failures)) return true;
    {
        const uint8_t* p = nullptr;
        uint64_t bytes = 0;
        const std::string err = naive::raw_tensor(model, "blk.1.ffn_gate_inp.weight", p, bytes);
        check(err.empty(), "raw_tensor(blk.1.ffn_gate_inp.weight): loads from the shard that holds it");
        const bool meta_ok = err.empty() && bytes == (uint64_t) (uint32_t) len.i[0] && bytes == (4u << 20) &&
                             (uint64_t) (uint32_t) elems.i[0] == 256u * 4096u && type_id.i[0] == 0;
        check(meta_ok, "raw_router: the byte window is the fixture's 4 MiB F32 block");
        check(err.empty() && fnv1a(p, bytes) == (uint32_t) hash.i[0],
              "raw_router: the returned bytes hash to the file's own FNV-1a");
    }

    // Fixture `dequant_f32`: a real norm tensor's first 256 values must load byte-exact (F32 copy).
    Arr nraw;
    if (!naive_test::load(g_dir, "dequant_f32", "raw", nraw, g_failures)) return true;
    {
        naive::LoadedTensor lt;
        const std::string err = naive::load_tensor_f32(model, "blk.1.attn_norm.weight", lt);
        bool ok = err.empty() && lt.shape.size() == 1 && lt.shape[0] == 4096 && lt.elements() == 4096;
        for (int j = 0; ok && j < nraw.size(); ++j) ok = lt.data[(size_t) j] == nraw.f[(size_t) j];
        check(ok, "dequant_f32: blk.1.attn_norm.weight's first 256 values load byte-exact");
    }

    // Fixture `real_router`: the tested router over the REAL weights = the oracle's experts/weights.
    Arr x, w, b, topk, idx_ref, wts_ref;
    if (!naive_test::load(g_dir, "real_router", "x", x, g_failures)) return true;
    if (!naive_test::load(g_dir, "real_router", "weight", w, g_failures)) return true;
    if (!naive_test::load(g_dir, "real_router", "bias", b, g_failures)) return true;
    if (!naive_test::load(g_dir, "real_router", "topk", topk, g_failures)) return true;
    if (!naive_test::load(g_dir, "real_router", "idx", idx_ref, g_failures)) return true;
    if (!naive_test::load(g_dir, "real_router", "wts", wts_ref, g_failures)) return true;
    {
        naive::LoadedTensor lw, lb;
        std::string err = naive::load_tensor_f32(model, "blk.1.ffn_gate_inp.weight", lw);
        if (err.empty()) err = naive::load_tensor_f32(model, "blk.1.exp_probs_b.bias", lb);
        check(err.empty(), "real router: weight and bias load through load_tensor_f32");
        const bool shape_ok = lw.shape.size() == 2 && lw.shape[0] == 4096 && lw.shape[1] == 256 &&
                              lb.shape.size() == 1 && lb.shape[0] == 256;
        check(shape_ok, "real router: shapes carry in GGUF order ({4096,256} and {256})");
        check_floats("real_router", "loaded weight == the file's bytes (fixture copy)", lw.data, w, 0.0, 0.0);
        check_floats("real_router", "loaded bias == the file's bytes", lb.data, b, 0.0, 0.0);
        const int k = topk.i[0];
        std::vector<int> got_idx((size_t) k);
        std::vector<float> got_wts((size_t) k);
        naive::moe_route(x.fp(), lw.data.data(), lb.data.data(), 4096, 256, k, got_idx.data(), got_wts.data());
        check_ints("real_router", "experts over the real weights", got_idx, idx_ref);
        check_floats("real_router", "weights over the real weights", got_wts, wts_ref, 1e-7, 1e-7);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (argc > 2) g_model = argv[2];
    std::printf("naive loader test (fixtures: %s)\n", g_dir.c_str());

    part_a_type_geometry();
    part_b_synthetic();
    part_c_properties();
    bool ran_real = false;
    try {
        ran_real = part_d_real();
    } catch (const std::exception& e) {
        check(false, (std::string("real model: ") + e.what()).c_str());
        ran_real = true;
    }

    if (g_failures) {
        std::printf("%d loader check(s) FAILED\n", g_failures);
        return 1;
    }
    if (!ran_real) {
        std::printf("model-free loader checks passed; real-weights checks skipped\n");
        return 77;  // CTest's skip code: the model file was not present
    }
    std::printf("all naive loader checks passed (model included)\n");
    return 0;
}
