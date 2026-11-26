// tests/naive/test_naive_runtime.cpp - the runtime layer of the strata-naive path
// (include/strata/naive/runtime.hpp): the state between rows.  The per-layer ARITHMETIC is pinned
// by the other three naive suites against fixtures from the real weights; this one pins what those
// suites do not cover - the running caches (DSA full-history K/V + E4M3 index keys, the window
// ring), the session rewind, the selection fast path, the sampler, and the GEN line - by running a
// whole small naive model end to end and comparing against an INDEPENDENT numpy reference
// (terminal/tools/naive_runtime_reference.py) that walks the same GGUF bytes with the vectorised decoders.
//
//   test_naive_runtime [repo-fixtures-dir] [scratch-dir] [case]
//     argv[1] is the CTest fixtures dir (its grandgrandparent is the repo root, where tools/ is);
//     scratch default: $TMPDIR/strata_naive_runtime;  case: unit | model | all (default all).
//     python3+numpy absent (or the reference failing) -> skip line + exit 77.
//
// The model is a synthetic GGUF the test writes itself: the shapes the layout gate hard-codes
// (n_embd 4096, vocab 152576) with everything else shrunk to what a CPU test can run in minutes -
// 3 layers (layer 0 DSA+dense, layer 1 window+MoE, layer 2 DSA+MoE), 8 heads, 16 experts top-8,
// window 8, index_top_k 2 so the real top-k path runs from the third token on.  Geometry, layout
// and metadata go through the SAME gates the engine runs (read_geometry, check_gguf_layout) - the
// test model is refused by those gates exactly when a runtime would be.
//
// What is compared, and how strictly:
//   * logits per case, elementwise, at the suites' tolerance (rel 0.008 - the fp32-vs-fp64
//     pre-round dot may flip a bf16 rounding boundary - or abs 2e-3);
//   * the next greedy token and the DSA selections per row per layer, exact ints;
//   * cache discipline, bit-exact: logits after a rewind to the branch point must equal the
//     logits of a fresh full prefill of the same rows (bitwise), and logits after continuing the
//     session with a generated token must equal the reference running the continued prompt.

#include "strata/naive/geometry.hpp"
#include "strata/naive/reader.hpp"
#include "strata/naive/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include "fixture_io.hpp"
#include "synthetic_model.hpp"

using namespace naive_synth;  // the shared synthetic model: build_model / build_metadata / write_gguf

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

// ------------------------------------------------------------------ part 1: the pure pieces
void unit_checks() {
    // GEN line: keys in the middle, ids start at the first token without an '=' (the main parser's
    // rule), comma-split ids, unknown keys skipped, temperature/top_k/seed parsed.
    {
        strata::naive::GenReq q;
        std::string err;
        const std::string line =
            "GEN 64 temperature=0.7 top_p=0.95 top_k=40 seed=12345 logprobs=5 10,200,3000,40000";
        if (!strata::naive::parse_gen_line(line, q, err)) check(false, "GEN parse accepts a full line");
        else check(q.max_new == 64 && q.temperature == 0.7f && q.top_p == 0.95f && q.top_k == 40 &&
                       q.seed == 12345ull && q.ids.size() == 4 && q.ids[3] == 40000,
                   "GEN parse: keys, ignored keys, comma ids");
        q.ids.clear();
        check(!strata::naive::parse_gen_line("GEN 8 temperature=1", q, err) && !err.empty(),
              "GEN without ids is refused");
        strata::naive::GenReq p;
        check(strata::naive::parse_gen_line("GEN 1 7,8,", p, err) && p.ids.size() == 2 && p.temperature == 0.f,
              "GEN greedy default and trailing comma");
    }
    // The fast stable top-k against the O(n^2) reference, tie-heavy vectors included.
    {
        std::mt19937 rng(7);
        bool same = true;
        for (int trial = 0; trial < 60 && same; ++trial) {
            const int n = 1 + (int) (rng() % 17), k = 1 + (int) (rng() % 6);
            std::vector<float> v((size_t) n);
            for (int i = 0; i < n; ++i) v[(size_t) i] = (float) (rng() % 4);   // many exact ties
            std::vector<int> a((size_t) std::min(n, k));
            strata::naive::stable_topk(v.data(), n, k, a.data());
            std::vector<int32_t> b;
            strata::naive::stable_topk_fast(v.data(), n, k, b);
            if (a.size() != b.size() || !std::equal(a.begin(), a.end(), b.begin())) {
                same = false;
                std::printf("  trial n=%d k=%d: %zu vs %zu\n", n, k, a.size(), b.size());
            }
        }
        check(same, "stable_topk_fast == stable_topk on tie-heavy vectors (incl. k >= n)");
    }
    // The sampler: greedy is the first argmax; a temperature draw replays bit-identically on seed.
    {
        std::vector<float> lg(64, -3.f);
        lg[9] = 1.f; lg[40] = 1.f; lg[63] = 0.5f;
        strata::naive::Sampler s;
        strata::naive::GenReq q;
        check(s.sample(lg.data(), (int) lg.size(), q, {}) == 9, "greedy takes the first argmax");
        q.temperature = 2.0f; q.top_k = 64; q.seed = 777;
        s.reseed(777);
        const int32_t a = s.sample(lg.data(), (int) lg.size(), q, {});
        s.reseed(777);
        const int32_t b = s.sample(lg.data(), (int) lg.size(), q, {});
        check(a == b && a != 63, "a sampled step replays bit-identically under its seed");
    }
}

// ------------------------------------------------------------------ part 3: the runtime vs numpy
std::string g_refdir;  // set by model_checks before any load

naive_test::Arr load_ref(const std::string& case_, const std::string& name) {
    naive_test::Arr a;
    int before = g_failures;
    if (!naive_test::load(g_refdir, case_, name, a, g_failures) || g_failures > before) {
        std::fprintf(stderr, "reference fixture %s__%s missing or unreadable\n", case_.c_str(), name.c_str());
        std::exit(3);
    }
    return a;
}

void compare_case_logits(const char* case_, const char* what, const std::vector<float>& got) {
    const naive_test::Arr ref = load_ref(case_, "logits");
    bool finite = true;
    for (float f : got) finite = finite && (f == f);            // check_floats lets NaN==NaN pass;
    check(finite, (std::string(what) + ": the logits are all finite").c_str());   // finiteness is a
    const int before = g_failures;                                              // a claim of its own
    naive_test::check_floats(case_, what, got, ref, 2e-3, 0.008, g_failures);
    if (g_failures > before) {                          // which value moved the most, for the log
        double worst = 0.0; size_t wi = 0;
        for (size_t i = 0; i < got.size() && i < ref.f.size(); ++i) {
            const double tol = 2e-3 + 0.008 * std::fabs((double) ref.f[i]);
            const double d = std::fabs((double) got[i] - (double) ref.f[i]);
            if (d > tol && d / tol > worst) { worst = d / tol; wi = i; }
        }
        std::printf("  worst: id %zu got %.6g ref %.6g (%.1fx over tolerance)\n", wi, (double) got[wi],
                    (double) ref.f[wi], worst);
    }
}

void compare_case_ints(const char* case_, const std::string& name, const char* what,
                       const std::vector<int>& got) {
    const naive_test::Arr ref = load_ref(case_, name);
    naive_test::check_ints(case_, what, got, ref, g_failures);
}

std::vector<int> sel_as_ints(const std::vector<int32_t>& s) {
    return std::vector<int>(s.begin(), s.end());
}

int model_checks(const std::filesystem::path& scratch, const std::filesystem::path& repo_root) {
    const auto gguf_path = scratch / "strata_naive_runtime.gguf";
    std::mt19937 rng(20261118);
    const auto t0 = std::chrono::steady_clock::now();
    // STRATA_NAIVE_TEST_REUSE=1 keeps an already-written model (the generator is deterministic;
    // handy when iterating on the python side).  The normal path always rewrites it.
    const bool reuse = std::getenv("STRATA_NAIVE_TEST_REUSE") != nullptr &&
                       std::filesystem::exists(gguf_path);
    if (!reuse && !write_gguf(gguf_path, build_model(rng), build_metadata())) return 3;
    std::printf("model: %s %s in %.1f s\n", reuse ? "reused" : "wrote", gguf_path.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Open it through the engine's own gates: geometry read, layout checked, runtime initialised.
    strata::GgufModel model(strata::gguf_split_paths(gguf_path.string()));
    strata::naive::NaiveGeometry g;
    if (auto e = strata::naive::read_geometry(model.meta(), g); !e.empty()) {
        std::fprintf(stderr, "read_geometry refused the test model: %s\n", e.c_str());
        return 3;
    }
    if (auto e = strata::naive::check_gguf_layout(model, g); !e.empty()) {
        std::fprintf(stderr, "check_gguf_layout refused the test model: %s\n", e.c_str());
        return 3;
    }
    std::string rerr;
    strata::naive::NaiveRuntime rt(model, g, 4, HISTORY, rerr);
    if (!rt.ok()) { std::fprintf(stderr, "runtime refused the test model: %s\n", rerr.c_str()); return 3; }
    check(true, "the test model passes read_geometry, check_gguf_layout and the runtime gate");

    // The reference: python3 + the vectorised numpy decoders.  Absent or failing -> CTest skip.
    const auto refdir = scratch / "ref";
    std::filesystem::create_directories(refdir);
    const std::string cmd = "python3 '" + (repo_root / "terminal/tools/naive_runtime_reference.py").string() +
                            "' '" + gguf_path.string() + "' '" + refdir.string() + "'";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::printf("SKIP: the numpy reference failed (rc %d: absent python3/numpy?)\n", rc);
        return 77;
    }
    g_refdir = refdir.string();

    const std::vector<int32_t> p1{10, 200, 3000, 40000};
    // ---- p1: prefill 4 rows (logits on the last), comparing the DSA selections row by row.
    for (size_t i = 0; i < p1.size(); ++i) {
        if (auto e = rt.forward_row(p1[i], i + 1 == p1.size()); !e.empty()) {
            std::fprintf(stderr, "forward_row: %s\n", e.c_str());
            return 3;
        }
        compare_case_ints("p1", "sel_l0_r" + std::to_string(i), "p1 row selection (DSA layer 0)",
                          sel_as_ints(rt.selection_at(0)));
        compare_case_ints("p1", "sel_l2_r" + std::to_string(i), "p1 row selection (DSA layer 2)",
                          sel_as_ints(rt.selection_at(2)));
        compare_case_ints("p1", "moe_l1_r" + std::to_string(i), "p1 row router (layer 1)",
                          sel_as_ints(rt.expert_at(1)));
        compare_case_ints("p1", "moe_l2_r" + std::to_string(i), "p1 row router (layer 2)",
                          sel_as_ints(rt.expert_at(2)));
    }
    compare_case_logits("p1", "p1 logits", rt.logits());
    const std::vector<int> next_ref = load_ref("p1", "next").i;
    strata::naive::GenReq greedy;
    strata::naive::Sampler sampler;
    const int32_t t1 = sampler.sample(rt.logits().data(), (int) rt.logits().size(), greedy, rt.session());
    check(t1 == next_ref[0], "p1 next greedy token matches the reference");

    // Continue decoding: feeding t1 from the caches must equal the reference running the CONTINUED
    // prompt (the caches hold rows 0..4 exactly like a fresh prefill of those rows).
    if (auto e = rt.forward_row(t1, true); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 3; }
    {
        const naive_test::Arr ref = load_ref("p1", "logits_step2");
        naive_test::check_floats("p1", "p1 step 2 (decoded from cache) logits", rt.logits(), ref, 2e-3, 0.008,
                                 g_failures);
        compare_case_ints("p1", "next_step2", "p1 step 2 greedy token (cache-fed)",
                          {(int) sampler.sample(rt.logits().data(), (int) rt.logits().size(), greedy,
                                                rt.session())});
    }
    compare_case_ints("p3", "sel_l0_r4", "decoded row selection (DSA layer 0)", sel_as_ints(rt.selection_at(0)));
    compare_case_ints("p3", "sel_l2_r4", "decoded row selection (DSA layer 2)", sel_as_ints(rt.selection_at(2)));
    compare_case_ints("p3", "moe_l1_r4", "decoded row router (layer 1)", sel_as_ints(rt.expert_at(1)));
    compare_case_ints("p3", "moe_l2_r4", "decoded row router (layer 2)", sel_as_ints(rt.expert_at(2)));

    // ---- p2: branch at token 2 (the session rewinds; the rings still reach).  Then the same p2
    // run fresh: bit-identical logits prove the caches hold exactly the branch's rows, no residue.
    const std::vector<int32_t> p2{10, 200, 999};
    const int64_t same = rt.common_prefix(p2);
    check(same == 2, "common_prefix finds the branch point");
    if (!rt.rewind(same > 0 ? same - 1 : 0)) { std::fprintf(stderr, "rewind refused\n"); return 3; }
    check(rt.session_len() == 1, "rewind cut the session to the branch");
    for (size_t i = 1; i < p2.size(); ++i)
        if (auto e = rt.forward_row(p2[i], i + 1 == p2.size()); !e.empty()) {
            std::fprintf(stderr, "p2 forward: %s\n", e.c_str());
            return 3;
        }
    compare_case_ints("p2", "sel_l0_r2", "p2 last-row selection (DSA layer 0)", sel_as_ints(rt.selection_at(0)));
    compare_case_ints("p2", "sel_l2_r2", "p2 last-row selection (DSA layer 2)", sel_as_ints(rt.selection_at(2)));
    compare_case_ints("p2", "moe_l1_r2", "p2 last-row router (layer 1)", sel_as_ints(rt.expert_at(1)));
    compare_case_ints("p2", "moe_l2_r2", "p2 last-row router (layer 2)", sel_as_ints(rt.expert_at(2)));
    const std::vector<float> rewound = rt.logits();
    compare_case_logits("p2", "p2 (rewound) logits", rewound);
    rt.reset();
    for (size_t i = 0; i < p2.size(); ++i)
        if (auto e = rt.forward_row(p2[i], i + 1 == p2.size()); !e.empty()) {
            std::fprintf(stderr, "p2 fresh: %s\n", e.c_str());
            return 3;
        }
    check(rewound == rt.logits(), "a rewound session's logits are BIT-identical to a fresh prefill");
    if (rewound != rt.logits()) {                                // which bits moved, for the log
        size_t bad = 0; double worst = 0.0; size_t wi = 0;
        for (size_t i = 0; i < rewound.size(); ++i) {
            if (rewound[i] != rt.logits()[i]) ++bad;
            const double d = std::fabs((double) rewound[i] - (double) rt.logits()[i]);
            if (d > worst) { worst = d; wi = i; }
        }
        std::printf("  diff: %zu of %zu values, worst %.3g at %zu (got %.6g want %.6g)\n", bad,
                    rewound.size(), worst, wi, (double) rewound[wi], (double) rt.logits()[wi]);
    }

    // ---- the ring's refusal boundary: past `len - cap + window - 1` the rings cannot reach, and
    // rewind() must say so (the caller then resets; a silent wrong-window read would be the bug).
    rt.reset();
    std::vector<int32_t> long_prompt;
    for (int i = 0; i < RING_CAP + 3; ++i) {
        long_prompt.push_back((int32_t) ((i * 7919) % 151000 + 1));
        if (auto e = rt.forward_row(long_prompt.back(), false); !e.empty()) {
            std::fprintf(stderr, "ring test forward: %s\n", e.c_str());
            return 3;
        }
    }
    const int64_t len = rt.session_len();
    check(!rt.rewind(len - RING_CAP + WINDOW - 2), "rewind past the ring's reach is REFUSED");
    check(rt.rewind(len - RING_CAP + WINDOW - 1), "rewind just inside the ring's reach succeeds");
    std::printf("(ring cap %lld, session %lld)\n", (long long) RING_CAP, (long long) len);
    return 0;
}

// Debug case: fingerprint x after every layer of the branch prompt, the same lines the reference's
// `xdump` mode prints (xor of the fp32 bits is bit-exact across implementations; the sum is slack).
int xdump_checks(const std::filesystem::path& scratch, bool pts) {
    const auto gguf_path = scratch / "strata_naive_runtime.gguf";
    strata::GgufModel model(strata::gguf_split_paths(gguf_path.string()));
    strata::naive::NaiveGeometry g;
    if (auto e = strata::naive::read_geometry(model.meta(), g); !e.empty()) return 3;
    std::string rerr;
    strata::naive::NaiveRuntime rt(model, g, 4, HISTORY, rerr);
    if (!rt.ok()) return 3;
    rt.dbg = pts ? 2 : 1;
    auto fp = [](const std::vector<float>& x) {
        uint32_t xo = 0; double s = 0.0;
        for (float f : x) { uint32_t b; std::memcpy(&b, &f, 4); xo ^= b; s += (double) f; }
        return std::make_pair(xo, s);
    };
    for (int32_t t : {10, 200, 999}) {
        rt.dbg_x.clear(); rt.dbg_pts.clear();
        if (auto e = rt.forward_row(t, false); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 3; }
        if (pts) {
            for (const auto& [name, v] : rt.dbg_pts) {
                auto [x, s] = fp(v);
                std::printf("%s %08x %.6f\n", name.c_str(), x, s);
                if (t == 10 && name == "emb") {           // element-level dump of the first row
                    std::ofstream out("logs/emb_cpp.bin", std::ios::binary);
                    out.write((const char*) v.data(), (std::streamsize) v.size() * 4);
                }
            }
        }
        else
            for (const auto& x : rt.dbg_x) { auto [v, s] = fp(x); std::printf("xor %08x sum %.6f\n", v, s); }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path scratch =
        argc > 2 ? std::filesystem::path(argv[2])
                 : std::filesystem::temp_directory_path() / "strata_naive_runtime";
    std::error_code ec;
    std::filesystem::create_directories(scratch, ec);
    // The repo root, found from the fixtures dir CTest passes (…/tests/naive/fixtures) or from the
    // cwd: the first parent that holds terminal/tools/naive_runtime_reference.py, the numpy reference below.
    std::filesystem::path root = argc > 1 ? std::filesystem::path(argv[1]) : "tests/naive/fixtures";
    root = std::filesystem::weakly_canonical(root.parent_path().parent_path().parent_path(), ec);
    for (int up = 0; up < 5 && !std::filesystem::exists(root / "terminal/tools/naive_runtime_reference.py"); ++up)
        root = root.parent_path();
    const std::string only = argc > 3 ? argv[3] : "all";

    if (only == "xdump") return xdump_checks(scratch, false);
    if (only == "xdbg2") return xdump_checks(scratch, true);
    if (only == "all" || only == "unit") unit_checks();
    if (only == "all" || only == "model") {
        const int rc = model_checks(scratch, root);
        if (rc) return rc;
    }
    if (g_failures) { std::printf("%d check(s) FAILED\n", g_failures); return 1; }
    std::printf("all naive runtime checks passed\n");
    return 0;
}
