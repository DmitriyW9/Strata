// tests/naive/test_naive_kernels.cpp - the Naive-N0.5-Flash trunk arithmetic against a reference.
//
// Each sub-command reads one fixture set (written by terminal/tools/naive_kernels_oracle.py, a numpy
// transcription of the model authors' NaiveRT torch references), runs the matching function from
// include/strata/naive/kernels.hpp over the fixture's inputs, and compares the result to the fixture's
// recorded reference: integers (indices, FP8 codes) must match exactly, floats within a tolerance set
// so a real arithmetic bug fails while the fp32/double accumulation and the two independent FP8 codecs
// (this header's frexp split, the oracle's table search) are allowed their last-bit spread.
//
// It loads no weights and touches no GPU: the fixtures hold every input and expected output, so the
// whole check runs on any box, in CI, with the model absent.  That is the point - the trunk's numbers
// are pinned here before any 81 GiB load or CUDA kernel exists.
//
//   test_naive_kernels [fixtures-dir]     all cases; nonzero exit and a printed reason on any mismatch
//
// The fixtures live in tests/naive/fixtures by default (committed); regenerate with
//   python terminal/tools/naive_kernels_oracle.py
// The fixture reader lives in fixture_io.hpp, shared with test_naive_loader.cpp.

#include "fixture_io.hpp"  // the shared fixture reader, also used by test_naive_loader.cpp
#include "strata/naive/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_dir = "tests/naive/fixtures";
int g_failures = 0;

using naive_test::Arr;

// The shared helpers keep their failure count in a parameter; these wrappers bind it to this test's
// counter so the case bodies below read as before.
bool load(const std::string& dir, const std::string& case_, const std::string& name, Arr& out) {
    return naive_test::load(dir, case_, name, out, g_failures);
}
void check_floats(const std::string& case_, const char* what, const std::vector<float>& got, const Arr& ref,
                  double abstol, double reltol) {
    naive_test::check_floats(case_, what, got, ref, abstol, reltol, g_failures);
}
void check_ints(const std::string& case_, const char* what, const std::vector<int>& got, const Arr& ref) {
    naive_test::check_ints(case_, what, got, ref, g_failures);
}

using Loader = bool (*)(Arr&);

// Runs one case's body; g_failures is checked before and after.
bool run_case(void (*body)()) {
    const int before = g_failures;
    body();
    return g_failures == before;
}

void case_rmsnorm() {
    const std::string c = "rmsnorm";
    Arr x, gamma, eps, ref;
    if (!load(g_dir, c, "x", x) || !load(g_dir, c, "gamma", gamma) || !load(g_dir, c, "eps", eps) ||
        !load(g_dir, c, "out", ref))
        return;
    std::vector<float> got((size_t) x.size());
    strata::naive::rmsnorm(x.fp(), gamma.fp(), x.size(), eps.f[0], got.data());
    check_floats(c, "rmsnorm", got, ref, 1e-5, 1e-5);
}

void run_rope(const std::string& c) {
    Arr x, pos, nrot, theta, ref;
    if (!load(g_dir, c, "x", x) || !load(g_dir, c, "pos", pos) || !load(g_dir, c, "n_rot", nrot) ||
        !load(g_dir, c, "theta", theta) || !load(g_dir, c, "out", ref))
        return;
    const int hd = x.cols, nr = nrot.i[0], P = x.rows;
    std::vector<float> got((size_t) x.size());
    for (int p = 0; p < P; ++p)
        strata::naive::rope_head(x.fp() + (size_t) p * hd, hd, nr, pos.i[p], theta.f[0], got.data() + (size_t) p * hd);
    check_floats(c, "rope_head", got, ref, 1e-4, 1e-4);
}

void run_attention(const std::string& c, bool with_sink) {
    Arr q, k, v, sink, first, last, ref;
    if (!load(g_dir, c, "q", q) || !load(g_dir, c, "k", k) || !load(g_dir, c, "v", v) ||
        !load(g_dir, c, "first", first) || !load(g_dir, c, "last", last) || !load(g_dir, c, "out", ref))
        return;
    if (with_sink && !load(g_dir, c, "sink", sink)) return;
    const int H = q.rows, K = q.cols, V = v.cols;
    std::vector<float> got((size_t) H * V);
    strata::naive::swa_attention(q.fp(), H, K, k.fp(), v.fp(), V, with_sink ? sink.fp() : nullptr, first.i[0],
                                 last.i[0], got.data());
    check_floats(c, "attention", got, ref, 1e-4, 1e-4);
}

void case_index_ln() {
    const std::string c = "index_ln";
    Arr x, w, b, eps, ref;
    if (!load(g_dir, c, "x", x) || !load(g_dir, c, "weight", w) || !load(g_dir, c, "bias", b) ||
        !load(g_dir, c, "eps", eps) || !load(g_dir, c, "out", ref))
        return;
    std::vector<float> got((size_t) x.size());
    strata::naive::index_key_layernorm(x.fp(), w.fp(), b.fp(), x.size(), eps.f[0], got.data());
    check_floats(c, "index_key_layernorm", got, ref, 1e-5, 1e-5);
}

void case_fp8_codec() {
    const std::string c = "fp8_codec";
    Arr in, code, decoded, table, val;
    if (!load(g_dir, c, "in_", in) || !load(g_dir, c, "code", code) || !load(g_dir, c, "decoded", decoded) ||
        !load(g_dir, c, "code_table", table) || !load(g_dir, c, "val_table", val))
        return;
    // every finite code decodes to the same value the oracle's table holds
    std::vector<float> got_dec((size_t) table.size());
    for (size_t j = 0; j < (size_t) table.size(); ++j) got_dec[j] = strata::naive::fp8_e4m3_decode((uint8_t) table.i[j]);
    check_floats(c, "decode table", got_dec, val, 0.0, 0.0);                 // exact: decode is bit-exact
    // every grid input encodes to the oracle's code and decodes to the oracle's dequantised value
    std::vector<int> got_code((size_t) in.size());
    std::vector<float> got_val((size_t) in.size());
    for (size_t j = 0; j < (size_t) in.size(); ++j) {
        got_code[j] = strata::naive::fp8_e4m3_encode(in.f[j]);
        got_val[j] = strata::naive::fp8_e4m3_decode((uint8_t) got_code[j]);
    }
    check_ints(c, "encode grid", got_code, code);
    check_floats(c, "decoded grid", got_val, decoded, 0.0, 0.0);
}

void case_fp8_rows() {
    const std::string c = "fp8_rows";
    Arr rows, deq, scale;
    if (!load(g_dir, c, "rows", rows) || !load(g_dir, c, "deq", deq) || !load(g_dir, c, "scale", scale)) return;
    const int m = rows.rows, n = rows.cols;
    std::vector<float> got((size_t) rows.size()), sc((size_t) m);
    strata::naive::fp8_rows_dequant(rows.fp(), m, n, got.data(), sc.data());
    check_floats(c, "rows dequant", got, deq, 1e-6, 1e-6);                  // the code must match exactly -> tiny tol
    check_floats(c, "rows scale", sc, scale, 1e-6, 1e-6);
}

void case_dsa_logits() {
    const std::string c = "dsa_logits";
    Arr iq, ki, w, ref;
    if (!load(g_dir, c, "iq", iq) || !load(g_dir, c, "ki", ki) || !load(g_dir, c, "w", w) ||
        !load(g_dir, c, "out", ref))
        return;
    const int H = iq.rows, HD = iq.cols, VIS = ki.rows;
    std::vector<float> got((size_t) VIS);
    strata::naive::dsa_index_logits(iq.fp(), H, HD, w.fp(), ki.fp(), VIS, got.data());
    check_floats(c, "dsa_index_logits", got, ref, 1e-5, 1e-4);
}

void case_stable_topk() {
    const std::string c = "stable_topk";
    Arr v, k, ref;
    if (!load(g_dir, c, "v", v) || !load(g_dir, c, "k", k) || !load(g_dir, c, "out", ref)) return;
    std::vector<int> got((size_t) k.i[0]);
    strata::naive::stable_topk(v.fp(), v.size(), k.i[0], got.data());
    check_ints(c, "stable_topk", got, ref);
}

void case_moe_route() {
    const std::string c = "moe_route";
    Arr x, w, b, topk, idx, wts, scores;
    if (!load(g_dir, c, "x", x) || !load(g_dir, c, "weight", w) || !load(g_dir, c, "bias", b) ||
        !load(g_dir, c, "topk", topk) || !load(g_dir, c, "idx", idx) || !load(g_dir, c, "wts", wts) ||
        !load(g_dir, c, "scores", scores))
        return;
    const int embd = x.size(), ne = w.rows, k = topk.i[0];
    std::vector<int> got_idx((size_t) k);
    std::vector<float> got_wts((size_t) k);
    strata::naive::moe_route(x.fp(), w.fp(), b.fp(), embd, ne, k, got_idx.data(), got_wts.data());
    check_ints(c, "router experts", got_idx, idx);
    check_floats(c, "router weights", got_wts, wts, 1e-6, 1e-6);
}

void case_moe_combine() {
    const std::string c = "moe_combine";
    Arr eo, w, ref;
    if (!load(g_dir, c, "expert_out", eo) || !load(g_dir, c, "w", w) || !load(g_dir, c, "out", ref)) return;
    std::vector<float> got((size_t) ref.size());
    strata::naive::moe_combine(eo.fp(), w.fp(), w.size(), eo.cols, got.data());
    check_floats(c, "moe_combine", got, ref, 1e-6, 1e-6);
}

void case_silu() {
    const std::string c = "silu";
    Arr g, ref;
    if (!load(g_dir, c, "gate", g) || !load(g_dir, c, "out", ref)) return;
    std::vector<float> got((size_t) g.size());
    for (size_t j = 0; j < (size_t) g.size(); ++j) got[j] = strata::naive::silu(g.f[j]);
    check_floats(c, "silu", got, ref, 1e-6, 1e-6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    std::printf("naive kernel parity (fixtures: %s)\n", g_dir.c_str());
    run_case(case_rmsnorm);
    run_rope("rope_swa");
    run_rope("rope_dsa");
    run_rope("rope_idx");
    run_attention("swa_attn", true);
    run_attention("dsa_attn", false);
    run_case(case_index_ln);
    run_case(case_fp8_codec);
    run_case(case_fp8_rows);
    run_case(case_dsa_logits);
    run_case(case_stable_topk);
    run_case(case_moe_route);
    run_case(case_moe_combine);
    run_case(case_silu);
    if (g_failures) {
        std::printf("%d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("all naive kernel checks passed\n");
    return 0;
}
