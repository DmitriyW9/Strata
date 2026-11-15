// tests/naive/fixture_io.hpp - the shared reader for the naive tests' binary fixtures.
//
// A fixture array is one file: an ASCII "<dtype> <rows> <cols>" header line (dtype 'f' float32, 'i'
// int32; flat arrays use rows=count, cols=1) followed by the raw little-endian values. The header
// stays human-scannable (`head -1` shows the shape); the body is exact float32/int32, which is what
// lets a 1M-element router matrix round-trip to C++ at its true bits without 17 MB of text. Both
// generators (tools/naive_kernels_oracle.py, tools/naive_real_fixture.py) write this format; both
// tests (test_naive_kernels.cpp, test_naive_loader.cpp) read it through the helpers here.
//
// Header-only like the code it tests: two executables including one definition cannot drift.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace naive_test {

/// One fixture array: the header's dtype ('f' or 'i'), shape, and flat values kept as both int and float.
struct Arr {
    char dtype = 'f';
    int rows = 0, cols = 1;
    std::vector<float> f;
    std::vector<int> i;
    int size() const { return rows * cols; }
    const float* fp() const { return f.data(); }
    const int* ip() const { return i.data(); }
};

/// Loads `<dir>/<case>__<name>.txt` in the oracle's binary format, exact at float32 bits. Reports the
/// path on any problem and returns false, which lets the caller abort just that case.
inline bool load(const std::string& dir, const std::string& case_, const std::string& name, Arr& out,
                 int& failures) {
    const std::string path = dir + "/" + case_ + "__" + name + ".txt";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "FAIL %s: cannot open fixture %s\n", case_.c_str(), path.c_str());
        ++failures;
        return false;
    }
    std::string header;
    std::getline(in, header);
    std::istringstream hs(header);
    std::string dt;
    if (!(hs >> dt >> out.rows >> out.cols) || dt.size() != 1 || (dt[0] != 'f' && dt[0] != 'i')) {
        std::fprintf(stderr, "FAIL %s: bad header '%s' in %s\n", case_.c_str(), header.c_str(), path.c_str());
        ++failures;
        return false;
    }
    out.dtype = dt[0];
    const int n = out.rows * out.cols;
    std::vector<uint8_t> raw((size_t) n * 4);
    in.read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size());
    if (in.gcount() != (std::streamsize) raw.size()) {
        std::fprintf(stderr, "FAIL %s: %s is short of %d values\n", case_.c_str(), path.c_str(), n);
        ++failures;
        return false;
    }
    out.f.resize((size_t) n);
    out.i.resize((size_t) n);
    for (int j = 0; j < n; ++j) {
        const uint8_t* p = raw.data() + (size_t) j * 4;
        const uint32_t u = (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
        if (out.dtype == 'f')
            std::memcpy(&out.f[(size_t) j], &u, 4);
        else
            out.i[(size_t) j] = (int) (int32_t) u;
    }
    return true;
}

inline void report(const std::string& case_, const char* what, int bad, double max_rel, int& failures) {
    if (bad == 0) {
        std::printf("  ok   %s %s\n", case_.c_str(), what);
    } else {
        std::printf("  FAIL %s %s: %d of the values differ (max rel %g)\n", case_.c_str(), what, bad, max_rel);
        ++failures;
    }
}

// Floats match when within a mixed absolute/relative tolerance; the fp32-vs-double accumulation and
// independent codecs leave a last-bit spread that must pass, a real formula bug must not.
inline void check_floats(const std::string& case_, const char* what, const std::vector<float>& got,
                         const Arr& ref, double abstol, double reltol, int& failures) {
    if ((int) got.size() != ref.size()) {
        std::printf("  FAIL %s %s: got %zu values, fixture has %d\n", case_.c_str(), what, got.size(), ref.size());
        ++failures;
        return;
    }
    int bad = 0;
    double max_rel = 0.0;
    for (size_t j = 0; j < got.size(); ++j) {
        const double a = got[j], b = ref.f[j];
        const double diff = std::fabs(a - b);
        const double tol = abstol + reltol * std::fabs(b);
        if (!(diff <= tol) && !std::isnan(a)) ++bad;                        // NaN in ref is matched by NaN in got
        else if (std::isnan(a) != std::isnan(b)) ++bad;
        const double rel = diff / std::max(std::fabs(b), 1e-12);
        if (rel > max_rel) max_rel = rel;
    }
    report(case_, what, bad, max_rel, failures);
}

inline void check_ints(const std::string& case_, const char* what, const std::vector<int>& got,
                       const Arr& ref, int& failures) {
    if ((int) got.size() != ref.size()) {
        std::printf("  FAIL %s %s: got %zu values, fixture has %d\n", case_.c_str(), what, got.size(), ref.size());
        ++failures;
        return;
    }
    int bad = 0;
    for (size_t j = 0; j < got.size(); ++j) bad += (got[j] != ref.i[j]);
    report(case_, what, bad, 0.0, failures);
}

}  // namespace naive_test
