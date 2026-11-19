// include/strata/naive/loader.hpp - reading the Naive-N0.5-Flash weights from the GGUF shards into
// fp32, on top of the metadata-only reader (naive/reader.hpp) and the mmap'd multi-shard view
// (artifact/gguf_reader.hpp).
//
// The model is 81 GiB across 4 shards, so nothing here reads a whole tensor unless asked: it hands
// back a pointer to a tensor's quantised bytes (the mmap, no copy) and dequantises into fp32 on demand.
// The dequant itself is ggml's own per-type converter (`ggml_get_type_traits(t).to_float`), the same
// reference the model authors' runtime loads through - not a re-derived IQ2/IQ3 decoder.  That is a
// deliberate correctness choice: ggml ships and tests those decoders, so the port does not risk a
// transcription of the 1024-entry IQ2 codebook.  What is checked instead, in tests/naive/, is the port's
// own layer: the byte count per tensor matches ggml's row size, the bytes the loader sees are the bytes
// in the file, and the small FP32 tensors (the router, the norms) come back exact - which is enough to
// run the tested router (kernels.hpp) over the real weights.
//
// Shape note, kept honest: GGUF stores dim 0 fastest, so a dequantised tensor is returned in that
// element order (for a {n_embd, n_expert} router, expert-major: row e is elements [e*n_embd, +n_embd)).
// `LoadedTensor` carries the shape so a caller knows this; nothing silently transposes it.
//
// This header includes ggml.h, so any target using it links ggml-base (like tests/core/dequant_q5_1_test).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::naive {

/// The GGUF type id and ggml's `ggml_type` share numbering: the ids the reader names (artifact/
/// gguf_reader.hpp: F32 0, Q8_0 8, Q6_K 14, IQ2_XXS 16, IQ2_XS 17, BF16 30) are ggml's enum values
/// for the same types (ggml.c's traits table spells them "f32", "q8_0", "iq2_xxs", ...), so a
/// TensorInfo's `type` casts straight to ggml's enum.  A test checks both tables row by row.
inline constexpr ggml_type ggml_of(uint32_t gguf_type) { return (ggml_type) gguf_type; }

/// Bytes of a tensor's payload from its ggml type and element count, computed ggml's own way
/// (`type_size` per block of `blck_size`).  Returns 0 for an unknown type or a ragged element count.
/// This is the loader's view of the size; the reader's independent `tensor_payload_bytes` must agree,
/// and a test checks they do for every tensor of the layout - along with both against a hard-coded
/// third table, so a wrong row fails in one place and names itself.
///
/// The `GGML_TYPE_COUNT` gate matters more than it looks: ggml's size functions are a switch that
/// aborts the process on an id outside its enum, and a GGUF's type field is an unvalidated u32 from
/// the file.  Refusing in data (return 0) rather than letting ggml abort is the difference between a
/// loader that reports a bad file and one that dies on it.  The gate also bounds the loader to the
/// types ggml carries (ids 0..42); the fork-only trellis ids this fork's `block_geometry` knows are
/// refused by name - none of them appear in the Naive layout (F32, Q8_0, Q6_K, IQ2_XXS, IQ2_XS, BF16).
inline uint64_t ggml_payload_bytes(uint32_t gguf_type, uint64_t elements) {
    if (gguf_type >= (uint32_t) GGML_TYPE_COUNT) return 0;  // outside ggml's enum: never reach its switch
    const ggml_type t = ggml_of(gguf_type);
    const int64_t bs = ggml_blck_size(t);
    const size_t ts = ggml_type_size(t);
    if (bs <= 0 || ts == 0 || elements % (uint64_t) bs != 0) return 0;
    return (elements / (uint64_t) bs) * (uint64_t) ts;
}

/// Dequantise `n` elements of a ggml-quantised block buffer into fp32, via ggml's reference converter.
/// Returns nullptr on success, or a reason naming the type when there is no CPU dequantiser for it.
/// The id is gated the same way as `ggml_payload_bytes`: `ggml_get_type_traits` indexes a fixed
/// table, so an id at or past GGML_TYPE_COUNT (u32 straight from a file) is refused before the call.
inline const char* dequant_to_float(uint32_t gguf_type, const uint8_t* src, float* dst, int64_t n) {
    if (gguf_type >= (uint32_t) GGML_TYPE_COUNT) return "this type id is outside ggml's type table";
    const ggml_type_traits* tr = ggml_get_type_traits(ggml_of(gguf_type));
    if (!tr || !tr->to_float) return "no CPU dequantizer for this ggml type";
    tr->to_float(src, dst, n);
    return nullptr;
}

/// A tensor read into fp32, in GGUF element order, with its shape and type carried along.
struct LoadedTensor {
    std::vector<float> data;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t elements() const { return data.size(); }
};

/// The raw (still quantised) bytes of one tensor - no copy, it points into the mmap - plus their count.
/// Returns "" or a reason naming the tensor.  This is what a bytes-hash test reads, and what the GPU
/// path will hand to a kernel without ever materialising fp32.
inline std::string raw_tensor(const GgufModel& model, const std::string& name, const uint8_t*& out,
                              uint64_t& bytes) {
    size_t shard = 0;
    const TensorInfo* t = model.find(name, &shard);
    if (!t) return "the model has no " + name;
    const uint64_t n = t->elements();
    const uint64_t expect = ggml_payload_bytes(t->type, n);
    if (expect == 0) return name + ": unknown or ragged ggml type " + t->type_name();
    if (!model.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
    out = model.shard(shard).tensor_data(*t);
    bytes = expect;
    return {};
}

/// Dequantise one named tensor into fp32.  Cheap for the FP32/BF16 tensors (the norms, the router); the
/// caller decides which large quantised stacks to bring back.  "" or a reason naming the tensor.
///
/// F32 is copied, not decoded: ggml's traits carry no `to_float` for F32 (upstream llama.cpp is the
/// same - an f32 tensor is raw values, llama.cpp loads it through its copy path, never a decoder), so
/// asking ggml to dequantise one would be refused by the line above.  The identity is a memcpy from
/// the mmap, which is also the whole cost of reading the router and the norms.
inline std::string load_tensor_f32(const GgufModel& model, const std::string& name, LoadedTensor& out) {
    size_t shard = 0;
    const TensorInfo* t = model.find(name, &shard);
    if (!t) return "the model has no " + name;
    const uint64_t n = t->elements();
    const uint64_t expect = ggml_payload_bytes(t->type, n);
    if (expect == 0) return name + ": unknown or ragged ggml type " + t->type_name();
    if (!model.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
    out.data.resize((size_t) n);
    out.shape = t->shape;
    out.type = t->type;
    if (t->type == 0 /* F32 */) {
        std::memcpy(out.data.data(), model.shard(shard).tensor_data(*t), (size_t) expect);
        return {};
    }
    if (const char* err = dequant_to_float(t->type, model.shard(shard).tensor_data(*t), out.data.data(),
                                           (int64_t) n))
        return name + ": " + err;
    return {};
}

/// One slice of a named 2-D quantised tensor, dequantised into fp32: the elements of column
/// `col_start .. col_start + cols` of a {rows, total_cols} tensor (GGUF order: dim 0 = `rows` is
/// fastest, so a column slice is a set of contiguous per-row blocks, and whole columns only).
/// This is how a token's embedding row is read: token_embd is {n_embd, vocab}, the row of token id
/// `t` is column t, and no other token is touched - a full decode of that tensor would be 2.4 GB.
/// Returns "" or a reason naming the tensor.
inline std::string load_tensor_cols(const GgufModel& model, const std::string& name, uint64_t col_start,
                                    uint64_t cols, LoadedTensor& out) {
    size_t shard = 0;
    const TensorInfo* t = model.find(name, &shard);
    if (!t) return "the model has no " + name;
    if (t->shape.size() != 2) return name + " is not a 2-D tensor";
    const uint64_t rows = t->shape[0];
    if (col_start + cols > t->shape[1])
        return name + ": columns [" + std::to_string(col_start) + "," +
               std::to_string(col_start + cols) + ") lie outside {" + std::to_string(t->shape[0]) + "," +
               std::to_string(t->shape[1]) + "}";
    const uint32_t ty = t->type;
    const int64_t bs = ggml_blck_size(ggml_of(ty));
    if (rows % (uint64_t) bs != 0)
        return name + ": a row of " + std::to_string(rows) + " elements is ragged for " + t->type_name();
    if (ggml_payload_bytes(ty, rows * t->shape[1]) == 0)
        return name + ": unknown ggml type " + t->type_name();
    if (!model.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
    out.shape = {rows, cols};
    out.type = ty;
    out.data.resize((size_t) (rows * cols));
    const uint8_t* base = model.shard(shard).tensor_data(*t);
    // Each column of a {rows, total_cols} tensor is rows elements long and starts at its own element
    // offset (GGUF order: dim 0 is fastest), so a column slice is a stack of per-column block runs.
    // ggml_payload_bytes turns an element offset into a byte offset, so block types stay aligned.
    for (uint64_t c = 0; c < cols; ++c) {
        const uint64_t elem0 = (col_start + c) * rows;  // column c = elements [elem0, elem0+rows)
        const uint8_t* src = base + ggml_payload_bytes(ty, elem0);
        float* dst = out.data.data() + (size_t) c * rows;
        if (ty == 0 /* F32 */) {
            std::memcpy(dst, src, (size_t) rows * 4);
            continue;
        }
        if (const char* err = dequant_to_float(ty, src, dst, (int64_t) rows))
            return name + ": " + err;
    }
    return {};
}

/// y[m] = sum_k x[k] * W_e[k, m] over ONE expert slice of a named 3-D quantised tensor, read row by
/// row with no copy of the slice.  The tensor's GGUF shape is {n_in, n_out, n_expert} ("GGUF
/// dimensions [input, output, expert]" - the file's own quantization.tensor_layout key), so expert e
/// is the element range [e * n_in * n_out, (e + 1) * n_in * n_out) and its element order matches a
/// 2-D {n_in, n_out} weight: output row m is the contiguous slice [m*n_in, m*n_in + n_in).
/// This is what makes a MoE FFN affordable on CPU: one expert of layer 1 is 8.4 M elements (~2 MB
/// quantised); the full stack is 6.4 G elements (12 GiB) and is never touched.
/// Returns "" or a reason naming the tensor.
inline std::string gemv_dequant_3d(const GgufModel& model, const std::string& name, uint64_t expert,
                                   const float* x, int64_t n_in, float* y, int64_t n_out) {
    size_t shard = 0;
    const TensorInfo* t = model.find(name, &shard);
    if (!t) return "the model has no " + name;
    if (t->shape.size() != 3) return name + " is not a 3-D expert stack";
    if ((int64_t) t->shape[0] != n_in)
        return name + " has n_in " + std::to_string(t->shape[0]) + ", this call reads " +
               std::to_string(n_in);
    if ((int64_t) t->shape[1] != n_out)
        return name + " has n_out " + std::to_string(t->shape[1]) + ", this call reads " +
               std::to_string(n_out);
    if (expert >= t->shape[2])
        return name + ": expert " + std::to_string(expert) + " is outside " +
               std::to_string(t->shape[2]) + " experts";
    const uint32_t ty = t->type;
    if (ggml_payload_bytes(ty, t->elements()) == 0)
        return name + ": unknown ggml type " + t->type_name();
    if (n_in <= 0 || n_in % (int64_t) ggml_blck_size(ggml_of(ty)) != 0)
        return name + ": a row of " + std::to_string(n_in) + " elements is ragged for " +
               t->type_name();
    if (n_out <= 0) return name + ": empty GEMV";
    if (!model.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
    const uint8_t* base = model.shard(shard).tensor_data(*t);
    const uint64_t slice_elems = (uint64_t) n_in * (uint64_t) n_out;
    const uint8_t* slice = base + ggml_payload_bytes(ty, expert * slice_elems);
    std::vector<float> row((size_t) n_in);
    for (int64_t m = 0; m < n_out; ++m) {
        const uint8_t* src = slice + ggml_payload_bytes(ty, (uint64_t) m * (uint64_t) n_in);
        if (const char* err = dequant_to_float(ty, src, row.data(), n_in))
            return name + ": " + err;
        double acc = 0.0;
        for (int64_t k = 0; k < n_in; ++k) acc += (double) x[k] * row[(size_t) k];
        y[m] = (float) acc;
    }
    return {};
}

/// y[m] = sum_k x[k] * W[k, m] over a named 2-D weight tensor, read through ggml's decoder one
/// output row at a time, with no fp32 copy of the weight.  The tensor's GGUF shape is
/// {n_embd, n_out} (dim 0 fastest), so output row m is the elements
/// [m*n_embd, m*n_embd + n_embd): a contiguous slice for the contiguous quant types (Q8_0 needs
/// n_embd % 32 == 0, and every Q8_0 weight this model carries has n_embd = 4096, so it holds;
/// Q6_K's row slices need n_embd % 256 == 0, which 4096 also satisfies - the check is written for
/// the row slice, not the whole tensor, so a row-aligned tensor whose total is ragged would work too).
/// A bf16 or f32 row dequantises through the same path as everything else.
///
/// The fp32 buffer is one row (n_embd floats, 16 KiB) reused per output row, so the peak memory of
/// a GEMV is the row, not the weight: 4 MiB for the 12288x4096 Q6_K Q projection, versus 192 MiB if
/// a whole dequantised weight were materialised.  The caller keeps its own accumulation order; here
/// each output row is summed in k order in double, the same widening the tested kernels use.
/// Returns "" or a reason naming the tensor (missing, wrong rank, wrong n_embd, ragged row).
inline std::string gemv_dequant(const GgufModel& model, const std::string& name, const float* x,
                                int64_t n_embd, float* y, int64_t n_out) {
    size_t shard = 0;
    const TensorInfo* t = model.find(name, &shard);
    if (!t) return "the model has no " + name;
    if (t->shape.size() != 2) return name + " is not a 2-D weight";
    if ((int64_t) t->shape[0] != n_embd || (int64_t) t->shape[1] != n_out)
        return name + " is {" + std::to_string(t->shape[0]) + "," + std::to_string(t->shape[1]) +
               "}, this call reads {" + std::to_string(n_embd) + "," + std::to_string(n_out) + "}";
    const uint64_t total = t->elements();
    if (ggml_payload_bytes(t->type, total) == 0)
        return name + ": unknown or ragged ggml type " + t->type_name();
    // The ROW, not only the tensor total, must be whole blocks: {48, 2} is 96 elements - whole
    // blocks - but a row of 48 is one and a half, and the m-th slice would start mid-block.
    if (n_embd <= 0 || n_embd % (int64_t) ggml_blck_size(ggml_of(t->type)) != 0)
        return name + ": a row of " + std::to_string(n_embd) + " elements is ragged for " +
               t->type_name();
    if (n_out <= 0) return name + ": empty GEMV";
    if (!model.in_bounds(*t, shard)) return name + ": payload lies outside its shard";
    const uint8_t* base = model.shard(shard).tensor_data(*t);
    std::vector<float> row((size_t) n_embd);
    for (int64_t m = 0; m < n_out; ++m) {
        // One output row through ggml's decoder: a contiguous block-slice of the tensor's bytes.
        const uint8_t* src = base + ggml_payload_bytes(t->type, (uint64_t) m * (uint64_t) n_embd);
        if (const char* err = dequant_to_float(t->type, src, row.data(), n_embd))
            return name + ": " + err;
        double acc = 0.0;
        for (int64_t k = 0; k < n_embd; ++k) acc += (double) x[k] * row[(size_t) k];
        y[m] = (float) acc;
    }
    return {};
}

}  // namespace strata::naive
