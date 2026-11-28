# terminal/tools/llama - strata-llama, the engine path that runs, through llama.cpp's own library,
# the models Strata's Flash-Next kernels were not written for (GigaChat4 first of them).  It links
# the vendored third_party/llama.cpp (with the GigaChat4 model added beside it) and speaks the same
# serve protocol as strata-naive (docs: the config's exe/args, run-gigachat4.sh beside this repo).
#
#   cmake -S terminal/tools/llama -B build-llama -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         [-DSTRATA_LLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;89"]
#   cmake --build build-llama --target strata-llama -j
#
# CUDA is off by default: a CPU build needs no toolkit and stays byte-identical to nothing - it is
# its own program, the strata binary does not gain a line from it.
cmake_minimum_required(VERSION 3.21)
project(strata_llama C CXX)

option(STRATA_LLAMA_CUDA "run strata-llama on NVIDIA cards (llama.cpp's CUDA backend)" OFF)
if(STRATA_LLAMA_CUDA)
  set(GGML_CUDA ON CACHE BOOL "strata-llama: llama.cpp's CUDA backend" FORCE)
endif()
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "strata-llama: no downloads at run time" FORCE)

# the vendored tree, with the GigaChat4 model added (see its UPSTREAM note in docs/GIGACHAT4.md)
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp"
    CACHE PATH "the llama.cpp tree strata-llama is built against")

set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_BENCHS    OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama_build)

add_executable(strata-llama strata_llama.cpp)
target_link_libraries(strata-llama PRIVATE llama)
target_compile_features(strata-llama PRIVATE cxx_std_17)
target_compile_definitions(strata-llama PRIVATE STRATA_VERSION="0.0.0-llama")
if(MSVC)
  target_compile_options(strata-llama PRIVATE /utf-8)
  set(STRATA_LLAMA_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/../vision/strata-utf8.manifest")
  if(EXISTS ${STRATA_LLAMA_MANIFEST})
    target_sources(strata-llama PRIVATE ${STRATA_LLAMA_MANIFEST})   # the console in UTF-8, like the other programs
  endif()
endif()
