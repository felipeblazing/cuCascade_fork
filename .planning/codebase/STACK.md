# Technology Stack

**Analysis Date:** 2026-04-02

## Languages

**Primary:**
- C++20 - All library source code (`src/memory/`, `src/data/`) and public headers (`include/cucascade/`)
- CUDA C++20 - GPU kernel code (`test/memory/test_gpu_kernels.cu`) and direct CUDA runtime calls throughout `src/`

**Secondary:**
- CMake 3.26.4+ - Build system (`CMakeLists.txt`, `cmake/`, `CMakePresets.json`)
- Python - Utility scripts (`scripts/generate_api_docs.py`, `scripts/compare_benchmarks.py`)

## Runtime

**Environment:**
- Linux only: `linux-64` and `linux-aarch64` (enforced via `pixi.toml` line 4)
- NVIDIA GPU required; compute capability >= 7.5 (Turing or newer)
- CUDA Toolkit 12.9+ (cuda-12 track) or 13+ (cuda-13 track)
- NUMA-aware host: requires `libnuma` (installed via `numactl` pixi dependency)

**Package Manager:**
- Pixi >= 0.59
- Config: `pixi.toml`
- Lockfile: `pixi.lock` (committed)
- Channels: `rapidsai-nightly`, `conda-forge` (default); `rapidsai`, `conda-forge` (cudf-stable feature)

## Frameworks

**Core:**
- RMM (RAPIDS Memory Manager) - GPU/host memory resource abstraction; provides `rmm::mr::device_memory_resource`, `rmm::cuda_stream_view`, `rmm::out_of_memory`, `rmm::bad_alloc`; pulled in via `find_package(rmm REQUIRED CONFIG)` from libcudf installation
- libcudf 26.06 (nightly) / 26.02 (stable) - Columnar data representation; provides `cudf::table`, `cudf::column`, `cudf::type_id`, `cudf::pack`/`unpack`; pulled in via `find_package(cudf REQUIRED CONFIG)`

**Testing:**
- Catch2 v2.13.10 - Unit test framework; fetched via `FetchContent` in `test/CMakeLists.txt`; test executable: `cucascade_tests`

**Benchmarking:**
- Google Benchmark v1.8.3 - Microbenchmark framework; fetched via `FetchContent` in `benchmark/CMakeLists.txt`; benchmark executable: `cucascade_benchmarks`

**Build/Dev:**
- Ninja - Build generator (configured in `CMakePresets.json`)
- sccache - Compiler cache for C, CXX, and CUDA compilers (`CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER`, `CMAKE_CUDA_COMPILER_LAUNCHER` in `CMakePresets.json`)
- clang-format v20.1.4 - Code formatting enforced via pre-commit (`.clang-format` at repo root)
- cmake-format / cmake-lint v0.6.13 - CMake file linting via pre-commit
- black v25.1.0 - Python formatting via pre-commit
- codespell v2.4.1 - Spell checking via pre-commit (ignore list: `.codespell_words`)
- Doxygen - API documentation generation; config: `Doxyfile`; output parsed by `scripts/generate_api_docs.py`

## Key Dependencies

**Critical:**
- `libcudf` 26.06 / 26.02 - Core data representation; `cudf::table` is the GPU-tier data container; all column type handling (LIST, STRUCT, STRING, DICTIONARY32, etc.) delegates to cudf
- `RMM` (via cudf) - `rmm::mr::device_memory_resource` is the base class for all custom allocators; `rmm::cuda_stream_view` is used throughout for CUDA stream propagation
- `CUDA::cudart` - Direct CUDA runtime API calls (`cudaMalloc`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`)
- `CUDA::nvml` - GPU topology discovery via NVML in `src/memory/topology_discovery.cpp`

**I/O Backends:**
- `kvikio` 26.06 / 26.02 - Async disk I/O with automatic GDS/POSIX fallback; used in `src/data/kvikio_io_backend.cpp` via `kvikio::FileHandle`; linked PRIVATE via `kvikio::kvikio`
- `libcufile` (cuFile / GDS) - NVIDIA GPUDirect Storage for direct GPU↔NVMe transfers; `<cufile.h>` used in `src/data/gds_io_backend.cpp`; found via `find_library(CUFILE_LIB cufile ...)` — optional at configure time, required at runtime for GDS backend

**Infrastructure:**
- `libnuma` - NUMA-aware pinned host memory allocation in `src/memory/numa_region_pinned_host_allocator.cpp`; found via `find_library(NUMA_LIB numa REQUIRED)`
- `Threads::Threads` (pthreads) - Thread support; `std::mutex`, `std::condition_variable`, `std::async` throughout
- `fmt` - Format library (pixi dependency; available in environment)
- `nvtx3::nvtx3-cpp` - NVIDIA NVTX profiling annotations; only linked when `CUCASCADE_NVTX=ON`; used via `CUCASCADE_FUNC_RANGE()` macro in `include/cucascade/error.hpp`

## Configuration

**Environment:**
- `CUDAARCHS` - Set by pixi environment activation to select CUDA architecture targets
  - cuda-13: `75-real;80-real;86-real;90a-real;100f-real;120a-real;120`
  - cuda-12: `75-real;80-real;86-real;90a-real;100f-real`
  - CMake fallback: `75 80 86 90`
- `CMAKE_PREFIX_PATH` - Passed through from pixi environment for dependency resolution

**Build options** (all in `CMakeLists.txt`):
- `CUCASCADE_BUILD_TESTS` (default ON) - Adds `test/` subdirectory
- `CUCASCADE_BUILD_BENCHMARKS` (default ON) - Adds `benchmark/` subdirectory
- `CUCASCADE_BUILD_SHARED_LIBS` (default ON) - Builds `libcucascade.so`
- `CUCASCADE_BUILD_STATIC_LIBS` (default ON) - Builds `libcucascade.a`
- `CUCASCADE_NVTX` (default OFF) - Enables NVTX profiling ranges
- `CUCASCADE_WARNINGS_AS_ERRORS` (default ON) - Treats all compiler warnings as errors

**Build configs** (via `CMakePresets.json`):
- `debug` → `build/debug/`
- `release` → `build/release/`
- `relwithdebinfo` → `build/relwithdebinfo/`

## Library Outputs

- `cucascade_shared` (`libcucascade.so`, versioned) - alias `cuCascade::cucascade_shared`
- `cucascade_static` (`libcucascade.a`) - alias `cuCascade::cucascade_static`
- `cuCascade::cucascade` - Default alias pointing to shared if available, else static
- `cucascade_tests` - Test executable linked against Catch2
- `cucascade_benchmarks` - Benchmark executable linked against Google Benchmark
- Headers installed to `include/`; CMake package config at `cmake/cuCascadeConfig.cmake.in`
- Consumers: `find_package(cuCascade)` then `target_link_libraries(... cuCascade::cucascade)`

## Platform Requirements

**Development:**
- Linux x86_64 or aarch64
- NVIDIA GPU with compute capability >= 7.5
- Pixi >= 0.59
- CUDA Toolkit 12.9+ or 13+

**Production / CI:**
- Build runner: `linux-amd64-cpu4`
- Test/Benchmark runner: `linux-amd64-gpu-t4-latest-1` (NVIDIA T4 GPU)

---

*Stack analysis: 2026-04-02*
