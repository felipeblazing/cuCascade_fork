# External Integrations

**Analysis Date:** 2026-04-02

## APIs & External Services

**NVIDIA CUDA Runtime:**
- CUDA Runtime API (`<cuda_runtime_api.h>`) - Used directly throughout `src/` for memory management, stream operations, and device-to-device/device-to-host copies
  - SDK/Client: `CUDA::cudart` (CMake target from `find_package(CUDAToolkit)`)
  - Auth: None (local GPU hardware)
  - Key calls: `cudaMalloc`, `cudaFree`, `cudaMallocHost`, `cudaFreeHost`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaStreamCreate`, `cudaStreamDestroy`

**NVIDIA NVML (GPU Management Library):**
- NVML API - Used in `src/memory/topology_discovery.cpp` to enumerate GPUs, discover NUMA affinity, PCI bus IDs, UUIDs, and PCIe topology between GPUs and NICs
  - SDK/Client: `CUDA::nvml` (CMake target)
  - Auth: None (local hardware access)
  - Output: `system_topology_info` struct with GPU, network device, and storage device info

**NVIDIA cuFile (GPUDirect Storage):**
- cuFile API (`<cufile.h>`) - Used in `src/data/gds_io_backend.cpp` for direct GPU↔NVMe I/O bypassing the CPU
  - SDK/Client: `libcufile` (found via `find_library(CUFILE_LIB cufile ...)` in `CMakeLists.txt`)
  - Auth: None (requires GDS-capable kernel driver and NVMe controller)
  - Key APIs: `cuFileDriverOpen`, `cuFileDriverClose`, `cuFileHandleRegister`, `cuFileHandleDeregister`, `cuFileBufRegister`, `cuFileBufDeregister`, `cuFileBatchIOSetUp`, `cuFileBatchIOSubmit`, `cuFileBatchIOGetStatus`, `cuFileBatchIODestroy`
  - Note: Optional at configure time — `CMakeLists.txt` emits a warning if not found; GDS backend will fail at runtime if absent

**NVIDIA kvikIO:**
- kvikIO library - Used in `src/data/kvikio_io_backend.cpp` for async disk I/O with automatic GDS/POSIX fallback
  - SDK/Client: `kvikio::kvikio` (CMake target from `find_package(kvikio REQUIRED)`)
  - Package: `libkvikio` 26.06 (nightly) / 26.02 (stable) via pixi
  - Auth: None
  - Key API: `kvikio::FileHandle` with `.write()`, `.read()`, `.pwrite()`, `.pread()` methods
  - Implementation: `src/data/kvikio_io_backend.cpp`

## Data Storage

**Databases:**
- None — no database integrations

**GPU Memory (Device Tier):**
- RMM-managed GPU device memory
  - Client: `rmm::mr::device_memory_resource` hierarchy in `include/cucascade/memory/`
  - Allocator chain: `reservation_aware_resource_adaptor` → upstream RMM pool resource
  - Key files: `include/cucascade/memory/reservation_aware_resource_adaptor.hpp`, `src/memory/reservation_aware_resource_adaptor.cpp`

**Host Memory (Pinned Tier):**
- NUMA-aware pinned host memory
  - Client: `libnuma` via `find_library(NUMA_LIB numa REQUIRED)`
  - Key files: `include/cucascade/memory/numa_region_pinned_host_allocator.hpp`, `src/memory/numa_region_pinned_host_allocator.cpp`
  - Also: `fixed_size_host_memory_resource` for fixed-block pinned allocation (`include/cucascade/memory/fixed_size_host_memory_resource.hpp`)

**File Storage (Disk Tier):**
- Local filesystem — custom binary format (`.cucs` convention, not enforced by extension)
  - Format: Custom binary with 32-byte `disk_file_header` (magic `0x43554353`, version, num_columns, data_offset), followed by serialized `column_metadata`, then 4KB-aligned column buffer data
  - Format spec: `include/cucascade/data/disk_file_format.hpp`
  - Implementation: `src/data/disk_data_representation.cpp`, `src/data/disk_file_format.cpp`
  - Three I/O backends selectable at runtime via `io_backend_type` enum:
    - `KVIKIO` — `src/data/kvikio_io_backend.cpp` (kvikIO, auto GDS/POSIX)
    - `GDS` — `src/data/gds_io_backend.cpp` (raw cuFile batch API, 64MB registered staging buffer split into 4MB slots)
    - `PIPELINE` — `src/data/pipeline_io_backend.cpp` (double-buffered pinned host pipeline, D2H + pwrite overlap)

**Caching:**
- None — no external cache service; memory tier hierarchy serves as in-memory cache

## Authentication & Identity

**Auth Provider:**
- None — library has no authentication or identity requirements

## Monitoring & Observability

**Profiling:**
- NVTX3 (NVIDIA Tools Extension) - Optional compile-time integration
  - Enabled via `CUCASCADE_NVTX=ON` CMake option
  - SDK/Client: `nvtx3::nvtx3-cpp` (only linked when option enabled)
  - Usage: `CUCASCADE_FUNC_RANGE()` macro defined in `include/cucascade/error.hpp` wraps function-level ranges in the `libcucascade` domain
  - Custom domain: `cucascade::libcucascade_domain` with name `"libcucascade"`

**Error Tracking:**
- None — errors propagate as C++ exceptions (`cucascade::cuda_error`, `cucascade::logic_error`, `rmm::out_of_memory`, `rmm::bad_alloc`)

**Logs:**
- None — no structured logging framework; errors surface via exceptions with file/line context injected by `CUCASCADE_FAIL` and `CUCASCADE_CUDA_TRY` macros

## CI/CD & Deployment

**Hosting:**
- No deployment target — this is a C++ library, not a service
- Distributed as `libcucascade.so` / `libcucascade.a` with CMake package config

**CI Pipeline:**
- Not determined from repository files (no `.github/`, `.gitlab-ci.yml`, or similar CI config found in the repo root)
- Build runner documented as `linux-amd64-cpu4`
- Test/Benchmark runner documented as `linux-amd64-gpu-t4-latest-1` (NVIDIA T4 GPU)

## Environment Configuration

**Required environment setup:**
- Pixi activates the correct conda environment which sets:
  - `CUDAARCHS` - CUDA architecture targets
  - `CMAKE_PREFIX_PATH` - Path for CMake to find all conda-installed dependencies (cudf, RMM, kvikIO, CUDAToolkit)
- No `.env` files detected in the repository

**Secrets / credentials:**
- None — no API keys, tokens, or external service credentials required

## Webhooks & Callbacks

**Incoming:**
- None — library has no network server or webhook receiver

**Outgoing:**
- None — library has no outbound HTTP or network calls

## System-Level Integrations

**Linux sysfs:**
- `topology_discovery` reads `/sys` filesystem entries to map PCIe topology between GPUs, NICs, and NVMe storage devices
  - Implementation: `src/memory/topology_discovery.cpp`
  - Data returned: `system_topology_info` with `gpu_topology_info`, `network_device_info`, `storage_device_info`

**POSIX I/O:**
- Direct `pread`/`pwrite` via POSIX file descriptors used in:
  - GDS backend `write_host`/`read_host` methods for metadata I/O (`src/data/gds_io_backend.cpp`)
  - Pipeline backend for all host I/O and non-direct-IO metadata paths (`src/data/pipeline_io_backend.cpp`)
  - `O_DIRECT` flag used for NVMe DMA-aligned bulk transfers in pipeline backend

---

*Integration audit: 2026-04-02*
