# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
cuCascade/
├── include/
│   └── cucascade/           # All public headers (installed)
│       ├── data/            # Data representation, batch, repository, converters, disk I/O
│       ├── memory/          # Memory spaces, reservations, allocators, topology
│       └── utils/           # Shared utilities (atomics, overloaded visitor)
│       cuda_utils.hpp       # CUDA utility helpers
│       error.hpp            # Error macros and exception types (CUCASCADE_CUDA_TRY, etc.)
├── src/
│   ├── data/                # Data subsystem implementations + internal headers
│   │   io_backend_internal.hpp  # Internal factory functions for GDS/pipeline backends
│   └── memory/              # Memory subsystem implementations
├── test/
│   ├── data/                # Data subsystem tests
│   ├── memory/              # Memory subsystem tests
│   ├── utils/               # Shared test utilities (cudf helpers, mock memory resources)
│   │   cudf_test_utils.hpp  # cuDF table creation helpers for tests
│   │   cudf_test_utils.cpp
│   │   mock_test_utils.hpp  # Mock memory space / reservation manager factories
│   │   test_memory_resources.hpp
│   └── unittest.cpp         # Catch2 main entry point
├── benchmark/
│   ├── benchmark_main.cpp           # Google Benchmark main
│   ├── benchmark_representation_converter.cpp
│   ├── benchmark_disk_converter.cpp
│   ├── visualize_results.ipynb      # Jupyter notebook for result visualization
│   └── README.md
├── cmake/
│   └── cuCascadeConfig.cmake.in     # Package config template for consumers
├── docs/                    # Architecture and development documentation (Markdown)
├── scripts/
│   ├── generate_api_docs.py
│   └── compare_benchmarks.py
├── CMakeLists.txt           # Root build definition
├── CMakePresets.json        # Build presets (debug, release, relwithdebinfo)
├── pixi.toml                # Pixi environment and dependency configuration
├── pixi.lock                # Committed lockfile
├── Doxyfile                 # Doxygen config — generates from include/ only
├── .clang-format            # clang-format v20 config
├── .pre-commit-config.yaml  # Pre-commit hooks (clang-format, cmake-format, codespell, black)
└── CLAUDE.md                # Project context and conventions for Claude
```

## Directory Purposes

**`include/cucascade/data/`:**
- Purpose: All public data-tier headers
- Key files:
  - `common.hpp` — `idata_representation` abstract base
  - `gpu_data_representation.hpp` — `gpu_table_representation` (wraps `cudf::table`)
  - `cpu_data_representation.hpp` — `host_data_representation`, `host_data_packed_representation`
  - `disk_data_representation.hpp` — `disk_data_representation` (RAII file lifecycle)
  - `data_batch.hpp` — `data_batch`, `batch_state`, `data_batch_processing_handle`, `idata_batch_probe`
  - `data_repository.hpp` — `idata_repository<PtrType>`, type aliases
  - `data_repository_manager.hpp` — `data_repository_manager<PtrType>`, `operator_port_key`
  - `representation_converter.hpp` — `representation_converter_registry`, `converter_key`, `register_builtin_converters()`
  - `disk_io_backend.hpp` — `idisk_io_backend`, `io_backend_type`, `make_io_backend()`
  - `disk_file_format.hpp` — `disk_file_header`, constants, serialization helpers

**`include/cucascade/memory/`:**
- Purpose: All public memory-tier headers
- Key files:
  - `common.hpp` — `Tier` enum, `memory_space_id`, `DeviceMemoryResourceFactoryFn`
  - `config.hpp` — `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config`, `memory_space_config` (variant)
  - `memory_space.hpp` — `memory_space` (non-copyable), `memory_space_hash`
  - `memory_reservation_manager.hpp` — `memory_reservation_manager`, all reservation strategy structs
  - `reservation_manager_configurator.hpp` — `reservation_manager_configurator` (fluent builder)
  - `reservation_aware_resource_adaptor.hpp` — GPU tier allocator with per-stream tracking
  - `fixed_size_host_memory_resource.hpp` — HOST tier fixed-block pool for pinned NUMA memory
  - `disk_access_limiter.hpp` — DISK tier byte-accounting resource (no actual allocator)
  - `disk_table.hpp` — `disk_table_allocation`, `generate_disk_file_path()`
  - `host_table.hpp` — `host_table_allocation`, `column_metadata` (shared by HOST and DISK tiers)
  - `host_table_packed.hpp` — `host_table_packed_allocation` (cudf::pack format)
  - `notification_channel.hpp` — `notification_channel`, `event_notifier`, `notify_on_exit`
  - `stream_pool.hpp` — `exclusive_stream_pool`, `borrowed_stream`
  - `topology_discovery.hpp` — `topology_discovery`, `system_topology_info`, `gpu_topology_info`
  - `memory_reservation.hpp` — `reservation`, `reserved_arena` base
  - `error.hpp` — `MemoryError` enum, `cucascade_out_of_memory`
  - `oom_handling_policy.hpp` — `oom_handling_policy`, `reservation_limit_policy`

**`include/cucascade/utils/`:**
- Purpose: Shared utility types used across subsystems
- Key files:
  - `atomics.hpp` — `atomic_peak_tracker<T>`, `atomic_bounded_counter<T>` (requires `std::integral<T>`)
  - `overloaded.hpp` — `overloaded` helper for `std::visit` with multiple lambdas

**`src/data/`:**
- Purpose: Implementation files for the data subsystem
- Key files:
  - `representation_converter.cpp` — `register_builtin_converters()`, converter implementations
  - `disk_data_representation.cpp` — `disk_data_representation` methods
  - `disk_file_format.cpp` — `serialize_column_metadata()`, `deserialize_column_metadata()`
  - `gds_io_backend.cpp` — raw cuFile/GDS batch I/O backend
  - `kvikio_io_backend.cpp` — kvikIO backend (GDS/POSIX fallback)
  - `pipeline_io_backend.cpp` — double-buffered pinned host pipeline backend
  - `io_backend_internal.hpp` — internal factory declarations (`make_gds_io_backend()`, `make_pipeline_io_backend()`)
  - `gpu_data_representation.cpp`, `cpu_data_representation.cpp` — GPU/host representation methods
  - `data_batch.cpp`, `data_repository.cpp`, `data_repository_manager.cpp`

**`src/memory/`:**
- Purpose: Implementation files for the memory subsystem
- Key files:
  - `memory_space.cpp` — `memory_space` construction, tier-specific allocator wiring
  - `memory_reservation_manager.cpp` — strategy execution, blocking wait logic
  - `reservation_manager_configurator.cpp` — fluent builder `build()` methods
  - `reservation_aware_resource_adaptor.cpp` — GPU allocator with per-stream tracking
  - `fixed_size_host_memory_resource.cpp` — HOST pool allocator
  - `disk_access_limiter.cpp` — DISK accounting resource
  - `topology_discovery.cpp` — NVML + sysfs queries
  - `notification_channel.cpp`, `stream_pool.cpp`, `memory_reservation.cpp`

**`test/data/`:**
- Purpose: Tests for data subsystem components
- Key files: `test_data_batch.cpp`, `test_data_repository.cpp`, `test_data_representation.cpp`, `test_disk_io_backend.cpp`, `test_gpu_disk_converters.cpp`, `test_disk_host_converters.cpp`, `test_representation_converter.cpp`

**`test/memory/`:**
- Purpose: Tests for memory subsystem components
- Key files: `test_memory_reservation_manager.cpp`, `test_topology_discovery.cpp`, `test_small_pinned_host_memory_resource.cpp`, `test_gpu_kernels.cu` / `test_gpu_kernels.cuh`

**`test/utils/`:**
- Purpose: Shared test infrastructure; not compiled into library
- Key files:
  - `cudf_test_utils.hpp` / `cudf_test_utils.cpp` — `create_simple_cudf_table()`, column generation helpers
  - `mock_test_utils.hpp` — `make_mock_memory_space()`, mock reservation manager factories
  - `test_memory_resources.hpp` — lightweight RMM memory resource stubs for testing

## Key File Locations

**Entry Points:**
- `include/cucascade/memory/reservation_manager_configurator.hpp` — system bootstrap (fluent builder)
- `include/cucascade/memory/memory_reservation_manager.hpp` — reservation request API
- `include/cucascade/data/data_repository_manager.hpp` — pipeline data routing
- `include/cucascade/data/representation_converter.hpp` — `register_builtin_converters()` + tier conversion

**Core Error Handling:**
- `include/cucascade/error.hpp` — `CUCASCADE_CUDA_TRY`, `CUCASCADE_FAIL`, `CUCASCADE_ASSERT_CUDA_SUCCESS`, `CUCASCADE_FUNC_RANGE()`, `cucascade::cuda_error`, `cucascade::logic_error`

**Disk I/O:**
- `include/cucascade/data/disk_io_backend.hpp` — `idisk_io_backend` interface + factory
- `include/cucascade/data/disk_file_format.hpp` — binary file format constants and serializers
- `include/cucascade/memory/disk_table.hpp` — `disk_table_allocation` (file path + metadata)
- `src/data/gds_io_backend.cpp` — GDS implementation
- `src/data/kvikio_io_backend.cpp` — kvikIO implementation
- `src/data/pipeline_io_backend.cpp` — pipeline (double-buffer) implementation

**Configuration:**
- `pixi.toml` — dependency environments (default/cuda-13-nightly/cuda-12-nightly/cuda-13-stable/cuda-12-stable)
- `CMakePresets.json` — build presets
- `CMakeLists.txt` — root build; links `CUDAToolkit`, `rmm`, `cudf`, `kvikio`, `numa`, `Threads`

**Testing:**
- `test/unittest.cpp` — Catch2 main
- `test/utils/cudf_test_utils.hpp` — reusable cuDF table factories
- `test/utils/mock_test_utils.hpp` — mock memory resources and spaces

## Naming Conventions

**Files:**
- Headers: `snake_case.hpp` (never `.h` for project headers)
- Source: `snake_case.cpp`; CUDA kernels: `snake_case.cu` / `snake_case.cuh`
- Test files: `test_<module_name>.cpp` — e.g., `test_disk_io_backend.cpp`
- Benchmark files: `benchmark_<module_name>.cpp` — e.g., `benchmark_disk_converter.cpp`

**Directories:**
- Source modules mirror header layout: `src/data/` mirrors `include/cucascade/data/`, `src/memory/` mirrors `include/cucascade/memory/`
- Test directories mirror source: `test/data/`, `test/memory/`

**Classes and types:**
- `snake_case` for all classes and structs: `memory_space`, `data_batch`, `disk_table_allocation`
- Interface classes prefixed with `i`: `idata_representation`, `idata_repository`, `idisk_io_backend`
- Config structs suffixed with `_config`: `gpu_memory_space_config`, `disk_memory_space_config`
- RAII handles suffixed with `_handle`: `data_batch_processing_handle`
- Hash structs suffixed with `_hash`: `memory_space_hash`, `converter_key_hash`
- Exception: `Tier` and `MemoryError` use PascalCase name with UPPER_CASE values

## Where to Add New Code

**New data representation type:**
- Header: `include/cucascade/data/<name>_data_representation.hpp` (derive from `idata_representation`)
- Implementation: `src/data/<name>_data_representation.cpp`
- Register converters: add to `register_builtin_converters()` in `src/data/representation_converter.cpp`
- Tests: `test/data/test_<name>_converters.cpp`

**New I/O backend:**
- Implement `idisk_io_backend` in `src/data/<name>_io_backend.cpp`
- Add value to `io_backend_type` enum in `include/cucascade/data/disk_io_backend.hpp`
- Add case to `make_io_backend()` factory in `src/data/` (declare internal factory in `src/data/io_backend_internal.hpp`)
- Tests: `test/data/test_disk_io_backend.cpp` (existing file; add new test cases)

**New memory tier:**
- Add enum value to `Tier` in `include/cucascade/memory/common.hpp`
- Add config struct in `include/cucascade/memory/config.hpp`; add to `memory_space_config` variant
- Add allocator class in `include/cucascade/memory/` and `src/memory/`
- Add constructor overload in `memory_space` (`include/cucascade/memory/memory_space.hpp`)
- Add allocator variant arm to `reserving_adaptor_type` in `memory_space.hpp`
- Add reservation strategy in `memory_reservation_manager.hpp`
- Add builder method in `reservation_manager_configurator.hpp`

**New reservation strategy:**
- Derive from `reservation_request_strategy` in `include/cucascade/memory/memory_reservation_manager.hpp`
- Implement `get_candidates()` in `src/memory/memory_reservation_manager.cpp`

**New utility:**
- Shared utilities: `include/cucascade/utils/<name>.hpp`
- File-local helpers: anonymous namespace in `.cpp` or test files

**New test:**
- Test file: `test/data/test_<feature>.cpp` or `test/memory/test_<feature>.cpp`
- Register in `test/data/CMakeLists.txt` or `test/memory/CMakeLists.txt`
- Use `test/utils/cudf_test_utils.hpp` for cuDF table construction
- Use `test/utils/mock_test_utils.hpp` for mock memory infrastructure

**New benchmark:**
- Benchmark file: `benchmark/benchmark_<feature>.cpp`
- Register in `benchmark/CMakeLists.txt`

## Special Directories

**`.planning/`:**
- Purpose: GSD workflow artifacts (codebase maps, plans, phase documents)
- Generated: By GSD commands
- Committed: Yes

**`.pixi/`:**
- Purpose: Pixi-managed conda environments (default, cuda-13-nightly, etc.)
- Generated: Yes (by pixi)
- Committed: No (only `pixi.toml` and `pixi.lock` are committed)

**`build/`:**
- Purpose: CMake build outputs (debug/, release/, relwithdebinfo/ subdirectories)
- Generated: Yes
- Committed: No

**`docs/`:**
- Purpose: Architecture and development documentation in Markdown
- Key files: `ARCHITECTURE.md`, `data_batch_state_transitions.md`, `data-management.md`, `memory-management.md`, `development-guide.md`, `topology-and-configuration.md`
- Generated: No (hand-authored)
- Committed: Yes

---

*Structure analysis: 2026-04-02*
