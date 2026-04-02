# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Two-subsystem layered library — memory management subsystem and data management subsystem — built on a three-tier memory hierarchy (GPU → HOST → DISK).

**Key Characteristics:**
- Three memory tiers represented by `Tier::GPU`, `Tier::HOST`, `Tier::DISK` enum values (`include/cucascade/memory/common.hpp`)
- Strategy pattern for reservation requests; converter registry pattern for data tier transitions
- RAII-first ownership: reservations, processing handles, streams, and disk files all release automatically on destruction
- Thread-safe state machine on every `data_batch` using mutex + condition variable
- C++20 concepts constrain template parameters at compile time; `std::variant` dispatches tier-specific allocator types at runtime

## Two Major Subsystems

### 1. Memory Subsystem (`cucascade::memory`)

Manages capacity accounting, reservation, and allocation within each memory space.

```
reservation_manager_configurator  (fluent builder)
         |
         v
memory_reservation_manager        (owns all memory_spaces)
         |
    per-tier memory_space instances
    ├── GPU:  reservation_aware_resource_adaptor  (wraps RMM pool, per-stream tracking)
    ├── HOST: fixed_size_host_memory_resource     (fixed-block pool of pinned NUMA memory)
    └── DISK: disk_access_limiter                 (byte-accounting only, no actual allocator)
```

### 2. Data Subsystem (`cucascade`)

Manages data lifecycle, tier-specific representations, and conversion between tiers.

```
data_repository_manager<PtrType>
         |
    per-operator idata_repository<PtrType>
         |
    data_batch  (state machine + unique_ptr<idata_representation>)
         |
    idata_representation  (abstract)
    ├── gpu_table_representation      (wraps cudf::table)
    ├── host_data_representation      (direct buffer copies to pinned host memory)
    ├── host_data_packed_representation (cudf::pack format)
    └── disk_data_representation      (file path + column_metadata + RAII file deletion)
```

## Layers

**Configuration Layer:**
- Purpose: Build memory space configs, optionally from NUMA/GPU topology
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`, `include/cucascade/memory/config.hpp`
- Contains: `reservation_manager_configurator` (fluent builder), `gpu_memory_space_config`, `host_memory_space_config`, `disk_memory_space_config`, `memory_space_config` (variant)
- Depends on: `topology_discovery`
- Used by: Application bootstrap code

**Topology Discovery Layer:**
- Purpose: Query NVML and Linux sysfs for GPU-NUMA-NIC-storage topology
- Location: `include/cucascade/memory/topology_discovery.hpp`, `src/memory/topology_discovery.cpp`
- Contains: `topology_discovery`, `system_topology_info`, `gpu_topology_info`, `network_device_info`, `storage_device_info`
- Depends on: NVML, Linux sysfs
- Used by: `reservation_manager_configurator`

**Reservation Manager Layer:**
- Purpose: Coordinate reservation requests across all memory spaces via strategy pattern
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`, `src/memory/memory_reservation_manager.cpp`
- Contains: `memory_reservation_manager`, strategy structs (`any_memory_space_in_tier`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`)
- Depends on: Memory resource layer, `notification_channel`
- Used by: Data layer, application code

**Memory Resource Layer:**
- Purpose: Tier-specific allocation and deallocation with reservation enforcement
- Location: `include/cucascade/memory/reservation_aware_resource_adaptor.hpp`, `include/cucascade/memory/fixed_size_host_memory_resource.hpp`, `include/cucascade/memory/disk_access_limiter.hpp`
- Contains: Per-tier allocators wrapping RMM upstream resources; per-stream/per-thread tracking via atomic counters
- Depends on: RMM, CUDA runtime, `notification_channel`, `atomics` utilities
- Used by: `memory_space`

**Memory Space Layer:**
- Purpose: Represent a single tier+device location; own its allocator; provide streams
- Location: `include/cucascade/memory/memory_space.hpp`, `src/memory/memory_space.cpp`
- Contains: `memory_space` (non-copyable, non-movable), `memory_space_id`, `memory_space_hash`
- `_reservation_allocator` is a `std::variant` selecting `reservation_aware_resource_adaptor` (GPU), `fixed_size_host_memory_resource` (HOST), or `disk_access_limiter` (DISK) at construction time
- Depends on: Config layer, memory resource layer
- Used by: `memory_reservation_manager`, `idata_representation`

**Data Representation Layer:**
- Purpose: Tier-specific data storage format; all derive from `idata_representation`
- Location: `include/cucascade/data/common.hpp`, `include/cucascade/data/gpu_data_representation.hpp`, `include/cucascade/data/cpu_data_representation.hpp`, `include/cucascade/data/disk_data_representation.hpp`
- Contains: `idata_representation` (abstract: `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, templated `cast<T>()`), four concrete types
- `disk_data_representation` owns a `disk_table_allocation` (file path + `column_metadata` vector); destructor deletes the file (RAII)
- Depends on: `memory_space`, cuDF (`cudf::table`)
- Used by: `data_batch`, converter registry

**Converter Registry Layer:**
- Purpose: Type-pair dispatch table for converting between representation types
- Location: `include/cucascade/data/representation_converter.hpp`, `src/data/representation_converter.cpp`
- Contains: `representation_converter_registry`, `converter_key` (`{source_type_index, target_type_index}`), `representation_converter_fn`
- Registration: `register_converter<SourceType, TargetType>(fn)` with static_assert constraints
- Lookup: `convert<TargetType>(source, memory_space, stream)` uses `typeid(source)` at runtime
- `register_builtin_converters()` registers GPU↔HOST and GPU↔DISK and HOST↔DISK converters; overload accepts `shared_ptr<idisk_io_backend>` to select I/O backend
- Depends on: `idata_representation`, `idisk_io_backend`
- Used by: `data_batch::convert_to()`, `data_batch::clone_to()`

**Disk I/O Backend Layer:**
- Purpose: Abstract disk I/O; concrete backends selectable at runtime
- Location: `include/cucascade/data/disk_io_backend.hpp`, `src/data/gds_io_backend.cpp`, `src/data/kvikio_io_backend.cpp`, `src/data/pipeline_io_backend.cpp`, `src/data/io_backend_internal.hpp`
- Contains: `idisk_io_backend` (abstract with `write_device`, `read_device`, `write_host`, `read_host`, `write_device_batch`, `read_device_batch`), `io_backend_type` enum (`KVIKIO`, `GDS`, `PIPELINE`), `make_io_backend(type)` factory
- `GDS` uses raw cuFile batch API; `KVIKIO` uses kvikIO with automatic GDS/POSIX fallback; `PIPELINE` uses double-buffered pinned host transfer for D2H overlap with disk writes
- Depends on: kvikIO, cuFile (GDS)
- Used by: built-in disk converters registered via `register_builtin_converters()`

**Data Batch Layer:**
- Purpose: Lifecycle management; state machine; processing-count reference counting
- Location: `include/cucascade/data/data_batch.hpp`, `src/data/data_batch.cpp`
- Contains: `data_batch` (owns `unique_ptr<idata_representation>`), `batch_state` enum, `data_batch_processing_handle` (RAII, holds `weak_ptr<data_batch>`), `idata_batch_probe` interface, `lock_for_processing_result`
- Allowed state transitions: `idle → in_transit | task_created`, `task_created → processing | idle`, `processing → idle`, `in_transit → idle`
- Depends on: `idata_representation`, `representation_converter_registry`
- Used by: `idata_repository`, application code

**Repository Layer:**
- Purpose: Partitioned, thread-safe collections of batches; blocking pop with state transition
- Location: `include/cucascade/data/data_repository.hpp`, `src/data/data_repository.cpp`
- Contains: `idata_repository<PtrType>` (template: `shared_ptr<data_batch>` or `unique_ptr<data_batch>`), `shared_data_repository`, `unique_data_repository`
- `pop_data_batch(target_state)` blocks on condition variable until a batch can transition
- `pop_data_batch_by_id()` / `get_data_batch_by_id()` for directed retrieval
- Depends on: `data_batch`
- Used by: `data_repository_manager`

**Repository Manager Layer:**
- Purpose: Top-level coordinator; operator-port keyed repository map; unique batch ID generation
- Location: `include/cucascade/data/data_repository_manager.hpp`, `src/data/data_repository_manager.cpp`
- Contains: `data_repository_manager<PtrType>`, `operator_port_key`, `shared_data_repository_manager`, `unique_data_repository_manager`
- Batch IDs generated atomically via `_next_data_batch_id` (`std::atomic<uint64_t>`)
- SFINAE selects copy vs. move in `add_data_batch_impl` based on `PtrType`
- Depends on: `idata_repository`
- Used by: Application pipeline code

## Data Flow

**Downgrade (GPU → HOST → DISK):**

1. Caller calls `memory_space::should_downgrade_memory()` to detect pressure
2. `memory_reservation_manager::request_reservation(any_memory_space_to_downgrade{src, target_tier}, size)` selects target space
3. Repository yields a `data_batch` via `pop_data_batch(batch_state::in_transit)`; batch state enters `in_transit`
4. `data_batch::convert_to<TargetRepresentation>(registry, target_memory_space, stream)` invokes the registered converter lambda
5. Converter produces a new `idata_representation` (e.g., `disk_data_representation` wrapping a `disk_table_allocation`)
6. `data_batch::set_data()` swaps in the new representation; batch returns to `idle`
7. Caller re-inserts batch into repository; old reservation released, triggering `notification_channel` wakeup

**Processing (read from any tier):**

1. Consumer calls `idata_repository::pop_data_batch(batch_state::task_created)` — increments `task_created_count`
2. Consumer calls `data_batch::try_to_lock_for_processing(memory_space_id)` — returns `data_batch_processing_handle`
3. Consumer accesses `data_batch::get_data()->cast<gpu_table_representation>().get_table()` (or appropriate type)
4. `data_batch_processing_handle` destructor decrements `_processing_count`; when zero, batch transitions back to `idle`

**State Management:**
- `data_batch` mutex protects all state transitions and `_processing_count`
- Blocking `wait_to_*` methods use `_internal_cv` on the batch mutex
- `idata_repository` propagates state change notifications via `_state_change_cv` pointer set on each batch

## Key Abstractions

**`idata_representation`:**
- Purpose: Uniform interface for tier-specific storage formats
- Location: `include/cucascade/data/common.hpp`
- Pattern: Abstract base with `get_size_in_bytes()`, `get_uncompressed_data_size_in_bytes()`, `clone()`, and templated `cast<T>()` (requires `std::derived_from<T, idata_representation>`)
- Concrete types: `gpu_table_representation` (`include/cucascade/data/gpu_data_representation.hpp`), `host_data_representation`, `host_data_packed_representation` (`include/cucascade/data/cpu_data_representation.hpp`), `disk_data_representation` (`include/cucascade/data/disk_data_representation.hpp`)

**`memory_space`:**
- Purpose: Owns a single tier+device memory budget and its allocator
- Location: `include/cucascade/memory/memory_space.hpp`
- Pattern: Non-copyable/non-movable; variant-based allocator dispatch; exposes `make_reservation_or_null()`, `should_downgrade_memory()`, `get_disk_mount_path()`
- Template helpers: `get_memory_resource_of<Tier::GPU>()` returns correctly typed allocator pointer

**`representation_converter_registry`:**
- Purpose: Type-pair dispatch for tier conversions
- Location: `include/cucascade/data/representation_converter.hpp`
- Pattern: `unordered_map<converter_key, representation_converter_fn>` keyed by `{typeid(Source), typeid(Target)}`; thread-safe with internal mutex

**`data_batch`:**
- Purpose: Unit of data movement; state machine + reference counting
- Location: `include/cucascade/data/data_batch.hpp`
- Pattern: Owns `unique_ptr<idata_representation>`; `data_batch_processing_handle` holds `weak_ptr` so handle doesn't keep batch alive; `idata_batch_probe` for external state observation callbacks

**`disk_table_allocation` / `disk_file_format`:**
- Purpose: On-disk file descriptor and binary format for a serialized cuDF table
- Location: `include/cucascade/memory/disk_table.hpp`, `include/cucascade/data/disk_file_format.hpp`
- Pattern: File starts with 32-byte `disk_file_header` (magic `0x43554353`, version, num_columns, metadata_size, data_offset); column metadata serialized depth-first; column data aligned to 4096-byte boundaries for GDS DMA

**`idisk_io_backend`:**
- Purpose: Abstraction over GDS, kvikIO, and pipeline I/O strategies
- Location: `include/cucascade/data/disk_io_backend.hpp`
- Pattern: Interface with `write_device`/`read_device`/`write_host`/`read_host` and batch variants; `make_io_backend(io_backend_type)` factory

**`notification_channel`:**
- Purpose: Cross-component signaling when reservations are released
- Location: `include/cucascade/memory/notification_channel.hpp`
- Pattern: Shared ownership via `shared_ptr`; `event_notifier` instances post notifications; `wait()` blocks until notified or shutdown

**`borrowed_stream`:**
- Purpose: RAII borrow of a CUDA stream from `exclusive_stream_pool`
- Location: `include/cucascade/memory/stream_pool.hpp`
- Pattern: Move-only; destructor calls `release_fn` returning stream to pool; acquire policies: `GROW` (create new) or `BLOCK` (wait)

**`reservation_request_strategy`:**
- Purpose: Strategy pattern determining candidate `memory_space` objects for a reservation
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Pattern: Abstract base `get_candidates(manager)`, concrete strategies: `any_memory_space_in_tier`, `specific_memory_space`, `any_memory_space_to_downgrade`, `any_memory_space_to_upgrade`, `any_memory_space_in_tier_with_preference`, `any_memory_space_in_tiers`

## Entry Points

**System Bootstrap:**
- Location: `include/cucascade/memory/reservation_manager_configurator.hpp`
- Triggers: Application initialization
- Responsibilities: Fluent builder collects GPU/HOST/DISK settings → `build()` emits `vector<memory_space_config>` → caller passes to `memory_reservation_manager` constructor

**Reservation Request:**
- Location: `include/cucascade/memory/memory_reservation_manager.hpp`
- Triggers: `request_reservation(strategy, size)`
- Responsibilities: Evaluate strategy → iterate candidate `memory_space` objects → call `make_reservation_or_null()` → block and wait on `_wait_cv` if none available

**Data Batch Submission:**
- Location: `include/cucascade/data/data_repository_manager.hpp`
- Triggers: Pipeline operator producing a batch
- Responsibilities: Assign unique batch ID via `get_next_data_batch_id()` → `add_data_batch()` routes to the operator-port's repository → `add_data_batch` notifies blocked consumers via `_cv`

**Tier Conversion:**
- Location: `include/cucascade/data/data_batch.hpp` (`convert_to<T>()`)
- Triggers: Memory pressure downgrade or upgrade request
- Responsibilities: Lock batch mutex → check `_processing_count == 0` → call `registry.convert<T>()` → replace `_data` with new representation

## Error Handling

**Strategy:** Exception-based with context macros; no error codes in normal API paths (except `MemoryError` enum which bridges to `std::error_code`).

**Patterns:**
- `CUCASCADE_CUDA_TRY(call)` — throws `cucascade::cuda_error` on CUDA runtime failure (defined in `include/cucascade/error.hpp`)
- `CUCASCADE_CUDA_TRY_ALLOC(call, bytes)` — throws `rmm::out_of_memory` for `cudaErrorMemoryAllocation`, `rmm::bad_alloc` otherwise
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — assert in debug builds; no-op in release; used in destructors and `noexcept` paths
- `CUCASCADE_FAIL(msg)` / `CUCASCADE_FAIL(msg, exception_type)` — throws `cucascade::logic_error` or custom type with file/line context
- `cucascade::memory::cucascade_out_of_memory` extends `rmm::out_of_memory` with `error_kind`, `requested_bytes`, `global_usage`, `pool_handle`
- `oom_handling_policy` interface (`include/cucascade/memory/oom_handling_policy.hpp`) allows pluggable OOM recovery; default `throw_on_oom_policy`
- `reservation_limit_policy` handles over-reservation: `ignore`, `fail`, or `increase` strategies

## Cross-Cutting Concerns

**Thread Safety:**
- All public methods on `memory_space`, `data_batch`, `idata_repository`, `data_repository_manager` are mutex-protected
- `representation_converter_registry` uses internal mutex for concurrent register/lookup
- Atomic counters (`atomic_bounded_counter`, `atomic_peak_tracker` in `include/cucascade/utils/atomics.hpp`) for lock-free allocation tracking in hot paths
- `notification_channel` provides cross-component async signaling with `shutdown()` support

**Ownership:**
- `memory_reservation_manager` owns all `memory_space` instances via `vector<unique_ptr<memory_space>>`
- `memory_space` owns its allocator and reservation adaptor
- `reservation` releases bytes back to its `memory_space` on destruction
- `data_batch` owns its `idata_representation` via `unique_ptr`
- `disk_data_representation` owns and deletes its backing file on destruction
- `data_batch_processing_handle` holds `weak_ptr<data_batch>` to avoid preventing batch destruction

**Validation:**
- `data_batch::convert_to()` and `clone_to()` assert `_processing_count == 0` before allowing representation swap
- `pop_data_batch(batch_state::processing)` throws immediately — callers must use `task_created` + `try_to_lock_for_processing()`
- `disk_data_representation::clone()` always throws `cucascade::logic_error` — disk representations must be materialized to another tier via converter

**Profiling (optional):**
- `CUCASCADE_FUNC_RANGE()` macro emits NVTX range when `CUCASCADE_NVTX` compile definition is present
- Custom domain: `cucascade::libcucascade_domain` (defined in `include/cucascade/error.hpp`)

---

*Architecture analysis: 2026-04-02*
