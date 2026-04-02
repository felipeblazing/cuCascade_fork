# Technology Stack: GDS & kvikIO I/O Optimization

**Project:** cuCascade Disk I/O Performance Optimization
**Researched:** 2026-04-02
**Focus:** cuFile batch API, kvikIO async API, GDS configuration for maximum NVMe throughput

## Executive Summary

The current cuCascade GDS backend achieves 582 MiB/s writes vs gdsio's 6.73 GiB/s -- an 11.8x gap. The kvikIO backend achieves 1.20 GiB/s -- a 5.6x gap. Both gaps are explained by specific, fixable implementation patterns identified below: unnecessary D2D staging copies, undersized transfer chunks, single-threaded submission, and unused parallelism settings in kvikIO. No new dependencies are needed; the existing cuFile (direct link) and kvikIO 26.06 libraries already expose every API required.

## What gdsio Does That cuCascade Does Not

**Confidence: HIGH** (verified from NVIDIA Benchmarking and Configuration Guide)

gdsio achieves 6.73 GiB/s writes and 13.35 GiB/s reads on the target NVMe. It does this through:

| gdsio Behavior | cuCascade Current Behavior | Impact |
|----------------|---------------------------|--------|
| Registers user GPU buffer directly with cuFile, then calls cuFileRead/cuFileWrite against it (zero-copy) | Allocates a separate 64MB staging buffer, copies data D2D into staging, then calls cuFile on staging | Extra D2D copy saturates GPU memory bandwidth; halves effective throughput |
| Uses multiple threads (typically 4-8 for sequential, up to 128 for random 4K) each submitting independent I/O | Single-threaded: builds batch params, submits one batch, polls to completion, repeats | Cannot saturate NVMe queue depth; leaves hardware idle between waves |
| Uses 1MB I/O size by default for sequential workloads (configurable via `-i`) | Uses 4MB slot size for batch entries | 4MB is reasonable; 1MB is gdsio's default but the real issue is concurrency not chunk size |
| Opens file once, issues all I/O against single fd | Opens file once per `write_device` call; `write_device_batch` falls back to per-entry `write_device` calls for >64MB | Repeated file open/close and cuFileHandleRegister/Deregister per call adds latency |
| Does NOT use a staging buffer at all -- reads/writes directly to/from the user's GPU allocation | Always copies through a 64MB pre-registered staging buffer | The staging buffer is the primary bottleneck |

## Recommended Optimization Stack

### cuFile Direct API (Eliminate Staging Buffer)

**Confidence: HIGH** (NVIDIA API Reference Guide, Best Practices Guide)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| cuFileRead/cuFileWrite | GDS r1.16+ (CUDA 13.1) | Direct GPU<->disk transfers | Zero-copy path when buffer is registered; avoids D2D staging entirely |
| cuFileBufRegister | GDS r1.16+ | Per-transfer buffer registration | Registers caller's GPU buffer for direct DMA; amortize cost over transfer |
| cuFileBatchIOSubmit | GDS r1.16+ | Parallel I/O for multi-buffer scatter/gather | Submit all column buffers as independent batch entries in a single call |

**Key insight:** The current implementation copies data into a staging buffer and then uses cuFile batch API on the staging buffer. Instead, register the caller's actual GPU buffer directly with cuFileBufRegister, then call cuFileRead/cuFileWrite (or batch API) directly on it. This eliminates the D2D copy entirely.

**Registration pattern for large transfers:**
```cpp
// Register caller's buffer temporarily for the duration of the transfer
registered_buffer reg(dev_ptr, size);  // RAII: cuFileBufRegister in ctor

// Direct I/O -- no staging copy needed
cufile_handle cfh(path, /*write_mode=*/true);
ssize_t ret = cuFileWrite(cfh.handle(), dev_ptr, size, file_offset, 0);

// reg destructor calls cuFileBufDeregister
```

**Registration cost/benefit analysis:**

| Approach | Overhead | Throughput | When to Use |
|----------|----------|------------|-------------|
| Register caller's buffer per-transfer | ~100-500us per registration (BAR mapping) | Near gdsio levels (zero-copy) | Large transfers (>1MB) where registration cost is amortized |
| Use cuFile internal bounce buffers (no registration) | None | Slower for large I/O (internal D2D copy), fine for small I/O | Small transfers (<1MB) or when buffer reuse is low |
| Pre-registered persistent staging buffer (current approach) | One-time registration | Poor -- adds D2D copy overhead | Never for bulk transfers; only valid for tiny metadata |

**4KB alignment requirements for zero-copy path:**
- File offset: must be 4KB aligned
- I/O size: must be 4KB aligned
- Device pointer base address: must be 4KB aligned (cudaMalloc provides this)
- Device pointer offset: must be 4KB aligned

Unaligned operations still work but trigger internal bounce buffer usage automatically.

### cuFile Batch API (For Multi-Buffer Scatter/Gather)

**Confidence: HIGH** (NVIDIA API Reference Guide)

The batch API is appropriate for `write_device_batch` / `read_device_batch` where multiple column buffers need to be written/read at different file offsets. Each column buffer should be a separate batch entry pointing directly to its own GPU memory (not staged).

**Corrected batch pattern:**
```cpp
// Register each column buffer individually
std::vector<registered_buffer> regs;
for (const auto& entry : entries) {
    regs.emplace_back(const_cast<void*>(entry.ptr), entry.size);
}

// Build batch params pointing to actual GPU buffers (not staging)
cufile_batch_guard batch(num_ops);
std::vector<CUfileIOParams_t> params(num_ops);
for (unsigned i = 0; i < num_ops; ++i) {
    params[i].mode                  = CUFILE_BATCH;
    params[i].u.batch.devPtr_base   = const_cast<void*>(entries[i].ptr);
    params[i].u.batch.devPtr_offset = 0;
    params[i].u.batch.file_offset   = entries[i].file_offset;
    params[i].u.batch.size          = entries[i].size;
    params[i].fh                    = cfh.handle();
    params[i].opcode                = CUFILE_WRITE;
    params[i].cookie                = reinterpret_cast<void*>(i);
}
submit_and_wait(batch, params);
```

**Batch API limitations to be aware of:**
- For any batch I/O size bigger than the bounce buffer size (`per_buffer_cache_size_kb`, default 16MB), cuFile uses compatibility (POSIX) mode for that entry. This does NOT apply when buffers are registered.
- Operations within a batch may be reordered relative to each other (good for throughput).
- Maximum batch size is capped by `properties.io_batch_size` from the driver.

### cuFile Stream Async API (Alternative to Batch)

**Confidence: MEDIUM** (API Reference Guide, MagnumIO samples)

cuFileReadAsync / cuFileWriteAsync provide CUDA stream-ordered I/O. This is an alternative to the batch API that integrates better with CUDA stream semantics.

| API | Purpose | Key Advantage |
|-----|---------|---------------|
| cuFileStreamRegister(stream, CU_FILE_STREAM_FIXED_AND_ALIGNED) | Register stream with optimization hints | Flag 0xF enables submission-time optimization; avoids deferred evaluation |
| cuFileWriteAsync(cfh, devPtr, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream) | Stream-ordered write | Integrates with CUDA stream graph; FIFO ordering with other stream work |
| cuFileReadAsync(cfh, devPtr, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream) | Stream-ordered read | Same stream integration benefits |

**When to prefer stream async over batch:**
- When I/O needs to be ordered with respect to GPU compute work on the same stream
- When the caller already has a CUDA stream context (cuCascade does via rmm::cuda_stream_view)
- When all parameters are known at submission time (use CU_FILE_STREAM_FIXED_AND_ALIGNED flag)

**When to prefer batch over stream async:**
- When submitting many independent scatter/gather operations (batch has lower per-entry overhead)
- When operations can be freely reordered for maximum NVMe queue utilization

**Recommendation:** Use batch API for the multi-buffer scatter/gather path (`write_device_batch` / `read_device_batch`). Use direct cuFileRead/cuFileWrite for the single-buffer path (`write_device` / `read_device`). Stream async API is a secondary optimization to explore after the primary D2D elimination.

### kvikIO Optimization

**Confidence: HIGH** (libkvikio docs, RAPIDS kvikio runtime settings)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| kvikio::FileHandle::read / write | 26.06 (already used) | Synchronous single-call I/O | Currently used; internally calls cuFileRead/cuFileWrite |
| kvikio::FileHandle::pread / pwrite | 26.06 (not used) | Parallel multi-threaded I/O | Splits transfer into task_size chunks across thread pool; key for throughput |
| kvikio::FileHandle::read_async / write_async | 26.06 (not used) | CUDA stream-ordered async I/O | Uses cuFileReadAsync/cuFileWriteAsync internally; proper stream integration |
| kvikio::defaults::set_thread_pool_nthreads() | 26.06 | Configure parallelism | Default is 1 thread (!) -- must increase for throughput |
| kvikio::defaults::set_task_size() | 26.06 | Configure chunk size | Default is 4 MiB per task; tune based on NVMe characteristics |
| kvikio::defaults::set_gds_threshold() | 26.06 | Bypass GDS for small I/O | Default 1 MiB; below this threshold, uses POSIX directly |
| kvikio::defaults::set_bounce_buffer_size() | 26.06 | Compat mode bounce buffer | Default 16 MiB; only matters if GDS is unavailable |

**Current kvikIO implementation problems:**

1. **Single-threaded I/O**: Uses `fh.write(dev_ptr, size, file_offset, 0)` which calls synchronous cuFileWrite in a single thread. For a 4 GiB transfer, this is one giant blocking call.

2. **Not using pread/pwrite**: The parallel methods split the transfer into `task_size` chunks and submit them across the thread pool. With default settings (1 thread, 4 MiB tasks), this would give no benefit. But with proper configuration:

```cpp
// At initialization time:
kvikio::defaults::set_thread_pool_nthreads(8);   // Match gdsio's thread count
kvikio::defaults::set_task_size(1ULL << 20);       // 1 MiB tasks (gdsio default)

// In write_device:
kvikio::FileHandle fh(path, "w");
auto fut = fh.pwrite(dev_ptr, size, file_offset);  // Parallel!
auto bytes_written = fut.get();
```

3. **Not using read_async/write_async**: For stream-ordered operation:

```cpp
kvikio::FileHandle fh(path, "w");
auto stream_future = fh.write_async(dev_ptr, size, file_offset, 0, stream.value());
stream_future.check_bytes_done();  // Synchronizes stream and returns byte count
```

**Note:** read_async/write_async require CUDA 12.2+ (available; we have CUDA 13.1). On older CUDA, they fall back to synchronous read/write after stream sync.

### GDS Configuration Tuning (cufile.json)

**Confidence: HIGH** (NVIDIA Configuration Guide, Release Notes)

No cufile.json exists in the project. The system uses defaults from `/etc/cufile.json`. A project-specific configuration should be created and pointed to via the `CUFILE_ENV_PATH_JSON` environment variable.

**Recommended cufile.json for NVMe throughput optimization:**

```json
{
    "logging": {
        "level": "ERROR"
    },
    "profile": {
        "nvtx": false,
        "cufile_stats": 0
    },
    "execution": {
        "max_io_threads": 8,
        "max_io_queue_depth": 128,
        "parallel_io": true,
        "min_io_threshold_size_kb": 8192,
        "max_request_parallelism": 4
    },
    "fs": {
        "generic": {
            "posix_unaligned_writes": false,
            "max_direct_io_size_kb": 16384,
            "max_device_cache_size_kb": 131072,
            "max_device_pinned_mem_size_kb": 33554432,
            "per_buffer_cache_size_kb": 16384
        }
    },
    "properties": {
        "use_poll_mode": false,
        "poll_mode_max_size_kb": 4,
        "allow_compat_mode": true,
        "force_compat_mode": false
    }
}
```

**Key parameters and their effects:**

| Parameter | Default | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `max_direct_io_size_kb` | 16384 (16MB) | 16384 | Controls max single I/O chunk; larger reduces stack overhead. Default is good. |
| `per_buffer_cache_size_kb` | 16384 (16MB) | 16384 | Max bounce buffer per entry. With registered buffers, bounce buffers are not used. Keep default. |
| `max_device_cache_size_kb` | 131072 (128MB) | 131072 | Total bounce buffer pool. With registration, less important. |
| `execution::max_io_threads` | 4 | 8 | Internal cuFile threadpool for parallel I/O splitting. Increase for NVMe saturation. |
| `execution::max_io_queue_depth` | 128 | 128 | Max pending work items. Default is fine. |
| `execution::parallel_io` | true | true | Must be true for any internal parallelism. |
| `execution::min_io_threshold_size_kb` | 8192 (8MB) | 8192 | Transfers larger than this get split for parallelism. Default is good. |
| `execution::max_request_parallelism` | 4 | 4 | Max sub-operations per split. Default is good. |
| `use_poll_mode` | false | false | Polling benefits small I/O (<4KB); not relevant for bulk transfers. |

**GDS v1.17 new feature -- gpu_bounce_buffer_slab_config:**
Version 1.17 adds tunable bounce buffer sizing per mode via `gpu_bounce_buffer_slab_config` in cufile.json. This replaces the uniform `per_buffer_cache_size_kb` with slab-based allocation. This is relevant if using unregistered buffers.

### Large Transfer Strategy

**Confidence: HIGH** (Best Practices Guide, Configuration Guide)

For the 4 GiB transfer case (primary benchmark), the strategy should be:

**Option A: Direct cuFileRead/cuFileWrite with registered buffer (recommended)**
1. Register the entire 4 GiB GPU buffer with cuFileBufRegister
2. Call cuFileWrite(cfh, dev_ptr, 4GiB, file_offset, 0) -- single call
3. cuFile internally splits this into `max_direct_io_size_kb` chunks (16MB default) and submits via its internal threadpool (`max_io_threads` workers)
4. Deregister after transfer

This is what gdsio does (transfer type 0: GPU_DIRECT). Single registration, single API call, internal parallelism handles the rest.

**Option B: Multi-threaded explicit cuFileRead/cuFileWrite**
1. Register the buffer
2. Spawn N threads, each calling cuFileWrite on a different offset range
3. Join threads
4. Deregister

This gives more explicit control and matches gdsio's multi-thread model. More complex but potentially higher throughput if cuFile's internal parallelism is insufficient.

**Option C: Batch API with direct buffer pointers**
1. Register the buffer
2. Split into N batch entries at different offsets, all pointing to the same registered buffer
3. Single cuFileBatchIOSubmit
4. Poll for completion
5. Deregister

Good for the scatter/gather use case (write_device_batch) but for a single contiguous buffer, Option A is simpler and equally fast.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| GDS data path | Direct registered buffer | Staging buffer (current) | Staging adds D2D copy; primary bottleneck |
| GDS API for single buffer | cuFileWrite + registration | cuFileBatchIOSubmit on staging slots | Batch on staging still requires D2D; batch has higher submission overhead for single buffer |
| GDS API for multi-buffer | cuFileBatchIOSubmit with direct pointers | Sequential cuFileWrite per buffer | Batch enables NVMe queue depth utilization across buffers |
| kvikIO parallelism | pwrite with 8 threads, 1MB tasks | Synchronous write (current) | Single-threaded write cannot saturate NVMe |
| Stream integration | cuFile stream async API | None (currently ignoring stream param) | Stream async enables overlap with GPU compute; secondary priority |

## Version Requirements

| Dependency | Minimum Version | Available Version | Notes |
|------------|----------------|-------------------|-------|
| libcufile (GDS) | r1.16 | CUDA 13.1 ships r1.17 | v1.16 removed batch I/O size limitation; v1.17 adds slab-based bounce buffers |
| libkvikio | 26.02 | 26.06 (nightly) | read_async/write_async available since CUDA 12.2 support was added |
| CUDA Toolkit | 12.2 (for stream async) | 13.1 | Stream-ordered cuFile APIs require CUDA 12.2+; fully supported |

## Installation

No new dependencies needed. Both cuFile (via CUDA toolkit) and kvikIO (via pixi) are already linked:

```cmake
# Already in CMakeLists.txt:
find_package(kvikio REQUIRED)
target_link_libraries(cucascade_objects PRIVATE kvikio::kvikio)
find_library(CUFILE_LIB cufile ...)
target_link_libraries(cucascade_objects PRIVATE ${CUFILE_LIB})
```

kvikIO thread pool configuration should be added at backend construction time:
```cpp
// In kvikio_io_backend constructor or initialization:
kvikio::defaults::set_thread_pool_nthreads(8);
kvikio::defaults::set_task_size(1ULL << 20);  // 1 MiB
```

## Summary of Changes Needed (Priority Order)

1. **Eliminate D2D staging copy in GDS backend** -- register caller's buffer directly, call cuFileWrite/cuFileRead on it. This is the single highest-impact change. Expected improvement: 5-10x on write path.

2. **Remove 64MB staging buffer fallback** -- for large transfers, register + direct I/O instead of sequential waves through staging. Eliminates the serial 64-wave bottleneck for 4 GiB transfers.

3. **Fix kvikIO to use pwrite/pread** -- switch from synchronous write() to parallel pwrite(), configure thread pool to 8 threads. Expected improvement: 3-5x.

4. **Fix write_device_batch / read_device_batch** -- register individual column buffers, submit batch with direct pointers instead of gathering into staging. Removes the >64MB fallback-to-sequential codepath.

5. **Add cufile.json configuration** -- tune `execution::max_io_threads` and verify `per_buffer_cache_size_kb` for the target NVMe.

6. **Explore cuFile stream async API** -- for proper CUDA stream integration. Lower priority; delivers correctness improvement (stream ordering) more than throughput.

## Sources

- [NVIDIA GDS Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) -- buffer registration strategy, alignment requirements, API selection matrix
- [NVIDIA GDS API Reference Guide](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) -- cuFileBufRegister cost, batch API details, stream API flags
- [NVIDIA GDS Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) -- cufile.json parameters, gdsio usage, throughput benchmarks
- [NVIDIA GDS Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) -- direct path vs bounce buffer, BAR space mapping
- [NVIDIA GDS Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html) -- v1.16/v1.17 features
- [libkvikio FileHandle API](https://docs.rapids.ai/api/libkvikio/stable/classkvikio_1_1filehandle) -- read_async/write_async/pread/pwrite signatures
- [libkvikio defaults API](https://docs.rapids.ai/api/libkvikio/stable/classkvikio_1_1defaults) -- thread pool, task_size, gds_threshold configuration
- [kvikIO Runtime Settings](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/) -- environment variables and defaults
- [NVIDIA MagnumIO GDS Samples](https://github.com/NVIDIA/MagnumIO/blob/main/gds/samples/README.md) -- batch and async sample code
- [NIXL GDS Plugin](https://github.com/ai-dynamo/nixl/tree/main/src/plugins/cuda_gds) -- real-world cufile.json configuration example
