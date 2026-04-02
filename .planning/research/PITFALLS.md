# Domain Pitfalls: GDS/cuFile I/O Optimization

**Domain:** GPUDirect Storage and kvikIO I/O backend optimization for GPU-disk data persistence
**Researched:** 2026-04-02
**Confidence:** HIGH (verified against NVIDIA official documentation, current codebase analysis, and peer-reviewed research)

## Critical Pitfalls

Mistakes that cause rewrites, major performance regressions, or silent throughput killers.

---

### Pitfall 1: Staging Buffer D2D Copy -- The Double-Bandwidth Tax

**What goes wrong:** Every GPU-to-disk transfer first copies data from the source GPU buffer into a pre-registered staging buffer (D2D copy on GPU), then GDS DMA's the staging buffer to disk. This doubles GPU memory bandwidth consumption and serializes what should be a direct DMA operation.

**Why it happens:** The current `gds_io_backend` allocates a 64MB staging buffer (`_staging_ptr`) via `cudaMalloc`, registers it with `cuFileBufRegister`, then copies user data into it before submitting I/O. This was a safe initial approach because RMM-allocated user buffers are not automatically registered with cuFile, and unregistered buffers trigger cuFile's *own* internal bounce buffer path anyway.

**How this manifests in cuCascade:** Lines 304-309 in `gds_io_backend.cpp` -- `cudaMemcpyAsync` D2D into staging, then `cudaStreamSynchronize`, then batch submit. For a 4 GiB transfer, this means 64 sequential waves of: D2D copy 64MB + wait + batch submit 16 chunks + wait. That serialization alone explains much of the 582 MiB/s vs 6.73 GiB/s gap.

**Consequences:**
- 2x GPU memory bandwidth usage per transfer
- Serialized 64MB waves instead of saturating NVMe queue depth
- 64 wave iterations for 4 GiB (each wave blocked on D2D sync + batch poll)
- Throughput capped at roughly 10% of raw gdsio baseline

**Prevention -- Direct Registration Strategy:**
Register the user's GPU buffer directly with `cuFileBufRegister` for the duration of the transfer, then deregister after completion. This eliminates the D2D copy entirely:
```cpp
registered_buffer user_reg(dev_ptr, size);  // RAII register
// Submit batch ops pointing directly at dev_ptr
cuFileBatchIOSubmit(batch, nr, params.data(), 0);
// Poll for completion
// ~registered_buffer() calls cuFileBufDeregister
```
The per-transfer registration cost is roughly 1-5ms, but it is amortized over multi-GiB transfers where the D2D copy overhead dwarfs it.

**Warning signs:**
- `nvidia-fs` stats (`cat /proc/driver/nvidia-fs/stats`) shows high bounce buffer usage even though you believe you are using registered buffers
- GPU memory bandwidth utilization is high during I/O but disk throughput is low
- Transfer time scales linearly with 2x the data size (D2D + disk, not just disk)

**Detection:** Profile with `nsys` -- you will see `cudaMemcpyAsync` D2D calls interleaved with cuFile operations. A direct path should show zero D2D copies during I/O.

**Optimization phase:** Phase 1 -- this is the single largest bottleneck (identified in PROJECT.md).

---

### Pitfall 2: T4 BAR1 Aperture Exhaustion (256MB Hard Limit)

**What goes wrong:** The NVIDIA T4 GPU has only **256MB of BAR1 memory**. `cuFileBufRegister` pins GPU memory in BAR1 space. Registering a 4 GiB user buffer will fail with `-12 (ENOMEM)` because it exceeds the BAR1 aperture. This silently forces cuFile to fall back to its internal bounce buffer path (extra copy), or worse, causes `cuFileBufRegister` to return an error that gets ignored.

**Why it happens:** Developers test on A100/H100 (BAR1 = 128GB/256GB) where registering multi-GiB buffers works fine, then deploy to T4 (BAR1 = 256MB) or other Turing cards where it fails. cuCascade's CI runs on T4 GPUs (`linux-amd64-gpu-t4-latest-1`).

**Consequences:**
- `cuFileBufRegister` silently fails; `is_registered()` returns false
- cuFile falls back to internal bounce buffers (extra copy, lower throughput)
- Performance regression appears only on T4, not on development GPUs
- If registration failure is treated as fatal, the entire backend breaks on T4

**Prevention -- Chunked Registration:**
Never register the entire multi-GiB buffer at once. Instead, register in chunks that fit within BAR1 limits (e.g., 128MB or less) and process in waves:
```cpp
constexpr std::size_t MAX_REGISTRATION_SIZE = 128ULL * 1024 * 1024;  // 128 MB
// For each wave:
std::size_t reg_size = std::min(remaining, MAX_REGISTRATION_SIZE);
registered_buffer chunk_reg(dev_ptr + offset, reg_size);
// Submit batch for this chunk, wait, deregister (RAII), advance offset
```
Alternatively, query BAR1 size at runtime via NVML or `nvidia-smi -q | grep "BAR1"` and adapt.

**Warning signs:**
- `cuFileBufRegister` returns non-SUCCESS status
- Tests pass on A100 but fail or run slowly on T4
- `nvidia-fs` stats show `Registered_MB` is 0 even though you called `cuFileBufRegister`

**Detection:** Check the return value of `cuFileBufRegister` AND log when `is_registered()` returns false. Also: `nvidia-smi -q | grep "BAR1"` to check available BAR1 on the target GPU.

**Optimization phase:** Phase 1 -- must be solved simultaneously with Pitfall 1 (direct registration requires BAR1-aware chunking).

---

### Pitfall 3: Alignment Violations Triggering Silent Bounce Buffer Fallback

**What goes wrong:** cuFile requires 4KB alignment on three parameters for optimal DMA: `devPtr_offset`, `file_offset`, and `size`. When any of these is not 4KB-aligned, cuFile silently falls back to its internal GPU bounce buffer path -- an extra D2D copy that halves throughput. The API does not error; it just runs slower.

**Why it happens:** Column data buffers from cuDF (via `rmm::device_buffer`) are allocated with RMM's default 256-byte alignment, not 4KB. Null mask sizes and string offset arrays often have sizes that are not multiples of 4096. The disk file format already aligns file offsets to 4KB (`DISK_FILE_ALIGNMENT = 4096`) but the GPU-side buffer offsets within a batch operation may not be aligned.

**How this manifests in cuCascade:** In `convert_gpu_to_disk`, the `io_batch_entry` list includes column data buffers whose `dev_ptr` values are RMM-allocated pointers. These satisfy cudaMalloc alignment (256 bytes) but not necessarily 4KB alignment. When these are passed as `devPtr_base` in batch params, cuFile internally detects the misalignment and routes through bounce buffers.

**Consequences:**
- Silent performance degradation (no error, no warning)
- Up to 50% throughput loss on writes, more on reads (read-modify-write for unaligned writes)
- Inconsistent benchmark results depending on RMM allocator behavior

**Prevention:**
1. **File layout:** Already done -- `DISK_FILE_ALIGNMENT = 4096` for file offsets (verified in `disk_file_format.hpp` line 36).
2. **GPU buffer registration:** When registering user buffers directly, ensure the base pointer is 4KB-aligned. `cudaMalloc` returns at least 256-byte aligned pointers; verify 4KB alignment: `assert(reinterpret_cast<uintptr_t>(ptr) % 4096 == 0)`.
3. **Batch entry offsets:** When building `CUfileIOParams_t`, ensure `devPtr_offset` is a multiple of 4096. If column buffers within a contiguous allocation are not 4KB-aligned, pad the layout.
4. **Transfer sizes:** Pad individual I/O operations to 4KB multiples. The file format already does this for column data regions.

**Warning signs:**
- `nvidia-fs` stats show non-zero bounce buffer usage (`cat /proc/driver/nvidia-fs/stats`)
- Throughput varies across runs or across column configurations
- Small columns (null masks, offsets arrays) are disproportionately slow

**Detection:** `cuFileDriverGetProperties()` returns `per_buffer_cache_size` -- if individual batch I/O operations exceed this, they go through compat mode. Also, add assertions: `CUCASCADE_ASSERT(offset % 4096 == 0)` for all batch params.

**Optimization phase:** Phase 1 -- alignment correctness is a prerequisite for direct registration to deliver full throughput.

---

### Pitfall 4: Per-Batch-Entry Size Exceeding `per_buffer_cache_size` Triggers Compat Mode

**What goes wrong:** For any single batch I/O entry whose size exceeds `properties.per_buffer_cache_size_kb` (default: 1024 KB = 1 MB), cuFile silently switches that entry to **compatible mode** (POSIX path through page cache). This means individual batch entries larger than 1MB do not use the GDS DMA path even if the buffer is registered, defeating the entire purpose of GDS.

**Why it happens:** The batch API documentation states: "For any batch I/O size bigger than the bounce buffer size (`properties.per_buffer_cache_size_kb`), cuFile uses compatible mode." This is per-entry, not per-batch. The current `gds_io_backend` uses 4MB `SLOT_SIZE` chunks, each of which exceeds the default 1MB `per_buffer_cache_size`.

**How this manifests in cuCascade:** Lines 253-256: `SLOT_SIZE = 4MB`, `NUM_SLOTS = 16`. Each batch entry is 4MB -- 4x the default `per_buffer_cache_size` threshold. This means *every single batch entry* may be falling back to POSIX compat mode, even with proper buffer registration.

**Consequences:**
- GDS DMA path is never used for batch entries, silently falling through to POSIX
- Entire batch API usage is a no-op performance-wise -- you get POSIX throughput at best
- You are paying the overhead of batch setup and polling without the DMA benefit

**Prevention:**
1. **Query the threshold at runtime:**
   ```cpp
   CUfileDrvProps_t props;
   cuFileDriverGetProperties(&props);
   auto max_batch_entry_size = props.per_buffer_cache_size * 1024;  // convert KB to bytes
   ```
2. **Size batch entries to fit:** Ensure each individual batch entry's `size` field is less than or equal to `per_buffer_cache_size_kb * 1024`. With the default of 1MB, use 1MB slot sizes for batch entries.
3. **Or increase the threshold:** Edit `/etc/cufile.json` and set `"per_buffer_cache_size"` to a larger value (e.g., 16384 for 16MB). But this consumes more GPU memory for bounce buffers.
4. **Or use cuFileRead/cuFileWrite (non-batch):** For large registered buffers, the synchronous API with cuFile's internal thread pool (`max_io_threads`, `parallel_io`) can be more effective than the batch API.

**Warning signs:**
- Batch API throughput matches or is worse than POSIX/pipeline backend
- `nvidia-fs` stats show low `GDS Read/Write` counts but high `Compat Read/Write` counts
- Increasing batch operation count does not improve throughput

**Detection:** Use `cuFileDriverGetProperties` to check `per_buffer_cache_size` and compare against your batch entry sizes. Also use cuFile Stats API (`cuFileStatsStart`/`cuFileStatsStop`) to check if operations are actually using GDS vs compat mode.

**Optimization phase:** Phase 1 -- directly impacts whether the batch API approach is even viable. Must be validated before committing to batch vs synchronous API strategy.

---

### Pitfall 5: Serial Wave Processing for Large Transfers

**What goes wrong:** For transfers exceeding the staging buffer size (64MB), the current implementation processes waves sequentially: copy wave then submit wave then wait then next wave. There is no overlap between NVMe I/O and the next wave's data preparation. For 4 GiB, this means 64 sequential wave cycles.

**Why it happens:** The `while (remaining > 0)` loop in both `write_device` and `read_device` is fully synchronous: D2D copy, then sync, then submit batch, then poll to completion, then advance to next wave. No double-buffering or pipelining.

**How this manifests in cuCascade:** Write path (lines 299-336): `cudaMemcpyAsync + cudaStreamSynchronize + submit_and_wait` per wave. The NVMe is idle during the D2D copy, and the GPU memory bus is idle during the NVMe I/O.

**Consequences:**
- NVMe utilization: roughly 50% (idle during D2D copy phases)
- GPU memory bandwidth utilization: roughly 50% (idle during I/O phases)
- Total throughput: roughly half of what overlapped execution would achieve
- The pipeline backend (`pipeline_io_backend`) already solves this with double-buffering

**Prevention -- Double-Buffered Registration Waves:**
With direct registration (Pitfall 1 fix), this pitfall largely disappears because there is no D2D copy to overlap. However, if chunked registration is needed (Pitfall 2), implement double-buffered waves:
```cpp
// Register chunk A, start I/O on A
// While A's I/O is in flight, register chunk B
// When A completes, deregister A, start I/O on B
// Register next chunk into A's slot, repeat
```
The key insight: cuFile operations are asynchronous (batch API) and `cuFileBufRegister`/`cuFileBufDeregister` can be done on different memory regions concurrently. Two registration slots enable full pipeline overlap.

**Warning signs:**
- Throughput scales with buffer size but plateaus at staging buffer size
- NVMe device utilization (via `iostat -x 1`) shows bursty rather than sustained I/O
- Large transfers are proportionally slower per-byte than small ones

**Detection:** `nsys` timeline shows alternating D2D and cuFile I/O blocks with idle gaps between them. No overlap between compute/copy and I/O.

**Optimization phase:** Phase 2 -- after direct registration is working, optimize the wave pipeline for BAR1-constrained GPUs.

---

## Moderate Pitfalls

Issues that cause 20-50% performance loss or implementation complexity.

---

### Pitfall 6: Mixing O_DIRECT and Buffered POSIX Writes on the Same File

**What goes wrong:** Opening the same file with both `O_DIRECT` (for GDS device writes) and without `O_DIRECT` (for POSIX host writes of header/metadata) causes the kernel to invalidate the page cache for that file. This invalidation can cost 50-100ms and must happen on every file open that crosses the boundary.

**Why it happens:** cuFile opens files with `O_DIRECT` for GDS DMA. If the header/metadata was previously written via a buffered POSIX `write()`, the page cache has stale data. The kernel must invalidate it before `O_DIRECT` can proceed, or vice versa. The Linux documentation explicitly warns: "Applications should avoid mixing O_DIRECT and normal I/O to the same file."

**How this manifests in cuCascade:** The codebase already fixed this by copying header+metadata to a GPU buffer and writing everything through `write_device_batch` in a single O_DIRECT file open (lines 1241-1243 in `representation_converter.cpp`: "avoids the 80ms page cache invalidation from mixing POSIX and O_DIRECT writes"). However, the `write_host()` and `read_host()` methods in `gds_io_backend` (lines 396-429) use buffered POSIX I/O without `O_DIRECT`. If these are ever called on the same file as `write_device`, the penalty returns.

**Consequences:**
- 50-100ms latency spike per file open after mode switch
- For many small files (one per data batch), this adds up significantly
- Difficult to diagnose because the penalty is in the VFS layer, not in cuFile

**Prevention:**
- Already partially addressed in cuCascade's converter code
- Ensure `read_host` (used for header reads in `convert_disk_to_gpu`) never shares a file descriptor or is called between `O_DIRECT` operations on the same file
- Consider reading the header via `read_device` + D2H copy instead of `read_host` to keep everything on the O_DIRECT path
- If using ext4, ensure the mount option `dioread_nolock` is set (default on kernel 6.x+) to avoid inode mutex contention on O_DIRECT writes

**Warning signs:**
- First I/O operation on a file is significantly slower than subsequent ones
- Latency spikes that correlate with file open/close rather than data size
- `dmesg` or `perf trace` shows page cache invalidation activity

**Detection:** Trace file descriptor flags with `strace -e openat` and look for the same file path opened with and without `O_DIRECT`.

**Optimization phase:** Phase 1 -- when restructuring the I/O path, ensure the header read path stays consistent.

---

### Pitfall 7: File Handle Open/Close Overhead Per Transfer

**What goes wrong:** `cuFileHandleRegister` performs filesystem compatibility checks and initializes GDS state for each file. Opening a new file handle for every `write_device` or `read_device` call adds roughly 0.5-2ms per operation. For the batch fallback path where `write_device_batch` calls `write_device` 64 times, this means 64 file opens and cuFile handle registrations.

**Why it happens:** The current `cufile_handle` class opens a POSIX fd + registers with cuFile in its constructor, and deregisters + closes in its destructor. Each call to `write_device`/`read_device` creates and destroys a `cufile_handle`. For the `write_device_batch` fallback path (line 459), this means N sequential file opens for N entries.

**Consequences:**
- Per-column I/O fallback path: N file opens for N columns (could be 20+ for wide tables)
- Each cuFileHandleRegister "evaluates the suitability of the file state and the file mount for GDS and initializes the file state of the critical performance path"
- This initialization cost is wasted when the same file is reopened immediately

**Prevention:**
- Cache file handles: open once per file path, reuse across operations, close at the end
- For batch operations, the current `write_device_batch` already opens once for the in-staging path -- ensure the direct registration path maintains this pattern
- Consider a file handle pool keyed by path if concurrent access is needed

**Warning signs:**
- High syscall count in `strace` output (`openat`, `close` dominating)
- Throughput improves dramatically for single large writes vs many small writes to the same file
- `cuFileHandleRegister` appearing multiple times per batch in profiling

**Detection:** `strace -c` to count syscall frequency. Look for `openat`/`close` counts matching column count.

**Optimization phase:** Phase 1 -- ensure the new direct-I/O path opens each file exactly once per conversion.

---

### Pitfall 8: kvikIO Default Thread Pool Size = 1

**What goes wrong:** kvikIO's default `KVIKIO_NTHREADS` is 1, meaning all parallel I/O is serialized through a single thread. For large transfers, kvikIO splits data into `KVIKIO_TASK_SIZE` (default 4MB) chunks and processes them through this thread pool. With 1 thread, there is zero I/O parallelism.

**Why it happens:** kvikIO defaults to the most conservative setting. Developers who expect kvikIO to automatically parallelize I/O are surprised when single-threaded behavior is the default.

**How this manifests in cuCascade:** The `kvikio_io_backend` (lines 33-91) calls `fh.write(dev_ptr, size, file_offset, 0)` which uses kvikIO's synchronous API. With `KVIKIO_NTHREADS=1`, even the internal chunking is serial. The measured 1.20 GiB/s write throughput is consistent with single-threaded NVMe I/O.

**Consequences:**
- Single-threaded I/O cannot saturate NVMe queue depth (most NVMe SSDs need 4-16 concurrent I/O operations)
- gdsio achieves 6.73 GiB/s write with 4 threads; kvikIO at 1 thread leaves 5x throughput on the table
- No benefit from NVMe's internal parallelism

**Prevention:**
- Set `KVIKIO_NTHREADS` to 4-16 via environment variable or runtime API:
  ```cpp
  kvikio::defaults::set_num_threads(8);
  ```
- Set `KVIKIO_TASK_SIZE` to match NVMe optimal I/O size (typically 1-4MB)
- For the synchronous `.write()`/`.read()` API, kvikIO handles the internal splitting; ensure the thread pool is sized to match NVMe's optimal queue depth
- Set thread pool configuration ONCE at backend construction time, before any I/O operations; `defaults::set_num_threads()` destroys all existing threads and creates a new pool, which is unsafe during active I/O

**Warning signs:**
- kvikIO throughput matches or is lower than single-threaded `dd`
- `KVIKIO_NTHREADS` is not set in the process environment
- `iostat -x` shows low `avgqu-sz` (average queue size) during kvikIO operations

**Detection:** `kvikio::defaults::get_num_threads()` to check the runtime value. Compare against `nproc` and NVMe capabilities.

**Optimization phase:** Phase 2 -- kvikIO optimization pass.

---

### Pitfall 9: cuFile Batch API Does Not Truly Batch at the Kernel Level

**What goes wrong:** Despite its name, `cuFileBatchIOSubmit` does not coalesce I/O requests into a single kernel submission. A 2025 NIXL benchmarking study found that batched requests are submitted individually internally. The batch API's benefit is reducing ioctl overhead (one system call instead of N), but each I/O entry is still processed as a separate NVMe command. For large I/O sizes (>1MB), there is no measurable advantage over submitting individual `cuFileWrite`/`cuFileRead` calls.

**Why it happens:** The cuFile batch API was designed to amortize CPU overhead for many small I/O operations, not to merge I/O at the storage level. NVMe SSDs already have their own internal command queue and do not benefit from application-level batching of large sequential writes.

**How this manifests in cuCascade:** The current implementation builds elaborate batch param arrays (lines 313-329) for 4MB chunks, but each 4MB write is submitted individually to the NVMe. The batch API overhead (allocating `CUfileIOParams_t` arrays, polling with `cuFileBatchIOGetStatus`) adds latency without improving throughput for these large chunks.

**Consequences:**
- Batch API setup/teardown overhead without corresponding throughput benefit for large entries
- False sense of parallelism -- developers assume batch = parallel, but it is serial at the storage layer
- For entries > 1MB, synchronous `cuFileRead`/`cuFileWrite` with cuFile's internal thread pool may outperform the batch API

**Prevention:**
- For large transfers with registered buffers: prefer `cuFileRead`/`cuFileWrite` synchronous API with cuFile's internal `parallel_io` and `max_io_threads` (configurable in `cufile.json`)
- Reserve the batch API for scenarios with many small I/O entries (< 1MB each)
- Test both approaches with your actual transfer sizes and compare throughput

**Warning signs:**
- Batch API throughput does not scale with batch size
- Individual `cuFileWrite` calls match or exceed batch throughput for same data
- NVMe queue depth (via `iostat`) stays at 1 regardless of batch size

**Detection:** Compare `cuFileBatchIOSubmit` with N entries vs N sequential `cuFileWrite` calls on the same data. If throughput is identical, the batch API is not providing value.

**Optimization phase:** Phase 1 -- API selection decision (batch vs synchronous) must be made before implementing the direct-registration path.

---

### Pitfall 10: ext4 O_DIRECT Write Lock Contention Without `dioread_nolock`

**What goes wrong:** On ext4 filesystems (which cuCascade uses on `/mnt/disk_2`), O_DIRECT writes require an exclusive inode lock by default. This serializes all concurrent writes to the same file and can reduce throughput by 30-80% compared to XFS or ext4-with-`dioread_nolock`.

**Why it happens:** ext4's default journaling mode takes an exclusive inode mutex for O_DIRECT writes to manage extent allocation. The `dioread_nolock` mount option (default since kernel 6.x) allocates uninitialized extents first and converts after I/O, avoiding the mutex.

**Consequences:**
- Concurrent writes from multiple threads to the same file are serialized
- Single-file throughput matches; multi-file or multi-thread throughput is degraded
- Affects both GDS and kvikIO (both use O_DIRECT)

**Prevention:**
- Verify the filesystem mount options: `mount | grep disk_2` should show `dioread_nolock`
- If using kernel 6.x+, this is likely already the default
- If not set, remount with: `mount -o remount,dioread_nolock /mnt/disk_2`
- Consider pre-allocating file space with `fallocate()` before writing to avoid extent allocation during I/O

**Warning signs:**
- Write throughput is significantly below read throughput (more than raw hardware ratio suggests)
- `perf top` shows ext4 journal/inode lock contention
- Multi-threaded write benchmarks show no scaling

**Detection:** `cat /proc/mounts | grep disk_2` and check for `dioread_nolock` in the mount options. Also: `uname -r` to check kernel version (6.x should have it by default).

**Optimization phase:** Phase 0 (environment setup) -- check before any benchmarking.

---

## Minor Pitfalls

Issues that cause <20% performance loss or debugging time.

---

### Pitfall 11: cuFileDriverOpen Latency on First I/O

**What goes wrong:** If `cuFileDriverOpen` is called lazily (on first I/O operation), the first transfer incurs a 50-200ms initialization penalty. This distorts benchmarks and causes latency spikes in production.

**Prevention:** The current implementation already handles this correctly via the `cufile_driver_guard` singleton (lines 43-75 in `gds_io_backend.cpp`), called in the `gds_io_backend` constructor. Ensure any refactoring preserves this eager initialization.

**Detection:** First-transfer latency is significantly higher than subsequent transfers. Benchmark warm-up iterations should be used.

**Optimization phase:** N/A -- already addressed. Preserve during refactoring.

---

### Pitfall 12: cuFileBatchIOGetStatus `nr` Parameter Misuse

**What goes wrong:** The `nr` parameter to `cuFileBatchIOGetStatus` is INPUT/OUTPUT. On input, it specifies the maximum number of events to return. On output, it contains the actual count of events returned. Passing `nr=0` on input means "return 0 events" -- the poll returns immediately without checking any completions, causing infinite loops.

**Why it happens:** Developers assume `nr` is output-only and pass an uninitialized or zero value.

**How this manifests in cuCascade:** The code already handles this correctly (line 210: `unsigned nr_to_poll = nr;` resets to max before each poll call), with a comment documenting the root cause (lines 187-194). However, if the poll loop is refactored, this subtle behavior is easy to re-introduce.

**Prevention:** Always reset `nr_to_poll` to the maximum expected count before each `cuFileBatchIOGetStatus` call. Add a comment explaining the INPUT/OUTPUT dual purpose.

**Detection:** Infinite loop or timeout in `submit_and_wait`. The `total_completed` counter never advances.

**Optimization phase:** N/A -- already addressed. Preserve the comment and pattern during refactoring.

---

### Pitfall 13: `cuFileBufRegister`/`cuFileBufDeregister` Churn in Hot Path

**What goes wrong:** Repeatedly registering and deregistering GPU buffers in a loop wastes BAR1 pinning/unpinning cycles. Each register/deregister pair costs roughly 1-5ms. If done per-column or per-chunk in a tight loop, this overhead accumulates.

**Prevention:**
- Register buffers as early as possible and keep them registered for the duration of multiple I/O operations
- For the direct-registration approach, register the entire chunk once, perform all batch I/O, then deregister once
- If using wave-based processing, minimize the number of register/deregister cycles by using larger waves (fewer but bigger)

**Warning signs:**
- `nvidia-fs` stats show high registration count (n) and deregistration count (free)
- Per-column write throughput is worse than single-column write throughput

**Detection:** `cat /proc/driver/nvidia-fs/stats` -- check the BAR1-map stats row for high `n` (registration) and `free` (deregistration) counts.

**Optimization phase:** Phase 1 -- bake into the direct-registration design from the start.

---

### Pitfall 14: cuFile's Internal Thread Pool Conflicts with Application Threading

**What goes wrong:** cuFile's synchronous APIs (`cuFileRead`/`cuFileWrite`) internally use a thread pool controlled by `max_io_threads` (default: 4) and `parallel_io` (default: true) in `cufile.json`. Developers unaware of this build their own parallelism on top, causing contention between cuFile's internal threads and application threads, or conversely fail to use it and get worse throughput than expected.

**Prevention:**
- Know that `cuFileRead`/`cuFileWrite` for large registered buffers are already internally parallelized
- Tune `max_io_threads` and `max_request_parallelism` in `cufile.json` rather than building application-level parallelism
- For the batch API, this internal parallelism does not apply -- each batch entry is processed individually

**Detection:** Check `cufile.json` for `parallel_io` and `max_io_threads` settings. Compare single-call throughput vs multi-threaded throughput.

**Optimization phase:** Phase 1 -- when choosing between batch API and synchronous API.

---

### Pitfall 15: cuFileBatchIOSetUp Max Entries Limit

**What goes wrong:** `cuFileBatchIOSetUp` has a maximum number of operations per batch, determined by `properties.io_batch_size` (default: 128). If you submit more entries than this limit, the setup call fails. For a 4 GiB transfer with 1MB chunks, that is 4096 entries -- far exceeding the 128 default.

**Prevention:**
- Query `cuFileDriverGetProperties()` at initialization to get `io_batch_size`
- Split large transfers into multiple batches of at most `io_batch_size` entries each
- Use larger chunk sizes (16MB instead of 1MB) to reduce entry count: 4 GiB / 16MB = 256 entries, needing only 2 batches

**Detection:** `cuFileBatchIOSetUp` returns an error code. The current code would throw via CUCASCADE_FAIL.

**Optimization phase:** Phase 1 -- if continuing to use batch API, respect this limit.

---

### Pitfall 16: cuFile Registration on RMM Suballocated Memory

**What goes wrong:** RMM's pool allocator returns suballocations from large `cudaMalloc` blocks. If you register a suballocation (e.g., a 1MB slice of a 256MB pool block), cuFile may register the entire underlying block or may fail. The behavior depends on the cuFile version and the alignment of the suballocation.

**Prevention:**
- Test registration with actual RMM-allocated buffers, not just raw `cudaMalloc`
- If registration fails on suballocated buffers, fall back to the staging path
- Consider using `rmm::mr::cuda_memory_resource` (non-pooling) for disk I/O buffers if pool-based registration proves unreliable
- The PROJECT.md constraint says "Cannot assume RMM-allocated GPU memory is registered with cuFile" -- handle registration failures gracefully

**Detection:** `registered_buffer::is_registered()` returns false. Test with pool allocator active.

**Optimization phase:** Phase 1 -- test early with RMM pool allocators in the loop.

---

### Pitfall 17: GDS Compatibility Mode Silently Enabled

**What goes wrong:** If the system does not meet GDS requirements (kernel module not loaded, filesystem not supported, etc.), cuFile silently falls back to "compatibility mode" which routes through CPU bounce buffers. The code works but throughput is limited to POSIX levels.

**Prevention:**
- Run `gds_check` tool during setup to verify GDS is properly configured
- Check `cufile.json` for `force_compat_mode: false` and `allow_compat_mode: true`
- Enable cuFile stats to verify GDS (not compat) mode is used during benchmarks
- kvikIO's `compat_mode()` API can check current mode

**Detection:** Throughput at POSIX levels (roughly 2-3 GB/s) instead of GDS levels (roughly 6-13 GB/s). cuFile stats show compat mode operations.

**Optimization phase:** Phase 0 -- verify GDS is functional before optimizing.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 0: Environment setup | ext4 `dioread_nolock` not set (Pitfall 10) | Check mount options before benchmarking. Verify kernel version. |
| Phase 0: Environment setup | GDS compat mode silently enabled (Pitfall 17) | Run `gds_check`; verify with cuFile stats. |
| Phase 0: Environment setup | Running benchmarks on wrong disk (/tmp vs /mnt/disk_2) | Verify mount point and device before benchmarking. |
| Phase 1: Direct registration for GDS | BAR1 exhaustion on T4 (Pitfall 2) | Chunked registration with 128MB max; query BAR1 at runtime. |
| Phase 1: Direct registration for GDS | Alignment violations causing silent bounce buffer (Pitfall 3) | Assert 4KB alignment on all three parameters; pad if needed. |
| Phase 1: Direct registration for GDS | Batch entry > per_buffer_cache_size (Pitfall 4) | Query threshold; size entries accordingly or switch to sync API. |
| Phase 1: API selection | Batch API does not truly batch (Pitfall 9) | Benchmark batch vs sync API; prefer sync for large registered buffers. |
| Phase 1: API selection | cuFile internal thread pool (Pitfall 14) | Tune cufile.json rather than adding application-level threads. |
| Phase 1: File I/O restructure | Handle open/close overhead (Pitfall 7) | Single file open per conversion, reuse handle. |
| Phase 1: File I/O restructure | Mixing O_DIRECT and POSIX (Pitfall 6) | Keep everything on O_DIRECT path; read header via device path. |
| Phase 1: Registration design | Register/deregister churn (Pitfall 13) | Register once per wave, not per column. |
| Phase 1: Registration design | RMM suballocated memory (Pitfall 16) | Test with pool allocators early; graceful fallback. |
| Phase 1: Batch sizing | cuFileBatchIOSetUp max entries (Pitfall 15) | Query `io_batch_size` property; split large transfers. |
| Phase 2: Wave pipelining (if needed) | Serial wave processing (Pitfall 5) | Double-buffered registration with two BAR1 slots. |
| Phase 2: kvikIO optimization | Thread pool size = 1 (Pitfall 8) | Set `KVIKIO_NTHREADS=8` and tune `KVIKIO_TASK_SIZE`. |
| All phases: Refactoring | `nr` parameter misuse (Pitfall 12) | Preserve the INPUT/OUTPUT handling pattern with comments. |

## Diagnostic Checklist

Before and after each optimization phase, run these checks:

```bash
# 1. Check nvidia-fs driver stats (bounce buffer usage, registration counts)
cat /proc/driver/nvidia-fs/stats

# 2. Check filesystem mount options (dioread_nolock for ext4)
mount | grep disk_2

# 3. Check BAR1 memory on target GPU
nvidia-smi -q | grep -A3 "BAR1"

# 4. Check cufile.json configuration
cat /etc/cufile.json | grep -E "per_buffer_cache_size|max_direct_io_size|max_io_threads|parallel_io|max_batch_io"

# 5. Check kvikIO settings
echo "KVIKIO_NTHREADS=${KVIKIO_NTHREADS:-not set (default 1)}"
echo "KVIKIO_TASK_SIZE=${KVIKIO_TASK_SIZE:-not set (default 4194304)}"
echo "KVIKIO_GDS_THRESHOLD=${KVIKIO_GDS_THRESHOLD:-not set (default 16384)}"

# 6. Monitor NVMe queue depth during I/O
iostat -x 1 nvme1n1

# 7. Profile with nsys to check for D2D copies during I/O
nsys profile --trace=cuda,nvtx ./cucascade_benchmarks --benchmark_filter=DiskConverter
```

## Sources

- [NVIDIA GPUDirect Storage Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) -- alignment, registration, bounce buffer behavior, I/O patterns (HIGH confidence)
- [NVIDIA GPUDirect Storage API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) -- cuFileBatchIOSubmit, cuFileBatchIOGetStatus, cuFileBufRegister API details (HIGH confidence)
- [NVIDIA GPUDirect Storage Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) -- cufile.json tunables, default values (HIGH confidence)
- [NVIDIA GPUDirect Storage O_DIRECT Requirements Guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html) -- O_DIRECT alignment, page cache behavior (HIGH confidence)
- [NVIDIA GPUDirect Storage Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) -- BAR1 aperture, internal chunking for large transfers (HIGH confidence)
- [NVIDIA GPUDirect Storage Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) -- nvidia-fs stats, BAR1 diagnostics (HIGH confidence)
- [kvikIO Runtime Settings Documentation](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/) -- KVIKIO_NTHREADS, KVIKIO_TASK_SIZE, KVIKIO_GDS_THRESHOLD defaults (HIGH confidence)
- [libkvikio C++ API Reference](https://docs.rapids.ai/api/libkvikio/stable/) -- FileHandle, defaults class (HIGH confidence)
- [NIXL Benchmarking Study (2025)](http://cs.iit.edu/~scs/assets/files/muradli2025gpudirect.pdf) -- batch API internal behavior, GPU DIRECT vs GPU BATCH comparison (MEDIUM confidence -- single peer-reviewed study)
- [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) -- T4 BAR1 = 256MB specification (HIGH confidence)
- [NVIDIA Developer Forums -- BAR size on T4](https://forums.developer.nvidia.com/t/ibv-reg-mr-and-bar-size-on-t4/201907) -- T4 BAR1 practical limits (MEDIUM confidence)
- [Percona -- ext4 O_DIRECT Performance](https://www.percona.com/blog/watch-out-for-disk-i-o-performance-issues-when-running-ext4/) -- ext4 dioread_nolock impact, 30-80% throughput drop without it (MEDIUM confidence)
- [Phoronix -- EXT4 Direct I/O Improvement Linux 6.3](https://www.phoronix.com/news/Linux-6.3-EXT4) -- dioread_nolock becoming default (MEDIUM confidence)
- Current codebase analysis: `src/data/gds_io_backend.cpp`, `src/data/kvikio_io_backend.cpp`, `src/data/representation_converter.cpp`, `include/cucascade/data/disk_file_format.hpp`
