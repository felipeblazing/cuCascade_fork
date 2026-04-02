# Feature Landscape: GPU-Disk I/O Throughput Optimization

**Domain:** High-performance GPU Direct Storage I/O optimization for NVMe SSDs
**Researched:** 2026-04-02
**Overall confidence:** HIGH (verified against NVIDIA GDS official docs, kvikIO docs, and codebase analysis)

## Performance Gap Analysis

| Metric | Current | Target (80%) | Hardware Max |
|--------|---------|--------------|--------------|
| GDS Write | 582 MiB/s | 5.38 GiB/s | 6.73 GiB/s |
| GDS Read | Not measured | 10.68 GiB/s | 13.35 GiB/s |
| kvikIO Write | 1.20 GiB/s | 5.38 GiB/s | 6.73 GiB/s |
| kvikIO Read | Not measured | 10.68 GiB/s | 13.35 GiB/s |

Gap: GDS is at 8.4% of max write throughput. kvikIO is at 17.3%. Both are 5-11x below target.

---

## Table Stakes

Features that are required to not be embarrassingly slow. Missing any of these means the backend cannot approach even 50% of hardware throughput.

### TS-1: Eliminate D2D Staging Copy in GDS Backend

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Current code copies GPU data to a 64MB staging buffer via cudaMemcpy D2D, then writes staging buffer to disk. This doubles GPU memory bandwidth usage and serializes operations. It is the single largest bottleneck. |
| **Expected Throughput Impact** | 2-4x improvement (from ~580 MiB/s to 1.5-2.5 GiB/s) |
| **Complexity** | Medium |
| **Confidence** | HIGH |

**Current code problem (gds_io_backend.cpp:299-310):**
```
cudaMemcpyAsync(staging_ptr, dev_ptr + offset, wave_bytes, D2D, stream);  // WASTE
cudaStreamSynchronize(stream);                                             // BLOCK
// Then batch submit from staging_ptr
```

**Fix:** Register the application's GPU buffer directly with `cuFileBufRegister()`, then issue cuFile batch operations pointing at the original buffer. The staging buffer and D2D copy become unnecessary.

Per NVIDIA Best Practices Guide: "If a user buffer is not registered, an intermediate pre-registered GPU buffer that is owned by the cuFile implementation is used, and there is an extra copy." By registering the source buffer, GDS performs DMA directly from application memory to NVMe, eliminating the D2D copy entirely.

**Registration cost:** cuFileBufRegister pins pages in BAR space (a few milliseconds). For a 4 GiB transfer, this is amortized over the multi-second I/O and is negligible. Register once per transfer, not per chunk.

**Depends on:** Nothing (standalone fix)

### TS-2: Eliminate 64MB Wave Serialization in GDS Backend

| Attribute | Value |
|-----------|-------|
| **Why Expected** | For 4 GiB transfers, the current code processes 64 sequential waves of 64MB each. Each wave: D2D copy, sync, batch submit, poll. This serializes what should be parallel NVMe operations. |
| **Expected Throughput Impact** | 3-5x improvement (the wave loop is the primary serialization bottleneck after D2D is fixed) |
| **Complexity** | Medium |
| **Confidence** | HIGH |

**Current code problem:** The `while (remaining > 0)` loop at line 299 processes STAGING_BUFFER_SIZE (64MB) at a time. With D2D staging removed (TS-1), a single `cuFileBufRegister` of the full buffer allows one batch submit of all chunks across the entire 4 GiB range, or at minimum much larger waves.

**Fix:** After TS-1, register the entire output buffer and submit all batch operations in one `cuFileBatchIOSubmit` call (or a small number of very large calls). The NVMe controller can then parallelize across its internal queues.

**Depends on:** TS-1 (eliminating staging enables submitting against the actual buffer)

### TS-3: Increase Chunk Size from 4MB to 1MB-16MB Range

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Current SLOT_SIZE is 4MB with 16 slots. gdsio achieves peak throughput with 1MB I/O sizes and 32+ threads. The 4MB slot size is not inherently bad, but the 16-slot cap limits total in-flight I/O to 64MB. |
| **Expected Throughput Impact** | 20-50% improvement beyond TS-1+TS-2 |
| **Complexity** | Low |
| **Confidence** | MEDIUM |

**Fix:** With TS-1 and TS-2 removing the staging buffer constraint, chunk size becomes a tuning parameter for the batch params. Use 1MB chunks (matching gdsio optimal) to maximize NVMe queue depth, or tune `max_direct_io_size_kb` (default 16MB). The key insight is that GDS internally chunks to `max_direct_io_size_kb` anyway, so submitting many 1MB operations lets the NVMe controller parallelize optimally.

**Depends on:** TS-1, TS-2

### TS-4: Configure kvikIO Thread Pool and Task Size

| Attribute | Value |
|-----------|-------|
| **Why Expected** | kvikIO defaults to KVIKIO_NTHREADS=1 (single thread) and KVIKIO_TASK_SIZE=4MiB. With one thread, a 4 GiB write is 1024 sequential 4MB cuFile calls. gdsio uses 4+ threads for peak throughput. |
| **Expected Throughput Impact** | 2-4x improvement for kvikIO (from 1.2 GiB/s toward 3-5 GiB/s) |
| **Complexity** | Low |
| **Confidence** | HIGH |

**Current code problem (kvikio_io_backend.cpp:41-42):**
```cpp
kvikio::FileHandle fh(path, "w");
auto bytes_written = fh.write(dev_ptr, size, file_offset, 0);  // synchronous, single-threaded
```

The synchronous `write()` method does NOT use the thread pool. Only `pwrite()` (the parallel async version) splits into tasks across the thread pool.

**Fix:** Two changes required:
1. Switch from `fh.write()` to `fh.pwrite()` to enable parallel task splitting
2. Set `KVIKIO_NTHREADS=8` (or configure programmatically via `kvikio::defaults::thread_pool_nthreads()`) and `KVIKIO_TASK_SIZE=1048576` (1MB)

Per kvikIO docs: "Reads specified bytes from the file into the device or host memory in parallel by partitioning the operation into tasks of task_size for execution in the default thread pool."

**Depends on:** Nothing (standalone fix, independent of GDS fixes)

### TS-5: Batch Read Path (disk-to-GPU) Must Use Single Batch Submit

| Attribute | Value |
|-----------|-------|
| **Why Expected** | The `convert_disk_to_gpu` function at line 1373 calls `read_device` per column buffer individually. For a table with many columns (strings have 3 buffers each: offsets, chars, null mask), this means many separate file opens and reads instead of one batch. |
| **Expected Throughput Impact** | 30-50% improvement for read path on multi-column tables |
| **Complexity** | Medium |
| **Confidence** | MEDIUM |

**Current code problem:** `reconstruct_column_from_disk` calls `alloc_and_read_from_disk` per buffer, which calls `backend.read_device()` per buffer. Each `read_device` opens the file, creates a cuFile handle, and does the I/O independently.

**Fix:** Collect all (dest_ptr, size, file_offset) tuples across all columns first (similar to how `collect_gpu_column_io_entries` works for writes), then call `read_device_batch` once. This enables:
- Single file open
- Single cuFile handle registration
- Single batch submit with all operations
- NVMe controller parallelizes all reads

**Depends on:** Nothing (standalone, but benefits compound with TS-1/TS-2)

### TS-6: Ensure Benchmarks Run on NVMe (/mnt/disk_2), Not /tmp

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Current benchmarks run on /tmp which may be tmpfs (RAM-backed) or a different disk. Baselines were measured on /dev/nvme1n1 at /mnt/disk_2. Comparing apples to oranges invalidates all measurements. |
| **Expected Throughput Impact** | Not a throughput fix, but required for valid measurement |
| **Complexity** | Low |
| **Confidence** | HIGH |

**Fix:** Configure benchmark disk_memory_space to use /mnt/disk_2. Verify with `mount` that it is the NVMe device.

**Depends on:** Nothing

---

## Differentiators

Advanced techniques that squeeze out remaining performance after table stakes are implemented. These bridge the gap from ~50-60% to 80%+ of hardware throughput.

### D-1: Direct Buffer Registration of RMM Device Buffers

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | After TS-1 eliminates the staging buffer, the application registers the caller's GPU buffer per-transfer. Going further: if the converter pre-allocates a contiguous device buffer (like cudf::pack does), register THAT once and reuse across many transfers. |
| **Expected Throughput Impact** | 5-15% beyond TS-1 (amortizes registration cost further) |
| **Complexity** | Medium |
| **Confidence** | MEDIUM |

**Approach:** For write path, `cudf::pack()` produces a contiguous GPU buffer. Register it once, submit all batch ops referencing it. For read path, allocate a single large contiguous device buffer, register it, read everything into it, then scatter to column buffers (or restructure to avoid scatter).

**Tradeoff:** cudf::pack allocates a temporary GPU copy. For a 4 GiB table this means 8 GiB peak GPU memory (original + packed). If memory is tight, the per-column direct approach (TS-1) is better.

**Depends on:** TS-1

### D-2: Overlap I/O Submission with Metadata Serialization

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | The write path serializes: plan layout, serialize metadata, copy header to GPU, THEN submit I/O. For large tables the I/O dominates, but the 32-byte header + metadata can be written via host path while GPU data writes proceed in parallel. |
| **Expected Throughput Impact** | 1-5% (small but free) |
| **Complexity** | Low |
| **Confidence** | HIGH |

**Approach:** Write metadata via `write_host` (POSIX) to offset 0, simultaneously submit GPU data via `write_device_batch` starting at data_offset. Current code already puts header+metadata in GPU memory to avoid mixing POSIX and O_DIRECT, but this adds an unnecessary H2D copy and D2D overhead for small metadata.

**Alternative:** Use two file opens -- one without O_DIRECT for metadata at offset 0, one with O_DIRECT for data. Or, write everything via O_DIRECT by padding metadata to 4KB alignment (which the format already does via DISK_FILE_ALIGNMENT).

**Depends on:** Nothing (orthogonal)

### D-3: Double-Buffered GDS Pipeline (Overlap NVMe I/O with Buffer Registration)

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | For very large transfers (>1 GiB), register buffer region A and start I/O while registering region B. When A completes, deregister A and start I/O on B while registering C. This hides registration latency. |
| **Expected Throughput Impact** | 5-10% for transfers > 1 GiB |
| **Complexity** | High |
| **Confidence** | LOW |

**Why LOW confidence:** cuFileBufRegister latency is "a few milliseconds" per NVIDIA docs. For a 4 GiB transfer at 6 GiB/s, total time is ~670ms. A few ms of registration is <1% overhead. This only matters if registration takes longer than expected or for smaller transfers.

**Depends on:** TS-1, TS-2

### D-4: Tune cufile.json Parameters

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | Default GDS settings may not be optimal for this NVMe. Key parameters: max_direct_io_size_kb (default 16MB), max_device_cache_size_kb (default 128MB), max_io_threads (default 4). |
| **Expected Throughput Impact** | 5-20% (hardware and config dependent) |
| **Complexity** | Low |
| **Confidence** | MEDIUM |

**Recommended tuning:**
- `max_direct_io_size_kb`: Try 16384 (default) and 32768. Larger values reduce driver call count.
- `max_device_cache_size_kb`: Only relevant when using unregistered buffers (after TS-1, less important)
- `max_io_threads`: Try 4, 8, 16. More threads enable higher NVMe queue depth.
- `max_request_parallelism`: Try 4, 8. Controls parallel buffer divisions per request.

**Depends on:** Nothing (standalone tuning)

### D-5: Use cuFileReadAsync/cuFileWriteAsync Stream API Instead of Batch API

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | The stream-ordered async API integrates with CUDA streams, enabling fire-and-forget I/O that overlaps with GPU compute. The batch API requires explicit polling (submit_and_wait loop). |
| **Expected Throughput Impact** | 0-10% (reduces CPU-side overhead, may improve latency) |
| **Complexity** | Medium |
| **Confidence** | LOW |

**Caveats:**
- NVIDIA docs note batch APIs are "experimental and might take a different form later"
- Stream APIs require CUDA 12.1+ (available in this project)
- Stream API has more complex parameter management (pointer-to-parameter design)
- For pure throughput (not latency), the batch API may be equivalent

**Depends on:** Nothing, but is an alternative to batch API, not additive

### D-6: Pre-Registered Persistent Buffer Pool

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | Instead of register/deregister per transfer, maintain a pool of pre-registered GPU buffers (like the current staging buffer, but larger and used for actual I/O rather than intermediate copies). |
| **Expected Throughput Impact** | 2-5% (eliminates per-transfer registration overhead) |
| **Complexity** | Medium |
| **Confidence** | MEDIUM |

**Approach:** At GDS backend construction, allocate and register e.g. 256MB of GPU memory. For transfers <= 256MB, use directly. For larger transfers, copy in waves (but now the waves are much larger than 64MB). This is the "hybrid" approach between current staging and full direct registration.

**Depends on:** Architecturally independent, but less valuable if TS-1 works well

### D-7: Contiguous Write via cudf::pack for GDS Path

| Attribute | Value |
|-----------|-------|
| **Value Proposition** | Instead of writing many small scatter-gather column buffers, use cudf::pack to produce a single contiguous GPU buffer, register it once, and write with one large sequential I/O. Sequential I/O is always faster than scatter-gather on NVMe. |
| **Expected Throughput Impact** | 10-30% over scatter-gather approach (especially for many-column tables) |
| **Complexity** | Medium |
| **Confidence** | MEDIUM |

**Tradeoff:** cudf::pack allocates an additional GPU-side copy (doubles peak GPU memory for the transfer duration). The pack operation itself takes GPU time. For tables that are already nearly contiguous (single wide column), the benefit is minimal.

**Depends on:** TS-1 (for buffer registration of the packed buffer)

---

## Anti-Features

Features that seem helpful but hurt performance, reliability, or maintainability. Do NOT implement these.

### AF-1: Per-Column Buffer Registration

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Calling cuFileBufRegister on each individual column buffer (null mask, data, offsets) | Registration takes milliseconds per call. A table with 100 columns and strings has 300+ buffers. 300 x 2ms = 600ms of registration overhead, negating any throughput gain. | Register one contiguous buffer (either the original allocation if contiguous, or a cudf::packed buffer), OR register the caller's full GPU buffer region once. |

### AF-2: Very Small Chunk Sizes (<256KB)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Using tiny chunk sizes (e.g., 64KB or 128KB) in batch submit to "maximize parallelism" | Excessive batch entries increase CPU-side submission overhead and per-operation status tracking. GDS internally chunks to max_direct_io_size_kb anyway. Many small ops do not help NVMe queue depth because the driver coalesces them. | Use 1MB-4MB chunks (matching gdsio optimal) or even larger (16MB) if buffer is pre-registered. Let the GDS driver handle internal chunking. |

### AF-3: Over-Registration (Registering All RMM Memory)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Registering all GPU memory with cuFileBufRegister at startup or registering the entire RMM pool | BAR space is limited. Over-registration can exhaust BAR memory, causing failures for other GPU operations. cuFileBufRegister "incurs a significant performance cost" per NVIDIA docs. | Register only the specific buffer being transferred, and only for the duration of the transfer. Deregister immediately after. |

### AF-4: Mixing O_DIRECT and Non-O_DIRECT Writes to Same File

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Writing metadata via POSIX (non-O_DIRECT) then data via GDS (O_DIRECT) to the same file | Causes page cache invalidation. The existing code comment notes "80ms page cache invalidation from mixing POSIX and O_DIRECT writes." This is already correctly avoided in the current implementation. | Continue current approach: either write everything via O_DIRECT (padding metadata to 4KB alignment) or write everything via POSIX. Do not mix. |

### AF-5: Synchronous cudaStreamSynchronize Between Every I/O Operation

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Calling cudaStreamSynchronize after every cudaMemcpy or between batch submits | Serializes GPU and I/O pipelines. The current code has sync after every D2D copy AND after every batch wave. | Minimize synchronization points. Ideally: one registration, one batch submit, one poll loop, one deregistration. Zero intermediate syncs. |

### AF-6: Using kvikIO Synchronous read()/write() for Large Transfers

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Calling `fh.write()` / `fh.read()` (synchronous single-threaded) | These do NOT use the kvikIO thread pool. A 4 GiB write becomes one sequential cuFile call. The thread pool parallelism that makes kvikIO competitive is only available via pread()/pwrite(). | Use `fh.pwrite()` / `fh.pread()` which partition the operation across the thread pool. |

---

## Feature Dependencies

```
TS-6 (benchmark on NVMe)  -- independent, do first for valid measurement
     |
TS-1 (eliminate D2D staging) --> TS-2 (eliminate wave serialization) --> TS-3 (tune chunk size)
     |                                                                      |
     +-------> D-1 (direct buffer registration)                            |
     |         D-7 (contiguous write via cudf::pack)                       |
     |                                                                      |
     +-------> D-3 (double-buffered registration)                          |
                                                                            |
TS-4 (kvikIO thread pool) -- independent, parallel track                   |
                                                                            |
TS-5 (batch read path) -- independent, parallel track                      |
                                                                            |
D-2 (overlap metadata)  -- independent                                     |
D-4 (cufile.json tuning) -- independent                                   |
D-5 (stream API) -- alternative to batch API, evaluate after TS-1/TS-2    |
D-6 (persistent buffer pool) -- alternative to per-transfer registration   |
```

## Impact-Ordered Summary

Optimization techniques ordered by expected throughput impact, from largest wins to smallest:

| Priority | ID | Technique | Expected Impact | Complexity | Track |
|----------|----|-----------|-----------------|------------|-------|
| 1 | TS-1 | Eliminate D2D staging copy | 2-4x | Medium | GDS |
| 2 | TS-2 | Eliminate 64MB wave serialization | 3-5x (cumulative with TS-1) | Medium | GDS |
| 3 | TS-4 | kvikIO thread pool + pwrite() | 2-4x | Low | kvikIO |
| 4 | TS-5 | Batch read path for disk-to-GPU | 30-50% | Medium | Both |
| 5 | TS-3 | Tune chunk size to 1MB | 20-50% | Low | GDS |
| 6 | D-7 | Contiguous write via cudf::pack | 10-30% | Medium | GDS |
| 7 | D-4 | cufile.json tuning | 5-20% | Low | GDS |
| 8 | D-1 | Direct RMM buffer registration | 5-15% | Medium | GDS |
| 9 | D-3 | Double-buffered registration | 5-10% | High | GDS |
| 10 | D-6 | Pre-registered buffer pool | 2-5% | Medium | GDS |
| 11 | D-2 | Overlap metadata serialization | 1-5% | Low | Both |
| 12 | D-5 | Stream API instead of batch | 0-10% | Medium | GDS |
| 13 | TS-6 | Benchmark on NVMe path | N/A (measurement) | Low | Both |

## MVP Recommendation

Prioritize (in order):

1. **TS-6**: Move benchmarks to /mnt/disk_2 for valid measurements (prerequisite for all optimization validation)
2. **TS-1 + TS-2**: Eliminate staging buffer and wave serialization in GDS backend (these are the root cause of ~10x gap)
3. **TS-4**: Switch kvikIO to pwrite()/pread() with 8 threads (root cause of kvikIO's ~5x gap)
4. **TS-5**: Batch the read path for disk-to-GPU converter
5. **TS-3 + D-4**: Tune chunk sizes and cufile.json parameters

**Defer:**
- D-1, D-7 (contiguous buffer approaches): Only if TS-1/TS-2 don't reach 80% target. These add GPU memory pressure.
- D-3 (double-buffered registration): LOW confidence, high complexity, small expected gain.
- D-5 (stream API): Evaluate only if batch API proves limiting after TS-1/TS-2.
- D-6 (persistent pool): Only if per-transfer registration proves too costly.

## Sources

- [NVIDIA GDS Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) -- buffer registration, alignment, thread tuning
- [NVIDIA GDS Benchmarking and Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) -- cufile.json parameters, gdsio baselines
- [NVIDIA GDS API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) -- cuFileBufRegister, batch API, stream API details
- [NVIDIA GDS Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) -- bounce buffer architecture
- [kvikIO Runtime Settings](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/) -- KVIKIO_NTHREADS, KVIKIO_TASK_SIZE defaults
- [kvikIO FileHandle API](https://docs.rapids.ai/api/libkvikio/stable/classkvikio_1_1filehandle) -- read vs pread parallelism
- [kvikIO GitHub](https://github.com/rapidsai/kvikio) -- implementation details
- Codebase analysis: `src/data/gds_io_backend.cpp`, `src/data/kvikio_io_backend.cpp`, `src/data/representation_converter.cpp`
