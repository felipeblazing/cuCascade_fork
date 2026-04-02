# Architecture Patterns

**Domain:** High-performance GPU-disk I/O pipeline optimization (GPUDirect Storage / kvikIO)
**Researched:** 2026-04-02

## Recommended Architecture

### Current Architecture (Problem Statement)

The GDS backend currently uses a **staging-buffer-with-waves** design that introduces two fundamental bottlenecks:

```
WRITE PATH (current):
  GPU source buffer
       |
       | cudaMemcpyD2D (entire 64MB wave)     <-- BOTTLENECK 1: unnecessary D2D copy
       v
  64MB registered staging buffer (16 x 4MB slots)
       |
       | cuFileBatchIOSubmit (16 ops, 4MB each) <-- BOTTLENECK 2: small ops, sequential waves
       v
  NVMe disk

  For 4 GiB: 64 sequential waves, each blocking on D2D + batch I/O.
  No overlap between waves.
```

```
READ PATH (current):
  NVMe disk
       |
       | cuFileBatchIOSubmit into staging
       v
  64MB registered staging buffer
       |
       | cudaMemcpyD2D to destination         <-- Same D2D overhead
       v
  GPU destination buffer
```

**Why this is slow:**
1. Every byte crosses GPU memory bus twice (source->staging, then staging->disk DMA). This halves effective GPU memory bandwidth.
2. 4MB slot size is 4x smaller than cuFile's `max_direct_io_size` (16MB default). The NVMe queue is underutilized.
3. Waves are fully serial: wave N must complete entirely before wave N+1 starts. No overlap of D2D copy with disk I/O.
4. `write_device_batch()` falls back to per-entry sequential `write_device()` calls when total > 64MB.

### Target Architecture: Direct-Register + Overlapping Waves

The optimized architecture eliminates the staging buffer for the data path by registering the user's GPU buffer directly with cuFile, and overlaps independent operations.

```
WRITE PATH (optimized, single large buffer):
  GPU source buffer
       |
       | cuFileBufRegister(source, size)       <-- Register once per transfer
       |
       | cuFileBatchIOSubmit(N ops, 16MB each) <-- Direct DMA, no D2D copy
       v
  NVMe disk

  For 4 GiB: single batch of 256 x 16MB ops, all submitted at once.
  cuFile library handles internal parallelism.
```

```
WRITE PATH (optimized, scattered column buffers via batch):
  GPU column buffers [buf0, buf1, buf2, ...]
       |
       | cuFileBufRegister each buffer         <-- Register per-buffer
       |
       | cuFileBatchIOSubmit(all entries)       <-- Single batch, all buffers
       v
  NVMe disk (column data at respective file offsets)
```

```
READ PATH (optimized, bulk single buffer):
  NVMe disk
       |
       | cuFileBufRegister(dest, size)
       | cuFileBatchIOSubmit(N ops, 16MB each) <-- Direct DMA into dest
       v
  GPU destination buffer

READ PATH (optimized, column reconstruction):
  NVMe disk
       |
       | For each column buffer: alloc, register, read_device
       v
  GPU column buffers (owned by cudf::column)
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `idisk_io_backend` | Abstract I/O interface (unchanged) | Converter layer via virtual dispatch |
| `gds_io_backend` (rewritten) | Direct GDS I/O with per-transfer registration | cuFile driver, registered_buffer |
| `registered_buffer` (enhanced) | RAII buffer registration lifecycle | cuFile API |
| `cufile_handle` | RAII file handle (unchanged) | cuFile API |
| `cufile_batch_guard` | RAII batch handle (unchanged) | cuFile API |
| `submit_and_wait()` | Batch submit + poll (enhanced) | cuFile batch API |
| `kvikio_io_backend` (tuned) | kvikIO with thread pool tuning | kvikIO FileHandle |
| Converter layer | Collects io_batch_entry, calls backend | Backend via `write_device_batch` |

### Data Flow

**Write (GPU -> Disk):**
1. Converter collects all column buffer pointers + file offsets into `io_batch_entry` vector
2. Converter copies header+metadata to a small device buffer (H2D)
3. Backend receives batch entries via `write_device_batch()`
4. Backend registers each unique GPU buffer with cuFile (or the contiguous source if single-buffer)
5. Backend builds CUfileIOParams_t array -- one entry per buffer, each pointing to its registered base
6. Backend submits single batch and polls for completion
7. Backend deregisters buffers (RAII)

**Read (Disk -> GPU):**
1. Converter reads header+metadata via `read_host()` (POSIX, small)
2. Converter allocates device buffers for each column via RMM
3. Converter calls `read_device()` per buffer (or `read_device_batch()` for all)
4. Backend registers destination buffer, submits cuFile read, deregisters

## Patterns to Follow

### Pattern 1: Direct Buffer Registration (Eliminate Staging Copy)

**What:** Register the user's actual GPU buffer with cuFile instead of copying to an intermediate staging buffer.

**When:** For any transfer where the source/destination GPU buffer is 4KB-aligned (which `cudaMalloc` guarantees) and the file offset is 4KB-aligned (which the disk file format already ensures via `DISK_FILE_ALIGNMENT`).

**Why:** The NVIDIA Best Practices Guide states: "If a user buffer is not registered, an intermediate pre-registered GPU buffer that is owned by the cuFile implementation is used, and there is an extra copy from there to the user buffer." By registering, we get direct DMA with zero D2D copies.

**Confidence:** HIGH -- directly from NVIDIA official documentation.

**Example:**
```cpp
void write_device(const std::string& path,
                  const void* dev_ptr,
                  std::size_t size,
                  std::size_t file_offset,
                  rmm::cuda_stream_view stream) override
{
  if (size == 0) return;

  cufile_handle cfh(path, true);

  // Register the source buffer directly -- no staging copy needed
  registered_buffer reg(const_cast<void*>(dev_ptr), size);

  // Build batch with 16MB chunks for optimal NVMe queue utilization
  constexpr std::size_t CHUNK_SIZE = 16ULL * 1024 * 1024;  // match max_direct_io_size
  auto num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;
  auto num_chunks_uint = static_cast<unsigned>(num_chunks);

  cufile_batch_guard batch(num_chunks_uint);
  std::vector<CUfileIOParams_t> params(num_chunks_uint);

  for (unsigned i = 0; i < num_chunks_uint; ++i) {
    auto chunk_offset = static_cast<std::size_t>(i) * CHUNK_SIZE;
    auto chunk_size   = std::min(CHUNK_SIZE, size - chunk_offset);

    std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
    params[i].mode                  = CUFILE_BATCH;
    params[i].u.batch.devPtr_base   = const_cast<void*>(dev_ptr);
    params[i].u.batch.devPtr_offset = static_cast<off_t>(chunk_offset);
    params[i].u.batch.file_offset   = static_cast<off_t>(file_offset + chunk_offset);
    params[i].u.batch.size          = chunk_size;
    params[i].fh                    = cfh.handle();
    params[i].opcode                = CUFILE_WRITE;
    params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
  }

  submit_and_wait(batch, params);
}
```

### Pattern 2: Larger Chunk Size (16MB Aligned to max_direct_io_size)

**What:** Use 16MB chunks instead of 4MB slots. This matches the `max_direct_io_size_kb` default of 16384 (16MB) in `/etc/cufile.json`.

**When:** Always, for the batch API path.

**Why:** The cuFile library internally splits operations at `max_direct_io_size` boundaries. Using chunks smaller than this threshold means the library processes each 4MB op as a separate IO call, reducing NVMe queue depth. Using 16MB chunks means one library-level IO call per chunk, and the library's internal parallel_io can further optimize within that.

**Confidence:** MEDIUM -- cuFile documentation states larger `max_direct_io_size` reduces calls to IO stack. The 16MB default is well-established but the exact throughput impact vs 4MB needs benchmarking.

### Pattern 3: Multi-Buffer Batch Registration for Column Writes

**What:** When `write_device_batch()` receives scattered column buffers, register each buffer individually and submit all as a single batch.

**When:** For the GPU->disk converter path where `collect_gpu_column_io_entries()` produces entries from different column buffers.

**Why:** Each `CUfileIOParams_t` in a batch can reference a different `devPtr_base`. By registering each column buffer separately, all writes go through a single batch submit with direct DMA from each buffer's location.

**Example:**
```cpp
void write_device_batch(const std::string& path,
                        const std::vector<io_batch_entry>& entries,
                        rmm::cuda_stream_view stream) override
{
  if (entries.empty()) return;

  // Deduplicate and register unique base pointers
  std::vector<std::unique_ptr<registered_buffer>> registrations;
  // ... (register each unique buffer)

  cufile_handle cfh(path, true);
  auto num_ops = static_cast<unsigned>(entries.size());
  cufile_batch_guard batch(num_ops);
  std::vector<CUfileIOParams_t> params(num_ops);

  for (unsigned i = 0; i < num_ops; ++i) {
    std::memset(&params[i], 0, sizeof(CUfileIOParams_t));
    params[i].mode                  = CUFILE_BATCH;
    params[i].u.batch.devPtr_base   = const_cast<void*>(entries[i].ptr);
    params[i].u.batch.devPtr_offset = 0;  // ptr IS the base for this entry
    params[i].u.batch.file_offset   = static_cast<off_t>(entries[i].file_offset);
    params[i].u.batch.size          = entries[i].size;
    params[i].fh                    = cfh.handle();
    params[i].opcode                = CUFILE_WRITE;
    params[i].cookie                = reinterpret_cast<void*>(static_cast<uintptr_t>(i));
  }

  submit_and_wait(batch, params);
}
```

**Confidence:** MEDIUM -- The batch API documentation confirms different `devPtr_base` per entry is valid, but registering many small buffers (null masks, offsets columns) may have nontrivial overhead. Needs benchmarking to determine if a gather-to-contiguous-then-register approach is better for many small entries.

### Pattern 4: Overlapping Waves for Very Large Transfers

**What:** When a transfer is large enough to benefit from pipelining (e.g., > 256MB), overlap batch submission of wave N+1 with polling completion of wave N using two batch handles.

**When:** For transfers that are too large to submit as a single batch (exceeding `io_batchsize` of 128 entries) or when hardware queue depth limits prevent single-shot submission.

**Why:** The Best Practices Guide states: "`min_nr` can be set to something less than `batch_size` and as the `min_nr` number of IOs are completed, that many numbers of I/Os can be submitted subsequently to the I/O pipeline resulting in an enhanced I/O throughput." This is the sliding window pattern.

**Confidence:** MEDIUM -- The principle is documented but the current architecture may not need this if direct registration makes single-batch submissions fast enough. The 4 GiB / 16MB = 256 entries fits within `io_batchsize` of 128, so two waves of 128 may be needed, but the simpler approach should be tried first.

**Example (sliding window):**
```cpp
// For transfers requiring > 128 batch entries
constexpr unsigned MAX_BATCH_OPS = 128;  // matches io_batchsize in cufile.json

// Wave 0: submit first 128 entries
cufile_batch_guard batch_a(MAX_BATCH_OPS);
// ... build params_a from entries[0..127]
submit_async(batch_a, params_a);  // non-blocking submit

// Wave 1: submit next 128 entries
cufile_batch_guard batch_b(MAX_BATCH_OPS);
// ... build params_b from entries[128..255]
submit_async(batch_b, params_b);

// Wait for wave 0, then wave 1
wait_for_completion(batch_a);
wait_for_completion(batch_b);
```

### Pattern 5: kvikIO Thread Pool Tuning

**What:** Configure kvikIO's internal thread pool for higher parallelism instead of the default single-threaded mode.

**When:** Always, when using the kvikIO backend.

**Why:** kvikIO defaults to `KVIKIO_NTHREADS=1` and `KVIKIO_TASK_SIZE=4MB`. For NVMe throughput, the NVIDIA blog on kvikIO showed that 16-64 threads with larger task sizes dramatically improve throughput. The current kvikIO backend makes a single synchronous `fh.write()` call which internally uses this thread pool.

**Confidence:** HIGH -- kvikIO documentation and the NVIDIA blog confirm these settings directly affect throughput.

**Configuration approach:**
```cpp
// At kvikio_io_backend construction time
kvikio_io_backend() {
  kvikio::defaults::thread_pool_nthreads(16);  // or environment KVIKIO_NTHREADS=16
  kvikio::defaults::task_size(16 * 1024 * 1024);  // 16MB task size
}
```

### Pattern 6: Fallback to Staging Buffer for Unregisterable Memory

**What:** Keep the staging buffer path as a fallback when direct registration fails (e.g., managed memory, or registration returns failure).

**When:** When `cuFileBufRegister` returns a non-success status.

**Why:** Not all GPU memory can be registered. cuFile registration requires `cudaMalloc`-allocated memory (not `cudaMallocManaged`). RMM pool allocators return suballocations from `cudaMalloc` blocks, which should be registerable, but edge cases exist.

**Confidence:** HIGH -- the existing staging path is correct and tested; keeping it as fallback is zero-risk.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Double-Buffered D2D Staging for GDS

**What:** Copying GPU data into a separate registered staging buffer before submitting to cuFile.

**Why bad:** This is the current architecture. It doubles GPU memory bandwidth consumption. Every byte travels: source GPU buffer -> staging GPU buffer (D2D copy) -> NVMe (DMA). The D2D copy step is entirely unnecessary when the source buffer itself can be registered.

**Instead:** Register the source buffer directly with `cuFileBufRegister`. The DMA engine reads directly from the source location.

### Anti-Pattern 2: Tiny Batch Slots (4MB)

**What:** Splitting I/O into 4MB operations.

**Why bad:** The cuFile library's `max_direct_io_size` is 16MB. Using 4MB ops means 4x more IO submissions, 4x more kernel driver round-trips, and lower NVMe queue utilization. The `parallel_io` feature in cuFile internally parallelizes within a single large request (up to `max_request_parallelism` of 4 sub-requests), but this only helps if each submitted op is >= `min_io_threshold_size_kb` (8MB default).

**Instead:** Use 16MB chunks (matching `max_direct_io_size`) or larger. Let cuFile's internal `parallel_io` handle sub-chunking optimally.

### Anti-Pattern 3: Falling Back to Sequential Per-Entry Writes

**What:** `write_device_batch()` degrades to a loop of `write_device()` calls when total bytes exceed the staging buffer.

**Why bad:** Each `write_device()` call opens a new batch handle, registers staging, copies D2D, submits, waits, then loops. For 4 GiB with 64MB staging, this is 64 sequential round-trips with no overlap.

**Instead:** With direct registration, there is no staging buffer limit. Submit all entries in a single (or two-wave) batch regardless of total size.

### Anti-Pattern 4: Opening/Registering Per Wave

**What:** Creating a new `cufile_batch_guard` and re-registering buffers for every 64MB wave.

**Why bad:** Batch setup and buffer registration have non-trivial overhead. Doing this 64 times for a 4 GiB transfer adds latency.

**Instead:** Register once, create one (or two) batch handles, submit all ops, wait once.

## Detailed Pipeline Architecture

### GDS Write Pipeline (Target Design)

```
Phase 1: Setup (one-time)
  - cuFileDriverOpen (singleton, already done)
  - cufile_handle: open file with O_CREAT | O_RDWR | O_DIRECT

Phase 2: Registration
  - For write_device(): cuFileBufRegister(source_ptr, size)
  - For write_device_batch(): cuFileBufRegister each unique buffer base

Phase 3: Batch Construction
  - Divide total transfer into 16MB chunks
  - Build CUfileIOParams_t array (up to io_batchsize=128 per batch)
  - Each entry: devPtr_base=registered_ptr, devPtr_offset=chunk_offset, file_offset, size

Phase 4: Submission + Completion
  If total ops <= 128:
    - Single cuFileBatchIOSubmit + poll loop
  If total ops > 128:
    - Wave 0: submit entries [0..127], begin polling
    - Wave 1: submit entries [128..255] while wave 0 completes
    - Wait for all waves

Phase 5: Teardown (RAII)
  - registered_buffer destructor: cuFileBufDeregister
  - cufile_batch_guard destructor: cuFileBatchIODestroy
  - cufile_handle destructor: cuFileHandleDeregister + close(fd)
```

### GDS Read Pipeline (Target Design)

```
Phase 1: Setup
  - cufile_handle: open file with O_RDONLY | O_DIRECT

Phase 2: Registration
  - cuFileBufRegister(dest_ptr, size)

Phase 3: Batch Construction
  - Same chunking as write (16MB chunks)

Phase 4: Submission + Completion
  - Same wave strategy as write

Phase 5: Teardown (RAII)
  - Same RAII cleanup
```

### Read Path Column Reconstruction (Current vs Target)

**Current:** Each `alloc_and_read_from_disk()` call issues a separate `read_device()`, which does a full staging-buffer wave cycle per column buffer. For a 10-column table, that is 10+ separate staging cycles.

**Target (simple, recommended first):** Keep per-column `read_device()` calls but each call now directly registers the destination buffer and does direct DMA. The overhead is per-call cuFileBufRegister/Deregister, but the D2D copy is eliminated.

**Target (advanced, if needed):** Add `read_device_batch()` to the converter read path -- collect all column buffer destinations, call backend once. Requires changes to `reconstruct_column_from_disk()` to collect entries first, then batch-read. This is more complex because each column's buffer is allocated by RMM independently. Worth doing only if per-column registration overhead is measured as significant.

### kvikIO Optimization Pipeline

```
Current:
  kvikio::FileHandle fh(path, "w");
  fh.write(dev_ptr, size, file_offset, 0);  // Single-threaded, default 4MB tasks

Target:
  // At backend construction:
  kvikio::defaults::thread_pool_nthreads(16);
  kvikio::defaults::task_size(16 * 1024 * 1024);

  // Per operation (same API):
  kvikio::FileHandle fh(path, "w");
  fh.write(dev_ptr, size, file_offset, 0);  // Now 16 threads, 16MB tasks
```

kvikIO internally uses cuFile when GDS is available, or falls back to POSIX. Its thread pool parallelizes by splitting the write into `task_size` chunks across `nthreads` workers. With 16 threads x 16MB tasks, a 4 GiB write becomes 256 tasks with 16 concurrent, matching gdsio's thread-pool approach.

## Buffer Registration Lifecycle

### Registration Scope Options

| Approach | Scope | Overhead | When to Use |
|----------|-------|----------|-------------|
| Per-transfer register/deregister | Single write_device/read_device call | Registration + deregistration per call | Default approach, simplest |
| Backend-owned pool of registered buffers | Backend lifetime | One-time registration | If per-transfer reg overhead is measured as bottleneck |
| Register at allocation, deregister at free | Buffer lifetime | Requires RMM integration | Out of scope (PROJECT.md constraint) |

**Recommended approach: Per-transfer registration.** Rationale:
1. Registration overhead is amortized over the transfer time (a 4 GiB transfer takes hundreds of ms; registration takes microseconds).
2. No need to manage a buffer pool or integrate with RMM allocation callbacks.
3. If registration fails (e.g., too many registered buffers), fall back to the existing staging path.
4. RAII `registered_buffer` class already exists and handles cleanup.

### Registration Alignment Requirements

- `cudaMalloc` returns 256-byte aligned pointers. cuFile requires no specific alignment for buffer registration itself.
- File offsets must be 4KB-aligned for optimal GDS performance. The disk file format already enforces this via `DISK_FILE_ALIGNMENT`.
- I/O sizes do not need to be aligned, but the best practices guide notes aligned operations avoid internal bounce buffers.

### Registration Failure Handling

```cpp
registered_buffer reg(dev_ptr, size);
if (!reg.is_registered()) {
  // Fall back to staging buffer path (existing implementation)
  write_device_via_staging(path, dev_ptr, size, file_offset, stream);
  return;
}
// Proceed with direct I/O using registered buffer
```

## How gdsio Achieves Its Throughput

Based on analysis of gdsio's documented behavior and `/etc/cufile.json` configuration:

1. **Thread parallelism:** gdsio uses `-w` worker threads (4 by default, user tested with 4). Each thread makes independent cuFile calls. This maps to `max_io_threads=4` in cufile.json.

2. **Large I/O sizes:** gdsio uses `-i` to set I/O size (typically 1MB). cuFile's `parallel_io` feature splits each 1MB call into up to `max_request_parallelism=4` sub-requests when the size exceeds `min_io_threshold_size_kb=8MB`.

3. **Queue depth:** `max_io_queue_depth=128` allows 128 pending operations. With 4 threads each submitting I/Os, the NVMe queue stays full.

4. **Direct buffer access:** gdsio registers its buffers, so there are no D2D staging copies.

5. **No metadata overhead:** gdsio writes raw data without file format headers, column metadata, or alignment padding.

**Key insight:** gdsio's 6.73 GiB/s write / 13.35 GiB/s read comes from 4 threads x direct registered buffers x 1MB I/O sizes x deep NVMe queue. The cuCascade GDS backend currently uses 1 thread x staging buffer x 4MB slots x 16 depth, which explains the 10x+ gap.

## Scalability Considerations

| Concern | At 64 MB | At 1 GiB | At 4 GiB |
|---------|----------|----------|----------|
| Batch entries | 4 (16MB chunks) | 64 entries | 256 entries (2 waves of 128) |
| Registration overhead | Negligible | Negligible | Negligible (~1 reg call) |
| Wave overlap benefit | None (single wave) | None (fits in 1 batch) | Moderate (2 waves) |
| File handle | 1 open/close | 1 open/close | 1 open/close |
| Memory overhead | 0 extra (no staging) | 0 extra | 0 extra |

## Refactoring Order (Build Dependencies)

These changes are ordered by dependency. Items in the same group are independent.

### Group 1: Foundation (independent, no API changes)

1. **Increase chunk size from 4MB to 16MB** in `gds_io_backend::write_device()` and `read_device()`.
   - Single constant change. No API change. Immediate throughput improvement by better matching `max_direct_io_size`.
   - Risk: LOW

2. **Tune kvikIO thread pool** at `kvikio_io_backend` construction.
   - Add two lines to constructor. No API change. Independent of all GDS changes.
   - Risk: LOW

### Group 2: Core optimization (depends on nothing, biggest impact)

3. **Rewrite `write_device()` to use direct buffer registration** instead of staging copy.
   - Replace: D2D copy + staging batch -> direct register + batch from source.
   - Keep staging path as fallback if registration fails.
   - This is the single highest-impact change (eliminates D2D copy for all writes).
   - Risk: MEDIUM (registration could fail for certain allocator configurations)

4. **Rewrite `read_device()` to use direct buffer registration.**
   - Same transformation as write path.
   - Risk: MEDIUM

### Group 3: Batch path (depends on Group 2 patterns)

5. **Rewrite `write_device_batch()` with per-buffer registration.**
   - Remove 64MB staging limit. Register each buffer, submit all in one batch.
   - Eliminates the fallback-to-sequential path.
   - Risk: MEDIUM (many small buffer registrations; may need gather fallback for tiny entries)

6. **Rewrite `read_device_batch()` with per-buffer registration.**
   - Same pattern as write batch.
   - Risk: MEDIUM

### Group 4: Large transfer optimization (depends on Group 2)

7. **Add wave overlap for transfers > 128 batch entries.**
   - Implement sliding window with two batch handles.
   - Only matters for single write_device/read_device calls with > 2 GiB.
   - Risk: LOW (additive, does not change < 128 entry path)

### Group 5: Cleanup (depends on all above)

8. **Remove or shrink staging buffer.**
   - Once direct registration is proven, the 64MB staging allocation can be removed (saving GPU memory) or kept as a small fallback buffer.
   - Risk: LOW

## Mapping to Existing Interface

All optimizations are **internal to the backend implementations**. The `idisk_io_backend` interface does not change:

```cpp
class idisk_io_backend {
 public:
  virtual void write_device(...) = 0;     // Unchanged
  virtual void read_device(...) = 0;      // Unchanged
  virtual void write_host(...) = 0;       // Unchanged (not optimized)
  virtual void read_host(...) = 0;        // Unchanged (not optimized)
  virtual void write_device_batch(...);   // Unchanged signature
  virtual void read_device_batch(...);    // Unchanged signature
};
```

The converter layer (`representation_converter.cpp`) also requires no changes. The `io_batch_entry` struct and collection pattern remain valid -- the only change is that the backend handles entries without staging.

## Sources

- [NVIDIA GPUDirect Storage Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) -- Buffer registration, max_direct_io_size, batch API sliding window. **HIGH confidence.**
- [NVIDIA GPUDirect Storage API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) -- cuFileBatchIOSubmit/GetStatus parameters, stream API, internal IO splitting. **HIGH confidence.**
- [NVIDIA GPUDirect Storage Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) -- Architecture layers, bounce buffer elimination, BAR1 chunking. **HIGH confidence.**
- [NVIDIA GPUDirect Storage Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) -- gdsio usage, cufile.json parameters, throughput benchmarking. **HIGH confidence.**
- [NVIDIA GPUDirect Storage Overview](https://developer.nvidia.com/blog/gpudirect-storage/) -- DMA architecture, performance measurements. **HIGH confidence.**
- [kvikIO Runtime Settings](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/) -- KVIKIO_NTHREADS, KVIKIO_TASK_SIZE defaults. **HIGH confidence.**
- [kvikIO C++ Documentation](https://docs.rapids.ai/api/libkvikio/stable/) -- FileHandle API, parallel IO internals. **HIGH confidence.**
- [NVIDIA MagnumIO GDS Samples](https://github.com/NVIDIA/MagnumIO/tree/main/gds/samples) -- Batch and async API usage patterns. **MEDIUM confidence** (samples illustrate API usage but not production patterns).
- `/etc/cufile.json` on target system -- Actual configuration values for max_direct_io_size, io_batchsize, parallel_io, etc. **HIGH confidence** (measured from target system).
