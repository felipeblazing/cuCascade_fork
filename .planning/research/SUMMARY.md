# Project Research Summary

**Project:** cuCascade Disk I/O Performance Optimization
**Domain:** High-performance GPU Direct Storage (GDS) and kvikIO I/O optimization for NVMe SSDs
**Researched:** 2026-04-02
**Confidence:** HIGH

## Executive Summary

cuCascade's disk I/O backends are achieving 8-17% of the hardware's demonstrated throughput: the GDS backend writes at 582 MiB/s and kvikIO at 1.20 GiB/s, against a measured hardware ceiling of 6.73 GiB/s writes and 13.35 GiB/s reads (per gdsio baselines). The root cause is not a missing dependency or an API limitation -- both cuFile and kvikIO already expose every API required to close this gap. The problem is a staging-buffer architecture that forces an unnecessary device-to-device memory copy on every transfer, serialized 64MB wave processing, and single-threaded kvikIO usage with default settings.

The recommended approach is a two-track optimization. Track 1 (GDS): eliminate the D2D staging copy by registering the caller's GPU buffer directly with cuFile, submit batch operations against the original buffer, and respect T4 BAR1 limits via chunked registration. Track 2 (kvikIO): switch from synchronous `write()`/`read()` to parallel `pwrite()`/`pread()` and configure the thread pool to 8-16 threads. No new dependencies are needed; both cuFile (via CUDA toolkit) and kvikIO 26.06 are already linked.

The key risks are: (1) T4 BAR1 aperture exhaustion -- the T4 has only 256MB of BAR1 memory, so registering a 4 GiB buffer at once will fail; chunked registration with a configurable cap is mandatory; (2) silent compat mode fallback -- cuFile silently falls back to POSIX bounce buffers when alignment or configuration is wrong, with no error, only degraded throughput; (3) the batch API may not truly batch at the kernel level for large entries, so synchronous cuFile with its internal thread pool may outperform batch for registered buffers. All three risks have well-defined mitigation strategies identified in the research.

## Key Findings

### Recommended Stack

No new dependencies are needed. The existing cuFile (GDS r1.17 via CUDA 13.1) and kvikIO 26.06 libraries expose every API required. The optimization is entirely about how these APIs are called, not which APIs are available.

**Core technologies (usage changes only):**
- **cuFileBufRegister/Deregister**: Per-transfer buffer registration -- enables zero-copy DMA by registering the caller's GPU buffer directly, eliminating the D2D staging copy
- **cuFileBatchIOSubmit (or cuFileWrite sync)**: Direct I/O against registered buffers -- the batch API for scatter/gather of column buffers, or synchronous API with cuFile's internal parallelism for single contiguous buffers
- **kvikio::FileHandle::pwrite/pread**: Parallel multi-threaded I/O -- replaces single-threaded `write()`/`read()`, splits transfers into task-sized chunks across the thread pool
- **kvikio::defaults::set_thread_pool_nthreads()**: Thread pool sizing -- must be set to 8-16 (default is 1)
- **cufile.json**: GDS configuration tuning -- `max_io_threads`, `per_buffer_cache_size_kb`, `parallel_io` parameters

**Critical version requirements:**
- libcufile r1.16+ (CUDA 13.1 ships r1.17) -- r1.16 removed batch I/O size limitations
- libkvikio 26.02+ (26.06 nightly available) -- pread/pwrite and async APIs available
- CUDA 12.2+ for stream-ordered cuFile APIs (CUDA 13.1 available, fully supported)

### Expected Features

**Must have (table stakes -- required to reach 50%+ of hardware throughput):**
- TS-1: Eliminate D2D staging copy in GDS backend (2-4x improvement)
- TS-2: Eliminate 64MB wave serialization (3-5x cumulative with TS-1)
- TS-4: Configure kvikIO thread pool and switch to pwrite/pread (2-4x for kvikIO)
- TS-5: Batch read path for disk-to-GPU converter (30-50% for multi-column tables)
- TS-6: Ensure benchmarks run on NVMe (/mnt/disk_2), not /tmp (measurement validity)

**Should have (to bridge from 50-60% to 80%+ of hardware throughput):**
- D-4: Tune cufile.json parameters (5-20%)
- D-7: Contiguous write via cudf::pack for GDS path (10-30% for many-column tables)
- D-1: Direct RMM buffer registration with amortized registration (5-15%)
- D-2: Overlap I/O submission with metadata serialization (1-5%)

**Defer (v2+):**
- D-3: Double-buffered GDS pipeline (LOW confidence, high complexity, small gain)
- D-5: cuFile stream async API (secondary to batch/sync; evaluate only if batch proves limiting)
- D-6: Pre-registered persistent buffer pool (only if per-transfer registration proves costly)

### Architecture Approach

The target architecture replaces the staging-buffer-with-waves pattern with a direct-register-and-submit pattern. The `idisk_io_backend` interface remains unchanged -- all optimizations are internal to the backend implementations. For writes, the caller's GPU buffer is registered directly with cuFile and batch operations point at the original memory. For reads, destination buffers are registered before DMA. The converter layer and `io_batch_entry` collection pattern remain valid. The kvikIO backend requires only configuration changes (thread pool, task size) and switching from `write()` to `pwrite()`.

**Major components (changes only, interface preserved):**
1. **gds_io_backend (rewritten internals)** -- Direct buffer registration + batch submit from source; chunked registration for BAR1 safety; staging buffer demoted to fallback path
2. **kvikio_io_backend (tuned)** -- Thread pool configured at construction (8-16 threads, 1-16MB task size); switch to pwrite/pread for parallel I/O
3. **registered_buffer (enhanced)** -- RAII registration with BAR1-aware size capping; graceful fallback when registration fails
4. **cufile.json (new)** -- Project-specific GDS configuration for NVMe throughput optimization

### Critical Pitfalls

1. **T4 BAR1 exhaustion (256MB limit)** -- cuFileBufRegister on a 4 GiB buffer will fail on T4. Use chunked registration capped at 128MB per chunk; query BAR1 at runtime via NVML. This must be solved simultaneously with the direct-registration change.

2. **Silent alignment-triggered bounce buffer fallback** -- cuFile requires 4KB alignment on devPtr_offset, file_offset, and I/O size. Misaligned operations silently route through internal bounce buffers, halving throughput with no error. Assert 4KB alignment on all batch params; pad as needed.

3. **Batch entry size exceeding per_buffer_cache_size triggers compat mode** -- Per-entry sizes exceeding the default 1MB `per_buffer_cache_size_kb` cause per-entry POSIX fallback. Either size entries to fit, increase the threshold in cufile.json, or use synchronous cuFileWrite with internal parallelism instead of batch API for large transfers.

4. **Batch API does not truly batch at the kernel level** -- A 2025 NIXL study found cuFileBatchIOSubmit processes entries individually. For large registered buffers, synchronous cuFileRead/cuFileWrite with cuFile's internal thread pool (`max_io_threads`) may outperform the batch API. Benchmark both approaches before committing.

5. **kvikIO single-threaded default** -- KVIKIO_NTHREADS defaults to 1. The synchronous `write()`/`read()` API does NOT use the thread pool. Must switch to `pwrite()`/`pread()` AND set thread count to 8-16.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 0: Environment Validation and Baseline Measurement
**Rationale:** Valid measurements are a prerequisite for validating any optimization. The current benchmarks may run on /tmp (tmpfs) rather than the NVMe device, and GDS compat mode may be silently active.
**Delivers:** Verified baseline numbers on the correct hardware path; confirmed GDS is using DMA not compat mode; verified ext4 mount options.
**Addresses:** TS-6 (benchmark on NVMe), environment validation
**Avoids:** Pitfall 10 (ext4 dioread_nolock), Pitfall 17 (silent compat mode)

### Phase 1: GDS Direct Registration (Core Optimization)
**Rationale:** The D2D staging copy and wave serialization account for the majority of the 11.8x throughput gap. This is the single highest-impact change. It must be done before any other GDS optimization because all subsequent improvements depend on direct registration.
**Delivers:** Rewritten `write_device()` and `read_device()` using direct buffer registration with BAR1-aware chunking; staging buffer demoted to fallback; decision on batch vs sync API based on benchmarking.
**Addresses:** TS-1 (eliminate D2D staging), TS-2 (eliminate wave serialization), TS-3 (tune chunk size)
**Avoids:** Pitfall 1 (D2D staging tax), Pitfall 2 (BAR1 exhaustion), Pitfall 3 (alignment violations), Pitfall 4 (compat mode per-entry), Pitfall 9 (batch vs sync API selection), Pitfall 13 (registration churn), Pitfall 16 (RMM suballocation)

### Phase 2: Batch Path and kvikIO Optimization
**Rationale:** After Phase 1 proves direct registration works for single-buffer transfers, extend to multi-buffer batch operations and optimize the kvikIO backend. These are independent tracks that can be parallelized.
**Delivers:** Rewritten `write_device_batch()` and `read_device_batch()` with per-buffer registration; kvikIO configured with parallel thread pool and pwrite/pread; batch read path for disk-to-GPU converter.
**Addresses:** TS-4 (kvikIO thread pool), TS-5 (batch read path), D-4 (cufile.json tuning)
**Avoids:** Pitfall 5 (serial wave processing for large batch), Pitfall 7 (file handle churn), Pitfall 8 (kvikIO single-threaded), Pitfall 15 (batch max entries)

### Phase 3: Advanced Throughput Tuning
**Rationale:** After the core optimizations are proven, apply advanced techniques to close remaining gap to 80%+ target. These are incremental improvements that depend on measuring the gap remaining after Phases 1-2.
**Delivers:** cufile.json tuning validated on NVMe hardware; contiguous write via cudf::pack for many-column tables (if scatter-gather proves suboptimal); metadata I/O overlap; potential staging buffer removal.
**Addresses:** D-7 (contiguous write), D-1 (amortized registration), D-2 (metadata overlap), D-4 (cufile.json finalization)
**Avoids:** Pitfall 6 (O_DIRECT/POSIX mixing), Anti-feature AF-1 (per-column registration)

### Phase Ordering Rationale

- Phase 0 before Phase 1 because valid measurement is required to know whether optimizations are working. Without confirming GDS is active (not compat mode) and benchmarks target NVMe, all subsequent throughput numbers are meaningless.
- Phase 1 before Phase 2 because direct buffer registration is the foundation. The batch path rewrite reuses the same registration pattern, and kvikIO optimization is independent but lower impact.
- Phase 2 kvikIO work is independent of Phase 2 GDS batch work -- these can be done in parallel by different developers or sequentially by the same developer.
- Phase 3 is measurement-driven -- which optimizations from Phase 3 are needed depends on the throughput achieved after Phases 1-2. If 80% target is reached, Phase 3 items become optional.
- The entire plan avoids all six anti-features identified: no per-column registration, no tiny chunks, no over-registration, no O_DIRECT mixing, no intermediate syncs, no kvikIO synchronous API for large transfers.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** The batch API vs synchronous API decision requires benchmarking on the target T4 hardware. The NIXL study suggests sync may win for large registered buffers, but this has not been validated in the cuCascade context. Run a focused benchmark comparing `cuFileWrite` (sync, registered buffer) vs `cuFileBatchIOSubmit` (batch, registered buffer) for 1GB and 4GB transfers.
- **Phase 1:** RMM suballocation registration behavior needs testing. The PROJECT.md constraint explicitly says "Cannot assume RMM-allocated GPU memory is registered with cuFile." Verify registration works with the actual RMM pool allocator in use.

Phases with standard patterns (skip research-phase):
- **Phase 0:** Standard environment validation -- check mount options, run gds_check, verify device path. Well-documented in NVIDIA configuration guide.
- **Phase 2 (kvikIO):** Well-documented API change: `write()` to `pwrite()`, set `thread_pool_nthreads()`. The kvikIO docs are clear and complete.
- **Phase 3:** All items are incremental tuning with documented parameters. cufile.json tuning is trial-and-error with clear knobs.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies needed. All APIs already available and documented. Verified against NVIDIA GDS official docs (Best Practices, API Reference, Configuration Guide) and kvikIO official docs. |
| Features | HIGH | Performance gap root causes identified through codebase analysis cross-referenced with NVIDIA documentation. Impact estimates based on gdsio baselines and documented API behavior. |
| Architecture | HIGH | Target architecture follows NVIDIA's own recommended patterns (direct registration, single batch submit). Interface compatibility confirmed -- `idisk_io_backend` does not change. |
| Pitfalls | HIGH | 17 pitfalls identified from official NVIDIA docs, peer-reviewed research (NIXL 2025), and codebase analysis. T4 BAR1 limit from hardware datasheet. Critical pitfalls have concrete mitigation strategies. |

**Overall confidence:** HIGH

### Gaps to Address

- **Batch vs sync API throughput on T4**: The NIXL study measured on different hardware. Must benchmark both approaches on the T4 CI runner to make a definitive API selection. Plan a focused benchmark in Phase 1 before committing to the implementation path.
- **RMM pool allocator registration behavior**: cuFileBufRegister on RMM suballocated memory is untested in this codebase. If registration fails silently for suballocations, the fallback to staging must remain. Test early in Phase 1.
- **Read path throughput**: All current benchmarks and baselines are write-focused. Read path throughput needs to be measured (gdsio shows 13.35 GiB/s read, roughly 2x write). The read-path optimizations may have different bottleneck profiles.
- **cufile.json `per_buffer_cache_size_kb` actual default**: PITFALLS.md identifies the default as 1MB (1024 KB) based on API docs, but the system `/etc/cufile.json` may have a different value. Must query at runtime via `cuFileDriverGetProperties()` during Phase 0.
- **kvikIO pwrite() interaction with GDS**: kvikIO internally uses cuFile when GDS is available. Confirm that pwrite() with GDS enabled uses cuFile's optimal path (not POSIX fallback) and that the thread pool does not conflict with cuFile's internal threads.

## Sources

### Primary (HIGH confidence)
- [NVIDIA GDS Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html) -- buffer registration, alignment, bounce buffer behavior, I/O patterns
- [NVIDIA GDS API Reference Guide](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) -- cuFileBufRegister, batch API, stream API, internal IO splitting
- [NVIDIA GDS Configuration Guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) -- cufile.json parameters, gdsio baselines, throughput benchmarking
- [NVIDIA GDS Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html) -- DMA architecture, BAR1 chunking, bounce buffer elimination
- [NVIDIA GDS O_DIRECT Requirements Guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html) -- O_DIRECT alignment, page cache behavior
- [NVIDIA GDS Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) -- nvidia-fs stats, BAR1 diagnostics
- [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) -- T4 BAR1 = 256MB specification
- [kvikIO Runtime Settings](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/) -- KVIKIO_NTHREADS, KVIKIO_TASK_SIZE defaults
- [libkvikio C++ API Reference](https://docs.rapids.ai/api/libkvikio/stable/) -- FileHandle pread/pwrite/read_async/write_async APIs

### Secondary (MEDIUM confidence)
- [NIXL GDS Benchmarking Study (2025)](http://cs.iit.edu/~scs/assets/files/muradli2025gpudirect.pdf) -- batch API internal behavior, GPU DIRECT vs GPU BATCH comparison
- [NVIDIA MagnumIO GDS Samples](https://github.com/NVIDIA/MagnumIO/tree/main/gds/samples) -- batch and async API usage patterns
- [NIXL GDS Plugin](https://github.com/ai-dynamo/nixl/tree/main/src/plugins/cuda_gds) -- real-world cufile.json configuration
- [Percona -- ext4 O_DIRECT Performance](https://www.percona.com/blog/watch-out-for-disk-i-o-performance-issues-when-running-ext4/) -- ext4 dioread_nolock impact

### Tertiary (LOW confidence)
- [NVIDIA Developer Forums -- BAR size on T4](https://forums.developer.nvidia.com/t/ibv-reg-mr-and-bar-size-on-t4/201907) -- T4 BAR1 practical limits (forum post, needs validation)

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
