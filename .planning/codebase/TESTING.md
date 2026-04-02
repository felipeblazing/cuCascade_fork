# Testing Patterns

**Analysis Date:** 2026-04-02

## Test Framework

**Runner:**
- Catch2 v2.13.10 (fetched via CMake `FetchContent` in `test/CMakeLists.txt`)
- Config: `test/CMakeLists.txt`
- Single test executable: `cucascade_tests` (all test `.cpp` files linked together)

**Benchmark Framework:**
- Google Benchmark v1.8.3 (fetched via `FetchContent`)
- Single benchmark executable: `cucascade_benchmarks`
- Main: `benchmark/benchmark_main.cpp` — contains only `BENCHMARK_MAIN();`

**Assertion Library:**
- Catch2 built-in: `REQUIRE`, `CHECK`, `REQUIRE_THROWS`, `REQUIRE_THROWS_AS`, `REQUIRE_THROWS_WITH`, `REQUIRE_NOTHROW`

**Run Commands:**
```bash
# From build directory (e.g., build/debug/):
./cucascade_tests                          # Run all tests
./cucascade_tests "[disk]"                 # Run tests with tag
./cucascade_tests "[.multi-device]"        # Include hidden multi-device tests
./cucascade_tests "~[.multi-device]"       # Exclude multi-device tests
./cucascade_tests "[memory_space]"         # Run by tag

# Benchmarks
./cucascade_benchmarks                     # Run all benchmarks
./cucascade_benchmarks --benchmark_filter=BM_ConvertGpuToDisk

# Via CTest (registered as single test target "cucascade_tests")
ctest --test-dir build/debug/
```

**Coverage:**
- Not enforced; no coverage target detected

## Test File Organization

**Location:**
- All test files are separate from source (not co-located)
- Tests live under `test/` mirroring the `src/` and `include/cucascade/` layout

**Naming:**
- `test_<module_name>.cpp` for test files
- `test_<module_name>.cu` / `test_<module_name>.cuh` for CUDA test files
- `benchmark_<module_name>.cpp` for benchmark files

**Directory structure:**
```
test/
├── unittest.cpp            # Custom main() + global GPU pool setup
├── CMakeLists.txt          # Single target collecting all test sources
├── data/
│   ├── test_data_batch.cpp
│   ├── test_data_repository.cpp
│   ├── test_data_repository_manager.cpp
│   ├── test_data_representation.cpp
│   ├── test_disk_host_converters.cpp
│   ├── test_disk_io_backend.cpp
│   ├── test_gpu_disk_converters.cpp
│   └── test_representation_converter.cpp
├── memory/
│   ├── test_gpu_kernels.cu
│   ├── test_gpu_kernels.cuh
│   ├── test_memory_reservation_manager.cpp
│   ├── test_small_pinned_host_memory_resource.cpp
│   └── test_topology_discovery.cpp
└── utils/
    ├── cudf_test_utils.hpp     # Stream-aware table comparison
    ├── cudf_test_utils.cpp
    ├── mock_test_utils.hpp     # Mock memory spaces, mock representations
    └── test_memory_resources.hpp  # Shared device resource wrapper
```

```
benchmark/
├── benchmark_main.cpp                  # BENCHMARK_MAIN() only
├── benchmark_disk_converter.cpp        # GPU/Host <-> Disk benchmarks
└── benchmark_representation_converter.cpp  # GPU <-> Host benchmarks
```

## Custom Test Runner (`test/unittest.cpp`)

The test runner sets up a global GPU memory pool before any tests run and installs a device-sync listener:

```cpp
// Global GPU pool setup (anonymous namespace)
class test_gpu_pool {
  // Initializes rmm::mr::cuda_async_memory_resource
  // Reads pool size from CUCASCADE_TEST_GPU_POOL_BYTES env var
  static constexpr std::size_t default_initial_bytes = 2ULL * 1024 * 1024 * 1024;
  static constexpr std::size_t default_max_bytes     = 10ULL * 1024 * 1024 * 1024;
};
test_gpu_pool global_pool;  // constructed at program start

// Device sync after each test case
struct device_sync_listener : Catch::TestEventListenerBase {
  void testCaseEnded(Catch::TestCaseStats const&) override { cudaDeviceSynchronize(); }
};
CATCH_REGISTER_LISTENER(device_sync_listener)

int main(int argc, char* argv[]) {
  int code = Catch::Session().run(argc, argv);
  std::fflush(stdout);
  std::fflush(stderr);
  return code;
}
```

## Test Structure

**Simple TEST_CASE pattern:**
```cpp
TEST_CASE("data_batch Construction", "[data_batch]")
{
  auto data = std::make_unique<mock_data_representation>(memory::Tier::GPU, 2048);
  data_batch batch(1, std::move(data));

  REQUIRE(batch.get_batch_id() == 1);
  REQUIRE(batch.get_current_tier() == memory::Tier::GPU);
  REQUIRE(batch.get_state() == batch_state::idle);
}
```

**Parameterized tests with GENERATE:**
```cpp
TEST_CASE("gpu disk round-trip numeric types", "[disk][gpu-converter][numeric]")
{
  auto type = GENERATE(cudf::type_id::INT8,
                       cudf::type_id::INT16,
                       cudf::type_id::INT32,
                       /* ... */);

  SECTION("100 rows") { gpu_disk_round_trip_test(make_typed_table(type, 100)); }
}
```

**BDD-style SCENARIO/GIVEN/WHEN/THEN (used for complex state flows in memory tests):**
```cpp
SCENARIO("Reservation Concepts on Single Gpu Manager", "[memory_space]")
{
  GIVEN("A single gpu manager")
  {
    auto manager = createSingleDeviceMemoryManager();
    WHEN("a reservation is made with overflow policy to ignore")
    {
      auto reservation = manager->request_reservation(...);
      THEN("upstream and others see it as allocated/unavailable")
      {
        CHECK(gpu_device_0->get_total_reserved_memory() == res_size);
      }
    }
  }
}
```

**Hidden tests (skipped by default):**
- Tag with `[.multi-device]` to hide from default runs: `TEST_CASE("...", "[memory_space][.multi-device]")`
- Run explicitly with `"[.multi-device]"` flag

## Section Separators in Test Files

Test files use section comment banners to group related tests:
```cpp
// =============================================================================
// Numeric type round-trip tests
// =============================================================================
```

## Mocking

**No mocking framework.** All mocks are hand-written in `test/utils/`.

**Mock memory space (`test/utils/mock_test_utils.hpp`):**
```cpp
// Factory function — returns shared_ptr<memory_space> for any tier
auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);
// Disk spaces use /tmp as mount path, 1 GiB capacity
```

**Mock data representation (`test/utils/mock_test_utils.hpp`):**
```cpp
class mock_data_representation : private mock_memory_space_holder,
                                 public idata_representation {
  // Holds its own memory_space via mock_memory_space_holder (multiple inheritance trick
  // to ensure memory_space is constructed before idata_representation base class)
  explicit mock_data_representation(memory::Tier tier, size_t size = 1024, size_t device_id = 0);
};
```

**Custom test representations (inline in test files):**
Test files define their own `idata_representation` subclasses inside anonymous namespaces when the test requires a specific type not available in mocks (e.g., `custom_test_representation` in `test/data/test_representation_converter.cpp`).

**Shared device resource (`test/utils/test_memory_resources.hpp`):**
```cpp
// Wraps RMM current device resource for sharing across memory spaces in tests
inline std::unique_ptr<rmm::mr::device_memory_resource>
  make_shared_current_device_resource(int, size_t);
```

**What to mock:**
- Memory spaces — always use `test::make_mock_memory_space()` rather than constructing `memory_space` with real allocators
- Converter registries — construct `representation_converter_registry` and call `register_builtin_converters()` directly

**What NOT to mock:**
- The converter functions themselves — round-trip tests use the real `register_builtin_converters()` to test actual I/O
- RMM memory resources — use the global pool from `unittest.cpp` or `make_shared_current_device_resource`

## Test Utilities

**`test::expect_cudf_tables_equal_on_stream(left, right, stream)` (`test/utils/cudf_test_utils.hpp`, `test/utils/cudf_test_utils.cpp`):**
- Stream-aware comparison; always synchronizes `stream` before comparing
- Compares row count, column count, type IDs, and raw data bytes per column
- On mismatch: prints hex context around the first differing byte to stdout with `[cudf-equal]` prefix
- Called via `REQUIRE(cudf_tables_have_equal_contents_on_stream(...))` or the throwing wrapper

**`test::create_simple_cudf_table(num_rows, num_columns, mr, stream)` (`test/utils/mock_test_utils.hpp`):**
- Overloaded: can be called with just row count, or row+column count
- Single-column: INT32 filled with `0x42`; two-column: INT32 (`0x11`) + INT64 (`0x22`)
- Default: 100 rows, 2 columns

**`test::create_conversion_test_configs()` (`test/utils/mock_test_utils.hpp`):**
- Returns `memory_space_config` vector for a single GPU (2 GiB) + single HOST (4 GiB)
- Uses `make_shared_current_device_resource` to reuse the global GPU pool

## Round-Trip Test Pattern

The predominant pattern for disk/converter tests is a helper function that performs the full conversion chain and asserts equality:

```cpp
namespace {

/// Round-trip test helper: GPU -> disk -> GPU, compare tables.
void gpu_disk_round_trip_test(std::unique_ptr<cudf::table> original_table)
{
  rmm::cuda_stream stream;
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  auto gpu_rep = std::make_unique<gpu_table_representation>(
    std::move(original_table), *gpu_space);

  auto disk_rep = registry.convert<disk_data_representation>(
    *gpu_rep, disk_space.get(), stream.view());

  auto gpu_rep2 = registry.convert<gpu_table_representation>(
    *disk_rep, gpu_space.get(), stream.view());

  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table(), gpu_rep2->get_table(), stream.view());
}

}  // namespace

TEST_CASE("gpu disk round-trip numeric types", "[disk][gpu-converter][numeric]")
{
  auto type = GENERATE(cudf::type_id::INT32, cudf::type_id::INT64, ...);
  SECTION("100 rows") { gpu_disk_round_trip_test(make_typed_table(type, 100)); }
}
```

All file-local helpers (test helpers, table factories) go in anonymous namespaces.

## Exception Testing

```cpp
// Assert that a specific exception type is thrown
REQUIRE_THROWS_AS(mr->allocate(stream, reservation_size * 2), rmm::bad_alloc);

// Assert that any exception is thrown
REQUIRE_THROWS(
  registry.convert<disk_data_representation>(*gpu_rep, disk_space.get(), stream.view()));

// Assert exception message
REQUIRE_THROWS_WITH(discovery.discover(),
  "CUDA_VISIBLE_DEVICES entry 99999999 is out of range");
```

## Benchmark Patterns

**Setup/Teardown:**
```cpp
// Global shared manager (no per-benchmark construction)
static std::shared_ptr<memory_reservation_manager> g_shared_memory_manager;

void DoSetup([[maybe_unused]] const benchmark::State& state)
{
  ensure_kvikio_direct_io();
  if (!g_shared_memory_manager) {
    g_shared_memory_manager =
      std::make_shared<memory_reservation_manager>(create_benchmark_configs());
  }
}

void DoTeardown([[maybe_unused]] const benchmark::State& state) { /* no-op */ }
```

**Benchmark function pattern:**
```cpp
void BM_ConvertGpuToDisk(benchmark::State& state)
{
  int64_t total_bytes = state.range(0);
  int num_columns     = static_cast<int>(state.range(1));

  // Setup outside timed loop
  auto gpu_rep = /* ... create GPU representation ... */;

  // Warmup (outside timed loop)
  auto warmup_result = registry->convert<disk_data_representation>(...);
  stream.synchronize();

  // Timed loop
  for ([[maybe_unused]] auto _ : state) {
    auto disk_result = registry->convert<disk_data_representation>(*gpu_rep, disk_space, stream.view());
    stream.synchronize();
  }

  // Report throughput
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(bytes_transferred));
  state.counters["columns"] = static_cast<double>(num_columns);
  state.counters["bytes"]   = static_cast<double>(bytes_transferred);
}
```

**Benchmark registration:**
```cpp
BENCHMARK(BM_ConvertGpuToDisk)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->Args({1 * MiB, 4})
  ->Args({64 * MiB, 4})
  ->Args({512 * MiB, 4})
  ->Args({4 * GiB, 4})
  ->Unit(benchmark::kMillisecond);

// Multi-range sweep:
BENCHMARK(BM_ConvertGpuToHost)
  ->Setup(DoSetup)
  ->Teardown(DoTeardown)
  ->RangeMultiplier(4)
  ->Ranges({{64 * KiB, 512 * MiB}, {2, 8}, {1, 4}})
  ->Unit(benchmark::kMillisecond);
```

Benchmark code lives in anonymous `namespace { }` blocks with `using namespace cucascade;` and `using namespace cucascade::memory;` inside the namespace.

## Test Tags

Tags follow `[category]` Catch2 syntax. Established tags in the codebase:

| Tag | Meaning |
|-----|---------|
| `[data_batch]` | data_batch state machine tests |
| `[memory_space]` | Memory space and reservation tests |
| `[disk]` | All disk I/O tests |
| `[io]` | I/O backend factory tests |
| `[kvikio]` | kvikIO-specific tests |
| `[gds]` | GDS-specific tests |
| `[gpu-converter]` | GPU <-> disk converter tests |
| `[format]` | Disk file format tests |
| `[numeric]` | Numeric column type tests |
| `[timestamp]` | Timestamp column tests |
| `[string]` | String column tests |
| `[list]` | List column tests |
| `[struct]` | Struct column tests |
| `[nested]` | Nested type tests |
| `[null]` | Null mask tests |
| `[dictionary]` | Dictionary column tests |
| `[sliced]` | Sliced column tests |
| `[backend]` | Backend selection tests |
| `[threading]` | Multi-threaded tests |
| `[gpu]` | GPU-specific tests requiring CUDA |
| `[.multi-device]` | Hidden by default; requires multiple GPUs |
| `[tracking]` | Memory tracking/peak tests |
| `[.overflow_policy]` | Hidden overflow policy tests |

## Test Types

**Unit Tests:**
- Test a single class/function in isolation with mock dependencies
- Examples: `test/data/test_data_batch.cpp`, `test/data/test_representation_converter.cpp`
- Use `mock_data_representation` and `make_mock_memory_space()` to avoid real allocators

**Integration/Round-Trip Tests:**
- Test full conversion pipelines with real I/O
- Examples: `test/data/test_gpu_disk_converters.cpp`, `test/data/test_disk_host_converters.cpp`
- Use `test::expect_cudf_tables_equal_on_stream()` for data correctness assertions
- Each column type (numeric, string, list, struct, dictionary, sliced) gets its own `TEST_CASE`

**Threading Tests:**
- Found in `test/data/test_data_batch.cpp` and `test/data/test_data_repository.cpp`
- Use `std::thread`, `std::atomic`, `std::mutex` directly
- Device sync via `device_sync_listener` after each test case

**CUDA Kernel Tests:**
- `test/memory/test_gpu_kernels.cu` tests GPU memory access patterns
- Kernel code in `.cu`, declarations in `.cuh`

---

*Testing analysis: 2026-04-02*
