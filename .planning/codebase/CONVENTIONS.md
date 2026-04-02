# Coding Conventions

**Analysis Date:** 2026-04-02

## Naming Patterns

**Files:**
- Headers: `snake_case.hpp` — never `.h` for project headers
- CUDA headers: `snake_case.cuh` (e.g., `test/memory/test_gpu_kernels.cuh`)
- Source: `snake_case.cpp` for C++, `snake_case.cu` for CUDA kernels
- Test files: `test_<module_name>.cpp` (e.g., `test/data/test_disk_io_backend.cpp`)
- Benchmark files: `benchmark_<module_name>.cpp` (e.g., `benchmark/benchmark_disk_converter.cpp`)

**Classes and Structs:**
- `snake_case` for all: `memory_space`, `data_batch`, `disk_data_representation`
- Interface classes prefixed with `i`: `idata_representation`, `idisk_io_backend`
- Config structs suffixed with `_config`: `gpu_memory_space_config`, `disk_memory_space_config`
- Hash structs suffixed with `_hash`: `converter_key_hash`
- RAII handles suffixed with `_handle`: `data_batch_processing_handle`
- Error category structs suffixed with `_category`: `memory_error_category`
- Exception: `Tier` enum uses PascalCase name, UPPER_CASE values: `Tier::GPU`, `Tier::HOST`, `Tier::DISK`
- Exception: `MemoryError` uses PascalCase name, UPPER_CASE values: `MemoryError::ALLOCATION_FAILED`

**Functions:**
- `snake_case`: `get_available_memory()`, `make_reservation_or_null()`
- Getters prefixed with `get_`: `get_tier()`, `get_device_id()`, `get_batch_id()`
- Boolean queries prefixed with `should_`, `has_`, or `is_`: `should_downgrade_memory()`
- Factory functions prefixed with `make_` or `create_`: `make_mock_memory_space()`, `create_simple_cudf_table()`
- Try-pattern methods prefixed with `try_to_`: `try_to_create_task()`, `try_to_lock_for_processing()`
- Blocking wait methods prefixed with `wait_to_`: `wait_to_create_task()`

**Variables:**
- Member variables prefixed with underscore: `_id`, `_capacity`, `_mutex`, `_disk_table`
- Local variables: `snake_case` — `gpu_device_0`, `reservation_size`
- Constants: `snake_case` — `expected_gpu_capacity`, `default_block_size`
- Compile-time constants: `constexpr` — `static constexpr std::size_t default_size{16};`
- Size literal constants use `ull` suffix and bit shifts: `2ull << 30` for 2 GB, `1UL << 20` for 1 MB
- Benchmark-local byte-unit constants: `constexpr uint64_t KiB = 1024ULL;`

**Type Aliases:**
- Function type aliases: PascalCase — `DeviceMemoryResourceFactoryFn`, `representation_converter_fn` (simpler aliases are `snake_case`)
- Enum classes: `snake_case` for names and values — `batch_state::idle`, `batch_state::in_transit`

**Macros:**
- All caps with `CUCASCADE_` prefix: `CUCASCADE_CUDA_TRY`, `CUCASCADE_FAIL`, `CUCASCADE_FUNC_RANGE`

## Code Style

**Formatting:**
- Tool: clang-format v20.1.4 (enforced by pre-commit hook via `.clang-format`)
- Column limit: 100 characters
- Indent width: 2 spaces (tabs never used)
- Standard: `c++20`
- Brace style: WebKit (`BreakBeforeBraces: WebKit`) — opening braces on same line for most constructs
- Pointer alignment: left — `void* ptr`, not `void *ptr`
- `AlignConsecutiveAssignments: true` — align consecutive assignment operators
- `AlignConsecutiveMacros: true` — align consecutive macro definitions
- `BinPackArguments: false` — all arguments on one line or each on its own line
- No trailing whitespace (enforced by pre-commit `trailing-whitespace` hook)
- Files must end with a newline (enforced by `end-of-file-fixer`)

**Linting:**
- cmake-format and cmake-lint for CMake files (line width 220, disabled code C0307)
- codespell for spell checking (ignore-words in `.codespell_words`)
- black for Python files
- clang-format for all C++/CUDA source
- `CUCASCADE_WARNINGS_AS_ERRORS=ON` by default — all warnings are errors

**Compiler Warnings:**
- `-Wall -Wextra -Wpedantic`
- `-Wcast-align -Wunused -Wconversion -Wsign-conversion`
- `-Wnull-dereference -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough`

## License Header

Every source file begins with an SPDX Apache 2.0 header block:

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */
```

Headers follow the license block immediately with `#pragma once` (never `#ifndef` guards).

## Include Organization

clang-format enforces `IncludeBlocks: Regroup` with `SortIncludes: true` and these priority groups (ascending priority = higher in file):

1. Quoted local includes (priority 1): `#include "utils/cudf_test_utils.hpp"`
2. Test/benchmark includes (priority 2): `#include <test/...>`
3. cuDF test includes (priority 3): `#include <cudf_test/...>`
4. cucascade includes (priority 4): `#include <cucascade/data/data_batch.hpp>`
5. cuDF includes (priority 5): `#include <cudf/table/table.hpp>`
6. Other RAPIDS includes (priority 6): `#include <kvikio/...>`, `#include <rmm/...>` (priority 7)
7. CCCL includes (priority 8): `#include <thrust/...>`
8. CUDA includes (priority 8): `#include <cuda_runtime_api.h>`
9. System includes with `.` (priority 9): `#include <filesystem>`
10. STL includes without `.` (priority 10): `#include <memory>`, `#include <vector>`

Example ordering from `test/data/test_gpu_disk_converters.cpp`:
```cpp
#include "utils/cudf_test_utils.hpp"     // quoted local
#include "utils/mock_test_utils.hpp"

#include <cucascade/data/disk_data_representation.hpp>  // cucascade
#include <cucascade/data/representation_converter.hpp>

#include <cudf/column/column_factories.hpp>  // cuDF

#include <rmm/cuda_stream.hpp>  // RMM

#include <catch2/catch.hpp>  // system with dot

#include <memory>   // STL
#include <vector>
```

## Namespace Usage

- Top-level: `cucascade`
- Subnamespaces: `cucascade::memory`, `cucascade::utils`
- Test namespace: `cucascade::test`
- No namespace indentation
- Close with comment: `}  // namespace cucascade` or `}  // namespace`
- Nested namespaces use traditional form: `namespace cucascade { namespace test {` (not C++17 `::`)
- Use anonymous `namespace { }` for file-local helpers in `.cpp` and test files
- `using namespace cucascade;` at file scope in test files is acceptable
- Specific test utilities imported explicitly: `using cucascade::test::create_simple_cudf_table;`
- C++17 nested namespace shorthand used in `test_memory_resources.hpp`: `namespace cucascade::test {`

## Error Handling

**Macros (defined in `include/cucascade/error.hpp`):**
- `CUCASCADE_CUDA_TRY(call)` — wraps CUDA runtime calls; throws `cucascade::cuda_error` on failure
- `CUCASCADE_CUDA_TRY(call, exception_type)` — two-arg form throws custom exception type
- `CUCASCADE_CUDA_TRY_ALLOC(call)` — throws `rmm::out_of_memory` for OOM, `rmm::bad_alloc` otherwise
- `CUCASCADE_CUDA_TRY_ALLOC(call, num_bytes)` — two-arg form includes requested bytes in message
- `CUCASCADE_ASSERT_CUDA_SUCCESS(call)` — assert-based check for noexcept/destructor contexts; in release builds the call executes but the error is discarded
- `CUCASCADE_FAIL(message)` — throws `cucascade::logic_error` with file/line context
- `CUCASCADE_FAIL(message, exception_type)` — throws custom exception type

**Exception types:**
- `cucascade::cuda_error` — inherits `std::runtime_error`; for CUDA runtime failures
- `cucascade::logic_error` — inherits `std::logic_error`; for programming errors
- `rmm::out_of_memory`, `rmm::bad_alloc` — for allocation failures
- `cucascade::memory::cucascade_out_of_memory` — extends `rmm::out_of_memory` with diagnostics

**Guidelines:**
- Use exceptions for errors; not error codes (except `MemoryError` which bridges to `std::error_code`)
- Use `std::runtime_error` or `std::invalid_argument` for general errors in constructors
- Destructors must be exception-safe: use `CUCASCADE_ASSERT_CUDA_SUCCESS` (not `CUCASCADE_CUDA_TRY`) and wrap filesystem operations in try/catch with silent discard (see `disk_data_representation::~disk_data_representation()` in `src/data/disk_data_representation.cpp`)
- Use `[[nodiscard]]` on getters and methods returning important values

## Documentation Style

All public API uses Javadoc-style `/** ... */` blocks:

```cpp
/**
 * @brief Get the tier of memory that this representation resides in
 *
 * @return Tier The memory tier
 */
memory::Tier get_current_tier() const { return _memory_space.get_tier(); }
```

- `@brief` — one-line summary
- `@param` — parameters
- `@return` — return values
- `@throws` — exception specifications
- `@tparam` — template parameters
- `@note` — important caveats
- `@code` / `@endcode` — inline code examples (used in `include/cucascade/data/representation_converter.hpp`)
- `@example` — usage examples
- Multi-line descriptions separated by a blank `*` line after `@brief`
- Trailing member docs use `///< description`
- Section separators: `//===----------------------------------------------------------------------===//`
- Inline comments use `//`
- `// clang-format off` / `// clang-format on` around macro blocks needing custom formatting

## RAII and Ownership Patterns

- `std::unique_ptr` for exclusive ownership (allocators, data representations, reservations, tables)
- `std::shared_ptr` for shared ownership (data batches, memory spaces in tests)
- `std::weak_ptr` for non-owning references (processing handles to batches)
- Explicitly delete copy/move when objects must be pinned: `= delete` (e.g., `representation_converter_registry`)
- Explicitly default move when move-only semantics are desired: `= default`
- `mutable std::mutex _mutex` for internal locking in thread-safe classes

## C++20 Features in Use

- `std::derived_from` concept in `requires` clauses: `requires std::derived_from<TargetType, idata_representation>`
- `static_assert` with `std::is_base_of_v` at template registration sites
- `[[nodiscard]]` attribute on getters
- `[[maybe_unused]]` on interface default parameters (e.g., `clone([[maybe_unused]] rmm::cuda_stream_view stream)`)
- Structured bindings: `auto [free_bytes, total_bytes] = rmm::available_device_memory();`
- `std::span` (in memory layer)
- Three-way comparison `<=>` (in `include/cucascade/memory/common.hpp`)

## NVTX Profiling

- Enabled via `CUCASCADE_NVTX` CMake option (default OFF)
- `CUCASCADE_FUNC_RANGE()` macro at function entry points for profiling
- Custom domain: `cucascade::libcucascade_domain` (in `include/cucascade/error.hpp`)
- Links `nvtx3::nvtx3-cpp` when enabled

---

*Convention analysis: 2026-04-02*
