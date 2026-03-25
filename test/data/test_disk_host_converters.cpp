/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "utils/cudf_test_utils.hpp"
#include "utils/mock_test_utils.hpp"

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/host_table.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch2/catch.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using namespace cucascade;

namespace {

/// Round-trip test helper: GPU -> host_data -> disk -> host_data -> GPU, compare tables.
void round_trip_test(std::unique_ptr<cudf::table> original_table)
{
  rmm::cuda_stream stream;
  auto gpu_space  = test::make_mock_memory_space(memory::Tier::GPU, 0);
  auto host_space = test::make_mock_memory_space(memory::Tier::HOST, 0);
  auto disk_space = test::make_mock_memory_space(memory::Tier::DISK, 0);

  representation_converter_registry registry;
  register_builtin_converters(registry);

  // Create GPU representation from the original table
  auto gpu_rep =
    std::make_unique<gpu_table_representation>(std::move(original_table), *gpu_space);

  // GPU -> host_data
  auto host_rep =
    registry.convert<host_data_representation>(*gpu_rep, host_space.get(), stream.view());

  // host_data -> disk
  auto disk_rep =
    registry.convert<disk_data_representation>(*host_rep, disk_space.get(), stream.view());

  // disk -> host_data
  auto host_rep2 =
    registry.convert<host_data_representation>(*disk_rep, host_space.get(), stream.view());

  // host_data -> GPU
  auto gpu_rep2 =
    registry.convert<gpu_table_representation>(*host_rep2, gpu_space.get(), stream.view());

  // Compare original (from gpu_rep) with round-tripped
  test::expect_cudf_tables_equal_on_stream(
    gpu_rep->get_table(), gpu_rep2->get_table(), stream.view());
}

/// Create a simple column of the given type with uninitialized data.
/// Returns a single-column cudf::table.
std::unique_ptr<cudf::table> make_typed_table(cudf::type_id type, cudf::size_type num_rows)
{
  rmm::cuda_stream stream;
  auto const dtype = cudf::data_type{type};
  std::vector<std::unique_ptr<cudf::column>> cols;
  if (cudf::is_numeric(dtype)) {
    // cudf::is_numeric includes BOOL8
    auto col = cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::UNALLOCATED,
                                         stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_timestamp(dtype)) {
    auto col = cudf::make_timestamp_column(dtype, num_rows, cudf::mask_state::UNALLOCATED,
                                           stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_duration(dtype)) {
    auto col = cudf::make_duration_column(dtype, num_rows, cudf::mask_state::UNALLOCATED,
                                          stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_fixed_point(dtype)) {
    auto const fp_dtype = cudf::data_type{type, -2};  // scale = -2
    auto col = cudf::make_fixed_point_column(fp_dtype, num_rows, cudf::mask_state::UNALLOCATED,
                                             stream.view());
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Create a table with ALL_NULL mask.
std::unique_ptr<cudf::table> make_typed_table_with_nulls(cudf::type_id type,
                                                         cudf::size_type num_rows)
{
  rmm::cuda_stream stream;
  auto const dtype = cudf::data_type{type};
  std::vector<std::unique_ptr<cudf::column>> cols;
  if (cudf::is_numeric(dtype)) {
    auto col =
      cudf::make_numeric_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_timestamp(dtype)) {
    auto col =
      cudf::make_timestamp_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_duration(dtype)) {
    auto col =
      cudf::make_duration_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream.view());
    cols.push_back(std::move(col));
  } else if (cudf::is_fixed_point(dtype)) {
    auto const fp_dtype = cudf::data_type{type, -2};
    auto col =
      cudf::make_fixed_point_column(fp_dtype, num_rows, cudf::mask_state::ALL_NULL, stream.view());
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

// =============================================================================
// Numeric type round-trip tests (TEST-01)
// =============================================================================

TEST_CASE("host_data disk round-trip numeric types", "[disk][converter][numeric]")
{
  auto type = GENERATE(cudf::type_id::INT8,
                       cudf::type_id::INT16,
                       cudf::type_id::INT32,
                       cudf::type_id::INT64,
                       cudf::type_id::UINT8,
                       cudf::type_id::UINT16,
                       cudf::type_id::UINT32,
                       cudf::type_id::UINT64,
                       cudf::type_id::FLOAT32,
                       cudf::type_id::FLOAT64);

  SECTION("100 rows") { round_trip_test(make_typed_table(type, 100)); }
}

// =============================================================================
// BOOL8 round-trip tests (TEST-02)
// =============================================================================

TEST_CASE("host_data disk round-trip bool type", "[disk][converter][bool]")
{
  round_trip_test(make_typed_table(cudf::type_id::BOOL8, 100));
}

// =============================================================================
// Timestamp round-trip tests (TEST-02)
// =============================================================================

TEST_CASE("host_data disk round-trip timestamp types", "[disk][converter][timestamp]")
{
  auto type = GENERATE(cudf::type_id::TIMESTAMP_DAYS,
                       cudf::type_id::TIMESTAMP_SECONDS,
                       cudf::type_id::TIMESTAMP_MILLISECONDS,
                       cudf::type_id::TIMESTAMP_MICROSECONDS,
                       cudf::type_id::TIMESTAMP_NANOSECONDS);

  round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Duration round-trip tests (TEST-02)
// =============================================================================

TEST_CASE("host_data disk round-trip duration types", "[disk][converter][duration]")
{
  auto type = GENERATE(cudf::type_id::DURATION_DAYS,
                       cudf::type_id::DURATION_SECONDS,
                       cudf::type_id::DURATION_MILLISECONDS,
                       cudf::type_id::DURATION_MICROSECONDS,
                       cudf::type_id::DURATION_NANOSECONDS);

  round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Decimal round-trip tests (TEST-03)
// =============================================================================

TEST_CASE("host_data disk round-trip decimal types", "[disk][converter][decimal]")
{
  auto type = GENERATE(
    cudf::type_id::DECIMAL32, cudf::type_id::DECIMAL64, cudf::type_id::DECIMAL128);

  round_trip_test(make_typed_table(type, 100));
}

// =============================================================================
// Null mask round-trip tests (TEST-08)
// =============================================================================

TEST_CASE("host_data disk round-trip all-null column", "[disk][converter][null]")
{
  round_trip_test(make_typed_table_with_nulls(cudf::type_id::INT32, 100));
}

TEST_CASE("host_data disk round-trip no-null column", "[disk][converter][null]")
{
  round_trip_test(make_typed_table(cudf::type_id::INT32, 100));
}

TEST_CASE("host_data disk round-trip mixed-null column", "[disk][converter][null]")
{
  // Create a column with alternating nulls
  rmm::cuda_stream stream;
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       100,
                                       cudf::mask_state::ALL_VALID,
                                       stream.view());

  // Set every other bit to null using cudf utilities
  auto const num_rows = 100;
  std::vector<uint8_t> host_mask(cudf::bitmask_allocation_size_bytes(num_rows), 0xFF);
  // Clear every other bit (set elements 0, 2, 4, ... to null)
  for (int i = 0; i < num_rows; i += 2) {
    auto byte_idx = static_cast<std::size_t>(i / 8);
    auto bit_idx  = static_cast<unsigned>(i % 8);
    host_mask[byte_idx] &= static_cast<uint8_t>(~(1u << bit_idx));
  }
  rmm::device_buffer dev_mask(host_mask.data(), host_mask.size(), stream.view());
  col->set_null_mask(std::move(dev_mask), 50);  // 50 nulls out of 100

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  round_trip_test(std::make_unique<cudf::table>(std::move(cols)));
}
