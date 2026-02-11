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

#include "utils/test_memory_resources.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch.hpp>

#include <cstddef>
#include <memory>

using namespace cucascade::memory;

static constexpr std::size_t host_capacity = 512UL << 20;  // 512 MB
static constexpr double limit_ratio        = 0.75;

static std::unique_ptr<memory_reservation_manager> create_manager()
{
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(2ULL << 30);
  builder.set_gpu_memory_resource_factory(cucascade::test::make_shared_current_device_resource);
  builder.set_reservation_fraction_per_gpu(limit_ratio);
  builder.set_per_host_capacity(host_capacity);
  builder.use_host_per_gpu();
  builder.set_reservation_fraction_per_host(limit_ratio);
  return std::make_unique<memory_reservation_manager>(builder.build());
}

SCENARIO("Blocks returned to free list after deallocation",
         "[memory_space][host_reservation][block_reuse]")
{
  auto manager = create_manager();

  auto* host_space = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  auto* mr = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(mr != nullptr);

  auto const block_size        = mr->get_block_size();
  std::size_t alloc_bytes      = block_size * 4;  // allocate 4 blocks
  std::size_t reservation_size = alloc_bytes;

  GIVEN("a host reservation and an allocation of 4 blocks")
  {
    auto reservation =
      manager->request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
    REQUIRE(reservation != nullptr);

    auto free_before  = mr->get_free_blocks();
    auto total_before = mr->get_total_blocks();

    auto blocks = mr->allocate_multiple_blocks(alloc_bytes, reservation.get());
    REQUIRE(blocks != nullptr);
    REQUIRE(blocks->size() == 4);

    auto free_after_alloc = mr->get_free_blocks();
    REQUIRE(free_after_alloc == free_before - 4);

    WHEN("blocks are released")
    {
      blocks.reset();

      THEN("free block count is restored") { REQUIRE(mr->get_free_blocks() == free_before); }

      THEN("total block count is unchanged (no new pools allocated)")
      {
        REQUIRE(mr->get_total_blocks() == total_before);
      }
    }
  }
}

SCENARIO("Blocks are reused across multiple allocate/release cycles",
         "[memory_space][host_reservation][block_reuse]")
{
  auto manager = create_manager();

  auto* host_space = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  auto* mr = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(mr != nullptr);

  auto const block_size        = mr->get_block_size();
  std::size_t alloc_bytes      = block_size * 4;
  std::size_t reservation_size = alloc_bytes;

  GIVEN("repeated allocate/release cycles")
  {
    auto total_blocks_initial = mr->get_total_blocks();

    for (int i = 0; i < 10; ++i) {
      auto reservation =
        manager->request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
      REQUIRE(reservation != nullptr);

      auto blocks = mr->allocate_multiple_blocks(alloc_bytes, reservation.get());
      REQUIRE(blocks != nullptr);
      REQUIRE(blocks->size() == 4);

      // Release blocks, then reservation
      blocks.reset();
      reservation.reset();
    }

    THEN("pool did not grow — blocks were reused")
    {
      REQUIRE(mr->get_total_blocks() == total_blocks_initial);
    }

    THEN("all blocks are free again") { REQUIRE(mr->get_free_blocks() == mr->get_total_blocks()); }
  }
}

SCENARIO("Freed blocks satisfy subsequent allocations without pool growth",
         "[memory_space][host_reservation][block_reuse]")
{
  auto manager = create_manager();

  auto* host_space = manager->get_memory_space(Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  auto* mr = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(mr != nullptr);

  auto const block_size        = mr->get_block_size();
  std::size_t alloc_bytes      = block_size * 4;
  std::size_t reservation_size = alloc_bytes;

  GIVEN("an allocation that is freed")
  {
    auto total_initial = mr->get_total_blocks();

    {
      auto res =
        manager->request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
      REQUIRE(res != nullptr);
      auto blocks = mr->allocate_multiple_blocks(alloc_bytes, res.get());
      REQUIRE(blocks != nullptr);
      // blocks and reservation destroyed here — blocks returned to free list
    }

    REQUIRE(mr->get_total_blocks() == total_initial);

    WHEN("the same amount is allocated again")
    {
      auto res =
        manager->request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
      REQUIRE(res != nullptr);
      auto blocks = mr->allocate_multiple_blocks(alloc_bytes, res.get());
      REQUIRE(blocks != nullptr);

      THEN("no new pool was allocated — freed blocks were reused")
      {
        REQUIRE(mr->get_total_blocks() == total_initial);
      }
    }
  }
}
