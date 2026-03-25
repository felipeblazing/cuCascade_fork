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

#include "io_backend_internal.hpp"

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/error.hpp>

#include <cstddef>
#include <string>

namespace cucascade {
namespace {

// =============================================================================
// registered_buffer -- RAII wrapper for cuFile buffer registration
// =============================================================================

class registered_buffer {
 public:
  registered_buffer(void* dev_ptr, std::size_t size) : _ptr(dev_ptr), _size(size), _registered(false)
  {
    // cufile.h is not available in this environment; buffer registration is a no-op.
    // When cuFile becomes available, this constructor will call cuFileBufRegister.
    (void)_ptr;
    (void)_size;
  }

  ~registered_buffer() noexcept
  {
    // When cuFile is available: if (_registered) { cuFileBufDeregister(_ptr); }
  }

  registered_buffer(const registered_buffer&)            = delete;
  registered_buffer& operator=(const registered_buffer&) = delete;
  registered_buffer(registered_buffer&&)                 = delete;
  registered_buffer& operator=(registered_buffer&&)      = delete;

  [[nodiscard]] bool is_registered() const noexcept { return _registered; }

 private:
  void* _ptr;
  std::size_t _size;
  bool _registered;
};

// =============================================================================
// gds_io_backend -- stub implementation (Phase 3/4 work)
// =============================================================================

class gds_io_backend : public idisk_io_backend {
 public:
  void write_device([[maybe_unused]] const std::string& path,
                    [[maybe_unused]] const void* dev_ptr,
                    [[maybe_unused]] std::size_t size,
                    [[maybe_unused]] std::size_t file_offset,
                    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    CUCASCADE_FAIL("gds_io_backend not yet implemented");
  }

  void read_device([[maybe_unused]] const std::string& path,
                   [[maybe_unused]] void* dev_ptr,
                   [[maybe_unused]] std::size_t size,
                   [[maybe_unused]] std::size_t file_offset,
                   [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    CUCASCADE_FAIL("gds_io_backend not yet implemented");
  }

  void write_host([[maybe_unused]] const std::string& path,
                  [[maybe_unused]] const void* host_ptr,
                  [[maybe_unused]] std::size_t size,
                  [[maybe_unused]] std::size_t file_offset) override
  {
    CUCASCADE_FAIL("gds_io_backend not yet implemented");
  }

  void read_host([[maybe_unused]] const std::string& path,
                 [[maybe_unused]] void* host_ptr,
                 [[maybe_unused]] std::size_t size,
                 [[maybe_unused]] std::size_t file_offset) override
  {
    CUCASCADE_FAIL("gds_io_backend not yet implemented");
  }
};

}  // namespace

std::unique_ptr<idisk_io_backend> make_gds_io_backend()
{
  return std::make_unique<gds_io_backend>();
}

}  // namespace cucascade
