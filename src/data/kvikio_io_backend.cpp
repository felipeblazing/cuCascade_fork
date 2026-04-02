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

#include <kvikio/file_handle.hpp>

#include <cstddef>
#include <future>
#include <string>

namespace cucascade {
namespace {

class kvikio_io_backend : public idisk_io_backend {
 public:
  void write_device(const std::string& path,
                    const void* dev_ptr,
                    std::size_t size,
                    std::size_t file_offset,
                    [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    // TODO: Use stream-ordered async API (read_async/write_async) for proper stream ordering
    kvikio::FileHandle fh(path, "w");
    auto bytes_written = fh.write(dev_ptr, size, file_offset, 0);
    if (bytes_written != size) {
      CUCASCADE_FAIL("kvikio write_device: short write (" + std::to_string(bytes_written) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }

  void read_device(const std::string& path,
                   void* dev_ptr,
                   std::size_t size,
                   std::size_t file_offset,
                   [[maybe_unused]] rmm::cuda_stream_view stream) override
  {
    // TODO: Use stream-ordered async API (read_async/write_async) for proper stream ordering
    kvikio::FileHandle fh(path, "r");
    auto bytes_read = fh.read(dev_ptr, size, file_offset, 0);
    if (bytes_read != size) {
      CUCASCADE_FAIL("kvikio read_device: short read (" + std::to_string(bytes_read) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }

  void write_host(const std::string& path,
                  const void* host_ptr,
                  std::size_t size,
                  std::size_t file_offset) override
  {
    kvikio::FileHandle fh(path, "w");
    auto fut           = fh.pwrite(host_ptr, size, file_offset);
    auto bytes_written = fut.get();
    if (bytes_written != size) {
      CUCASCADE_FAIL("kvikio write_host: short write (" + std::to_string(bytes_written) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }

  void read_host(const std::string& path,
                 void* host_ptr,
                 std::size_t size,
                 std::size_t file_offset) override
  {
    kvikio::FileHandle fh(path, "r");
    auto fut        = fh.pread(host_ptr, size, file_offset);
    auto bytes_read = fut.get();
    if (bytes_read != size) {
      CUCASCADE_FAIL("kvikio read_host: short read (" + std::to_string(bytes_read) + " of " +
                     std::to_string(size) + " bytes)");
    }
  }
};

}  // namespace

std::unique_ptr<idisk_io_backend> make_io_backend(io_backend_type type)
{
  switch (type) {
    case io_backend_type::KVIKIO: return std::make_unique<kvikio_io_backend>();
    case io_backend_type::GDS: return make_gds_io_backend();
    case io_backend_type::PIPELINE: return make_pipeline_io_backend();
    default: CUCASCADE_FAIL("Unknown io_backend_type");
  }
}

}  // namespace cucascade
