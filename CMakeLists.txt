# =============================================================================
# Copyright 2025, cuCascade Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# Set project name and languages
project(
  cuCascade
  VERSION 0.1.0
  LANGUAGES C CXX CUDA
  DESCRIPTION "GPU memory management and data representation library")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Option to build tests
option(BUILD_TESTS "Build the test suite" ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find RMM from libcudf installation
find_package(rmm REQUIRED CONFIG)

# Find cudf for data representation support
find_package(cudf REQUIRED CONFIG)

# Find fmt for string formatting
find_package(fmt REQUIRED CONFIG)

# Define the main library
add_library(cucascade STATIC)

# Add source directories
add_subdirectory(src/memory)
add_subdirectory(src/data)

# Set include directories for the library
target_include_directories(
  cucascade PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                   $<INSTALL_INTERFACE:include>)

# Link dependencies
target_link_libraries(
  cucascade
  PUBLIC rmm::rmm
         cudf::cudf
         CUDA::cudart
         CUDA::nvml
         Threads::Threads
         fmt::fmt
         numa)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
endif()

# Build tests if enabled
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
