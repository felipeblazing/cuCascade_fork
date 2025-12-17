# cuCascade

A high-performance GPU memory management library for data-intensive applications requiring intelligent tiered memory allocation across GPU, host, and disk storage.

# Overview

cuCascade provides a robust memory management and data representation framework extracted from the Sirius project. It enables efficient allocation and reservation across multiple memory tiers while automatically discovering hardware topology for optimal data placement.

**Key Features:**
- **Tiered Memory Management**: Seamlessly manage GPU (fastest), pinned host (medium), and disk (largest capacity) memory tiers
- **Hardware Topology Discovery**: Automatic detection of NUMA regions and GPU-CPU affinity for optimal memory placement
- **Thread-Safe Operations**: Lock-free memory reservation and data batch management
- **cuDF Integration**: Native support for GPU DataFrames with batch processing capabilities

# Getting Started

```bash
# Option A: Using Pixi (recommended)
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build

# Option B: Using CMake directly
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
cmake --preset release
cmake --build build/release

# Verify installation
cd build/release && ctest --output-on-failure
```

# Requirements

- **OS/Arch**: Linux (x86_64, aarch64)
- **Compiler**: C++20 compatible compiler
- **Build Tools**: CMake 4.1+, Ninja
- **GPU/Drivers**: CUDA 13+, compatible NVIDIA driver
- **Dependencies**: libcudf 25.10+

# Usage

```cpp
#include <memory/memory_reservation_manager.hpp>
#include <memory/topology_discovery.hpp>

// Discover hardware topology
auto topology = cucascade::discover_topology();

// Create memory reservation manager
auto manager = cucascade::MemoryReservationManager(topology);

// Reserve GPU memory
auto gpu_reservation = manager.reserve(cucascade::MemoryTier::GPU, size_bytes);

// Reserve host memory with NUMA awareness
auto host_reservation = manager.reserve(cucascade::MemoryTier::HOST, size_bytes);
```

- More examples: See `test/` directory for comprehensive usage examples

# Contribution Guidelines

- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart:

```bash
git clone https://github.com/nvidia/cuCascade.git
cd cuCascade
pixi install
pixi run build
pixi run test
```

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) for code quality checks including C++/CUDA formatting (clang-format), CMake linting, spell checking, and more.

```bash
# Run all checks manually
pixi run lint

# Install hooks to run automatically on every commit
pixi run lint-install

# Update hook versions
pre-commit autoupdate
```

## Security

- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.
- Report vulnerabilities via: https://www.nvidia.com/object/submit-security-vulnerability.html

## Support

- Level: Experimental
- How to get help: GitHub Issues
- For NVIDIA product security concerns: https://www.nvidia.com/en-us/security

# Project Structure

```
cuCascade/
├── include/
│   ├── data/           # Data representation headers
│   ├── memory/         # Memory management headers
│   ├── log/            # Logging utilities
│   └── helper/         # Common helper headers
├── src/
│   ├── data/           # Data representation implementation
│   └── memory/         # Memory management implementation
├── test/               # Unit tests
├── CMakeLists.txt      # Main CMake configuration
├── CMakePresets.json   # CMake presets for build configurations
└── pixi.toml           # Pixi dependency management
```

# References

- [RAPIDS cuDF](https://github.com/rapidsai/cudf) - GPU DataFrame library
- [Pixi](https://pixi.sh/) - Package management tool

# License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
