# ─────────────────────────────────────────────────────────────────────────────
# Synapse-AI — top-level build
#
# Two products, one clean boundary:
#   • synapse_engine   — the from-scratch neural-net engine. SYCL lives ONLY here,
#                        compiled through AdaptiveCpp. Its PUBLIC headers are pure
#                        C++/STL so the GUI can consume them without SYCL.
#   • synapse_dashboard — the Qt6/QML learning dashboard. Compiled by the normal
#                        Clang toolchain; links the engine as a library.
#   • examples/        — the original SYCL sandbox files, kept as reference.
#
# Build:  cmake --preset default  &&  cmake --build --preset default
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.25)
project(synapse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)          # matches AdaptiveCpp's SSCP flow
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default AdaptiveCpp target: 'generic' = single-pass JIT that runs on whatever
# device is present (GPU if available, CPU/OpenMP otherwise). This is what makes
# the engine hardware-agnostic. Overridable via -DACPP_TARGETS=... or the env var.
if(NOT DEFINED ACPP_TARGETS AND NOT DEFINED ENV{ACPP_TARGETS})
  set(ACPP_TARGETS "generic" CACHE STRING "AdaptiveCpp compilation targets")
endif()

# Found once at the top level so add_sycl_to_target() is visible to every
# subdirectory that needs it (engine + examples).
find_package(AdaptiveCpp REQUIRED)

enable_testing()

add_subdirectory(engine)
add_subdirectory(dashboard)
add_subdirectory(examples)
add_subdirectory(tests)
