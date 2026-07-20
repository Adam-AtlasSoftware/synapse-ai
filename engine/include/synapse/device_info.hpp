#pragma once
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Public engine header — PURE C++/STL. Deliberately contains NO `sycl/sycl.hpp`.
//
// This is the whole point of the engine/GUI boundary: the dashboard (compiled by
// the ordinary Clang toolchain, with no AdaptiveCpp involved) can include this and
// call into the engine, while every SYCL detail stays hidden inside the .cpp files
// that AdaptiveCpp compiles. If you ever see a SYCL type leak into a header like
// this one, the toolchain separation is broken.
// ─────────────────────────────────────────────────────────────────────────────
namespace synapse {

// Name of the SYCL device the engine selected (e.g. a GPU model, or an OpenMP
// host device when no accelerator is available).
std::string active_device_name();

// Runs a trivial vector-add kernel on the device and checks the result.
// Returns true on success. This proves that device code compiled by AdaptiveCpp
// actually executes when called across the static-library boundary.
bool device_self_test();

}  // namespace synapse
