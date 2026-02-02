// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - Present, Light Transport Entertainment, Inc.
#include "performance.hh"

#include <chrono>

namespace tinyusdz {
namespace performance {

double now() {
  auto t = std::chrono::system_clock::now();

  // to milliseconds.
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()); 

  return double(ms.count());
}

} // namespace performance
} // namespace tinyusdz
