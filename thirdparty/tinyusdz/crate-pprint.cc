// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
#include "crate-pprint.hh"

#include <string>

namespace std {

std::ostream &operator<<(std::ostream &os, const tinyusdz::crate::Index &i) {
  os << std::to_string(i.value);
  return os;
}

} // namespace std

