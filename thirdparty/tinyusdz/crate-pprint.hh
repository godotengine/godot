// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
#pragma once

#include <iostream>

#include "crate-format.hh"

namespace std {

std::ostream &operator<<(std::ostream &os, const tinyusdz::crate::Index &i);

} // namespace std

namespace tinyusdz {

} // namespace tinyusdz
