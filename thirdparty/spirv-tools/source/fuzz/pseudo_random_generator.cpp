// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/fuzz/pseudo_random_generator.h"

#include <cassert>

namespace spvtools {
namespace fuzz {

PseudoRandomGenerator::PseudoRandomGenerator(uint32_t seed) : mt_(seed) {}

PseudoRandomGenerator::~PseudoRandomGenerator() = default;

uint32_t PseudoRandomGenerator::RandomUint32(uint32_t bound) {
  assert(bound > 0 && "Bound must be positive");
  return std::uniform_int_distribution<uint32_t>(0, bound - 1)(mt_);
}

uint64_t PseudoRandomGenerator::RandomUint64(uint64_t bound) {
  assert(bound > 0 && "Bound must be positive");
  return std::uniform_int_distribution<uint64_t>(0, bound - 1)(mt_);
}

bool PseudoRandomGenerator::RandomBool() {
  return static_cast<bool>(std::uniform_int_distribution<>(0, 1)(mt_));
}

uint32_t PseudoRandomGenerator::RandomPercentage() {
  // We use 101 because we want a result in the closed interval [0, 100], and
  // RandomUint32 is not inclusive of its bound.
  return RandomUint32(101);
}

double PseudoRandomGenerator::RandomDouble() {
  return std::uniform_real_distribution<double>(0.0, 1.0)(mt_);
}

}  // namespace fuzz
}  // namespace spvtools
