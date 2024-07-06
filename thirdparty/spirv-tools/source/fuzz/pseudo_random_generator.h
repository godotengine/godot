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

#ifndef SOURCE_FUZZ_PSEUDO_RANDOM_GENERATOR_H_
#define SOURCE_FUZZ_PSEUDO_RANDOM_GENERATOR_H_

#include <random>

#include "source/fuzz/random_generator.h"

namespace spvtools {
namespace fuzz {

// Generates random data from a pseudo-random number generator.
class PseudoRandomGenerator : public RandomGenerator {
 public:
  explicit PseudoRandomGenerator(uint32_t seed);

  ~PseudoRandomGenerator() override;

  uint32_t RandomUint32(uint32_t bound) override;

  uint64_t RandomUint64(uint64_t bound) override;

  uint32_t RandomPercentage() override;

  bool RandomBool() override;

  double RandomDouble() override;

 private:
  std::mt19937 mt_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_PSEUDO_RANDOM_GENERATOR_H_
