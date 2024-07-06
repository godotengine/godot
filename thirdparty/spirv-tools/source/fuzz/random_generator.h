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

#ifndef SOURCE_FUZZ_RANDOM_GENERATOR_H_
#define SOURCE_FUZZ_RANDOM_GENERATOR_H_

#include <stdint.h>

namespace spvtools {
namespace fuzz {

class RandomGenerator {
 public:
  RandomGenerator();

  virtual ~RandomGenerator();

  // Returns a value in the half-open interval [0, bound).
  virtual uint32_t RandomUint32(uint32_t bound) = 0;

  // Returns a value in the half-open interval [0, bound).
  virtual uint64_t RandomUint64(uint64_t bound) = 0;

  // Returns a value in the closed interval [0, 100].
  virtual uint32_t RandomPercentage() = 0;

  // Returns a boolean.
  virtual bool RandomBool() = 0;

  // Returns a double in the closed interval [0, 1]
  virtual double RandomDouble() = 0;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_RANDOM_GENERATOR_H_
