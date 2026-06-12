// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_DIVIDE_H_
#define DRACO_CORE_DIVIDE_H_
// An implementation of the divide by multiply algorithm
// https://gmplib.org/~tege/divcnst-pldi94.pdf
// This file is based off libvpx's divide.h.

#include <stdint.h>

#include <climits>

namespace draco {

struct fastdiv_elem {
  unsigned mult;
  unsigned shift;
};

extern const struct fastdiv_elem vp10_fastdiv_tab[256];

static inline unsigned fastdiv(unsigned x, int y) {
  unsigned t =
      ((uint64_t)x * vp10_fastdiv_tab[y].mult) >> (sizeof(x) * CHAR_BIT);
  return (t + x) >> vp10_fastdiv_tab[y].shift;
}

}  // namespace draco

#endif  // DRACO_CORE_DIVIDE_H_
