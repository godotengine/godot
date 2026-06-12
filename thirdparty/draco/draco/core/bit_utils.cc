// Copyright 2017 The Draco Authors.
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
#include "draco/core/bit_utils.h"

namespace draco {

void ConvertSignedIntsToSymbols(const int32_t *in, int in_values,
                                uint32_t *out) {
  // Convert the quantized values into a format more suitable for entropy
  // encoding.
  // Put the sign bit into LSB pos and shift the rest one bit left.
  for (int i = 0; i < in_values; ++i) {
    out[i] = ConvertSignedIntToSymbol(in[i]);
  }
}

void ConvertSymbolsToSignedInts(const uint32_t *in, int in_values,
                                int32_t *out) {
  for (int i = 0; i < in_values; ++i) {
    out[i] = ConvertSymbolToSignedInt(in[i]);
  }
}

}  // namespace draco
