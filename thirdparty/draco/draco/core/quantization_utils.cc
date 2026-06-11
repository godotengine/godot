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
#include "draco/core/quantization_utils.h"

namespace draco {

Quantizer::Quantizer() : inverse_delta_(1.f) {}

void Quantizer::Init(float range, int32_t max_quantized_value) {
  inverse_delta_ = static_cast<float>(max_quantized_value) / range;
}

void Quantizer::Init(float delta) { inverse_delta_ = 1.f / delta; }

Dequantizer::Dequantizer() : delta_(1.f) {}

bool Dequantizer::Init(float range, int32_t max_quantized_value) {
  if (max_quantized_value <= 0) {
    return false;
  }
  delta_ = range / static_cast<float>(max_quantized_value);
  return true;
}

bool Dequantizer::Init(float delta) {
  delta_ = delta;
  return true;
}

}  // namespace draco
