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
// A set of classes for quantizing and dequantizing of floating point values
// into integers.
// The quantization works on all floating point numbers within (-range, +range)
// interval producing integers in range
// (-max_quantized_value, +max_quantized_value).

#ifndef DRACO_CORE_QUANTIZATION_UTILS_H_
#define DRACO_CORE_QUANTIZATION_UTILS_H_

#include <stdint.h>

#include <cmath>

#include "draco/core/macros.h"

namespace draco {

// Class for quantizing single precision floating point values. The values
// should be centered around zero and be within interval (-range, +range), where
// the range is specified in the Init() method. Alternatively, the quantization
// can be defined by |delta| that specifies the distance between two quantized
// values. Note that the quantizer always snaps the values to the nearest
// integer value. E.g. for |delta| == 1.f, values -0.4f and 0.4f would be
// both quantized to 0 while value 0.6f would be quantized to 1. If a value
// lies exactly between two quantized states, it is always rounded up. E.g.,
// for |delta| == 1.f, value -0.5f would be quantized to 0 while 0.5f would be
// quantized to 1.
class Quantizer {
 public:
  Quantizer();
  void Init(float range, int32_t max_quantized_value);
  void Init(float delta);
  inline int32_t QuantizeFloat(float val) const {
    val *= inverse_delta_;
    return static_cast<int32_t>(floor(val + 0.5f));
  }
  inline int32_t operator()(float val) const { return QuantizeFloat(val); }

 private:
  float inverse_delta_;
};

// Class for dequantizing values that were previously quantized using the
// Quantizer class.
class Dequantizer {
 public:
  Dequantizer();

  // Initializes the dequantizer. Both parameters must correspond to the values
  // provided to the initializer of the Quantizer class.
  // Returns false when the initialization fails.
  bool Init(float range, int32_t max_quantized_value);

  // Initializes the dequantizer using the |delta| between two quantized values.
  bool Init(float delta);

  inline float DequantizeFloat(int32_t val) const {
    return static_cast<float>(val) * delta_;
  }
  inline float operator()(int32_t val) const { return DequantizeFloat(val); }

 private:
  float delta_;
};

}  // namespace draco

#endif  // DRACO_CORE_QUANTIZATION_UTILS_H_
