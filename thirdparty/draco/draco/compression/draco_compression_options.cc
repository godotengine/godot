// Copyright 2022 The Draco Authors.
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
#include "draco/compression/draco_compression_options.h"

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

SpatialQuantizationOptions::SpatialQuantizationOptions(int quantization_bits) {
  SetQuantizationBits(quantization_bits);
}

void SpatialQuantizationOptions::SetQuantizationBits(int quantization_bits) {
  mode_ = LOCAL_QUANTIZATION_BITS;
  quantization_bits_ = quantization_bits;
}

bool SpatialQuantizationOptions::AreQuantizationBitsDefined() const {
  return mode_ == LOCAL_QUANTIZATION_BITS;
}

SpatialQuantizationOptions &SpatialQuantizationOptions::SetGrid(float spacing) {
  mode_ = GLOBAL_GRID;
  spacing_ = spacing;
  return *this;
}

bool SpatialQuantizationOptions::operator==(
    const SpatialQuantizationOptions &other) const {
  if (mode_ != other.mode_) {
    return false;
  }
  if (mode_ == LOCAL_QUANTIZATION_BITS) {
    if (quantization_bits_ != other.quantization_bits_) {
      return false;
    }
  } else if (mode_ == GLOBAL_GRID) {
    if (spacing_ != other.spacing_) {
      return false;
    }
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
