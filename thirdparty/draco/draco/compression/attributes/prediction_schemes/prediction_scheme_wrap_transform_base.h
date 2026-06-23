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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_WRAP_TRANSFORM_BASE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_WRAP_TRANSFORM_BASE_H_

#include <limits>
#include <vector>

#include "draco/compression/config/compression_shared.h"
#include "draco/core/macros.h"

namespace draco {

// PredictionSchemeWrapTransform uses the min and max bounds of the original
// data to wrap stored correction values around these bounds centered at 0,
// i.e., when the range of the original values O is between <MIN, MAX> and
// N = MAX-MIN, we can then store any correction X = O - P, as:
//        X + N,   if X < -N / 2
//        X - N,   if X > N / 2
//        X        otherwise
// To unwrap this value, the decoder then simply checks whether the final
// corrected value F = P + X is out of the bounds of the input values.
// All out of bounds values are unwrapped using
//        F + N,   if F < MIN
//        F - N,   if F > MAX
// This wrapping can reduce the number of unique values, which translates to a
// better entropy of the stored values and better compression rates.
template <typename DataTypeT>
class PredictionSchemeWrapTransformBase {
 public:
  PredictionSchemeWrapTransformBase()
      : num_components_(0),
        min_value_(0),
        max_value_(0),
        max_dif_(0),
        max_correction_(0),
        min_correction_(0) {}

  static constexpr PredictionSchemeTransformType GetType() {
    return PREDICTION_TRANSFORM_WRAP;
  }

  void Init(int num_components) {
    num_components_ = num_components;
    clamped_value_.resize(num_components);
  }

  bool AreCorrectionsPositive() const { return false; }

  inline const DataTypeT *ClampPredictedValue(
      const DataTypeT *predicted_val) const {
    for (int i = 0; i < this->num_components(); ++i) {
      if (predicted_val[i] > max_value_) {
        clamped_value_[i] = max_value_;
      } else if (predicted_val[i] < min_value_) {
        clamped_value_[i] = min_value_;
      } else {
        clamped_value_[i] = predicted_val[i];
      }
    }
    return clamped_value_.data();
  }

  // TODO(b/199760123): Consider refactoring to avoid this dummy.
  int quantization_bits() const {
    DRACO_DCHECK(false);
    return -1;
  }

 protected:
  bool InitCorrectionBounds() {
    const int64_t dif =
        static_cast<int64_t>(max_value_) - static_cast<int64_t>(min_value_);
    if (dif < 0 || dif >= std::numeric_limits<DataTypeT>::max()) {
      return false;
    }
    max_dif_ = 1 + static_cast<DataTypeT>(dif);
    max_correction_ = max_dif_ / 2;
    min_correction_ = -max_correction_;
    if ((max_dif_ & 1) == 0) {
      max_correction_ -= 1;
    }
    return true;
  }

  inline int num_components() const { return num_components_; }
  inline DataTypeT min_value() const { return min_value_; }
  inline void set_min_value(const DataTypeT &v) { min_value_ = v; }
  inline DataTypeT max_value() const { return max_value_; }
  inline void set_max_value(const DataTypeT &v) { max_value_ = v; }
  inline DataTypeT max_dif() const { return max_dif_; }
  inline DataTypeT min_correction() const { return min_correction_; }
  inline DataTypeT max_correction() const { return max_correction_; }

 private:
  int num_components_;
  DataTypeT min_value_;
  DataTypeT max_value_;
  DataTypeT max_dif_;
  DataTypeT max_correction_;
  DataTypeT min_correction_;
  // This is in fact just a tmp variable to avoid reallocation.
  mutable std::vector<DataTypeT> clamped_value_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_WRAP_TRANSFORM_BASE_H_
