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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_DECODING_TRANSFORM_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_DECODING_TRANSFORM_H_

#include <cmath>

#include "draco/compression/attributes/normal_compression_utils.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_canonicalized_transform_base.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/macros.h"
#include "draco/core/math_utils.h"
#include "draco/core/vector_d.h"

namespace draco {

// Class for converting correction values transformed by the canonicalized
// normal octahedron transform back to the original values. See the
// corresponding encoder for more details.
template <typename DataTypeT>
class PredictionSchemeNormalOctahedronCanonicalizedDecodingTransform
    : public PredictionSchemeNormalOctahedronCanonicalizedTransformBase<
          DataTypeT> {
 public:
  typedef VectorD<DataTypeT, 2> Point2;
  typedef DataTypeT CorrType;
  typedef DataTypeT DataType;

  PredictionSchemeNormalOctahedronCanonicalizedDecodingTransform() {}

  // Dummy to fulfill concept.
  void Init(int num_components) {}

  bool DecodeTransformData(DecoderBuffer *buffer) {
    DataTypeT max_quantized_value, center_value;
    if (!buffer->Decode(&max_quantized_value)) {
      return false;
    }
    if (!buffer->Decode(&center_value)) {
      return false;
    }
    (void)center_value;
    if (!this->set_max_quantized_value(max_quantized_value)) {
      return false;
    }
    // Account for reading wrong values, e.g., due to fuzzing.
    if (this->quantization_bits() < 2) {
      return false;
    }
    if (this->quantization_bits() > 30) {
      return false;
    }
    return true;
  }

  inline void ComputeOriginalValue(const DataType *pred_vals,
                                   const CorrType *corr_vals,
                                   DataType *out_orig_vals) const {
    DRACO_DCHECK_LE(pred_vals[0], 2 * this->center_value());
    DRACO_DCHECK_LE(pred_vals[1], 2 * this->center_value());
    DRACO_DCHECK_LE(corr_vals[0], 2 * this->center_value());
    DRACO_DCHECK_LE(corr_vals[1], 2 * this->center_value());

    DRACO_DCHECK_LE(0, pred_vals[0]);
    DRACO_DCHECK_LE(0, pred_vals[1]);
    DRACO_DCHECK_LE(0, corr_vals[0]);
    DRACO_DCHECK_LE(0, corr_vals[1]);

    const Point2 pred = Point2(pred_vals[0], pred_vals[1]);
    const Point2 corr = Point2(corr_vals[0], corr_vals[1]);
    const Point2 orig = ComputeOriginalValue(pred, corr);

    out_orig_vals[0] = orig[0];
    out_orig_vals[1] = orig[1];
  }

 private:
  Point2 ComputeOriginalValue(Point2 pred, Point2 corr) const {
    const Point2 t(this->center_value(), this->center_value());
    pred = pred - t;
    const bool pred_is_in_diamond = this->IsInDiamond(pred[0], pred[1]);
    if (!pred_is_in_diamond) {
      this->InvertDiamond(&pred[0], &pred[1]);
    }
    const bool pred_is_in_bottom_left = this->IsInBottomLeft(pred);
    const int32_t rotation_count = this->GetRotationCount(pred);
    if (!pred_is_in_bottom_left) {
      pred = this->RotatePoint(pred, rotation_count);
    }
    Point2 orig(this->ModMax(AddAsUnsigned(pred[0], corr[0])),
                this->ModMax(AddAsUnsigned(pred[1], corr[1])));
    if (!pred_is_in_bottom_left) {
      const int32_t reverse_rotation_count = (4 - rotation_count) % 4;
      orig = this->RotatePoint(orig, reverse_rotation_count);
    }
    if (!pred_is_in_diamond) {
      this->InvertDiamond(&orig[0], &orig[1]);
    }
    orig = orig + t;
    return orig;
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_CANONICALIZED_DECODING_TRANSFORM_H_
