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
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_DECODING_TRANSFORM_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_DECODING_TRANSFORM_H_

#include <cmath>

#include "draco/compression/attributes/normal_compression_utils.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_transform_base.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/macros.h"
#include "draco/core/vector_d.h"
#include "draco/draco_features.h"

namespace draco {

// Class for converting correction values transformed by the octahedral normal
// transform back to the original values. See the corresponding encoder for more
// details.
template <typename DataTypeT>
class PredictionSchemeNormalOctahedronDecodingTransform
    : public PredictionSchemeNormalOctahedronTransformBase<DataTypeT> {
 public:
  typedef VectorD<DataTypeT, 2> Point2;
  typedef DataTypeT CorrType;
  typedef DataTypeT DataType;

  PredictionSchemeNormalOctahedronDecodingTransform() {}

  // Dummy function to fulfill concept.
  void Init(int num_components) {}
  bool DecodeTransformData(DecoderBuffer *buffer) {
    DataTypeT max_quantized_value, center_value;
    if (!buffer->Decode(&max_quantized_value)) {
      return false;
    }
    if (buffer->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
      if (!buffer->Decode(&center_value)) {
        return false;
      }
    }
    (void)center_value;
    return this->set_max_quantized_value(max_quantized_value);
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
  Point2 ComputeOriginalValue(Point2 pred, const Point2 &corr) const {
    const Point2 t(this->center_value(), this->center_value());
    typedef typename std::make_unsigned<DataTypeT>::type UnsignedDataTypeT;
    typedef VectorD<UnsignedDataTypeT, 2> Point2u;

    // Perform the addition in unsigned type to avoid signed integer overflow.
    // Note that the result will be the same (for non-overflowing values).
    pred = Point2(Point2u(pred) - Point2u(t));

    const bool pred_is_in_diamond = this->IsInDiamond(pred[0], pred[1]);
    if (!pred_is_in_diamond) {
      this->InvertDiamond(&pred[0], &pred[1]);
    }

    // Perform the addition in unsigned type to avoid signed integer overflow.
    // Note that the result will be the same (for non-overflowing values).
    Point2 orig(Point2u(pred) + Point2u(corr));

    orig[0] = this->ModMax(orig[0]);
    orig[1] = this->ModMax(orig[1]);
    if (!pred_is_in_diamond) {
      this->InvertDiamond(&orig[0], &orig[1]);
    }

    // Perform the addition in unsigned type to avoid signed integer overflow.
    // Note that the result will be the same (for non-overflowing values).
    orig = Point2(Point2u(orig) + Point2u(t));
    return orig;
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_NORMAL_OCTAHEDRON_DECODING_TRANSFORM_H_
#endif
