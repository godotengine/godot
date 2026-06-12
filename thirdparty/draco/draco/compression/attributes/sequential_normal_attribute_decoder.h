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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_NORMAL_ATTRIBUTE_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_NORMAL_ATTRIBUTE_DECODER_H_

#include "draco/attributes/attribute_octahedron_transform.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_decoder_factory.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_canonicalized_decoding_transform.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_decoding_transform.h"
#include "draco/compression/attributes/sequential_integer_attribute_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// Decoder for attributes encoded with SequentialNormalAttributeEncoder.
class SequentialNormalAttributeDecoder
    : public SequentialIntegerAttributeDecoder {
 public:
  SequentialNormalAttributeDecoder();
  bool Init(PointCloudDecoder *decoder, int attribute_id) override;

 protected:
  int32_t GetNumValueComponents() const override {
    return 2;  // We quantize everything into two components.
  }
  bool DecodeIntegerValues(const std::vector<PointIndex> &point_ids,
                           DecoderBuffer *in_buffer) override;
  bool DecodeDataNeededByPortableTransform(
      const std::vector<PointIndex> &point_ids,
      DecoderBuffer *in_buffer) override;
  bool StoreValues(uint32_t num_points) override;

 private:
  AttributeOctahedronTransform octahedral_transform_;

  std::unique_ptr<PredictionSchemeTypedDecoderInterface<int32_t>>
  CreateIntPredictionScheme(
      PredictionSchemeMethod method,
      PredictionSchemeTransformType transform_type) override {
    switch (transform_type) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
      case PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON: {
        typedef PredictionSchemeNormalOctahedronDecodingTransform<int32_t>
            Transform;
        // At this point the decoder has not read the quantization bits,
        // which is why we must construct the transform by default.
        // See Transform.DecodeTransformData for more details.
        return CreatePredictionSchemeForDecoder<int32_t, Transform>(
            method, attribute_id(), decoder());
      }
#endif
      case PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON_CANONICALIZED: {
        typedef PredictionSchemeNormalOctahedronCanonicalizedDecodingTransform<
            int32_t>
            Transform;
        // At this point the decoder has not read the quantization bits,
        // which is why we must construct the transform by default.
        // See Transform.DecodeTransformData for more details.
        return CreatePredictionSchemeForDecoder<int32_t, Transform>(
            method, attribute_id(), decoder());
      }
      default:
        return nullptr;  // Currently, we support only octahedron transform and
                         // octahedron transform canonicalized.
    }
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_NORMAL_ATTRIBUTE_DECODER_H_
