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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_DECODER_H_

#include "draco/attributes/attribute_quantization_transform.h"
#include "draco/compression/attributes/attributes_decoder.h"

namespace draco {

// Decodes attributes encoded with the KdTreeAttributesEncoder.
class KdTreeAttributesDecoder : public AttributesDecoder {
 public:
  KdTreeAttributesDecoder();

 protected:
  bool DecodePortableAttributes(DecoderBuffer *in_buffer) override;
  bool DecodeDataNeededByPortableTransforms(DecoderBuffer *in_buffer) override;
  bool TransformAttributesToOriginalFormat() override;

 private:
  template <int level_t, typename OutIteratorT>
  bool DecodePoints(int total_dimensionality, int num_expected_points,
                    DecoderBuffer *in_buffer, OutIteratorT *out_iterator);

  template <typename SignedDataTypeT>
  bool TransformAttributeBackToSignedType(PointAttribute *att,
                                          int num_processed_signed_components);

  std::vector<AttributeQuantizationTransform>
      attribute_quantization_transforms_;
  std::vector<int32_t> min_signed_values_;
  std::vector<std::unique_ptr<PointAttribute>> quantized_portable_attributes_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_DECODER_H_
