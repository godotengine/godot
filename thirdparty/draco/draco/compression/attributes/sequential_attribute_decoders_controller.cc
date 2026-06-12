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
#include "draco/compression/attributes/sequential_attribute_decoders_controller.h"
#ifdef DRACO_NORMAL_ENCODING_SUPPORTED
#include "draco/compression/attributes/sequential_normal_attribute_decoder.h"
#endif
#include "draco/compression/attributes/sequential_quantization_attribute_decoder.h"
#include "draco/compression/config/compression_shared.h"

namespace draco {

SequentialAttributeDecodersController::SequentialAttributeDecodersController(
    std::unique_ptr<PointsSequencer> sequencer)
    : sequencer_(std::move(sequencer)) {}

bool SequentialAttributeDecodersController::DecodeAttributesDecoderData(
    DecoderBuffer *buffer) {
  if (!AttributesDecoder::DecodeAttributesDecoderData(buffer)) {
    return false;
  }
  // Decode unique ids of all sequential encoders and create them.
  const int32_t num_attributes = GetNumAttributes();
  sequential_decoders_.resize(num_attributes);
  for (int i = 0; i < num_attributes; ++i) {
    uint8_t decoder_type;
    if (!buffer->Decode(&decoder_type)) {
      return false;
    }
    // Create the decoder from the id.
    sequential_decoders_[i] = CreateSequentialDecoder(decoder_type);
    if (!sequential_decoders_[i]) {
      return false;
    }
    if (!sequential_decoders_[i]->Init(GetDecoder(), GetAttributeId(i))) {
      return false;
    }
  }
  return true;
}

bool SequentialAttributeDecodersController::DecodeAttributes(
    DecoderBuffer *buffer) {
  if (!sequencer_ || !sequencer_->GenerateSequence(&point_ids_)) {
    return false;
  }
  // Initialize point to attribute value mapping for all decoded attributes.
  const int32_t num_attributes = GetNumAttributes();
  for (int i = 0; i < num_attributes; ++i) {
    PointAttribute *const pa =
        GetDecoder()->point_cloud()->attribute(GetAttributeId(i));
    if (!sequencer_->UpdatePointToAttributeIndexMapping(pa)) {
      return false;
    }
  }
  return AttributesDecoder::DecodeAttributes(buffer);
}

bool SequentialAttributeDecodersController::DecodePortableAttributes(
    DecoderBuffer *in_buffer) {
  const int32_t num_attributes = GetNumAttributes();
  for (int i = 0; i < num_attributes; ++i) {
    if (!sequential_decoders_[i]->DecodePortableAttribute(point_ids_,
                                                          in_buffer)) {
      return false;
    }
  }
  return true;
}

bool SequentialAttributeDecodersController::
    DecodeDataNeededByPortableTransforms(DecoderBuffer *in_buffer) {
  const int32_t num_attributes = GetNumAttributes();
  for (int i = 0; i < num_attributes; ++i) {
    if (!sequential_decoders_[i]->DecodeDataNeededByPortableTransform(
            point_ids_, in_buffer)) {
      return false;
    }
  }
  return true;
}

bool SequentialAttributeDecodersController::
    TransformAttributesToOriginalFormat() {
  const int32_t num_attributes = GetNumAttributes();
  for (int i = 0; i < num_attributes; ++i) {
    // Check whether the attribute transform should be skipped.
    if (GetDecoder()->options()) {
      const PointAttribute *const attribute =
          sequential_decoders_[i]->attribute();
      const PointAttribute *const portable_attribute =
          sequential_decoders_[i]->GetPortableAttribute();
      if (portable_attribute &&
          GetDecoder()->options()->GetAttributeBool(
              attribute->attribute_type(), "skip_attribute_transform", false)) {
        // Attribute transform should not be performed. In this case, we replace
        // the output geometry attribute with the portable attribute.
        // TODO(ostava): We can potentially avoid this copy by introducing a new
        // mechanism that would allow to use the final attributes as portable
        // attributes for predictors that may need them.
        sequential_decoders_[i]->attribute()->CopyFrom(*portable_attribute);
        continue;
      }
    }
    if (!sequential_decoders_[i]->TransformAttributeToOriginalFormat(
            point_ids_)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<SequentialAttributeDecoder>
SequentialAttributeDecodersController::CreateSequentialDecoder(
    uint8_t decoder_type) {
  switch (decoder_type) {
    case SEQUENTIAL_ATTRIBUTE_ENCODER_GENERIC:
      return std::unique_ptr<SequentialAttributeDecoder>(
          new SequentialAttributeDecoder());
    case SEQUENTIAL_ATTRIBUTE_ENCODER_INTEGER:
      return std::unique_ptr<SequentialAttributeDecoder>(
          new SequentialIntegerAttributeDecoder());
    case SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION:
      return std::unique_ptr<SequentialAttributeDecoder>(
          new SequentialQuantizationAttributeDecoder());
#ifdef DRACO_NORMAL_ENCODING_SUPPORTED
    case SEQUENTIAL_ATTRIBUTE_ENCODER_NORMALS:
      return std::unique_ptr<SequentialNormalAttributeDecoder>(
          new SequentialNormalAttributeDecoder());
#endif
    default:
      break;
  }
  // Unknown or unsupported decoder type.
  return nullptr;
}

}  // namespace draco
