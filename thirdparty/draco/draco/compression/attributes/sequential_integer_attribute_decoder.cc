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
#include "draco/compression/attributes/sequential_integer_attribute_decoder.h"

#include "draco/compression/attributes/prediction_schemes/prediction_scheme_decoder_factory.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_wrap_decoding_transform.h"
#include "draco/compression/entropy/symbol_decoding.h"

namespace draco {

SequentialIntegerAttributeDecoder::SequentialIntegerAttributeDecoder() {}

bool SequentialIntegerAttributeDecoder::Init(PointCloudDecoder *decoder,
                                             int attribute_id) {
  if (!SequentialAttributeDecoder::Init(decoder, attribute_id)) {
    return false;
  }
  return true;
}

bool SequentialIntegerAttributeDecoder::TransformAttributeToOriginalFormat(
    const std::vector<PointIndex> &point_ids) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder() &&
      decoder()->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    return true;  // Don't revert the transform here for older files.
  }
#endif
  return StoreValues(static_cast<uint32_t>(point_ids.size()));
}

bool SequentialIntegerAttributeDecoder::DecodeValues(
    const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
  // Decode prediction scheme.
  int8_t prediction_scheme_method;
  if (!in_buffer->Decode(&prediction_scheme_method)) {
    return false;
  }
  // Check that decoded prediction scheme method type is valid.
  if (prediction_scheme_method < PREDICTION_NONE ||
      prediction_scheme_method >= NUM_PREDICTION_SCHEMES) {
    return false;
  }
  if (prediction_scheme_method != PREDICTION_NONE) {
    int8_t prediction_transform_type;
    if (!in_buffer->Decode(&prediction_transform_type)) {
      return false;
    }
    // Check that decoded prediction scheme transform type is valid.
    if (prediction_transform_type < PREDICTION_TRANSFORM_NONE ||
        prediction_transform_type >= NUM_PREDICTION_SCHEME_TRANSFORM_TYPES) {
      return false;
    }
    prediction_scheme_ = CreateIntPredictionScheme(
        static_cast<PredictionSchemeMethod>(prediction_scheme_method),
        static_cast<PredictionSchemeTransformType>(prediction_transform_type));
  }

  if (prediction_scheme_) {
    if (!InitPredictionScheme(prediction_scheme_.get())) {
      return false;
    }
  }

  if (!DecodeIntegerValues(point_ids, in_buffer)) {
    return false;
  }

#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  const int32_t num_values = static_cast<uint32_t>(point_ids.size());
  if (decoder() &&
      decoder()->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    // For older files, revert the transform right after we decode the data.
    if (!StoreValues(num_values)) {
      return false;
    }
  }
#endif
  return true;
}

std::unique_ptr<PredictionSchemeTypedDecoderInterface<int32_t>>
SequentialIntegerAttributeDecoder::CreateIntPredictionScheme(
    PredictionSchemeMethod method,
    PredictionSchemeTransformType transform_type) {
  if (transform_type != PREDICTION_TRANSFORM_WRAP) {
    return nullptr;  // For now we support only wrap transform.
  }
  return CreatePredictionSchemeForDecoder<
      int32_t, PredictionSchemeWrapDecodingTransform<int32_t>>(
      method, attribute_id(), decoder());
}

bool SequentialIntegerAttributeDecoder::DecodeIntegerValues(
    const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer) {
  const int num_components = GetNumValueComponents();
  if (num_components <= 0) {
    return false;
  }
  const size_t num_entries = point_ids.size();
  const size_t num_values = num_entries * num_components;
  PreparePortableAttribute(static_cast<int>(num_entries), num_components);
  int32_t *const portable_attribute_data = GetPortableAttributeData();
  if (portable_attribute_data == nullptr) {
    return false;
  }
  uint8_t compressed;
  if (!in_buffer->Decode(&compressed)) {
    return false;
  }
  if (compressed > 0) {
    // Decode compressed values.
    if (!DecodeSymbols(static_cast<uint32_t>(num_values), num_components,
                       in_buffer,
                       reinterpret_cast<uint32_t *>(portable_attribute_data))) {
      return false;
    }
  } else {
    // Decode the integer data directly.
    // Get the number of bytes for a given entry.
    uint8_t num_bytes;
    if (!in_buffer->Decode(&num_bytes)) {
      return false;
    }
    if (num_bytes == DataTypeLength(DT_INT32)) {
      if (portable_attribute()->buffer()->data_size() <
          sizeof(int32_t) * num_values) {
        return false;
      }
      if (!in_buffer->Decode(portable_attribute_data,
                             sizeof(int32_t) * num_values)) {
        return false;
      }
    } else {
      if (portable_attribute()->buffer()->data_size() <
          num_bytes * num_values) {
        return false;
      }
      if (in_buffer->remaining_size() <
          static_cast<int64_t>(num_bytes) * static_cast<int64_t>(num_values)) {
        return false;
      }
      for (size_t i = 0; i < num_values; ++i) {
        if (!in_buffer->Decode(portable_attribute_data + i, num_bytes)) {
          return false;
        }
      }
    }
  }

  if (num_values > 0 && (prediction_scheme_ == nullptr ||
                         !prediction_scheme_->AreCorrectionsPositive())) {
    // Convert the values back to the original signed format.
    ConvertSymbolsToSignedInts(
        reinterpret_cast<const uint32_t *>(portable_attribute_data),
        static_cast<int>(num_values), portable_attribute_data);
  }

  // If the data was encoded with a prediction scheme, we must revert it.
  if (prediction_scheme_) {
    if (!prediction_scheme_->DecodePredictionData(in_buffer)) {
      return false;
    }

    if (num_values > 0) {
      if (!prediction_scheme_->ComputeOriginalValues(
              portable_attribute_data, portable_attribute_data,
              static_cast<int>(num_values), num_components, point_ids.data())) {
        return false;
      }
    }
  }
  return true;
}

bool SequentialIntegerAttributeDecoder::StoreValues(uint32_t num_values) {
  switch (attribute()->data_type()) {
    case DT_UINT8:
      StoreTypedValues<uint8_t>(num_values);
      break;
    case DT_INT8:
      StoreTypedValues<int8_t>(num_values);
      break;
    case DT_UINT16:
      StoreTypedValues<uint16_t>(num_values);
      break;
    case DT_INT16:
      StoreTypedValues<int16_t>(num_values);
      break;
    case DT_UINT32:
      StoreTypedValues<uint32_t>(num_values);
      break;
    case DT_INT32:
      StoreTypedValues<int32_t>(num_values);
      break;
    default:
      return false;
  }
  return true;
}

template <typename AttributeTypeT>
void SequentialIntegerAttributeDecoder::StoreTypedValues(uint32_t num_values) {
  const int num_components = attribute()->num_components();
  const int entry_size = sizeof(AttributeTypeT) * num_components;
  const std::unique_ptr<AttributeTypeT[]> att_val(
      new AttributeTypeT[num_components]);
  const int32_t *const portable_attribute_data = GetPortableAttributeData();
  int val_id = 0;
  int out_byte_pos = 0;
  for (uint32_t i = 0; i < num_values; ++i) {
    for (int c = 0; c < num_components; ++c) {
      const AttributeTypeT value =
          static_cast<AttributeTypeT>(portable_attribute_data[val_id++]);
      att_val[c] = value;
    }
    // Store the integer value into the attribute buffer.
    attribute()->buffer()->Write(out_byte_pos, att_val.get(), entry_size);
    out_byte_pos += entry_size;
  }
}

void SequentialIntegerAttributeDecoder::PreparePortableAttribute(
    int num_entries, int num_components) {
  GeometryAttribute ga;
  ga.Init(attribute()->attribute_type(), nullptr, num_components, DT_INT32,
          false, num_components * DataTypeLength(DT_INT32), 0);
  std::unique_ptr<PointAttribute> port_att(new PointAttribute(ga));
  port_att->SetIdentityMapping();
  port_att->Reset(num_entries);
  port_att->set_unique_id(attribute()->unique_id());
  SetPortableAttribute(std::move(port_att));
}

}  // namespace draco
