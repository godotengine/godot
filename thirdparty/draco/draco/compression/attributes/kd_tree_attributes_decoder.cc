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
#include "draco/compression/attributes/kd_tree_attributes_decoder.h"

#include "draco/compression/attributes/kd_tree_attributes_shared.h"
#include "draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_decoder.h"
#include "draco/compression/point_cloud/algorithms/float_points_tree_decoder.h"
#include "draco/compression/point_cloud/point_cloud_decoder.h"
#include "draco/core/draco_types.h"
#include "draco/core/varint_decoding.h"

namespace draco {

// attribute, offset_dimensionality, data_type, data_size, num_components
using AttributeTuple =
    std::tuple<PointAttribute *, uint32_t, DataType, uint32_t, uint32_t>;

// Output iterator that is used to decode values directly into the data buffer
// of the modified PointAttribute.
// The extension of this iterator beyond the DT_UINT32 concerns itself only with
// the size of the data for efficiency, not the type.  DataType is conveyed in
// but is an unused field populated for any future logic/special casing.
// DT_UINT32 and all other 4-byte types are naturally supported from the size of
// data in the kd tree encoder.  DT_UINT16 and DT_UINT8 are supported by way
// of byte copies into a temporary memory buffer.
template <class CoeffT>
class PointAttributeVectorOutputIterator {
  typedef PointAttributeVectorOutputIterator<CoeffT> Self;

 public:
  PointAttributeVectorOutputIterator(
      PointAttributeVectorOutputIterator &&that) = default;

  explicit PointAttributeVectorOutputIterator(
      const std::vector<AttributeTuple> &atts)
      : attributes_(atts), point_id_(0) {
    DRACO_DCHECK_GE(atts.size(), 1);
    uint32_t required_decode_bytes = 0;
    for (auto index = 0; index < attributes_.size(); index++) {
      const AttributeTuple &att = attributes_[index];
      required_decode_bytes = (std::max)(required_decode_bytes,
                                         std::get<3>(att) * std::get<4>(att));
    }
    memory_.resize(required_decode_bytes);
    data_ = memory_.data();
  }

  const Self &operator++() {
    ++point_id_;
    return *this;
  }

  // We do not want to do ANY copying of this constructor so this particular
  // operator is disabled for performance reasons.
  // Self operator++(int) {
  //   Self copy = *this;
  //   ++point_id_;
  //   return copy;
  // }

  Self &operator*() { return *this; }
  // Still needed in some cases.
  // TODO(b/199760123): Remove.
  // hardcoded to 3 based on legacy usage.
  const Self &operator=(const VectorD<CoeffT, 3> &val) {
    DRACO_DCHECK_EQ(attributes_.size(), 1);  // Expect only ONE attribute.
    AttributeTuple &att = attributes_[0];
    PointAttribute *attribute = std::get<0>(att);
    const AttributeValueIndex avi = attribute->mapped_index(point_id_);
    if (avi >= static_cast<uint32_t>(attribute->size())) {
      return *this;
    }
    const uint32_t &offset = std::get<1>(att);
    DRACO_DCHECK_EQ(offset, 0);  // expected to be zero
    attribute->SetAttributeValue(avi, &val[0] + offset);
    return *this;
  }
  // Additional operator taking std::vector as argument.
  const Self &operator=(const std::vector<CoeffT> &val) {
    for (auto index = 0; index < attributes_.size(); index++) {
      AttributeTuple &att = attributes_[index];
      PointAttribute *attribute = std::get<0>(att);
      const AttributeValueIndex avi = attribute->mapped_index(point_id_);
      if (avi >= static_cast<uint32_t>(attribute->size())) {
        return *this;
      }
      const uint32_t &offset = std::get<1>(att);
      const uint32_t &data_size = std::get<3>(att);
      const uint32_t &num_components = std::get<4>(att);
      const uint32_t *data_source = val.data() + offset;
      if (data_size < 4) {  // handle uint16_t, uint8_t
        // selectively copy data bytes
        uint8_t *data_counter = data_;
        for (uint32_t index = 0; index < num_components;
             index += 1, data_counter += data_size) {
          std::memcpy(data_counter, data_source + index, data_size);
        }
        // redirect to copied data
        data_source = reinterpret_cast<uint32_t *>(data_);
      }
      attribute->SetAttributeValue(avi, data_source);
    }
    return *this;
  }

 private:
  // preallocated memory for buffering different data sizes.  Never reallocated.
  std::vector<uint8_t> memory_;
  uint8_t *data_;
  std::vector<AttributeTuple> attributes_;
  PointIndex point_id_;

  // NO COPY
  PointAttributeVectorOutputIterator(
      const PointAttributeVectorOutputIterator &that) = delete;
  PointAttributeVectorOutputIterator &operator=(
      PointAttributeVectorOutputIterator const &) = delete;
};

KdTreeAttributesDecoder::KdTreeAttributesDecoder() {}

bool KdTreeAttributesDecoder::DecodePortableAttributes(
    DecoderBuffer *in_buffer) {
  if (in_buffer->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 3)) {
    // Old bitstream does everything in the
    // DecodeDataNeededByPortableTransforms() method.
    return true;
  }
  uint8_t compression_level = 0;
  if (!in_buffer->Decode(&compression_level)) {
    return false;
  }
  const int32_t num_points = GetDecoder()->point_cloud()->num_points();

  // Decode data using the kd tree decoding into integer (portable) attributes.
  // We first need to go over all attributes and create a new portable storage
  // for those attributes that need it (floating point attributes that have to
  // be dequantized after decoding).

  const int num_attributes = GetNumAttributes();
  uint32_t total_dimensionality = 0;  // position is a required dimension
  std::vector<AttributeTuple> atts(num_attributes);

  for (int i = 0; i < GetNumAttributes(); ++i) {
    const int att_id = GetAttributeId(i);
    PointAttribute *const att = GetDecoder()->point_cloud()->attribute(att_id);
    // All attributes have the same number of values and identity mapping
    // between PointIndex and AttributeValueIndex.
    att->Reset(num_points);
    att->SetIdentityMapping();

    PointAttribute *target_att = nullptr;
    if (att->data_type() == DT_UINT32 || att->data_type() == DT_UINT16 ||
        att->data_type() == DT_UINT8) {
      // We can decode to these attributes directly.
      target_att = att;
    } else if (att->data_type() == DT_INT32 || att->data_type() == DT_INT16 ||
               att->data_type() == DT_INT8) {
      // Prepare storage for data that is used to convert unsigned values back
      // to the signed ones.
      for (int c = 0; c < att->num_components(); ++c) {
        min_signed_values_.push_back(0);
      }
      target_att = att;
    } else if (att->data_type() == DT_FLOAT32) {
      // Create a portable attribute that will hold the decoded data. We will
      // dequantize the decoded data to the final attribute later on.
      const int num_components = att->num_components();
      GeometryAttribute va;
      va.Init(att->attribute_type(), nullptr, num_components, DT_UINT32, false,
              num_components * DataTypeLength(DT_UINT32), 0);
      std::unique_ptr<PointAttribute> port_att(new PointAttribute(va));
      port_att->SetIdentityMapping();
      port_att->Reset(num_points);
      quantized_portable_attributes_.push_back(std::move(port_att));
      target_att = quantized_portable_attributes_.back().get();
    } else {
      // Unsupported type.
      return false;
    }
    // Add attribute to the output iterator used by the core algorithm.
    const DataType data_type = target_att->data_type();
    const uint32_t data_size = (std::max)(0, DataTypeLength(data_type));
    const uint32_t num_components = target_att->num_components();
    atts[i] = std::make_tuple(target_att, total_dimensionality, data_type,
                              data_size, num_components);
    total_dimensionality += num_components;
  }
  typedef PointAttributeVectorOutputIterator<uint32_t> OutIt;
  OutIt out_it(atts);

  switch (compression_level) {
    case 0: {
      if (!DecodePoints<0, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 1: {
      if (!DecodePoints<1, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 2: {
      if (!DecodePoints<2, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 3: {
      if (!DecodePoints<3, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 4: {
      if (!DecodePoints<4, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 5: {
      if (!DecodePoints<5, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    case 6: {
      if (!DecodePoints<6, OutIt>(total_dimensionality, num_points, in_buffer,
                                  &out_it)) {
        return false;
      }
      break;
    }
    default:
      return false;
  }
  return true;
}

template <int level_t, typename OutIteratorT>
bool KdTreeAttributesDecoder::DecodePoints(int total_dimensionality,
                                           int num_expected_points,
                                           DecoderBuffer *in_buffer,
                                           OutIteratorT *out_iterator) {
  DynamicIntegerPointsKdTreeDecoder<level_t> decoder(total_dimensionality);
  if (!decoder.DecodePoints(in_buffer, *out_iterator, num_expected_points) ||
      decoder.num_decoded_points() != num_expected_points) {
    return false;
  }
  return true;
}

bool KdTreeAttributesDecoder::DecodeDataNeededByPortableTransforms(
    DecoderBuffer *in_buffer) {
  if (in_buffer->bitstream_version() >= DRACO_BITSTREAM_VERSION(2, 3)) {
    // Decode quantization data for each attribute that need it.
    // TODO(ostava): This should be moved to AttributeQuantizationTransform.
    std::vector<float> min_value;
    for (int i = 0; i < GetNumAttributes(); ++i) {
      const int att_id = GetAttributeId(i);
      const PointAttribute *const att =
          GetDecoder()->point_cloud()->attribute(att_id);
      if (att->data_type() == DT_FLOAT32) {
        const int num_components = att->num_components();
        min_value.resize(num_components);
        if (!in_buffer->Decode(&min_value[0], sizeof(float) * num_components)) {
          return false;
        }
        float max_value_dif;
        if (!in_buffer->Decode(&max_value_dif)) {
          return false;
        }
        uint8_t quantization_bits;
        if (!in_buffer->Decode(&quantization_bits) || quantization_bits > 31) {
          return false;
        }
        AttributeQuantizationTransform transform;
        if (!transform.SetParameters(quantization_bits, min_value.data(),
                                     num_components, max_value_dif)) {
          return false;
        }
        const int num_transforms =
            static_cast<int>(attribute_quantization_transforms_.size());
        if (!transform.TransferToAttribute(
                quantized_portable_attributes_[num_transforms].get())) {
          return false;
        }
        attribute_quantization_transforms_.push_back(transform);
      }
    }

    // Decode transform data for signed integer attributes.
    for (int i = 0; i < min_signed_values_.size(); ++i) {
      int32_t val;
      if (!DecodeVarint(&val, in_buffer)) {
        return false;
      }
      min_signed_values_[i] = val;
    }
    return true;
  }
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  // Handle old bitstream
  // Figure out the total dimensionality of the point cloud
  const uint32_t attribute_count = GetNumAttributes();
  uint32_t total_dimensionality = 0;  // position is a required dimension
  std::vector<AttributeTuple> atts(attribute_count);
  for (auto attribute_index = 0;
       static_cast<uint32_t>(attribute_index) < attribute_count;
       attribute_index += 1)  // increment the dimensionality as needed...
  {
    const int att_id = GetAttributeId(attribute_index);
    PointAttribute *const att = GetDecoder()->point_cloud()->attribute(att_id);
    const DataType data_type = att->data_type();
    const uint32_t data_size = (std::max)(0, DataTypeLength(data_type));
    const uint32_t num_components = att->num_components();
    if (data_size > 4) {
      return false;
    }

    atts[attribute_index] = std::make_tuple(
        att, total_dimensionality, data_type, data_size, num_components);
    // everything is treated as 32bit in the encoder.
    total_dimensionality += num_components;
  }

  const int att_id = GetAttributeId(0);
  PointAttribute *const att = GetDecoder()->point_cloud()->attribute(att_id);
  att->SetIdentityMapping();
  // Decode method
  uint8_t method;
  if (!in_buffer->Decode(&method)) {
    return false;
  }
  if (method == KdTreeAttributesEncodingMethod::kKdTreeQuantizationEncoding) {
    // This method only supports one attribute with exactly three components.
    if (atts.size() != 1 || std::get<4>(atts[0]) != 3) {
      return false;
    }
    uint8_t compression_level = 0;
    if (!in_buffer->Decode(&compression_level)) {
      return false;
    }
    uint32_t num_points = 0;
    if (!in_buffer->Decode(&num_points)) {
      return false;
    }
    att->Reset(num_points);
    FloatPointsTreeDecoder decoder;
    decoder.set_num_points_from_header(num_points);
    PointAttributeVectorOutputIterator<float> out_it(atts);
    if (!decoder.DecodePointCloud(in_buffer, out_it)) {
      return false;
    }
  } else if (method == KdTreeAttributesEncodingMethod::kKdTreeIntegerEncoding) {
    uint8_t compression_level = 0;
    if (!in_buffer->Decode(&compression_level)) {
      return false;
    }
    if (6 < compression_level) {
      DRACO_LOGE(
          "KdTreeAttributesDecoder: compression level %i not supported.\n",
          compression_level);
      return false;
    }

    uint32_t num_points;
    if (!in_buffer->Decode(&num_points)) {
      return false;
    }

    for (auto attribute_index = 0;
         static_cast<uint32_t>(attribute_index) < attribute_count;
         attribute_index += 1) {
      const int att_id = GetAttributeId(attribute_index);
      PointAttribute *const attr =
          GetDecoder()->point_cloud()->attribute(att_id);
      attr->Reset(num_points);
      attr->SetIdentityMapping();
    }

    PointAttributeVectorOutputIterator<uint32_t> out_it(atts);

    switch (compression_level) {
      case 0: {
        DynamicIntegerPointsKdTreeDecoder<0> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 1: {
        DynamicIntegerPointsKdTreeDecoder<1> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 2: {
        DynamicIntegerPointsKdTreeDecoder<2> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 3: {
        DynamicIntegerPointsKdTreeDecoder<3> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 4: {
        DynamicIntegerPointsKdTreeDecoder<4> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 5: {
        DynamicIntegerPointsKdTreeDecoder<5> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      case 6: {
        DynamicIntegerPointsKdTreeDecoder<6> decoder(total_dimensionality);
        if (!decoder.DecodePoints(in_buffer, out_it)) {
          return false;
        }
        break;
      }
      default:
        return false;
    }
  } else {
    // Invalid method.
    return false;
  }
  return true;
#else
  return false;
#endif
}

template <typename SignedDataTypeT>
bool KdTreeAttributesDecoder::TransformAttributeBackToSignedType(
    PointAttribute *att, int num_processed_signed_components) {
  typedef typename std::make_unsigned<SignedDataTypeT>::type UnsignedType;
  std::vector<UnsignedType> unsigned_val(att->num_components());
  std::vector<SignedDataTypeT> signed_val(att->num_components());

  for (AttributeValueIndex avi(0); avi < static_cast<uint32_t>(att->size());
       ++avi) {
    att->GetValue(avi, &unsigned_val[0]);
    for (int c = 0; c < att->num_components(); ++c) {
      // Up-cast |unsigned_val| to int32_t to ensure we don't overflow it for
      // smaller data types. But first check that the up-casting does not cause
      // signed integer overflow.
      if (unsigned_val[c] > std::numeric_limits<int32_t>::max()) {
        return false;
      }
      signed_val[c] = static_cast<SignedDataTypeT>(
          static_cast<int32_t>(unsigned_val[c]) +
          min_signed_values_[num_processed_signed_components + c]);
    }
    att->SetAttributeValue(avi, &signed_val[0]);
  }
  return true;
}

bool KdTreeAttributesDecoder::TransformAttributesToOriginalFormat() {
  if (quantized_portable_attributes_.empty() && min_signed_values_.empty()) {
    return true;
  }
  int num_processed_quantized_attributes = 0;
  int num_processed_signed_components = 0;
  // Dequantize attributes that needed it.
  for (int i = 0; i < GetNumAttributes(); ++i) {
    const int att_id = GetAttributeId(i);
    PointAttribute *const att = GetDecoder()->point_cloud()->attribute(att_id);
    if (att->data_type() == DT_INT32 || att->data_type() == DT_INT16 ||
        att->data_type() == DT_INT8) {
      std::vector<uint32_t> unsigned_val(att->num_components());
      std::vector<int32_t> signed_val(att->num_components());
      // Values are stored as unsigned in the attribute, make them signed again.
      if (att->data_type() == DT_INT32) {
        if (!TransformAttributeBackToSignedType<int32_t>(
                att, num_processed_signed_components)) {
          return false;
        }
      } else if (att->data_type() == DT_INT16) {
        if (!TransformAttributeBackToSignedType<int16_t>(
                att, num_processed_signed_components)) {
          return false;
        }
      } else if (att->data_type() == DT_INT8) {
        if (!TransformAttributeBackToSignedType<int8_t>(
                att, num_processed_signed_components)) {
          return false;
        }
      }
      num_processed_signed_components += att->num_components();
    } else if (att->data_type() == DT_FLOAT32) {
      // TODO(ostava): This code should be probably moved out to attribute
      // transform and shared with the SequentialQuantizationAttributeDecoder.

      const PointAttribute *const src_att =
          quantized_portable_attributes_[num_processed_quantized_attributes]
              .get();

      const AttributeQuantizationTransform &transform =
          attribute_quantization_transforms_
              [num_processed_quantized_attributes];

      num_processed_quantized_attributes++;

      if (GetDecoder()->options()->GetAttributeBool(
              att->attribute_type(), "skip_attribute_transform", false)) {
        // Attribute transform should not be performed. In this case, we replace
        // the output geometry attribute with the portable attribute.
        // TODO(ostava): We can potentially avoid this copy by introducing a new
        // mechanism that would allow to use the final attributes as portable
        // attributes for predictors that may need them.
        att->CopyFrom(*src_att);
        continue;
      }

      // Convert all quantized values back to floats.
      const int32_t max_quantized_value =
          (1u << static_cast<uint32_t>(transform.quantization_bits())) - 1;
      const int num_components = att->num_components();
      const int entry_size = sizeof(float) * num_components;
      const std::unique_ptr<float[]> att_val(new float[num_components]);
      int quant_val_id = 0;
      int out_byte_pos = 0;
      Dequantizer dequantizer;
      if (!dequantizer.Init(transform.range(), max_quantized_value)) {
        return false;
      }
      const uint32_t *const portable_attribute_data =
          reinterpret_cast<const uint32_t *>(
              src_att->GetAddress(AttributeValueIndex(0)));
      for (uint32_t i = 0; i < src_att->size(); ++i) {
        for (int c = 0; c < num_components; ++c) {
          float value = dequantizer.DequantizeFloat(
              portable_attribute_data[quant_val_id++]);
          value = value + transform.min_value(c);
          att_val[c] = value;
        }
        // Store the floating point value into the attribute buffer.
        att->buffer()->Write(out_byte_pos, att_val.get(), entry_size);
        out_byte_pos += entry_size;
      }
    }
  }
  return true;
}

}  // namespace draco
