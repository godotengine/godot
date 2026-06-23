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
#include "draco/attributes/point_attribute.h"

#include <tuple>
#include <unordered_map>
using std::unordered_map;

// Shortcut for typed conditionals.
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

namespace draco {

PointAttribute::PointAttribute()
    : num_unique_entries_(0), identity_mapping_(false) {}

PointAttribute::PointAttribute(const GeometryAttribute &att)
    : GeometryAttribute(att),
      num_unique_entries_(0),
      identity_mapping_(false) {}

void PointAttribute::Init(Type attribute_type, int8_t num_components,
                          DataType data_type, bool normalized,
                          size_t num_attribute_values) {
  attribute_buffer_ = std::unique_ptr<DataBuffer>(new DataBuffer());
  GeometryAttribute::Init(attribute_type, attribute_buffer_.get(),
                          num_components, data_type, normalized,
                          DataTypeLength(data_type) * num_components, 0);
  Reset(num_attribute_values);
  SetIdentityMapping();
}

void PointAttribute::CopyFrom(const PointAttribute &src_att) {
  if (buffer() == nullptr) {
    // If the destination attribute doesn't have a valid buffer, create it.
    attribute_buffer_ = std::unique_ptr<DataBuffer>(new DataBuffer());
    ResetBuffer(attribute_buffer_.get(), 0, 0);
  }
  if (!GeometryAttribute::CopyFrom(src_att)) {
    return;
  }
  identity_mapping_ = src_att.identity_mapping_;
  num_unique_entries_ = src_att.num_unique_entries_;
  indices_map_ = src_att.indices_map_;
  if (src_att.attribute_transform_data_) {
    attribute_transform_data_ = std::unique_ptr<AttributeTransformData>(
        new AttributeTransformData(*src_att.attribute_transform_data_));
  } else {
    attribute_transform_data_ = nullptr;
  }
}

bool PointAttribute::Reset(size_t num_attribute_values) {
  if (attribute_buffer_ == nullptr) {
    attribute_buffer_ = std::unique_ptr<DataBuffer>(new DataBuffer());
  }
  const int64_t entry_size = DataTypeLength(data_type()) * num_components();
  if (!attribute_buffer_->Update(nullptr, num_attribute_values * entry_size)) {
    return false;
  }
  // Assign the new buffer to the parent attribute.
  ResetBuffer(attribute_buffer_.get(), entry_size, 0);
  num_unique_entries_ = static_cast<uint32_t>(num_attribute_values);
  return true;
}

void PointAttribute::Resize(size_t new_num_unique_entries) {
  num_unique_entries_ = static_cast<uint32_t>(new_num_unique_entries);
  attribute_buffer_->Resize(new_num_unique_entries * byte_stride());
}

#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
AttributeValueIndex::ValueType PointAttribute::DeduplicateValues(
    const GeometryAttribute &in_att) {
  return DeduplicateValues(in_att, AttributeValueIndex(0));
}

AttributeValueIndex::ValueType PointAttribute::DeduplicateValues(
    const GeometryAttribute &in_att, AttributeValueIndex in_att_offset) {
  AttributeValueIndex::ValueType unique_vals = 0;
  switch (in_att.data_type()) {
    // Currently we support only float, uint8, and uint16 arguments.
    case DT_FLOAT32:
      unique_vals = DeduplicateTypedValues<float>(in_att, in_att_offset);
      break;
    case DT_INT8:
      unique_vals = DeduplicateTypedValues<int8_t>(in_att, in_att_offset);
      break;
    case DT_UINT8:
    case DT_BOOL:
      unique_vals = DeduplicateTypedValues<uint8_t>(in_att, in_att_offset);
      break;
    case DT_UINT16:
      unique_vals = DeduplicateTypedValues<uint16_t>(in_att, in_att_offset);
      break;
    case DT_INT16:
      unique_vals = DeduplicateTypedValues<int16_t>(in_att, in_att_offset);
      break;
    case DT_UINT32:
      unique_vals = DeduplicateTypedValues<uint32_t>(in_att, in_att_offset);
      break;
    case DT_INT32:
      unique_vals = DeduplicateTypedValues<int32_t>(in_att, in_att_offset);
      break;
    default:
      return -1;  // Unsupported data type.
  }
  if (unique_vals == 0) {
    return -1;  // Unexpected error.
  }
  return unique_vals;
}

// Helper function for calling UnifyDuplicateAttributes<T,num_components_t>
// with the correct template arguments.
// Returns the number of unique attribute values.
template <typename T>
AttributeValueIndex::ValueType PointAttribute::DeduplicateTypedValues(
    const GeometryAttribute &in_att, AttributeValueIndex in_att_offset) {
  // Select the correct method to call based on the number of attribute
  // components.
  switch (in_att.num_components()) {
    case 1:
      return DeduplicateFormattedValues<T, 1>(in_att, in_att_offset);
    case 2:
      return DeduplicateFormattedValues<T, 2>(in_att, in_att_offset);
    case 3:
      return DeduplicateFormattedValues<T, 3>(in_att, in_att_offset);
    case 4:
      return DeduplicateFormattedValues<T, 4>(in_att, in_att_offset);
    default:
      return 0;
  }
}

template <typename T, int num_components_t>
AttributeValueIndex::ValueType PointAttribute::DeduplicateFormattedValues(
    const GeometryAttribute &in_att, AttributeValueIndex in_att_offset) {
  // We want to detect duplicates using a hash map but we cannot hash floating
  // point numbers directly so bit-copy floats to the same sized integers and
  // hash them.

  // First we need to determine which int type to use (1, 2, 4 or 8 bytes).
  // Note, this is done at compile time using std::conditional struct.
  // Conditional is in form <bool-expression, true, false>. If bool-expression
  // is true the "true" branch is used and vice versa. All at compile time.
  typedef conditional_t<sizeof(T) == 1, uint8_t,
                        conditional_t<sizeof(T) == 2, uint16_t,
                                      conditional_t<sizeof(T) == 4, uint32_t,
                                                    /*else*/ uint64_t>>>
      HashType;

  AttributeValueIndex unique_vals(0);
  typedef std::array<T, num_components_t> AttributeValue;
  typedef std::array<HashType, num_components_t> AttributeHashableValue;
  typedef unordered_map<AttributeHashableValue, AttributeValueIndex,
                        HashArray<AttributeHashableValue>>
      ValueToIndexMap;

  // Hash map storing index of the first attribute with a given value.
  ValueToIndexMap value_to_index_map;
  AttributeValue att_value;
  AttributeHashableValue hashable_value;
  IndexTypeVector<AttributeValueIndex, AttributeValueIndex> value_map(
      num_unique_entries_);
  for (AttributeValueIndex i(0); i < num_unique_entries_; ++i) {
    const AttributeValueIndex att_pos = i + in_att_offset;
    att_value = in_att.GetValue<T, num_components_t>(att_pos);
    // Convert the value to hashable type. Bit-copy real attributes to integers.
    memcpy(&(hashable_value[0]), &(att_value[0]), sizeof(att_value));

    typename ValueToIndexMap::iterator it;
    bool inserted;
    std::tie(it, inserted) = value_to_index_map.insert(
        std::pair<AttributeHashableValue, AttributeValueIndex>(hashable_value,
                                                               unique_vals));

    // Try to update the hash map with a new entry pointing to the latest unique
    // vertex index.
    if (!inserted) {
      // Duplicated value found. Update index mapping.
      value_map[i] = it->second;
    } else {
      // New unique value.
      SetAttributeValue(unique_vals, &att_value);
      // Update index mapping.
      value_map[i] = unique_vals;

      ++unique_vals;
    }
  }
  if (unique_vals == num_unique_entries_) {
    return unique_vals.value();  // Nothing has changed.
  }
  if (is_mapping_identity()) {
    // Change identity mapping to the explicit one.
    // The number of points is equal to the number of old unique values.
    SetExplicitMapping(num_unique_entries_);
    // Update the explicit map.
    for (uint32_t i = 0; i < num_unique_entries_; ++i) {
      SetPointMapEntry(PointIndex(i), value_map[AttributeValueIndex(i)]);
    }
  } else {
    // Update point to value map using the mapping between old and new values.
    for (PointIndex i(0); i < static_cast<uint32_t>(indices_map_.size()); ++i) {
      SetPointMapEntry(i, value_map[indices_map_[i]]);
    }
  }
  num_unique_entries_ = unique_vals.value();
  return num_unique_entries_;
}
#endif

#ifdef DRACO_TRANSCODER_SUPPORTED
void PointAttribute::RemoveUnusedValues() {
  if (is_mapping_identity()) {
    return;  // For identity mapping, all values are always used.
  }
  // For explicit mapping we need to check if any point is mapped to a value.
  // If not we can delete the value.
  IndexTypeVector<AttributeValueIndex, bool> is_value_used(size(), false);
  int num_used_values = 0;
  for (PointIndex pi(0); pi < indices_map_.size(); ++pi) {
    const AttributeValueIndex avi = indices_map_[pi];
    if (!is_value_used[avi]) {
      is_value_used[avi] = true;
      num_used_values++;
    }
  }
  if (num_used_values == size()) {
    return;  // All values are used.
  }

  // Remap the values and update the point to value mapping.
  IndexTypeVector<AttributeValueIndex, AttributeValueIndex>
      old_to_new_value_map(size(), kInvalidAttributeValueIndex);
  AttributeValueIndex new_avi(0);
  for (AttributeValueIndex avi(0); avi < size(); ++avi) {
    if (!is_value_used[avi]) {
      continue;
    }
    if (avi != new_avi) {
      SetAttributeValue(new_avi, GetAddress(avi));
    }
    old_to_new_value_map[avi] = new_avi++;
  }

  // Remap all points to the new attribute values.
  for (PointIndex pi(0); pi < indices_map_.size(); ++pi) {
    indices_map_[pi] = old_to_new_value_map[indices_map_[pi]];
  }

  num_unique_entries_ = num_used_values;
}
#endif

}  // namespace draco
