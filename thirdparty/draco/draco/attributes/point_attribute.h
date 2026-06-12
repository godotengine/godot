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
#ifndef DRACO_ATTRIBUTES_POINT_ATTRIBUTE_H_
#define DRACO_ATTRIBUTES_POINT_ATTRIBUTE_H_

#include <memory>

#include "draco/attributes/attribute_transform_data.h"
#include "draco/attributes/geometry_attribute.h"
#include "draco/core/draco_index_type_vector.h"
#include "draco/core/hash_utils.h"
#include "draco/core/macros.h"
#include "draco/draco_features.h"

namespace draco {

// Class for storing point specific data about each attribute. In general,
// multiple points stored in a point cloud can share the same attribute value
// and this class provides the necessary mapping between point ids and attribute
// value ids.
class PointAttribute : public GeometryAttribute {
 public:
  PointAttribute();
  explicit PointAttribute(const GeometryAttribute &att);

  // Make sure the move constructor is defined (needed for better performance
  // when new attributes are added to PointCloud).
  PointAttribute(PointAttribute &&attribute) = default;
  PointAttribute &operator=(PointAttribute &&attribute) = default;

  // Initializes a point attribute. By default the attribute will be set to
  // identity mapping between point indices and attribute values. To set custom
  // mapping use SetExplicitMapping() function.
  void Init(Type attribute_type, int8_t num_components, DataType data_type,
            bool normalized, size_t num_attribute_values);

  // Copies attribute data from the provided |src_att| attribute.
  void CopyFrom(const PointAttribute &src_att);

  // Prepares the attribute storage for the specified number of entries.
  bool Reset(size_t num_attribute_values);

  size_t size() const { return num_unique_entries_; }
  AttributeValueIndex mapped_index(PointIndex point_index) const {
    if (identity_mapping_) {
      return AttributeValueIndex(point_index.value());
    }
    return indices_map_[point_index];
  }
  DataBuffer *buffer() const { return attribute_buffer_.get(); }
  bool is_mapping_identity() const { return identity_mapping_; }
  size_t indices_map_size() const {
    if (is_mapping_identity()) {
      return 0;
    }
    return indices_map_.size();
  }

  const uint8_t *GetAddressOfMappedIndex(PointIndex point_index) const {
    return GetAddress(mapped_index(point_index));
  }

  // Sets the new number of unique attribute entries for the attribute. The
  // function resizes the attribute storage to hold |num_attribute_values|
  // entries.
  // All previous entries with AttributeValueIndex < |num_attribute_values|
  // are preserved. Caller needs to ensure that the PointAttribute is still
  // valid after the resizing operation (that is, each point is mapped to a
  // valid attribute value).
  void Resize(size_t new_num_unique_entries);

  // Functions for setting the type of mapping between point indices and
  // attribute entry ids.
  // This function sets the mapping to implicit, where point indices are equal
  // to attribute entry indices.
  void SetIdentityMapping() {
    identity_mapping_ = true;
    indices_map_.clear();
  }
  // This function sets the mapping to be explicitly using the indices_map_
  // array that needs to be initialized by the caller.
  void SetExplicitMapping(size_t num_points) {
    identity_mapping_ = false;
    indices_map_.resize(num_points, kInvalidAttributeValueIndex);
  }

  // Set an explicit map entry for a specific point index.
  void SetPointMapEntry(PointIndex point_index,
                        AttributeValueIndex entry_index) {
    DRACO_DCHECK(!identity_mapping_);
    indices_map_[point_index] = entry_index;
  }

  // Same as GeometryAttribute::GetValue(), but using point id as the input.
  // Mapping to attribute value index is performed automatically.
  void GetMappedValue(PointIndex point_index, void *out_data) const {
    return GetValue(mapped_index(point_index), out_data);
  }

#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
  // Deduplicate |in_att| values into |this| attribute. |in_att| can be equal
  // to |this|.
  // Returns -1 if the deduplication failed.
  AttributeValueIndex::ValueType DeduplicateValues(
      const GeometryAttribute &in_att);

  // Same as above but the values read from |in_att| are sampled with the
  // provided offset |in_att_offset|.
  AttributeValueIndex::ValueType DeduplicateValues(
      const GeometryAttribute &in_att, AttributeValueIndex in_att_offset);
#endif

  // Set attribute transform data for the attribute. The data is used to store
  // the type and parameters of the transform that is applied on the attribute
  // data (optional).
  void SetAttributeTransformData(
      std::unique_ptr<AttributeTransformData> transform_data) {
    attribute_transform_data_ = std::move(transform_data);
  }
  const AttributeTransformData *GetAttributeTransformData() const {
    return attribute_transform_data_.get();
  }

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Removes unused values from the attribute. Value is unused when no point
  // is mapped to the value. Only applicable when the mapping is not identity.
  void RemoveUnusedValues();
#endif

 private:
#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
  template <typename T>
  AttributeValueIndex::ValueType DeduplicateTypedValues(
      const GeometryAttribute &in_att, AttributeValueIndex in_att_offset);
  template <typename T, int COMPONENTS_COUNT>
  AttributeValueIndex::ValueType DeduplicateFormattedValues(
      const GeometryAttribute &in_att, AttributeValueIndex in_att_offset);
#endif

  // Data storage for attribute values. GeometryAttribute itself doesn't own its
  // buffer so we need to allocate it here.
  std::unique_ptr<DataBuffer> attribute_buffer_;

  // Mapping between point ids and attribute value ids.
  IndexTypeVector<PointIndex, AttributeValueIndex> indices_map_;
  AttributeValueIndex::ValueType num_unique_entries_;
  // Flag when the mapping between point ids and attribute values is identity.
  bool identity_mapping_;

  // If an attribute contains transformed data (e.g. quantized), we can specify
  // the attribute transform here and use it to transform the attribute back to
  // its original format.
  std::unique_ptr<AttributeTransformData> attribute_transform_data_;

  friend struct PointAttributeHasher;
};

// Hash functor for the PointAttribute class.
struct PointAttributeHasher {
  size_t operator()(const PointAttribute &attribute) const {
    GeometryAttributeHasher base_hasher;
    size_t hash = base_hasher(attribute);
    hash = HashCombine(attribute.identity_mapping_, hash);
    hash = HashCombine(attribute.num_unique_entries_, hash);
    hash = HashCombine(attribute.indices_map_.size(), hash);
    if (!attribute.indices_map_.empty()) {
      const uint64_t indices_hash = FingerprintString(
          reinterpret_cast<const char *>(attribute.indices_map_.data()),
          attribute.indices_map_.size());
      hash = HashCombine(indices_hash, hash);
    }
    if (attribute.attribute_buffer_ != nullptr) {
      const uint64_t buffer_hash = FingerprintString(
          reinterpret_cast<const char *>(attribute.attribute_buffer_->data()),
          attribute.attribute_buffer_->data_size());
      hash = HashCombine(buffer_hash, hash);
    }
    return hash;
  }
};

}  // namespace draco

#endif  // DRACO_ATTRIBUTES_POINT_ATTRIBUTE_H_
