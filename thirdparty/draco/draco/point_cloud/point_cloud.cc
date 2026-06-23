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
#include "draco/point_cloud/point_cloud.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#ifdef DRACO_TRANSCODER_SUPPORTED
#include "draco/attributes/point_attribute.h"
#endif

namespace draco {

PointCloud::PointCloud() : num_points_(0) {}

#ifdef DRACO_TRANSCODER_SUPPORTED
void PointCloud::Copy(const PointCloud &src) {
  num_points_ = src.num_points_;
  for (int i = 0; i < GeometryAttribute::NAMED_ATTRIBUTES_COUNT; ++i) {
    named_attribute_index_[i] = src.named_attribute_index_[i];
  }
  attributes_.resize(src.attributes_.size());
  for (int i = 0; i < src.attributes_.size(); ++i) {
    attributes_[i] = std::unique_ptr<PointAttribute>(new PointAttribute());
    attributes_[i]->CopyFrom(*src.attributes_[i]);
  }
  compression_enabled_ = src.compression_enabled_;
  compression_options_ = src.compression_options_;
  CopyMetadata(src);
}

void PointCloud::CopyMetadata(const PointCloud &src) {
  if (src.metadata_ == nullptr) {
    metadata_ = nullptr;
  } else {
    // Copy base metadata.
    const GeometryMetadata *const metadata = src.metadata_.get();
    metadata_.reset(new GeometryMetadata(*metadata));
  }
}
#endif

int32_t PointCloud::NumNamedAttributes(GeometryAttribute::Type type) const {
  if (type == GeometryAttribute::INVALID ||
      type >= GeometryAttribute::NAMED_ATTRIBUTES_COUNT) {
    return 0;
  }
  return static_cast<int32_t>(named_attribute_index_[type].size());
}

int32_t PointCloud::GetNamedAttributeId(GeometryAttribute::Type type) const {
  return GetNamedAttributeId(type, 0);
}

int32_t PointCloud::GetNamedAttributeId(GeometryAttribute::Type type,
                                        int i) const {
  if (NumNamedAttributes(type) <= i) {
    return -1;
  }
  return named_attribute_index_[type][i];
}

const PointAttribute *PointCloud::GetNamedAttribute(
    GeometryAttribute::Type type) const {
  return GetNamedAttribute(type, 0);
}

const PointAttribute *PointCloud::GetNamedAttribute(
    GeometryAttribute::Type type, int i) const {
  const int32_t att_id = GetNamedAttributeId(type, i);
  if (att_id == -1) {
    return nullptr;
  }
  return attributes_[att_id].get();
}

const PointAttribute *PointCloud::GetNamedAttributeByUniqueId(
    GeometryAttribute::Type type, uint32_t unique_id) const {
  for (size_t att_id = 0; att_id < named_attribute_index_[type].size();
       ++att_id) {
    if (attributes_[named_attribute_index_[type][att_id]]->unique_id() ==
        unique_id) {
      return attributes_[named_attribute_index_[type][att_id]].get();
    }
  }
  return nullptr;
}

const PointAttribute *PointCloud::GetAttributeByUniqueId(
    uint32_t unique_id) const {
  const int32_t att_id = GetAttributeIdByUniqueId(unique_id);
  if (att_id == -1) {
    return nullptr;
  }
  return attributes_[att_id].get();
}

#ifdef DRACO_TRANSCODER_SUPPORTED
const PointAttribute *PointCloud::GetNamedAttributeByName(
    GeometryAttribute::Type type, const std::string &name) const {
  const auto &index = named_attribute_index_;
  for (size_t i = 0; i < index[type].size(); ++i) {
    const PointAttribute *const att = attributes_[index[type][i]].get();
    if (att->name() == name) {
      return att;
    }
  }
  return nullptr;
}
#endif  // DRACO_TRANSCODER_SUPPORTED

int32_t PointCloud::GetAttributeIdByUniqueId(uint32_t unique_id) const {
  for (size_t att_id = 0; att_id < attributes_.size(); ++att_id) {
    if (attributes_[att_id]->unique_id() == unique_id) {
      return static_cast<int32_t>(att_id);
    }
  }
  return -1;
}

int PointCloud::AddAttribute(std::unique_ptr<PointAttribute> pa) {
  SetAttribute(static_cast<int>(attributes_.size()), std::move(pa));
  return static_cast<int>(attributes_.size() - 1);
}

int PointCloud::AddAttribute(
    const GeometryAttribute &att, bool identity_mapping,
    AttributeValueIndex::ValueType num_attribute_values) {
  auto pa = CreateAttribute(att, identity_mapping, num_attribute_values);
  if (!pa) {
    return -1;
  }
  const int32_t att_id = AddAttribute(std::move(pa));
  return att_id;
}

std::unique_ptr<PointAttribute> PointCloud::CreateAttribute(
    const GeometryAttribute &att, bool identity_mapping,
    AttributeValueIndex::ValueType num_attribute_values) const {
  if (att.attribute_type() == GeometryAttribute::INVALID) {
    return nullptr;
  }
  std::unique_ptr<PointAttribute> pa =
      std::unique_ptr<PointAttribute>(new PointAttribute(att));
  // Initialize point cloud specific attribute data.
  if (!identity_mapping) {
    // First create mapping between indices.
    pa->SetExplicitMapping(num_points_);
  } else {
    pa->SetIdentityMapping();
    num_attribute_values = std::max(num_points_, num_attribute_values);
  }
  if (num_attribute_values > 0) {
    pa->Reset(num_attribute_values);
  }
  return pa;
}

void PointCloud::SetAttribute(int att_id, std::unique_ptr<PointAttribute> pa) {
  DRACO_DCHECK(att_id >= 0);
  if (static_cast<int>(attributes_.size()) <= att_id) {
    attributes_.resize(att_id + 1);
  }
  if (pa->attribute_type() < GeometryAttribute::NAMED_ATTRIBUTES_COUNT) {
    named_attribute_index_[pa->attribute_type()].push_back(att_id);
  }
  pa->set_unique_id(att_id);
  attributes_[att_id] = std::move(pa);
}

void PointCloud::DeleteAttribute(int att_id) {
  if (att_id < 0 || att_id >= attributes_.size()) {
    return;  // Attribute does not exist.
  }
  const GeometryAttribute::Type att_type =
      attributes_[att_id]->attribute_type();
  const uint32_t unique_id = attribute(att_id)->unique_id();
  attributes_.erase(attributes_.begin() + att_id);
  // Remove metadata if applicable.
  if (metadata_) {
    metadata_->DeleteAttributeMetadataByUniqueId(unique_id);
  }

  // Remove the attribute from the named attribute list if applicable.
  if (att_type < GeometryAttribute::NAMED_ATTRIBUTES_COUNT) {
    const auto it = std::find(named_attribute_index_[att_type].begin(),
                              named_attribute_index_[att_type].end(), att_id);
    if (it != named_attribute_index_[att_type].end()) {
      named_attribute_index_[att_type].erase(it);
    }
  }

  // Update ids of all subsequent named attributes (decrease them by one).
  for (int i = 0; i < GeometryAttribute::NAMED_ATTRIBUTES_COUNT; ++i) {
    for (int j = 0; j < named_attribute_index_[i].size(); ++j) {
      if (named_attribute_index_[i][j] > att_id) {
        named_attribute_index_[i][j]--;
      }
    }
  }
}

#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
void PointCloud::DeduplicatePointIds() {
  // Hashing function for a single vertex.
  auto point_hash = [this](PointIndex p) {
    PointIndex::ValueType hash = 0;
    for (int32_t i = 0; i < this->num_attributes(); ++i) {
      const AttributeValueIndex att_id = attribute(i)->mapped_index(p);
      hash = static_cast<uint32_t>(HashCombine(att_id.value(), hash));
    }
    return hash;
  };
  // Comparison function between two vertices.
  auto point_compare = [this](PointIndex p0, PointIndex p1) {
    for (int32_t i = 0; i < this->num_attributes(); ++i) {
      const AttributeValueIndex att_id0 = attribute(i)->mapped_index(p0);
      const AttributeValueIndex att_id1 = attribute(i)->mapped_index(p1);
      if (att_id0 != att_id1) {
        return false;
      }
    }
    return true;
  };

  std::unordered_map<PointIndex, PointIndex, decltype(point_hash),
                     decltype(point_compare)>
      unique_point_map(num_points_, point_hash, point_compare);
  int32_t num_unique_points = 0;
  IndexTypeVector<PointIndex, PointIndex> index_map(num_points_);
  std::vector<PointIndex> unique_points;
  // Go through all vertices and find their duplicates.
  for (PointIndex i(0); i < num_points_; ++i) {
    const auto it = unique_point_map.find(i);
    if (it != unique_point_map.end()) {
      index_map[i] = it->second;
    } else {
      unique_point_map.insert(std::make_pair(i, PointIndex(num_unique_points)));
      index_map[i] = num_unique_points++;
      unique_points.push_back(i);
    }
  }
  if (num_unique_points == num_points_) {
    return;  // All vertices are already unique.
  }

  ApplyPointIdDeduplication(index_map, unique_points);
  set_num_points(num_unique_points);
}

void PointCloud::ApplyPointIdDeduplication(
    const IndexTypeVector<PointIndex, PointIndex> &id_map,
    const std::vector<PointIndex> &unique_point_ids) {
  int32_t num_unique_points = 0;
  for (PointIndex i : unique_point_ids) {
    const PointIndex new_point_id = id_map[i];
    if (new_point_id >= num_unique_points) {
      // New unique vertex reached. Copy attribute indices to the proper
      // position.
      for (int32_t a = 0; a < num_attributes(); ++a) {
        attribute(a)->SetPointMapEntry(new_point_id,
                                       attribute(a)->mapped_index(i));
      }
      num_unique_points = new_point_id.value() + 1;
    }
  }
  for (int32_t a = 0; a < num_attributes(); ++a) {
    attribute(a)->SetExplicitMapping(num_unique_points);
  }
}
#endif

#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
bool PointCloud::DeduplicateAttributeValues() {
  // Go over all attributes and create mapping between duplicate entries.
  if (num_points() == 0) {
    return true;  // Nothing to deduplicate.
  }
  // Deduplicate all attributes.
  for (int32_t att_id = 0; att_id < num_attributes(); ++att_id) {
    if (!attribute(att_id)->DeduplicateValues(*attribute(att_id))) {
      return false;
    }
  }
  return true;
}
#endif

// TODO(b/199760503): Consider to cache the BBox.
BoundingBox PointCloud::ComputeBoundingBox() const {
  BoundingBox bounding_box;
  auto pc_att = GetNamedAttribute(GeometryAttribute::POSITION);
  if (pc_att == nullptr) {
    // Return default invalid bounding box.
    return bounding_box;
  }

  // TODO(b/199760503): Make the BoundingBox a template type, it may not be easy
  // because PointCloud is not a template.
  // Or simply add some preconditioning here to make sure the position attribute
  // is valid, because the current code works only if the position attribute is
  // defined with 3 components of DT_FLOAT32.
  // Consider using pc_att->ConvertValue<float, 3>(i, &p[0]) (Enforced
  // transformation from Vector with any dimension to Vector3f)
  Vector3f p;
  for (AttributeValueIndex i(0); i < static_cast<uint32_t>(pc_att->size());
       ++i) {
    pc_att->GetValue(i, &p[0]);
    bounding_box.Update(p);
  }
  return bounding_box;
}
}  // namespace draco
