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
#ifndef DRACO_POINT_CLOUD_POINT_CLOUD_H_
#define DRACO_POINT_CLOUD_POINT_CLOUD_H_

#include "draco/attributes/point_attribute.h"
#include "draco/core/bounding_box.h"
#include "draco/core/vector_d.h"
#include "draco/draco_features.h"
#include "draco/metadata/geometry_metadata.h"

#ifdef DRACO_TRANSCODER_SUPPORTED
#include "draco/compression/draco_compression_options.h"
#endif

namespace draco {

// PointCloud is a collection of n-dimensional points that are described by a
// set of PointAttributes that can represent data such as positions or colors
// of individual points (see point_attribute.h).
class PointCloud {
 public:
  PointCloud();
  virtual ~PointCloud() = default;

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Copies all data from the |src| point cloud.
  void Copy(const PointCloud &src);
#endif

  // Returns the number of named attributes of a given type.
  int32_t NumNamedAttributes(GeometryAttribute::Type type) const;

  // Returns attribute id of the first named attribute with a given type or -1
  // when the attribute is not used by the point cloud.
  int32_t GetNamedAttributeId(GeometryAttribute::Type type) const;

  // Returns the id of the i-th named attribute of a given type.
  int32_t GetNamedAttributeId(GeometryAttribute::Type type, int i) const;

  // Returns the first named attribute of a given type or nullptr if the
  // attribute is not used by the point cloud.
  const PointAttribute *GetNamedAttribute(GeometryAttribute::Type type) const;

  // Returns the i-th named attribute of a given type.
  const PointAttribute *GetNamedAttribute(GeometryAttribute::Type type,
                                          int i) const;

  // Returns the named attribute of a given unique id.
  const PointAttribute *GetNamedAttributeByUniqueId(
      GeometryAttribute::Type type, uint32_t id) const;

  // Returns the attribute of a given unique id.
  const PointAttribute *GetAttributeByUniqueId(uint32_t id) const;
  int32_t GetAttributeIdByUniqueId(uint32_t unique_id) const;

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Returns the named attribute with a given name.
  const PointAttribute *GetNamedAttributeByName(GeometryAttribute::Type type,
                                                const std::string &name) const;
#endif  // DRACO_TRANSCODER_SUPPORTED

  int32_t num_attributes() const {
    return static_cast<int32_t>(attributes_.size());
  }
  const PointAttribute *attribute(int32_t att_id) const {
    DRACO_DCHECK_LE(0, att_id);
    DRACO_DCHECK_LT(att_id, static_cast<int32_t>(attributes_.size()));
    return attributes_[att_id].get();
  }

  // Returned attribute can be modified, but it's caller's responsibility to
  // maintain the attribute's consistency with draco::PointCloud.
  PointAttribute *attribute(int32_t att_id) {
    DRACO_DCHECK_LE(0, att_id);
    DRACO_DCHECK_LT(att_id, static_cast<int32_t>(attributes_.size()));
    return attributes_[att_id].get();
  }

  // Adds a new attribute to the point cloud.
  // Returns the attribute id.
  int AddAttribute(std::unique_ptr<PointAttribute> pa);

  // Creates and adds a new attribute to the point cloud. The attribute has
  // properties derived from the provided GeometryAttribute |att|.
  // If |identity_mapping| is set to true, the attribute will use identity
  // mapping between point indices and attribute value indices (i.e., each
  // point has a unique attribute value). If |identity_mapping| is false, the
  // mapping between point indices and attribute value indices is set to
  // explicit, and it needs to be initialized manually using the
  // PointAttribute::SetPointMapEntry() method. |num_attribute_values| can be
  // used to specify the number of attribute values that are going to be
  // stored in the newly created attribute. Returns attribute id of the newly
  // created attribute or -1 in case of failure.
  int AddAttribute(const GeometryAttribute &att, bool identity_mapping,
                   AttributeValueIndex::ValueType num_attribute_values);

  // Creates and returns a new attribute or nullptr in case of failure. This
  // method is similar to AddAttribute(), except that it returns the new
  // attribute instead of adding it to the point cloud.
  std::unique_ptr<PointAttribute> CreateAttribute(
      const GeometryAttribute &att, bool identity_mapping,
      AttributeValueIndex::ValueType num_attribute_values) const;

  // Assigns an attribute id to a given PointAttribute. If an attribute with
  // the same attribute id already exists, it is deleted.
  virtual void SetAttribute(int att_id, std::unique_ptr<PointAttribute> pa);

  // Deletes an attribute with specified attribute id. Note that this changes
  // attribute ids of all subsequent attributes.
  virtual void DeleteAttribute(int att_id);

#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
  // Deduplicates all attribute values (all attribute entries with the same
  // value are merged into a single entry).
  virtual bool DeduplicateAttributeValues();
#endif

#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
  // Removes duplicate point ids (two point ids are duplicate when all of their
  // attributes are mapped to the same entry ids).
  virtual void DeduplicatePointIds();
#endif

  // Get bounding box.
  BoundingBox ComputeBoundingBox() const;

  // Add metadata.
  void AddMetadata(std::unique_ptr<GeometryMetadata> metadata) {
    metadata_ = std::move(metadata);
  }

  // Add metadata for an attribute.
  void AddAttributeMetadata(int32_t att_id,
                            std::unique_ptr<AttributeMetadata> metadata) {
    if (!metadata_) {
      metadata_ = std::unique_ptr<GeometryMetadata>(new GeometryMetadata());
    }
    const int32_t att_unique_id = attribute(att_id)->unique_id();
    metadata->set_att_unique_id(att_unique_id);
    metadata_->AddAttributeMetadata(std::move(metadata));
  }

  const AttributeMetadata *GetAttributeMetadataByAttributeId(
      int32_t att_id) const {
    if (metadata_ == nullptr) {
      return nullptr;
    }
    const uint32_t unique_id = attribute(att_id)->unique_id();
    return metadata_->GetAttributeMetadataByUniqueId(unique_id);
  }

  // Returns the attribute metadata that has the requested metadata entry.
  const AttributeMetadata *GetAttributeMetadataByStringEntry(
      const std::string &name, const std::string &value) const {
    if (metadata_ == nullptr) {
      return nullptr;
    }
    return metadata_->GetAttributeMetadataByStringEntry(name, value);
  }

  // Returns the first attribute that has the requested metadata entry.
  int GetAttributeIdByMetadataEntry(const std::string &name,
                                    const std::string &value) const {
    if (metadata_ == nullptr) {
      return -1;
    }
    const AttributeMetadata *att_metadata =
        metadata_->GetAttributeMetadataByStringEntry(name, value);
    if (!att_metadata) {
      return -1;
    }
    return GetAttributeIdByUniqueId(att_metadata->att_unique_id());
  }

  // Get a const pointer of the metadata of the point cloud.
  const GeometryMetadata *GetMetadata() const { return metadata_.get(); }

  // Get a pointer to the metadata of the point cloud.
  GeometryMetadata *metadata() { return metadata_.get(); }

  // Returns the number of n-dimensional points stored within the point cloud.
  PointIndex::ValueType num_points() const { return num_points_; }

  // Sets the number of points. It's the caller's responsibility to ensure the
  // new number is valid with respect to the PointAttributes stored in the point
  // cloud.
  void set_num_points(PointIndex::ValueType num) { num_points_ = num; }

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Enables or disables Draco geometry compression for this mesh.
  void SetCompressionEnabled(bool enabled) { compression_enabled_ = enabled; }
  bool IsCompressionEnabled() const { return compression_enabled_; }

  // Sets |options| that configure Draco geometry compression. This does not
  // enable or disable compression.
  void SetCompressionOptions(const DracoCompressionOptions &options) {
    compression_options_ = options;
  }
  const DracoCompressionOptions &GetCompressionOptions() const {
    return compression_options_;
  }
  DracoCompressionOptions &GetCompressionOptions() {
    return compression_options_;
  }
#endif  // DRACO_TRANSCODER_SUPPORTED

 protected:
#ifdef DRACO_TRANSCODER_SUPPORTED
  // Copies metadata from the |src| point cloud.
  void CopyMetadata(const PointCloud &src);
#endif

#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
  // Applies id mapping of deduplicated points (called by DeduplicatePointIds).
  virtual void ApplyPointIdDeduplication(
      const IndexTypeVector<PointIndex, PointIndex> &id_map,
      const std::vector<PointIndex> &unique_point_ids);
#endif

 private:
  // Metadata for the point cloud.
  std::unique_ptr<GeometryMetadata> metadata_;

  // Attributes describing the point cloud.
  std::vector<std::unique_ptr<PointAttribute>> attributes_;

  // Ids of named attributes of the given type.
  std::vector<int32_t>
      named_attribute_index_[GeometryAttribute::NAMED_ATTRIBUTES_COUNT];

  // The number of n-dimensional points. All point attribute values are stored
  // in corresponding PointAttribute instances in the |attributes_| array.
  PointIndex::ValueType num_points_;

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Compression options for this geometry.
  // TODO(vytyaz): Store encoded bitstream that this geometry compresses into.
  bool compression_enabled_ = false;
  DracoCompressionOptions compression_options_;
#endif  // DRACO_TRANSCODER_SUPPORTED

  friend struct PointCloudHasher;
};

// Functor for computing a hash from data stored within a point cloud.
// Note that this can be quite slow. Two point clouds will have the same hash
// only when all points have the same order and when all attribute values are
// exactly the same.
struct PointCloudHasher {
  size_t operator()(const PointCloud &pc) const {
    size_t hash = pc.num_points_;
    hash = HashCombine(pc.attributes_.size(), hash);
    for (int i = 0; i < GeometryAttribute::NAMED_ATTRIBUTES_COUNT; ++i) {
      hash = HashCombine(pc.named_attribute_index_[i].size(), hash);
      for (int j = 0; j < static_cast<int>(pc.named_attribute_index_[i].size());
           ++j) {
        hash = HashCombine(pc.named_attribute_index_[i][j], hash);
      }
    }
    // Hash attributes.
    for (int i = 0; i < static_cast<int>(pc.attributes_.size()); ++i) {
      PointAttributeHasher att_hasher;
      hash = HashCombine(att_hasher(*pc.attributes_[i]), hash);
    }
    // Hash metadata.
    GeometryMetadataHasher metadata_hasher;
    if (pc.metadata_) {
      hash = HashCombine(metadata_hasher(*pc.metadata_), hash);
    }
    return hash;
  }
};

}  // namespace draco

#endif  // DRACO_POINT_CLOUD_POINT_CLOUD_H_
