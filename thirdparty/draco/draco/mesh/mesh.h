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
#ifndef DRACO_MESH_MESH_H_
#define DRACO_MESH_MESH_H_

#include <memory>
#include <unordered_map>

#include "draco/attributes/geometry_indices.h"
#include "draco/core/hash_utils.h"
#include "draco/core/macros.h"
#include "draco/core/status.h"
#include "draco/draco_features.h"
#ifdef DRACO_TRANSCODER_SUPPORTED
#include "draco/material/material_library.h"
#include "draco/mesh/mesh_features.h"
#include "draco/mesh/mesh_indices.h"
#include "draco/metadata/structural_metadata.h"
#endif
#include "draco/point_cloud/point_cloud.h"

namespace draco {

// List of different variants of mesh attributes.
enum MeshAttributeElementType {
  // All corners attached to a vertex share the same attribute value. A typical
  // example are the vertex positions and often vertex colors.
  MESH_VERTEX_ATTRIBUTE = 0,
  // The most general attribute where every corner of the mesh can have a
  // different attribute value. Often used for texture coordinates or normals.
  MESH_CORNER_ATTRIBUTE,
  // All corners of a single face share the same value.
  MESH_FACE_ATTRIBUTE
};

// Mesh class can be used to represent general triangular meshes. Internally,
// Mesh is just an extended PointCloud with extra connectivity data that defines
// what points are connected together in triangles.
class Mesh : public PointCloud {
 public:
  typedef std::array<PointIndex, 3> Face;

  Mesh();

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Copies all data from the |src| mesh.
  void Copy(const Mesh &src);
#endif

  void AddFace(const Face &face) { faces_.push_back(face); }

  void SetFace(FaceIndex face_id, const Face &face) {
    if (face_id >= static_cast<uint32_t>(faces_.size())) {
      faces_.resize(face_id.value() + 1, Face());
    }
    faces_[face_id] = face;
  }

  // Sets the total number of faces. Creates new empty faces or deletes
  // existing ones if necessary.
  void SetNumFaces(size_t num_faces) { faces_.resize(num_faces, Face()); }

  FaceIndex::ValueType num_faces() const {
    return static_cast<uint32_t>(faces_.size());
  }
  const Face &face(FaceIndex face_id) const {
    DRACO_DCHECK_LE(0, face_id.value());
    DRACO_DCHECK_LT(face_id.value(), static_cast<int>(faces_.size()));
    return faces_[face_id];
  }

  void SetAttribute(int att_id, std::unique_ptr<PointAttribute> pa) override {
    PointCloud::SetAttribute(att_id, std::move(pa));
    if (static_cast<int>(attribute_data_.size()) <= att_id) {
      attribute_data_.resize(att_id + 1);
    }
  }

  void DeleteAttribute(int att_id) override {
    PointCloud::DeleteAttribute(att_id);
    if (att_id >= 0 && att_id < static_cast<int>(attribute_data_.size())) {
      attribute_data_.erase(attribute_data_.begin() + att_id);
    }
#ifdef DRACO_TRANSCODER_SUPPORTED
    UpdateMeshFeaturesAfterDeletedAttribute(att_id);
#endif
  }

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Adds a point attribute |att| to the mesh and returns the index of the
  // newly inserted attribute. Attribute connectivity data is specified in
  // |corner_to_value| array that contains mapping between face corners and
  // attribute value indices.
  // The purpose of this function is to allow users to add attributes with
  // arbitrary connectivity to an existing mesh. New points will be
  // automatically created if needed.
  int32_t AddAttributeWithConnectivity(
      std::unique_ptr<PointAttribute> att,
      const IndexTypeVector<CornerIndex, AttributeValueIndex> &corner_to_value);

  // Adds a point attribute |att| to the mesh and returns the index of the
  // newly inserted attribute. The inserted attribute must have the same
  // connectivity as the position attribute of the mesh (that is, the attribute
  // values are defined per-vertex). Each attribute value entry in |att|
  // corresponds to the corresponding attribute value entry in the position
  // attribute (AttributeValueIndex in both attributes refer to the same
  // spatial vertex).
  // Returns -1 in case of error.
  int32_t AddPerVertexAttribute(std::unique_ptr<PointAttribute> att);

  // Removes points that are not mapped to any face of the mesh. All attribute
  // values are going to be removed as well.
  void RemoveIsolatedPoints();

  // Adds a point attribute |att| to the mesh and returns the index of the
  // newly inserted attribute. Attribute values are mapped 1:1 to face indices.
  // Returns -1 in case of error.
  int32_t AddPerFaceAttribute(std::unique_ptr<PointAttribute> att);
#endif  // DRACO_TRANSCODER_SUPPORTED

  MeshAttributeElementType GetAttributeElementType(int att_id) const {
    return attribute_data_[att_id].element_type;
  }

  void SetAttributeElementType(int att_id, MeshAttributeElementType et) {
    attribute_data_[att_id].element_type = et;
  }

  // Returns the point id of for a corner |ci|.
  inline PointIndex CornerToPointId(int ci) const {
    if (ci < 0 || static_cast<uint32_t>(ci) == kInvalidCornerIndex.value()) {
      return kInvalidPointIndex;
    }
    return this->face(FaceIndex(ci / 3))[ci % 3];
  }

  // Returns the point id of a corner |ci|.
  inline PointIndex CornerToPointId(CornerIndex ci) const {
    return this->CornerToPointId(ci.value());
  }

  struct AttributeData {
    AttributeData() : element_type(MESH_CORNER_ATTRIBUTE) {}
    MeshAttributeElementType element_type;
  };

#ifdef DRACO_TRANSCODER_SUPPORTED
  void SetName(const std::string &name) { name_ = name; }
  const std::string &GetName() const { return name_; }
  const MaterialLibrary &GetMaterialLibrary() const {
    return material_library_;
  }
  MaterialLibrary &GetMaterialLibrary() { return material_library_; }

  // Removes all materials that are not referenced by any face of the mesh.
  // Optional argument |remove_unused_material_indices| can be used to control
  // whether unusued material indices are removed as well (default = true).
  // If material indices are not removed, the unused material indices will
  // point to empty (default) materials.
  void RemoveUnusedMaterials();
  void RemoveUnusedMaterials(bool remove_unused_material_indices);

  // Library that contains non-material textures.
  const TextureLibrary &GetNonMaterialTextureLibrary() const {
    return non_material_texture_library_;
  }
  TextureLibrary &GetNonMaterialTextureLibrary() {
    return non_material_texture_library_;
  }

  // Mesh feature ID sets as defined by EXT_mesh_features glTF extension.
  MeshFeaturesIndex AddMeshFeatures(
      std::unique_ptr<MeshFeatures> mesh_features) {
    mesh_features_.push_back(std::move(mesh_features));
    mesh_features_material_mask_.push_back({});
    return MeshFeaturesIndex(mesh_features_.size() - 1);
  }
  int NumMeshFeatures() const { return mesh_features_.size(); }
  const MeshFeatures &GetMeshFeatures(MeshFeaturesIndex index) const {
    return *mesh_features_[index];
  }
  MeshFeatures &GetMeshFeatures(MeshFeaturesIndex index) {
    return *mesh_features_[index];
  }

  // Removes mesh features from the mesh. Note that removing a mesh feature does
  // not delete any associated data such as vertex attributes or feature
  // textures.
  void RemoveMeshFeatures(MeshFeaturesIndex index) {
    mesh_features_.erase(mesh_features_.begin() + index.value());
    mesh_features_material_mask_.erase(mesh_features_material_mask_.begin() +
                                       index.value());
  }

  // Returns true if an attribute with |att_id| is being used by any mesh
  // features attached to the mesh.
  bool IsAttributeUsedByMeshFeatures(int att_id) const;

  // Restricts given mesh features to faces mapped to a material with
  // |material_index|. Note that single mesh features can be restricted to
  // multiple materials.
  void AddMeshFeaturesMaterialMask(MeshFeaturesIndex index,
                                   int material_index) {
    mesh_features_material_mask_[index].push_back(material_index);
  }

  size_t NumMeshFeaturesMaterialMasks(MeshFeaturesIndex index) const {
    return mesh_features_material_mask_[index].size();
  }
  int GetMeshFeaturesMaterialMask(MeshFeaturesIndex index,
                                  int mask_index) const {
    return mesh_features_material_mask_[index][mask_index];
  }

  // Updates mesh features texture pointer to point to a new |texture_library|.
  // The current texture pointer is used to determine the texture index in the
  // new texture library via a given |texture_to_index_map|.
  static void UpdateMeshFeaturesTexturePointer(
      const std::unordered_map<const Texture *, int> &texture_to_index_map,
      TextureLibrary *texture_library, MeshFeatures *mesh_features);

  // Copies over mesh features from |source_mesh| and stores them in
  // |target_mesh| as long as the mesh features material mask is valid for
  // given |material_index|.
  static void CopyMeshFeaturesForMaterial(const Mesh &source_mesh,
                                          Mesh *target_mesh,
                                          int material_index);

  // Structural metadata.
  const StructuralMetadata &GetStructuralMetadata() const {
    return structural_metadata_;
  }
  StructuralMetadata &GetStructuralMetadata() { return structural_metadata_; }

  // Property attributes indices as defined by EXT_structural_metadata glTF
  // extension.
  int AddPropertyAttributesIndex(int property_attribute_index) {
    property_attributes_.push_back(property_attribute_index);
    property_attributes_material_mask_.push_back({});
    return property_attributes_.size() - 1;
  }
  int NumPropertyAttributesIndices() const {
    return property_attributes_.size();
  }
  const int &GetPropertyAttributesIndex(int index) const {
    return property_attributes_[index];
  }
  int &GetPropertyAttributesIndex(int index) {
    return property_attributes_[index];
  }
  void RemovePropertyAttributesIndex(int index) {
    property_attributes_.erase(property_attributes_.begin() + index);
    property_attributes_material_mask_.erase(
        property_attributes_material_mask_.begin() + index);
  }

  // Restricts given property attributes indices to faces mapped to a material
  // with |material_index|. Note that single property attribute can be
  // restricted to multiple materials.
  void AddPropertyAttributesIndexMaterialMask(int index, int material_index) {
    property_attributes_material_mask_[index].push_back(material_index);
  }

  size_t NumPropertyAttributesIndexMaterialMasks(int index) const {
    return property_attributes_material_mask_[index].size();
  }
  int GetPropertyAttributesIndexMaterialMask(int index, int mask_index) const {
    return property_attributes_material_mask_[index][mask_index];
  }

  // Copies over property attributes indices from |source_mesh| and stores them
  // in |target_mesh| as long as the property attributes indices material mask
  // is valid for given |material_index|.
  static void CopyPropertyAttributesIndicesForMaterial(const Mesh &source_mesh,
                                                       Mesh *target_mesh,
                                                       int material_index);
#endif  // DRACO_TRANSCODER_SUPPORTED

 protected:
#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
  // Extends the point deduplication to face corners. This method is called from
  // the PointCloud::DeduplicatePointIds() and it remaps all point ids stored in
  // |faces_| to the new deduplicated point ids using the map |id_map|.
  void ApplyPointIdDeduplication(
      const IndexTypeVector<PointIndex, PointIndex> &id_map,
      const std::vector<PointIndex> &unique_point_ids) override;
#endif

  // Exposes |faces_|. Use |faces_| at your own risk. DO NOT store the
  // reference: the |faces_| object is destroyed with the mesh.
  IndexTypeVector<FaceIndex, Face> &faces() { return faces_; }

 private:
#ifdef DRACO_TRANSCODER_SUPPORTED
  // Updates attribute indices associated to all mesh features after a mesh
  // attribute is deleted.
  void UpdateMeshFeaturesAfterDeletedAttribute(int att_id);
#endif
  // Mesh specific per-attribute data.
  std::vector<AttributeData> attribute_data_;

  // Vertex indices valid for all attributes. Each attribute has its own map
  // that converts vertex indices into attribute indices.
  IndexTypeVector<FaceIndex, Face> faces_;

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Mesh name.
  std::string name_;

  // Materials applied to to this mesh.
  MaterialLibrary material_library_;

  // Sets of feature IDs as defined by EXT_mesh_features glTF extension.
  IndexTypeVector<MeshFeaturesIndex, std::unique_ptr<MeshFeatures>>
      mesh_features_;

  // When the Mesh contains multiple materials, |mesh_features_material_mask_|
  // can be used to limit specific MeshFeaturesIndex to a vector of material
  // indices. If for a given mesh feature index, the material indices are empty,
  // the corresponding mesh features are applied to the entire mesh.
  IndexTypeVector<MeshFeaturesIndex, std::vector<int>>
      mesh_features_material_mask_;

  // Indices pointing to property attributes in draco::StructuralMetadata.
  std::vector<int> property_attributes_;

  // When the Mesh contains multiple materials, this mask can be used to limit
  // specific index into |property_attributes_| to a vector of material indices.
  // If for a given property attributes index, the material indices are empty,
  // the corresponding property attributes are applied to the entire mesh.
  std::vector<std::vector<int>> property_attributes_material_mask_;

  // Texture library for storing non-material textures used by this mesh, e.g.,
  // textures containing mesh feature IDs of EXT_mesh_features glTF extension.
  // If the mesh is part of the scene then the textures are stored in the scene.
  // Note that mesh features contain pointers to non-material textures. It is
  // responsibility of class user to update these pointers when updating the
  // textures. See Mesh::Copy() for example.
  TextureLibrary non_material_texture_library_;

  // Structural metadata defined by the EXT_structural_metadata glTF extension.
  StructuralMetadata structural_metadata_;
#endif  // DRACO_TRANSCODER_SUPPORTED
  friend struct MeshHasher;
};

// Functor for computing a hash from data stored within a mesh.
// Note that this can be quite slow. Two meshes will have the same hash only
// when they have exactly the same connectivity and attribute values.
struct MeshHasher {
  size_t operator()(const Mesh &mesh) const {
    PointCloudHasher pc_hasher;
    size_t hash = pc_hasher(mesh);
    // Hash faces.
    for (FaceIndex i(0); i < static_cast<uint32_t>(mesh.faces_.size()); ++i) {
      for (int j = 0; j < 3; ++j) {
        hash = HashCombine(mesh.faces_[i][j].value(), hash);
      }
    }
    return hash;
  }
};

}  // namespace draco

#endif  // DRACO_MESH_MESH_H_
