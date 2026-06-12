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
#include "draco/mesh/mesh.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace draco {

// Shortcut for typed conditionals.
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

Mesh::Mesh() {}

#ifdef DRACO_TRANSCODER_SUPPORTED
void Mesh::Copy(const Mesh &src) {
  PointCloud::Copy(src);
  name_ = src.name_;
  faces_ = src.faces_;
  attribute_data_ = src.attribute_data_;
  material_library_.Copy(src.material_library_);

  // Copy mesh feature ID sets.
  mesh_features_.clear();
  for (MeshFeaturesIndex i(0); i < src.NumMeshFeatures(); i++) {
    std::unique_ptr<MeshFeatures> mesh_features(new MeshFeatures());
    mesh_features->Copy(src.GetMeshFeatures(i));
    AddMeshFeatures(std::move(mesh_features));
  }
  mesh_features_material_mask_ = src.mesh_features_material_mask_;

  // Copy non-material textures.
  non_material_texture_library_.Copy(src.non_material_texture_library_);

  // Update pointers to non-material textures in mesh feature ID sets.
  if (non_material_texture_library_.NumTextures() != 0) {
    const auto texture_to_index_map =
        src.non_material_texture_library_.ComputeTextureToIndexMap();
    for (MeshFeaturesIndex j(0); j < NumMeshFeatures(); ++j) {
      Mesh::UpdateMeshFeaturesTexturePointer(texture_to_index_map,
                                             &non_material_texture_library_,
                                             &GetMeshFeatures(j));
    }
  }

  // Copy structural metadata.
  structural_metadata_.Copy(src.structural_metadata_);
  property_attributes_ = src.property_attributes_;
  property_attributes_material_mask_ = src.property_attributes_material_mask_;
}

namespace {
// A helper struct that augments a point index with an attribute value index.
// A unique combination of |point_index| and |attribute_value_index|
// corresponds to a unique point on the mesh. Used to identify unique points
// after a new attribute is added to the mesh.
struct AugmentedPointData {
  PointIndex point_index;
  AttributeValueIndex attribute_value_index;
  bool operator<(const AugmentedPointData &pd) const {
    if (point_index < pd.point_index) {
      return true;
    }
    if (point_index > pd.point_index) {
      return false;
    }
    return attribute_value_index < pd.attribute_value_index;
  }
};
}  // namespace

int32_t Mesh::AddAttributeWithConnectivity(
    std::unique_ptr<PointAttribute> att,
    const IndexTypeVector<CornerIndex, AttributeValueIndex> &corner_to_value) {
  // Map between augmented point and new point indices (one augmented point
  // corresponds to one PointIndex).
  std::map<AugmentedPointData, PointIndex> old_to_new_point_map;

  // Map between corners and the new point indices.
  IndexTypeVector<CornerIndex, PointIndex> corner_to_point(num_faces() * 3,
                                                           kInvalidPointIndex);

  // Flag whether a given existing point index has been used. Used to ensure
  // that mapping between existing and new point indices that are smaller
  // than num_points() is identity. In other words, we want to keep indices of
  // the existing points intact and add new points to end.
  IndexTypeVector<PointIndex, bool> is_point_used(num_points(), false);

  int new_num_points = num_points();
  for (CornerIndex ci(0); ci < num_faces() * 3; ++ci) {
    AugmentedPointData apd;
    apd.point_index = CornerToPointId(ci);
    apd.attribute_value_index = corner_to_value[ci];
    const auto it = old_to_new_point_map.find(apd);
    if (it != old_to_new_point_map.end()) {
      // Augmented point is already mapped to a point index. Reuse it.
      corner_to_point[ci] = it->second;
    } else {
      // New combination of point index + attribute value index. Map it to a
      // unique point index.
      PointIndex new_point_index;
      if (!is_point_used[apd.point_index]) {
        // Reuse the existing (old) point index.
        new_point_index = apd.point_index;
        is_point_used[apd.point_index] = true;
      } else {
        // Add a new point index to the end.
        new_point_index = PointIndex(new_num_points++);
      }
      old_to_new_point_map[apd] = new_point_index;
      corner_to_point[ci] = new_point_index;
    }
  }

  // Update point to attribute value mapping for the new attribute.
  att->SetExplicitMapping(new_num_points);
  for (CornerIndex ci(0); ci < num_faces() * 3; ++ci) {
    att->SetPointMapEntry(corner_to_point[ci], corner_to_value[ci]);
  }

  // Update point to attribute value mapping on the remaining attributes if
  // needed.
  if (new_num_points > num_points()) {
    set_num_points(new_num_points);

    // Setup attributes for the new number of points.
    for (int ai = 0; ai < num_attributes(); ++ai) {
      const bool mapping_was_identity = attribute(ai)->is_mapping_identity();
      attribute(ai)->SetExplicitMapping(new_num_points);
      if (mapping_was_identity) {
        // Convert all old points from identity to explicit mapping.
        for (AttributeValueIndex avi(0); avi < attribute(ai)->size(); ++avi) {
          attribute(ai)->SetPointMapEntry(PointIndex(avi.value()), avi);
        }
      }
    }

    for (CornerIndex ci(0); ci < num_faces() * 3; ++ci) {
      const PointIndex old_point_index = CornerToPointId(ci);
      const PointIndex new_point_index = corner_to_point[ci];
      if (old_point_index == new_point_index) {
        continue;
      }
      // Update point to value mapping for all existing attributes.
      for (int ai = 0; ai < num_attributes(); ++ai) {
        attribute(ai)->SetPointMapEntry(
            new_point_index, attribute(ai)->mapped_index(old_point_index));
      }
      // Update mapping between the corner and the new point index.
      faces_[FaceIndex(ci.value() / 3)][ci.value() % 3] = new_point_index;
    }
  }

  // If any of the old points have not been used, initialize dummy mapping for
  // the new attribute.
  for (PointIndex pi(0); pi < is_point_used.size(); ++pi) {
    if (!is_point_used[pi]) {
      att->SetPointMapEntry(pi, AttributeValueIndex(0));
    }
  }

  return PointCloud::AddAttribute(std::move(att));
}

int32_t Mesh::AddPerVertexAttribute(std::unique_ptr<PointAttribute> att) {
  const PointAttribute *const pos_att =
      GetNamedAttribute(GeometryAttribute::POSITION);
  if (pos_att == nullptr) {
    return -1;
  }
  if (att->size() != pos_att->size()) {
    return -1;  // Number of values must be same as in the position attribute.
  }

  if (pos_att->is_mapping_identity()) {
    att->SetIdentityMapping();
  } else {
    // Copy point to attribute value mapping from the position attribute to
    // |att|.
    att->SetExplicitMapping(num_points());
    for (PointIndex pi(0); pi < num_points(); ++pi) {
      att->SetPointMapEntry(pi, pos_att->mapped_index(pi));
    }
  }

  return PointCloud::AddAttribute(std::move(att));
}

void Mesh::RemoveIsolatedPoints() {
  // For each point, check if it is mapped to a face.
  IndexTypeVector<PointIndex, bool> is_point_used(num_points(), false);
  int num_used_points = 0;
  for (FaceIndex fi(0); fi < num_faces(); ++fi) {
    const auto &f = face(fi);
    for (int c = 0; c < 3; ++c) {
      if (!is_point_used[f[c]]) {
        num_used_points++;
        is_point_used[f[c]] = true;
      }
    }
  }
  if (num_used_points == num_points()) {
    return;  // All points are used.
  }

  // Create mapping between the old and new point indices.
  IndexTypeVector<PointIndex, PointIndex> old_to_new_point_map(
      num_points(), kInvalidPointIndex);
  PointIndex new_point_index(0);
  for (PointIndex pi(0); pi < num_points(); ++pi) {
    if (is_point_used[pi]) {
      old_to_new_point_map[pi] = new_point_index++;
    }
  }

  // Update point to attribute value index map for all attributes.
  for (int ai = 0; ai < num_attributes(); ++ai) {
    PointAttribute *att = attribute(ai);
    if (att->is_mapping_identity()) {
      // When the attribute uses identity mapping we need to reorder to the
      // attribute values to match the new point indices.
      for (PointIndex pi(0); pi < num_points(); ++pi) {
        const PointIndex new_pi = old_to_new_point_map[pi];
        if (new_pi == pi || new_pi == kInvalidPointIndex) {
          continue;
        }
        att->SetAttributeValue(
            AttributeValueIndex(new_pi.value()),
            att->GetAddress(AttributeValueIndex(pi.value())));
      }
      att->Resize(num_used_points);
    } else {
      // For explicitly mapped attributes, we first update the point to
      // attribute value mapping and then we remove all unused values from the
      // attribute.
      for (PointIndex pi(0); pi < num_points(); ++pi) {
        const PointIndex new_pi = old_to_new_point_map[pi];
        if (new_pi == pi || new_pi == kInvalidPointIndex) {
          continue;
        }
        att->SetPointMapEntry(new_pi, att->mapped_index(pi));
      }
      att->SetExplicitMapping(num_used_points);

      att->RemoveUnusedValues();
    }
  }

  // Update the mapping between faces and point indices.
  for (FaceIndex fi(0); fi < num_faces(); ++fi) {
    auto &f = faces_[fi];
    for (int c = 0; c < 3; ++c) {
      f[c] = old_to_new_point_map[f[c]];
    }
  }

  set_num_points(num_used_points);
}

void Mesh::RemoveUnusedMaterials() { RemoveUnusedMaterials(true); }

void Mesh::RemoveUnusedMaterials(bool remove_unused_material_indices) {
  const int mat_att_index = GetNamedAttributeId(GeometryAttribute::MATERIAL);
  if (mat_att_index == -1) {
    // Remove all materials except for the first one.
    while (GetMaterialLibrary().NumMaterials() > 1) {
      GetMaterialLibrary().RemoveMaterial(1);
    }
    GetMaterialLibrary().RemoveUnusedTextures();
    return;
  }
  auto mat_att = attribute(mat_att_index);

  // Deduplicate attribute values in the material attribute to ensure that one
  // attribute value index corresponds to one unique material index.
  // Note that this does not remove unused material indices.
  mat_att->DeduplicateValues(*mat_att);

  // Gather all material indices that are referenced by faces of the mesh.
  const int num_materials = GetMaterialLibrary().NumMaterials();
  std::vector<bool> is_material_used(num_materials, false);
  int num_used_materials = 0;

  // Helper function that updates |is_material_used| for the processed mesh.
  auto update_used_materials = [&is_material_used, &num_used_materials, mat_att,
                                num_materials](PointIndex pi) {
    uint32_t mat_index = 0;
    mat_att->GetMappedValue(pi, &mat_index);
    if (mat_index < num_materials) {
      if (!is_material_used[mat_index]) {
        is_material_used[mat_index] = true;
        num_used_materials++;
      }
    }
  };

  if (num_faces() > 0) {
    for (FaceIndex fi(0); fi < num_faces(); ++fi) {
      update_used_materials(faces_[fi][0]);
    }
  } else {
    // Handle the mesh as a point cloud and check materials used by points.
    for (PointIndex pi(0); pi < num_points(); ++pi) {
      update_used_materials(pi);
    }
  }

  // Check if any of the (unused) materials is used by mesh features. If so,
  // user should remove unused mesh features first.
  for (MeshFeaturesIndex mfi(0); mfi < NumMeshFeatures(); ++mfi) {
    for (int mask_index = 0; mask_index < NumMeshFeaturesMaterialMasks(mfi);
         ++mask_index) {
      const int mat_index = GetMeshFeaturesMaterialMask(mfi, mask_index);
      if (mat_index < num_materials && !is_material_used[mat_index]) {
        is_material_used[mat_index] = true;
        num_used_materials++;
      }
    }
  }

  // Check if any of the (unused) materials is used by property attributes
  // indices. If so, user should remove unused property attributes indices
  // first.
  for (int i = 0; i < NumPropertyAttributesIndices(); ++i) {
    for (int mask_index = 0;
         mask_index < NumPropertyAttributesIndexMaterialMasks(i);
         ++mask_index) {
      const int mat_index =
          GetPropertyAttributesIndexMaterialMask(i, mask_index);
      if (mat_index < num_materials && !is_material_used[mat_index]) {
        is_material_used[mat_index] = true;
        num_used_materials++;
      }
    }
  }

  if (num_used_materials == num_materials) {
    return;  // All materials are used, don't do anything.
  }

  // Remove unused materials from the material library or replace them with
  // default materials if we do not remove unused material indices.
  for (int mi = num_materials - 1; mi >= 0; --mi) {
    if (!is_material_used[mi] && mi < GetMaterialLibrary().NumMaterials()) {
      if (remove_unused_material_indices) {
        GetMaterialLibrary().RemoveMaterial(mi);
      } else {
        GetMaterialLibrary().MutableMaterial(mi)->Clear();
      }
    }
  }
  GetMaterialLibrary().RemoveUnusedTextures();

  if (!remove_unused_material_indices) {
    // All the code below handles updating of material indices. Since we do not
    // want to update them, we can return early.
    return;
  }

  // Compute map between old and new material indices.
  std::vector<int> old_to_new_material_index_map(num_materials, -1);
  for (int mi = 0, new_material_index = 0; mi < num_materials; ++mi) {
    if (is_material_used[mi]) {
      old_to_new_material_index_map[mi] = new_material_index;
      ++new_material_index;
    }
  }
  IndexTypeVector<AttributeValueIndex, int>
      old_to_new_material_attribute_value_index_map(mat_att->size(), -1);
  for (AttributeValueIndex avi(0); avi < mat_att->size(); ++avi) {
    uint32_t mat_index = 0;
    mat_att->GetValue(avi, &mat_index);
    if (mat_index < num_materials && is_material_used[mat_index]) {
      old_to_new_material_attribute_value_index_map[avi] =
          old_to_new_material_index_map[mat_index];
    }
  }

  // Update attribute values with the new number of materials.
  mat_att->Reset(num_used_materials);

  // Set identity mapping between AttributeValueIndex and material indices.
  for (AttributeValueIndex avi(0); avi < mat_att->size(); ++avi) {
    const uint32_t mat_index = avi.value();
    mat_att->SetAttributeValue(avi, &mat_index);
  }

  // Update mapping between points and attribute values.
  for (PointIndex pi(0); pi < num_points(); ++pi) {
    const AttributeValueIndex old_avi = mat_att->mapped_index(pi);
    mat_att->SetPointMapEntry(
        pi, AttributeValueIndex(
                old_to_new_material_attribute_value_index_map[old_avi]));
  }

  // Update material indices on mesh features.
  for (MeshFeaturesIndex mfi(0); mfi < NumMeshFeatures(); ++mfi) {
    for (int mask_index = 0; mask_index < NumMeshFeaturesMaterialMasks(mfi);
         ++mask_index) {
      const int old_mat_index = GetMeshFeaturesMaterialMask(mfi, mask_index);
      if (old_mat_index < num_materials && is_material_used[old_mat_index]) {
        mesh_features_material_mask_[mfi][mask_index] =
            old_to_new_material_index_map[old_mat_index];
      }
    }
  }

  // Update material indices on property attributes incices.
  for (int i = 0; i < NumPropertyAttributesIndices(); ++i) {
    for (int mask_index = 0;
         mask_index < NumPropertyAttributesIndexMaterialMasks(i);
         ++mask_index) {
      const int old_mat_index =
          GetPropertyAttributesIndexMaterialMask(i, mask_index);
      if (old_mat_index < num_materials && is_material_used[old_mat_index]) {
        property_attributes_material_mask_[i][mask_index] =
            old_to_new_material_index_map[old_mat_index];
      }
    }
  }
}

bool Mesh::IsAttributeUsedByMeshFeatures(int att_id) const {
  for (MeshFeaturesIndex mfi(0); mfi < NumMeshFeatures(); ++mfi) {
    const auto &mf = GetMeshFeatures(mfi);
    if (mf.GetAttributeIndex() == att_id) {
      return true;
    }
  }
  return false;
}

void Mesh::UpdateMeshFeaturesTexturePointer(
    const std::unordered_map<const Texture *, int> &texture_to_index_map,
    TextureLibrary *texture_library, MeshFeatures *mesh_features) {
  TextureMap &texture_map = mesh_features->GetTextureMap();
  if (texture_map.texture() == nullptr) {
    return;
  }
  const auto it = texture_to_index_map.find(texture_map.texture());
  DRACO_DCHECK(it != texture_to_index_map.end());
  const int texture_index = it->second;
  DRACO_DCHECK(texture_index < texture_library->NumTextures());
  texture_map.SetTexture(texture_library->GetTexture(texture_index));
}

void Mesh::CopyMeshFeaturesForMaterial(const Mesh &source_mesh,
                                       Mesh *target_mesh, int material_index) {
  for (MeshFeaturesIndex mfi(0); mfi < source_mesh.NumMeshFeatures(); ++mfi) {
    // Mesh features is used if it doesn't have any material mask or if one
    // of the material masks matches |material_index|.
    bool is_used = source_mesh.NumMeshFeaturesMaterialMasks(mfi) == 0;
    for (int mask_index = 0;
         !is_used && mask_index < source_mesh.NumMeshFeaturesMaterialMasks(mfi);
         ++mask_index) {
      if (source_mesh.GetMeshFeaturesMaterialMask(mfi, mask_index) ==
          material_index) {
        is_used = true;
      }
    }
    if (is_used) {
      // Copy over the mesh features to the target mesh. Note that texture
      // pointers are not updated at this step.
      std::unique_ptr<MeshFeatures> new_mf(new MeshFeatures());
      new_mf->Copy(source_mesh.GetMeshFeatures(mfi));
      target_mesh->AddMeshFeatures(std::move(new_mf));
    }
  }
}

void Mesh::CopyPropertyAttributesIndicesForMaterial(const Mesh &source_mesh,
                                                    Mesh *target_mesh,
                                                    int material_index) {
  for (int i = 0; i < source_mesh.NumPropertyAttributesIndices(); ++i) {
    // Property attributes index is used if it doesn't have any material mask or
    // if one of the material masks matches |material_index|.
    bool is_used = source_mesh.NumPropertyAttributesIndexMaterialMasks(i) == 0;
    for (int mask_index = 0;
         !is_used &&
         mask_index < source_mesh.NumPropertyAttributesIndexMaterialMasks(i);
         ++mask_index) {
      if (source_mesh.GetPropertyAttributesIndexMaterialMask(i, mask_index) ==
          material_index) {
        is_used = true;
      }
    }
    if (is_used) {
      // Copy over the property attributes index to the target mesh.
      target_mesh->AddPropertyAttributesIndex(
          source_mesh.GetPropertyAttributesIndex(i));
    }
  }
}

void Mesh::UpdateMeshFeaturesAfterDeletedAttribute(int att_id) {
  for (MeshFeaturesIndex mfi(0); mfi < NumMeshFeatures(); ++mfi) {
    auto &mf = GetMeshFeatures(mfi);
    if (mf.GetAttributeIndex() == att_id) {
      // Mesh features is no longer associated with a vertex attribute.
      mf.SetAttributeIndex(-1);
    } else if (mf.GetAttributeIndex() > att_id) {
      // Attribute index decremented by one.
      mf.SetAttributeIndex(mf.GetAttributeIndex() - 1);
    }
  }
}

int32_t Mesh::AddPerFaceAttribute(std::unique_ptr<PointAttribute> att) {
  IndexTypeVector<CornerIndex, AttributeValueIndex> corner_map(num_faces() * 3);
  for (CornerIndex ci(0); ci < num_faces() * 3; ++ci) {
    corner_map[ci] = AttributeValueIndex(ci.value() / 3);
  }
  return AddAttributeWithConnectivity(std::move(att), corner_map);
}
#endif  // DRACO_TRANSCODER_SUPPORTED

#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
void Mesh::ApplyPointIdDeduplication(
    const IndexTypeVector<PointIndex, PointIndex> &id_map,
    const std::vector<PointIndex> &unique_point_ids) {
  PointCloud::ApplyPointIdDeduplication(id_map, unique_point_ids);
  for (FaceIndex f(0); f < num_faces(); ++f) {
    for (int32_t c = 0; c < 3; ++c) {
      faces_[f][c] = id_map[faces_[f][c]];
    }
  }
}
#endif

}  // namespace draco
