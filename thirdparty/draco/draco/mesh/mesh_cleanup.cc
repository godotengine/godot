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
#include "draco/mesh/mesh_cleanup.h"

#include <array>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "draco/core/hash_utils.h"

namespace draco {

Status MeshCleanup::Cleanup(Mesh *mesh, const MeshCleanupOptions &options) {
  if (!options.remove_degenerated_faces && !options.remove_unused_attributes &&
      !options.remove_duplicate_faces && !options.make_geometry_manifold) {
    return OkStatus();  // Nothing to cleanup.
  }
  const PointAttribute *const pos_att =
      mesh->GetNamedAttribute(GeometryAttribute::POSITION);
  if (pos_att == nullptr) {
    return Status(Status::DRACO_ERROR, "Missing position attribute.");
  }

  if (options.remove_degenerated_faces) {
    RemoveDegeneratedFaces(mesh);
  }

  if (options.remove_duplicate_faces) {
    RemoveDuplicateFaces(mesh);
  }

  if (options.remove_unused_attributes) {
    RemoveUnusedAttributes(mesh);
  }

  return OkStatus();
}

void MeshCleanup::RemoveDegeneratedFaces(Mesh *mesh) {
  const PointAttribute *const pos_att =
      mesh->GetNamedAttribute(GeometryAttribute::POSITION);
  FaceIndex::ValueType num_degenerated_faces = 0;
  // Array for storing position indices on a face.
  std::array<AttributeValueIndex, 3> pos_indices;
  for (FaceIndex f(0); f < mesh->num_faces(); ++f) {
    const Mesh::Face &face = mesh->face(f);
    for (int p = 0; p < 3; ++p) {
      pos_indices[p] = pos_att->mapped_index(face[p]);
    }
    if (pos_indices[0] == pos_indices[1] || pos_indices[0] == pos_indices[2] ||
        pos_indices[1] == pos_indices[2]) {
      ++num_degenerated_faces;
    } else if (num_degenerated_faces > 0) {
      // Copy the face to its new location.
      mesh->SetFace(f - num_degenerated_faces, face);
    }
  }
  if (num_degenerated_faces > 0) {
    mesh->SetNumFaces(mesh->num_faces() - num_degenerated_faces);
  }
}

void MeshCleanup::RemoveDuplicateFaces(Mesh *mesh) {
  std::unordered_set<Mesh::Face, HashArray<Mesh::Face>> is_face_used;

  uint32_t num_duplicate_faces = 0;
  for (FaceIndex fi(0); fi < mesh->num_faces(); ++fi) {
    auto face = mesh->face(fi);

    // Shift the face indices until the smallest index is the first one.
    while (face[0] > face[1] || face[0] > face[2]) {
      // Shift to the left.
      std::swap(face[0], face[1]);
      std::swap(face[1], face[2]);
    }
    // Check if have encountered the same face before.
    if (is_face_used.find(face) != is_face_used.end()) {
      // Duplicate face. Ignore it.
      num_duplicate_faces++;
    } else {
      // Insert new face to the set.
      is_face_used.insert(face);
      if (num_duplicate_faces > 0) {
        // Copy the face to its new location.
        mesh->SetFace(fi - num_duplicate_faces, face);
      }
    }
  }
  if (num_duplicate_faces > 0) {
    mesh->SetNumFaces(mesh->num_faces() - num_duplicate_faces);
  }
}

void MeshCleanup::RemoveUnusedAttributes(Mesh *mesh) {
  // Array that is going to store whether a corresponding point is used.
  std::vector<bool> is_point_used;
  PointIndex::ValueType num_new_points = 0;
  is_point_used.resize(mesh->num_points(), false);
  for (FaceIndex f(0); f < mesh->num_faces(); ++f) {
    const Mesh::Face &face = mesh->face(f);
    for (int p = 0; p < 3; ++p) {
      if (!is_point_used[face[p].value()]) {
        is_point_used[face[p].value()] = true;
        ++num_new_points;
      }
    }
  }

  bool points_changed = false;
  const PointIndex::ValueType num_original_points = mesh->num_points();
  // Map from old points to the new ones.
  IndexTypeVector<PointIndex, PointIndex> point_map(num_original_points);
  if (num_new_points < static_cast<int>(mesh->num_points())) {
    // Some of the points were removed. We need to remap the old points to the
    // new ones.
    num_new_points = 0;
    for (PointIndex i(0); i < num_original_points; ++i) {
      if (is_point_used[i.value()]) {
        point_map[i] = num_new_points++;
      } else {
        point_map[i] = kInvalidPointIndex;
      }
    }
    // Go over faces and update their points.
    for (FaceIndex f(0); f < mesh->num_faces(); ++f) {
      Mesh::Face face = mesh->face(f);
      for (int p = 0; p < 3; ++p) {
        face[p] = point_map[face[p]];
      }
      mesh->SetFace(f, face);
    }
    // Set the new number of points.
    mesh->set_num_points(num_new_points);
    points_changed = true;
  } else {
    // No points were removed. Initialize identity map between the old and new
    // points.
    for (PointIndex i(0); i < num_original_points; ++i) {
      point_map[i] = i;
    }
  }

  // Update index mapping for attributes.
  IndexTypeVector<AttributeValueIndex, uint8_t> is_att_index_used;
  IndexTypeVector<AttributeValueIndex, AttributeValueIndex> att_index_map;
  for (int a = 0; a < mesh->num_attributes(); ++a) {
    PointAttribute *const att = mesh->attribute(a);
    // First detect which attribute entries are used (included in a point).
    is_att_index_used.assign(att->size(), 0);
    att_index_map.clear();
    AttributeValueIndex::ValueType num_used_entries = 0;
    for (PointIndex i(0); i < num_original_points; ++i) {
      if (point_map[i] != kInvalidPointIndex) {
        const AttributeValueIndex entry_id = att->mapped_index(i);
        if (!is_att_index_used[entry_id]) {
          is_att_index_used[entry_id] = 1;
          ++num_used_entries;
        }
      }
    }
    bool att_indices_changed = false;
    // If there are some unused attribute entries, remap the attribute values
    // in the attribute buffer.
    if (num_used_entries < static_cast<int>(att->size())) {
      att_index_map.resize(att->size());
      num_used_entries = 0;
      for (AttributeValueIndex i(0); i < static_cast<uint32_t>(att->size());
           ++i) {
        if (is_att_index_used[i]) {
          att_index_map[i] = num_used_entries;
          if (i > num_used_entries) {
            const uint8_t *const src_add = att->GetAddress(i);
            att->buffer()->Write(
                att->GetBytePos(AttributeValueIndex(num_used_entries)), src_add,
                att->byte_stride());
          }
          ++num_used_entries;
        }
      }
      // Update the number of unique entries in the vertex buffer.
      att->Resize(num_used_entries);
      att_indices_changed = true;
    }
    // If either the points or attribute indices have changed, we need to
    // update the attribute index mapping.
    if (points_changed || att_indices_changed) {
      if (att->is_mapping_identity()) {
        // The mapping was identity. It'll remain identity only if the
        // number of point and attribute indices is still the same.
        if (num_used_entries != static_cast<int>(mesh->num_points())) {
          // We need to create an explicit mapping.
          // First we need to initialize the explicit map to the original
          // number of points to recreate the original identity map.
          att->SetExplicitMapping(num_original_points);
          // Set the entries of the explicit map to identity.
          for (PointIndex::ValueType i = 0; i < num_original_points; ++i) {
            att->SetPointMapEntry(PointIndex(i), AttributeValueIndex(i));
          }
        }
      }
      if (!att->is_mapping_identity()) {
        // Explicit mapping between points and local attribute indices.
        for (PointIndex i(0); i < num_original_points; ++i) {
          // The new point id that maps to the currently processed attribute
          // entry.
          const PointIndex new_point_id = point_map[i];
          if (new_point_id == kInvalidPointIndex) {
            continue;
          }
          // Index of the currently processed attribute entry in the original
          // mesh.
          const AttributeValueIndex original_entry_index = att->mapped_index(i);
          // New index of the same entry after unused entries were removed.
          const AttributeValueIndex new_entry_index =
              att_indices_changed ? att_index_map[original_entry_index]
                                  : original_entry_index;

          // Update the mapping. Note that the new point index is always smaller
          // than the processed index |i|, making this operation safe.
          att->SetPointMapEntry(new_point_id, new_entry_index);
        }
        // If the number of points changed, we need to set a new explicit map
        // size.
        att->SetExplicitMapping(mesh->num_points());
      }
    }
  }
}

Status MeshCleanup::MakeGeometryManifold(Mesh *mesh) {
  return Status(Status::DRACO_ERROR, "Unsupported function.");
}

}  // namespace draco
