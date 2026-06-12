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
#include "draco/mesh/mesh_attribute_corner_table.h"

#include "draco/mesh/corner_table_iterators.h"
#include "draco/mesh/mesh_misc_functions.h"

namespace draco {

MeshAttributeCornerTable::MeshAttributeCornerTable()
    : no_interior_seams_(true), corner_table_(nullptr), valence_cache_(*this) {}

bool MeshAttributeCornerTable::InitEmpty(const CornerTable *table) {
  if (table == nullptr) {
    return false;
  }
  valence_cache_.ClearValenceCache();
  valence_cache_.ClearValenceCacheInaccurate();
  is_edge_on_seam_.assign(table->num_corners(), false);
  is_vertex_on_seam_.assign(table->num_vertices(), false);
  corner_to_vertex_map_.assign(table->num_corners(), kInvalidVertexIndex);
  vertex_to_attribute_entry_id_map_.reserve(table->num_vertices());
  vertex_to_left_most_corner_map_.reserve(table->num_vertices());
  corner_table_ = table;
  no_interior_seams_ = true;
  return true;
}

bool MeshAttributeCornerTable::InitFromAttribute(const Mesh *mesh,
                                                 const CornerTable *table,
                                                 const PointAttribute *att) {
  if (!InitEmpty(table)) {
    return false;
  }
  valence_cache_.ClearValenceCache();
  valence_cache_.ClearValenceCacheInaccurate();

  // Find all necessary data for encoding attributes. For now we check which of
  // the mesh vertices is part of an attribute seam, because seams require
  // special handling.
  for (CornerIndex c(0); c < corner_table_->num_corners(); ++c) {
    const FaceIndex f = corner_table_->Face(c);
    if (corner_table_->IsDegenerated(f)) {
      continue;  // Ignore corners on degenerated faces.
    }
    const CornerIndex opp_corner = corner_table_->Opposite(c);
    if (opp_corner == kInvalidCornerIndex) {
      // Boundary. Mark it as seam edge.
      is_edge_on_seam_[c.value()] = true;
      // Mark seam vertices.
      VertexIndex v;
      v = corner_table_->Vertex(corner_table_->Next(c));
      is_vertex_on_seam_[v.value()] = true;
      v = corner_table_->Vertex(corner_table_->Previous(c));
      is_vertex_on_seam_[v.value()] = true;
      continue;
    }
    if (opp_corner < c) {
      continue;  // Opposite corner was already processed.
    }

    CornerIndex act_c(c), act_sibling_c(opp_corner);
    for (int i = 0; i < 2; ++i) {
      // Get the sibling corners. I.e., the two corners attached to the same
      // vertex but divided by the seam edge.
      act_c = corner_table_->Next(act_c);
      act_sibling_c = corner_table_->Previous(act_sibling_c);
      const PointIndex point_id = mesh->CornerToPointId(act_c.value());
      const PointIndex sibling_point_id =
          mesh->CornerToPointId(act_sibling_c.value());
      if (att->mapped_index(point_id) != att->mapped_index(sibling_point_id)) {
        no_interior_seams_ = false;
        is_edge_on_seam_[c.value()] = true;
        is_edge_on_seam_[opp_corner.value()] = true;
        // Mark seam vertices.
        is_vertex_on_seam_[corner_table_
                               ->Vertex(corner_table_->Next(CornerIndex(c)))
                               .value()] = true;
        is_vertex_on_seam_[corner_table_
                               ->Vertex(corner_table_->Previous(CornerIndex(c)))
                               .value()] = true;
        is_vertex_on_seam_
            [corner_table_->Vertex(corner_table_->Next(opp_corner)).value()] =
                true;
        is_vertex_on_seam_[corner_table_
                               ->Vertex(corner_table_->Previous(opp_corner))
                               .value()] = true;
        break;
      }
    }
  }
  RecomputeVertices(mesh, att);
  return true;
}

void MeshAttributeCornerTable::AddSeamEdge(CornerIndex c) {
  DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
  is_edge_on_seam_[c.value()] = true;
  // Mark seam vertices.
  is_vertex_on_seam_[corner_table_->Vertex(corner_table_->Next(c)).value()] =
      true;
  is_vertex_on_seam_[corner_table_->Vertex(corner_table_->Previous(c))
                         .value()] = true;

  const CornerIndex opp_corner = corner_table_->Opposite(c);
  if (opp_corner != kInvalidCornerIndex) {
    no_interior_seams_ = false;
    is_edge_on_seam_[opp_corner.value()] = true;
    is_vertex_on_seam_[corner_table_->Vertex(corner_table_->Next(opp_corner))
                           .value()] = true;
    is_vertex_on_seam_
        [corner_table_->Vertex(corner_table_->Previous(opp_corner)).value()] =
            true;
  }
}

bool MeshAttributeCornerTable::RecomputeVertices(const Mesh *mesh,
                                                 const PointAttribute *att) {
  DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
  if (mesh != nullptr && att != nullptr) {
    return RecomputeVerticesInternal<true>(mesh, att);
  } else {
    return RecomputeVerticesInternal<false>(nullptr, nullptr);
  }
}

template <bool init_vertex_to_attribute_entry_map>
bool MeshAttributeCornerTable::RecomputeVerticesInternal(
    const Mesh *mesh, const PointAttribute *att) {
  DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
  vertex_to_attribute_entry_id_map_.clear();
  vertex_to_left_most_corner_map_.clear();
  int num_new_vertices = 0;
  for (VertexIndex v(0); v < corner_table_->num_vertices(); ++v) {
    const CornerIndex c = corner_table_->LeftMostCorner(v);
    if (c == kInvalidCornerIndex) {
      continue;  // Isolated vertex?
    }
    AttributeValueIndex first_vert_id(num_new_vertices++);
    if (init_vertex_to_attribute_entry_map) {
      const PointIndex point_id = mesh->CornerToPointId(c.value());
      vertex_to_attribute_entry_id_map_.push_back(att->mapped_index(point_id));
    } else {
      // Identity mapping
      vertex_to_attribute_entry_id_map_.push_back(first_vert_id);
    }
    CornerIndex first_c = c;
    CornerIndex act_c;
    // Check if the vertex is on a seam edge, if it is we need to find the first
    // attribute entry on the seam edge when traversing in the CCW direction.
    if (is_vertex_on_seam_[v.value()]) {
      // Try to swing left on the modified corner table. We need to get the
      // first corner that defines an attribute seam.
      act_c = SwingLeft(first_c);
      while (act_c != kInvalidCornerIndex) {
        first_c = act_c;
        act_c = SwingLeft(act_c);
        if (act_c == c) {
          // We reached the initial corner which shouldn't happen when we swing
          // left from |c|.
          return false;
        }
      }
    }
    corner_to_vertex_map_[first_c.value()] = VertexIndex(first_vert_id.value());
    vertex_to_left_most_corner_map_.push_back(first_c);
    act_c = corner_table_->SwingRight(first_c);
    while (act_c != kInvalidCornerIndex && act_c != first_c) {
      if (IsCornerOppositeToSeamEdge(corner_table_->Next(act_c))) {
        first_vert_id = AttributeValueIndex(num_new_vertices++);
        if (init_vertex_to_attribute_entry_map) {
          const PointIndex point_id = mesh->CornerToPointId(act_c.value());
          vertex_to_attribute_entry_id_map_.push_back(
              att->mapped_index(point_id));
        } else {
          // Identity mapping.
          vertex_to_attribute_entry_id_map_.push_back(first_vert_id);
        }
        vertex_to_left_most_corner_map_.push_back(act_c);
      }
      corner_to_vertex_map_[act_c.value()] = VertexIndex(first_vert_id.value());
      act_c = corner_table_->SwingRight(act_c);
    }
  }
  return true;
}

int MeshAttributeCornerTable::Valence(VertexIndex v) const {
  if (v == kInvalidVertexIndex) {
    return -1;
  }
  return ConfidentValence(v);
}

int MeshAttributeCornerTable::ConfidentValence(VertexIndex v) const {
  DRACO_DCHECK_LT(v.value(), num_vertices());
  draco::VertexRingIterator<MeshAttributeCornerTable> vi(this, v);
  int valence = 0;
  for (; !vi.End(); vi.Next()) {
    ++valence;
  }
  return valence;
}

}  // namespace draco
