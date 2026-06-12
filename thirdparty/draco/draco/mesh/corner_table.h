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
#ifndef DRACO_MESH_CORNER_TABLE_H_
#define DRACO_MESH_CORNER_TABLE_H_

#include <array>
#include <memory>

#include "draco/attributes/geometry_indices.h"
#include "draco/core/draco_index_type_vector.h"
#include "draco/core/macros.h"
#include "draco/draco_features.h"
#include "draco/mesh/valence_cache.h"

namespace draco {

// CornerTable is used to represent connectivity of triangular meshes.
// For every corner of all faces, the corner table stores the index of the
// opposite corner in the neighboring face (if it exists) as illustrated in the
// figure below (see corner |c| and it's opposite corner |o|).
//
//     *
//    /c\
//   /   \
//  /n   p\
// *-------*
//  \     /
//   \   /
//    \o/
//     *
//
// All corners are defined by unique CornerIndex and each triplet of corners
// that define a single face id always ordered consecutively as:
//     { 3 * FaceIndex, 3 * FaceIndex + 1, 3 * FaceIndex +2 }.
// This representation of corners allows CornerTable to easily retrieve Next and
// Previous corners on any face (see corners |n| and |p| in the figure above).
// Using the Next, Previous, and Opposite corners then enables traversal of any
// 2-manifold surface.
// If the CornerTable is constructed from a non-manifold surface, the input
// non-manifold edges and vertices are automatically split.
class CornerTable {
 public:
  // Corner table face type.
  typedef std::array<VertexIndex, 3> FaceType;

  CornerTable();
  static std::unique_ptr<CornerTable> Create(
      const IndexTypeVector<FaceIndex, FaceType> &faces);

  // Initializes the CornerTable from provides set of indexed faces.
  // The input faces can represent a non-manifold topology, in which case the
  // non-manifold edges and vertices are going to be split.
  bool Init(const IndexTypeVector<FaceIndex, FaceType> &faces);

  // Resets the corner table to the given number of invalid faces.
  bool Reset(int num_faces);

  // Resets the corner table to the given number of invalid faces and vertices.
  bool Reset(int num_faces, int num_vertices);

  inline int num_vertices() const {
    return static_cast<int>(vertex_corners_.size());
  }
  inline int num_corners() const {
    return static_cast<int>(corner_to_vertex_map_.size());
  }
  inline int num_faces() const {
    return static_cast<int>(corner_to_vertex_map_.size() / 3);
  }

  inline CornerIndex Opposite(CornerIndex corner) const {
    if (corner == kInvalidCornerIndex) {
      return corner;
    }
    return opposite_corners_[corner];
  }
  inline CornerIndex Next(CornerIndex corner) const {
    if (corner == kInvalidCornerIndex) {
      return corner;
    }
    return LocalIndex(++corner) ? corner : corner - 3;
  }
  inline CornerIndex Previous(CornerIndex corner) const {
    if (corner == kInvalidCornerIndex) {
      return corner;
    }
    return LocalIndex(corner) ? corner - 1 : corner + 2;
  }
  inline VertexIndex Vertex(CornerIndex corner) const {
    if (corner == kInvalidCornerIndex) {
      return kInvalidVertexIndex;
    }
    return ConfidentVertex(corner);
  }
  inline VertexIndex ConfidentVertex(CornerIndex corner) const {
    DRACO_DCHECK_GE(corner.value(), 0);
    DRACO_DCHECK_LT(corner.value(), num_corners());
    return corner_to_vertex_map_[corner];
  }
  inline FaceIndex Face(CornerIndex corner) const {
    if (corner == kInvalidCornerIndex) {
      return kInvalidFaceIndex;
    }
    return FaceIndex(corner.value() / 3);
  }
  inline CornerIndex FirstCorner(FaceIndex face) const {
    if (face == kInvalidFaceIndex) {
      return kInvalidCornerIndex;
    }
    return CornerIndex(face.value() * 3);
  }
  inline std::array<CornerIndex, 3> AllCorners(FaceIndex face) const {
    const CornerIndex ci = CornerIndex(face.value() * 3);
    return {{ci, ci + 1, ci + 2}};
  }
  inline int LocalIndex(CornerIndex corner) const { return corner.value() % 3; }

  inline FaceType FaceData(FaceIndex face) const {
    const CornerIndex first_corner = FirstCorner(face);
    FaceType face_data;
    for (int i = 0; i < 3; ++i) {
      face_data[i] = corner_to_vertex_map_[first_corner + i];
    }
    return face_data;
  }

  void SetFaceData(FaceIndex face, FaceType data) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    const CornerIndex first_corner = FirstCorner(face);
    for (int i = 0; i < 3; ++i) {
      corner_to_vertex_map_[first_corner + i] = data[i];
    }
  }

  // Returns the left-most corner of a single vertex 1-ring. If a vertex is not
  // on a boundary (in which case it has a full 1-ring), this function returns
  // any of the corners mapped to the given vertex.
  inline CornerIndex LeftMostCorner(VertexIndex v) const {
    return vertex_corners_[v];
  }

  // Returns the parent vertex index of a given corner table vertex.
  VertexIndex VertexParent(VertexIndex vertex) const {
    if (vertex.value() < static_cast<uint32_t>(num_original_vertices_)) {
      return vertex;
    }
    return non_manifold_vertex_parents_[vertex - num_original_vertices_];
  }

  // Returns true if the corner is valid.
  inline bool IsValid(CornerIndex c) const {
    return Vertex(c) != kInvalidVertexIndex;
  }

  // Returns the valence (or degree) of a vertex.
  // Returns -1 if the given vertex index is not valid.
  int Valence(VertexIndex v) const;
  // Same as above but does not check for validity and does not return -1
  int ConfidentValence(VertexIndex v) const;
  // Returns the valence of the vertex at the given corner.
  inline int Valence(CornerIndex c) const {
    if (c == kInvalidCornerIndex) {
      return -1;
    }
    return ConfidentValence(c);
  }
  inline int ConfidentValence(CornerIndex c) const {
    DRACO_DCHECK_LT(c.value(), num_corners());
    return ConfidentValence(ConfidentVertex(c));
  }

  // Returns true if the specified vertex is on a boundary.
  inline bool IsOnBoundary(VertexIndex vert) const {
    const CornerIndex corner = LeftMostCorner(vert);
    if (SwingLeft(corner) == kInvalidCornerIndex) {
      return true;
    }
    return false;
  }

  //     *-------*
  //    / \     / \
  //   /   \   /   \
  //  /   sl\c/sr   \
  // *-------v-------*
  // Returns the corner on the adjacent face on the right that maps to
  // the same vertex as the given corner (sr in the above diagram).
  inline CornerIndex SwingRight(CornerIndex corner) const {
    return Previous(Opposite(Previous(corner)));
  }
  // Returns the corner on the left face that maps to the same vertex as the
  // given corner (sl in the above diagram).
  inline CornerIndex SwingLeft(CornerIndex corner) const {
    return Next(Opposite(Next(corner)));
  }

  // Get opposite corners on the left and right faces respectively (see image
  // below, where L and R are the left and right corners of a corner X.
  //
  // *-------*-------*
  //  \L    /X\    R/
  //   \   /   \   /
  //    \ /     \ /
  //     *-------*
  inline CornerIndex GetLeftCorner(CornerIndex corner_id) const {
    if (corner_id == kInvalidCornerIndex) {
      return kInvalidCornerIndex;
    }
    return Opposite(Previous(corner_id));
  }
  inline CornerIndex GetRightCorner(CornerIndex corner_id) const {
    if (corner_id == kInvalidCornerIndex) {
      return kInvalidCornerIndex;
    }
    return Opposite(Next(corner_id));
  }

  // Returns the number of new vertices that were created as a result of
  // splitting of non-manifold vertices of the input geometry.
  int NumNewVertices() const { return num_vertices() - num_original_vertices_; }
  int NumOriginalVertices() const { return num_original_vertices_; }

  // Returns the number of faces with duplicated vertex indices.
  int NumDegeneratedFaces() const { return num_degenerated_faces_; }

  // Returns the number of isolated vertices (vertices that have
  // vertex_corners_ mapping set to kInvalidCornerIndex.
  int NumIsolatedVertices() const { return num_isolated_vertices_; }

  bool IsDegenerated(FaceIndex face) const;

  // Methods that modify an existing corner table.
  // Sets the opposite corner mapping between two corners. Caller must ensure
  // that the indices are valid.
  inline void SetOppositeCorner(CornerIndex corner_id,
                                CornerIndex opp_corner_id) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    opposite_corners_[corner_id] = opp_corner_id;
  }

  // Sets opposite corners for both input corners.
  inline void SetOppositeCorners(CornerIndex corner_0, CornerIndex corner_1) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    if (corner_0 != kInvalidCornerIndex) {
      SetOppositeCorner(corner_0, corner_1);
    }
    if (corner_1 != kInvalidCornerIndex) {
      SetOppositeCorner(corner_1, corner_0);
    }
  }

  // Updates mapping between a corner and a vertex.
  inline void MapCornerToVertex(CornerIndex corner_id, VertexIndex vert_id) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    corner_to_vertex_map_[corner_id] = vert_id;
  }

  VertexIndex AddNewVertex() {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    // Add a new invalid vertex.
    vertex_corners_.push_back(kInvalidCornerIndex);
    return VertexIndex(static_cast<uint32_t>(vertex_corners_.size() - 1));
  }

  // Adds a new face connected to three vertices. Note that connectivity is not
  // automatically updated and all opposite corners need to be set explicitly.
  FaceIndex AddNewFace(const std::array<VertexIndex, 3> &vertices) {
    // Add a new invalid face.
    const FaceIndex new_face_index(num_faces());
    for (int i = 0; i < 3; ++i) {
      corner_to_vertex_map_.push_back(vertices[i]);
      SetLeftMostCorner(vertices[i],
                        CornerIndex(corner_to_vertex_map_.size() - 1));
    }
    opposite_corners_.resize(corner_to_vertex_map_.size(), kInvalidCornerIndex);
    return new_face_index;
  }

  // Sets a new left most corner for a given vertex.
  void SetLeftMostCorner(VertexIndex vert, CornerIndex corner) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    if (vert != kInvalidVertexIndex) {
      vertex_corners_[vert] = corner;
    }
  }

  // Updates the vertex to corner map on a specified vertex. This should be
  // called in cases where the mapping may be invalid (e.g. when the corner
  // table was constructed manually).
  void UpdateVertexToCornerMap(VertexIndex vert) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    const CornerIndex first_c = vertex_corners_[vert];
    if (first_c == kInvalidCornerIndex) {
      return;  // Isolated vertex.
    }
    CornerIndex act_c = SwingLeft(first_c);
    CornerIndex c = first_c;
    while (act_c != kInvalidCornerIndex && act_c != first_c) {
      c = act_c;
      act_c = SwingLeft(act_c);
    }
    if (act_c != first_c) {
      vertex_corners_[vert] = c;
    }
  }

  // Sets the new number of vertices. It's a responsibility of the caller to
  // ensure that no corner is mapped beyond the range of the new number of
  // vertices.
  inline void SetNumVertices(int num_vertices) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    vertex_corners_.resize(num_vertices, kInvalidCornerIndex);
  }

  // Makes a vertex isolated (not attached to any corner).
  void MakeVertexIsolated(VertexIndex vert) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    vertex_corners_[vert] = kInvalidCornerIndex;
  }

  // Returns true if a vertex is not attached to any face.
  inline bool IsVertexIsolated(VertexIndex v) const {
    return LeftMostCorner(v) == kInvalidCornerIndex;
  }

  // Makes a given face invalid (all corners are marked as invalid).
  void MakeFaceInvalid(FaceIndex face) {
    DRACO_DCHECK(GetValenceCache().IsCacheEmpty());
    if (face != kInvalidFaceIndex) {
      const CornerIndex first_corner = FirstCorner(face);
      for (int i = 0; i < 3; ++i) {
        corner_to_vertex_map_[first_corner + i] = kInvalidVertexIndex;
      }
    }
  }

  // Updates mapping between faces and a vertex using the corners mapped to
  // the provided vertex.
  void UpdateFaceToVertexMap(const VertexIndex vertex);

  // Allows access to an internal object for caching valences.  The object can
  // be instructed to cache or uncache all valences and then its interfaces
  // queried directly for valences with differing performance/confidence
  // qualities.  If the mesh or table is modified the cache should be discarded
  // and not relied on as it does not automatically update or invalidate for
  // performance reasons.
  const draco::ValenceCache<CornerTable> &GetValenceCache() const {
    return valence_cache_;
  }

 private:
  // Computes opposite corners mapping from the data stored in
  // |corner_to_vertex_map_|.
  bool ComputeOppositeCorners(int *num_vertices);

  // Finds and breaks non-manifold edges in the 1-ring neighborhood around
  // vertices (vertices themselves will be split in the ComputeVertexCorners()
  // function if necessary).
  bool BreakNonManifoldEdges();

  // Computes the lookup map for going from a vertex to a corner. This method
  // can handle non-manifold vertices by splitting them into multiple manifold
  // vertices.
  bool ComputeVertexCorners(int num_vertices);

  // Each three consecutive corners represent one face.
  IndexTypeVector<CornerIndex, VertexIndex> corner_to_vertex_map_;
  IndexTypeVector<CornerIndex, CornerIndex> opposite_corners_;
  IndexTypeVector<VertexIndex, CornerIndex> vertex_corners_;

  int num_original_vertices_;
  int num_degenerated_faces_;
  int num_isolated_vertices_;
  IndexTypeVector<VertexIndex, VertexIndex> non_manifold_vertex_parents_;

  draco::ValenceCache<CornerTable> valence_cache_;
};

// A special case to denote an invalid corner table triangle.
static constexpr CornerTable::FaceType kInvalidFace(
    {{kInvalidVertexIndex, kInvalidVertexIndex, kInvalidVertexIndex}});

}  // namespace draco

#endif  // DRACO_MESH_CORNER_TABLE_H_
