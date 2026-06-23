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
#ifndef DRACO_COMPRESSION_MESH_TRAVERSER_TRAVERSER_BASE_H_
#define DRACO_COMPRESSION_MESH_TRAVERSER_TRAVERSER_BASE_H_

#include "draco/mesh/corner_table.h"

namespace draco {

// Class providing the basic traversal functionality needed by traversers (such
// as the DepthFirstTraverser, see depth_first_traverser.h). It keeps a pointer
// to the corner table that is used for the traversal, plus it provides a basic
// bookkeeping of visited faces and vertices during the traversal.
template <class CornerTableT, class TraversalObserverT>
class TraverserBase {
 public:
  typedef CornerTableT CornerTable;
  typedef TraversalObserverT TraversalObserver;

  TraverserBase() : corner_table_(nullptr) {}
  virtual ~TraverserBase() = default;

  virtual void Init(const CornerTable *corner_table,
                    TraversalObserver traversal_observer) {
    corner_table_ = corner_table;
    is_face_visited_.assign(corner_table->num_faces(), false);
    is_vertex_visited_.assign(corner_table_->num_vertices(), false);
    traversal_observer_ = traversal_observer;
  }

  const CornerTable &GetCornerTable() const { return *corner_table_; }

  inline bool IsFaceVisited(FaceIndex face_id) const {
    if (face_id == kInvalidFaceIndex) {
      return true;  // Invalid faces are always considered as visited.
    }
    return is_face_visited_[face_id.value()];
  }

  // Returns true if the face containing the given corner was visited.
  inline bool IsFaceVisited(CornerIndex corner_id) const {
    if (corner_id == kInvalidCornerIndex) {
      return true;  // Invalid faces are always considered as visited.
    }
    return is_face_visited_[corner_id.value() / 3];
  }

  inline void MarkFaceVisited(FaceIndex face_id) {
    is_face_visited_[face_id.value()] = true;
  }
  inline bool IsVertexVisited(VertexIndex vert_id) const {
    return is_vertex_visited_[vert_id.value()];
  }
  inline void MarkVertexVisited(VertexIndex vert_id) {
    is_vertex_visited_[vert_id.value()] = true;
  }

  inline const CornerTable *corner_table() const { return corner_table_; }
  inline const TraversalObserverT &traversal_observer() const {
    return traversal_observer_;
  }
  inline TraversalObserverT &traversal_observer() {
    return traversal_observer_;
  }

 private:
  const CornerTable *corner_table_;
  TraversalObserverT traversal_observer_;
  std::vector<bool> is_face_visited_;
  std::vector<bool> is_vertex_visited_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_TRAVERSER_TRAVERSER_BASE_H_
