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
#ifndef DRACO_COMPRESSION_MESH_TRAVERSER_DEPTH_FIRST_TRAVERSER_H_
#define DRACO_COMPRESSION_MESH_TRAVERSER_DEPTH_FIRST_TRAVERSER_H_

#include <vector>

#include "draco/compression/mesh/traverser/traverser_base.h"
#include "draco/mesh/corner_table.h"

namespace draco {

// Basic traverser that traverses a mesh in a DFS like fashion using the
// CornerTable data structure. The necessary bookkeeping is available via the
// TraverserBase. Callbacks are handled through template argument
// TraversalObserverT.
//
// TraversalObserverT can perform an action on a traversal event such as newly
// visited face, or corner, but it does not affect the traversal itself.
//
// Concept TraversalObserverT requires:
//
// public:
// void OnNewFaceVisited(FaceIndex face);
//    - Called whenever a previously unvisited face is reached.
//
// void OnNewVertexVisited(VertexIndex vert, CornerIndex corner)
//    - Called when a new vertex is visited. |corner| is used to indicate the
//      which of the vertex's corners has been reached.

template <class CornerTableT, class TraversalObserverT>
class DepthFirstTraverser
    : public TraverserBase<CornerTableT, TraversalObserverT> {
 public:
  typedef CornerTableT CornerTable;
  typedef TraversalObserverT TraversalObserver;
  typedef TraverserBase<CornerTable, TraversalObserver> Base;

  DepthFirstTraverser() {}

  // Called before any traversing starts.
  void OnTraversalStart() {}

  // Called when all the traversing is done.
  void OnTraversalEnd() {}

  bool TraverseFromCorner(CornerIndex corner_id) {
    if (this->IsFaceVisited(corner_id)) {
      return true;  // Already traversed.
    }

    corner_traversal_stack_.clear();
    corner_traversal_stack_.push_back(corner_id);
    // For the first face, check the remaining corners as they may not be
    // processed yet.
    const VertexIndex next_vert =
        this->corner_table()->Vertex(this->corner_table()->Next(corner_id));
    const VertexIndex prev_vert =
        this->corner_table()->Vertex(this->corner_table()->Previous(corner_id));
    if (next_vert == kInvalidVertexIndex || prev_vert == kInvalidVertexIndex) {
      return false;
    }
    if (!this->IsVertexVisited(next_vert)) {
      this->MarkVertexVisited(next_vert);
      this->traversal_observer().OnNewVertexVisited(
          next_vert, this->corner_table()->Next(corner_id));
    }
    if (!this->IsVertexVisited(prev_vert)) {
      this->MarkVertexVisited(prev_vert);
      this->traversal_observer().OnNewVertexVisited(
          prev_vert, this->corner_table()->Previous(corner_id));
    }

    // Start the actual traversal.
    while (!corner_traversal_stack_.empty()) {
      // Currently processed corner.
      corner_id = corner_traversal_stack_.back();
      FaceIndex face_id(corner_id.value() / 3);
      // Make sure the face hasn't been visited yet.
      if (corner_id == kInvalidCornerIndex || this->IsFaceVisited(face_id)) {
        // This face has been already traversed.
        corner_traversal_stack_.pop_back();
        continue;
      }
      while (true) {
        this->MarkFaceVisited(face_id);
        this->traversal_observer().OnNewFaceVisited(face_id);
        const VertexIndex vert_id = this->corner_table()->Vertex(corner_id);
        if (vert_id == kInvalidVertexIndex) {
          return false;
        }
        if (!this->IsVertexVisited(vert_id)) {
          const bool on_boundary = this->corner_table()->IsOnBoundary(vert_id);
          this->MarkVertexVisited(vert_id);
          this->traversal_observer().OnNewVertexVisited(vert_id, corner_id);
          if (!on_boundary) {
            corner_id = this->corner_table()->GetRightCorner(corner_id);
            face_id = FaceIndex(corner_id.value() / 3);
            continue;
          }
        }
        // The current vertex has been already visited or it was on a boundary.
        // We need to determine whether we can visit any of it's neighboring
        // faces.
        const CornerIndex right_corner_id =
            this->corner_table()->GetRightCorner(corner_id);
        const CornerIndex left_corner_id =
            this->corner_table()->GetLeftCorner(corner_id);
        const FaceIndex right_face_id(
            (right_corner_id == kInvalidCornerIndex
                 ? kInvalidFaceIndex
                 : FaceIndex(right_corner_id.value() / 3)));
        const FaceIndex left_face_id(
            (left_corner_id == kInvalidCornerIndex
                 ? kInvalidFaceIndex
                 : FaceIndex(left_corner_id.value() / 3)));
        if (this->IsFaceVisited(right_face_id)) {
          // Right face has been already visited.
          if (this->IsFaceVisited(left_face_id)) {
            // Both neighboring faces are visited. End reached.
            corner_traversal_stack_.pop_back();
            break;  // Break from the while (true) loop.
          } else {
            // Go to the left face.
            corner_id = left_corner_id;
            face_id = left_face_id;
          }
        } else {
          // Right face was not visited.
          if (this->IsFaceVisited(left_face_id)) {
            // Left face visited, go to the right one.
            corner_id = right_corner_id;
            face_id = right_face_id;
          } else {
            // Both neighboring faces are unvisited, we need to visit both of
            // them.

            // Split the traversal.
            // First make the top of the current corner stack point to the left
            // face (this one will be processed second).
            corner_traversal_stack_.back() = left_corner_id;
            // Add a new corner to the top of the stack (right face needs to
            // be traversed first).
            corner_traversal_stack_.push_back(right_corner_id);
            // Break from the while (true) loop.
            break;
          }
        }
      }
    }
    return true;
  }

 private:
  std::vector<CornerIndex> corner_traversal_stack_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_TRAVERSER_DEPTH_FIRST_TRAVERSER_H_
