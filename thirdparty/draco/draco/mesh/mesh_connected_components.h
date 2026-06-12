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
#ifndef DRACO_MESH_MESH_CONNECTED_COMPONENTS_H_
#define DRACO_MESH_MESH_CONNECTED_COMPONENTS_H_

#include <vector>

#include "draco/mesh/corner_table.h"

namespace draco {

// Class for detecting connected components on an input mesh defined by a
// corner table. Degenerated faces and their vertices are not assigned to any
// component.
class MeshConnectedComponents {
 public:
  MeshConnectedComponents() = default;

  // Initializes the class with the component data of the input mesh. No other
  // method should be called before this one.
  template <class CornerTableT = CornerTable>
  void FindConnectedComponents(const CornerTableT *corner_table);
  int NumConnectedComponents() const { return components_.size(); }

  struct ConnectedComponent {
    std::vector<int> vertices;
    std::vector<int> faces;
    std::vector<int> boundary_edges;
  };

  const ConnectedComponent &GetConnectedComponent(int index) const {
    return components_[index];
  }

  // Returns the id of an component attached to a given vertex. Returns -1 when
  // the vertex was not assigned to any component.
  int GetConnectedComponentIdAtVertex(int vertex_id) const {
    return vertex_to_component_map_[vertex_id];
  }

  // Returns the number of vertices that belong to the input component.
  int NumConnectedComponentVertices(int component_id) const {
    return components_[component_id].vertices.size();
  }

  // Returns the i-th vertex of the input component.
  int GetConnectedComponentVertex(int component_id, int i) const {
    return components_[component_id].vertices[i];
  }

  // Returns the id of an component attached to a given face. Returns -1 when
  // the face was not assigned to any component.
  int GetConnectedComponentIdAtFace(int face_id) const {
    return face_to_component_map_[face_id];
  }

  // Returns the number of faces that belong to the input component.
  int NumConnectedComponentFaces(int component_id) const {
    return components_[component_id].faces.size();
  }

  // Returns the i-th face of the input component.
  int GetConnectedComponentFace(int component_id, int i) const {
    return components_[component_id].faces[i];
  }

  // Returns the number of boundary edges that belong to the input component.
  int NumConnectedComponentBoundaryEdges(int component_id) const {
    return components_[component_id].boundary_edges.size();
  }

  // Returns the i-th boundary edge of the input component.
  int GetConnectedComponentBoundaryEdge(int component_id, int i) const {
    return components_[component_id].boundary_edges[i];
  }

 private:
  std::vector<int> vertex_to_component_map_;
  std::vector<int> face_to_component_map_;
  std::vector<int> boundary_corner_to_component_map_;
  std::vector<ConnectedComponent> components_;
};

template <class CornerTableT>
void MeshConnectedComponents::FindConnectedComponents(
    const CornerTableT *corner_table) {
  components_.clear();
  vertex_to_component_map_.assign(corner_table->num_vertices(), -1);
  face_to_component_map_.assign(corner_table->num_faces(), -1);
  boundary_corner_to_component_map_.assign(corner_table->num_corners(), -1);
  std::vector<int> is_face_visited(corner_table->num_faces(), false);
  std::vector<int> face_stack;
  // Go over all faces of the mesh and for each unvisited face, recursively
  // traverse its neighborhood and mark all traversed faces as visited. All
  // faces visited during one traversal belong to one mesh component.
  for (int face_id = 0; face_id < corner_table->num_faces(); ++face_id) {
    if (is_face_visited[face_id]) {
      continue;
    }
    if (corner_table->IsDegenerated(FaceIndex(face_id))) {
      continue;
    }
    const int component_id = components_.size();
    components_.push_back(ConnectedComponent());
    face_stack.push_back(face_id);
    is_face_visited[face_id] = true;
    while (!face_stack.empty()) {
      const int act_face_id = face_stack.back();
      if (face_to_component_map_[act_face_id] == -1) {
        face_to_component_map_[act_face_id] = component_id;
        components_[component_id].faces.push_back(act_face_id);
      }
      face_stack.pop_back();
      // Gather all neighboring faces.
      std::array<CornerIndex, 3> corners =
          corner_table->AllCorners(FaceIndex(act_face_id));
      for (int c = 0; c < 3; ++c) {
        // Update vertex to component mapping.
        const int vertex_id = corner_table->Vertex(corners[c]).value();
        if (vertex_to_component_map_[vertex_id] == -1) {
          vertex_to_component_map_[vertex_id] = component_id;
          components_[component_id].vertices.push_back(vertex_id);
        }
        // Traverse component to neighboring faces (add the faces to the stack).
        const CornerIndex opp_corner = corner_table->Opposite(corners[c]);
        if (opp_corner == kInvalidCornerIndex) {
          if (boundary_corner_to_component_map_[corners[c].value()] == -1) {
            boundary_corner_to_component_map_[corners[c].value()] =
                component_id;
            components_[component_id].boundary_edges.push_back(
                corners[c].value());
          }
          continue;  // Invalid corner (mesh boundary).
        }

        const int opp_face_id = corner_table->Face(opp_corner).value();
        if (is_face_visited[opp_face_id]) {
          continue;  // Opposite face has been already reached.
        }
        is_face_visited[opp_face_id] = true;
        face_stack.push_back(opp_face_id);
      }
    }
  }
}

}  // namespace draco

#endif  // DRACO_MESH_MESH_CONNECTED_COMPONENTS_H_
