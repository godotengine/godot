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
#ifndef DRACO_MESH_MESH_CLEANUP_H_
#define DRACO_MESH_MESH_CLEANUP_H_

#include "draco/core/status.h"
#include "draco/draco_features.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Options used by the MeshCleanup class.
struct MeshCleanupOptions {
  // If true, the cleanup tool removes any face where two or more vertices
  // share the same position index.
  bool remove_degenerated_faces = true;

  // If true, the cleanup tool removes all duplicate faces. A pair of faces is
  // duplicate if both faces share the same position indices on all vertices
  // (that is, position values have to be duduplicated). Note that all
  // non-position properties are currently ignored.
  bool remove_duplicate_faces = true;

  // If true, the cleanup tool removes any unused attribute value or unused
  // point id. For example, it can be used to remove isolated vertices.
  bool remove_unused_attributes = true;

  // If true, the cleanup tool splits vertices along non-manifold edges and
  // vertices. This ensures that the connectivity defined by position indices
  // is manifold.
  bool make_geometry_manifold = false;
};

// Tool that can be used for removing bad or unused data from draco::Meshes.
class MeshCleanup {
 public:
  // Performs in-place cleanup of the input mesh according to the input options.
  static Status Cleanup(Mesh *mesh, const MeshCleanupOptions &options);

 private:
  static void RemoveDegeneratedFaces(Mesh *mesh);
  static void RemoveDuplicateFaces(Mesh *mesh);
  static void RemoveUnusedAttributes(Mesh *mesh);
  static Status MakeGeometryManifold(Mesh *mesh);
};

}  // namespace draco

#endif  // DRACO_MESH_MESH_CLEANUP_H_
