// Copyright 2017 The Draco Authors.
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
#ifndef DRACO_MESH_MESH_SPLITTER_H_
#define DRACO_MESH_MESH_SPLITTER_H_

#include <memory>
#include <vector>

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED
#include "draco/core/status_or.h"
#include "draco/mesh/mesh.h"
#include "draco/mesh/mesh_connected_components.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"

namespace draco {

// Class that can be used to split a single mesh into multiple sub-meshes
// according to specified criteria.
class MeshSplitter {
 public:
  typedef std::vector<std::unique_ptr<Mesh>> MeshVector;
  MeshSplitter();

  // Sets a flag that tells the splitter to preserve all materials on the input
  // mesh during mesh splitting. When set, the materials used on sub-meshes are
  // going to be copied over. Any redundant materials on sub-meshes are going to
  // be deleted but material indices may still be preserved depending on the
  // SetRemoveUnusedMaterialIndices() flag.
  // Default = false.
  void SetPreserveMaterials(bool flag) { preserve_materials_ = flag; }

  // Sets a flag that tells the splitter to delete any unused material indices
  // on the generated sub-meshes. This option is currently used only when
  // SetPreserveMaterials() was set to true. If this option is set to false, the
  // material indices of the MATERIAL attribute will be the same as in the
  // source mesh. If the flag is true, then the unused material indices will be
  // removed and they may no longer correspond to the source mesh. Note that
  // when this flag is false, any unused materials would be replaced with empty
  // (default) materials.
  // Default = true.
  void SetRemoveUnusedMaterialIndices(bool flag) {
    remove_unused_material_indices_ = flag;
  }

  // Sets a flag that tells the splitter to preserve all mesh features on the
  // input mesh during mesh splitting. When set, the mesh features used on
  // sub-meshes are going to be copied over. Any redundant mesh features on
  // sub-meshes are going to be deleted.
  // Default = false.
  void SetPreserveMeshFeatures(bool flag) { preserve_mesh_features_ = flag; }

  // Sets a flag that tells the splitter to preserve structural metadata on the
  // input mesh during mesh splitting. When set, the structural metadata like
  // property attributes indices used on sub-meshes are going to be copied over.
  // Any redundant structural metadata on sub-meshes are going to be deleted.
  // Default = false.
  void SetPreserveStructuralMetadata(bool flag) {
    preserve_structural_metadata_ = flag;
  }

  // By default, the splitter will attempt to deduplicate vertices after
  // splitting the mesh. This means lower memory usage and smaller output glTFs
  // after encoding. However, for very large meshes, this may become an
  // expensive operation. If that becomes an issue, you might want to consider
  // disabling deduplication with |SetDeduplicateVertices(false)|.
  //
  // Note that at this moment, disabling deduplication works ONLY for point
  // clouds.
  void SetDeduplicateVertices(bool flag) { deduplicate_vertices_ = flag; }

  // Splits the input |mesh| according to attribute values stored in the
  // specified attribute. If the |mesh| contains faces, the attribute values
  // need to be defined per-face, that is, all points attached to a single face
  // must share the same attribute value. Meshes without faces are treated as
  // point clouds and the attribute values can be defined per-point. Each
  // attribute value (AttributeValueIndex) is mapped to a single output mesh. If
  // an AttributeValueIndex is unused, no mesh is created for the given value.
  StatusOr<MeshVector> SplitMesh(const Mesh &mesh, uint32_t split_attribute_id);

  // Splits the input |mesh| into separate components defined in
  // |connected_components|. That is, all faces associated with a given
  // component index will be stored in the same mesh. The number of generated
  // meshes will correspond to |connected_components.NumConnectedComponents()|.
  StatusOr<MeshVector> SplitMeshToComponents(
      const Mesh &mesh, const MeshConnectedComponents &connected_components);

  // Returns attribute index on each split mesh that corresponds to the
  // |source_mesh_att_index| of the source Mesh.
  // Must be called after SplitMesh() or SplitMeshToComponents().
  int GetSplitMeshAttributeIndex(int source_mesh_att_index) const;

 private:
  struct WorkData {
    std::vector<int> num_sub_mesh_elements;
    bool split_by_materials = false;
  };

  template <typename BuilderT>
  StatusOr<MeshVector> SplitMeshInternal(const Mesh &mesh,
                                         int split_attribute_id);

  StatusOr<MeshVector> FinalizeMeshes(const Mesh &mesh,
                                      const WorkData &work_data,
                                      MeshVector out_meshes) const;

  bool preserve_materials_;
  bool remove_unused_material_indices_;
  bool preserve_mesh_features_;
  bool preserve_structural_metadata_;
  bool deduplicate_vertices_;

  // Map between attribute ids of the input and output meshes.
  std::vector<int> att_id_map_;

  template <typename BuilderT>
  friend class MeshSplitterInternal;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_MESH_MESH_SPLITTER_H_
