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
#ifndef DRACO_COMPRESSION_MESH_TRAVERSER_MESH_TRAVERSAL_SEQUENCER_H_
#define DRACO_COMPRESSION_MESH_TRAVERSER_MESH_TRAVERSAL_SEQUENCER_H_

#include "draco/attributes/geometry_indices.h"
#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/compression/attributes/points_sequencer.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Sequencer that generates point sequence in an order given by a deterministic
// traversal on the mesh surface. Note that all attributes encoded with this
// sequence must share the same connectivity.
// TODO(b/199760123): Consider refactoring such that this is an observer.
template <class TraverserT>
class MeshTraversalSequencer : public PointsSequencer {
 public:
  MeshTraversalSequencer(const Mesh *mesh,
                         const MeshAttributeIndicesEncodingData *encoding_data)
      : mesh_(mesh), encoding_data_(encoding_data), corner_order_(nullptr) {}
  void SetTraverser(const TraverserT &t) { traverser_ = t; }

  // Function that can be used to set an order in which the mesh corners should
  // be processed. This is an optional flag used usually only by the encoder
  // to match the same corner order that is going to be used by the decoder.
  // Note that |corner_order| should contain only one corner per face (it can
  // have all corners but only the first encountered corner for each face is
  // going to be used to start a traversal). If the corner order is not set, the
  // corners are processed sequentially based on their ids.
  void SetCornerOrder(const std::vector<CornerIndex> &corner_order) {
    corner_order_ = &corner_order;
  }

  bool UpdatePointToAttributeIndexMapping(PointAttribute *attribute) override {
    const auto *corner_table = traverser_.corner_table();
    attribute->SetExplicitMapping(mesh_->num_points());
    const size_t num_faces = mesh_->num_faces();
    const size_t num_points = mesh_->num_points();
    for (FaceIndex f(0); f < static_cast<uint32_t>(num_faces); ++f) {
      const auto &face = mesh_->face(f);
      for (int p = 0; p < 3; ++p) {
        const PointIndex point_id = face[p];
        const VertexIndex vert_id =
            corner_table->Vertex(CornerIndex(3 * f.value() + p));
        if (vert_id == kInvalidVertexIndex) {
          return false;
        }
        const AttributeValueIndex att_entry_id(
            encoding_data_
                ->vertex_to_encoded_attribute_value_index_map[vert_id.value()]);
        if (point_id >= num_points || att_entry_id.value() >= num_points) {
          // There cannot be more attribute values than the number of points.
          return false;
        }
        attribute->SetPointMapEntry(point_id, att_entry_id);
      }
    }
    return true;
  }

 protected:
  bool GenerateSequenceInternal() override {
    // Preallocate memory for storing point indices. We expect the number of
    // points to be the same as the number of corner table vertices.
    out_point_ids()->reserve(traverser_.corner_table()->num_vertices());

    traverser_.OnTraversalStart();
    if (corner_order_) {
      for (uint32_t i = 0; i < corner_order_->size(); ++i) {
        if (!ProcessCorner(corner_order_->at(i))) {
          return false;
        }
      }
    } else {
      const int32_t num_faces = traverser_.corner_table()->num_faces();
      for (int i = 0; i < num_faces; ++i) {
        if (!ProcessCorner(CornerIndex(3 * i))) {
          return false;
        }
      }
    }
    traverser_.OnTraversalEnd();
    return true;
  }

 private:
  bool ProcessCorner(CornerIndex corner_id) {
    return traverser_.TraverseFromCorner(corner_id);
  }

  TraverserT traverser_;
  const Mesh *mesh_;
  const MeshAttributeIndicesEncodingData *encoding_data_;
  const std::vector<CornerIndex> *corner_order_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_TRAVERSER_MESH_TRAVERSAL_SEQUENCER_H_
