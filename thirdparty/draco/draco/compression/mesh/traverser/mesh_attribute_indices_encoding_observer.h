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
#ifndef DRACO_COMPRESSION_MESH_TRAVERSER_MESH_ATTRIBUTE_INDICES_ENCODING_OBSERVER_H_
#define DRACO_COMPRESSION_MESH_TRAVERSER_MESH_ATTRIBUTE_INDICES_ENCODING_OBSERVER_H_

#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/compression/attributes/points_sequencer.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Class that can be used to generate encoding (and decoding) order of attribute
// values based on the traversal of the encoded mesh. The class should be used
// as the TraversalObserverT member of a Traverser class such as the
// DepthFirstTraverser (depth_first_traverser.h).
// TODO(b/199760123): Rename to AttributeIndicesCodingTraverserObserver.
template <class CornerTableT>
class MeshAttributeIndicesEncodingObserver {
 public:
  MeshAttributeIndicesEncodingObserver()
      : att_connectivity_(nullptr),
        encoding_data_(nullptr),
        mesh_(nullptr),
        sequencer_(nullptr) {}
  MeshAttributeIndicesEncodingObserver(
      const CornerTableT *connectivity, const Mesh *mesh,
      PointsSequencer *sequencer,
      MeshAttributeIndicesEncodingData *encoding_data)
      : att_connectivity_(connectivity),
        encoding_data_(encoding_data),
        mesh_(mesh),
        sequencer_(sequencer) {}

  // Interface for TraversalObserverT

  void OnNewFaceVisited(FaceIndex /* face */) {}

  inline void OnNewVertexVisited(VertexIndex vertex, CornerIndex corner) {
    const PointIndex point_id =
        mesh_->face(FaceIndex(corner.value() / 3))[corner.value() % 3];
    // Append the visited attribute to the encoding order.
    sequencer_->AddPointId(point_id);

    // Keep track of visited corners.
    encoding_data_->encoded_attribute_value_index_to_corner_map.push_back(
        corner);

    encoding_data_
        ->vertex_to_encoded_attribute_value_index_map[vertex.value()] =
        encoding_data_->num_values;

    encoding_data_->num_values++;
  }

 private:
  const CornerTableT *att_connectivity_;
  MeshAttributeIndicesEncodingData *encoding_data_;
  const Mesh *mesh_;
  PointsSequencer *sequencer_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_TRAVERSER_MESH_ATTRIBUTE_INDICES_ENCODING_OBSERVER_H_
