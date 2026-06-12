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
#ifndef DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_H_
#define DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_H_

#include <unordered_map>
#include <unordered_set>

#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/compression/mesh/mesh_edgebreaker_decoder_impl_interface.h"
#include "draco/compression/mesh/mesh_edgebreaker_shared.h"
#include "draco/compression/mesh/traverser/mesh_traversal_sequencer.h"
#include "draco/core/decoder_buffer.h"
#include "draco/draco_features.h"
#include "draco/mesh/corner_table.h"
#include "draco/mesh/mesh_attribute_corner_table.h"

namespace draco {

// Implementation of the edgebreaker decoder that decodes data encoded with the
// MeshEdgebreakerEncoderImpl class. The implementation of the decoder is based
// on the algorithm presented in Isenburg et al'02 "Spirale Reversi: Reverse
// decoding of the Edgebreaker encoding". Note that the encoding is still based
// on the standard edgebreaker method as presented in "3D Compression
// Made Simple: Edgebreaker on a Corner-Table" by Rossignac at al.'01.
// http://www.cc.gatech.edu/~jarek/papers/CornerTableSMI.pdf. One difference is
// caused by the properties of the spirale reversi algorithm that decodes the
// symbols from the last one to the first one. To make the decoding more
// efficient, we encode all symbols in the reverse order, therefore the decoder
// can process them one by one.
// The main advantage of the spirale reversi method is that the partially
// decoded mesh has valid connectivity data at any time during the decoding
// process (valid with respect to the decoded portion of the mesh). The standard
// Edgebreaker decoder used two passes (forward decoding + zipping) which not
// only prevented us from having a valid connectivity but it was also slower.
// The main benefit of having the valid connectivity is that we can use the
// known connectivity to predict encoded symbols that can improve the
// compression rate.
template <class TraversalDecoderT>
class MeshEdgebreakerDecoderImpl : public MeshEdgebreakerDecoderImplInterface {
 public:
  MeshEdgebreakerDecoderImpl();
  bool Init(MeshEdgebreakerDecoder *decoder) override;

  const MeshAttributeCornerTable *GetAttributeCornerTable(
      int att_id) const override;
  const MeshAttributeIndicesEncodingData *GetAttributeEncodingData(
      int att_id) const override;

  bool CreateAttributesDecoder(int32_t att_decoder_id) override;
  bool DecodeConnectivity() override;
  bool OnAttributesDecoded() override;
  MeshEdgebreakerDecoder *GetDecoder() const override { return decoder_; }
  const CornerTable *GetCornerTable() const override {
    return corner_table_.get();
  }

 private:
  // Creates a vertex traversal sequencer for the specified |TraverserT| type.
  template <class TraverserT>
  std::unique_ptr<PointsSequencer> CreateVertexTraversalSequencer(
      MeshAttributeIndicesEncodingData *encoding_data);

  // Decodes connectivity between vertices (vertex indices).
  // Returns the number of vertices created by the decoder or -1 on error.
  int DecodeConnectivity(int num_symbols);

  // Returns true if the current symbol was part of a topology split event. This
  // means that the current face was connected to the left edge of a face
  // encoded with the TOPOLOGY_S symbol. |out_symbol_edge| can be used to
  // identify which edge of the source symbol was connected to the TOPOLOGY_S
  // symbol.
  bool IsTopologySplit(int encoder_symbol_id, EdgeFaceName *out_face_edge,
                       int *out_encoder_split_symbol_id) {
    if (topology_split_data_.size() == 0) {
      return false;
    }
    if (topology_split_data_.back().source_symbol_id >
        static_cast<uint32_t>(encoder_symbol_id)) {
      // Something is wrong; if the desired source symbol is greater than the
      // current encoder_symbol_id, we missed it, or the input was tampered
      // (|encoder_symbol_id| keeps decreasing).
      // Return invalid symbol id to notify the decoder that there was an
      // error.
      *out_encoder_split_symbol_id = -1;
      return true;
    }
    if (topology_split_data_.back().source_symbol_id != encoder_symbol_id) {
      return false;
    }
    *out_face_edge =
        static_cast<EdgeFaceName>(topology_split_data_.back().source_edge);
    *out_encoder_split_symbol_id = topology_split_data_.back().split_symbol_id;
    // Remove the latest split event.
    topology_split_data_.pop_back();
    return true;
  }

  // Decodes event data for hole and topology split events and stores them for
  // future use.
  // Returns the number of parsed bytes, or -1 on error.
  int32_t DecodeHoleAndTopologySplitEvents(DecoderBuffer *decoder_buffer);

  // Decodes all non-position attribute connectivity on the currently
  // processed face.
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  bool DecodeAttributeConnectivitiesOnFaceLegacy(CornerIndex corner);
#endif
  bool DecodeAttributeConnectivitiesOnFace(CornerIndex corner);

  // Initializes mapping between corners and point ids.
  bool AssignPointsToCorners(int num_connectivity_verts);

  bool IsFaceVisited(CornerIndex corner_id) const {
    if (corner_id < 0) {
      return true;  // Invalid corner signalizes that the face does not exist.
    }
    return visited_faces_[corner_table_->Face(corner_id).value()];
  }

  void SetOppositeCorners(CornerIndex corner_0, CornerIndex corner_1) {
    corner_table_->SetOppositeCorner(corner_0, corner_1);
    corner_table_->SetOppositeCorner(corner_1, corner_0);
  }

  MeshEdgebreakerDecoder *decoder_;

  std::unique_ptr<CornerTable> corner_table_;

  // Stack used for storing corners that need to be traversed when decoding
  // mesh vertices. New corner is added for each initial face and a split
  // symbol, and one corner is removed when the end symbol is reached.
  // Stored as member variable to prevent frequent memory reallocations when
  // handling meshes with lots of disjoint components.  Originally, we used
  // recursive functions to handle this behavior, but that can cause stack
  // memory overflow when compressing huge meshes.
  std::vector<CornerIndex> corner_traversal_stack_;

  // Array stores the number of visited visited for each mesh traversal.
  std::vector<int> vertex_traversal_length_;

  // List of decoded topology split events.
  std::vector<TopologySplitEventData> topology_split_data_;

  // List of decoded hole events.
  std::vector<HoleEventData> hole_event_data_;

  // Configuration of the initial face for each mesh component.
  std::vector<bool> init_face_configurations_;

  // Initial corner for each traversal.
  std::vector<CornerIndex> init_corners_;

  // Id of the last processed input symbol.
  int last_symbol_id_;

  // Id of the last decoded vertex.
  int last_vert_id_;

  // Id of the last decoded face.
  int last_face_id_;

  // Array for marking visited faces.
  std::vector<bool> visited_faces_;
  // Array for marking visited vertices.
  std::vector<bool> visited_verts_;
  // Array for marking vertices on open boundaries.
  std::vector<bool> is_vert_hole_;

  // The number of new vertices added by the encoder (because of non-manifold
  // vertices on the input mesh).
  // If there are no non-manifold edges/vertices on the input mesh, this should
  // be 0.
  int num_new_vertices_;
  // For every newly added vertex, this array stores it's mapping to the
  // parent vertex id of the encoded mesh.
  std::unordered_map<int, int> new_to_parent_vertex_map_;
  // The number of vertices that were encoded (can be different from the number
  // of vertices of the input mesh).
  int num_encoded_vertices_;

  // Array for storing the encoded corner ids in the order their associated
  // vertices were decoded.
  std::vector<int32_t> processed_corner_ids_;

  // Array storing corners in the order they were visited during the
  // connectivity decoding (always storing the tip corner of each newly visited
  // face).
  std::vector<int> processed_connectivity_corners_;

  MeshAttributeIndicesEncodingData pos_encoding_data_;

  // Id of an attributes decoder that uses |pos_encoding_data_|.
  int pos_data_decoder_id_;

  // Data for non-position attributes used by the decoder.
  struct AttributeData {
    AttributeData() : decoder_id(-1), is_connectivity_used(true) {}
    // Id of the attribute decoder that was used to decode this attribute data.
    int decoder_id;
    MeshAttributeCornerTable connectivity_data;
    // Flag that can mark the connectivity_data invalid. In such case the base
    // corner table of the mesh should be used instead.
    bool is_connectivity_used;
    MeshAttributeIndicesEncodingData encoding_data;
    // Opposite corners to attribute seam edges.
    std::vector<int32_t> attribute_seam_corners;
  };
  std::vector<AttributeData> attribute_data_;

  TraversalDecoderT traversal_decoder_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_H_
