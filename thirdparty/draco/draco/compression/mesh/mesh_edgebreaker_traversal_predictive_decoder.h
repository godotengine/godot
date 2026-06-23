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
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
#ifndef DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_TRAVERSAL_PREDICTIVE_DECODER_H_
#define DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_TRAVERSAL_PREDICTIVE_DECODER_H_

#include "draco/compression/mesh/mesh_edgebreaker_traversal_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// Decoder for traversal encoded with the
// MeshEdgebreakerTraversalPredictiveEncoder. The decoder maintains valences
// of the decoded portion of the traversed mesh and it uses them to predict
// symbols that are about to be decoded.
class MeshEdgebreakerTraversalPredictiveDecoder
    : public MeshEdgebreakerTraversalDecoder {
 public:
  MeshEdgebreakerTraversalPredictiveDecoder()
      : corner_table_(nullptr),
        num_vertices_(0),
        last_symbol_(-1),
        predicted_symbol_(-1) {}
  void Init(MeshEdgebreakerDecoderImplInterface *decoder) {
    MeshEdgebreakerTraversalDecoder::Init(decoder);
    corner_table_ = decoder->GetCornerTable();
  }
  void SetNumEncodedVertices(int num_vertices) { num_vertices_ = num_vertices; }

  bool Start(DecoderBuffer *out_buffer) {
    if (!MeshEdgebreakerTraversalDecoder::Start(out_buffer)) {
      return false;
    }
    int32_t num_split_symbols;
    if (!out_buffer->Decode(&num_split_symbols) || num_split_symbols < 0)
      return false;
    if (num_split_symbols >= num_vertices_) {
      return false;
    }
    // Set the valences of all initial vertices to 0.
    vertex_valences_.resize(num_vertices_, 0);
    if (!prediction_decoder_.StartDecoding(out_buffer)) {
      return false;
    }
    return true;
  }

  inline uint32_t DecodeSymbol() {
    // First check if we have a predicted symbol.
    if (predicted_symbol_ != -1) {
      // Double check that the predicted symbol was predicted correctly.
      if (prediction_decoder_.DecodeNextBit()) {
        last_symbol_ = predicted_symbol_;
        return predicted_symbol_;
      }
    }
    // We don't have a predicted symbol or the symbol was mis-predicted.
    // Decode it directly.
    last_symbol_ = MeshEdgebreakerTraversalDecoder::DecodeSymbol();
    return last_symbol_;
  }

  inline void NewActiveCornerReached(CornerIndex corner) {
    const CornerIndex next = corner_table_->Next(corner);
    const CornerIndex prev = corner_table_->Previous(corner);
    // Update valences.
    switch (last_symbol_) {
      case TOPOLOGY_C:
      case TOPOLOGY_S:
        vertex_valences_[corner_table_->Vertex(next).value()] += 1;
        vertex_valences_[corner_table_->Vertex(prev).value()] += 1;
        break;
      case TOPOLOGY_R:
        vertex_valences_[corner_table_->Vertex(corner).value()] += 1;
        vertex_valences_[corner_table_->Vertex(next).value()] += 1;
        vertex_valences_[corner_table_->Vertex(prev).value()] += 2;
        break;
      case TOPOLOGY_L:
        vertex_valences_[corner_table_->Vertex(corner).value()] += 1;
        vertex_valences_[corner_table_->Vertex(next).value()] += 2;
        vertex_valences_[corner_table_->Vertex(prev).value()] += 1;
        break;
      case TOPOLOGY_E:
        vertex_valences_[corner_table_->Vertex(corner).value()] += 2;
        vertex_valences_[corner_table_->Vertex(next).value()] += 2;
        vertex_valences_[corner_table_->Vertex(prev).value()] += 2;
        break;
      default:
        break;
    }
    // Compute the new predicted symbol.
    if (last_symbol_ == TOPOLOGY_C || last_symbol_ == TOPOLOGY_R) {
      const VertexIndex pivot =
          corner_table_->Vertex(corner_table_->Next(corner));
      if (vertex_valences_[pivot.value()] < 6) {
        predicted_symbol_ = TOPOLOGY_R;
      } else {
        predicted_symbol_ = TOPOLOGY_C;
      }
    } else {
      predicted_symbol_ = -1;
    }
  }

  inline void MergeVertices(VertexIndex dest, VertexIndex source) {
    // Update valences on the merged vertices.
    vertex_valences_[dest.value()] += vertex_valences_[source.value()];
  }

 private:
  const CornerTable *corner_table_;
  int num_vertices_;
  std::vector<int> vertex_valences_;
  BinaryDecoder prediction_decoder_;
  int last_symbol_;
  int predicted_symbol_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_TRAVERSAL_PREDICTIVE_DECODER_H_
#endif
