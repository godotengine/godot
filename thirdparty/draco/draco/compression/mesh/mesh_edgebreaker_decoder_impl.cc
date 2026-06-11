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
#include "draco/compression/mesh/mesh_edgebreaker_decoder_impl.h"

#include <algorithm>
#include <cstdint>

#include "draco/compression/attributes/sequential_attribute_decoders_controller.h"
#include "draco/compression/mesh/mesh_edgebreaker_decoder.h"
#include "draco/compression/mesh/mesh_edgebreaker_traversal_predictive_decoder.h"
#include "draco/compression/mesh/mesh_edgebreaker_traversal_valence_decoder.h"
#include "draco/compression/mesh/traverser/depth_first_traverser.h"
#include "draco/compression/mesh/traverser/max_prediction_degree_traverser.h"
#include "draco/compression/mesh/traverser/mesh_attribute_indices_encoding_observer.h"
#include "draco/compression/mesh/traverser/mesh_traversal_sequencer.h"
#include "draco/compression/mesh/traverser/traverser_base.h"
#include "draco/mesh/corner_table_iterators.h"

namespace draco {

// Types of "free" edges that are used during topology decoding.
// A free edge is an edge that is connected to one face only.
// All edge types are stored in the opposite_corner_id_ array, where each
// edge "e" is uniquely identified by the opposite corner "C" in its parent
// triangle:
//          *
//         /C\
//        /   \
//       /  e  \
//      *-------*
// For more description about how the edges are used, see comment inside
// ZipConnectivity() method.

template <class TraversalDecoder>
MeshEdgebreakerDecoderImpl<TraversalDecoder>::MeshEdgebreakerDecoderImpl()
    : decoder_(nullptr),
      last_symbol_id_(-1),
      last_vert_id_(-1),
      last_face_id_(-1),
      num_new_vertices_(0),
      num_encoded_vertices_(0),
      pos_data_decoder_id_(-1) {}

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::Init(
    MeshEdgebreakerDecoder *decoder) {
  decoder_ = decoder;
  return true;
}

template <class TraversalDecoder>
const MeshAttributeCornerTable *
MeshEdgebreakerDecoderImpl<TraversalDecoder>::GetAttributeCornerTable(
    int att_id) const {
  for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
    const int decoder_id = attribute_data_[i].decoder_id;
    if (decoder_id < 0 || decoder_id >= decoder_->num_attributes_decoders()) {
      continue;
    }
    const AttributesDecoderInterface *const dec =
        decoder_->attributes_decoder(decoder_id);
    for (int j = 0; j < dec->GetNumAttributes(); ++j) {
      if (dec->GetAttributeId(j) == att_id) {
        if (attribute_data_[i].is_connectivity_used) {
          return &attribute_data_[i].connectivity_data;
        }
        return nullptr;
      }
    }
  }
  return nullptr;
}

template <class TraversalDecoder>
const MeshAttributeIndicesEncodingData *
MeshEdgebreakerDecoderImpl<TraversalDecoder>::GetAttributeEncodingData(
    int att_id) const {
  for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
    const int decoder_id = attribute_data_[i].decoder_id;
    if (decoder_id < 0 || decoder_id >= decoder_->num_attributes_decoders()) {
      continue;
    }
    const AttributesDecoderInterface *const dec =
        decoder_->attributes_decoder(decoder_id);
    for (int j = 0; j < dec->GetNumAttributes(); ++j) {
      if (dec->GetAttributeId(j) == att_id) {
        return &attribute_data_[i].encoding_data;
      }
    }
  }
  return &pos_encoding_data_;
}

template <class TraversalDecoder>
template <class TraverserT>
std::unique_ptr<PointsSequencer>
MeshEdgebreakerDecoderImpl<TraversalDecoder>::CreateVertexTraversalSequencer(
    MeshAttributeIndicesEncodingData *encoding_data) {
  typedef typename TraverserT::TraversalObserver AttObserver;
  typedef typename TraverserT::CornerTable CornerTable;

  const Mesh *mesh = decoder_->mesh();
  std::unique_ptr<MeshTraversalSequencer<TraverserT>> traversal_sequencer(
      new MeshTraversalSequencer<TraverserT>(mesh, encoding_data));

  AttObserver att_observer(corner_table_.get(), mesh, traversal_sequencer.get(),
                           encoding_data);

  TraverserT att_traverser;
  att_traverser.Init(corner_table_.get(), att_observer);

  traversal_sequencer->SetTraverser(att_traverser);
  return std::move(traversal_sequencer);
}

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::CreateAttributesDecoder(
    int32_t att_decoder_id) {
  int8_t att_data_id;
  if (!decoder_->buffer()->Decode(&att_data_id)) {
    return false;
  }
  uint8_t decoder_type;
  if (!decoder_->buffer()->Decode(&decoder_type)) {
    return false;
  }

  if (att_data_id >= 0) {
    if (att_data_id >= attribute_data_.size()) {
      return false;  // Unexpected attribute data.
    }

    // Ensure that the attribute data is not mapped to a different attributes
    // decoder already.
    if (attribute_data_[att_data_id].decoder_id >= 0) {
      return false;
    }

    attribute_data_[att_data_id].decoder_id = att_decoder_id;
  } else {
    // Assign the attributes decoder to |pos_encoding_data_|.
    if (pos_data_decoder_id_ >= 0) {
      return false;  // Some other decoder is already using the data. Error.
    }
    pos_data_decoder_id_ = att_decoder_id;
  }

  MeshTraversalMethod traversal_method = MESH_TRAVERSAL_DEPTH_FIRST;
  if (decoder_->bitstream_version() >= DRACO_BITSTREAM_VERSION(1, 2)) {
    uint8_t traversal_method_encoded;
    if (!decoder_->buffer()->Decode(&traversal_method_encoded)) {
      return false;
    }
    // Check that decoded traversal method is valid.
    if (traversal_method_encoded >= NUM_TRAVERSAL_METHODS) {
      return false;
    }
    traversal_method =
        static_cast<MeshTraversalMethod>(traversal_method_encoded);
  }

  const Mesh *mesh = decoder_->mesh();
  std::unique_ptr<PointsSequencer> sequencer;

  if (decoder_type == MESH_VERTEX_ATTRIBUTE) {
    // Per-vertex attribute decoder.

    MeshAttributeIndicesEncodingData *encoding_data = nullptr;
    if (att_data_id < 0) {
      encoding_data = &pos_encoding_data_;
    } else {
      encoding_data = &attribute_data_[att_data_id].encoding_data;
      // Mark the attribute connectivity data invalid to ensure it's not used
      // later on.
      attribute_data_[att_data_id].is_connectivity_used = false;
    }
    // Defining sequencer via a traversal scheme.
    if (traversal_method == MESH_TRAVERSAL_PREDICTION_DEGREE) {
      typedef MeshAttributeIndicesEncodingObserver<CornerTable> AttObserver;
      typedef MaxPredictionDegreeTraverser<CornerTable, AttObserver>
          AttTraverser;
      sequencer = CreateVertexTraversalSequencer<AttTraverser>(encoding_data);
    } else if (traversal_method == MESH_TRAVERSAL_DEPTH_FIRST) {
      typedef MeshAttributeIndicesEncodingObserver<CornerTable> AttObserver;
      typedef DepthFirstTraverser<CornerTable, AttObserver> AttTraverser;
      sequencer = CreateVertexTraversalSequencer<AttTraverser>(encoding_data);
    } else {
      return false;  // Unsupported method
    }
  } else {
    if (traversal_method != MESH_TRAVERSAL_DEPTH_FIRST) {
      return false;  // Unsupported method.
    }
    if (att_data_id < 0) {
      return false;  // Attribute data must be specified.
    }

    // Per-corner attribute decoder.

    typedef MeshAttributeIndicesEncodingObserver<MeshAttributeCornerTable>
        AttObserver;
    typedef DepthFirstTraverser<MeshAttributeCornerTable, AttObserver>
        AttTraverser;

    MeshAttributeIndicesEncodingData *const encoding_data =
        &attribute_data_[att_data_id].encoding_data;
    const MeshAttributeCornerTable *const corner_table =
        &attribute_data_[att_data_id].connectivity_data;

    std::unique_ptr<MeshTraversalSequencer<AttTraverser>> traversal_sequencer(
        new MeshTraversalSequencer<AttTraverser>(mesh, encoding_data));

    AttObserver att_observer(corner_table, mesh, traversal_sequencer.get(),
                             encoding_data);

    AttTraverser att_traverser;
    att_traverser.Init(corner_table, att_observer);

    traversal_sequencer->SetTraverser(att_traverser);
    sequencer = std::move(traversal_sequencer);
  }

  if (!sequencer) {
    return false;
  }

  std::unique_ptr<SequentialAttributeDecodersController> att_controller(
      new SequentialAttributeDecodersController(std::move(sequencer)));

  return decoder_->SetAttributesDecoder(att_decoder_id,
                                        std::move(att_controller));
}

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::DecodeConnectivity() {
  num_new_vertices_ = 0;
  new_to_parent_vertex_map_.clear();
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
    uint32_t num_new_verts;
    if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
      if (!decoder_->buffer()->Decode(&num_new_verts)) {
        return false;
      }
    } else {
      if (!DecodeVarint(&num_new_verts, decoder_->buffer())) {
        return false;
      }
    }
    num_new_vertices_ = num_new_verts;
  }
#endif

  uint32_t num_encoded_vertices;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_->buffer()->Decode(&num_encoded_vertices)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_encoded_vertices, decoder_->buffer())) {
      return false;
    }
  }
  num_encoded_vertices_ = num_encoded_vertices;

  uint32_t num_faces;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_->buffer()->Decode(&num_faces)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_faces, decoder_->buffer())) {
      return false;
    }
  }
  if (num_faces > std::numeric_limits<CornerIndex::ValueType>::max() / 3) {
    return false;  // Draco cannot handle this many faces.
  }

  if (static_cast<uint32_t>(num_encoded_vertices_) > num_faces * 3) {
    return false;  // There cannot be more vertices than 3 * num_faces.
  }

  // Minimum number of edges of the mesh assuming each edge is shared between
  // two faces.
  const uint32_t min_num_face_edges = 3 * num_faces / 2;

  // Maximum number of edges that can exist between |num_encoded_vertices_|.
  // This is based on graph theory assuming simple connected graph.
  const uint64_t num_encoded_vertices_64 =
      static_cast<uint64_t>(num_encoded_vertices_);
  const uint64_t max_num_vertex_edges =
      num_encoded_vertices_64 * (num_encoded_vertices_64 - 1) / 2;
  if (max_num_vertex_edges < min_num_face_edges) {
    // It is impossible to construct a manifold mesh with these properties.
    return false;
  }

  uint8_t num_attribute_data;
  if (!decoder_->buffer()->Decode(&num_attribute_data)) {
    return false;
  }

  uint32_t num_encoded_symbols;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_->buffer()->Decode(&num_encoded_symbols)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_encoded_symbols, decoder_->buffer())) {
      return false;
    }
  }

  if (num_faces < num_encoded_symbols) {
    // Number of faces needs to be the same or greater than the number of
    // symbols (it can be greater because the initial face may not be encoded as
    // a symbol).
    return false;
  }
  const uint32_t max_encoded_faces =
      num_encoded_symbols + (num_encoded_symbols / 3);
  if (num_faces > max_encoded_faces) {
    // Faces can only be 1 1/3 times bigger than number of encoded symbols. This
    // could only happen if all new encoded components started with interior
    // triangles. E.g. A mesh with multiple tetrahedrons.
    return false;
  }

  uint32_t num_encoded_split_symbols;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_->buffer()->Decode(&num_encoded_split_symbols)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_encoded_split_symbols, decoder_->buffer())) {
      return false;
    }
  }

  if (num_encoded_split_symbols > num_encoded_symbols) {
    return false;  // Split symbols are a sub-set of all symbols.
  }

  // Decode topology (connectivity).
  vertex_traversal_length_.clear();
  corner_table_ = std::unique_ptr<CornerTable>(new CornerTable());
  if (corner_table_ == nullptr) {
    return false;
  }
  processed_corner_ids_.clear();
  processed_corner_ids_.reserve(num_faces);
  processed_connectivity_corners_.clear();
  processed_connectivity_corners_.reserve(num_faces);
  topology_split_data_.clear();
  hole_event_data_.clear();
  init_face_configurations_.clear();
  init_corners_.clear();

  last_symbol_id_ = -1;
  last_face_id_ = -1;
  last_vert_id_ = -1;

  attribute_data_.clear();
  // Add one attribute data for each attribute decoder.
  attribute_data_.resize(num_attribute_data);

  if (!corner_table_->Reset(
          num_faces, num_encoded_vertices_ + num_encoded_split_symbols)) {
    return false;
  }

  // Start with all vertices marked as holes (boundaries).
  // Only vertices decoded with TOPOLOGY_C symbol (and the initial face) will
  // be marked as non hole vertices. We need to allocate the array larger
  // because split symbols can create extra vertices during the decoding
  // process (these extra vertices are then eliminated during deduplication).
  is_vert_hole_.assign(num_encoded_vertices_ + num_encoded_split_symbols, true);

#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  int32_t topology_split_decoded_bytes = -1;
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
    uint32_t encoded_connectivity_size;
    if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
      if (!decoder_->buffer()->Decode(&encoded_connectivity_size)) {
        return false;
      }
    } else {
      if (!DecodeVarint(&encoded_connectivity_size, decoder_->buffer())) {
        return false;
      }
    }
    if (encoded_connectivity_size == 0 ||
        encoded_connectivity_size > decoder_->buffer()->remaining_size()) {
      return false;
    }
    DecoderBuffer event_buffer;
    event_buffer.Init(
        decoder_->buffer()->data_head() + encoded_connectivity_size,
        decoder_->buffer()->remaining_size() - encoded_connectivity_size,
        decoder_->buffer()->bitstream_version());
    // Decode hole and topology split events.
    topology_split_decoded_bytes =
        DecodeHoleAndTopologySplitEvents(&event_buffer);
    if (topology_split_decoded_bytes == -1) {
      return false;
    }

  } else
#endif
  {
    if (DecodeHoleAndTopologySplitEvents(decoder_->buffer()) == -1) {
      return false;
    }
  }

  traversal_decoder_.Init(this);
  // Add one extra vertex for each split symbol.
  traversal_decoder_.SetNumEncodedVertices(num_encoded_vertices_ +
                                           num_encoded_split_symbols);
  traversal_decoder_.SetNumAttributeData(num_attribute_data);

  DecoderBuffer traversal_end_buffer;
  if (!traversal_decoder_.Start(&traversal_end_buffer)) {
    return false;
  }

  const int num_connectivity_verts = DecodeConnectivity(num_encoded_symbols);
  if (num_connectivity_verts == -1) {
    return false;
  }

  // Set the main buffer to the end of the traversal.
  decoder_->buffer()->Init(traversal_end_buffer.data_head(),
                           traversal_end_buffer.remaining_size(),
                           decoder_->buffer()->bitstream_version());

#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
    // Skip topology split data that was already decoded earlier.
    decoder_->buffer()->Advance(topology_split_decoded_bytes);
  }
#endif

  // Decode connectivity of non-position attributes.
  if (!attribute_data_.empty()) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
    if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 1)) {
      for (CornerIndex ci(0); ci < corner_table_->num_corners(); ci += 3) {
        if (!DecodeAttributeConnectivitiesOnFaceLegacy(ci)) {
          return false;
        }
      }

    } else
#endif
    {
      for (CornerIndex ci(0); ci < corner_table_->num_corners(); ci += 3) {
        if (!DecodeAttributeConnectivitiesOnFace(ci)) {
          return false;
        }
      }
    }
  }
  traversal_decoder_.Done();

  // Decode attribute connectivity.
  // Prepare data structure for decoding non-position attribute connectivity.
  for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
    attribute_data_[i].connectivity_data.InitEmpty(corner_table_.get());
    // Add all seams.
    for (int32_t c : attribute_data_[i].attribute_seam_corners) {
      attribute_data_[i].connectivity_data.AddSeamEdge(CornerIndex(c));
    }
    // Recompute vertices from the newly added seam edges.
    if (!attribute_data_[i].connectivity_data.RecomputeVertices(nullptr,
                                                                nullptr)) {
      return false;
    }
  }

  pos_encoding_data_.Init(corner_table_->num_vertices());
  for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
    // For non-position attributes, preallocate the vertex to value mapping
    // using the maximum number of vertices from the base corner table and the
    // attribute corner table (since the attribute decoder may use either of
    // it).
    int32_t att_connectivity_verts =
        attribute_data_[i].connectivity_data.num_vertices();
    if (att_connectivity_verts < corner_table_->num_vertices()) {
      att_connectivity_verts = corner_table_->num_vertices();
    }
    attribute_data_[i].encoding_data.Init(att_connectivity_verts);
  }
  if (!AssignPointsToCorners(num_connectivity_verts)) {
    return false;
  }
  return true;
}

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::OnAttributesDecoded() {
  return true;
}

template <class TraversalDecoder>
int MeshEdgebreakerDecoderImpl<TraversalDecoder>::DecodeConnectivity(
    int num_symbols) {
  // Algorithm does the reverse decoding of the symbols encoded with the
  // edgebreaker method. The reverse decoding always keeps track of the active
  // edge identified by its opposite corner (active corner). New faces are
  // always added to this active edge. There may be multiple active corners at
  // one time that either correspond to separate mesh components or to
  // sub-components of one mesh that are going to be merged together using the
  // TOPOLOGY_S symbol. We can store these active edges on a stack, because the
  // decoder always processes only the latest active edge. TOPOLOGY_S then
  // removes the top edge from the stack and TOPOLOGY_E adds a new edge to the
  // stack.
  std::vector<CornerIndex> active_corner_stack;

  // Additional active edges may be added as a result of topology split events.
  // They can be added in arbitrary order, but we always know the split symbol
  // id they belong to, so we can address them using this symbol id.
  std::unordered_map<int, CornerIndex> topology_split_active_corners;

  // Vector used for storing vertices that were marked as isolated during the
  // decoding process. Currently used only when the mesh doesn't contain any
  // non-position connectivity data.
  std::vector<VertexIndex> invalid_vertices;
  const bool remove_invalid_vertices = attribute_data_.empty();

  int max_num_vertices = static_cast<int>(is_vert_hole_.size());
  int num_faces = 0;
  for (int symbol_id = 0; symbol_id < num_symbols; ++symbol_id) {
    const FaceIndex face(num_faces++);
    // Used to flag cases where we need to look for topology split events.
    bool check_topology_split = false;
    const uint32_t symbol = traversal_decoder_.DecodeSymbol();
    if (symbol == TOPOLOGY_C) {
      // Create a new face between two edges on the open boundary.
      // The first edge is opposite to the corner "a" from the image below.
      // The other edge is opposite to the corner "b" that can be reached
      // through a CCW traversal around the vertex "v".
      // One new active boundary edge is created, opposite to the new corner
      // "x".
      //
      //     *-------*
      //    / \     / \
      //   /   \   /   \
      //  /     \ /     \
      // *-------v-------*
      //  \b    /x\    a/
      //   \   /   \   /
      //    \ /  C  \ /
      //     *.......*

      // Find the corner "b" from the corner "a" which is the corner on the
      // top of the active stack.
      if (active_corner_stack.empty()) {
        return -1;
      }

      const CornerIndex corner_a = active_corner_stack.back();
      const VertexIndex vertex_x =
          corner_table_->Vertex(corner_table_->Next(corner_a));
      const CornerIndex corner_b =
          corner_table_->Next(corner_table_->LeftMostCorner(vertex_x));

      if (corner_a == corner_b) {
        // All matched corners must be different.
        return -1;
      }
      if (corner_table_->Opposite(corner_a) != kInvalidCornerIndex ||
          corner_table_->Opposite(corner_b) != kInvalidCornerIndex) {
        // One of the corners is already opposite to an existing face, which
        // should not happen unless the input was tampered with.
        return -1;
      }

      // New tip corner.
      const CornerIndex corner(3 * face.value());
      // Update opposite corner mappings.
      SetOppositeCorners(corner_a, corner + 1);
      SetOppositeCorners(corner_b, corner + 2);

      // Update vertex mapping.
      const VertexIndex vert_a_prev =
          corner_table_->Vertex(corner_table_->Previous(corner_a));
      const VertexIndex vert_b_next =
          corner_table_->Vertex(corner_table_->Next(corner_b));
      if (vertex_x == vert_a_prev || vertex_x == vert_b_next) {
        // Encoding is invalid, because face vertices are degenerate.
        return -1;
      }
      corner_table_->MapCornerToVertex(corner, vertex_x);
      corner_table_->MapCornerToVertex(corner + 1, vert_b_next);
      corner_table_->MapCornerToVertex(corner + 2, vert_a_prev);
      corner_table_->SetLeftMostCorner(vert_a_prev, corner + 2);
      // Mark the vertex |x| as interior.
      is_vert_hole_[vertex_x.value()] = false;
      // Update the corner on the active stack.
      active_corner_stack.back() = corner;
    } else if (symbol == TOPOLOGY_R || symbol == TOPOLOGY_L) {
      // Create a new face extending from the open boundary edge opposite to the
      // corner "a" from the image below. Two new boundary edges are created
      // opposite to corners "r" and "l". New active corner is set to either "r"
      // or "l" depending on the decoded symbol. One new vertex is created
      // at the opposite corner to corner "a".
      //     *-------*
      //    /a\     / \
      //   /   \   /   \
      //  /     \ /     \
      // *-------v-------*
      //  .l   r.
      //   .   .
      //    . .
      //     *
      if (active_corner_stack.empty()) {
        return -1;
      }
      const CornerIndex corner_a = active_corner_stack.back();
      if (corner_table_->Opposite(corner_a) != kInvalidCornerIndex) {
        // Active corner is already opposite to an existing face, which should
        // not happen unless the input was tampered with.
        return -1;
      }

      // First corner on the new face is either corner "l" or "r".
      const CornerIndex corner(3 * face.value());
      CornerIndex opp_corner, corner_l, corner_r;
      if (symbol == TOPOLOGY_R) {
        // "r" is the new first corner.
        opp_corner = corner + 2;
        corner_l = corner + 1;
        corner_r = corner;
      } else {
        // "l" is the new first corner.
        opp_corner = corner + 1;
        corner_l = corner;
        corner_r = corner + 2;
      }
      SetOppositeCorners(opp_corner, corner_a);
      // Update vertex mapping.
      const VertexIndex new_vert_index = corner_table_->AddNewVertex();

      if (corner_table_->num_vertices() > max_num_vertices) {
        return -1;  // Unexpected number of decoded vertices.
      }

      corner_table_->MapCornerToVertex(opp_corner, new_vert_index);
      corner_table_->SetLeftMostCorner(new_vert_index, opp_corner);

      const VertexIndex vertex_r =
          corner_table_->Vertex(corner_table_->Previous(corner_a));
      corner_table_->MapCornerToVertex(corner_r, vertex_r);
      // Update left-most corner on the vertex on the |corner_r|.
      corner_table_->SetLeftMostCorner(vertex_r, corner_r);

      corner_table_->MapCornerToVertex(
          corner_l, corner_table_->Vertex(corner_table_->Next(corner_a)));
      active_corner_stack.back() = corner;
      check_topology_split = true;
    } else if (symbol == TOPOLOGY_S) {
      // Create a new face that merges two last active edges from the active
      // stack. No new vertex is created, but two vertices at corners "p" and
      // "n" need to be merged into a single vertex.
      //
      // *-------v-------*
      //  \a   p/x\n   b/
      //   \   /   \   /
      //    \ /  S  \ /
      //     *.......*
      //
      if (active_corner_stack.empty()) {
        return -1;
      }
      const CornerIndex corner_b = active_corner_stack.back();
      active_corner_stack.pop_back();

      // Corner "a" can correspond either to a normal active edge, or to an edge
      // created from the topology split event.
      const auto it = topology_split_active_corners.find(symbol_id);
      if (it != topology_split_active_corners.end()) {
        // Topology split event. Move the retrieved edge to the stack.
        active_corner_stack.push_back(it->second);
      }
      if (active_corner_stack.empty()) {
        return -1;
      }
      const CornerIndex corner_a = active_corner_stack.back();

      if (corner_a == corner_b) {
        // All matched corners must be different.
        return -1;
      }
      if (corner_table_->Opposite(corner_a) != kInvalidCornerIndex ||
          corner_table_->Opposite(corner_b) != kInvalidCornerIndex) {
        // One of the corners is already opposite to an existing face, which
        // should not happen unless the input was tampered with.
        return -1;
      }

      // First corner on the new face is corner "x" from the image above.
      const CornerIndex corner(3 * face.value());
      // Update the opposite corner mapping.
      SetOppositeCorners(corner_a, corner + 2);
      SetOppositeCorners(corner_b, corner + 1);
      // Update vertices. For the vertex at corner "x", use the vertex id from
      // the corner "p".
      const VertexIndex vertex_p =
          corner_table_->Vertex(corner_table_->Previous(corner_a));
      corner_table_->MapCornerToVertex(corner, vertex_p);
      corner_table_->MapCornerToVertex(
          corner + 1, corner_table_->Vertex(corner_table_->Next(corner_a)));
      const VertexIndex vert_b_prev =
          corner_table_->Vertex(corner_table_->Previous(corner_b));
      corner_table_->MapCornerToVertex(corner + 2, vert_b_prev);
      corner_table_->SetLeftMostCorner(vert_b_prev, corner + 2);
      CornerIndex corner_n = corner_table_->Next(corner_b);
      const VertexIndex vertex_n = corner_table_->Vertex(corner_n);
      traversal_decoder_.MergeVertices(vertex_p, vertex_n);
      // Update the left most corner on the newly merged vertex.
      corner_table_->SetLeftMostCorner(vertex_p,
                                       corner_table_->LeftMostCorner(vertex_n));

      // Also update the vertex id at corner "n" and all corners that are
      // connected to it in the CCW direction.
      const CornerIndex first_corner = corner_n;
      while (corner_n != kInvalidCornerIndex) {
        corner_table_->MapCornerToVertex(corner_n, vertex_p);
        corner_n = corner_table_->SwingLeft(corner_n);
        if (corner_n == first_corner) {
          // We reached the start again which should not happen for split
          // symbols.
          return -1;
        }
      }
      // Make sure the old vertex n is now mapped to an invalid corner (make it
      // isolated).
      corner_table_->MakeVertexIsolated(vertex_n);
      if (remove_invalid_vertices) {
        invalid_vertices.push_back(vertex_n);
      }
      active_corner_stack.back() = corner;
    } else if (symbol == TOPOLOGY_E) {
      const CornerIndex corner(3 * face.value());
      const VertexIndex first_vert_index = corner_table_->AddNewVertex();
      // Create three new vertices at the corners of the new face.
      corner_table_->MapCornerToVertex(corner, first_vert_index);
      corner_table_->MapCornerToVertex(corner + 1,
                                       corner_table_->AddNewVertex());
      corner_table_->MapCornerToVertex(corner + 2,
                                       corner_table_->AddNewVertex());

      if (corner_table_->num_vertices() > max_num_vertices) {
        return -1;  // Unexpected number of decoded vertices.
      }

      corner_table_->SetLeftMostCorner(first_vert_index, corner);
      corner_table_->SetLeftMostCorner(first_vert_index + 1, corner + 1);
      corner_table_->SetLeftMostCorner(first_vert_index + 2, corner + 2);
      // Add the tip corner to the active stack.
      active_corner_stack.push_back(corner);
      check_topology_split = true;
    } else {
      // Error. Unknown symbol decoded.
      return -1;
    }
    // Inform the traversal decoder that a new corner has been reached.
    traversal_decoder_.NewActiveCornerReached(active_corner_stack.back());

    if (check_topology_split) {
      // Check for topology splits happens only for TOPOLOGY_L, TOPOLOGY_R and
      // TOPOLOGY_E symbols because those are the symbols that correspond to
      // faces that can be directly connected a TOPOLOGY_S face through the
      // topology split event.
      // If a topology split is detected, we need to add a new active edge
      // onto the active_corner_stack because it will be used later when the
      // corresponding TOPOLOGY_S event is decoded.

      // Symbol id used by the encoder (reverse).
      const int encoder_symbol_id = num_symbols - symbol_id - 1;
      EdgeFaceName split_edge;
      int encoder_split_symbol_id;
      while (IsTopologySplit(encoder_symbol_id, &split_edge,
                             &encoder_split_symbol_id)) {
        if (encoder_split_symbol_id < 0) {
          return -1;  // Wrong split symbol id.
        }
        // Symbol was part of a topology split. Now we need to determine which
        // edge should be added to the active edges stack.
        const CornerIndex act_top_corner = active_corner_stack.back();
        // The current symbol has one active edge (stored in act_top_corner) and
        // two remaining inactive edges that are attached to it.
        //              *
        //             / \
        //  left_edge /   \ right_edge
        //           /     \
        //          *.......*
        //         active_edge

        CornerIndex new_active_corner;
        if (split_edge == RIGHT_FACE_EDGE) {
          new_active_corner = corner_table_->Next(act_top_corner);
        } else {
          new_active_corner = corner_table_->Previous(act_top_corner);
        }
        // Add the new active edge.
        // Convert the encoder split symbol id to decoder symbol id.
        const int decoder_split_symbol_id =
            num_symbols - encoder_split_symbol_id - 1;
        topology_split_active_corners[decoder_split_symbol_id] =
            new_active_corner;
      }
    }
  }
  if (corner_table_->num_vertices() > max_num_vertices) {
    return -1;  // Unexpected number of decoded vertices.
  }
  // Decode start faces and connect them to the faces from the active stack.
  while (!active_corner_stack.empty()) {
    const CornerIndex corner = active_corner_stack.back();
    active_corner_stack.pop_back();
    const bool interior_face =
        traversal_decoder_.DecodeStartFaceConfiguration();
    if (interior_face) {
      // The start face is interior, we need to find three corners that are
      // opposite to it. The first opposite corner "a" is the corner from the
      // top of the active corner stack and the remaining two corners "b" and
      // "c" are then the next corners from the left-most corners of vertices
      // "n" and "x" respectively.
      //
      //           *-------*
      //          / \     / \
      //         /   \   /   \
      //        /     \ /     \
      //       *-------p-------*
      //      / \a    . .    c/ \
      //     /   \   .   .   /   \
      //    /     \ .  I  . /     \
      //   *-------n.......x------*
      //    \     / \     / \     /
      //     \   /   \   /   \   /
      //      \ /     \b/     \ /
      //       *-------*-------*
      //

      if (num_faces >= corner_table_->num_faces()) {
        return -1;  // More faces than expected added to the mesh.
      }

      const CornerIndex corner_a = corner;
      const VertexIndex vert_n =
          corner_table_->Vertex(corner_table_->Next(corner_a));
      const CornerIndex corner_b =
          corner_table_->Next(corner_table_->LeftMostCorner(vert_n));

      const VertexIndex vert_x =
          corner_table_->Vertex(corner_table_->Next(corner_b));
      const CornerIndex corner_c =
          corner_table_->Next(corner_table_->LeftMostCorner(vert_x));

      if (corner == corner_b || corner == corner_c || corner_b == corner_c) {
        // All matched corners must be different.
        return -1;
      }
      if (corner_table_->Opposite(corner) != kInvalidCornerIndex ||
          corner_table_->Opposite(corner_b) != kInvalidCornerIndex ||
          corner_table_->Opposite(corner_c) != kInvalidCornerIndex) {
        // One of the corners is already opposite to an existing face, which
        // should not happen unless the input was tampered with.
        return -1;
      }

      const VertexIndex vert_p =
          corner_table_->Vertex(corner_table_->Next(corner_c));

      const FaceIndex face(num_faces++);
      // The first corner of the initial face is the corner opposite to "a".
      const CornerIndex new_corner(3 * face.value());
      SetOppositeCorners(new_corner, corner);
      SetOppositeCorners(new_corner + 1, corner_b);
      SetOppositeCorners(new_corner + 2, corner_c);

      // Map new corners to existing vertices.
      corner_table_->MapCornerToVertex(new_corner, vert_x);
      corner_table_->MapCornerToVertex(new_corner + 1, vert_p);
      corner_table_->MapCornerToVertex(new_corner + 2, vert_n);

      // Mark all three vertices as interior.
      for (int ci = 0; ci < 3; ++ci) {
        is_vert_hole_[corner_table_->Vertex(new_corner + ci).value()] = false;
      }

      init_face_configurations_.push_back(true);
      init_corners_.push_back(new_corner);
    } else {
      // The initial face wasn't interior and the traversal had to start from
      // an open boundary. In this case no new face is added, but we need to
      // keep record about the first opposite corner to this boundary.
      init_face_configurations_.push_back(false);
      init_corners_.push_back(corner);
    }
  }
  if (num_faces != corner_table_->num_faces()) {
    return -1;  // Unexpected number of decoded faces.
  }

  int num_vertices = corner_table_->num_vertices();
  // If any vertex was marked as isolated, we want to remove it from the corner
  // table to ensure that all vertices in range <0, num_vertices> are valid.
  for (const VertexIndex invalid_vert : invalid_vertices) {
    // Find the last valid vertex and swap it with the isolated vertex.
    VertexIndex src_vert(num_vertices - 1);
    while (corner_table_->LeftMostCorner(src_vert) == kInvalidCornerIndex) {
      // The last vertex is invalid, proceed to the previous one.
      src_vert = VertexIndex(--num_vertices - 1);
    }
    if (src_vert < invalid_vert) {
      continue;  // No need to swap anything.
    }

    // Remap all corners mapped to |src_vert| to |invalid_vert|.
    VertexCornersIterator<CornerTable> vcit(corner_table_.get(), src_vert);
    for (; !vcit.End(); ++vcit) {
      const CornerIndex cid = vcit.Corner();
      if (corner_table_->Vertex(cid) != src_vert) {
        // Vertex mapped to |cid| was not |src_vert|. This indicates corrupted
        // data and we should terminate the decoding.
        return -1;
      }
      corner_table_->MapCornerToVertex(cid, invalid_vert);
    }
    corner_table_->SetLeftMostCorner(invalid_vert,
                                     corner_table_->LeftMostCorner(src_vert));

    // Make the |src_vert| invalid.
    corner_table_->MakeVertexIsolated(src_vert);
    is_vert_hole_[invalid_vert.value()] = is_vert_hole_[src_vert.value()];
    is_vert_hole_[src_vert.value()] = false;

    // The last vertex is now invalid.
    num_vertices--;
  }
  return num_vertices;
}

template <class TraversalDecoder>
int32_t
MeshEdgebreakerDecoderImpl<TraversalDecoder>::DecodeHoleAndTopologySplitEvents(
    DecoderBuffer *decoder_buffer) {
  // Prepare a new decoder from the provided buffer offset.
  uint32_t num_topology_splits;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_buffer->Decode(&num_topology_splits)) {
      return -1;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_topology_splits, decoder_buffer)) {
      return -1;
    }
  }
  if (num_topology_splits > 0) {
    if (num_topology_splits >
        static_cast<uint32_t>(corner_table_->num_faces())) {
      return -1;
    }
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
    if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(1, 2)) {
      for (uint32_t i = 0; i < num_topology_splits; ++i) {
        TopologySplitEventData event_data;
        if (!decoder_buffer->Decode(&event_data.split_symbol_id)) {
          return -1;
        }
        if (!decoder_buffer->Decode(&event_data.source_symbol_id)) {
          return -1;
        }
        uint8_t edge_data;
        if (!decoder_buffer->Decode(&edge_data)) {
          return -1;
        }
        event_data.source_edge = edge_data & 1;
        topology_split_data_.push_back(event_data);
      }

    } else
#endif
    {
      // Decode source and split symbol ids using delta and varint coding. See
      // description in mesh_edgebreaker_encoder_impl.cc for more details.
      int last_source_symbol_id = 0;
      for (uint32_t i = 0; i < num_topology_splits; ++i) {
        TopologySplitEventData event_data;
        uint32_t delta;
        if (!DecodeVarint<uint32_t>(&delta, decoder_buffer)) {
          return -1;
        }
        event_data.source_symbol_id = delta + last_source_symbol_id;
        if (!DecodeVarint<uint32_t>(&delta, decoder_buffer)) {
          return -1;
        }
        if (delta > event_data.source_symbol_id) {
          return -1;
        }
        event_data.split_symbol_id =
            event_data.source_symbol_id - static_cast<int32_t>(delta);
        last_source_symbol_id = event_data.source_symbol_id;
        topology_split_data_.push_back(event_data);
      }
      // Split edges are decoded from a direct bit decoder.
      decoder_buffer->StartBitDecoding(false, nullptr);
      for (uint32_t i = 0; i < num_topology_splits; ++i) {
        uint32_t edge_data;
        if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
          decoder_buffer->DecodeLeastSignificantBits32(2, &edge_data);
        } else {
          decoder_buffer->DecodeLeastSignificantBits32(1, &edge_data);
        }
        TopologySplitEventData &event_data = topology_split_data_[i];
        event_data.source_edge = edge_data & 1;
      }
      decoder_buffer->EndBitDecoding();
    }
  }
  uint32_t num_hole_events = 0;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 0)) {
    if (!decoder_buffer->Decode(&num_hole_events)) {
      return -1;
    }
  } else if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(2, 1)) {
    if (!DecodeVarint(&num_hole_events, decoder_buffer)) {
      return -1;
    }
  }
#endif
  if (num_hole_events > 0) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
    if (decoder_->bitstream_version() < DRACO_BITSTREAM_VERSION(1, 2)) {
      for (uint32_t i = 0; i < num_hole_events; ++i) {
        HoleEventData event_data;
        if (!decoder_buffer->Decode(&event_data)) {
          return -1;
        }
        hole_event_data_.push_back(event_data);
      }

    } else
#endif
    {
      // Decode hole symbol ids using delta and varint coding.
      int last_symbol_id = 0;
      for (uint32_t i = 0; i < num_hole_events; ++i) {
        HoleEventData event_data;
        uint32_t delta;
        if (!DecodeVarint<uint32_t>(&delta, decoder_buffer)) {
          return -1;
        }
        event_data.symbol_id = delta + last_symbol_id;
        last_symbol_id = event_data.symbol_id;
        hole_event_data_.push_back(event_data);
      }
    }
  }
  return static_cast<int32_t>(decoder_buffer->decoded_size());
}

#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::
    DecodeAttributeConnectivitiesOnFaceLegacy(CornerIndex corner) {
  // Three corners of the face.
  const CornerIndex corners[3] = {corner, corner_table_->Next(corner),
                                  corner_table_->Previous(corner)};

  for (int c = 0; c < 3; ++c) {
    const CornerIndex opp_corner = corner_table_->Opposite(corners[c]);
    if (opp_corner == kInvalidCornerIndex) {
      // Don't decode attribute seams on boundary edges (every boundary edge
      // is automatically an attribute seam).
      for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
        attribute_data_[i].attribute_seam_corners.push_back(corners[c].value());
      }
      continue;
    }

    for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
      const bool is_seam = traversal_decoder_.DecodeAttributeSeam(i);
      if (is_seam) {
        attribute_data_[i].attribute_seam_corners.push_back(corners[c].value());
      }
    }
  }
  return true;
}
#endif

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<
    TraversalDecoder>::DecodeAttributeConnectivitiesOnFace(CornerIndex corner) {
  // Three corners of the face.
  const CornerIndex corners[3] = {corner, corner_table_->Next(corner),
                                  corner_table_->Previous(corner)};

  const FaceIndex src_face_id = corner_table_->Face(corner);
  for (int c = 0; c < 3; ++c) {
    const CornerIndex opp_corner = corner_table_->Opposite(corners[c]);
    if (opp_corner == kInvalidCornerIndex) {
      // Don't decode attribute seams on boundary edges (every boundary edge
      // is automatically an attribute seam).
      for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
        attribute_data_[i].attribute_seam_corners.push_back(corners[c].value());
      }
      continue;
    }
    const FaceIndex opp_face_id = corner_table_->Face(opp_corner);
    // Don't decode edges when the opposite face has been already processed.
    if (opp_face_id < src_face_id) {
      continue;
    }

    for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
      const bool is_seam = traversal_decoder_.DecodeAttributeSeam(i);
      if (is_seam) {
        attribute_data_[i].attribute_seam_corners.push_back(corners[c].value());
      }
    }
  }
  return true;
}

template <class TraversalDecoder>
bool MeshEdgebreakerDecoderImpl<TraversalDecoder>::AssignPointsToCorners(
    int num_connectivity_verts) {
  // Map between the existing and deduplicated point ids.
  // Note that at this point we have one point id for each corner of the
  // mesh so there is corner_table_->num_corners() point ids.
  decoder_->mesh()->SetNumFaces(corner_table_->num_faces());

  if (attribute_data_.empty()) {
    // We have connectivity for position only. In this case all vertex indices
    // are equal to point indices.
    for (FaceIndex f(0); f < decoder_->mesh()->num_faces(); ++f) {
      Mesh::Face face;
      const CornerIndex start_corner(3 * f.value());
      for (int c = 0; c < 3; ++c) {
        // Get the vertex index on the corner and use it as a point index.
        const int32_t vert_id = corner_table_->Vertex(start_corner + c).value();
        face[c] = vert_id;
      }
      decoder_->mesh()->SetFace(f, face);
    }
    decoder_->point_cloud()->set_num_points(num_connectivity_verts);
    return true;
  }
  // Else we need to deduplicate multiple attributes.

  // Map between point id and an associated corner id. Only one corner for
  // each point is stored. The corners are used to sample the attribute values
  // in the last stage of the deduplication.
  std::vector<int32_t> point_to_corner_map;
  // Map between every corner and their new point ids.
  std::vector<int32_t> corner_to_point_map(corner_table_->num_corners());
  for (int v = 0; v < corner_table_->num_vertices(); ++v) {
    CornerIndex c = corner_table_->LeftMostCorner(VertexIndex(v));
    if (c == kInvalidCornerIndex) {
      continue;  // Isolated vertex.
    }
    CornerIndex deduplication_first_corner = c;
    if (is_vert_hole_[v]) {
      // If the vertex is on a boundary, start deduplication from the left most
      // corner that is guaranteed to lie on the boundary.
      deduplication_first_corner = c;
    } else {
      // If we are not on the boundary we need to find the first seam (of any
      // attribute).
      for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
        if (!attribute_data_[i].connectivity_data.IsCornerOnSeam(c)) {
          continue;  // No seam for this attribute, ignore it.
        }
        // Else there needs to be at least one seam edge.

        // At this point, we use identity mapping between corners and point ids.
        const VertexIndex vert_id =
            attribute_data_[i].connectivity_data.Vertex(c);
        CornerIndex act_c = corner_table_->SwingRight(c);
        bool seam_found = false;
        while (act_c != c) {
          if (act_c == kInvalidCornerIndex) {
            return false;
          }
          if (attribute_data_[i].connectivity_data.Vertex(act_c) != vert_id) {
            // Attribute seam found. Stop.
            deduplication_first_corner = act_c;
            seam_found = true;
            break;
          }
          act_c = corner_table_->SwingRight(act_c);
        }
        if (seam_found) {
          break;  // No reason to process other attributes if we found a seam.
        }
      }
    }

    // Do a deduplication pass over the corners on the processed vertex.
    // At this point each corner corresponds to one point id and our goal is to
    // merge similar points into a single point id.
    // We do a single pass in a clockwise direction over the corners and we add
    // a new point id whenever one of the attributes change.
    c = deduplication_first_corner;
    // Create a new point.
    corner_to_point_map[c.value()] =
        static_cast<uint32_t>(point_to_corner_map.size());
    point_to_corner_map.push_back(c.value());
    // Traverse in CW direction.
    CornerIndex prev_c = c;
    c = corner_table_->SwingRight(c);
    while (c != kInvalidCornerIndex && c != deduplication_first_corner) {
      bool attribute_seam = false;
      for (uint32_t i = 0; i < attribute_data_.size(); ++i) {
        if (attribute_data_[i].connectivity_data.Vertex(c) !=
            attribute_data_[i].connectivity_data.Vertex(prev_c)) {
          // Attribute index changed from the previous corner. We need to add a
          // new point here.
          attribute_seam = true;
          break;
        }
      }
      if (attribute_seam) {
        corner_to_point_map[c.value()] =
            static_cast<uint32_t>(point_to_corner_map.size());
        point_to_corner_map.push_back(c.value());
      } else {
        corner_to_point_map[c.value()] = corner_to_point_map[prev_c.value()];
      }
      prev_c = c;
      c = corner_table_->SwingRight(c);
    }
  }
  // Add faces.
  for (FaceIndex f(0); f < decoder_->mesh()->num_faces(); ++f) {
    Mesh::Face face;
    for (int c = 0; c < 3; ++c) {
      // Remap old points to the new ones.
      face[c] = corner_to_point_map[3 * f.value() + c];
    }
    decoder_->mesh()->SetFace(f, face);
  }
  decoder_->point_cloud()->set_num_points(
      static_cast<uint32_t>(point_to_corner_map.size()));
  return true;
}

template class MeshEdgebreakerDecoderImpl<MeshEdgebreakerTraversalDecoder>;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
template class MeshEdgebreakerDecoderImpl<
    MeshEdgebreakerTraversalPredictiveDecoder>;
#endif
template class MeshEdgebreakerDecoderImpl<
    MeshEdgebreakerTraversalValenceDecoder>;
}  // namespace draco
