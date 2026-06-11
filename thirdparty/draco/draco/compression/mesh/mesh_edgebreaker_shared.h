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
#ifndef DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_SHARED_H_
#define DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_SHARED_H_

#include <stdint.h>

namespace draco {

// Shared declarations used by both edgebreaker encoder and decoder.

// A variable length encoding for storing all possible topology configurations
// during traversal of mesh's surface. The configurations are based on visited
// state of neighboring triangles around a currently processed face corner.
// Note that about half of the encountered configurations is expected to be of
// type TOPOLOGY_C. It's guaranteed that the encoding will use at most 2 bits
// per triangle for meshes with no holes and up to 6 bits per triangle for
// general meshes. In addition, the encoding will take up to 4 bits per triangle
// for each non-position attribute attached to the mesh.
//
//     *-------*          *-------*          *-------*
//    / \     / \        / \     / \        / \     / \
//   /   \   /   \      /   \   /   \      /   \   /   \
//  /     \ /     \    /     \ /     \    /     \ /     \
// *-------v-------*  *-------v-------*  *-------v-------*
//  \     /x\     /          /x\     /    \     /x\
//   \   /   \   /          /   \   /      \   /   \
//    \ /  C  \ /          /  L  \ /        \ /  R  \
//     *-------*          *-------*          *-------*
//
//     *       *
//    / \     / \
//   /   \   /   \
//  /     \ /     \
// *-------v-------*          v
//  \     /x\     /          /x\
//   \   /   \   /          /   \
//    \ /  S  \ /          /  E  \
//     *-------*          *-------*
//
enum EdgebreakerTopologyBitPattern {
  TOPOLOGY_C = 0x0,  // 0
  TOPOLOGY_S = 0x1,  // 1 0 0
  TOPOLOGY_L = 0x3,  // 1 1 0
  TOPOLOGY_R = 0x5,  // 1 0 1
  TOPOLOGY_E = 0x7,  // 1 1 1
  // A special symbol that's not actually encoded, but it can be used to mark
  // the initial face that triggers the mesh encoding of a single connected
  // component.
  TOPOLOGY_INIT_FACE,
  // A special value used to indicate an invalid symbol.
  TOPOLOGY_INVALID
};

enum EdgebreakerSymbol {
  EDGEBREAKER_SYMBOL_C = 0,
  EDGEBREAKER_SYMBOL_S,
  EDGEBREAKER_SYMBOL_L,
  EDGEBREAKER_SYMBOL_R,
  EDGEBREAKER_SYMBOL_E,
  EDGEBREAKER_SYMBOL_INVALID
};

// Bit-length of symbols in the EdgebreakerTopologyBitPattern stored as a
// lookup table for faster indexing.
constexpr int32_t edge_breaker_topology_bit_pattern_length[] = {1, 3, 0, 3,
                                                                0, 3, 0, 3};

// Zero-indexed symbol id for each of topology pattern.
constexpr EdgebreakerSymbol edge_breaker_topology_to_symbol_id[] = {
    EDGEBREAKER_SYMBOL_C,       EDGEBREAKER_SYMBOL_S,
    EDGEBREAKER_SYMBOL_INVALID, EDGEBREAKER_SYMBOL_L,
    EDGEBREAKER_SYMBOL_INVALID, EDGEBREAKER_SYMBOL_R,
    EDGEBREAKER_SYMBOL_INVALID, EDGEBREAKER_SYMBOL_E};

// Reverse mapping between symbol id and topology pattern symbol.
constexpr EdgebreakerTopologyBitPattern edge_breaker_symbol_to_topology_id[] = {
    TOPOLOGY_C, TOPOLOGY_S, TOPOLOGY_L, TOPOLOGY_R, TOPOLOGY_E};

// Types of edges used during mesh traversal relative to the tip vertex of a
// visited triangle.
enum EdgeFaceName : uint8_t { LEFT_FACE_EDGE = 0, RIGHT_FACE_EDGE = 1 };

// Struct used for storing data about a source face that connects to an
// already traversed face that was either the initial face or a face encoded
// with either topology S (split) symbol. Such connection can be only caused by
// topology changes on the traversed surface (if its genus != 0, i.e. when the
// surface has topological handles or holes).
// For each occurrence of such event we always encode the split symbol id,
// source symbol id and source edge id (left, or right). There will be always
// exactly two occurrences of this event for every topological handle on the
// traversed mesh and one occurrence for a hole.
struct TopologySplitEventData {
  uint32_t split_symbol_id;
  uint32_t source_symbol_id;
  // We need to use uint32_t instead of EdgeFaceName because the most recent
  // version of gcc does not allow that when optimizations are turned on.
  uint32_t source_edge : 1;
};

// Hole event is used to store info about the first symbol that reached a
// vertex of so far unvisited hole. This can happen only on either the initial
// face or during a regular traversal when TOPOLOGY_S is encountered.
struct HoleEventData {
  int32_t symbol_id;
  HoleEventData() : symbol_id(0) {}
  explicit HoleEventData(int32_t sym_id) : symbol_id(sym_id) {}
};

// List of supported modes for valence based edgebreaker coding.
enum EdgebreakerValenceCodingMode {
  EDGEBREAKER_VALENCE_MODE_2_7 = 0,  // Use contexts for valences in range 2-7.
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_SHARED_H_
