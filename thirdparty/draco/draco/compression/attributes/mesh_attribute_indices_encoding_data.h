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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_MESH_ATTRIBUTE_INDICES_ENCODING_DATA_H_
#define DRACO_COMPRESSION_ATTRIBUTES_MESH_ATTRIBUTE_INDICES_ENCODING_DATA_H_

#include <inttypes.h>

#include <vector>

#include "draco/attributes/geometry_indices.h"

namespace draco {

// Data used for encoding and decoding of mesh attributes.
struct MeshAttributeIndicesEncodingData {
  MeshAttributeIndicesEncodingData() : num_values(0) {}

  void Init(int num_vertices) {
    vertex_to_encoded_attribute_value_index_map.resize(num_vertices);

    // We expect to store one value for each vertex.
    encoded_attribute_value_index_to_corner_map.reserve(num_vertices);
  }

  // Array for storing the corner ids in the order their associated attribute
  // entries were encoded/decoded. For every encoded attribute value entry we
  // store exactly one corner. I.e., this is the mapping between an encoded
  // attribute entry ids and corner ids. This map is needed for example by
  // prediction schemes. Note that not all corners are included in this map,
  // e.g., if multiple corners share the same attribute value, only one of these
  // corners will be usually included.
  std::vector<CornerIndex> encoded_attribute_value_index_to_corner_map;

  // Map for storing encoding order of attribute entries for each vertex.
  // i.e. Mapping between vertices and their corresponding attribute entry ids
  // that are going to be used by the decoder.
  // -1 if an attribute entry hasn't been encoded/decoded yet.
  std::vector<int32_t> vertex_to_encoded_attribute_value_index_map;

  // Total number of encoded/decoded attribute entries.
  int num_values;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_MESH_ATTRIBUTE_INDICES_ENCODING_DATA_H_
