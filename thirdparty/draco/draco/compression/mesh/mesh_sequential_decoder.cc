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
#include "draco/compression/mesh/mesh_sequential_decoder.h"

#include <cstdint>
#include <limits>

#include "draco/compression/attributes/linear_sequencer.h"
#include "draco/compression/attributes/sequential_attribute_decoders_controller.h"
#include "draco/compression/entropy/symbol_decoding.h"
#include "draco/core/varint_decoding.h"

namespace draco {

MeshSequentialDecoder::MeshSequentialDecoder() {}

bool MeshSequentialDecoder::DecodeConnectivity() {
  uint32_t num_faces;
  uint32_t num_points;
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
  if (bitstream_version() < DRACO_BITSTREAM_VERSION(2, 2)) {
    if (!buffer()->Decode(&num_faces)) {
      return false;
    }
    if (!buffer()->Decode(&num_points)) {
      return false;
    }

  } else
#endif
  {
    if (!DecodeVarint(&num_faces, buffer())) {
      return false;
    }
    if (!DecodeVarint(&num_points, buffer())) {
      return false;
    }
  }

  // Check that num_faces and num_points are valid values.
  const uint64_t faces_64 = static_cast<uint64_t>(num_faces);
  // Compressed sequential encoding can only handle (2^32 - 1) / 3 indices.
  if (faces_64 > 0xffffffff / 3) {
    return false;
  }
  if (faces_64 > buffer()->remaining_size() / 3) {
    // The number of faces is unreasonably high, because face indices do not
    // fit in the remaining size of the buffer.
    return false;
  }
  uint8_t connectivity_method;
  if (!buffer()->Decode(&connectivity_method)) {
    return false;
  }
  if (connectivity_method == 0) {
    if (!DecodeAndDecompressIndices(num_faces)) {
      return false;
    }
  } else {
    if (num_points < 256) {
      // Decode indices as uint8_t.
      for (uint32_t i = 0; i < num_faces; ++i) {
        Mesh::Face face;
        for (int j = 0; j < 3; ++j) {
          uint8_t val;
          if (!buffer()->Decode(&val)) {
            return false;
          }
          face[j] = val;
        }
        mesh()->AddFace(face);
      }
    } else if (num_points < (1 << 16)) {
      // Decode indices as uint16_t.
      for (uint32_t i = 0; i < num_faces; ++i) {
        Mesh::Face face;
        for (int j = 0; j < 3; ++j) {
          uint16_t val;
          if (!buffer()->Decode(&val)) {
            return false;
          }
          face[j] = val;
        }
        mesh()->AddFace(face);
      }
    } else if (num_points < (1 << 21) &&
               bitstream_version() >= DRACO_BITSTREAM_VERSION(2, 2)) {
      // Decode indices as uint32_t.
      for (uint32_t i = 0; i < num_faces; ++i) {
        Mesh::Face face;
        for (int j = 0; j < 3; ++j) {
          uint32_t val;
          if (!DecodeVarint(&val, buffer())) {
            return false;
          }
          face[j] = val;
        }
        mesh()->AddFace(face);
      }
    } else {
      // Decode faces as uint32_t (default).
      for (uint32_t i = 0; i < num_faces; ++i) {
        Mesh::Face face;
        for (int j = 0; j < 3; ++j) {
          uint32_t val;
          if (!buffer()->Decode(&val)) {
            return false;
          }
          face[j] = val;
        }
        mesh()->AddFace(face);
      }
    }
  }
  point_cloud()->set_num_points(num_points);
  return true;
}

bool MeshSequentialDecoder::CreateAttributesDecoder(int32_t att_decoder_id) {
  // Always create the basic attribute decoder.
  return SetAttributesDecoder(
      att_decoder_id,
      std::unique_ptr<AttributesDecoder>(
          new SequentialAttributeDecodersController(
              std::unique_ptr<PointsSequencer>(
                  new LinearSequencer(point_cloud()->num_points())))));
}

bool MeshSequentialDecoder::DecodeAndDecompressIndices(uint32_t num_faces) {
  // Get decoded indices differences that were encoded with an entropy code.
  std::vector<uint32_t> indices_buffer(num_faces * 3);
  if (!DecodeSymbols(num_faces * 3, 1, buffer(), indices_buffer.data())) {
    return false;
  }
  // Reconstruct the indices from the differences.
  // See MeshSequentialEncoder::CompressAndEncodeIndices() for more details.
  int32_t last_index_value = 0;  // This will always be >= 0.
  int vertex_index = 0;
  for (uint32_t i = 0; i < num_faces; ++i) {
    Mesh::Face face;
    for (int j = 0; j < 3; ++j) {
      const uint32_t encoded_val = indices_buffer[vertex_index++];
      int32_t index_diff = (encoded_val >> 1);
      if (encoded_val & 1) {
        if (index_diff > last_index_value) {
          // Subtracting index_diff would result in a negative index.
          return false;
        }
        index_diff = -index_diff;
      } else {
        if (index_diff >
            (std::numeric_limits<int32_t>::max() - last_index_value)) {
          // Adding index_diff to last_index_value would overflow.
          return false;
        }
      }
      const int32_t index_value = index_diff + last_index_value;
      face[j] = index_value;
      last_index_value = index_value;
    }
    mesh()->AddFace(face);
  }
  return true;
}

}  // namespace draco
