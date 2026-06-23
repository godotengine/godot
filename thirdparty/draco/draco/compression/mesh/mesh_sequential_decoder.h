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
#ifndef DRACO_COMPRESSION_MESH_MESH_SEQUENTIAL_DECODER_H_
#define DRACO_COMPRESSION_MESH_MESH_SEQUENTIAL_DECODER_H_

#include "draco/compression/mesh/mesh_decoder.h"

namespace draco {

// Class for decoding data encoded by MeshSequentialEncoder.
class MeshSequentialDecoder : public MeshDecoder {
 public:
  MeshSequentialDecoder();

 protected:
  bool DecodeConnectivity() override;
  bool CreateAttributesDecoder(int32_t att_decoder_id) override;

 private:
  // Decodes face indices that were compressed with an entropy code.
  // Returns false on error.
  bool DecodeAndDecompressIndices(uint32_t num_faces);
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_SEQUENTIAL_DECODER_H_
