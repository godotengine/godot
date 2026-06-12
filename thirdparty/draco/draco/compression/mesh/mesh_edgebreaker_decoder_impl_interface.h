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
#ifndef DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_INTERFACE_H_
#define DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_INTERFACE_H_

#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/mesh/mesh_attribute_corner_table.h"

namespace draco {

// Forward declaration is necessary here to avoid circular dependencies.
class MeshEdgebreakerDecoder;

// Abstract interface used by MeshEdgebreakerDecoder to interact with the actual
// implementation of the edgebreaker decoding method.
class MeshEdgebreakerDecoderImplInterface {
 public:
  virtual ~MeshEdgebreakerDecoderImplInterface() = default;
  virtual bool Init(MeshEdgebreakerDecoder *decoder) = 0;

  virtual const MeshAttributeCornerTable *GetAttributeCornerTable(
      int att_id) const = 0;
  virtual const MeshAttributeIndicesEncodingData *GetAttributeEncodingData(
      int att_id) const = 0;
  virtual bool CreateAttributesDecoder(int32_t att_decoder_id) = 0;
  virtual bool DecodeConnectivity() = 0;
  virtual bool OnAttributesDecoded() = 0;

  virtual MeshEdgebreakerDecoder *GetDecoder() const = 0;
  virtual const CornerTable *GetCornerTable() const = 0;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_IMPL_INTERFACE_H_
