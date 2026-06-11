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
#ifndef DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_H_
#define DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_H_

#include "draco/compression/mesh/mesh_decoder.h"
#include "draco/compression/mesh/mesh_edgebreaker_decoder_impl_interface.h"
#include "draco/draco_features.h"

namespace draco {

// Class for decoding data encoded by MeshEdgebreakerEncoder.
class MeshEdgebreakerDecoder : public MeshDecoder {
 public:
  MeshEdgebreakerDecoder();

  const CornerTable *GetCornerTable() const override {
    return impl_->GetCornerTable();
  }

  const MeshAttributeCornerTable *GetAttributeCornerTable(
      int att_id) const override {
    return impl_->GetAttributeCornerTable(att_id);
  }

  const MeshAttributeIndicesEncodingData *GetAttributeEncodingData(
      int att_id) const override {
    return impl_->GetAttributeEncodingData(att_id);
  }

 protected:
  bool InitializeDecoder() override;
  bool CreateAttributesDecoder(int32_t att_decoder_id) override;
  bool DecodeConnectivity() override;
  bool OnAttributesDecoded() override;

  std::unique_ptr<MeshEdgebreakerDecoderImplInterface> impl_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_EDGEBREAKER_DECODER_H_
