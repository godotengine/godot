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
#ifndef DRACO_COMPRESSION_MESH_MESH_DECODER_H_
#define DRACO_COMPRESSION_MESH_MESH_DECODER_H_

#include "draco/compression/attributes/mesh_attribute_indices_encoding_data.h"
#include "draco/compression/point_cloud/point_cloud_decoder.h"
#include "draco/mesh/mesh.h"
#include "draco/mesh/mesh_attribute_corner_table.h"

namespace draco {

// Class that reconstructs a 3D mesh from input data that was encoded by
// MeshEncoder.
class MeshDecoder : public PointCloudDecoder {
 public:
  MeshDecoder();

  EncodedGeometryType GetGeometryType() const override {
    return TRIANGULAR_MESH;
  }

  // The main entry point for mesh decoding.
  Status Decode(const DecoderOptions &options, DecoderBuffer *in_buffer,
                Mesh *out_mesh);

  // Returns the base connectivity of the decoded mesh (or nullptr if it is not
  // initialized).
  virtual const CornerTable *GetCornerTable() const { return nullptr; }

  // Returns the attribute connectivity data or nullptr if it does not exist.
  virtual const MeshAttributeCornerTable *GetAttributeCornerTable(
      int /* att_id */) const {
    return nullptr;
  }

  // Returns the decoding data for a given attribute or nullptr when the data
  // does not exist.
  virtual const MeshAttributeIndicesEncodingData *GetAttributeEncodingData(
      int /* att_id */) const {
    return nullptr;
  }

  Mesh *mesh() const { return mesh_; }

 protected:
  bool DecodeGeometryData() override;
  virtual bool DecodeConnectivity() = 0;

 private:
  Mesh *mesh_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_MESH_MESH_DECODER_H_
