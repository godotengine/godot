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
#ifndef DRACO_COMPRESSION_DECODE_H_
#define DRACO_COMPRESSION_DECODE_H_

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/config/decoder_options.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/status_or.h"
#include "draco/draco_features.h"
#include "draco/mesh/mesh.h"

namespace draco {

// Class responsible for decoding of meshes and point clouds that were
// compressed by a Draco encoder.
class Decoder {
 public:
  // Returns the geometry type encoded in the input |in_buffer|.
  // The return value is one of POINT_CLOUD, MESH or INVALID_GEOMETRY in case
  // the input data is invalid.
  // The decoded geometry type can be used to choose an appropriate decoding
  // function for a given geometry type (see below).
  static StatusOr<EncodedGeometryType> GetEncodedGeometryType(
      DecoderBuffer *in_buffer);

  // Decodes point cloud from the provided buffer. The buffer must be filled
  // with data that was encoded with either the EncodePointCloudToBuffer or
  // EncodeMeshToBuffer methods in encode.h. In case the input buffer contains
  // mesh, the returned instance can be down-casted to Mesh.
  StatusOr<std::unique_ptr<PointCloud>> DecodePointCloudFromBuffer(
      DecoderBuffer *in_buffer);

  // Decodes a triangular mesh from the provided buffer. The mesh must be filled
  // with data that was encoded using the EncodeMeshToBuffer method in encode.h.
  // The function will return nullptr in case the input is invalid or if it was
  // encoded with the EncodePointCloudToBuffer method.
  StatusOr<std::unique_ptr<Mesh>> DecodeMeshFromBuffer(
      DecoderBuffer *in_buffer);

  // Decodes the buffer into a provided geometry. If the geometry is
  // incompatible with the encoded data. For example, when |out_geometry| is
  // draco::Mesh while the data contains a point cloud, the function will return
  // an error status.
  Status DecodeBufferToGeometry(DecoderBuffer *in_buffer,
                                PointCloud *out_geometry);
  Status DecodeBufferToGeometry(DecoderBuffer *in_buffer, Mesh *out_geometry);

  // When set, the decoder is going to skip attribute transform for a given
  // attribute type. For example for quantized attributes, the decoder would
  // skip the dequantization step and the returned geometry would contain an
  // attribute with quantized values. The attribute would also contain an
  // instance of AttributeTransform class that is used to describe the skipped
  // transform, including all parameters that are needed to perform the
  // transform manually.
  void SetSkipAttributeTransform(GeometryAttribute::Type att_type);

  // Returns the options instance used by the decoder that can be used by users
  // to control the decoding process.
  DecoderOptions *options() { return &options_; }

 private:
  DecoderOptions options_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_DECODE_H_
