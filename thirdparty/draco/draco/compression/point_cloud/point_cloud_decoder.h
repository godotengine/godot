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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_DECODER_H_
#define DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_DECODER_H_

#include "draco/compression/attributes/attributes_decoder_interface.h"
#include "draco/compression/config/compression_shared.h"
#include "draco/compression/config/decoder_options.h"
#include "draco/core/status.h"
#include "draco/point_cloud/point_cloud.h"

namespace draco {

// Abstract base class for all point cloud and mesh decoders. It provides a
// basic functionality that is shared between different decoders.
class PointCloudDecoder {
 public:
  PointCloudDecoder();
  virtual ~PointCloudDecoder() = default;

  virtual EncodedGeometryType GetGeometryType() const { return POINT_CLOUD; }

  // Decodes a Draco header int other provided |out_header|.
  // Returns false on error.
  static Status DecodeHeader(DecoderBuffer *buffer, DracoHeader *out_header);

  // The main entry point for point cloud decoding.
  Status Decode(const DecoderOptions &options, DecoderBuffer *in_buffer,
                PointCloud *out_point_cloud);

  bool SetAttributesDecoder(
      int att_decoder_id, std::unique_ptr<AttributesDecoderInterface> decoder) {
    if (att_decoder_id < 0) {
      return false;
    }
    if (att_decoder_id >= static_cast<int>(attributes_decoders_.size())) {
      attributes_decoders_.resize(att_decoder_id + 1);
    }
    attributes_decoders_[att_decoder_id] = std::move(decoder);
    return true;
  }

  // Returns an attribute containing decoded data in their portable form that
  // is guaranteed to be the same for both encoder and decoder. I.e., it returns
  // an attribute before it was transformed back into its final form which may
  // be slightly different (non-portable) across platforms. For example, for
  // attributes encoded with quantization, this method returns an attribute
  // that contains the quantized values (before the dequantization step).
  const PointAttribute *GetPortableAttribute(int32_t point_attribute_id);

  uint16_t bitstream_version() const {
    return DRACO_BITSTREAM_VERSION(version_major_, version_minor_);
  }

  const AttributesDecoderInterface *attributes_decoder(int dec_id) {
    return attributes_decoders_[dec_id].get();
  }
  int32_t num_attributes_decoders() const {
    return static_cast<int32_t>(attributes_decoders_.size());
  }

  // Get a mutable pointer to the decoded point cloud. This is intended to be
  // used mostly by other decoder subsystems.
  PointCloud *point_cloud() { return point_cloud_; }
  const PointCloud *point_cloud() const { return point_cloud_; }

  DecoderBuffer *buffer() { return buffer_; }
  const DecoderOptions *options() const { return options_; }

 protected:
  // Can be implemented by derived classes to perform any custom initialization
  // of the decoder. Called in the Decode() method.
  virtual bool InitializeDecoder() { return true; }

  // Creates an attribute decoder.
  virtual bool CreateAttributesDecoder(int32_t att_decoder_id) = 0;
  virtual bool DecodeGeometryData() { return true; }
  virtual bool DecodePointAttributes();

  virtual bool DecodeAllAttributes();
  virtual bool OnAttributesDecoded() { return true; }

  Status DecodeMetadata();

 private:
  // Point cloud that is being filled in by the decoder.
  PointCloud *point_cloud_;

  std::vector<std::unique_ptr<AttributesDecoderInterface>> attributes_decoders_;

  // Map between attribute id and decoder id.
  std::vector<int32_t> attribute_to_decoder_map_;

  // Input buffer holding the encoded data.
  DecoderBuffer *buffer_;

  // Bit-stream version of the encoder that encoded the input data.
  uint8_t version_major_;
  uint8_t version_minor_;

  const DecoderOptions *options_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_DECODER_H_
