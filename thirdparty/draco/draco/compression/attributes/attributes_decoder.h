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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_H_

#include <vector>

#include "draco/compression/attributes/attributes_decoder_interface.h"
#include "draco/compression/point_cloud/point_cloud_decoder.h"
#include "draco/core/decoder_buffer.h"
#include "draco/draco_features.h"
#include "draco/point_cloud/point_cloud.h"

namespace draco {

// Base class for decoding one or more attributes that were encoded with a
// matching AttributesEncoder. It is a basic implementation of
// AttributesDecoderInterface that provides functionality that is shared between
// all AttributesDecoders.
class AttributesDecoder : public AttributesDecoderInterface {
 public:
  AttributesDecoder();
  virtual ~AttributesDecoder() = default;

  // Called after all attribute decoders are created. It can be used to perform
  // any custom initialization.
  bool Init(PointCloudDecoder *decoder, PointCloud *pc) override;

  // Decodes any attribute decoder specific data from the |in_buffer|.
  bool DecodeAttributesDecoderData(DecoderBuffer *in_buffer) override;

  int32_t GetAttributeId(int i) const override {
    return point_attribute_ids_[i];
  }
  int32_t GetNumAttributes() const override {
    return static_cast<int32_t>(point_attribute_ids_.size());
  }
  PointCloudDecoder *GetDecoder() const override {
    return point_cloud_decoder_;
  }

  // Decodes attribute data from the source buffer.
  bool DecodeAttributes(DecoderBuffer *in_buffer) override {
    if (!DecodePortableAttributes(in_buffer)) {
      return false;
    }
    if (!DecodeDataNeededByPortableTransforms(in_buffer)) {
      return false;
    }
    if (!TransformAttributesToOriginalFormat()) {
      return false;
    }
    return true;
  }

 protected:
  int32_t GetLocalIdForPointAttribute(int32_t point_attribute_id) const {
    const int id_map_size =
        static_cast<int>(point_attribute_to_local_id_map_.size());
    if (point_attribute_id >= id_map_size) {
      return -1;
    }
    return point_attribute_to_local_id_map_[point_attribute_id];
  }
  virtual bool DecodePortableAttributes(DecoderBuffer *in_buffer) = 0;
  virtual bool DecodeDataNeededByPortableTransforms(DecoderBuffer *in_buffer) {
    return true;
  }
  virtual bool TransformAttributesToOriginalFormat() { return true; }

 private:
  // List of attribute ids that need to be decoded with this decoder.
  std::vector<int32_t> point_attribute_ids_;

  // Map between point attribute id and the local id (i.e., the inverse of the
  // |point_attribute_ids_|.
  std::vector<int32_t> point_attribute_to_local_id_map_;

  PointCloudDecoder *point_cloud_decoder_;
  PointCloud *point_cloud_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_H_
