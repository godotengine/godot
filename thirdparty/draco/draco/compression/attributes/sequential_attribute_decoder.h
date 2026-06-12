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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_ATTRIBUTE_DECODER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_ATTRIBUTE_DECODER_H_

#include "draco/compression/attributes/prediction_schemes/prediction_scheme_interface.h"
#include "draco/compression/point_cloud/point_cloud_decoder.h"
#include "draco/draco_features.h"

namespace draco {

// A base class for decoding attribute values encoded by the
// SequentialAttributeEncoder.
class SequentialAttributeDecoder {
 public:
  SequentialAttributeDecoder();
  virtual ~SequentialAttributeDecoder() = default;

  virtual bool Init(PointCloudDecoder *decoder, int attribute_id);

  // Initialization for a specific attribute. This can be used mostly for
  // standalone decoding of an attribute without an PointCloudDecoder.
  virtual bool InitializeStandalone(PointAttribute *attribute);

  // Performs lossless decoding of the portable attribute data.
  virtual bool DecodePortableAttribute(const std::vector<PointIndex> &point_ids,
                                       DecoderBuffer *in_buffer);

  // Decodes any data needed to revert portable transform of the decoded
  // attribute.
  virtual bool DecodeDataNeededByPortableTransform(
      const std::vector<PointIndex> &point_ids, DecoderBuffer *in_buffer);

  // Reverts transformation performed by encoder in
  // SequentialAttributeEncoder::TransformAttributeToPortableFormat() method.
  virtual bool TransformAttributeToOriginalFormat(
      const std::vector<PointIndex> &point_ids);

  const PointAttribute *GetPortableAttribute();

  const PointAttribute *attribute() const { return attribute_; }
  PointAttribute *attribute() { return attribute_; }
  int attribute_id() const { return attribute_id_; }
  PointCloudDecoder *decoder() const { return decoder_; }

 protected:
  // Should be used to initialize newly created prediction scheme.
  // Returns false when the initialization failed (in which case the scheme
  // cannot be used).
  virtual bool InitPredictionScheme(PredictionSchemeInterface *ps);

  // The actual implementation of the attribute decoding. Should be overridden
  // for specialized decoders.
  virtual bool DecodeValues(const std::vector<PointIndex> &point_ids,
                            DecoderBuffer *in_buffer);

  void SetPortableAttribute(std::unique_ptr<PointAttribute> att) {
    portable_attribute_ = std::move(att);
  }

  PointAttribute *portable_attribute() { return portable_attribute_.get(); }

 private:
  PointCloudDecoder *decoder_;
  PointAttribute *attribute_;
  int attribute_id_;

  // Storage for decoded portable attribute (after lossless decoding).
  std::unique_ptr<PointAttribute> portable_attribute_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_SEQUENTIAL_ATTRIBUTE_DECODER_H_
