// Copyright 2017 The Draco Authors.
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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_INTERFACE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_INTERFACE_H_

#include <vector>

#include "draco/core/decoder_buffer.h"
#include "draco/point_cloud/point_cloud.h"

namespace draco {

class PointCloudDecoder;

// Interface class for decoding one or more attributes that were encoded with a
// matching AttributesEncoder. It provides only the basic interface
// that is used by the PointCloudDecoder. The actual decoding must be
// implemented in derived classes using the DecodeAttributes() method.
class AttributesDecoderInterface {
 public:
  AttributesDecoderInterface() = default;
  virtual ~AttributesDecoderInterface() = default;

  // Called after all attribute decoders are created. It can be used to perform
  // any custom initialization.
  virtual bool Init(PointCloudDecoder *decoder, PointCloud *pc) = 0;

  // Decodes any attribute decoder specific data from the |in_buffer|.
  virtual bool DecodeAttributesDecoderData(DecoderBuffer *in_buffer) = 0;

  // Decode attribute data from the source buffer. Needs to be implemented by
  // the derived classes.
  virtual bool DecodeAttributes(DecoderBuffer *in_buffer) = 0;

  virtual int32_t GetAttributeId(int i) const = 0;
  virtual int32_t GetNumAttributes() const = 0;
  virtual PointCloudDecoder *GetDecoder() const = 0;

  // Returns an attribute containing data processed by the attribute transform.
  // (see TransformToPortableFormat() method). This data is guaranteed to be
  // same for encoder and decoder and it can be used by predictors.
  virtual const PointAttribute *GetPortableAttribute(
      int32_t /* point_attribute_id */) {
    return nullptr;
  }
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_ATTRIBUTES_DECODER_INTERFACE_H_
