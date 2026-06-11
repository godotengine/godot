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
#ifndef DRACO_METADATA_METADATA_DECODER_H_
#define DRACO_METADATA_METADATA_DECODER_H_

#include "draco/core/decoder_buffer.h"
#include "draco/metadata/geometry_metadata.h"
#include "draco/metadata/metadata.h"

namespace draco {

// Class for decoding the metadata.
class MetadataDecoder {
 public:
  MetadataDecoder();
  bool DecodeMetadata(DecoderBuffer *in_buffer, Metadata *metadata);
  bool DecodeGeometryMetadata(DecoderBuffer *in_buffer,
                              GeometryMetadata *metadata);

 private:
  bool DecodeMetadata(Metadata *metadata);
  bool DecodeEntries(Metadata *metadata);
  bool DecodeEntry(Metadata *metadata);
  bool DecodeName(std::string *name);

  DecoderBuffer *buffer_;
};
}  // namespace draco

#endif  // DRACO_METADATA_METADATA_DECODER_H_
