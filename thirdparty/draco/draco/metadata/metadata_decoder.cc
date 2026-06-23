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
#include "draco/metadata/metadata_decoder.h"

#include <string>

#include "draco/core/varint_decoding.h"

namespace draco {

MetadataDecoder::MetadataDecoder() : buffer_(nullptr) {}

bool MetadataDecoder::DecodeMetadata(DecoderBuffer *in_buffer,
                                     Metadata *metadata) {
  if (!metadata) {
    return false;
  }
  buffer_ = in_buffer;
  return DecodeMetadata(metadata);
}

bool MetadataDecoder::DecodeGeometryMetadata(DecoderBuffer *in_buffer,
                                             GeometryMetadata *metadata) {
  if (!metadata) {
    return false;
  }
  buffer_ = in_buffer;
  uint32_t num_att_metadata = 0;
  if (!DecodeVarint(&num_att_metadata, buffer_)) {
    return false;
  }
  // Decode attribute metadata.
  for (uint32_t i = 0; i < num_att_metadata; ++i) {
    uint32_t att_unique_id;
    if (!DecodeVarint(&att_unique_id, buffer_)) {
      return false;
    }
    std::unique_ptr<AttributeMetadata> att_metadata =
        std::unique_ptr<AttributeMetadata>(new AttributeMetadata());
    att_metadata->set_att_unique_id(att_unique_id);
    if (!DecodeMetadata(static_cast<Metadata *>(att_metadata.get()))) {
      return false;
    }
    metadata->AddAttributeMetadata(std::move(att_metadata));
  }
  return DecodeMetadata(static_cast<Metadata *>(metadata));
}

bool MetadataDecoder::DecodeMetadata(Metadata *metadata) {
  // Limit metadata nesting depth to avoid stack overflow in destructor.
  constexpr int kMaxSubmetadataLevel = 1000;

  struct MetadataTuple {
    Metadata *parent_metadata;
    Metadata *decoded_metadata;
    int level;
  };
  std::vector<MetadataTuple> metadata_stack;
  metadata_stack.push_back({nullptr, metadata, 0});
  while (!metadata_stack.empty()) {
    const MetadataTuple mp = metadata_stack.back();
    metadata_stack.pop_back();
    metadata = mp.decoded_metadata;

    if (mp.parent_metadata != nullptr) {
      if (mp.level > kMaxSubmetadataLevel) {
        return false;
      }
      std::string sub_metadata_name;
      if (!DecodeName(&sub_metadata_name)) {
        return false;
      }
      std::unique_ptr<Metadata> sub_metadata =
          std::unique_ptr<Metadata>(new Metadata());
      metadata = sub_metadata.get();
      if (!mp.parent_metadata->AddSubMetadata(sub_metadata_name,
                                              std::move(sub_metadata))) {
        return false;
      }
    }
    if (metadata == nullptr) {
      return false;
    }

    uint32_t num_entries = 0;
    if (!DecodeVarint(&num_entries, buffer_)) {
      return false;
    }
    for (uint32_t i = 0; i < num_entries; ++i) {
      if (!DecodeEntry(metadata)) {
        return false;
      }
    }
    uint32_t num_sub_metadata = 0;
    if (!DecodeVarint(&num_sub_metadata, buffer_)) {
      return false;
    }
    if (num_sub_metadata > buffer_->remaining_size()) {
      // The decoded number of metadata items is unreasonably high.
      return false;
    }
    for (uint32_t i = 0; i < num_sub_metadata; ++i) {
      metadata_stack.push_back(
          {metadata, nullptr, mp.parent_metadata ? mp.level + 1 : mp.level});
    }
  }
  return true;
}

bool MetadataDecoder::DecodeEntry(Metadata *metadata) {
  std::string entry_name;
  if (!DecodeName(&entry_name)) {
    return false;
  }
  uint32_t data_size = 0;
  if (!DecodeVarint(&data_size, buffer_)) {
    return false;
  }
  if (data_size == 0) {
    return false;
  }
  if (data_size > buffer_->remaining_size()) {
    return false;
  }
  std::vector<uint8_t> entry_value(data_size);
  if (!buffer_->Decode(&entry_value[0], data_size)) {
    return false;
  }
  metadata->AddEntryBinary(entry_name, entry_value);
  return true;
}

bool MetadataDecoder::DecodeName(std::string *name) {
  uint8_t name_len = 0;
  if (!buffer_->Decode(&name_len)) {
    return false;
  }
  name->resize(name_len);
  if (name_len == 0) {
    return true;
  }
  if (!buffer_->Decode(&name->at(0), name_len)) {
    return false;
  }
  return true;
}
}  // namespace draco
