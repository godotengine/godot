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
#include "draco/metadata/geometry_metadata.h"

#include <utility>

namespace draco {

AttributeMetadata::AttributeMetadata(const AttributeMetadata &metadata)
    : Metadata(metadata) {
  att_unique_id_ = metadata.att_unique_id_;
}

GeometryMetadata::GeometryMetadata(const GeometryMetadata &metadata)
    : Metadata(metadata) {
  for (size_t i = 0; i < metadata.att_metadatas_.size(); ++i) {
    att_metadatas_.push_back(std::unique_ptr<AttributeMetadata>(
        new AttributeMetadata(*metadata.att_metadatas_[i])));
  }
}

const AttributeMetadata *GeometryMetadata::GetAttributeMetadataByStringEntry(
    const std::string &entry_name, const std::string &entry_value) const {
  for (auto &&att_metadata : att_metadatas_) {
    std::string value;
    if (!att_metadata->GetEntryString(entry_name, &value)) {
      continue;
    }
    if (value == entry_value) {
      return att_metadata.get();
    }
  }
  // No attribute has the requested entry.
  return nullptr;
}

bool GeometryMetadata::AddAttributeMetadata(
    std::unique_ptr<AttributeMetadata> att_metadata) {
  if (!att_metadata) {
    return false;
  }
  att_metadatas_.push_back(std::move(att_metadata));
  return true;
}
}  // namespace draco
