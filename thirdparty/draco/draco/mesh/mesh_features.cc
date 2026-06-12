// Copyright 2022 The Draco Authors.
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
#include "draco/mesh/mesh_features.h"

#include <string>
#include <vector>

#ifdef DRACO_TRANSCODER_SUPPORTED

namespace draco {

MeshFeatures::MeshFeatures()
    : feature_count_(0),
      null_feature_id_(-1),
      attribute_index_(-1),
      property_table_index_(-1) {}

void MeshFeatures::Copy(const MeshFeatures &src) {
  label_ = src.label_;
  feature_count_ = src.feature_count_;
  null_feature_id_ = src.null_feature_id_;
  attribute_index_ = src.attribute_index_;
  texture_map_.Copy(src.texture_map_);
  texture_channels_ = src.texture_channels_;
  property_table_index_ = src.property_table_index_;
}

void MeshFeatures::SetLabel(const std::string &label) { label_ = label; }

const std::string &MeshFeatures::GetLabel() const { return label_; }

void MeshFeatures::SetFeatureCount(int feature_count) {
  feature_count_ = feature_count;
}

int MeshFeatures::GetFeatureCount() const { return feature_count_; }

void MeshFeatures::SetNullFeatureId(int null_feature_id) {
  null_feature_id_ = null_feature_id;
}

int MeshFeatures::GetNullFeatureId() const { return null_feature_id_; }

void MeshFeatures::SetAttributeIndex(int attribute_index) {
  attribute_index_ = attribute_index;
}

int MeshFeatures::GetAttributeIndex() const { return attribute_index_; }

void MeshFeatures::SetTextureMap(const TextureMap &texture_map) {
  texture_map_.Copy(texture_map);
}

void MeshFeatures::SetTextureMap(Texture *texture, int tex_coord_index) {
  texture_map_.SetProperties(TextureMap::GENERIC, tex_coord_index);
  texture_map_.SetTexture(texture);
}

const TextureMap &MeshFeatures::GetTextureMap() const { return texture_map_; }

TextureMap &MeshFeatures::GetTextureMap() { return texture_map_; }

void MeshFeatures::SetTextureChannels(
    const std::vector<int> &texture_channels) {
  texture_channels_ = texture_channels;
}

const std::vector<int> &MeshFeatures::GetTextureChannels() const {
  return texture_channels_;
}

std::vector<int> &MeshFeatures::GetTextureChannels() {
  return texture_channels_;
}

void MeshFeatures::SetPropertyTableIndex(int property_table_index) {
  property_table_index_ = property_table_index;
}

int MeshFeatures::GetPropertyTableIndex() const {
  return property_table_index_;
}

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
