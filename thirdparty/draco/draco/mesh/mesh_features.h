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
#ifndef DRACO_MESH_MESH_FEATURES_H_
#define DRACO_MESH_MESH_FEATURES_H_

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED
#include <string>
#include <vector>

#include "draco/texture/texture_library.h"
#include "draco/texture/texture_map.h"

namespace draco {

// Describes a mesh feature ID set according to the EXT_mesh_features glTF
// extension. Feature IDs are either associated with geometry vertices or with
// texture pixels and stored in a geometry attribute or in texture channels,
// respectively. Optionally, the feature ID set may be associated with a
// property table defined in the EXT_structural_metadata glTF extension.
class MeshFeatures {
 public:
  // Creates an empty feature ID set that is associated neither with vertices,
  // nor with texture pixels, nor with property tables.
  MeshFeatures();

  // Copies all data from |src| mesh feature ID set.
  void Copy(const MeshFeatures &src);

  // Label assigned to this feature ID set.
  void SetLabel(const std::string &label);
  const std::string &GetLabel() const;

  // The number of unique features in this feature ID set.
  void SetFeatureCount(int feature_count);
  int GetFeatureCount() const;

  // Non-negative null feature ID value indicating the absence of an associated
  // feature. The value of -1 indicates that the null feature ID is not set.
  void SetNullFeatureId(int null_feature_id);
  int GetNullFeatureId() const;

  // Index of the feature ID vertex attribute in draco::Mesh or -1 if the
  // feature ID is not associated with vertices.
  void SetAttributeIndex(int attribute_index);
  int GetAttributeIndex() const;

  // Feature ID texture map and texture channels containing feature IDs
  // associated with texture pixels. Only used when |attribute_index_| is -1.
  // The RGBA channels are numbered from 0 to 3. See the glTF extension
  // documentation for reconstruction of feature ID from the channel values.
  void SetTextureMap(const TextureMap &texture_map);
  void SetTextureMap(Texture *texture, int tex_coord_index);
  const TextureMap &GetTextureMap() const;
  TextureMap &GetTextureMap();
  void SetTextureChannels(const std::vector<int> &texture_channels);
  const std::vector<int> &GetTextureChannels() const;
  std::vector<int> &GetTextureChannels();

  // Non-negative index of the property table this feature ID set is associated
  // with. Property tables are defined in the EXT_structural_metadata glTF
  // extension. The value of -1 indicates that this feature ID set is not
  // associated with any property tables.
  void SetPropertyTableIndex(int property_table_index);
  int GetPropertyTableIndex() const;

 private:
  std::string label_;
  int feature_count_;
  int null_feature_id_;
  int attribute_index_;
  TextureMap texture_map_;
  std::vector<int> texture_channels_;
  int property_table_index_;
};

}  // namespace draco

#endif  // DRACO_TRANSCODER_SUPPORTED
#endif  // DRACO_MESH_MESH_FEATURES_H_
