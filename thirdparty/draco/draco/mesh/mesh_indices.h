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
#ifdef DRACO_TRANSCODER_SUPPORTED
#ifndef DRACO_MESH_MESH_INDICES_H_
#define DRACO_MESH_MESH_INDICES_H_

#include <inttypes.h>

#include <limits>

#include "draco/core/draco_index_type.h"

namespace draco {

// Index of a mesh feature ID set.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, MeshFeaturesIndex)

// Constants denoting invalid indices.
static constexpr MeshFeaturesIndex kInvalidMeshFeaturesIndex(
    std::numeric_limits<uint32_t>::max());

}  // namespace draco

#endif  // DRACO_MESH_MESH_INDICES_H_
#endif  // DRACO_TRANSCODER_SUPPORTED
