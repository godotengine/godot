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
#ifndef DRACO_ATTRIBUTES_GEOMETRY_INDICES_H_
#define DRACO_ATTRIBUTES_GEOMETRY_INDICES_H_

#include <inttypes.h>

#include <limits>

#include "draco/core/draco_index_type.h"

namespace draco {

// Index of an attribute value entry stored in a GeometryAttribute.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, AttributeValueIndex)
// Index of a point in a PointCloud.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, PointIndex)
// Vertex index in a Mesh or CornerTable.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, VertexIndex)
// Corner index that identifies a corner in a Mesh or CornerTable.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, CornerIndex)
// Face index for Mesh and CornerTable.
DEFINE_NEW_DRACO_INDEX_TYPE(uint32_t, FaceIndex)

// Constants denoting invalid indices.
static constexpr AttributeValueIndex kInvalidAttributeValueIndex(
    std::numeric_limits<uint32_t>::max());
static constexpr PointIndex kInvalidPointIndex(
    std::numeric_limits<uint32_t>::max());
static constexpr VertexIndex kInvalidVertexIndex(
    std::numeric_limits<uint32_t>::max());
static constexpr CornerIndex kInvalidCornerIndex(
    std::numeric_limits<uint32_t>::max());
static constexpr FaceIndex kInvalidFaceIndex(
    std::numeric_limits<uint32_t>::max());

// TODO(ostava): Add strongly typed indices for attribute id and unique
// attribute id.

}  // namespace draco

#endif  // DRACO_ATTRIBUTES_GEOMETRY_INDICES_H_
