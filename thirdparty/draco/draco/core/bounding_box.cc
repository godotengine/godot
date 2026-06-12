// Copyright 2018 The Draco Authors.
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

#include "draco/core/bounding_box.h"

namespace draco {

BoundingBox::BoundingBox()
    : BoundingBox(Vector3f(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()),
                  Vector3f(std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest())) {}

BoundingBox::BoundingBox(const Vector3f &min_point, const Vector3f &max_point)
    : min_point_(min_point), max_point_(max_point) {}

const bool BoundingBox::IsValid() const {
  return GetMinPoint()[0] != std::numeric_limits<float>::max() &&
         GetMinPoint()[1] != std::numeric_limits<float>::max() &&
         GetMinPoint()[2] != std::numeric_limits<float>::max() &&
         GetMaxPoint()[0] != std::numeric_limits<float>::lowest() &&
         GetMaxPoint()[1] != std::numeric_limits<float>::lowest() &&
         GetMaxPoint()[2] != std::numeric_limits<float>::lowest();
}

}  // namespace draco
