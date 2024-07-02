// Copyright 2021 The Manifold Authors.
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

#pragma once
#include "public.h"
#include "sparse.h"
#include "vec.h"

namespace manifold {

/** @ingroup Private */
class Collider {
 public:
  Collider() {}
  Collider(const VecView<const Box>& leafBB,
           const VecView<const uint32_t>& leafMorton);
  bool Transform(glm::mat4x3);
  void UpdateBoxes(const VecView<const Box>& leafBB);
  template <const bool selfCollision = false, const bool inverted = false,
            typename T>
  SparseIndices Collisions(const VecView<const T>& queriesIn) const;

 private:
  Vec<Box> nodeBBox_;
  Vec<int> nodeParent_;
  // even nodes are leaves, odd nodes are internal, root is 1
  Vec<thrust::pair<int, int>> internalChildren_;

  int NumInternal() const { return internalChildren_.size(); };
  int NumLeaves() const {
    return internalChildren_.empty() ? 0 : (NumInternal() + 1);
  };
};

}  // namespace manifold
