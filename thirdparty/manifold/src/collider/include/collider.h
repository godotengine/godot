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
#include "sparse.h"
#include "structs.h"
#include "vec_dh.h"

namespace manifold {

/** @ingroup Private */
class Collider {
 public:
  Collider() {}
  Collider(const VecDH<Box>& leafBB, const VecDH<uint32_t>& leafMorton);
  // Aborts and returns false if transform is not axis aligned.
  bool Transform(glm::mat4x3);
  void UpdateBoxes(const VecDH<Box>& leafBB);
  // Collisions returns a sparse result, where i is the querry index and j is
  // the leaf index where their bounding boxes overlap.
  template <typename T>
  SparseIndices Collisions(const VecDH<T>& querriesIn) const;

 private:
  VecDH<Box> nodeBBox_;
  VecDH<int> nodeParent_;
  // even nodes are leaves, odd nodes are internal, root is 1
  VecDH<thrust::pair<int, int>> internalChildren_;

  int NumInternal() const { return internalChildren_.size(); };
  int NumLeaves() const { return NumInternal() + 1; };
};

}  // namespace manifold
