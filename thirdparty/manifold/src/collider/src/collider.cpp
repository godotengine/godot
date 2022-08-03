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

#include "collider.h"

#include "par.h"
#include "utils.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Adjustable parameters
constexpr int kInitialLength = 128;
constexpr int kLengthMultiple = 4;
// Fundamental constants
constexpr int kRoot = 1;

#ifdef _MSC_VER

#ifndef _WINDEF_
typedef unsigned long DWORD;
#endif

uint32_t __inline ctz(uint32_t value) {
  DWORD trailing_zero = 0;

  if (_BitScanForward(&trailing_zero, value)) {
    return trailing_zero;
  } else {
    // This is undefined, I better choose 32 than 0
    return 32;
  }
}

uint32_t __inline clz(uint32_t value) {
  DWORD leading_zero = 0;

  if (_BitScanReverse(&leading_zero, value)) {
    return 31 - leading_zero;
  } else {
    // Same remarks as above
    return 32;
  }
}
#endif

namespace {
using namespace manifold;

__host__ __device__ bool IsLeaf(int node) { return node % 2 == 0; }
__host__ __device__ bool IsInternal(int node) { return node % 2 == 1; }
__host__ __device__ int Node2Internal(int node) { return (node - 1) / 2; }
__host__ __device__ int Internal2Node(int internal) { return internal * 2 + 1; }
__host__ __device__ int Node2Leaf(int node) { return node / 2; }
__host__ __device__ int Leaf2Node(int leaf) { return leaf * 2; }

struct CreateRadixTree {
  int* nodeParent_;
  thrust::pair<int, int>* internalChildren_;
  const VecD<uint32_t> leafMorton_;

  __host__ __device__ int PrefixLength(uint32_t a, uint32_t b) const {
// count-leading-zeros is used to find the number of identical highest-order
// bits
#ifdef __CUDA_ARCH__
    return __clz(a ^ b);
#else

#ifdef _MSC_VER
    // return __lzcnt(a ^ b);
    return clz(a ^ b);
#else
    return __builtin_clz(a ^ b);
#endif

#endif
  }

  __host__ __device__ int PrefixLength(int i, int j) const {
    if (j < 0 || j >= leafMorton_.size()) {
      return -1;
    } else {
      int out;
      if (leafMorton_[i] == leafMorton_[j])
        // use index to disambiguate
        out = 32 +
              PrefixLength(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
      else
        out = PrefixLength(leafMorton_[i], leafMorton_[j]);
      return out;
    }
  }

  __host__ __device__ int RangeEnd(int i) const {
    // Determine direction of range (+1 or -1)
    int dir = PrefixLength(i, i + 1) - PrefixLength(i, i - 1);
    dir = (dir > 0) - (dir < 0);
    // Compute conservative range length with exponential increase
    int commonPrefix = PrefixLength(i, i - dir);
    int max_length = kInitialLength;
    while (PrefixLength(i, i + dir * max_length) > commonPrefix)
      max_length *= kLengthMultiple;
    // Compute precise range length with binary search
    int length = 0;
    for (int step = max_length / 2; step > 0; step /= 2) {
      if (PrefixLength(i, i + dir * (length + step)) > commonPrefix)
        length += step;
    }
    return i + dir * length;
  }

  __host__ __device__ int FindSplit(int first, int last) const {
    int commonPrefix = PrefixLength(first, last);
    // Find the furthest object that shares more than commonPrefix bits with the
    // first one, using binary search.
    int split = first;
    int step = last - first;
    do {
      step = (step + 1) >> 1;  // divide by 2, rounding up
      int newSplit = split + step;
      if (newSplit < last) {
        int splitPrefix = PrefixLength(first, newSplit);
        if (splitPrefix > commonPrefix) split = newSplit;
      }
    } while (step > 1);
    return split;
  }

  __host__ __device__ void operator()(int internal) {
    int first = internal;
    // Find the range of objects with a common prefix
    int last = RangeEnd(first);
    if (first > last) thrust::swap(first, last);
    // Determine where the next-highest difference occurs
    int split = FindSplit(first, last);
    int child1 = split == first ? Leaf2Node(split) : Internal2Node(split);
    ++split;
    int child2 = split == last ? Leaf2Node(split) : Internal2Node(split);
    // Record parent_child relationships.
    internalChildren_[internal].first = child1;
    internalChildren_[internal].second = child2;
    int node = Internal2Node(internal);
    nodeParent_[child1] = node;
    nodeParent_[child2] = node;
  }
};

template <typename T>
struct FindCollisions {
  thrust::pair<int*, int*> querryTri_;
  int* numOverlaps_;
  const int maxOverlaps_;
  const Box* nodeBBox_;
  const thrust::pair<int, int>* internalChildren_;

  __host__ __device__ int RecordCollision(int node,
                                          const thrust::tuple<T, int>& query) {
    const T& queryObj = thrust::get<0>(query);
    const int queryIdx = thrust::get<1>(query);

    bool overlaps = nodeBBox_[node].DoesOverlap(queryObj);
    if (overlaps && IsLeaf(node)) {
      int pos = AtomicAdd(*numOverlaps_, 1);
      if (pos >= maxOverlaps_)
        return -1;  // Didn't allocate enough memory; bail out
      querryTri_.first[pos] = queryIdx;
      querryTri_.second[pos] = Node2Leaf(node);
    }
    return overlaps && IsInternal(node);  // Should traverse into node
  }

  __host__ __device__ void operator()(thrust::tuple<T, int> query) {
    // stack cannot overflow because radix tree has max depth 30 (Morton code) +
    // 32 (index).
    int stack[64];
    int top = -1;
    // Depth-first search
    int node = kRoot;
    while (1) {
      int internal = Node2Internal(node);
      int child1 = internalChildren_[internal].first;
      int child2 = internalChildren_[internal].second;

      int traverse1 = RecordCollision(child1, query);
      if (traverse1 < 0) return;
      int traverse2 = RecordCollision(child2, query);
      if (traverse2 < 0) return;

      if (!traverse1 && !traverse2) {
        if (top < 0) break;   // done
        node = stack[top--];  // get a saved node
      } else {
        node = traverse1 ? child1 : child2;  // go here next
        if (traverse1 && traverse2) {
          stack[++top] = child2;  // save the other for later
        }
      }
    }
  }
};

struct BuildInternalBoxes {
  Box* nodeBBox_;
  int* counter_;
  const int* nodeParent_;
  const thrust::pair<int, int>* internalChildren_;

  __host__ __device__ void operator()(int leaf) {
    int node = Leaf2Node(leaf);
    do {
      node = nodeParent_[node];
      int internal = Node2Internal(node);
      if (AtomicAdd(counter_[internal], 1) == 0) return;
      nodeBBox_[node] = nodeBBox_[internalChildren_[internal].first].Union(
          nodeBBox_[internalChildren_[internal].second]);
    } while (node != kRoot);
  }
};

struct TransformBox {
  const glm::mat4x3 transform;
  __host__ __device__ void operator()(Box& box) {
    box = box.Transform(transform);
  }
};
}  // namespace

namespace manifold {

/**
 * Creates a Bounding Volume Hierarchy (BVH) from an input set of axis-aligned
 * bounding boxes and corresponding Morton codes. It is assumed these vectors
 * are already sorted by increasing Morton code.
 */
Collider::Collider(const VecDH<Box>& leafBB,
                   const VecDH<uint32_t>& leafMorton) {
  ASSERT(leafBB.size() == leafMorton.size(), userErr,
         "vectors must be the same length");
  int num_nodes = 2 * leafBB.size() - 1;
  // assign and allocate members
  nodeBBox_.resize(num_nodes);
  nodeParent_.resize(num_nodes, -1);
  internalChildren_.resize(leafBB.size() - 1, thrust::make_pair(-1, -1));
  // organize tree
  for_each_n(autoPolicy(NumInternal()), countAt(0), NumInternal(),
             CreateRadixTree(
                 {nodeParent_.ptrD(), internalChildren_.ptrD(), leafMorton}));
  UpdateBoxes(leafBB);
}

/**
 * For a vector of querry objects, this returns a sparse array of overlaps
 * between the querries and the bounding boxes of the collider. Querries are
 * normally axis-aligned bounding boxes. Points can also be used, and this case
 * overlaps are defined as lying in the XY projection of the bounding box.
 */
template <typename T>
SparseIndices Collider::Collisions(const VecDH<T>& querriesIn) const {
  int maxOverlaps = querriesIn.size() * 4;
  SparseIndices querryTri(maxOverlaps);
  int nOverlaps = 0;
  while (1) {
    // scalar number of overlaps found
    VecDH<int> nOverlapsD(1, 0);
    // calculate Bounding Box overlaps
    for_each_n(
        autoPolicy(querriesIn.size()), zip(querriesIn.cbegin(), countAt(0)),
        querriesIn.size(),
        FindCollisions<T>({querryTri.ptrDpq(), nOverlapsD.ptrD(), maxOverlaps,
                           nodeBBox_.ptrD(), internalChildren_.ptrD()}));
    nOverlaps = nOverlapsD[0];
    if (nOverlaps <= maxOverlaps)
      break;
    else {  // if not enough memory was allocated, guess how much will be needed
      int lastQuery = querryTri.Get(0).back();
      float ratio = static_cast<float>(querriesIn.size()) / lastQuery;
      if (ratio > 1000)  // do not trust the ratio if it is too large
        maxOverlaps *= 2;
      else
        maxOverlaps *= 2 * ratio;
      querryTri.Resize(maxOverlaps);
    }
  }
  // remove unused part of array
  querryTri.Resize(nOverlaps);
  return querryTri;
}

/**
 * Recalculate the collider's internal bounding boxes without changing the
 * hierarchy.
 */
void Collider::UpdateBoxes(const VecDH<Box>& leafBB) {
  ASSERT(leafBB.size() == NumLeaves(), userErr,
         "must have the same number of updated boxes as original");
  // copy in leaf node Boxs
  strided_range<VecDH<Box>::Iter> leaves(nodeBBox_.begin(), nodeBBox_.end(), 2);
  auto policy = autoPolicy(NumInternal());
  copy(policy, leafBB.cbegin(), leafBB.cend(), leaves.begin());
  // create global counters
  VecDH<int> counter(NumInternal(), 0);
  // kernel over leaves to save internal Boxs
  for_each_n(
      policy, countAt(0), NumLeaves(),
      BuildInternalBoxes({nodeBBox_.ptrD(), counter.ptrD(), nodeParent_.ptrD(),
                          internalChildren_.ptrD()}));
}

/**
 * Apply axis-aligned transform to all bounding boxes. If transform is not
 * axis-aligned, abort and return false to indicate recalculation is necessary.
 */
bool Collider::Transform(glm::mat4x3 transform) {
  bool axisAligned = true;
  for (int row : {0, 1, 2}) {
    int count = 0;
    for (int col : {0, 1, 2}) {
      if (transform[col][row] == 0.0f) ++count;
    }
    if (count != 2) axisAligned = false;
  }
  if (axisAligned) {
    for_each(autoPolicy(nodeBBox_.size()), nodeBBox_.begin(), nodeBBox_.end(),
             TransformBox({transform}));
  }
  return axisAligned;
}

template SparseIndices Collider::Collisions<Box>(const VecDH<Box>&) const;

template SparseIndices Collider::Collisions<glm::vec3>(
    const VecDH<glm::vec3>&) const;

}  // namespace manifold
