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
constexpr int kSequentialThreshold = 512;
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

bool IsLeaf(int node) { return node % 2 == 0; }
bool IsInternal(int node) { return node % 2 == 1; }
int Node2Internal(int node) { return (node - 1) / 2; }
int Internal2Node(int internal) { return internal * 2 + 1; }
int Node2Leaf(int node) { return node / 2; }
int Leaf2Node(int leaf) { return leaf * 2; }

struct CreateRadixTree {
  VecView<int> nodeParent_;
  VecView<thrust::pair<int, int>> internalChildren_;
  const VecView<const uint32_t> leafMorton_;

  int PrefixLength(uint32_t a, uint32_t b) const {
// count-leading-zeros is used to find the number of identical highest-order
// bits
#ifdef _MSC_VER
    // return __lzcnt(a ^ b);
    return clz(a ^ b);
#else
    return __builtin_clz(a ^ b);
#endif
  }

  int PrefixLength(int i, int j) const {
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

  int RangeEnd(int i) const {
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

  int FindSplit(int first, int last) const {
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

  void operator()(int internal) {
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

template <typename T, const bool selfCollision, typename Recorder>
struct FindCollisions {
  VecView<const Box> nodeBBox_;
  VecView<const thrust::pair<int, int>> internalChildren_;
  Recorder recorder;

  int RecordCollision(int node, thrust::tuple<T, int>& query) {
    const T& queryObj = thrust::get<0>(query);
    const int queryIdx = thrust::get<1>(query);

    bool overlaps = nodeBBox_[node].DoesOverlap(queryObj);
    if (overlaps && IsLeaf(node)) {
      int leafIdx = Node2Leaf(node);
      if (!selfCollision || leafIdx != queryIdx) {
        recorder.record(queryIdx, leafIdx);
      }
    }
    return overlaps && IsInternal(node);  // Should traverse into node
  }

  void operator()(thrust::tuple<T, int> query) {
    // stack cannot overflow because radix tree has max depth 30 (Morton code) +
    // 32 (index).
    int stack[64];
    int top = -1;
    // Depth-first search
    int node = kRoot;
    const int queryIdx = thrust::get<1>(query);
    // same implies that this query do not have any collision
    if (recorder.earlyexit(queryIdx)) return;
    while (1) {
      int internal = Node2Internal(node);
      int child1 = internalChildren_[internal].first;
      int child2 = internalChildren_[internal].second;

      int traverse1 = RecordCollision(child1, query);
      int traverse2 = RecordCollision(child2, query);

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
    recorder.end(queryIdx);
  }
};

struct CountCollisions {
  VecView<int> counts;
  VecView<char> empty;
  void record(int queryIdx, int _leafIdx) { counts[queryIdx]++; }
  bool earlyexit(int _queryIdx) { return false; }
  void end(int queryIdx) {
    if (counts[queryIdx] == 0) empty[queryIdx] = 1;
  }
};

template <const bool inverted>
struct SeqCollisionRecorder {
  SparseIndices& queryTri_;
  void record(int queryIdx, int leafIdx) const {
    if (inverted)
      queryTri_.Add(leafIdx, queryIdx);
    else
      queryTri_.Add(queryIdx, leafIdx);
  }
  bool earlyexit(int queryIdx) const { return false; }
  void end(int queryIdx) const {}
};

template <const bool inverted>
struct ParCollisionRecorder {
  SparseIndices& queryTri;
  VecView<int> counts;
  VecView<char> empty;
  void record(int queryIdx, int leafIdx) {
    int pos = counts[queryIdx]++;
    if (inverted)
      queryTri.Set(pos, leafIdx, queryIdx);
    else
      queryTri.Set(pos, queryIdx, leafIdx);
  }
  bool earlyexit(int queryIdx) const { return empty[queryIdx] == 1; }
  void end(int queryIdx) const {}
};

struct BuildInternalBoxes {
  VecView<Box> nodeBBox_;
  VecView<int> counter_;
  const VecView<int> nodeParent_;
  const VecView<thrust::pair<int, int>> internalChildren_;

  void operator()(int leaf) {
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
  void operator()(Box& box) { box = box.Transform(transform); }
};
}  // namespace

namespace manifold {

/**
 * Creates a Bounding Volume Hierarchy (BVH) from an input set of axis-aligned
 * bounding boxes and corresponding Morton codes. It is assumed these vectors
 * are already sorted by increasing Morton code.
 */
Collider::Collider(const VecView<const Box>& leafBB,
                   const VecView<const uint32_t>& leafMorton) {
  ZoneScoped;
  ASSERT(leafBB.size() == leafMorton.size(), userErr,
         "vectors must be the same length");
  int num_nodes = 2 * leafBB.size() - 1;
  // assign and allocate members
  nodeBBox_.resize(num_nodes);
  nodeParent_.resize(num_nodes, -1);
  internalChildren_.resize(leafBB.size() - 1, thrust::make_pair(-1, -1));
  // organize tree
  for_each_n(autoPolicy(NumInternal()), countAt(0), NumInternal(),
             CreateRadixTree({nodeParent_, internalChildren_, leafMorton}));
  UpdateBoxes(leafBB);
}

/**
 * For a vector of query objects, this returns a sparse array of overlaps
 * between the queries and the bounding boxes of the collider. Queries are
 * normally axis-aligned bounding boxes. Points can also be used, and this case
 * overlaps are defined as lying in the XY projection of the bounding box. If
 * the query vector is the leaf vector, set selfCollision to true, which will
 * then not report any collisions between an index and itself.
 */
template <const bool selfCollision, const bool inverted, typename T>
SparseIndices Collider::Collisions(const VecView<const T>& queriesIn) const {
  ZoneScoped;
  // note that the length is 1 larger than the number of queries so the last
  // element can store the sum when using exclusive scan
  if (queriesIn.size() < kSequentialThreshold) {
    SparseIndices queryTri;
    for_each_n(ExecutionPolicy::Seq, zip(queriesIn.cbegin(), countAt(0)),
               queriesIn.size(),
               FindCollisions<T, selfCollision, SeqCollisionRecorder<inverted>>{
                   nodeBBox_, internalChildren_, {queryTri}});
    return queryTri;
  } else {
    // compute the number of collisions to determine the size for allocation and
    // offset, this avoids the need for atomic
    Vec<int> counts(queriesIn.size() + 1, 0);
    Vec<char> empty(queriesIn.size(), 0);
    for_each_n(ExecutionPolicy::Par, zip(queriesIn.cbegin(), countAt(0)),
               queriesIn.size(),
               FindCollisions<T, selfCollision, CountCollisions>{
                   nodeBBox_, internalChildren_, {counts, empty}});
    // compute start index for each query and total count
    exclusive_scan(ExecutionPolicy::Par, counts.begin(), counts.end(),
                   counts.begin(), 0, std::plus<int>());
    if (counts.back() == 0) return SparseIndices(0);
    SparseIndices queryTri(counts.back());
    // actually recording collisions
    for_each_n(ExecutionPolicy::Par, zip(queriesIn.cbegin(), countAt(0)),
               queriesIn.size(),
               FindCollisions<T, selfCollision, ParCollisionRecorder<inverted>>{
                   nodeBBox_, internalChildren_, {queryTri, counts, empty}});
    return queryTri;
  }
}

/**
 * Recalculate the collider's internal bounding boxes without changing the
 * hierarchy.
 */
void Collider::UpdateBoxes(const VecView<const Box>& leafBB) {
  ZoneScoped;
  ASSERT(leafBB.size() == NumLeaves(), userErr,
         "must have the same number of updated boxes as original");
  // copy in leaf node Boxes
  strided_range<Vec<Box>::Iter> leaves(nodeBBox_.begin(), nodeBBox_.end(), 2);
  auto policy = autoPolicy(NumInternal());
  copy(policy, leafBB.cbegin(), leafBB.cend(), leaves.begin());
  // create global counters
  Vec<int> counter(NumInternal(), 0);
  // kernel over leaves to save internal Boxes
  for_each_n(
      policy, countAt(0), NumLeaves(),
      BuildInternalBoxes({nodeBBox_, counter, nodeParent_, internalChildren_}));
}

/**
 * Apply axis-aligned transform to all bounding boxes. If transform is not
 * axis-aligned, abort and return false to indicate recalculation is necessary.
 */
bool Collider::Transform(glm::mat4x3 transform) {
  ZoneScoped;
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

template SparseIndices Collider::Collisions<true, false, Box>(
    const VecView<const Box>&) const;

template SparseIndices Collider::Collisions<false, false, Box>(
    const VecView<const Box>&) const;

template SparseIndices Collider::Collisions<false, false, glm::vec3>(
    const VecView<const glm::vec3>&) const;

template SparseIndices Collider::Collisions<true, true, Box>(
    const VecView<const Box>&) const;

template SparseIndices Collider::Collisions<false, true, Box>(
    const VecView<const Box>&) const;

template SparseIndices Collider::Collisions<false, true, glm::vec3>(
    const VecView<const glm::vec3>&) const;

}  // namespace manifold
