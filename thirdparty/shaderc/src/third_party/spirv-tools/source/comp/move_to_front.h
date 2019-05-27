// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_COMP_MOVE_TO_FRONT_H_
#define SOURCE_COMP_MOVE_TO_FRONT_H_

#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace spvtools {
namespace comp {

// Log(n) move-to-front implementation. Implements the following functions:
// Insert - pushes value to the front of the mtf sequence
//          (only unique values allowed).
// Remove - remove value from the sequence.
// ValueFromRank - access value by its 1-indexed rank in the sequence.
// RankFromValue - get the rank of the given value in the sequence.
// Accessing a value with ValueFromRank or RankFromValue moves the value to the
// front of the sequence (rank of 1).
//
// The implementation is based on an AVL-based order statistic tree. The tree
// is ordered by timestamps issued when values are inserted or accessed (recent
// values go to the left side of the tree, old values are gradually rotated to
// the right side).
//
// Terminology
// rank: 1-indexed rank showing how recently the value was inserted or accessed.
// node: handle used internally to access node data.
// size: size of the subtree of a node (including the node).
// height: distance from a node to the farthest leaf.
class MoveToFront {
 public:
  explicit MoveToFront(size_t reserve_capacity = 4) {
    nodes_.reserve(reserve_capacity);

    // Create NIL node.
    nodes_.emplace_back(Node());
  }

  virtual ~MoveToFront() = default;

  // Inserts value in the move-to-front sequence. Does nothing if the value is
  // already in the sequence. Returns true if insertion was successful.
  // The inserted value is placed at the front of the sequence (rank 1).
  bool Insert(uint32_t value);

  // Removes value from move-to-front sequence. Returns false iff the value
  // was not found.
  bool Remove(uint32_t value);

  // Computes 1-indexed rank of value in the move-to-front sequence and moves
  // the value to the front. Example:
  // Before the call: 4 8 2 1 7
  // RankFromValue(8) returns 2
  // After the call: 8 4 2 1 7
  // Returns true iff the value was found in the sequence.
  bool RankFromValue(uint32_t value, uint32_t* rank);

  // Returns value corresponding to a 1-indexed rank in the move-to-front
  // sequence and moves the value to the front. Example:
  // Before the call: 4 8 2 1 7
  // ValueFromRank(2) returns 8
  // After the call: 8 4 2 1 7
  // Returns true iff the rank is within bounds [1, GetSize()].
  bool ValueFromRank(uint32_t rank, uint32_t* value);

  // Moves the value to the front of the sequence.
  // Returns false iff value is not in the sequence.
  bool Promote(uint32_t value);

  // Returns true iff the move-to-front sequence contains the value.
  bool HasValue(uint32_t value) const;

  // Returns the number of elements in the move-to-front sequence.
  uint32_t GetSize() const { return SizeOf(root_); }

 protected:
  // Internal tree data structure uses handles instead of pointers. Leaves and
  // root parent reference a singleton under handle 0. Although dereferencing
  // a null pointer is not possible, inappropriate access to handle 0 would
  // cause an assertion. Handles are not garbage collected if value was
  // deprecated
  // with DeprecateValue(). But handles are recycled when a node is
  // repositioned.

  // Internal tree data structure node.
  struct Node {
    // Timestamp from a logical clock which updates every time the element is
    // accessed through ValueFromRank or RankFromValue.
    uint32_t timestamp = 0;
    // The size of the node's subtree, including the node.
    // SizeOf(LeftOf(node)) + SizeOf(RightOf(node)) + 1.
    uint32_t size = 0;
    // Handles to connected nodes.
    uint32_t left = 0;
    uint32_t right = 0;
    uint32_t parent = 0;
    // Distance to the farthest leaf.
    // Leaves have height 0, real nodes at least 1.
    uint32_t height = 0;
    // Stored value.
    uint32_t value = 0;
  };

  // Creates node and sets correct values. Non-NIL nodes should be created only
  // through this function. If the node with this value has been created
  // previously
  // and since orphaned, reuses the old node instead of creating a new one.
  uint32_t CreateNode(uint32_t timestamp, uint32_t value);

  // Node accessor methods. Naming is designed to be similar to natural
  // language as these functions tend to be used in sequences, for example:
  // ParentOf(LeftestDescendentOf(RightOf(node)))

  // Returns value of the node referenced by |handle|.
  uint32_t ValueOf(uint32_t node) const { return nodes_.at(node).value; }

  // Returns left child of |node|.
  uint32_t LeftOf(uint32_t node) const { return nodes_.at(node).left; }

  // Returns right child of |node|.
  uint32_t RightOf(uint32_t node) const { return nodes_.at(node).right; }

  // Returns parent of |node|.
  uint32_t ParentOf(uint32_t node) const { return nodes_.at(node).parent; }

  // Returns timestamp of |node|.
  uint32_t TimestampOf(uint32_t node) const {
    assert(node);
    return nodes_.at(node).timestamp;
  }

  // Returns size of |node|.
  uint32_t SizeOf(uint32_t node) const { return nodes_.at(node).size; }

  // Returns height of |node|.
  uint32_t HeightOf(uint32_t node) const { return nodes_.at(node).height; }

  // Returns mutable reference to value of |node|.
  uint32_t& MutableValueOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).value;
  }

  // Returns mutable reference to handle of left child of |node|.
  uint32_t& MutableLeftOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).left;
  }

  // Returns mutable reference to handle of right child of |node|.
  uint32_t& MutableRightOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).right;
  }

  // Returns mutable reference to handle of parent of |node|.
  uint32_t& MutableParentOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).parent;
  }

  // Returns mutable reference to timestamp of |node|.
  uint32_t& MutableTimestampOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).timestamp;
  }

  // Returns mutable reference to size of |node|.
  uint32_t& MutableSizeOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).size;
  }

  // Returns mutable reference to height of |node|.
  uint32_t& MutableHeightOf(uint32_t node) {
    assert(node);
    return nodes_.at(node).height;
  }

  // Returns true iff |node| is left child of its parent.
  bool IsLeftChild(uint32_t node) const {
    assert(node);
    return LeftOf(ParentOf(node)) == node;
  }

  // Returns true iff |node| is right child of its parent.
  bool IsRightChild(uint32_t node) const {
    assert(node);
    return RightOf(ParentOf(node)) == node;
  }

  // Returns true iff |node| has no relatives.
  bool IsOrphan(uint32_t node) const {
    assert(node);
    return !ParentOf(node) && !LeftOf(node) && !RightOf(node);
  }

  // Returns true iff |node| is in the tree.
  bool IsInTree(uint32_t node) const {
    assert(node);
    return node == root_ || !IsOrphan(node);
  }

  // Returns the height difference between right and left subtrees.
  int BalanceOf(uint32_t node) const {
    return int(HeightOf(RightOf(node))) - int(HeightOf(LeftOf(node)));
  }

  // Updates size and height of the node, assuming that the children have
  // correct values.
  void UpdateNode(uint32_t node);

  // Returns the most LeftOf(LeftOf(... descendent which is not leaf.
  uint32_t LeftestDescendantOf(uint32_t node) const {
    uint32_t parent = 0;
    while (node) {
      parent = node;
      node = LeftOf(node);
    }
    return parent;
  }

  // Returns the most RightOf(RightOf(... descendent which is not leaf.
  uint32_t RightestDescendantOf(uint32_t node) const {
    uint32_t parent = 0;
    while (node) {
      parent = node;
      node = RightOf(node);
    }
    return parent;
  }

  // Inserts node in the tree. The node must be an orphan.
  void InsertNode(uint32_t node);

  // Removes node from the tree. May change value_to_node_ if removal uses a
  // scapegoat. Returns the removed (orphaned) handle for recycling. The
  // returned handle may not be equal to |node| if scapegoat was used.
  uint32_t RemoveNode(uint32_t node);

  // Rotates |node| left, reassigns all connections and returns the node
  // which takes place of the |node|.
  uint32_t RotateLeft(const uint32_t node);

  // Rotates |node| right, reassigns all connections and returns the node
  // which takes place of the |node|.
  uint32_t RotateRight(const uint32_t node);

  // Root node handle. The tree is empty if root_ is 0.
  uint32_t root_ = 0;

  // Incremented counters for next timestamp and value.
  uint32_t next_timestamp_ = 1;

  // Holds all tree nodes. Indices of this vector are node handles.
  std::vector<Node> nodes_;

  // Maps ids to node handles.
  std::unordered_map<uint32_t, uint32_t> value_to_node_;

  // Cache for the last accessed value in the sequence.
  uint32_t last_accessed_value_ = 0;
  bool last_accessed_value_valid_ = false;
};

class MultiMoveToFront {
 public:
  // Inserts |value| to sequence with handle |mtf|.
  // Returns false if |mtf| already has |value|.
  bool Insert(uint64_t mtf, uint32_t value) {
    if (GetMtf(mtf).Insert(value)) {
      val_to_mtfs_[value].insert(mtf);
      return true;
    }
    return false;
  }

  // Removes |value| from sequence with handle |mtf|.
  // Returns false if |mtf| doesn't have |value|.
  bool Remove(uint64_t mtf, uint32_t value) {
    if (GetMtf(mtf).Remove(value)) {
      val_to_mtfs_[value].erase(mtf);
      return true;
    }
    assert(val_to_mtfs_[value].count(mtf) == 0);
    return false;
  }

  // Removes |value| from all sequences which have it.
  void RemoveFromAll(uint32_t value) {
    auto it = val_to_mtfs_.find(value);
    if (it == val_to_mtfs_.end()) return;

    auto& mtfs_containing_value = it->second;
    for (uint64_t mtf : mtfs_containing_value) {
      GetMtf(mtf).Remove(value);
    }

    val_to_mtfs_.erase(value);
  }

  // Computes rank of |value| in sequence |mtf|.
  // Returns false if |mtf| doesn't have |value|.
  bool RankFromValue(uint64_t mtf, uint32_t value, uint32_t* rank) {
    return GetMtf(mtf).RankFromValue(value, rank);
  }

  // Finds |value| with |rank| in sequence |mtf|.
  // Returns false if |rank| is out of bounds.
  bool ValueFromRank(uint64_t mtf, uint32_t rank, uint32_t* value) {
    return GetMtf(mtf).ValueFromRank(rank, value);
  }

  // Returns size of |mtf| sequence.
  uint32_t GetSize(uint64_t mtf) { return GetMtf(mtf).GetSize(); }

  // Promotes |value| in all sequences which have it.
  void Promote(uint32_t value) {
    const auto it = val_to_mtfs_.find(value);
    if (it == val_to_mtfs_.end()) return;

    const auto& mtfs_containing_value = it->second;
    for (uint64_t mtf : mtfs_containing_value) {
      GetMtf(mtf).Promote(value);
    }
  }

  // Inserts |value| in sequence |mtf| or promotes if it's already there.
  void InsertOrPromote(uint64_t mtf, uint32_t value) {
    if (!Insert(mtf, value)) {
      GetMtf(mtf).Promote(value);
    }
  }

  // Returns if |mtf| sequence has |value|.
  bool HasValue(uint64_t mtf, uint32_t value) {
    return GetMtf(mtf).HasValue(value);
  }

 private:
  // Returns actual MoveToFront object corresponding to |handle|.
  // As multiple operations are often performed consecutively for the same
  // sequence, the last returned value is cached.
  MoveToFront& GetMtf(uint64_t handle) {
    if (!cached_mtf_ || cached_handle_ != handle) {
      cached_handle_ = handle;
      cached_mtf_ = &mtfs_[handle];
    }

    return *cached_mtf_;
  }

  // Container holding MoveToFront objects. Map key is sequence handle.
  std::map<uint64_t, MoveToFront> mtfs_;

  // Container mapping value to sequences which contain that value.
  std::unordered_map<uint32_t, std::set<uint64_t>> val_to_mtfs_;

  // Cache for the last accessed sequence.
  uint64_t cached_handle_ = 0;
  MoveToFront* cached_mtf_ = nullptr;
};

}  // namespace comp
}  // namespace spvtools

#endif  // SOURCE_COMP_MOVE_TO_FRONT_H_
