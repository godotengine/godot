// Copyright (c) 2018 Google Inc.
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

#include "source/comp/move_to_front.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace spvtools {
namespace comp {

bool MoveToFront::Insert(uint32_t value) {
  auto it = value_to_node_.find(value);
  if (it != value_to_node_.end() && IsInTree(it->second)) return false;

  const uint32_t old_size = GetSize();
  (void)old_size;

  InsertNode(CreateNode(next_timestamp_++, value));

  last_accessed_value_ = value;
  last_accessed_value_valid_ = true;

  assert(value_to_node_.count(value));
  assert(old_size + 1 == GetSize());
  return true;
}

bool MoveToFront::Remove(uint32_t value) {
  auto it = value_to_node_.find(value);
  if (it == value_to_node_.end()) return false;

  if (!IsInTree(it->second)) return false;

  if (last_accessed_value_ == value) last_accessed_value_valid_ = false;

  const uint32_t orphan = RemoveNode(it->second);
  (void)orphan;
  // The node of |value| is still alive but it's orphaned now. Can still be
  // reused later.
  assert(!IsInTree(orphan));
  assert(ValueOf(orphan) == value);
  return true;
}

bool MoveToFront::RankFromValue(uint32_t value, uint32_t* rank) {
  if (last_accessed_value_valid_ && last_accessed_value_ == value) {
    *rank = 1;
    return true;
  }

  const uint32_t old_size = GetSize();
  if (old_size == 1) {
    if (ValueOf(root_) == value) {
      *rank = 1;
      return true;
    } else {
      return false;
    }
  }

  const auto it = value_to_node_.find(value);
  if (it == value_to_node_.end()) {
    return false;
  }

  uint32_t target = it->second;

  if (!IsInTree(target)) {
    return false;
  }

  uint32_t node = target;
  *rank = 1 + SizeOf(LeftOf(node));
  while (node) {
    if (IsRightChild(node)) *rank += 1 + SizeOf(LeftOf(ParentOf(node)));
    node = ParentOf(node);
  }

  // Don't update timestamp if the node has rank 1.
  if (*rank != 1) {
    // Update timestamp and reposition the node.
    target = RemoveNode(target);
    assert(ValueOf(target) == value);
    assert(old_size == GetSize() + 1);
    MutableTimestampOf(target) = next_timestamp_++;
    InsertNode(target);
    assert(old_size == GetSize());
  }

  last_accessed_value_ = value;
  last_accessed_value_valid_ = true;
  return true;
}

bool MoveToFront::HasValue(uint32_t value) const {
  const auto it = value_to_node_.find(value);
  if (it == value_to_node_.end()) {
    return false;
  }

  return IsInTree(it->second);
}

bool MoveToFront::Promote(uint32_t value) {
  if (last_accessed_value_valid_ && last_accessed_value_ == value) {
    return true;
  }

  const uint32_t old_size = GetSize();
  if (old_size == 1) return ValueOf(root_) == value;

  const auto it = value_to_node_.find(value);
  if (it == value_to_node_.end()) {
    return false;
  }

  uint32_t target = it->second;

  if (!IsInTree(target)) {
    return false;
  }

  // Update timestamp and reposition the node.
  target = RemoveNode(target);
  assert(ValueOf(target) == value);
  assert(old_size == GetSize() + 1);

  MutableTimestampOf(target) = next_timestamp_++;
  InsertNode(target);
  assert(old_size == GetSize());

  last_accessed_value_ = value;
  last_accessed_value_valid_ = true;
  return true;
}

bool MoveToFront::ValueFromRank(uint32_t rank, uint32_t* value) {
  if (last_accessed_value_valid_ && rank == 1) {
    *value = last_accessed_value_;
    return true;
  }

  const uint32_t old_size = GetSize();
  if (rank <= 0 || rank > old_size) {
    return false;
  }

  if (old_size == 1) {
    *value = ValueOf(root_);
    return true;
  }

  const bool update_timestamp = (rank != 1);

  uint32_t node = root_;
  while (node) {
    const uint32_t left_subtree_num_nodes = SizeOf(LeftOf(node));
    if (rank == left_subtree_num_nodes + 1) {
      // This is the node we are looking for.
      // Don't update timestamp if the node has rank 1.
      if (update_timestamp) {
        node = RemoveNode(node);
        assert(old_size == GetSize() + 1);
        MutableTimestampOf(node) = next_timestamp_++;
        InsertNode(node);
        assert(old_size == GetSize());
      }
      *value = ValueOf(node);
      last_accessed_value_ = *value;
      last_accessed_value_valid_ = true;
      return true;
    }

    if (rank < left_subtree_num_nodes + 1) {
      // Descend into the left subtree. The rank is still valid.
      node = LeftOf(node);
    } else {
      // Descend into the right subtree. We leave behind the left subtree and
      // the current node, adjust the |rank| accordingly.
      rank -= left_subtree_num_nodes + 1;
      node = RightOf(node);
    }
  }

  assert(0);
  return false;
}

uint32_t MoveToFront::CreateNode(uint32_t timestamp, uint32_t value) {
  uint32_t handle = static_cast<uint32_t>(nodes_.size());
  const auto result = value_to_node_.emplace(value, handle);
  if (result.second) {
    // Create new node.
    nodes_.emplace_back(Node());
    Node& node = nodes_.back();
    node.timestamp = timestamp;
    node.value = value;
    node.size = 1;
    // Non-NIL nodes start with height 1 because their NIL children are
    // leaves.
    node.height = 1;
  } else {
    // Reuse old node.
    handle = result.first->second;
    assert(!IsInTree(handle));
    assert(ValueOf(handle) == value);
    assert(SizeOf(handle) == 1);
    assert(HeightOf(handle) == 1);
    MutableTimestampOf(handle) = timestamp;
  }

  return handle;
}

void MoveToFront::InsertNode(uint32_t node) {
  assert(!IsInTree(node));
  assert(SizeOf(node) == 1);
  assert(HeightOf(node) == 1);
  assert(TimestampOf(node));

  if (!root_) {
    root_ = node;
    return;
  }

  uint32_t iter = root_;
  uint32_t parent = 0;

  // Will determine if |node| will become the right or left child after
  // insertion (but before balancing).
  bool right_child = true;

  // Find the node which will become |node|'s parent after insertion
  // (but before balancing).
  while (iter) {
    parent = iter;
    assert(TimestampOf(iter) != TimestampOf(node));
    right_child = TimestampOf(iter) > TimestampOf(node);
    iter = right_child ? RightOf(iter) : LeftOf(iter);
  }

  assert(parent);

  // Connect node and parent.
  MutableParentOf(node) = parent;
  if (right_child)
    MutableRightOf(parent) = node;
  else
    MutableLeftOf(parent) = node;

  // Insertion is finished. Start the balancing process.
  bool needs_rebalancing = true;
  parent = ParentOf(node);

  while (parent) {
    UpdateNode(parent);

    if (needs_rebalancing) {
      const int parent_balance = BalanceOf(parent);

      if (RightOf(parent) == node) {
        // Added node to the right subtree.
        if (parent_balance > 1) {
          // Parent is right heavy, rotate left.
          if (BalanceOf(node) < 0) RotateRight(node);
          parent = RotateLeft(parent);
        } else if (parent_balance == 0 || parent_balance == -1) {
          // Parent is balanced or left heavy, no need to balance further.
          needs_rebalancing = false;
        }
      } else {
        // Added node to the left subtree.
        if (parent_balance < -1) {
          // Parent is left heavy, rotate right.
          if (BalanceOf(node) > 0) RotateLeft(node);
          parent = RotateRight(parent);
        } else if (parent_balance == 0 || parent_balance == 1) {
          // Parent is balanced or right heavy, no need to balance further.
          needs_rebalancing = false;
        }
      }
    }

    assert(BalanceOf(parent) >= -1 && (BalanceOf(parent) <= 1));

    node = parent;
    parent = ParentOf(parent);
  }
}

uint32_t MoveToFront::RemoveNode(uint32_t node) {
  if (LeftOf(node) && RightOf(node)) {
    // If |node| has two children, then use another node as scapegoat and swap
    // their contents. We pick the scapegoat on the side of the tree which has
    // more nodes.
    const uint32_t scapegoat = SizeOf(LeftOf(node)) >= SizeOf(RightOf(node))
                                   ? RightestDescendantOf(LeftOf(node))
                                   : LeftestDescendantOf(RightOf(node));
    assert(scapegoat);
    std::swap(MutableValueOf(node), MutableValueOf(scapegoat));
    std::swap(MutableTimestampOf(node), MutableTimestampOf(scapegoat));
    value_to_node_[ValueOf(node)] = node;
    value_to_node_[ValueOf(scapegoat)] = scapegoat;
    node = scapegoat;
  }

  // |node| may have only one child at this point.
  assert(!RightOf(node) || !LeftOf(node));

  uint32_t parent = ParentOf(node);
  uint32_t child = RightOf(node) ? RightOf(node) : LeftOf(node);

  // Orphan |node| and reconnect parent and child.
  if (child) MutableParentOf(child) = parent;

  if (parent) {
    if (LeftOf(parent) == node)
      MutableLeftOf(parent) = child;
    else
      MutableRightOf(parent) = child;
  }

  MutableParentOf(node) = 0;
  MutableLeftOf(node) = 0;
  MutableRightOf(node) = 0;
  UpdateNode(node);
  const uint32_t orphan = node;

  if (root_ == node) root_ = child;

  // Removal is finished. Start the balancing process.
  bool needs_rebalancing = true;
  node = child;

  while (parent) {
    UpdateNode(parent);

    if (needs_rebalancing) {
      const int parent_balance = BalanceOf(parent);

      if (parent_balance == 1 || parent_balance == -1) {
        // The height of the subtree was not changed.
        needs_rebalancing = false;
      } else {
        if (RightOf(parent) == node) {
          // Removed node from the right subtree.
          if (parent_balance < -1) {
            // Parent is left heavy, rotate right.
            const uint32_t sibling = LeftOf(parent);
            if (BalanceOf(sibling) > 0) RotateLeft(sibling);
            parent = RotateRight(parent);
          }
        } else {
          // Removed node from the left subtree.
          if (parent_balance > 1) {
            // Parent is right heavy, rotate left.
            const uint32_t sibling = RightOf(parent);
            if (BalanceOf(sibling) < 0) RotateRight(sibling);
            parent = RotateLeft(parent);
          }
        }
      }
    }

    assert(BalanceOf(parent) >= -1 && (BalanceOf(parent) <= 1));

    node = parent;
    parent = ParentOf(parent);
  }

  return orphan;
}

uint32_t MoveToFront::RotateLeft(const uint32_t node) {
  const uint32_t pivot = RightOf(node);
  assert(pivot);

  // LeftOf(pivot) gets attached to node in place of pivot.
  MutableRightOf(node) = LeftOf(pivot);
  if (RightOf(node)) MutableParentOf(RightOf(node)) = node;

  // Pivot gets attached to ParentOf(node) in place of node.
  MutableParentOf(pivot) = ParentOf(node);
  if (!ParentOf(node))
    root_ = pivot;
  else if (IsLeftChild(node))
    MutableLeftOf(ParentOf(node)) = pivot;
  else
    MutableRightOf(ParentOf(node)) = pivot;

  // Node is child of pivot.
  MutableLeftOf(pivot) = node;
  MutableParentOf(node) = pivot;

  // Update both node and pivot. Pivot is the new parent of node, so node should
  // be updated first.
  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

uint32_t MoveToFront::RotateRight(const uint32_t node) {
  const uint32_t pivot = LeftOf(node);
  assert(pivot);

  // RightOf(pivot) gets attached to node in place of pivot.
  MutableLeftOf(node) = RightOf(pivot);
  if (LeftOf(node)) MutableParentOf(LeftOf(node)) = node;

  // Pivot gets attached to ParentOf(node) in place of node.
  MutableParentOf(pivot) = ParentOf(node);
  if (!ParentOf(node))
    root_ = pivot;
  else if (IsLeftChild(node))
    MutableLeftOf(ParentOf(node)) = pivot;
  else
    MutableRightOf(ParentOf(node)) = pivot;

  // Node is child of pivot.
  MutableRightOf(pivot) = node;
  MutableParentOf(node) = pivot;

  // Update both node and pivot. Pivot is the new parent of node, so node should
  // be updated first.
  UpdateNode(node);
  UpdateNode(pivot);

  return pivot;
}

void MoveToFront::UpdateNode(uint32_t node) {
  MutableSizeOf(node) = 1 + SizeOf(LeftOf(node)) + SizeOf(RightOf(node));
  MutableHeightOf(node) =
      1 + std::max(HeightOf(LeftOf(node)), HeightOf(RightOf(node)));
}

}  // namespace comp
}  // namespace spvtools
