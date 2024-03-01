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

#ifndef SOURCE_UTIL_ILIST_NODE_H_
#define SOURCE_UTIL_ILIST_NODE_H_

#include <cassert>

namespace spvtools {
namespace utils {

template <class NodeType>
class IntrusiveList;

// IntrusiveNodeBase is the base class for nodes in an IntrusiveList.
// See the comments in ilist.h on how to use the class.

template <class NodeType>
class IntrusiveNodeBase {
 public:
  // Creates a new node that is not in a list.
  inline IntrusiveNodeBase();
  inline IntrusiveNodeBase(const IntrusiveNodeBase&);
  inline IntrusiveNodeBase& operator=(const IntrusiveNodeBase&);
  inline IntrusiveNodeBase(IntrusiveNodeBase&& that);

  // Destroys a node.  It is an error to destroy a node that is part of a
  // list, unless it is a sentinel.
  virtual ~IntrusiveNodeBase();

  IntrusiveNodeBase& operator=(IntrusiveNodeBase&& that);

  // Returns true if |this| is in a list.
  inline bool IsInAList() const;

  // Returns the node that comes after the given node in the list, if one
  // exists.  If the given node is not in a list or is at the end of the list,
  // the return value is nullptr.
  inline NodeType* NextNode() const;

  // Returns the node that comes before the given node in the list, if one
  // exists.  If the given node is not in a list or is at the start of the
  // list, the return value is nullptr.
  inline NodeType* PreviousNode() const;

  // Inserts the given node immediately before |pos| in the list.
  // If the given node is already in a list, it will first be removed
  // from that list.
  //
  // It is assumed that the given node is of type NodeType.  It is an error if
  // |pos| is not already in a list.
  inline void InsertBefore(NodeType* pos);

  // Inserts the given node immediately after |pos| in the list.
  // If the given node is already in a list, it will first be removed
  // from that list.
  //
  // It is assumed that the given node is of type NodeType.  It is an error if
  // |pos| is not already in a list, or if |pos| is equal to |this|.
  inline void InsertAfter(NodeType* pos);

  // Removes the given node from the list.  It is assumed that the node is
  // in a list.  Note that this does not free any storage related to the node,
  // it becomes the caller's responsibility to free the storage.
  inline void RemoveFromList();

 protected:
  // Replaces |this| with |target|.  |this| is a sentinel if and only if
  // |target| is also a sentinel.
  //
  // If neither node is a sentinel, |target| takes
  // the place of |this|.  It is assumed that |target| is not in a list.
  //
  // If both are sentinels, then it will cause all of the
  // nodes in the list containing |this| to be moved to the list containing
  // |target|.  In this case, it is assumed that |target| is an empty list.
  //
  // No storage will be deleted.
  void ReplaceWith(NodeType* target);

  // Returns true if |this| is the sentinel node of an empty list.
  bool IsEmptyList();

  // The pointers to the next and previous nodes in the list.
  // If the current node is not part of a list, then |next_node_| and
  // |previous_node_| are equal to |nullptr|.
  NodeType* next_node_;
  NodeType* previous_node_;

  // Only true for the sentinel node stored in the list itself.
  bool is_sentinel_;

  friend IntrusiveList<NodeType>;
};

// Implementation of IntrusiveNodeBase

template <class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase()
    : next_node_(nullptr), previous_node_(nullptr), is_sentinel_(false) {}

template <class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase(
    const IntrusiveNodeBase&) {
  next_node_ = nullptr;
  previous_node_ = nullptr;
  is_sentinel_ = false;
}

template <class NodeType>
inline IntrusiveNodeBase<NodeType>& IntrusiveNodeBase<NodeType>::operator=(
    const IntrusiveNodeBase&) {
  assert(!is_sentinel_);
  if (IsInAList()) {
    RemoveFromList();
  }
  return *this;
}

template <class NodeType>
inline IntrusiveNodeBase<NodeType>::IntrusiveNodeBase(IntrusiveNodeBase&& that)
    : next_node_(nullptr),
      previous_node_(nullptr),
      is_sentinel_(that.is_sentinel_) {
  if (is_sentinel_) {
    next_node_ = this;
    previous_node_ = this;
  }
  that.ReplaceWith(this);
}

template <class NodeType>
IntrusiveNodeBase<NodeType>::~IntrusiveNodeBase() {
  assert(is_sentinel_ || !IsInAList());
}

template <class NodeType>
IntrusiveNodeBase<NodeType>& IntrusiveNodeBase<NodeType>::operator=(
    IntrusiveNodeBase&& that) {
  that.ReplaceWith(this);
  return *this;
}

template <class NodeType>
inline bool IntrusiveNodeBase<NodeType>::IsInAList() const {
  return next_node_ != nullptr;
}

template <class NodeType>
inline NodeType* IntrusiveNodeBase<NodeType>::NextNode() const {
  if (!next_node_->is_sentinel_) return next_node_;
  return nullptr;
}

template <class NodeType>
inline NodeType* IntrusiveNodeBase<NodeType>::PreviousNode() const {
  if (!previous_node_->is_sentinel_) return previous_node_;
  return nullptr;
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::InsertBefore(NodeType* pos) {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(pos->IsInAList() && "Pos should already be in a list.");
  if (this->IsInAList()) this->RemoveFromList();

  this->next_node_ = pos;
  this->previous_node_ = pos->previous_node_;
  pos->previous_node_ = static_cast<NodeType*>(this);
  this->previous_node_->next_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::InsertAfter(NodeType* pos) {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(pos->IsInAList() && "Pos should already be in a list.");
  assert(this != pos && "Can't insert a node after itself.");

  if (this->IsInAList()) {
    this->RemoveFromList();
  }

  this->previous_node_ = pos;
  this->next_node_ = pos->next_node_;
  pos->next_node_ = static_cast<NodeType*>(this);
  this->next_node_->previous_node_ = static_cast<NodeType*>(this);
}

template <class NodeType>
inline void IntrusiveNodeBase<NodeType>::RemoveFromList() {
  assert(!this->is_sentinel_ && "Sentinel nodes cannot be moved around.");
  assert(this->IsInAList() &&
         "Cannot remove a node from a list if it is not in a list.");

  this->next_node_->previous_node_ = this->previous_node_;
  this->previous_node_->next_node_ = this->next_node_;
  this->next_node_ = nullptr;
  this->previous_node_ = nullptr;
}

template <class NodeType>
void IntrusiveNodeBase<NodeType>::ReplaceWith(NodeType* target) {
  if (this->is_sentinel_) {
    assert(target->IsEmptyList() &&
           "If target is not an empty list, the nodes in that list would not "
           "be linked to a sentinel.");
  } else {
    assert(IsInAList() && "The node being replaced must be in a list.");
    assert(!target->is_sentinel_ &&
           "Cannot turn a sentinel node into one that is not.");
  }

  if (!this->IsEmptyList()) {
    // Link target into the same position that |this| was in.
    target->next_node_ = this->next_node_;
    target->previous_node_ = this->previous_node_;
    target->next_node_->previous_node_ = target;
    target->previous_node_->next_node_ = target;

    // Reset |this| to itself default value.
    if (!this->is_sentinel_) {
      // Reset |this| so that it is not in a list.
      this->next_node_ = nullptr;
      this->previous_node_ = nullptr;
    } else {
      // Set |this| so that it is the head of an empty list.
      // We cannot treat sentinel nodes like others because it is invalid for
      // a sentinel node to not be in a list.
      this->next_node_ = static_cast<NodeType*>(this);
      this->previous_node_ = static_cast<NodeType*>(this);
    }
  } else {
    // If |this| points to itself, it must be a sentinel node with an empty
    // list.  Reset |this| so that it is the head of an empty list.  We want
    // |target| to be the same.  The asserts above guarantee that.
  }
}

template <class NodeType>
bool IntrusiveNodeBase<NodeType>::IsEmptyList() {
  if (next_node_ == this) {
    assert(is_sentinel_ &&
           "None sentinel nodes should never point to themselves.");
    assert(previous_node_ == this &&
           "Inconsistency with the previous and next nodes.");
    return true;
  }
  return false;
}

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_ILIST_NODE_H_
