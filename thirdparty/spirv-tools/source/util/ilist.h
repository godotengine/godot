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

#ifndef SOURCE_UTIL_ILIST_H_
#define SOURCE_UTIL_ILIST_H_

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "source/util/ilist_node.h"

namespace spvtools {
namespace utils {

// An IntrusiveList is a generic implementation of a doubly-linked list.  The
// intended convention for using this container is:
//
//      class Node : public IntrusiveNodeBase<Node> {
//        // Note that "Node", the class being defined is the template.
//        // Must have a default constructor accessible to List.
//        // Add whatever data is needed in the node
//      };
//
//      using List = IntrusiveList<Node>;
//
// You can also inherit from IntrusiveList instead of a typedef if you want to
// add more functionality.
//
// The condition on the template for IntrusiveNodeBase is there to add some type
// checking to the container.  The compiler will still allow inserting elements
// of type IntrusiveNodeBase<Node>, but that would be an error. This assumption
// allows NextNode and PreviousNode to return pointers to Node, and casting will
// not be required by the user.

template <class NodeType>
class IntrusiveList {
 public:
  static_assert(
      std::is_base_of<IntrusiveNodeBase<NodeType>, NodeType>::value,
      "The type from the node must be derived from IntrusiveNodeBase, with "
      "itself in the template.");

  // Creates an empty list.
  inline IntrusiveList();

  // Moves the contents of the given list to the list being constructed.
  IntrusiveList(IntrusiveList&&);

  // Destroys the list.  Note that the elements of the list will not be deleted,
  // but they will be removed from the list.
  virtual ~IntrusiveList();

  // Moves all of the elements in the list on the RHS to the list on the LHS.
  IntrusiveList& operator=(IntrusiveList&&);

  // Basetype for iterators so an IntrusiveList can be traversed like STL
  // containers.
  template <class T>
  class iterator_template {
   public:
    iterator_template(const iterator_template& i) : node_(i.node_) {}

    iterator_template& operator++() {
      node_ = node_->next_node_;
      return *this;
    }

    iterator_template& operator--() {
      node_ = node_->previous_node_;
      return *this;
    }

    iterator_template& operator=(const iterator_template& i) {
      node_ = i.node_;
      return *this;
    }

    T& operator*() const { return *node_; }
    T* operator->() const { return node_; }

    friend inline bool operator==(const iterator_template& lhs,
                                  const iterator_template& rhs) {
      return lhs.node_ == rhs.node_;
    }
    friend inline bool operator!=(const iterator_template& lhs,
                                  const iterator_template& rhs) {
      return !(lhs == rhs);
    }

    // Moves the nodes in |list| to the list that |this| points to.  The
    // positions of the nodes will be immediately before the element pointed to
    // by the iterator.  The return value will be an iterator pointing to the
    // first of the newly inserted elements.
    iterator_template MoveBefore(IntrusiveList* list) {
      if (list->empty()) return *this;

      NodeType* first_node = list->sentinel_.next_node_;
      NodeType* last_node = list->sentinel_.previous_node_;

      this->node_->previous_node_->next_node_ = first_node;
      first_node->previous_node_ = this->node_->previous_node_;

      last_node->next_node_ = this->node_;
      this->node_->previous_node_ = last_node;

      list->sentinel_.next_node_ = &list->sentinel_;
      list->sentinel_.previous_node_ = &list->sentinel_;

      return iterator(first_node);
    }

    // Define standard iterator types needs so this class can be
    // used with <algorithms>.
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

   protected:
    iterator_template() = delete;
    inline iterator_template(T* node) { node_ = node; }
    T* node_;

    friend IntrusiveList;
  };

  using iterator = iterator_template<NodeType>;
  using const_iterator = iterator_template<const NodeType>;

  // Various types of iterators for the start (begin) and one past the end (end)
  // of the list.
  //
  // Decrementing |end()| iterator will give and iterator pointing to the last
  // element in the list, if one exists.
  //
  // Incrementing |end()| iterator will give |begin()|.
  //
  // Decrementing |begin()| will give |end()|.
  //
  // TODO: Not marking these functions as noexcept because Visual Studio 2013
  // does not support it.  When we no longer care about that compiler, we should
  // mark these as noexcept.
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

  // Appends |node| to the end of the list.  If |node| is already in a list, it
  // will be removed from that list first.
  void push_back(NodeType* node);

  // Returns true if the list is empty.
  bool empty() const;

  // Makes the current list empty.
  inline void clear();

  // Returns references to the first or last element in the list.  It is an
  // error to call these functions on an empty list.
  NodeType& front();
  NodeType& back();
  const NodeType& front() const;
  const NodeType& back() const;

  // Transfers [|first|, |last|) from |other| into the list at |where|.
  //
  // If |other| is |this|, no change is made.
  void Splice(iterator where, IntrusiveList<NodeType>* other, iterator first,
              iterator last);

 protected:
  // Doing a deep copy of the list does not make sense if the list does not own
  // the data.  It is not clear who will own the newly created data.  Making
  // copies illegal for that reason.
  IntrusiveList(const IntrusiveList&) = delete;
  IntrusiveList& operator=(const IntrusiveList&) = delete;

  // This function will assert if it finds the list containing |node| is not in
  // a valid state.
  static void Check(NodeType* node);

  // A special node used to represent both the start and end of the list,
  // without being part of the list.
  NodeType sentinel_;
};

// Implementation of IntrusiveList

template <class NodeType>
inline IntrusiveList<NodeType>::IntrusiveList() : sentinel_() {
  sentinel_.next_node_ = &sentinel_;
  sentinel_.previous_node_ = &sentinel_;
  sentinel_.is_sentinel_ = true;
}

template <class NodeType>
IntrusiveList<NodeType>::IntrusiveList(IntrusiveList&& list) : sentinel_() {
  sentinel_.next_node_ = &sentinel_;
  sentinel_.previous_node_ = &sentinel_;
  sentinel_.is_sentinel_ = true;
  list.sentinel_.ReplaceWith(&sentinel_);
}

template <class NodeType>
IntrusiveList<NodeType>::~IntrusiveList() {
  clear();
}

template <class NodeType>
IntrusiveList<NodeType>& IntrusiveList<NodeType>::operator=(
    IntrusiveList<NodeType>&& list) {
  list.sentinel_.ReplaceWith(&sentinel_);
  return *this;
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::iterator
IntrusiveList<NodeType>::begin() {
  return iterator(sentinel_.next_node_);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::iterator
IntrusiveList<NodeType>::end() {
  return iterator(&sentinel_);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::begin() const {
  return const_iterator(sentinel_.next_node_);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::end() const {
  return const_iterator(&sentinel_);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::cbegin() const {
  return const_iterator(sentinel_.next_node_);
}

template <class NodeType>
inline typename IntrusiveList<NodeType>::const_iterator
IntrusiveList<NodeType>::cend() const {
  return const_iterator(&sentinel_);
}

template <class NodeType>
void IntrusiveList<NodeType>::push_back(NodeType* node) {
  node->InsertBefore(&sentinel_);
}

template <class NodeType>
bool IntrusiveList<NodeType>::empty() const {
  return sentinel_.NextNode() == nullptr;
}

template <class NodeType>
void IntrusiveList<NodeType>::clear() {
  while (!empty()) {
    front().RemoveFromList();
  }
}

template <class NodeType>
NodeType& IntrusiveList<NodeType>::front() {
  NodeType* node = sentinel_.NextNode();
  assert(node != nullptr && "Can't get the front of an empty list.");
  return *node;
}

template <class NodeType>
NodeType& IntrusiveList<NodeType>::back() {
  NodeType* node = sentinel_.PreviousNode();
  assert(node != nullptr && "Can't get the back of an empty list.");
  return *node;
}

template <class NodeType>
const NodeType& IntrusiveList<NodeType>::front() const {
  NodeType* node = sentinel_.NextNode();
  assert(node != nullptr && "Can't get the front of an empty list.");
  return *node;
}

template <class NodeType>
const NodeType& IntrusiveList<NodeType>::back() const {
  NodeType* node = sentinel_.PreviousNode();
  assert(node != nullptr && "Can't get the back of an empty list.");
  return *node;
}

template <class NodeType>
void IntrusiveList<NodeType>::Splice(iterator where,
                                     IntrusiveList<NodeType>* other,
                                     iterator first, iterator last) {
  if (first == last) return;
  if (other == this) return;

  NodeType* first_prev = first.node_->previous_node_;
  NodeType* where_next = where.node_->next_node_;

  // Attach first.
  where.node_->next_node_ = first.node_;
  first.node_->previous_node_ = where.node_;

  // Attach last.
  where_next->previous_node_ = last.node_->previous_node_;
  last.node_->previous_node_->next_node_ = where_next;

  // Fixup other.
  first_prev->next_node_ = last.node_;
  last.node_->previous_node_ = first_prev;
}

template <class NodeType>
void IntrusiveList<NodeType>::Check(NodeType* start) {
  int sentinel_count = 0;
  NodeType* p = start;
  do {
    assert(p != nullptr);
    assert(p->next_node_->previous_node_ == p);
    assert(p->previous_node_->next_node_ == p);
    if (p->is_sentinel_) sentinel_count++;
    p = p->next_node_;
  } while (p != start);
  assert(sentinel_count == 1 && "List should have exactly 1 sentinel node.");
  (void)sentinel_count;

  p = start;
  do {
    assert(p != nullptr);
    assert(p->previous_node_->next_node_ == p);
    assert(p->next_node_->previous_node_ == p);
    if (p->is_sentinel_) sentinel_count++;
    p = p->previous_node_;
  } while (p != start);
}

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_ILIST_H_
