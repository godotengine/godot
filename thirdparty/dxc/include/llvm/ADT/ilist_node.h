//==-- llvm/ADT/ilist_node.h - Intrusive Linked List Helper ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ilist_node class template, which is a convenient
// base class for creating classes that can be used with ilists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ILIST_NODE_H
#define LLVM_ADT_ILIST_NODE_H

namespace llvm {

template<typename NodeTy>
struct ilist_traits;

/// ilist_half_node - Base class that provides prev services for sentinels.
///
template<typename NodeTy>
class ilist_half_node {
  friend struct ilist_traits<NodeTy>;
  NodeTy *Prev;
protected:
  NodeTy *getPrev() { return Prev; }
  const NodeTy *getPrev() const { return Prev; }
  void setPrev(NodeTy *P) { Prev = P; }
  ilist_half_node() : Prev(nullptr) {}
};

template<typename NodeTy>
struct ilist_nextprev_traits;

/// ilist_node - Base class that provides next/prev services for nodes
/// that use ilist_nextprev_traits or ilist_default_traits.
///
template<typename NodeTy>
class ilist_node : private ilist_half_node<NodeTy> {
  friend struct ilist_nextprev_traits<NodeTy>;
  friend struct ilist_traits<NodeTy>;
  NodeTy *Next;
  NodeTy *getNext() { return Next; }
  const NodeTy *getNext() const { return Next; }
  void setNext(NodeTy *N) { Next = N; }
protected:
  ilist_node() : Next(nullptr) {}

public:
  /// @name Adjacent Node Accessors
  /// @{

  /// \brief Get the previous node, or 0 for the list head.
  NodeTy *getPrevNode() {
    NodeTy *Prev = this->getPrev();

    // Check for sentinel.
    if (Prev && !Prev->getNext()) // HLSL Change: Prev may be nullptr
      return nullptr;

    return Prev;
  }

  /// \brief Get the previous node, or 0 for the list head.
  const NodeTy *getPrevNode() const {
    const NodeTy *Prev = this->getPrev();

    // Check for sentinel.
    if (Prev && !Prev->getNext()) // HLSL Change: Prev may be nullptr
      return nullptr;

    return Prev;
  }

  /// \brief Get the next node, or 0 for the list tail.
  NodeTy *getNextNode() {
    NodeTy *Next = getNext();

    // Check for sentinel.
    if (Next && !Next->getNext()) // HLSL Change: Next may be nullptr
      return nullptr;

    return Next;
  }

  /// \brief Get the next node, or 0 for the list tail.
  const NodeTy *getNextNode() const {
    const NodeTy *Next = getNext();

    // Check for sentinel.
    if (Next && !Next->getNext()) // HLSL Change: Next may be nullptr
      return nullptr;

    return Next;
  }

  /// @}
};

} // End llvm namespace

#endif
