//==--- ImmutableList.h - Immutable (functional) list interface --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImmutableList class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMMUTABLELIST_H
#define LLVM_ADT_IMMUTABLELIST_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

template <typename T> class ImmutableListFactory;

template <typename T>
class ImmutableListImpl : public FoldingSetNode {
  T Head;
  const ImmutableListImpl* Tail;

  ImmutableListImpl(const T& head, const ImmutableListImpl* tail = 0)
    : Head(head), Tail(tail) {}

  friend class ImmutableListFactory<T>;

  void operator=(const ImmutableListImpl&) = delete;
  ImmutableListImpl(const ImmutableListImpl&) = delete;

public:
  const T& getHead() const { return Head; }
  const ImmutableListImpl* getTail() const { return Tail; }

  static inline void Profile(FoldingSetNodeID& ID, const T& H,
                             const ImmutableListImpl* L){
    ID.AddPointer(L);
    ID.Add(H);
  }

  void Profile(FoldingSetNodeID& ID) {
    Profile(ID, Head, Tail);
  }
};

/// ImmutableList - This class represents an immutable (functional) list.
///  It is implemented as a smart pointer (wraps ImmutableListImpl), so it
///  it is intended to always be copied by value as if it were a pointer.
///  This interface matches ImmutableSet and ImmutableMap.  ImmutableList
///  objects should almost never be created directly, and instead should
///  be created by ImmutableListFactory objects that manage the lifetime
///  of a group of lists.  When the factory object is reclaimed, all lists
///  created by that factory are released as well.
template <typename T>
class ImmutableList {
public:
  typedef T value_type;
  typedef ImmutableListFactory<T> Factory;

private:
  const ImmutableListImpl<T>* X;

public:
  // This constructor should normally only be called by ImmutableListFactory<T>.
  // There may be cases, however, when one needs to extract the internal pointer
  // and reconstruct a list object from that pointer.
  ImmutableList(const ImmutableListImpl<T>* x = 0) : X(x) {}

  const ImmutableListImpl<T>* getInternalPointer() const {
    return X;
  }

  class iterator {
    const ImmutableListImpl<T>* L;
  public:
    iterator() : L(0) {}
    iterator(ImmutableList l) : L(l.getInternalPointer()) {}

    iterator& operator++() { L = L->getTail(); return *this; }
    bool operator==(const iterator& I) const { return L == I.L; }
    bool operator!=(const iterator& I) const { return L != I.L; }
    const value_type& operator*() const { return L->getHead(); }
    ImmutableList getList() const { return L; }
  };

  /// begin - Returns an iterator referring to the head of the list, or
  ///  an iterator denoting the end of the list if the list is empty.
  iterator begin() const { return iterator(X); }

  /// end - Returns an iterator denoting the end of the list.  This iterator
  ///  does not refer to a valid list element.
  iterator end() const { return iterator(); }

  /// isEmpty - Returns true if the list is empty.
  bool isEmpty() const { return !X; }

  bool contains(const T& V) const {
    for (iterator I = begin(), E = end(); I != E; ++I) {
      if (*I == V)
        return true;
    }
    return false;
  }

  /// isEqual - Returns true if two lists are equal.  Because all lists created
  ///  from the same ImmutableListFactory are uniqued, this has O(1) complexity
  ///  because it the contents of the list do not need to be compared.  Note
  ///  that you should only compare two lists created from the same
  ///  ImmutableListFactory.
  bool isEqual(const ImmutableList& L) const { return X == L.X; }

  bool operator==(const ImmutableList& L) const { return isEqual(L); }

  /// getHead - Returns the head of the list.
  const T& getHead() {
    assert (!isEmpty() && "Cannot get the head of an empty list.");
    return X->getHead();
  }

  /// getTail - Returns the tail of the list, which is another (possibly empty)
  ///  ImmutableList.
  ImmutableList getTail() {
    return X ? X->getTail() : 0;
  }

  void Profile(FoldingSetNodeID& ID) const {
    ID.AddPointer(X);
  }
};

template <typename T>
class ImmutableListFactory {
  typedef ImmutableListImpl<T> ListTy;
  typedef FoldingSet<ListTy>   CacheTy;

  CacheTy Cache;
  uintptr_t Allocator;

  bool ownsAllocator() const {
    return Allocator & 0x1 ? false : true;
  }

  BumpPtrAllocator& getAllocator() const {
    return *reinterpret_cast<BumpPtrAllocator*>(Allocator & ~0x1);
  }

public:
  ImmutableListFactory()
    : Allocator(reinterpret_cast<uintptr_t>(new BumpPtrAllocator())) {}

  ImmutableListFactory(BumpPtrAllocator& Alloc)
  : Allocator(reinterpret_cast<uintptr_t>(&Alloc) | 0x1) {}

  ~ImmutableListFactory() {
    if (ownsAllocator()) delete &getAllocator();
  }

  ImmutableList<T> concat(const T& Head, ImmutableList<T> Tail) {
    // Profile the new list to see if it already exists in our cache.
    FoldingSetNodeID ID;
    void* InsertPos;

    const ListTy* TailImpl = Tail.getInternalPointer();
    ListTy::Profile(ID, Head, TailImpl);
    ListTy* L = Cache.FindNodeOrInsertPos(ID, InsertPos);

    if (!L) {
      // The list does not exist in our cache.  Create it.
      BumpPtrAllocator& A = getAllocator();
      L = (ListTy*) A.Allocate<ListTy>();
      new (L) ListTy(Head, TailImpl);

      // Insert the new list into the cache.
      Cache.InsertNode(L, InsertPos);
    }

    return L;
  }

  ImmutableList<T> add(const T& D, ImmutableList<T> L) {
    return concat(D, L);
  }

  ImmutableList<T> getEmptyList() const {
    return ImmutableList<T>(0);
  }

  ImmutableList<T> create(const T& X) {
    return Concat(X, getEmptyList());
  }
};

//===----------------------------------------------------------------------===//
// Partially-specialized Traits.
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

template<typename T> struct DenseMapInfo;
template<typename T> struct DenseMapInfo<ImmutableList<T> > {
  static inline ImmutableList<T> getEmptyKey() {
    return reinterpret_cast<ImmutableListImpl<T>*>(-1);
  }
  static inline ImmutableList<T> getTombstoneKey() {
    return reinterpret_cast<ImmutableListImpl<T>*>(-2);
  }
  static unsigned getHashValue(ImmutableList<T> X) {
    uintptr_t PtrVal = reinterpret_cast<uintptr_t>(X.getInternalPointer());
    return (unsigned((uintptr_t)PtrVal) >> 4) ^
           (unsigned((uintptr_t)PtrVal) >> 9);
  }
  static bool isEqual(ImmutableList<T> X1, ImmutableList<T> X2) {
    return X1 == X2;
  }
};

template <typename T> struct isPodLike;
template <typename T>
struct isPodLike<ImmutableList<T> > { static const bool value = true; };

} // end llvm namespace

#endif
