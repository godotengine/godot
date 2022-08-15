//==- llvm/Support/Recycler.h - Recycling Allocator --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Recycler class template.  See the doxygen comment for
// Recycler for more details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RECYCLER_H
#define LLVM_SUPPORT_RECYCLER_H

#include "llvm/ADT/ilist.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {

/// PrintRecyclingAllocatorStats - Helper for RecyclingAllocator for
/// printing statistics.
///
void PrintRecyclerStats(size_t Size, size_t Align, size_t FreeListSize);

/// RecyclerStruct - Implementation detail for Recycler. This is a
/// class that the recycler imposes on free'd memory to carve out
/// next/prev pointers.
struct RecyclerStruct {
  RecyclerStruct *Prev, *Next;
};

template<>
struct ilist_traits<RecyclerStruct> :
    public ilist_default_traits<RecyclerStruct> {
  static RecyclerStruct *getPrev(const RecyclerStruct *t) { return t->Prev; }
  static RecyclerStruct *getNext(const RecyclerStruct *t) { return t->Next; }
  static void setPrev(RecyclerStruct *t, RecyclerStruct *p) { t->Prev = p; }
  static void setNext(RecyclerStruct *t, RecyclerStruct *n) { t->Next = n; }

  mutable RecyclerStruct Sentinel;
  RecyclerStruct *createSentinel() const {
    return &Sentinel;
  }
  static void destroySentinel(RecyclerStruct *) {}

  RecyclerStruct *provideInitialHead() const { return createSentinel(); }
  RecyclerStruct *ensureHead(RecyclerStruct*) const { return createSentinel(); }
  static void noteHead(RecyclerStruct*, RecyclerStruct*) {}

  static void deleteNode(RecyclerStruct *) {
    llvm_unreachable("Recycler's ilist_traits shouldn't see a deleteNode call!");
  }
};

/// Recycler - This class manages a linked-list of deallocated nodes
/// and facilitates reusing deallocated memory in place of allocating
/// new memory.
///
template<class T, size_t Size = sizeof(T), size_t Align = AlignOf<T>::Alignment>
class Recycler {
  /// FreeList - Doubly-linked list of nodes that have deleted contents and
  /// are not in active use.
  ///
  iplist<RecyclerStruct> FreeList;

public:
  ~Recycler() {
    // If this fails, either the callee has lost track of some allocation,
    // or the callee isn't tracking allocations and should just call
    // clear() before deleting the Recycler.
    assert(FreeList.empty() && "Non-empty recycler deleted!");
  }

  /// clear - Release all the tracked allocations to the allocator. The
  /// recycler must be free of any tracked allocations before being
  /// deleted; calling clear is one way to ensure this.
  template<class AllocatorType>
  void clear(AllocatorType &Allocator) {
    while (!FreeList.empty()) {
      T *t = reinterpret_cast<T *>(FreeList.remove(FreeList.begin()));
      Allocator.Deallocate(t);
    }
  }

  /// Special case for BumpPtrAllocator which has an empty Deallocate()
  /// function.
  ///
  /// There is no need to traverse the free list, pulling all the objects into
  /// cache.
  void clear(BumpPtrAllocator&) {
    FreeList.clearAndLeakNodesUnsafely();
  }

  template<class SubClass, class AllocatorType>
  SubClass *Allocate(AllocatorType &Allocator) {
    static_assert(AlignOf<SubClass>::Alignment <= Align,
                  "Recycler allocation alignment is less than object align!");
    static_assert(sizeof(SubClass) <= Size,
                  "Recycler allocation size is less than object size!");
    return !FreeList.empty() ?
           reinterpret_cast<SubClass *>(FreeList.remove(FreeList.begin())) :
           static_cast<SubClass *>(Allocator.Allocate(Size, Align));
  }

  template<class AllocatorType>
  T *Allocate(AllocatorType &Allocator) {
    return Allocate<T>(Allocator);
  }

  template<class SubClass, class AllocatorType>
  void Deallocate(AllocatorType & /*Allocator*/, SubClass* Element) {
    FreeList.push_front(reinterpret_cast<RecyclerStruct *>(Element));
  }

  void PrintStats() {
    PrintRecyclerStats(Size, Align, FreeList.size());
  }
};

}

#endif
