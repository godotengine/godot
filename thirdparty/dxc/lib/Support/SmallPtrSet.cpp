//===- llvm/ADT/SmallPtrSet.cpp - 'Normally small' pointer set ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallPtrSet class.  See SmallPtrSet.h for an
// overview of the algorithm.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdlib>

using namespace llvm;

void SmallPtrSetImplBase::shrink_and_clear() {
  assert(!isSmall() && "Can't shrink a small set!");
  delete[] CurArray; // HLSL Change: Use overridable operator delete

  // Reduce the number of buckets.
  CurArraySize = NumElements > 16 ? 1 << (Log2_32_Ceil(NumElements) + 1) : 32;
  NumElements = NumTombstones = 0;

  // Install the new array.  Clear all the buckets to empty.
  CurArray = new const void*[CurArraySize]; // HLSL Change: Use overridable operator new
  assert(CurArray && "Failed to allocate memory?");
  memset(CurArray, -1, CurArraySize*sizeof(void*));
}

std::pair<const void *const *, bool>
SmallPtrSetImplBase::insert_imp(const void *Ptr) {
  if (isSmall()) {
    // Check to see if it is already in the set.
    for (const void **APtr = SmallArray, **E = SmallArray+NumElements;
         APtr != E; ++APtr)
      if (*APtr == Ptr)
        return std::make_pair(APtr, false);

    // Nope, there isn't.  If we stay small, just 'pushback' now.
    if (NumElements < CurArraySize) {
      SmallArray[NumElements++] = Ptr;
      return std::make_pair(SmallArray + (NumElements - 1), true);
    }
    // Otherwise, hit the big set case, which will call grow.
  }

  if (LLVM_UNLIKELY(NumElements * 4 >= CurArraySize * 3)) {
    // If more than 3/4 of the array is full, grow.
    Grow(CurArraySize < 64 ? 128 : CurArraySize*2);
  } else if (LLVM_UNLIKELY(CurArraySize - (NumElements + NumTombstones) <
                           CurArraySize / 8)) {
    // If fewer of 1/8 of the array is empty (meaning that many are filled with
    // tombstones), rehash.
    Grow(CurArraySize);
  }
  
  // Okay, we know we have space.  Find a hash bucket.
  const void **Bucket = const_cast<const void**>(FindBucketFor(Ptr));
  if (*Bucket == Ptr)
    return std::make_pair(Bucket, false); // Already inserted, good.

  // Otherwise, insert it!
  if (*Bucket == getTombstoneMarker())
    --NumTombstones;
  *Bucket = Ptr;
  ++NumElements;  // Track density.
  return std::make_pair(Bucket, true);
}

bool SmallPtrSetImplBase::erase_imp(const void * Ptr) {
  if (isSmall()) {
    // Check to see if it is in the set.
    for (const void **APtr = SmallArray, **E = SmallArray+NumElements;
         APtr != E; ++APtr)
      if (*APtr == Ptr) {
        // If it is in the set, replace this element.
        *APtr = E[-1];
        E[-1] = getEmptyMarker();
        --NumElements;
        return true;
      }
    
    return false;
  }
  
  // Okay, we know we have space.  Find a hash bucket.
  void **Bucket = const_cast<void**>(FindBucketFor(Ptr));
  if (*Bucket != Ptr) return false;  // Not in the set?

  // Set this as a tombstone.
  *Bucket = getTombstoneMarker();
  --NumElements;
  ++NumTombstones;
  return true;
}

const void * const *SmallPtrSetImplBase::FindBucketFor(const void *Ptr) const {
  unsigned Bucket = DenseMapInfo<void *>::getHashValue(Ptr) & (CurArraySize-1);
  unsigned ArraySize = CurArraySize;
  unsigned ProbeAmt = 1;
  const void *const *Array = CurArray;
  const void *const *Tombstone = nullptr;
  while (1) {
    // If we found an empty bucket, the pointer doesn't exist in the set.
    // Return a tombstone if we've seen one so far, or the empty bucket if
    // not.
    if (LLVM_LIKELY(Array[Bucket] == getEmptyMarker()))
      return Tombstone ? Tombstone : Array+Bucket;

    // Found Ptr's bucket?
    if (LLVM_LIKELY(Array[Bucket] == Ptr))
      return Array+Bucket;

    // If this is a tombstone, remember it.  If Ptr ends up not in the set, we
    // prefer to return it than something that would require more probing.
    if (Array[Bucket] == getTombstoneMarker() && !Tombstone)
      Tombstone = Array+Bucket;  // Remember the first tombstone found.
    
    // It's a hash collision or a tombstone. Reprobe.
    Bucket = (Bucket + ProbeAmt++) & (ArraySize-1);
  }
}

/// Grow - Allocate a larger backing store for the buckets and move it over.
///
void SmallPtrSetImplBase::Grow(unsigned NewSize) {
  // Allocate at twice as many buckets, but at least 128.
  unsigned OldSize = CurArraySize;
  
  const void **OldBuckets = CurArray;
  bool WasSmall = isSmall();
  
  // Install the new array.  Clear all the buckets to empty.
  CurArray = new const void*[NewSize]; // HLSL Change: Use overridable operator new
  assert(CurArray && "Failed to allocate memory?");
  CurArraySize = NewSize;
  memset(CurArray, -1, NewSize*sizeof(void*));
  
  // Copy over all the elements.
  if (WasSmall) {
    // Small sets store their elements in order.
    for (const void **BucketPtr = OldBuckets, **E = OldBuckets+NumElements;
         BucketPtr != E; ++BucketPtr) {
      const void *Elt = *BucketPtr;
      *const_cast<void**>(FindBucketFor(Elt)) = const_cast<void*>(Elt);
    }
  } else {
    // Copy over all valid entries.
    for (const void **BucketPtr = OldBuckets, **E = OldBuckets+OldSize;
         BucketPtr != E; ++BucketPtr) {
      // Copy over the element if it is valid.
      const void *Elt = *BucketPtr;
      if (Elt != getTombstoneMarker() && Elt != getEmptyMarker())
        *const_cast<void**>(FindBucketFor(Elt)) = const_cast<void*>(Elt);
    }
    
    delete[] OldBuckets; // HLSL Change: Use overridable operator delete
    NumTombstones = 0;
  }
}

SmallPtrSetImplBase::SmallPtrSetImplBase(const void **SmallStorage,
                                 const SmallPtrSetImplBase& that) {
  SmallArray = SmallStorage;

  // If we're becoming small, prepare to insert into our stack space
  if (that.isSmall()) {
    CurArray = SmallArray;
  // Otherwise, allocate new heap space (unless we were the same size)
  } else {
    CurArray = new const void*[that.CurArraySize]; // HLSL Change: Use overridable operator new
    assert(CurArray && "Failed to allocate memory?");
  }
  
  // Copy over the new array size
  CurArraySize = that.CurArraySize;

  // Copy over the contents from the other set
  memcpy(CurArray, that.CurArray, sizeof(void*)*CurArraySize);
  
  NumElements = that.NumElements;
  NumTombstones = that.NumTombstones;
}

SmallPtrSetImplBase::SmallPtrSetImplBase(const void **SmallStorage,
                                         unsigned SmallSize,
                                         SmallPtrSetImplBase &&that) {
  SmallArray = SmallStorage;

  // Copy over the basic members.
  CurArraySize = that.CurArraySize;
  NumElements = that.NumElements;
  NumTombstones = that.NumTombstones;

  // When small, just copy into our small buffer.
  if (that.isSmall()) {
    CurArray = SmallArray;
    memcpy(CurArray, that.CurArray, sizeof(void *) * CurArraySize);
  } else {
    // Otherwise, we steal the large memory allocation and no copy is needed.
    CurArray = that.CurArray;
    that.CurArray = that.SmallArray;
  }

  // Make the "that" object small and empty.
  that.CurArraySize = SmallSize;
  assert(that.CurArray == that.SmallArray);
  that.NumElements = 0;
  that.NumTombstones = 0;
}

/// CopyFrom - implement operator= from a smallptrset that has the same pointer
/// type, but may have a different small size.
void SmallPtrSetImplBase::CopyFrom(const SmallPtrSetImplBase &RHS) {
  assert(&RHS != this && "Self-copy should be handled by the caller.");

  if (isSmall() && RHS.isSmall())
    assert(CurArraySize == RHS.CurArraySize &&
           "Cannot assign sets with different small sizes");

  // If we're becoming small, prepare to insert into our stack space
  if (RHS.isSmall()) {
    if (!isSmall())
      delete[] CurArray; // HLSL Change: Use overridable operator delete
    CurArray = SmallArray;
  // Otherwise, allocate new heap space (unless we were the same size)
  } else if (CurArraySize != RHS.CurArraySize) {
    if (isSmall())
      CurArray = new const void*[RHS.CurArraySize]; // HLSL Change: Use overridable operator new
    else {
      // HLSL Change Begins: Use overridable operator new
      const void **T = new const void*[RHS.CurArraySize];
      std::memcpy(T, CurArray, std::min(CurArraySize, RHS.CurArraySize));
      delete[] CurArray;
      CurArray = T;
      // HLSL Change Ends
    }
    assert(CurArray && "Failed to allocate memory?");
  }
  
  // Copy over the new array size
  CurArraySize = RHS.CurArraySize;

  // Copy over the contents from the other set
  memcpy(CurArray, RHS.CurArray, sizeof(void*)*CurArraySize);
  
  NumElements = RHS.NumElements;
  NumTombstones = RHS.NumTombstones;
}

void SmallPtrSetImplBase::MoveFrom(unsigned SmallSize,
                                   SmallPtrSetImplBase &&RHS) {
  assert(&RHS != this && "Self-move should be handled by the caller.");

  if (!isSmall())
    delete[] CurArray; // HLSL Change: Use overridable operator delete

  if (RHS.isSmall()) {
    // Copy a small RHS rather than moving.
    CurArray = SmallArray;
    memcpy(CurArray, RHS.CurArray, sizeof(void*)*RHS.CurArraySize);
  } else {
    CurArray = RHS.CurArray;
    RHS.CurArray = RHS.SmallArray;
  }

  // Copy the rest of the trivial members.
  CurArraySize = RHS.CurArraySize;
  NumElements = RHS.NumElements;
  NumTombstones = RHS.NumTombstones;

  // Make the RHS small and empty.
  RHS.CurArraySize = SmallSize;
  assert(RHS.CurArray == RHS.SmallArray);
  RHS.NumElements = 0;
  RHS.NumTombstones = 0;
}

void SmallPtrSetImplBase::swap(SmallPtrSetImplBase &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither set is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->CurArray, RHS.CurArray);
    std::swap(this->CurArraySize, RHS.CurArraySize);
    std::swap(this->NumElements, RHS.NumElements);
    std::swap(this->NumTombstones, RHS.NumTombstones);
    return;
  }

  // FIXME: From here on we assume that both sets have the same small size.

  // If only RHS is small, copy the small elements into LHS and move the pointer
  // from LHS to RHS.
  if (!this->isSmall() && RHS.isSmall()) {
    std::copy(RHS.SmallArray, RHS.SmallArray+RHS.CurArraySize,
              this->SmallArray);
    std::swap(this->NumElements, RHS.NumElements);
    std::swap(this->CurArraySize, RHS.CurArraySize);
    RHS.CurArray = this->CurArray;
    RHS.NumTombstones = this->NumTombstones;
    this->CurArray = this->SmallArray;
    this->NumTombstones = 0;
    return;
  }

  // If only LHS is small, copy the small elements into RHS and move the pointer
  // from RHS to LHS.
  if (this->isSmall() && !RHS.isSmall()) {
    std::copy(this->SmallArray, this->SmallArray+this->CurArraySize,
              RHS.SmallArray);
    std::swap(RHS.NumElements, this->NumElements);
    std::swap(RHS.CurArraySize, this->CurArraySize);
    this->CurArray = RHS.CurArray;
    this->NumTombstones = RHS.NumTombstones;
    RHS.CurArray = RHS.SmallArray;
    RHS.NumTombstones = 0;
    return;
  }

  // Both a small, just swap the small elements.
  assert(this->isSmall() && RHS.isSmall());
  assert(this->CurArraySize == RHS.CurArraySize);
  std::swap_ranges(this->SmallArray, this->SmallArray+this->CurArraySize,
                   RHS.SmallArray);
  std::swap(this->NumElements, RHS.NumElements);
}

SmallPtrSetImplBase::~SmallPtrSetImplBase() {
  if (!isSmall())
    delete[] CurArray; // HLSL Change: Use overridable operator delete
}
