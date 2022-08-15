//===--- Allocator.h - Simple memory allocation abstraction -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the MallocAllocator and BumpPtrAllocator interfaces. Both
/// of these conform to an LLVM "Allocator" concept which consists of an
/// Allocate method accepting a size and alignment, and a Deallocate accepting
/// a pointer and size. Further, the LLVM "Allocator" concept has overloads of
/// Allocate and Deallocate for setting size and alignment based on the final
/// type. These overloads are typically provided by a base class template \c
/// AllocatorBase.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALLOCATOR_H
#define LLVM_SUPPORT_ALLOCATOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Memory.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace llvm {

/// \brief CRTP base class providing obvious overloads for the core \c
/// Allocate() methods of LLVM-style allocators.
///
/// This base class both documents the full public interface exposed by all
/// LLVM-style allocators, and redirects all of the overloads to a single core
/// set of methods which the derived class must define.
template <typename DerivedT> class AllocatorBase {
public:
  /// \brief Allocate \a Size bytes of \a Alignment aligned memory. This method
  /// must be implemented by \c DerivedT.
  void *Allocate(size_t Size, size_t Alignment) {
#ifdef __clang__
    static_assert(static_cast<void *(AllocatorBase::*)(size_t, size_t)>(
                      &AllocatorBase::Allocate) !=
                      static_cast<void *(DerivedT::*)(size_t, size_t)>(
                          &DerivedT::Allocate),
                  "Class derives from AllocatorBase without implementing the "
                  "core Allocate(size_t, size_t) overload!");
#endif
    return static_cast<DerivedT *>(this)->Allocate(Size, Alignment);
  }

  /// \brief Deallocate \a Ptr to \a Size bytes of memory allocated by this
  /// allocator.
  void Deallocate(const void *Ptr, size_t Size) {
#ifdef __clang__
    static_assert(static_cast<void (AllocatorBase::*)(const void *, size_t)>(
                      &AllocatorBase::Deallocate) !=
                      static_cast<void (DerivedT::*)(const void *, size_t)>(
                          &DerivedT::Deallocate),
                  "Class derives from AllocatorBase without implementing the "
                  "core Deallocate(void *) overload!");
#endif
    return static_cast<DerivedT *>(this)->Deallocate(Ptr, Size);
  }

  // The rest of these methods are helpers that redirect to one of the above
  // core methods.

  /// \brief Allocate space for a sequence of objects without constructing them.
  template <typename T> T *Allocate(size_t Num = 1) {
    return static_cast<T *>(Allocate(Num * sizeof(T), AlignOf<T>::Alignment));
  }

  /// \brief Deallocate space for a sequence of objects without constructing them.
  template <typename T>
  typename std::enable_if<
      !std::is_same<typename std::remove_cv<T>::type, void>::value, void>::type
  Deallocate(T *Ptr, size_t Num = 1) {
    Deallocate(static_cast<const void *>(Ptr), Num * sizeof(T));
  }
};

class MallocAllocator : public AllocatorBase<MallocAllocator> {
public:
  void Reset() {}

  _Ret_notnull_   // HLSL Change - SAL
  LLVM_ATTRIBUTE_RETURNS_NONNULL void *Allocate(size_t Size,
                                                size_t /*Alignment*/) {
    return ::operator new(Size); // HLSL Change: use overridable operator new and throw on OOM
  }

  // Pull in base class overloads.
  using AllocatorBase<MallocAllocator>::Allocate;

  void Deallocate(const void *Ptr, size_t /*Size*/) {
    ::operator delete(const_cast<void*>(Ptr)); // HLSL Change: use overridable operator delete
  }

  // Pull in base class overloads.
  using AllocatorBase<MallocAllocator>::Deallocate;

  void PrintStats() const {}
};

namespace detail {

// We call out to an external function to actually print the message as the
// printing code uses Allocator.h in its implementation.
void printBumpPtrAllocatorStats(unsigned NumSlabs, size_t BytesAllocated,
                                size_t TotalMemory);
} // End namespace detail.

/// \brief Allocate memory in an ever growing pool, as if by bump-pointer.
///
/// This isn't strictly a bump-pointer allocator as it uses backing slabs of
/// memory rather than relying on a boundless contiguous heap. However, it has
/// bump-pointer semantics in that it is a monotonically growing pool of memory
/// where every allocation is found by merely allocating the next N bytes in
/// the slab, or the next N bytes in the next slab.
///
/// Note that this also has a threshold for forcing allocations above a certain
/// size into their own slab.
///
/// The BumpPtrAllocatorImpl template defaults to using a MallocAllocator
/// object, which wraps malloc, to allocate memory, but it can be changed to
/// use a custom allocator.
template <typename AllocatorT = MallocAllocator, size_t SlabSize = 4096,
          size_t SizeThreshold = SlabSize>
class BumpPtrAllocatorImpl
    : public AllocatorBase<
          BumpPtrAllocatorImpl<AllocatorT, SlabSize, SizeThreshold>> {
public:
  static_assert(SizeThreshold <= SlabSize,
                "The SizeThreshold must be at most the SlabSize to ensure "
                "that objects larger than a slab go into their own memory "
                "allocation.");

  BumpPtrAllocatorImpl()
      : CurPtr(nullptr), End(nullptr), BytesAllocated(0), Allocator() {}
  template <typename T>
  BumpPtrAllocatorImpl(T &&Allocator)
      : CurPtr(nullptr), End(nullptr), BytesAllocated(0),
        Allocator(std::forward<T &&>(Allocator)) {}

  // Manually implement a move constructor as we must clear the old allocator's
  // slabs as a matter of correctness.
  BumpPtrAllocatorImpl(BumpPtrAllocatorImpl &&Old)
      : CurPtr(Old.CurPtr), End(Old.End), Slabs(std::move(Old.Slabs)),
        CustomSizedSlabs(std::move(Old.CustomSizedSlabs)),
        BytesAllocated(Old.BytesAllocated),
        Allocator(std::move(Old.Allocator)) {
    Old.CurPtr = Old.End = nullptr;
    Old.BytesAllocated = 0;
    Old.Slabs.clear();
    Old.CustomSizedSlabs.clear();
  }

  ~BumpPtrAllocatorImpl() {
    DeallocateSlabs(Slabs.begin(), Slabs.end());
    DeallocateCustomSizedSlabs();
  }

  BumpPtrAllocatorImpl &operator=(BumpPtrAllocatorImpl &&RHS) {
    DeallocateSlabs(Slabs.begin(), Slabs.end());
    DeallocateCustomSizedSlabs();

    CurPtr = RHS.CurPtr;
    End = RHS.End;
    BytesAllocated = RHS.BytesAllocated;
    Slabs = std::move(RHS.Slabs);
    CustomSizedSlabs = std::move(RHS.CustomSizedSlabs);
    Allocator = std::move(RHS.Allocator);

    RHS.CurPtr = RHS.End = nullptr;
    RHS.BytesAllocated = 0;
    RHS.Slabs.clear();
    RHS.CustomSizedSlabs.clear();
    return *this;
  }

  /// \brief Deallocate all but the current slab and reset the current pointer
  /// to the beginning of it, freeing all memory allocated so far.
  void Reset() {
    DeallocateCustomSizedSlabs();
    CustomSizedSlabs.clear();

    if (Slabs.empty())
      return;

    // Reset the state.
    BytesAllocated = 0;
    CurPtr = (char *)Slabs.front();
    End = CurPtr + SlabSize;

    // Deallocate all but the first slab, and deallocate all custom-sized slabs.
    DeallocateSlabs(std::next(Slabs.begin()), Slabs.end());
    Slabs.erase(std::next(Slabs.begin()), Slabs.end());
  }

  /// \brief Allocate space at the specified alignment.
  LLVM_ATTRIBUTE_RETURNS_NONNULL LLVM_ATTRIBUTE_RETURNS_NOALIAS void *
  Allocate(size_t Size, size_t Alignment) {
    assert(Alignment > 0 && "0-byte alignnment is not allowed. Use 1 instead.");

    // Keep track of how many bytes we've allocated.
    BytesAllocated += Size;

    size_t Adjustment = alignmentAdjustment(CurPtr, Alignment);
    assert(Adjustment + Size >= Size && "Adjustment + Size must not overflow");

    // Check if we have enough space.
    if (Adjustment + Size <= size_t(End - CurPtr)) {
      char *AlignedPtr = CurPtr + Adjustment;
      CurPtr = AlignedPtr + Size;
      // Update the allocation point of this memory block in MemorySanitizer.
      // Without this, MemorySanitizer messages for values originated from here
      // will point to the allocation of the entire slab.
      __msan_allocated_memory(AlignedPtr, Size);
      return AlignedPtr;
    }

    // If Size is really big, allocate a separate slab for it.
    size_t PaddedSize = Size + Alignment - 1;
    if (PaddedSize > SizeThreshold) {
      void *NewSlab = Allocator.Allocate(PaddedSize, 0);
      CustomSizedSlabs.push_back(std::make_pair(NewSlab, PaddedSize));

      uintptr_t AlignedAddr = alignAddr(NewSlab, Alignment);
      assert(AlignedAddr + Size <= (uintptr_t)NewSlab + PaddedSize);
      char *AlignedPtr = (char*)AlignedAddr;
      __msan_allocated_memory(AlignedPtr, Size);
      return AlignedPtr;
    }

    // Otherwise, start a new slab and try again.
    StartNewSlab();
    uintptr_t AlignedAddr = alignAddr(CurPtr, Alignment);
    assert(AlignedAddr + Size <= (uintptr_t)End &&
           "Unable to allocate memory!");
    char *AlignedPtr = (char*)AlignedAddr;
    CurPtr = AlignedPtr + Size;
    __msan_allocated_memory(AlignedPtr, Size);
    return AlignedPtr;
  }

  // Pull in base class overloads.
  using AllocatorBase<BumpPtrAllocatorImpl>::Allocate;

  void Deallocate(const void * /*Ptr*/, size_t /*Size*/) {}

  // Pull in base class overloads.
  using AllocatorBase<BumpPtrAllocatorImpl>::Deallocate;

  size_t GetNumSlabs() const { return Slabs.size() + CustomSizedSlabs.size(); }

  size_t getTotalMemory() const {
    size_t TotalMemory = 0;
    for (auto I = Slabs.begin(), E = Slabs.end(); I != E; ++I)
      TotalMemory += computeSlabSize(std::distance(Slabs.begin(), I));
    for (auto &PtrAndSize : CustomSizedSlabs)
      TotalMemory += PtrAndSize.second;
    return TotalMemory;
  }

  void PrintStats() const {
    detail::printBumpPtrAllocatorStats(Slabs.size(), BytesAllocated,
                                       getTotalMemory());
  }

private:
  /// \brief The current pointer into the current slab.
  ///
  /// This points to the next free byte in the slab.
  char *CurPtr;

  /// \brief The end of the current slab.
  char *End;

  /// \brief The slabs allocated so far.
  SmallVector<void *, 4> Slabs;

  /// \brief Custom-sized slabs allocated for too-large allocation requests.
  SmallVector<std::pair<void *, size_t>, 0> CustomSizedSlabs;

  /// \brief How many bytes we've allocated.
  ///
  /// Used so that we can compute how much space was wasted.
  size_t BytesAllocated;

  /// \brief The allocator instance we use to get slabs of memory.
  AllocatorT Allocator;

  static size_t computeSlabSize(unsigned SlabIdx) {
    // Scale the actual allocated slab size based on the number of slabs
    // allocated. Every 128 slabs allocated, we double the allocated size to
    // reduce allocation frequency, but saturate at multiplying the slab size by
    // 2^30.
    return SlabSize * ((size_t)1 << std::min<size_t>(30, SlabIdx / 128));
  }

  /// \brief Allocate a new slab and move the bump pointers over into the new
  /// slab, modifying CurPtr and End.
  void StartNewSlab() {
    size_t AllocatedSlabSize = computeSlabSize(Slabs.size());

    Slabs.reserve(Slabs.size() + 1); // HLSL Change: Prevent leak on push_back exception
    void *NewSlab = Allocator.Allocate(AllocatedSlabSize, 0);
    Slabs.push_back(NewSlab);
    CurPtr = (char *)(NewSlab);
    End = ((char *)NewSlab) + AllocatedSlabSize;
  }

  /// \brief Deallocate a sequence of slabs.
  void DeallocateSlabs(SmallVectorImpl<void *>::iterator I,
                       SmallVectorImpl<void *>::iterator E) {
    for (; I != E; ++I) {
      size_t AllocatedSlabSize =
          computeSlabSize(std::distance(Slabs.begin(), I));
      Allocator.Deallocate(*I, AllocatedSlabSize);
    }
  }

  /// \brief Deallocate all memory for custom sized slabs.
  void DeallocateCustomSizedSlabs() {
    for (auto &PtrAndSize : CustomSizedSlabs) {
      void *Ptr = PtrAndSize.first;
      size_t Size = PtrAndSize.second;
      Allocator.Deallocate(Ptr, Size);
    }
  }

  template <typename T> friend class SpecificBumpPtrAllocator;
};

/// \brief The standard BumpPtrAllocator which just uses the default template
/// paramaters.
typedef BumpPtrAllocatorImpl<> BumpPtrAllocator;

/// \brief A BumpPtrAllocator that allows only elements of a specific type to be
/// allocated.
///
/// This allows calling the destructor in DestroyAll() and when the allocator is
/// destroyed.
template <typename T> class SpecificBumpPtrAllocator {
  BumpPtrAllocator Allocator;

public:
  SpecificBumpPtrAllocator() : Allocator() {}
  SpecificBumpPtrAllocator(SpecificBumpPtrAllocator &&Old)
      : Allocator(std::move(Old.Allocator)) {}
  ~SpecificBumpPtrAllocator() { DestroyAll(); }

  SpecificBumpPtrAllocator &operator=(SpecificBumpPtrAllocator &&RHS) {
    Allocator = std::move(RHS.Allocator);
    return *this;
  }

  /// Call the destructor of each allocated object and deallocate all but the
  /// current slab and reset the current pointer to the beginning of it, freeing
  /// all memory allocated so far.
  void DestroyAll() {
    auto DestroyElements = [](char *Begin, char *End) {
      assert(Begin == (char*)alignAddr(Begin, alignOf<T>()));
      for (char *Ptr = Begin; Ptr + sizeof(T) <= End; Ptr += sizeof(T))
        reinterpret_cast<T *>(Ptr)->~T();
    };

    for (auto I = Allocator.Slabs.begin(), E = Allocator.Slabs.end(); I != E;
         ++I) {
      size_t AllocatedSlabSize = BumpPtrAllocator::computeSlabSize(
          std::distance(Allocator.Slabs.begin(), I));
      char *Begin = (char*)alignAddr(*I, alignOf<T>());
      char *End = *I == Allocator.Slabs.back() ? Allocator.CurPtr
                                               : (char *)*I + AllocatedSlabSize;

      DestroyElements(Begin, End);
    }

    for (auto &PtrAndSize : Allocator.CustomSizedSlabs) {
      void *Ptr = PtrAndSize.first;
      size_t Size = PtrAndSize.second;
      DestroyElements((char*)alignAddr(Ptr, alignOf<T>()), (char *)Ptr + Size);
    }

    Allocator.Reset();
  }

  /// \brief Allocate space for an array of objects without constructing them.
  T *Allocate(size_t num = 1) { return Allocator.Allocate<T>(num); }
};

}  // end namespace llvm

// HLSL Change Starts - undef min due to conflict with std::min
#ifdef min
#undef min
#endif
// HLSL Change Ends

template <typename AllocatorT, size_t SlabSize, size_t SizeThreshold>
void *operator new(size_t Size,
                   llvm::BumpPtrAllocatorImpl<AllocatorT, SlabSize,
                                              SizeThreshold> &Allocator) {
  struct S {
    char c;
    union {
      double D;
      long double LD;
      long long L;
      void *P;
    } x;
  };
  return Allocator.Allocate(
      Size, std::min((size_t)llvm::NextPowerOf2(Size), offsetof(S, x)));
}

template <typename AllocatorT, size_t SlabSize, size_t SizeThreshold>
void operator delete(
    void *, llvm::BumpPtrAllocatorImpl<AllocatorT, SlabSize, SizeThreshold> &) {
}

#endif // LLVM_SUPPORT_ALLOCATOR_H
