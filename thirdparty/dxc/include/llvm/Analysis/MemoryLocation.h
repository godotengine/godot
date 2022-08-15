//===- MemoryLocation.h - Memory location descriptions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides utility analysis objects describing memory locations.
/// These are used both by the Alias Analysis infrastructure and more
/// specialized memory analysis layers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMORYLOCATION_H
#define LLVM_ANALYSIS_MEMORYLOCATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Metadata.h"

namespace llvm {

class LoadInst;
class StoreInst;
class MemTransferInst;
class MemIntrinsic;
class TargetLibraryInfo;

/// Representation for a specific memory location.
///
/// This abstraction can be used to represent a specific location in memory.
/// The goal of the location is to represent enough information to describe
/// abstract aliasing, modification, and reference behaviors of whatever
/// value(s) are stored in memory at the particular location.
///
/// The primary user of this interface is LLVM's Alias Analysis, but other
/// memory analyses such as MemoryDependence can use it as well.
class MemoryLocation {
public:
  /// UnknownSize - This is a special value which can be used with the
  /// size arguments in alias queries to indicate that the caller does not
  /// know the sizes of the potential memory references.
  enum : uint64_t { UnknownSize = ~UINT64_C(0) };

  /// The address of the start of the location.
  const Value *Ptr;

  /// The maximum size of the location, in address-units, or
  /// UnknownSize if the size is not known.
  ///
  /// Note that an unknown size does not mean the pointer aliases the entire
  /// virtual address space, because there are restrictions on stepping out of
  /// one object and into another. See
  /// http://llvm.org/docs/LangRef.html#pointeraliasing
  uint64_t Size;

  /// The metadata nodes which describes the aliasing of the location (each
  /// member is null if that kind of information is unavailable).
  AAMDNodes AATags;

  /// Return a location with information about the memory reference by the given
  /// instruction.
  static MemoryLocation get(const LoadInst *LI);
  static MemoryLocation get(const StoreInst *SI);
  static MemoryLocation get(const VAArgInst *VI);
  static MemoryLocation get(const AtomicCmpXchgInst *CXI);
  static MemoryLocation get(const AtomicRMWInst *RMWI);
  static MemoryLocation get(const Instruction *Inst) {
    if (auto *I = dyn_cast<LoadInst>(Inst))
      return get(I);
    else if (auto *I = dyn_cast<StoreInst>(Inst))
      return get(I);
    else if (auto *I = dyn_cast<VAArgInst>(Inst))
      return get(I);
    else if (auto *I = dyn_cast<AtomicCmpXchgInst>(Inst))
      return get(I);
    else if (auto *I = dyn_cast<AtomicRMWInst>(Inst))
      return get(I);
    llvm_unreachable("unsupported memory instruction");
  }

  /// Return a location representing the source of a memory transfer.
  static MemoryLocation getForSource(const MemTransferInst *MTI);

  /// Return a location representing the destination of a memory set or
  /// transfer.
  static MemoryLocation getForDest(const MemIntrinsic *MI);

  /// Return a location representing a particular argument of a call.
  static MemoryLocation getForArgument(ImmutableCallSite CS, unsigned ArgIdx,
                                       const TargetLibraryInfo &TLI);

  explicit MemoryLocation(const Value *Ptr = nullptr,
                          uint64_t Size = UnknownSize,
                          const AAMDNodes &AATags = AAMDNodes())
      : Ptr(Ptr), Size(Size), AATags(AATags) {}

  MemoryLocation getWithNewPtr(const Value *NewPtr) const {
    MemoryLocation Copy(*this);
    Copy.Ptr = NewPtr;
    return Copy;
  }

  MemoryLocation getWithNewSize(uint64_t NewSize) const {
    MemoryLocation Copy(*this);
    Copy.Size = NewSize;
    return Copy;
  }

  MemoryLocation getWithoutAATags() const {
    MemoryLocation Copy(*this);
    Copy.AATags = AAMDNodes();
    return Copy;
  }

  bool operator==(const MemoryLocation &Other) const {
    return Ptr == Other.Ptr && Size == Other.Size && AATags == Other.AATags;
  }
};

// Specialize DenseMapInfo for MemoryLocation.
template <> struct DenseMapInfo<MemoryLocation> {
  static inline MemoryLocation getEmptyKey() {
    return MemoryLocation(DenseMapInfo<const Value *>::getEmptyKey(), 0);
  }
  static inline MemoryLocation getTombstoneKey() {
    return MemoryLocation(DenseMapInfo<const Value *>::getTombstoneKey(), 0);
  }
  static unsigned getHashValue(const MemoryLocation &Val) {
    return DenseMapInfo<const Value *>::getHashValue(Val.Ptr) ^
           DenseMapInfo<uint64_t>::getHashValue(Val.Size) ^
           DenseMapInfo<AAMDNodes>::getHashValue(Val.AATags);
  }
  static bool isEqual(const MemoryLocation &LHS, const MemoryLocation &RHS) {
    return LHS == RHS;
  }
};
}

#endif
