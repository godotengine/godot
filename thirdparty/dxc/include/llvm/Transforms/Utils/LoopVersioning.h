//===- LoopVersioning.h - Utility to version a loop -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility class to perform loop versioning.  The versioned
// loop speculates that otherwise may-aliasing memory accesses don't overlap and
// emits checks to prove this.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPVERSIONING_H
#define LLVM_TRANSFORMS_UTILS_LOOPVERSIONING_H

#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class Loop;
class LoopAccessInfo;
class LoopInfo;

/// \brief This class emits a version of the loop where run-time checks ensure
/// that may-alias pointers can't overlap.
///
/// It currently only supports single-exit loops and assumes that the loop
/// already has a preheader.
class LoopVersioning {
public:
  LoopVersioning(const LoopAccessInfo &LAI, Loop *L, LoopInfo *LI,
                 DominatorTree *DT,
                 const SmallVector<int, 8> *PtrToPartition = nullptr);

  /// \brief Returns true if we need memchecks to disambiguate may-aliasing
  /// accesses.
  bool needsRuntimeChecks() const;

  /// \brief Performs the CFG manipulation part of versioning the loop including
  /// the DominatorTree and LoopInfo updates.
  ///
  /// The loop that was used to construct the class will be the "versioned" loop
  /// i.e. the loop that will receive control if all the memchecks pass.
  ///
  /// This allows the loop transform pass to operate on the same loop regardless
  /// of whether versioning was necessary or not:
  ///
  ///    for each loop L:
  ///        analyze L
  ///        if versioning is necessary version L
  ///        transform L
  void versionLoop(Pass *P);

  /// \brief Adds the necessary PHI nodes for the versioned loops based on the
  /// loop-defined values used outside of the loop.
  ///
  /// This needs to be called after versionLoop if there are defs in the loop
  /// that are used outside the loop.  FIXME: this should be invoked internally
  /// by versionLoop and made private.
  void addPHINodes(const SmallVectorImpl<Instruction *> &DefsUsedOutside);

  /// \brief Returns the versioned loop.  Control flows here if pointers in the
  /// loop don't alias (i.e. all memchecks passed).  (This loop is actually the
  /// same as the original loop that we got constructed with.)
  Loop *getVersionedLoop() { return VersionedLoop; }

  /// \brief Returns the fall-back loop.  Control flows here if pointers in the
  /// loop may alias (i.e. one of the memchecks failed).
  Loop *getNonVersionedLoop() { return NonVersionedLoop; }

private:
  /// \brief The original loop.  This becomes the "versioned" one.  I.e.,
  /// control flows here if pointers in the loop don't alias.
  Loop *VersionedLoop;
  /// \brief The fall-back loop.  I.e. control flows here if pointers in the
  /// loop may alias (memchecks failed).
  Loop *NonVersionedLoop;

  /// \brief For each memory pointer it contains the partitionId it is used in.
  /// If nullptr, no partitioning is used.
  ///
  /// The I-th entry corresponds to I-th entry in LAI.getRuntimePointerCheck().
  /// If the pointer is used in multiple partitions the entry is set to -1.
  const SmallVector<int, 8> *PtrToPartition;

  /// \brief This maps the instructions from VersionedLoop to their counterpart
  /// in NonVersionedLoop.
  ValueToValueMapTy VMap;

  /// \brief Analyses used.
  const LoopAccessInfo &LAI;
  LoopInfo *LI;
  DominatorTree *DT;
};
}

#endif
