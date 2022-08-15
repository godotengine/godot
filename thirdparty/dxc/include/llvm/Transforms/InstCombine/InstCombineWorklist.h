//===- InstCombineWorklist.h - Worklist for InstCombine pass ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINEWORKLIST_H
#define LLVM_TRANSFORMS_INSTCOMBINE_INSTCOMBINEWORKLIST_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "instcombine"

namespace llvm {

/// InstCombineWorklist - This is the worklist management logic for
/// InstCombine.
class InstCombineWorklist {
  SmallVector<Instruction*, 256> Worklist;
  DenseMap<Instruction*, unsigned> WorklistMap;

  void operator=(const InstCombineWorklist&RHS) = delete;
  InstCombineWorklist(const InstCombineWorklist&) = delete;
public:
  InstCombineWorklist() {}

  InstCombineWorklist(InstCombineWorklist &&Arg)
      : Worklist(std::move(Arg.Worklist)),
        WorklistMap(std::move(Arg.WorklistMap)) {}
  InstCombineWorklist &operator=(InstCombineWorklist &&RHS) {
    Worklist = std::move(RHS.Worklist);
    WorklistMap = std::move(RHS.WorklistMap);
    return *this;
  }

  bool isEmpty() const { return Worklist.empty(); }

  /// Add - Add the specified instruction to the worklist if it isn't already
  /// in it.
  void Add(Instruction *I) {
    if (WorklistMap.insert(std::make_pair(I, Worklist.size())).second) {
      DEBUG(dbgs() << "IC: ADD: " << *I << '\n');
      Worklist.push_back(I);
    }
  }

  void AddValue(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      Add(I);
  }

  /// AddInitialGroup - Add the specified batch of stuff in reverse order.
  /// which should only be done when the worklist is empty and when the group
  /// has no duplicates.
  void AddInitialGroup(Instruction *const *List, unsigned NumEntries) {
    assert(Worklist.empty() && "Worklist must be empty to add initial group");
    Worklist.reserve(NumEntries+16);
    WorklistMap.resize(NumEntries);
    DEBUG(dbgs() << "IC: ADDING: " << NumEntries << " instrs to worklist\n");
    for (unsigned Idx = 0; NumEntries; --NumEntries) {
      Instruction *I = List[NumEntries-1];
      WorklistMap.insert(std::make_pair(I, Idx++));
      Worklist.push_back(I);
    }
  }

  // Remove - remove I from the worklist if it exists.
  void Remove(Instruction *I) {
    DenseMap<Instruction*, unsigned>::iterator It = WorklistMap.find(I);
    if (It == WorklistMap.end()) return; // Not in worklist.

    // Don't bother moving everything down, just null out the slot.
    Worklist[It->second] = nullptr;

    WorklistMap.erase(It);
  }

  Instruction *RemoveOne() {
    Instruction *I = Worklist.pop_back_val();
    WorklistMap.erase(I);
    return I;
  }

  /// AddUsersToWorkList - When an instruction is simplified, add all users of
  /// the instruction to the work lists because they might get more simplified
  /// now.
  ///
  void AddUsersToWorkList(Instruction &I) {
    for (User *U : I.users())
      Add(cast<Instruction>(U));
  }


  /// Zap - check that the worklist is empty and nuke the backing store for
  /// the map if it is large.
  void Zap() {
    assert(WorklistMap.empty() && "Worklist empty, but map not?");

    // Do an explicit clear, this shrinks the map if needed.
    WorklistMap.clear();
  }
};

} // end namespace llvm.

#undef DEBUG_TYPE

#endif
