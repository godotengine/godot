//===- IntervalPartition.h - Interval partition Calculation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the IntervalPartition class, which
// calculates and represents the interval partition of a function, or a
// preexisting interval partition.
//
// In this way, the interval partition may be used to reduce a flow graph down
// to its degenerate single node interval partition (unless it is irreducible).
//
// TODO: The IntervalPartition class should take a bool parameter that tells
// whether it should add the "tails" of an interval to an interval itself or if
// they should be represented as distinct intervals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INTERVALPARTITION_H
#define LLVM_ANALYSIS_INTERVALPARTITION_H

#include "llvm/Analysis/Interval.h"
#include "llvm/Pass.h"
#include <map>

namespace llvm {
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//
// IntervalPartition - This class builds and holds an "interval partition" for
// a function.  This partition divides the control flow graph into a set of
// maximal intervals, as defined with the properties above.  Intuitively, an
// interval is a (possibly nonexistent) loop with a "tail" of non-looping
// nodes following it.
//
class IntervalPartition : public FunctionPass {
  typedef std::map<BasicBlock*, Interval*> IntervalMapTy;
  IntervalMapTy IntervalMap;

  typedef std::vector<Interval*> IntervalListTy;
  Interval *RootInterval;
  std::vector<Interval*> Intervals;

public:
  static char ID; // Pass identification, replacement for typeid

  IntervalPartition() : FunctionPass(ID), RootInterval(nullptr) {
    initializeIntervalPartitionPass(*PassRegistry::getPassRegistry());
  }

  // run - Calculate the interval partition for this function
  bool runOnFunction(Function &F) override;

  // IntervalPartition ctor - Build a reduced interval partition from an
  // existing interval graph.  This takes an additional boolean parameter to
  // distinguish it from a copy constructor.  Always pass in false for now.
  //
  IntervalPartition(IntervalPartition &I, bool);

  // print - Show contents in human readable format...
  void print(raw_ostream &O, const Module* = nullptr) const override;

  // getRootInterval() - Return the root interval that contains the starting
  // block of the function.
  inline Interval *getRootInterval() { return RootInterval; }

  // isDegeneratePartition() - Returns true if the interval partition contains
  // a single interval, and thus cannot be simplified anymore.
  bool isDegeneratePartition() { return Intervals.size() == 1; }

  // TODO: isIrreducible - look for triangle graph.

  // getBlockInterval - Return the interval that a basic block exists in.
  inline Interval *getBlockInterval(BasicBlock *BB) {
    IntervalMapTy::iterator I = IntervalMap.find(BB);
    return I != IntervalMap.end() ? I->second : nullptr;
  }

  // getAnalysisUsage - Implement the Pass API
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  // Interface to Intervals vector...
  const std::vector<Interval*> &getIntervals() const { return Intervals; }

  // releaseMemory - Reset state back to before function was analyzed
  void releaseMemory() override;

private:
  // addIntervalToPartition - Add an interval to the internal list of intervals,
  // and then add mappings from all of the basic blocks in the interval to the
  // interval itself (in the IntervalMap).
  //
  void addIntervalToPartition(Interval *I);

  // updatePredecessors - Interval generation only sets the successor fields of
  // the interval data structures.  After interval generation is complete,
  // run through all of the intervals and propagate successor info as
  // predecessor info.
  //
  void updatePredecessors(Interval *Int);
};

} // End llvm namespace

#endif
