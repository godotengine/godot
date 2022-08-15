//===- BlockFrequencyInfo.h - Block Frequency Analysis ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFO_H

#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include <climits>

namespace llvm {

class BranchProbabilityInfo;
template <class BlockT> class BlockFrequencyInfoImpl;

/// BlockFrequencyInfo pass uses BlockFrequencyInfoImpl implementation to
/// estimate IR basic block frequencies.
class BlockFrequencyInfo : public FunctionPass {
  typedef BlockFrequencyInfoImpl<BasicBlock> ImplType;
  std::unique_ptr<ImplType> BFI;

public:
  static char ID;

  BlockFrequencyInfo();

  ~BlockFrequencyInfo() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
  void releaseMemory() override;
  void print(raw_ostream &O, const Module *M) const override;
  const Function *getFunction() const;
  void view() const;

  /// getblockFreq - Return block frequency. Return 0 if we don't have the
  /// information. Please note that initial frequency is equal to ENTRY_FREQ. It
  /// means that we should not rely on the value itself, but only on the
  /// comparison to the other block frequencies. We do this to avoid using of
  /// floating points.
  BlockFrequency getBlockFreq(const BasicBlock *BB) const;

  // Print the block frequency Freq to OS using the current functions entry
  // frequency to convert freq into a relative decimal form.
  raw_ostream &printBlockFreq(raw_ostream &OS, const BlockFrequency Freq) const;

  // Convenience method that attempts to look up the frequency associated with
  // BB and print it to OS.
  raw_ostream &printBlockFreq(raw_ostream &OS, const BasicBlock *BB) const;

  uint64_t getEntryFreq() const;

};

}

#endif
