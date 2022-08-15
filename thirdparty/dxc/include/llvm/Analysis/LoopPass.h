//===- LoopPass.h - LoopPass class ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines LoopPass class. All loop optimization
// and transformation passes are derived from LoopPass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPPASS_H
#define LLVM_ANALYSIS_LOOPPASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/LegacyPassManagers.h"
#include "llvm/Pass.h"
#include <deque>

namespace llvm {

class LPPassManager;
class Function;
class PMStack;

class LoopPass : public Pass {
public:
  explicit LoopPass(char &pid) : Pass(PT_Loop, pid) {}

  /// getPrinterPass - Get a pass to print the function corresponding
  /// to a Loop.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override;

  // runOnLoop - This method should be implemented by the subclass to perform
  // whatever action is necessary for the specified Loop.
  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) = 0;

  using llvm::Pass::doInitialization;
  using llvm::Pass::doFinalization;

  // Initialization and finalization hooks.
  virtual bool doInitialization(Loop *L, LPPassManager &LPM) {
    return false;
  }

  // Finalization hook does not supply Loop because at this time
  // loop nest is completely different.
  virtual bool doFinalization() { return false; }

  // Check if this pass is suitable for the current LPPassManager, if
  // available. This pass P is not suitable for a LPPassManager if P
  // is not preserving higher level analysis info used by other
  // LPPassManager passes. In such case, pop LPPassManager from the
  // stack. This will force assignPassManager() to create new
  // LPPassManger as expected.
  void preparePassManager(PMStack &PMS) override;

  /// Assign pass manager to manage this pass
  void assignPassManager(PMStack &PMS, PassManagerType PMT) override;

  ///  Return what kind of Pass Manager can manage this pass.
  PassManagerType getPotentialPassManagerType() const override {
    return PMT_LoopPassManager;
  }

  //===--------------------------------------------------------------------===//
  /// SimpleAnalysis - Provides simple interface to update analysis info
  /// maintained by various passes. Note, if required this interface can
  /// be extracted into a separate abstract class but it would require
  /// additional use of multiple inheritance in Pass class hierarchy, something
  /// we are trying to avoid.

  /// Each loop pass can override these simple analysis hooks to update
  /// desired analysis information.
  /// cloneBasicBlockAnalysis - Clone analysis info associated with basic block.
  virtual void cloneBasicBlockAnalysis(BasicBlock *F, BasicBlock *T, Loop *L) {}

  /// deleteAnalysisValue - Delete analysis info associated with value V.
  virtual void deleteAnalysisValue(Value *V, Loop *L) {}

  /// Delete analysis info associated with Loop L.
  /// Called to notify a Pass that a loop has been deleted and any
  /// associated analysis values can be deleted.
  virtual void deleteAnalysisLoop(Loop *L) {}

protected:
  /// skipOptnoneFunction - Containing function has Attribute::OptimizeNone
  /// and most transformation passes should skip it.
  bool skipOptnoneFunction(const Loop *L) const;
};

class LPPassManager : public FunctionPass, public PMDataManager {
public:
  static char ID;
  explicit LPPassManager();

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnFunction(Function &F) override;

  /// Pass Manager itself does not invalidate any analysis info.
  // LPPassManager needs LoopInfo.
  void getAnalysisUsage(AnalysisUsage &Info) const override;

  StringRef getPassName() const override {
    return "Loop Pass Manager";
  }

  PMDataManager *getAsPMDataManager() override { return this; }
  Pass *getAsPass() override { return this; }

  /// Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) override;

  LoopPass *getContainedPass(unsigned N) {
    assert(N < PassVector.size() && "Pass number out of range!");
    LoopPass *LP = static_cast<LoopPass *>(PassVector[N]);
    return LP;
  }

  PassManagerType getPassManagerType() const override {
    return PMT_LoopPassManager;
  }

public:
  // Delete loop from the loop queue and loop nest (LoopInfo).
  void deleteLoopFromQueue(Loop *L);

  // Insert loop into the loop queue and add it as a child of the
  // given parent.
  void insertLoop(Loop *L, Loop *ParentLoop);

  // Insert a loop into the loop queue.
  void insertLoopIntoQueue(Loop *L);

  // Reoptimize this loop. LPPassManager will re-insert this loop into the
  // queue. This allows LoopPass to change loop nest for the loop. This
  // utility may send LPPassManager into infinite loops so use caution.
  void redoLoop(Loop *L);

  //===--------------------------------------------------------------------===//
  /// SimpleAnalysis - Provides simple interface to update analysis info
  /// maintained by various passes. Note, if required this interface can
  /// be extracted into a separate abstract class but it would require
  /// additional use of multiple inheritance in Pass class hierarchy, something
  /// we are trying to avoid.

  /// cloneBasicBlockSimpleAnalysis - Invoke cloneBasicBlockAnalysis hook for
  /// all passes that implement simple analysis interface.
  void cloneBasicBlockSimpleAnalysis(BasicBlock *From, BasicBlock *To, Loop *L);

  /// deleteSimpleAnalysisValue - Invoke deleteAnalysisValue hook for all passes
  /// that implement simple analysis interface.
  void deleteSimpleAnalysisValue(Value *V, Loop *L);

  /// Invoke deleteAnalysisLoop hook for all passes that implement simple
  /// analysis interface.
  void deleteSimpleAnalysisLoop(Loop *L);

private:
  std::deque<Loop *> LQ;
  bool skipThisLoop;
  bool redoThisLoop;
  LoopInfo *LI;
  Loop *CurrentLoop;
};

} // End llvm namespace

#endif
