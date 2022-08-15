//===- LoopPass.cpp - Loop Pass and Loop Pass Manager ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LoopPass and LPPassManager. All loop optimization
// and transformation passes are derived from LoopPass. LPPassManager is
// responsible for managing LoopPasses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "loop-pass-manager"

namespace {

/// PrintLoopPass - Print a Function corresponding to a Loop.
///
class PrintLoopPass : public LoopPass {
private:
  std::string Banner;
  raw_ostream &Out;       // raw_ostream to print on.

public:
  static char ID;
  PrintLoopPass(const std::string &B, raw_ostream &o)
      : LoopPass(ID), Banner(B), Out(o) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnLoop(Loop *L, LPPassManager &) override {
    Out << Banner;
    for (Loop::block_iterator b = L->block_begin(), be = L->block_end();
         b != be;
         ++b) {
      if (*b)
        (*b)->print(Out);
      else
        Out << "Printing <null> block";
    }
    return false;
  }
};

char PrintLoopPass::ID = 0;
}

//===----------------------------------------------------------------------===//
// LPPassManager
//

char LPPassManager::ID = 0;

LPPassManager::LPPassManager()
  : FunctionPass(ID), PMDataManager() {
  skipThisLoop = false;
  redoThisLoop = false;
  LI = nullptr;
  CurrentLoop = nullptr;
}

/// Delete loop from the loop queue and loop hierarchy (LoopInfo).
void LPPassManager::deleteLoopFromQueue(Loop *L) {

  LI->updateUnloop(L);

  // Notify passes that the loop is being deleted.
  deleteSimpleAnalysisLoop(L);

  // If L is current loop then skip rest of the passes and let
  // runOnFunction remove L from LQ. Otherwise, remove L from LQ now
  // and continue applying other passes on CurrentLoop.
  if (CurrentLoop == L)
    skipThisLoop = true;

  delete L;

  if (skipThisLoop)
    return;

  for (std::deque<Loop *>::iterator I = LQ.begin(),
         E = LQ.end(); I != E; ++I) {
    if (*I == L) {
      LQ.erase(I);
      break;
    }
  }
}

// Inset loop into loop nest (LoopInfo) and loop queue (LQ).
void LPPassManager::insertLoop(Loop *L, Loop *ParentLoop) {

  assert (CurrentLoop != L && "Cannot insert CurrentLoop");

  // Insert into loop nest
  if (ParentLoop)
    ParentLoop->addChildLoop(L);
  else
    LI->addTopLevelLoop(L);

  insertLoopIntoQueue(L);
}

void LPPassManager::insertLoopIntoQueue(Loop *L) {
  // Insert L into loop queue
  if (L == CurrentLoop)
    redoLoop(L);
  else if (!L->getParentLoop())
    // This is top level loop.
    LQ.push_front(L);
  else {
    // Insert L after the parent loop.
    for (std::deque<Loop *>::iterator I = LQ.begin(),
           E = LQ.end(); I != E; ++I) {
      if (*I == L->getParentLoop()) {
        // deque does not support insert after.
        ++I;
        LQ.insert(I, 1, L);
        break;
      }
    }
  }
}

// Reoptimize this loop. LPPassManager will re-insert this loop into the
// queue. This allows LoopPass to change loop nest for the loop. This
// utility may send LPPassManager into infinite loops so use caution.
void LPPassManager::redoLoop(Loop *L) {
  assert (CurrentLoop == L && "Can redo only CurrentLoop");
  redoThisLoop = true;
}

/// cloneBasicBlockSimpleAnalysis - Invoke cloneBasicBlockAnalysis hook for
/// all loop passes.
void LPPassManager::cloneBasicBlockSimpleAnalysis(BasicBlock *From,
                                                  BasicBlock *To, Loop *L) {
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    LoopPass *LP = getContainedPass(Index);
    LP->cloneBasicBlockAnalysis(From, To, L);
  }
}

/// deleteSimpleAnalysisValue - Invoke deleteAnalysisValue hook for all passes.
void LPPassManager::deleteSimpleAnalysisValue(Value *V, Loop *L) {
  if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;
         ++BI) {
      Instruction &I = *BI;
      deleteSimpleAnalysisValue(&I, L);
    }
  }
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    LoopPass *LP = getContainedPass(Index);
    LP->deleteAnalysisValue(V, L);
  }
}

/// Invoke deleteAnalysisLoop hook for all passes.
void LPPassManager::deleteSimpleAnalysisLoop(Loop *L) {
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    LoopPass *LP = getContainedPass(Index);
    LP->deleteAnalysisLoop(L);
  }
}


// Recurse through all subloops and all loops  into LQ.
static void addLoopIntoQueue(Loop *L, std::deque<Loop *> &LQ) {
  LQ.push_back(L);
  for (Loop::reverse_iterator I = L->rbegin(), E = L->rend(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);
}

/// Pass Manager itself does not invalidate any analysis info.
void LPPassManager::getAnalysisUsage(AnalysisUsage &Info) const {
  // LPPassManager needs LoopInfo. In the long term LoopInfo class will
  // become part of LPPassManager.
  Info.addRequired<LoopInfoWrapperPass>();
  Info.setPreservesAll();
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the function, and if so, return true.
bool LPPassManager::runOnFunction(Function &F) {
  auto &LIWP = getAnalysis<LoopInfoWrapperPass>();
  LI = &LIWP.getLoopInfo();
  bool Changed = false;

  // Collect inherited analysis from Module level pass manager.
  populateInheritedAnalysis(TPM->activeStack);

  // Populate the loop queue in reverse program order. There is no clear need to
  // process sibling loops in either forward or reverse order. There may be some
  // advantage in deleting uses in a later loop before optimizing the
  // definitions in an earlier loop. If we find a clear reason to process in
  // forward order, then a forward variant of LoopPassManager should be created.
  //
  // Note that LoopInfo::iterator visits loops in reverse program
  // order. Here, reverse_iterator gives us a forward order, and the LoopQueue
  // reverses the order a third time by popping from the back.
  for (LoopInfo::reverse_iterator I = LI->rbegin(), E = LI->rend(); I != E; ++I)
    addLoopIntoQueue(*I, LQ);

  if (LQ.empty()) // No loops, skip calling finalizers
    return false;

  // Initialization
  for (std::deque<Loop *>::const_iterator I = LQ.begin(), E = LQ.end();
       I != E; ++I) {
    Loop *L = *I;
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      LoopPass *P = getContainedPass(Index);
      Changed |= P->doInitialization(L, *this);
    }
  }

  // Walk Loops
  while (!LQ.empty()) {

    CurrentLoop  = LQ.back();
    skipThisLoop = false;
    redoThisLoop = false;

    // Run all passes on the current Loop.
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      LoopPass *P = getContainedPass(Index);

      dumpPassInfo(P, EXECUTION_MSG, ON_LOOP_MSG,
                   CurrentLoop->getHeader()->getName());
      dumpRequiredSet(P);

      initializeAnalysisImpl(P);

      {
        PassManagerPrettyStackEntry X(P, *CurrentLoop->getHeader());
        TimeRegion PassTimer(getPassTimer(P));

        Changed |= P->runOnLoop(CurrentLoop, *this);
      }

      if (Changed)
        dumpPassInfo(P, MODIFICATION_MSG, ON_LOOP_MSG,
                     skipThisLoop ? "<deleted>" :
                                    CurrentLoop->getHeader()->getName());
      dumpPreservedSet(P);

      if (!skipThisLoop) {
        // Manually check that this loop is still healthy. This is done
        // instead of relying on LoopInfo::verifyLoop since LoopInfo
        // is a function pass and it's really expensive to verify every
        // loop in the function every time. That level of checking can be
        // enabled with the -verify-loop-info option.
        {
          TimeRegion PassTimer(getPassTimer(&LIWP));
          CurrentLoop->verifyLoop();
        }

        // Then call the regular verifyAnalysis functions.
        verifyPreservedAnalysis(P);

        F.getContext().yield();
      }

      removeNotPreservedAnalysis(P);
      recordAvailableAnalysis(P);
      removeDeadPasses(P,
                       skipThisLoop ? "<deleted>" :
                                      CurrentLoop->getHeader()->getName(),
                       ON_LOOP_MSG);

      if (skipThisLoop)
        // Do not run other passes on this loop.
        break;
    }

    // If the loop was deleted, release all the loop passes. This frees up
    // some memory, and avoids trouble with the pass manager trying to call
    // verifyAnalysis on them.
    if (skipThisLoop)
      for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
        Pass *P = getContainedPass(Index);
        freePass(P, "<deleted>", ON_LOOP_MSG);
      }

    // Pop the loop from queue after running all passes.
    LQ.pop_back();

    if (redoThisLoop)
      LQ.push_back(CurrentLoop);
  }

  // Finalization
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    LoopPass *P = getContainedPass(Index);
    Changed |= P->doFinalization();
  }

  return Changed;
}

/// Print passes managed by this manager
void LPPassManager::dumpPassStructure(unsigned Offset) {
  errs().indent(Offset*2) << "Loop Pass Manager\n";
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    Pass *P = getContainedPass(Index);
    P->dumpPassStructure(Offset + 1);
    dumpLastUses(P, Offset+1);
  }
}


//===----------------------------------------------------------------------===//
// LoopPass

Pass *LoopPass::createPrinterPass(raw_ostream &O,
                                  const std::string &Banner) const {
  return new PrintLoopPass(Banner, O);
}

// Check if this pass is suitable for the current LPPassManager, if
// available. This pass P is not suitable for a LPPassManager if P
// is not preserving higher level analysis info used by other
// LPPassManager passes. In such case, pop LPPassManager from the
// stack. This will force assignPassManager() to create new
// LPPassManger as expected.
void LoopPass::preparePassManager(PMStack &PMS) {

  // Find LPPassManager
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_LoopPassManager)
    PMS.pop();

  // If this pass is destroying high level information that is used
  // by other passes that are managed by LPM then do not insert
  // this pass in current LPM. Use new LPPassManager.
  if (PMS.top()->getPassManagerType() == PMT_LoopPassManager &&
      !PMS.top()->preserveHigherLevelAnalysis(this))
    PMS.pop();
}

/// Assign pass manager to manage this pass.
void LoopPass::assignPassManager(PMStack &PMS,
                                 PassManagerType PreferredType) {
  std::unique_ptr<LoopPass> thisPtr(this); // HLSL Change
  // Find LPPassManager
  while (!PMS.empty() &&
         PMS.top()->getPassManagerType() > PMT_LoopPassManager)
    PMS.pop();

  LPPassManager *LPPM;
  if (PMS.top()->getPassManagerType() == PMT_LoopPassManager)
    LPPM = (LPPassManager*)PMS.top();
  else {
    // Create new Loop Pass Manager if it does not exist.
    assert (!PMS.empty() && "Unable to create Loop Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Loop Pass Manager
    LPPM = new LPPassManager();
    LPPM->populateInheritedAnalysis(PMS);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(LPPM);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    Pass *P = LPPM->getAsPass();
    TPM->schedulePass(P);

    // [4] Push new manager into PMS
    PMS.push(LPPM);
  }

  thisPtr.release(); // HLSL Change
  LPPM->add(this);
}

// Containing function has Attribute::OptimizeNone and transformation
// passes should skip it.
bool LoopPass::skipOptnoneFunction(const Loop *L) const {
  const Function *F = L->getHeader()->getParent();
  if (F && F->hasFnAttribute(Attribute::OptimizeNone)) {
    // FIXME: Report this to dbgs() only once per function.
    DEBUG(dbgs() << "Skipping pass '" << getPassName()
          << "' in function " << F->getName() << "\n");
    // FIXME: Delete loop from pass manager's queue?
    return true;
  }
  return false;
}
