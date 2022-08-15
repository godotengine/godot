//===- CodeMetrics.cpp - Code cost measurements ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements code cost measurement utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "code-metrics"

using namespace llvm;

static void completeEphemeralValues(SmallVector<const Value *, 16> &WorkSet,
                                    SmallPtrSetImpl<const Value*> &EphValues) {
  SmallPtrSet<const Value *, 32> Visited;

  // Make sure that all of the items in WorkSet are in our EphValues set.
  EphValues.insert(WorkSet.begin(), WorkSet.end());

  // Note: We don't speculate PHIs here, so we'll miss instruction chains kept
  // alive only by ephemeral values.

  while (!WorkSet.empty()) {
    const Value *V = WorkSet.front();
    WorkSet.erase(WorkSet.begin());

    if (!Visited.insert(V).second)
      continue;

    // If all uses of this value are ephemeral, then so is this value.
    bool FoundNEUse = false;
    for (const User *I : V->users())
      if (!EphValues.count(I)) {
        FoundNEUse = true;
        break;
      }

    if (FoundNEUse)
      continue;

    EphValues.insert(V);
    DEBUG(dbgs() << "Ephemeral Value: " << *V << "\n");

    if (const User *U = dyn_cast<User>(V))
      for (const Value *J : U->operands()) {
        if (isSafeToSpeculativelyExecute(J))
          WorkSet.push_back(J);
      }
  }
}

// Find all ephemeral values.
void CodeMetrics::collectEphemeralValues(
    const Loop *L, AssumptionCache *AC,
    SmallPtrSetImpl<const Value *> &EphValues) {
  SmallVector<const Value *, 16> WorkSet;

  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    Instruction *I = cast<Instruction>(AssumeVH);

    // Filter out call sites outside of the loop so we don't to a function's
    // worth of work for each of its loops (and, in the common case, ephemeral
    // values in the loop are likely due to @llvm.assume calls in the loop).
    if (!L->contains(I->getParent()))
      continue;

    WorkSet.push_back(I);
  }

  completeEphemeralValues(WorkSet, EphValues);
}

void CodeMetrics::collectEphemeralValues(
    const Function *F, AssumptionCache *AC,
    SmallPtrSetImpl<const Value *> &EphValues) {
  SmallVector<const Value *, 16> WorkSet;

  for (auto &AssumeVH : AC->assumptions()) {
    if (!AssumeVH)
      continue;
    Instruction *I = cast<Instruction>(AssumeVH);
    assert(I->getParent()->getParent() == F &&
           "Found assumption for the wrong function!");
    WorkSet.push_back(I);
  }

  completeEphemeralValues(WorkSet, EphValues);
}

/// analyzeBasicBlock - Fill in the current structure with information gleaned
/// from the specified block.
void CodeMetrics::analyzeBasicBlock(const BasicBlock *BB,
                                    const TargetTransformInfo &TTI,
                                    SmallPtrSetImpl<const Value*> &EphValues) {
  ++NumBlocks;
  unsigned NumInstsBeforeThisBB = NumInsts;
  for (BasicBlock::const_iterator II = BB->begin(), E = BB->end();
       II != E; ++II) {
    // Skip ephemeral values.
    if (EphValues.count(II))
      continue;

    // Special handling for calls.
    if (isa<CallInst>(II) || isa<InvokeInst>(II)) {
      ImmutableCallSite CS(cast<Instruction>(II));

      if (const Function *F = CS.getCalledFunction()) {
        // If a function is both internal and has a single use, then it is
        // extremely likely to get inlined in the future (it was probably
        // exposed by an interleaved devirtualization pass).
        if (!CS.isNoInline() && F->hasInternalLinkage() && F->hasOneUse())
          ++NumInlineCandidates;

        // If this call is to function itself, then the function is recursive.
        // Inlining it into other functions is a bad idea, because this is
        // basically just a form of loop peeling, and our metrics aren't useful
        // for that case.
        if (F == BB->getParent())
          isRecursive = true;

        if (TTI.isLoweredToCall(F))
          ++NumCalls;
      } else {
        // We don't want inline asm to count as a call - that would prevent loop
        // unrolling. The argument setup cost is still real, though.
        if (!isa<InlineAsm>(CS.getCalledValue()))
          ++NumCalls;
      }
    }

    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (!AI->isStaticAlloca())
        this->usesDynamicAlloca = true;
    }

    if (isa<ExtractElementInst>(II) || II->getType()->isVectorTy())
      ++NumVectorInsts;

    if (const CallInst *CI = dyn_cast<CallInst>(II))
      if (CI->cannotDuplicate())
        notDuplicatable = true;

    if (const InvokeInst *InvI = dyn_cast<InvokeInst>(II))
      if (InvI->cannotDuplicate())
        notDuplicatable = true;

    NumInsts += TTI.getUserCost(&*II);
  }

  if (isa<ReturnInst>(BB->getTerminator()))
    ++NumRets;

  // We never want to inline functions that contain an indirectbr.  This is
  // incorrect because all the blockaddress's (in static global initializers
  // for example) would be referring to the original function, and this indirect
  // jump would jump from the inlined copy of the function into the original
  // function which is extremely undefined behavior.
  // FIXME: This logic isn't really right; we can safely inline functions
  // with indirectbr's as long as no other function or global references the
  // blockaddress of a block within the current function.  And as a QOI issue,
  // if someone is using a blockaddress without an indirectbr, and that
  // reference somehow ends up in another function or global, we probably
  // don't want to inline this function.
  notDuplicatable |= isa<IndirectBrInst>(BB->getTerminator());

  // Remember NumInsts for this BB.
  NumBBInsts[BB] = NumInsts - NumInstsBeforeThisBB;
}
