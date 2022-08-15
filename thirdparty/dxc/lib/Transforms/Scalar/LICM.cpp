//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs loop invariant code motion, attempting to remove as much
// code from the body of a loop as possible.  It does this by either hoisting
// code into the preheader block, or by sinking code to the exit blocks if it is
// safe.  This pass also promotes must-aliased memory locations in the loop to
// live in registers, thus hoisting and sinking "invariant" loads and stores.
//
// This pass uses alias analysis for two purposes:
//
//  1. Moving loop invariant loads and calls out of loops.  If we can determine
//     that a load or call inside of a loop never aliases anything stored to,
//     we can hoist it or sink it like any other instruction.
//  2. Scalar Promotion of Memory - If there is a store instruction inside of
//     the loop, we try to move the store to happen AFTER the loop instead of
//     inside of the loop.  This can only happen if a few conditions are true:
//       A. The pointer stored through is loop invariant
//       B. There are no stores or loads in the loop which _may_ alias the
//          pointer.  There are no calls in the loop which mod/ref the pointer.
//     If these conditions are true, we can promote the loads and stores in the
//     loop of the pointer to use a temporary alloca'd variable.  We then use
//     the SSAUpdater to construct the appropriate SSA form for the value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "licm"

STATISTIC(NumSunk      , "Number of instructions sunk out of loop");
STATISTIC(NumHoisted   , "Number of instructions hoisted out of loop");
STATISTIC(NumMovedLoads, "Number of load insts hoisted or sunk");
STATISTIC(NumMovedCalls, "Number of call insts hoisted or sunk");
STATISTIC(NumPromoted  , "Number of memory locations promoted to registers");

#if 0 // HLSL Change Starts - option pending
static cl::opt<bool>
DisablePromotion("disable-licm-promotion", cl::Hidden,
                 cl::desc("Disable memory promotion in LICM pass"));
#else
static bool DisablePromotion = false;
#endif // HLSL Change Ends

static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI);
static bool isNotUsedInLoop(const Instruction &I, const Loop *CurLoop);
static bool hoist(Instruction &I, BasicBlock *Preheader);
static bool sink(Instruction &I, const LoopInfo *LI, const DominatorTree *DT,
                 const Loop *CurLoop, AliasSetTracker *CurAST );
static bool isGuaranteedToExecute(const Instruction &Inst,
                                  const DominatorTree *DT,
                                  const Loop *CurLoop,
                                  const LICMSafetyInfo *SafetyInfo);
static bool isSafeToExecuteUnconditionally(const Instruction &Inst,
                                           const DominatorTree *DT,
                                           const TargetLibraryInfo *TLI,
                                           const Loop *CurLoop,
                                           const LICMSafetyInfo *SafetyInfo,
                                           const Instruction *CtxI = nullptr);
static bool pointerInvalidatedByLoop(Value *V, uint64_t Size,
                                     const AAMDNodes &AAInfo, 
                                     AliasSetTracker *CurAST);
static Instruction *CloneInstructionInExitBlock(const Instruction &I,
                                                BasicBlock &ExitBlock,
                                                PHINode &PN,
                                                const LoopInfo *LI);
static bool canSinkOrHoistInst(Instruction &I, AliasAnalysis *AA,
                               DominatorTree *DT, TargetLibraryInfo *TLI,
                               Loop *CurLoop, AliasSetTracker *CurAST,
                               LICMSafetyInfo *SafetyInfo);

namespace {
  struct LICM : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    LICM() : LoopPass(ID) {
      initializeLICMPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }

    using llvm::Pass::doFinalization;

    bool doFinalization() override {
      assert(LoopToAliasSetMap.empty() && "Didn't free loop alias sets");
      return false;
    }

  private:
    AliasAnalysis *AA;       // Current AliasAnalysis information
    LoopInfo      *LI;       // Current LoopInfo
    DominatorTree *DT;       // Dominator Tree for the current Loop.

    TargetLibraryInfo *TLI;  // TargetLibraryInfo for constant folding.

    // State that is updated as we process loops.
    bool Changed;            // Set to true when we change anything.
    BasicBlock *Preheader;   // The preheader block of the current loop...
    Loop *CurLoop;           // The current loop we are working on...
    AliasSetTracker *CurAST; // AliasSet information for the current loop...
    DenseMap<Loop*, AliasSetTracker*> LoopToAliasSetMap;

    /// cloneBasicBlockAnalysis - Simple Analysis hook. Clone alias set info.
    void cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To,
                                 Loop *L) override;

    /// deleteAnalysisValue - Simple Analysis hook. Delete value V from alias
    /// set.
    void deleteAnalysisValue(Value *V, Loop *L) override;

    /// Simple Analysis hook. Delete loop L from alias set map.
    void deleteAnalysisLoop(Loop *L) override;
  };
}

char LICM::ID = 0;
INITIALIZE_PASS_BEGIN(LICM, "licm", "Loop Invariant Code Motion", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(LICM, "licm", "Loop Invariant Code Motion", false, false)

Pass *llvm::createLICMPass() { return new LICM(); }

/// Hoist expressions out of the specified loop. Note, alias info for inner
/// loop is not preserved so it is not a good idea to run LICM multiple
/// times on one loop.
///
bool LICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  Changed = false;

  // Get our Loop and Alias Analysis information...
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  assert(L->isLCSSAForm(*DT) && "Loop is not in LCSSA form.");

  CurAST = new AliasSetTracker(*AA);
  // Collect Alias info from subloops.
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *InnerL = *LoopItr;
    AliasSetTracker *InnerAST = LoopToAliasSetMap[InnerL];
    assert(InnerAST && "Where is my AST?");

    // What if InnerLoop was modified by other passes ?
    CurAST->add(*InnerAST);

    // Once we've incorporated the inner loop's AST into ours, we don't need the
    // subloop's anymore.
    delete InnerAST;
    LoopToAliasSetMap.erase(InnerL);
  }

  CurLoop = L;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();

  // Loop over the body of this loop, looking for calls, invokes, and stores.
  // Because subloops have already been incorporated into AST, we skip blocks in
  // subloops.
  //
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    if (LI->getLoopFor(BB) == L)        // Ignore blocks in subloops.
      CurAST->add(*BB);                 // Incorporate the specified basic block
  }

  // Compute loop safety information.
  LICMSafetyInfo SafetyInfo;
  computeLICMSafetyInfo(&SafetyInfo, CurLoop);

  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  // Traverse the body of the loop in depth first order on the dominator tree so
  // that we are guaranteed to see definitions before we see uses.  This allows
  // us to sink instructions in one pass, without iteration.  After sinking
  // instructions, we perform another pass to hoist them out of the loop.
  //
  if (L->hasDedicatedExits())
    Changed |= sinkRegion(DT->getNode(L->getHeader()), AA, LI, DT, TLI, CurLoop,
                          CurAST, &SafetyInfo);
  if (Preheader)
    Changed |= hoistRegion(DT->getNode(L->getHeader()), AA, LI, DT, TLI,
                           CurLoop, CurAST, &SafetyInfo);

  // Now that all loop invariants have been removed from the loop, promote any
  // memory references to scalars that we can.
  if (!DisablePromotion && (Preheader || L->hasDedicatedExits())) {
    SmallVector<BasicBlock *, 8> ExitBlocks;
    SmallVector<Instruction *, 8> InsertPts;
    PredIteratorCache PIC;

    // Loop over all of the alias sets in the tracker object.
    for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
         I != E; ++I)
      Changed |= promoteLoopAccessesToScalars(*I, ExitBlocks, InsertPts, 
                                              PIC, LI, DT, CurLoop, 
                                              CurAST, &SafetyInfo);

    // Once we have promoted values across the loop body we have to recursively
    // reform LCSSA as any nested loop may now have values defined within the
    // loop used in the outer loop.
    // FIXME: This is really heavy handed. It would be a bit better to use an
    // SSAUpdater strategy during promotion that was LCSSA aware and reformed
    // it as it went.
    if (Changed)
      formLCSSARecursively(*L, *DT, LI,
                           getAnalysisIfAvailable<ScalarEvolution>());
  }

  // Check that neither this loop nor its parent have had LCSSA broken. LICM is
  // specifically moving instructions across the loop boundary and so it is
  // especially in need of sanity checking here.
  assert(L->isLCSSAForm(*DT) && "Loop not left in LCSSA form after LICM!");
  assert((!L->getParentLoop() || L->getParentLoop()->isLCSSAForm(*DT)) &&
         "Parent loop not left in LCSSA form after LICM!");

  // Clear out loops state information for the next iteration
  CurLoop = nullptr;
  Preheader = nullptr;

  // If this loop is nested inside of another one, save the alias information
  // for when we process the outer loop.
  if (L->getParentLoop())
    LoopToAliasSetMap[L] = CurAST;
  else
    delete CurAST;
  return Changed;
}

/// Walk the specified region of the CFG (defined by all blocks dominated by
/// the specified block, and that are in the current loop) in reverse depth 
/// first order w.r.t the DominatorTree.  This allows us to visit uses before
/// definitions, allowing us to sink a loop body in one pass without iteration.
///
bool llvm::sinkRegion(DomTreeNode *N, AliasAnalysis *AA, LoopInfo *LI,
                      DominatorTree *DT, TargetLibraryInfo *TLI, Loop *CurLoop,
                      AliasSetTracker *CurAST, LICMSafetyInfo *SafetyInfo) {

  // Verify inputs.
  assert(N != nullptr && AA != nullptr && LI != nullptr && 
         DT != nullptr && CurLoop != nullptr && CurAST != nullptr && 
         SafetyInfo != nullptr && "Unexpected input to sinkRegion");

  // Set changed as false.
  bool Changed = false;
  // Get basic block
  BasicBlock *BB = N->getBlock();
  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return Changed;

  // We are processing blocks in reverse dfo, so process children first.
  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |=
        sinkRegion(Children[i], AA, LI, DT, TLI, CurLoop, CurAST, SafetyInfo);
  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (inSubLoop(BB,CurLoop,LI)) return Changed;

  for (BasicBlock::iterator II = BB->end(); II != BB->begin(); ) {
    Instruction &I = *--II;

    // If the instruction is dead, we would try to sink it because it isn't used
    // in the loop, instead, just delete it.
    if (isInstructionTriviallyDead(&I, TLI)) {
      DEBUG(dbgs() << "LICM deleting dead inst: " << I << '\n');
      ++II;
      CurAST->deleteValue(&I);
      I.eraseFromParent();
      Changed = true;
      continue;
    }

    // Check to see if we can sink this instruction to the exit blocks
    // of the loop.  We can do this if the all users of the instruction are
    // outside of the loop.  In this case, it doesn't even matter if the
    // operands of the instruction are loop invariant.
    //
    if (isNotUsedInLoop(I, CurLoop) &&
        canSinkOrHoistInst(I, AA, DT, TLI, CurLoop, CurAST, SafetyInfo)) {
      ++II;
      Changed |= sink(I, LI, DT, CurLoop, CurAST);
    }
  }
  return Changed;
}

/// Walk the specified region of the CFG (defined by all blocks dominated by
/// the specified block, and that are in the current loop) in depth first
/// order w.r.t the DominatorTree.  This allows us to visit definitions before
/// uses, allowing us to hoist a loop body in one pass without iteration.
///
bool llvm::hoistRegion(DomTreeNode *N, AliasAnalysis *AA, LoopInfo *LI,
                       DominatorTree *DT, TargetLibraryInfo *TLI, Loop *CurLoop,
                       AliasSetTracker *CurAST, LICMSafetyInfo *SafetyInfo) {
  // Verify inputs.
  assert(N != nullptr && AA != nullptr && LI != nullptr && 
         DT != nullptr && CurLoop != nullptr && CurAST != nullptr && 
         SafetyInfo != nullptr && "Unexpected input to hoistRegion");
  // Set changed as false.
  bool Changed = false;
  // Get basic block
  BasicBlock *BB = N->getBlock();
  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(BB)) return Changed;
  // Only need to process the contents of this block if it is not part of a
  // subloop (which would already have been processed).
  if (!inSubLoop(BB, CurLoop, LI))
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E; ) {
      Instruction &I = *II++;
      // Try constant folding this instruction.  If all the operands are
      // constants, it is technically hoistable, but it would be better to just
      // fold it.
      if (Constant *C = ConstantFoldInstruction(
              &I, I.getModule()->getDataLayout(), TLI)) {
        DEBUG(dbgs() << "LICM folding inst: " << I << "  --> " << *C << '\n');
        CurAST->copyValue(&I, C);
        CurAST->deleteValue(&I);
        I.replaceAllUsesWith(C);
        I.eraseFromParent();
        continue;
      }

      // Try hoisting the instruction out to the preheader.  We can only do this
      // if all of the operands of the instruction are loop invariant and if it
      // is safe to hoist the instruction.
      //
      if (CurLoop->hasLoopInvariantOperands(&I) &&
          canSinkOrHoistInst(I, AA, DT, TLI, CurLoop, CurAST, SafetyInfo) &&
          isSafeToExecuteUnconditionally(I, DT, TLI, CurLoop, SafetyInfo,
                                 CurLoop->getLoopPreheader()->getTerminator()))
        Changed |= hoist(I, CurLoop->getLoopPreheader());
    }

  const std::vector<DomTreeNode*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |=
        hoistRegion(Children[i], AA, LI, DT, TLI, CurLoop, CurAST, SafetyInfo);
  return Changed;
}

/// Computes loop safety information, checks loop body & header
/// for the possiblity of may throw exception.
///
void llvm::computeLICMSafetyInfo(LICMSafetyInfo * SafetyInfo, Loop * CurLoop) {
  assert(CurLoop != nullptr && "CurLoop cant be null");
  BasicBlock *Header = CurLoop->getHeader();
  // Setting default safety values.
  SafetyInfo->MayThrow = false;
  SafetyInfo->HeaderMayThrow = false;
  // Iterate over header and compute dafety info.
  for (BasicBlock::iterator I = Header->begin(), E = Header->end();
       (I != E) && !SafetyInfo->HeaderMayThrow; ++I)
    SafetyInfo->HeaderMayThrow |= I->mayThrow();
  
  SafetyInfo->MayThrow = SafetyInfo->HeaderMayThrow;
  // Iterate over loop instructions and compute safety info. 
  for (Loop::block_iterator BB = CurLoop->block_begin(), 
       BBE = CurLoop->block_end(); (BB != BBE) && !SafetyInfo->MayThrow ; ++BB)
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end();
         (I != E) && !SafetyInfo->MayThrow; ++I)
      SafetyInfo->MayThrow |= I->mayThrow();
}

/// canSinkOrHoistInst - Return true if the hoister and sinker can handle this
/// instruction.
///
bool canSinkOrHoistInst(Instruction &I, AliasAnalysis *AA, DominatorTree *DT,
                        TargetLibraryInfo *TLI, Loop *CurLoop,
                        AliasSetTracker *CurAST, LICMSafetyInfo *SafetyInfo) {
  // Loads have extra constraints we have to verify before we can hoist them.
  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    if (!LI->isUnordered())
      return false;        // Don't hoist volatile/atomic loads!

    // Loads from constant memory are always safe to move, even if they end up
    // in the same alias set as something that ends up being modified.
    if (AA->pointsToConstantMemory(LI->getOperand(0)))
      return true;
    if (LI->getMetadata(LLVMContext::MD_invariant_load))
      return true;

    // Don't hoist loads which have may-aliased stores in loop.
    uint64_t Size = 0;
    if (LI->getType()->isSized())
      Size = AA->getTypeStoreSize(LI->getType());

    AAMDNodes AAInfo;
    LI->getAAMetadata(AAInfo);

    return !pointerInvalidatedByLoop(LI->getOperand(0), Size, AAInfo, CurAST);
  } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
    // Don't sink or hoist dbg info; it's legal, but not useful.
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    // Handle simple cases by querying alias analysis.
    AliasAnalysis::ModRefBehavior Behavior = AA->getModRefBehavior(CI);
    if (Behavior == AliasAnalysis::DoesNotAccessMemory)
      return true;
    if (AliasAnalysis::onlyReadsMemory(Behavior)) {
      // If this call only reads from memory and there are no writes to memory
      // in the loop, we can hoist or sink the call as appropriate.
      bool FoundMod = false;
      for (AliasSetTracker::iterator I = CurAST->begin(), E = CurAST->end();
           I != E; ++I) {
        AliasSet &AS = *I;
        if (!AS.isForwardingAliasSet() && AS.isMod()) {
          FoundMod = true;
          break;
        }
      }
      if (!FoundMod) return true;
    }

    // FIXME: This should use mod/ref information to see if we can hoist or
    // sink the call.

    return false;
  }

  // Only these instructions are hoistable/sinkable.
  if (!isa<BinaryOperator>(I) && !isa<CastInst>(I) && !isa<SelectInst>(I) &&
      !isa<GetElementPtrInst>(I) && !isa<CmpInst>(I) &&
      !isa<InsertElementInst>(I) && !isa<ExtractElementInst>(I) &&
      !isa<ShuffleVectorInst>(I) && !isa<ExtractValueInst>(I) &&
      !isa<InsertValueInst>(I))
    return false;

  // TODO: Plumb the context instruction through to make hoisting and sinking
  // more powerful. Hoisting of loads already works due to the special casing
  // above. 
  return isSafeToExecuteUnconditionally(I, DT, TLI, CurLoop, SafetyInfo,
                                        nullptr);
}

/// Returns true if a PHINode is a trivially replaceable with an
/// Instruction.
/// This is true when all incoming values are that instruction.
/// This pattern occurs most often with LCSSA PHI nodes.
///
static bool isTriviallyReplacablePHI(const PHINode &PN, const Instruction &I) {
  for (const Value *IncValue : PN.incoming_values())
    if (IncValue != &I)
      return false;

  return true;
}

/// Return true if the only users of this instruction are outside of
/// the loop. If this is true, we can sink the instruction to the exit
/// blocks of the loop.
///
static bool isNotUsedInLoop(const Instruction &I, const Loop *CurLoop) {
  for (const User *U : I.users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (const PHINode *PN = dyn_cast<PHINode>(UI)) {
      // A PHI node where all of the incoming values are this instruction are
      // special -- they can just be RAUW'ed with the instruction and thus
      // don't require a use in the predecessor. This is a particular important
      // special case because it is the pattern found in LCSSA form.
      if (isTriviallyReplacablePHI(*PN, I)) {
        if (CurLoop->contains(PN))
          return false;
        else
          continue;
      }

      // Otherwise, PHI node uses occur in predecessor blocks if the incoming
      // values. Check for such a use being inside the loop.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == &I)
          if (CurLoop->contains(PN->getIncomingBlock(i)))
            return false;

      continue;
    }

    if (CurLoop->contains(UI))
      return false;
  }
  return true;
}

static Instruction *CloneInstructionInExitBlock(const Instruction &I,
                                                BasicBlock &ExitBlock,
                                                PHINode &PN,
                                                const LoopInfo *LI) {
  Instruction *New = I.clone();
  ExitBlock.getInstList().insert(ExitBlock.getFirstInsertionPt(), New);
  if (!I.getName().empty()) New->setName(I.getName() + ".le");

  // Build LCSSA PHI nodes for any in-loop operands. Note that this is
  // particularly cheap because we can rip off the PHI node that we're
  // replacing for the number and blocks of the predecessors.
  // OPT: If this shows up in a profile, we can instead finish sinking all
  // invariant instructions, and then walk their operands to re-establish
  // LCSSA. That will eliminate creating PHI nodes just to nuke them when
  // sinking bottom-up.
  for (User::op_iterator OI = New->op_begin(), OE = New->op_end(); OI != OE;
       ++OI)
    if (Instruction *OInst = dyn_cast<Instruction>(*OI))
      if (Loop *OLoop = LI->getLoopFor(OInst->getParent()))
        if (!OLoop->contains(&PN)) {
          PHINode *OpPN =
              PHINode::Create(OInst->getType(), PN.getNumIncomingValues(),
                              OInst->getName() + ".lcssa", ExitBlock.begin());
          for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
            OpPN->addIncoming(OInst, PN.getIncomingBlock(i));
          *OI = OpPN;
        }
  return New;
}

/// When an instruction is found to only be used outside of the loop, this
/// function moves it to the exit blocks and patches up SSA form as needed.
/// This method is guaranteed to remove the original instruction from its
/// position, and may either delete it or move it to outside of the loop.
///
static bool sink(Instruction &I, const LoopInfo *LI, const DominatorTree *DT,
                 const Loop *CurLoop, AliasSetTracker *CurAST ) {
  DEBUG(dbgs() << "LICM sinking instruction: " << I << "\n");
  bool Changed = false;
  if (isa<LoadInst>(I)) ++NumMovedLoads;
  else if (isa<CallInst>(I)) ++NumMovedCalls;
  ++NumSunk;
  Changed = true;

#ifndef NDEBUG
  SmallVector<BasicBlock *, 32> ExitBlocks;
  CurLoop->getUniqueExitBlocks(ExitBlocks);
  SmallPtrSet<BasicBlock *, 32> ExitBlockSet(ExitBlocks.begin(), 
                                             ExitBlocks.end());
#endif

  // Clones of this instruction. Don't create more than one per exit block!
  SmallDenseMap<BasicBlock *, Instruction *, 32> SunkCopies;

  // If this instruction is only used outside of the loop, then all users are
  // PHI nodes in exit blocks due to LCSSA form. Just RAUW them with clones of
  // the instruction.
  while (!I.use_empty()) {
    Value::user_iterator UI = I.user_begin();
    auto *User = cast<Instruction>(*UI);
    if (!DT->isReachableFromEntry(User->getParent())) {
      User->replaceUsesOfWith(&I, UndefValue::get(I.getType()));
      continue;
    }
    // The user must be a PHI node.
    PHINode *PN = cast<PHINode>(User);

    // Surprisingly, instructions can be used outside of loops without any
    // exits.  This can only happen in PHI nodes if the incoming block is
    // unreachable.
    Use &U = UI.getUse();
    BasicBlock *BB = PN->getIncomingBlock(U);
    if (!DT->isReachableFromEntry(BB)) {
      U = UndefValue::get(I.getType());
      continue;
    }

    BasicBlock *ExitBlock = PN->getParent();
    assert(ExitBlockSet.count(ExitBlock) &&
           "The LCSSA PHI is not in an exit block!");

    Instruction *New;
    auto It = SunkCopies.find(ExitBlock);
    if (It != SunkCopies.end())
      New = It->second;
    else
      New = SunkCopies[ExitBlock] =
            CloneInstructionInExitBlock(I, *ExitBlock, *PN, LI);

    PN->replaceAllUsesWith(New);
    PN->eraseFromParent();
  }

  CurAST->deleteValue(&I);
  I.eraseFromParent();
  return Changed;
}

/// When an instruction is found to only use loop invariant operands that
/// is safe to hoist, this instruction is called to do the dirty work.
///
static bool hoist(Instruction &I, BasicBlock *Preheader) {
  DEBUG(dbgs() << "LICM hoisting to " << Preheader->getName() << ": "
        << I << "\n");
  // Move the new node to the Preheader, before its terminator.
  I.moveBefore(Preheader->getTerminator());

  if (isa<LoadInst>(I)) ++NumMovedLoads;
  else if (isa<CallInst>(I)) ++NumMovedCalls;
  ++NumHoisted;
  return true;
}

/// Only sink or hoist an instruction if it is not a trapping instruction,
/// or if the instruction is known not to trap when moved to the preheader.
/// or if it is a trapping instruction and is guaranteed to execute.
static bool isSafeToExecuteUnconditionally(const Instruction &Inst, 
                                           const DominatorTree *DT,
                                           const TargetLibraryInfo *TLI,
                                           const Loop *CurLoop,
                                           const LICMSafetyInfo *SafetyInfo,
                                           const Instruction *CtxI) {
  if (isSafeToSpeculativelyExecute(&Inst, CtxI, DT, TLI))
    return true;

  return isGuaranteedToExecute(Inst, DT, CurLoop, SafetyInfo);
}

static bool isGuaranteedToExecute(const Instruction &Inst,
                                  const DominatorTree *DT,
                                  const Loop *CurLoop,
                                  const LICMSafetyInfo * SafetyInfo) {

  // We have to check to make sure that the instruction dominates all
  // of the exit blocks.  If it doesn't, then there is a path out of the loop
  // which does not execute this instruction, so we can't hoist it.

  // If the instruction is in the header block for the loop (which is very
  // common), it is always guaranteed to dominate the exit blocks.  Since this
  // is a common case, and can save some work, check it now.
  if (Inst.getParent() == CurLoop->getHeader())
    // If there's a throw in the header block, we can't guarantee we'll reach
    // Inst.
    return !SafetyInfo->HeaderMayThrow;

  // Somewhere in this loop there is an instruction which may throw and make us
  // exit the loop.
  if (SafetyInfo->MayThrow)
    return false;

  // Get the exit blocks for the current loop.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  CurLoop->getExitBlocks(ExitBlocks);

  // Verify that the block dominates each of the exit blocks of the loop.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (!DT->dominates(Inst.getParent(), ExitBlocks[i]))
      return false;

  // As a degenerate case, if the loop is statically infinite then we haven't
  // proven anything since there are no exit blocks.
  if (ExitBlocks.empty())
    return false;

  return true;
}

namespace {
  class LoopPromoter : public LoadAndStorePromoter {
    Value *SomePtr;  // Designated pointer to store to.
    SmallPtrSetImpl<Value*> &PointerMustAliases;
    SmallVectorImpl<BasicBlock*> &LoopExitBlocks;
    SmallVectorImpl<Instruction*> &LoopInsertPts;
    PredIteratorCache &PredCache;
    AliasSetTracker &AST;
    LoopInfo &LI;
    DebugLoc DL;
    int Alignment;
    AAMDNodes AATags;

    Value *maybeInsertLCSSAPHI(Value *V, BasicBlock *BB) const {
      if (Instruction *I = dyn_cast<Instruction>(V))
        if (Loop *L = LI.getLoopFor(I->getParent()))
          if (!L->contains(BB)) {
            // We need to create an LCSSA PHI node for the incoming value and
            // store that.
            PHINode *PN = PHINode::Create(
                I->getType(), PredCache.size(BB),
                I->getName() + ".lcssa", BB->begin());
            for (BasicBlock *Pred : PredCache.get(BB))
              PN->addIncoming(I, Pred);
            return PN;
          }
      return V;
    }

  public:
    LoopPromoter(Value *SP,
                 ArrayRef<const Instruction *> Insts,
                 SSAUpdater &S, SmallPtrSetImpl<Value *> &PMA,
                 SmallVectorImpl<BasicBlock *> &LEB,
                 SmallVectorImpl<Instruction *> &LIP, PredIteratorCache &PIC,
                 AliasSetTracker &ast, LoopInfo &li, DebugLoc dl, int alignment,
                 const AAMDNodes &AATags)
        : LoadAndStorePromoter(Insts, S), SomePtr(SP), PointerMustAliases(PMA),
          LoopExitBlocks(LEB), LoopInsertPts(LIP), PredCache(PIC), AST(ast),
          LI(li), DL(dl), Alignment(alignment), AATags(AATags) {}

    bool isInstInList(Instruction *I,
                      const SmallVectorImpl<Instruction*> &) const override {
      Value *Ptr;
      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        Ptr = LI->getOperand(0);
      else
        Ptr = cast<StoreInst>(I)->getPointerOperand();
      return PointerMustAliases.count(Ptr);
    }

    void doExtraRewritesBeforeFinalDeletion() const override {
      // Insert stores after in the loop exit blocks.  Each exit block gets a
      // store of the live-out values that feed them.  Since we've already told
      // the SSA updater about the defs in the loop and the preheader
      // definition, it is all set and we can start using it.
      for (unsigned i = 0, e = LoopExitBlocks.size(); i != e; ++i) {
        BasicBlock *ExitBlock = LoopExitBlocks[i];
        Value *LiveInValue = SSA.GetValueInMiddleOfBlock(ExitBlock);
        LiveInValue = maybeInsertLCSSAPHI(LiveInValue, ExitBlock);
        Value *Ptr = maybeInsertLCSSAPHI(SomePtr, ExitBlock);
        Instruction *InsertPos = LoopInsertPts[i];
        StoreInst *NewSI = new StoreInst(LiveInValue, Ptr, InsertPos);
        NewSI->setAlignment(Alignment);
        NewSI->setDebugLoc(DL);
        if (AATags) NewSI->setAAMetadata(AATags);
      }
    }

    void replaceLoadWithValue(LoadInst *LI, Value *V) const override {
      // Update alias analysis.
      AST.copyValue(LI, V);
    }
    void instructionDeleted(Instruction *I) const override {
      AST.deleteValue(I);
    }
  };
} // end anon namespace

/// Try to promote memory values to scalars by sinking stores out of the
/// loop and moving loads to before the loop.  We do this by looping over
/// the stores in the loop, looking for stores to Must pointers which are
/// loop invariant.
///
bool llvm::promoteLoopAccessesToScalars(AliasSet &AS,
                                        SmallVectorImpl<BasicBlock*>&ExitBlocks,
                                        SmallVectorImpl<Instruction*>&InsertPts,
                                        PredIteratorCache &PIC, LoopInfo *LI, 
                                        DominatorTree *DT, Loop *CurLoop, 
                                        AliasSetTracker *CurAST, 
                                        LICMSafetyInfo * SafetyInfo) { 
  // Verify inputs.
  assert(LI != nullptr && DT != nullptr && 
         CurLoop != nullptr && CurAST != nullptr && 
         SafetyInfo != nullptr && 
         "Unexpected Input to promoteLoopAccessesToScalars");
  // Initially set Changed status to false.
  bool Changed = false;
  // We can promote this alias set if it has a store, if it is a "Must" alias
  // set, if the pointer is loop invariant, and if we are not eliminating any
  // volatile loads or stores.
  if (AS.isForwardingAliasSet() || !AS.isMod() || !AS.isMustAlias() ||
      AS.isVolatile() || !CurLoop->isLoopInvariant(AS.begin()->getValue()))
    return Changed;

  assert(!AS.empty() &&
         "Must alias set should have at least one pointer element in it!");

  Value *SomePtr = AS.begin()->getValue();
  BasicBlock * Preheader = CurLoop->getLoopPreheader();

  // It isn't safe to promote a load/store from the loop if the load/store is
  // conditional.  For example, turning:
  //
  //    for () { if (c) *P += 1; }
  //
  // into:
  //
  //    tmp = *P;  for () { if (c) tmp +=1; } *P = tmp;
  //
  // is not safe, because *P may only be valid to access if 'c' is true.
  //
  // It is safe to promote P if all uses are direct load/stores and if at
  // least one is guaranteed to be executed.
  bool GuaranteedToExecute = false;

  SmallVector<Instruction*, 64> LoopUses;
  SmallPtrSet<Value*, 4> PointerMustAliases;

  // We start with an alignment of one and try to find instructions that allow
  // us to prove better alignment.
  unsigned Alignment = 1;
  AAMDNodes AATags;
  bool HasDedicatedExits = CurLoop->hasDedicatedExits();

  // Check that all of the pointers in the alias set have the same type.  We
  // cannot (yet) promote a memory location that is loaded and stored in
  // different sizes.  While we are at it, collect alignment and AA info.
  for (AliasSet::iterator ASI = AS.begin(), E = AS.end(); ASI != E; ++ASI) {
    Value *ASIV = ASI->getValue();
    PointerMustAliases.insert(ASIV);

    // Check that all of the pointers in the alias set have the same type.  We
    // cannot (yet) promote a memory location that is loaded and stored in
    // different sizes.
    if (SomePtr->getType() != ASIV->getType())
      return Changed;

    for (User *U : ASIV->users()) {
      // Ignore instructions that are outside the loop.
      Instruction *UI = dyn_cast<Instruction>(U);
      if (!UI || !CurLoop->contains(UI))
        continue;

      // If there is an non-load/store instruction in the loop, we can't promote
      // it.
      if (const LoadInst *load = dyn_cast<LoadInst>(UI)) {
        assert(!load->isVolatile() && "AST broken");
        if (!load->isSimple())
          return Changed;
      } else if (const StoreInst *store = dyn_cast<StoreInst>(UI)) {
        // Stores *of* the pointer are not interesting, only stores *to* the
        // pointer.
        if (UI->getOperand(1) != ASIV)
          continue;
        assert(!store->isVolatile() && "AST broken");
        if (!store->isSimple())
          return Changed;
        // Don't sink stores from loops without dedicated block exits. Exits
        // containing indirect branches are not transformed by loop simplify,
        // make sure we catch that. An additional load may be generated in the
        // preheader for SSA updater, so also avoid sinking when no preheader
        // is available.
        if (!HasDedicatedExits || !Preheader)
          return Changed;

        // Note that we only check GuaranteedToExecute inside the store case
        // so that we do not introduce stores where they did not exist before
        // (which would break the LLVM concurrency model).

        // If the alignment of this instruction allows us to specify a more
        // restrictive (and performant) alignment and if we are sure this
        // instruction will be executed, update the alignment.
        // Larger is better, with the exception of 0 being the best alignment.
        unsigned InstAlignment = store->getAlignment();
        if ((InstAlignment > Alignment || InstAlignment == 0) && Alignment != 0)
          if (isGuaranteedToExecute(*UI, DT, CurLoop, SafetyInfo)) {
            GuaranteedToExecute = true;
            Alignment = InstAlignment;
          }

        if (!GuaranteedToExecute)
          GuaranteedToExecute = isGuaranteedToExecute(*UI, DT, 
                                                      CurLoop, SafetyInfo);

      } else
        return Changed; // Not a load or store.

      // Merge the AA tags.
      if (LoopUses.empty()) {
        // On the first load/store, just take its AA tags.
        UI->getAAMetadata(AATags);
      } else if (AATags) {
        UI->getAAMetadata(AATags, /* Merge = */ true);
      }

      LoopUses.push_back(UI);
    }
  }

  // If there isn't a guaranteed-to-execute instruction, we can't promote.
  if (!GuaranteedToExecute)
    return Changed;

  // Otherwise, this is safe to promote, lets do it!
  DEBUG(dbgs() << "LICM: Promoting value stored to in loop: " <<*SomePtr<<'\n');
  Changed = true;
  ++NumPromoted;

  // Grab a debug location for the inserted loads/stores; given that the
  // inserted loads/stores have little relation to the original loads/stores,
  // this code just arbitrarily picks a location from one, since any debug
  // location is better than none.
  DebugLoc DL = LoopUses[0]->getDebugLoc();

  // Figure out the loop exits and their insertion points, if this is the
  // first promotion.
  if (ExitBlocks.empty()) {
    CurLoop->getUniqueExitBlocks(ExitBlocks);
    InsertPts.resize(ExitBlocks.size());
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      InsertPts[i] = ExitBlocks[i]->getFirstInsertionPt();
  }

  // We use the SSAUpdater interface to insert phi nodes as required.
  SmallVector<PHINode*, 16> NewPHIs;
  SSAUpdater SSA(&NewPHIs);
  LoopPromoter Promoter(SomePtr, LoopUses, SSA,
                        PointerMustAliases, ExitBlocks,
                        InsertPts, PIC, *CurAST, *LI, DL, Alignment, AATags);

  // Set up the preheader to have a definition of the value.  It is the live-out
  // value from the preheader that uses in the loop will use.
  LoadInst *PreheaderLoad =
    new LoadInst(SomePtr, SomePtr->getName()+".promoted",
                 Preheader->getTerminator());
  PreheaderLoad->setAlignment(Alignment);
  PreheaderLoad->setDebugLoc(DL);
  if (AATags) PreheaderLoad->setAAMetadata(AATags);
  SSA.AddAvailableValue(Preheader, PreheaderLoad);

  // Rewrite all the loads in the loop and remember all the definitions from
  // stores in the loop.
  Promoter.run(LoopUses);

  // If the SSAUpdater didn't use the load in the preheader, just zap it now.
  if (PreheaderLoad->use_empty())
    PreheaderLoad->eraseFromParent();

  return Changed;
}

/// Simple Analysis hook. Clone alias set info.
///
void LICM::cloneBasicBlockAnalysis(BasicBlock *From, BasicBlock *To, Loop *L) {
  AliasSetTracker *AST = LoopToAliasSetMap.lookup(L);
  if (!AST)
    return;

  AST->copyValue(From, To);
}

/// Simple Analysis hook. Delete value V from alias set
///
void LICM::deleteAnalysisValue(Value *V, Loop *L) {
  AliasSetTracker *AST = LoopToAliasSetMap.lookup(L);
  if (!AST)
    return;

  AST->deleteValue(V);
}

/// Simple Analysis hook. Delete value L from alias set map.
///
void LICM::deleteAnalysisLoop(Loop *L) {
  AliasSetTracker *AST = LoopToAliasSetMap.lookup(L);
  if (!AST)
    return;

  delete AST;
  LoopToAliasSetMap.erase(L);
}


/// Return true if the body of this loop may store into the memory
/// location pointed to by V.
///
static bool pointerInvalidatedByLoop(Value *V, uint64_t Size,
                                     const AAMDNodes &AAInfo, 
                                     AliasSetTracker *CurAST) {
  // Check to see if any of the basic blocks in CurLoop invalidate *V.
  return CurAST->getAliasSetForPointer(V, Size, AAInfo).isMod();
}

/// Little predicate that returns true if the specified basic block is in
/// a subloop of the current one, not the current one itself.
///
static bool inSubLoop(BasicBlock *BB, Loop *CurLoop, LoopInfo *LI) {
  assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
  return LI->getLoopFor(BB) != CurLoop;
}

