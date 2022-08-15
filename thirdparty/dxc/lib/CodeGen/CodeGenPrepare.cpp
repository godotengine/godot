//===- CodeGenPrepare.cpp - Prepare a function for code generation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach. It should eventually be removed.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/BypassSlowDivision.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "codegenprepare"

STATISTIC(NumBlocksElim, "Number of blocks eliminated");
STATISTIC(NumPHIsElim,   "Number of trivial PHIs eliminated");
STATISTIC(NumGEPsElim,   "Number of GEPs converted to casts");
STATISTIC(NumCmpUses, "Number of uses of Cmp expressions replaced with uses of "
                      "sunken Cmps");
STATISTIC(NumCastUses, "Number of uses of Cast expressions replaced with uses "
                       "of sunken Casts");
STATISTIC(NumMemoryInsts, "Number of memory instructions whose address "
                          "computations were sunk");
STATISTIC(NumExtsMoved,  "Number of [s|z]ext instructions combined with loads");
STATISTIC(NumExtUses,    "Number of uses of [s|z]ext instructions optimized");
STATISTIC(NumRetsDup,    "Number of return instructions duplicated");
STATISTIC(NumDbgValueMoved, "Number of debug value instructions moved");
STATISTIC(NumSelectsExpanded, "Number of selects turned into branches");
STATISTIC(NumAndCmpsMoved, "Number of and/cmp's pushed into branches");
STATISTIC(NumStoreExtractExposed, "Number of store(extractelement) exposed");

static cl::opt<bool> DisableBranchOpts(
  "disable-cgp-branch-opts", cl::Hidden, cl::init(false),
  cl::desc("Disable branch optimizations in CodeGenPrepare"));

static cl::opt<bool>
    DisableGCOpts("disable-cgp-gc-opts", cl::Hidden, cl::init(false),
                  cl::desc("Disable GC optimizations in CodeGenPrepare"));

static cl::opt<bool> DisableSelectToBranch(
  "disable-cgp-select2branch", cl::Hidden, cl::init(false),
  cl::desc("Disable select to branch conversion."));

static cl::opt<bool> AddrSinkUsingGEPs(
  "addr-sink-using-gep", cl::Hidden, cl::init(false),
  cl::desc("Address sinking in CGP using GEPs."));

static cl::opt<bool> EnableAndCmpSinking(
   "enable-andcmp-sinking", cl::Hidden, cl::init(true),
   cl::desc("Enable sinkinig and/cmp into branches."));

static cl::opt<bool> DisableStoreExtract(
    "disable-cgp-store-extract", cl::Hidden, cl::init(false),
    cl::desc("Disable store(extract) optimizations in CodeGenPrepare"));

static cl::opt<bool> StressStoreExtract(
    "stress-cgp-store-extract", cl::Hidden, cl::init(false),
    cl::desc("Stress test store(extract) optimizations in CodeGenPrepare"));

static cl::opt<bool> DisableExtLdPromotion(
    "disable-cgp-ext-ld-promotion", cl::Hidden, cl::init(false),
    cl::desc("Disable ext(promotable(ld)) -> promoted(ext(ld)) optimization in "
             "CodeGenPrepare"));

static cl::opt<bool> StressExtLdPromotion(
    "stress-cgp-ext-ld-promotion", cl::Hidden, cl::init(false),
    cl::desc("Stress test ext(promotable(ld)) -> promoted(ext(ld)) "
             "optimization in CodeGenPrepare"));

namespace {
typedef SmallPtrSet<Instruction *, 16> SetOfInstrs;
struct TypeIsSExt {
  Type *Ty;
  bool IsSExt;
  TypeIsSExt(Type *Ty, bool IsSExt) : Ty(Ty), IsSExt(IsSExt) {}
};
typedef DenseMap<Instruction *, TypeIsSExt> InstrToOrigTy;
class TypePromotionTransaction;

  class CodeGenPrepare : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetMachine *TM;
    const TargetLowering *TLI;
    const TargetTransformInfo *TTI;
    const TargetLibraryInfo *TLInfo;

    /// CurInstIterator - As we scan instructions optimizing them, this is the
    /// next instruction to optimize.  Xforms that can invalidate this should
    /// update it.
    BasicBlock::iterator CurInstIterator;

    /// Keeps track of non-local addresses that have been sunk into a block.
    /// This allows us to avoid inserting duplicate code for blocks with
    /// multiple load/stores of the same address.
    ValueMap<Value*, Value*> SunkAddrs;

    /// Keeps track of all instructions inserted for the current function.
    SetOfInstrs InsertedInsts;
    /// Keeps track of the type of the related instruction before their
    /// promotion for the current function.
    InstrToOrigTy PromotedInsts;

    /// ModifiedDT - If CFG is modified in anyway.
    bool ModifiedDT;

    /// OptSize - True if optimizing for size.
    bool OptSize;

    /// DataLayout for the Function being processed.
    const DataLayout *DL;

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit CodeGenPrepare(const TargetMachine *TM = nullptr)
        : FunctionPass(ID), TM(TM), TLI(nullptr), TTI(nullptr), DL(nullptr) {
        initializeCodeGenPreparePass(*PassRegistry::getPassRegistry());
      }
    bool runOnFunction(Function &F) override;

    StringRef getPassName() const override { return "CodeGen Prepare"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
    }

  private:
    bool EliminateFallThrough(Function &F);
    bool EliminateMostlyEmptyBlocks(Function &F);
    bool CanMergeBlocks(const BasicBlock *BB, const BasicBlock *DestBB) const;
    void EliminateMostlyEmptyBlock(BasicBlock *BB);
    bool OptimizeBlock(BasicBlock &BB, bool& ModifiedDT);
    bool OptimizeInst(Instruction *I, bool& ModifiedDT);
    bool OptimizeMemoryInst(Instruction *I, Value *Addr,
                            Type *AccessTy, unsigned AS);
    bool OptimizeInlineAsmInst(CallInst *CS);
    bool OptimizeCallInst(CallInst *CI, bool& ModifiedDT);
    bool MoveExtToFormExtLoad(Instruction *&I);
    bool OptimizeExtUses(Instruction *I);
    bool OptimizeSelectInst(SelectInst *SI);
    bool OptimizeShuffleVectorInst(ShuffleVectorInst *SI);
    bool OptimizeExtractElementInst(Instruction *Inst);
    bool DupRetToEnableTailCallOpts(BasicBlock *BB);
    bool PlaceDbgValues(Function &F);
    bool sinkAndCmp(Function &F);
    bool ExtLdPromotion(TypePromotionTransaction &TPT, LoadInst *&LI,
                        Instruction *&Inst,
                        const SmallVectorImpl<Instruction *> &Exts,
                        unsigned CreatedInstCost);
    bool splitBranchCondition(Function &F);
    bool simplifyOffsetableRelocate(Instruction &I);
  };
}

char CodeGenPrepare::ID = 0;
INITIALIZE_TM_PASS(CodeGenPrepare, "codegenprepare",
                   "Optimize for code generation", false, false)

FunctionPass *llvm::createCodeGenPreparePass(const TargetMachine *TM) {
  return new CodeGenPrepare(TM);
}

bool CodeGenPrepare::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  DL = &F.getParent()->getDataLayout();

  bool EverMadeChange = false;
  // Clear per function information.
  InsertedInsts.clear();
  PromotedInsts.clear();

  ModifiedDT = false;
  if (TM)
    TLI = TM->getSubtargetImpl(F)->getTargetLowering();
  TLInfo = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  OptSize = F.hasFnAttribute(Attribute::OptimizeForSize);

  /// This optimization identifies DIV instructions that can be
  /// profitably bypassed and carried out with a shorter, faster divide.
  if (!OptSize && TLI && TLI->isSlowDivBypassed()) {
    const DenseMap<unsigned int, unsigned int> &BypassWidths =
       TLI->getBypassSlowDivWidths();
    for (Function::iterator I = F.begin(); I != F.end(); I++)
      EverMadeChange |= bypassSlowDivision(F, I, BypassWidths);
  }

  // Eliminate blocks that contain only PHI nodes and an
  // unconditional branch.
  EverMadeChange |= EliminateMostlyEmptyBlocks(F);

  // llvm.dbg.value is far away from the value then iSel may not be able
  // handle it properly. iSel will drop llvm.dbg.value if it can not
  // find a node corresponding to the value.
  EverMadeChange |= PlaceDbgValues(F);

  // If there is a mask, compare against zero, and branch that can be combined
  // into a single target instruction, push the mask and compare into branch
  // users. Do this before OptimizeBlock -> OptimizeInst ->
  // OptimizeCmpExpression, which perturbs the pattern being searched for.
  if (!DisableBranchOpts) {
    EverMadeChange |= sinkAndCmp(F);
    EverMadeChange |= splitBranchCondition(F);
  }

  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (Function::iterator I = F.begin(); I != F.end(); ) {
      BasicBlock *BB = I++;
      bool ModifiedDTOnIteration = false;
      MadeChange |= OptimizeBlock(*BB, ModifiedDTOnIteration);

      // Restart BB iteration if the dominator tree of the Function was changed
      if (ModifiedDTOnIteration)
        break;
    }
    EverMadeChange |= MadeChange;
  }

  SunkAddrs.clear();

  if (!DisableBranchOpts) {
    MadeChange = false;
    SmallPtrSet<BasicBlock*, 8> WorkList;
    for (BasicBlock &BB : F) {
      SmallVector<BasicBlock *, 2> Successors(succ_begin(&BB), succ_end(&BB));
      MadeChange |= ConstantFoldTerminator(&BB, true);
      if (!MadeChange) continue;

      for (SmallVectorImpl<BasicBlock*>::iterator
             II = Successors.begin(), IE = Successors.end(); II != IE; ++II)
        if (pred_begin(*II) == pred_end(*II))
          WorkList.insert(*II);
    }

    // Delete the dead blocks and any of their dead successors.
    MadeChange |= !WorkList.empty();
    while (!WorkList.empty()) {
      BasicBlock *BB = *WorkList.begin();
      WorkList.erase(BB);
      SmallVector<BasicBlock*, 2> Successors(succ_begin(BB), succ_end(BB));

      DeleteDeadBlock(BB);

      for (SmallVectorImpl<BasicBlock*>::iterator
             II = Successors.begin(), IE = Successors.end(); II != IE; ++II)
        if (pred_begin(*II) == pred_end(*II))
          WorkList.insert(*II);
    }

    // Merge pairs of basic blocks with unconditional branches, connected by
    // a single edge.
    if (EverMadeChange || MadeChange)
      MadeChange |= EliminateFallThrough(F);

    EverMadeChange |= MadeChange;
  }

  if (!DisableGCOpts) {
    SmallVector<Instruction *, 2> Statepoints;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isStatepoint(I))
          Statepoints.push_back(&I);
    for (auto &I : Statepoints)
      EverMadeChange |= simplifyOffsetableRelocate(*I);
  }

  return EverMadeChange;
}

/// EliminateFallThrough - Merge basic blocks which are connected
/// by a single edge, where one of the basic blocks has a single successor
/// pointing to the other basic block, which has a single predecessor.
bool CodeGenPrepare::EliminateFallThrough(Function &F) {
  bool Changed = false;
  // Scan all of the blocks in the function, except for the entry block.
  for (Function::iterator I = std::next(F.begin()), E = F.end(); I != E;) {
    BasicBlock *BB = I++;
    // If the destination block has a single pred, then this is a trivial
    // edge, just collapse it.
    BasicBlock *SinglePred = BB->getSinglePredecessor();

    // Don't merge if BB's address is taken.
    if (!SinglePred || SinglePred == BB || BB->hasAddressTaken()) continue;

    BranchInst *Term = dyn_cast<BranchInst>(SinglePred->getTerminator());
    if (Term && !Term->isConditional()) {
      Changed = true;
      DEBUG(dbgs() << "To merge:\n"<< *SinglePred << "\n\n\n");
      // Remember if SinglePred was the entry block of the function.
      // If so, we will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(BB, nullptr);

      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());

      // We have erased a block. Update the iterator.
      I = BB;
    }
  }
  return Changed;
}

/// EliminateMostlyEmptyBlocks - eliminate blocks that contain only PHI nodes,
/// debug info directives, and an unconditional branch.  Passes before isel
/// (e.g. LSR/loopsimplify) often split edges in ways that are non-optimal for
/// isel.  Start by eliminating these blocks so we can split them the way we
/// want them.
bool CodeGenPrepare::EliminateMostlyEmptyBlocks(Function &F) {
  bool MadeChange = false;
  // Note that this intentionally skips the entry block.
  for (Function::iterator I = std::next(F.begin()), E = F.end(); I != E;) {
    BasicBlock *BB = I++;

    // If this block doesn't end with an uncond branch, ignore it.
    BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isUnconditional())
      continue;

    // If the instruction before the branch (skipping debug info) isn't a phi
    // node, then other stuff is happening here.
    BasicBlock::iterator BBI = BI;
    if (BBI != BB->begin()) {
      --BBI;
      while (isa<DbgInfoIntrinsic>(BBI)) {
        if (BBI == BB->begin())
          break;
        --BBI;
      }
      if (!isa<DbgInfoIntrinsic>(BBI) && !isa<PHINode>(BBI))
        continue;
    }

    // Do not break infinite loops.
    BasicBlock *DestBB = BI->getSuccessor(0);
    if (DestBB == BB)
      continue;

    if (!CanMergeBlocks(BB, DestBB))
      continue;

    EliminateMostlyEmptyBlock(BB);
    MadeChange = true;
  }
  return MadeChange;
}

/// CanMergeBlocks - Return true if we can merge BB into DestBB if there is a
/// single uncond branch between them, and BB contains no other non-phi
/// instructions.
bool CodeGenPrepare::CanMergeBlocks(const BasicBlock *BB,
                                    const BasicBlock *DestBB) const {
  // We only want to eliminate blocks whose phi nodes are used by phi nodes in
  // the successor.  If there are more complex condition (e.g. preheaders),
  // don't mess around with them.
  BasicBlock::const_iterator BBI = BB->begin();
  while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
    for (const User *U : PN->users()) {
      const Instruction *UI = cast<Instruction>(U);
      if (UI->getParent() != DestBB || !isa<PHINode>(UI))
        return false;
      // If User is inside DestBB block and it is a PHINode then check
      // incoming value. If incoming value is not from BB then this is
      // a complex condition (e.g. preheaders) we want to avoid here.
      if (UI->getParent() == DestBB) {
        if (const PHINode *UPN = dyn_cast<PHINode>(UI))
          for (unsigned I = 0, E = UPN->getNumIncomingValues(); I != E; ++I) {
            Instruction *Insn = dyn_cast<Instruction>(UPN->getIncomingValue(I));
            if (Insn && Insn->getParent() == BB &&
                Insn->getParent() != UPN->getIncomingBlock(I))
              return false;
          }
      }
    }
  }

  // If BB and DestBB contain any common predecessors, then the phi nodes in BB
  // and DestBB may have conflicting incoming values for the block.  If so, we
  // can't merge the block.
  const PHINode *DestBBPN = dyn_cast<PHINode>(DestBB->begin());
  if (!DestBBPN) return true;  // no conflict.

  // Collect the preds of BB.
  SmallPtrSet<const BasicBlock*, 16> BBPreds;
  if (const PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
    // It is faster to get preds from a PHI than with pred_iterator.
    for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
      BBPreds.insert(BBPN->getIncomingBlock(i));
  } else {
    BBPreds.insert(pred_begin(BB), pred_end(BB));
  }

  // Walk the preds of DestBB.
  for (unsigned i = 0, e = DestBBPN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = DestBBPN->getIncomingBlock(i);
    if (BBPreds.count(Pred)) {   // Common predecessor?
      BBI = DestBB->begin();
      while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
        const Value *V1 = PN->getIncomingValueForBlock(Pred);
        const Value *V2 = PN->getIncomingValueForBlock(BB);

        // If V2 is a phi node in BB, look up what the mapped value will be.
        if (const PHINode *V2PN = dyn_cast<PHINode>(V2))
          if (V2PN->getParent() == BB)
            V2 = V2PN->getIncomingValueForBlock(Pred);

        // If there is a conflict, bail out.
        if (V1 != V2) return false;
      }
    }
  }

  return true;
}


/// EliminateMostlyEmptyBlock - Eliminate a basic block that have only phi's and
/// an unconditional branch in it.
void CodeGenPrepare::EliminateMostlyEmptyBlock(BasicBlock *BB) {
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  BasicBlock *DestBB = BI->getSuccessor(0);

  DEBUG(dbgs() << "MERGING MOSTLY EMPTY BLOCKS - BEFORE:\n" << *BB << *DestBB);

  // If the destination block has a single pred, then this is a trivial edge,
  // just collapse it.
  if (BasicBlock *SinglePred = DestBB->getSinglePredecessor()) {
    if (SinglePred != DestBB) {
      // Remember if SinglePred was the entry block of the function.  If so, we
      // will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(DestBB, nullptr);

      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());

      DEBUG(dbgs() << "AFTER:\n" << *DestBB << "\n\n\n");
      return;
    }
  }

  // Otherwise, we have multiple predecessors of BB.  Update the PHIs in DestBB
  // to handle the new incoming edges it is about to have.
  PHINode *PN;
  for (BasicBlock::iterator BBI = DestBB->begin();
       (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
    // Remove the incoming value for BB, and remember it.
    Value *InVal = PN->removeIncomingValue(BB, false);

    // Two options: either the InVal is a phi node defined in BB or it is some
    // value that dominates BB.
    PHINode *InValPhi = dyn_cast<PHINode>(InVal);
    if (InValPhi && InValPhi->getParent() == BB) {
      // Add all of the input values of the input PHI as inputs of this phi.
      for (unsigned i = 0, e = InValPhi->getNumIncomingValues(); i != e; ++i)
        PN->addIncoming(InValPhi->getIncomingValue(i),
                        InValPhi->getIncomingBlock(i));
    } else {
      // Otherwise, add one instance of the dominating value for each edge that
      // we will be adding.
      if (PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
        for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
          PN->addIncoming(InVal, BBPN->getIncomingBlock(i));
      } else {
        for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
          PN->addIncoming(InVal, *PI);
      }
    }
  }

  // The PHIs are now updated, change everything that refers to BB to use
  // DestBB and remove BB.
  BB->replaceAllUsesWith(DestBB);
  BB->eraseFromParent();
  ++NumBlocksElim;

  DEBUG(dbgs() << "AFTER:\n" << *DestBB << "\n\n\n");
}

// Computes a map of base pointer relocation instructions to corresponding
// derived pointer relocation instructions given a vector of all relocate calls
static void computeBaseDerivedRelocateMap(
    const SmallVectorImpl<User *> &AllRelocateCalls,
    DenseMap<IntrinsicInst *, SmallVector<IntrinsicInst *, 2>> &
        RelocateInstMap) {
  // Collect information in two maps: one primarily for locating the base object
  // while filling the second map; the second map is the final structure holding
  // a mapping between Base and corresponding Derived relocate calls
  DenseMap<std::pair<unsigned, unsigned>, IntrinsicInst *> RelocateIdxMap;
  for (auto &U : AllRelocateCalls) {
    GCRelocateOperands ThisRelocate(U);
    IntrinsicInst *I = cast<IntrinsicInst>(U);
    auto K = std::make_pair(ThisRelocate.getBasePtrIndex(),
                            ThisRelocate.getDerivedPtrIndex());
    RelocateIdxMap.insert(std::make_pair(K, I));
  }
  for (auto &Item : RelocateIdxMap) {
    std::pair<unsigned, unsigned> Key = Item.first;
    if (Key.first == Key.second)
      // Base relocation: nothing to insert
      continue;

    IntrinsicInst *I = Item.second;
    auto BaseKey = std::make_pair(Key.first, Key.first);

    // We're iterating over RelocateIdxMap so we cannot modify it.
    auto MaybeBase = RelocateIdxMap.find(BaseKey);
    if (MaybeBase == RelocateIdxMap.end())
      // TODO: We might want to insert a new base object relocate and gep off
      // that, if there are enough derived object relocates.
      continue;

    RelocateInstMap[MaybeBase->second].push_back(I);
  }
}

// Accepts a GEP and extracts the operands into a vector provided they're all
// small integer constants
static bool getGEPSmallConstantIntOffsetV(GetElementPtrInst *GEP,
                                          SmallVectorImpl<Value *> &OffsetV) {
  for (unsigned i = 1; i < GEP->getNumOperands(); i++) {
    // Only accept small constant integer operands
    auto Op = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!Op || Op->getZExtValue() > 20)
      return false;
  }

  for (unsigned i = 1; i < GEP->getNumOperands(); i++)
    OffsetV.push_back(GEP->getOperand(i));
  return true;
}

// Takes a RelocatedBase (base pointer relocation instruction) and Targets to
// replace, computes a replacement, and affects it.
static bool
simplifyRelocatesOffABase(IntrinsicInst *RelocatedBase,
                          const SmallVectorImpl<IntrinsicInst *> &Targets) {
  bool MadeChange = false;
  for (auto &ToReplace : Targets) {
    GCRelocateOperands MasterRelocate(RelocatedBase);
    GCRelocateOperands ThisRelocate(ToReplace);

    assert(ThisRelocate.getBasePtrIndex() == MasterRelocate.getBasePtrIndex() &&
           "Not relocating a derived object of the original base object");
    if (ThisRelocate.getBasePtrIndex() == ThisRelocate.getDerivedPtrIndex()) {
      // A duplicate relocate call. TODO: coalesce duplicates.
      continue;
    }

    Value *Base = ThisRelocate.getBasePtr();
    auto Derived = dyn_cast<GetElementPtrInst>(ThisRelocate.getDerivedPtr());
    if (!Derived || Derived->getPointerOperand() != Base)
      continue;

    SmallVector<Value *, 2> OffsetV;
    if (!getGEPSmallConstantIntOffsetV(Derived, OffsetV))
      continue;

    // Create a Builder and replace the target callsite with a gep
    assert(RelocatedBase->getNextNode() && "Should always have one since it's not a terminator");

    // Insert after RelocatedBase
    IRBuilder<> Builder(RelocatedBase->getNextNode());
    Builder.SetCurrentDebugLocation(ToReplace->getDebugLoc());

    // If gc_relocate does not match the actual type, cast it to the right type.
    // In theory, there must be a bitcast after gc_relocate if the type does not
    // match, and we should reuse it to get the derived pointer. But it could be
    // cases like this:
    // bb1:
    //  ...
    //  %g1 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(...)
    //  br label %merge
    //
    // bb2:
    //  ...
    //  %g2 = call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(...)
    //  br label %merge
    //
    // merge:
    //  %p1 = phi i8 addrspace(1)* [ %g1, %bb1 ], [ %g2, %bb2 ]
    //  %cast = bitcast i8 addrspace(1)* %p1 in to i32 addrspace(1)*
    //
    // In this case, we can not find the bitcast any more. So we insert a new bitcast
    // no matter there is already one or not. In this way, we can handle all cases, and
    // the extra bitcast should be optimized away in later passes.
    Instruction *ActualRelocatedBase = RelocatedBase;
    if (RelocatedBase->getType() != Base->getType()) {
      ActualRelocatedBase =
          cast<Instruction>(Builder.CreateBitCast(RelocatedBase, Base->getType()));
    }
    Value *Replacement = Builder.CreateGEP(
        Derived->getSourceElementType(), ActualRelocatedBase, makeArrayRef(OffsetV));
    Instruction *ReplacementInst = cast<Instruction>(Replacement);
    Replacement->takeName(ToReplace);
    // If the newly generated derived pointer's type does not match the original derived
    // pointer's type, cast the new derived pointer to match it. Same reasoning as above.
    Instruction *ActualReplacement = ReplacementInst;
    if (ReplacementInst->getType() != ToReplace->getType()) {
      ActualReplacement =
          cast<Instruction>(Builder.CreateBitCast(ReplacementInst, ToReplace->getType()));
    }
    ToReplace->replaceAllUsesWith(ActualReplacement);
    ToReplace->eraseFromParent();

    MadeChange = true;
  }
  return MadeChange;
}

// Turns this:
//
// %base = ...
// %ptr = gep %base + 15
// %tok = statepoint (%fun, i32 0, i32 0, i32 0, %base, %ptr)
// %base' = relocate(%tok, i32 4, i32 4)
// %ptr' = relocate(%tok, i32 4, i32 5)
// %val = load %ptr'
//
// into this:
//
// %base = ...
// %ptr = gep %base + 15
// %tok = statepoint (%fun, i32 0, i32 0, i32 0, %base, %ptr)
// %base' = gc.relocate(%tok, i32 4, i32 4)
// %ptr' = gep %base' + 15
// %val = load %ptr'
bool CodeGenPrepare::simplifyOffsetableRelocate(Instruction &I) {
  bool MadeChange = false;
  SmallVector<User *, 2> AllRelocateCalls;

  for (auto *U : I.users())
    if (isGCRelocate(dyn_cast<Instruction>(U)))
      // Collect all the relocate calls associated with a statepoint
      AllRelocateCalls.push_back(U);

  // We need atleast one base pointer relocation + one derived pointer
  // relocation to mangle
  if (AllRelocateCalls.size() < 2)
    return false;

  // RelocateInstMap is a mapping from the base relocate instruction to the
  // corresponding derived relocate instructions
  DenseMap<IntrinsicInst *, SmallVector<IntrinsicInst *, 2>> RelocateInstMap;
  computeBaseDerivedRelocateMap(AllRelocateCalls, RelocateInstMap);
  if (RelocateInstMap.empty())
    return false;

  for (auto &Item : RelocateInstMap)
    // Item.first is the RelocatedBase to offset against
    // Item.second is the vector of Targets to replace
    MadeChange = simplifyRelocatesOffABase(Item.first, Item.second);
  return MadeChange;
}

/// SinkCast - Sink the specified cast instruction into its user blocks
static bool SinkCast(CastInst *CI) {
  BasicBlock *DefBB = CI->getParent();

  /// InsertedCasts - Only insert a cast in each block once.
  DenseMap<BasicBlock*, CastInst*> InsertedCasts;

  bool MadeChange = false;
  for (Value::user_iterator UI = CI->user_begin(), E = CI->user_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this cast is used in.  For PHI's this is the
    // appropriate predecessor block.
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      UserBB = PN->getIncomingBlock(TheUse);
    }

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // If this user is in the same block as the cast, don't change the cast.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cast into this block, use it.
    CastInst *&InsertedCast = InsertedCasts[UserBB];

    if (!InsertedCast) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedCast =
        CastInst::Create(CI->getOpcode(), CI->getOperand(0), CI->getType(), "",
                         InsertPt);
    }

    // Replace a use of the cast with a use of the new cast.
    TheUse = InsertedCast;
    MadeChange = true;
    ++NumCastUses;
  }

  // If we removed all uses, nuke the cast.
  if (CI->use_empty()) {
    CI->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

/// OptimizeNoopCopyExpression - If the specified cast instruction is a noop
/// copy (e.g. it's casting from one pointer type to another, i32->i8 on PPC),
/// sink it into user blocks to reduce the number of virtual
/// registers that must be created and coalesced.
///
/// Return true if any changes are made.
///
static bool OptimizeNoopCopyExpression(CastInst *CI, const TargetLowering &TLI,
                                       const DataLayout &DL) {
  // If this is a noop copy,
  EVT SrcVT = TLI.getValueType(DL, CI->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(DL, CI->getType());

  // This is an fp<->int conversion?
  if (SrcVT.isInteger() != DstVT.isInteger())
    return false;

  // If this is an extension, it will be a zero or sign extension, which
  // isn't a noop.
  if (SrcVT.bitsLT(DstVT)) return false;

  // If these values will be promoted, find out what they will be promoted
  // to.  This helps us consider truncates on PPC as noop copies when they
  // are.
  if (TLI.getTypeAction(CI->getContext(), SrcVT) ==
      TargetLowering::TypePromoteInteger)
    SrcVT = TLI.getTypeToTransformTo(CI->getContext(), SrcVT);
  if (TLI.getTypeAction(CI->getContext(), DstVT) ==
      TargetLowering::TypePromoteInteger)
    DstVT = TLI.getTypeToTransformTo(CI->getContext(), DstVT);

  // If, after promotion, these are the same types, this is a noop copy.
  if (SrcVT != DstVT)
    return false;

  return SinkCast(CI);
}

/// CombineUAddWithOverflow - try to combine CI into a call to the
/// llvm.uadd.with.overflow intrinsic if possible.
///
/// Return true if any changes were made.
static bool CombineUAddWithOverflow(CmpInst *CI) {
  Value *A, *B;
  Instruction *AddI;
  if (!match(CI,
             m_UAddWithOverflow(m_Value(A), m_Value(B), m_Instruction(AddI))))
    return false;

  Type *Ty = AddI->getType();
  if (!isa<IntegerType>(Ty))
    return false;

  // We don't want to move around uses of condition values this late, so we we
  // check if it is legal to create the call to the intrinsic in the basic
  // block containing the icmp:

  if (AddI->getParent() != CI->getParent() && !AddI->hasOneUse())
    return false;

#ifndef NDEBUG
  // Someday m_UAddWithOverflow may get smarter, but this is a safe assumption
  // for now:
  if (AddI->hasOneUse())
    assert(*AddI->user_begin() == CI && "expected!");
#endif

  Module *M = CI->getParent()->getParent()->getParent();
  Value *F = Intrinsic::getDeclaration(M, Intrinsic::uadd_with_overflow, Ty);

  auto *InsertPt = AddI->hasOneUse() ? CI : AddI;

  auto *UAddWithOverflow =
      CallInst::Create(F, {A, B}, "uadd.overflow", InsertPt);
  auto *UAdd = ExtractValueInst::Create(UAddWithOverflow, 0, "uadd", InsertPt);
  auto *Overflow =
      ExtractValueInst::Create(UAddWithOverflow, 1, "overflow", InsertPt);

  CI->replaceAllUsesWith(Overflow);
  AddI->replaceAllUsesWith(UAdd);
  CI->eraseFromParent();
  AddI->eraseFromParent();
  return true;
}

/// SinkCmpExpression - Sink the given CmpInst into user blocks to reduce
/// the number of virtual registers that must be created and coalesced.  This is
/// a clear win except on targets with multiple condition code registers
///  (PowerPC), where it might lose; some adjustment may be wanted there.
///
/// Return true if any changes are made.
static bool SinkCmpExpression(CmpInst *CI) {
  BasicBlock *DefBB = CI->getParent();

  /// InsertedCmp - Only insert a cmp in each block once.
  DenseMap<BasicBlock*, CmpInst*> InsertedCmps;

  bool MadeChange = false;
  for (Value::user_iterator UI = CI->user_begin(), E = CI->user_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // Don't bother for PHI nodes.
    if (isa<PHINode>(User))
      continue;

    // Figure out which BB this cmp is used in.
    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the cmp, don't change the cmp.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cmp into this block, use it.
    CmpInst *&InsertedCmp = InsertedCmps[UserBB];

    if (!InsertedCmp) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedCmp =
        CmpInst::Create(CI->getOpcode(),
                        CI->getPredicate(),  CI->getOperand(0),
                        CI->getOperand(1), "", InsertPt);
    }

    // Replace a use of the cmp with a use of the new cmp.
    TheUse = InsertedCmp;
    MadeChange = true;
    ++NumCmpUses;
  }

  // If we removed all uses, nuke the cmp.
  if (CI->use_empty()) {
    CI->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

static bool OptimizeCmpExpression(CmpInst *CI) {
  if (SinkCmpExpression(CI))
    return true;

  if (CombineUAddWithOverflow(CI))
    return true;

  return false;
}

/// isExtractBitsCandidateUse - Check if the candidates could
/// be combined with shift instruction, which includes:
/// 1. Truncate instruction
/// 2. And instruction and the imm is a mask of the low bits:
/// imm & (imm+1) == 0
static bool isExtractBitsCandidateUse(Instruction *User) {
  if (!isa<TruncInst>(User)) {
    if (User->getOpcode() != Instruction::And ||
        !isa<ConstantInt>(User->getOperand(1)))
      return false;

    const APInt &Cimm = cast<ConstantInt>(User->getOperand(1))->getValue();

    if ((Cimm & (Cimm + 1)).getBoolValue())
      return false;
  }
  return true;
}

/// SinkShiftAndTruncate - sink both shift and truncate instruction
/// to the use of truncate's BB.
static bool
SinkShiftAndTruncate(BinaryOperator *ShiftI, Instruction *User, ConstantInt *CI,
                     DenseMap<BasicBlock *, BinaryOperator *> &InsertedShifts,
                     const TargetLowering &TLI, const DataLayout &DL) {
  BasicBlock *UserBB = User->getParent();
  DenseMap<BasicBlock *, CastInst *> InsertedTruncs;
  TruncInst *TruncI = dyn_cast<TruncInst>(User);
  bool MadeChange = false;

  for (Value::user_iterator TruncUI = TruncI->user_begin(),
                            TruncE = TruncI->user_end();
       TruncUI != TruncE;) {

    Use &TruncTheUse = TruncUI.getUse();
    Instruction *TruncUser = cast<Instruction>(*TruncUI);
    // Preincrement use iterator so we don't invalidate it.

    ++TruncUI;

    int ISDOpcode = TLI.InstructionOpcodeToISD(TruncUser->getOpcode());
    if (!ISDOpcode)
      continue;

    // If the use is actually a legal node, there will not be an
    // implicit truncate.
    // FIXME: always querying the result type is just an
    // approximation; some nodes' legality is determined by the
    // operand or other means. There's no good way to find out though.
    if (TLI.isOperationLegalOrCustom(
            ISDOpcode, TLI.getValueType(DL, TruncUser->getType(), true)))
      continue;

    // Don't bother for PHI nodes.
    if (isa<PHINode>(TruncUser))
      continue;

    BasicBlock *TruncUserBB = TruncUser->getParent();

    if (UserBB == TruncUserBB)
      continue;

    BinaryOperator *&InsertedShift = InsertedShifts[TruncUserBB];
    CastInst *&InsertedTrunc = InsertedTruncs[TruncUserBB];

    if (!InsertedShift && !InsertedTrunc) {
      BasicBlock::iterator InsertPt = TruncUserBB->getFirstInsertionPt();
      // Sink the shift
      if (ShiftI->getOpcode() == Instruction::AShr)
        InsertedShift =
            BinaryOperator::CreateAShr(ShiftI->getOperand(0), CI, "", InsertPt);
      else
        InsertedShift =
            BinaryOperator::CreateLShr(ShiftI->getOperand(0), CI, "", InsertPt);

      // Sink the trunc
      BasicBlock::iterator TruncInsertPt = TruncUserBB->getFirstInsertionPt();
      TruncInsertPt++;

      InsertedTrunc = CastInst::Create(TruncI->getOpcode(), InsertedShift,
                                       TruncI->getType(), "", TruncInsertPt);

      MadeChange = true;

      TruncTheUse = InsertedTrunc;
    }
  }
  return MadeChange;
}

/// OptimizeExtractBits - sink the shift *right* instruction into user blocks if
/// the uses could potentially be combined with this shift instruction and
/// generate BitExtract instruction. It will only be applied if the architecture
/// supports BitExtract instruction. Here is an example:
/// BB1:
///   %x.extract.shift = lshr i64 %arg1, 32
/// BB2:
///   %x.extract.trunc = trunc i64 %x.extract.shift to i16
/// ==>
///
/// BB2:
///   %x.extract.shift.1 = lshr i64 %arg1, 32
///   %x.extract.trunc = trunc i64 %x.extract.shift.1 to i16
///
/// CodeGen will recoginze the pattern in BB2 and generate BitExtract
/// instruction.
/// Return true if any changes are made.
static bool OptimizeExtractBits(BinaryOperator *ShiftI, ConstantInt *CI,
                                const TargetLowering &TLI,
                                const DataLayout &DL) {
  BasicBlock *DefBB = ShiftI->getParent();

  /// Only insert instructions in each block once.
  DenseMap<BasicBlock *, BinaryOperator *> InsertedShifts;

  bool shiftIsLegal = TLI.isTypeLegal(TLI.getValueType(DL, ShiftI->getType()));

  bool MadeChange = false;
  for (Value::user_iterator UI = ShiftI->user_begin(), E = ShiftI->user_end();
       UI != E;) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);
    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // Don't bother for PHI nodes.
    if (isa<PHINode>(User))
      continue;

    if (!isExtractBitsCandidateUse(User))
      continue;

    BasicBlock *UserBB = User->getParent();

    if (UserBB == DefBB) {
      // If the shift and truncate instruction are in the same BB. The use of
      // the truncate(TruncUse) may still introduce another truncate if not
      // legal. In this case, we would like to sink both shift and truncate
      // instruction to the BB of TruncUse.
      // for example:
      // BB1:
      // i64 shift.result = lshr i64 opnd, imm
      // trunc.result = trunc shift.result to i16
      //
      // BB2:
      //   ----> We will have an implicit truncate here if the architecture does
      //   not have i16 compare.
      // cmp i16 trunc.result, opnd2
      //
      if (isa<TruncInst>(User) && shiftIsLegal
          // If the type of the truncate is legal, no trucate will be
          // introduced in other basic blocks.
          &&
          (!TLI.isTypeLegal(TLI.getValueType(DL, User->getType()))))
        MadeChange =
            SinkShiftAndTruncate(ShiftI, User, CI, InsertedShifts, TLI, DL);

      continue;
    }
    // If we have already inserted a shift into this block, use it.
    BinaryOperator *&InsertedShift = InsertedShifts[UserBB];

    if (!InsertedShift) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();

      if (ShiftI->getOpcode() == Instruction::AShr)
        InsertedShift =
            BinaryOperator::CreateAShr(ShiftI->getOperand(0), CI, "", InsertPt);
      else
        InsertedShift =
            BinaryOperator::CreateLShr(ShiftI->getOperand(0), CI, "", InsertPt);

      MadeChange = true;
    }

    // Replace a use of the shift with a use of the new shift.
    TheUse = InsertedShift;
  }

  // If we removed all uses, nuke the shift.
  if (ShiftI->use_empty())
    ShiftI->eraseFromParent();

  return MadeChange;
}

//  ScalarizeMaskedLoad() translates masked load intrinsic, like 
// <16 x i32 > @llvm.masked.load( <16 x i32>* %addr, i32 align,
//                               <16 x i1> %mask, <16 x i32> %passthru)
// to a chain of basic blocks, whith loading element one-by-one if
// the appropriate mask bit is set
// 
//  %1 = bitcast i8* %addr to i32*
//  %2 = extractelement <16 x i1> %mask, i32 0
//  %3 = icmp eq i1 %2, true
//  br i1 %3, label %cond.load, label %else
//
//cond.load:                                        ; preds = %0
//  %4 = getelementptr i32* %1, i32 0
//  %5 = load i32* %4
//  %6 = insertelement <16 x i32> undef, i32 %5, i32 0
//  br label %else
//
//else:                                             ; preds = %0, %cond.load
//  %res.phi.else = phi <16 x i32> [ %6, %cond.load ], [ undef, %0 ]
//  %7 = extractelement <16 x i1> %mask, i32 1
//  %8 = icmp eq i1 %7, true
//  br i1 %8, label %cond.load1, label %else2
//
//cond.load1:                                       ; preds = %else
//  %9 = getelementptr i32* %1, i32 1
//  %10 = load i32* %9
//  %11 = insertelement <16 x i32> %res.phi.else, i32 %10, i32 1
//  br label %else2
//
//else2:                                            ; preds = %else, %cond.load1
//  %res.phi.else3 = phi <16 x i32> [ %11, %cond.load1 ], [ %res.phi.else, %else ]
//  %12 = extractelement <16 x i1> %mask, i32 2
//  %13 = icmp eq i1 %12, true
//  br i1 %13, label %cond.load4, label %else5
//
static void ScalarizeMaskedLoad(CallInst *CI) {
  Value *Ptr  = CI->getArgOperand(0);
  Value *Src0 = CI->getArgOperand(3);
  Value *Mask = CI->getArgOperand(2);
  VectorType *VecType = dyn_cast<VectorType>(CI->getType());
  Type *EltTy = VecType->getElementType();

  assert(VecType && "Unexpected return type of masked load intrinsic");

  IRBuilder<> Builder(CI->getContext());
  Instruction *InsertPt = CI;
  BasicBlock *IfBlock = CI->getParent();
  BasicBlock *CondBlock = nullptr;
  BasicBlock *PrevIfBlock = CI->getParent();
  Builder.SetInsertPoint(InsertPt);

  Builder.SetCurrentDebugLocation(CI->getDebugLoc());

  // Bitcast %addr fron i8* to EltTy*
  Type *NewPtrType =
    EltTy->getPointerTo(cast<PointerType>(Ptr->getType())->getAddressSpace());
  Value *FirstEltPtr = Builder.CreateBitCast(Ptr, NewPtrType);
  Value *UndefVal = UndefValue::get(VecType);

  // The result vector
  Value *VResult = UndefVal;

  PHINode *Phi = nullptr;
  Value *PrevPhi = UndefVal;

  unsigned VectorWidth = VecType->getNumElements();
  for (unsigned Idx = 0; Idx < VectorWidth; ++Idx) {

    // Fill the "else" block, created in the previous iteration
    //
    //  %res.phi.else3 = phi <16 x i32> [ %11, %cond.load1 ], [ %res.phi.else, %else ]
    //  %mask_1 = extractelement <16 x i1> %mask, i32 Idx
    //  %to_load = icmp eq i1 %mask_1, true
    //  br i1 %to_load, label %cond.load, label %else
    //
    if (Idx > 0) {
      Phi = Builder.CreatePHI(VecType, 2, "res.phi.else");
      Phi->addIncoming(VResult, CondBlock);
      Phi->addIncoming(PrevPhi, PrevIfBlock);
      PrevPhi = Phi;
      VResult = Phi;
    }

    Value *Predicate = Builder.CreateExtractElement(Mask, Builder.getInt32(Idx));
    Value *Cmp = Builder.CreateICmp(ICmpInst::ICMP_EQ, Predicate,
                                    ConstantInt::get(Predicate->getType(), 1));

    // Create "cond" block
    //
    //  %EltAddr = getelementptr i32* %1, i32 0
    //  %Elt = load i32* %EltAddr
    //  VResult = insertelement <16 x i32> VResult, i32 %Elt, i32 Idx
    //
    CondBlock = IfBlock->splitBasicBlock(InsertPt, "cond.load");
    Builder.SetInsertPoint(InsertPt);

    Value *Gep =
        Builder.CreateInBoundsGEP(EltTy, FirstEltPtr, Builder.getInt32(Idx));
    LoadInst* Load = Builder.CreateLoad(Gep, false);
    VResult = Builder.CreateInsertElement(VResult, Load, Builder.getInt32(Idx));

    // Create "else" block, fill it in the next iteration
    BasicBlock *NewIfBlock = CondBlock->splitBasicBlock(InsertPt, "else");
    Builder.SetInsertPoint(InsertPt);
    Instruction *OldBr = IfBlock->getTerminator();
    BranchInst::Create(CondBlock, NewIfBlock, Cmp, OldBr);
    OldBr->eraseFromParent();
    PrevIfBlock = IfBlock;
    IfBlock = NewIfBlock;
  }

  Phi = Builder.CreatePHI(VecType, 2, "res.phi.select");
  Phi->addIncoming(VResult, CondBlock);
  Phi->addIncoming(PrevPhi, PrevIfBlock);
  Value *NewI = Builder.CreateSelect(Mask, Phi, Src0);
  CI->replaceAllUsesWith(NewI);
  CI->eraseFromParent();
}

//  ScalarizeMaskedStore() translates masked store intrinsic, like
// void @llvm.masked.store(<16 x i32> %src, <16 x i32>* %addr, i32 align,
//                               <16 x i1> %mask)
// to a chain of basic blocks, that stores element one-by-one if
// the appropriate mask bit is set
//
//   %1 = bitcast i8* %addr to i32*
//   %2 = extractelement <16 x i1> %mask, i32 0
//   %3 = icmp eq i1 %2, true
//   br i1 %3, label %cond.store, label %else
//
// cond.store:                                       ; preds = %0
//   %4 = extractelement <16 x i32> %val, i32 0
//   %5 = getelementptr i32* %1, i32 0
//   store i32 %4, i32* %5
//   br label %else
// 
// else:                                             ; preds = %0, %cond.store
//   %6 = extractelement <16 x i1> %mask, i32 1
//   %7 = icmp eq i1 %6, true
//   br i1 %7, label %cond.store1, label %else2
// 
// cond.store1:                                      ; preds = %else
//   %8 = extractelement <16 x i32> %val, i32 1
//   %9 = getelementptr i32* %1, i32 1
//   store i32 %8, i32* %9
//   br label %else2
//   . . .
static void ScalarizeMaskedStore(CallInst *CI) {
  Value *Ptr  = CI->getArgOperand(1);
  Value *Src = CI->getArgOperand(0);
  Value *Mask = CI->getArgOperand(3);

  VectorType *VecType = dyn_cast<VectorType>(Src->getType());
  Type *EltTy = VecType->getElementType();

  assert(VecType && "Unexpected data type in masked store intrinsic");

  IRBuilder<> Builder(CI->getContext());
  Instruction *InsertPt = CI;
  BasicBlock *IfBlock = CI->getParent();
  Builder.SetInsertPoint(InsertPt);
  Builder.SetCurrentDebugLocation(CI->getDebugLoc());

  // Bitcast %addr fron i8* to EltTy*
  Type *NewPtrType =
    EltTy->getPointerTo(cast<PointerType>(Ptr->getType())->getAddressSpace());
  Value *FirstEltPtr = Builder.CreateBitCast(Ptr, NewPtrType);

  unsigned VectorWidth = VecType->getNumElements();
  for (unsigned Idx = 0; Idx < VectorWidth; ++Idx) {

    // Fill the "else" block, created in the previous iteration
    //
    //  %mask_1 = extractelement <16 x i1> %mask, i32 Idx
    //  %to_store = icmp eq i1 %mask_1, true
    //  br i1 %to_load, label %cond.store, label %else
    //
    Value *Predicate = Builder.CreateExtractElement(Mask, Builder.getInt32(Idx));
    Value *Cmp = Builder.CreateICmp(ICmpInst::ICMP_EQ, Predicate,
                                    ConstantInt::get(Predicate->getType(), 1));

    // Create "cond" block
    //
    //  %OneElt = extractelement <16 x i32> %Src, i32 Idx
    //  %EltAddr = getelementptr i32* %1, i32 0
    //  %store i32 %OneElt, i32* %EltAddr
    //
    BasicBlock *CondBlock = IfBlock->splitBasicBlock(InsertPt, "cond.store");
    Builder.SetInsertPoint(InsertPt);
    
    Value *OneElt = Builder.CreateExtractElement(Src, Builder.getInt32(Idx));
    Value *Gep =
        Builder.CreateInBoundsGEP(EltTy, FirstEltPtr, Builder.getInt32(Idx));
    Builder.CreateStore(OneElt, Gep);

    // Create "else" block, fill it in the next iteration
    BasicBlock *NewIfBlock = CondBlock->splitBasicBlock(InsertPt, "else");
    Builder.SetInsertPoint(InsertPt);
    Instruction *OldBr = IfBlock->getTerminator();
    BranchInst::Create(CondBlock, NewIfBlock, Cmp, OldBr);
    OldBr->eraseFromParent();
    IfBlock = NewIfBlock;
  }
  CI->eraseFromParent();
}

bool CodeGenPrepare::OptimizeCallInst(CallInst *CI, bool& ModifiedDT) {
  BasicBlock *BB = CI->getParent();

  // Lower inline assembly if we can.
  // If we found an inline asm expession, and if the target knows how to
  // lower it to normal LLVM code, do so now.
  if (TLI && isa<InlineAsm>(CI->getCalledValue())) {
    if (TLI->ExpandInlineAsm(CI)) {
      // Avoid invalidating the iterator.
      CurInstIterator = BB->begin();
      // Avoid processing instructions out of order, which could cause
      // reuse before a value is defined.
      SunkAddrs.clear();
      return true;
    }
    // Sink address computing for memory operands into the block.
    if (OptimizeInlineAsmInst(CI))
      return true;
  }

  // Align the pointer arguments to this call if the target thinks it's a good
  // idea
  unsigned MinSize, PrefAlign;
  if (TLI && TLI->shouldAlignPointerArgs(CI, MinSize, PrefAlign)) {
    for (auto &Arg : CI->arg_operands()) {
      // We want to align both objects whose address is used directly and
      // objects whose address is used in casts and GEPs, though it only makes
      // sense for GEPs if the offset is a multiple of the desired alignment and
      // if size - offset meets the size threshold.
      if (!Arg->getType()->isPointerTy())
        continue;
      APInt Offset(DL->getPointerSizeInBits(
                       cast<PointerType>(Arg->getType())->getAddressSpace()),
                   0);
      Value *Val = Arg->stripAndAccumulateInBoundsConstantOffsets(*DL, Offset);
      uint64_t Offset2 = Offset.getLimitedValue();
      if ((Offset2 & (PrefAlign-1)) != 0)
        continue;
      AllocaInst *AI;
      if ((AI = dyn_cast<AllocaInst>(Val)) && AI->getAlignment() < PrefAlign &&
          DL->getTypeAllocSize(AI->getAllocatedType()) >= MinSize + Offset2)
        AI->setAlignment(PrefAlign);
      // Global variables can only be aligned if they are defined in this
      // object (i.e. they are uniquely initialized in this object), and
      // over-aligning global variables that have an explicit section is
      // forbidden.
      GlobalVariable *GV;
      if ((GV = dyn_cast<GlobalVariable>(Val)) && GV->hasUniqueInitializer() &&
          !GV->hasSection() && GV->getAlignment() < PrefAlign &&
          DL->getTypeAllocSize(GV->getType()->getElementType()) >=
              MinSize + Offset2)
        GV->setAlignment(PrefAlign);
    }
    // If this is a memcpy (or similar) then we may be able to improve the
    // alignment
    if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(CI)) {
      unsigned Align = getKnownAlignment(MI->getDest(), *DL);
      if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI))
        Align = std::min(Align, getKnownAlignment(MTI->getSource(), *DL));
      if (Align > MI->getAlignment())
        MI->setAlignment(ConstantInt::get(MI->getAlignmentType(), Align));
    }
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
  if (II) {
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::objectsize: {
      // Lower all uses of llvm.objectsize.*
      bool Min = (cast<ConstantInt>(II->getArgOperand(1))->getZExtValue() == 1);
      Type *ReturnTy = CI->getType();
      Constant *RetVal = ConstantInt::get(ReturnTy, Min ? 0 : -1ULL);

      // Substituting this can cause recursive simplifications, which can
      // invalidate our iterator.  Use a WeakVH to hold onto it in case this
      // happens.
      WeakVH IterHandle(CurInstIterator);

      replaceAndRecursivelySimplify(CI, RetVal,
                                    TLInfo, nullptr);

      // If the iterator instruction was recursively deleted, start over at the
      // start of the block.
      if (IterHandle != CurInstIterator) {
        CurInstIterator = BB->begin();
        SunkAddrs.clear();
      }
      return true;
    }
    case Intrinsic::masked_load: {
      // Scalarize unsupported vector masked load
      if (!TTI->isLegalMaskedLoad(CI->getType(), 1)) {
        ScalarizeMaskedLoad(CI);
        ModifiedDT = true;
        return true;
      }
      return false;
    }
    case Intrinsic::masked_store: {
      if (!TTI->isLegalMaskedStore(CI->getArgOperand(0)->getType(), 1)) {
        ScalarizeMaskedStore(CI);
        ModifiedDT = true;
        return true;
      }
      return false;
    }
#if 0 // HLSL Change - remove platform intrinsics
    case Intrinsic::aarch64_stlxr:
    case Intrinsic::aarch64_stxr: {
      ZExtInst *ExtVal = dyn_cast<ZExtInst>(CI->getArgOperand(0));
      if (!ExtVal || !ExtVal->hasOneUse() ||
          ExtVal->getParent() == CI->getParent())
        return false;
      // Sink a zext feeding stlxr/stxr before it, so it can be folded into it.
      ExtVal->moveBefore(CI);
      // Mark this instruction as "inserted by CGP", so that other
      // optimizations don't touch it.
      InsertedInsts.insert(ExtVal);
      return true;
    }
#endif // HLSL Change - remove platform intrinsics
    }

    if (TLI) {
      // Unknown address space.
      // TODO: Target hook to pick which address space the intrinsic cares
      // about?
      unsigned AddrSpace = ~0u;
      SmallVector<Value*, 2> PtrOps;
      Type *AccessTy;
      if (TLI->GetAddrModeArguments(II, PtrOps, AccessTy, AddrSpace))
        while (!PtrOps.empty())
          if (OptimizeMemoryInst(II, PtrOps.pop_back_val(), AccessTy, AddrSpace))
            return true;
    }
  }

  // From here on out we're working with named functions.
  if (!CI->getCalledFunction()) return false;

  // Lower all default uses of _chk calls.  This is very similar
  // to what InstCombineCalls does, but here we are only lowering calls
  // to fortified library functions (e.g. __memcpy_chk) that have the default
  // "don't know" as the objectsize.  Anything else should be left alone.
  FortifiedLibCallSimplifier Simplifier(TLInfo, true);
  if (Value *V = Simplifier.optimizeCall(CI)) {
    CI->replaceAllUsesWith(V);
    CI->eraseFromParent();
    return true;
  }
  return false;
}

/// DupRetToEnableTailCallOpts - Look for opportunities to duplicate return
/// instructions to the predecessor to enable tail call optimizations. The
/// case it is currently looking for is:
/// @code
/// bb0:
///   %tmp0 = tail call i32 @f0()
///   br label %return
/// bb1:
///   %tmp1 = tail call i32 @f1()
///   br label %return
/// bb2:
///   %tmp2 = tail call i32 @f2()
///   br label %return
/// return:
///   %retval = phi i32 [ %tmp0, %bb0 ], [ %tmp1, %bb1 ], [ %tmp2, %bb2 ]
///   ret i32 %retval
/// @endcode
///
/// =>
///
/// @code
/// bb0:
///   %tmp0 = tail call i32 @f0()
///   ret i32 %tmp0
/// bb1:
///   %tmp1 = tail call i32 @f1()
///   ret i32 %tmp1
/// bb2:
///   %tmp2 = tail call i32 @f2()
///   ret i32 %tmp2
/// @endcode
bool CodeGenPrepare::DupRetToEnableTailCallOpts(BasicBlock *BB) {
  if (!TLI)
    return false;

  ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator());
  if (!RI)
    return false;

  PHINode *PN = nullptr;
  BitCastInst *BCI = nullptr;
  Value *V = RI->getReturnValue();
  if (V) {
    BCI = dyn_cast<BitCastInst>(V);
    if (BCI)
      V = BCI->getOperand(0);

    PN = dyn_cast<PHINode>(V);
    if (!PN)
      return false;
  }

  if (PN && PN->getParent() != BB)
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  // See llvm::isInTailCallPosition().
  const Function *F = BB->getParent();
  AttributeSet CallerAttrs = F->getAttributes();
  if (CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt) ||
      CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
    return false;

  // Make sure there are no instructions between the PHI and return, or that the
  // return is the first instruction in the block.
  if (PN) {
    BasicBlock::iterator BI = BB->begin();
    do { ++BI; } while (isa<DbgInfoIntrinsic>(BI));
    if (&*BI == BCI)
      // Also skip over the bitcast.
      ++BI;
    if (&*BI != RI)
      return false;
  } else {
    BasicBlock::iterator BI = BB->begin();
    while (isa<DbgInfoIntrinsic>(BI)) ++BI;
    if (&*BI != RI)
      return false;
  }

  /// Only dup the ReturnInst if the CallInst is likely to be emitted as a tail
  /// call.
  SmallVector<CallInst*, 4> TailCalls;
  if (PN) {
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(PN->getIncomingValue(I));
      // Make sure the phi value is indeed produced by the tail call.
      if (CI && CI->hasOneUse() && CI->getParent() == PN->getIncomingBlock(I) &&
          TLI->mayBeEmittedAsTailCall(CI))
        TailCalls.push_back(CI);
    }
  } else {
    SmallPtrSet<BasicBlock*, 4> VisitedBBs;
    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
      if (!VisitedBBs.insert(*PI).second)
        continue;

      BasicBlock::InstListType &InstList = (*PI)->getInstList();
      BasicBlock::InstListType::reverse_iterator RI = InstList.rbegin();
      BasicBlock::InstListType::reverse_iterator RE = InstList.rend();
      do { ++RI; } while (RI != RE && isa<DbgInfoIntrinsic>(&*RI));
      if (RI == RE)
        continue;

      CallInst *CI = dyn_cast<CallInst>(&*RI);
      if (CI && CI->use_empty() && TLI->mayBeEmittedAsTailCall(CI))
        TailCalls.push_back(CI);
    }
  }

  bool Changed = false;
  for (unsigned i = 0, e = TailCalls.size(); i != e; ++i) {
    CallInst *CI = TailCalls[i];
    CallSite CS(CI);

    // Conservatively require the attributes of the call to match those of the
    // return. Ignore noalias because it doesn't affect the call sequence.
    AttributeSet CalleeAttrs = CS.getAttributes();
    if (AttrBuilder(CalleeAttrs, AttributeSet::ReturnIndex).
          removeAttribute(Attribute::NoAlias) !=
        AttrBuilder(CalleeAttrs, AttributeSet::ReturnIndex).
          removeAttribute(Attribute::NoAlias))
      continue;

    // Make sure the call instruction is followed by an unconditional branch to
    // the return block.
    BasicBlock *CallBB = CI->getParent();
    BranchInst *BI = dyn_cast<BranchInst>(CallBB->getTerminator());
    if (!BI || !BI->isUnconditional() || BI->getSuccessor(0) != BB)
      continue;

    // Duplicate the return into CallBB.
    (void)FoldReturnIntoUncondBranch(RI, BB, CallBB);
    ModifiedDT = Changed = true;
    ++NumRetsDup;
  }

  // If we eliminated all predecessors of the block, delete the block now.
  if (Changed && !BB->hasAddressTaken() && pred_begin(BB) == pred_end(BB))
    BB->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Memory Optimization
//===----------------------------------------------------------------------===//

namespace {

/// ExtAddrMode - This is an extended version of TargetLowering::AddrMode
/// which holds actual Value*'s for register values.
struct ExtAddrMode : public TargetLowering::AddrMode {
  Value *BaseReg;
  Value *ScaledReg;
  ExtAddrMode() : BaseReg(nullptr), ScaledReg(nullptr) {}
  void print(raw_ostream &OS) const;
  void dump() const;

  bool operator==(const ExtAddrMode& O) const {
    return (BaseReg == O.BaseReg) && (ScaledReg == O.ScaledReg) &&
           (BaseGV == O.BaseGV) && (BaseOffs == O.BaseOffs) &&
           (HasBaseReg == O.HasBaseReg) && (Scale == O.Scale);
  }
};

#ifndef NDEBUG
static inline raw_ostream &operator<<(raw_ostream &OS, const ExtAddrMode &AM) {
  AM.print(OS);
  return OS;
}
#endif

void ExtAddrMode::print(raw_ostream &OS) const {
  bool NeedPlus = false;
  OS << "[";
  if (BaseGV) {
    OS << (NeedPlus ? " + " : "")
       << "GV:";
    BaseGV->printAsOperand(OS, /*PrintType=*/false);
    NeedPlus = true;
  }

  if (BaseOffs) {
    OS << (NeedPlus ? " + " : "")
       << BaseOffs;
    NeedPlus = true;
  }

  if (BaseReg) {
    OS << (NeedPlus ? " + " : "")
       << "Base:";
    BaseReg->printAsOperand(OS, /*PrintType=*/false);
    NeedPlus = true;
  }
  if (Scale) {
    OS << (NeedPlus ? " + " : "")
       << Scale << "*";
    ScaledReg->printAsOperand(OS, /*PrintType=*/false);
  }

  OS << ']';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ExtAddrMode::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif

/// \brief This class provides transaction based operation on the IR.
/// Every change made through this class is recorded in the internal state and
/// can be undone (rollback) until commit is called.
class TypePromotionTransaction {

  /// \brief This represents the common interface of the individual transaction.
  /// Each class implements the logic for doing one specific modification on
  /// the IR via the TypePromotionTransaction.
  class TypePromotionAction {
  protected:
    /// The Instruction modified.
    Instruction *Inst;

  public:
    /// \brief Constructor of the action.
    /// The constructor performs the related action on the IR.
    TypePromotionAction(Instruction *Inst) : Inst(Inst) {}

    virtual ~TypePromotionAction() {}

    /// \brief Undo the modification done by this action.
    /// When this method is called, the IR must be in the same state as it was
    /// before this action was applied.
    /// \pre Undoing the action works if and only if the IR is in the exact same
    /// state as it was directly after this action was applied.
    virtual void undo() = 0;

    /// \brief Advocate every change made by this action.
    /// When the results on the IR of the action are to be kept, it is important
    /// to call this function, otherwise hidden information may be kept forever.
    virtual void commit() {
      // Nothing to be done, this action is not doing anything.
    }
  };

  /// \brief Utility to remember the position of an instruction.
  class InsertionHandler {
    /// Position of an instruction.
    /// Either an instruction:
    /// - Is the first in a basic block: BB is used.
    /// - Has a previous instructon: PrevInst is used.
    union {
      Instruction *PrevInst;
      BasicBlock *BB;
    } Point;
    /// Remember whether or not the instruction had a previous instruction.
    bool HasPrevInstruction;

  public:
    /// \brief Record the position of \p Inst.
    InsertionHandler(Instruction *Inst) {
      BasicBlock::iterator It = Inst;
      HasPrevInstruction = (It != (Inst->getParent()->begin()));
      if (HasPrevInstruction)
        Point.PrevInst = --It;
      else
        Point.BB = Inst->getParent();
    }

    /// \brief Insert \p Inst at the recorded position.
    void insert(Instruction *Inst) {
      if (HasPrevInstruction) {
        if (Inst->getParent())
          Inst->removeFromParent();
        Inst->insertAfter(Point.PrevInst);
      } else {
        Instruction *Position = Point.BB->getFirstInsertionPt();
        if (Inst->getParent())
          Inst->moveBefore(Position);
        else
          Inst->insertBefore(Position);
      }
    }
  };

  /// \brief Move an instruction before another.
  class InstructionMoveBefore : public TypePromotionAction {
    /// Original position of the instruction.
    InsertionHandler Position;

  public:
    /// \brief Move \p Inst before \p Before.
    InstructionMoveBefore(Instruction *Inst, Instruction *Before)
        : TypePromotionAction(Inst), Position(Inst) {
      DEBUG(dbgs() << "Do: move: " << *Inst << "\nbefore: " << *Before << "\n");
      Inst->moveBefore(Before);
    }

    /// \brief Move the instruction back to its original position.
    void undo() override {
      DEBUG(dbgs() << "Undo: moveBefore: " << *Inst << "\n");
      Position.insert(Inst);
    }
  };

  /// \brief Set the operand of an instruction with a new value.
  class OperandSetter : public TypePromotionAction {
    /// Original operand of the instruction.
    Value *Origin;
    /// Index of the modified instruction.
    unsigned Idx;

  public:
    /// \brief Set \p Idx operand of \p Inst with \p NewVal.
    OperandSetter(Instruction *Inst, unsigned Idx, Value *NewVal)
        : TypePromotionAction(Inst), Idx(Idx) {
      DEBUG(dbgs() << "Do: setOperand: " << Idx << "\n"
                   << "for:" << *Inst << "\n"
                   << "with:" << *NewVal << "\n");
      Origin = Inst->getOperand(Idx);
      Inst->setOperand(Idx, NewVal);
    }

    /// \brief Restore the original value of the instruction.
    void undo() override {
      DEBUG(dbgs() << "Undo: setOperand:" << Idx << "\n"
                   << "for: " << *Inst << "\n"
                   << "with: " << *Origin << "\n");
      Inst->setOperand(Idx, Origin);
    }
  };

  /// \brief Hide the operands of an instruction.
  /// Do as if this instruction was not using any of its operands.
  class OperandsHider : public TypePromotionAction {
    /// The list of original operands.
    SmallVector<Value *, 4> OriginalValues;

  public:
    /// \brief Remove \p Inst from the uses of the operands of \p Inst.
    OperandsHider(Instruction *Inst) : TypePromotionAction(Inst) {
      DEBUG(dbgs() << "Do: OperandsHider: " << *Inst << "\n");
      unsigned NumOpnds = Inst->getNumOperands();
      OriginalValues.reserve(NumOpnds);
      for (unsigned It = 0; It < NumOpnds; ++It) {
        // Save the current operand.
        Value *Val = Inst->getOperand(It);
        OriginalValues.push_back(Val);
        // Set a dummy one.
        // We could use OperandSetter here, but that would implied an overhead
        // that we are not willing to pay.
        Inst->setOperand(It, UndefValue::get(Val->getType()));
      }
    }

    /// \brief Restore the original list of uses.
    void undo() override {
      DEBUG(dbgs() << "Undo: OperandsHider: " << *Inst << "\n");
      for (unsigned It = 0, EndIt = OriginalValues.size(); It != EndIt; ++It)
        Inst->setOperand(It, OriginalValues[It]);
    }
  };

  /// \brief Build a truncate instruction.
  class TruncBuilder : public TypePromotionAction {
    Value *Val;
  public:
    /// \brief Build a truncate instruction of \p Opnd producing a \p Ty
    /// result.
    /// trunc Opnd to Ty.
    TruncBuilder(Instruction *Opnd, Type *Ty) : TypePromotionAction(Opnd) {
      IRBuilder<> Builder(Opnd);
      Val = Builder.CreateTrunc(Opnd, Ty, "promoted");
      DEBUG(dbgs() << "Do: TruncBuilder: " << *Val << "\n");
    }

    /// \brief Get the built value.
    Value *getBuiltValue() { return Val; }

    /// \brief Remove the built instruction.
    void undo() override {
      DEBUG(dbgs() << "Undo: TruncBuilder: " << *Val << "\n");
      if (Instruction *IVal = dyn_cast<Instruction>(Val))
        IVal->eraseFromParent();
    }
  };

  /// \brief Build a sign extension instruction.
  class SExtBuilder : public TypePromotionAction {
    Value *Val;
  public:
    /// \brief Build a sign extension instruction of \p Opnd producing a \p Ty
    /// result.
    /// sext Opnd to Ty.
    SExtBuilder(Instruction *InsertPt, Value *Opnd, Type *Ty)
        : TypePromotionAction(InsertPt) {
      IRBuilder<> Builder(InsertPt);
      Val = Builder.CreateSExt(Opnd, Ty, "promoted");
      DEBUG(dbgs() << "Do: SExtBuilder: " << *Val << "\n");
    }

    /// \brief Get the built value.
    Value *getBuiltValue() { return Val; }

    /// \brief Remove the built instruction.
    void undo() override {
      DEBUG(dbgs() << "Undo: SExtBuilder: " << *Val << "\n");
      if (Instruction *IVal = dyn_cast<Instruction>(Val))
        IVal->eraseFromParent();
    }
  };

  /// \brief Build a zero extension instruction.
  class ZExtBuilder : public TypePromotionAction {
    Value *Val;
  public:
    /// \brief Build a zero extension instruction of \p Opnd producing a \p Ty
    /// result.
    /// zext Opnd to Ty.
    ZExtBuilder(Instruction *InsertPt, Value *Opnd, Type *Ty)
        : TypePromotionAction(InsertPt) {
      IRBuilder<> Builder(InsertPt);
      Val = Builder.CreateZExt(Opnd, Ty, "promoted");
      DEBUG(dbgs() << "Do: ZExtBuilder: " << *Val << "\n");
    }

    /// \brief Get the built value.
    Value *getBuiltValue() { return Val; }

    /// \brief Remove the built instruction.
    void undo() override {
      DEBUG(dbgs() << "Undo: ZExtBuilder: " << *Val << "\n");
      if (Instruction *IVal = dyn_cast<Instruction>(Val))
        IVal->eraseFromParent();
    }
  };

  /// \brief Mutate an instruction to another type.
  class TypeMutator : public TypePromotionAction {
    /// Record the original type.
    Type *OrigTy;

  public:
    /// \brief Mutate the type of \p Inst into \p NewTy.
    TypeMutator(Instruction *Inst, Type *NewTy)
        : TypePromotionAction(Inst), OrigTy(Inst->getType()) {
      DEBUG(dbgs() << "Do: MutateType: " << *Inst << " with " << *NewTy
                   << "\n");
      Inst->mutateType(NewTy);
    }

    /// \brief Mutate the instruction back to its original type.
    void undo() override {
      DEBUG(dbgs() << "Undo: MutateType: " << *Inst << " with " << *OrigTy
                   << "\n");
      Inst->mutateType(OrigTy);
    }
  };

  /// \brief Replace the uses of an instruction by another instruction.
  class UsesReplacer : public TypePromotionAction {
    /// Helper structure to keep track of the replaced uses.
    struct InstructionAndIdx {
      /// The instruction using the instruction.
      Instruction *Inst;
      /// The index where this instruction is used for Inst.
      unsigned Idx;
      InstructionAndIdx(Instruction *Inst, unsigned Idx)
          : Inst(Inst), Idx(Idx) {}
    };

    /// Keep track of the original uses (pair Instruction, Index).
    SmallVector<InstructionAndIdx, 4> OriginalUses;
    typedef SmallVectorImpl<InstructionAndIdx>::iterator use_iterator;

  public:
    /// \brief Replace all the use of \p Inst by \p New.
    UsesReplacer(Instruction *Inst, Value *New) : TypePromotionAction(Inst) {
      DEBUG(dbgs() << "Do: UsersReplacer: " << *Inst << " with " << *New
                   << "\n");
      // Record the original uses.
      for (Use &U : Inst->uses()) {
        Instruction *UserI = cast<Instruction>(U.getUser());
        OriginalUses.push_back(InstructionAndIdx(UserI, U.getOperandNo()));
      }
      // Now, we can replace the uses.
      Inst->replaceAllUsesWith(New);
    }

    /// \brief Reassign the original uses of Inst to Inst.
    void undo() override {
      DEBUG(dbgs() << "Undo: UsersReplacer: " << *Inst << "\n");
      for (use_iterator UseIt = OriginalUses.begin(),
                        EndIt = OriginalUses.end();
           UseIt != EndIt; ++UseIt) {
        UseIt->Inst->setOperand(UseIt->Idx, Inst);
      }
    }
  };

  /// \brief Remove an instruction from the IR.
  class InstructionRemover : public TypePromotionAction {
    /// Original position of the instruction.
    InsertionHandler Inserter;
    /// Helper structure to hide all the link to the instruction. In other
    /// words, this helps to do as if the instruction was removed.
    OperandsHider Hider;
    /// Keep track of the uses replaced, if any.
    UsesReplacer *Replacer;

  public:
    /// \brief Remove all reference of \p Inst and optinally replace all its
    /// uses with New.
    /// \pre If !Inst->use_empty(), then New != nullptr
    InstructionRemover(Instruction *Inst, Value *New = nullptr)
        : TypePromotionAction(Inst), Inserter(Inst), Hider(Inst),
          Replacer(nullptr) {
      if (New)
        Replacer = new UsesReplacer(Inst, New);
      DEBUG(dbgs() << "Do: InstructionRemover: " << *Inst << "\n");
      Inst->removeFromParent();
    }

    ~InstructionRemover() override { delete Replacer; }

    /// \brief Really remove the instruction.
    void commit() override { delete Inst; }

    /// \brief Resurrect the instruction and reassign it to the proper uses if
    /// new value was provided when build this action.
    void undo() override {
      DEBUG(dbgs() << "Undo: InstructionRemover: " << *Inst << "\n");
      Inserter.insert(Inst);
      if (Replacer)
        Replacer->undo();
      Hider.undo();
    }
  };

public:
  /// Restoration point.
  /// The restoration point is a pointer to an action instead of an iterator
  /// because the iterator may be invalidated but not the pointer.
  typedef const TypePromotionAction *ConstRestorationPt;
  /// Advocate every changes made in that transaction.
  void commit();
  /// Undo all the changes made after the given point.
  void rollback(ConstRestorationPt Point);
  /// Get the current restoration point.
  ConstRestorationPt getRestorationPoint() const;

  /// \name API for IR modification with state keeping to support rollback.
  /// @{
  /// Same as Instruction::setOperand.
  void setOperand(Instruction *Inst, unsigned Idx, Value *NewVal);
  /// Same as Instruction::eraseFromParent.
  void eraseInstruction(Instruction *Inst, Value *NewVal = nullptr);
  /// Same as Value::replaceAllUsesWith.
  void replaceAllUsesWith(Instruction *Inst, Value *New);
  /// Same as Value::mutateType.
  void mutateType(Instruction *Inst, Type *NewTy);
  /// Same as IRBuilder::createTrunc.
  Value *createTrunc(Instruction *Opnd, Type *Ty);
  /// Same as IRBuilder::createSExt.
  Value *createSExt(Instruction *Inst, Value *Opnd, Type *Ty);
  /// Same as IRBuilder::createZExt.
  Value *createZExt(Instruction *Inst, Value *Opnd, Type *Ty);
  /// Same as Instruction::moveBefore.
  void moveBefore(Instruction *Inst, Instruction *Before);
  /// @}

private:
  /// The ordered list of actions made so far.
  SmallVector<std::unique_ptr<TypePromotionAction>, 16> Actions;
  typedef SmallVectorImpl<std::unique_ptr<TypePromotionAction>>::iterator CommitPt;
};

void TypePromotionTransaction::setOperand(Instruction *Inst, unsigned Idx,
                                          Value *NewVal) {
  Actions.push_back(
      make_unique<TypePromotionTransaction::OperandSetter>(Inst, Idx, NewVal));
}

void TypePromotionTransaction::eraseInstruction(Instruction *Inst,
                                                Value *NewVal) {
  Actions.push_back(
      make_unique<TypePromotionTransaction::InstructionRemover>(Inst, NewVal));
}

void TypePromotionTransaction::replaceAllUsesWith(Instruction *Inst,
                                                  Value *New) {
  Actions.push_back(make_unique<TypePromotionTransaction::UsesReplacer>(Inst, New));
}

void TypePromotionTransaction::mutateType(Instruction *Inst, Type *NewTy) {
  Actions.push_back(make_unique<TypePromotionTransaction::TypeMutator>(Inst, NewTy));
}

Value *TypePromotionTransaction::createTrunc(Instruction *Opnd,
                                             Type *Ty) {
  std::unique_ptr<TruncBuilder> Ptr(new TruncBuilder(Opnd, Ty));
  Value *Val = Ptr->getBuiltValue();
  Actions.push_back(std::move(Ptr));
  return Val;
}

Value *TypePromotionTransaction::createSExt(Instruction *Inst,
                                            Value *Opnd, Type *Ty) {
  std::unique_ptr<SExtBuilder> Ptr(new SExtBuilder(Inst, Opnd, Ty));
  Value *Val = Ptr->getBuiltValue();
  Actions.push_back(std::move(Ptr));
  return Val;
}

Value *TypePromotionTransaction::createZExt(Instruction *Inst,
                                            Value *Opnd, Type *Ty) {
  std::unique_ptr<ZExtBuilder> Ptr(new ZExtBuilder(Inst, Opnd, Ty));
  Value *Val = Ptr->getBuiltValue();
  Actions.push_back(std::move(Ptr));
  return Val;
}

void TypePromotionTransaction::moveBefore(Instruction *Inst,
                                          Instruction *Before) {
  Actions.push_back(
      make_unique<TypePromotionTransaction::InstructionMoveBefore>(Inst, Before));
}

TypePromotionTransaction::ConstRestorationPt
TypePromotionTransaction::getRestorationPoint() const {
  return !Actions.empty() ? Actions.back().get() : nullptr;
}

void TypePromotionTransaction::commit() {
  for (CommitPt It = Actions.begin(), EndIt = Actions.end(); It != EndIt;
       ++It)
    (*It)->commit();
  Actions.clear();
}

void TypePromotionTransaction::rollback(
    TypePromotionTransaction::ConstRestorationPt Point) {
  while (!Actions.empty() && Point != Actions.back().get()) {
    std::unique_ptr<TypePromotionAction> Curr = Actions.pop_back_val();
    Curr->undo();
  }
}

/// \brief A helper class for matching addressing modes.
///
/// This encapsulates the logic for matching the target-legal addressing modes.
class AddressingModeMatcher {
  SmallVectorImpl<Instruction*> &AddrModeInsts;
  const TargetMachine &TM;
  const TargetLowering &TLI;
  const DataLayout &DL;

  /// AccessTy/MemoryInst - This is the type for the access (e.g. double) and
  /// the memory instruction that we're computing this address for.
  Type *AccessTy;
  unsigned AddrSpace;
  Instruction *MemoryInst;

  /// AddrMode - This is the addressing mode that we're building up.  This is
  /// part of the return value of this addressing mode matching stuff.
  ExtAddrMode &AddrMode;

  /// The instructions inserted by other CodeGenPrepare optimizations.
  const SetOfInstrs &InsertedInsts;
  /// A map from the instructions to their type before promotion.
  InstrToOrigTy &PromotedInsts;
  /// The ongoing transaction where every action should be registered.
  TypePromotionTransaction &TPT;

  /// IgnoreProfitability - This is set to true when we should not do
  /// profitability checks.  When true, IsProfitableToFoldIntoAddressingMode
  /// always returns true.
  bool IgnoreProfitability;

  AddressingModeMatcher(SmallVectorImpl<Instruction *> &AMI,
                        const TargetMachine &TM, Type *AT, unsigned AS,
                        Instruction *MI, ExtAddrMode &AM,
                        const SetOfInstrs &InsertedInsts,
                        InstrToOrigTy &PromotedInsts,
                        TypePromotionTransaction &TPT)
      : AddrModeInsts(AMI), TM(TM),
        TLI(*TM.getSubtargetImpl(*MI->getParent()->getParent())
                 ->getTargetLowering()),
        DL(MI->getModule()->getDataLayout()), AccessTy(AT), AddrSpace(AS),
        MemoryInst(MI), AddrMode(AM), InsertedInsts(InsertedInsts),
        PromotedInsts(PromotedInsts), TPT(TPT) {
    IgnoreProfitability = false;
  }
public:

  /// Match - Find the maximal addressing mode that a load/store of V can fold,
  /// give an access type of AccessTy.  This returns a list of involved
  /// instructions in AddrModeInsts.
  /// \p InsertedInsts The instructions inserted by other CodeGenPrepare
  /// optimizations.
  /// \p PromotedInsts maps the instructions to their type before promotion.
  /// \p The ongoing transaction where every action should be registered.
  static ExtAddrMode Match(Value *V, Type *AccessTy, unsigned AS,
                           Instruction *MemoryInst,
                           SmallVectorImpl<Instruction*> &AddrModeInsts,
                           const TargetMachine &TM,
                           const SetOfInstrs &InsertedInsts,
                           InstrToOrigTy &PromotedInsts,
                           TypePromotionTransaction &TPT) {
    ExtAddrMode Result;

    bool Success = AddressingModeMatcher(AddrModeInsts, TM, AccessTy, AS,
                                         MemoryInst, Result, InsertedInsts,
                                         PromotedInsts, TPT).MatchAddr(V, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");
    return Result;
  }
private:
  bool MatchScaledValue(Value *ScaleReg, int64_t Scale, unsigned Depth);
  bool MatchAddr(Value *V, unsigned Depth);
  bool MatchOperationAddr(User *Operation, unsigned Opcode, unsigned Depth,
                          bool *MovedAway = nullptr);
  bool IsProfitableToFoldIntoAddressingMode(Instruction *I,
                                            ExtAddrMode &AMBefore,
                                            ExtAddrMode &AMAfter);
  bool ValueAlreadyLiveAtInst(Value *Val, Value *KnownLive1, Value *KnownLive2);
  bool IsPromotionProfitable(unsigned NewCost, unsigned OldCost,
                             Value *PromotedOperand) const;
};

/// MatchScaledValue - Try adding ScaleReg*Scale to the current addressing mode.
/// Return true and update AddrMode if this addr mode is legal for the target,
/// false if not.
bool AddressingModeMatcher::MatchScaledValue(Value *ScaleReg, int64_t Scale,
                                             unsigned Depth) {
  // If Scale is 1, then this is the same as adding ScaleReg to the addressing
  // mode.  Just process that directly.
  if (Scale == 1)
    return MatchAddr(ScaleReg, Depth);

  // If the scale is 0, it takes nothing to add this.
  if (Scale == 0)
    return true;

  // If we already have a scale of this value, we can add to it, otherwise, we
  // need an available scale field.
  if (AddrMode.Scale != 0 && AddrMode.ScaledReg != ScaleReg)
    return false;

  ExtAddrMode TestAddrMode = AddrMode;

  // Add scale to turn X*4+X*3 -> X*7.  This could also do things like
  // [A+B + A*7] -> [B+A*8].
  TestAddrMode.Scale += Scale;
  TestAddrMode.ScaledReg = ScaleReg;

  // If the new address isn't legal, bail out.
  if (!TLI.isLegalAddressingMode(DL, TestAddrMode, AccessTy, AddrSpace))
    return false;

  // It was legal, so commit it.
  AddrMode = TestAddrMode;

  // Okay, we decided that we can add ScaleReg+Scale to AddrMode.  Check now
  // to see if ScaleReg is actually X+C.  If so, we can turn this into adding
  // X*Scale + C*Scale to addr mode.
  ConstantInt *CI = nullptr; Value *AddLHS = nullptr;
  if (isa<Instruction>(ScaleReg) &&  // not a constant expr.
      match(ScaleReg, m_Add(m_Value(AddLHS), m_ConstantInt(CI)))) {
    TestAddrMode.ScaledReg = AddLHS;
    TestAddrMode.BaseOffs += CI->getSExtValue()*TestAddrMode.Scale;

    // If this addressing mode is legal, commit it and remember that we folded
    // this instruction.
    if (TLI.isLegalAddressingMode(DL, TestAddrMode, AccessTy, AddrSpace)) {
      AddrModeInsts.push_back(cast<Instruction>(ScaleReg));
      AddrMode = TestAddrMode;
      return true;
    }
  }

  // Otherwise, not (x+c)*scale, just return what we have.
  return true;
}

/// MightBeFoldableInst - This is a little filter, which returns true if an
/// addressing computation involving I might be folded into a load/store
/// accessing it.  This doesn't need to be perfect, but needs to accept at least
/// the set of instructions that MatchOperationAddr can.
static bool MightBeFoldableInst(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    // Don't touch identity bitcasts.
    if (I->getType() == I->getOperand(0)->getType())
      return false;
    return I->getType()->isPointerTy() || I->getType()->isIntegerTy();
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return true;
  case Instruction::IntToPtr:
    // We know the input is intptr_t, so this is foldable.
    return true;
  case Instruction::Add:
    return true;
  case Instruction::Mul:
  case Instruction::Shl:
    // Can only handle X*C and X << C.
    return isa<ConstantInt>(I->getOperand(1));
  case Instruction::GetElementPtr:
    return true;
  default:
    return false;
  }
}

/// \brief Check whether or not \p Val is a legal instruction for \p TLI.
/// \note \p Val is assumed to be the product of some type promotion.
/// Therefore if \p Val has an undefined state in \p TLI, this is assumed
/// to be legal, as the non-promoted value would have had the same state.
static bool isPromotedInstructionLegal(const TargetLowering &TLI,
                                       const DataLayout &DL, Value *Val) {
  Instruction *PromotedInst = dyn_cast<Instruction>(Val);
  if (!PromotedInst)
    return false;
  int ISDOpcode = TLI.InstructionOpcodeToISD(PromotedInst->getOpcode());
  // If the ISDOpcode is undefined, it was undefined before the promotion.
  if (!ISDOpcode)
    return true;
  // Otherwise, check if the promoted instruction is legal or not.
  return TLI.isOperationLegalOrCustom(
      ISDOpcode, TLI.getValueType(DL, PromotedInst->getType()));
}

/// \brief Hepler class to perform type promotion.
class TypePromotionHelper {
  /// \brief Utility function to check whether or not a sign or zero extension
  /// of \p Inst with \p ConsideredExtType can be moved through \p Inst by
  /// either using the operands of \p Inst or promoting \p Inst.
  /// The type of the extension is defined by \p IsSExt.
  /// In other words, check if:
  /// ext (Ty Inst opnd1 opnd2 ... opndN) to ConsideredExtType.
  /// #1 Promotion applies:
  /// ConsideredExtType Inst (ext opnd1 to ConsideredExtType, ...).
  /// #2 Operand reuses:
  /// ext opnd1 to ConsideredExtType.
  /// \p PromotedInsts maps the instructions to their type before promotion.
  static bool canGetThrough(const Instruction *Inst, Type *ConsideredExtType,
                            const InstrToOrigTy &PromotedInsts, bool IsSExt);

  /// \brief Utility function to determine if \p OpIdx should be promoted when
  /// promoting \p Inst.
  static bool shouldExtOperand(const Instruction *Inst, int OpIdx) {
    if (isa<SelectInst>(Inst) && OpIdx == 0)
      return false;
    return true;
  }

  /// \brief Utility function to promote the operand of \p Ext when this
  /// operand is a promotable trunc or sext or zext.
  /// \p PromotedInsts maps the instructions to their type before promotion.
  /// \p CreatedInstsCost[out] contains the cost of all instructions
  /// created to promote the operand of Ext.
  /// Newly added extensions are inserted in \p Exts.
  /// Newly added truncates are inserted in \p Truncs.
  /// Should never be called directly.
  /// \return The promoted value which is used instead of Ext.
  static Value *promoteOperandForTruncAndAnyExt(
      Instruction *Ext, TypePromotionTransaction &TPT,
      InstrToOrigTy &PromotedInsts, unsigned &CreatedInstsCost,
      SmallVectorImpl<Instruction *> *Exts,
      SmallVectorImpl<Instruction *> *Truncs, const TargetLowering &TLI);

  /// \brief Utility function to promote the operand of \p Ext when this
  /// operand is promotable and is not a supported trunc or sext.
  /// \p PromotedInsts maps the instructions to their type before promotion.
  /// \p CreatedInstsCost[out] contains the cost of all the instructions
  /// created to promote the operand of Ext.
  /// Newly added extensions are inserted in \p Exts.
  /// Newly added truncates are inserted in \p Truncs.
  /// Should never be called directly.
  /// \return The promoted value which is used instead of Ext.
  static Value *promoteOperandForOther(Instruction *Ext,
                                       TypePromotionTransaction &TPT,
                                       InstrToOrigTy &PromotedInsts,
                                       unsigned &CreatedInstsCost,
                                       SmallVectorImpl<Instruction *> *Exts,
                                       SmallVectorImpl<Instruction *> *Truncs,
                                       const TargetLowering &TLI, bool IsSExt);

  /// \see promoteOperandForOther.
  static Value *signExtendOperandForOther(
      Instruction *Ext, TypePromotionTransaction &TPT,
      InstrToOrigTy &PromotedInsts, unsigned &CreatedInstsCost,
      SmallVectorImpl<Instruction *> *Exts,
      SmallVectorImpl<Instruction *> *Truncs, const TargetLowering &TLI) {
    return promoteOperandForOther(Ext, TPT, PromotedInsts, CreatedInstsCost,
                                  Exts, Truncs, TLI, true);
  }

  /// \see promoteOperandForOther.
  static Value *zeroExtendOperandForOther(
      Instruction *Ext, TypePromotionTransaction &TPT,
      InstrToOrigTy &PromotedInsts, unsigned &CreatedInstsCost,
      SmallVectorImpl<Instruction *> *Exts,
      SmallVectorImpl<Instruction *> *Truncs, const TargetLowering &TLI) {
    return promoteOperandForOther(Ext, TPT, PromotedInsts, CreatedInstsCost,
                                  Exts, Truncs, TLI, false);
  }

public:
  /// Type for the utility function that promotes the operand of Ext.
  typedef Value *(*Action)(Instruction *Ext, TypePromotionTransaction &TPT,
                           InstrToOrigTy &PromotedInsts,
                           unsigned &CreatedInstsCost,
                           SmallVectorImpl<Instruction *> *Exts,
                           SmallVectorImpl<Instruction *> *Truncs,
                           const TargetLowering &TLI);
  /// \brief Given a sign/zero extend instruction \p Ext, return the approriate
  /// action to promote the operand of \p Ext instead of using Ext.
  /// \return NULL if no promotable action is possible with the current
  /// sign extension.
  /// \p InsertedInsts keeps track of all the instructions inserted by the
  /// other CodeGenPrepare optimizations. This information is important
  /// because we do not want to promote these instructions as CodeGenPrepare
  /// will reinsert them later. Thus creating an infinite loop: create/remove.
  /// \p PromotedInsts maps the instructions to their type before promotion.
  static Action getAction(Instruction *Ext, const SetOfInstrs &InsertedInsts,
                          const TargetLowering &TLI,
                          const InstrToOrigTy &PromotedInsts);
};

bool TypePromotionHelper::canGetThrough(const Instruction *Inst,
                                        Type *ConsideredExtType,
                                        const InstrToOrigTy &PromotedInsts,
                                        bool IsSExt) {
  // The promotion helper does not know how to deal with vector types yet.
  // To be able to fix that, we would need to fix the places where we
  // statically extend, e.g., constants and such.
  if (Inst->getType()->isVectorTy())
    return false;

  // We can always get through zext.
  if (isa<ZExtInst>(Inst))
    return true;

  // sext(sext) is ok too.
  if (IsSExt && isa<SExtInst>(Inst))
    return true;

  // We can get through binary operator, if it is legal. In other words, the
  // binary operator must have a nuw or nsw flag.
  const BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst);
  if (BinOp && isa<OverflowingBinaryOperator>(BinOp) &&
      ((!IsSExt && BinOp->hasNoUnsignedWrap()) ||
       (IsSExt && BinOp->hasNoSignedWrap())))
    return true;

  // Check if we can do the following simplification.
  // ext(trunc(opnd)) --> ext(opnd)
  if (!isa<TruncInst>(Inst))
    return false;

  Value *OpndVal = Inst->getOperand(0);
  // Check if we can use this operand in the extension.
  // If the type is larger than the result type of the extension,
  // we cannot.
  if (!OpndVal->getType()->isIntegerTy() ||
      OpndVal->getType()->getIntegerBitWidth() >
          ConsideredExtType->getIntegerBitWidth())
    return false;

  // If the operand of the truncate is not an instruction, we will not have
  // any information on the dropped bits.
  // (Actually we could for constant but it is not worth the extra logic).
  Instruction *Opnd = dyn_cast<Instruction>(OpndVal);
  if (!Opnd)
    return false;

  // Check if the source of the type is narrow enough.
  // I.e., check that trunc just drops extended bits of the same kind of
  // the extension.
  // #1 get the type of the operand and check the kind of the extended bits.
  const Type *OpndType;
  InstrToOrigTy::const_iterator It = PromotedInsts.find(Opnd);
  if (It != PromotedInsts.end() && It->second.IsSExt == IsSExt)
    OpndType = It->second.Ty;
  else if ((IsSExt && isa<SExtInst>(Opnd)) || (!IsSExt && isa<ZExtInst>(Opnd)))
    OpndType = Opnd->getOperand(0)->getType();
  else
    return false;

  // #2 check that the truncate just drop extended bits.
  if (Inst->getType()->getIntegerBitWidth() >= OpndType->getIntegerBitWidth())
    return true;

  return false;
}

TypePromotionHelper::Action TypePromotionHelper::getAction(
    Instruction *Ext, const SetOfInstrs &InsertedInsts,
    const TargetLowering &TLI, const InstrToOrigTy &PromotedInsts) {
  assert((isa<SExtInst>(Ext) || isa<ZExtInst>(Ext)) &&
         "Unexpected instruction type");
  Instruction *ExtOpnd = dyn_cast<Instruction>(Ext->getOperand(0));
  Type *ExtTy = Ext->getType();
  bool IsSExt = isa<SExtInst>(Ext);
  // If the operand of the extension is not an instruction, we cannot
  // get through.
  // If it, check we can get through.
  if (!ExtOpnd || !canGetThrough(ExtOpnd, ExtTy, PromotedInsts, IsSExt))
    return nullptr;

  // Do not promote if the operand has been added by codegenprepare.
  // Otherwise, it means we are undoing an optimization that is likely to be
  // redone, thus causing potential infinite loop.
  if (isa<TruncInst>(ExtOpnd) && InsertedInsts.count(ExtOpnd))
    return nullptr;

  // SExt or Trunc instructions.
  // Return the related handler.
  if (isa<SExtInst>(ExtOpnd) || isa<TruncInst>(ExtOpnd) ||
      isa<ZExtInst>(ExtOpnd))
    return promoteOperandForTruncAndAnyExt;

  // Regular instruction.
  // Abort early if we will have to insert non-free instructions.
  if (!ExtOpnd->hasOneUse() && !TLI.isTruncateFree(ExtTy, ExtOpnd->getType()))
    return nullptr;
  return IsSExt ? signExtendOperandForOther : zeroExtendOperandForOther;
}

Value *TypePromotionHelper::promoteOperandForTruncAndAnyExt(
    llvm::Instruction *SExt, TypePromotionTransaction &TPT,
    InstrToOrigTy &PromotedInsts, unsigned &CreatedInstsCost,
    SmallVectorImpl<Instruction *> *Exts,
    SmallVectorImpl<Instruction *> *Truncs, const TargetLowering &TLI) {
  // By construction, the operand of SExt is an instruction. Otherwise we cannot
  // get through it and this method should not be called.
  Instruction *SExtOpnd = cast<Instruction>(SExt->getOperand(0));
  Value *ExtVal = SExt;
  bool HasMergedNonFreeExt = false;
  if (isa<ZExtInst>(SExtOpnd)) {
    // Replace s|zext(zext(opnd))
    // => zext(opnd).
    HasMergedNonFreeExt = !TLI.isExtFree(SExtOpnd);
    Value *ZExt =
        TPT.createZExt(SExt, SExtOpnd->getOperand(0), SExt->getType());
    TPT.replaceAllUsesWith(SExt, ZExt);
    TPT.eraseInstruction(SExt);
    ExtVal = ZExt;
  } else {
    // Replace z|sext(trunc(opnd)) or sext(sext(opnd))
    // => z|sext(opnd).
    TPT.setOperand(SExt, 0, SExtOpnd->getOperand(0));
  }
  CreatedInstsCost = 0;

  // Remove dead code.
  if (SExtOpnd->use_empty())
    TPT.eraseInstruction(SExtOpnd);

  // Check if the extension is still needed.
  Instruction *ExtInst = dyn_cast<Instruction>(ExtVal);
  if (!ExtInst || ExtInst->getType() != ExtInst->getOperand(0)->getType()) {
    if (ExtInst) {
      if (Exts)
        Exts->push_back(ExtInst);
      CreatedInstsCost = !TLI.isExtFree(ExtInst) && !HasMergedNonFreeExt;
    }
    return ExtVal;
  }

  // At this point we have: ext ty opnd to ty.
  // Reassign the uses of ExtInst to the opnd and remove ExtInst.
  Value *NextVal = ExtInst->getOperand(0);
  TPT.eraseInstruction(ExtInst, NextVal);
  return NextVal;
}

Value *TypePromotionHelper::promoteOperandForOther(
    Instruction *Ext, TypePromotionTransaction &TPT,
    InstrToOrigTy &PromotedInsts, unsigned &CreatedInstsCost,
    SmallVectorImpl<Instruction *> *Exts,
    SmallVectorImpl<Instruction *> *Truncs, const TargetLowering &TLI,
    bool IsSExt) {
  // By construction, the operand of Ext is an instruction. Otherwise we cannot
  // get through it and this method should not be called.
  Instruction *ExtOpnd = cast<Instruction>(Ext->getOperand(0));
  CreatedInstsCost = 0;
  if (!ExtOpnd->hasOneUse()) {
    // ExtOpnd will be promoted.
    // All its uses, but Ext, will need to use a truncated value of the
    // promoted version.
    // Create the truncate now.
    Value *Trunc = TPT.createTrunc(Ext, ExtOpnd->getType());
    if (Instruction *ITrunc = dyn_cast<Instruction>(Trunc)) {
      ITrunc->removeFromParent();
      // Insert it just after the definition.
      ITrunc->insertAfter(ExtOpnd);
      if (Truncs)
        Truncs->push_back(ITrunc);
    }

    TPT.replaceAllUsesWith(ExtOpnd, Trunc);
    // Restore the operand of Ext (which has been replace by the previous call
    // to replaceAllUsesWith) to avoid creating a cycle trunc <-> sext.
    TPT.setOperand(Ext, 0, ExtOpnd);
  }

  // Get through the Instruction:
  // 1. Update its type.
  // 2. Replace the uses of Ext by Inst.
  // 3. Extend each operand that needs to be extended.

  // Remember the original type of the instruction before promotion.
  // This is useful to know that the high bits are sign extended bits.
  PromotedInsts.insert(std::pair<Instruction *, TypeIsSExt>(
      ExtOpnd, TypeIsSExt(ExtOpnd->getType(), IsSExt)));
  // Step #1.
  TPT.mutateType(ExtOpnd, Ext->getType());
  // Step #2.
  TPT.replaceAllUsesWith(Ext, ExtOpnd);
  // Step #3.
  Instruction *ExtForOpnd = Ext;

  DEBUG(dbgs() << "Propagate Ext to operands\n");
  for (int OpIdx = 0, EndOpIdx = ExtOpnd->getNumOperands(); OpIdx != EndOpIdx;
       ++OpIdx) {
    DEBUG(dbgs() << "Operand:\n" << *(ExtOpnd->getOperand(OpIdx)) << '\n');
    if (ExtOpnd->getOperand(OpIdx)->getType() == Ext->getType() ||
        !shouldExtOperand(ExtOpnd, OpIdx)) {
      DEBUG(dbgs() << "No need to propagate\n");
      continue;
    }
    // Check if we can statically extend the operand.
    Value *Opnd = ExtOpnd->getOperand(OpIdx);
    if (const ConstantInt *Cst = dyn_cast<ConstantInt>(Opnd)) {
      DEBUG(dbgs() << "Statically extend\n");
      unsigned BitWidth = Ext->getType()->getIntegerBitWidth();
      APInt CstVal = IsSExt ? Cst->getValue().sext(BitWidth)
                            : Cst->getValue().zext(BitWidth);
      TPT.setOperand(ExtOpnd, OpIdx, ConstantInt::get(Ext->getType(), CstVal));
      continue;
    }
    // UndefValue are typed, so we have to statically sign extend them.
    if (isa<UndefValue>(Opnd)) {
      DEBUG(dbgs() << "Statically extend\n");
      TPT.setOperand(ExtOpnd, OpIdx, UndefValue::get(Ext->getType()));
      continue;
    }

    // Otherwise we have to explicity sign extend the operand.
    // Check if Ext was reused to extend an operand.
    if (!ExtForOpnd) {
      // If yes, create a new one.
      DEBUG(dbgs() << "More operands to ext\n");
      Value *ValForExtOpnd = IsSExt ? TPT.createSExt(Ext, Opnd, Ext->getType())
        : TPT.createZExt(Ext, Opnd, Ext->getType());
      if (!isa<Instruction>(ValForExtOpnd)) {
        TPT.setOperand(ExtOpnd, OpIdx, ValForExtOpnd);
        continue;
      }
      ExtForOpnd = cast<Instruction>(ValForExtOpnd);
    }
    if (Exts)
      Exts->push_back(ExtForOpnd);
    TPT.setOperand(ExtForOpnd, 0, Opnd);

    // Move the sign extension before the insertion point.
    TPT.moveBefore(ExtForOpnd, ExtOpnd);
    TPT.setOperand(ExtOpnd, OpIdx, ExtForOpnd);
    CreatedInstsCost += !TLI.isExtFree(ExtForOpnd);
    // If more sext are required, new instructions will have to be created.
    ExtForOpnd = nullptr;
  }
  if (ExtForOpnd == Ext) {
    DEBUG(dbgs() << "Extension is useless now\n");
    TPT.eraseInstruction(Ext);
  }
  return ExtOpnd;
}

/// IsPromotionProfitable - Check whether or not promoting an instruction
/// to a wider type was profitable.
/// \p NewCost gives the cost of extension instructions created by the
/// promotion.
/// \p OldCost gives the cost of extension instructions before the promotion
/// plus the number of instructions that have been
/// matched in the addressing mode the promotion.
/// \p PromotedOperand is the value that has been promoted.
/// \return True if the promotion is profitable, false otherwise.
bool AddressingModeMatcher::IsPromotionProfitable(
    unsigned NewCost, unsigned OldCost, Value *PromotedOperand) const {
  DEBUG(dbgs() << "OldCost: " << OldCost << "\tNewCost: " << NewCost << '\n');
  // The cost of the new extensions is greater than the cost of the
  // old extension plus what we folded.
  // This is not profitable.
  if (NewCost > OldCost)
    return false;
  if (NewCost < OldCost)
    return true;
  // The promotion is neutral but it may help folding the sign extension in
  // loads for instance.
  // Check that we did not create an illegal instruction.
  return isPromotedInstructionLegal(TLI, DL, PromotedOperand);
}

/// MatchOperationAddr - Given an instruction or constant expr, see if we can
/// fold the operation into the addressing mode.  If so, update the addressing
/// mode and return true, otherwise return false without modifying AddrMode.
/// If \p MovedAway is not NULL, it contains the information of whether or
/// not AddrInst has to be folded into the addressing mode on success.
/// If \p MovedAway == true, \p AddrInst will not be part of the addressing
/// because it has been moved away.
/// Thus AddrInst must not be added in the matched instructions.
/// This state can happen when AddrInst is a sext, since it may be moved away.
/// Therefore, AddrInst may not be valid when MovedAway is true and it must
/// not be referenced anymore.
bool AddressingModeMatcher::MatchOperationAddr(User *AddrInst, unsigned Opcode,
                                               unsigned Depth,
                                               bool *MovedAway) {
  // Avoid exponential behavior on extremely deep expression trees.
  if (Depth >= 5) return false;

  // By default, all matched instructions stay in place.
  if (MovedAway)
    *MovedAway = false;

  switch (Opcode) {
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return MatchAddr(AddrInst->getOperand(0), Depth);
  case Instruction::IntToPtr: {
    auto AS = AddrInst->getType()->getPointerAddressSpace();
    auto PtrTy = MVT::getIntegerVT(DL.getPointerSizeInBits(AS));
    // This inttoptr is a no-op if the integer type is pointer sized.
    if (TLI.getValueType(DL, AddrInst->getOperand(0)->getType()) == PtrTy)
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  }
  case Instruction::BitCast:
    // BitCast is always a noop, and we can handle it as long as it is
    // int->int or pointer->pointer (we don't want int<->fp or something).
    if ((AddrInst->getOperand(0)->getType()->isPointerTy() ||
         AddrInst->getOperand(0)->getType()->isIntegerTy()) &&
        // Don't touch identity bitcasts.  These were probably put here by LSR,
        // and we don't want to mess around with them.  Assume it knows what it
        // is doing.
        AddrInst->getOperand(0)->getType() != AddrInst->getType())
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  case Instruction::AddrSpaceCast: {
    unsigned SrcAS
      = AddrInst->getOperand(0)->getType()->getPointerAddressSpace();
    unsigned DestAS = AddrInst->getType()->getPointerAddressSpace();
    if (TLI.isNoopAddrSpaceCast(SrcAS, DestAS))
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  }
  case Instruction::Add: {
    // Check to see if we can merge in the RHS then the LHS.  If so, we win.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();
    // Start a transaction at this point.
    // The LHS may match but not the RHS.
    // Therefore, we need a higher level restoration point to undo partially
    // matched operation.
    TypePromotionTransaction::ConstRestorationPt LastKnownGood =
        TPT.getRestorationPoint();

    if (MatchAddr(AddrInst->getOperand(1), Depth+1) &&
        MatchAddr(AddrInst->getOperand(0), Depth+1))
      return true;

    // Restore the old addr mode info.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    TPT.rollback(LastKnownGood);

    // Otherwise this was over-aggressive.  Try merging in the LHS then the RHS.
    if (MatchAddr(AddrInst->getOperand(0), Depth+1) &&
        MatchAddr(AddrInst->getOperand(1), Depth+1))
      return true;

    // Otherwise we definitely can't merge the ADD in.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    TPT.rollback(LastKnownGood);
    break;
  }
  //case Instruction::Or:
  // TODO: We can handle "Or Val, Imm" iff this OR is equivalent to an ADD.
  //break;
  case Instruction::Mul:
  case Instruction::Shl: {
    // Can only handle X*C and X << C.
    ConstantInt *RHS = dyn_cast<ConstantInt>(AddrInst->getOperand(1));
    if (!RHS)
      return false;
    int64_t Scale = RHS->getSExtValue();
    if (Opcode == Instruction::Shl)
      Scale = 1LL << Scale;

    return MatchScaledValue(AddrInst->getOperand(0), Scale, Depth);
  }
  case Instruction::GetElementPtr: {
    // Scan the GEP.  We check it if it contains constant offsets and at most
    // one variable offset.
    int VariableOperand = -1;
    unsigned VariableScale = 0;

    int64_t ConstantOffset = 0;
    gep_type_iterator GTI = gep_type_begin(AddrInst);
    for (unsigned i = 1, e = AddrInst->getNumOperands(); i != e; ++i, ++GTI) {
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        const StructLayout *SL = DL.getStructLayout(STy);
        unsigned Idx =
          cast<ConstantInt>(AddrInst->getOperand(i))->getZExtValue();
        ConstantOffset += SL->getElementOffset(Idx);
      } else {
        uint64_t TypeSize = DL.getTypeAllocSize(GTI.getIndexedType());
        if (ConstantInt *CI = dyn_cast<ConstantInt>(AddrInst->getOperand(i))) {
          ConstantOffset += CI->getSExtValue()*TypeSize;
        } else if (TypeSize) {  // Scales of zero don't do anything.
          // We only allow one variable index at the moment.
          if (VariableOperand != -1)
            return false;

          // Remember the variable index.
          VariableOperand = i;
          VariableScale = TypeSize;
        }
      }
    }

    // A common case is for the GEP to only do a constant offset.  In this case,
    // just add it to the disp field and check validity.
    if (VariableOperand == -1) {
      AddrMode.BaseOffs += ConstantOffset;
      if (ConstantOffset == 0 ||
          TLI.isLegalAddressingMode(DL, AddrMode, AccessTy, AddrSpace)) {
        // Check to see if we can fold the base pointer in too.
        if (MatchAddr(AddrInst->getOperand(0), Depth+1))
          return true;
      }
      AddrMode.BaseOffs -= ConstantOffset;
      return false;
    }

    // Save the valid addressing mode in case we can't match.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // See if the scale and offset amount is valid for this target.
    AddrMode.BaseOffs += ConstantOffset;

    // Match the base operand of the GEP.
    if (!MatchAddr(AddrInst->getOperand(0), Depth+1)) {
      // If it couldn't be matched, just stuff the value in a register.
      if (AddrMode.HasBaseReg) {
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
    }

    // Match the remaining variable portion of the GEP.
    if (!MatchScaledValue(AddrInst->getOperand(VariableOperand), VariableScale,
                          Depth)) {
      // If it couldn't be matched, try stuffing the base into a register
      // instead of matching it, and retrying the match of the scale.
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
      if (AddrMode.HasBaseReg)
        return false;
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
      AddrMode.BaseOffs += ConstantOffset;
      if (!MatchScaledValue(AddrInst->getOperand(VariableOperand),
                            VariableScale, Depth)) {
        // If even that didn't work, bail.
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
    }

    return true;
  }
  case Instruction::SExt:
  case Instruction::ZExt: {
    Instruction *Ext = dyn_cast<Instruction>(AddrInst);
    if (!Ext)
      return false;

    // Try to move this ext out of the way of the addressing mode.
    // Ask for a method for doing so.
    TypePromotionHelper::Action TPH =
        TypePromotionHelper::getAction(Ext, InsertedInsts, TLI, PromotedInsts);
    if (!TPH)
      return false;

    TypePromotionTransaction::ConstRestorationPt LastKnownGood =
        TPT.getRestorationPoint();
    unsigned CreatedInstsCost = 0;
    unsigned ExtCost = !TLI.isExtFree(Ext);
    Value *PromotedOperand =
        TPH(Ext, TPT, PromotedInsts, CreatedInstsCost, nullptr, nullptr, TLI);
    // SExt has been moved away.
    // Thus either it will be rematched later in the recursive calls or it is
    // gone. Anyway, we must not fold it into the addressing mode at this point.
    // E.g.,
    // op = add opnd, 1
    // idx = ext op
    // addr = gep base, idx
    // is now:
    // promotedOpnd = ext opnd            <- no match here
    // op = promoted_add promotedOpnd, 1  <- match (later in recursive calls)
    // addr = gep base, op                <- match
    if (MovedAway)
      *MovedAway = true;

    assert(PromotedOperand &&
           "TypePromotionHelper should have filtered out those cases");

    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    if (!MatchAddr(PromotedOperand, Depth) ||
        // The total of the new cost is equals to the cost of the created
        // instructions.
        // The total of the old cost is equals to the cost of the extension plus
        // what we have saved in the addressing mode.
        !IsPromotionProfitable(CreatedInstsCost,
                               ExtCost + (AddrModeInsts.size() - OldSize),
                               PromotedOperand)) {
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
      DEBUG(dbgs() << "Sign extension does not pay off: rollback\n");
      TPT.rollback(LastKnownGood);
      return false;
    }
    return true;
  }
  }
  return false;
}

/// MatchAddr - If we can, try to add the value of 'Addr' into the current
/// addressing mode.  If Addr can't be added to AddrMode this returns false and
/// leaves AddrMode unmodified.  This assumes that Addr is either a pointer type
/// or intptr_t for the target.
///
bool AddressingModeMatcher::MatchAddr(Value *Addr, unsigned Depth) {
  // Start a transaction at this point that we will rollback if the matching
  // fails.
  TypePromotionTransaction::ConstRestorationPt LastKnownGood =
      TPT.getRestorationPoint();
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Addr)) {
    // Fold in immediates if legal for the target.
    AddrMode.BaseOffs += CI->getSExtValue();
    if (TLI.isLegalAddressingMode(DL, AddrMode, AccessTy, AddrSpace))
      return true;
    AddrMode.BaseOffs -= CI->getSExtValue();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(Addr)) {
    // If this is a global variable, try to fold it into the addressing mode.
    if (!AddrMode.BaseGV) {
      AddrMode.BaseGV = GV;
      if (TLI.isLegalAddressingMode(DL, AddrMode, AccessTy, AddrSpace))
        return true;
      AddrMode.BaseGV = nullptr;
    }
  } else if (Instruction *I = dyn_cast<Instruction>(Addr)) {
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // Check to see if it is possible to fold this operation.
    bool MovedAway = false;
    if (MatchOperationAddr(I, I->getOpcode(), Depth, &MovedAway)) {
      // This instruction may have been move away. If so, there is nothing
      // to check here.
      if (MovedAway)
        return true;
      // Okay, it's possible to fold this.  Check to see if it is actually
      // *profitable* to do so.  We use a simple cost model to avoid increasing
      // register pressure too much.
      if (I->hasOneUse() ||
          IsProfitableToFoldIntoAddressingMode(I, BackupAddrMode, AddrMode)) {
        AddrModeInsts.push_back(I);
        return true;
      }

      // It isn't profitable to do this, roll back.
      //cerr << "NOT FOLDING: " << *I;
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
      TPT.rollback(LastKnownGood);
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    if (MatchOperationAddr(CE, CE->getOpcode(), Depth))
      return true;
    TPT.rollback(LastKnownGood);
  } else if (isa<ConstantPointerNull>(Addr)) {
    // Null pointer gets folded without affecting the addressing mode.
    return true;
  }

  // Worse case, the target should support [reg] addressing modes. :)
  if (!AddrMode.HasBaseReg) {
    AddrMode.HasBaseReg = true;
    AddrMode.BaseReg = Addr;
    // Still check for legality in case the target supports [imm] but not [i+r].
    if (TLI.isLegalAddressingMode(DL, AddrMode, AccessTy, AddrSpace))
      return true;
    AddrMode.HasBaseReg = false;
    AddrMode.BaseReg = nullptr;
  }

  // If the base register is already taken, see if we can do [r+r].
  if (AddrMode.Scale == 0) {
    AddrMode.Scale = 1;
    AddrMode.ScaledReg = Addr;
    if (TLI.isLegalAddressingMode(DL, AddrMode, AccessTy, AddrSpace))
      return true;
    AddrMode.Scale = 0;
    AddrMode.ScaledReg = nullptr;
  }
  // Couldn't match.
  TPT.rollback(LastKnownGood);
  return false;
}

/// IsOperandAMemoryOperand - Check to see if all uses of OpVal by the specified
/// inline asm call are due to memory operands.  If so, return true, otherwise
/// return false.
static bool IsOperandAMemoryOperand(CallInst *CI, InlineAsm *IA, Value *OpVal,
                                    const TargetMachine &TM) {
  const Function *F = CI->getParent()->getParent();
  const TargetLowering *TLI = TM.getSubtargetImpl(*F)->getTargetLowering();
  const TargetRegisterInfo *TRI = TM.getSubtargetImpl(*F)->getRegisterInfo();
  TargetLowering::AsmOperandInfoVector TargetConstraints =
      TLI->ParseConstraints(F->getParent()->getDataLayout(), TRI,
                            ImmutableCallSite(CI));
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];

    // Compute the constraint code and ConstraintType to use.
    TLI->ComputeConstraintToUse(OpInfo, SDValue());

    // If this asm operand is our Value*, and if it isn't an indirect memory
    // operand, we can't fold it!
    if (OpInfo.CallOperandVal == OpVal &&
        (OpInfo.ConstraintType != TargetLowering::C_Memory ||
         !OpInfo.isIndirect))
      return false;
  }

  return true;
}

/// FindAllMemoryUses - Recursively walk all the uses of I until we find a
/// memory use.  If we find an obviously non-foldable instruction, return true.
/// Add the ultimately found memory instructions to MemoryUses.
static bool FindAllMemoryUses(
    Instruction *I,
    SmallVectorImpl<std::pair<Instruction *, unsigned>> &MemoryUses,
    SmallPtrSetImpl<Instruction *> &ConsideredInsts, const TargetMachine &TM) {
  // If we already considered this instruction, we're done.
  if (!ConsideredInsts.insert(I).second)
    return false;

  // If this is an obviously unfoldable instruction, bail out.
  if (!MightBeFoldableInst(I))
    return true;

  // Loop over all the uses, recursively processing them.
  for (Use &U : I->uses()) {
    Instruction *UserI = cast<Instruction>(U.getUser());

    if (LoadInst *LI = dyn_cast<LoadInst>(UserI)) {
      MemoryUses.push_back(std::make_pair(LI, U.getOperandNo()));
      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(UserI)) {
      unsigned opNo = U.getOperandNo();
      if (opNo == 0) return true; // Storing addr, not into addr.
      MemoryUses.push_back(std::make_pair(SI, opNo));
      continue;
    }

    if (CallInst *CI = dyn_cast<CallInst>(UserI)) {
      InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledValue());
      if (!IA) return true;

      // If this is a memory operand, we're cool, otherwise bail out.
      if (!IsOperandAMemoryOperand(CI, IA, I, TM))
        return true;
      continue;
    }

    if (FindAllMemoryUses(UserI, MemoryUses, ConsideredInsts, TM))
      return true;
  }

  return false;
}

/// ValueAlreadyLiveAtInst - Retrn true if Val is already known to be live at
/// the use site that we're folding it into.  If so, there is no cost to
/// include it in the addressing mode.  KnownLive1 and KnownLive2 are two values
/// that we know are live at the instruction already.
bool AddressingModeMatcher::ValueAlreadyLiveAtInst(Value *Val,Value *KnownLive1,
                                                   Value *KnownLive2) {
  // If Val is either of the known-live values, we know it is live!
  if (Val == nullptr || Val == KnownLive1 || Val == KnownLive2)
    return true;

  // All values other than instructions and arguments (e.g. constants) are live.
  if (!isa<Instruction>(Val) && !isa<Argument>(Val)) return true;

  // If Val is a constant sized alloca in the entry block, it is live, this is
  // true because it is just a reference to the stack/frame pointer, which is
  // live for the whole function.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Val))
    if (AI->isStaticAlloca())
      return true;

  // Check to see if this value is already used in the memory instruction's
  // block.  If so, it's already live into the block at the very least, so we
  // can reasonably fold it.
  return Val->isUsedInBasicBlock(MemoryInst->getParent());
}

/// IsProfitableToFoldIntoAddressingMode - It is possible for the addressing
/// mode of the machine to fold the specified instruction into a load or store
/// that ultimately uses it.  However, the specified instruction has multiple
/// uses.  Given this, it may actually increase register pressure to fold it
/// into the load.  For example, consider this code:
///
///     X = ...
///     Y = X+1
///     use(Y)   -> nonload/store
///     Z = Y+1
///     load Z
///
/// In this case, Y has multiple uses, and can be folded into the load of Z
/// (yielding load [X+2]).  However, doing this will cause both "X" and "X+1" to
/// be live at the use(Y) line.  If we don't fold Y into load Z, we use one
/// fewer register.  Since Y can't be folded into "use(Y)" we don't increase the
/// number of computations either.
///
/// Note that this (like most of CodeGenPrepare) is just a rough heuristic.  If
/// X was live across 'load Z' for other reasons, we actually *would* want to
/// fold the addressing mode in the Z case.  This would make Y die earlier.
bool AddressingModeMatcher::
IsProfitableToFoldIntoAddressingMode(Instruction *I, ExtAddrMode &AMBefore,
                                     ExtAddrMode &AMAfter) {
  if (IgnoreProfitability) return true;

  // AMBefore is the addressing mode before this instruction was folded into it,
  // and AMAfter is the addressing mode after the instruction was folded.  Get
  // the set of registers referenced by AMAfter and subtract out those
  // referenced by AMBefore: this is the set of values which folding in this
  // address extends the lifetime of.
  //
  // Note that there are only two potential values being referenced here,
  // BaseReg and ScaleReg (global addresses are always available, as are any
  // folded immediates).
  Value *BaseReg = AMAfter.BaseReg, *ScaledReg = AMAfter.ScaledReg;

  // If the BaseReg or ScaledReg was referenced by the previous addrmode, their
  // lifetime wasn't extended by adding this instruction.
  if (ValueAlreadyLiveAtInst(BaseReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    BaseReg = nullptr;
  if (ValueAlreadyLiveAtInst(ScaledReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    ScaledReg = nullptr;

  // If folding this instruction (and it's subexprs) didn't extend any live
  // ranges, we're ok with it.
  if (!BaseReg && !ScaledReg)
    return true;

  // If all uses of this instruction are ultimately load/store/inlineasm's,
  // check to see if their addressing modes will include this instruction.  If
  // so, we can fold it into all uses, so it doesn't matter if it has multiple
  // uses.
  SmallVector<std::pair<Instruction*,unsigned>, 16> MemoryUses;
  SmallPtrSet<Instruction*, 16> ConsideredInsts;
  if (FindAllMemoryUses(I, MemoryUses, ConsideredInsts, TM))
    return false;  // Has a non-memory, non-foldable use!

  // Now that we know that all uses of this instruction are part of a chain of
  // computation involving only operations that could theoretically be folded
  // into a memory use, loop over each of these uses and see if they could
  // *actually* fold the instruction.
  SmallVector<Instruction*, 32> MatchedAddrModeInsts;
  for (unsigned i = 0, e = MemoryUses.size(); i != e; ++i) {
    Instruction *User = MemoryUses[i].first;
    unsigned OpNo = MemoryUses[i].second;

    // Get the access type of this use.  If the use isn't a pointer, we don't
    // know what it accesses.
    Value *Address = User->getOperand(OpNo);
    PointerType *AddrTy = dyn_cast<PointerType>(Address->getType());
    if (!AddrTy)
      return false;
    Type *AddressAccessTy = AddrTy->getElementType();
    unsigned AS = AddrTy->getAddressSpace();

    // Do a match against the root of this address, ignoring profitability. This
    // will tell us if the addressing mode for the memory operation will
    // *actually* cover the shared instruction.
    ExtAddrMode Result;
    TypePromotionTransaction::ConstRestorationPt LastKnownGood =
        TPT.getRestorationPoint();
    AddressingModeMatcher Matcher(MatchedAddrModeInsts, TM, AddressAccessTy, AS,
                                  MemoryInst, Result, InsertedInsts,
                                  PromotedInsts, TPT);
    Matcher.IgnoreProfitability = true;
    bool Success = Matcher.MatchAddr(Address, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");

    // The match was to check the profitability, the changes made are not
    // part of the original matcher. Therefore, they should be dropped
    // otherwise the original matcher will not present the right state.
    TPT.rollback(LastKnownGood);

    // If the match didn't cover I, then it won't be shared by it.
    if (std::find(MatchedAddrModeInsts.begin(), MatchedAddrModeInsts.end(),
                  I) == MatchedAddrModeInsts.end())
      return false;

    MatchedAddrModeInsts.clear();
  }

  return true;
}

} // end anonymous namespace

/// IsNonLocalValue - Return true if the specified values are defined in a
/// different basic block than BB.
static bool IsNonLocalValue(Value *V, BasicBlock *BB) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() != BB;
  return false;
}

/// OptimizeMemoryInst - Load and Store Instructions often have
/// addressing modes that can do significant amounts of computation.  As such,
/// instruction selection will try to get the load or store to do as much
/// computation as possible for the program.  The problem is that isel can only
/// see within a single block.  As such, we sink as much legal addressing mode
/// stuff into the block as possible.
///
/// This method is used to optimize both load/store and inline asms with memory
/// operands.
bool CodeGenPrepare::OptimizeMemoryInst(Instruction *MemoryInst, Value *Addr,
                                        Type *AccessTy, unsigned AddrSpace) {
  Value *Repl = Addr;

  // Try to collapse single-value PHI nodes.  This is necessary to undo
  // unprofitable PRE transformations.
  SmallVector<Value*, 8> worklist;
  SmallPtrSet<Value*, 16> Visited;
  worklist.push_back(Addr);

  // Use a worklist to iteratively look through PHI nodes, and ensure that
  // the addressing mode obtained from the non-PHI roots of the graph
  // are equivalent.
  Value *Consensus = nullptr;
  unsigned NumUsesConsensus = 0;
  bool IsNumUsesConsensusValid = false;
  SmallVector<Instruction*, 16> AddrModeInsts;
  ExtAddrMode AddrMode;
  TypePromotionTransaction TPT;
  TypePromotionTransaction::ConstRestorationPt LastKnownGood =
      TPT.getRestorationPoint();
  while (!worklist.empty()) {
    Value *V = worklist.back();
    worklist.pop_back();

    // Break use-def graph loops.
    if (!Visited.insert(V).second) {
      Consensus = nullptr;
      break;
    }

    // For a PHI node, push all of its incoming values.
    if (PHINode *P = dyn_cast<PHINode>(V)) {
      for (Value *IncValue : P->incoming_values())
        worklist.push_back(IncValue);
      continue;
    }

    // For non-PHIs, determine the addressing mode being computed.
    SmallVector<Instruction*, 16> NewAddrModeInsts;
    ExtAddrMode NewAddrMode = AddressingModeMatcher::Match(
      V, AccessTy, AddrSpace, MemoryInst, NewAddrModeInsts, *TM,
      InsertedInsts, PromotedInsts, TPT);

    // This check is broken into two cases with very similar code to avoid using
    // getNumUses() as much as possible. Some values have a lot of uses, so
    // calling getNumUses() unconditionally caused a significant compile-time
    // regression.
    if (!Consensus) {
      Consensus = V;
      AddrMode = NewAddrMode;
      AddrModeInsts = NewAddrModeInsts;
      continue;
    } else if (NewAddrMode == AddrMode) {
      if (!IsNumUsesConsensusValid) {
        NumUsesConsensus = Consensus->getNumUses();
        IsNumUsesConsensusValid = true;
      }

      // Ensure that the obtained addressing mode is equivalent to that obtained
      // for all other roots of the PHI traversal.  Also, when choosing one
      // such root as representative, select the one with the most uses in order
      // to keep the cost modeling heuristics in AddressingModeMatcher
      // applicable.
      unsigned NumUses = V->getNumUses();
      if (NumUses > NumUsesConsensus) {
        Consensus = V;
        NumUsesConsensus = NumUses;
        AddrModeInsts = NewAddrModeInsts;
      }
      continue;
    }

    Consensus = nullptr;
    break;
  }

  // If the addressing mode couldn't be determined, or if multiple different
  // ones were determined, bail out now.
  if (!Consensus) {
    TPT.rollback(LastKnownGood);
    return false;
  }
  TPT.commit();

  // Check to see if any of the instructions supersumed by this addr mode are
  // non-local to I's BB.
  bool AnyNonLocal = false;
  for (unsigned i = 0, e = AddrModeInsts.size(); i != e; ++i) {
    if (IsNonLocalValue(AddrModeInsts[i], MemoryInst->getParent())) {
      AnyNonLocal = true;
      break;
    }
  }

  // If all the instructions matched are already in this BB, don't do anything.
  if (!AnyNonLocal) {
    DEBUG(dbgs() << "CGP: Found      local addrmode: " << AddrMode << "\n");
    return false;
  }

  // Insert this computation right after this user.  Since our caller is
  // scanning from the top of the BB to the bottom, reuse of the expr are
  // guaranteed to happen later.
  IRBuilder<> Builder(MemoryInst);

  // Now that we determined the addressing expression we want to use and know
  // that we have to sink it into this block.  Check to see if we have already
  // done this for some other load/store instr in this block.  If so, reuse the
  // computation.
  Value *&SunkAddr = SunkAddrs[Addr];
  if (SunkAddr) {
    DEBUG(dbgs() << "CGP: Reusing nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst << "\n");
    if (SunkAddr->getType() != Addr->getType())
      SunkAddr = Builder.CreateBitCast(SunkAddr, Addr->getType());
  } else if (AddrSinkUsingGEPs ||
             (!AddrSinkUsingGEPs.getNumOccurrences() && TM &&
              TM->getSubtargetImpl(*MemoryInst->getParent()->getParent())
                  ->useAA())) {
    // By default, we use the GEP-based method when AA is used later. This
    // prevents new inttoptr/ptrtoint pairs from degrading AA capabilities.
    DEBUG(dbgs() << "CGP: SINKING nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst << "\n");
    Type *IntPtrTy = DL->getIntPtrType(Addr->getType());
    Value *ResultPtr = nullptr, *ResultIndex = nullptr;

    // First, find the pointer.
    if (AddrMode.BaseReg && AddrMode.BaseReg->getType()->isPointerTy()) {
      ResultPtr = AddrMode.BaseReg;
      AddrMode.BaseReg = nullptr;
    }

    if (AddrMode.Scale && AddrMode.ScaledReg->getType()->isPointerTy()) {
      // We can't add more than one pointer together, nor can we scale a
      // pointer (both of which seem meaningless).
      if (ResultPtr || AddrMode.Scale != 1)
        return false;

      ResultPtr = AddrMode.ScaledReg;
      AddrMode.Scale = 0;
    }

    if (AddrMode.BaseGV) {
      if (ResultPtr)
        return false;

      ResultPtr = AddrMode.BaseGV;
    }

    // If the real base value actually came from an inttoptr, then the matcher
    // will look through it and provide only the integer value. In that case,
    // use it here.
    if (!ResultPtr && AddrMode.BaseReg) {
      ResultPtr =
        Builder.CreateIntToPtr(AddrMode.BaseReg, Addr->getType(), "sunkaddr");
      AddrMode.BaseReg = nullptr;
    } else if (!ResultPtr && AddrMode.Scale == 1) {
      ResultPtr =
        Builder.CreateIntToPtr(AddrMode.ScaledReg, Addr->getType(), "sunkaddr");
      AddrMode.Scale = 0;
    }

    if (!ResultPtr &&
        !AddrMode.BaseReg && !AddrMode.Scale && !AddrMode.BaseOffs) {
      SunkAddr = Constant::getNullValue(Addr->getType());
    } else if (!ResultPtr) {
      return false;
    } else {
      Type *I8PtrTy =
          Builder.getInt8PtrTy(Addr->getType()->getPointerAddressSpace());
      Type *I8Ty = Builder.getInt8Ty();

      // Start with the base register. Do this first so that subsequent address
      // matching finds it last, which will prevent it from trying to match it
      // as the scaled value in case it happens to be a mul. That would be
      // problematic if we've sunk a different mul for the scale, because then
      // we'd end up sinking both muls.
      if (AddrMode.BaseReg) {
        Value *V = AddrMode.BaseReg;
        if (V->getType() != IntPtrTy)
          V = Builder.CreateIntCast(V, IntPtrTy, /*isSigned=*/true, "sunkaddr");

        ResultIndex = V;
      }

      // Add the scale value.
      if (AddrMode.Scale) {
        Value *V = AddrMode.ScaledReg;
        if (V->getType() == IntPtrTy) {
          // done.
        } else if (cast<IntegerType>(IntPtrTy)->getBitWidth() <
                   cast<IntegerType>(V->getType())->getBitWidth()) {
          V = Builder.CreateTrunc(V, IntPtrTy, "sunkaddr");
        } else {
          // It is only safe to sign extend the BaseReg if we know that the math
          // required to create it did not overflow before we extend it. Since
          // the original IR value was tossed in favor of a constant back when
          // the AddrMode was created we need to bail out gracefully if widths
          // do not match instead of extending it.
          Instruction *I = dyn_cast_or_null<Instruction>(ResultIndex);
          if (I && (ResultIndex != AddrMode.BaseReg))
            I->eraseFromParent();
          return false;
        }

        if (AddrMode.Scale != 1)
          V = Builder.CreateMul(V, ConstantInt::get(IntPtrTy, AddrMode.Scale),
                                "sunkaddr");
        if (ResultIndex)
          ResultIndex = Builder.CreateAdd(ResultIndex, V, "sunkaddr");
        else
          ResultIndex = V;
      }

      // Add in the Base Offset if present.
      if (AddrMode.BaseOffs) {
        Value *V = ConstantInt::get(IntPtrTy, AddrMode.BaseOffs);
        if (ResultIndex) {
          // We need to add this separately from the scale above to help with
          // SDAG consecutive load/store merging.
          if (ResultPtr->getType() != I8PtrTy)
            ResultPtr = Builder.CreateBitCast(ResultPtr, I8PtrTy);
          ResultPtr = Builder.CreateGEP(I8Ty, ResultPtr, ResultIndex, "sunkaddr");
        }

        ResultIndex = V;
      }

      if (!ResultIndex) {
        SunkAddr = ResultPtr;
      } else {
        if (ResultPtr->getType() != I8PtrTy)
          ResultPtr = Builder.CreateBitCast(ResultPtr, I8PtrTy);
        SunkAddr = Builder.CreateGEP(I8Ty, ResultPtr, ResultIndex, "sunkaddr");
      }

      if (SunkAddr->getType() != Addr->getType())
        SunkAddr = Builder.CreateBitCast(SunkAddr, Addr->getType());
    }
  } else {
    DEBUG(dbgs() << "CGP: SINKING nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst << "\n");
    Type *IntPtrTy = DL->getIntPtrType(Addr->getType());
    Value *Result = nullptr;

    // Start with the base register. Do this first so that subsequent address
    // matching finds it last, which will prevent it from trying to match it
    // as the scaled value in case it happens to be a mul. That would be
    // problematic if we've sunk a different mul for the scale, because then
    // we'd end up sinking both muls.
    if (AddrMode.BaseReg) {
      Value *V = AddrMode.BaseReg;
      if (V->getType()->isPointerTy())
        V = Builder.CreatePtrToInt(V, IntPtrTy, "sunkaddr");
      if (V->getType() != IntPtrTy)
        V = Builder.CreateIntCast(V, IntPtrTy, /*isSigned=*/true, "sunkaddr");
      Result = V;
    }

    // Add the scale value.
    if (AddrMode.Scale) {
      Value *V = AddrMode.ScaledReg;
      if (V->getType() == IntPtrTy) {
        // done.
      } else if (V->getType()->isPointerTy()) {
        V = Builder.CreatePtrToInt(V, IntPtrTy, "sunkaddr");
      } else if (cast<IntegerType>(IntPtrTy)->getBitWidth() <
                 cast<IntegerType>(V->getType())->getBitWidth()) {
        V = Builder.CreateTrunc(V, IntPtrTy, "sunkaddr");
      } else {
        // It is only safe to sign extend the BaseReg if we know that the math
        // required to create it did not overflow before we extend it. Since
        // the original IR value was tossed in favor of a constant back when
        // the AddrMode was created we need to bail out gracefully if widths
        // do not match instead of extending it.
        Instruction *I = dyn_cast_or_null<Instruction>(Result);
        if (I && (Result != AddrMode.BaseReg))
          I->eraseFromParent();
        return false;
      }
      if (AddrMode.Scale != 1)
        V = Builder.CreateMul(V, ConstantInt::get(IntPtrTy, AddrMode.Scale),
                              "sunkaddr");
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    // Add in the BaseGV if present.
    if (AddrMode.BaseGV) {
      Value *V = Builder.CreatePtrToInt(AddrMode.BaseGV, IntPtrTy, "sunkaddr");
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    // Add in the Base Offset if present.
    if (AddrMode.BaseOffs) {
      Value *V = ConstantInt::get(IntPtrTy, AddrMode.BaseOffs);
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    if (!Result)
      SunkAddr = Constant::getNullValue(Addr->getType());
    else
      SunkAddr = Builder.CreateIntToPtr(Result, Addr->getType(), "sunkaddr");
  }

  MemoryInst->replaceUsesOfWith(Repl, SunkAddr);

  // If we have no uses, recursively delete the value and all dead instructions
  // using it.
  if (Repl->use_empty()) {
    // This can cause recursive deletion, which can invalidate our iterator.
    // Use a WeakVH to hold onto it in case this happens.
    WeakVH IterHandle(CurInstIterator);
    BasicBlock *BB = CurInstIterator->getParent();

    RecursivelyDeleteTriviallyDeadInstructions(Repl, TLInfo);

    if (IterHandle != CurInstIterator) {
      // If the iterator instruction was recursively deleted, start over at the
      // start of the block.
      CurInstIterator = BB->begin();
      SunkAddrs.clear();
    }
  }
  ++NumMemoryInsts;
  return true;
}

/// OptimizeInlineAsmInst - If there are any memory operands, use
/// OptimizeMemoryInst to sink their address computing into the block when
/// possible / profitable.
bool CodeGenPrepare::OptimizeInlineAsmInst(CallInst *CS) {
  bool MadeChange = false;

  const TargetRegisterInfo *TRI =
      TM->getSubtargetImpl(*CS->getParent()->getParent())->getRegisterInfo();
  TargetLowering::AsmOperandInfoVector TargetConstraints =
      TLI->ParseConstraints(*DL, TRI, CS);
  unsigned ArgNo = 0;
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];

    // Compute the constraint code and ConstraintType to use.
    TLI->ComputeConstraintToUse(OpInfo, SDValue());

    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        OpInfo.isIndirect) {
      Value *OpVal = CS->getArgOperand(ArgNo++);
      MadeChange |= OptimizeMemoryInst(CS, OpVal, OpVal->getType(), ~0u);
    } else if (OpInfo.Type == InlineAsm::isInput)
      ArgNo++;
  }

  return MadeChange;
}

/// \brief Check if all the uses of \p Inst are equivalent (or free) zero or
/// sign extensions.
static bool hasSameExtUse(Instruction *Inst, const TargetLowering &TLI) {
  assert(!Inst->use_empty() && "Input must have at least one use");
  const Instruction *FirstUser = cast<Instruction>(*Inst->user_begin());
  bool IsSExt = isa<SExtInst>(FirstUser);
  Type *ExtTy = FirstUser->getType();
  for (const User *U : Inst->users()) {
    const Instruction *UI = cast<Instruction>(U);
    if ((IsSExt && !isa<SExtInst>(UI)) || (!IsSExt && !isa<ZExtInst>(UI)))
      return false;
    Type *CurTy = UI->getType();
    // Same input and output types: Same instruction after CSE.
    if (CurTy == ExtTy)
      continue;

    // If IsSExt is true, we are in this situation:
    // a = Inst
    // b = sext ty1 a to ty2
    // c = sext ty1 a to ty3
    // Assuming ty2 is shorter than ty3, this could be turned into:
    // a = Inst
    // b = sext ty1 a to ty2
    // c = sext ty2 b to ty3
    // However, the last sext is not free.
    if (IsSExt)
      return false;

    // This is a ZExt, maybe this is free to extend from one type to another.
    // In that case, we would not account for a different use.
    Type *NarrowTy;
    Type *LargeTy;
    if (ExtTy->getScalarType()->getIntegerBitWidth() >
        CurTy->getScalarType()->getIntegerBitWidth()) {
      NarrowTy = CurTy;
      LargeTy = ExtTy;
    } else {
      NarrowTy = ExtTy;
      LargeTy = CurTy;
    }

    if (!TLI.isZExtFree(NarrowTy, LargeTy))
      return false;
  }
  // All uses are the same or can be derived from one another for free.
  return true;
}

/// \brief Try to form ExtLd by promoting \p Exts until they reach a
/// load instruction.
/// If an ext(load) can be formed, it is returned via \p LI for the load
/// and \p Inst for the extension.
/// Otherwise LI == nullptr and Inst == nullptr.
/// When some promotion happened, \p TPT contains the proper state to
/// revert them.
///
/// \return true when promoting was necessary to expose the ext(load)
/// opportunity, false otherwise.
///
/// Example:
/// \code
/// %ld = load i32* %addr
/// %add = add nuw i32 %ld, 4
/// %zext = zext i32 %add to i64
/// \endcode
/// =>
/// \code
/// %ld = load i32* %addr
/// %zext = zext i32 %ld to i64
/// %add = add nuw i64 %zext, 4
/// \encode
/// Thanks to the promotion, we can match zext(load i32*) to i64.
bool CodeGenPrepare::ExtLdPromotion(TypePromotionTransaction &TPT,
                                    LoadInst *&LI, Instruction *&Inst,
                                    const SmallVectorImpl<Instruction *> &Exts,
                                    unsigned CreatedInstsCost = 0) {
  // Iterate over all the extensions to see if one form an ext(load).
  for (auto I : Exts) {
    // Check if we directly have ext(load).
    if ((LI = dyn_cast<LoadInst>(I->getOperand(0)))) {
      Inst = I;
      // No promotion happened here.
      return false;
    }
    // Check whether or not we want to do any promotion.
    if (!TLI || !TLI->enableExtLdPromotion() || DisableExtLdPromotion)
      continue;
    // Get the action to perform the promotion.
    TypePromotionHelper::Action TPH = TypePromotionHelper::getAction(
        I, InsertedInsts, *TLI, PromotedInsts);
    // Check if we can promote.
    if (!TPH)
      continue;
    // Save the current state.
    TypePromotionTransaction::ConstRestorationPt LastKnownGood =
        TPT.getRestorationPoint();
    SmallVector<Instruction *, 4> NewExts;
    unsigned NewCreatedInstsCost = 0;
    unsigned ExtCost = !TLI->isExtFree(I);
    // Promote.
    Value *PromotedVal = TPH(I, TPT, PromotedInsts, NewCreatedInstsCost,
                             &NewExts, nullptr, *TLI);
    assert(PromotedVal &&
           "TypePromotionHelper should have filtered out those cases");

    // We would be able to merge only one extension in a load.
    // Therefore, if we have more than 1 new extension we heuristically
    // cut this search path, because it means we degrade the code quality.
    // With exactly 2, the transformation is neutral, because we will merge
    // one extension but leave one. However, we optimistically keep going,
    // because the new extension may be removed too.
    long long TotalCreatedInstsCost = CreatedInstsCost + NewCreatedInstsCost;
    TotalCreatedInstsCost -= ExtCost;
    if (!StressExtLdPromotion &&
        (TotalCreatedInstsCost > 1 ||
         !isPromotedInstructionLegal(*TLI, *DL, PromotedVal))) {
      // The promotion is not profitable, rollback to the previous state.
      TPT.rollback(LastKnownGood);
      continue;
    }
    // The promotion is profitable.
    // Check if it exposes an ext(load).
    (void)ExtLdPromotion(TPT, LI, Inst, NewExts, TotalCreatedInstsCost);
    if (LI && (StressExtLdPromotion || NewCreatedInstsCost <= ExtCost ||
               // If we have created a new extension, i.e., now we have two
               // extensions. We must make sure one of them is merged with
               // the load, otherwise we may degrade the code quality.
               (LI->hasOneUse() || hasSameExtUse(LI, *TLI))))
      // Promotion happened.
      return true;
    // If this does not help to expose an ext(load) then, rollback.
    TPT.rollback(LastKnownGood);
  }
  // None of the extension can form an ext(load).
  LI = nullptr;
  Inst = nullptr;
  return false;
}

/// MoveExtToFormExtLoad - Move a zext or sext fed by a load into the same
/// basic block as the load, unless conditions are unfavorable. This allows
/// SelectionDAG to fold the extend into the load.
/// \p I[in/out] the extension may be modified during the process if some
/// promotions apply.
///
bool CodeGenPrepare::MoveExtToFormExtLoad(Instruction *&I) {
  // Try to promote a chain of computation if it allows to form
  // an extended load.
  TypePromotionTransaction TPT;
  TypePromotionTransaction::ConstRestorationPt LastKnownGood =
    TPT.getRestorationPoint();
  SmallVector<Instruction *, 1> Exts;
  Exts.push_back(I);
  // Look for a load being extended.
  LoadInst *LI = nullptr;
  Instruction *OldExt = I;
  bool HasPromoted = ExtLdPromotion(TPT, LI, I, Exts);
  if (!LI || !I) {
    assert(!HasPromoted && !LI && "If we did not match any load instruction "
                                  "the code must remain the same");
    I = OldExt;
    return false;
  }

  // If they're already in the same block, there's nothing to do.
  // Make the cheap checks first if we did not promote.
  // If we promoted, we need to check if it is indeed profitable.
  if (!HasPromoted && LI->getParent() == I->getParent())
    return false;

  EVT VT = TLI->getValueType(*DL, I->getType());
  EVT LoadVT = TLI->getValueType(*DL, LI->getType());

  // If the load has other users and the truncate is not free, this probably
  // isn't worthwhile.
  if (!LI->hasOneUse() && TLI &&
      (TLI->isTypeLegal(LoadVT) || !TLI->isTypeLegal(VT)) &&
      !TLI->isTruncateFree(I->getType(), LI->getType())) {
    I = OldExt;
    TPT.rollback(LastKnownGood);
    return false;
  }

  // Check whether the target supports casts folded into loads.
  unsigned LType;
  if (isa<ZExtInst>(I))
    LType = ISD::ZEXTLOAD;
  else {
    assert(isa<SExtInst>(I) && "Unexpected ext type!");
    LType = ISD::SEXTLOAD;
  }
  if (TLI && !TLI->isLoadExtLegal(LType, VT, LoadVT)) {
    I = OldExt;
    TPT.rollback(LastKnownGood);
    return false;
  }

  // Move the extend into the same block as the load, so that SelectionDAG
  // can fold it.
  TPT.commit();
  I->removeFromParent();
  I->insertAfter(LI);
  ++NumExtsMoved;
  return true;
}

bool CodeGenPrepare::OptimizeExtUses(Instruction *I) {
  BasicBlock *DefBB = I->getParent();

  // If the result of a {s|z}ext and its source are both live out, rewrite all
  // other uses of the source with result of extension.
  Value *Src = I->getOperand(0);
  if (Src->hasOneUse())
    return false;

  // Only do this xform if truncating is free.
  if (TLI && !TLI->isTruncateFree(I->getType(), Src->getType()))
    return false;

  // Only safe to perform the optimization if the source is also defined in
  // this block.
  if (!isa<Instruction>(Src) || DefBB != cast<Instruction>(Src)->getParent())
    return false;

  bool DefIsLiveOut = false;
  for (User *U : I->users()) {
    Instruction *UI = cast<Instruction>(U);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = UI->getParent();
    if (UserBB == DefBB) continue;
    DefIsLiveOut = true;
    break;
  }
  if (!DefIsLiveOut)
    return false;

  // Make sure none of the uses are PHI nodes.
  for (User *U : Src->users()) {
    Instruction *UI = cast<Instruction>(U);
    BasicBlock *UserBB = UI->getParent();
    if (UserBB == DefBB) continue;
    // Be conservative. We don't want this xform to end up introducing
    // reloads just before load / store instructions.
    if (isa<PHINode>(UI) || isa<LoadInst>(UI) || isa<StoreInst>(UI))
      return false;
  }

  // InsertedTruncs - Only insert one trunc in each block once.
  DenseMap<BasicBlock*, Instruction*> InsertedTruncs;

  bool MadeChange = false;
  for (Use &U : Src->uses()) {
    Instruction *User = cast<Instruction>(U.getUser());

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;

    // Both src and def are live in this block. Rewrite the use.
    Instruction *&InsertedTrunc = InsertedTruncs[UserBB];

    if (!InsertedTrunc) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedTrunc = new TruncInst(I, Src->getType(), "", InsertPt);
      InsertedInsts.insert(InsertedTrunc);
    }

    // Replace a use of the {s|z}ext source with a use of the result.
    U = InsertedTrunc;
    ++NumExtUses;
    MadeChange = true;
  }

  return MadeChange;
}

/// isFormingBranchFromSelectProfitable - Returns true if a SelectInst should be
/// turned into an explicit branch.
static bool isFormingBranchFromSelectProfitable(SelectInst *SI) {
  // FIXME: This should use the same heuristics as IfConversion to determine
  // whether a select is better represented as a branch.  This requires that
  // branch probability metadata is preserved for the select, which is not the
  // case currently.

  CmpInst *Cmp = dyn_cast<CmpInst>(SI->getCondition());

  // If the branch is predicted right, an out of order CPU can avoid blocking on
  // the compare.  Emit cmovs on compares with a memory operand as branches to
  // avoid stalls on the load from memory.  If the compare has more than one use
  // there's probably another cmov or setcc around so it's not worth emitting a
  // branch.
  if (!Cmp)
    return false;

  Value *CmpOp0 = Cmp->getOperand(0);
  Value *CmpOp1 = Cmp->getOperand(1);

  // We check that the memory operand has one use to avoid uses of the loaded
  // value directly after the compare, making branches unprofitable.
  return Cmp->hasOneUse() &&
         ((isa<LoadInst>(CmpOp0) && CmpOp0->hasOneUse()) ||
          (isa<LoadInst>(CmpOp1) && CmpOp1->hasOneUse()));
}


/// If we have a SelectInst that will likely profit from branch prediction,
/// turn it into a branch.
bool CodeGenPrepare::OptimizeSelectInst(SelectInst *SI) {
  bool VectorCond = !SI->getCondition()->getType()->isIntegerTy(1);

  // Can we convert the 'select' to CF ?
  if (DisableSelectToBranch || OptSize || !TLI || VectorCond)
    return false;

  TargetLowering::SelectSupportKind SelectKind;
  if (VectorCond)
    SelectKind = TargetLowering::VectorMaskSelect;
  else if (SI->getType()->isVectorTy())
    SelectKind = TargetLowering::ScalarCondVectorVal;
  else
    SelectKind = TargetLowering::ScalarValSelect;

  // Do we have efficient codegen support for this kind of 'selects' ?
  if (TLI->isSelectSupported(SelectKind)) {
    // We have efficient codegen support for the select instruction.
    // Check if it is profitable to keep this 'select'.
    if (!TLI->isPredictableSelectExpensive() ||
        !isFormingBranchFromSelectProfitable(SI))
      return false;
  }

  ModifiedDT = true;

  // First, we split the block containing the select into 2 blocks.
  BasicBlock *StartBlock = SI->getParent();
  BasicBlock::iterator SplitPt = ++(BasicBlock::iterator(SI));
  BasicBlock *NextBlock = StartBlock->splitBasicBlock(SplitPt, "select.end");

  // Create a new block serving as the landing pad for the branch.
  BasicBlock *SmallBlock = BasicBlock::Create(SI->getContext(), "select.mid",
                                             NextBlock->getParent(), NextBlock);

  // Move the unconditional branch from the block with the select in it into our
  // landing pad block.
  StartBlock->getTerminator()->eraseFromParent();
  BranchInst::Create(NextBlock, SmallBlock);

  // Insert the real conditional branch based on the original condition.
  BranchInst::Create(NextBlock, SmallBlock, SI->getCondition(), SI);

  // The select itself is replaced with a PHI Node.
  PHINode *PN = PHINode::Create(SI->getType(), 2, "", NextBlock->begin());
  PN->takeName(SI);
  PN->addIncoming(SI->getTrueValue(), StartBlock);
  PN->addIncoming(SI->getFalseValue(), SmallBlock);
  SI->replaceAllUsesWith(PN);
  SI->eraseFromParent();

  // Instruct OptimizeBlock to skip to the next block.
  CurInstIterator = StartBlock->end();
  ++NumSelectsExpanded;
  return true;
}

static bool isBroadcastShuffle(ShuffleVectorInst *SVI) {
  SmallVector<int, 16> Mask(SVI->getShuffleMask());
  int SplatElem = -1;
  for (unsigned i = 0; i < Mask.size(); ++i) {
    if (SplatElem != -1 && Mask[i] != -1 && Mask[i] != SplatElem)
      return false;
    SplatElem = Mask[i];
  }

  return true;
}

/// Some targets have expensive vector shifts if the lanes aren't all the same
/// (e.g. x86 only introduced "vpsllvd" and friends with AVX2). In these cases
/// it's often worth sinking a shufflevector splat down to its use so that
/// codegen can spot all lanes are identical.
bool CodeGenPrepare::OptimizeShuffleVectorInst(ShuffleVectorInst *SVI) {
  BasicBlock *DefBB = SVI->getParent();

  // Only do this xform if variable vector shifts are particularly expensive.
  if (!TLI || !TLI->isVectorShiftByScalarCheap(SVI->getType()))
    return false;

  // We only expect better codegen by sinking a shuffle if we can recognise a
  // constant splat.
  if (!isBroadcastShuffle(SVI))
    return false;

  // InsertedShuffles - Only insert a shuffle in each block once.
  DenseMap<BasicBlock*, Instruction*> InsertedShuffles;

  bool MadeChange = false;
  for (User *U : SVI->users()) {
    Instruction *UI = cast<Instruction>(U);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = UI->getParent();
    if (UserBB == DefBB) continue;

    // For now only apply this when the splat is used by a shift instruction.
    if (!UI->isShift()) continue;

    // Everything checks out, sink the shuffle if the user's block doesn't
    // already have a copy.
    Instruction *&InsertedShuffle = InsertedShuffles[UserBB];

    if (!InsertedShuffle) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedShuffle = new ShuffleVectorInst(SVI->getOperand(0),
                                              SVI->getOperand(1),
                                              SVI->getOperand(2), "", InsertPt);
    }

    UI->replaceUsesOfWith(SVI, InsertedShuffle);
    MadeChange = true;
  }

  // If we removed all uses, nuke the shuffle.
  if (SVI->use_empty()) {
    SVI->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

namespace {
/// \brief Helper class to promote a scalar operation to a vector one.
/// This class is used to move downward extractelement transition.
/// E.g.,
/// a = vector_op <2 x i32>
/// b = extractelement <2 x i32> a, i32 0
/// c = scalar_op b
/// store c
///
/// =>
/// a = vector_op <2 x i32>
/// c = vector_op a (equivalent to scalar_op on the related lane)
/// * d = extractelement <2 x i32> c, i32 0
/// * store d
/// Assuming both extractelement and store can be combine, we get rid of the
/// transition.
class VectorPromoteHelper {
  /// DataLayout associated with the current module.
  const DataLayout &DL;

  /// Used to perform some checks on the legality of vector operations.
  const TargetLowering &TLI;

  /// Used to estimated the cost of the promoted chain.
  const TargetTransformInfo &TTI;

  /// The transition being moved downwards.
  Instruction *Transition;
  /// The sequence of instructions to be promoted.
  SmallVector<Instruction *, 4> InstsToBePromoted;
  /// Cost of combining a store and an extract.
  unsigned StoreExtractCombineCost;
  /// Instruction that will be combined with the transition.
  Instruction *CombineInst;

  /// \brief The instruction that represents the current end of the transition.
  /// Since we are faking the promotion until we reach the end of the chain
  /// of computation, we need a way to get the current end of the transition.
  Instruction *getEndOfTransition() const {
    if (InstsToBePromoted.empty())
      return Transition;
    return InstsToBePromoted.back();
  }

  /// \brief Return the index of the original value in the transition.
  /// E.g., for "extractelement <2 x i32> c, i32 1" the original value,
  /// c, is at index 0.
  unsigned getTransitionOriginalValueIdx() const {
    assert(isa<ExtractElementInst>(Transition) &&
           "Other kind of transitions are not supported yet");
    return 0;
  }

  /// \brief Return the index of the index in the transition.
  /// E.g., for "extractelement <2 x i32> c, i32 0" the index
  /// is at index 1.
  unsigned getTransitionIdx() const {
    assert(isa<ExtractElementInst>(Transition) &&
           "Other kind of transitions are not supported yet");
    return 1;
  }

  /// \brief Get the type of the transition.
  /// This is the type of the original value.
  /// E.g., for "extractelement <2 x i32> c, i32 1" the type of the
  /// transition is <2 x i32>.
  Type *getTransitionType() const {
    return Transition->getOperand(getTransitionOriginalValueIdx())->getType();
  }

  /// \brief Promote \p ToBePromoted by moving \p Def downward through.
  /// I.e., we have the following sequence:
  /// Def = Transition <ty1> a to <ty2>
  /// b = ToBePromoted <ty2> Def, ...
  /// =>
  /// b = ToBePromoted <ty1> a, ...
  /// Def = Transition <ty1> ToBePromoted to <ty2>
  void promoteImpl(Instruction *ToBePromoted);

  /// \brief Check whether or not it is profitable to promote all the
  /// instructions enqueued to be promoted.
  bool isProfitableToPromote() {
    Value *ValIdx = Transition->getOperand(getTransitionOriginalValueIdx());
    unsigned Index = isa<ConstantInt>(ValIdx)
                         ? cast<ConstantInt>(ValIdx)->getZExtValue()
                         : -1;
    Type *PromotedType = getTransitionType();

    StoreInst *ST = cast<StoreInst>(CombineInst);
    unsigned AS = ST->getPointerAddressSpace();
    unsigned Align = ST->getAlignment();
    // Check if this store is supported.
    if (!TLI.allowsMisalignedMemoryAccesses(
            TLI.getValueType(DL, ST->getValueOperand()->getType()), AS,
            Align)) {
      // If this is not supported, there is no way we can combine
      // the extract with the store.
      return false;
    }

    // The scalar chain of computation has to pay for the transition
    // scalar to vector.
    // The vector chain has to account for the combining cost.
    uint64_t ScalarCost =
        TTI.getVectorInstrCost(Transition->getOpcode(), PromotedType, Index);
    uint64_t VectorCost = StoreExtractCombineCost;
    for (const auto &Inst : InstsToBePromoted) {
      // Compute the cost.
      // By construction, all instructions being promoted are arithmetic ones.
      // Moreover, one argument is a constant that can be viewed as a splat
      // constant.
      Value *Arg0 = Inst->getOperand(0);
      bool IsArg0Constant = isa<UndefValue>(Arg0) || isa<ConstantInt>(Arg0) ||
                            isa<ConstantFP>(Arg0);
      TargetTransformInfo::OperandValueKind Arg0OVK =
          IsArg0Constant ? TargetTransformInfo::OK_UniformConstantValue
                         : TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Arg1OVK =
          !IsArg0Constant ? TargetTransformInfo::OK_UniformConstantValue
                          : TargetTransformInfo::OK_AnyValue;
      ScalarCost += TTI.getArithmeticInstrCost(
          Inst->getOpcode(), Inst->getType(), Arg0OVK, Arg1OVK);
      VectorCost += TTI.getArithmeticInstrCost(Inst->getOpcode(), PromotedType,
                                               Arg0OVK, Arg1OVK);
    }
    DEBUG(dbgs() << "Estimated cost of computation to be promoted:\nScalar: "
                 << ScalarCost << "\nVector: " << VectorCost << '\n');
    return ScalarCost > VectorCost;
  }

  /// \brief Generate a constant vector with \p Val with the same
  /// number of elements as the transition.
  /// \p UseSplat defines whether or not \p Val should be replicated
  /// accross the whole vector.
  /// In other words, if UseSplat == true, we generate <Val, Val, ..., Val>,
  /// otherwise we generate a vector with as many undef as possible:
  /// <undef, ..., undef, Val, undef, ..., undef> where \p Val is only
  /// used at the index of the extract.
  Value *getConstantVector(Constant *Val, bool UseSplat) const {
    unsigned ExtractIdx = UINT_MAX;
    if (!UseSplat) {
      // If we cannot determine where the constant must be, we have to
      // use a splat constant.
      Value *ValExtractIdx = Transition->getOperand(getTransitionIdx());
      if (ConstantInt *CstVal = dyn_cast<ConstantInt>(ValExtractIdx))
        ExtractIdx = CstVal->getSExtValue();
      else
        UseSplat = true;
    }

    unsigned End = getTransitionType()->getVectorNumElements();
    if (UseSplat)
      return ConstantVector::getSplat(End, Val);

    SmallVector<Constant *, 4> ConstVec;
    UndefValue *UndefVal = UndefValue::get(Val->getType());
    for (unsigned Idx = 0; Idx != End; ++Idx) {
      if (Idx == ExtractIdx)
        ConstVec.push_back(Val);
      else
        ConstVec.push_back(UndefVal);
    }
    return ConstantVector::get(ConstVec);
  }

  /// \brief Check if promoting to a vector type an operand at \p OperandIdx
  /// in \p Use can trigger undefined behavior.
  static bool canCauseUndefinedBehavior(const Instruction *Use,
                                        unsigned OperandIdx) {
    // This is not safe to introduce undef when the operand is on
    // the right hand side of a division-like instruction.
    if (OperandIdx != 1)
      return false;
    switch (Use->getOpcode()) {
    default:
      return false;
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::SRem:
    case Instruction::URem:
      return true;
    case Instruction::FDiv:
    case Instruction::FRem:
      return !Use->hasNoNaNs();
    }
    llvm_unreachable(nullptr);
  }

public:
  VectorPromoteHelper(const DataLayout &DL, const TargetLowering &TLI,
                      const TargetTransformInfo &TTI, Instruction *Transition,
                      unsigned CombineCost)
      : DL(DL), TLI(TLI), TTI(TTI), Transition(Transition),
        StoreExtractCombineCost(CombineCost), CombineInst(nullptr) {
    assert(Transition && "Do not know how to promote null");
  }

  /// \brief Check if we can promote \p ToBePromoted to \p Type.
  bool canPromote(const Instruction *ToBePromoted) const {
    // We could support CastInst too.
    return isa<BinaryOperator>(ToBePromoted);
  }

  /// \brief Check if it is profitable to promote \p ToBePromoted
  /// by moving downward the transition through.
  bool shouldPromote(const Instruction *ToBePromoted) const {
    // Promote only if all the operands can be statically expanded.
    // Indeed, we do not want to introduce any new kind of transitions.
    for (const Use &U : ToBePromoted->operands()) {
      const Value *Val = U.get();
      if (Val == getEndOfTransition()) {
        // If the use is a division and the transition is on the rhs,
        // we cannot promote the operation, otherwise we may create a
        // division by zero.
        if (canCauseUndefinedBehavior(ToBePromoted, U.getOperandNo()))
          return false;
        continue;
      }
      if (!isa<ConstantInt>(Val) && !isa<UndefValue>(Val) &&
          !isa<ConstantFP>(Val))
        return false;
    }
    // Check that the resulting operation is legal.
    int ISDOpcode = TLI.InstructionOpcodeToISD(ToBePromoted->getOpcode());
    if (!ISDOpcode)
      return false;
    return StressStoreExtract ||
           TLI.isOperationLegalOrCustom(
               ISDOpcode, TLI.getValueType(DL, getTransitionType(), true));
  }

  /// \brief Check whether or not \p Use can be combined
  /// with the transition.
  /// I.e., is it possible to do Use(Transition) => AnotherUse?
  bool canCombine(const Instruction *Use) { return isa<StoreInst>(Use); }

  /// \brief Record \p ToBePromoted as part of the chain to be promoted.
  void enqueueForPromotion(Instruction *ToBePromoted) {
    InstsToBePromoted.push_back(ToBePromoted);
  }

  /// \brief Set the instruction that will be combined with the transition.
  void recordCombineInstruction(Instruction *ToBeCombined) {
    assert(canCombine(ToBeCombined) && "Unsupported instruction to combine");
    CombineInst = ToBeCombined;
  }

  /// \brief Promote all the instructions enqueued for promotion if it is
  /// is profitable.
  /// \return True if the promotion happened, false otherwise.
  bool promote() {
    // Check if there is something to promote.
    // Right now, if we do not have anything to combine with,
    // we assume the promotion is not profitable.
    if (InstsToBePromoted.empty() || !CombineInst)
      return false;

    // Check cost.
    if (!StressStoreExtract && !isProfitableToPromote())
      return false;

    // Promote.
    for (auto &ToBePromoted : InstsToBePromoted)
      promoteImpl(ToBePromoted);
    InstsToBePromoted.clear();
    return true;
  }
};
} // End of anonymous namespace.

void VectorPromoteHelper::promoteImpl(Instruction *ToBePromoted) {
  // At this point, we know that all the operands of ToBePromoted but Def
  // can be statically promoted.
  // For Def, we need to use its parameter in ToBePromoted:
  // b = ToBePromoted ty1 a
  // Def = Transition ty1 b to ty2
  // Move the transition down.
  // 1. Replace all uses of the promoted operation by the transition.
  // = ... b => = ... Def.
  assert(ToBePromoted->getType() == Transition->getType() &&
         "The type of the result of the transition does not match "
         "the final type");
  ToBePromoted->replaceAllUsesWith(Transition);
  // 2. Update the type of the uses.
  // b = ToBePromoted ty2 Def => b = ToBePromoted ty1 Def.
  Type *TransitionTy = getTransitionType();
  ToBePromoted->mutateType(TransitionTy);
  // 3. Update all the operands of the promoted operation with promoted
  // operands.
  // b = ToBePromoted ty1 Def => b = ToBePromoted ty1 a.
  for (Use &U : ToBePromoted->operands()) {
    Value *Val = U.get();
    Value *NewVal = nullptr;
    if (Val == Transition)
      NewVal = Transition->getOperand(getTransitionOriginalValueIdx());
    else if (isa<UndefValue>(Val) || isa<ConstantInt>(Val) ||
             isa<ConstantFP>(Val)) {
      // Use a splat constant if it is not safe to use undef.
      NewVal = getConstantVector(
          cast<Constant>(Val),
          isa<UndefValue>(Val) ||
              canCauseUndefinedBehavior(ToBePromoted, U.getOperandNo()));
    } else
      llvm_unreachable("Did you modified shouldPromote and forgot to update "
                       "this?");
    ToBePromoted->setOperand(U.getOperandNo(), NewVal);
  }
  Transition->removeFromParent();
  Transition->insertAfter(ToBePromoted);
  Transition->setOperand(getTransitionOriginalValueIdx(), ToBePromoted);
}

/// Some targets can do store(extractelement) with one instruction.
/// Try to push the extractelement towards the stores when the target
/// has this feature and this is profitable.
bool CodeGenPrepare::OptimizeExtractElementInst(Instruction *Inst) {
  unsigned CombineCost = UINT_MAX;
  if (DisableStoreExtract || !TLI ||
      (!StressStoreExtract &&
       !TLI->canCombineStoreAndExtract(Inst->getOperand(0)->getType(),
                                       Inst->getOperand(1), CombineCost)))
    return false;

  // At this point we know that Inst is a vector to scalar transition.
  // Try to move it down the def-use chain, until:
  // - We can combine the transition with its single use
  //   => we got rid of the transition.
  // - We escape the current basic block
  //   => we would need to check that we are moving it at a cheaper place and
  //      we do not do that for now.
  BasicBlock *Parent = Inst->getParent();
  DEBUG(dbgs() << "Found an interesting transition: " << *Inst << '\n');
  VectorPromoteHelper VPH(*DL, *TLI, *TTI, Inst, CombineCost);
  // If the transition has more than one use, assume this is not going to be
  // beneficial.
  while (Inst->hasOneUse()) {
    Instruction *ToBePromoted = cast<Instruction>(*Inst->user_begin());
    DEBUG(dbgs() << "Use: " << *ToBePromoted << '\n');

    if (ToBePromoted->getParent() != Parent) {
      DEBUG(dbgs() << "Instruction to promote is in a different block ("
                   << ToBePromoted->getParent()->getName()
                   << ") than the transition (" << Parent->getName() << ").\n");
      return false;
    }

    if (VPH.canCombine(ToBePromoted)) {
      DEBUG(dbgs() << "Assume " << *Inst << '\n'
                   << "will be combined with: " << *ToBePromoted << '\n');
      VPH.recordCombineInstruction(ToBePromoted);
      bool Changed = VPH.promote();
      NumStoreExtractExposed += Changed;
      return Changed;
    }

    DEBUG(dbgs() << "Try promoting.\n");
    if (!VPH.canPromote(ToBePromoted) || !VPH.shouldPromote(ToBePromoted))
      return false;

    DEBUG(dbgs() << "Promoting is possible... Enqueue for promotion!\n");

    VPH.enqueueForPromotion(ToBePromoted);
    Inst = ToBePromoted;
  }
  return false;
}

bool CodeGenPrepare::OptimizeInst(Instruction *I, bool& ModifiedDT) {
  // Bail out if we inserted the instruction to prevent optimizations from
  // stepping on each other's toes.
  if (InsertedInsts.count(I))
    return false;

  if (PHINode *P = dyn_cast<PHINode>(I)) {
    // It is possible for very late stage optimizations (such as SimplifyCFG)
    // to introduce PHI nodes too late to be cleaned up.  If we detect such a
    // trivial PHI, go ahead and zap it here.
    if (Value *V = SimplifyInstruction(P, *DL, TLInfo, nullptr)) {
      P->replaceAllUsesWith(V);
      P->eraseFromParent();
      ++NumPHIsElim;
      return true;
    }
    return false;
  }

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    // If the source of the cast is a constant, then this should have
    // already been constant folded.  The only reason NOT to constant fold
    // it is if something (e.g. LSR) was careful to place the constant
    // evaluation in a block other than then one that uses it (e.g. to hoist
    // the address of globals out of a loop).  If this is the case, we don't
    // want to forward-subst the cast.
    if (isa<Constant>(CI->getOperand(0)))
      return false;

    if (TLI && OptimizeNoopCopyExpression(CI, *TLI, *DL))
      return true;

    if (isa<ZExtInst>(I) || isa<SExtInst>(I)) {
      /// Sink a zext or sext into its user blocks if the target type doesn't
      /// fit in one register
      if (TLI &&
          TLI->getTypeAction(CI->getContext(),
                             TLI->getValueType(*DL, CI->getType())) ==
              TargetLowering::TypeExpandInteger) {
        return SinkCast(CI);
      } else {
        bool MadeChange = MoveExtToFormExtLoad(I);
        return MadeChange | OptimizeExtUses(I);
      }
    }
    return false;
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (!TLI || !TLI->hasMultipleConditionRegisters())
      return OptimizeCmpExpression(CI);

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (TLI) {
      unsigned AS = LI->getPointerAddressSpace();
      return OptimizeMemoryInst(I, I->getOperand(0), LI->getType(), AS);
    }
    return false;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (TLI) {
      unsigned AS = SI->getPointerAddressSpace();
      return OptimizeMemoryInst(I, SI->getOperand(1),
                                SI->getOperand(0)->getType(), AS);
    }
    return false;
  }

  BinaryOperator *BinOp = dyn_cast<BinaryOperator>(I);

  if (BinOp && (BinOp->getOpcode() == Instruction::AShr ||
                BinOp->getOpcode() == Instruction::LShr)) {
    ConstantInt *CI = dyn_cast<ConstantInt>(BinOp->getOperand(1));
    if (TLI && CI && TLI->hasExtractBitsInsn())
      return OptimizeExtractBits(BinOp, CI, *TLI, *DL);

    return false;
  }

  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    if (GEPI->hasAllZeroIndices()) {
      /// The GEP operand must be a pointer, so must its result -> BitCast
      Instruction *NC = new BitCastInst(GEPI->getOperand(0), GEPI->getType(),
                                        GEPI->getName(), GEPI);
      GEPI->replaceAllUsesWith(NC);
      GEPI->eraseFromParent();
      ++NumGEPsElim;
      OptimizeInst(NC, ModifiedDT);
      return true;
    }
    return false;
  }

  if (CallInst *CI = dyn_cast<CallInst>(I))
    return OptimizeCallInst(CI, ModifiedDT);

  if (SelectInst *SI = dyn_cast<SelectInst>(I))
    return OptimizeSelectInst(SI);

  if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I))
    return OptimizeShuffleVectorInst(SVI);

  if (isa<ExtractElementInst>(I))
    return OptimizeExtractElementInst(I);

  return false;
}

// In this pass we look for GEP and cast instructions that are used
// across basic blocks and rewrite them to improve basic-block-at-a-time
// selection.
bool CodeGenPrepare::OptimizeBlock(BasicBlock &BB, bool& ModifiedDT) {
  SunkAddrs.clear();
  bool MadeChange = false;

  CurInstIterator = BB.begin();
  while (CurInstIterator != BB.end()) {
    MadeChange |= OptimizeInst(CurInstIterator++, ModifiedDT);
    if (ModifiedDT)
      return true;
  }
  MadeChange |= DupRetToEnableTailCallOpts(&BB);

  return MadeChange;
}

// llvm.dbg.value is far away from the value then iSel may not be able
// handle it properly. iSel will drop llvm.dbg.value if it can not
// find a node corresponding to the value.
bool CodeGenPrepare::PlaceDbgValues(Function &F) {
  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    Instruction *PrevNonDbgInst = nullptr;
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
      Instruction *Insn = BI++;
      DbgValueInst *DVI = dyn_cast<DbgValueInst>(Insn);
      // Leave dbg.values that refer to an alloca alone. These
      // instrinsics describe the address of a variable (= the alloca)
      // being taken.  They should not be moved next to the alloca
      // (and to the beginning of the scope), but rather stay close to
      // where said address is used.
      if (!DVI || (DVI->getValue() && isa<AllocaInst>(DVI->getValue()))) {
        PrevNonDbgInst = Insn;
        continue;
      }

      Instruction *VI = dyn_cast_or_null<Instruction>(DVI->getValue());
      if (VI && VI != PrevNonDbgInst && !VI->isTerminator()) {
        DEBUG(dbgs() << "Moving Debug Value before :\n" << *DVI << ' ' << *VI);
        DVI->removeFromParent();
        if (isa<PHINode>(VI))
          DVI->insertBefore(VI->getParent()->getFirstInsertionPt());
        else
          DVI->insertAfter(VI);
        MadeChange = true;
        ++NumDbgValueMoved;
      }
    }
  }
  return MadeChange;
}

// If there is a sequence that branches based on comparing a single bit
// against zero that can be combined into a single instruction, and the
// target supports folding these into a single instruction, sink the
// mask and compare into the branch uses. Do this before OptimizeBlock ->
// OptimizeInst -> OptimizeCmpExpression, which perturbs the pattern being
// searched for.
bool CodeGenPrepare::sinkAndCmp(Function &F) {
  if (!EnableAndCmpSinking)
    return false;
  if (!TLI || !TLI->isMaskAndBranchFoldingLegal())
    return false;
  bool MadeChange = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;

    // Does this BB end with the following?
    //   %andVal = and %val, #single-bit-set
    //   %icmpVal = icmp %andResult, 0
    //   br i1 %cmpVal label %dest1, label %dest2"
    BranchInst *Brcc = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Brcc || !Brcc->isConditional())
      continue;
    ICmpInst *Cmp = dyn_cast<ICmpInst>(Brcc->getOperand(0));
    if (!Cmp || Cmp->getParent() != BB)
      continue;
    ConstantInt *Zero = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (!Zero || !Zero->isZero())
      continue;
    Instruction *And = dyn_cast<Instruction>(Cmp->getOperand(0));
    if (!And || And->getOpcode() != Instruction::And || And->getParent() != BB)
      continue;
    ConstantInt* Mask = dyn_cast<ConstantInt>(And->getOperand(1));
    if (!Mask || !Mask->getUniqueInteger().isPowerOf2())
      continue;
    DEBUG(dbgs() << "found and; icmp ?,0; brcc\n"); DEBUG(BB->dump());

    // Push the "and; icmp" for any users that are conditional branches.
    // Since there can only be one branch use per BB, we don't need to keep
    // track of which BBs we insert into.
    for (Value::use_iterator UI = Cmp->use_begin(), E = Cmp->use_end();
         UI != E; ) {
      Use &TheUse = *UI;
      // Find brcc use.
      BranchInst *BrccUser = dyn_cast<BranchInst>(*UI);
      ++UI;
      if (!BrccUser || !BrccUser->isConditional())
        continue;
      BasicBlock *UserBB = BrccUser->getParent();
      if (UserBB == BB) continue;
      DEBUG(dbgs() << "found Brcc use\n");

      // Sink the "and; icmp" to use.
      MadeChange = true;
      BinaryOperator *NewAnd =
        BinaryOperator::CreateAnd(And->getOperand(0), And->getOperand(1), "",
                                  BrccUser);
      CmpInst *NewCmp =
        CmpInst::Create(Cmp->getOpcode(), Cmp->getPredicate(), NewAnd, Zero,
                        "", BrccUser);
      TheUse = NewCmp;
      ++NumAndCmpsMoved;
      DEBUG(BrccUser->getParent()->dump());
    }
  }
  return MadeChange;
}

/// \brief Retrieve the probabilities of a conditional branch. Returns true on
/// success, or returns false if no or invalid metadata was found.
static bool extractBranchMetadata(BranchInst *BI,
                                  uint64_t &ProbTrue, uint64_t &ProbFalse) {
  assert(BI->isConditional() &&
         "Looking for probabilities on unconditional branch?");
  auto *ProfileData = BI->getMetadata(LLVMContext::MD_prof);
  if (!ProfileData || ProfileData->getNumOperands() != 3)
    return false;

  const auto *CITrue =
      mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(1));
  const auto *CIFalse =
      mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(2));
  if (!CITrue || !CIFalse)
    return false;

  ProbTrue = CITrue->getValue().getZExtValue();
  ProbFalse = CIFalse->getValue().getZExtValue();

  return true;
}

/// \brief Scale down both weights to fit into uint32_t.
static void scaleWeights(uint64_t &NewTrue, uint64_t &NewFalse) {
  uint64_t NewMax = (NewTrue > NewFalse) ? NewTrue : NewFalse;
  uint32_t Scale = (NewMax / UINT32_MAX) + 1;
  NewTrue = NewTrue / Scale;
  NewFalse = NewFalse / Scale;
}

/// \brief Some targets prefer to split a conditional branch like:
/// \code
///   %0 = icmp ne i32 %a, 0
///   %1 = icmp ne i32 %b, 0
///   %or.cond = or i1 %0, %1
///   br i1 %or.cond, label %TrueBB, label %FalseBB
/// \endcode
/// into multiple branch instructions like:
/// \code
///   bb1:
///     %0 = icmp ne i32 %a, 0
///     br i1 %0, label %TrueBB, label %bb2
///   bb2:
///     %1 = icmp ne i32 %b, 0
///     br i1 %1, label %TrueBB, label %FalseBB
/// \endcode
/// This usually allows instruction selection to do even further optimizations
/// and combine the compare with the branch instruction. Currently this is
/// applied for targets which have "cheap" jump instructions.
///
/// FIXME: Remove the (equivalent?) implementation in SelectionDAG.
///
bool CodeGenPrepare::splitBranchCondition(Function &F) {
  if (!TM || !TM->Options.EnableFastISel || !TLI || TLI->isJumpExpensive())
    return false;

  bool MadeChange = false;
  for (auto &BB : F) {
    // Does this BB end with the following?
    //   %cond1 = icmp|fcmp|binary instruction ...
    //   %cond2 = icmp|fcmp|binary instruction ...
    //   %cond.or = or|and i1 %cond1, cond2
    //   br i1 %cond.or label %dest1, label %dest2"
    BinaryOperator *LogicOp;
    BasicBlock *TBB, *FBB;
    if (!match(BB.getTerminator(), m_Br(m_OneUse(m_BinOp(LogicOp)), TBB, FBB)))
      continue;

    unsigned Opc;
    Value *Cond1, *Cond2;
    if (match(LogicOp, m_And(m_OneUse(m_Value(Cond1)),
                             m_OneUse(m_Value(Cond2)))))
      Opc = Instruction::And;
    else if (match(LogicOp, m_Or(m_OneUse(m_Value(Cond1)),
                                 m_OneUse(m_Value(Cond2)))))
      Opc = Instruction::Or;
    else
      continue;

    if (!match(Cond1, m_CombineOr(m_Cmp(), m_BinOp())) ||
        !match(Cond2, m_CombineOr(m_Cmp(), m_BinOp()))   )
      continue;

    DEBUG(dbgs() << "Before branch condition splitting\n"; BB.dump());

    // Create a new BB.
    auto *InsertBefore = std::next(Function::iterator(BB))
        .getNodePtrUnchecked();
    auto TmpBB = BasicBlock::Create(BB.getContext(),
                                    BB.getName() + ".cond.split",
                                    BB.getParent(), InsertBefore);

    // Update original basic block by using the first condition directly by the
    // branch instruction and removing the no longer needed and/or instruction.
    auto *Br1 = cast<BranchInst>(BB.getTerminator());
    Br1->setCondition(Cond1);
    LogicOp->eraseFromParent();

    // Depending on the conditon we have to either replace the true or the false
    // successor of the original branch instruction.
    if (Opc == Instruction::And)
      Br1->setSuccessor(0, TmpBB);
    else
      Br1->setSuccessor(1, TmpBB);

    // Fill in the new basic block.
    auto *Br2 = IRBuilder<>(TmpBB).CreateCondBr(Cond2, TBB, FBB);
    if (auto *I = dyn_cast<Instruction>(Cond2)) {
      I->removeFromParent();
      I->insertBefore(Br2);
    }

    // Update PHI nodes in both successors. The original BB needs to be
    // replaced in one succesor's PHI nodes, because the branch comes now from
    // the newly generated BB (NewBB). In the other successor we need to add one
    // incoming edge to the PHI nodes, because both branch instructions target
    // now the same successor. Depending on the original branch condition
    // (and/or) we have to swap the successors (TrueDest, FalseDest), so that
    // we perfrom the correct update for the PHI nodes.
    // This doesn't change the successor order of the just created branch
    // instruction (or any other instruction).
    if (Opc == Instruction::Or)
      std::swap(TBB, FBB);

    // Replace the old BB with the new BB.
    for (auto &I : *TBB) {
      PHINode *PN = dyn_cast<PHINode>(&I);
      if (!PN)
        break;
      int i;
      while ((i = PN->getBasicBlockIndex(&BB)) >= 0)
        PN->setIncomingBlock(i, TmpBB);
    }

    // Add another incoming edge form the new BB.
    for (auto &I : *FBB) {
      PHINode *PN = dyn_cast<PHINode>(&I);
      if (!PN)
        break;
      auto *Val = PN->getIncomingValueForBlock(&BB);
      PN->addIncoming(Val, TmpBB);
    }

    // Update the branch weights (from SelectionDAGBuilder::
    // FindMergedConditions).
    if (Opc == Instruction::Or) {
      // Codegen X | Y as:
      // BB1:
      //   jmp_if_X TBB
      //   jmp TmpBB
      // TmpBB:
      //   jmp_if_Y TBB
      //   jmp FBB
      //

      // We have flexibility in setting Prob for BB1 and Prob for NewBB.
      // The requirement is that
      //   TrueProb for BB1 + (FalseProb for BB1 * TrueProb for TmpBB)
      //     = TrueProb for orignal BB.
      // Assuming the orignal weights are A and B, one choice is to set BB1's
      // weights to A and A+2B, and set TmpBB's weights to A and 2B. This choice
      // assumes that
      //   TrueProb for BB1 == FalseProb for BB1 * TrueProb for TmpBB.
      // Another choice is to assume TrueProb for BB1 equals to TrueProb for
      // TmpBB, but the math is more complicated.
      uint64_t TrueWeight, FalseWeight;
      if (extractBranchMetadata(Br1, TrueWeight, FalseWeight)) {
        uint64_t NewTrueWeight = TrueWeight;
        uint64_t NewFalseWeight = TrueWeight + 2 * FalseWeight;
        scaleWeights(NewTrueWeight, NewFalseWeight);
        Br1->setMetadata(LLVMContext::MD_prof, MDBuilder(Br1->getContext())
                         .createBranchWeights(TrueWeight, FalseWeight));

        NewTrueWeight = TrueWeight;
        NewFalseWeight = 2 * FalseWeight;
        scaleWeights(NewTrueWeight, NewFalseWeight);
        Br2->setMetadata(LLVMContext::MD_prof, MDBuilder(Br2->getContext())
                         .createBranchWeights(TrueWeight, FalseWeight));
      }
    } else {
      // Codegen X & Y as:
      // BB1:
      //   jmp_if_X TmpBB
      //   jmp FBB
      // TmpBB:
      //   jmp_if_Y TBB
      //   jmp FBB
      //
      //  This requires creation of TmpBB after CurBB.

      // We have flexibility in setting Prob for BB1 and Prob for TmpBB.
      // The requirement is that
      //   FalseProb for BB1 + (TrueProb for BB1 * FalseProb for TmpBB)
      //     = FalseProb for orignal BB.
      // Assuming the orignal weights are A and B, one choice is to set BB1's
      // weights to 2A+B and B, and set TmpBB's weights to 2A and B. This choice
      // assumes that
      //   FalseProb for BB1 == TrueProb for BB1 * FalseProb for TmpBB.
      uint64_t TrueWeight, FalseWeight;
      if (extractBranchMetadata(Br1, TrueWeight, FalseWeight)) {
        uint64_t NewTrueWeight = 2 * TrueWeight + FalseWeight;
        uint64_t NewFalseWeight = FalseWeight;
        scaleWeights(NewTrueWeight, NewFalseWeight);
        Br1->setMetadata(LLVMContext::MD_prof, MDBuilder(Br1->getContext())
                         .createBranchWeights(TrueWeight, FalseWeight));

        NewTrueWeight = 2 * TrueWeight;
        NewFalseWeight = FalseWeight;
        scaleWeights(NewTrueWeight, NewFalseWeight);
        Br2->setMetadata(LLVMContext::MD_prof, MDBuilder(Br2->getContext())
                         .createBranchWeights(TrueWeight, FalseWeight));
      }
    }

    // Note: No point in getting fancy here, since the DT info is never
    // available to CodeGenPrepare.
    ModifiedDT = true;

    MadeChange = true;

    DEBUG(dbgs() << "After branch condition splitting\n"; BB.dump();
          TmpBB->dump());
  }
  return MadeChange;
}
