//===- DeadStoreElimination.cpp - Fast Dead Store Elimination -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a trivial dead store elimination that only considers
// basic-block local redundant stores.
//
// FIXME: This should eventually be extended to be a post-dominator tree
// traversal.  Doing so would be pretty trivial.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "dse"

STATISTIC(NumFastStores, "Number of stores deleted");
STATISTIC(NumFastOther , "Number of other instrs removed");

namespace {
  struct DSE : public FunctionPass {
    AliasAnalysis *AA;
    MemoryDependenceAnalysis *MD;
    DominatorTree *DT;
    const TargetLibraryInfo *TLI;
    unsigned ScanLimit; // HLSL Change - Add ScanLimit

    static char ID; // Pass identification, replacement for typeid
    DSE(unsigned ScanLimit = 0) :
      FunctionPass(ID), AA(nullptr), MD(nullptr), DT(nullptr), ScanLimit(ScanLimit) {// HLSL Change - Add ScanLimit
      initializeDSEPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
      if (skipOptnoneFunction(F))
        return false;

      AA = &getAnalysis<AliasAnalysis>();
      MD = &getAnalysis<MemoryDependenceAnalysis>();
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      TLI = AA->getTargetLibraryInfo();

      bool Changed = false;
      for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
        // Only check non-dead blocks.  Dead blocks may have strange pointer
        // cycles that will confuse alias analysis.
        if (DT->isReachableFromEntry(I))
          Changed |= runOnBasicBlock(*I);

      AA = nullptr; MD = nullptr; DT = nullptr;
      return Changed;
    }

    bool runOnBasicBlock(BasicBlock &BB);
    bool HandleFree(CallInst *F);
    bool handleEndBlock(BasicBlock &BB);
    void RemoveAccessedObjects(const MemoryLocation &LoadedLoc,
                               SmallSetVector<Value *, 16> &DeadStackObjects,
                               const DataLayout &DL);

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MemoryDependenceAnalysis>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<MemoryDependenceAnalysis>();
    }
  };
}

char DSE::ID = 0;
INITIALIZE_PASS_BEGIN(DSE, "dse", "Dead Store Elimination", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(DSE, "dse", "Dead Store Elimination", false, false)

FunctionPass *llvm::createDeadStoreEliminationPass(unsigned ScanLimit) { return new DSE(ScanLimit); } // HLSL Change - add ScanLimit

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// DeleteDeadInstruction - Delete this instruction.  Before we do, go through
/// and zero out all the operands of this instruction.  If any of them become
/// dead, delete them and the computation tree that feeds them.
///
/// If ValueSet is non-null, remove any deleted instructions from it as well.
///
static void DeleteDeadInstruction(Instruction *I,
                               MemoryDependenceAnalysis &MD,
                               const TargetLibraryInfo *TLI,
                               SmallSetVector<Value*, 16> *ValueSet = nullptr) {
  SmallVector<Instruction*, 32> NowDeadInsts;

  NowDeadInsts.push_back(I);
  --NumFastOther;

  // Before we touch this instruction, remove it from memdep!
  do {
    Instruction *DeadInst = NowDeadInsts.pop_back_val();
    ++NumFastOther;

    // This instruction is dead, zap it, in stages.  Start by removing it from
    // MemDep, which needs to know the operands and needs it to be in the
    // function.
    MD.removeInstruction(DeadInst);

    for (unsigned op = 0, e = DeadInst->getNumOperands(); op != e; ++op) {
      Value *Op = DeadInst->getOperand(op);
      DeadInst->setOperand(op, nullptr);

      // If this operand just became dead, add it to the NowDeadInsts list.
      if (!Op->use_empty()) continue;

      if (Instruction *OpI = dyn_cast<Instruction>(Op))
        if (isInstructionTriviallyDead(OpI, TLI))
          NowDeadInsts.push_back(OpI);
    }

    DeadInst->eraseFromParent();

    if (ValueSet) ValueSet->remove(DeadInst);
  } while (!NowDeadInsts.empty());
}


/// hasMemoryWrite - Does this instruction write some memory?  This only returns
/// true for things that we can analyze with other helpers below.
static bool hasMemoryWrite(Instruction *I, const TargetLibraryInfo *TLI) {
  if (isa<StoreInst>(I))
    return true;
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
    case Intrinsic::init_trampoline:
    case Intrinsic::lifetime_end:
      return true;
    }
  }
  if (auto CS = CallSite(I)) {
    if (Function *F = CS.getCalledFunction()) {
      if (TLI && TLI->has(LibFunc::strcpy) &&
          F->getName() == TLI->getName(LibFunc::strcpy)) {
        return true;
      }
      if (TLI && TLI->has(LibFunc::strncpy) &&
          F->getName() == TLI->getName(LibFunc::strncpy)) {
        return true;
      }
      if (TLI && TLI->has(LibFunc::strcat) &&
          F->getName() == TLI->getName(LibFunc::strcat)) {
        return true;
      }
      if (TLI && TLI->has(LibFunc::strncat) &&
          F->getName() == TLI->getName(LibFunc::strncat)) {
        return true;
      }
    }
  }
  return false;
}

/// getLocForWrite - Return a Location stored to by the specified instruction.
/// If isRemovable returns true, this function and getLocForRead completely
/// describe the memory operations for this instruction.
static MemoryLocation getLocForWrite(Instruction *Inst, AliasAnalysis &AA) {
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
    return MemoryLocation::get(SI);

  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(Inst)) {
    // memcpy/memmove/memset.
    MemoryLocation Loc = MemoryLocation::getForDest(MI);
    return Loc;
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
  if (!II)
    return MemoryLocation();

  switch (II->getIntrinsicID()) {
  default:
    return MemoryLocation(); // Unhandled intrinsic.
  case Intrinsic::init_trampoline:
    // FIXME: We don't know the size of the trampoline, so we can't really
    // handle it here.
    return MemoryLocation(II->getArgOperand(0));
  case Intrinsic::lifetime_end: {
    uint64_t Len = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
    return MemoryLocation(II->getArgOperand(1), Len);
  }
  }
}

/// getLocForRead - Return the location read by the specified "hasMemoryWrite"
/// instruction if any.
static MemoryLocation getLocForRead(Instruction *Inst, AliasAnalysis &AA) {
  assert(hasMemoryWrite(Inst, AA.getTargetLibraryInfo()) &&
         "Unknown instruction case");

  // The only instructions that both read and write are the mem transfer
  // instructions (memcpy/memmove).
  if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(Inst))
    return MemoryLocation::getForSource(MTI);
  return MemoryLocation();
}


/// isRemovable - If the value of this instruction and the memory it writes to
/// is unused, may we delete this instruction?
static bool isRemovable(Instruction *I) {
  // Don't remove volatile/atomic stores.
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isUnordered();

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default: llvm_unreachable("doesn't pass 'hasMemoryWrite' predicate");
    case Intrinsic::lifetime_end:
      // Never remove dead lifetime_end's, e.g. because it is followed by a
      // free.
      return false;
    case Intrinsic::init_trampoline:
      // Always safe to remove init_trampoline.
      return true;

    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
      // Don't remove volatile memory intrinsics.
      return !cast<MemIntrinsic>(II)->isVolatile();
    }
  }

  if (auto CS = CallSite(I))
    return CS.getInstruction()->use_empty();

  return false;
}


/// isShortenable - Returns true if this instruction can be safely shortened in
/// length.
static bool isShortenable(Instruction *I) {
  // Don't shorten stores for now
  if (isa<StoreInst>(I))
    return false;

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
      default: return false;
      case Intrinsic::memset:
      case Intrinsic::memcpy:
        // Do shorten memory intrinsics.
        return true;
    }
  }

  // Don't shorten libcalls calls for now.

  return false;
}

/// getStoredPointerOperand - Return the pointer that is being written to.
static Value *getStoredPointerOperand(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return MI->getDest();

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    default: llvm_unreachable("Unexpected intrinsic!");
    case Intrinsic::init_trampoline:
      return II->getArgOperand(0);
    }
  }

  CallSite CS(I);
  // All the supported functions so far happen to have dest as their first
  // argument.
  return CS.getArgument(0);
}

static uint64_t getPointerSize(const Value *V, const DataLayout &DL,
                               const TargetLibraryInfo *TLI) {
  uint64_t Size;
  if (getObjectSize(V, Size, DL, TLI))
    return Size;
  return MemoryLocation::UnknownSize;
}

namespace {
  enum OverwriteResult
  {
    OverwriteComplete,
    OverwriteEnd,
    OverwriteUnknown
  };
}

/// isOverwrite - Return 'OverwriteComplete' if a store to the 'Later' location
/// completely overwrites a store to the 'Earlier' location.
/// 'OverwriteEnd' if the end of the 'Earlier' location is completely
/// overwritten by 'Later', or 'OverwriteUnknown' if nothing can be determined
static OverwriteResult isOverwrite(const MemoryLocation &Later,
                                   const MemoryLocation &Earlier,
                                   const DataLayout &DL,
                                   const TargetLibraryInfo *TLI,
                                   int64_t &EarlierOff, int64_t &LaterOff) {
  const Value *P1 = Earlier.Ptr->stripPointerCasts();
  const Value *P2 = Later.Ptr->stripPointerCasts();

  // If the start pointers are the same, we just have to compare sizes to see if
  // the later store was larger than the earlier store.
  if (P1 == P2) {
    // If we don't know the sizes of either access, then we can't do a
    // comparison.
    if (Later.Size == MemoryLocation::UnknownSize ||
        Earlier.Size == MemoryLocation::UnknownSize)
      return OverwriteUnknown;

    // Make sure that the Later size is >= the Earlier size.
    if (Later.Size >= Earlier.Size)
      return OverwriteComplete;
  }

  // Otherwise, we have to have size information, and the later store has to be
  // larger than the earlier one.
  if (Later.Size == MemoryLocation::UnknownSize ||
      Earlier.Size == MemoryLocation::UnknownSize)
    return OverwriteUnknown;

  // Check to see if the later store is to the entire object (either a global,
  // an alloca, or a byval/inalloca argument).  If so, then it clearly
  // overwrites any other store to the same object.
  const Value *UO1 = GetUnderlyingObject(P1, DL),
              *UO2 = GetUnderlyingObject(P2, DL);

  // If we can't resolve the same pointers to the same object, then we can't
  // analyze them at all.
  if (UO1 != UO2)
    return OverwriteUnknown;

  // If the "Later" store is to a recognizable object, get its size.
  uint64_t ObjectSize = getPointerSize(UO2, DL, TLI);
  if (ObjectSize != MemoryLocation::UnknownSize)
    if (ObjectSize == Later.Size && ObjectSize >= Earlier.Size)
      return OverwriteComplete;

  // Okay, we have stores to two completely different pointers.  Try to
  // decompose the pointer into a "base + constant_offset" form.  If the base
  // pointers are equal, then we can reason about the two stores.
  EarlierOff = 0;
  LaterOff = 0;
  const Value *BP1 = GetPointerBaseWithConstantOffset(P1, EarlierOff, DL);
  const Value *BP2 = GetPointerBaseWithConstantOffset(P2, LaterOff, DL);

  // If the base pointers still differ, we have two completely different stores.
  if (BP1 != BP2)
    return OverwriteUnknown;

  // The later store completely overlaps the earlier store if:
  //
  // 1. Both start at the same offset and the later one's size is greater than
  //    or equal to the earlier one's, or
  //
  //      |--earlier--|
  //      |--   later   --|
  //
  // 2. The earlier store has an offset greater than the later offset, but which
  //    still lies completely within the later store.
  //
  //        |--earlier--|
  //    |-----  later  ------|
  //
  // We have to be careful here as *Off is signed while *.Size is unsigned.
  if (EarlierOff >= LaterOff &&
      Later.Size >= Earlier.Size &&
      uint64_t(EarlierOff - LaterOff) + Earlier.Size <= Later.Size)
    return OverwriteComplete;

  // The other interesting case is if the later store overwrites the end of
  // the earlier store
  //
  //      |--earlier--|
  //                |--   later   --|
  //
  // In this case we may want to trim the size of earlier to avoid generating
  // writes to addresses which will definitely be overwritten later
  if (LaterOff > EarlierOff &&
      LaterOff < int64_t(EarlierOff + Earlier.Size) &&
      int64_t(LaterOff + Later.Size) >= int64_t(EarlierOff + Earlier.Size))
    return OverwriteEnd;

  // Otherwise, they don't completely overlap.
  return OverwriteUnknown;
}

/// isPossibleSelfRead - If 'Inst' might be a self read (i.e. a noop copy of a
/// memory region into an identical pointer) then it doesn't actually make its
/// input dead in the traditional sense.  Consider this case:
///
///   memcpy(A <- B)
///   memcpy(A <- A)
///
/// In this case, the second store to A does not make the first store to A dead.
/// The usual situation isn't an explicit A<-A store like this (which can be
/// trivially removed) but a case where two pointers may alias.
///
/// This function detects when it is unsafe to remove a dependent instruction
/// because the DSE inducing instruction may be a self-read.
static bool isPossibleSelfRead(Instruction *Inst,
                               const MemoryLocation &InstStoreLoc,
                               Instruction *DepWrite, AliasAnalysis &AA) {
  // Self reads can only happen for instructions that read memory.  Get the
  // location read.
  MemoryLocation InstReadLoc = getLocForRead(Inst, AA);
  if (!InstReadLoc.Ptr) return false;  // Not a reading instruction.

  // If the read and written loc obviously don't alias, it isn't a read.
  if (AA.isNoAlias(InstReadLoc, InstStoreLoc)) return false;

  // Okay, 'Inst' may copy over itself.  However, we can still remove a the
  // DepWrite instruction if we can prove that it reads from the same location
  // as Inst.  This handles useful cases like:
  //   memcpy(A <- B)
  //   memcpy(A <- B)
  // Here we don't know if A/B may alias, but we do know that B/B are must
  // aliases, so removing the first memcpy is safe (assuming it writes <= #
  // bytes as the second one.
  MemoryLocation DepReadLoc = getLocForRead(DepWrite, AA);

  if (DepReadLoc.Ptr && AA.isMustAlias(InstReadLoc.Ptr, DepReadLoc.Ptr))
    return false;

  // If DepWrite doesn't read memory or if we can't prove it is a must alias,
  // then it can't be considered dead.
  return true;
}


//===----------------------------------------------------------------------===//
// DSE Pass
//===----------------------------------------------------------------------===//

bool DSE::runOnBasicBlock(BasicBlock &BB) {
  bool MadeChange = false;

  // Do a top-down walk on the BB.
  for (BasicBlock::iterator BBI = BB.begin(), BBE = BB.end(); BBI != BBE; ) {
    Instruction *Inst = BBI++;

    // Handle 'free' calls specially.
    if (CallInst *F = isFreeCall(Inst, TLI)) {
      MadeChange |= HandleFree(F);
      continue;
    }

    // If we find something that writes memory, get its memory dependence.
    if (!hasMemoryWrite(Inst, TLI))
      continue;

    MemDepResult InstDep = MD->getDependency(Inst, ScanLimit);

    // Ignore any store where we can't find a local dependence.
    // FIXME: cross-block DSE would be fun. :)
    if (!InstDep.isDef() && !InstDep.isClobber())
      continue;

    // If we're storing the same value back to a pointer that we just
    // loaded from, then the store can be removed.
    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      if (LoadInst *DepLoad = dyn_cast<LoadInst>(InstDep.getInst())) {
        if (SI->getPointerOperand() == DepLoad->getPointerOperand() &&
            SI->getOperand(0) == DepLoad && isRemovable(SI)) {
          DEBUG(dbgs() << "DSE: Remove Store Of Load from same pointer:\n  "
                       << "LOAD: " << *DepLoad << "\n  STORE: " << *SI << '\n');

          // DeleteDeadInstruction can delete the current instruction.  Save BBI
          // in case we need it.
          WeakVH NextInst(BBI);

          DeleteDeadInstruction(SI, *MD, TLI);

          if (!NextInst)  // Next instruction deleted.
            BBI = BB.begin();
          else if (BBI != BB.begin())  // Revisit this instruction if possible.
            --BBI;
          ++NumFastStores;
          MadeChange = true;
          continue;
        }
      }
    }

    // Figure out what location is being stored to.
    MemoryLocation Loc = getLocForWrite(Inst, *AA);

    // If we didn't get a useful location, fail.
    if (!Loc.Ptr)
      continue;

    while (InstDep.isDef() || InstDep.isClobber()) {
      // Get the memory clobbered by the instruction we depend on.  MemDep will
      // skip any instructions that 'Loc' clearly doesn't interact with.  If we
      // end up depending on a may- or must-aliased load, then we can't optimize
      // away the store and we bail out.  However, if we depend on on something
      // that overwrites the memory location we *can* potentially optimize it.
      //
      // Find out what memory location the dependent instruction stores.
      Instruction *DepWrite = InstDep.getInst();
      MemoryLocation DepLoc = getLocForWrite(DepWrite, *AA);
      // If we didn't get a useful location, or if it isn't a size, bail out.
      if (!DepLoc.Ptr)
        break;

      // If we find a write that is a) removable (i.e., non-volatile), b) is
      // completely obliterated by the store to 'Loc', and c) which we know that
      // 'Inst' doesn't load from, then we can remove it.
      if (isRemovable(DepWrite) &&
          !isPossibleSelfRead(Inst, Loc, DepWrite, *AA)) {
        int64_t InstWriteOffset, DepWriteOffset;
        const DataLayout &DL = BB.getModule()->getDataLayout();
        OverwriteResult OR =
            isOverwrite(Loc, DepLoc, DL, AA->getTargetLibraryInfo(),
                        DepWriteOffset, InstWriteOffset);
        if (OR == OverwriteComplete) {
          DEBUG(dbgs() << "DSE: Remove Dead Store:\n  DEAD: "
                << *DepWrite << "\n  KILLER: " << *Inst << '\n');

          // Delete the store and now-dead instructions that feed it.
          DeleteDeadInstruction(DepWrite, *MD, TLI);
          ++NumFastStores;
          MadeChange = true;

          // DeleteDeadInstruction can delete the current instruction in loop
          // cases, reset BBI.
          BBI = Inst;
          if (BBI != BB.begin())
            --BBI;
          break;
        } else if (OR == OverwriteEnd && isShortenable(DepWrite)) {
          // TODO: base this on the target vector size so that if the earlier
          // store was too small to get vector writes anyway then its likely
          // a good idea to shorten it
          // Power of 2 vector writes are probably always a bad idea to optimize
          // as any store/memset/memcpy is likely using vector instructions so
          // shortening it to not vector size is likely to be slower
          MemIntrinsic* DepIntrinsic = cast<MemIntrinsic>(DepWrite);
          unsigned DepWriteAlign = DepIntrinsic->getAlignment();
          if (llvm::isPowerOf2_64(InstWriteOffset) ||
              ((DepWriteAlign != 0) && InstWriteOffset % DepWriteAlign == 0)) {

            DEBUG(dbgs() << "DSE: Remove Dead Store:\n  OW END: "
                  << *DepWrite << "\n  KILLER (offset "
                  << InstWriteOffset << ", "
                  << DepLoc.Size << ")"
                  << *Inst << '\n');

            Value* DepWriteLength = DepIntrinsic->getLength();
            Value* TrimmedLength = ConstantInt::get(DepWriteLength->getType(),
                                                    InstWriteOffset -
                                                    DepWriteOffset);
            DepIntrinsic->setLength(TrimmedLength);
            MadeChange = true;
          }
        }
      }

      // If this is a may-aliased store that is clobbering the store value, we
      // can keep searching past it for another must-aliased pointer that stores
      // to the same location.  For example, in:
      //   store -> P
      //   store -> Q
      //   store -> P
      // we can remove the first store to P even though we don't know if P and Q
      // alias.
      if (DepWrite == &BB.front()) break;

      // Can't look past this instruction if it might read 'Loc'.
      if (AA->getModRefInfo(DepWrite, Loc) & AliasAnalysis::Ref)
        break;

      InstDep = MD->getPointerDependencyFrom(Loc, false, DepWrite, &BB);
    }
  }

  // If this block ends in a return, unwind, or unreachable, all allocas are
  // dead at its end, which means stores to them are also dead.
  if (BB.getTerminator()->getNumSuccessors() == 0)
    MadeChange |= handleEndBlock(BB);

  return MadeChange;
}

/// Find all blocks that will unconditionally lead to the block BB and append
/// them to F.
static void FindUnconditionalPreds(SmallVectorImpl<BasicBlock *> &Blocks,
                                   BasicBlock *BB, DominatorTree *DT) {
  for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
    BasicBlock *Pred = *I;
    if (Pred == BB) continue;
    TerminatorInst *PredTI = Pred->getTerminator();
    if (PredTI->getNumSuccessors() != 1)
      continue;

    if (DT->isReachableFromEntry(Pred))
      Blocks.push_back(Pred);
  }
}

/// HandleFree - Handle frees of entire structures whose dependency is a store
/// to a field of that structure.
bool DSE::HandleFree(CallInst *F) {
  bool MadeChange = false;

  MemoryLocation Loc = MemoryLocation(F->getOperand(0));
  SmallVector<BasicBlock *, 16> Blocks;
  Blocks.push_back(F->getParent());
  const DataLayout &DL = F->getModule()->getDataLayout();

  while (!Blocks.empty()) {
    BasicBlock *BB = Blocks.pop_back_val();
    Instruction *InstPt = BB->getTerminator();
    if (BB == F->getParent()) InstPt = F;

    MemDepResult Dep = MD->getPointerDependencyFrom(Loc, false, InstPt, BB);
    while (Dep.isDef() || Dep.isClobber()) {
      Instruction *Dependency = Dep.getInst();
      if (!hasMemoryWrite(Dependency, TLI) || !isRemovable(Dependency))
        break;

      Value *DepPointer =
          GetUnderlyingObject(getStoredPointerOperand(Dependency), DL);

      // Check for aliasing.
      if (!AA->isMustAlias(F->getArgOperand(0), DepPointer))
        break;

      Instruction *Next = std::next(BasicBlock::iterator(Dependency));

      // DCE instructions only used to calculate that store
      DeleteDeadInstruction(Dependency, *MD, TLI);
      ++NumFastStores;
      MadeChange = true;

      // Inst's old Dependency is now deleted. Compute the next dependency,
      // which may also be dead, as in
      //    s[0] = 0;
      //    s[1] = 0; // This has just been deleted.
      //    free(s);
      Dep = MD->getPointerDependencyFrom(Loc, false, Next, BB);
    }

    if (Dep.isNonLocal())
      FindUnconditionalPreds(Blocks, BB, DT);
  }

  return MadeChange;
}

/// handleEndBlock - Remove dead stores to stack-allocated locations in the
/// function end block.  Ex:
/// %A = alloca i32
/// ...
/// store i32 1, i32* %A
/// ret void
bool DSE::handleEndBlock(BasicBlock &BB) {
  bool MadeChange = false;

  // Keep track of all of the stack objects that are dead at the end of the
  // function.
  SmallSetVector<Value*, 16> DeadStackObjects;

  // Find all of the alloca'd pointers in the entry block.
  BasicBlock *Entry = BB.getParent()->begin();
  for (BasicBlock::iterator I = Entry->begin(), E = Entry->end(); I != E; ++I) {
    if (isa<AllocaInst>(I))
      DeadStackObjects.insert(I);

    // Okay, so these are dead heap objects, but if the pointer never escapes
    // then it's leaked by this function anyways.
    else if (isAllocLikeFn(I, TLI) && !PointerMayBeCaptured(I, true, true))
      DeadStackObjects.insert(I);
  }

  // Treat byval or inalloca arguments the same, stores to them are dead at the
  // end of the function.
  for (Function::arg_iterator AI = BB.getParent()->arg_begin(),
       AE = BB.getParent()->arg_end(); AI != AE; ++AI)
    if (AI->hasByValOrInAllocaAttr())
      DeadStackObjects.insert(AI);

  const DataLayout &DL = BB.getModule()->getDataLayout();

  // Scan the basic block backwards
  for (BasicBlock::iterator BBI = BB.end(); BBI != BB.begin(); ){
    --BBI;

    // If we find a store, check to see if it points into a dead stack value.
    if (hasMemoryWrite(BBI, TLI) && isRemovable(BBI)) {
      // See through pointer-to-pointer bitcasts
      SmallVector<Value *, 4> Pointers;
      GetUnderlyingObjects(getStoredPointerOperand(BBI), Pointers, DL);

      // Stores to stack values are valid candidates for removal.
      bool AllDead = true;
      for (SmallVectorImpl<Value *>::iterator I = Pointers.begin(),
           E = Pointers.end(); I != E; ++I)
        if (!DeadStackObjects.count(*I)) {
          AllDead = false;
          break;
        }

      if (AllDead) {
        Instruction *Dead = BBI++;

        DEBUG(dbgs() << "DSE: Dead Store at End of Block:\n  DEAD: "
                     << *Dead << "\n  Objects: ";
              for (SmallVectorImpl<Value *>::iterator I = Pointers.begin(),
                   E = Pointers.end(); I != E; ++I) {
                dbgs() << **I;
                if (std::next(I) != E)
                  dbgs() << ", ";
              }
              dbgs() << '\n');

        // DCE instructions only used to calculate that store.
        DeleteDeadInstruction(Dead, *MD, TLI, &DeadStackObjects);
        ++NumFastStores;
        MadeChange = true;
        continue;
      }
    }

    // Remove any dead non-memory-mutating instructions.
    if (isInstructionTriviallyDead(BBI, TLI)) {
      Instruction *Inst = BBI++;
      DeleteDeadInstruction(Inst, *MD, TLI, &DeadStackObjects);
      ++NumFastOther;
      MadeChange = true;
      continue;
    }

    if (isa<AllocaInst>(BBI)) {
      // Remove allocas from the list of dead stack objects; there can't be
      // any references before the definition.
      DeadStackObjects.remove(BBI);
      continue;
    }

    if (auto CS = CallSite(BBI)) {
      // Remove allocation function calls from the list of dead stack objects; 
      // there can't be any references before the definition.
      if (isAllocLikeFn(BBI, TLI))
        DeadStackObjects.remove(BBI);

      // If this call does not access memory, it can't be loading any of our
      // pointers.
      if (AA->doesNotAccessMemory(CS))
        continue;

      // If the call might load from any of our allocas, then any store above
      // the call is live.
      DeadStackObjects.remove_if([&](Value *I) {
        // See if the call site touches the value.
        AliasAnalysis::ModRefResult A = AA->getModRefInfo(
            CS, I, getPointerSize(I, DL, AA->getTargetLibraryInfo()));

        return A == AliasAnalysis::ModRef || A == AliasAnalysis::Ref;
      });

      // If all of the allocas were clobbered by the call then we're not going
      // to find anything else to process.
      if (DeadStackObjects.empty())
        break;

      continue;
    }

    MemoryLocation LoadedLoc;

    // If we encounter a use of the pointer, it is no longer considered dead
    if (LoadInst *L = dyn_cast<LoadInst>(BBI)) {
      if (!L->isUnordered()) // Be conservative with atomic/volatile load
        break;
      LoadedLoc = MemoryLocation::get(L);
    } else if (VAArgInst *V = dyn_cast<VAArgInst>(BBI)) {
      LoadedLoc = MemoryLocation::get(V);
    } else if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(BBI)) {
      LoadedLoc = MemoryLocation::getForSource(MTI);
    } else if (!BBI->mayReadFromMemory()) {
      // Instruction doesn't read memory.  Note that stores that weren't removed
      // above will hit this case.
      continue;
    } else {
      // Unknown inst; assume it clobbers everything.
      break;
    }

    // Remove any allocas from the DeadPointer set that are loaded, as this
    // makes any stores above the access live.
    RemoveAccessedObjects(LoadedLoc, DeadStackObjects, DL);

    // If all of the allocas were clobbered by the access then we're not going
    // to find anything else to process.
    if (DeadStackObjects.empty())
      break;
  }

  return MadeChange;
}

/// RemoveAccessedObjects - Check to see if the specified location may alias any
/// of the stack objects in the DeadStackObjects set.  If so, they become live
/// because the location is being loaded.
void DSE::RemoveAccessedObjects(const MemoryLocation &LoadedLoc,
                                SmallSetVector<Value *, 16> &DeadStackObjects,
                                const DataLayout &DL) {
  const Value *UnderlyingPointer = GetUnderlyingObject(LoadedLoc.Ptr, DL);

  // A constant can't be in the dead pointer set.
  if (isa<Constant>(UnderlyingPointer))
    return;

  // If the kill pointer can be easily reduced to an alloca, don't bother doing
  // extraneous AA queries.
  if (isa<AllocaInst>(UnderlyingPointer) || isa<Argument>(UnderlyingPointer)) {
    DeadStackObjects.remove(const_cast<Value*>(UnderlyingPointer));
    return;
  }

  // Remove objects that could alias LoadedLoc.
  DeadStackObjects.remove_if([&](Value *I) {
    // See if the loaded location could alias the stack location.
    MemoryLocation StackLoc(I,
                            getPointerSize(I, DL, AA->getTargetLibraryInfo()));
    return !AA->isNoAlias(StackLoc, LoadedLoc);
  });
}
