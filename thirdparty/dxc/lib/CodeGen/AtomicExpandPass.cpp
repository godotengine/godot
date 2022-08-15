//===-- AtomicExpandPass.cpp - Expand atomic instructions -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass (at IR level) to replace atomic instructions with
// either (intrinsic-based) load-linked/store-conditional loops or AtomicCmpXchg.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "atomic-expand"

namespace {
  class AtomicExpand: public FunctionPass {
    const TargetMachine *TM;
    const TargetLowering *TLI;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit AtomicExpand(const TargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM), TLI(nullptr) {
      initializeAtomicExpandPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

  private:
    bool bracketInstWithFences(Instruction *I, AtomicOrdering Order,
                               bool IsStore, bool IsLoad);
    bool expandAtomicLoad(LoadInst *LI);
    bool expandAtomicLoadToLL(LoadInst *LI);
    bool expandAtomicLoadToCmpXchg(LoadInst *LI);
    bool expandAtomicStore(StoreInst *SI);
    bool tryExpandAtomicRMW(AtomicRMWInst *AI);
    bool expandAtomicRMWToLLSC(AtomicRMWInst *AI);
    bool expandAtomicRMWToCmpXchg(AtomicRMWInst *AI);
    bool expandAtomicCmpXchg(AtomicCmpXchgInst *CI);
    bool isIdempotentRMW(AtomicRMWInst *AI);
    bool simplifyIdempotentRMW(AtomicRMWInst *AI);
  };
}

char AtomicExpand::ID = 0;
char &llvm::AtomicExpandID = AtomicExpand::ID;
INITIALIZE_TM_PASS(AtomicExpand, "atomic-expand",
    "Expand Atomic calls in terms of either load-linked & store-conditional or cmpxchg",
    false, false)

FunctionPass *llvm::createAtomicExpandPass(const TargetMachine *TM) {
  return new AtomicExpand(TM);
}

bool AtomicExpand::runOnFunction(Function &F) {
  if (!TM || !TM->getSubtargetImpl(F)->enableAtomicExpand())
    return false;
  TLI = TM->getSubtargetImpl(F)->getTargetLowering();

  SmallVector<Instruction *, 1> AtomicInsts;

  // Changing control-flow while iterating through it is a bad idea, so gather a
  // list of all atomic instructions before we start.
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (I->isAtomic())
      AtomicInsts.push_back(&*I);
  }

  bool MadeChange = false;
  for (auto I : AtomicInsts) {
    auto LI = dyn_cast<LoadInst>(I);
    auto SI = dyn_cast<StoreInst>(I);
    auto RMWI = dyn_cast<AtomicRMWInst>(I);
    auto CASI = dyn_cast<AtomicCmpXchgInst>(I);
    assert((LI || SI || RMWI || CASI || isa<FenceInst>(I)) &&
           "Unknown atomic instruction");

    auto FenceOrdering = Monotonic;
    bool IsStore, IsLoad;
    if (TLI->getInsertFencesForAtomic()) {
      if (LI && isAtLeastAcquire(LI->getOrdering())) {
        FenceOrdering = LI->getOrdering();
        LI->setOrdering(Monotonic);
        IsStore = false;
        IsLoad = true;
      } else if (SI && isAtLeastRelease(SI->getOrdering())) {
        FenceOrdering = SI->getOrdering();
        SI->setOrdering(Monotonic);
        IsStore = true;
        IsLoad = false;
      } else if (RMWI && (isAtLeastRelease(RMWI->getOrdering()) ||
                          isAtLeastAcquire(RMWI->getOrdering()))) {
        FenceOrdering = RMWI->getOrdering();
        RMWI->setOrdering(Monotonic);
        IsStore = IsLoad = true;
      } else if (CASI && !TLI->hasLoadLinkedStoreConditional() &&
                 (isAtLeastRelease(CASI->getSuccessOrdering()) ||
                  isAtLeastAcquire(CASI->getSuccessOrdering()))) {
        // If a compare and swap is lowered to LL/SC, we can do smarter fence
        // insertion, with a stronger one on the success path than on the
        // failure path. As a result, fence insertion is directly done by
        // expandAtomicCmpXchg in that case.
        FenceOrdering = CASI->getSuccessOrdering();
        CASI->setSuccessOrdering(Monotonic);
        CASI->setFailureOrdering(Monotonic);
        IsStore = IsLoad = true;
      }

      if (FenceOrdering != Monotonic) {
        MadeChange |= bracketInstWithFences(I, FenceOrdering, IsStore, IsLoad);
      }
    }

    if (LI && TLI->shouldExpandAtomicLoadInIR(LI)) {
      MadeChange |= expandAtomicLoad(LI);
    } else if (SI && TLI->shouldExpandAtomicStoreInIR(SI)) {
      MadeChange |= expandAtomicStore(SI);
    } else if (RMWI) {
      // There are two different ways of expanding RMW instructions:
      // - into a load if it is idempotent
      // - into a Cmpxchg/LL-SC loop otherwise
      // we try them in that order.

      if (isIdempotentRMW(RMWI) && simplifyIdempotentRMW(RMWI)) {
        MadeChange = true;
      } else {
        MadeChange |= tryExpandAtomicRMW(RMWI);
      }
    } else if (CASI && TLI->hasLoadLinkedStoreConditional()) {
      MadeChange |= expandAtomicCmpXchg(CASI);
    }
  }
  return MadeChange;
}

bool AtomicExpand::bracketInstWithFences(Instruction *I, AtomicOrdering Order,
                                         bool IsStore, bool IsLoad) {
  IRBuilder<> Builder(I);

  auto LeadingFence = TLI->emitLeadingFence(Builder, Order, IsStore, IsLoad);

  auto TrailingFence = TLI->emitTrailingFence(Builder, Order, IsStore, IsLoad);
  // The trailing fence is emitted before the instruction instead of after
  // because there is no easy way of setting Builder insertion point after
  // an instruction. So we must erase it from the BB, and insert it back
  // in the right place.
  // We have a guard here because not every atomic operation generates a
  // trailing fence.
  if (TrailingFence) {
    TrailingFence->removeFromParent();
    TrailingFence->insertAfter(I);
  }

  return (LeadingFence || TrailingFence);
}

bool AtomicExpand::expandAtomicLoad(LoadInst *LI) {
  if (TLI->hasLoadLinkedStoreConditional())
    return expandAtomicLoadToLL(LI);
  else
    return expandAtomicLoadToCmpXchg(LI);
}

bool AtomicExpand::expandAtomicLoadToLL(LoadInst *LI) {
  IRBuilder<> Builder(LI);

  // On some architectures, load-linked instructions are atomic for larger
  // sizes than normal loads. For example, the only 64-bit load guaranteed
  // to be single-copy atomic by ARM is an ldrexd (A3.5.3).
  Value *Val =
      TLI->emitLoadLinked(Builder, LI->getPointerOperand(), LI->getOrdering());

  LI->replaceAllUsesWith(Val);
  LI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicLoadToCmpXchg(LoadInst *LI) {
  IRBuilder<> Builder(LI);
  AtomicOrdering Order = LI->getOrdering();
  Value *Addr = LI->getPointerOperand();
  Type *Ty = cast<PointerType>(Addr->getType())->getElementType();
  Constant *DummyVal = Constant::getNullValue(Ty);

  Value *Pair = Builder.CreateAtomicCmpXchg(
      Addr, DummyVal, DummyVal, Order,
      AtomicCmpXchgInst::getStrongestFailureOrdering(Order));
  Value *Loaded = Builder.CreateExtractValue(Pair, 0, "loaded");

  LI->replaceAllUsesWith(Loaded);
  LI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicStore(StoreInst *SI) {
  // This function is only called on atomic stores that are too large to be
  // atomic if implemented as a native store. So we replace them by an
  // atomic swap, that can be implemented for example as a ldrex/strex on ARM
  // or lock cmpxchg8/16b on X86, as these are atomic for larger sizes.
  // It is the responsibility of the target to only signal expansion via
  // shouldExpandAtomicRMW in cases where this is required and possible.
  IRBuilder<> Builder(SI);
  AtomicRMWInst *AI =
      Builder.CreateAtomicRMW(AtomicRMWInst::Xchg, SI->getPointerOperand(),
                              SI->getValueOperand(), SI->getOrdering());
  SI->eraseFromParent();

  // Now we have an appropriate swap instruction, lower it as usual.
  return tryExpandAtomicRMW(AI);
}

bool AtomicExpand::tryExpandAtomicRMW(AtomicRMWInst *AI) {
  switch (TLI->shouldExpandAtomicRMWInIR(AI)) {
  case TargetLoweringBase::AtomicRMWExpansionKind::None:
    return false;
  case TargetLoweringBase::AtomicRMWExpansionKind::LLSC: {
    assert(TLI->hasLoadLinkedStoreConditional() &&
           "TargetLowering requested we expand AtomicRMW instruction into "
           "load-linked/store-conditional combos, but such instructions aren't "
           "supported");

    return expandAtomicRMWToLLSC(AI);
  }
  case TargetLoweringBase::AtomicRMWExpansionKind::CmpXChg: {
    return expandAtomicRMWToCmpXchg(AI);
  }
  }
  llvm_unreachable("Unhandled case in tryExpandAtomicRMW");
}

/// Emit IR to implement the given atomicrmw operation on values in registers,
/// returning the new value.
static Value *performAtomicOp(AtomicRMWInst::BinOp Op, IRBuilder<> &Builder,
                              Value *Loaded, Value *Inc) {
  Value *NewVal;
  switch (Op) {
  case AtomicRMWInst::Xchg:
    return Inc;
  case AtomicRMWInst::Add:
    return Builder.CreateAdd(Loaded, Inc, "new");
  case AtomicRMWInst::Sub:
    return Builder.CreateSub(Loaded, Inc, "new");
  case AtomicRMWInst::And:
    return Builder.CreateAnd(Loaded, Inc, "new");
  case AtomicRMWInst::Nand:
    return Builder.CreateNot(Builder.CreateAnd(Loaded, Inc), "new");
  case AtomicRMWInst::Or:
    return Builder.CreateOr(Loaded, Inc, "new");
  case AtomicRMWInst::Xor:
    return Builder.CreateXor(Loaded, Inc, "new");
  case AtomicRMWInst::Max:
    NewVal = Builder.CreateICmpSGT(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::Min:
    NewVal = Builder.CreateICmpSLE(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::UMax:
    NewVal = Builder.CreateICmpUGT(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  case AtomicRMWInst::UMin:
    NewVal = Builder.CreateICmpULE(Loaded, Inc);
    return Builder.CreateSelect(NewVal, Loaded, Inc, "new");
  default:
    llvm_unreachable("Unknown atomic op");
  }
}

bool AtomicExpand::expandAtomicRMWToLLSC(AtomicRMWInst *AI) {
  AtomicOrdering MemOpOrder = AI->getOrdering();
  Value *Addr = AI->getPointerOperand();
  BasicBlock *BB = AI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();

  // Given: atomicrmw some_op iN* %addr, iN %incr ordering
  //
  // The standard expansion we produce is:
  //     [...]
  //     fence?
  // atomicrmw.start:
  //     %loaded = @load.linked(%addr)
  //     %new = some_op iN %loaded, %incr
  //     %stored = @store_conditional(%new, %addr)
  //     %try_again = icmp i32 ne %stored, 0
  //     br i1 %try_again, label %loop, label %atomicrmw.end
  // atomicrmw.end:
  //     fence?
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(AI, "atomicrmw.end");
  BasicBlock *LoopBB =  BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

  // This grabs the DebugLoc from AI.
  IRBuilder<> Builder(AI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we might want a fence too. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  Value *Loaded = TLI->emitLoadLinked(Builder, Addr, MemOpOrder);

  Value *NewVal =
      performAtomicOp(AI->getOperation(), Builder, Loaded, AI->getValOperand());

  Value *StoreSuccess =
      TLI->emitStoreConditional(Builder, NewVal, Addr, MemOpOrder);
  Value *TryAgain = Builder.CreateICmpNE(
      StoreSuccess, ConstantInt::get(IntegerType::get(Ctx, 32), 0), "tryagain");
  Builder.CreateCondBr(TryAgain, LoopBB, ExitBB);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());

  AI->replaceAllUsesWith(Loaded);
  AI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicRMWToCmpXchg(AtomicRMWInst *AI) {
  AtomicOrdering MemOpOrder =
      AI->getOrdering() == Unordered ? Monotonic : AI->getOrdering();
  Value *Addr = AI->getPointerOperand();
  BasicBlock *BB = AI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();

  // Given: atomicrmw some_op iN* %addr, iN %incr ordering
  //
  // The standard expansion we produce is:
  //     [...]
  //     %init_loaded = load atomic iN* %addr
  //     br label %loop
  // loop:
  //     %loaded = phi iN [ %init_loaded, %entry ], [ %new_loaded, %loop ]
  //     %new = some_op iN %loaded, %incr
  //     %pair = cmpxchg iN* %addr, iN %loaded, iN %new
  //     %new_loaded = extractvalue { iN, i1 } %pair, 0
  //     %success = extractvalue { iN, i1 } %pair, 1
  //     br i1 %success, label %atomicrmw.end, label %loop
  // atomicrmw.end:
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(AI, "atomicrmw.end");
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

  // This grabs the DebugLoc from AI.
  IRBuilder<> Builder(AI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we want a load. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  LoadInst *InitLoaded = Builder.CreateLoad(Addr);
  // Atomics require at least natural alignment.
  InitLoaded->setAlignment(AI->getType()->getPrimitiveSizeInBits());
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  PHINode *Loaded = Builder.CreatePHI(AI->getType(), 2, "loaded");
  Loaded->addIncoming(InitLoaded, BB);

  Value *NewVal =
      performAtomicOp(AI->getOperation(), Builder, Loaded, AI->getValOperand());

  Value *Pair = Builder.CreateAtomicCmpXchg(
      Addr, Loaded, NewVal, MemOpOrder,
      AtomicCmpXchgInst::getStrongestFailureOrdering(MemOpOrder));
  Value *NewLoaded = Builder.CreateExtractValue(Pair, 0, "newloaded");
  Loaded->addIncoming(NewLoaded, LoopBB);

  Value *Success = Builder.CreateExtractValue(Pair, 1, "success");
  Builder.CreateCondBr(Success, ExitBB, LoopBB);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());

  AI->replaceAllUsesWith(NewLoaded);
  AI->eraseFromParent();

  return true;
}

bool AtomicExpand::expandAtomicCmpXchg(AtomicCmpXchgInst *CI) {
  AtomicOrdering SuccessOrder = CI->getSuccessOrdering();
  AtomicOrdering FailureOrder = CI->getFailureOrdering();
  Value *Addr = CI->getPointerOperand();
  BasicBlock *BB = CI->getParent();
  Function *F = BB->getParent();
  LLVMContext &Ctx = F->getContext();
  // If getInsertFencesForAtomic() returns true, then the target does not want
  // to deal with memory orders, and emitLeading/TrailingFence should take care
  // of everything. Otherwise, emitLeading/TrailingFence are no-op and we
  // should preserve the ordering.
  AtomicOrdering MemOpOrder =
      TLI->getInsertFencesForAtomic() ? Monotonic : SuccessOrder;

  // Given: cmpxchg some_op iN* %addr, iN %desired, iN %new success_ord fail_ord
  //
  // The full expansion we produce is:
  //     [...]
  //     fence?
  // cmpxchg.start:
  //     %loaded = @load.linked(%addr)
  //     %should_store = icmp eq %loaded, %desired
  //     br i1 %should_store, label %cmpxchg.trystore,
  //                          label %cmpxchg.failure
  // cmpxchg.trystore:
  //     %stored = @store_conditional(%new, %addr)
  //     %success = icmp eq i32 %stored, 0
  //     br i1 %success, label %cmpxchg.success, label %loop/%cmpxchg.failure
  // cmpxchg.success:
  //     fence?
  //     br label %cmpxchg.end
  // cmpxchg.failure:
  //     fence?
  //     br label %cmpxchg.end
  // cmpxchg.end:
  //     %success = phi i1 [true, %cmpxchg.success], [false, %cmpxchg.failure]
  //     %restmp = insertvalue { iN, i1 } undef, iN %loaded, 0
  //     %res = insertvalue { iN, i1 } %restmp, i1 %success, 1
  //     [...]
  BasicBlock *ExitBB = BB->splitBasicBlock(CI, "cmpxchg.end");
  auto FailureBB = BasicBlock::Create(Ctx, "cmpxchg.failure", F, ExitBB);
  auto SuccessBB = BasicBlock::Create(Ctx, "cmpxchg.success", F, FailureBB);
  auto TryStoreBB = BasicBlock::Create(Ctx, "cmpxchg.trystore", F, SuccessBB);
  auto LoopBB = BasicBlock::Create(Ctx, "cmpxchg.start", F, TryStoreBB);

  // This grabs the DebugLoc from CI
  IRBuilder<> Builder(CI);

  // The split call above "helpfully" added a branch at the end of BB (to the
  // wrong place), but we might want a fence too. It's easiest to just remove
  // the branch entirely.
  std::prev(BB->end())->eraseFromParent();
  Builder.SetInsertPoint(BB);
  TLI->emitLeadingFence(Builder, SuccessOrder, /*IsStore=*/true,
                        /*IsLoad=*/true);
  Builder.CreateBr(LoopBB);

  // Start the main loop block now that we've taken care of the preliminaries.
  Builder.SetInsertPoint(LoopBB);
  Value *Loaded = TLI->emitLoadLinked(Builder, Addr, MemOpOrder);
  Value *ShouldStore =
      Builder.CreateICmpEQ(Loaded, CI->getCompareOperand(), "should_store");

  // If the cmpxchg doesn't actually need any ordering when it fails, we can
  // jump straight past that fence instruction (if it exists).
  Builder.CreateCondBr(ShouldStore, TryStoreBB, FailureBB);

  Builder.SetInsertPoint(TryStoreBB);
  Value *StoreSuccess = TLI->emitStoreConditional(
      Builder, CI->getNewValOperand(), Addr, MemOpOrder);
  StoreSuccess = Builder.CreateICmpEQ(
      StoreSuccess, ConstantInt::get(Type::getInt32Ty(Ctx), 0), "success");
  Builder.CreateCondBr(StoreSuccess, SuccessBB,
                       CI->isWeak() ? FailureBB : LoopBB);

  // Make sure later instructions don't get reordered with a fence if necessary.
  Builder.SetInsertPoint(SuccessBB);
  TLI->emitTrailingFence(Builder, SuccessOrder, /*IsStore=*/true,
                         /*IsLoad=*/true);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(FailureBB);
  TLI->emitTrailingFence(Builder, FailureOrder, /*IsStore=*/true,
                         /*IsLoad=*/true);
  Builder.CreateBr(ExitBB);

  // Finally, we have control-flow based knowledge of whether the cmpxchg
  // succeeded or not. We expose this to later passes by converting any
  // subsequent "icmp eq/ne %loaded, %oldval" into a use of an appropriate PHI.

  // Setup the builder so we can create any PHIs we need.
  Builder.SetInsertPoint(ExitBB, ExitBB->begin());
  PHINode *Success = Builder.CreatePHI(Type::getInt1Ty(Ctx), 2);
  Success->addIncoming(ConstantInt::getTrue(Ctx), SuccessBB);
  Success->addIncoming(ConstantInt::getFalse(Ctx), FailureBB);

  // Look for any users of the cmpxchg that are just comparing the loaded value
  // against the desired one, and replace them with the CFG-derived version.
  SmallVector<ExtractValueInst *, 2> PrunedInsts;
  for (auto User : CI->users()) {
    ExtractValueInst *EV = dyn_cast<ExtractValueInst>(User);
    if (!EV)
      continue;

    assert(EV->getNumIndices() == 1 && EV->getIndices()[0] <= 1 &&
           "weird extraction from { iN, i1 }");

    if (EV->getIndices()[0] == 0)
      EV->replaceAllUsesWith(Loaded);
    else
      EV->replaceAllUsesWith(Success);

    PrunedInsts.push_back(EV);
  }

  // We can remove the instructions now we're no longer iterating through them.
  for (auto EV : PrunedInsts)
    EV->eraseFromParent();

  if (!CI->use_empty()) {
    // Some use of the full struct return that we don't understand has happened,
    // so we've got to reconstruct it properly.
    Value *Res;
    Res = Builder.CreateInsertValue(UndefValue::get(CI->getType()), Loaded, 0);
    Res = Builder.CreateInsertValue(Res, Success, 1);

    CI->replaceAllUsesWith(Res);
  }

  CI->eraseFromParent();
  return true;
}

bool AtomicExpand::isIdempotentRMW(AtomicRMWInst* RMWI) {
  auto C = dyn_cast<ConstantInt>(RMWI->getValOperand());
  if(!C)
    return false;

  AtomicRMWInst::BinOp Op = RMWI->getOperation();
  switch(Op) {
    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor:
      return C->isZero();
    case AtomicRMWInst::And:
      return C->isMinusOne();
    // FIXME: we could also treat Min/Max/UMin/UMax by the INT_MIN/INT_MAX/...
    default:
      return false;
  }
}

bool AtomicExpand::simplifyIdempotentRMW(AtomicRMWInst* RMWI) {
  if (auto ResultingLoad = TLI->lowerIdempotentRMWIntoFencedLoad(RMWI)) {
    if (TLI->shouldExpandAtomicLoadInIR(ResultingLoad))
      expandAtomicLoad(ResultingLoad);
    return true;
  }
  return false;
}
