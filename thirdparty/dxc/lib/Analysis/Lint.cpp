//===-- Lint.cpp - Check for common errors in LLVM IR ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass statically checks for common and easily-identified constructs
// which produce undefined or likely unintended behavior in LLVM IR.
//
// It is not a guarantee of correctness, in two ways. First, it isn't
// comprehensive. There are checks which could be done statically which are
// not yet implemented. Some of these are indicated by TODO comments, but
// those aren't comprehensive either. Second, many conditions cannot be
// checked statically. This pass does no dynamic instrumentation, so it
// can't check for all possible problems.
//
// Another limitation is that it assumes all code will be executed. A store
// through a null pointer in a basic block which is never reached is harmless,
// but this pass will warn about it anyway. This is the main reason why most
// of these checks live here instead of in the Verifier pass.
//
// Optimization passes may make conditions that this pass checks for more or
// less obvious. If an optimization pass appears to be introducing a warning,
// it may be that the optimization pass is merely exposing an existing
// condition in the code.
//
// This code may be run before instcombine. In many cases, instcombine checks
// for the same kinds of things and turns instructions with undefined behavior
// into unreachable (or equivalent). Because of this, this pass makes some
// effort to look through bitcasts and so on.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Lint.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  namespace MemRef {
    static const unsigned Read     = 1;
    static const unsigned Write    = 2;
    static const unsigned Callee   = 4;
    static const unsigned Branchee = 8;
  }

  class Lint : public FunctionPass, public InstVisitor<Lint> {
    friend class InstVisitor<Lint>;

    void visitFunction(Function &F);

    void visitCallSite(CallSite CS);
    void visitMemoryReference(Instruction &I, Value *Ptr,
                              uint64_t Size, unsigned Align,
                              Type *Ty, unsigned Flags);
    void visitEHBeginCatch(IntrinsicInst *II);
    void visitEHEndCatch(IntrinsicInst *II);

    void visitCallInst(CallInst &I);
    void visitInvokeInst(InvokeInst &I);
    void visitReturnInst(ReturnInst &I);
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitXor(BinaryOperator &I);
    void visitSub(BinaryOperator &I);
    void visitLShr(BinaryOperator &I);
    void visitAShr(BinaryOperator &I);
    void visitShl(BinaryOperator &I);
    void visitSDiv(BinaryOperator &I);
    void visitUDiv(BinaryOperator &I);
    void visitSRem(BinaryOperator &I);
    void visitURem(BinaryOperator &I);
    void visitAllocaInst(AllocaInst &I);
    void visitVAArgInst(VAArgInst &I);
    void visitIndirectBrInst(IndirectBrInst &I);
    void visitExtractElementInst(ExtractElementInst &I);
    void visitInsertElementInst(InsertElementInst &I);
    void visitUnreachableInst(UnreachableInst &I);

    Value *findValue(Value *V, const DataLayout &DL, bool OffsetOk) const;
    Value *findValueImpl(Value *V, const DataLayout &DL, bool OffsetOk,
                         SmallPtrSetImpl<Value *> &Visited) const;

  public:
    Module *Mod;
    AliasAnalysis *AA;
    AssumptionCache *AC;
    DominatorTree *DT;
    TargetLibraryInfo *TLI;

    std::string Messages;
    raw_string_ostream MessagesStr;

    static char ID; // Pass identification, replacement for typeid
    Lint() : FunctionPass(ID), MessagesStr(Messages) {
      initializeLintPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
    }
    void print(raw_ostream &O, const Module *M) const override {}

    void WriteValues(ArrayRef<const Value *> Vs) {
      for (const Value *V : Vs) {
        if (!V)
          continue;
        if (isa<Instruction>(V)) {
          MessagesStr << *V << '\n';
        } else {
          V->printAsOperand(MessagesStr, true, Mod);
          MessagesStr << '\n';
        }
      }
    }

    /// \brief A check failed, so printout out the condition and the message.
    ///
    /// This provides a nice place to put a breakpoint if you want to see why
    /// something is not correct.
    void CheckFailed(const Twine &Message) { MessagesStr << Message << '\n'; }

    /// \brief A check failed (with values to print).
    ///
    /// This calls the Message-only version so that the above is easier to set
    /// a breakpoint on.
    template <typename T1, typename... Ts>
    void CheckFailed(const Twine &Message, const T1 &V1, const Ts &...Vs) {
      CheckFailed(Message);
      WriteValues({V1, Vs...});
    }
  };
}

char Lint::ID = 0;
INITIALIZE_PASS_BEGIN(Lint, "lint", "Statically lint-checks LLVM IR",
                      false, true)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(Lint, "lint", "Statically lint-checks LLVM IR",
                    false, true)

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, ...) \
    do { if (!(C)) { CheckFailed(__VA_ARGS__); return; } } while (0)

// Lint::run - This is the main Analysis entry point for a
// function.
//
bool Lint::runOnFunction(Function &F) {
  Mod = F.getParent();
  AA = &getAnalysis<AliasAnalysis>();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  visit(F);
  dbgs() << MessagesStr.str();
  Messages.clear();
  return false;
}

void Lint::visitFunction(Function &F) {
  // This isn't undefined behavior, it's just a little unusual, and it's a
  // fairly common mistake to neglect to name a function.
  Assert(F.hasName() || F.hasLocalLinkage(),
         "Unusual: Unnamed function with non-local linkage", &F);

  // TODO: Check for irreducible control flow.
}

void Lint::visitCallSite(CallSite CS) {
  Instruction &I = *CS.getInstruction();
  Value *Callee = CS.getCalledValue();
  const DataLayout &DL = CS->getModule()->getDataLayout();

  visitMemoryReference(I, Callee, MemoryLocation::UnknownSize, 0, nullptr,
                       MemRef::Callee);

  if (Function *F = dyn_cast<Function>(findValue(Callee, DL,
                                                 /*OffsetOk=*/false))) {
    Assert(CS.getCallingConv() == F->getCallingConv(),
           "Undefined behavior: Caller and callee calling convention differ",
           &I);

    FunctionType *FT = F->getFunctionType();
    unsigned NumActualArgs = CS.arg_size();

    Assert(FT->isVarArg() ? FT->getNumParams() <= NumActualArgs
                          : FT->getNumParams() == NumActualArgs,
           "Undefined behavior: Call argument count mismatches callee "
           "argument count",
           &I);

    Assert(FT->getReturnType() == I.getType(),
           "Undefined behavior: Call return type mismatches "
           "callee return type",
           &I);

    // Check argument types (in case the callee was casted) and attributes.
    // TODO: Verify that caller and callee attributes are compatible.
    Function::arg_iterator PI = F->arg_begin(), PE = F->arg_end();
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    for (; AI != AE; ++AI) {
      Value *Actual = *AI;
      if (PI != PE) {
        Argument *Formal = PI++;
        Assert(Formal->getType() == Actual->getType(),
               "Undefined behavior: Call argument type mismatches "
               "callee parameter type",
               &I);

        // Check that noalias arguments don't alias other arguments. This is
        // not fully precise because we don't know the sizes of the dereferenced
        // memory regions.
        if (Formal->hasNoAliasAttr() && Actual->getType()->isPointerTy())
          for (CallSite::arg_iterator BI = CS.arg_begin(); BI != AE; ++BI)
            if (AI != BI && (*BI)->getType()->isPointerTy()) {
              AliasResult Result = AA->alias(*AI, *BI);
              Assert(Result != MustAlias && Result != PartialAlias,
                     "Unusual: noalias argument aliases another argument", &I);
            }

        // Check that an sret argument points to valid memory.
        if (Formal->hasStructRetAttr() && Actual->getType()->isPointerTy()) {
          Type *Ty =
            cast<PointerType>(Formal->getType())->getElementType();
          visitMemoryReference(I, Actual, AA->getTypeStoreSize(Ty),
                               DL.getABITypeAlignment(Ty), Ty,
                               MemRef::Read | MemRef::Write);
        }
      }
    }
  }

  if (CS.isCall() && cast<CallInst>(CS.getInstruction())->isTailCall())
    for (CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
         AI != AE; ++AI) {
      Value *Obj = findValue(*AI, DL, /*OffsetOk=*/true);
      Assert(!isa<AllocaInst>(Obj),
             "Undefined behavior: Call with \"tail\" keyword references "
             "alloca",
             &I);
    }


  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
    switch (II->getIntrinsicID()) {
    default: break;

    // TODO: Check more intrinsics

    case Intrinsic::memcpy: {
      MemCpyInst *MCI = cast<MemCpyInst>(&I);
      // TODO: If the size is known, use it.
      visitMemoryReference(I, MCI->getDest(), MemoryLocation::UnknownSize,
                           MCI->getAlignment(), nullptr, MemRef::Write);
      visitMemoryReference(I, MCI->getSource(), MemoryLocation::UnknownSize,
                           MCI->getAlignment(), nullptr, MemRef::Read);

      // Check that the memcpy arguments don't overlap. The AliasAnalysis API
      // isn't expressive enough for what we really want to do. Known partial
      // overlap is not distinguished from the case where nothing is known.
      uint64_t Size = 0;
      if (const ConstantInt *Len =
              dyn_cast<ConstantInt>(findValue(MCI->getLength(), DL,
                                              /*OffsetOk=*/false)))
        if (Len->getValue().isIntN(32))
          Size = Len->getValue().getZExtValue();
      Assert(AA->alias(MCI->getSource(), Size, MCI->getDest(), Size) !=
                 MustAlias,
             "Undefined behavior: memcpy source and destination overlap", &I);
      break;
    }
    case Intrinsic::memmove: {
      MemMoveInst *MMI = cast<MemMoveInst>(&I);
      // TODO: If the size is known, use it.
      visitMemoryReference(I, MMI->getDest(), MemoryLocation::UnknownSize,
                           MMI->getAlignment(), nullptr, MemRef::Write);
      visitMemoryReference(I, MMI->getSource(), MemoryLocation::UnknownSize,
                           MMI->getAlignment(), nullptr, MemRef::Read);
      break;
    }
    case Intrinsic::memset: {
      MemSetInst *MSI = cast<MemSetInst>(&I);
      // TODO: If the size is known, use it.
      visitMemoryReference(I, MSI->getDest(), MemoryLocation::UnknownSize,
                           MSI->getAlignment(), nullptr, MemRef::Write);
      break;
    }

    case Intrinsic::vastart:
      Assert(I.getParent()->getParent()->isVarArg(),
             "Undefined behavior: va_start called in a non-varargs function",
             &I);

      visitMemoryReference(I, CS.getArgument(0), MemoryLocation::UnknownSize, 0,
                           nullptr, MemRef::Read | MemRef::Write);
      break;
    case Intrinsic::vacopy:
      visitMemoryReference(I, CS.getArgument(0), MemoryLocation::UnknownSize, 0,
                           nullptr, MemRef::Write);
      visitMemoryReference(I, CS.getArgument(1), MemoryLocation::UnknownSize, 0,
                           nullptr, MemRef::Read);
      break;
    case Intrinsic::vaend:
      visitMemoryReference(I, CS.getArgument(0), MemoryLocation::UnknownSize, 0,
                           nullptr, MemRef::Read | MemRef::Write);
      break;

    case Intrinsic::stackrestore:
      // Stackrestore doesn't read or write memory, but it sets the
      // stack pointer, which the compiler may read from or write to
      // at any time, so check it for both readability and writeability.
      visitMemoryReference(I, CS.getArgument(0), MemoryLocation::UnknownSize, 0,
                           nullptr, MemRef::Read | MemRef::Write);
      break;

    case Intrinsic::eh_begincatch:
      visitEHBeginCatch(II);
      break;
    case Intrinsic::eh_endcatch:
      visitEHEndCatch(II);
      break;
    }
}

void Lint::visitCallInst(CallInst &I) {
  return visitCallSite(&I);
}

void Lint::visitInvokeInst(InvokeInst &I) {
  return visitCallSite(&I);
}

void Lint::visitReturnInst(ReturnInst &I) {
  Function *F = I.getParent()->getParent();
  Assert(!F->doesNotReturn(),
         "Unusual: Return statement in function with noreturn attribute", &I);

  if (Value *V = I.getReturnValue()) {
    Value *Obj =
        findValue(V, F->getParent()->getDataLayout(), /*OffsetOk=*/true);
    Assert(!isa<AllocaInst>(Obj), "Unusual: Returning alloca value", &I);
  }
}

// TODO: Check that the reference is in bounds.
// TODO: Check readnone/readonly function attributes.
void Lint::visitMemoryReference(Instruction &I,
                                Value *Ptr, uint64_t Size, unsigned Align,
                                Type *Ty, unsigned Flags) {
  // If no memory is being referenced, it doesn't matter if the pointer
  // is valid.
  if (Size == 0)
    return;

  Value *UnderlyingObject =
      findValue(Ptr, I.getModule()->getDataLayout(), /*OffsetOk=*/true);
  Assert(!isa<ConstantPointerNull>(UnderlyingObject),
         "Undefined behavior: Null pointer dereference", &I);
  Assert(!isa<UndefValue>(UnderlyingObject),
         "Undefined behavior: Undef pointer dereference", &I);
  Assert(!isa<ConstantInt>(UnderlyingObject) ||
             !cast<ConstantInt>(UnderlyingObject)->isAllOnesValue(),
         "Unusual: All-ones pointer dereference", &I);
  Assert(!isa<ConstantInt>(UnderlyingObject) ||
             !cast<ConstantInt>(UnderlyingObject)->isOne(),
         "Unusual: Address one pointer dereference", &I);

  if (Flags & MemRef::Write) {
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(UnderlyingObject))
      Assert(!GV->isConstant(), "Undefined behavior: Write to read-only memory",
             &I);
    Assert(!isa<Function>(UnderlyingObject) &&
               !isa<BlockAddress>(UnderlyingObject),
           "Undefined behavior: Write to text section", &I);
  }
  if (Flags & MemRef::Read) {
    Assert(!isa<Function>(UnderlyingObject), "Unusual: Load from function body",
           &I);
    Assert(!isa<BlockAddress>(UnderlyingObject),
           "Undefined behavior: Load from block address", &I);
  }
  if (Flags & MemRef::Callee) {
    Assert(!isa<BlockAddress>(UnderlyingObject),
           "Undefined behavior: Call to block address", &I);
  }
  if (Flags & MemRef::Branchee) {
    Assert(!isa<Constant>(UnderlyingObject) ||
               isa<BlockAddress>(UnderlyingObject),
           "Undefined behavior: Branch to non-blockaddress", &I);
  }

  // Check for buffer overflows and misalignment.
  // Only handles memory references that read/write something simple like an
  // alloca instruction or a global variable.
  auto &DL = I.getModule()->getDataLayout();
  int64_t Offset = 0;
  if (Value *Base = GetPointerBaseWithConstantOffset(Ptr, Offset, DL)) {
    // OK, so the access is to a constant offset from Ptr.  Check that Ptr is
    // something we can handle and if so extract the size of this base object
    // along with its alignment.
    uint64_t BaseSize = MemoryLocation::UnknownSize;
    unsigned BaseAlign = 0;

    if (AllocaInst *AI = dyn_cast<AllocaInst>(Base)) {
      Type *ATy = AI->getAllocatedType();
      if (!AI->isArrayAllocation() && ATy->isSized())
        BaseSize = DL.getTypeAllocSize(ATy);
      BaseAlign = AI->getAlignment();
      if (BaseAlign == 0 && ATy->isSized())
        BaseAlign = DL.getABITypeAlignment(ATy);
    } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
      // If the global may be defined differently in another compilation unit
      // then don't warn about funky memory accesses.
      if (GV->hasDefinitiveInitializer()) {
        Type *GTy = GV->getType()->getElementType();
        if (GTy->isSized())
          BaseSize = DL.getTypeAllocSize(GTy);
        BaseAlign = GV->getAlignment();
        if (BaseAlign == 0 && GTy->isSized())
          BaseAlign = DL.getABITypeAlignment(GTy);
      }
    }

    // Accesses from before the start or after the end of the object are not
    // defined.
    Assert(Size == MemoryLocation::UnknownSize ||
               BaseSize == MemoryLocation::UnknownSize ||
               (Offset >= 0 && Offset + Size <= BaseSize),
           "Undefined behavior: Buffer overflow", &I);

    // Accesses that say that the memory is more aligned than it is are not
    // defined.
    if (Align == 0 && Ty && Ty->isSized())
      Align = DL.getABITypeAlignment(Ty);
    Assert(!BaseAlign || Align <= MinAlign(BaseAlign, Offset),
           "Undefined behavior: Memory reference address is misaligned", &I);
  }
}

void Lint::visitLoadInst(LoadInst &I) {
  visitMemoryReference(I, I.getPointerOperand(),
                       AA->getTypeStoreSize(I.getType()), I.getAlignment(),
                       I.getType(), MemRef::Read);
}

void Lint::visitStoreInst(StoreInst &I) {
  visitMemoryReference(I, I.getPointerOperand(),
                       AA->getTypeStoreSize(I.getOperand(0)->getType()),
                       I.getAlignment(),
                       I.getOperand(0)->getType(), MemRef::Write);
}

void Lint::visitXor(BinaryOperator &I) {
  Assert(!isa<UndefValue>(I.getOperand(0)) || !isa<UndefValue>(I.getOperand(1)),
         "Undefined result: xor(undef, undef)", &I);
}

void Lint::visitSub(BinaryOperator &I) {
  Assert(!isa<UndefValue>(I.getOperand(0)) || !isa<UndefValue>(I.getOperand(1)),
         "Undefined result: sub(undef, undef)", &I);
}

void Lint::visitLShr(BinaryOperator &I) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(
          findValue(I.getOperand(1), I.getModule()->getDataLayout(),
                    /*OffsetOk=*/false)))
    Assert(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
           "Undefined result: Shift count out of range", &I);
}

void Lint::visitAShr(BinaryOperator &I) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(findValue(
          I.getOperand(1), I.getModule()->getDataLayout(), /*OffsetOk=*/false)))
    Assert(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
           "Undefined result: Shift count out of range", &I);
}

void Lint::visitShl(BinaryOperator &I) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(findValue(
          I.getOperand(1), I.getModule()->getDataLayout(), /*OffsetOk=*/false)))
    Assert(CI->getValue().ult(cast<IntegerType>(I.getType())->getBitWidth()),
           "Undefined result: Shift count out of range", &I);
}

static bool
allPredsCameFromLandingPad(BasicBlock *BB,
                           SmallSet<BasicBlock *, 4> &VisitedBlocks) {
  VisitedBlocks.insert(BB);
  if (BB->isLandingPad())
    return true;
  // If we find a block with no predecessors, the search failed.
  if (pred_empty(BB))
    return false;
  for (BasicBlock *Pred : predecessors(BB)) {
    if (VisitedBlocks.count(Pred))
      continue;
    if (!allPredsCameFromLandingPad(Pred, VisitedBlocks))
      return false;
  }
  return true;
}

static bool
allSuccessorsReachEndCatch(BasicBlock *BB, BasicBlock::iterator InstBegin,
                           IntrinsicInst **SecondBeginCatch,
                           SmallSet<BasicBlock *, 4> &VisitedBlocks) {
  VisitedBlocks.insert(BB);
  for (BasicBlock::iterator I = InstBegin, E = BB->end(); I != E; ++I) {
    IntrinsicInst *IC = dyn_cast<IntrinsicInst>(I);
    if (IC && IC->getIntrinsicID() == Intrinsic::eh_endcatch)
      return true;
    // If we find another begincatch while looking for an endcatch,
    // that's also an error.
    if (IC && IC->getIntrinsicID() == Intrinsic::eh_begincatch) {
      *SecondBeginCatch = IC;
      return false;
    }
  }

  // If we reach a block with no successors while searching, the
  // search has failed.
  if (succ_empty(BB))
    return false;
  // Otherwise, search all of the successors.
  for (BasicBlock *Succ : successors(BB)) {
    if (VisitedBlocks.count(Succ))
      continue;
    if (!allSuccessorsReachEndCatch(Succ, Succ->begin(), SecondBeginCatch,
                                    VisitedBlocks))
      return false;
  }
  return true;
}

void Lint::visitEHBeginCatch(IntrinsicInst *II) {
  // The checks in this function make a potentially dubious assumption about
  // the CFG, namely that any block involved in a catch is only used for the
  // catch.  This will very likely be true of IR generated by a front end,
  // but it may cease to be true, for example, if the IR is run through a
  // pass which combines similar blocks.
  //
  // In general, if we encounter a block the isn't dominated by the catch
  // block while we are searching the catch block's successors for a call
  // to end catch intrinsic, then it is possible that it will be legal for
  // a path through this block to never reach a call to llvm.eh.endcatch.
  // An analogous statement could be made about our search for a landing
  // pad among the catch block's predecessors.
  //
  // What is actually required is that no path is possible at runtime that
  // reaches a call to llvm.eh.begincatch without having previously visited
  // a landingpad instruction and that no path is possible at runtime that
  // calls llvm.eh.begincatch and does not subsequently call llvm.eh.endcatch
  // (mentally adjusting for the fact that in reality these calls will be
  // removed before code generation).
  //
  // Because this is a lint check, we take a pessimistic approach and warn if
  // the control flow is potentially incorrect.

  SmallSet<BasicBlock *, 4> VisitedBlocks;
  BasicBlock *CatchBB = II->getParent();

  // The begin catch must occur in a landing pad block or all paths
  // to it must have come from a landing pad.
  Assert(allPredsCameFromLandingPad(CatchBB, VisitedBlocks),
         "llvm.eh.begincatch may be reachable without passing a landingpad",
         II);

  // Reset the visited block list.
  VisitedBlocks.clear();

  IntrinsicInst *SecondBeginCatch = nullptr;

  // This has to be called before it is asserted.  Otherwise, the first assert
  // below can never be hit.
  bool EndCatchFound = allSuccessorsReachEndCatch(
      CatchBB, std::next(static_cast<BasicBlock::iterator>(II)),
      &SecondBeginCatch, VisitedBlocks);
  Assert(
      SecondBeginCatch == nullptr,
      "llvm.eh.begincatch may be called a second time before llvm.eh.endcatch",
      II, SecondBeginCatch);
  Assert(EndCatchFound,
         "Some paths from llvm.eh.begincatch may not reach llvm.eh.endcatch",
         II);
}

static bool allPredCameFromBeginCatch(
    BasicBlock *BB, BasicBlock::reverse_iterator InstRbegin,
    IntrinsicInst **SecondEndCatch, SmallSet<BasicBlock *, 4> &VisitedBlocks) {
  VisitedBlocks.insert(BB);
  // Look for a begincatch in this block.
  for (BasicBlock::reverse_iterator RI = InstRbegin, RE = BB->rend(); RI != RE;
       ++RI) {
    IntrinsicInst *IC = dyn_cast<IntrinsicInst>(&*RI);
    if (IC && IC->getIntrinsicID() == Intrinsic::eh_begincatch)
      return true;
    // If we find another end catch before we find a begin catch, that's
    // an error.
    if (IC && IC->getIntrinsicID() == Intrinsic::eh_endcatch) {
      *SecondEndCatch = IC;
      return false;
    }
    // If we encounter a landingpad instruction, the search failed.
    if (isa<LandingPadInst>(*RI))
      return false;
  }
  // If while searching we find a block with no predeccesors,
  // the search failed.
  if (pred_empty(BB))
    return false;
  // Search any predecessors we haven't seen before.
  for (BasicBlock *Pred : predecessors(BB)) {
    if (VisitedBlocks.count(Pred))
      continue;
    if (!allPredCameFromBeginCatch(Pred, Pred->rbegin(), SecondEndCatch,
                                   VisitedBlocks))
      return false;
  }
  return true;
}

void Lint::visitEHEndCatch(IntrinsicInst *II) {
  // The check in this function makes a potentially dubious assumption about
  // the CFG, namely that any block involved in a catch is only used for the
  // catch.  This will very likely be true of IR generated by a front end,
  // but it may cease to be true, for example, if the IR is run through a
  // pass which combines similar blocks.
  //
  // In general, if we encounter a block the isn't post-dominated by the
  // end catch block while we are searching the end catch block's predecessors
  // for a call to the begin catch intrinsic, then it is possible that it will
  // be legal for a path to reach the end catch block without ever having
  // called llvm.eh.begincatch.
  //
  // What is actually required is that no path is possible at runtime that
  // reaches a call to llvm.eh.endcatch without having previously visited
  // a call to llvm.eh.begincatch (mentally adjusting for the fact that in
  // reality these calls will be removed before code generation).
  //
  // Because this is a lint check, we take a pessimistic approach and warn if
  // the control flow is potentially incorrect.

  BasicBlock *EndCatchBB = II->getParent();

  // Alls paths to the end catch call must pass through a begin catch call.

  // If llvm.eh.begincatch wasn't called in the current block, we'll use this
  // lambda to recursively look for it in predecessors.
  SmallSet<BasicBlock *, 4> VisitedBlocks;
  IntrinsicInst *SecondEndCatch = nullptr;

  // This has to be called before it is asserted.  Otherwise, the first assert
  // below can never be hit.
  bool BeginCatchFound =
      allPredCameFromBeginCatch(EndCatchBB, BasicBlock::reverse_iterator(II),
                                &SecondEndCatch, VisitedBlocks);
  Assert(
      SecondEndCatch == nullptr,
      "llvm.eh.endcatch may be called a second time after llvm.eh.begincatch",
      II, SecondEndCatch);
  Assert(BeginCatchFound,
         "llvm.eh.endcatch may be reachable without passing llvm.eh.begincatch",
         II);
}

static bool isZero(Value *V, const DataLayout &DL, DominatorTree *DT,
                   AssumptionCache *AC) {
  // Assume undef could be zero.
  if (isa<UndefValue>(V))
    return true;

  VectorType *VecTy = dyn_cast<VectorType>(V->getType());
  if (!VecTy) {
    unsigned BitWidth = V->getType()->getIntegerBitWidth();
    APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
    computeKnownBits(V, KnownZero, KnownOne, DL, 0, AC,
                     dyn_cast<Instruction>(V), DT);
    return KnownZero.isAllOnesValue();
  }

  // Per-component check doesn't work with zeroinitializer
  Constant *C = dyn_cast<Constant>(V);
  if (!C)
    return false;

  if (C->isZeroValue())
    return true;

  // For a vector, KnownZero will only be true if all values are zero, so check
  // this per component
  unsigned BitWidth = VecTy->getElementType()->getIntegerBitWidth();
  for (unsigned I = 0, N = VecTy->getNumElements(); I != N; ++I) {
    Constant *Elem = C->getAggregateElement(I);
    if (isa<UndefValue>(Elem))
      return true;

    APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
    computeKnownBits(Elem, KnownZero, KnownOne, DL);
    if (KnownZero.isAllOnesValue())
      return true;
  }

  return false;
}

void Lint::visitSDiv(BinaryOperator &I) {
  Assert(!isZero(I.getOperand(1), I.getModule()->getDataLayout(), DT, AC),
         "Undefined behavior: Division by zero", &I);
}

void Lint::visitUDiv(BinaryOperator &I) {
  Assert(!isZero(I.getOperand(1), I.getModule()->getDataLayout(), DT, AC),
         "Undefined behavior: Division by zero", &I);
}

void Lint::visitSRem(BinaryOperator &I) {
  Assert(!isZero(I.getOperand(1), I.getModule()->getDataLayout(), DT, AC),
         "Undefined behavior: Division by zero", &I);
}

void Lint::visitURem(BinaryOperator &I) {
  Assert(!isZero(I.getOperand(1), I.getModule()->getDataLayout(), DT, AC),
         "Undefined behavior: Division by zero", &I);
}

void Lint::visitAllocaInst(AllocaInst &I) {
  if (isa<ConstantInt>(I.getArraySize()))
    // This isn't undefined behavior, it's just an obvious pessimization.
    Assert(&I.getParent()->getParent()->getEntryBlock() == I.getParent(),
           "Pessimization: Static alloca outside of entry block", &I);

  // TODO: Check for an unusual size (MSB set?)
}

void Lint::visitVAArgInst(VAArgInst &I) {
  visitMemoryReference(I, I.getOperand(0), MemoryLocation::UnknownSize, 0,
                       nullptr, MemRef::Read | MemRef::Write);
}

void Lint::visitIndirectBrInst(IndirectBrInst &I) {
  visitMemoryReference(I, I.getAddress(), MemoryLocation::UnknownSize, 0,
                       nullptr, MemRef::Branchee);

  Assert(I.getNumDestinations() != 0,
         "Undefined behavior: indirectbr with no destinations", &I);
}

void Lint::visitExtractElementInst(ExtractElementInst &I) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(
          findValue(I.getIndexOperand(), I.getModule()->getDataLayout(),
                    /*OffsetOk=*/false)))
    Assert(CI->getValue().ult(I.getVectorOperandType()->getNumElements()),
           "Undefined result: extractelement index out of range", &I);
}

void Lint::visitInsertElementInst(InsertElementInst &I) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(
          findValue(I.getOperand(2), I.getModule()->getDataLayout(),
                    /*OffsetOk=*/false)))
    Assert(CI->getValue().ult(I.getType()->getNumElements()),
           "Undefined result: insertelement index out of range", &I);
}

void Lint::visitUnreachableInst(UnreachableInst &I) {
  // This isn't undefined behavior, it's merely suspicious.
  Assert(&I == I.getParent()->begin() ||
             std::prev(BasicBlock::iterator(&I))->mayHaveSideEffects(),
         "Unusual: unreachable immediately preceded by instruction without "
         "side effects",
         &I);
}

/// findValue - Look through bitcasts and simple memory reference patterns
/// to identify an equivalent, but more informative, value.  If OffsetOk
/// is true, look through getelementptrs with non-zero offsets too.
///
/// Most analysis passes don't require this logic, because instcombine
/// will simplify most of these kinds of things away. But it's a goal of
/// this Lint pass to be useful even on non-optimized IR.
Value *Lint::findValue(Value *V, const DataLayout &DL, bool OffsetOk) const {
  SmallPtrSet<Value *, 4> Visited;
  return findValueImpl(V, DL, OffsetOk, Visited);
}

/// findValueImpl - Implementation helper for findValue.
Value *Lint::findValueImpl(Value *V, const DataLayout &DL, bool OffsetOk,
                           SmallPtrSetImpl<Value *> &Visited) const {
  // Detect self-referential values.
  if (!Visited.insert(V).second)
    return UndefValue::get(V->getType());

  // TODO: Look through sext or zext cast, when the result is known to
  // be interpreted as signed or unsigned, respectively.
  // TODO: Look through eliminable cast pairs.
  // TODO: Look through calls with unique return values.
  // TODO: Look through vector insert/extract/shuffle.
  V = OffsetOk ? GetUnderlyingObject(V, DL) : V->stripPointerCasts();
  if (LoadInst *L = dyn_cast<LoadInst>(V)) {
    BasicBlock::iterator BBI = L;
    BasicBlock *BB = L->getParent();
    SmallPtrSet<BasicBlock *, 4> VisitedBlocks;
    for (;;) {
      if (!VisitedBlocks.insert(BB).second)
        break;
      if (Value *U = FindAvailableLoadedValue(L->getPointerOperand(),
                                              BB, BBI, 6, AA))
        return findValueImpl(U, DL, OffsetOk, Visited);
      if (BBI != BB->begin()) break;
      BB = BB->getUniquePredecessor();
      if (!BB) break;
      BBI = BB->end();
    }
  } else if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (Value *W = PN->hasConstantValue())
      if (W != V)
        return findValueImpl(W, DL, OffsetOk, Visited);
  } else if (CastInst *CI = dyn_cast<CastInst>(V)) {
    if (CI->isNoopCast(DL))
      return findValueImpl(CI->getOperand(0), DL, OffsetOk, Visited);
  } else if (ExtractValueInst *Ex = dyn_cast<ExtractValueInst>(V)) {
    if (Value *W = FindInsertedValue(Ex->getAggregateOperand(),
                                     Ex->getIndices()))
      if (W != V)
        return findValueImpl(W, DL, OffsetOk, Visited);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    // Same as above, but for ConstantExpr instead of Instruction.
    if (Instruction::isCast(CE->getOpcode())) {
      if (CastInst::isNoopCast(Instruction::CastOps(CE->getOpcode()),
                               CE->getOperand(0)->getType(), CE->getType(),
                               DL.getIntPtrType(V->getType())))
        return findValueImpl(CE->getOperand(0), DL, OffsetOk, Visited);
    } else if (CE->getOpcode() == Instruction::ExtractValue) {
      ArrayRef<unsigned> Indices = CE->getIndices();
      if (Value *W = FindInsertedValue(CE->getOperand(0), Indices))
        if (W != V)
          return findValueImpl(W, DL, OffsetOk, Visited);
    }
  }

  // As a last resort, try SimplifyInstruction or constant folding.
  if (Instruction *Inst = dyn_cast<Instruction>(V)) {
    if (Value *W = SimplifyInstruction(Inst, DL, TLI, DT, AC))
      return findValueImpl(W, DL, OffsetOk, Visited);
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (Value *W = ConstantFoldConstantExpression(CE, DL, TLI))
      if (W != V)
        return findValueImpl(W, DL, OffsetOk, Visited);
  }

  return V;
}

//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createLintPass() {
  return new Lint();
}

/// lintFunction - Check a function for errors, printing messages on stderr.
///
void llvm::lintFunction(const Function &f) {
  Function &F = const_cast<Function&>(f);
  assert(!F.isDeclaration() && "Cannot lint external functions");

  legacy::FunctionPassManager FPM(F.getParent());
  Lint *V = new Lint();
  FPM.add(V);
  FPM.run(F);
}

/// lintModule - Check a module for errors, printing messages on stderr.
///
void llvm::lintModule(const Module &M) {
  legacy::PassManager PM;
  Lint *V = new Lint();
  PM.add(V);
  PM.run(const_cast<Module&>(M));
}
