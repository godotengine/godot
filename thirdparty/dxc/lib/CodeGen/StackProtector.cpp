//===-- StackProtector.cpp - Stack Protector Insertion --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass inserts stack protectors into functions which need them. A variable
// with a random value in it is stored onto the stack before the local variables
// are allocated. Upon exiting the block, the stored value is checked. If it's
// changed, then there was some sort of violation and the program aborts.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/StackProtector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <cstdlib>
using namespace llvm;

#define DEBUG_TYPE "stack-protector"

STATISTIC(NumFunProtected, "Number of functions protected");
STATISTIC(NumAddrTaken, "Number of local variables that have their address"
                        " taken.");

static cl::opt<bool> EnableSelectionDAGSP("enable-selectiondag-sp",
                                          cl::init(true), cl::Hidden);

char StackProtector::ID = 0;
INITIALIZE_PASS(StackProtector, "stack-protector", "Insert stack protectors",
                false, true)

FunctionPass *llvm::createStackProtectorPass(const TargetMachine *TM) {
  return new StackProtector(TM);
}

StackProtector::SSPLayoutKind
StackProtector::getSSPLayout(const AllocaInst *AI) const {
  return AI ? Layout.lookup(AI) : SSPLK_None;
}

void StackProtector::adjustForColoring(const AllocaInst *From,
                                       const AllocaInst *To) {
  // When coloring replaces one alloca with another, transfer the SSPLayoutKind
  // tag from the remapped to the target alloca. The remapped alloca should
  // have a size smaller than or equal to the replacement alloca.
  SSPLayoutMap::iterator I = Layout.find(From);
  if (I != Layout.end()) {
    SSPLayoutKind Kind = I->second;
    Layout.erase(I);

    // Transfer the tag, but make sure that SSPLK_AddrOf does not overwrite
    // SSPLK_SmallArray or SSPLK_LargeArray, and make sure that
    // SSPLK_SmallArray does not overwrite SSPLK_LargeArray.
    I = Layout.find(To);
    if (I == Layout.end())
      Layout.insert(std::make_pair(To, Kind));
    else if (I->second != SSPLK_LargeArray && Kind != SSPLK_AddrOf)
      I->second = Kind;
  }
}

bool StackProtector::runOnFunction(Function &Fn) {
  F = &Fn;
  M = F->getParent();
  DominatorTreeWrapperPass *DTWP =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTWP ? &DTWP->getDomTree() : nullptr;
  TLI = TM->getSubtargetImpl(Fn)->getTargetLowering();

  Attribute Attr = Fn.getFnAttribute("stack-protector-buffer-size");
  if (Attr.isStringAttribute() &&
      Attr.getValueAsString().getAsInteger(10, SSPBufferSize))
      return false; // Invalid integer string

  if (!RequiresStackProtector())
    return false;

  ++NumFunProtected;
  return InsertStackProtectors();
}

/// \param [out] IsLarge is set to true if a protectable array is found and
/// it is "large" ( >= ssp-buffer-size).  In the case of a structure with
/// multiple arrays, this gets set if any of them is large.
bool StackProtector::ContainsProtectableArray(Type *Ty, bool &IsLarge,
                                              bool Strong,
                                              bool InStruct) const {
  if (!Ty)
    return false;
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    if (!AT->getElementType()->isIntegerTy(8)) {
      // If we're on a non-Darwin platform or we're inside of a structure, don't
      // add stack protectors unless the array is a character array.
      // However, in strong mode any array, regardless of type and size,
      // triggers a protector.
      if (!Strong && (InStruct || !Trip.isOSDarwin()))
        return false;
    }

    // If an array has more than SSPBufferSize bytes of allocated space, then we
    // emit stack protectors.
    if (SSPBufferSize <= M->getDataLayout().getTypeAllocSize(AT)) {
      IsLarge = true;
      return true;
    }

    if (Strong)
      // Require a protector for all arrays in strong mode
      return true;
  }

  const StructType *ST = dyn_cast<StructType>(Ty);
  if (!ST)
    return false;

  bool NeedsProtector = false;
  for (StructType::element_iterator I = ST->element_begin(),
                                    E = ST->element_end();
       I != E; ++I)
    if (ContainsProtectableArray(*I, IsLarge, Strong, true)) {
      // If the element is a protectable array and is large (>= SSPBufferSize)
      // then we are done.  If the protectable array is not large, then
      // keep looking in case a subsequent element is a large array.
      if (IsLarge)
        return true;
      NeedsProtector = true;
    }

  return NeedsProtector;
}

bool StackProtector::HasAddressTaken(const Instruction *AI) {
  for (const User *U : AI->users()) {
    if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (AI == SI->getValueOperand())
        return true;
    } else if (const PtrToIntInst *SI = dyn_cast<PtrToIntInst>(U)) {
      if (AI == SI->getOperand(0))
        return true;
    } else if (isa<CallInst>(U)) {
      return true;
    } else if (isa<InvokeInst>(U)) {
      return true;
    } else if (const SelectInst *SI = dyn_cast<SelectInst>(U)) {
      if (HasAddressTaken(SI))
        return true;
    } else if (const PHINode *PN = dyn_cast<PHINode>(U)) {
      // Keep track of what PHI nodes we have already visited to ensure
      // they are only visited once.
      if (VisitedPHIs.insert(PN).second)
        if (HasAddressTaken(PN))
          return true;
    } else if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (HasAddressTaken(GEP))
        return true;
    } else if (const BitCastInst *BI = dyn_cast<BitCastInst>(U)) {
      if (HasAddressTaken(BI))
        return true;
    }
  }
  return false;
}

/// \brief Check whether or not this function needs a stack protector based
/// upon the stack protector level.
///
/// We use two heuristics: a standard (ssp) and strong (sspstrong).
/// The standard heuristic which will add a guard variable to functions that
/// call alloca with a either a variable size or a size >= SSPBufferSize,
/// functions with character buffers larger than SSPBufferSize, and functions
/// with aggregates containing character buffers larger than SSPBufferSize. The
/// strong heuristic will add a guard variables to functions that call alloca
/// regardless of size, functions with any buffer regardless of type and size,
/// functions with aggregates that contain any buffer regardless of type and
/// size, and functions that contain stack-based variables that have had their
/// address taken.
bool StackProtector::RequiresStackProtector() {
  bool Strong = false;
  bool NeedsProtector = false;
  if (F->hasFnAttribute(Attribute::StackProtectReq)) {
    NeedsProtector = true;
    Strong = true; // Use the same heuristic as strong to determine SSPLayout
  } else if (F->hasFnAttribute(Attribute::StackProtectStrong))
    Strong = true;
  else if (!F->hasFnAttribute(Attribute::StackProtect))
    return false;

  for (const BasicBlock &BB : *F) {
    for (const Instruction &I : BB) {
      if (const AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
        if (AI->isArrayAllocation()) {
          // SSP-Strong: Enable protectors for any call to alloca, regardless
          // of size.
          if (Strong)
            return true;

          if (const auto *CI = dyn_cast<ConstantInt>(AI->getArraySize())) {
            if (CI->getLimitedValue(SSPBufferSize) >= SSPBufferSize) {
              // A call to alloca with size >= SSPBufferSize requires
              // stack protectors.
              Layout.insert(std::make_pair(AI, SSPLK_LargeArray));
              NeedsProtector = true;
            } else if (Strong) {
              // Require protectors for all alloca calls in strong mode.
              Layout.insert(std::make_pair(AI, SSPLK_SmallArray));
              NeedsProtector = true;
            }
          } else {
            // A call to alloca with a variable size requires protectors.
            Layout.insert(std::make_pair(AI, SSPLK_LargeArray));
            NeedsProtector = true;
          }
          continue;
        }

        bool IsLarge = false;
        if (ContainsProtectableArray(AI->getAllocatedType(), IsLarge, Strong)) {
          Layout.insert(std::make_pair(AI, IsLarge ? SSPLK_LargeArray
                                                   : SSPLK_SmallArray));
          NeedsProtector = true;
          continue;
        }

        if (Strong && HasAddressTaken(AI)) {
          ++NumAddrTaken;
          Layout.insert(std::make_pair(AI, SSPLK_AddrOf));
          NeedsProtector = true;
        }
      }
    }
  }

  return NeedsProtector;
}

static bool InstructionWillNotHaveChain(const Instruction *I) {
  return !I->mayHaveSideEffects() && !I->mayReadFromMemory() &&
         isSafeToSpeculativelyExecute(I);
}

/// Identify if RI has a previous instruction in the "Tail Position" and return
/// it. Otherwise return 0.
///
/// This is based off of the code in llvm::isInTailCallPosition. The difference
/// is that it inverts the first part of llvm::isInTailCallPosition since
/// isInTailCallPosition is checking if a call is in a tail call position, and
/// we are searching for an unknown tail call that might be in the tail call
/// position. Once we find the call though, the code uses the same refactored
/// code, returnTypeIsEligibleForTailCall.
static CallInst *FindPotentialTailCall(BasicBlock *BB, ReturnInst *RI,
                                       const TargetLoweringBase *TLI) {
  // Establish a reasonable upper bound on the maximum amount of instructions we
  // will look through to find a tail call.
  unsigned SearchCounter = 0;
  const unsigned MaxSearch = 4;
  bool NoInterposingChain = true;

  for (BasicBlock::reverse_iterator I = std::next(BB->rbegin()), E = BB->rend();
       I != E && SearchCounter < MaxSearch; ++I) {
    Instruction *Inst = &*I;

    // Skip over debug intrinsics and do not allow them to affect our MaxSearch
    // counter.
    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    // If we find a call and the following conditions are satisifed, then we
    // have found a tail call that satisfies at least the target independent
    // requirements of a tail call:
    //
    // 1. The call site has the tail marker.
    //
    // 2. The call site either will not cause the creation of a chain or if a
    // chain is necessary there are no instructions in between the callsite and
    // the call which would create an interposing chain.
    //
    // 3. The return type of the function does not impede tail call
    // optimization.
    if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
      if (CI->isTailCall() &&
          (InstructionWillNotHaveChain(CI) || NoInterposingChain) &&
          returnTypeIsEligibleForTailCall(BB->getParent(), CI, RI, *TLI))
        return CI;
    }

    // If we did not find a call see if we have an instruction that may create
    // an interposing chain.
    NoInterposingChain =
        NoInterposingChain && InstructionWillNotHaveChain(Inst);

    // Increment max search.
    SearchCounter++;
  }

  return nullptr;
}

/// Insert code into the entry block that stores the __stack_chk_guard
/// variable onto the stack:
///
///   entry:
///     StackGuardSlot = alloca i8*
///     StackGuard = load __stack_chk_guard
///     call void @llvm.stackprotect.create(StackGuard, StackGuardSlot)
///
/// Returns true if the platform/triple supports the stackprotectorcreate pseudo
/// node.
static bool CreatePrologue(Function *F, Module *M, ReturnInst *RI,
                           const TargetLoweringBase *TLI, const Triple &TT,
                           AllocaInst *&AI, Value *&StackGuardVar) {
  bool SupportsSelectionDAGSP = false;
  PointerType *PtrTy = Type::getInt8PtrTy(RI->getContext());
  unsigned AddressSpace, Offset;
  if (TLI->getStackCookieLocation(AddressSpace, Offset)) {
    Constant *OffsetVal =
        ConstantInt::get(Type::getInt32Ty(RI->getContext()), Offset);

    StackGuardVar =
        ConstantExpr::getIntToPtr(OffsetVal, PointerType::get(PtrTy,
                                                              AddressSpace));
  } else if (TT.isOSOpenBSD()) {
    StackGuardVar = M->getOrInsertGlobal("__guard_local", PtrTy);
    cast<GlobalValue>(StackGuardVar)
        ->setVisibility(GlobalValue::HiddenVisibility);
  } else {
    SupportsSelectionDAGSP = true;
    StackGuardVar = M->getOrInsertGlobal("__stack_chk_guard", PtrTy);
  }

  IRBuilder<> B(&F->getEntryBlock().front());
  AI = B.CreateAlloca(PtrTy, nullptr, "StackGuardSlot");
  LoadInst *LI = B.CreateLoad(StackGuardVar, "StackGuard");
  B.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::stackprotector),
               {LI, AI});

  return SupportsSelectionDAGSP;
}

/// InsertStackProtectors - Insert code into the prologue and epilogue of the
/// function.
///
///  - The prologue code loads and stores the stack guard onto the stack.
///  - The epilogue checks the value stored in the prologue against the original
///    value. It calls __stack_chk_fail if they differ.
bool StackProtector::InsertStackProtectors() {
  bool HasPrologue = false;
  bool SupportsSelectionDAGSP =
      EnableSelectionDAGSP && !TM->Options.EnableFastISel;
  AllocaInst *AI = nullptr;       // Place on stack that stores the stack guard.
  Value *StackGuardVar = nullptr; // The stack guard variable.

  for (Function::iterator I = F->begin(), E = F->end(); I != E;) {
    BasicBlock *BB = I++;
    ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator());
    if (!RI)
      continue;

    if (!HasPrologue) {
      HasPrologue = true;
      SupportsSelectionDAGSP &=
          CreatePrologue(F, M, RI, TLI, Trip, AI, StackGuardVar);
    }

    if (SupportsSelectionDAGSP) {
      // Since we have a potential tail call, insert the special stack check
      // intrinsic.
      Instruction *InsertionPt = nullptr;
      if (CallInst *CI = FindPotentialTailCall(BB, RI, TLI)) {
        InsertionPt = CI;
      } else {
        InsertionPt = RI;
        // At this point we know that BB has a return statement so it *DOES*
        // have a terminator.
        assert(InsertionPt != nullptr &&
               "BB must have a terminator instruction at this point.");
      }

      Function *Intrinsic =
          Intrinsic::getDeclaration(M, Intrinsic::stackprotectorcheck);
      CallInst::Create(Intrinsic, StackGuardVar, "", InsertionPt);
    } else {
      // If we do not support SelectionDAG based tail calls, generate IR level
      // tail calls.
      //
      // For each block with a return instruction, convert this:
      //
      //   return:
      //     ...
      //     ret ...
      //
      // into this:
      //
      //   return:
      //     ...
      //     %1 = load __stack_chk_guard
      //     %2 = load StackGuardSlot
      //     %3 = cmp i1 %1, %2
      //     br i1 %3, label %SP_return, label %CallStackCheckFailBlk
      //
      //   SP_return:
      //     ret ...
      //
      //   CallStackCheckFailBlk:
      //     call void @__stack_chk_fail()
      //     unreachable

      // Create the FailBB. We duplicate the BB every time since the MI tail
      // merge pass will merge together all of the various BB into one including
      // fail BB generated by the stack protector pseudo instruction.
      BasicBlock *FailBB = CreateFailBB();

      // Split the basic block before the return instruction.
      BasicBlock *NewBB = BB->splitBasicBlock(RI, "SP_return");

      // Update the dominator tree if we need to.
      if (DT && DT->isReachableFromEntry(BB)) {
        DT->addNewBlock(NewBB, BB);
        DT->addNewBlock(FailBB, BB);
      }

      // Remove default branch instruction to the new BB.
      BB->getTerminator()->eraseFromParent();

      // Move the newly created basic block to the point right after the old
      // basic block so that it's in the "fall through" position.
      NewBB->moveAfter(BB);

      // Generate the stack protector instructions in the old basic block.
      IRBuilder<> B(BB);
      LoadInst *LI1 = B.CreateLoad(StackGuardVar);
      LoadInst *LI2 = B.CreateLoad(AI);
      Value *Cmp = B.CreateICmpEQ(LI1, LI2);
      unsigned SuccessWeight =
          BranchProbabilityInfo::getBranchWeightStackProtector(true);
      unsigned FailureWeight =
          BranchProbabilityInfo::getBranchWeightStackProtector(false);
      MDNode *Weights = MDBuilder(F->getContext())
                            .createBranchWeights(SuccessWeight, FailureWeight);
      B.CreateCondBr(Cmp, NewBB, FailBB, Weights);
    }
  }

  // Return if we didn't modify any basic blocks. i.e., there are no return
  // statements in the function.
  if (!HasPrologue)
    return false;

  return true;
}

/// CreateFailBB - Create a basic block to jump to when the stack protector
/// check fails.
BasicBlock *StackProtector::CreateFailBB() {
  LLVMContext &Context = F->getContext();
  BasicBlock *FailBB = BasicBlock::Create(Context, "CallStackCheckFailBlk", F);
  IRBuilder<> B(FailBB);
  if (Trip.isOSOpenBSD()) {
    Constant *StackChkFail =
        M->getOrInsertFunction("__stack_smash_handler",
                               Type::getVoidTy(Context),
                               Type::getInt8PtrTy(Context), nullptr);

    B.CreateCall(StackChkFail, B.CreateGlobalStringPtr(F->getName(), "SSH"));
  } else {
    Constant *StackChkFail =
        M->getOrInsertFunction("__stack_chk_fail", Type::getVoidTy(Context),
                               nullptr);
    B.CreateCall(StackChkFail, {});
  }
  B.CreateUnreachable();
  return FailBB;
}
