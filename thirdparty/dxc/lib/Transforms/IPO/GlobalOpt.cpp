//===- GlobalOpt.cpp - Optimize Global Variables --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms simple global variables that never have their address
// taken.  If obviously true, it marks read/write globals as constant, deletes
// variables only stored to, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "dxc/DXIL/DxilModule.h" // HLSL Change - Entrypoint testing
#include <algorithm>
#include <deque>
using namespace llvm;

#define DEBUG_TYPE "globalopt"

STATISTIC(NumMarked    , "Number of globals marked constant");
STATISTIC(NumUnnamed   , "Number of globals marked unnamed_addr");
STATISTIC(NumSRA       , "Number of aggregate globals broken into scalars");
STATISTIC(NumHeapSRA   , "Number of heap objects SRA'd");
STATISTIC(NumSubstitute,"Number of globals with initializers stored into them");
STATISTIC(NumDeleted   , "Number of globals deleted");
STATISTIC(NumFnDeleted , "Number of functions deleted");
STATISTIC(NumGlobUses  , "Number of global uses devirtualized");
STATISTIC(NumLocalized , "Number of globals localized");
STATISTIC(NumShrunkToBool  , "Number of global vars shrunk to booleans");
STATISTIC(NumFastCallFns   , "Number of functions converted to fastcc");
STATISTIC(NumCtorsEvaluated, "Number of static ctors evaluated");
STATISTIC(NumNestRemoved   , "Number of nest attributes removed");
STATISTIC(NumAliasesResolved, "Number of global aliases resolved");
STATISTIC(NumAliasesRemoved, "Number of global aliases eliminated");
STATISTIC(NumCXXDtorsRemoved, "Number of global C++ destructors removed");

namespace {
  struct GlobalOpt : public ModulePass {
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }
    static char ID; // Pass identification, replacement for typeid
    GlobalOpt() : ModulePass(ID) {
      initializeGlobalOptPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override;

  private:
    bool OptimizeFunctions(Module &M);
    bool OptimizeGlobalVars(Module &M);
    bool OptimizeGlobalAliases(Module &M);
    bool ProcessGlobal(GlobalVariable *GV,Module::global_iterator &GVI);
    bool ProcessInternalGlobal(GlobalVariable *GV,Module::global_iterator &GVI,
                               const GlobalStatus &GS);
    bool OptimizeEmptyGlobalCXXDtors(Function *CXAAtExitFn);

    TargetLibraryInfo *TLI;
    SmallSet<const Comdat *, 8> NotDiscardableComdats;
  };
}

char GlobalOpt::ID = 0;
INITIALIZE_PASS_BEGIN(GlobalOpt, "globalopt",
                "Global Variable Optimizer", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(GlobalOpt, "globalopt",
                "Global Variable Optimizer", false, false)

ModulePass *llvm::createGlobalOptimizerPass() { return new GlobalOpt(); }

/// isLeakCheckerRoot - Is this global variable possibly used by a leak checker
/// as a root?  If so, we might not really want to eliminate the stores to it.
static bool isLeakCheckerRoot(GlobalVariable *GV) {
  // A global variable is a root if it is a pointer, or could plausibly contain
  // a pointer.  There are two challenges; one is that we could have a struct
  // the has an inner member which is a pointer.  We recurse through the type to
  // detect these (up to a point).  The other is that we may actually be a union
  // of a pointer and another type, and so our LLVM type is an integer which
  // gets converted into a pointer, or our type is an [i8 x #] with a pointer
  // potentially contained here.

  if (GV->hasPrivateLinkage())
    return false;

  SmallVector<Type *, 4> Types;
  Types.push_back(cast<PointerType>(GV->getType())->getElementType());

  unsigned Limit = 20;
  do {
    Type *Ty = Types.pop_back_val();
    switch (Ty->getTypeID()) {
      default: break;
      case Type::PointerTyID: return true;
      case Type::ArrayTyID:
      case Type::VectorTyID: {
        SequentialType *STy = cast<SequentialType>(Ty);
        Types.push_back(STy->getElementType());
        break;
      }
      case Type::StructTyID: {
        StructType *STy = cast<StructType>(Ty);
        if (STy->isOpaque()) return true;
        for (StructType::element_iterator I = STy->element_begin(),
                 E = STy->element_end(); I != E; ++I) {
          Type *InnerTy = *I;
          if (isa<PointerType>(InnerTy)) return true;
          if (isa<CompositeType>(InnerTy))
            Types.push_back(InnerTy);
        }
        break;
      }
    }
    if (--Limit == 0) return true;
  } while (!Types.empty());
  return false;
}

/// Given a value that is stored to a global but never read, determine whether
/// it's safe to remove the store and the chain of computation that feeds the
/// store.
static bool IsSafeComputationToRemove(Value *V, const TargetLibraryInfo *TLI) {
  do {
    if (isa<Constant>(V))
      return true;
    if (!V->hasOneUse())
      return false;
    if (isa<LoadInst>(V) || isa<InvokeInst>(V) || isa<Argument>(V) ||
        isa<GlobalValue>(V))
      return false;
    if (isAllocationFn(V, TLI))
      return true;

    Instruction *I = cast<Instruction>(V);
    if (I->mayHaveSideEffects())
      return false;
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (!GEP->hasAllConstantIndices())
        return false;
    } else if (I->getNumOperands() != 1) {
      return false;
    }

    V = I->getOperand(0);
  } while (1);
}

/// CleanupPointerRootUsers - This GV is a pointer root.  Loop over all users
/// of the global and clean up any that obviously don't assign the global a
/// value that isn't dynamically allocated.
///
static bool CleanupPointerRootUsers(GlobalVariable *GV,
                                    const TargetLibraryInfo *TLI) {
  // A brief explanation of leak checkers.  The goal is to find bugs where
  // pointers are forgotten, causing an accumulating growth in memory
  // usage over time.  The common strategy for leak checkers is to whitelist the
  // memory pointed to by globals at exit.  This is popular because it also
  // solves another problem where the main thread of a C++ program may shut down
  // before other threads that are still expecting to use those globals.  To
  // handle that case, we expect the program may create a singleton and never
  // destroy it.

  bool Changed = false;

  // If Dead[n].first is the only use of a malloc result, we can delete its
  // chain of computation and the store to the global in Dead[n].second.
  SmallVector<std::pair<Instruction *, Instruction *>, 32> Dead;

  // Constants can't be pointers to dynamically allocated memory.
  for (Value::user_iterator UI = GV->user_begin(), E = GV->user_end();
       UI != E;) {
    User *U = *UI++;
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      Value *V = SI->getValueOperand();
      if (isa<Constant>(V)) {
        Changed = true;
        SI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(V)) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, SI));
      }
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(U)) {
      if (isa<Constant>(MSI->getValue())) {
        Changed = true;
        MSI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(MSI->getValue())) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, MSI));
      }
    } else if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(U)) {
      GlobalVariable *MemSrc = dyn_cast<GlobalVariable>(MTI->getSource());
      if (MemSrc && MemSrc->isConstant()) {
        Changed = true;
        MTI->eraseFromParent();
      } else if (Instruction *I = dyn_cast<Instruction>(MemSrc)) {
        if (I->hasOneUse())
          Dead.push_back(std::make_pair(I, MTI));
      }
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->use_empty()) {
        CE->destroyConstant();
        Changed = true;
      }
    } else if (Constant *C = dyn_cast<Constant>(U)) {
      if (isSafeToDestroyConstant(C)) {
        C->destroyConstant();
        // This could have invalidated UI, start over from scratch.
        Dead.clear();
        CleanupPointerRootUsers(GV, TLI);
        return true;
      }
    }
  }

  for (int i = 0, e = Dead.size(); i != e; ++i) {
    if (IsSafeComputationToRemove(Dead[i].first, TLI)) {
      Dead[i].second->eraseFromParent();
      Instruction *I = Dead[i].first;
      do {
        if (isAllocationFn(I, TLI))
          break;
        Instruction *J = dyn_cast<Instruction>(I->getOperand(0));
        if (!J)
          break;
        I->eraseFromParent();
        I = J;
      } while (1);
      I->eraseFromParent();
    }
  }

  return Changed;
}

/// CleanupConstantGlobalUsers - We just marked GV constant.  Loop over all
/// users of the global, cleaning up the obvious ones.  This is largely just a
/// quick scan over the use list to clean up the easy and obvious cruft.  This
/// returns true if it made a change.
static bool CleanupConstantGlobalUsers(Value *V, Constant *Init,
                                       const DataLayout &DL,
                                       TargetLibraryInfo *TLI) {
  bool Changed = false;
  // Note that we need to use a weak value handle for the worklist items. When
  // we delete a constant array, we may also be holding pointer to one of its
  // elements (or an element of one of its elements if we're dealing with an
  // array of arrays) in the worklist.
  SmallVector<WeakVH, 8> WorkList(V->user_begin(), V->user_end());
  while (!WorkList.empty()) {
    Value *UV = WorkList.pop_back_val();
    if (!UV)
      continue;

    User *U = cast<User>(UV);

    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (Init) {
        // Replace the load with the initializer.
        LI->replaceAllUsesWith(Init);
        LI->eraseFromParent();
        Changed = true;
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // Store must be unreachable or storing Init into the global.
      SI->eraseFromParent();
      Changed = true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::GetElementPtr) {
        Constant *SubInit = nullptr;
        if (Init)
          SubInit = ConstantFoldLoadThroughGEPConstantExpr(Init, CE);
        Changed |= CleanupConstantGlobalUsers(CE, SubInit, DL, TLI);
      } else if ((CE->getOpcode() == Instruction::BitCast &&
                  CE->getType()->isPointerTy()) ||
                 CE->getOpcode() == Instruction::AddrSpaceCast) {
        // Pointer cast, delete any stores and memsets to the global.
        Changed |= CleanupConstantGlobalUsers(CE, nullptr, DL, TLI);
      }

      if (CE->use_empty()) {
        CE->destroyConstant();
        Changed = true;
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      // Do not transform "gepinst (gep constexpr (GV))" here, because forming
      // "gepconstexpr (gep constexpr (GV))" will cause the two gep's to fold
      // and will invalidate our notion of what Init is.
      Constant *SubInit = nullptr;
      if (!isa<ConstantExpr>(GEP->getOperand(0))) {
        ConstantExpr *CE = dyn_cast_or_null<ConstantExpr>(
            ConstantFoldInstruction(GEP, DL, TLI));
        if (Init && CE && CE->getOpcode() == Instruction::GetElementPtr)
          SubInit = ConstantFoldLoadThroughGEPConstantExpr(Init, CE);

        // If the initializer is an all-null value and we have an inbounds GEP,
        // we already know what the result of any load from that GEP is.
        // TODO: Handle splats.
        if (Init && isa<ConstantAggregateZero>(Init) && GEP->isInBounds())
          SubInit = Constant::getNullValue(GEP->getType()->getElementType());
      }
      Changed |= CleanupConstantGlobalUsers(GEP, SubInit, DL, TLI);

      if (GEP->use_empty()) {
        GEP->eraseFromParent();
        Changed = true;
      }
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(U)) { // memset/cpy/mv
      if (MI->getRawDest() == V) {
        MI->eraseFromParent();
        Changed = true;
      }

    } else if (Constant *C = dyn_cast<Constant>(U)) {
      // If we have a chain of dead constantexprs or other things dangling from
      // us, and if they are all dead, nuke them without remorse.
      if (isSafeToDestroyConstant(C)) {
        C->destroyConstant();
        CleanupConstantGlobalUsers(V, Init, DL, TLI);
        return true;
      }
    }
  }
  return Changed;
}

/// isSafeSROAElementUse - Return true if the specified instruction is a safe
/// user of a derived expression from a global that we want to SROA.
static bool isSafeSROAElementUse(Value *V) {
  // We might have a dead and dangling constant hanging off of here.
  if (Constant *C = dyn_cast<Constant>(V))
    return isSafeToDestroyConstant(C);

  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // Loads are ok.
  if (isa<LoadInst>(I)) return true;

  // Stores *to* the pointer are ok.
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getOperand(0) != V;

  // Otherwise, it must be a GEP.
  GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I);
  if (!GEPI) return false;

  if (GEPI->getNumOperands() < 3 || !isa<Constant>(GEPI->getOperand(1)) ||
      !cast<Constant>(GEPI->getOperand(1))->isNullValue())
    return false;

  for (User *U : GEPI->users())
    if (!isSafeSROAElementUse(U))
      return false;
  return true;
}


/// IsUserOfGlobalSafeForSRA - U is a direct user of the specified global value.
/// Look at it and its uses and decide whether it is safe to SROA this global.
///
static bool IsUserOfGlobalSafeForSRA(User *U, GlobalValue *GV) {
  // The user of the global must be a GEP Inst or a ConstantExpr GEP.
  if (!isa<GetElementPtrInst>(U) &&
      (!isa<ConstantExpr>(U) ||
       cast<ConstantExpr>(U)->getOpcode() != Instruction::GetElementPtr))
    return false;

  // Check to see if this ConstantExpr GEP is SRA'able.  In particular, we
  // don't like < 3 operand CE's, and we don't like non-constant integer
  // indices.  This enforces that all uses are 'gep GV, 0, C, ...' for some
  // value of C.
  if (U->getNumOperands() < 3 || !isa<Constant>(U->getOperand(1)) ||
      !cast<Constant>(U->getOperand(1))->isNullValue() ||
      !isa<ConstantInt>(U->getOperand(2)))
    return false;

  gep_type_iterator GEPI = gep_type_begin(U), E = gep_type_end(U);
  ++GEPI;  // Skip over the pointer index.

  // If this is a use of an array allocation, do a bit more checking for sanity.
  if (ArrayType *AT = dyn_cast<ArrayType>(*GEPI)) {
    uint64_t NumElements = AT->getNumElements();
    ConstantInt *Idx = cast<ConstantInt>(U->getOperand(2));

    // Check to make sure that index falls within the array.  If not,
    // something funny is going on, so we won't do the optimization.
    //
    if (Idx->getZExtValue() >= NumElements)
      return false;

    // We cannot scalar repl this level of the array unless any array
    // sub-indices are in-range constants.  In particular, consider:
    // A[0][i].  We cannot know that the user isn't doing invalid things like
    // allowing i to index an out-of-range subscript that accesses A[1].
    //
    // Scalar replacing *just* the outer index of the array is probably not
    // going to be a win anyway, so just give up.
    for (++GEPI; // Skip array index.
         GEPI != E;
         ++GEPI) {
      uint64_t NumElements;
      if (ArrayType *SubArrayTy = dyn_cast<ArrayType>(*GEPI))
        NumElements = SubArrayTy->getNumElements();
      else if (VectorType *SubVectorTy = dyn_cast<VectorType>(*GEPI))
        NumElements = SubVectorTy->getNumElements();
      else {
        assert((*GEPI)->isStructTy() &&
               "Indexed GEP type is not array, vector, or struct!");
        continue;
      }

      ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPI.getOperand());
      if (!IdxVal || IdxVal->getZExtValue() >= NumElements)
        return false;
    }
  }

  for (User *UU : U->users())
    if (!isSafeSROAElementUse(UU))
      return false;

  return true;
}

/// GlobalUsersSafeToSRA - Look at all uses of the global and decide whether it
/// is safe for us to perform this transformation.
///
static bool GlobalUsersSafeToSRA(GlobalValue *GV) {
  for (User *U : GV->users())
    if (!IsUserOfGlobalSafeForSRA(U, GV))
      return false;

  return true;
}


/// SRAGlobal - Perform scalar replacement of aggregates on the specified global
/// variable.  This opens the door for other optimizations by exposing the
/// behavior of the program in a more fine-grained way.  We have determined that
/// this transformation is safe already.  We return the first global variable we
/// insert so that the caller can reprocess it.
static GlobalVariable *SRAGlobal(GlobalVariable *GV, const DataLayout &DL) {
  // Make sure this global only has simple uses that we can SRA.
  if (!GlobalUsersSafeToSRA(GV))
    return nullptr;

  assert(GV->hasLocalLinkage() && !GV->isConstant());
  Constant *Init = GV->getInitializer();
  Type *Ty = Init->getType();

  std::vector<GlobalVariable*> NewGlobals;
  Module::GlobalListType &Globals = GV->getParent()->getGlobalList();

  // Get the alignment of the global, either explicit or target-specific.
  unsigned StartAlignment = GV->getAlignment();
  if (StartAlignment == 0)
    StartAlignment = DL.getABITypeAlignment(GV->getType());

  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    NewGlobals.reserve(STy->getNumElements());
    const StructLayout &Layout = *DL.getStructLayout(STy);
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      Constant *In = Init->getAggregateElement(i);
      assert(In && "Couldn't get element of initializer?");
      GlobalVariable *NGV = new GlobalVariable(STy->getElementType(i), false,
                                               GlobalVariable::InternalLinkage,
                                               In, GV->getName()+"."+Twine(i),
                                               GV->getThreadLocalMode(),
                                              GV->getType()->getAddressSpace());
      Globals.insert(GV, NGV);
      NewGlobals.push_back(NGV);

      // Calculate the known alignment of the field.  If the original aggregate
      // had 256 byte alignment for example, something might depend on that:
      // propagate info to each field.
      uint64_t FieldOffset = Layout.getElementOffset(i);
      unsigned NewAlign = (unsigned)MinAlign(StartAlignment, FieldOffset);
      if (NewAlign > DL.getABITypeAlignment(STy->getElementType(i)))
        NGV->setAlignment(NewAlign);
    }
  } else if (SequentialType *STy = dyn_cast<SequentialType>(Ty)) {
    unsigned NumElements = 0;
    if (ArrayType *ATy = dyn_cast<ArrayType>(STy))
      NumElements = ATy->getNumElements();
    else
      NumElements = cast<VectorType>(STy)->getNumElements();

    if (NumElements > 16 && GV->hasNUsesOrMore(16))
      return nullptr; // It's not worth it.
    NewGlobals.reserve(NumElements);

    uint64_t EltSize = DL.getTypeAllocSize(STy->getElementType());
    unsigned EltAlign = DL.getABITypeAlignment(STy->getElementType());
    for (unsigned i = 0, e = NumElements; i != e; ++i) {
      Constant *In = Init->getAggregateElement(i);
      assert(In && "Couldn't get element of initializer?");

      GlobalVariable *NGV = new GlobalVariable(STy->getElementType(), false,
                                               GlobalVariable::InternalLinkage,
                                               In, GV->getName()+"."+Twine(i),
                                               GV->getThreadLocalMode(),
                                              GV->getType()->getAddressSpace());
      Globals.insert(GV, NGV);
      NewGlobals.push_back(NGV);

      // Calculate the known alignment of the field.  If the original aggregate
      // had 256 byte alignment for example, something might depend on that:
      // propagate info to each field.
      unsigned NewAlign = (unsigned)MinAlign(StartAlignment, EltSize*i);
      if (NewAlign > EltAlign)
        NGV->setAlignment(NewAlign);
    }
  }

  if (NewGlobals.empty())
    return nullptr;

  DEBUG(dbgs() << "PERFORMING GLOBAL SRA ON: " << *GV);

  Constant *NullInt =Constant::getNullValue(Type::getInt32Ty(GV->getContext()));

  // Loop over all of the uses of the global, replacing the constantexpr geps,
  // with smaller constantexpr geps or direct references.
  while (!GV->use_empty()) {
    User *GEP = GV->user_back();
    assert(((isa<ConstantExpr>(GEP) &&
             cast<ConstantExpr>(GEP)->getOpcode()==Instruction::GetElementPtr)||
            isa<GetElementPtrInst>(GEP)) && "NonGEP CE's are not SRAable!");

    // Ignore the 1th operand, which has to be zero or else the program is quite
    // broken (undefined).  Get the 2nd operand, which is the structure or array
    // index.
    unsigned Val = cast<ConstantInt>(GEP->getOperand(2))->getZExtValue();
    if (Val >= NewGlobals.size()) Val = 0; // Out of bound array access.

    Value *NewPtr = NewGlobals[Val];
    Type *NewTy = NewGlobals[Val]->getValueType();

    // Form a shorter GEP if needed.
    if (GEP->getNumOperands() > 3) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEP)) {
        SmallVector<Constant*, 8> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = CE->getNumOperands(); i != e; ++i)
          Idxs.push_back(CE->getOperand(i));
        NewPtr =
            ConstantExpr::getGetElementPtr(NewTy, cast<Constant>(NewPtr), Idxs);
      } else {
        GetElementPtrInst *GEPI = cast<GetElementPtrInst>(GEP);
        SmallVector<Value*, 8> Idxs;
        Idxs.push_back(NullInt);
        for (unsigned i = 3, e = GEPI->getNumOperands(); i != e; ++i)
          Idxs.push_back(GEPI->getOperand(i));
        NewPtr = GetElementPtrInst::Create(
            NewTy, NewPtr, Idxs, GEPI->getName() + "." + Twine(Val), GEPI);
      }
    }
    GEP->replaceAllUsesWith(NewPtr);

    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(GEP))
      GEPI->eraseFromParent();
    else
      cast<ConstantExpr>(GEP)->destroyConstant();
  }

  // Delete the old global, now that it is dead.
  Globals.erase(GV);
  ++NumSRA;

  // Loop over the new globals array deleting any globals that are obviously
  // dead.  This can arise due to scalarization of a structure or an array that
  // has elements that are dead.
  unsigned FirstGlobal = 0;
  for (unsigned i = 0, e = NewGlobals.size(); i != e; ++i)
    if (NewGlobals[i]->use_empty()) {
      Globals.erase(NewGlobals[i]);
      if (FirstGlobal == i) ++FirstGlobal;
    }

  return FirstGlobal != NewGlobals.size() ? NewGlobals[FirstGlobal] : nullptr;
}

/// AllUsesOfValueWillTrapIfNull - Return true if all users of the specified
/// value will trap if the value is dynamically null.  PHIs keeps track of any
/// phi nodes we've seen to avoid reprocessing them.
static bool AllUsesOfValueWillTrapIfNull(const Value *V,
                                        SmallPtrSetImpl<const PHINode*> &PHIs) {
  for (const User *U : V->users())
    if (isa<LoadInst>(U)) {
      // Will trap.
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getOperand(0) == V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Storing the value.
      }
    } else if (const CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledValue() != V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Not calling the ptr
      }
    } else if (const InvokeInst *II = dyn_cast<InvokeInst>(U)) {
      if (II->getCalledValue() != V) {
        //cerr << "NONTRAPPING USE: " << *U;
        return false;  // Not calling the ptr
      }
    } else if (const BitCastInst *CI = dyn_cast<BitCastInst>(U)) {
      if (!AllUsesOfValueWillTrapIfNull(CI, PHIs)) return false;
    } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U)) {
      if (!AllUsesOfValueWillTrapIfNull(GEPI, PHIs)) return false;
    } else if (const PHINode *PN = dyn_cast<PHINode>(U)) {
      // If we've already seen this phi node, ignore it, it has already been
      // checked.
      if (PHIs.insert(PN).second && !AllUsesOfValueWillTrapIfNull(PN, PHIs))
        return false;
    } else if (isa<ICmpInst>(U) &&
               isa<ConstantPointerNull>(U->getOperand(1))) {
      // Ignore icmp X, null
    } else {
      //cerr << "NONTRAPPING USE: " << *U;
      return false;
    }

  return true;
}

/// AllUsesOfLoadedValueWillTrapIfNull - Return true if all uses of any loads
/// from GV will trap if the loaded value is null.  Note that this also permits
/// comparisons of the loaded value against null, as a special case.
static bool AllUsesOfLoadedValueWillTrapIfNull(const GlobalVariable *GV) {
  for (const User *U : GV->users())
    if (const LoadInst *LI = dyn_cast<LoadInst>(U)) {
      SmallPtrSet<const PHINode*, 8> PHIs;
      if (!AllUsesOfValueWillTrapIfNull(LI, PHIs))
        return false;
    } else if (isa<StoreInst>(U)) {
      // Ignore stores to the global.
    } else {
      // We don't know or understand this user, bail out.
      //cerr << "UNKNOWN USER OF GLOBAL!: " << *U;
      return false;
    }
  return true;
}

static bool OptimizeAwayTrappingUsesOfValue(Value *V, Constant *NewV) {
  bool Changed = false;
  for (auto UI = V->user_begin(), E = V->user_end(); UI != E; ) {
    Instruction *I = cast<Instruction>(*UI++);
    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      LI->setOperand(0, NewV);
      Changed = true;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (SI->getOperand(1) == V) {
        SI->setOperand(1, NewV);
        Changed = true;
      }
    } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
      CallSite CS(I);
      if (CS.getCalledValue() == V) {
        // Calling through the pointer!  Turn into a direct call, but be careful
        // that the pointer is not also being passed as an argument.
        CS.setCalledFunction(NewV);
        Changed = true;
        bool PassedAsArg = false;
        for (unsigned i = 0, e = CS.arg_size(); i != e; ++i)
          if (CS.getArgument(i) == V) {
            PassedAsArg = true;
            CS.setArgument(i, NewV);
          }

        if (PassedAsArg) {
          // Being passed as an argument also.  Be careful to not invalidate UI!
          UI = V->user_begin();
        }
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      Changed |= OptimizeAwayTrappingUsesOfValue(CI,
                                ConstantExpr::getCast(CI->getOpcode(),
                                                      NewV, CI->getType()));
      if (CI->use_empty()) {
        Changed = true;
        CI->eraseFromParent();
      }
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      // Should handle GEP here.
      SmallVector<Constant*, 8> Idxs;
      Idxs.reserve(GEPI->getNumOperands()-1);
      for (User::op_iterator i = GEPI->op_begin() + 1, e = GEPI->op_end();
           i != e; ++i)
        if (Constant *C = dyn_cast<Constant>(*i))
          Idxs.push_back(C);
        else
          break;
      if (Idxs.size() == GEPI->getNumOperands()-1)
        Changed |= OptimizeAwayTrappingUsesOfValue(
            GEPI, ConstantExpr::getGetElementPtr(nullptr, NewV, Idxs));
      if (GEPI->use_empty()) {
        Changed = true;
        GEPI->eraseFromParent();
      }
    }
  }

  return Changed;
}


/// OptimizeAwayTrappingUsesOfLoads - The specified global has only one non-null
/// value stored into it.  If there are uses of the loaded value that would trap
/// if the loaded value is dynamically null, then we know that they cannot be
/// reachable with a null optimize away the load.
static bool OptimizeAwayTrappingUsesOfLoads(GlobalVariable *GV, Constant *LV,
                                            const DataLayout &DL,
                                            TargetLibraryInfo *TLI) {
  bool Changed = false;

  // Keep track of whether we are able to remove all the uses of the global
  // other than the store that defines it.
  bool AllNonStoreUsesGone = true;

  // Replace all uses of loads with uses of uses of the stored value.
  for (Value::user_iterator GUI = GV->user_begin(), E = GV->user_end(); GUI != E;){
    User *GlobalUser = *GUI++;
    if (LoadInst *LI = dyn_cast<LoadInst>(GlobalUser)) {
      Changed |= OptimizeAwayTrappingUsesOfValue(LI, LV);
      // If we were able to delete all uses of the loads
      if (LI->use_empty()) {
        LI->eraseFromParent();
        Changed = true;
      } else {
        AllNonStoreUsesGone = false;
      }
    } else if (isa<StoreInst>(GlobalUser)) {
      // Ignore the store that stores "LV" to the global.
      assert(GlobalUser->getOperand(1) == GV &&
             "Must be storing *to* the global");
    } else {
      AllNonStoreUsesGone = false;

      // If we get here we could have other crazy uses that are transitively
      // loaded.
      assert((isa<PHINode>(GlobalUser) || isa<SelectInst>(GlobalUser) ||
              isa<ConstantExpr>(GlobalUser) || isa<CmpInst>(GlobalUser) ||
              isa<BitCastInst>(GlobalUser) ||
              isa<GetElementPtrInst>(GlobalUser)) &&
             "Only expect load and stores!");
    }
  }

  if (Changed) {
    DEBUG(dbgs() << "OPTIMIZED LOADS FROM STORED ONCE POINTER: " << *GV);
    ++NumGlobUses;
  }

  // If we nuked all of the loads, then none of the stores are needed either,
  // nor is the global.
  if (AllNonStoreUsesGone) {
    if (isLeakCheckerRoot(GV)) {
      Changed |= CleanupPointerRootUsers(GV, TLI);
    } else {
      Changed = true;
      CleanupConstantGlobalUsers(GV, nullptr, DL, TLI);
    }
    if (GV->use_empty()) {
      DEBUG(dbgs() << "  *** GLOBAL NOW DEAD!\n");
      Changed = true;
      GV->eraseFromParent();
      ++NumDeleted;
    }
  }
  return Changed;
}

/// ConstantPropUsersOf - Walk the use list of V, constant folding all of the
/// instructions that are foldable.
static void ConstantPropUsersOf(Value *V, const DataLayout &DL,
                                TargetLibraryInfo *TLI) {
  for (Value::user_iterator UI = V->user_begin(), E = V->user_end(); UI != E; )
    if (Instruction *I = dyn_cast<Instruction>(*UI++))
      if (Constant *NewC = ConstantFoldInstruction(I, DL, TLI)) {
        I->replaceAllUsesWith(NewC);

        // Advance UI to the next non-I use to avoid invalidating it!
        // Instructions could multiply use V.
        while (UI != E && *UI == I)
          ++UI;
        I->eraseFromParent();
      }
}

/// OptimizeGlobalAddressOfMalloc - This function takes the specified global
/// variable, and transforms the program as if it always contained the result of
/// the specified malloc.  Because it is always the result of the specified
/// malloc, there is no reason to actually DO the malloc.  Instead, turn the
/// malloc into a global, and any loads of GV as uses of the new global.
static GlobalVariable *
OptimizeGlobalAddressOfMalloc(GlobalVariable *GV, CallInst *CI, Type *AllocTy,
                              ConstantInt *NElements, const DataLayout &DL,
                              TargetLibraryInfo *TLI) {
  DEBUG(errs() << "PROMOTING GLOBAL: " << *GV << "  CALL = " << *CI << '\n');

  Type *GlobalType;
  if (NElements->getZExtValue() == 1)
    GlobalType = AllocTy;
  else
    // If we have an array allocation, the global variable is of an array.
    GlobalType = ArrayType::get(AllocTy, NElements->getZExtValue());

  // Create the new global variable.  The contents of the malloc'd memory is
  // undefined, so initialize with an undef value.
  GlobalVariable *NewGV = new GlobalVariable(*GV->getParent(),
                                             GlobalType, false,
                                             GlobalValue::InternalLinkage,
                                             UndefValue::get(GlobalType),
                                             GV->getName()+".body",
                                             GV,
                                             GV->getThreadLocalMode());

  // If there are bitcast users of the malloc (which is typical, usually we have
  // a malloc + bitcast) then replace them with uses of the new global.  Update
  // other users to use the global as well.
  BitCastInst *TheBC = nullptr;
  while (!CI->use_empty()) {
    Instruction *User = cast<Instruction>(CI->user_back());
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(User)) {
      if (BCI->getType() == NewGV->getType()) {
        BCI->replaceAllUsesWith(NewGV);
        BCI->eraseFromParent();
      } else {
        BCI->setOperand(0, NewGV);
      }
    } else {
      if (!TheBC)
        TheBC = new BitCastInst(NewGV, CI->getType(), "newgv", CI);
      User->replaceUsesOfWith(CI, TheBC);
    }
  }

  Constant *RepValue = NewGV;
  if (NewGV->getType() != GV->getType()->getElementType())
    RepValue = ConstantExpr::getBitCast(RepValue,
                                        GV->getType()->getElementType());

  // If there is a comparison against null, we will insert a global bool to
  // keep track of whether the global was initialized yet or not.
  GlobalVariable *InitBool =
    new GlobalVariable(Type::getInt1Ty(GV->getContext()), false,
                       GlobalValue::InternalLinkage,
                       ConstantInt::getFalse(GV->getContext()),
                       GV->getName()+".init", GV->getThreadLocalMode());
  bool InitBoolUsed = false;

  // Loop over all uses of GV, processing them in turn.
  while (!GV->use_empty()) {
    if (StoreInst *SI = dyn_cast<StoreInst>(GV->user_back())) {
      // The global is initialized when the store to it occurs.
      new StoreInst(ConstantInt::getTrue(GV->getContext()), InitBool, false, 0,
                    SI->getOrdering(), SI->getSynchScope(), SI);
      SI->eraseFromParent();
      continue;
    }

    LoadInst *LI = cast<LoadInst>(GV->user_back());
    while (!LI->use_empty()) {
      Use &LoadUse = *LI->use_begin();
      ICmpInst *ICI = dyn_cast<ICmpInst>(LoadUse.getUser());
      if (!ICI) {
        LoadUse = RepValue;
        continue;
      }

      // Replace the cmp X, 0 with a use of the bool value.
      // Sink the load to where the compare was, if atomic rules allow us to.
      Value *LV = new LoadInst(InitBool, InitBool->getName()+".val", false, 0,
                               LI->getOrdering(), LI->getSynchScope(),
                               LI->isUnordered() ? (Instruction*)ICI : LI);
      InitBoolUsed = true;
      switch (ICI->getPredicate()) {
      default: llvm_unreachable("Unknown ICmp Predicate!");
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT:   // X < null -> always false
        LV = ConstantInt::getFalse(GV->getContext());
        break;
      case ICmpInst::ICMP_ULE:
      case ICmpInst::ICMP_SLE:
      case ICmpInst::ICMP_EQ:
        LV = BinaryOperator::CreateNot(LV, "notinit", ICI);
        break;
      case ICmpInst::ICMP_NE:
      case ICmpInst::ICMP_UGE:
      case ICmpInst::ICMP_SGE:
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT:
        break;  // no change.
      }
      ICI->replaceAllUsesWith(LV);
      ICI->eraseFromParent();
    }
    LI->eraseFromParent();
  }

  // If the initialization boolean was used, insert it, otherwise delete it.
  if (!InitBoolUsed) {
    while (!InitBool->use_empty())  // Delete initializations
      cast<StoreInst>(InitBool->user_back())->eraseFromParent();
    delete InitBool;
  } else
    GV->getParent()->getGlobalList().insert(GV, InitBool);

  // Now the GV is dead, nuke it and the malloc..
  GV->eraseFromParent();
  CI->eraseFromParent();

  // To further other optimizations, loop over all users of NewGV and try to
  // constant prop them.  This will promote GEP instructions with constant
  // indices into GEP constant-exprs, which will allow global-opt to hack on it.
  ConstantPropUsersOf(NewGV, DL, TLI);
  if (RepValue != NewGV)
    ConstantPropUsersOf(RepValue, DL, TLI);

  return NewGV;
}

/// ValueIsOnlyUsedLocallyOrStoredToOneGlobal - Scan the use-list of V checking
/// to make sure that there are no complex uses of V.  We permit simple things
/// like dereferencing the pointer, but not storing through the address, unless
/// it is to the specified global.
static bool ValueIsOnlyUsedLocallyOrStoredToOneGlobal(const Instruction *V,
                                                      const GlobalVariable *GV,
                                        SmallPtrSetImpl<const PHINode*> &PHIs) {
  for (const User *U : V->users()) {
    const Instruction *Inst = cast<Instruction>(U);

    if (isa<LoadInst>(Inst) || isa<CmpInst>(Inst)) {
      continue; // Fine, ignore.
    }

    if (const StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      if (SI->getOperand(0) == V && SI->getOperand(1) != GV)
        return false;  // Storing the pointer itself... bad.
      continue; // Otherwise, storing through it, or storing into GV... fine.
    }

    // Must index into the array and into the struct.
    if (isa<GetElementPtrInst>(Inst) && Inst->getNumOperands() >= 3) {
      if (!ValueIsOnlyUsedLocallyOrStoredToOneGlobal(Inst, GV, PHIs))
        return false;
      continue;
    }

    if (const PHINode *PN = dyn_cast<PHINode>(Inst)) {
      // PHIs are ok if all uses are ok.  Don't infinitely recurse through PHI
      // cycles.
      if (PHIs.insert(PN).second)
        if (!ValueIsOnlyUsedLocallyOrStoredToOneGlobal(PN, GV, PHIs))
          return false;
      continue;
    }

    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(Inst)) {
      if (!ValueIsOnlyUsedLocallyOrStoredToOneGlobal(BCI, GV, PHIs))
        return false;
      continue;
    }

    return false;
  }
  return true;
}

/// ReplaceUsesOfMallocWithGlobal - The Alloc pointer is stored into GV
/// somewhere.  Transform all uses of the allocation into loads from the
/// global and uses of the resultant pointer.  Further, delete the store into
/// GV.  This assumes that these value pass the
/// 'ValueIsOnlyUsedLocallyOrStoredToOneGlobal' predicate.
static void ReplaceUsesOfMallocWithGlobal(Instruction *Alloc,
                                          GlobalVariable *GV) {
  while (!Alloc->use_empty()) {
    Instruction *U = cast<Instruction>(*Alloc->user_begin());
    Instruction *InsertPt = U;
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      // If this is the store of the allocation into the global, remove it.
      if (SI->getOperand(1) == GV) {
        SI->eraseFromParent();
        continue;
      }
    } else if (PHINode *PN = dyn_cast<PHINode>(U)) {
      // Insert the load in the corresponding predecessor, not right before the
      // PHI.
      InsertPt = PN->getIncomingBlock(*Alloc->use_begin())->getTerminator();
    } else if (isa<BitCastInst>(U)) {
      // Must be bitcast between the malloc and store to initialize the global.
      ReplaceUsesOfMallocWithGlobal(U, GV);
      U->eraseFromParent();
      continue;
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U)) {
      // If this is a "GEP bitcast" and the user is a store to the global, then
      // just process it as a bitcast.
      if (GEPI->hasAllZeroIndices() && GEPI->hasOneUse())
        if (StoreInst *SI = dyn_cast<StoreInst>(GEPI->user_back()))
          if (SI->getOperand(1) == GV) {
            // Must be bitcast GEP between the malloc and store to initialize
            // the global.
            ReplaceUsesOfMallocWithGlobal(GEPI, GV);
            GEPI->eraseFromParent();
            continue;
          }
    }

    // Insert a load from the global, and use it instead of the malloc.
    Value *NL = new LoadInst(GV, GV->getName()+".val", InsertPt);
    U->replaceUsesOfWith(Alloc, NL);
  }
}

/// LoadUsesSimpleEnoughForHeapSRA - Verify that all uses of V (a load, or a phi
/// of a load) are simple enough to perform heap SRA on.  This permits GEP's
/// that index through the array and struct field, icmps of null, and PHIs.
static bool LoadUsesSimpleEnoughForHeapSRA(const Value *V,
                        SmallPtrSetImpl<const PHINode*> &LoadUsingPHIs,
                        SmallPtrSetImpl<const PHINode*> &LoadUsingPHIsPerLoad) {
  // We permit two users of the load: setcc comparing against the null
  // pointer, and a getelementptr of a specific form.
  for (const User *U : V->users()) {
    const Instruction *UI = cast<Instruction>(U);

    // Comparison against null is ok.
    if (const ICmpInst *ICI = dyn_cast<ICmpInst>(UI)) {
      if (!isa<ConstantPointerNull>(ICI->getOperand(1)))
        return false;
      continue;
    }

    // getelementptr is also ok, but only a simple form.
    if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(UI)) {
      // Must index into the array and into the struct.
      if (GEPI->getNumOperands() < 3)
        return false;

      // Otherwise the GEP is ok.
      continue;
    }

    if (const PHINode *PN = dyn_cast<PHINode>(UI)) {
      if (!LoadUsingPHIsPerLoad.insert(PN).second)
        // This means some phi nodes are dependent on each other.
        // Avoid infinite looping!
        return false;
      if (!LoadUsingPHIs.insert(PN).second)
        // If we have already analyzed this PHI, then it is safe.
        continue;

      // Make sure all uses of the PHI are simple enough to transform.
      if (!LoadUsesSimpleEnoughForHeapSRA(PN,
                                          LoadUsingPHIs, LoadUsingPHIsPerLoad))
        return false;

      continue;
    }

    // Otherwise we don't know what this is, not ok.
    return false;
  }

  return true;
}


/// AllGlobalLoadUsesSimpleEnoughForHeapSRA - If all users of values loaded from
/// GV are simple enough to perform HeapSRA, return true.
static bool AllGlobalLoadUsesSimpleEnoughForHeapSRA(const GlobalVariable *GV,
                                                    Instruction *StoredVal) {
  SmallPtrSet<const PHINode*, 32> LoadUsingPHIs;
  SmallPtrSet<const PHINode*, 32> LoadUsingPHIsPerLoad;
  for (const User *U : GV->users())
    if (const LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (!LoadUsesSimpleEnoughForHeapSRA(LI, LoadUsingPHIs,
                                          LoadUsingPHIsPerLoad))
        return false;
      LoadUsingPHIsPerLoad.clear();
    }

  // If we reach here, we know that all uses of the loads and transitive uses
  // (through PHI nodes) are simple enough to transform.  However, we don't know
  // that all inputs the to the PHI nodes are in the same equivalence sets.
  // Check to verify that all operands of the PHIs are either PHIS that can be
  // transformed, loads from GV, or MI itself.
  for (const PHINode *PN : LoadUsingPHIs) {
    for (unsigned op = 0, e = PN->getNumIncomingValues(); op != e; ++op) {
      Value *InVal = PN->getIncomingValue(op);

      // PHI of the stored value itself is ok.
      if (InVal == StoredVal) continue;

      if (const PHINode *InPN = dyn_cast<PHINode>(InVal)) {
        // One of the PHIs in our set is (optimistically) ok.
        if (LoadUsingPHIs.count(InPN))
          continue;
        return false;
      }

      // Load from GV is ok.
      if (const LoadInst *LI = dyn_cast<LoadInst>(InVal))
        if (LI->getOperand(0) == GV)
          continue;

      // UNDEF? NULL?

      // Anything else is rejected.
      return false;
    }
  }

  return true;
}

static Value *GetHeapSROAValue(Value *V, unsigned FieldNo,
               DenseMap<Value*, std::vector<Value*> > &InsertedScalarizedValues,
                   std::vector<std::pair<PHINode*, unsigned> > &PHIsToRewrite) {
  std::vector<Value*> &FieldVals = InsertedScalarizedValues[V];

  if (FieldNo >= FieldVals.size())
    FieldVals.resize(FieldNo+1);

  // If we already have this value, just reuse the previously scalarized
  // version.
  if (Value *FieldVal = FieldVals[FieldNo])
    return FieldVal;

  // Depending on what instruction this is, we have several cases.
  Value *Result;
  if (LoadInst *LI = dyn_cast<LoadInst>(V)) {
    // This is a scalarized version of the load from the global.  Just create
    // a new Load of the scalarized global.
    Result = new LoadInst(GetHeapSROAValue(LI->getOperand(0), FieldNo,
                                           InsertedScalarizedValues,
                                           PHIsToRewrite),
                          LI->getName()+".f"+Twine(FieldNo), LI);
  } else {
    PHINode *PN = cast<PHINode>(V);
    // PN's type is pointer to struct.  Make a new PHI of pointer to struct
    // field.

    PointerType *PTy = cast<PointerType>(PN->getType());
    StructType *ST = cast<StructType>(PTy->getElementType());

    unsigned AS = PTy->getAddressSpace();
    PHINode *NewPN =
      PHINode::Create(PointerType::get(ST->getElementType(FieldNo), AS),
                     PN->getNumIncomingValues(),
                     PN->getName()+".f"+Twine(FieldNo), PN);
    Result = NewPN;
    PHIsToRewrite.push_back(std::make_pair(PN, FieldNo));
  }

  return FieldVals[FieldNo] = Result;
}

/// RewriteHeapSROALoadUser - Given a load instruction and a value derived from
/// the load, rewrite the derived value to use the HeapSRoA'd load.
static void RewriteHeapSROALoadUser(Instruction *LoadUser,
             DenseMap<Value*, std::vector<Value*> > &InsertedScalarizedValues,
                   std::vector<std::pair<PHINode*, unsigned> > &PHIsToRewrite) {
  // If this is a comparison against null, handle it.
  if (ICmpInst *SCI = dyn_cast<ICmpInst>(LoadUser)) {
    assert(isa<ConstantPointerNull>(SCI->getOperand(1)));
    // If we have a setcc of the loaded pointer, we can use a setcc of any
    // field.
    Value *NPtr = GetHeapSROAValue(SCI->getOperand(0), 0,
                                   InsertedScalarizedValues, PHIsToRewrite);

    Value *New = new ICmpInst(SCI, SCI->getPredicate(), NPtr,
                              Constant::getNullValue(NPtr->getType()),
                              SCI->getName());
    SCI->replaceAllUsesWith(New);
    SCI->eraseFromParent();
    return;
  }

  // Handle 'getelementptr Ptr, Idx, i32 FieldNo ...'
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(LoadUser)) {
    assert(GEPI->getNumOperands() >= 3 && isa<ConstantInt>(GEPI->getOperand(2))
           && "Unexpected GEPI!");

    // Load the pointer for this field.
    unsigned FieldNo = cast<ConstantInt>(GEPI->getOperand(2))->getZExtValue();
    Value *NewPtr = GetHeapSROAValue(GEPI->getOperand(0), FieldNo,
                                     InsertedScalarizedValues, PHIsToRewrite);

    // Create the new GEP idx vector.
    SmallVector<Value*, 8> GEPIdx;
    GEPIdx.push_back(GEPI->getOperand(1));
    GEPIdx.append(GEPI->op_begin()+3, GEPI->op_end());

    Value *NGEPI = GetElementPtrInst::Create(GEPI->getResultElementType(), NewPtr, GEPIdx,
                                             GEPI->getName(), GEPI);
    GEPI->replaceAllUsesWith(NGEPI);
    GEPI->eraseFromParent();
    return;
  }

  // Recursively transform the users of PHI nodes.  This will lazily create the
  // PHIs that are needed for individual elements.  Keep track of what PHIs we
  // see in InsertedScalarizedValues so that we don't get infinite loops (very
  // antisocial).  If the PHI is already in InsertedScalarizedValues, it has
  // already been seen first by another load, so its uses have already been
  // processed.
  PHINode *PN = cast<PHINode>(LoadUser);
  if (!InsertedScalarizedValues.insert(std::make_pair(PN,
                                              std::vector<Value*>())).second)
    return;

  // If this is the first time we've seen this PHI, recursively process all
  // users.
  for (auto UI = PN->user_begin(), E = PN->user_end(); UI != E;) {
    Instruction *User = cast<Instruction>(*UI++);
    RewriteHeapSROALoadUser(User, InsertedScalarizedValues, PHIsToRewrite);
  }
}

/// RewriteUsesOfLoadForHeapSRoA - We are performing Heap SRoA on a global.  Ptr
/// is a value loaded from the global.  Eliminate all uses of Ptr, making them
/// use FieldGlobals instead.  All uses of loaded values satisfy
/// AllGlobalLoadUsesSimpleEnoughForHeapSRA.
static void RewriteUsesOfLoadForHeapSRoA(LoadInst *Load,
               DenseMap<Value*, std::vector<Value*> > &InsertedScalarizedValues,
                   std::vector<std::pair<PHINode*, unsigned> > &PHIsToRewrite) {
  for (auto UI = Load->user_begin(), E = Load->user_end(); UI != E;) {
    Instruction *User = cast<Instruction>(*UI++);
    RewriteHeapSROALoadUser(User, InsertedScalarizedValues, PHIsToRewrite);
  }

  if (Load->use_empty()) {
    Load->eraseFromParent();
    InsertedScalarizedValues.erase(Load);
  }
}

/// PerformHeapAllocSRoA - CI is an allocation of an array of structures.  Break
/// it up into multiple allocations of arrays of the fields.
static GlobalVariable *PerformHeapAllocSRoA(GlobalVariable *GV, CallInst *CI,
                                            Value *NElems, const DataLayout &DL,
                                            const TargetLibraryInfo *TLI) {
  DEBUG(dbgs() << "SROA HEAP ALLOC: " << *GV << "  MALLOC = " << *CI << '\n');
  Type *MAT = getMallocAllocatedType(CI, TLI);
  StructType *STy = cast<StructType>(MAT);

  // There is guaranteed to be at least one use of the malloc (storing
  // it into GV).  If there are other uses, change them to be uses of
  // the global to simplify later code.  This also deletes the store
  // into GV.
  ReplaceUsesOfMallocWithGlobal(CI, GV);

  // Okay, at this point, there are no users of the malloc.  Insert N
  // new mallocs at the same place as CI, and N globals.
  std::vector<Value*> FieldGlobals;
  std::vector<Value*> FieldMallocs;

  unsigned AS = GV->getType()->getPointerAddressSpace();
  for (unsigned FieldNo = 0, e = STy->getNumElements(); FieldNo != e;++FieldNo){
    Type *FieldTy = STy->getElementType(FieldNo);
    PointerType *PFieldTy = PointerType::get(FieldTy, AS);

    GlobalVariable *NGV =
      new GlobalVariable(*GV->getParent(),
                         PFieldTy, false, GlobalValue::InternalLinkage,
                         Constant::getNullValue(PFieldTy),
                         GV->getName() + ".f" + Twine(FieldNo), GV,
                         GV->getThreadLocalMode());
    FieldGlobals.push_back(NGV);

    unsigned TypeSize = DL.getTypeAllocSize(FieldTy);
    if (StructType *ST = dyn_cast<StructType>(FieldTy))
      TypeSize = DL.getStructLayout(ST)->getSizeInBytes();
    Type *IntPtrTy = DL.getIntPtrType(CI->getType());
    Value *NMI = CallInst::CreateMalloc(CI, IntPtrTy, FieldTy,
                                        ConstantInt::get(IntPtrTy, TypeSize),
                                        NElems, nullptr,
                                        CI->getName() + ".f" + Twine(FieldNo));
    FieldMallocs.push_back(NMI);
    new StoreInst(NMI, NGV, CI);
  }

  // The tricky aspect of this transformation is handling the case when malloc
  // fails.  In the original code, malloc failing would set the result pointer
  // of malloc to null.  In this case, some mallocs could succeed and others
  // could fail.  As such, we emit code that looks like this:
  //    F0 = malloc(field0)
  //    F1 = malloc(field1)
  //    F2 = malloc(field2)
  //    if (F0 == 0 || F1 == 0 || F2 == 0) {
  //      if (F0) { free(F0); F0 = 0; }
  //      if (F1) { free(F1); F1 = 0; }
  //      if (F2) { free(F2); F2 = 0; }
  //    }
  // The malloc can also fail if its argument is too large.
  Constant *ConstantZero = ConstantInt::get(CI->getArgOperand(0)->getType(), 0);
  Value *RunningOr = new ICmpInst(CI, ICmpInst::ICMP_SLT, CI->getArgOperand(0),
                                  ConstantZero, "isneg");
  for (unsigned i = 0, e = FieldMallocs.size(); i != e; ++i) {
    Value *Cond = new ICmpInst(CI, ICmpInst::ICMP_EQ, FieldMallocs[i],
                             Constant::getNullValue(FieldMallocs[i]->getType()),
                               "isnull");
    RunningOr = BinaryOperator::CreateOr(RunningOr, Cond, "tmp", CI);
  }

  // Split the basic block at the old malloc.
  BasicBlock *OrigBB = CI->getParent();
  BasicBlock *ContBB = OrigBB->splitBasicBlock(CI, "malloc_cont");

  // Create the block to check the first condition.  Put all these blocks at the
  // end of the function as they are unlikely to be executed.
  BasicBlock *NullPtrBlock = BasicBlock::Create(OrigBB->getContext(),
                                                "malloc_ret_null",
                                                OrigBB->getParent());

  // Remove the uncond branch from OrigBB to ContBB, turning it into a cond
  // branch on RunningOr.
  OrigBB->getTerminator()->eraseFromParent();
  BranchInst::Create(NullPtrBlock, ContBB, RunningOr, OrigBB);

  // Within the NullPtrBlock, we need to emit a comparison and branch for each
  // pointer, because some may be null while others are not.
  for (unsigned i = 0, e = FieldGlobals.size(); i != e; ++i) {
    Value *GVVal = new LoadInst(FieldGlobals[i], "tmp", NullPtrBlock);
    Value *Cmp = new ICmpInst(*NullPtrBlock, ICmpInst::ICMP_NE, GVVal,
                              Constant::getNullValue(GVVal->getType()));
    BasicBlock *FreeBlock = BasicBlock::Create(Cmp->getContext(), "free_it",
                                               OrigBB->getParent());
    BasicBlock *NextBlock = BasicBlock::Create(Cmp->getContext(), "next",
                                               OrigBB->getParent());
    Instruction *BI = BranchInst::Create(FreeBlock, NextBlock,
                                         Cmp, NullPtrBlock);

    // Fill in FreeBlock.
    CallInst::CreateFree(GVVal, BI);
    new StoreInst(Constant::getNullValue(GVVal->getType()), FieldGlobals[i],
                  FreeBlock);
    BranchInst::Create(NextBlock, FreeBlock);

    NullPtrBlock = NextBlock;
  }

  BranchInst::Create(ContBB, NullPtrBlock);

  // CI is no longer needed, remove it.
  CI->eraseFromParent();

  /// InsertedScalarizedLoads - As we process loads, if we can't immediately
  /// update all uses of the load, keep track of what scalarized loads are
  /// inserted for a given load.
  DenseMap<Value*, std::vector<Value*> > InsertedScalarizedValues;
  InsertedScalarizedValues[GV] = FieldGlobals;

  std::vector<std::pair<PHINode*, unsigned> > PHIsToRewrite;

  // Okay, the malloc site is completely handled.  All of the uses of GV are now
  // loads, and all uses of those loads are simple.  Rewrite them to use loads
  // of the per-field globals instead.
  for (auto UI = GV->user_begin(), E = GV->user_end(); UI != E;) {
    Instruction *User = cast<Instruction>(*UI++);

    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      RewriteUsesOfLoadForHeapSRoA(LI, InsertedScalarizedValues, PHIsToRewrite);
      continue;
    }

    // Must be a store of null.
    StoreInst *SI = cast<StoreInst>(User);
    assert(isa<ConstantPointerNull>(SI->getOperand(0)) &&
           "Unexpected heap-sra user!");

    // Insert a store of null into each global.
    for (unsigned i = 0, e = FieldGlobals.size(); i != e; ++i) {
      PointerType *PT = cast<PointerType>(FieldGlobals[i]->getType());
      Constant *Null = Constant::getNullValue(PT->getElementType());
      new StoreInst(Null, FieldGlobals[i], SI);
    }
    // Erase the original store.
    SI->eraseFromParent();
  }

  // While we have PHIs that are interesting to rewrite, do it.
  while (!PHIsToRewrite.empty()) {
    PHINode *PN = PHIsToRewrite.back().first;
    unsigned FieldNo = PHIsToRewrite.back().second;
    PHIsToRewrite.pop_back();
    PHINode *FieldPN = cast<PHINode>(InsertedScalarizedValues[PN][FieldNo]);
    assert(FieldPN->getNumIncomingValues() == 0 &&"Already processed this phi");

    // Add all the incoming values.  This can materialize more phis.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      Value *InVal = PN->getIncomingValue(i);
      InVal = GetHeapSROAValue(InVal, FieldNo, InsertedScalarizedValues,
                               PHIsToRewrite);
      FieldPN->addIncoming(InVal, PN->getIncomingBlock(i));
    }
  }

  // Drop all inter-phi links and any loads that made it this far.
  for (DenseMap<Value*, std::vector<Value*> >::iterator
       I = InsertedScalarizedValues.begin(), E = InsertedScalarizedValues.end();
       I != E; ++I) {
    if (PHINode *PN = dyn_cast<PHINode>(I->first))
      PN->dropAllReferences();
    else if (LoadInst *LI = dyn_cast<LoadInst>(I->first))
      LI->dropAllReferences();
  }

  // Delete all the phis and loads now that inter-references are dead.
  for (DenseMap<Value*, std::vector<Value*> >::iterator
       I = InsertedScalarizedValues.begin(), E = InsertedScalarizedValues.end();
       I != E; ++I) {
    if (PHINode *PN = dyn_cast<PHINode>(I->first))
      PN->eraseFromParent();
    else if (LoadInst *LI = dyn_cast<LoadInst>(I->first))
      LI->eraseFromParent();
  }

  // The old global is now dead, remove it.
  GV->eraseFromParent();

  ++NumHeapSRA;
  return cast<GlobalVariable>(FieldGlobals[0]);
}

/// TryToOptimizeStoreOfMallocToGlobal - This function is called when we see a
/// pointer global variable with a single value stored it that is a malloc or
/// cast of malloc.
static bool TryToOptimizeStoreOfMallocToGlobal(GlobalVariable *GV, CallInst *CI,
                                               Type *AllocTy,
                                               AtomicOrdering Ordering,
                                               Module::global_iterator &GVI,
                                               const DataLayout &DL,
                                               TargetLibraryInfo *TLI) {
  // If this is a malloc of an abstract type, don't touch it.
  if (!AllocTy->isSized())
    return false;

  // We can't optimize this global unless all uses of it are *known* to be
  // of the malloc value, not of the null initializer value (consider a use
  // that compares the global's value against zero to see if the malloc has
  // been reached).  To do this, we check to see if all uses of the global
  // would trap if the global were null: this proves that they must all
  // happen after the malloc.
  if (!AllUsesOfLoadedValueWillTrapIfNull(GV))
    return false;

  // We can't optimize this if the malloc itself is used in a complex way,
  // for example, being stored into multiple globals.  This allows the
  // malloc to be stored into the specified global, loaded icmp'd, and
  // GEP'd.  These are all things we could transform to using the global
  // for.
  SmallPtrSet<const PHINode*, 8> PHIs;
  if (!ValueIsOnlyUsedLocallyOrStoredToOneGlobal(CI, GV, PHIs))
    return false;

  // If we have a global that is only initialized with a fixed size malloc,
  // transform the program to use global memory instead of malloc'd memory.
  // This eliminates dynamic allocation, avoids an indirection accessing the
  // data, and exposes the resultant global to further GlobalOpt.
  // We cannot optimize the malloc if we cannot determine malloc array size.
  Value *NElems = getMallocArraySize(CI, DL, TLI, true);
  if (!NElems)
    return false;

  if (ConstantInt *NElements = dyn_cast<ConstantInt>(NElems))
    // Restrict this transformation to only working on small allocations
    // (2048 bytes currently), as we don't want to introduce a 16M global or
    // something.
    if (NElements->getZExtValue() * DL.getTypeAllocSize(AllocTy) < 2048) {
      GVI = OptimizeGlobalAddressOfMalloc(GV, CI, AllocTy, NElements, DL, TLI);
      return true;
    }

  // If the allocation is an array of structures, consider transforming this
  // into multiple malloc'd arrays, one for each field.  This is basically
  // SRoA for malloc'd memory.

  if (Ordering != NotAtomic)
    return false;

  // If this is an allocation of a fixed size array of structs, analyze as a
  // variable size array.  malloc [100 x struct],1 -> malloc struct, 100
  if (NElems == ConstantInt::get(CI->getArgOperand(0)->getType(), 1))
    if (ArrayType *AT = dyn_cast<ArrayType>(AllocTy))
      AllocTy = AT->getElementType();

  StructType *AllocSTy = dyn_cast<StructType>(AllocTy);
  if (!AllocSTy)
    return false;

  // This the structure has an unreasonable number of fields, leave it
  // alone.
  if (AllocSTy->getNumElements() <= 16 && AllocSTy->getNumElements() != 0 &&
      AllGlobalLoadUsesSimpleEnoughForHeapSRA(GV, CI)) {

    // If this is a fixed size array, transform the Malloc to be an alloc of
    // structs.  malloc [100 x struct],1 -> malloc struct, 100
    if (ArrayType *AT = dyn_cast<ArrayType>(getMallocAllocatedType(CI, TLI))) {
      Type *IntPtrTy = DL.getIntPtrType(CI->getType());
      unsigned TypeSize = DL.getStructLayout(AllocSTy)->getSizeInBytes();
      Value *AllocSize = ConstantInt::get(IntPtrTy, TypeSize);
      Value *NumElements = ConstantInt::get(IntPtrTy, AT->getNumElements());
      Instruction *Malloc = CallInst::CreateMalloc(CI, IntPtrTy, AllocSTy,
                                                   AllocSize, NumElements,
                                                   nullptr, CI->getName());
      Instruction *Cast = new BitCastInst(Malloc, CI->getType(), "tmp", CI);
      CI->replaceAllUsesWith(Cast);
      CI->eraseFromParent();
      if (BitCastInst *BCI = dyn_cast<BitCastInst>(Malloc))
        CI = cast<CallInst>(BCI->getOperand(0));
      else
        CI = cast<CallInst>(Malloc);
    }

    GVI = PerformHeapAllocSRoA(GV, CI, getMallocArraySize(CI, DL, TLI, true),
                               DL, TLI);
    return true;
  }

  return false;
}

// OptimizeOnceStoredGlobal - Try to optimize globals based on the knowledge
// that only one value (besides its initializer) is ever stored to the global.
static bool OptimizeOnceStoredGlobal(GlobalVariable *GV, Value *StoredOnceVal,
                                     AtomicOrdering Ordering,
                                     Module::global_iterator &GVI,
                                     const DataLayout &DL,
                                     TargetLibraryInfo *TLI) {
  // Ignore no-op GEPs and bitcasts.
  StoredOnceVal = StoredOnceVal->stripPointerCasts();

  // If we are dealing with a pointer global that is initialized to null and
  // only has one (non-null) value stored into it, then we can optimize any
  // users of the loaded value (often calls and loads) that would trap if the
  // value was null.
  if (GV->getInitializer()->getType()->isPointerTy() &&
      GV->getInitializer()->isNullValue()) {
    if (Constant *SOVC = dyn_cast<Constant>(StoredOnceVal)) {
      if (GV->getInitializer()->getType() != SOVC->getType())
        SOVC = ConstantExpr::getBitCast(SOVC, GV->getInitializer()->getType());

      // Optimize away any trapping uses of the loaded value.
      if (OptimizeAwayTrappingUsesOfLoads(GV, SOVC, DL, TLI))
        return true;
    } else if (CallInst *CI = extractMallocCall(StoredOnceVal, TLI)) {
      Type *MallocType = getMallocAllocatedType(CI, TLI);
      if (MallocType &&
          TryToOptimizeStoreOfMallocToGlobal(GV, CI, MallocType, Ordering, GVI,
                                             DL, TLI))
        return true;
    }
  }

  return false;
}

/// TryToShrinkGlobalToBoolean - At this point, we have learned that the only
/// two values ever stored into GV are its initializer and OtherVal.  See if we
/// can shrink the global into a boolean and select between the two values
/// whenever it is used.  This exposes the values to other scalar optimizations.
static bool TryToShrinkGlobalToBoolean(GlobalVariable *GV, Constant *OtherVal) {
  Type *GVElType = GV->getType()->getElementType();

  // If GVElType is already i1, it is already shrunk.  If the type of the GV is
  // an FP value, pointer or vector, don't do this optimization because a select
  // between them is very expensive and unlikely to lead to later
  // simplification.  In these cases, we typically end up with "cond ? v1 : v2"
  // where v1 and v2 both require constant pool loads, a big loss.
  if (GVElType == Type::getInt1Ty(GV->getContext()) ||
      GVElType->isFloatingPointTy() ||
      GVElType->isPointerTy() || GVElType->isVectorTy())
    return false;

  // Walk the use list of the global seeing if all the uses are load or store.
  // If there is anything else, bail out.
  for (User *U : GV->users())
    if (!isa<LoadInst>(U) && !isa<StoreInst>(U))
      return false;

  DEBUG(dbgs() << "   *** SHRINKING TO BOOL: " << *GV);

  // Create the new global, initializing it to false.
  GlobalVariable *NewGV = new GlobalVariable(Type::getInt1Ty(GV->getContext()),
                                             false,
                                             GlobalValue::InternalLinkage,
                                        ConstantInt::getFalse(GV->getContext()),
                                             GV->getName()+".b",
                                             GV->getThreadLocalMode(),
                                             GV->getType()->getAddressSpace());
  GV->getParent()->getGlobalList().insert(GV, NewGV);

  Constant *InitVal = GV->getInitializer();
  assert(InitVal->getType() != Type::getInt1Ty(GV->getContext()) &&
         "No reason to shrink to bool!");

  // If initialized to zero and storing one into the global, we can use a cast
  // instead of a select to synthesize the desired value.
  bool IsOneZero = false;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(OtherVal))
    IsOneZero = InitVal->isNullValue() && CI->isOne();

  while (!GV->use_empty()) {
    Instruction *UI = cast<Instruction>(GV->user_back());
    if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Change the store into a boolean store.
      bool StoringOther = SI->getOperand(0) == OtherVal;
      // Only do this if we weren't storing a loaded value.
      Value *StoreVal;
      if (StoringOther || SI->getOperand(0) == InitVal) {
        StoreVal = ConstantInt::get(Type::getInt1Ty(GV->getContext()),
                                    StoringOther);
      } else {
        // Otherwise, we are storing a previously loaded copy.  To do this,
        // change the copy from copying the original value to just copying the
        // bool.
        Instruction *StoredVal = cast<Instruction>(SI->getOperand(0));

        // If we've already replaced the input, StoredVal will be a cast or
        // select instruction.  If not, it will be a load of the original
        // global.
        if (LoadInst *LI = dyn_cast<LoadInst>(StoredVal)) {
          assert(LI->getOperand(0) == GV && "Not a copy!");
          // Insert a new load, to preserve the saved value.
          StoreVal = new LoadInst(NewGV, LI->getName()+".b", false, 0,
                                  LI->getOrdering(), LI->getSynchScope(), LI);
        } else {
          assert((isa<CastInst>(StoredVal) || isa<SelectInst>(StoredVal)) &&
                 "This is not a form that we understand!");
          StoreVal = StoredVal->getOperand(0);
          assert(isa<LoadInst>(StoreVal) && "Not a load of NewGV!");
        }
      }
      new StoreInst(StoreVal, NewGV, false, 0,
                    SI->getOrdering(), SI->getSynchScope(), SI);
    } else {
      // Change the load into a load of bool then a select.
      LoadInst *LI = cast<LoadInst>(UI);
      LoadInst *NLI = new LoadInst(NewGV, LI->getName()+".b", false, 0,
                                   LI->getOrdering(), LI->getSynchScope(), LI);
      Value *NSI;
      if (IsOneZero)
        NSI = new ZExtInst(NLI, LI->getType(), "", LI);
      else
        NSI = SelectInst::Create(NLI, OtherVal, InitVal, "", LI);
      NSI->takeName(LI);
      LI->replaceAllUsesWith(NSI);
    }
    UI->eraseFromParent();
  }

  // Retain the name of the old global variable. People who are debugging their
  // programs may expect these variables to be named the same.
  NewGV->takeName(GV);
  GV->eraseFromParent();
  return true;
}


/// ProcessGlobal - Analyze the specified global variable and optimize it if
/// possible.  If we make a change, return true.
bool GlobalOpt::ProcessGlobal(GlobalVariable *GV,
                              Module::global_iterator &GVI) {
  // Do more involved optimizations if the global is internal.
  GV->removeDeadConstantUsers();

  if (GV->use_empty()) {
    DEBUG(dbgs() << "GLOBAL DEAD: " << *GV);
    GV->eraseFromParent();
    ++NumDeleted;
    return true;
  }

  if (!GV->hasLocalLinkage())
    return false;

  GlobalStatus GS;

  if (GlobalStatus::analyzeGlobal(GV, GS))
    return false;

  if (!GS.IsCompared && !GV->hasUnnamedAddr()) {
    GV->setUnnamedAddr(true);
    NumUnnamed++;
  }

  if (GV->isConstant() || !GV->hasInitializer())
    return false;

  return ProcessInternalGlobal(GV, GVI, GS);
}

// HLSL Change Begin
static bool isEntryPoint(const llvm::Function* Func) {
  const llvm::Module* Mod = Func->getParent();
  return Mod->HasDxilModule()
    ? Mod->GetDxilModule().IsEntryOrPatchConstantFunction(Func)
    : Func->getName() == "main"; // Original logic for non-HLSL
}
// HLSL Change End

/// ProcessInternalGlobal - Analyze the specified global variable and optimize
/// it if possible.  If we make a change, return true.
bool GlobalOpt::ProcessInternalGlobal(GlobalVariable *GV,
                                      Module::global_iterator &GVI,
                                      const GlobalStatus &GS) {
  auto &DL = GV->getParent()->getDataLayout();


  // If this is a first class global and has only one accessing function
  // and this function is main (which we know is not recursive), we replace
  // the global with a local alloca in this function.
  //
  // NOTE: It doesn't make sense to promote non-single-value types since we
  // are just replacing static memory to stack memory.
  //
  // If the global is in different address space, don't bring it to stack.
  if (!GS.HasMultipleAccessingFunctions &&
      GS.AccessingFunction && !GS.HasNonInstructionUser &&
      GV->getType()->getElementType()->isSingleValueType() &&
      isEntryPoint(GS.AccessingFunction) && // HLSL Change - Generalize entrypoint testing
      GS.AccessingFunction->hasExternalLinkage() &&
      GV->getType()->getAddressSpace() == 0) {
    DEBUG(dbgs() << "LOCALIZING GLOBAL: " << *GV);
    Instruction &FirstI = const_cast<Instruction&>(*GS.AccessingFunction
                                                   ->getEntryBlock().begin());
    Type *ElemTy = GV->getType()->getElementType();
    // FIXME: Pass Global's alignment when globals have alignment
    AllocaInst *Alloca = new AllocaInst(ElemTy, nullptr,
                                        GV->getName(), &FirstI);
    if (!isa<UndefValue>(GV->getInitializer()))
      new StoreInst(GV->getInitializer(), Alloca, &FirstI);

    GV->replaceAllUsesWith(Alloca);
    GV->eraseFromParent();
    ++NumLocalized;
    return true;
  }

  // If the global is never loaded (but may be stored to), it is dead.
  // Delete it now.
  if (!GS.IsLoaded) {
    DEBUG(dbgs() << "GLOBAL NEVER LOADED: " << *GV);

    bool Changed;
    if (isLeakCheckerRoot(GV)) {
      // Delete any constant stores to the global.
      Changed = CleanupPointerRootUsers(GV, TLI);
    } else {
      // Delete any stores we can find to the global.  We may not be able to
      // make it completely dead though.
      Changed = CleanupConstantGlobalUsers(GV, GV->getInitializer(), DL, TLI);
    }

    // If the global is dead now, delete it.
    if (GV->use_empty()) {
      GV->eraseFromParent();
      ++NumDeleted;
      Changed = true;
    }
    return Changed;

  } else if (GS.StoredType <= GlobalStatus::InitializerStored) {
    DEBUG(dbgs() << "MARKING CONSTANT: " << *GV << "\n");
    GV->setConstant(true);

    // Clean up any obviously simplifiable users now.
    CleanupConstantGlobalUsers(GV, GV->getInitializer(), DL, TLI);

    // If the global is dead now, just nuke it.
    if (GV->use_empty()) {
      DEBUG(dbgs() << "   *** Marking constant allowed us to simplify "
            << "all users and delete global!\n");
      GV->eraseFromParent();
      ++NumDeleted;
    }

    ++NumMarked;
    return true;
  } else if (!GV->getInitializer()->getType()->isSingleValueType()) {
    const DataLayout &DL = GV->getParent()->getDataLayout();
    if (GlobalVariable *FirstNewGV = SRAGlobal(GV, DL)) {
      GVI = FirstNewGV; // Don't skip the newly produced globals!
      return true;
    }
  } else if (GS.StoredType == GlobalStatus::StoredOnce) {
    // If the initial value for the global was an undef value, and if only
    // one other value was stored into it, we can just change the
    // initializer to be the stored value, then delete all stores to the
    // global.  This allows us to mark it constant.
    if (Constant *SOVConstant = dyn_cast<Constant>(GS.StoredOnceValue))
      if (isa<UndefValue>(GV->getInitializer())) {
        // Change the initial value here.
        GV->setInitializer(SOVConstant);

        // Clean up any obviously simplifiable users now.
        CleanupConstantGlobalUsers(GV, GV->getInitializer(), DL, TLI);

        if (GV->use_empty()) {
          DEBUG(dbgs() << "   *** Substituting initializer allowed us to "
                       << "simplify all users and delete global!\n");
          GV->eraseFromParent();
          ++NumDeleted;
        } else {
          GVI = GV;
        }
        ++NumSubstitute;
        return true;
      }

    // Try to optimize globals based on the knowledge that only one value
    // (besides its initializer) is ever stored to the global.
    if (OptimizeOnceStoredGlobal(GV, GS.StoredOnceValue, GS.Ordering, GVI,
                                 DL, TLI))
      return true;

    // Otherwise, if the global was not a boolean, we can shrink it to be a
    // boolean.
    if (Constant *SOVConstant = dyn_cast<Constant>(GS.StoredOnceValue)) {
      if (GS.Ordering == NotAtomic) {
        if (TryToShrinkGlobalToBoolean(GV, SOVConstant)) {
          ++NumShrunkToBool;
          return true;
        }
      }
    }
  }

  return false;
}

/// ChangeCalleesToFastCall - Walk all of the direct calls of the specified
/// function, changing them to FastCC.
static void ChangeCalleesToFastCall(Function *F) {
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    CallSite CS(cast<Instruction>(U));
    CS.setCallingConv(CallingConv::Fast);
  }
}

static AttributeSet StripNest(LLVMContext &C, const AttributeSet &Attrs) {
  for (unsigned i = 0, e = Attrs.getNumSlots(); i != e; ++i) {
    unsigned Index = Attrs.getSlotIndex(i);
    if (!Attrs.getSlotAttributes(i).hasAttribute(Index, Attribute::Nest))
      continue;

    // There can be only one.
    return Attrs.removeAttribute(C, Index, Attribute::Nest);
  }

  return Attrs;
}

static void RemoveNestAttribute(Function *F) {
  F->setAttributes(StripNest(F->getContext(), F->getAttributes()));
  for (User *U : F->users()) {
    if (isa<BlockAddress>(U))
      continue;
    CallSite CS(cast<Instruction>(U));
    CS.setAttributes(StripNest(F->getContext(), CS.getAttributes()));
  }
}

/// Return true if this is a calling convention that we'd like to change.  The
/// idea here is that we don't want to mess with the convention if the user
/// explicitly requested something with performance implications like coldcc,
/// GHC, or anyregcc.
static bool isProfitableToMakeFastCC(Function *F) {
  CallingConv::ID CC = F->getCallingConv();
  // FIXME: Is it worth transforming x86_stdcallcc and x86_fastcallcc?
  return CC == CallingConv::C || CC == CallingConv::X86_ThisCall;
}

bool GlobalOpt::OptimizeFunctions(Module &M) {
  bool Changed = false;
  // Optimize functions.
  for (Module::iterator FI = M.begin(), E = M.end(); FI != E; ) {
    Function *F = FI++;
    // Functions without names cannot be referenced outside this module.
    if (!F->hasName() && !F->isDeclaration() && !F->hasLocalLinkage())
      F->setLinkage(GlobalValue::InternalLinkage);

    const Comdat *C = F->getComdat();
    bool inComdat = C && NotDiscardableComdats.count(C);
    F->removeDeadConstantUsers();
    if ((!inComdat || F->hasLocalLinkage()) && F->isDefTriviallyDead()) {
      F->eraseFromParent();
      Changed = true;
      ++NumFnDeleted;
    } else if (F->hasLocalLinkage()) {
      if (isProfitableToMakeFastCC(F) && !F->isVarArg() &&
          !F->hasAddressTaken()) {
        // If this function has a calling convention worth changing, is not a
        // varargs function, and is only called directly, promote it to use the
        // Fast calling convention.
        F->setCallingConv(CallingConv::Fast);
        ChangeCalleesToFastCall(F);
        ++NumFastCallFns;
        Changed = true;
      }

      if (F->getAttributes().hasAttrSomewhere(Attribute::Nest) &&
          !F->hasAddressTaken()) {
        // The function is not used by a trampoline intrinsic, so it is safe
        // to remove the 'nest' attribute.
        RemoveNestAttribute(F);
        ++NumNestRemoved;
        Changed = true;
      }
    }
  }
  return Changed;
}

bool GlobalOpt::OptimizeGlobalVars(Module &M) {
  bool Changed = false;

  for (Module::global_iterator GVI = M.global_begin(), E = M.global_end();
       GVI != E; ) {
    GlobalVariable *GV = GVI++;
    // Global variables without names cannot be referenced outside this module.
    if (!GV->hasName() && !GV->isDeclaration() && !GV->hasLocalLinkage())
      GV->setLinkage(GlobalValue::InternalLinkage);
    // Simplify the initializer.
    if (GV->hasInitializer())
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GV->getInitializer())) {
        auto &DL = M.getDataLayout();
        Constant *New = ConstantFoldConstantExpression(CE, DL, TLI);
        if (New && New != CE)
          GV->setInitializer(New);
      }

    if (GV->isDiscardableIfUnused()) {
      if (const Comdat *C = GV->getComdat())
        if (NotDiscardableComdats.count(C) && !GV->hasLocalLinkage())
          continue;
      Changed |= ProcessGlobal(GV, GVI);
    }
  }
  return Changed;
}

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL);

/// isSimpleEnoughValueToCommit - Return true if the specified constant can be
/// handled by the code generator.  We don't want to generate something like:
///   void *X = &X/42;
/// because the code generator doesn't have a relocation that can handle that.
///
/// This function should be called if C was not found (but just got inserted)
/// in SimpleConstants to avoid having to rescan the same constants all the
/// time.
static bool
isSimpleEnoughValueToCommitHelper(Constant *C,
                                  SmallPtrSetImpl<Constant *> &SimpleConstants,
                                  const DataLayout &DL) {
  // Simple global addresses are supported, do not allow dllimport or
  // thread-local globals.
  if (auto *GV = dyn_cast<GlobalValue>(C))
    return !GV->hasDLLImportStorageClass() && !GV->isThreadLocal();

  // Simple integer, undef, constant aggregate zero, etc are all supported.
  if (C->getNumOperands() == 0 || isa<BlockAddress>(C))
    return true;

  // Aggregate values are safe if all their elements are.
  if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
      isa<ConstantVector>(C)) {
    for (Value *Op : C->operands())
      if (!isSimpleEnoughValueToCommit(cast<Constant>(Op), SimpleConstants, DL))
        return false;
    return true;
  }

  // We don't know exactly what relocations are allowed in constant expressions,
  // so we allow &global+constantoffset, which is safe and uniformly supported
  // across targets.
  ConstantExpr *CE = cast<ConstantExpr>(C);
  switch (CE->getOpcode()) {
  case Instruction::BitCast:
    // Bitcast is fine if the casted value is fine.
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
    // int <=> ptr is fine if the int type is the same size as the
    // pointer type.
    if (DL.getTypeSizeInBits(CE->getType()) !=
        DL.getTypeSizeInBits(CE->getOperand(0)->getType()))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  // GEP is fine if it is simple + constant offset.
  case Instruction::GetElementPtr:
    for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
      if (!isa<ConstantInt>(CE->getOperand(i)))
        return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);

  case Instruction::Add:
    // We allow simple+cst.
    if (!isa<ConstantInt>(CE->getOperand(1)))
      return false;
    return isSimpleEnoughValueToCommit(CE->getOperand(0), SimpleConstants, DL);
  }
  return false;
}

static inline bool
isSimpleEnoughValueToCommit(Constant *C,
                            SmallPtrSetImpl<Constant *> &SimpleConstants,
                            const DataLayout &DL) {
  // If we already checked this constant, we win.
  if (!SimpleConstants.insert(C).second)
    return true;
  // Check the constant.
  return isSimpleEnoughValueToCommitHelper(C, SimpleConstants, DL);
}


/// isSimpleEnoughPointerToCommit - Return true if this constant is simple
/// enough for us to understand.  In particular, if it is a cast to anything
/// other than from one pointer type to another pointer type, we punt.
/// We basically just support direct accesses to globals and GEP's of
/// globals.  This should be kept up to date with CommitValueTo.
static bool isSimpleEnoughPointerToCommit(Constant *C) {
  // Conservatively, avoid aggregate types. This is because we don't
  // want to worry about them partially overlapping other stores.
  if (!cast<PointerType>(C->getType())->getElementType()->isSingleValueType())
    return false;

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    // Do not allow weak/*_odr/linkonce linkage or external globals.
    return GV->hasUniqueInitializer();

  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Handle a constantexpr gep.
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0)) &&
        cast<GEPOperator>(CE)->isInBounds()) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      if (!GV->hasUniqueInitializer())
        return false;

      // The first index must be zero.
      ConstantInt *CI = dyn_cast<ConstantInt>(*std::next(CE->op_begin()));
      if (!CI || !CI->isZero()) return false;

      // The remaining indices must be compile-time known integers within the
      // notional bounds of the corresponding static array types.
      if (!CE->isGEPWithNoNotionalOverIndexing())
        return false;

      return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);

    // A constantexpr bitcast from a pointer to another pointer is a no-op,
    // and we know how to evaluate it by moving the bitcast from the pointer
    // operand to the value operand.
    } else if (CE->getOpcode() == Instruction::BitCast &&
               isa<GlobalVariable>(CE->getOperand(0))) {
      // Do not allow weak/*_odr/linkonce/dllimport/dllexport linkage or
      // external globals.
      return cast<GlobalVariable>(CE->getOperand(0))->hasUniqueInitializer();
    }
  }

  return false;
}

/// EvaluateStoreInto - Evaluate a piece of a constantexpr store into a global
/// initializer.  This returns 'Init' modified to reflect 'Val' stored into it.
/// At this point, the GEP operands of Addr [0, OpNo) have been stepped into.
static Constant *EvaluateStoreInto(Constant *Init, Constant *Val,
                                   ConstantExpr *Addr, unsigned OpNo) {
  // Base case of the recursion.
  if (OpNo == Addr->getNumOperands()) {
    assert(Val->getType() == Init->getType() && "Type mismatch!");
    return Val;
  }

  SmallVector<Constant*, 32> Elts;
  if (StructType *STy = dyn_cast<StructType>(Init->getType())) {
    // Break up the constant into its elements.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i)
      Elts.push_back(Init->getAggregateElement(i));

    // Replace the element that we are supposed to.
    ConstantInt *CU = cast<ConstantInt>(Addr->getOperand(OpNo));
    unsigned Idx = CU->getZExtValue();
    assert(Idx < STy->getNumElements() && "Struct index out of range!");
    Elts[Idx] = EvaluateStoreInto(Elts[Idx], Val, Addr, OpNo+1);

    // Return the modified struct.
    return ConstantStruct::get(STy, Elts);
  }

  ConstantInt *CI = cast<ConstantInt>(Addr->getOperand(OpNo));
  SequentialType *InitTy = cast<SequentialType>(Init->getType());

  uint64_t NumElts;
  if (ArrayType *ATy = dyn_cast<ArrayType>(InitTy))
    NumElts = ATy->getNumElements();
  else
    NumElts = InitTy->getVectorNumElements();

  // Break up the array into elements.
  for (uint64_t i = 0, e = NumElts; i != e; ++i)
    Elts.push_back(Init->getAggregateElement(i));

  assert(CI->getZExtValue() < NumElts);
  Elts[CI->getZExtValue()] =
    EvaluateStoreInto(Elts[CI->getZExtValue()], Val, Addr, OpNo+1);

  if (Init->getType()->isArrayTy())
    return ConstantArray::get(cast<ArrayType>(InitTy), Elts);
  return ConstantVector::get(Elts);
}

/// CommitValueTo - We have decided that Addr (which satisfies the predicate
/// isSimpleEnoughPointerToCommit) should get Val as its value.  Make it happen.
static void CommitValueTo(Constant *Val, Constant *Addr) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    assert(GV->hasInitializer());
    GV->setInitializer(Val);
    return;
  }

  ConstantExpr *CE = cast<ConstantExpr>(Addr);
  GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
  GV->setInitializer(EvaluateStoreInto(GV->getInitializer(), Val, CE, 2));
}

namespace {

/// Evaluator - This class evaluates LLVM IR, producing the Constant
/// representing each SSA instruction.  Changes to global variables are stored
/// in a mapping that can be iterated over after the evaluation is complete.
/// Once an evaluation call fails, the evaluation object should not be reused.
class Evaluator {
public:
  Evaluator(const DataLayout &DL, const TargetLibraryInfo *TLI)
      : DL(DL), TLI(TLI) {
    ValueStack.emplace_back();
  }

  ~Evaluator() {
    for (auto &Tmp : AllocaTmps)
      // If there are still users of the alloca, the program is doing something
      // silly, e.g. storing the address of the alloca somewhere and using it
      // later.  Since this is undefined, we'll just make it be null.
      if (!Tmp->use_empty())
        Tmp->replaceAllUsesWith(Constant::getNullValue(Tmp->getType()));
  }

  /// EvaluateFunction - Evaluate a call to function F, returning true if
  /// successful, false if we can't evaluate it.  ActualArgs contains the formal
  /// arguments for the function.
  bool EvaluateFunction(Function *F, Constant *&RetVal,
                        const SmallVectorImpl<Constant*> &ActualArgs);

  /// EvaluateBlock - Evaluate all instructions in block BB, returning true if
  /// successful, false if we can't evaluate it.  NewBB returns the next BB that
  /// control flows into, or null upon return.
  bool EvaluateBlock(BasicBlock::iterator CurInst, BasicBlock *&NextBB);

  Constant *getVal(Value *V) {
    if (Constant *CV = dyn_cast<Constant>(V)) return CV;
    Constant *R = ValueStack.back().lookup(V);
    assert(R && "Reference to an uncomputed value!");
    return R;
  }

  void setVal(Value *V, Constant *C) {
    ValueStack.back()[V] = C;
  }

  const DenseMap<Constant*, Constant*> &getMutatedMemory() const {
    return MutatedMemory;
  }

  const SmallPtrSetImpl<GlobalVariable*> &getInvariants() const {
    return Invariants;
  }

private:
  Constant *ComputeLoadResult(Constant *P);

  /// ValueStack - As we compute SSA register values, we store their contents
  /// here. The back of the deque contains the current function and the stack
  /// contains the values in the calling frames.
  std::deque<DenseMap<Value*, Constant*>> ValueStack;

  /// CallStack - This is used to detect recursion.  In pathological situations
  /// we could hit exponential behavior, but at least there is nothing
  /// unbounded.
  SmallVector<Function*, 4> CallStack;

  /// MutatedMemory - For each store we execute, we update this map.  Loads
  /// check this to get the most up-to-date value.  If evaluation is successful,
  /// this state is committed to the process.
  DenseMap<Constant*, Constant*> MutatedMemory;

  /// AllocaTmps - To 'execute' an alloca, we create a temporary global variable
  /// to represent its body.  This vector is needed so we can delete the
  /// temporary globals when we are done.
  SmallVector<std::unique_ptr<GlobalVariable>, 32> AllocaTmps;

  /// Invariants - These global variables have been marked invariant by the
  /// static constructor.
  SmallPtrSet<GlobalVariable*, 8> Invariants;

  /// SimpleConstants - These are constants we have checked and know to be
  /// simple enough to live in a static initializer of a global.
  SmallPtrSet<Constant*, 8> SimpleConstants;

  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
};

}  // anonymous namespace

/// ComputeLoadResult - Return the value that would be computed by a load from
/// P after the stores reflected by 'memory' have been performed.  If we can't
/// decide, return null.
Constant *Evaluator::ComputeLoadResult(Constant *P) {
  // If this memory location has been recently stored, use the stored value: it
  // is the most up-to-date.
  DenseMap<Constant*, Constant*>::const_iterator I = MutatedMemory.find(P);
  if (I != MutatedMemory.end()) return I->second;

  // Access it.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(P)) {
    if (GV->hasDefinitiveInitializer())
      return GV->getInitializer();
    return nullptr;
  }

  // Handle a constantexpr getelementptr.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(P))
    if (CE->getOpcode() == Instruction::GetElementPtr &&
        isa<GlobalVariable>(CE->getOperand(0))) {
      GlobalVariable *GV = cast<GlobalVariable>(CE->getOperand(0));
      if (GV->hasDefinitiveInitializer())
        return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(), CE);
    }

  return nullptr;  // don't know how to evaluate.
}

/// EvaluateBlock - Evaluate all instructions in block BB, returning true if
/// successful, false if we can't evaluate it.  NewBB returns the next BB that
/// control flows into, or null upon return.
bool Evaluator::EvaluateBlock(BasicBlock::iterator CurInst,
                              BasicBlock *&NextBB) {
  // This is the main evaluation loop.
  while (1) {
    Constant *InstResult = nullptr;

    DEBUG(dbgs() << "Evaluating Instruction: " << *CurInst << "\n");

    if (StoreInst *SI = dyn_cast<StoreInst>(CurInst)) {
      if (!SI->isSimple()) {
        DEBUG(dbgs() << "Store is not simple! Can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }
      Constant *Ptr = getVal(SI->getOperand(1));
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        DEBUG(dbgs() << "Folding constant ptr expression: " << *Ptr);
        Ptr = ConstantFoldConstantExpression(CE, DL, TLI);
        DEBUG(dbgs() << "; To: " << *Ptr << "\n");
      }
      if (!isSimpleEnoughPointerToCommit(Ptr)) {
        // If this is too complex for us to commit, reject it.
        DEBUG(dbgs() << "Pointer is too complex for us to evaluate store.");
        return false;
      }

      Constant *Val = getVal(SI->getOperand(0));

      // If this might be too difficult for the backend to handle (e.g. the addr
      // of one global variable divided by another) then we can't commit it.
      if (!isSimpleEnoughValueToCommit(Val, SimpleConstants, DL)) {
        DEBUG(dbgs() << "Store value is too complex to evaluate store. " << *Val
              << "\n");
        return false;
      }

      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        if (CE->getOpcode() == Instruction::BitCast) {
          DEBUG(dbgs() << "Attempting to resolve bitcast on constant ptr.\n");
          // If we're evaluating a store through a bitcast, then we need
          // to pull the bitcast off the pointer type and push it onto the
          // stored value.
          Ptr = CE->getOperand(0);

          Type *NewTy = cast<PointerType>(Ptr->getType())->getElementType();

          // In order to push the bitcast onto the stored value, a bitcast
          // from NewTy to Val's type must be legal.  If it's not, we can try
          // introspecting NewTy to find a legal conversion.
          while (!Val->getType()->canLosslesslyBitCastTo(NewTy)) {
            // If NewTy is a struct, we can convert the pointer to the struct
            // into a pointer to its first member.
            // FIXME: This could be extended to support arrays as well.
            if (StructType *STy = dyn_cast<StructType>(NewTy)) {
              NewTy = STy->getTypeAtIndex(0U);

              IntegerType *IdxTy = IntegerType::get(NewTy->getContext(), 32);
              Constant *IdxZero = ConstantInt::get(IdxTy, 0, false);
              Constant * const IdxList[] = {IdxZero, IdxZero};

              Ptr = ConstantExpr::getGetElementPtr(nullptr, Ptr, IdxList);
              if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
                Ptr = ConstantFoldConstantExpression(CE, DL, TLI);

            // If we can't improve the situation by introspecting NewTy,
            // we have to give up.
            } else {
              DEBUG(dbgs() << "Failed to bitcast constant ptr, can not "
                    "evaluate.\n");
              return false;
            }
          }

          // If we found compatible types, go ahead and push the bitcast
          // onto the stored value.
          Val = ConstantExpr::getBitCast(Val, NewTy);

          DEBUG(dbgs() << "Evaluated bitcast: " << *Val << "\n");
        }
      }

      MutatedMemory[Ptr] = Val;
    } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(CurInst)) {
      InstResult = ConstantExpr::get(BO->getOpcode(),
                                     getVal(BO->getOperand(0)),
                                     getVal(BO->getOperand(1)));
      DEBUG(dbgs() << "Found a BinaryOperator! Simplifying: " << *InstResult
            << "\n");
    } else if (CmpInst *CI = dyn_cast<CmpInst>(CurInst)) {
      InstResult = ConstantExpr::getCompare(CI->getPredicate(),
                                            getVal(CI->getOperand(0)),
                                            getVal(CI->getOperand(1)));
      DEBUG(dbgs() << "Found a CmpInst! Simplifying: " << *InstResult
            << "\n");
    } else if (CastInst *CI = dyn_cast<CastInst>(CurInst)) {
      InstResult = ConstantExpr::getCast(CI->getOpcode(),
                                         getVal(CI->getOperand(0)),
                                         CI->getType());
      DEBUG(dbgs() << "Found a Cast! Simplifying: " << *InstResult
            << "\n");
    } else if (SelectInst *SI = dyn_cast<SelectInst>(CurInst)) {
      InstResult = ConstantExpr::getSelect(getVal(SI->getOperand(0)),
                                           getVal(SI->getOperand(1)),
                                           getVal(SI->getOperand(2)));
      DEBUG(dbgs() << "Found a Select! Simplifying: " << *InstResult
            << "\n");
    } else if (auto *EVI = dyn_cast<ExtractValueInst>(CurInst)) {
      InstResult = ConstantExpr::getExtractValue(
          getVal(EVI->getAggregateOperand()), EVI->getIndices());
      DEBUG(dbgs() << "Found an ExtractValueInst! Simplifying: " << *InstResult
                   << "\n");
    } else if (auto *IVI = dyn_cast<InsertValueInst>(CurInst)) {
      InstResult = ConstantExpr::getInsertValue(
          getVal(IVI->getAggregateOperand()),
          getVal(IVI->getInsertedValueOperand()), IVI->getIndices());
      DEBUG(dbgs() << "Found an InsertValueInst! Simplifying: " << *InstResult
                   << "\n");
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurInst)) {
      Constant *P = getVal(GEP->getOperand(0));
      SmallVector<Constant*, 8> GEPOps;
      for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end();
           i != e; ++i)
        GEPOps.push_back(getVal(*i));
      InstResult =
          ConstantExpr::getGetElementPtr(GEP->getSourceElementType(), P, GEPOps,
                                         cast<GEPOperator>(GEP)->isInBounds());
      DEBUG(dbgs() << "Found a GEP! Simplifying: " << *InstResult
            << "\n");
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurInst)) {

      if (!LI->isSimple()) {
        DEBUG(dbgs() << "Found a Load! Not a simple load, can not evaluate.\n");
        return false;  // no volatile/atomic accesses.
      }

      Constant *Ptr = getVal(LI->getOperand(0));
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr)) {
        Ptr = ConstantFoldConstantExpression(CE, DL, TLI);
        DEBUG(dbgs() << "Found a constant pointer expression, constant "
              "folding: " << *Ptr << "\n");
      }
      InstResult = ComputeLoadResult(Ptr);
      if (!InstResult) {
        DEBUG(dbgs() << "Failed to compute load result. Can not evaluate load."
              "\n");
        return false; // Could not evaluate load.
      }

      DEBUG(dbgs() << "Evaluated load: " << *InstResult << "\n");
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(CurInst)) {
      if (AI->isArrayAllocation()) {
        DEBUG(dbgs() << "Found an array alloca. Can not evaluate.\n");
        return false;  // Cannot handle array allocs.
      }
      Type *Ty = AI->getType()->getElementType();
      AllocaTmps.push_back(
          make_unique<GlobalVariable>(Ty, false, GlobalValue::InternalLinkage,
                                      UndefValue::get(Ty), AI->getName()));
      InstResult = AllocaTmps.back().get();
      DEBUG(dbgs() << "Found an alloca. Result: " << *InstResult << "\n");
    } else if (isa<CallInst>(CurInst) || isa<InvokeInst>(CurInst)) {
      CallSite CS(CurInst);

      // Debug info can safely be ignored here.
      if (isa<DbgInfoIntrinsic>(CS.getInstruction())) {
        DEBUG(dbgs() << "Ignoring debug info.\n");
        ++CurInst;
        continue;
      }

      // Cannot handle inline asm.
      if (isa<InlineAsm>(CS.getCalledValue())) {
        DEBUG(dbgs() << "Found inline asm, can not evaluate.\n");
        return false;
      }

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CS.getInstruction())) {
        if (MemSetInst *MSI = dyn_cast<MemSetInst>(II)) {
          if (MSI->isVolatile()) {
            DEBUG(dbgs() << "Can not optimize a volatile memset " <<
                  "intrinsic.\n");
            return false;
          }
          Constant *Ptr = getVal(MSI->getDest());
          Constant *Val = getVal(MSI->getValue());
          Constant *DestVal = ComputeLoadResult(getVal(Ptr));
          if (Val->isNullValue() && DestVal && DestVal->isNullValue()) {
            // This memset is a no-op.
            DEBUG(dbgs() << "Ignoring no-op memset.\n");
            ++CurInst;
            continue;
          }
        }

        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
          DEBUG(dbgs() << "Ignoring lifetime intrinsic.\n");
          ++CurInst;
          continue;
        }

        if (II->getIntrinsicID() == Intrinsic::invariant_start) {
          // We don't insert an entry into Values, as it doesn't have a
          // meaningful return value.
          if (!II->use_empty()) {
            DEBUG(dbgs() << "Found unused invariant_start. Can't evaluate.\n");
            return false;
          }
          ConstantInt *Size = cast<ConstantInt>(II->getArgOperand(0));
          Value *PtrArg = getVal(II->getArgOperand(1));
          Value *Ptr = PtrArg->stripPointerCasts();
          if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
            Type *ElemTy = cast<PointerType>(GV->getType())->getElementType();
            if (!Size->isAllOnesValue() &&
                Size->getValue().getLimitedValue() >=
                    DL.getTypeStoreSize(ElemTy)) {
              Invariants.insert(GV);
              DEBUG(dbgs() << "Found a global var that is an invariant: " << *GV
                    << "\n");
            } else {
              DEBUG(dbgs() << "Found a global var, but can not treat it as an "
                    "invariant.\n");
            }
          }
          // Continue even if we do nothing.
          ++CurInst;
          continue;
        }

        DEBUG(dbgs() << "Unknown intrinsic. Can not evaluate.\n");
        return false;
      }

      // Resolve function pointers.
      Function *Callee = dyn_cast<Function>(getVal(CS.getCalledValue()));
      if (!Callee || Callee->mayBeOverridden()) {
        DEBUG(dbgs() << "Can not resolve function pointer.\n");
        return false;  // Cannot resolve.
      }

      SmallVector<Constant*, 8> Formals;
      for (User::op_iterator i = CS.arg_begin(), e = CS.arg_end(); i != e; ++i)
        Formals.push_back(getVal(*i));

      if (Callee->isDeclaration()) {
        // If this is a function we can constant fold, do it.
        if (Constant *C = ConstantFoldCall(Callee, Formals, TLI)) {
          InstResult = C;
          DEBUG(dbgs() << "Constant folded function call. Result: " <<
                *InstResult << "\n");
        } else {
          DEBUG(dbgs() << "Can not constant fold function call.\n");
          return false;
        }
      } else {
        if (Callee->getFunctionType()->isVarArg()) {
          DEBUG(dbgs() << "Can not constant fold vararg function call.\n");
          return false;
        }

        Constant *RetVal = nullptr;
        // Execute the call, if successful, use the return value.
        ValueStack.emplace_back();
        if (!EvaluateFunction(Callee, RetVal, Formals)) {
          DEBUG(dbgs() << "Failed to evaluate function.\n");
          return false;
        }
        ValueStack.pop_back();
        InstResult = RetVal;

        if (InstResult) {
          DEBUG(dbgs() << "Successfully evaluated function. Result: " <<
                InstResult << "\n\n");
        } else {
          DEBUG(dbgs() << "Successfully evaluated function. Result: 0\n\n");
        }
      }
    } else if (isa<TerminatorInst>(CurInst)) {
      DEBUG(dbgs() << "Found a terminator instruction.\n");

      if (BranchInst *BI = dyn_cast<BranchInst>(CurInst)) {
        if (BI->isUnconditional()) {
          NextBB = BI->getSuccessor(0);
        } else {
          ConstantInt *Cond =
            dyn_cast<ConstantInt>(getVal(BI->getCondition()));
          if (!Cond) return false;  // Cannot determine.

          NextBB = BI->getSuccessor(!Cond->getZExtValue());
        }
      } else if (SwitchInst *SI = dyn_cast<SwitchInst>(CurInst)) {
        ConstantInt *Val =
          dyn_cast<ConstantInt>(getVal(SI->getCondition()));
        if (!Val) return false;  // Cannot determine.
        NextBB = SI->findCaseValue(Val).getCaseSuccessor();
      } else if (IndirectBrInst *IBI = dyn_cast<IndirectBrInst>(CurInst)) {
        Value *Val = getVal(IBI->getAddress())->stripPointerCasts();
        if (BlockAddress *BA = dyn_cast<BlockAddress>(Val))
          NextBB = BA->getBasicBlock();
        else
          return false;  // Cannot determine.
      } else if (isa<ReturnInst>(CurInst)) {
        NextBB = nullptr;
      } else {
        // invoke, unwind, resume, unreachable.
        DEBUG(dbgs() << "Can not handle terminator.");
        return false;  // Cannot handle this terminator.
      }

      // We succeeded at evaluating this block!
      DEBUG(dbgs() << "Successfully evaluated block.\n");
      return true;
    } else {
      // Did not know how to evaluate this!
      DEBUG(dbgs() << "Failed to evaluate block due to unhandled instruction."
            "\n");
      return false;
    }

    if (!CurInst->use_empty()) {
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(InstResult))
        InstResult = ConstantFoldConstantExpression(CE, DL, TLI);

      setVal(CurInst, InstResult);
    }

    // If we just processed an invoke, we finished evaluating the block.
    if (InvokeInst *II = dyn_cast<InvokeInst>(CurInst)) {
      NextBB = II->getNormalDest();
      DEBUG(dbgs() << "Found an invoke instruction. Finished Block.\n\n");
      return true;
    }

    // Advance program counter.
    ++CurInst;
  }
}

/// EvaluateFunction - Evaluate a call to function F, returning true if
/// successful, false if we can't evaluate it.  ActualArgs contains the formal
/// arguments for the function.
bool Evaluator::EvaluateFunction(Function *F, Constant *&RetVal,
                                 const SmallVectorImpl<Constant*> &ActualArgs) {
  // Check to see if this function is already executing (recursion).  If so,
  // bail out.  TODO: we might want to accept limited recursion.
  if (std::find(CallStack.begin(), CallStack.end(), F) != CallStack.end())
    return false;

  CallStack.push_back(F);

  // Initialize arguments to the incoming values specified.
  unsigned ArgNo = 0;
  for (Function::arg_iterator AI = F->arg_begin(), E = F->arg_end(); AI != E;
       ++AI, ++ArgNo)
    setVal(AI, ActualArgs[ArgNo]);

  // ExecutedBlocks - We only handle non-looping, non-recursive code.  As such,
  // we can only evaluate any one basic block at most once.  This set keeps
  // track of what we have executed so we can detect recursive cases etc.
  SmallPtrSet<BasicBlock*, 32> ExecutedBlocks;

  // CurBB - The current basic block we're evaluating.
  BasicBlock *CurBB = F->begin();

  BasicBlock::iterator CurInst = CurBB->begin();

  while (1) {
    BasicBlock *NextBB = nullptr; // Initialized to avoid compiler warnings.
    DEBUG(dbgs() << "Trying to evaluate BB: " << *CurBB << "\n");

    if (!EvaluateBlock(CurInst, NextBB))
      return false;

    if (!NextBB) {
      // Successfully running until there's no next block means that we found
      // the return.  Fill it the return value and pop the call stack.
      ReturnInst *RI = cast<ReturnInst>(CurBB->getTerminator());
      if (RI->getNumOperands())
        RetVal = getVal(RI->getOperand(0));
      CallStack.pop_back();
      return true;
    }

    // Okay, we succeeded in evaluating this control flow.  See if we have
    // executed the new block before.  If so, we have a looping function,
    // which we cannot evaluate in reasonable time.
    if (!ExecutedBlocks.insert(NextBB).second)
      return false;  // looped!

    // Okay, we have never been in this block before.  Check to see if there
    // are any PHI nodes.  If so, evaluate them with information about where
    // we came from.
    PHINode *PN = nullptr;
    for (CurInst = NextBB->begin();
         (PN = dyn_cast<PHINode>(CurInst)); ++CurInst)
      setVal(PN, getVal(PN->getIncomingValueForBlock(CurBB)));

    // Advance to the next block.
    CurBB = NextBB;
  }
}

/// EvaluateStaticConstructor - Evaluate static constructors in the function, if
/// we can.  Return true if we can, false otherwise.
static bool EvaluateStaticConstructor(Function *F, const DataLayout &DL,
                                      const TargetLibraryInfo *TLI) {
  // Call the function.
  Evaluator Eval(DL, TLI);
  Constant *RetValDummy;
  bool EvalSuccess = Eval.EvaluateFunction(F, RetValDummy,
                                           SmallVector<Constant*, 0>());

  if (EvalSuccess) {
    ++NumCtorsEvaluated;

    // We succeeded at evaluation: commit the result.
    DEBUG(dbgs() << "FULLY EVALUATED GLOBAL CTOR FUNCTION '"
          << F->getName() << "' to " << Eval.getMutatedMemory().size()
          << " stores.\n");
    for (DenseMap<Constant*, Constant*>::const_iterator I =
           Eval.getMutatedMemory().begin(), E = Eval.getMutatedMemory().end();
         I != E; ++I)
      CommitValueTo(I->second, I->first);
    for (GlobalVariable *GV : Eval.getInvariants())
      GV->setConstant(true);
  }

  return EvalSuccess;
}

// HLSL Change: changed calling convention to __cdecl
static int __cdecl compareNames(Constant *const *A, Constant *const *B) {
  return (*A)->getName().compare((*B)->getName());
}

static void setUsedInitializer(GlobalVariable &V,
                               const SmallPtrSet<GlobalValue *, 8> &Init) {
  if (Init.empty()) {
    V.eraseFromParent();
    return;
  }

  // Type of pointer to the array of pointers.
  PointerType *Int8PtrTy = Type::getInt8PtrTy(V.getContext(), 0);

  SmallVector<llvm::Constant *, 8> UsedArray;
  for (GlobalValue *GV : Init) {
    Constant *Cast
      = ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, Int8PtrTy);
    UsedArray.push_back(Cast);
  }
  // Sort to get deterministic order.
  array_pod_sort(UsedArray.begin(), UsedArray.end(), compareNames);
  ArrayType *ATy = ArrayType::get(Int8PtrTy, UsedArray.size());

  Module *M = V.getParent();
  V.removeFromParent();
  GlobalVariable *NV =
      new GlobalVariable(*M, ATy, false, llvm::GlobalValue::AppendingLinkage,
                         llvm::ConstantArray::get(ATy, UsedArray), "");
  NV->takeName(&V);
  NV->setSection("llvm.metadata");
  delete &V;
}

namespace {
/// \brief An easy to access representation of llvm.used and llvm.compiler.used.
class LLVMUsed {
  SmallPtrSet<GlobalValue *, 8> Used;
  SmallPtrSet<GlobalValue *, 8> CompilerUsed;
  GlobalVariable *UsedV;
  GlobalVariable *CompilerUsedV;

public:
  LLVMUsed(Module &M) {
    UsedV = collectUsedGlobalVariables(M, Used, false);
    CompilerUsedV = collectUsedGlobalVariables(M, CompilerUsed, true);
  }
  typedef SmallPtrSet<GlobalValue *, 8>::iterator iterator;
  typedef iterator_range<iterator> used_iterator_range;
  iterator usedBegin() { return Used.begin(); }
  iterator usedEnd() { return Used.end(); }
  used_iterator_range used() {
    return used_iterator_range(usedBegin(), usedEnd());
  }
  iterator compilerUsedBegin() { return CompilerUsed.begin(); }
  iterator compilerUsedEnd() { return CompilerUsed.end(); }
  used_iterator_range compilerUsed() {
    return used_iterator_range(compilerUsedBegin(), compilerUsedEnd());
  }
  bool usedCount(GlobalValue *GV) const { return Used.count(GV); }
  bool compilerUsedCount(GlobalValue *GV) const {
    return CompilerUsed.count(GV);
  }
  bool usedErase(GlobalValue *GV) { return Used.erase(GV); }
  bool compilerUsedErase(GlobalValue *GV) { return CompilerUsed.erase(GV); }
  bool usedInsert(GlobalValue *GV) { return Used.insert(GV).second; }
  bool compilerUsedInsert(GlobalValue *GV) {
    return CompilerUsed.insert(GV).second;
  }

  void syncVariablesAndSets() {
    if (UsedV)
      setUsedInitializer(*UsedV, Used);
    if (CompilerUsedV)
      setUsedInitializer(*CompilerUsedV, CompilerUsed);
  }
};
}

static bool hasUseOtherThanLLVMUsed(GlobalAlias &GA, const LLVMUsed &U) {
  if (GA.use_empty()) // No use at all.
    return false;

  assert((!U.usedCount(&GA) || !U.compilerUsedCount(&GA)) &&
         "We should have removed the duplicated "
         "element from llvm.compiler.used");
  if (!GA.hasOneUse())
    // Strictly more than one use. So at least one is not in llvm.used and
    // llvm.compiler.used.
    return true;

  // Exactly one use. Check if it is in llvm.used or llvm.compiler.used.
  return !U.usedCount(&GA) && !U.compilerUsedCount(&GA);
}

static bool hasMoreThanOneUseOtherThanLLVMUsed(GlobalValue &V,
                                               const LLVMUsed &U) {
  unsigned N = 2;
  assert((!U.usedCount(&V) || !U.compilerUsedCount(&V)) &&
         "We should have removed the duplicated "
         "element from llvm.compiler.used");
  if (U.usedCount(&V) || U.compilerUsedCount(&V))
    ++N;
  return V.hasNUsesOrMore(N);
}

static bool mayHaveOtherReferences(GlobalAlias &GA, const LLVMUsed &U) {
  if (!GA.hasLocalLinkage())
    return true;

  return U.usedCount(&GA) || U.compilerUsedCount(&GA);
}

static bool hasUsesToReplace(GlobalAlias &GA, const LLVMUsed &U,
                             bool &RenameTarget) {
  RenameTarget = false;
  bool Ret = false;
  if (hasUseOtherThanLLVMUsed(GA, U))
    Ret = true;

  // If the alias is externally visible, we may still be able to simplify it.
  if (!mayHaveOtherReferences(GA, U))
    return Ret;

  // If the aliasee has internal linkage, give it the name and linkage
  // of the alias, and delete the alias.  This turns:
  //   define internal ... @f(...)
  //   @a = alias ... @f
  // into:
  //   define ... @a(...)
  Constant *Aliasee = GA.getAliasee();
  GlobalValue *Target = cast<GlobalValue>(Aliasee->stripPointerCasts());
  if (!Target->hasLocalLinkage())
    return Ret;

  // Do not perform the transform if multiple aliases potentially target the
  // aliasee. This check also ensures that it is safe to replace the section
  // and other attributes of the aliasee with those of the alias.
  if (hasMoreThanOneUseOtherThanLLVMUsed(*Target, U))
    return Ret;

  RenameTarget = true;
  return true;
}

bool GlobalOpt::OptimizeGlobalAliases(Module &M) {
  bool Changed = false;
  LLVMUsed Used(M);

  for (GlobalValue *GV : Used.used())
    Used.compilerUsedErase(GV);

  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
       I != E;) {
    Module::alias_iterator J = I++;
    // Aliases without names cannot be referenced outside this module.
    if (!J->hasName() && !J->isDeclaration() && !J->hasLocalLinkage())
      J->setLinkage(GlobalValue::InternalLinkage);
    // If the aliasee may change at link time, nothing can be done - bail out.
    if (J->mayBeOverridden())
      continue;

    Constant *Aliasee = J->getAliasee();
    GlobalValue *Target = dyn_cast<GlobalValue>(Aliasee->stripPointerCasts());
    // We can't trivially replace the alias with the aliasee if the aliasee is
    // non-trivial in some way.
    // TODO: Try to handle non-zero GEPs of local aliasees.
    if (!Target)
      continue;
    Target->removeDeadConstantUsers();

    // Make all users of the alias use the aliasee instead.
    bool RenameTarget;
    if (!hasUsesToReplace(*J, Used, RenameTarget))
      continue;

    J->replaceAllUsesWith(ConstantExpr::getBitCast(Aliasee, J->getType()));
    ++NumAliasesResolved;
    Changed = true;

    if (RenameTarget) {
      // Give the aliasee the name, linkage and other attributes of the alias.
      Target->takeName(J);
      Target->setLinkage(J->getLinkage());
      Target->setVisibility(J->getVisibility());
      Target->setDLLStorageClass(J->getDLLStorageClass());

      if (Used.usedErase(J))
        Used.usedInsert(Target);

      if (Used.compilerUsedErase(J))
        Used.compilerUsedInsert(Target);
    } else if (mayHaveOtherReferences(*J, Used))
      continue;

    // Delete the alias.
    M.getAliasList().erase(J);
    ++NumAliasesRemoved;
    Changed = true;
  }

  Used.syncVariablesAndSets();

  return Changed;
}

static Function *FindCXAAtExit(Module &M, TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::cxa_atexit))
    return nullptr;

  Function *Fn = M.getFunction(TLI->getName(LibFunc::cxa_atexit));

  if (!Fn)
    return nullptr;

  FunctionType *FTy = Fn->getFunctionType();

  // Checking that the function has the right return type, the right number of
  // parameters and that they all have pointer types should be enough.
  if (!FTy->getReturnType()->isIntegerTy() ||
      FTy->getNumParams() != 3 ||
      !FTy->getParamType(0)->isPointerTy() ||
      !FTy->getParamType(1)->isPointerTy() ||
      !FTy->getParamType(2)->isPointerTy())
    return nullptr;

  return Fn;
}

/// cxxDtorIsEmpty - Returns whether the given function is an empty C++
/// destructor and can therefore be eliminated.
/// Note that we assume that other optimization passes have already simplified
/// the code so we only look for a function with a single basic block, where
/// the only allowed instructions are 'ret', 'call' to an empty C++ dtor and
/// other side-effect free instructions.
static bool cxxDtorIsEmpty(const Function &Fn,
                           SmallPtrSet<const Function *, 8> &CalledFunctions) {
  // FIXME: We could eliminate C++ destructors if they're readonly/readnone and
  // nounwind, but that doesn't seem worth doing.
  if (Fn.isDeclaration())
    return false;

  if (++Fn.begin() != Fn.end())
    return false;

  const BasicBlock &EntryBlock = Fn.getEntryBlock();
  for (BasicBlock::const_iterator I = EntryBlock.begin(), E = EntryBlock.end();
       I != E; ++I) {
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      // Ignore debug intrinsics.
      if (isa<DbgInfoIntrinsic>(CI))
        continue;

      const Function *CalledFn = CI->getCalledFunction();

      if (!CalledFn)
        return false;

      SmallPtrSet<const Function *, 8> NewCalledFunctions(CalledFunctions);

      // Don't treat recursive functions as empty.
      if (!NewCalledFunctions.insert(CalledFn).second)
        return false;

      if (!cxxDtorIsEmpty(*CalledFn, NewCalledFunctions))
        return false;
    } else if (isa<ReturnInst>(*I))
      return true; // We're done.
    else if (I->mayHaveSideEffects())
      return false; // Destructor with side effects, bail.
  }

  return false;
}

bool GlobalOpt::OptimizeEmptyGlobalCXXDtors(Function *CXAAtExitFn) {
  /// Itanium C++ ABI p3.3.5:
  ///
  ///   After constructing a global (or local static) object, that will require
  ///   destruction on exit, a termination function is registered as follows:
  ///
  ///   extern "C" int __cxa_atexit ( void (*f)(void *), void *p, void *d );
  ///
  ///   This registration, e.g. __cxa_atexit(f,p,d), is intended to cause the
  ///   call f(p) when DSO d is unloaded, before all such termination calls
  ///   registered before this one. It returns zero if registration is
  ///   successful, nonzero on failure.

  // This pass will look for calls to __cxa_atexit where the function is trivial
  // and remove them.
  bool Changed = false;

  for (auto I = CXAAtExitFn->user_begin(), E = CXAAtExitFn->user_end();
       I != E;) {
    // We're only interested in calls. Theoretically, we could handle invoke
    // instructions as well, but neither llvm-gcc nor clang generate invokes
    // to __cxa_atexit.
    CallInst *CI = dyn_cast<CallInst>(*I++);
    if (!CI)
      continue;

    Function *DtorFn =
      dyn_cast<Function>(CI->getArgOperand(0)->stripPointerCasts());
    if (!DtorFn)
      continue;

    SmallPtrSet<const Function *, 8> CalledFunctions;
    if (!cxxDtorIsEmpty(*DtorFn, CalledFunctions))
      continue;

    // Just remove the call.
    CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
    CI->eraseFromParent();

    ++NumCXXDtorsRemoved;

    Changed |= true;
  }

  return Changed;
}

bool GlobalOpt::runOnModule(Module &M) {
  bool Changed = false;

  auto &DL = M.getDataLayout();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    NotDiscardableComdats.clear();
    for (const GlobalVariable &GV : M.globals())
      if (const Comdat *C = GV.getComdat())
        if (!GV.isDiscardableIfUnused() || !GV.use_empty())
          NotDiscardableComdats.insert(C);
    for (Function &F : M)
      if (const Comdat *C = F.getComdat())
        if (!F.isDefTriviallyDead())
          NotDiscardableComdats.insert(C);
    for (GlobalAlias &GA : M.aliases())
      if (const Comdat *C = GA.getComdat())
        if (!GA.isDiscardableIfUnused() || !GA.use_empty())
          NotDiscardableComdats.insert(C);

    // Delete functions that are trivially dead, ccc -> fastcc
    LocalChange |= OptimizeFunctions(M);

    // Optimize global_ctors list.
    LocalChange |= optimizeGlobalCtorsList(M, [&](Function *F) {
      return EvaluateStaticConstructor(F, DL, TLI);
    });

    // Optimize non-address-taken globals.
    LocalChange |= OptimizeGlobalVars(M);

    // Resolve aliases, when possible.
    LocalChange |= OptimizeGlobalAliases(M);

    // Try to remove trivial global destructors if they are not removed
    // already.
    Function *CXAAtExitFn = FindCXAAtExit(M, TLI);
    if (CXAAtExitFn)
      LocalChange |= OptimizeEmptyGlobalCXXDtors(CXAAtExitFn);

    Changed |= LocalChange;
  }

  // TODO: Move all global ctors functions to the end of the module for code
  // layout.

  return Changed;
}
