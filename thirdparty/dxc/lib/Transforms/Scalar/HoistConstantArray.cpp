//===- HoistConstantArray.cpp - Code to perform constant array hoisting ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (C) Microsoft Corporation. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements hoisting of constant local arrays to global arrays.
// The idea is to change the array initialization from function local memory
// using alloca and stores to global constant memory using a global variable
// and constant initializer. We only hoist arrays that have all constant elements.
// The frontend will hoist the arrays if they are declared static, but we can
// hoist any array that is only ever initialized with constant data.
//
// This transformation was developed to work with the dxil produced from the
// hlsl compiler. Hoisting the array to use a constant initializer should allow
// a dxil backend compiler to generate more efficent code than a local array.
// For example, it could use an immediate constant pool to represent the array.
//
// We limit hoisting to those arrays that are initialized by constant values.
// We still hoist if the array is partially initialized as long as no
// non-constant values are written. The uninitialized values will be hoisted 
// as undef values.
//
// Improvements:
// Currently we do not merge arrays that have the same constant values. We
// create the global variables with `unnamed_addr` set which means they
// can be merged with other constants. We should probably use a separate
// pass to merge all the unnamed_addr constants.
//
// Example:
//
// float main(int i : I) : SV_Target{
//   float A[] = { 1, 2, 3 };
//   return A[i];
// }
//
// Without array hoisting, we generate the following dxil
//
// define void @main() {
// entry:
//   %0 = call i32 @dx.op.loadInput.i32(i32 4, i32 0, i32 0, i8 0, i32 undef)
//   %A = alloca[3 x float], align 4
//   %1 = getelementptr inbounds[3 x float], [3 x float] * %A, i32 0, i32 0
//   store float 1.000000e+00, float* %1, align 4
//   %2 = getelementptr inbounds[3 x float], [3 x float] * %A, i32 0, i32 1
//   store float 2.000000e+00, float* %2, align 4
//   %3 = getelementptr inbounds[3 x float], [3 x float] * %A, i32 0, i32 2
//   store float 3.000000e+00, float* %3, align 4
//   %arrayidx = getelementptr inbounds[3 x float], [3 x float] * %A, i32 0, i32 %0
//   %4 = load float, float* %arrayidx, align 4, !tbaa !14
//   call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %4);
//   ret void
// }
//
// With array hoisting enabled we generate this dxil
//
// @A.hca = internal unnamed_addr constant [3 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00]
// define void @main() {
// entry:
//   %0 = call i32 @dx.op.loadInput.i32(i32 4, i32 0, i32 0, i8 0, i32 undef)
//   %arrayidx = getelementptr inbounds[3 x float], [3 x float] * @A.hca, i32 0, i32 %0
//   %1 = load float, float* %arrayidx, align 4, !tbaa !14
//   call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %1)
//   ret void
// }
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Pass.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Analysis/ValueTracking.h"
using namespace llvm;

namespace {
  class CandidateArray;

  //===--------------------------------------------------------------------===//
  // HoistConstantArray pass implementation
  //
  class HoistConstantArray : public ModulePass {
  public:
    static char ID; // Pass identification, replacement for typeid
    HoistConstantArray() : ModulePass(ID) {
      initializeHoistConstantArrayPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
    }
  private:
    bool runOnFunction(Function &F);
    std::vector<AllocaInst *> findCandidateAllocas(Function &F);
    void hoistArray(const CandidateArray &candidate);
    void removeLocalArrayStores(const CandidateArray &candidate);
 };

  // Represents an array we are considering for hoisting.
  // Contains helper routines for analyzing if hoisting is possible
  // and creating the global variable for the hoisted array.
  class CandidateArray {
  public:
    explicit CandidateArray(AllocaInst *);
    bool IsConstArray() const { return m_IsConstArray; }
    void AnalyzeUses();
    GlobalVariable *GetGlobalArray() const;
    AllocaInst *GetLocalArray() const { return m_Alloca; }
    std::vector<StoreInst*> GetArrayStores() const;

  private:
    AllocaInst *m_Alloca;
    ArrayType *m_ArrayType;
    std::vector<Constant *> m_Values;
    bool m_IsConstArray;

    bool AnalyzeStore(StoreInst *SI);
    bool StoreConstant(int64_t index, Constant *value);
    void EnsureSize();
    void GetArrayStores(GEPOperator *gep,
                        std::vector<StoreInst *> &stores) const;
    bool AllArrayUsersAreGEPOrLifetime(std::vector<GEPOperator *> &geps);
    bool AllGEPUsersAreValid(GEPOperator *gep);
    UndefValue *UndefElement();
  };
}

// Returns the ArrayType for the alloca or nullptr if the alloca
// does not allocate an array.
static ArrayType *getAllocaArrayType(AllocaInst *allocaInst) {
  return dyn_cast<ArrayType>(allocaInst->getType()->getPointerElementType());
}

// Check if the instruction is an alloca that we should consider for hoisting.
// The alloca must allocate and array of primitive types.
static AllocaInst *isHoistableArrayAlloca(Instruction *I) {
  AllocaInst *allocaInst = dyn_cast<AllocaInst>(I);
  if (!allocaInst)
    return nullptr;

  ArrayType *arrayTy = getAllocaArrayType(allocaInst);
  if (!arrayTy)
    return nullptr;

  if (!arrayTy->getElementType()->isSingleValueType())
    return nullptr;

  return allocaInst;
}

// ----------------------------------------------------------------------------
// CandidateArray implementation
// ----------------------------------------------------------------------------

// Create the candidate array for the alloca.
CandidateArray::CandidateArray(AllocaInst *AI)
  : m_Alloca(AI)
  , m_Values()
  , m_IsConstArray(false)
{
  assert(isHoistableArrayAlloca(AI));
  m_ArrayType = getAllocaArrayType(AI);
}

// Get the global variable with a constant initializer for the array.
// Only valid to call if the array has been analyzed as a constant array.
GlobalVariable *CandidateArray::GetGlobalArray() const {
  assert(IsConstArray());
  Constant *initializer = ConstantArray::get(m_ArrayType, m_Values);
  Module *M = m_Alloca->getModule();
  GlobalVariable *GV = new GlobalVariable(*M, m_ArrayType, true, GlobalVariable::LinkageTypes::InternalLinkage, initializer, Twine(m_Alloca->getName()) + ".hca");
  GV->setUnnamedAddr(true);
  return GV;
}

// Get a list of all the stores that write to the array through one or more
// GetElementPtrInst operations.
std::vector<StoreInst *> CandidateArray::GetArrayStores() const {
  std::vector<StoreInst *> stores;
  for (User *U : m_Alloca->users())
    if (GEPOperator *gep = dyn_cast<GEPOperator>(U))
      GetArrayStores(gep, stores);
  return stores;
}

// Recursively collect all the stores that write to the pointer/buffer
// referred to by this GetElementPtrInst.
void CandidateArray::GetArrayStores(GEPOperator *gep,
                                    std::vector<StoreInst *> &stores) const {
  for (User *GU : gep->users()) {
    if (StoreInst *SI = dyn_cast<StoreInst>(GU)) {
      stores.push_back(SI);
    }
    else if (GEPOperator *GEPI = dyn_cast<GEPOperator>(GU)) {
      GetArrayStores(GEPI, stores);
    }
  }
}
// Check to see that all the users of the array are GEPs or lifetime intrinsics.
// If so, populate the `geps` vector with a list of all geps that use the array.
bool CandidateArray::AllArrayUsersAreGEPOrLifetime(std::vector<GEPOperator *> &geps) {
  for (User *U : m_Alloca->users()) {
    // Allow users that are only used by lifetime intrinsics.
    if (isa<BitCastInst>(U) && onlyUsedByLifetimeMarkers(U))
      continue;

    GEPOperator *gep = dyn_cast<GEPOperator>(U);
    if (!gep)
      return false;

    geps.push_back(gep);
  }

  return true;
}

// Check that all gep uses are valid.
// A valid use is either
//  1. A store of a constant value that does not overwrite an existing constant
//     with a different value.
//  2. A load instruction.
//  3. Another GetElementPtrInst that itself only has valid uses (recursively)
// Any other use is considered invalid.
bool CandidateArray::AllGEPUsersAreValid(GEPOperator *gep) {
  for (User *U : gep->users()) {
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (!AnalyzeStore(SI))
        return false;
    }
    else if (GEPOperator *recursive_gep = dyn_cast<GEPOperator>(U)) {
      if (!AllGEPUsersAreValid(recursive_gep))
        return false;
    }
    else if (!isa<LoadInst>(U)) {
      return false;
    }
  }

  return true;
}

// Analyze all uses of the array to see if it qualifes as a constant array.
// We check the following conditions:
//  1. Make sure alloca is only used by GEP and lifetime intrinsics.
//  2. Make sure GEP is only used in load/store.
//  3. Make sure all stores have constant indicies.
//  4. Make sure all stores are constants.
//  5. Make sure all stores to same location are the same constant.
void CandidateArray::AnalyzeUses() {
  m_IsConstArray = false;
  std::vector<GEPOperator *> geps;
  if (!AllArrayUsersAreGEPOrLifetime(geps))
    return;

  for (GEPOperator *gep : geps)
    if (!AllGEPUsersAreValid(gep))
      return;

  m_IsConstArray = true;
}

// Analyze a store to see if it is a valid constant store.
// A valid store will write a constant value to a known (constant) location.
bool CandidateArray::AnalyzeStore(StoreInst *SI) {
  if (!isa<Constant>(SI->getValueOperand()))
    return false;
  // Walk up the ladder of GetElementPtr instructions to accumulate the index
  int64_t index = 0;
  for (auto iter = SI->getPointerOperand(); iter != m_Alloca;) {
    GEPOperator *gep = cast<GEPOperator>(iter);
    if (!gep->hasAllConstantIndices())
      return false;

    // Deal with the 'extra 0' index from what might have been a global pointer
    // https://www.llvm.org/docs/GetElementPtr.html#why-is-the-extra-0-index-required
    if ((gep->getNumIndices() == 2) && (gep->getPointerOperand() == m_Alloca)) {
      // Non-zero offset is unexpected, but could occur in the wild. Bail out if
      // we see it.
      ConstantInt *ptrOffset = cast<ConstantInt>(gep->getOperand(1));
      if (!ptrOffset->isZero())
        return false;
    }
    else if (gep->getNumIndices() != 1) {
      return false;
    }

    // Accumulate the index
    ConstantInt *c = cast<ConstantInt>(gep->getOperand(gep->getNumIndices()));
    index += c->getSExtValue();

    iter = gep->getPointerOperand();
  }

  return StoreConstant(index, cast<Constant>(SI->getValueOperand()));
}

// Check if the store is valid and record the value if so.
// A valid constant store is either:
//  1. A store of a new constant
//  2. A store of the same constant to the same location
bool CandidateArray::StoreConstant(int64_t index, Constant *value) {
  EnsureSize();
  size_t i = static_cast<size_t>(index);
  if (i >= m_Values.size())
    return false;
  if (m_Values[i] == UndefElement())
    m_Values[i] = value;

  return m_Values[i] == value;
}

// We lazily create the values array until we have a store of a
// constant that we need to remember. This avoids memory overhead
// for obviously non-constant arrays.
void CandidateArray::EnsureSize() {
  if (m_Values.size() == 0) {
    m_Values.resize(m_ArrayType->getNumElements(), UndefElement());
  }
  assert(m_Values.size() == m_ArrayType->getNumElements());
}

// Get an undef value of the correct type for the array.
UndefValue *CandidateArray::UndefElement() {
  return UndefValue::get(m_ArrayType->getElementType());
}


// ----------------------------------------------------------------------------
// Pass Implementation
// ----------------------------------------------------------------------------

// Find the allocas that are candidates for array hoisting in the function.
std::vector<AllocaInst*> HoistConstantArray::findCandidateAllocas(Function &F) {
  std::vector<AllocaInst*> candidates;
  for (Instruction &I : F.getEntryBlock())
    if (AllocaInst *allocaInst = isHoistableArrayAlloca(&I))
        candidates.push_back(allocaInst);

  return candidates;
}

// Remove local stores to the array.
// We remove them explicitly rather than relying on DCE to find they are dead.
// Other uses (e.g. geps) can be easily cleaned up by DCE.
void HoistConstantArray::removeLocalArrayStores(const CandidateArray &candidate) {
  std::vector<StoreInst*> stores = candidate.GetArrayStores();
  for (StoreInst *store : stores)
    store->eraseFromParent();
}

// Hoist an array from a local to a global.
void HoistConstantArray::hoistArray(const CandidateArray &candidate) {
  assert(candidate.IsConstArray());

  removeLocalArrayStores(candidate);
  AllocaInst *local = candidate.GetLocalArray();
  GlobalVariable *global = candidate.GetGlobalArray();
  local->replaceAllUsesWith(global);
  local->eraseFromParent();
}

// Perform array hoisting on a single function.
bool HoistConstantArray::runOnFunction(Function &F) {
  bool changed = false;
  std::vector<AllocaInst *> candidateAllocas = findCandidateAllocas(F);

  for (AllocaInst *AI : candidateAllocas) {
    CandidateArray candidate(AI);
    candidate.AnalyzeUses();
    if (candidate.IsConstArray()) {
      hoistArray(candidate);
      changed |= true;
    }
  }

  return changed;
}

char HoistConstantArray::ID = 0;
INITIALIZE_PASS(HoistConstantArray, "hlsl-hca", "Hoist constant arrays", false, false)

bool HoistConstantArray::runOnModule(Module &M) {
  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    changed |= runOnFunction(F);
  }

  return changed;
}

ModulePass *llvm::createHoistConstantArrayPass() {
  return new HoistConstantArray();
}

