//===- LowerTypePasses.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HlslIntrinsicOp.h"
#include "llvm/Pass.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/ValueTracking.h"
#include <vector>

using namespace llvm;
using namespace hlsl;

static ArrayType* CreateNestArrayTy(Type* FinalEltTy,
  ArrayRef<ArrayType*> nestArrayTys) {
  Type* newAT = FinalEltTy;
  for (auto ArrayTy = nestArrayTys.rbegin(), E = nestArrayTys.rend(); ArrayTy != E;
    ++ArrayTy)
    newAT = ArrayType::get(newAT, (*ArrayTy)->getNumElements());
  return cast<ArrayType>(newAT);
}

//===----------------------------------------------------------------------===//
// Lower one type to another type.
//===----------------------------------------------------------------------===//
namespace {
class LowerTypePass : public ModulePass {
public:
  explicit LowerTypePass(char &ID)
      : ModulePass(ID) {}

  bool runOnModule(Module &M) override;
private:
  bool runOnFunction(Function &F, bool HasDbgInfo);
  AllocaInst *lowerAlloca(AllocaInst *A);
  GlobalVariable *lowerInternalGlobal(GlobalVariable *GV);
protected:
  virtual bool needToLower(Value *V) = 0;
  virtual void lowerUseWithNewValue(Value *V, Value *NewV) = 0;
  virtual Type *lowerType(Type *Ty) = 0;
  virtual Constant *lowerInitVal(Constant *InitVal, Type *NewTy) = 0;
  virtual StringRef getGlobalPrefix() = 0;
  virtual void initialize(Module &M) {};
};

AllocaInst *LowerTypePass::lowerAlloca(AllocaInst *A) {
  IRBuilder<> AllocaBuilder(A);
  Type *NewTy = lowerType(A->getAllocatedType());
  AllocaInst *NewA = AllocaBuilder.CreateAlloca(NewTy);
  NewA->setAlignment(A->getAlignment());
  return NewA;
}

GlobalVariable *LowerTypePass::lowerInternalGlobal(GlobalVariable *GV) {
  Type *NewTy = lowerType(GV->getType()->getPointerElementType());
  // So set init val to undef.
  Constant *InitVal = UndefValue::get(NewTy);
  if (GV->hasInitializer()) {
    Constant *OldInitVal = GV->getInitializer();
    if (isa<ConstantAggregateZero>(OldInitVal))
      InitVal = ConstantAggregateZero::get(NewTy);
    else if (!isa<UndefValue>(OldInitVal)) {
      InitVal = lowerInitVal(OldInitVal, NewTy);
    }
  }

  bool isConst = GV->isConstant();
  GlobalVariable::ThreadLocalMode TLMode = GV->getThreadLocalMode();
  unsigned AddressSpace = GV->getType()->getAddressSpace();
  GlobalValue::LinkageTypes linkage = GV->getLinkage();

  Module *M = GV->getParent();
  GlobalVariable *NewGV = new llvm::GlobalVariable(
      *M, NewTy, /*IsConstant*/ isConst, linkage,
      /*InitVal*/ InitVal, GV->getName() + getGlobalPrefix(),
      /*InsertBefore*/ nullptr, TLMode, AddressSpace);
  NewGV->setAlignment(GV->getAlignment());
  return NewGV;
}

bool LowerTypePass::runOnFunction(Function &F, bool HasDbgInfo) {
  std::vector<AllocaInst *> workList;
  // Scan the entry basic block, adding allocas to the worklist.
  BasicBlock &BB = F.getEntryBlock();
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (!isa<AllocaInst>(I))
      continue;
    AllocaInst *A = cast<AllocaInst>(I);
    if (needToLower(A))
      workList.emplace_back(A);
  }
  LLVMContext &Context = F.getContext();
  for (AllocaInst *A : workList) {
    AllocaInst *NewA = lowerAlloca(A);
    if (HasDbgInfo) {
      // Migrate debug info.
      DbgDeclareInst *DDI = llvm::FindAllocaDbgDeclare(A);
      if (DDI) DDI->setOperand(0, MetadataAsValue::get(Context, LocalAsMetadata::get(NewA)));
    }
    // Replace users.
    lowerUseWithNewValue(A, NewA);
    // Remove alloca.
    A->eraseFromParent();
  }
  return true;
}

bool LowerTypePass::runOnModule(Module &M) {
  initialize(M);
  // Load up debug information, to cross-reference values and the instructions
  // used to load them.
  bool HasDbgInfo = llvm::hasDebugInfo(M);
  llvm::DebugInfoFinder Finder;
  if (HasDbgInfo) {
    Finder.processModule(M);
  }

  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    runOnFunction(F, HasDbgInfo);
  }

  // Work on internal global.
  std::vector<GlobalVariable *> vecGVs;
  for (GlobalVariable &GV : M.globals()) {
    if (dxilutil::IsStaticGlobal(&GV) || dxilutil::IsSharedMemoryGlobal(&GV)) {
      if (needToLower(&GV) && !GV.user_empty())
        vecGVs.emplace_back(&GV);
    }
  }

  for (GlobalVariable *GV : vecGVs) {
    GlobalVariable *NewGV = lowerInternalGlobal(GV);
    // Add debug info.
    if (HasDbgInfo) {
      HLModule::UpdateGlobalVariableDebugInfo(GV, Finder, NewGV);
    }
    // Replace users.
    lowerUseWithNewValue(GV, NewGV);
    // Remove GV.
    GV->removeDeadConstantUsers();
    GV->eraseFromParent();
  }

  return true;
}

}


//===----------------------------------------------------------------------===//
// DynamicIndexingVector to Array.
//===----------------------------------------------------------------------===//

namespace {
class DynamicIndexingVectorToArray : public LowerTypePass {
  bool ReplaceAllVectors;
public:
  explicit DynamicIndexingVectorToArray(bool ReplaceAll = false)
      : LowerTypePass(ID), ReplaceAllVectors(ReplaceAll) {}
  static char ID; // Pass identification, replacement for typeid
  void applyOptions(PassOptions O) override;
  void dumpConfig(raw_ostream &OS) override;
protected:
  bool needToLower(Value *V) override;
  void lowerUseWithNewValue(Value *V, Value *NewV) override;
  Type *lowerType(Type *Ty) override;
  Constant *lowerInitVal(Constant *InitVal, Type *NewTy) override;
  StringRef getGlobalPrefix() override { return ".v"; }

private:
  bool HasVectorDynamicIndexing(Value *V);
  void ReplaceVecGEP(Value *GEP, ArrayRef<Value *> idxList, Value *A,
                     IRBuilder<> &Builder);
  void ReplaceVecArrayGEP(Value *GEP, ArrayRef<Value *> idxList, Value *A,
                          IRBuilder<> &Builder);
  void ReplaceVectorWithArray(Value *Vec, Value *Array);
  void ReplaceVectorArrayWithArray(Value *VecArray, Value *Array);
  void ReplaceStaticIndexingOnVector(Value *V);
  void ReplaceAddrSpaceCast(ConstantExpr *CE,
                            Value *A, IRBuilder<> &Builder);
};

void DynamicIndexingVectorToArray::applyOptions(PassOptions O) {
  GetPassOptionBool(O, "ReplaceAllVectors", &ReplaceAllVectors,
                    ReplaceAllVectors);
}
void DynamicIndexingVectorToArray::dumpConfig(raw_ostream &OS) {
  ModulePass::dumpConfig(OS);
  OS << ",ReplaceAllVectors=" << ReplaceAllVectors;
}

void DynamicIndexingVectorToArray::ReplaceStaticIndexingOnVector(Value *V) {
  for (auto U = V->user_begin(), E = V->user_end(); U != E;) {
    Value *User = *(U++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Only work on element access for vector.
      if (GEP->getNumOperands() == 3) {
        auto Idx = GEP->idx_begin();
        // Skip the pointer idx.
        Idx++;
        ConstantInt *constIdx = cast<ConstantInt>(Idx);
        // AllocaInst for Call user.
        AllocaInst *TmpAI = nullptr;
        for (auto GEPU = GEP->user_begin(), GEPE = GEP->user_end();
             GEPU != GEPE;) {
          Instruction *GEPUser = cast<Instruction>(*(GEPU++));

          IRBuilder<> Builder(GEPUser);

          if (LoadInst *ldInst = dyn_cast<LoadInst>(GEPUser)) {
            // Change
            //    ld a->x
            // into
            //    b = ld a
            //    b.x
            Value *ldVal = Builder.CreateLoad(V);
            Value *Elt = Builder.CreateExtractElement(ldVal, constIdx);
            ldInst->replaceAllUsesWith(Elt);
            ldInst->eraseFromParent();
          } else if (CallInst *CI = dyn_cast<CallInst>(GEPUser)) {
            // Change
            //    call a->x
            // into
            //   tmp = alloca
            //   b = ld a
            //   st b.x, tmp
            //   call tmp
            //   b = ld a
            //   b.x = ld tmp
            //   st b, a
            if (TmpAI == nullptr) {
              Type *Ty = GEP->getType()->getPointerElementType();
              IRBuilder<> AllocaB(CI->getParent()
                                      ->getParent()
                                      ->getEntryBlock()
                                      .getFirstInsertionPt());
              TmpAI = AllocaB.CreateAlloca(Ty);
            }
            Value *ldVal = Builder.CreateLoad(V);
            Value *Elt = Builder.CreateExtractElement(ldVal, constIdx);
            Builder.CreateStore(Elt, TmpAI);

            CI->replaceUsesOfWith(GEP, TmpAI);

            Builder.SetInsertPoint(CI->getNextNode());
            Elt = Builder.CreateLoad(TmpAI);

            ldVal = Builder.CreateLoad(V);
            ldVal = Builder.CreateInsertElement(ldVal, Elt, constIdx);
            Builder.CreateStore(ldVal, V);
          } else {
            // Change
            //    st val, a->x
            // into
            //    tmp = ld a
            //    tmp.x = val
            //    st tmp, a
            // Must be store inst here.
            StoreInst *stInst = cast<StoreInst>(GEPUser);
            Value *val = stInst->getValueOperand();
            Value *ldVal = Builder.CreateLoad(V);
            ldVal = Builder.CreateInsertElement(ldVal, val, constIdx);
            Builder.CreateStore(ldVal, V);
            stInst->eraseFromParent();
          }
        }
        GEP->eraseFromParent();
      } else if (GEP->getNumIndices() == 1) {
        Value *Idx = *GEP->idx_begin();
        if (ConstantInt *C = dyn_cast<ConstantInt>(Idx)) {
          if (C->getLimitedValue() == 0) {
            GEP->replaceAllUsesWith(V);
            GEP->eraseFromParent();
          }
        }
      }
    }
  }
}

bool DynamicIndexingVectorToArray::needToLower(Value *V) {
  Type *Ty = V->getType()->getPointerElementType();
  if (dyn_cast<VectorType>(Ty)) {
    if (isa<GlobalVariable>(V) || ReplaceAllVectors) {
      return true;
    }
    // Don't lower local vector which only static indexing.
    if (HasVectorDynamicIndexing(V)) {
      return true;
    } else {
      // Change vector indexing with ld st.
      ReplaceStaticIndexingOnVector(V);
      return false;
    }
  } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    // Array must be replaced even without dynamic indexing to remove vector
    // type in dxil.
    // TODO: optimize static array index in later pass.
    Type *EltTy = dxilutil::GetArrayEltTy(AT);
    return isa<VectorType>(EltTy);
  }
  return false;
}

void DynamicIndexingVectorToArray::ReplaceVecGEP(Value *GEP, ArrayRef<Value *> idxList,
                                       Value *A, IRBuilder<> &Builder) {
  Value *newGEP = Builder.CreateGEP(A, idxList);
  if (GEP->getType()->getPointerElementType()->isVectorTy()) {
    ReplaceVectorWithArray(GEP, newGEP);
  } else {
    GEP->replaceAllUsesWith(newGEP);
  }
}

void DynamicIndexingVectorToArray::ReplaceAddrSpaceCast(ConstantExpr *CE,
                                              Value *A, IRBuilder<> &Builder) {
  // create new AddrSpaceCast.
  Value *NewAddrSpaceCast = Builder.CreateAddrSpaceCast(
    A,
    PointerType::get(A->getType()->getPointerElementType(),
                      CE->getType()->getPointerAddressSpace()));
  ReplaceVectorWithArray(CE, NewAddrSpaceCast);
}

void DynamicIndexingVectorToArray::ReplaceVectorWithArray(Value *Vec, Value *A) {
  unsigned size = Vec->getType()->getPointerElementType()->getVectorNumElements();
  for (auto U = Vec->user_begin(); U != Vec->user_end();) {
    User *User = (*U++);

    // GlobalVariable user.
    if (ConstantExpr * CE = dyn_cast<ConstantExpr>(User)) {
      if (User->user_empty())
        continue;
      if (GEPOperator *GEP = dyn_cast<GEPOperator>(User)) {
        IRBuilder<> Builder(Vec->getContext());
        SmallVector<Value *, 4> idxList(GEP->idx_begin(), GEP->idx_end());
        ReplaceVecGEP(GEP, idxList, A, Builder);
        continue;
      } else if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        IRBuilder<> Builder(Vec->getContext());
        ReplaceAddrSpaceCast(CE, A, Builder);
        continue;
      }
      DXASSERT(0, "not implemented yet");
    }
    // Instrution user.
    Instruction *UserInst = cast<Instruction>(User);
    IRBuilder<> Builder(UserInst);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      SmallVector<Value *, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      ReplaceVecGEP(cast<GEPOperator>(GEP), idxList, A, Builder);
      GEP->eraseFromParent();
    } else if (LoadInst *ldInst = dyn_cast<LoadInst>(User)) {
      // If ld whole struct, need to split the load.
      Value *newLd = UndefValue::get(ldInst->getType());
      Value *zero = Builder.getInt32(0);
      unsigned align = ldInst->getAlignment();
      for (unsigned i = 0; i < size; i++) {
        Value *idx = Builder.getInt32(i);
        Value *GEP = Builder.CreateInBoundsGEP(A, {zero, idx});
        LoadInst *Elt = Builder.CreateLoad(GEP);
        Elt->setAlignment(align);
        newLd = Builder.CreateInsertElement(newLd, Elt, i);
      }
      ldInst->replaceAllUsesWith(newLd);
      ldInst->eraseFromParent();
    } else if (StoreInst *stInst = dyn_cast<StoreInst>(User)) {
      Value *val = stInst->getValueOperand();
      Value *zero = Builder.getInt32(0);
      unsigned align = stInst->getAlignment();
      for (unsigned i = 0; i < size; i++) {
        Value *Elt = Builder.CreateExtractElement(val, i);
        Value *idx = Builder.getInt32(i);
        Value *GEP = Builder.CreateInBoundsGEP(A, {zero, idx});
        StoreInst *EltSt = Builder.CreateStore(Elt, GEP);
        EltSt->setAlignment(align);
      }
      stInst->eraseFromParent();
    } else if (BitCastInst *castInst = dyn_cast<BitCastInst>(User)) {
      DXASSERT(onlyUsedByLifetimeMarkers(castInst),
               "expected bitcast to only be used by lifetime intrinsics");
      castInst->setOperand(0, A);
    } else {
      // Vector parameter should be lowered.
      // No function call should use vector.
      DXASSERT(0, "not implement yet");
    }
  }
}

void DynamicIndexingVectorToArray::ReplaceVecArrayGEP(Value *GEP,
                                            ArrayRef<Value *> idxList, Value *A,
                                            IRBuilder<> &Builder) {
  Value *newGEP = Builder.CreateGEP(A, idxList);
  Type *Ty = GEP->getType()->getPointerElementType();
  if (Ty->isVectorTy()) {
    ReplaceVectorWithArray(GEP, newGEP);
  } else if (Ty->isArrayTy()) {
    ReplaceVectorArrayWithArray(GEP, newGEP);
  } else {
    DXASSERT(Ty->isSingleValueType(), "must be vector subscript here");
    GEP->replaceAllUsesWith(newGEP);
  }
}

void DynamicIndexingVectorToArray::ReplaceVectorArrayWithArray(Value *VA, Value *A) {
  for (auto U = VA->user_begin(); U != VA->user_end();) {
    User *User = *(U++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      IRBuilder<> Builder(GEP);
      SmallVector<Value *, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      ReplaceVecArrayGEP(GEP, idxList, A, Builder);
      GEP->eraseFromParent();
    } else if (GEPOperator *GEPOp = dyn_cast<GEPOperator>(User)) {
      IRBuilder<> Builder(GEPOp->getContext());
      SmallVector<Value *, 4> idxList(GEPOp->idx_begin(), GEPOp->idx_end());
      ReplaceVecArrayGEP(GEPOp, idxList, A, Builder);
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(User)) {
      BCI->setOperand(0, A);
    } else {
      DXASSERT(0, "Array pointer should only used by GEP");
    }
  }
}

void DynamicIndexingVectorToArray::lowerUseWithNewValue(Value *V, Value *NewV) {
  Type *Ty = V->getType()->getPointerElementType();
  // Replace V with NewV.
  if (Ty->isVectorTy()) {
    ReplaceVectorWithArray(V, NewV);
  } else {
    ReplaceVectorArrayWithArray(V, NewV);
  }
}

Type *DynamicIndexingVectorToArray::lowerType(Type *Ty) {
  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    return ArrayType::get(VT->getElementType(), VT->getNumElements());
  } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    SmallVector<ArrayType *, 4> nestArrayTys;
    nestArrayTys.emplace_back(AT);

    Type *EltTy = AT->getElementType();
    // support multi level of array
    while (EltTy->isArrayTy()) {
      ArrayType *ElAT = cast<ArrayType>(EltTy);
      nestArrayTys.emplace_back(ElAT);
      EltTy = ElAT->getElementType();
    }
    if (EltTy->isVectorTy()) {
      Type *vecAT = ArrayType::get(EltTy->getVectorElementType(),
                                   EltTy->getVectorNumElements());
      return CreateNestArrayTy(vecAT, nestArrayTys);
    }
    return nullptr;
  }
  return nullptr;
}

Constant *DynamicIndexingVectorToArray::lowerInitVal(Constant *InitVal, Type *NewTy) {
  Type *VecTy = InitVal->getType();
  ArrayType *ArrayTy = cast<ArrayType>(NewTy);
  if (VecTy->isVectorTy()) {
    SmallVector<Constant *, 4> Elts;
    for (unsigned i = 0; i < VecTy->getVectorNumElements(); i++) {
      Elts.emplace_back(InitVal->getAggregateElement(i));
    }
    return ConstantArray::get(ArrayTy, Elts);
  } else {
    ArrayType *AT = cast<ArrayType>(VecTy);
    ArrayType *EltArrayTy = cast<ArrayType>(ArrayTy->getElementType());
    SmallVector<Constant *, 4> Elts;
    for (unsigned i = 0; i < AT->getNumElements(); i++) {
      Constant *Elt = lowerInitVal(InitVal->getAggregateElement(i), EltArrayTy);
      Elts.emplace_back(Elt);
    }
    return ConstantArray::get(ArrayTy, Elts);
  }
}

bool DynamicIndexingVectorToArray::HasVectorDynamicIndexing(Value *V) {
  return dxilutil::HasDynamicIndexing(V);
}

}

char DynamicIndexingVectorToArray::ID = 0;

INITIALIZE_PASS(DynamicIndexingVectorToArray, "dynamic-vector-to-array",
  "Replace dynamic indexing vector with array", false,
  false)

// Public interface to the DynamicIndexingVectorToArray pass
ModulePass *llvm::createDynamicIndexingVectorToArrayPass(bool ReplaceAllVector) {
  return new DynamicIndexingVectorToArray(ReplaceAllVector);
}

//===----------------------------------------------------------------------===//
// Flatten multi dim array into 1 dim.
//===----------------------------------------------------------------------===//

namespace {

class MultiDimArrayToOneDimArray : public LowerTypePass {
public:
  explicit MultiDimArrayToOneDimArray() : LowerTypePass(ID) {}
  static char ID; // Pass identification, replacement for typeid
protected:
  bool needToLower(Value *V) override;
  void lowerUseWithNewValue(Value *V, Value *NewV) override;
  Type *lowerType(Type *Ty) override;
  Constant *lowerInitVal(Constant *InitVal, Type *NewTy) override;
  StringRef getGlobalPrefix() override { return ".1dim"; }
  bool isSafeToLowerArray(Value *V);
};

// Recurse users, looking for any direct users of array or sub-array type,
// other than lifetime markers:
bool MultiDimArrayToOneDimArray::isSafeToLowerArray(Value *V) {
  if (!V->getType()->getPointerElementType()->isArrayTy())
    return true;
  for (auto it = V->user_begin(); it != V->user_end();) {
    User *U = *it++;
    if (isa<BitCastOperator>(U)) {
      // Bitcast is ok because source type can be changed.
      continue;
    } else if (isa<GEPOperator>(U) || isa<AddrSpaceCastInst>(U) ||
               isa<ConstantExpr>(U)) {
      if (!isSafeToLowerArray(U))
        return false;
    } else {
      return false;
    }
  }
  return true;
}


bool MultiDimArrayToOneDimArray::needToLower(Value *V) {
  Type *Ty = V->getType()->getPointerElementType();
  ArrayType *AT = dyn_cast<ArrayType>(Ty);
  if (!AT)
    return false;
  if (!isa<ArrayType>(AT->getElementType())) {
    return false;
  } else {
    // Merge all GEP.
    dxilutil::MergeGepUse(V);
    return isSafeToLowerArray(V);
  }
}

void ReplaceMultiDimGEP(User *GEP, Value *OneDim, IRBuilder<> &Builder) {
  gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

  Value *PtrOffset = GEPIt.getOperand();
  ++GEPIt;
  Value *ArrayIdx = GEPIt.getOperand();
  ++GEPIt;
  Value *VecIdx = nullptr;
  SmallVector<Value*,8> StructIdxs;
  for (; GEPIt != E; ++GEPIt) {
    if (GEPIt->isArrayTy()) {
      unsigned arraySize = GEPIt->getArrayNumElements();
      Value *V = GEPIt.getOperand();
      ArrayIdx = Builder.CreateMul(ArrayIdx, Builder.getInt32(arraySize));
      ArrayIdx = Builder.CreateAdd(V, ArrayIdx);
    } else if (isa<StructType>(*GEPIt)) {
      // Replaces multi-dim array of struct, with single-dim array of struct
      StructIdxs.push_back(PtrOffset);
      StructIdxs.push_back(ArrayIdx);
      while (GEPIt != E) {
        StructIdxs.push_back(GEPIt.getOperand());
        ++GEPIt;
      }
      break;
    } else {
      DXASSERT_NOMSG(isa<VectorType>(*GEPIt));
      VecIdx = GEPIt.getOperand();
    }
  }
  Value *NewGEP = nullptr;
  if (StructIdxs.size())
    NewGEP = Builder.CreateGEP(OneDim, StructIdxs);
  else if (!VecIdx)
    NewGEP = Builder.CreateGEP(OneDim, {PtrOffset, ArrayIdx});
  else
    NewGEP = Builder.CreateGEP(OneDim, {PtrOffset, ArrayIdx, VecIdx});

  GEP->replaceAllUsesWith(NewGEP);
}

void MultiDimArrayToOneDimArray::lowerUseWithNewValue(Value *MultiDim, Value *OneDim) {
  LLVMContext &Context = MultiDim->getContext();
  // All users should be element type.
  // Replace users of AI or GV.
  for (auto it = MultiDim->user_begin(); it != MultiDim->user_end();) {
    User *U = *(it++);
    if (U->user_empty())
      continue;
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      BCI->setOperand(0, OneDim);
      continue;
    }

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      IRBuilder<> Builder(Context);
      if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
        // NewGEP must be GEPOperator too.
        // No instruction will be build.
        ReplaceMultiDimGEP(U, OneDim, Builder);
      } else if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        Value *NewAddrSpaceCast = Builder.CreateAddrSpaceCast(
          OneDim,
          PointerType::get(OneDim->getType()->getPointerElementType(),
                           CE->getType()->getPointerAddressSpace()));
        lowerUseWithNewValue(CE, NewAddrSpaceCast);
      } else {
        DXASSERT(0, "not implemented");
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      IRBuilder<> Builder(GEP);
      ReplaceMultiDimGEP(U, OneDim, Builder);
      GEP->eraseFromParent();
    } else {
      DXASSERT(0, "not implemented");
    }
  }
}

Type *MultiDimArrayToOneDimArray::lowerType(Type *Ty) {
  ArrayType *AT = cast<ArrayType>(Ty);
  unsigned arraySize = AT->getNumElements();

  Type *EltTy = AT->getElementType();
  // support multi level of array
  while (EltTy->isArrayTy()) {
    ArrayType *ElAT = cast<ArrayType>(EltTy);
    arraySize *= ElAT->getNumElements();
    EltTy = ElAT->getElementType();
  }

  return ArrayType::get(EltTy, arraySize);
}

void FlattenMultiDimConstArray(Constant *V, std::vector<Constant *> &Elts) {
  if (!V->getType()->isArrayTy()) {
    Elts.emplace_back(V);
  } else {
    ArrayType *AT = cast<ArrayType>(V->getType());
    for (unsigned i = 0; i < AT->getNumElements(); i++) {
      FlattenMultiDimConstArray(V->getAggregateElement(i), Elts);
    }
  }
}

Constant *MultiDimArrayToOneDimArray::lowerInitVal(Constant *InitVal, Type *NewTy) {
  if (InitVal) {
    // MultiDim array init should be done by store.
    if (isa<ConstantAggregateZero>(InitVal))
      InitVal = ConstantAggregateZero::get(NewTy);
    else if (isa<UndefValue>(InitVal))
      InitVal = UndefValue::get(NewTy);
    else {
      std::vector<Constant *> Elts;
      FlattenMultiDimConstArray(InitVal, Elts);
      InitVal = ConstantArray::get(cast<ArrayType>(NewTy), Elts);
    }
  } else {
    InitVal = UndefValue::get(NewTy);
  }
  return InitVal;
}

}

char MultiDimArrayToOneDimArray::ID = 0;

INITIALIZE_PASS(MultiDimArrayToOneDimArray, "multi-dim-one-dim",
  "Flatten multi-dim array into one-dim array", false,
  false)

// Public interface to the SROA_Parameter_HLSL pass
ModulePass *llvm::createMultiDimArrayToOneDimArrayPass() {
  return new MultiDimArrayToOneDimArray();
}

//===----------------------------------------------------------------------===//
// Lower resource into handle.
//===----------------------------------------------------------------------===//

namespace {

class ResourceToHandle : public LowerTypePass {
public:
  explicit ResourceToHandle() : LowerTypePass(ID) {}
  static char ID; // Pass identification, replacement for typeid
protected:
  bool needToLower(Value *V) override;
  void lowerUseWithNewValue(Value *V, Value *NewV) override;
  Type *lowerType(Type *Ty) override;
  Constant *lowerInitVal(Constant *InitVal, Type *NewTy) override;
  StringRef getGlobalPrefix() override { return ".res"; }
  void initialize(Module &M) override;
private:
  void ReplaceResourceWithHandle(Value *ResPtr, Value *HandlePtr);
  void ReplaceResourceGEPWithHandleGEP(Value *GEP, ArrayRef<Value *> idxList,
                                       Value *A, IRBuilder<> &Builder);
  void ReplaceResourceArrayWithHandleArray(Value *VA, Value *A);

  Type *m_HandleTy;
  HLModule *m_pHLM;
  bool  m_bIsLib;
};

void ResourceToHandle::initialize(Module &M) {
  DXASSERT(M.HasHLModule(), "require HLModule");
  m_pHLM = &M.GetHLModule();
  m_HandleTy = m_pHLM->GetOP()->GetHandleType();
  m_bIsLib = m_pHLM->GetShaderModel()->IsLib();
}

bool ResourceToHandle::needToLower(Value *V) {
  Type *Ty = V->getType()->getPointerElementType();
  Ty = dxilutil::GetArrayEltTy(Ty);
  return (dxilutil::IsHLSLObjectType(Ty) &&
          !HLModule::IsStreamOutputType(Ty)) &&
         // Skip lib profile.
         !m_bIsLib;
}

Type *ResourceToHandle::lowerType(Type *Ty) {
  if ((dxilutil::IsHLSLObjectType(Ty) && !HLModule::IsStreamOutputType(Ty))) {
    return m_HandleTy;
  }

  ArrayType *AT = cast<ArrayType>(Ty);

  SmallVector<ArrayType *, 4> nestArrayTys;
  nestArrayTys.emplace_back(AT);

  Type *EltTy = AT->getElementType();
  // support multi level of array
  while (EltTy->isArrayTy()) {
    ArrayType *ElAT = cast<ArrayType>(EltTy);
    nestArrayTys.emplace_back(ElAT);
    EltTy = ElAT->getElementType();
  }

  return CreateNestArrayTy(m_HandleTy, nestArrayTys);
}

Constant *ResourceToHandle::lowerInitVal(Constant *InitVal, Type *NewTy) {
  DXASSERT(isa<UndefValue>(InitVal), "resource cannot have real init val");
  return UndefValue::get(NewTy);
}

void ResourceToHandle::ReplaceResourceWithHandle(Value *ResPtr,
                                                 Value *HandlePtr) {
  for (auto it = ResPtr->user_begin(); it != ResPtr->user_end();) {
    User *U = *(it++);
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      IRBuilder<> Builder(LI);
      Value *Handle = Builder.CreateLoad(HandlePtr);
      Type *ResTy = LI->getType();
      // Used by createHandle or Store.
      for (auto ldIt = LI->user_begin(); ldIt != LI->user_end();) {
        User *ldU = *(ldIt++);
        if (StoreInst *SI = dyn_cast<StoreInst>(ldU)) {
          Value *TmpRes = HLModule::EmitHLOperationCall(
              Builder, HLOpcodeGroup::HLCast,
              (unsigned)HLCastOpcode::HandleToResCast, ResTy, {Handle},
              *m_pHLM->GetModule());
          SI->replaceUsesOfWith(LI, TmpRes);
        } else {
          CallInst *CI = cast<CallInst>(ldU);
          DXASSERT(hlsl::GetHLOpcodeGroupByName(CI->getCalledFunction()) == HLOpcodeGroup::HLCreateHandle,
                   "must be createHandle");
          CI->replaceAllUsesWith(Handle);
          CI->eraseFromParent();
        }
      }
      LI->eraseFromParent();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      Value *Res = SI->getValueOperand();
      IRBuilder<> Builder(SI);
      // CreateHandle from Res.
      Value *Handle = HLModule::EmitHLOperationCall(
          Builder, HLOpcodeGroup::HLCreateHandle,
          /*opcode*/ 0, m_HandleTy, {Res}, *m_pHLM->GetModule());
      // Store Handle to HandlePtr.
      Builder.CreateStore(Handle, HandlePtr);
      // Remove resource Store.
      SI->eraseFromParent();
    } else if (U->user_empty() && isa<GEPOperator>(U)) {
      continue;
    } else {
      CallInst *CI = cast<CallInst>(U);
      IRBuilder<> Builder(CI);
      HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
      // Allow user function to use res ptr as argument.
      if (group == HLOpcodeGroup::NotHL) {
          Value *TmpResPtr = Builder.CreateBitCast(HandlePtr, ResPtr->getType());
          CI->replaceUsesOfWith(ResPtr, TmpResPtr);
      } else {
        DXASSERT(0, "invalid operation on resource");
      }
    }
  }
}

void ResourceToHandle::ReplaceResourceGEPWithHandleGEP(
    Value *GEP, ArrayRef<Value *> idxList, Value *A, IRBuilder<> &Builder) {
  Value *newGEP = Builder.CreateGEP(A, idxList);
  Type *Ty = GEP->getType()->getPointerElementType();
  if (Ty->isArrayTy()) {
    ReplaceResourceArrayWithHandleArray(GEP, newGEP);
  } else {
    DXASSERT(dxilutil::IsHLSLObjectType(Ty), "must be resource type here");
    ReplaceResourceWithHandle(GEP, newGEP);
  }
}

void ResourceToHandle::ReplaceResourceArrayWithHandleArray(Value *VA,
                                                           Value *A) {
  for (auto U = VA->user_begin(); U != VA->user_end();) {
    User *User = *(U++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      IRBuilder<> Builder(GEP);
      SmallVector<Value *, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      ReplaceResourceGEPWithHandleGEP(GEP, idxList, A, Builder);
      GEP->eraseFromParent();
    } else if (GEPOperator *GEPOp = dyn_cast<GEPOperator>(User)) {
      IRBuilder<> Builder(GEPOp->getContext());
      SmallVector<Value *, 4> idxList(GEPOp->idx_begin(), GEPOp->idx_end());
      ReplaceResourceGEPWithHandleGEP(GEPOp, idxList, A, Builder);
    } else {
      DXASSERT(0, "Array pointer should only used by GEP");
    }
  }
}

void ResourceToHandle::lowerUseWithNewValue(Value *V, Value *NewV) {
  Type *Ty = V->getType()->getPointerElementType();
  // Replace V with NewV.
  if (Ty->isArrayTy()) {
    ReplaceResourceArrayWithHandleArray(V, NewV);
  } else {
    ReplaceResourceWithHandle(V, NewV);
  }
}

}

char ResourceToHandle::ID = 0;

INITIALIZE_PASS(ResourceToHandle, "resource-handle",
  "Lower resource into handle", false,
  false)

// Public interface to the ResourceToHandle pass
ModulePass *llvm::createResourceToHandlePass() {
  return new ResourceToHandle();
}
