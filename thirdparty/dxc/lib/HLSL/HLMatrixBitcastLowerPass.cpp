///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixBitcastLowerPass.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLMatrixLowerPass.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace hlsl;
using namespace hlsl::HLMatrixLower;

// Matrix Bitcast lower.
// After linking Lower matrix bitcast patterns like:
//  %169 = bitcast [72 x float]* %0 to [6 x %class.matrix.float.4.3]*
//  %conv.i = fptoui float %164 to i32
//  %arrayidx.i = getelementptr inbounds [6 x %class.matrix.float.4.3], [6 x %class.matrix.float.4.3]* %169, i32 0, i32 %conv.i
//  %170 = bitcast %class.matrix.float.4.3* %arrayidx.i to <12 x float>*

namespace {

// Translate matrix type to array type.
Type *LowerMatrixTypeToOneDimArray(Type *Ty) {
  if (HLMatrixType MatTy = HLMatrixType::dyn_cast(Ty)) {
    Type *EltTy = MatTy.getElementTypeForReg();
    return ArrayType::get(EltTy, MatTy.getNumElements());
  }
  else {
    return Ty;
  }
}

Type *LowerMatrixArrayPointerToOneDimArray(Type *Ty) {
  unsigned addrSpace = Ty->getPointerAddressSpace();
  Ty = Ty->getPointerElementType();

  unsigned arraySize = 1;
  while (Ty->isArrayTy()) {
    arraySize *= Ty->getArrayNumElements();
    Ty = Ty->getArrayElementType();
  }

  HLMatrixType MatTy = HLMatrixType::cast(Ty);
  arraySize *= MatTy.getNumElements();

  Ty = ArrayType::get(MatTy.getElementTypeForReg(), arraySize);
  return PointerType::get(Ty, addrSpace);
}

Type *TryLowerMatTy(Type *Ty) {
  Type *VecTy = nullptr;
  if (HLMatrixType::isMatrixArrayPtr(Ty)) {
    VecTy = LowerMatrixArrayPointerToOneDimArray(Ty);
  } else if (isa<PointerType>(Ty) && HLMatrixType::isa(Ty->getPointerElementType())) {
    VecTy = LowerMatrixTypeToOneDimArray(
        Ty->getPointerElementType());
    VecTy = PointerType::get(VecTy, Ty->getPointerAddressSpace());
  }
  return VecTy;
}

class MatrixBitcastLowerPass : public FunctionPass {

public:
  static char ID; // Pass identification, replacement for typeid
  explicit MatrixBitcastLowerPass() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "Matrix Bitcast lower"; }
  bool runOnFunction(Function &F) override {
    bool bUpdated = false;
    std::unordered_set<BitCastInst*> matCastSet;
    for (auto blkIt = F.begin(); blkIt != F.end(); ++blkIt) {
      BasicBlock *BB = blkIt;
      for (auto iIt = BB->begin(); iIt != BB->end(); ) {
        Instruction *I = (iIt++);
        if (BitCastInst *BCI = dyn_cast<BitCastInst>(I)) {
          // Mutate mat to vec.
          Type *ToTy = BCI->getType();
          if (TryLowerMatTy(ToTy)) {
            matCastSet.insert(BCI);
            bUpdated = true;
          }
        }
      }
    }

    DxilModule &DM = F.getParent()->GetOrCreateDxilModule();
    // Remove bitcast which has CallInst user.
    if (DM.GetShaderModel()->IsLib()) {
      for (auto it = matCastSet.begin(); it != matCastSet.end();) {
        BitCastInst *BCI = *(it++);
        if (hasCallUser(BCI)) {
          matCastSet.erase(BCI);
        }
      }
    }

    // Lower matrix first.
    for (BitCastInst *BCI : matCastSet) {
      lowerMatrix(BCI, BCI->getOperand(0));
    }
    return bUpdated;
  }
private:
  void lowerMatrix(Instruction *M, Value *A);
  bool hasCallUser(Instruction *M);
};

}

bool MatrixBitcastLowerPass::hasCallUser(Instruction *M) {
  for (auto it = M->user_begin(); it != M->user_end();) {
    User *U = *(it++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      Type *EltTy = GEP->getType()->getPointerElementType();
      if (HLMatrixType::isa(EltTy)) {
        if (hasCallUser(GEP))
          return true;
      } else {
        DXASSERT(0, "invalid GEP for matrix");
      }
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      if (hasCallUser(BCI))
        return true;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (isa<VectorType>(LI->getType())) {
      } else {
        DXASSERT(0, "invalid load for matrix");
      }
    } else if (StoreInst *ST = dyn_cast<StoreInst>(U)) {
      Value *V = ST->getValueOperand();
      if (isa<VectorType>(V->getType())) {
      } else {
        DXASSERT(0, "invalid load for matrix");
      }
    } else if (isa<CallInst>(U)) {
      return true;
    } else {
      DXASSERT(0, "invalid use of matrix");
    }
  }
  return false;
}

namespace {
Value *CreateEltGEP(Value *A, unsigned i, Value *zeroIdx,
                    IRBuilder<> &Builder) {
  Value *GEP = nullptr;
  if (GetElementPtrInst *GEPA = dyn_cast<GetElementPtrInst>(A)) {
    // A should be gep oneDimArray, 0, index * matSize
    // Here add eltIdx to index * matSize foreach elt.
    Instruction *EltGEP = GEPA->clone();
    unsigned eltIdx = EltGEP->getNumOperands() - 1;
    Value *NewIdx =
        Builder.CreateAdd(EltGEP->getOperand(eltIdx), Builder.getInt32(i));
    EltGEP->setOperand(eltIdx, NewIdx);
    Builder.Insert(EltGEP);
    GEP = EltGEP;
  } else {
    GEP = Builder.CreateInBoundsGEP(A, {zeroIdx, Builder.getInt32(i)});
  }
  return GEP;
}
} // namespace

void MatrixBitcastLowerPass::lowerMatrix(Instruction *M, Value *A) {
  for (auto it = M->user_begin(); it != M->user_end();) {
    User *U = *(it++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      Type *EltTy = GEP->getType()->getPointerElementType();
      if (HLMatrixType::isa(EltTy)) {
        // Change gep matrixArray, 0, index
        // into
        //   gep oneDimArray, 0, index * matSize
        IRBuilder<> Builder(GEP);
        SmallVector<Value *, 2> idxList(GEP->idx_begin(), GEP->idx_end());
        DXASSERT(idxList.size() == 2,
                 "else not one dim matrix array index to matrix");

        HLMatrixType MatTy = HLMatrixType::cast(EltTy);
        Value *matSize = Builder.getInt32(MatTy.getNumElements());
        idxList.back() = Builder.CreateMul(idxList.back(), matSize);
        Value *NewGEP = Builder.CreateGEP(A, idxList);
        lowerMatrix(GEP, NewGEP);
        DXASSERT(GEP->user_empty(), "else lower matrix fail");
        GEP->eraseFromParent();
      } else {
        DXASSERT(0, "invalid GEP for matrix");
      }
    } else if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      lowerMatrix(BCI, A);
      DXASSERT(BCI->user_empty(), "else lower matrix fail");
      BCI->eraseFromParent();
    } else if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (VectorType *Ty = dyn_cast<VectorType>(LI->getType())) {
        IRBuilder<> Builder(LI);
        Value *zeroIdx = Builder.getInt32(0);
        unsigned vecSize = Ty->getNumElements();
        Value *NewVec = UndefValue::get(LI->getType());
        for (unsigned i = 0; i < vecSize; i++) {
          Value *GEP = CreateEltGEP(A, i, zeroIdx, Builder);
          Value *Elt = Builder.CreateLoad(GEP);
          NewVec = Builder.CreateInsertElement(NewVec, Elt, i);
        }
        LI->replaceAllUsesWith(NewVec);
        LI->eraseFromParent();
      } else {
        DXASSERT(0, "invalid load for matrix");
      }
    } else if (StoreInst *ST = dyn_cast<StoreInst>(U)) {
      Value *V = ST->getValueOperand();
      if (VectorType *Ty = dyn_cast<VectorType>(V->getType())) {
        IRBuilder<> Builder(LI);
        Value *zeroIdx = Builder.getInt32(0);
        unsigned vecSize = Ty->getNumElements();
        for (unsigned i = 0; i < vecSize; i++) {
          Value *GEP = CreateEltGEP(A, i, zeroIdx, Builder);
          Value *Elt = Builder.CreateExtractElement(V, i);
          Builder.CreateStore(Elt, GEP);
        }
        ST->eraseFromParent();
      } else {
        DXASSERT(0, "invalid load for matrix");
      }
    } else {
      DXASSERT(0, "invalid use of matrix");
    }
  }
}

char MatrixBitcastLowerPass::ID = 0;
FunctionPass *llvm::createMatrixBitcastLowerPass() { return new MatrixBitcastLowerPass(); }

INITIALIZE_PASS(MatrixBitcastLowerPass, "matrixbitcastlower", "Matrix Bitcast lower", false, false)
