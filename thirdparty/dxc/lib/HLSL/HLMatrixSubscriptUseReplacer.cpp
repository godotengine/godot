///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixSubscriptUseReplacer.cpp                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "HLMatrixSubscriptUseReplacer.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/Global.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

using namespace llvm;
using namespace hlsl;

HLMatrixSubscriptUseReplacer::HLMatrixSubscriptUseReplacer(CallInst* Call, Value *LoweredPtr, Value *TempLoweredMatrix,
  SmallVectorImpl<Value*> &ElemIndices, bool AllowLoweredPtrGEPs, std::vector<Instruction*> &DeadInsts)
  : LoweredPtr(LoweredPtr), ElemIndices(ElemIndices), DeadInsts(DeadInsts),
    AllowLoweredPtrGEPs(AllowLoweredPtrGEPs), TempLoweredMatrix(TempLoweredMatrix)
{
  HasScalarResult = !Call->getType()->getPointerElementType()->isVectorTy();

  for (Value *ElemIdx : ElemIndices) {
    if (!isa<Constant>(ElemIdx)) {
      HasDynamicElemIndex = true;
      break;
    }
  }

  if (TempLoweredMatrix)
    LoweredTy = TempLoweredMatrix->getType();
  else
    LoweredTy = LoweredPtr->getType()->getPointerElementType();

  replaceUses(Call, /* GEPIdx */ nullptr);
}

void HLMatrixSubscriptUseReplacer::replaceUses(Instruction* PtrInst, Value* SubIdxVal) {
  // We handle any number of load/stores of the subscript,
  // whether through a GEP or not, but there should really only be one.
  while (!PtrInst->use_empty()) {
    llvm::Use &Use = *PtrInst->use_begin();
    Instruction *UserInst = cast<Instruction>(Use.getUser());

    bool DeleteUserInst = true;
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UserInst)) {
      // Recurse on GEPs
      DXASSERT(GEP->getNumIndices() >= 1 && GEP->getNumIndices() <= 2,
        "Unexpected GEP on constant matrix subscript.");
      DXASSERT(cast<ConstantInt>(GEP->idx_begin()->get())->isZero(),
        "Unexpected nonzero first index of constant matrix subscript GEP.");

      Value *NewSubIdxVal = SubIdxVal;
      if (GEP->getNumIndices() == 2) {
        DXASSERT(!HasScalarResult && SubIdxVal == nullptr,
          "Unexpected GEP on matrix subscript scalar value.");
        NewSubIdxVal = (GEP->idx_begin() + 1)->get();
      }

      replaceUses(GEP, NewSubIdxVal);
    }
    else {
      IRBuilder<> UserBuilder(UserInst);

      if (Value *ScalarElemIdx = tryGetScalarIndex(SubIdxVal, UserBuilder)) {
        // We are accessing a scalar element
        if (AllowLoweredPtrGEPs) {
          // Simply make the instruction point to the element in the lowered pointer
          DeleteUserInst = false;
          Value *ElemPtr = UserBuilder.CreateGEP(LoweredPtr, { UserBuilder.getInt32(0), ScalarElemIdx });
          Use.set(ElemPtr);
        }
        else {
          bool IsDynamicIndex = !isa<Constant>(ScalarElemIdx);
          cacheLoweredMatrix(IsDynamicIndex, UserBuilder);
          if (LoadInst *Load = dyn_cast<LoadInst>(UserInst)) {
            Value *Elem = loadElem(ScalarElemIdx, UserBuilder);
            Load->replaceAllUsesWith(Elem);
          }
          else if (StoreInst *Store = dyn_cast<StoreInst>(UserInst)) {
            storeElem(ScalarElemIdx, Store->getValueOperand(), UserBuilder);
            flushLoweredMatrix(UserBuilder);
          }
          else {
            llvm_unreachable("Unexpected matrix subscript use.");
          }
        }
      }
      else {
        // We are accessing a vector given by ElemIndices
        cacheLoweredMatrix(HasDynamicElemIndex, UserBuilder);
        if (LoadInst *Load = dyn_cast<LoadInst>(UserInst)) {
          Value *Vector = loadVector(UserBuilder);
          Load->replaceAllUsesWith(Vector);
        }
        else if (StoreInst *Store = dyn_cast<StoreInst>(UserInst)) {
          storeVector(Store->getValueOperand(), UserBuilder);
          flushLoweredMatrix(UserBuilder);
        }
        else {
          llvm_unreachable("Unexpected matrix subscript use.");
        }
      }
    }

    // We replaced this use, mark it dead
    if (DeleteUserInst) {
      DXASSERT(UserInst->use_empty(), "Matrix subscript user should be dead at this point.");
      Use.set(UndefValue::get(Use->getType()));
      DeadInsts.emplace_back(UserInst);
    }
  }
}

Value *HLMatrixSubscriptUseReplacer::tryGetScalarIndex(Value *SubIdxVal, IRBuilder<> &Builder) {
  if (SubIdxVal == nullptr) {
    // mat[0] case, returns a vector
    if (!HasScalarResult) return nullptr;

    // mat._11 case
    DXASSERT_NOMSG(ElemIndices.size() == 1);
    return ElemIndices[0];
  }

  if (ConstantInt *SubIdxConst = dyn_cast<ConstantInt>(SubIdxVal)) {
    // mat[0][0], mat[i][0] or mat._11_12[0] cases.
    uint64_t SubIdx = SubIdxConst->getLimitedValue();
    DXASSERT(SubIdx < ElemIndices.size(), "Unexpected out of range constant matrix subindex.");
    return ElemIndices[SubIdx];
  }

  // mat[0][j] or mat[i][j] case.
  // We need to dynamically index into the level 1 element indices
  if (LazyTempElemIndicesArrayAlloca == nullptr) {
    // The level 2 index is dynamic, use it to index a temporary array of the level 1 indices.
    IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));
    ArrayType *ArrayTy = ArrayType::get(AllocaBuilder.getInt32Ty(), ElemIndices.size());
    LazyTempElemIndicesArrayAlloca = AllocaBuilder.CreateAlloca(ArrayTy);
  }

  // Store level 1 indices in the temporary array
  Value *GEPIndices[2] = { Builder.getInt32(0), nullptr };
  for (unsigned SubIdx = 0; SubIdx < ElemIndices.size(); ++SubIdx) {
    GEPIndices[1] = Builder.getInt32(SubIdx);
    Value *TempArrayElemPtr = Builder.CreateGEP(LazyTempElemIndicesArrayAlloca, GEPIndices);
    Builder.CreateStore(ElemIndices[SubIdx], TempArrayElemPtr);
  }

  // Dynamically index using the subindex
  GEPIndices[1] = SubIdxVal;
  Value *ElemIdxPtr = Builder.CreateGEP(LazyTempElemIndicesArrayAlloca, GEPIndices);
  return Builder.CreateLoad(ElemIdxPtr);
}

// Unless we are allowed to GEP directly into the lowered matrix,
// we must load the vector in memory in order to read or write any elements.
// If we're going to dynamically index, we need to copy the vector into a temporary array.
// Further loadElem/storeElem calls depend on how we cached the matrix here.
void HLMatrixSubscriptUseReplacer::cacheLoweredMatrix(bool ForDynamicIndexing, IRBuilder<> &Builder) {
  // If we can GEP right into the lowered pointer, no need for caching
  if (AllowLoweredPtrGEPs) return;

  // Load without memory to register representation conversion,
  // since the point is to mimic pointer semantics
  if (!TempLoweredMatrix)
    TempLoweredMatrix = Builder.CreateLoad(LoweredPtr);

  if (!ForDynamicIndexing) return;

  // To handle mat[i] cases, we need to copy the matrix elements to
  // an array which we can dynamically index.
  VectorType *MatVecTy = cast<VectorType>(TempLoweredMatrix->getType());

  // Lazily create the temporary array alloca
  if (LazyTempElemArrayAlloca == nullptr) {
    ArrayType *TempElemArrayTy = ArrayType::get(MatVecTy->getElementType(), MatVecTy->getNumElements());
    IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));
    LazyTempElemArrayAlloca = AllocaBuilder.CreateAlloca(TempElemArrayTy);
  }

  // Copy the matrix elements to the temporary array
  Value *GEPIndices[2] = { Builder.getInt32(0), nullptr };
  for (unsigned ElemIdx = 0; ElemIdx < MatVecTy->getNumElements(); ++ElemIdx) {
    Value *VecElem = Builder.CreateExtractElement(TempLoweredMatrix, static_cast<uint64_t>(ElemIdx));
    GEPIndices[1] = Builder.getInt32(ElemIdx);
    Value *TempArrayElemPtr = Builder.CreateGEP(LazyTempElemArrayAlloca, GEPIndices);
    Builder.CreateStore(VecElem, TempArrayElemPtr);
  }

  // Null out the vector form so we know to use the array
  TempLoweredMatrix = nullptr;
}

Value *HLMatrixSubscriptUseReplacer::loadElem(Value *Idx, IRBuilder<> &Builder) {
  if (AllowLoweredPtrGEPs) {
    Value *ElemPtr = Builder.CreateGEP(LoweredPtr, { Builder.getInt32(0), Idx });
    return Builder.CreateLoad(ElemPtr);
  }
  else if (TempLoweredMatrix == nullptr) {
    DXASSERT_NOMSG(LazyTempElemArrayAlloca != nullptr);

    Value *TempArrayElemPtr = Builder.CreateGEP(LazyTempElemArrayAlloca, { Builder.getInt32(0), Idx });
    return Builder.CreateLoad(TempArrayElemPtr);
  }
  else {
    DXASSERT_NOMSG(isa<ConstantInt>(Idx));
    return Builder.CreateExtractElement(TempLoweredMatrix, Idx);
  }
}

void HLMatrixSubscriptUseReplacer::storeElem(Value *Idx, Value *Elem, IRBuilder<> &Builder) {
  if (AllowLoweredPtrGEPs) {
    Value *ElemPtr = Builder.CreateGEP(LoweredPtr, { Builder.getInt32(0), Idx });
    Builder.CreateStore(Elem, ElemPtr);
  }
  else if (TempLoweredMatrix == nullptr) {
    DXASSERT_NOMSG(LazyTempElemArrayAlloca != nullptr);

    Value *GEPIndices[2] = { Builder.getInt32(0), Idx };
    Value *TempArrayElemPtr = Builder.CreateGEP(LazyTempElemArrayAlloca, GEPIndices);
    Builder.CreateStore(Elem, TempArrayElemPtr);
  }
  else {
    DXASSERT_NOMSG(isa<ConstantInt>(Idx));
    TempLoweredMatrix = Builder.CreateInsertElement(TempLoweredMatrix, Elem, Idx);
  }
}

Value *HLMatrixSubscriptUseReplacer::loadVector(IRBuilder<> &Builder) {
  if (TempLoweredMatrix != nullptr) {
    // We can optimize this as a shuffle
    SmallVector<Constant*, 4> ShuffleIndices;
    ShuffleIndices.reserve(ElemIndices.size());
    for (Value *ElemIdx : ElemIndices)
      ShuffleIndices.emplace_back(cast<Constant>(ElemIdx));
    Constant* ShuffleVector = ConstantVector::get(ShuffleIndices);
    return Builder.CreateShuffleVector(TempLoweredMatrix, TempLoweredMatrix, ShuffleVector);
  }

  // Otherwise load elements one by one
  // Lowered form may be array when AllowLoweredPtrGEPs == true.
  Type* ElemTy = LoweredTy->isVectorTy() ? LoweredTy->getScalarType() :
              cast<ArrayType>(LoweredTy)->getArrayElementType();
  VectorType *VecTy = VectorType::get(ElemTy, static_cast<unsigned>(ElemIndices.size()));
  Value *Result = UndefValue::get(VecTy);
  for (unsigned SubIdx = 0; SubIdx < ElemIndices.size(); ++SubIdx) {
    Value *Elem = loadElem(ElemIndices[SubIdx], Builder);
    Result = Builder.CreateInsertElement(Result, Elem, static_cast<uint64_t>(SubIdx));
  }

  return Result;
}

void HLMatrixSubscriptUseReplacer::storeVector(Value *Vec, IRBuilder<> &Builder) {
  // We can't shuffle vectors of different sizes together, so insert one by one.
  DXASSERT(cast<FixedVectorType>(Vec->getType())->getNumElements() == ElemIndices.size(),
    "Matrix subscript stored vector element count mismatch.");

  for (unsigned SubIdx = 0; SubIdx < ElemIndices.size(); ++SubIdx) {
    Value *Elem = Builder.CreateExtractElement(Vec, static_cast<uint64_t>(SubIdx));
    storeElem(ElemIndices[SubIdx], Elem, Builder);
  }
}

void HLMatrixSubscriptUseReplacer::flushLoweredMatrix(IRBuilder<> &Builder) {
  // If GEPs are allowed, no flushing is necessary, we modified the source elements directly.
  if (AllowLoweredPtrGEPs) return;

  if (TempLoweredMatrix == nullptr) {
    // First re-create the vector from the temporary array
    DXASSERT_NOMSG(LazyTempElemArrayAlloca != nullptr);

    VectorType *LoweredMatrixTy = cast<VectorType>(LoweredTy);
    TempLoweredMatrix = UndefValue::get(LoweredMatrixTy);
    Value *GEPIndices[2] = { Builder.getInt32(0), nullptr };
    for (unsigned ElemIdx = 0; ElemIdx < LoweredMatrixTy->getNumElements(); ++ElemIdx) {
      GEPIndices[1] = Builder.getInt32(ElemIdx);
      Value *TempArrayElemPtr = Builder.CreateGEP(LazyTempElemArrayAlloca, GEPIndices);
      Value *NewElem = Builder.CreateLoad(TempArrayElemPtr);
      TempLoweredMatrix = Builder.CreateInsertElement(TempLoweredMatrix, NewElem, static_cast<uint64_t>(ElemIdx));
    }
  }

  // Store back the lowered matrix to its pointer
  Builder.CreateStore(TempLoweredMatrix, LoweredPtr);
  TempLoweredMatrix = nullptr;
}
