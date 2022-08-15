///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixType.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/Support/Global.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"

using namespace llvm;
using namespace hlsl;

HLMatrixType::HLMatrixType(Type *RegReprElemTy, unsigned NumRows, unsigned NumColumns)
  : RegReprElemTy(RegReprElemTy), NumRows(NumRows), NumColumns(NumColumns) {
  DXASSERT(RegReprElemTy != nullptr && (RegReprElemTy->isIntegerTy() || RegReprElemTy->isFloatingPointTy()),
    "Invalid matrix element type.");
  DXASSERT(NumRows >= 1 && NumRows <= 4 && NumColumns >= 1 && NumColumns <= 4,
    "Invalid matrix dimensions.");
}

Type *HLMatrixType::getElementType(bool MemRepr) const {
  // Bool i1s become i32s
  return MemRepr && RegReprElemTy->isIntegerTy(1)
    ? IntegerType::get(RegReprElemTy->getContext(), 32)
    : RegReprElemTy;
}

unsigned HLMatrixType::getRowMajorIndex(unsigned RowIdx, unsigned ColIdx) const {
  return getRowMajorIndex(RowIdx, ColIdx, NumRows, NumColumns);
}

unsigned HLMatrixType::getColumnMajorIndex(unsigned RowIdx, unsigned ColIdx) const {
  return getColumnMajorIndex(RowIdx, ColIdx, NumRows, NumColumns);
}

unsigned HLMatrixType::getRowMajorIndex(unsigned RowIdx, unsigned ColIdx, unsigned NumRows, unsigned NumColumns) {
  DXASSERT_NOMSG(RowIdx < NumRows && ColIdx < NumColumns);
  return RowIdx * NumColumns + ColIdx;
}

unsigned HLMatrixType::getColumnMajorIndex(unsigned RowIdx, unsigned ColIdx, unsigned NumRows, unsigned NumColumns) {
  DXASSERT_NOMSG(RowIdx < NumRows && ColIdx < NumColumns);
  return ColIdx * NumRows + RowIdx;
}

VectorType *HLMatrixType::getLoweredVectorType(bool MemRepr) const {
  return VectorType::get(getElementType(MemRepr), getNumElements());
}

Value *HLMatrixType::emitLoweredMemToReg(Value *Val, IRBuilder<> &Builder) const {
  DXASSERT(Val->getType()->getScalarType() == getElementTypeForMem(), "Lowered matrix type mismatch.");
  if (RegReprElemTy->isIntegerTy(1)) {
    Val = Builder.CreateICmpNE(Val, Constant::getNullValue(Val->getType()), "tobool");
  }
  return Val;
}

Value *HLMatrixType::emitLoweredRegToMem(Value *Val, IRBuilder<> &Builder) const {
  DXASSERT(Val->getType()->getScalarType() == RegReprElemTy, "Lowered matrix type mismatch.");
  if (RegReprElemTy->isIntegerTy(1)) {
    Type *MemReprTy = Val->getType()->isVectorTy() ? getLoweredVectorTypeForMem() : getElementTypeForMem();
    Val = Builder.CreateZExt(Val, MemReprTy, "frombool");
  }
  return Val;
}

Value *HLMatrixType::emitLoweredLoad(Value *Ptr, IRBuilder<> &Builder) const {
  return emitLoweredMemToReg(Builder.CreateLoad(Ptr), Builder);
}

StoreInst *HLMatrixType::emitLoweredStore(Value *Val, Value *Ptr, IRBuilder<> &Builder) const {
  return Builder.CreateStore(emitLoweredRegToMem(Val, Builder), Ptr);
}

Value *HLMatrixType::emitLoweredVectorRowToCol(Value *VecVal, IRBuilder<> &Builder) const {
  DXASSERT(VecVal->getType() == getLoweredVectorTypeForReg(), "Lowered matrix type mismatch.");
  if (NumRows == 1 || NumColumns == 1) return VecVal;

  SmallVector<int, 16> ShuffleIndices;
  for (unsigned ColIdx = 0; ColIdx < NumColumns; ++ColIdx)
    for (unsigned RowIdx = 0; RowIdx < NumRows; ++RowIdx)
      ShuffleIndices.emplace_back((int)getRowMajorIndex(RowIdx, ColIdx));
  return Builder.CreateShuffleVector(VecVal, VecVal, ShuffleIndices, "row2col");
}

Value *HLMatrixType::emitLoweredVectorColToRow(Value *VecVal, IRBuilder<> &Builder) const {
  DXASSERT(VecVal->getType() == getLoweredVectorTypeForReg(), "Lowered matrix type mismatch.");
  if (NumRows == 1 || NumColumns == 1) return VecVal;

  SmallVector<int, 16> ShuffleIndices;
  for (unsigned RowIdx = 0; RowIdx < NumRows; ++RowIdx)
    for (unsigned ColIdx = 0; ColIdx < NumColumns; ++ColIdx)
      ShuffleIndices.emplace_back((int)getColumnMajorIndex(RowIdx, ColIdx));
  return Builder.CreateShuffleVector(VecVal, VecVal, ShuffleIndices, "col2row");
}

bool HLMatrixType::isa(Type *Ty) {
  StructType *StructTy = llvm::dyn_cast<StructType>(Ty);
  return StructTy != nullptr && !StructTy->isLiteral() && StructTy->getName().startswith(StructNamePrefix);
}

bool HLMatrixType::isMatrixPtr(Type *Ty) {
  PointerType *PtrTy = llvm::dyn_cast<PointerType>(Ty);
  return PtrTy != nullptr && isa(PtrTy->getElementType());
}

bool HLMatrixType::isMatrixArray(Type *Ty) {
  ArrayType *ArrayTy = llvm::dyn_cast<ArrayType>(Ty);
  if (ArrayTy == nullptr) return false;
  while (ArrayType *NestedArrayTy = llvm::dyn_cast<ArrayType>(ArrayTy->getElementType()))
    ArrayTy = NestedArrayTy;
  return isa(ArrayTy->getElementType());
}

bool HLMatrixType::isMatrixArrayPtr(Type *Ty) {
  PointerType *PtrTy = llvm::dyn_cast<PointerType>(Ty);
  if (PtrTy == nullptr) return false;
  return isMatrixArray(PtrTy->getElementType());
}

bool HLMatrixType::isMatrixPtrOrArrayPtr(Type *Ty) {
  PointerType *PtrTy = llvm::dyn_cast<PointerType>(Ty);
  if (PtrTy == nullptr) return false;
  Ty = PtrTy->getElementType();
  while (ArrayType *ArrayTy = llvm::dyn_cast<ArrayType>(Ty))
    Ty = Ty->getArrayElementType();
  return isa(Ty);
}

bool HLMatrixType::isMatrixOrPtrOrArrayPtr(Type *Ty) {
  if (PointerType *PtrTy = llvm::dyn_cast<PointerType>(Ty)) Ty = PtrTy->getElementType();
  while (ArrayType *ArrayTy = llvm::dyn_cast<ArrayType>(Ty)) Ty = ArrayTy->getElementType();
  return isa(Ty);
}

// Converts a matrix, matrix pointer, or matrix array pointer type to its lowered equivalent.
// If the type is not matrix-derived, the original type is returned.
// Does not lower struct types containing matrices.
Type *HLMatrixType::getLoweredType(Type *Ty, bool MemRepr) {
  if (PointerType *PtrTy = llvm::dyn_cast<PointerType>(Ty)) {
    // Pointees are always in memory representation
    Type *LoweredElemTy = getLoweredType(PtrTy->getElementType(), /* MemRepr */ true);
    return LoweredElemTy == PtrTy->getElementType()
      ? Ty : PointerType::get(LoweredElemTy, PtrTy->getAddressSpace());
  }
  else if (ArrayType *ArrayTy = llvm::dyn_cast<ArrayType>(Ty)) {
    // Arrays are always in memory and so their elements are in memory representation
    Type *LoweredElemTy = getLoweredType(ArrayTy->getElementType(), /* MemRepr */ true);
    return LoweredElemTy == ArrayTy->getElementType()
      ? Ty : ArrayType::get(LoweredElemTy, ArrayTy->getNumElements());
  }
  else if (HLMatrixType MatrixTy = HLMatrixType::dyn_cast(Ty)) {
    return MatrixTy.getLoweredVectorType(MemRepr);
  }
  else return Ty;
}

HLMatrixType HLMatrixType::cast(Type *Ty) {
  DXASSERT_NOMSG(isa(Ty));
  StructType *StructTy = llvm::cast<StructType>(Ty);
  DXASSERT_NOMSG(Ty->getNumContainedTypes() == 1);
  ArrayType *RowArrayTy = llvm::cast<ArrayType>(StructTy->getElementType(0));
  DXASSERT_NOMSG(RowArrayTy->getNumElements() >= 1 && RowArrayTy->getNumElements() <= 4);
  VectorType *RowTy = llvm::cast<VectorType>(RowArrayTy->getElementType());
  DXASSERT_NOMSG(RowTy->getNumElements() >= 1 && RowTy->getNumElements() <= 4);
  return HLMatrixType(RowTy->getElementType(), RowArrayTy->getNumElements(), RowTy->getNumElements());
}

HLMatrixType HLMatrixType::dyn_cast(Type *Ty) {
  return isa(Ty) ? cast(Ty) : HLMatrixType();
}
