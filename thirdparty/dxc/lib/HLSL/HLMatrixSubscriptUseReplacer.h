///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixSubscriptUseReplacer.h                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include <vector>

namespace llvm {
class Value;
class AllocaInst;
class CallInst;
class Instruction;
class Function;
} // namespace llvm

namespace hlsl {
// Implements recursive replacement of a matrix subscript's uses,
// from a pointer to a matrix value to a pointer to its lowered vector version,
// whether directly or through GEPs in the case of two-level indexing like mat[i][j].
// This has to handle one or two levels of indices, each of which either
// constant or dynamic: mat[0], mat[i], mat[0][0], mat[i][0], mat[0][j], mat[i][j],
// plus the equivalent element accesses: mat._11, mat._11_12, mat._11_12[0], mat._11_12[i]
class HLMatrixSubscriptUseReplacer {
public:
  // The constructor does everything
  HLMatrixSubscriptUseReplacer(llvm::CallInst* Call, llvm::Value *LoweredPtr, llvm::Value *TempLoweredMatrix,
    llvm::SmallVectorImpl<llvm::Value*> &ElemIndices, bool AllowLoweredPtrGEPs,
    std::vector<llvm::Instruction*> &DeadInsts);

private:
  void replaceUses(llvm::Instruction* PtrInst, llvm::Value* SubIdxVal);
  llvm::Value *tryGetScalarIndex(llvm::Value *SubIdxVal, llvm::IRBuilder<> &Builder);
  void cacheLoweredMatrix(bool ForDynamicIndexing, llvm::IRBuilder<> &Builder);
  llvm::Value *loadElem(llvm::Value *Idx, llvm::IRBuilder<> &Builder);
  void storeElem(llvm::Value *Idx, llvm::Value *Elem, llvm::IRBuilder<> &Builder);
  llvm::Value *loadVector(llvm::IRBuilder<> &Builder);
  void storeVector(llvm::Value *Vec, llvm::IRBuilder<> &Builder);
  void flushLoweredMatrix(llvm::IRBuilder<> &Builder);

private:
  llvm::Value *LoweredPtr;
  llvm::SmallVectorImpl<llvm::Value*> &ElemIndices;
  std::vector<llvm::Instruction*> &DeadInsts;
  bool AllowLoweredPtrGEPs = false;
  bool HasScalarResult = false;
  bool HasDynamicElemIndex = false;
  llvm::Type *LoweredTy = nullptr;

  // The entire lowered matrix as loaded from LoweredPtr,
  // nullptr if we copied it to a temporary array.
  llvm::Value *TempLoweredMatrix = nullptr;

  // We allocate this if the level 1 indices are not all constants,
  // so we can dynamically index the lowered matrix vector.
  llvm::AllocaInst *LazyTempElemArrayAlloca = nullptr;

  // We'll allocate this lazily if we have a dynamic level 2 index (mat[0][i]),
  // so we can dynamically index the level 1 indices.
  llvm::AllocaInst *LazyTempElemIndicesArrayAlloca = nullptr;
};
} // namespace hlsl
