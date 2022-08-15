///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilEliminateVector.cpp                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// A pass to remove vector instructions, especially in situations where      //
// optimizations are turned off.                                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/Pass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/DIBuilder.h"

#include "llvm/Analysis/DxilValueCache.h"

#include <vector>

using namespace llvm;

namespace {

class DxilEliminateVector : public FunctionPass {
public:
  static char ID;
  DxilEliminateVector() : FunctionPass(ID) {
    initializeDxilEliminateVectorPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DxilValueCache>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesAll(); // DxilValueCache is safe. CFG is not changed, so DT is okay.
  }

  bool TryRewriteDebugInfoForVector(InsertElementInst *IE);
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override { return "Dxil Eliminate Vector"; }
};

char DxilEliminateVector::ID;
}

static
MetadataAsValue *GetAsMetadata(Instruction *I) {
  if (auto *L = LocalAsMetadata::getIfExists(I)) {
    if (auto *DINode = MetadataAsValue::getIfExists(I->getContext(), L)) {
      return DINode;
    }
  }
  return nullptr;
}

static bool IsZeroInitializer(Value *V) {
  Constant *C = dyn_cast<Constant>(V);
  return C && C->isZeroValue();
}

static
bool CollectVectorElements(Value *V, SmallVector<Value *, 4> &Elements) {
  if (InsertElementInst *IE = dyn_cast<InsertElementInst>(V)) {

    Value *Vec = IE->getOperand(0);
    Value *Element = IE->getOperand(1);
    Value *Index = IE->getOperand(2);

    if (!isa<UndefValue>(Vec) && !IsZeroInitializer(Vec)) {
      if (!CollectVectorElements(Vec, Elements))
        return false;
    }

    ConstantInt *ConstIndex = dyn_cast<ConstantInt>(Index);
    if (!ConstIndex)
      return false;

    uint64_t IdxValue = ConstIndex->getLimitedValue();
    if (IdxValue < 4) {
      if (Elements.size() <= IdxValue)
        Elements.resize(IdxValue+1);
      Elements[IdxValue] = Element;
    }

    return true;
  }

  return false;
}

bool DxilEliminateVector::TryRewriteDebugInfoForVector(InsertElementInst *IE) {

  // If this is not ever used as meta-data, there's no debug
  MetadataAsValue *DebugI = GetAsMetadata(IE);
  if (!DebugI)
    return false;

  // Collect @dbg.value instructions
  SmallVector<DbgValueInst *, 4> DbgValueInsts;
  for (User *U : DebugI->users()) {
    if (DbgValueInst *DbgValueI = dyn_cast<DbgValueInst>(U)) {
      DbgValueInsts.push_back(DbgValueI);
    }
  }

  if (!DbgValueInsts.size())
    return false;

  SmallVector<Value *, 4> Elements;
  if (!CollectVectorElements(IE, Elements))
    return false;

  DIBuilder DIB(*IE->getModule());
  const DataLayout &DL = IE->getModule()->getDataLayout();

  // Go through the elements and create @dbg.value with bit-piece
  // expressions for them.
  bool Changed = false;
  for (DbgValueInst *DVI : DbgValueInsts) {

    DIExpression *ParentExpr = DVI->getExpression();
    unsigned BitpieceOffset = 0;
    if (ParentExpr->isBitPiece())
      BitpieceOffset = ParentExpr->getBitPieceOffset();

    for (unsigned i = 0; i < Elements.size(); i++) {
      if (!Elements[i])
        continue;

      unsigned ElementSize = DL.getTypeSizeInBits(Elements[i]->getType());
      unsigned ElementAlign = DL.getTypeAllocSizeInBits(Elements[i]->getType());
      DIExpression *Expr = DIB.createBitPieceExpression(BitpieceOffset + i * ElementAlign, ElementSize);
      DIB.insertDbgValueIntrinsic(Elements[i], 0, DVI->getVariable(), Expr, DVI->getDebugLoc(), DVI);

      Changed = true;
    }

    DVI->eraseFromParent();
  }

  return Changed;
}

bool DxilEliminateVector::runOnFunction(Function &F) {

  auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DxilValueCache *DVC = &getAnalysis<DxilValueCache>();

  std::vector<Instruction *> VectorInsts;
  std::vector<AllocaInst *> VectorAllocas;

  // Collect the vector insts and allocas.
  for (auto &BB : F) {
    for (auto &I : BB)
      if (isa<InsertElementInst>(&I) || isa<ExtractElementInst>(&I))
        VectorInsts.push_back(&I);
      else if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
        if (AI->getAllocatedType()->isVectorTy() && llvm::isAllocaPromotable(AI))
          VectorAllocas.push_back(AI);
      }
  }

  if (!VectorInsts.size())
    return false;

  bool Changed = false;

  // Promote the allocas if they exist. They could very well exist
  // because of precise.
  if (VectorAllocas.size()) {
    PromoteMemToReg(VectorAllocas, *DT);
    Changed = true;
  }

  // Iteratively try to remove them, untill all gone or unable to
  // do it anymore.
  unsigned Attempts = VectorInsts.size();
  for (unsigned i = 0; i < Attempts; i++) {
    bool LocalChange = false;

    for (unsigned j = 0; j < VectorInsts.size();) {
      auto *I = VectorInsts[j];
      bool Remove = false;

      if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
        TryRewriteDebugInfoForVector(IE);
      }

      if (Value *V = DVC->GetValue(I, DT)) {
        I->replaceAllUsesWith(V);
        Remove = true;
      }
      else if (I->user_empty()) {
        Remove = true;
      }

      // Do the remove
      if (Remove) {
        LocalChange = true;
        I->eraseFromParent();
        VectorInsts.erase(VectorInsts.begin() + j);
      }
      else {
        j++;
      }
    }

    Changed |= LocalChange;
    if (!LocalChange)
      break;
  }

  return Changed;
}

Pass *llvm::createDxilEliminateVectorPass() {
  return new DxilEliminateVector();
}

INITIALIZE_PASS(DxilEliminateVector, "dxil-elim-vector", "Dxil Eliminate Vectors", false, false)
