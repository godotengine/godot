///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilConvergent.cpp                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Mark convergent for hlsl.                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_os_ostream.h"

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/HLSL/DxilConvergentName.h"

using namespace llvm;
using namespace hlsl;



///////////////////////////////////////////////////////////////////////////////
// DxilConvergent.
// Mark convergent to avoid sample coordnate calculation sink into control flow.
//
namespace {

class DxilConvergentMark : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilConvergentMark() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DxilConvergentMark";
  }

  bool runOnModule(Module &M) override {
    if (M.HasHLModule()) {
      const ShaderModel *SM = M.GetHLModule().GetShaderModel();
      if (!SM->IsPS() && !SM->IsLib() && (!SM->IsSM66Plus() || (!SM->IsCS() && !SM->IsMS() && !SM->IsAS())))
        return false;
    }
    bool bUpdated = false;

    for (Function &F : M.functions()) {
      if (F.isDeclaration())
        continue;

      // Compute postdominator relation.
      DominatorTreeBase<BasicBlock> PDR(true);
      PDR.recalculate(F);
      for (BasicBlock &bb : F.getBasicBlockList()) {
        for (auto it = bb.begin(); it != bb.end();) {
          Instruction *I = (it++);
          if (Value *V = FindConvergentOperand(I)) {
            if (PropagateConvergent(V, &F, PDR)) {
              // TODO: emit warning here.
            }
            bUpdated = true;
          }
        }
      }
    }

    return bUpdated;
  }

private:
  void MarkConvergent(Value *V, IRBuilder<> &Builder, Module &M);
  Value *FindConvergentOperand(Instruction *I);
  bool PropagateConvergent(Value *V, Function *F,
                           DominatorTreeBase<BasicBlock> &PostDom);
  bool PropagateConvergentImpl(Value *V, Function *F,
                           DominatorTreeBase<BasicBlock> &PostDom, std::set<Value*>& visited);
};

char DxilConvergentMark::ID = 0;

void DxilConvergentMark::MarkConvergent(Value *V, IRBuilder<> &Builder,
                                        Module &M) {
  Type *Ty = V->getType()->getScalarType();
  // Only work on vector/scalar types.
  if (Ty->isAggregateType() ||
      Ty->isPointerTy())
    return;
  FunctionType *FT = FunctionType::get(Ty, Ty, false);
  std::string str = kConvergentFunctionPrefix;
  raw_string_ostream os(str);
  Ty->print(os);
  os.flush();
  Function *ConvF = cast<Function>(M.getOrInsertFunction(str, FT));
  ConvF->addFnAttr(Attribute::AttrKind::Convergent);
  if (VectorType *VT = dyn_cast<VectorType>(V->getType())) {
    Value *ConvV = UndefValue::get(V->getType());
    std::vector<ExtractElementInst *> extractList(VT->getNumElements());
    for (unsigned i = 0; i < VT->getNumElements(); i++) {
      ExtractElementInst *EltV =
          cast<ExtractElementInst>(Builder.CreateExtractElement(V, i));
      extractList[i] = EltV;
      Value *EltC = Builder.CreateCall(ConvF, {EltV});
      ConvV = Builder.CreateInsertElement(ConvV, EltC, i);
    }
    V->replaceAllUsesWith(ConvV);
    for (ExtractElementInst *E : extractList) {
      E->setOperand(0, V);
    }
  } else {
    CallInst *ConvV = Builder.CreateCall(ConvF, {V});
    V->replaceAllUsesWith(ConvV);
    ConvV->setOperand(0, V);
  }
}

bool DxilConvergentMark::PropagateConvergent(
    Value *V, Function *F, DominatorTreeBase<BasicBlock> &PostDom) {
  std::set<Value *> visited;
  return PropagateConvergentImpl(V, F, PostDom, visited);
}

bool DxilConvergentMark::PropagateConvergentImpl(Value *V, Function *F,
  DominatorTreeBase<BasicBlock> &PostDom, std::set<Value*>& visited) {
  // Don't go through already visted nodes
  if (visited.find(V) != visited.end())
    return false;
  // Mark as visited
  visited.insert(V);
  // Skip constant.
  if (isa<Constant>(V))
    return false;
  // Skip phi which cannot sink.
  if (isa<PHINode>(V))
    return false;
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    BasicBlock *BB = I->getParent();
    if (PostDom.dominates(BB, &F->getEntryBlock())) {
      IRBuilder<> Builder(I->getNextNode());
      MarkConvergent(I, Builder, *F->getParent());
      return false;
    } else {
      // Propagete to each operand of I.
      for (Use &U : I->operands()) {
        PropagateConvergentImpl(U.get(), F, PostDom, visited);
      }
      // return true for report warning.
      // TODO: static indexing cbuffer is fine.
      return true;
    }
  } else {
    IRBuilder<> EntryBuilder(F->getEntryBlock().getFirstInsertionPt());
    MarkConvergent(V, EntryBuilder, *F->getParent());
    return false;
  }
}

Value *DxilConvergentMark::FindConvergentOperand(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (hlsl::GetHLOpcodeGroup(CI->getCalledFunction()) ==
        HLOpcodeGroup::HLIntrinsic) {
      IntrinsicOp IOP = static_cast<IntrinsicOp>(GetHLOpcode(CI));
      switch (IOP) {
      case IntrinsicOp::IOP_ddx:
      case IntrinsicOp::IOP_ddx_fine:
      case IntrinsicOp::IOP_ddx_coarse:
      case IntrinsicOp::IOP_ddy:
      case IntrinsicOp::IOP_ddy_fine:
      case IntrinsicOp::IOP_ddy_coarse:
        return CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx);
      case IntrinsicOp::MOP_Sample:
      case IntrinsicOp::MOP_SampleBias:
      case IntrinsicOp::MOP_SampleCmp:
      case IntrinsicOp::MOP_CalculateLevelOfDetail:
      case IntrinsicOp::MOP_CalculateLevelOfDetailUnclamped:
        return CI->getArgOperand(HLOperandIndex::kSampleCoordArgIndex);
      case IntrinsicOp::MOP_WriteSamplerFeedback:
      case IntrinsicOp::MOP_WriteSamplerFeedbackBias:
        return CI->getArgOperand(HLOperandIndex::kWriteSamplerFeedbackCoordArgIndex);
      default:
        // No other ops have convergent operands.
        break;
      }
    }
  }
  return nullptr;
}

} // namespace

INITIALIZE_PASS(DxilConvergentMark, "hlsl-dxil-convergent-mark",
                "Mark convergent", false, false)

ModulePass *llvm::createDxilConvergentMarkPass() {
  return new DxilConvergentMark();
}

namespace {

class DxilConvergentClear : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilConvergentClear() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DxilConvergentClear";
  }

  bool runOnModule(Module &M) override {
    std::vector<Function *> convergentList;
    for (Function &F : M.functions()) {
      if (F.getName().startswith(kConvergentFunctionPrefix)) {
        convergentList.emplace_back(&F);
      }
    }

    for (Function *F : convergentList) {
      ClearConvergent(F);
    }
    return convergentList.size();
  }

private:
  void ClearConvergent(Function *F);
};

char DxilConvergentClear::ID = 0;

void DxilConvergentClear::ClearConvergent(Function *F) {
  // Replace all users with arg.
  for (auto it = F->user_begin(); it != F->user_end();) {
    CallInst *CI = cast<CallInst>(*(it++));
    Value *arg = CI->getArgOperand(0);
    CI->replaceAllUsesWith(arg);
    CI->eraseFromParent();
  }

  F->eraseFromParent();
}

} // namespace

INITIALIZE_PASS(DxilConvergentClear, "hlsl-dxil-convergent-clear",
                "Clear convergent before dxil emit", false, false)

ModulePass *llvm::createDxilConvergentClearPass() {
  return new DxilConvergentClear();
}
