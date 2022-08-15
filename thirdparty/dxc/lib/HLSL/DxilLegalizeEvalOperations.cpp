///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilLegalizeEvalOperations.cpp                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HlslIntrinsicOp.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLOperations.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace hlsl;

// Make sure src of EvalOperations are from function parameter.
// This is needed in order to translate EvaluateAttribute operations that traces
// back to LoadInput operations during translation stage. Promoting load/store
// instructions beforehand will allow us to easily trace back to loadInput from
// function call.
namespace {

class DxilLegalizeEvalOperations : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilLegalizeEvalOperations() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXIL Legalize EvalOperations";
  }

  bool runOnModule(Module &M) override {
    for (Function &F : M.getFunctionList()) {
      hlsl::HLOpcodeGroup group = hlsl::GetHLOpcodeGroup(&F);
      if (group == HLOpcodeGroup::HLIntrinsic) {
        std::vector<CallInst *> EvalFunctionCalls;
        // Find all EvaluateAttribute calls
        for (User *U : F.users()) {
          if (CallInst *CI = dyn_cast<CallInst>(U)) {
            IntrinsicOp evalOp =
                static_cast<IntrinsicOp>(hlsl::GetHLOpcode(CI));
            if (evalOp == IntrinsicOp::IOP_EvaluateAttributeAtSample ||
                evalOp == IntrinsicOp::IOP_EvaluateAttributeCentroid ||
                evalOp == IntrinsicOp::IOP_EvaluateAttributeSnapped ||
                evalOp == IntrinsicOp::IOP_GetAttributeAtVertex) {
              EvalFunctionCalls.push_back(CI);
            }
          }
        }
        if (EvalFunctionCalls.empty()) {
          continue;
        }
        // Start from the call instruction, find all allocas that this call
        // uses.
        std::unordered_set<AllocaInst *> allocas;
        for (CallInst *CI : EvalFunctionCalls) {
          FindAllocasForEvalOperations(CI, allocas);
        }
        SSAUpdater SSA;
        SmallVector<Instruction *, 4> Insts;
        for (AllocaInst *AI : allocas) {
          for (User *user : AI->users()) {
            if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
              Insts.emplace_back(cast<Instruction>(user));
            }
          }
          LoadAndStorePromoter(Insts, SSA).run(Insts);
          Insts.clear();
        }
      }
    }
    return true;
  }

private:
  void FindAllocasForEvalOperations(Value *val,
                                    std::unordered_set<AllocaInst *> &allocas);
};

char DxilLegalizeEvalOperations::ID = 0;

// Find allocas for EvaluateAttribute operations
void DxilLegalizeEvalOperations::FindAllocasForEvalOperations(
    Value *val, std::unordered_set<AllocaInst *> &allocas) {
  Value *CurVal = val;
  while (!isa<AllocaInst>(CurVal)) {
    if (CallInst *CI = dyn_cast<CallInst>(CurVal)) {
      CurVal = CI->getOperand(HLOperandIndex::kUnaryOpSrc0Idx);
    } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(CurVal)) {
      Value *arg0 =
          IE->getOperand(0); // Could be another insertelement or undef
      Value *arg1 = IE->getOperand(1);
      FindAllocasForEvalOperations(arg0, allocas);
      CurVal = arg1;
    } else if (ShuffleVectorInst *SV = dyn_cast<ShuffleVectorInst>(CurVal)) {
      Value *arg0 = SV->getOperand(0);
      Value *arg1 = SV->getOperand(1);
      FindAllocasForEvalOperations(
          arg0, allocas); // Shuffle vector could come from different allocas
      CurVal = arg1;
    } else if (ExtractElementInst *EE = dyn_cast<ExtractElementInst>(CurVal)) {
      CurVal = EE->getOperand(0);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(CurVal)) {
      CurVal = LI->getOperand(0);
    } else {
      break;
    }
  }
  if (AllocaInst *AI = dyn_cast<AllocaInst>(CurVal)) {
    allocas.insert(AI);
  }
}
} // namespace

ModulePass *llvm::createDxilLegalizeEvalOperationsPass() {
  return new DxilLegalizeEvalOperations();
}

INITIALIZE_PASS(DxilLegalizeEvalOperations,
                "hlsl-dxil-legalize-eval-operations",
                "DXIL legalize eval operations", false, false)
