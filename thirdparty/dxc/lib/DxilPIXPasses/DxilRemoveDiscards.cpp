///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRemoveDiscards.cpp                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to remove all instances of the discard instruction        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;
using namespace hlsl;

class DxilRemoveDiscards : public ModulePass {

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilRemoveDiscards() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "DXIL Remove all discard instructions";
  }
  bool runOnModule(Module &M) override;
};

bool DxilRemoveDiscards::runOnModule(Module &M) {
  // This pass removes all instances of the discard instruction within the
  // shader.
  DxilModule &DM = M.GetOrCreateDxilModule();

  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();
  Function *DiscardFunction =
      HlslOP->GetOpFunc(DXIL::OpCode::Discard, Type::getVoidTy(Ctx));
  auto DiscardFunctionUses = DiscardFunction->uses();

  bool Modified = false;

  for (auto FI = DiscardFunctionUses.begin();
       FI != DiscardFunctionUses.end();) {
    auto &FunctionUse = *FI++;
    auto FunctionUser = FunctionUse.getUser();
    auto instruction = cast<Instruction>(FunctionUser);
    instruction->eraseFromParent();
    Modified = true;
  }

  return Modified;
}

char DxilRemoveDiscards::ID = 0;

ModulePass *llvm::createDxilRemoveDiscardsPass() {
  return new DxilRemoveDiscards();
}

INITIALIZE_PASS(DxilRemoveDiscards, "hlsl-dxil-remove-discards",
                "HLSL DXIL Remove all discard instructions", false, false)
