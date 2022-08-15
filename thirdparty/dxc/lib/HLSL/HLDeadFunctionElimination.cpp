///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLDeadFunctionElimination.cpp                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/HLModule.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;
using namespace hlsl;

namespace {
class HLDeadFunctionElimination : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit HLDeadFunctionElimination () : ModulePass(ID) {}

  StringRef getPassName() const override { return "Remove all unused function except entry from HLModule"; }

  bool runOnModule(Module &M) override {
    if (M.HasHLModule()) {
      HLModule &HLM = M.GetHLModule();

      bool IsLib = HLM.GetShaderModel()->IsLib();
      // Remove unused functions except entry and patch constant func.
      // For library profile, only remove unused external functions.
      Function *EntryFunc = HLM.GetEntryFunction();
      Function *PatchConstantFunc = HLM.GetPatchConstantFunction();

      bool bChanged = false;
      while (dxilutil::RemoveUnusedFunctions(M, EntryFunc, PatchConstantFunc,
                                             IsLib))
        bChanged = true;
      return bChanged;
    }

    return false;
  }
};
}

char HLDeadFunctionElimination::ID = 0;

ModulePass *llvm::createHLDeadFunctionEliminationPass() {
  return new HLDeadFunctionElimination();
}

INITIALIZE_PASS(HLDeadFunctionElimination, "hl-dfe", "Remove all unused function except entry from HLModule", false, false)
