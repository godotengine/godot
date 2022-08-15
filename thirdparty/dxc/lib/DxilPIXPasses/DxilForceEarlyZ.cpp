///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilOutputColorBecomesConstant.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to turn on the early-z flag                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace hlsl;

class DxilForceEarlyZ : public ModulePass {

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilForceEarlyZ() : ModulePass(ID) {}
  StringRef getPassName() const override { return "DXIL Force Early Z"; }
  bool runOnModule(Module &M) override;
};

bool DxilForceEarlyZ::runOnModule(Module &M) {
  // This pass adds the force-early-z flag

  DxilModule &DM = M.GetOrCreateDxilModule();

  DM.m_ShaderFlags.SetForceEarlyDepthStencil(true);

  DM.ReEmitDxilResources();

  return true;
}

char DxilForceEarlyZ::ID = 0;

ModulePass *llvm::createDxilForceEarlyZPass() { return new DxilForceEarlyZ(); }

INITIALIZE_PASS(
    DxilForceEarlyZ, "hlsl-dxil-force-early-z",
    "HLSL DXIL Force the early Z global flag, if shader has no discard calls",
    false, false)
