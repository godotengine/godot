///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRenameResourcesPass.cpp                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilGenerationPass.h"

using namespace llvm;
using namespace hlsl;

// Rename resources with prefix

namespace {

class DxilRenameResources : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilRenameResources()
      : ModulePass(ID) {}

  void applyOptions(PassOptions O) override {
    GetPassOptionBool(O, "from-binding", &m_bFromBinding, false);
    GetPassOptionBool(O, "keep-name", &m_bKeepName, true);
    StringRef prefix;
    GetPassOption(O, "prefix", &prefix);
    m_Prefix = prefix.str();
  }

  StringRef getPassName() const override {
    return "DXIL rename resources";
  }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    bool bChanged = false;
    if (m_bFromBinding) {
      bChanged = DM.RenameResourceGlobalsWithBinding(m_bKeepName);
    }
    if (!m_Prefix.empty()) {
      bChanged |= DM.RenameResourcesWithPrefix(m_Prefix);
    }
    if (bChanged) {
      DM.ReEmitDxilResources();
    }
    return bChanged;
  }

private:
  bool m_bFromBinding;
  bool m_bKeepName;
  std::string m_Prefix;
};

char DxilRenameResources::ID = 0;

}

ModulePass *llvm::createDxilRenameResourcesPass() {
  return new DxilRenameResources();
}

INITIALIZE_PASS(DxilRenameResources,
                "dxil-rename-resources",
                "DXIL rename resources", false, false)
