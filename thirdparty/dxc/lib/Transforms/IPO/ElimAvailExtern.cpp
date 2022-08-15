//===-- ElimAvailExtern.cpp - DCE unreachable internal functions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transform is designed to eliminate available external global
// definitions from the program, turning them into declarations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Pass.h"
using namespace llvm;

#define DEBUG_TYPE "elim-avail-extern"

STATISTIC(NumFunctions, "Number of functions removed");
STATISTIC(NumVariables, "Number of global variables removed");

namespace {
  struct EliminateAvailableExternally : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    EliminateAvailableExternally() : ModulePass(ID) {
      initializeEliminateAvailableExternallyPass(
          *PassRegistry::getPassRegistry());
    }

    // run - Do the EliminateAvailableExternally pass on the specified module,
    // optionally updating the specified callgraph to reflect the changes.
    //
    bool runOnModule(Module &M) override;
  };
}

char EliminateAvailableExternally::ID = 0;
INITIALIZE_PASS(EliminateAvailableExternally, "elim-avail-extern",
                "Eliminate Available Externally Globals", false, false)

ModulePass *llvm::createEliminateAvailableExternallyPass() {
  return new EliminateAvailableExternally();
}

bool EliminateAvailableExternally::runOnModule(Module &M) {
  bool Changed = false;

  // Drop initializers of available externally global variables.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasAvailableExternallyLinkage())
      continue;
    if (I->hasInitializer()) {
      Constant *Init = I->getInitializer();
      I->setInitializer(nullptr);
      if (isSafeToDestroyConstant(Init))
        Init->destroyConstant();
    }
    I->removeDeadConstantUsers();
    I->setLinkage(GlobalValue::ExternalLinkage);
    NumVariables++;
  }

  // Drop the bodies of available externally functions.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (!I->hasAvailableExternallyLinkage())
      continue;
    if (!I->isDeclaration())
      // This will set the linkage to external
      I->deleteBody();
    I->removeDeadConstantUsers();
    NumFunctions++;
  }

  return Changed;
}
