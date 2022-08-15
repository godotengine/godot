///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLPreprocess.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Preprocess HLModule after inline.                                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"

#include "dxc/HLSL/DxilGenerationPass.h"

using namespace llvm;

///////////////////////////////////////////////////////////////////////////////
// HLPreprocess.
// Inliner will create stacksave stackstore if there are allocas inside block
// other than entry block. HLPreprocess will remove stacksave and stackstore and
// put all allocas inside entry block.
//
namespace {

class HLPreprocess : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit HLPreprocess() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "Preprocess HLModule after inline";
  }

  bool runOnModule(Module &M) override {
    bool bUpdated = false;
    // Remove stacksave and stackstore.
    // Get the two intrinsics we care about.
    Function *StackSave = Intrinsic::getDeclaration(&M, Intrinsic::stacksave);
    Function *StackRestore =
        Intrinsic::getDeclaration(&M, Intrinsic::stackrestore);
    // If has user, remove user first.
    if (!StackSave->user_empty() || !StackRestore->user_empty()) {
      for (auto it = StackRestore->user_begin();
           it != StackRestore->user_end();) {
        Instruction *I = cast<Instruction>(*(it++));
        I->eraseFromParent();
      }

      for (auto it = StackSave->user_begin(); it != StackSave->user_end(); ) {
        Instruction *I = cast<Instruction>(*(it++));
        I->eraseFromParent();
      }
      bUpdated = true;
    }

    StackSave->eraseFromParent();
    StackRestore->eraseFromParent();

    // If stacksave/store is present, it means alloca not in the
    // entry block. However, there could be other cases where allocas
    // could be present in the non-entry blocks.
    // Therefore, always go through all non-entry blocks and
    // make sure all allocas are moved to the entry block.
    for (Function &F : M.functions()) {
      bUpdated |= MoveAllocasToEntryBlock(&F);
    }

    return bUpdated;
  }

private:
  bool MoveAllocasToEntryBlock(Function *F);
};

char HLPreprocess::ID = 0;

// Make sure all allocas are in entry block.
bool HLPreprocess::MoveAllocasToEntryBlock(Function *F) {
  bool changed = false;
  if (F->getBasicBlockList().size() < 2)
    return changed;
  BasicBlock &Entry = F->getEntryBlock();
  IRBuilder<> Builder(Entry.getFirstInsertionPt());

  for (auto bb = F->begin(); bb != F->end(); bb++) {
    BasicBlock *BB = bb;
    if (BB == &Entry)
      continue;
    for (auto it = BB->begin(); it != BB->end();) {
      Instruction *I = (it++);
      if (isa<AllocaInst>(I)) {
        I->removeFromParent();
        Builder.Insert(I);
        changed = true;
      }
    }
  }
  return changed;
}

} // namespace

ModulePass *llvm::createHLPreprocessPass() { return new HLPreprocess(); }

INITIALIZE_PASS(HLPreprocess, "hl-preprocess",
                "Preprocess HLModule after inline", false, false)
