//===- BitcodeWriterPass.cpp - Bitcode writing pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BitcodeWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
using namespace llvm;

PreservedAnalyses BitcodeWriterPass::run(Module &M) {
  WriteBitcodeToFile(&M, OS, ShouldPreserveUseListOrder);
  return PreservedAnalyses::all();
}

namespace {
  class WriteBitcodePass : public ModulePass {
    raw_ostream &OS; // raw_ostream to print on
    bool ShouldPreserveUseListOrder;

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit WriteBitcodePass(raw_ostream &o, bool ShouldPreserveUseListOrder)
        : ModulePass(ID), OS(o),
          ShouldPreserveUseListOrder(ShouldPreserveUseListOrder) {}

    StringRef getPassName() const override { return "Bitcode Writer"; }

    bool runOnModule(Module &M) override {
      WriteBitcodeToFile(&M, OS, ShouldPreserveUseListOrder);
      return false;
    }
  };
}

char WriteBitcodePass::ID = 0;

ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str,
                                          bool ShouldPreserveUseListOrder) {
  return new WriteBitcodePass(Str, ShouldPreserveUseListOrder);
}
