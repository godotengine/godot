//===--- IRPrintingPasses.cpp - Module and Function printing passes -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PrintModulePass and PrintFunctionPass implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/AssemblyAnnotationWriter.h" // HLSL Change
#include "llvm/IR/DebugInfoMetadata.h" // HLSL Change
#include "llvm/IR/IntrinsicInst.h" // HLSL Change
#include "llvm/Support/FormattedStream.h" // HLSL Change
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// HLSL Change - Begin
namespace {
class DxilAAW : public llvm::AssemblyAnnotationWriter {
public:
  ~DxilAAW() {}
  void printInfoComment(const Value &V, formatted_raw_ostream &OS) override {
    using namespace llvm;
    if (const Instruction *I = dyn_cast<Instruction>(&V)) {
      if (isa<DbgInfoIntrinsic>(I)) {
        DILocalVariable *Var = nullptr;
        DIExpression *Expr = nullptr;
        if (const DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(I)) {
          Var = DI->getVariable();
          Expr = DI->getExpression();
        }
        else if (const DbgValueInst *DI = dyn_cast<DbgValueInst>(I)) {
          Var = DI->getVariable();
          Expr = DI->getExpression();
        }

        if (Var && Expr) {
          OS << " ; var:\"" << Var->getName() << "\"" << " ";
          Expr->printAsBody(OS);
        }
      }
      else {
        DebugLoc Loc = I->getDebugLoc();
        if (Loc && Loc.getLine() != 0)
          OS << " ; line:" << Loc.getLine() << " col:" << Loc.getCol();
      }
    }
  }
};

}
// HLSL Change - End

PrintModulePass::PrintModulePass() : OS(dbgs()) {}
PrintModulePass::PrintModulePass(raw_ostream &OS, const std::string &Banner,
                                 bool ShouldPreserveUseListOrder)
    : OS(OS), Banner(Banner),
      ShouldPreserveUseListOrder(ShouldPreserveUseListOrder) {}

PreservedAnalyses PrintModulePass::run(Module &M) {
  DxilAAW AAW; // HLSL Change
  OS << Banner;
  M.print(OS, &AAW, ShouldPreserveUseListOrder); // HLSL Change
  return PreservedAnalyses::all();
}

PrintFunctionPass::PrintFunctionPass() : OS(dbgs()) {}
PrintFunctionPass::PrintFunctionPass(raw_ostream &OS, const std::string &Banner)
    : OS(OS), Banner(Banner) {}

PreservedAnalyses PrintFunctionPass::run(Function &F) {
  OS << Banner << static_cast<Value &>(F);
  return PreservedAnalyses::all();
}

namespace {

class PrintModulePassWrapper : public ModulePass {
  PrintModulePass P;

public:
  static char ID;
  PrintModulePassWrapper() : ModulePass(ID) {}
  PrintModulePassWrapper(raw_ostream &OS, const std::string &Banner,
                         bool ShouldPreserveUseListOrder)
      : ModulePass(ID), P(OS, Banner, ShouldPreserveUseListOrder) {}

  bool runOnModule(Module &M) override {
    P.run(M);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

class PrintFunctionPassWrapper : public FunctionPass {
  PrintFunctionPass P;

public:
  static char ID;
  PrintFunctionPassWrapper() : FunctionPass(ID) {}
  PrintFunctionPassWrapper(raw_ostream &OS, const std::string &Banner)
      : FunctionPass(ID), P(OS, Banner) {}

  // This pass just prints a banner followed by the function as it's processed.
  bool runOnFunction(Function &F) override {
    P.run(F);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

class PrintBasicBlockPass : public BasicBlockPass {
  raw_ostream &Out;
  std::string Banner;

public:
  static char ID;
  PrintBasicBlockPass() : BasicBlockPass(ID), Out(dbgs()) {}
  PrintBasicBlockPass(raw_ostream &Out, const std::string &Banner)
      : BasicBlockPass(ID), Out(Out), Banner(Banner) {}

  bool runOnBasicBlock(BasicBlock &BB) override {
    Out << Banner << BB;
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

}

char PrintModulePassWrapper::ID = 0;
INITIALIZE_PASS(PrintModulePassWrapper, "print-module",
                "Print module to stderr", false, false)
char PrintFunctionPassWrapper::ID = 0;
INITIALIZE_PASS(PrintFunctionPassWrapper, "print-function",
                "Print function to stderr", false, false)
char PrintBasicBlockPass::ID = 0;
INITIALIZE_PASS(PrintBasicBlockPass, "print-bb", "Print BB to stderr", false,
                false)

ModulePass *llvm::createPrintModulePass(llvm::raw_ostream &OS,
                                        const std::string &Banner,
                                        bool ShouldPreserveUseListOrder) {
  return new PrintModulePassWrapper(OS, Banner, ShouldPreserveUseListOrder);
}

FunctionPass *llvm::createPrintFunctionPass(llvm::raw_ostream &OS,
                                            const std::string &Banner) {
  return new PrintFunctionPassWrapper(OS, Banner);
}

BasicBlockPass *llvm::createPrintBasicBlockPass(llvm::raw_ostream &OS,
                                                const std::string &Banner) {
  return new PrintBasicBlockPass(OS, Banner);
}
