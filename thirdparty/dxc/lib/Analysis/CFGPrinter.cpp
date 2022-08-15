//===- CFGPrinter.cpp - DOT printer for the control flow graph ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a '-dot-cfg' analysis pass, which emits the
// cfg.<fnname>.dot file for each function in the program, with a graph of the
// CFG for that function.
//
// The other main feature of this file is that it implements the
// Function::viewCFG method, which is useful for debugging passes which operate
// on the CFG.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
using namespace llvm;

namespace {
  struct CFGViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGViewer() : FunctionPass(ID) {
      // initializeCFGOnlyViewerPass(*PassRegistry::getPassRegistry()); // HLSL Change - initialize up front
    }

    bool runOnFunction(Function &F) override {
      // HLSL Change Starts
      if (OSOverride != nullptr) {
        *OSOverride << "\ngraph: " << "cfg" << F.getName() << ".dot\n";
        llvm::WriteGraph(*OSOverride, (const Function*)&F, false, F.getName());
        return false;
      }
      // HLSL Change Ends
      F.viewCFG();
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGViewer::ID = 0;
INITIALIZE_PASS(CFGViewer, "view-cfg", "View CFG of function", false, true)

namespace {
  struct CFGOnlyViewer : public FunctionPass {
    static char ID; // Pass identifcation, replacement for typeid
    CFGOnlyViewer() : FunctionPass(ID) {
      // initializeCFGOnlyViewerPass(*PassRegistry::getPassRegistry()); // HLSL Change - initialize up front
    }

    bool runOnFunction(Function &F) override {
      // HLSL Change Starts
      if (OSOverride != nullptr) {
        *OSOverride << "\ngraph: " << "cfg" << F.getName() << ".dot\n";
        llvm::WriteGraph(*OSOverride, (const Function*)&F, true, F.getName());
        return false;
      }
      // HLSL Change Ends
      F.viewCFGOnly();
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyViewer::ID = 0;
INITIALIZE_PASS(CFGOnlyViewer, "view-cfg-only",
                "View CFG of function (with no function bodies)", false, true)

namespace {
  struct CFGPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGPrinter() : FunctionPass(ID) {
      // initializeCFGPrinterPass(*PassRegistry::getPassRegistry()); // HLSL Change - initialize up front
    }

    bool runOnFunction(Function &F) override {
      // HLSL Change Starts
      if (OSOverride != nullptr) {
        *OSOverride << "\ngraph: " << "cfg." << F.getName() << ".dot\n";
        llvm::WriteGraph(*OSOverride, (const Function*)&F, false, F.getName());
        return false;
      }
      // HLSL Change Ends

      std::string Filename = ("cfg." + F.getName() + ".dot").str();
      errs() << "Writing '" << Filename << "'...";

      std::error_code EC;
      raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

      if (!EC)
        WriteGraph(File, (const Function*)&F);
      else
        errs() << "  error opening file for writing!";
      errs() << "\n";
      return false;
    }

    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGPrinter::ID = 0;
INITIALIZE_PASS(CFGPrinter, "dot-cfg", "Print CFG of function to 'dot' file", 
                false, true)

namespace {
  struct CFGOnlyPrinter : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    CFGOnlyPrinter() : FunctionPass(ID) {
      // initializeCFGOnlyPrinterPass(*PassRegistry::getPassRegistry()); // HLSL Change - initialize up front
    }

    bool runOnFunction(Function &F) override {
      // HLSL Change Starts
      if (OSOverride != nullptr) {
        *OSOverride << "\ngraph: " << "cfg." << F.getName() << ".dot\n";
        llvm::WriteGraph(*OSOverride, (const Function*)&F, true, F.getName());
        return false;
      }
      // HLSL Change Ends
      std::string Filename = ("cfg." + F.getName() + ".dot").str();
      errs() << "Writing '" << Filename << "'...";

      std::error_code EC;
      raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

      if (!EC)
        WriteGraph(File, (const Function*)&F, true);
      else
        errs() << "  error opening file for writing!";
      errs() << "\n";
      return false;
    }
    void print(raw_ostream &OS, const Module* = nullptr) const override {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }
  };
}

char CFGOnlyPrinter::ID = 0;
INITIALIZE_PASS(CFGOnlyPrinter, "dot-cfg-only",
   "Print CFG of function to 'dot' file (with no function bodies)",
   false, true)

/// viewCFG - This function is meant for use from the debugger.  You can just
/// say 'call F->viewCFG()' and a ghostview window should pop up from the
/// program, displaying the CFG of the current function.  This depends on there
/// being a 'dot' and 'gv' program in your path.
///
void Function::viewCFG() const {
  ViewGraph(this, "cfg" + getName());
}

/// viewCFGOnly - This function is meant for use from the debugger.  It works
/// just like viewCFG, but it does not include the contents of basic blocks
/// into the nodes, just the label.  If you are only interested in the CFG
/// this can make the graph smaller.
///
void Function::viewCFGOnly() const {
  ViewGraph(this, "cfg" + getName(), true);
}

FunctionPass *llvm::createCFGPrinterPass () {
  return new CFGPrinter();
}

FunctionPass *llvm::createCFGOnlyPrinterPass () {
  return new CFGOnlyPrinter();
}

// HLSL Change Starts
void llvm::initializeCFGPrinterPasses(PassRegistry &Registry) {
  initializeCFGPrinterPass(Registry);
  initializeCFGOnlyPrinterPass(Registry);
  initializeCFGViewerPass(Registry);
  initializeCFGOnlyViewerPass(Registry);
}
// HLSL Change Ends
