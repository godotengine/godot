//===- MetaRenamer.cpp - Rename everything with metasyntatic names --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass renames everything with metasyntatic names. The intent is to use
// this pass after bugpoint reduction to conceal the nature of the original
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {

  // This PRNG is from the ISO C spec. It is intentionally simple and
  // unsuitable for cryptographic use. We're just looking for enough
  // variety to surprise and delight users.
  struct PRNG {
    unsigned long next;

    void srand(unsigned int seed) {
      next = seed;
    }

    int rand() {
      next = next * 1103515245 + 12345;
      return (unsigned int)(next / 65536) % 32768;
    }
  };

  struct MetaRenamer : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    MetaRenamer() : ModulePass(ID) {
      initializeMetaRenamerPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }

    bool runOnModule(Module &M) override {
      static const char *const metaNames[] = {
        // See http://en.wikipedia.org/wiki/Metasyntactic_variable
        "foo", "bar", "baz", "quux", "barney", "snork", "zot", "blam", "hoge",
        "wibble", "wobble", "widget", "wombat", "ham", "eggs", "pluto", "spam"
      };

      // Seed our PRNG with simple additive sum of ModuleID. We're looking to
      // simply avoid always having the same function names, and we need to
      // remain deterministic.
      unsigned int randSeed = 0;
      for (std::string::const_iterator I = M.getModuleIdentifier().begin(),
           E = M.getModuleIdentifier().end(); I != E; ++I)
        randSeed += *I;

      PRNG prng;
      prng.srand(randSeed);

      // Rename all aliases
      for (Module::alias_iterator AI = M.alias_begin(), AE = M.alias_end();
           AI != AE; ++AI) {
        StringRef Name = AI->getName();
        if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1))
          continue;

        AI->setName("alias");
      }
      
      // Rename all global variables
      for (Module::global_iterator GI = M.global_begin(), GE = M.global_end();
           GI != GE; ++GI) {
        StringRef Name = GI->getName();
        if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1))
          continue;

        GI->setName("global");
      }

      // Rename all struct types
      TypeFinder StructTypes;
      StructTypes.run(M, true);
      for (unsigned i = 0, e = StructTypes.size(); i != e; ++i) {
        StructType *STy = StructTypes[i];
        if (STy->isLiteral() || STy->getName().empty()) continue;

        SmallString<128> NameStorage;
        STy->setName((Twine("struct.") + metaNames[prng.rand() %
                     array_lengthof(metaNames)]).toStringRef(NameStorage));
      }

      // Rename all functions
      for (Module::iterator FI = M.begin(), FE = M.end();
           FI != FE; ++FI) {
        StringRef Name = FI->getName();
        if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1))
          continue;

        FI->setName(metaNames[prng.rand() % array_lengthof(metaNames)]);
        runOnFunction(*FI);
      }
      return true;
    }

    bool runOnFunction(Function &F) {
      for (Function::arg_iterator AI = F.arg_begin(), AE = F.arg_end();
           AI != AE; ++AI)
        if (!AI->getType()->isVoidTy())
          AI->setName("arg");

      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
        BB->setName("bb");

        for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
          if (!I->getType()->isVoidTy())
            I->setName("tmp");
      }
      return true;
    }
  };
}

char MetaRenamer::ID = 0;
INITIALIZE_PASS(MetaRenamer, "metarenamer", 
                "Assign new names to everything", false, false)
//===----------------------------------------------------------------------===//
//
// MetaRenamer - Rename everything with metasyntactic names.
//
ModulePass *llvm::createMetaRenamerPass() {
  return new MetaRenamer();
}
