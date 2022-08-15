//===- AliasDebugger.cpp - Simple Alias Analysis Use Checker --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This simple pass checks alias analysis users to ensure that if they
// create a new value, they do not query AA without informing it of the value.
// It acts as a shim over any other AA pass you want.
//
// Yes keeping track of every value in the program is expensive, but this is 
// a debugging pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <set>
using namespace llvm;

namespace {
  
  class AliasDebugger : public ModulePass, public AliasAnalysis {

    //What we do is simple.  Keep track of every value the AA could
    //know about, and verify that queries are one of those.
    //A query to a value that didn't exist when the AA was created
    //means someone forgot to update the AA when creating new values

    std::set<const Value*> Vals;
    
  public:
    static char ID; // Class identification, replacement for typeinfo
    AliasDebugger() : ModulePass(ID) {
      initializeAliasDebuggerPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override {
      InitializeAliasAnalysis(this, &M.getDataLayout()); // set up super class

      for(Module::global_iterator I = M.global_begin(),
            E = M.global_end(); I != E; ++I) {
        Vals.insert(&*I);
        for (User::const_op_iterator OI = I->op_begin(),
             OE = I->op_end(); OI != OE; ++OI)
          Vals.insert(*OI);
      }

      for(Module::iterator I = M.begin(),
            E = M.end(); I != E; ++I){
        Vals.insert(&*I);
        if(!I->isDeclaration()) {
          for (Function::arg_iterator AI = I->arg_begin(), AE = I->arg_end();
               AI != AE; ++AI) 
            Vals.insert(&*AI);     
          for (Function::const_iterator FI = I->begin(), FE = I->end();
               FI != FE; ++FI) 
            for (BasicBlock::const_iterator BI = FI->begin(), BE = FI->end();
                 BI != BE; ++BI) {
              Vals.insert(&*BI);
              for (User::const_op_iterator OI = BI->op_begin(),
                   OE = BI->op_end(); OI != OE; ++OI)
                Vals.insert(*OI);
            }
        }
        
      }
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                         // Does not transform code
    }

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    void *getAdjustedAnalysisPointer(AnalysisID PI) override {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
    
    //------------------------------------------------
    // Implement the AliasAnalysis API
    //
    AliasResult alias(const MemoryLocation &LocA,
                      const MemoryLocation &LocB) override {
      assert(Vals.find(LocA.Ptr) != Vals.end() &&
             "Never seen value in AA before");
      assert(Vals.find(LocB.Ptr) != Vals.end() &&
             "Never seen value in AA before");
      return AliasAnalysis::alias(LocA, LocB);
    }

    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) override {
      assert(Vals.find(Loc.Ptr) != Vals.end() && "Never seen value in AA before");
      return AliasAnalysis::getModRefInfo(CS, Loc);
    }

    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) override {
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }

    bool pointsToConstantMemory(const MemoryLocation &Loc,
                                bool OrLocal) override {
      assert(Vals.find(Loc.Ptr) != Vals.end() && "Never seen value in AA before");
      return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
    }

    void deleteValue(Value *V) override {
      assert(Vals.find(V) != Vals.end() && "Never seen value in AA before");
      AliasAnalysis::deleteValue(V);
    }

  };
}

char AliasDebugger::ID = 0;
INITIALIZE_AG_PASS(AliasDebugger, AliasAnalysis, "debug-aa",
                   "AA use debugger", false, true, false)

Pass *llvm::createAliasDebugger() { return new AliasDebugger(); }

