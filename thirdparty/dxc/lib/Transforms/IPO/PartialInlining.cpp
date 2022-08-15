//===- PartialInlining.cpp - Inline parts of functions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs partial inlining, typically by inlining an if statement
// that surrounds the body of the function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
using namespace llvm;

#define DEBUG_TYPE "partialinlining"

STATISTIC(NumPartialInlined, "Number of functions partially inlined");

namespace {
  struct PartialInliner : public ModulePass {
    void getAnalysisUsage(AnalysisUsage &AU) const override { }
    static char ID; // Pass identification, replacement for typeid
    PartialInliner() : ModulePass(ID) {
      initializePartialInlinerPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module& M) override;

  private:
    Function* unswitchFunction(Function* F);
  };
}

char PartialInliner::ID = 0;
INITIALIZE_PASS(PartialInliner, "partial-inliner",
                "Partial Inliner", false, false)

ModulePass* llvm::createPartialInliningPass() { return new PartialInliner(); }

Function* PartialInliner::unswitchFunction(Function* F) {
  // First, verify that this function is an unswitching candidate...
  BasicBlock* entryBlock = F->begin();
  BranchInst *BR = dyn_cast<BranchInst>(entryBlock->getTerminator());
  if (!BR || BR->isUnconditional())
    return nullptr;
  
  BasicBlock* returnBlock = nullptr;
  BasicBlock* nonReturnBlock = nullptr;
  unsigned returnCount = 0;
  for (BasicBlock *BB : successors(entryBlock)) {
    if (isa<ReturnInst>(BB->getTerminator())) {
      returnBlock = BB;
      returnCount++;
    } else
      nonReturnBlock = BB;
  }
  
  if (returnCount != 1)
    return nullptr;
  
  // Clone the function, so that we can hack away on it.
  ValueToValueMapTy VMap;
  Function* duplicateFunction = CloneFunction(F, VMap,
                                              /*ModuleLevelChanges=*/false);
  duplicateFunction->setLinkage(GlobalValue::InternalLinkage);
  F->getParent()->getFunctionList().push_back(duplicateFunction);
  BasicBlock* newEntryBlock = cast<BasicBlock>(VMap[entryBlock]);
  BasicBlock* newReturnBlock = cast<BasicBlock>(VMap[returnBlock]);
  BasicBlock* newNonReturnBlock = cast<BasicBlock>(VMap[nonReturnBlock]);
  
  // Go ahead and update all uses to the duplicate, so that we can just
  // use the inliner functionality when we're done hacking.
  F->replaceAllUsesWith(duplicateFunction);
  
  // Special hackery is needed with PHI nodes that have inputs from more than
  // one extracted block.  For simplicity, just split the PHIs into a two-level
  // sequence of PHIs, some of which will go in the extracted region, and some
  // of which will go outside.
  BasicBlock* preReturn = newReturnBlock;
  newReturnBlock = newReturnBlock->splitBasicBlock(
                                              newReturnBlock->getFirstNonPHI());
  BasicBlock::iterator I = preReturn->begin();
  BasicBlock::iterator Ins = newReturnBlock->begin();
  while (I != preReturn->end()) {
    PHINode* OldPhi = dyn_cast<PHINode>(I);
    if (!OldPhi) break;
    
    PHINode* retPhi = PHINode::Create(OldPhi->getType(), 2, "", Ins);
    OldPhi->replaceAllUsesWith(retPhi);
    Ins = newReturnBlock->getFirstNonPHI();
    
    retPhi->addIncoming(I, preReturn);
    retPhi->addIncoming(OldPhi->getIncomingValueForBlock(newEntryBlock),
                        newEntryBlock);
    OldPhi->removeIncomingValue(newEntryBlock);
    
    ++I;
  }
  newEntryBlock->getTerminator()->replaceUsesOfWith(preReturn, newReturnBlock);
  
  // Gather up the blocks that we're going to extract.
  std::vector<BasicBlock*> toExtract;
  toExtract.push_back(newNonReturnBlock);
  for (Function::iterator FI = duplicateFunction->begin(),
       FE = duplicateFunction->end(); FI != FE; ++FI)
    if (&*FI != newEntryBlock && &*FI != newReturnBlock &&
        &*FI != newNonReturnBlock)
      toExtract.push_back(FI);
      
  // The CodeExtractor needs a dominator tree.
  DominatorTree DT;
  DT.recalculate(*duplicateFunction);

  // Extract the body of the if.
  Function* extractedFunction
    = CodeExtractor(toExtract, &DT).extractCodeRegion();
  
  InlineFunctionInfo IFI;
  
  // Inline the top-level if test into all callers.
  std::vector<User *> Users(duplicateFunction->user_begin(),
                            duplicateFunction->user_end());
  for (std::vector<User*>::iterator UI = Users.begin(), UE = Users.end();
       UI != UE; ++UI)
    if (CallInst *CI = dyn_cast<CallInst>(*UI))
      InlineFunction(CI, IFI);
    else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI))
      InlineFunction(II, IFI);
  
  // Ditch the duplicate, since we're done with it, and rewrite all remaining
  // users (function pointers, etc.) back to the original function.
  duplicateFunction->replaceAllUsesWith(F);
  duplicateFunction->eraseFromParent();
  
  ++NumPartialInlined;
  
  return extractedFunction;
}

bool PartialInliner::runOnModule(Module& M) {
  std::vector<Function*> worklist;
  worklist.reserve(M.size());
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    if (!FI->use_empty() && !FI->isDeclaration())
      worklist.push_back(&*FI);
    
  bool changed = false;
  while (!worklist.empty()) {
    Function* currFunc = worklist.back();
    worklist.pop_back();
  
    if (currFunc->use_empty()) continue;
    
    bool recursive = false;
    for (User *U : currFunc->users())
      if (Instruction* I = dyn_cast<Instruction>(U))
        if (I->getParent()->getParent() == currFunc) {
          recursive = true;
          break;
        }
    if (recursive) continue;
          
    
    if (Function* newFunc = unswitchFunction(currFunc)) {
      worklist.push_back(newFunc);
      changed = true;
    }
    
  }
  
  return changed;
}
