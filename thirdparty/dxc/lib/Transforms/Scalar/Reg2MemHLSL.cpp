//===- Reg2MemHLSL.cpp - Convert registers to allocas ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
#include <list>
using namespace llvm;

#define DEBUG_TYPE "reg2mem_hlsl"

STATISTIC(NumRegsDemotedHlsl, "Number of registers demoted");
STATISTIC(NumPhisDemotedHlsl, "Number of phi-nodes demoted");

namespace {
  struct RegToMemHlsl : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    RegToMemHlsl() : FunctionPass(ID) {
      initializeRegToMemHlslPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
    }

    bool valueEscapes(const Instruction *Inst) const {
      for (const User *U : Inst->users()) {
        const Instruction *UI = cast<Instruction>(U);
        if (isa<PHINode>(UI))
          return true;
      }
      return false;
    }

    bool runOnFunction(Function &F) override;
  };

  /// DemotePHIToStack - This function takes a virtual register computed by a
  /// PHI node and replaces it with a slot in the stack frame allocated via
  /// alloca.
  /// The PHI node is deleted. It returns the pointer to the alloca inserted.
  /// The difference of HLSL version is the new Alloca will be loaded for each
  /// use, for case a phi inside loop be used outside the loop.
  AllocaInst *DemotePHIToStack_HLSL(PHINode *P, Instruction *AllocaPoint) {
    if (P->use_empty()) {
      P->eraseFromParent();
      return nullptr;
    }

    IRBuilder<> AllocaBuilder(P);
    if (!AllocaPoint) {
      Function *F = P->getParent()->getParent();
      AllocaPoint = F->getEntryBlock().begin();
    }
    AllocaBuilder.SetInsertPoint(AllocaPoint);

    // Create a stack slot to hold the value.
    AllocaInst *Slot = AllocaBuilder.CreateAlloca(P->getType(), nullptr, P->getName() + ".reg2mem");

    // Insert a load in place of the PHI and replace all uses.
    BasicBlock::iterator InsertPt = P;

    for (; isa<PHINode>(InsertPt) || isa<LandingPadInst>(InsertPt); ++InsertPt)
      /* empty */; // Don't insert before PHI nodes or landingpad instrs.

	std::vector<Instruction*> WorkList;
    for (auto U = P->user_begin(); U != P->user_end();) {
      Instruction *I = cast<Instruction>(*(U++));
	  WorkList.emplace_back(I);
    }

    for (Instruction *I : WorkList) {
      IRBuilder<> Builder(I);
      Value *Load = Builder.CreateLoad(Slot);
      I->replaceUsesOfWith(P, Load);
    }

    // Iterate over each operand inserting a store in each predecessor.
    // This should be done after load inserting because store for phi must be
    // after all other instructions of the incoming block.
    for (unsigned i = 0, e = P->getNumIncomingValues(); i < e; ++i) {
      if (InvokeInst *II = dyn_cast<InvokeInst>(P->getIncomingValue(i))) {
        assert(II->getParent() != P->getIncomingBlock(i) &&
               "Invoke edge not supported yet");
        (void)II;
      }
      Value *V = P->getIncomingValue(i);
      // Skip undef
      if (isa<UndefValue>(V))
        continue;
      new StoreInst(P->getIncomingValue(i), Slot,
                    P->getIncomingBlock(i)->getTerminator());
    }

    // Delete PHI.
    P->eraseFromParent();
    return Slot;
  }

  /// DemoteRegToStack - This function takes a virtual register computed by an
  /// Instruction and replaces it with a slot in the stack frame, allocated via
  /// alloca.  This allows the CFG to be changed around without fear of
  /// invalidating the SSA information for the value.  It returns the pointer to
  /// the alloca inserted to create a stack slot for I.
  /// The difference of HLSL version is for I is Alloca, only replace new Alloca with
  /// old alloca, and HLSL don't have InvokeInst
  AllocaInst *DemoteRegToStack_HLSL(Instruction &I, bool VolatileLoads,
                                     Instruction *AllocaPoint) {
    if (I.use_empty()) {
      I.eraseFromParent();
      return nullptr;
    }

    IRBuilder<> AllocaBuilder(&I);
    if (!AllocaPoint) {
      Function *F = I.getParent()->getParent();
      AllocaPoint = F->getEntryBlock().begin();
    }
    AllocaBuilder.SetInsertPoint(AllocaPoint);

    if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
      // Create a stack slot to hold the value.
      AllocaInst *Slot = AllocaBuilder.CreateAlloca(AI->getAllocatedType(), nullptr, I.getName() + ".reg2mem");
	  I.replaceAllUsesWith(Slot);
	  I.eraseFromParent();
	  return Slot;
    }

    // Create a stack slot to hold the value.
    AllocaInst *Slot = AllocaBuilder.CreateAlloca(I.getType(), nullptr, I.getName() + ".reg2mem");;

    // Change all of the users of the instruction to read from the stack slot.
    while (!I.use_empty()) {
      Instruction *U = cast<Instruction>(I.user_back());
      if (PHINode *PN = dyn_cast<PHINode>(U)) {
        // If this is a PHI node, we can't insert a load of the value before the
        // use.  Instead insert the load in the predecessor block corresponding
        // to the incoming value.
        //
        // Note that if there are multiple edges from a basic block to this PHI
        // node that we cannot have multiple loads. The problem is that the
        // resulting PHI node will have multiple values (from each load) coming
        // in
        // from the same block, which is illegal SSA form. For this reason, we
        // keep track of and reuse loads we insert.
        DenseMap<BasicBlock *, Value *> Loads;
        for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
          if (PN->getIncomingValue(i) == &I) {
            Value *&V = Loads[PN->getIncomingBlock(i)];
            if (!V) {
              // Insert the load into the predecessor block
              V = new LoadInst(Slot, I.getName() + ".reload", VolatileLoads,
                               PN->getIncomingBlock(i)->getTerminator());
            }
            PN->setIncomingValue(i, V);
          }

      } else {
        // If this is a normal instruction, just insert a load.
        Value *V =
            new LoadInst(Slot, I.getName() + ".reload", VolatileLoads, U);
        U->replaceUsesOfWith(&I, V);
      }
    }

    // Insert stores of the computed value into the stack slot. We have to be
    // careful if I is an invoke instruction, because we can't insert the store
    // AFTER the terminator instruction.
    BasicBlock::iterator InsertPt;
    if (!isa<TerminatorInst>(I)) {
      InsertPt = &I;
      ++InsertPt;
      for (; isa<PHINode>(InsertPt) || isa<LandingPadInst>(InsertPt);
           ++InsertPt)
        /* empty */; // Don't insert before PHI nodes or landingpad instrs.
    } else {
      InvokeInst &II = cast<InvokeInst>(I);
      InsertPt = II.getNormalDest()->getFirstInsertionPt();
    }

    new StoreInst(&I, Slot, InsertPt);
    return Slot;
  }
}

char RegToMemHlsl::ID = 0;
INITIALIZE_PASS_BEGIN(RegToMemHlsl, "reg2mem_hlsl", "Demote values with phi-node usage to stack slots",
                false, false)
INITIALIZE_PASS_END(RegToMemHlsl,   "reg2mem_hlsl", "Demote values with phi-node usage to stack slots",
                false, false)

bool RegToMemHlsl::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  // Insert all new allocas into entry block.
  BasicBlock *BBEntry = &F.getEntryBlock();
  assert(pred_empty(BBEntry) &&
         "Entry block to function must not have predecessors!");

  // Find first non-alloca instruction and create insertion point. This is
  // safe if block is well-formed: it always have terminator, otherwise
  // we'll get and assertion.
  BasicBlock::iterator I = BBEntry->begin();
  while (isa<AllocaInst>(I)) ++I;

  CastInst *AllocaInsertionPoint =
    new BitCastInst(Constant::getNullValue(Type::getInt32Ty(F.getContext())),
                    Type::getInt32Ty(F.getContext()),
                    "reg2mem_hlsl alloca point", I);

  // Find the escaped instructions. But don't create stack slots for
  // allocas in entry block.
  std::list<Instruction*> WorkList;
  for (Function::iterator ibb = F.begin(), ibe = F.end();
       ibb != ibe; ++ibb)
    for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
         iib != iie; ++iib) {
      if (!(isa<AllocaInst>(iib) && iib->getParent() == BBEntry) &&
          valueEscapes(iib)) {
        WorkList.push_front(&*iib);
      }
    }

  // Demote escaped instructions
  NumRegsDemotedHlsl += WorkList.size();
  for (std::list<Instruction*>::iterator ilb = WorkList.begin(),
       ile = WorkList.end(); ilb != ile; ++ilb)
    DemoteRegToStack_HLSL(**ilb, false, AllocaInsertionPoint);

  WorkList.clear();

  // Find all phi's
  for (Function::iterator ibb = F.begin(), ibe = F.end();
       ibb != ibe; ++ibb)
    for (BasicBlock::iterator iib = ibb->begin(), iie = ibb->end();
         iib != iie; ++iib)
      if (isa<PHINode>(iib))
        WorkList.push_front(&*iib);

  // Demote phi nodes
  NumPhisDemotedHlsl += WorkList.size();
  for (std::list<Instruction*>::iterator ilb = WorkList.begin(),
       ile = WorkList.end(); ilb != ile; ++ilb)
    DemotePHIToStack_HLSL(cast<PHINode>(*ilb), AllocaInsertionPoint);

  return true;
}


// createDemoteRegisterToMemoryHlsl - Provide an entry point to create this pass.
char &llvm::DemoteRegisterToMemoryHlslID = RegToMemHlsl::ID;
FunctionPass *llvm::createDemoteRegisterToMemoryHlslPass() {
  return new RegToMemHlsl();
}
