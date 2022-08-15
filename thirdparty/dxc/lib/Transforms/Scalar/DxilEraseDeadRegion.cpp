//===- DxilEraseDeadRegion.cpp - Heuristically Remove Dead Region ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:
//   1. Identify potentially dead regions by finding blocks with multiple
//      predecessors but no PHIs
//   2. Find common dominant ancestor of all the predecessors
//   3. Ensure original block post-dominates the ancestor
//   4. Ensure no instructions in the region have side effects (not including
//      original block and ancestor)
//   5. Remove all blocks in the region (excluding original block and ancestor)
//

#include "llvm/Pass.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ADT/SetVector.h"

#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilMetadataHelper.h"
#include "dxc/HLSL/DxilNoops.h"
#include "dxc/DXIL/DxilUtil.h"

#include <unordered_map>
#include <unordered_set>

using namespace llvm;
using namespace hlsl;

// TODO: Could probably move this to a common place at some point.
namespace {

struct MiniDCE {
  // Use a set vector because the same value could be added more than once, which
  // could lead to double free.
  SetVector<Instruction *> Worklist;
  void EraseAndProcessOperands(Instruction *TopI);
};

void MiniDCE::EraseAndProcessOperands(Instruction *TopI) {
  Worklist.clear();
  for (Value *Op : TopI->operands()) {
    if (Instruction *OpI = dyn_cast<Instruction>(Op))
      Worklist.insert(OpI);
  }
  TopI->eraseFromParent();
  TopI = nullptr;

  while (Worklist.size()) {
    Instruction *I = Worklist.pop_back_val();
    if (llvm::isInstructionTriviallyDead(I)) {
      for (Value *Op : I->operands()) {
        if (Instruction *OpI = dyn_cast<Instruction>(Op))
          Worklist.insert(OpI);
      }
      I->eraseFromParent();
    }
  }
}

}

struct DxilEraseDeadRegion : public FunctionPass {
  static char ID;

  DxilEraseDeadRegion() : FunctionPass(ID) {
    initializeDxilEraseDeadRegionPass(*PassRegistry::getPassRegistry());
  }

  std::unordered_map<BasicBlock *, bool> m_SafeBlocks;
  MiniDCE m_DCE;

  // Replace all uses of every instruction in a block with undefs
  void UndefBasicBlock(BasicBlock* BB) {
    while (BB->begin() != BB->end()) {
      Instruction *I = &BB->back();
      if (!I->user_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      m_DCE.EraseAndProcessOperands(I);
    }
  }

  // Wave Ops are marked as having side effects to avoid moving them across
  // control flow. But they're safe to remove if unused.
  bool IsWaveIntrinsic(Instruction *I) {
    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      if (hlsl::OP::IsDxilOpFuncCallInst(CI)) {
        DXIL::OpCode opcode = hlsl::OP::GetDxilOpFuncCallInst(CI);
        if (hlsl::OP::IsDxilOpWave(opcode))
          return true;
      }
    }
    return false;
  }

  // This function takes in a basic block, and a *complete* region that this
  // block is in, and checks whether it's safe to delete this block as part of
  // this region (i.e. if any values defined in the block are used outside of
  // the region) and whether it's safe to delete the block in general (side
  // effects).
  bool SafeToDeleteBlock(BasicBlock *BB, const std::set<BasicBlock*> &Region) {
    assert(Region.count(BB)); // Region must be a complete region that contains the block.

    auto FindIt = m_SafeBlocks.find(BB);
    if (FindIt != m_SafeBlocks.end()) {
      return FindIt->second;
    }

    // Make sure all insts are safe to delete
    // (no side effects, etc.)
    bool ValuesReferencedOutsideOfBlock = false;
    bool ValuesReferencedOutsideOfRegion = false;
    for (Instruction &I : *BB) {
      for (User *U : I.users()) {
        if (Instruction *UI = dyn_cast<Instruction>(U)) {
          BasicBlock *UB = UI->getParent();
          if (UB != BB) {
            ValuesReferencedOutsideOfBlock = true;
            if (!Region.count(UB))
              ValuesReferencedOutsideOfRegion = true;
          }
        }
      }

      // Wave intrinsics are technically read-only and safe to delete
      if (IsWaveIntrinsic(&I))
        continue;

      if (I.mayHaveSideEffects() && !hlsl::IsNop(&I)) {
        m_SafeBlocks[BB] = false;
        return false;
      }
    }

    if (ValuesReferencedOutsideOfRegion)
      return false;

    // If the block's defs are entirely referenced within the block itself,
    // it'll remain safe to delete no matter the region.
    if (!ValuesReferencedOutsideOfBlock)
      m_SafeBlocks[BB] = true;

    return true;
  }

  // Find a region of blocks between `Begin` and `End` that are entirely self
  // contained and produce no values that leave the region.
  bool FindDeadRegion(DominatorTree *DT, PostDominatorTree *PDT, BasicBlock *Begin, BasicBlock *End, std::set<BasicBlock *> &Region) {
    std::vector<BasicBlock *> WorkList;
    auto ProcessSuccessors = [DT, PDT, &WorkList, Begin, End, &Region](BasicBlock *BB) {
      for (BasicBlock *Succ : successors(BB)) {
        if (Succ == End) continue;
        if (Region.count(Succ)) continue;
        // Make sure it's safely inside the region.
        if (!DT->properlyDominates(Begin, Succ) || !PDT->properlyDominates(End, Succ))
          return false;
        WorkList.push_back(Succ);
        Region.insert(Succ);
      }
      return true;
    };

    if (!ProcessSuccessors(Begin))
      return false;

    while (WorkList.size()) {
      BasicBlock *BB = WorkList.back();
      WorkList.pop_back();
      if (!ProcessSuccessors(BB))
        return false;
    }

    if (Region.empty())
      return false;

    for (BasicBlock *BB : Region) {
      // Give up if there are any edges coming from outside of the region
      // anywhere other than `Begin`.
      for (auto PredIt = llvm::pred_begin(BB); PredIt != llvm::pred_end(BB); PredIt++) {
        BasicBlock *PredBB = *PredIt;
        if (PredBB != Begin && !Region.count(PredBB))
          return false;
      }
      // Check side effects etc.
      if (!this->SafeToDeleteBlock(BB, Region))
        return false;
    }
    return true;
  }

  static bool IsMetadataKind(LLVMContext &Ctx, unsigned TargetID, StringRef MDKind) {
    unsigned ID = 0;
    if (Ctx.findMDKindID(MDKind, &ID))
      return TargetID == ID;
    return false;
  }

  static bool HasUnsafeMetadata(Instruction *I) {
    SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
    I->getAllMetadata(MDs);

    LLVMContext &Context = I->getContext();
    for (auto &p : MDs) {
      if (p.first == (unsigned)LLVMContext::MD_dbg)
        continue;
      if (IsMetadataKind(Context, p.first, DxilMDHelper::kDxilControlFlowHintMDName))
        continue;
      return true;
    }

    return false;
  }

  bool TrySimplify(DominatorTree *DT, PostDominatorTree *PDT, LoopInfo *LI, BasicBlock *BB) {
    // Give up if BB has any Phis
    if (BB->begin() != BB->end() && isa<PHINode>(BB->begin()))
      return false;

    std::vector<BasicBlock *> Predecessors(pred_begin(BB), pred_end(BB));
    if (Predecessors.size() < 2) return false;

    // Find the common ancestor of all the predecessors
    BasicBlock *Common = DT->findNearestCommonDominator(Predecessors[0], Predecessors[1]);
    if (!Common) return false;
    for (unsigned i = 2; i < Predecessors.size(); i++) {
      Common = DT->findNearestCommonDominator(Common, Predecessors[i]);
      if (!Common) return false;
    }

    // If there are any metadata on Common block's branch, give up.
    if (HasUnsafeMetadata(Common->getTerminator()))
      return false;

    if (!DT->properlyDominates(Common, BB))
      return false;
    if (!PDT->properlyDominates(BB, Common))
      return false;

    std::set<BasicBlock *> Region;
    if (!this->FindDeadRegion(DT, PDT, Common, BB, Region))
      return false;

    // Replace Common's branch with an unconditional branch to BB
    m_DCE.EraseAndProcessOperands(Common->getTerminator());
    BranchInst::Create(BB, Common);

    DeleteRegion(Region, LI);

    return true;
  }

  // Only call this after all the incoming branches have
  // been removed.
  void DeleteRegion(std::set<BasicBlock *> &Region, LoopInfo *LI) {
    for (BasicBlock *BB : Region) {
      UndefBasicBlock(BB);
      // Don't leave any dangling pointers in the LoopInfo for subsequent iterations.
      // But don't bother to delete the (possibly now empty) Loop objects, just leave them empty.
      LI->removeBlock(BB);
    }

    // All blocks should be empty now, so walking the set is fine
    for (BasicBlock *BB : Region) {
      assert((BB->size() == 0) && "Trying to delete a non-empty basic block!");
      BB->eraseFromParent();
    }
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTree>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  // Go through list of all the loops and delete ones that definitely don't contribute any outputs.
  // Delete the loop if there's no side effects in the loop, and the loop has no exit values at all.
  bool TryRemoveSimpleDeadLoops(LoopInfo *LI) {
    bool Changed = false;
    SmallVector<Loop *, 4> LoopWorklist;
    for (Loop *L : *LI) {
      LoopWorklist.push_back(L);
    }

    std::set<BasicBlock *> LoopRegion;
    while (LoopWorklist.size()) {
      Loop *L = LoopWorklist.pop_back_val();

      // Skip empty loops.
      if (L->block_begin() == L->block_end())
        continue;

      // If there's not a single exit block, give up. Those cases can probably
      // be handled by normal region deletion heuristic anyways.
      BasicBlock *ExitBB = L->getExitBlock();
      if (!ExitBB) {
        for (Loop *ChildLoop : *L)
          LoopWorklist.push_back(ChildLoop);
        continue;
      }

      LoopRegion.clear();
      for (BasicBlock *BB : L->getBlocks())
        LoopRegion.insert(BB);

      bool LoopSafeToDelete = true;
      for (BasicBlock *BB : L->getBlocks()) {
        if (!this->SafeToDeleteBlock(BB, LoopRegion)) {
          LoopSafeToDelete = false;
          break;
        }
      }

      if (LoopSafeToDelete) {
        // Re-branch anything that went to the loop's header to the loop's sole exit.
        assert(!isa<PHINode>(ExitBB->front()) && "There must be no values escaping from the loop");
        BasicBlock *HeaderBB = L->getHeader();
        for (auto PredIt = llvm::pred_begin(HeaderBB), PredEnd = llvm::pred_end(HeaderBB);
          PredIt != PredEnd;)
        {
          BasicBlock *PredBB = *PredIt;
          PredIt++;

          TerminatorInst *TI = PredBB->getTerminator();
          TI->replaceUsesOfWith(HeaderBB, ExitBB);
        }

        DeleteRegion(LoopRegion, LI);
        Changed = true;
      }
      else {
        for (Loop *ChildLoop : *L)
          LoopWorklist.push_back(ChildLoop);
      }
    }
    return Changed;
  }

  bool runOnFunction(Function &F) override {
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *PDT = &getAnalysis<PostDominatorTree>();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

    bool Changed = false;
    while (1) {
      bool LocalChanged = false;

      LocalChanged |= hlsl::dxilutil::DeleteDeadAllocas(F);
      LocalChanged |= this->TryRemoveSimpleDeadLoops(LI);

      for (Function::iterator It = F.begin(), E = F.end(); It != E; It++) {
        BasicBlock &BB = *It;
        if (this->TrySimplify(DT, PDT, LI, &BB)) {
          LocalChanged = true;
          break;
        }
      }

      Changed |= LocalChanged;
      if (!LocalChanged)
        break;
    }

    return Changed;
  }
};

char DxilEraseDeadRegion::ID;

Pass *llvm::createDxilEraseDeadRegionPass() {
  return new DxilEraseDeadRegion();
}

INITIALIZE_PASS_BEGIN(DxilEraseDeadRegion, "dxil-erase-dead-region", "Dxil Erase Dead Region", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree)
INITIALIZE_PASS_END(DxilEraseDeadRegion, "dxil-erase-dead-region", "Dxil Erase Dead Region", false, false)

