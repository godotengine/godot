//===- DxilFixConstArrayInitializer.cpp - Special Construct Initializer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/CFG.h"
#include "llvm/Transforms/Scalar.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/HLModule.h"

#include <unordered_map>
#include <limits>

using namespace llvm;

namespace {

class DxilFixConstArrayInitializer : public ModulePass {
public:
  static char ID;
  DxilFixConstArrayInitializer() : ModulePass(ID) {
    initializeDxilFixConstArrayInitializerPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  StringRef getPassName() const override { return "Dxil Fix Const Array Initializer"; }
};

char DxilFixConstArrayInitializer::ID;
}

static bool TryFixGlobalVariable(GlobalVariable &GV, BasicBlock *EntryBlock, const std::unordered_map<Instruction *, unsigned> &InstOrder) {
  // Only proceed if the variable has an undef initializer
  if (!GV.hasInitializer() || !isa<UndefValue>(GV.getInitializer()))
    return false;

  // Only handle cases when it's an array of scalars.
  Type *Ty = GV.getType()->getPointerElementType();
  if (!Ty->isArrayTy())
    return false;

  // Don't handle arrays that are too big
  if (Ty->getArrayNumElements() > 1024)
    return false;

  Type *ElementTy = Ty->getArrayElementType();

  // Only handle arrays of scalar types
  if (ElementTy->isAggregateType() || ElementTy->isVectorTy())
    return false;

  // The instruction index at which point we no longer consider it
  // safe to fold Stores. It's the earliest store with non-constant index,
  // earliest store with non-constant value, or a load
  unsigned FirstUnsafeIndex = std::numeric_limits<unsigned>::max();

  SmallVector<StoreInst *, 8> PossibleFoldableStores;

  // First do a pass to find the boundary for where we could fold stores. Get a
  // list of stores that may be folded.
  for (User *U : GV.users()) {
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
      bool AllConstIndices = GEP->hasAllConstantIndices();
      unsigned NumIndices = GEP->getNumIndices();

      if (NumIndices != 2)
        return false;

      for (User *GEPUser : GEP->users()) {
        if (StoreInst *Store = dyn_cast<StoreInst>(GEPUser)) {
          if (Store->getParent() != EntryBlock)
            continue;
          unsigned StoreIndex = InstOrder.at(Store);
          if (!AllConstIndices || !isa<Constant>(Store->getValueOperand())) {
            FirstUnsafeIndex = std::min(StoreIndex, FirstUnsafeIndex);
            continue;
          }
          PossibleFoldableStores.push_back(Store);
        }
        else if (LoadInst *Load = dyn_cast<LoadInst>(GEPUser)) {
          if (Load->getParent() != EntryBlock)
            continue;
          FirstUnsafeIndex = std::min(FirstUnsafeIndex, InstOrder.at(Load));
        }
        // If we have something weird like chained GEPS, or bitcasts, give up.
        else {
          return false;
        }
      }
    }
  }
  
  SmallVector<Constant *, 16> InitValue;
  SmallVector<unsigned, 16>   LatestStores;
  SmallVector<StoreInst *, 8> StoresToRemove;

  InitValue.resize(Ty->getArrayNumElements());
  LatestStores.resize(Ty->getArrayNumElements());

  for (StoreInst *Store : PossibleFoldableStores) {
    unsigned StoreIndex = InstOrder.at(Store);
    // Skip stores that are out of bounds
    if (StoreIndex >= FirstUnsafeIndex)
      continue;

    GEPOperator *GEP = cast<GEPOperator>(Store->getPointerOperand());
    uint64_t Index = cast<ConstantInt>(GEP->getOperand(2))->getLimitedValue();

    if (LatestStores[Index] <= StoreIndex) {
      InitValue[Index] = cast<Constant>(Store->getValueOperand());
      LatestStores[Index] = StoreIndex;
    }
    StoresToRemove.push_back(Store);
  }

  // Give up if we have missing indices
  for (Constant *C : InitValue)
    if (!C)
      return false;

  GV.setInitializer(ConstantArray::get(cast<ArrayType>(Ty), InitValue));

  for (StoreInst *Store : StoresToRemove)
    Store->eraseFromParent();

  return true;
}

bool DxilFixConstArrayInitializer::runOnModule(Module &M) {
  BasicBlock *EntryBlock = nullptr;

  if (M.HasDxilModule()) {
    hlsl::DxilModule &DM = M.GetDxilModule();
    if (DM.GetEntryFunction()) {
      EntryBlock = &DM.GetEntryFunction()->getEntryBlock();
    }
  }
  else if (M.HasHLModule()) {
    hlsl::HLModule &HM = M.GetHLModule();
    if (HM.GetEntryFunction())
      EntryBlock = &HM.GetEntryFunction()->getEntryBlock();
  }

  if (!EntryBlock)
    return false;

  // If some block might branch to the entry for some reason (like if it's a loop header),
  // give up now. Have to make sure this block is not preceeded by anything.
  if (pred_begin(EntryBlock) != pred_end(EntryBlock))
    return false;

  // Find the instruction order for everything in the entry block.
  std::unordered_map<Instruction *, unsigned> InstOrder;
  for (Instruction &I : *EntryBlock) {
    InstOrder[&I] = InstOrder.size();
  }

  bool Changed = false;
  for (GlobalVariable &GV : M.globals()) {
    Changed = TryFixGlobalVariable(GV, EntryBlock, InstOrder);
  }

  return Changed;
}


Pass *llvm::createDxilFixConstArrayInitializerPass() {
  return new DxilFixConstArrayInitializer();
}

INITIALIZE_PASS(DxilFixConstArrayInitializer, "dxil-fix-array-init", "Dxil Fix Array Initializer", false, false)