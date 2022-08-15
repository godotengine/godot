//===- LoadCombine.cpp - Combine Adjacent Loads ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation combines adjacent loads.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "load-combine"

STATISTIC(NumLoadsAnalyzed, "Number of loads analyzed for combining");
STATISTIC(NumLoadsCombined, "Number of loads combined");

namespace {
struct PointerOffsetPair {
  Value *Pointer;
  uint64_t Offset;
};

struct LoadPOPPair {
  LoadPOPPair() = default;
  LoadPOPPair(LoadInst *L, PointerOffsetPair P, unsigned O)
      : Load(L), POP(P), InsertOrder(O) {}
  LoadInst *Load;
  PointerOffsetPair POP;
  /// \brief The new load needs to be created before the first load in IR order.
  unsigned InsertOrder;
};

class LoadCombine : public BasicBlockPass {
  LLVMContext *C;
  AliasAnalysis *AA;

public:
  LoadCombine() : BasicBlockPass(ID), C(nullptr), AA(nullptr) {
    initializeSROAPass(*PassRegistry::getPassRegistry());
  }
  
  using llvm::Pass::doInitialization;
  bool doInitialization(Function &) override;
  bool runOnBasicBlock(BasicBlock &BB) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  StringRef getPassName() const override { return "LoadCombine"; }
  static char ID;

  typedef IRBuilder<true, TargetFolder> BuilderTy;

private:
  BuilderTy *Builder;

  PointerOffsetPair getPointerOffsetPair(LoadInst &);
  bool combineLoads(DenseMap<const Value *, SmallVector<LoadPOPPair, 8>> &);
  bool aggregateLoads(SmallVectorImpl<LoadPOPPair> &);
  bool combineLoads(SmallVectorImpl<LoadPOPPair> &);
};
}

bool LoadCombine::doInitialization(Function &F) {
  DEBUG(dbgs() << "LoadCombine function: " << F.getName() << "\n");
  C = &F.getContext();
  return true;
}

PointerOffsetPair LoadCombine::getPointerOffsetPair(LoadInst &LI) {
  PointerOffsetPair POP;
  POP.Pointer = LI.getPointerOperand();
  POP.Offset = 0;
  while (isa<BitCastInst>(POP.Pointer) || isa<GetElementPtrInst>(POP.Pointer)) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(POP.Pointer)) {
      auto &DL = LI.getModule()->getDataLayout();
      unsigned BitWidth = DL.getPointerTypeSizeInBits(GEP->getType());
      APInt Offset(BitWidth, 0);
      if (GEP->accumulateConstantOffset(DL, Offset))
        POP.Offset += Offset.getZExtValue();
      else
        // Can't handle GEPs with variable indices.
        return POP;
      POP.Pointer = GEP->getPointerOperand();
    } else if (auto *BC = dyn_cast<BitCastInst>(POP.Pointer))
      POP.Pointer = BC->getOperand(0);
  }
  return POP;
}

bool LoadCombine::combineLoads(
    DenseMap<const Value *, SmallVector<LoadPOPPair, 8>> &LoadMap) {
  bool Combined = false;
  for (auto &Loads : LoadMap) {
    if (Loads.second.size() < 2)
      continue;
    std::sort(Loads.second.begin(), Loads.second.end(),
              [](const LoadPOPPair &A, const LoadPOPPair &B) {
      return A.POP.Offset < B.POP.Offset;
    });
    if (aggregateLoads(Loads.second))
      Combined = true;
  }
  return Combined;
}

/// \brief Try to aggregate loads from a sorted list of loads to be combined.
///
/// It is guaranteed that no writes occur between any of the loads. All loads
/// have the same base pointer. There are at least two loads.
bool LoadCombine::aggregateLoads(SmallVectorImpl<LoadPOPPair> &Loads) {
  assert(Loads.size() >= 2 && "Insufficient loads!");
  LoadInst *BaseLoad = nullptr;
  SmallVector<LoadPOPPair, 8> AggregateLoads;
  bool Combined = false;
  uint64_t PrevOffset = -1ull;
  uint64_t PrevSize = 0;
  for (auto &L : Loads) {
    if (PrevOffset == -1ull) {
      BaseLoad = L.Load;
      PrevOffset = L.POP.Offset;
      PrevSize = L.Load->getModule()->getDataLayout().getTypeStoreSize(
          L.Load->getType());
      AggregateLoads.push_back(L);
      continue;
    }
    if (L.Load->getAlignment() > BaseLoad->getAlignment())
      continue;
    if (L.POP.Offset > PrevOffset + PrevSize) {
      // No other load will be combinable
      if (combineLoads(AggregateLoads))
        Combined = true;
      AggregateLoads.clear();
      PrevOffset = -1;
      continue;
    }
    if (L.POP.Offset != PrevOffset + PrevSize)
      // This load is offset less than the size of the last load.
      // FIXME: We may want to handle this case.
      continue;
    PrevOffset = L.POP.Offset;
    PrevSize = L.Load->getModule()->getDataLayout().getTypeStoreSize(
        L.Load->getType());
    AggregateLoads.push_back(L);
  }
  if (combineLoads(AggregateLoads))
    Combined = true;
  return Combined;
}

/// \brief Given a list of combinable load. Combine the maximum number of them.
bool LoadCombine::combineLoads(SmallVectorImpl<LoadPOPPair> &Loads) {
  // Remove loads from the end while the size is not a power of 2.
  unsigned TotalSize = 0;
  for (const auto &L : Loads)
    TotalSize += L.Load->getType()->getPrimitiveSizeInBits();
  while (TotalSize != 0 && !isPowerOf2_32(TotalSize))
    TotalSize -= Loads.pop_back_val().Load->getType()->getPrimitiveSizeInBits();
  if (Loads.size() < 2)
    return false;

  DEBUG({
    dbgs() << "***** Combining Loads ******\n";
    for (const auto &L : Loads) {
      dbgs() << L.POP.Offset << ": " << *L.Load << "\n";
    }
  });

  // Find first load. This is where we put the new load.
  LoadPOPPair FirstLP;
  FirstLP.InsertOrder = -1u;
  for (const auto &L : Loads)
    if (L.InsertOrder < FirstLP.InsertOrder)
      FirstLP = L;

  unsigned AddressSpace =
      FirstLP.POP.Pointer->getType()->getPointerAddressSpace();

  Builder->SetInsertPoint(FirstLP.Load);
  Value *Ptr = Builder->CreateConstGEP1_64(
      Builder->CreatePointerCast(Loads[0].POP.Pointer,
                                 Builder->getInt8PtrTy(AddressSpace)),
      Loads[0].POP.Offset);
  LoadInst *NewLoad = new LoadInst(
      Builder->CreatePointerCast(
          Ptr, PointerType::get(IntegerType::get(Ptr->getContext(), TotalSize),
                                Ptr->getType()->getPointerAddressSpace())),
      Twine(Loads[0].Load->getName()) + ".combined", false,
      Loads[0].Load->getAlignment(), FirstLP.Load);

  for (const auto &L : Loads) {
    Builder->SetInsertPoint(L.Load);
    Value *V = Builder->CreateExtractInteger(
        L.Load->getModule()->getDataLayout(), NewLoad,
        cast<IntegerType>(L.Load->getType()),
        L.POP.Offset - Loads[0].POP.Offset, "combine.extract");
    L.Load->replaceAllUsesWith(V);
  }

  NumLoadsCombined = NumLoadsCombined + Loads.size();
  return true;
}

bool LoadCombine::runOnBasicBlock(BasicBlock &BB) {
  if (skipOptnoneFunction(BB))
    return false;

  AA = &getAnalysis<AliasAnalysis>();

  IRBuilder<true, TargetFolder> TheBuilder(
      BB.getContext(), TargetFolder(BB.getModule()->getDataLayout()));
  Builder = &TheBuilder;

  DenseMap<const Value *, SmallVector<LoadPOPPair, 8>> LoadMap;
  AliasSetTracker AST(*AA);

  bool Combined = false;
  unsigned Index = 0;
  for (auto &I : BB) {
    if (I.mayThrow() || (I.mayWriteToMemory() && AST.containsUnknown(&I))) {
      if (combineLoads(LoadMap))
        Combined = true;
      LoadMap.clear();
      AST.clear();
      continue;
    }
    LoadInst *LI = dyn_cast<LoadInst>(&I);
    if (!LI)
      continue;
    ++NumLoadsAnalyzed;
    if (!LI->isSimple() || !LI->getType()->isIntegerTy())
      continue;
    auto POP = getPointerOffsetPair(*LI);
    if (!POP.Pointer)
      continue;
    LoadMap[POP.Pointer].push_back(LoadPOPPair(LI, POP, Index++));
    AST.add(LI);
  }
  if (combineLoads(LoadMap))
    Combined = true;
  return Combined;
}

void LoadCombine::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();

  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
}

char LoadCombine::ID = 0;

BasicBlockPass *llvm::createLoadCombinePass() {
  return new LoadCombine();
}

INITIALIZE_PASS_BEGIN(LoadCombine, "load-combine", "Combine Adjacent Loads",
                      false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(LoadCombine, "load-combine", "Combine Adjacent Loads",
                    false, false)

