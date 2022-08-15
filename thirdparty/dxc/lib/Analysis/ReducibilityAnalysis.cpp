///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ReducibilityAnalysis.cpp                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/Analysis/ReducibilityAnalysis.h"
#include "dxc/Support/Global.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/CFG.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace llvm;
using llvm::legacy::PassManager;
using llvm::legacy::FunctionPassManager;
using std::vector;
using std::unordered_map;
using std::unordered_set;

#define DEBUG_TYPE "reducibility"


//===----------------------------------------------------------------------===//
//                    Reducibility Analysis Pass
//
// The pass implements T1-T2 graph reducibility test.
// The algorithm can be found in "Engineering a Compiler" text by 
// Keith Cooper and Linda Torczon.
//
//===----------------------------------------------------------------------===//
namespace ReducibilityAnalysisNS {

class ReducibilityAnalysis : public FunctionPass {
public:
  static char ID;

  ReducibilityAnalysis()
      : FunctionPass(ID), m_Action(IrreducibilityAction::ThrowException),
        m_bReducible(true) {}

  explicit ReducibilityAnalysis(IrreducibilityAction Action)
      : FunctionPass(ID), m_Action(Action), m_bReducible(true) {}

  virtual bool runOnFunction(Function &F);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }

  bool IsReducible() const { return m_bReducible; }

private:
  IrreducibilityAction m_Action;
  bool m_bReducible;
};

char ReducibilityAnalysis::ID = 0;

struct Node {
  typedef unordered_set<unsigned> IdxSet;
  IdxSet m_Succ;
  IdxSet m_Pred;
};

class NodeWorklist {
public:
  NodeWorklist(size_t MaxSize) : m_Size(0) { m_Data.resize(MaxSize); }

  size_t Size() const { return m_Size; }
  unsigned Get(size_t idx) const { return m_Data[idx]; }
  void PushBack(unsigned Val) { m_Data[m_Size++] = Val; }
  void Clear() { m_Size = 0; }

private:
  size_t m_Size;
  vector<unsigned> m_Data;
};

static bool IsEntryNode(size_t NodeIdx) {
  return NodeIdx == 0;
}

bool ReducibilityAnalysis::runOnFunction(Function &F) {
  m_bReducible = true;
  if (F.empty()) return false;
  IFTBOOL(F.size() < UINT32_MAX, DXC_E_DATA_TOO_LARGE);

  vector<Node> Nodes(F.size());
  unordered_map<BasicBlock*, unsigned> BasicBlockToNodeIdxMap;

  //
  // Initialize.
  //
  unsigned iNode = 0;
  for (BasicBlock &BB : F) {
    BasicBlockToNodeIdxMap[&BB] = iNode++;
  }

  for (BasicBlock &BB : F) {
    BasicBlock *pBB = &BB;
    unsigned N = BasicBlockToNodeIdxMap[pBB];

    for (succ_iterator itSucc = succ_begin(pBB), endSucc = succ_end(pBB); itSucc != endSucc; ++itSucc) {
      BasicBlock *pSuccBB = *itSucc;
      unsigned SuccNode = BasicBlockToNodeIdxMap[pSuccBB];
      Nodes[N].m_Succ.insert(SuccNode);
    }

    for (pred_iterator itPred = pred_begin(pBB), endPred = pred_end(pBB); itPred != endPred; ++itPred) {
      BasicBlock *pPredBB = *itPred;
      unsigned PredNode = BasicBlockToNodeIdxMap[pPredBB];
      Nodes[N].m_Pred.insert(PredNode);
    }
  }

  //
  // Reduce.
  //
  NodeWorklist Q1(Nodes.size()), Q2(Nodes.size());
  NodeWorklist *pReady = &Q1, *pWaiting = &Q2;
  for (unsigned i = 0; i < Nodes.size(); i++) {
    pReady->PushBack(i);
  }

  for (;;) {
    bool bChanged = false;
    pWaiting->Clear();
    
    for (unsigned iNode = 0; iNode < pReady->Size(); iNode++) {
      unsigned N = pReady->Get(iNode);
      Node *pNode = &Nodes[N];

      // T1: self-edge.
      auto itSucc = pNode->m_Succ.find(N);
      if (itSucc != pNode->m_Succ.end()) {
        pWaiting->PushBack(N);
        pNode->m_Succ.erase(itSucc);
        auto s1 = pNode->m_Pred.erase(N); DXASSERT_LOCALVAR(s1, s1 == 1, "otherwise check Pred/Succ sets");

        bChanged = true;
        continue;
      }

      // T2: single predecessor.
      if (pNode->m_Pred.size() == 1) {
        unsigned PredNode = *pNode->m_Pred.begin();
        Node *pPredNode = &Nodes[PredNode];
        auto s1 = pPredNode->m_Succ.erase(N); DXASSERT_LOCALVAR(s1, s1 == 1, "otherwise check Pred/Succ sets");
        // Do not update N's sets, as N is discarded and never looked at again.

        for (auto itSucc = pNode->m_Succ.begin(), endSucc = pNode->m_Succ.end(); itSucc != endSucc; ++itSucc) {
          unsigned SuccNode = *itSucc;
          Node *pSuccNode = &Nodes[SuccNode];
          auto s2 = pSuccNode->m_Pred.erase(N); DXASSERT_LOCALVAR(s2, s2, "otherwise check Pred/Succ sets");
          pPredNode->m_Succ.insert(SuccNode);
          pSuccNode->m_Pred.insert(PredNode);
        }

        bChanged = true;
        continue;
      }

      // Unreachable.
      if (pNode->m_Pred.size() == 0 && !IsEntryNode(N)) {
        for (auto itSucc = pNode->m_Succ.begin(), endSucc = pNode->m_Succ.end(); itSucc != endSucc; ++itSucc) {
          unsigned SuccNode = *itSucc;
          Node *pSuccNode = &Nodes[SuccNode];
          auto s1 = pSuccNode->m_Pred.erase(N); DXASSERT_LOCALVAR(s1, s1, "otherwise check Pred/Succ sets");
        }

        bChanged = true;
        continue;
      }

      // Could not reduce.
      pWaiting->PushBack(N);
    }

    if (pWaiting->Size() == 1) {
      break;
    }

    if (!bChanged) {
      m_bReducible = false;
      break;
    }

    std::swap(pReady, pWaiting);
  }

  if (!IsReducible()) {
    switch (m_Action) {
    case IrreducibilityAction::ThrowException:
      DEBUG(dbgs() << "Function '" << F.getName() << "' is irreducible. Aborting compilation.\n");
      IFT(DXC_E_IRREDUCIBLE_CFG);
      break;

    case IrreducibilityAction::PrintLog:
      DEBUG(dbgs() << "Function '" << F.getName() << "' is irreducible\n");
      break;

    case IrreducibilityAction::Ignore:
      break;

    default:
      DXASSERT(false, "otherwise incorrect action passed to the constructor");
    }
  }

  return false;
}

}


using namespace ReducibilityAnalysisNS;

// Publicly exposed interface to pass...
char &llvm::ReducibilityAnalysisID = ReducibilityAnalysis::ID;


INITIALIZE_PASS_BEGIN(ReducibilityAnalysis, "red", "Reducibility Analysis", true, true)
INITIALIZE_PASS_END(ReducibilityAnalysis, "red", "Reducibility Analysis", true, true)

namespace llvm {

FunctionPass *createReducibilityAnalysisPass(IrreducibilityAction Action) {
  return new ReducibilityAnalysis(Action);
}

bool IsReducible(const Module &M, IrreducibilityAction Action) {
  PassManager PM;
  ReducibilityAnalysis *pRA = new ReducibilityAnalysis(Action);
  PM.add(pRA);
  PM.run(const_cast<Module&>(M));

  return pRA->IsReducible();
}

bool IsReducible(const Function &f, IrreducibilityAction Action) {
  Function &F = const_cast<Function&>(f);
  DXASSERT(!F.isDeclaration(), "otherwise the caller is asking to check an external function");

  FunctionPassManager FPM(F.getParent());
  ReducibilityAnalysis *pRA = new ReducibilityAnalysis(Action);
  FPM.add(pRA);
  FPM.doInitialization();
  FPM.run(F);

  return pRA->IsReducible();
}

}
