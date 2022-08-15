///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// WaveSensitivityAnalysis.cpp                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides support for doing analysis that are aware of wave      //
// intrinsics.                                                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilValidation.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/Support/Global.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/PostDominators.h"

#ifdef _WIN32
#include <winerror.h>
#endif
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>

using namespace llvm;
using namespace std;

namespace hlsl {

// WaveSensitivityAnalysis is created to validate Gradient operations.
// Gradient operations require all neighbor lanes to be active when calculated,
// compiler will enable lanes to meet this requirement. If a wave operation
// contributed to gradient operation, it will get unexpected result because the
// active lanes are modified.
// To avoid unexpected result, validation will fail if gradient operations
// are dependent on wave-sensitive data or control flow.

class WaveSensitivityAnalyzer : public WaveSensitivityAnalysis {
private:
  enum WaveSensitivity {
    KnownSensitive,
    KnownNotSensitive,
    Unknown
  };
  PostDominatorTree *pPDT;
  map<Instruction *, WaveSensitivity> InstState;
  map<BasicBlock *, WaveSensitivity> BBState;
  std::vector<Instruction *> InstWorkList;
  std::vector<PHINode *> UnknownPhis; // currently unknown phis. Indicate cycles after Analyze
  std::vector<BasicBlock *> BBWorkList;
  bool CheckBBState(BasicBlock *BB, WaveSensitivity WS);
  WaveSensitivity GetInstState(Instruction *I);
  void UpdateBlock(BasicBlock *BB, WaveSensitivity WS);
  void UpdateInst(Instruction *I, WaveSensitivity WS);
  void VisitInst(Instruction *I);
public:
  WaveSensitivityAnalyzer(PostDominatorTree &PDT) : pPDT(&PDT) {}
  void Analyze(Function *F);
  void Analyze();
  bool IsWaveSensitive(Instruction *op);
};

WaveSensitivityAnalysis* WaveSensitivityAnalysis::create(PostDominatorTree &PDT) {
  return new WaveSensitivityAnalyzer(PDT);
}

// Analyze the given function's instructions as wave-sensitive or not
void WaveSensitivityAnalyzer::Analyze(Function *F) {
  // Add all blocks but the entry in reverse order so they come out in order
  auto it = F->getBasicBlockList().end();
  for ( it-- ; it != F->getBasicBlockList().begin(); it--)
    BBWorkList.emplace_back(&*it);
  // Add entry block as non-sensitive
  UpdateBlock(&*it, KnownNotSensitive);

  // First analysis
  Analyze();

  // If any phis with explored preds remain unknown
  // it has to be in a loop that don't include wave sensitivity
  // Update each as such and redo Analyze to mark the descendents
  while (!UnknownPhis.empty() || !InstWorkList.empty() || !BBWorkList.empty()) {
    while (!UnknownPhis.empty()) {
      PHINode *Phi = UnknownPhis.back();
      UnknownPhis.pop_back();
      // UnknownPhis might have actually known phis that were changed. skip them
      if (Unknown == GetInstState(Phi)) {
        // If any of the preds have not been visited, we can't assume a cycle yet
        bool allPredsVisited = true;
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (!BBState.count(Phi->getIncomingBlock(i))) {
            allPredsVisited = false;
            break;
          }
        }
#ifndef NDEBUG
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (Instruction *IArg = dyn_cast<Instruction>(Phi->getIncomingValue(i))) {
            DXASSERT_LOCALVAR(IArg, GetInstState(IArg) != KnownSensitive,
                   "Unknown wave-status Phi argument should not be able to be known sensitive");
          }
        }
#endif
        if (allPredsVisited)
          UpdateInst(Phi, KnownNotSensitive);
      }
    }
    Analyze();
  }
#ifndef NDEBUG
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      DXASSERT_LOCALVAR(I, Unknown != GetInstState(&I), "Wave sensitivity analysis exited without finding results for all instructions");
    }
  }
#endif
}

// Analyze the member instruction and BBlock worklists
void WaveSensitivityAnalyzer::Analyze() {
  while (!InstWorkList.empty() || !BBWorkList.empty()) {
    // Process the instruction work list.
    while (!InstWorkList.empty()) {
      Instruction *I = InstWorkList.back();
      InstWorkList.pop_back();

      // "I" got into the work list because it made a transition.
      for (User *U : I->users()) {
        Instruction *UI = cast<Instruction>(U);
        VisitInst(UI);
      }
    }

    // Process one entry of the basic block work list.
    if (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      // Notify all instructions in this basic block that they need to
      // be reevaluated (eg, a block previously though to be insensitive
      // is now sensitive).
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        VisitInst(I);
    }
  }
}

bool WaveSensitivityAnalyzer::CheckBBState(BasicBlock *BB, WaveSensitivity WS) {
  auto c = BBState.find(BB);
  if (c == BBState.end()) {
    return WS == Unknown;
  }
  else {
    return (*c).second == WS;
  }
}

WaveSensitivityAnalyzer::WaveSensitivity
WaveSensitivityAnalyzer::GetInstState(Instruction *I) {
  auto c = InstState.find(I);
  if (c == InstState.end())
    return Unknown;
  return (*c).second;
}

void WaveSensitivityAnalyzer::UpdateBlock(BasicBlock *BB, WaveSensitivity WS) {
  auto c = BBState.find(BB);
  // Do not update if an entry is already found and it hasn't changed or
  // has already been marked as wave sensitive (an insensitive term might
  // try to mark it as such, but this effectively implements the 'any pred'
  // rule).
  if (c != BBState.end() && ((*c).second == WS || (*c).second == KnownSensitive))
    return;
  BBState[BB] = WS;
  BBWorkList.push_back(BB);
}

void WaveSensitivityAnalyzer::UpdateInst(Instruction *I, WaveSensitivity WS) {
  auto c = InstState.find(I);
  if (c == InstState.end() || (*c).second != WS) {
    InstState[I] = WS;
    InstWorkList.push_back(I);
    if (TerminatorInst * TI = dyn_cast<TerminatorInst>(I)) {
      BasicBlock *CurBB = TI->getParent();
      for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
        BasicBlock *BB = TI->getSuccessor(i);
        // Only propagate WS when BB not post dom CurBB.
        WaveSensitivity TmpWS = pPDT->properlyDominates(BB, CurBB)
                                    ? WaveSensitivity::KnownNotSensitive
                                    : WS;
        UpdateBlock(BB, TmpWS);
      }
    }
  }
}

void WaveSensitivityAnalyzer::VisitInst(Instruction *I) {
  unsigned firstArg = 0;
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (OP::IsDxilOpFuncCallInst(CI)) {
      firstArg = 1;
      OP::OpCode opcode = OP::GetDxilOpFuncCallInst(CI);
      if (OP::IsDxilOpWave(opcode)) {
        UpdateInst(I, KnownSensitive);
        return;
      }
    }
  }


  if (CheckBBState(I->getParent(), KnownSensitive)) {
    UpdateInst(I, KnownSensitive);
    return;
  }

  // Catch control flow wave sensitive for phi.
  if (PHINode *Phi = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
      BasicBlock *BB = Phi->getIncomingBlock(i);
      WaveSensitivity WS = GetInstState(BB->getTerminator());
      if (WS == KnownSensitive) {
        UpdateInst(I, KnownSensitive);
        return;
      } else if (Unknown == GetInstState(I)) {
        UnknownPhis.emplace_back(Phi);
      }
    }
  }

  bool allKnownNotSensitive = true;
  for (unsigned i = firstArg; i < I->getNumOperands(); ++i) {
    Value *V = I->getOperand(i);
    if (Instruction *IArg = dyn_cast<Instruction>(V)) {
      WaveSensitivity WS = GetInstState(IArg);
      if (WS == KnownSensitive) {
        UpdateInst(I, KnownSensitive);
        return;
      } else if (WS == Unknown) {
        allKnownNotSensitive = false;
      }
    }
  }
  if (allKnownNotSensitive) {
    UpdateInst(I, KnownNotSensitive);
  }
}

bool WaveSensitivityAnalyzer::IsWaveSensitive(Instruction *op) {
  auto c = InstState.find(op);
  if(c == InstState.end()) {
    DXASSERT(false, "Instruction sensitivity not foud. Analysis didn't complete!");
    return false;
  }
  DXASSERT((*c).second != Unknown, "else analysis is missing a case");
  return (*c).second == KnownSensitive;
}

} // namespace hlsl
