//===- LexicalScopes.cpp - Collecting lexical scope info ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements LexicalScopes analysis.
//
// This pass collects lexical scope information and maps machine instructions
// to respective lexical scopes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "lexicalscopes"

/// reset - Reset the instance so that it's prepared for another function.
void LexicalScopes::reset() {
  MF = nullptr;
  CurrentFnLexicalScope = nullptr;
  LexicalScopeMap.clear();
  AbstractScopeMap.clear();
  InlinedLexicalScopeMap.clear();
  AbstractScopesList.clear();
}

/// initialize - Scan machine function and constuct lexical scope nest.
void LexicalScopes::initialize(const MachineFunction &Fn) {
  reset();
  MF = &Fn;
  SmallVector<InsnRange, 4> MIRanges;
  DenseMap<const MachineInstr *, LexicalScope *> MI2ScopeMap;
  extractLexicalScopes(MIRanges, MI2ScopeMap);
  if (CurrentFnLexicalScope) {
    constructScopeNest(CurrentFnLexicalScope);
    assignInstructionRanges(MIRanges, MI2ScopeMap);
  }
}

/// extractLexicalScopes - Extract instruction ranges for each lexical scopes
/// for the given machine function.
void LexicalScopes::extractLexicalScopes(
    SmallVectorImpl<InsnRange> &MIRanges,
    DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap) {

  // Scan each instruction and create scopes. First build working set of scopes.
  for (const auto &MBB : *MF) {
    const MachineInstr *RangeBeginMI = nullptr;
    const MachineInstr *PrevMI = nullptr;
    const DILocation *PrevDL = nullptr;
    for (const auto &MInsn : MBB) {
      // Check if instruction has valid location information.
      const DILocation *MIDL = MInsn.getDebugLoc();
      if (!MIDL) {
        PrevMI = &MInsn;
        continue;
      }

      // If scope has not changed then skip this instruction.
      if (MIDL == PrevDL) {
        PrevMI = &MInsn;
        continue;
      }

      // Ignore DBG_VALUE. It does not contribute to any instruction in output.
      if (MInsn.isDebugValue())
        continue;

      if (RangeBeginMI) {
        // If we have already seen a beginning of an instruction range and
        // current instruction scope does not match scope of first instruction
        // in this range then create a new instruction range.
        InsnRange R(RangeBeginMI, PrevMI);
        MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
        MIRanges.push_back(R);
      }

      // This is a beginning of a new instruction range.
      RangeBeginMI = &MInsn;

      // Reset previous markers.
      PrevMI = &MInsn;
      PrevDL = MIDL;
    }

    // Create last instruction range.
    if (RangeBeginMI && PrevMI && PrevDL) {
      InsnRange R(RangeBeginMI, PrevMI);
      MIRanges.push_back(R);
      MI2ScopeMap[RangeBeginMI] = getOrCreateLexicalScope(PrevDL);
    }
  }
}

/// findLexicalScope - Find lexical scope, either regular or inlined, for the
/// given DebugLoc. Return NULL if not found.
LexicalScope *LexicalScopes::findLexicalScope(const DILocation *DL) {
  DILocalScope *Scope = DL->getScope();
  if (!Scope)
    return nullptr;

  // The scope that we were created with could have an extra file - which
  // isn't what we care about in this case.
  if (auto *File = dyn_cast<DILexicalBlockFile>(Scope))
    Scope = File->getScope();

  if (auto *IA = DL->getInlinedAt()) {
    auto I = InlinedLexicalScopeMap.find(std::make_pair(Scope, IA));
    return I != InlinedLexicalScopeMap.end() ? &I->second : nullptr;
  }
  return findLexicalScope(Scope);
}

/// getOrCreateLexicalScope - Find lexical scope for the given DebugLoc. If
/// not available then create new lexical scope.
LexicalScope *LexicalScopes::getOrCreateLexicalScope(const DILocalScope *Scope,
                                                     const DILocation *IA) {
  if (IA) {
    // Create an abstract scope for inlined function.
    getOrCreateAbstractScope(Scope);
    // Create an inlined scope for inlined function.
    return getOrCreateInlinedScope(Scope, IA);
  }

  return getOrCreateRegularScope(Scope);
}

/// getOrCreateRegularScope - Find or create a regular lexical scope.
LexicalScope *
LexicalScopes::getOrCreateRegularScope(const DILocalScope *Scope) {
  if (auto *File = dyn_cast<DILexicalBlockFile>(Scope))
    Scope = File->getScope();

  auto I = LexicalScopeMap.find(Scope);
  if (I != LexicalScopeMap.end())
    return &I->second;

  // FIXME: Should the following dyn_cast be DILexicalBlock?
  LexicalScope *Parent = nullptr;
  if (auto *Block = dyn_cast<DILexicalBlockBase>(Scope))
    Parent = getOrCreateLexicalScope(Block->getScope());
  I = LexicalScopeMap.emplace(std::piecewise_construct,
                              std::forward_as_tuple(Scope),
                              std::forward_as_tuple(Parent, Scope, nullptr,
                                                    false)).first;

  if (!Parent) {
    assert(cast<DISubprogram>(Scope)->describes(MF->getFunction()));
    assert(!CurrentFnLexicalScope);
    CurrentFnLexicalScope = &I->second;
  }

  return &I->second;
}

/// getOrCreateInlinedScope - Find or create an inlined lexical scope.
LexicalScope *
LexicalScopes::getOrCreateInlinedScope(const DILocalScope *Scope,
                                       const DILocation *InlinedAt) {
  std::pair<const DILocalScope *, const DILocation *> P(Scope, InlinedAt);
  auto I = InlinedLexicalScopeMap.find(P);
  if (I != InlinedLexicalScopeMap.end())
    return &I->second;

  LexicalScope *Parent;
  if (auto *Block = dyn_cast<DILexicalBlockBase>(Scope))
    Parent = getOrCreateInlinedScope(Block->getScope(), InlinedAt);
  else
    Parent = getOrCreateLexicalScope(InlinedAt);

  I = InlinedLexicalScopeMap.emplace(std::piecewise_construct,
                                     std::forward_as_tuple(P),
                                     std::forward_as_tuple(Parent, Scope,
                                                           InlinedAt, false))
          .first;
  return &I->second;
}

/// getOrCreateAbstractScope - Find or create an abstract lexical scope.
LexicalScope *
LexicalScopes::getOrCreateAbstractScope(const DILocalScope *Scope) {
  assert(Scope && "Invalid Scope encoding!");

  if (auto *File = dyn_cast<DILexicalBlockFile>(Scope))
    Scope = File->getScope();
  auto I = AbstractScopeMap.find(Scope);
  if (I != AbstractScopeMap.end())
    return &I->second;

  // FIXME: Should the following isa be DILexicalBlock?
  LexicalScope *Parent = nullptr;
  if (auto *Block = dyn_cast<DILexicalBlockBase>(Scope))
    Parent = getOrCreateAbstractScope(Block->getScope());

  I = AbstractScopeMap.emplace(std::piecewise_construct,
                               std::forward_as_tuple(Scope),
                               std::forward_as_tuple(Parent, Scope,
                                                     nullptr, true)).first;
  if (isa<DISubprogram>(Scope))
    AbstractScopesList.push_back(&I->second);
  return &I->second;
}

/// constructScopeNest
void LexicalScopes::constructScopeNest(LexicalScope *Scope) {
  assert(Scope && "Unable to calculate scope dominance graph!");
  SmallVector<LexicalScope *, 4> WorkStack;
  WorkStack.push_back(Scope);
  unsigned Counter = 0;
  while (!WorkStack.empty()) {
    LexicalScope *WS = WorkStack.back();
    const SmallVectorImpl<LexicalScope *> &Children = WS->getChildren();
    bool visitedChildren = false;
    for (SmallVectorImpl<LexicalScope *>::const_iterator SI = Children.begin(),
                                                         SE = Children.end();
         SI != SE; ++SI) {
      LexicalScope *ChildScope = *SI;
      if (!ChildScope->getDFSOut()) {
        WorkStack.push_back(ChildScope);
        visitedChildren = true;
        ChildScope->setDFSIn(++Counter);
        break;
      }
    }
    if (!visitedChildren) {
      WorkStack.pop_back();
      WS->setDFSOut(++Counter);
    }
  }
}

/// assignInstructionRanges - Find ranges of instructions covered by each
/// lexical scope.
void LexicalScopes::assignInstructionRanges(
    SmallVectorImpl<InsnRange> &MIRanges,
    DenseMap<const MachineInstr *, LexicalScope *> &MI2ScopeMap) {

  LexicalScope *PrevLexicalScope = nullptr;
  for (SmallVectorImpl<InsnRange>::const_iterator RI = MIRanges.begin(),
                                                  RE = MIRanges.end();
       RI != RE; ++RI) {
    const InsnRange &R = *RI;
    LexicalScope *S = MI2ScopeMap.lookup(R.first);
    assert(S && "Lost LexicalScope for a machine instruction!");
    if (PrevLexicalScope && !PrevLexicalScope->dominates(S))
      PrevLexicalScope->closeInsnRange(S);
    S->openInsnRange(R.first);
    S->extendInsnRange(R.second);
    PrevLexicalScope = S;
  }

  if (PrevLexicalScope)
    PrevLexicalScope->closeInsnRange();
}

/// getMachineBasicBlocks - Populate given set using machine basic blocks which
/// have machine instructions that belong to lexical scope identified by
/// DebugLoc.
void LexicalScopes::getMachineBasicBlocks(
    const DILocation *DL, SmallPtrSetImpl<const MachineBasicBlock *> &MBBs) {
  MBBs.clear();
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return;

  if (Scope == CurrentFnLexicalScope) {
    for (const auto &MBB : *MF)
      MBBs.insert(&MBB);
    return;
  }

  SmallVectorImpl<InsnRange> &InsnRanges = Scope->getRanges();
  for (SmallVectorImpl<InsnRange>::iterator I = InsnRanges.begin(),
                                            E = InsnRanges.end();
       I != E; ++I) {
    InsnRange &R = *I;
    MBBs.insert(R.first->getParent());
  }
}

/// dominates - Return true if DebugLoc's lexical scope dominates at least one
/// machine instruction's lexical scope in a given machine basic block.
bool LexicalScopes::dominates(const DILocation *DL, MachineBasicBlock *MBB) {
  LexicalScope *Scope = getOrCreateLexicalScope(DL);
  if (!Scope)
    return false;

  // Current function scope covers all basic blocks in the function.
  if (Scope == CurrentFnLexicalScope && MBB->getParent() == MF)
    return true;

  bool Result = false;
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
    if (const DILocation *IDL = I->getDebugLoc())
      if (LexicalScope *IScope = getOrCreateLexicalScope(IDL))
        if (Scope->dominates(IScope))
          return true;
  }
  return Result;
}

/// dump - Print data structures.
void LexicalScope::dump(unsigned Indent) const {
#ifndef NDEBUG
  raw_ostream &err = dbgs();
  err.indent(Indent);
  err << "DFSIn: " << DFSIn << " DFSOut: " << DFSOut << "\n";
  const MDNode *N = Desc;
  err.indent(Indent);
  N->dump();
  if (AbstractScope)
    err << std::string(Indent, ' ') << "Abstract Scope\n";

  if (!Children.empty())
    err << std::string(Indent + 2, ' ') << "Children ...\n";
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    if (Children[i] != this)
      Children[i]->dump(Indent + 2);
#endif
}
