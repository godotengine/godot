///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ControlDependence.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Control dependence is computed using algorithm in Figure 7.9 from [AK].   //
//                                                                           //
// References                                                                //
// [AK] Optimizing Compilers for Modern Architectures by Allen and Kennedy.  //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/ControlDependence.h"
#include "dxc/Support/Global.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace hlsl;


const BasicBlockSet &ControlDependence::GetCDBlocks(BasicBlock *pBB) const {
  auto it = m_ControlDependence.find(pBB);
  if (it != m_ControlDependence.end())
    return it->second;
  else
    return m_EmptyBBSet;
}

void ControlDependence::print(raw_ostream &OS) {
  OS << "Control dependence for function '" << m_pFunc->getName() << "'\n";
  for (auto &it : m_ControlDependence) {
    BasicBlock *pBB = it.first;
    OS << "Block " << pBB->getName() << ": { ";
    bool bFirst = true;
    for (BasicBlock *pBB2 : it.second) {
      if (!bFirst) OS << ", ";
      OS << pBB2->getName();
      bFirst = false;
    }
    OS << " }\n";
  }
  OS << "\n";
}

void ControlDependence::dump() {
  print(dbgs());
}

void ControlDependence::Compute(Function *F, PostDomRelationType &PostDomRel) {
  m_pFunc = F;

  // Compute reverse topological order of PDT.
  BasicBlockVector RevTopOrder;
  BasicBlockSet VisitedBBs;
  for (BasicBlock *pBB : PostDomRel.getRoots()) {
    ComputeRevTopOrderRec(PostDomRel, pBB, RevTopOrder, VisitedBBs);
  }
  DXASSERT_NOMSG(RevTopOrder.size() == VisitedBBs.size());

  // Compute control dependence relation.
  for (size_t iBB = 0; iBB < RevTopOrder.size(); iBB++) {
    BasicBlock *x = RevTopOrder[iBB];

    // For each y = pred(x): if ipostdom(y) != x then add "x is control dependent on y"
    for (auto itPred = pred_begin(x), endPred = pred_end(x); itPred != endPred; ++itPred) {
      BasicBlock *y = *itPred;  // predecessor of x
      BasicBlock *pPredIDomBB = GetIPostDom(PostDomRel, y);
      if (pPredIDomBB != x) {
        m_ControlDependence[x].insert(y);
      }
    }

    // For all z such that ipostdom(z) = x
    for (DomTreeNode *child : PostDomRel.getNode(x)->getChildren()) {
      BasicBlock *z = child->getBlock();

      auto it = m_ControlDependence.find(z);
      if (it == m_ControlDependence.end())
        continue;

      // For all y in CDG(z)
      for (BasicBlock *y : it->second) {
        // if ipostdom(y) != x then add "x is control dependent on y" 
        BasicBlock *pPredIDomBB = GetIPostDom(PostDomRel, y);
        if (pPredIDomBB != x) {
          m_ControlDependence[x].insert(y);
        }
      }
    }
  }
}

void ControlDependence::Clear() {
  m_pFunc = nullptr;
  m_ControlDependence.clear();
  m_EmptyBBSet.clear();
}

BasicBlock *ControlDependence::GetIPostDom(PostDomRelationType &PostDomRel, BasicBlock *pBB) {
  auto *pPDTNode = PostDomRel.getNode(pBB);
  auto *pIDomNode = pPDTNode->getIDom();
  BasicBlock *pIDomBB = pIDomNode != nullptr ? pIDomNode->getBlock() : nullptr;
  return pIDomBB;
}

void ControlDependence::ComputeRevTopOrderRec(PostDomRelationType &PostDomRel,
                                              BasicBlock *pBB,
                                              BasicBlockVector &RevTopOrder,
                                              BasicBlockSet &VisitedBBs) {
  if (VisitedBBs.find(pBB) != VisitedBBs.end()) {
    return;
  }
  VisitedBBs.insert(pBB);

  SmallVector<BasicBlock *, 8> Descendants;
  PostDomRel.getDescendants(pBB, Descendants);
  for (BasicBlock *pDescBB : Descendants) {
    if (pDescBB != pBB)
      ComputeRevTopOrderRec(PostDomRel, pDescBB, RevTopOrder, VisitedBBs);
  }

  RevTopOrder.emplace_back(pBB);
}
