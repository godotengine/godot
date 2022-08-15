//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PostDominators.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
using namespace llvm;

#define DEBUG_TYPE "postdomtree"

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

char PostDominatorTree::ID = 0;
INITIALIZE_PASS(PostDominatorTree, "postdomtree",
                "Post-Dominator Tree Construction", true, true)

bool PostDominatorTree::runOnFunction(Function &F) {
  DT->recalculate(F);
  return false;
}

PostDominatorTree::~PostDominatorTree() {
  delete DT;
}

void PostDominatorTree::print(raw_ostream &OS, const Module *) const {
  DT->print(OS);
}


FunctionPass* llvm::createPostDomTree() {
  return new PostDominatorTree();
}

