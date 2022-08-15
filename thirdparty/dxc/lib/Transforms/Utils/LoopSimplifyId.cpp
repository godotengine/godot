//===- LoopSimplifyId.cpp - ID for the Loop Canonicalization Pass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//

#include "llvm/Pass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

using namespace llvm;

char LoopSimplify::ID = 0;

// Publicly exposed interface to pass...
// This is in a separate file instead of LoopSimplify.cpp which brings in many dependencies
// unnecessary increasing the size of dxilconv.dll.
char &llvm::LoopSimplifyID = LoopSimplify::ID;