//===- llvm/Transforms/Utils/UnrollLoop.h - Unrolling utilities -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H
#define LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class AssumptionCache;
class Loop;
class LoopInfo;
class LPPassManager;
class MDNode;
class Pass;

bool UnrollLoop(Loop *L, unsigned Count, unsigned TripCount, bool AllowRuntime,
                bool AllowExpensiveTripCount, unsigned TripMultiple,
                LoopInfo *LI, Pass *PP, LPPassManager *LPM,
                AssumptionCache *AC);

bool UnrollRuntimeLoopProlog(Loop *L, unsigned Count,
                             bool AllowExpensiveTripCount, LoopInfo *LI,
                             LPPassManager *LPM);

MDNode *GetUnrollMetadata(MDNode *LoopID, StringRef Name);
}

#endif
