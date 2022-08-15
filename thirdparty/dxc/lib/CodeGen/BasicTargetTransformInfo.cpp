//===- BasicTargetTransformInfo.cpp - Basic target-independent TTI impl ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the implementation of a basic TargetTransformInfo pass
/// predicated on the target abstractions present in the target independent
/// code generator. It uses these (primarily TargetLowering) to model as much
/// of the TTI query interface as possible. It is included by most targets so
/// that they can specialize only a small subset of the query space.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "basictti"

// This flag is used by the template base class for BasicTTIImpl, and here to
// provide a definition.
cl::opt<unsigned>
    llvm::PartialUnrollingThreshold("partial-unrolling-threshold", cl::init(0),
                                    cl::desc("Threshold for partial unrolling"),
                                    cl::Hidden);

BasicTTIImpl::BasicTTIImpl(const TargetMachine *TM, Function &F)
    : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
      TLI(ST->getTargetLowering()) {}
