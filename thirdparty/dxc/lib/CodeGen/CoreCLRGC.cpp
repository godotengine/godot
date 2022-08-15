//===-- CoreCLRGC.cpp - CoreCLR Runtime GC Strategy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a GCStrategy for the CoreCLR Runtime.
// The strategy is similar to Statepoint-example GC, but differs from it in
// certain aspects, such as:
// 1) Base-pointers need not be explicitly tracked and reported for
//    interior pointers
// 2) Uses a different format for encoding stack-maps
// 3) Location of Safe-point polls: polls are only needed before loop-back edges
//    and before tail-calls (not needed at function-entry)
//
// The above differences in behavior are to be implemented in upcoming checkins.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"

using namespace llvm;

namespace {
class CoreCLRGC : public GCStrategy {
public:
  CoreCLRGC() {
    UseStatepoints = true;
    // These options are all gc.root specific, we specify them so that the
    // gc.root lowering code doesn't run.
    InitRoots = false;
    NeededSafePoints = 0;
    UsesMetadata = false;
    CustomRoots = false;
  }
  Optional<bool> isGCManagedPointer(const Value *V) const override {
    // Method is only valid on pointer typed values.
    PointerType *PT = cast<PointerType>(V->getType());
    // We pick addrspace(1) as our GC managed heap.
    return (1 == PT->getAddressSpace());
  }
};
}

static GCRegistry::Add<CoreCLRGC> X("coreclr", "CoreCLR-compatible GC");

namespace llvm {
void linkCoreCLRGC() {}
}
