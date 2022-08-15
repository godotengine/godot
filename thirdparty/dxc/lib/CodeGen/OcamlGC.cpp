//===-- OcamlGC.cpp - Ocaml frametable GC strategy ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering for the llvm.gc* intrinsics compatible with
// Objective Caml 3.10.0, which uses a liveness-accurate static stack map.
//
// The frametable emitter is in OcamlGCPrinter.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

namespace {
class OcamlGC : public GCStrategy {
public:
  OcamlGC();
};
}

static GCRegistry::Add<OcamlGC> X("ocaml", "ocaml 3.10-compatible GC");

void llvm::linkOcamlGC() {}

OcamlGC::OcamlGC() {
  NeededSafePoints = 1 << GC::PostCall;
  UsesMetadata = true;
}
