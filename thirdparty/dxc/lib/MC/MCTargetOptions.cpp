//===- lib/MC/MCTargetOptions.cpp - MC Target Options --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCTargetOptions.h"

#if 0 // HLSL Change

namespace llvm {

MCTargetOptions::MCTargetOptions()
    : SanitizeAddress(false), MCRelaxAll(false), MCNoExecStack(false),
      MCFatalWarnings(false), MCSaveTempLabels(false),
      MCUseDwarfDirectory(false), ShowMCEncoding(false), ShowMCInst(false),
      AsmVerbose(false), DwarfVersion(0), ABIName() {}

StringRef MCTargetOptions::getABIName() const {
  return ABIName;
}

} // end namespace llvm

#endif // HLSL Change
