//===- MCTargetOptions.h - MC Target Options -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONS_H
#define LLVM_MC_MCTARGETOPTIONS_H

#include <string>

namespace llvm {

class StringRef;

class MCTargetOptions {
public:
  enum AsmInstrumentation {
    AsmInstrumentationNone,
    AsmInstrumentationAddress
  };

  /// Enables AddressSanitizer instrumentation at machine level.
  bool SanitizeAddress : 1;

  bool MCRelaxAll : 1;
  bool MCNoExecStack : 1;
  bool MCFatalWarnings : 1;
  bool MCSaveTempLabels : 1;
  bool MCUseDwarfDirectory : 1;
  bool ShowMCEncoding : 1;
  bool ShowMCInst : 1;
  bool AsmVerbose : 1;
  int DwarfVersion;
  /// getABIName - If this returns a non-empty string this represents the
  /// textual name of the ABI that we want the backend to use, e.g. o32, or
  /// aapcs-linux.
  StringRef getABIName() const { // HLSL Change - inline implementation
    return ABIName;
  }
  ::std::string ABIName;
  MCTargetOptions()   // HLSL Change - inline implementation
      : SanitizeAddress(false), MCRelaxAll(false), MCNoExecStack(false),
      MCFatalWarnings(false), MCSaveTempLabels(false),
      MCUseDwarfDirectory(false), ShowMCEncoding(false), ShowMCInst(false),
      AsmVerbose(false), DwarfVersion(0), ABIName() {}
};

inline bool operator==(const MCTargetOptions &LHS, const MCTargetOptions &RHS) {
#define ARE_EQUAL(X) LHS.X == RHS.X
  return (ARE_EQUAL(SanitizeAddress) &&
          ARE_EQUAL(MCRelaxAll) &&
          ARE_EQUAL(MCNoExecStack) &&
          ARE_EQUAL(MCFatalWarnings) &&
          ARE_EQUAL(MCSaveTempLabels) &&
          ARE_EQUAL(MCUseDwarfDirectory) &&
          ARE_EQUAL(ShowMCEncoding) &&
          ARE_EQUAL(ShowMCInst) &&
          ARE_EQUAL(AsmVerbose) &&
          ARE_EQUAL(DwarfVersion) &&
          ARE_EQUAL(ABIName));
#undef ARE_EQUAL
}

inline bool operator!=(const MCTargetOptions &LHS, const MCTargetOptions &RHS) {
  return !(LHS == RHS);
}

} // end namespace llvm

#endif
