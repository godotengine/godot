//===- Verifier.h - LLVM IR Verifier ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system, and for checking that transformations
// haven't done something bad.
//
// Note that this does not provide full 'java style' security and verifications,
// instead it just tries to ensure that code is well formed.
//
// To see what specifically is checked, look at the top of Verifier.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_VERIFIER_H
#define LLVM_IR_VERIFIER_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {

class Function;
class FunctionPass;
class ModulePass;
class Module;
class PreservedAnalyses;
class raw_ostream;

/// \brief Check a function for errors, useful for use when debugging a
/// pass.
///
/// If there are no errors, the function returns false. If an error is found,
/// a message describing the error is written to OS (if non-null) and true is
/// returned.
bool verifyFunction(const Function &F, raw_ostream *OS = nullptr);

/// \brief Check a module for errors.
///
/// If there are no errors, the function returns false. If an error is found,
/// a message describing the error is written to OS (if non-null) and true is
/// returned.
bool verifyModule(const Module &M, raw_ostream *OS = nullptr);

/// \brief Create a verifier pass.
///
/// Check a module or function for validity. This is essentially a pass wrapped
/// around the above verifyFunction and verifyModule routines and
/// functionality. When the pass detects a verification error it is always
/// printed to stderr, and by default they are fatal. You can override that by
/// passing \c false to \p FatalErrors.
///
/// Note that this creates a pass suitable for the legacy pass manager. It has
/// nothing to do with \c VerifierPass.
FunctionPass *createVerifierPass(bool FatalErrors = true);

class VerifierPass {
  bool FatalErrors;

public:
  explicit VerifierPass(bool FatalErrors = true) : FatalErrors(FatalErrors) {}

  PreservedAnalyses run(Module &M);
  PreservedAnalyses run(Function &F);

  static StringRef name() { return "VerifierPass"; }
};

} // End llvm namespace

#endif
