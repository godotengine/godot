//===-- OrcTargetSupport.h - Code to support specific targets  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Target specific code for Orc, e.g. callback assembly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H
#define LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H

#include "IndirectionUtils.h"

namespace llvm {
namespace orc {

class OrcX86_64 {
public:
  static const char *ResolverBlockName;

  /// @brief Insert module-level inline callback asm into module M for the
  /// symbols managed by JITResolveCallbackHandler J.
  static void insertResolverBlock(Module &M,
                                  JITCompileCallbackManagerBase &JCBM);

  /// @brief Get a label name from the given index.
  typedef std::function<std::string(unsigned)> LabelNameFtor;

  /// @brief Insert the requested number of trampolines into the given module.
  /// @param M Module to insert the call block into.
  /// @param NumCalls Number of calls to create in the call block.
  /// @param StartIndex Optional argument specifying the index suffix to start
  ///                   with.
  /// @return A functor that provides the symbol name for each entry in the call
  ///         block.
  ///
  static LabelNameFtor insertCompileCallbackTrampolines(
                                                    Module &M,
                                                    TargetAddress TrampolineAddr,
                                                    unsigned NumCalls,
                                                    unsigned StartIndex = 0);

};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_ORCTARGETSUPPORT_H
