//===- llvm/Support/Host.h - Host machine characteristics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Methods for querying the nature of the host machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HOST_H
#define LLVM_SUPPORT_HOST_H

#include "llvm/ADT/StringMap.h"

#if defined(__linux__) || defined(__GNU__)
#include <endian.h>
#else
#if !defined(BYTE_ORDER) && !defined(LLVM_ON_WIN32)
#include <machine/endian.h>
#endif
#endif

#include <string>

namespace llvm {
namespace sys {

#if defined(BYTE_ORDER) && defined(BIG_ENDIAN) && BYTE_ORDER == BIG_ENDIAN
  static const bool IsBigEndianHost = true;
#else
  static const bool IsBigEndianHost = false;
#endif

  static const bool IsLittleEndianHost = !IsBigEndianHost;

  /// getDefaultTargetTriple() - Return the default target triple the compiler
  /// has been configured to produce code for.
  ///
  /// The target triple is a string in the format of:
  ///   CPU_TYPE-VENDOR-OPERATING_SYSTEM
  /// or
  ///   CPU_TYPE-VENDOR-KERNEL-OPERATING_SYSTEM

  // HLSL Change - single triple supported, no allocation required
#ifdef _WIN32
  const char *getDefaultTargetTriple();
#else
  std::string getDefaultTargetTriple();
#endif

  /// getProcessTriple() - Return an appropriate target triple for generating
  /// code to be loaded into the current process, e.g. when using the JIT.
  std::string getProcessTriple();

  /// getHostCPUName - Get the LLVM name for the host CPU. The particular format
  /// of the name is target dependent, and suitable for passing as -mcpu to the
  /// target which matches the host.
  ///
  /// \return - The host CPU name, or empty if the CPU could not be determined.
  StringRef getHostCPUName();

  /// getHostCPUFeatures - Get the LLVM names for the host CPU features.
  /// The particular format of the names are target dependent, and suitable for
  /// passing as -mattr to the target which matches the host.
  ///
  /// \param Features - A string mapping feature names to either
  /// true (if enabled) or false (if disabled). This routine makes no guarantees
  /// about exactly which features may appear in this map, except that they are
  /// all valid LLVM feature names.
  ///
  /// \return - True on success.
  bool getHostCPUFeatures(StringMap<bool> &Features);
}
}

#endif
