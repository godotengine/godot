//===- Signals.cpp - Signal Handling support --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Signals.h"
#include "llvm/Config/config.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Signals.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Signals.inc"
#endif
