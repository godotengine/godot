//===-- COM.cpp - Implement COM utility classes -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements utility classes related to COM.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/COM.h"

#include "llvm/Config/config.h"

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/COM.inc"
#elif LLVM_ON_WIN32
#include "Windows/COM.inc"
#endif
