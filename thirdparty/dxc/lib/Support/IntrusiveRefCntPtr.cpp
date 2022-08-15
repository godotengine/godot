//== IntrusiveRefCntPtr.cpp - Smart Refcounting Pointer ----------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntrusiveRefCntPtr.h"

using namespace llvm;

void RefCountedBaseVPTR::anchor() { }
