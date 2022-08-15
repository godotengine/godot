//===-- Comdat.cpp - Implement Metadata classes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Comdat class.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Comdat.h"
#include "llvm/ADT/StringMap.h"
using namespace llvm;

Comdat::Comdat(SelectionKind SK, StringMapEntry<Comdat> *Name)
    : Name(Name), SK(SK) {}

Comdat::Comdat(Comdat &&C) : Name(C.Name), SK(C.SK) {}

Comdat::Comdat() : Name(nullptr), SK(Comdat::Any) {}

StringRef Comdat::getName() const { return Name->first(); }
