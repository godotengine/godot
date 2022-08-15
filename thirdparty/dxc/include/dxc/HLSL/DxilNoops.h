///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilNoops.h                                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm {
  class Instruction;
}

namespace hlsl {
static const llvm::StringRef kNoopName = "dx.noop";
static const llvm::StringRef kPreservePrefix = "dx.preserve.";
static const llvm::StringRef kNothingName = "dx.nothing.a";
static const llvm::StringRef kPreserveName = "dx.preserve.value.a";

bool IsPreserveRelatedValue(llvm::Instruction *I);
bool IsPreserve(llvm::Instruction *S);
bool IsNop(llvm::Instruction *I);
}
