///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLOperationLower.h                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower functions to lower HL operations to DXIL operations.                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <unordered_set>

namespace llvm {
class Instruction;
class Function;
}

namespace hlsl {
class HLModule;
class DxilResourceBase;
class HLSLExtensionsCodegenHelper;

void TranslateBuiltinOperations(
    HLModule &HLM, HLSLExtensionsCodegenHelper *extCodegenHelper,
    std::unordered_set<llvm::Instruction *> &UpdateCounterSet);
}