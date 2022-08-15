///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixLowerPass.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides a high level matrix lower pass.                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace llvm {
class ModulePass;
class PassRegistry;

/// \brief Create and return a pass that lower high level matrix.
/// Note that this pass is designed for use with the legacy pass manager.
ModulePass *createHLMatrixLowerPass();
void initializeHLMatrixLowerPassPass(llvm::PassRegistry&);

}
