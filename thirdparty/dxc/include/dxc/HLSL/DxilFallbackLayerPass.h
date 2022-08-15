///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilFallbackLayerPass.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides passes used by the Ray Tracing Fallback Layer          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace llvm {
    ModulePass *createDxilUpdateMetadataPass();
    ModulePass *createDxilPatchShaderRecordBindingsPass();

    void initializeDxilUpdateMetadataPass(llvm::PassRegistry&);
    void initializeDxilPatchShaderRecordBindingsPass(llvm::PassRegistry&);
}