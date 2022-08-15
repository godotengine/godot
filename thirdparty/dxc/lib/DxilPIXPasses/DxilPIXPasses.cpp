///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPIXPasses.cpp                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides SetupRegistryPassForPIX.                                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/WinIncludes.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;
using namespace hlsl;

namespace hlsl {

HRESULT SetupRegistryPassForPIX() {
  try {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
#include "DxilPIXPasses.inc"
  }
  CATCH_CPP_RETURN_HRESULT();
  return S_OK;
}

} // namespace hlsl
