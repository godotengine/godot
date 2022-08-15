///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Global.cpp                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //

//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"

#include <system_error>

#include "dxc/Support/WinIncludes.h"


void CheckLLVMErrorCode(const std::error_code &ec) {
  if (ec) {
    DXASSERT(ec.category() == std::system_category(), "unexpected LLVM exception code");
    throw hlsl::Exception(HRESULT_FROM_WIN32(ec.value()));
  }
}

static_assert(unsigned(DXC_E_OVERLAPPING_SEMANTICS) == 0x80AA0001, "Sanity check for DXC errors failed");
