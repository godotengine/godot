///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDia.cpp                                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDia.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"

HRESULT dxil_dia::StringRefToBSTR(llvm::StringRef value, BSTR *pRetVal) {
  try {
    wchar_t *wide;
    size_t sideSize;
    if (!Unicode::UTF8BufferToWideBuffer(value.data(), value.size(), &wide,
      &sideSize))
      return E_FAIL;
    *pRetVal = SysAllocString(wide);
    delete[] wide;
  }
  CATCH_CPP_RETURN_HRESULT();
  return S_OK;
}

HRESULT dxil_dia::ENotImpl() {
  return E_NOTIMPL;
}
