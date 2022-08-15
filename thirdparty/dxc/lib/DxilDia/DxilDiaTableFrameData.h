///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableFrameData.h                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include "dia2.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class FrameDataTable : public impl::TableBase<IDiaEnumFrameData, IDiaFrameData> {
public:
  FrameDataTable(IMalloc *pMalloc, Session *pSession);

  // HLSL inlines functions for a program, so no data to return.
  STDMETHODIMP frameByRVA(
    /* [in] */ DWORD relativeVirtualAddress,
    /* [retval][out] */ IDiaFrameData **frame) override { return ENotImpl(); }

  STDMETHODIMP frameByVA(
    /* [in] */ ULONGLONG virtualAddress,
    /* [retval][out] */ IDiaFrameData **frame) override { return ENotImpl(); }

  HRESULT GetItem(DWORD index, IDiaFrameData **ppItem) override;
};

}  // namespace dxil_dia
