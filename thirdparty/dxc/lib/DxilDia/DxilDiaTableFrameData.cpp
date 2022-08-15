///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableFrameData.cpp                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableFrameData.h"

#include "DxilDiaSession.h"

dxil_dia::FrameDataTable::FrameDataTable(IMalloc *pMalloc, Session *pSession)
  : impl::TableBase<IDiaEnumFrameData, IDiaFrameData>(pMalloc, pSession, Table::Kind::FrameData) {
}

HRESULT dxil_dia::FrameDataTable::GetItem(DWORD index, IDiaFrameData **ppItem) {
  *ppItem = nullptr;
  return E_FAIL;
}