///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSegmentMap.cpp                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableSegmentMap.h"

#include "DxilDiaSession.h"

dxil_dia::SegmentMapTable::SegmentMapTable(IMalloc *pMalloc, Session *pSession)
  : impl::TableBase<IDiaEnumSegments, IDiaSegment>(pMalloc, pSession, Table::Kind::SegmentMap) {
}

HRESULT dxil_dia::SegmentMapTable::GetItem(DWORD index, IDiaSegment **ppItem) {
  *ppItem = nullptr;
  return E_FAIL;
}