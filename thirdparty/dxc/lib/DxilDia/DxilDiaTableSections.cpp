///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSections.cpp                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableSections.h"

#include "DxilDiaSession.h"

dxil_dia::SectionsTable::SectionsTable(IMalloc *pMalloc, Session *pSession)
  : impl::TableBase<IDiaEnumSectionContribs, IDiaSectionContrib>(pMalloc, pSession, Table::Kind::Sections) {
}

HRESULT dxil_dia::SectionsTable::GetItem(DWORD index, IDiaSectionContrib **ppItem) {
  *ppItem = nullptr;
  return E_FAIL;
}
