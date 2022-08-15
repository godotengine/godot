///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableInputAssemblyFile.cpp                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableInputAssemblyFile.h"

#include "DxilDiaSession.h"

dxil_dia::InputAssemblyFilesTable::InputAssemblyFilesTable(IMalloc *pMalloc, Session *pSession)
  : impl::TableBase<IDiaEnumInputAssemblyFiles, IDiaInputAssemblyFile>(pMalloc, pSession, Table::Kind::InputAssemblyFile) {

}

HRESULT dxil_dia::InputAssemblyFilesTable::GetItem(DWORD index, IDiaInputAssemblyFile **ppItem) {
  *ppItem = nullptr;
  return E_FAIL;
}