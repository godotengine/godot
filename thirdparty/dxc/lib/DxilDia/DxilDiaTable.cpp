///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTable.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTable.h"

#include "DxilDiaSession.h"
#include "DxilDiaTableFrameData.h"
#include "DxilDiaTableInjectedSources.h"
#include "DxilDiaTableInputAssemblyFile.h"
#include "DxilDiaTableLineNumbers.h"
#include "DxilDiaTableSections.h"
#include "DxilDiaTableSegmentMap.h"
#include "DxilDiaTableSourceFiles.h"
#include "DxilDiaTableSymbols.h"

HRESULT dxil_dia::Table::Create(
    /* [in] */ Session *pSession,
    /* [in] */ Table::Kind kind,
    /* [out] */ IDiaTable **ppTable) {
  try {
    *ppTable = nullptr;
    IMalloc *pMalloc = pSession->GetMallocNoRef();
    switch (kind) {
    case Table::Kind::Symbols: *ppTable = CreateOnMalloc<SymbolsTable>(pMalloc, pSession); break;
    case Table::Kind::SourceFiles: *ppTable = CreateOnMalloc<SourceFilesTable>(pMalloc, pSession); break;
    case Table::Kind::LineNumbers: *ppTable = CreateOnMalloc<LineNumbersTable>(pMalloc, pSession); break;
    case Table::Kind::Sections: *ppTable = CreateOnMalloc<SectionsTable>(pMalloc, pSession); break;
    case Table::Kind::SegmentMap: *ppTable = CreateOnMalloc<SegmentMapTable>(pMalloc, pSession); break;
    case Table::Kind::InjectedSource: *ppTable = CreateOnMalloc<InjectedSourcesTable>(pMalloc, pSession); break;
    case Table::Kind::FrameData: *ppTable = CreateOnMalloc<FrameDataTable>(pMalloc, pSession); break;
    case Table::Kind::InputAssemblyFile: *ppTable = CreateOnMalloc<InputAssemblyFilesTable>(pMalloc, pSession); break;
    default: return E_FAIL;
    }
    if (*ppTable == nullptr)
      return E_OUTOFMEMORY;
    (*ppTable)->AddRef();
    return S_OK;
  } CATCH_CPP_RETURN_HRESULT();
}
