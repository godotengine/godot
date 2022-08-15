///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaEnumTables.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaEnumTables.h"

#include "DxilDia.h"
#include "DxilDiaSession.h"
#include "DxilDiaTable.h"

STDMETHODIMP dxil_dia::EnumTables::get_Count(_Out_ LONG *pRetVal) {
  *pRetVal = ((unsigned)Table::LastKind - (unsigned)Table::FirstKind) + 1;
  return S_OK;
}

STDMETHODIMP dxil_dia::EnumTables::Item(
    /* [in] */ VARIANT index,
    /* [retval][out] */ IDiaTable **table) {
  // Avoid pulling in additional variant support (could have used VariantChangeType instead).
  DWORD indexVal;
  switch (index.vt) {
  case VT_UI4:
    indexVal = index.uintVal;
    break;
  case VT_I4:
    IFR(IntToDWord(index.intVal, &indexVal));
    break;
  default:
    return E_INVALIDARG;
  }
  if (indexVal > (unsigned)Table::LastKind) {
    return E_INVALIDARG;
  }
  HRESULT hr = S_OK;
  if (!m_tables[indexVal]) {
    DxcThreadMalloc TM(m_pMalloc);
    hr = Table::Create(m_pSession, (Table::Kind)indexVal, &m_tables[indexVal]);
  }
  m_tables[indexVal].p->AddRef();
  *table = m_tables[indexVal];
  return hr;
}

STDMETHODIMP dxil_dia::EnumTables::Next(
    ULONG celt,
    IDiaTable **rgelt,
    ULONG *pceltFetched) {
  DxcThreadMalloc TM(m_pMalloc);
  ULONG fetched = 0;
  while (fetched < celt && m_next <= (unsigned)Table::LastKind) {
    HRESULT hr = S_OK;
    if (!m_tables[m_next]) {
      DxcThreadMalloc TM(m_pMalloc);
      hr = Table::Create(m_pSession, (Table::Kind)m_next, &m_tables[m_next]);
      if (FAILED(hr)) {
        return hr; // TODO: this leaks prior tables.
      }
    }
    m_tables[m_next].p->AddRef();
    rgelt[fetched] = m_tables[m_next];
    ++m_next, ++fetched;
  }
  if (pceltFetched != nullptr)
    *pceltFetched = fetched;
  return (fetched == celt) ? S_OK : S_FALSE;
}

STDMETHODIMP dxil_dia::EnumTables::Reset() {
  m_next = 0;
  return S_OK;
}

HRESULT dxil_dia::EnumTables::Create(
    /* [in] */ dxil_dia::Session *pSession,
    /* [out] */ IDiaEnumTables **ppEnumTables) {
  *ppEnumTables = CreateOnMalloc<EnumTables>(pSession->GetMallocNoRef(), pSession);
  if (*ppEnumTables == nullptr)
    return E_OUTOFMEMORY;
  (*ppEnumTables)->AddRef();
  return S_OK;
}
