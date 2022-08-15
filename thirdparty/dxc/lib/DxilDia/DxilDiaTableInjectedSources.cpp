///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableInjectedSources.cpp                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableInjectedSources.h"

#include "DxilDia.h"
#include "DxilDiaSession.h"
#include "DxilDiaTable.h"

llvm::MDTuple *dxil_dia::InjectedSource::NameContent() {
  return llvm::cast<llvm::MDTuple>(m_pSession->Contents()->getOperand(m_index));
}

llvm::StringRef dxil_dia::InjectedSource::Name() {
  return llvm::dyn_cast<llvm::MDString>(NameContent()->getOperand(0))->getString();
}

llvm::StringRef dxil_dia::InjectedSource::Content() {
  return llvm::dyn_cast<llvm::MDString>(NameContent()->getOperand(1))->getString();
}

STDMETHODIMP dxil_dia::InjectedSource::get_length(_Out_ ULONGLONG *pRetVal) {
  *pRetVal = Content().size();
  return S_OK;
}

STDMETHODIMP dxil_dia::InjectedSource::get_filename(BSTR *pRetVal) {
  DxcThreadMalloc TM(m_pMalloc);
  return StringRefToBSTR(Name(), pRetVal);
}

STDMETHODIMP dxil_dia::InjectedSource::get_objectFilename(BSTR *pRetVal) {
  *pRetVal = nullptr;
  return S_OK;
}

STDMETHODIMP dxil_dia::InjectedSource::get_virtualFilename(BSTR *pRetVal) {
  return get_filename(pRetVal);
}

STDMETHODIMP dxil_dia::InjectedSource::get_source(
  /* [in] */ DWORD cbData,
  /* [out] */ DWORD *pcbData,
  /* [size_is][out] */ BYTE *pbData) {
  if (pbData == nullptr) {
    if (pcbData != nullptr) {
      *pcbData = Content().size();
    }
    return S_OK;
  }

  cbData = std::min((DWORD)Content().size(), cbData);
  memcpy(pbData, Content().begin(), cbData);
  if (pcbData) {
    *pcbData = cbData;
  }
  return S_OK;
}

dxil_dia::InjectedSourcesTable::InjectedSourcesTable(
  IMalloc *pMalloc,
  Session *pSession)
  : impl::TableBase<IDiaEnumInjectedSources,
                    IDiaInjectedSource>(pMalloc, pSession, Table::Kind::InjectedSource) {
  // Count the number of source files available.
  // m_count = m_pSession->InfoRef().compile_unit_count();
  m_count =
    (m_pSession->Contents() == nullptr) ? 0 : m_pSession->Contents()->getNumOperands();
}

HRESULT dxil_dia::InjectedSourcesTable::GetItem(DWORD index, IDiaInjectedSource **ppItem) {
  if (index >= m_count)
    return E_INVALIDARG;
  unsigned itemIndex = index;
  if (m_count == m_indexList.size())
    itemIndex = m_indexList[index];
  *ppItem = CreateOnMalloc<InjectedSource>(m_pMalloc, m_pSession, itemIndex);
  if (*ppItem == nullptr)
    return E_OUTOFMEMORY;
  (*ppItem)->AddRef();
  return S_OK;
}

void dxil_dia::InjectedSourcesTable::Init(llvm::StringRef filename) {
  for (unsigned i = 0; i < m_pSession->Contents()->getNumOperands(); ++i) {
    llvm::StringRef fn =
      llvm::dyn_cast<llvm::MDString>(m_pSession->Contents()->getOperand(i)->getOperand(0))
      ->getString();
    if (fn.equals(filename)) {
      m_indexList.emplace_back(i);
    }
  }
  m_count = m_indexList.size();
}