///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSourceFiles.cpp                                               //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableSourceFiles.h"

#include "DxilDiaSession.h"

dxil_dia::SourceFile::SourceFile(IMalloc *pMalloc, Session *pSession, DWORD index)
  : m_pMalloc(pMalloc), m_pSession(pSession), m_index(index) {}

llvm::MDTuple *dxil_dia::SourceFile::NameContent() const {
  return llvm::cast<llvm::MDTuple>(m_pSession->Contents()->getOperand(m_index));
}
llvm::StringRef dxil_dia::SourceFile::Name() const {
  return llvm::dyn_cast<llvm::MDString>(NameContent()->getOperand(0))->getString();
}

STDMETHODIMP dxil_dia::SourceFile::get_uniqueId(
  /* [retval][out] */ DWORD *pRetVal) {
  *pRetVal = m_index;
  return S_OK;
}

STDMETHODIMP dxil_dia::SourceFile::get_fileName(
  /* [retval][out] */ BSTR *pRetVal) {
  DxcThreadMalloc TM(m_pMalloc);
  return StringRefToBSTR(Name(), pRetVal);
}

dxil_dia::SourceFilesTable::SourceFilesTable(
  IMalloc *pMalloc,
  Session *pSession)
  : impl::TableBase<IDiaEnumSourceFiles, IDiaSourceFile>(pMalloc, pSession, Table::Kind::SourceFiles) {
    m_count =
      (m_pSession->Contents() == nullptr) ? 0 : m_pSession->Contents()->getNumOperands();
    m_items.assign(m_count, nullptr);
  }

dxil_dia::SourceFilesTable::SourceFilesTable(
    IMalloc *pMalloc,
    Session *pSession,
    std::vector<CComPtr<IDiaSourceFile>> &&items)
    : impl::TableBase<IDiaEnumSourceFiles, IDiaSourceFile>(pMalloc, pSession, Table::Kind::SourceFiles),
      m_items(std::move(items)) {
    m_count = m_items.size();
}

HRESULT dxil_dia::SourceFilesTable::GetItem(DWORD index, IDiaSourceFile **ppItem) {
  if (!m_items[index]) {
    m_items[index] = CreateOnMalloc<SourceFile>(m_pMalloc, m_pSession, index);
    if (m_items[index] == nullptr)
      return E_OUTOFMEMORY;
  }
  m_items[index].p->AddRef();
  *ppItem = m_items[index];
  (*ppItem)->AddRef();
  return S_OK;
}
