///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSourceFiles.h                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <vector>

#include "dia2.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Metadata.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {

class Session;

class SourceFile : public IDiaSourceFile {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<Session> m_pSession;
  DWORD m_index;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaSourceFile>(this, iid, ppvObject);
  }

  SourceFile(IMalloc *pMalloc, Session *pSession, DWORD index);

  llvm::MDTuple *NameContent() const;

  llvm::StringRef Name() const;

  STDMETHODIMP get_uniqueId(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_fileName(
    /* [retval][out] */ BSTR *pRetVal) override;

  STDMETHODIMP get_checksumType(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_compilands(
    /* [retval][out] */ IDiaEnumSymbols **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_checksum(
    /* [in] */ DWORD cbData,
    /* [out] */ DWORD *pcbData,
    /* [size_is][out] */ BYTE *pbData) override { return ENotImpl(); }
};

class SourceFilesTable : public impl::TableBase<IDiaEnumSourceFiles, IDiaSourceFile> {
public:
  SourceFilesTable(IMalloc *pMalloc, Session *pSession);
  SourceFilesTable(IMalloc *pMalloc, Session *pSession,
                   std::vector<CComPtr<IDiaSourceFile>> &&items);

  HRESULT GetItem(DWORD index, IDiaSourceFile **ppItem) override;

private:
  std::vector<CComPtr<IDiaSourceFile>> m_items;
};

}  // namespace dxil_dia
