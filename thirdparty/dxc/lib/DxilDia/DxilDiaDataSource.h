///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaDataSource.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <memory>

#include "dia2.h"

#include "dxc/DXIL/DxilModule.h"
#include "dxc/Support/Global.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class DataSource : public IDiaDataSource {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  std::shared_ptr<llvm::Module> m_module;
  std::shared_ptr<llvm::LLVMContext> m_context;
  std::shared_ptr<llvm::DebugInfoFinder> m_finder;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()

  STDMETHODIMP QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaDataSource>(this, iid, ppvObject);
  }

  DataSource(IMalloc *pMalloc);

  ~DataSource();

  STDMETHODIMP get_lastError(BSTR *pRetVal) override;

  STDMETHODIMP loadDataFromPdb(_In_ LPCOLESTR pdbPath) override { return ENotImpl(); }

  STDMETHODIMP loadAndValidateDataFromPdb(
    _In_ LPCOLESTR pdbPath,
    _In_ GUID *pcsig70,
    _In_ DWORD sig,
    _In_ DWORD age) override { return ENotImpl(); }

  STDMETHODIMP loadDataForExe(
    _In_ LPCOLESTR executable,
    _In_ LPCOLESTR searchPath,
    _In_ IUnknown *pCallback) override { return ENotImpl(); }

  STDMETHODIMP loadDataFromIStream(_In_ IStream *pIStream) override;

  STDMETHODIMP openSession(_COM_Outptr_ IDiaSession **ppSession) override;

  HRESULT STDMETHODCALLTYPE loadDataFromCodeViewInfo(
    _In_ LPCOLESTR executable,
    _In_ LPCOLESTR searchPath,
    _In_ DWORD cbCvInfo,
    _In_ BYTE *pbCvInfo,
    _In_ IUnknown *pCallback) override { return ENotImpl(); }

  HRESULT STDMETHODCALLTYPE loadDataFromMiscInfo(
    _In_ LPCOLESTR executable,
    _In_ LPCOLESTR searchPath,
    _In_ DWORD timeStampExe,
    _In_ DWORD timeStampDbg,
    _In_ DWORD sizeOfExe,
    _In_ DWORD cbMiscInfo,
    _In_ BYTE *pbMiscInfo,
    _In_ IUnknown *pCallback) override { return ENotImpl(); }
};

}  // namespace dxil_dia
