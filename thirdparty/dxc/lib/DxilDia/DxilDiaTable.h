///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTable.h                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include "dia2.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"

namespace dxil_dia {

class Session;

namespace Table {
enum class Kind {
  Symbols,
  SourceFiles,
  LineNumbers,
  Sections,
  SegmentMap,
  InjectedSource,
  FrameData,
  InputAssemblyFile
};
static constexpr Kind FirstKind = Kind::Symbols;
static constexpr Kind LastKind = Kind::InputAssemblyFile;

HRESULT Create(
    /* [in] */ Session *pSession,
    /* [in] */ Kind kind,
    /* [out] */ IDiaTable **ppTable);
}  // namespace Table

namespace impl {

template<typename T, typename TItem>
class TableBase : public IDiaTable, public T {
public:
  // COM Interfaces do not have virtual destructors; they instead rely on
  // AddRef/Release matching calls for managing object lifetimes. This
  // template is inherited by the implementing table types (which is fine),
  // and it also provides the base implementation of the COM's memory
  // management callbacks (which is not fine: once a table goes out of scope
  // a method in this class will invoke the object's destructor -- which, being
  // non-virtual, will be this class' instead of the derived table's.) Therefore,
  // we introduce a virtual destructor.
  virtual ~TableBase() {
    DXASSERT(m_dwRef == 0, "deleting COM table with active references");
  }

protected:
  static constexpr LPCWSTR TableNames[] = {
    L"Symbols",
    L"SourceFiles",
    L"LineNumbers",
    L"Sections",
    L"SegmentMap",
    L"InjectedSource",
    L"FrameData",
    L"InputAssemblyFiles"
  };

  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<Session> m_pSession;
  unsigned m_next;
  unsigned m_count;
  Table::Kind m_kind;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaTable, T, IEnumUnknown>(this, iid, ppvObject);
  }

  TableBase(IMalloc *pMalloc, Session *pSession, Table::Kind kind) {
    m_pMalloc = pMalloc;
    m_pSession = pSession;
    m_kind = kind;
    m_next = 0;
    m_count = 0;
  }

  // IEnumUnknown implementation.
  STDMETHODIMP Next(
    _In_  ULONG celt,
    _Out_writes_to_(celt, *pceltFetched)  IUnknown **rgelt,
    _Out_opt_  ULONG *pceltFetched) override {
    DxcThreadMalloc TM(m_pMalloc);
    ULONG fetched = 0;
    while (fetched < celt && m_next < m_count) {
      HRESULT hr = Item(m_next, &rgelt[fetched]);
      if (FAILED(hr)) {
        return hr; // TODO: this leaks prior tables.
      }
      ++m_next, ++fetched;
    }
    if (pceltFetched != nullptr)
      *pceltFetched = fetched;
    return (fetched == celt) ? S_OK : S_FALSE;
  }

  STDMETHODIMP Skip(ULONG celt) override {
    if (celt + m_next <= m_count) {
      m_next += celt;
      return S_OK;
    }
    return S_FALSE;
  }

  STDMETHODIMP Reset(void) override {
    m_next = 0;
    return S_OK;
  }

  STDMETHODIMP Clone(IEnumUnknown **ppenum) override {
    return ENotImpl();
  }

  // IDiaTable implementation.
  STDMETHODIMP get__NewEnum(IUnknown **pRetVal) override {
    return ENotImpl();
  }

  STDMETHODIMP get_name(BSTR *pRetVal) override {
    *pRetVal = SysAllocString(TableNames[(unsigned)m_kind]);
    return (*pRetVal) ? S_OK : E_OUTOFMEMORY;
  }

  STDMETHODIMP get_Count(_Out_ LONG *pRetVal) override {
    *pRetVal = m_count;
    return S_OK;
  }

  STDMETHODIMP Item(DWORD index, _COM_Outptr_ IUnknown **table) override {
    if (index >= m_count)
      return E_INVALIDARG;
    return GetItem(index, (TItem **)table);
  }

  // T implementation (partial).
  STDMETHODIMP Clone(_COM_Outptr_ T **ppenum) override {
    *ppenum = nullptr;
    return ENotImpl();
  }
  STDMETHODIMP Next(
    /* [in] */ ULONG celt,
    /* [out] */ TItem **rgelt,
    /* [out] */ ULONG *pceltFetched) override {
    DxcThreadMalloc TM(m_pMalloc);
    ULONG fetched = 0;
    while (fetched < celt && m_next < m_count) {
      HRESULT hr = GetItem(m_next, &rgelt[fetched]);
      if (FAILED(hr)) {
        return hr; // TODO: this leaks prior items.
      }
      ++m_next, ++fetched;
    }
    if (pceltFetched != nullptr)
      *pceltFetched = fetched;
    return (fetched == celt) ? S_OK : S_FALSE;
  }
  STDMETHODIMP Item(
    /* [in] */ DWORD index,
    /* [retval][out] */ TItem **ppItem) override {
    DxcThreadMalloc TM(m_pMalloc);
    if (index >= m_count)
      return E_INVALIDARG;
    return GetItem(index, ppItem);
  }

  virtual HRESULT GetItem(DWORD index, TItem **ppItem) = 0;
};
}  // namespace impl
}  // namespace dxil_dia
