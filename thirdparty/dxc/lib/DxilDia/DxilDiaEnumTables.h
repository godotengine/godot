///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaEnumTable.h                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <array>

#include "dia2.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {

class Session;

class EnumTables : public IDiaEnumTables {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
protected:
  CComPtr<Session> m_pSession;
  unsigned m_next;
public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaEnumTables>(this, iid, ppvObject);
  }

  EnumTables(IMalloc *pMalloc, Session *pSession)
      : m_pMalloc(pMalloc), m_pSession(pSession), m_dwRef(0), m_next(0) {
    m_tables.fill(nullptr);
  }

  STDMETHODIMP get__NewEnum(
    /* [retval][out] */ IUnknown **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_Count(_Out_ LONG *pRetVal) override;

  STDMETHODIMP Item(
    /* [in] */ VARIANT index,
    /* [retval][out] */ IDiaTable **table) override;

  STDMETHODIMP Next(
    ULONG celt,
    IDiaTable **rgelt,
    ULONG *pceltFetched) override;

  STDMETHODIMP Skip(
    /* [in] */ ULONG celt) override { return ENotImpl(); }

  STDMETHODIMP Reset(void) override;

  STDMETHODIMP Clone(
    /* [out] */ IDiaEnumTables **ppenum) override { return ENotImpl(); }

  static HRESULT Create(Session *pSession,
                        IDiaEnumTables **ppEnumTables);
private:
  std::array<CComPtr<IDiaTable>, (int)Table::LastKind+1> m_tables;
};

}  // namespace dxil_dia
