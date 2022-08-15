///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSymbols.cpp                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaTableSymbols.h"

#include <comdef.h>

#include "dxc/Support/Unicode.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_os_ostream.h"

#include "DxilDiaSession.h"

dxil_dia::Symbol::~Symbol() = default;

void dxil_dia::Symbol::Init(Session *pSession, DWORD ID, DWORD symTag) {
  DXASSERT_ARGS(m_pSession == nullptr, "Double init on symbol %d", ID);
  m_pSession = pSession;
  m_ID = ID;
  m_symTag = symTag;
}

STDMETHODIMP dxil_dia::Symbol::get_symIndexId(
  /* [retval][out] */ DWORD *pRetVal) {
  *pRetVal = m_ID;
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_symTag(
  /* [retval][out] */ DWORD *pRetVal) {
  *pRetVal = m_symTag;
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_name(
  /* [retval][out] */ BSTR *pRetVal) {
  DxcThreadMalloc TM(m_pSession->GetMallocNoRef());
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = nullptr;
  if (!m_hasName) {
    return S_FALSE;
  }
  return m_name.CopyTo(pRetVal);
}

STDMETHODIMP dxil_dia::Symbol::get_lexicalParent(
  /* [retval][out] */ IDiaSymbol **pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = nullptr;

  DWORD dwParentID = this->m_lexicalParent;
  if (dwParentID == 0) {
    return S_FALSE;
  }

  Symbol *pParent;
  IFR(m_pSession->SymMgr().GetSymbolByID(dwParentID, &pParent));

  *pRetVal = pParent;
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_type(
  /* [retval][out] */ IDiaSymbol **pRetVal) {
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_dataKind(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;
  if (!m_hasDataKind) {
    return S_FALSE;
  }
  *pRetVal = m_dataKind;
  return m_dataKind ? S_OK : S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_locationType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = LocIsNull;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_sourceFileName(
  /* [retval][out] */ BSTR *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = nullptr;
  if (!m_hasSourceFileName) {
    return S_FALSE;
  }
  *pRetVal = m_sourceFileName.Copy();
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_value(
  /* [retval][out] */ VARIANT *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  ZeroMemory(pRetVal, sizeof(*pRetVal));
  if (!m_hasValue) {
    return S_FALSE;
  }
  return VariantCopy(pRetVal, &m_value);
}

STDMETHODIMP dxil_dia::Symbol::get_baseType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal != nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = btNoType;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_count(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_offset(
  /* [retval][out] */ LONG *pRetVal) {
  if (pRetVal != nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_length(
  /* [retval][out] */ ULONGLONG *pRetVal) {
  if (pRetVal != nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_lexicalParentId(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;
  if (!m_hasLexicalParent) {
    return S_FALSE;
  }
  *pRetVal = m_lexicalParent;
  return S_OK;
}

void dxil_dia::SymbolChildrenEnumerator::Init(std::vector<CComPtr<Symbol>> &&syms) {
  std::swap(syms, m_symbols);
  m_pos = m_symbols.begin();
}

HRESULT STDMETHODCALLTYPE dxil_dia::SymbolChildrenEnumerator::get_Count(
  /* [retval][out] */ LONG *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = m_symbols.size();
  return S_OK;
}

HRESULT STDMETHODCALLTYPE dxil_dia::SymbolChildrenEnumerator::Item(
  /* [in] */ DWORD index,
  /* [retval][out] */ IDiaSymbol **symbol) {
  if (symbol == nullptr) {
    return E_INVALIDARG;
  }
  *symbol = nullptr;

  if (index < 0 || index > m_symbols.size()) {
    return E_INVALIDARG;
  }

  *symbol = m_symbols[index];
  (*symbol)->AddRef();
  return S_OK;
}

HRESULT STDMETHODCALLTYPE dxil_dia::SymbolChildrenEnumerator::Reset(void) {
  m_pos = m_symbols.begin();
  return S_OK;
}

HRESULT STDMETHODCALLTYPE dxil_dia::SymbolChildrenEnumerator::Next(
  /* [in] */ ULONG celt,
  /* [out] */ IDiaSymbol **rgelt,
  /* [out] */ ULONG *pceltFetched) {
  DxcThreadMalloc TM(m_pMalloc);
  if (rgelt == nullptr || pceltFetched == nullptr) {
    return E_INVALIDARG;
  }

  *pceltFetched = 0;
  ZeroMemory(rgelt, sizeof(*rgelt) * celt);

  for (; *pceltFetched < celt && m_pos != m_symbols.end(); ++m_pos, ++rgelt, ++(*pceltFetched)) {
    *rgelt = *m_pos;
    (*rgelt)->AddRef();
  }

  return (*pceltFetched == celt) ? S_OK : S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::findChildren(
  /* [in] */ enum SymTagEnum symtag,
  /* [in] */ LPCOLESTR name,
  /* [in] */ DWORD compareFlags,
  /* [out] */ IDiaEnumSymbols **ppResult) {
  return findChildrenEx(symtag, name, compareFlags, ppResult);;
}

STDMETHODIMP dxil_dia::Symbol::findChildrenEx(
  /* [in] */ enum SymTagEnum symtag,
  /* [in] */ LPCOLESTR name,
  /* [in] */ DWORD compareFlags,
  /* [out] */ IDiaEnumSymbols **ppResult)  {
  DxcThreadMalloc TM(m_pMalloc);
  if (ppResult == nullptr) {
    return E_INVALIDARG;
  }
  *ppResult = nullptr;

  CComPtr<SymbolChildrenEnumerator> ret = SymbolChildrenEnumerator::Alloc(m_pMalloc);
  if (!ret) {
    return E_OUTOFMEMORY;
  }

  std::vector<CComPtr<Symbol>> children;
  IFR(GetChildren(&children));

  if (symtag != nsNone) {
    std::vector<CComPtr<Symbol>> tmp;
    tmp.reserve(children.size());
    for (const auto & c : children) {
      if (c->m_symTag == symtag) {
        tmp.emplace_back(c);
      }
    }
    std::swap(tmp, children);
  }

  if (name != nullptr && compareFlags != nsNone) {
    std::vector<CComPtr<Symbol>> tmp;
    tmp.reserve(children.size());
    for (const auto & c : children) {
      CComBSTR cName;
      IFR(c->get_name(&cName));
      // Careful with the string comparison function we use as it can make us pull in new dependencies
      // CompareStringOrdinal lives in kernel32.dll
      if (CompareStringOrdinal(cName, cName.Length(), name, -1, (BOOL)(compareFlags == nsfCaseInsensitive)) != CSTR_EQUAL) {
        continue;
      }

      if (c->m_symTag == symtag) {
        tmp.emplace_back(c);
      }
    }
    std::swap(tmp, children);
  }

  ret->Init(std::move(children));

  *ppResult = ret.Detach();
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_isAggregated(
  /* [retval][out] */ BOOL *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = false;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_registerType(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_sizeInUdt(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_liveRangeStartAddressOffset(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;

  SymbolManager::LiveRange LR;
  IFR(m_pSession->SymMgr().GetLiveRangeOf(this, &LR));
  *pRetVal = LR.Start;
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_liveRangeLength(
  /* [retval][out] */ ULONGLONG *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = 0;

  SymbolManager::LiveRange LR;
  IFR(m_pSession->SymMgr().GetLiveRangeOf(this, &LR));
  *pRetVal = LR.Length;
  return S_OK;
}

STDMETHODIMP dxil_dia::Symbol::get_offsetInUdt(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_numericProperties(
  /* [in] */ DWORD cnt,
  /* [out] */ DWORD *pcnt,
  /* [size_is][out] */ DWORD *pProperties) {
  if (pcnt == nullptr || pProperties == nullptr) {
    return E_INVALIDARG;
  }

  ZeroMemory(pProperties, sizeof(*pProperties) * cnt);
  *pcnt = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_numberOfRegisterIndices(
  /* [retval][out] */ DWORD *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }

  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Symbol::get_isHLSLData(
  /* [retval][out] */ BOOL *pRetVal) {
  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = false;
  if (!m_hasIsHLSLData) {
    return S_FALSE;
  }
  *pRetVal = m_isHLSLData;
  return S_OK;
}

dxil_dia::SymbolsTable::SymbolsTable(IMalloc *pMalloc, Session *pSession)
  : impl::TableBase<IDiaEnumSymbols, IDiaSymbol>(pMalloc, pSession, Table::Kind::Symbols) {
  m_count = pSession->SymMgr().NumSymbols();
}

HRESULT dxil_dia::SymbolsTable::GetItem(DWORD index, IDiaSymbol **ppItem) {
  if (ppItem == nullptr) {
    return E_INVALIDARG;
  }
  *ppItem = nullptr;

  Symbol *ret = nullptr;
  const DWORD dwSymID = index + 1;
  IFR(m_pSession->SymMgr().GetSymbolByID(dwSymID, &ret));

  *ppItem = ret;
  return S_OK;
}
