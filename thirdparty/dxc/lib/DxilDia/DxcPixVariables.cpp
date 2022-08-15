///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixVariables.cpp                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines DXC's PIX api for exposing llvm::DIVariables.                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


#include "dxc/Support/WinIncludes.h"

#include "dxc/dxcpix.h"
#include "dxc/Support/microcom.h"
#include "dxc/Support/Global.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "DxcPixBase.h"
#include "DxcPixDxilDebugInfo.h"
#include "DxcPixDxilStorage.h"
#include "DxcPixLiveVariables.h"
#include "DxcPixTypes.h"
#include "DxilDiaSession.h"

#include <set>
#include <vector>

namespace dxil_debug_info
{
template <typename T>
class DxcPixVariable : public IDxcPixVariable
{
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  T *m_pVariable;
  VariableInfo const *m_pVarInfo;
  llvm::DIType *m_pType;

  DxcPixVariable(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *DxilDebugInfo,
      T *pVariable,
      VariableInfo const *pVarInfo)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(DxilDebugInfo)
    , m_pVariable(pVariable)
    , m_pVarInfo(pVarInfo)
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    m_pType = m_pVariable->getType().resolve(EmptyMap);
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixVariable)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixVariable>(this, iid, ppvObject);
  }

public:
  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType** ppType) override;

  STDMETHODIMP GetStorage(
      _COM_Outptr_ IDxcPixDxilStorage** ppStorage) override;
};

}  // namespace dxil_debug_info

template <typename T>
STDMETHODIMP dxil_debug_info::DxcPixVariable<T>::GetName(
    _Outptr_result_z_ BSTR* Name)
{
  *Name = CComBSTR(CA2W(m_pVariable->getName().data())).Detach();
  return S_OK;
}

template <typename T>
STDMETHODIMP dxil_debug_info::DxcPixVariable<T>::GetType(
    _Outptr_result_z_ IDxcPixType **ppType
)
{
  return dxil_debug_info::CreateDxcPixType(
      m_pDxilDebugInfo,
      m_pType,
      ppType);
}

template <typename T>
STDMETHODIMP dxil_debug_info::DxcPixVariable<T>::GetStorage(
    _COM_Outptr_ IDxcPixDxilStorage **ppStorage
)
{
  const unsigned InitialOffsetInBits = 0;
  return CreateDxcPixStorage(
      m_pDxilDebugInfo,
      m_pType,
      m_pVarInfo,
      InitialOffsetInBits,
      ppStorage);
}

namespace dxil_debug_info
{
class DxcPixDxilLiveVariables : public IDxcPixDxilLiveVariables
{
private:
  DXC_MICROCOM_TM_REF_FIELDS();
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  std::vector<const VariableInfo *> m_LiveVars;

  DxcPixDxilLiveVariables(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      std::vector<const VariableInfo *> LiveVars
  ) : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_LiveVars(std::move(LiveVars))
  {
#ifndef NDEBUG
    for (auto VarAndInfo : m_LiveVars)
    {
        assert(llvm::isa<llvm::DIGlobalVariable>(VarAndInfo->m_Variable) ||
               llvm::isa<llvm::DILocalVariable>(VarAndInfo->m_Variable));
    }
#endif  // !NDEBUG
  }

  STDMETHODIMP CreateDxcPixVariable(
      IDxcPixVariable** ppVariable,
      VariableInfo const *VarInfo) const;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL();
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilLiveVariables);

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) final {
    return DoBasicQueryInterface<IDxcPixDxilLiveVariables>(this, iid, ppvObject);
  }

  STDMETHODIMP GetCount(
      _Outptr_ DWORD *dwSize) override;

  STDMETHODIMP GetVariableByIndex(
      _In_ DWORD Index,
      _Outptr_result_z_ IDxcPixVariable** ppVariable) override;

  STDMETHODIMP GetVariableByName(
      _In_ LPCWSTR Name,
      _Outptr_result_z_ IDxcPixVariable** ppVariable) override;
};

}  // namespace dxil_debug_info
STDMETHODIMP dxil_debug_info::DxcPixDxilLiveVariables::GetCount(
    _Outptr_ DWORD *dwSize) {
  *dwSize = m_LiveVars.size();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixDxilLiveVariables::CreateDxcPixVariable(
    IDxcPixVariable** ppVariable,
    VariableInfo const* VarInfo) const
{
  auto *Var = VarInfo->m_Variable;
  if (auto *DILV = llvm::dyn_cast<llvm::DILocalVariable>(Var))
  {
    return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixVariable<llvm::DILocalVariable>>(
        ppVariable,
        m_pMalloc,
        m_pDxilDebugInfo,
        DILV,
        VarInfo);
  }
  else if (auto *DIGV = llvm::dyn_cast<llvm::DIGlobalVariable>(Var))
  {
    return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixVariable<llvm::DIGlobalVariable>>(
        ppVariable,
        m_pMalloc,
        m_pDxilDebugInfo,
        DIGV,
        VarInfo);
  }

  return E_UNEXPECTED;
}

STDMETHODIMP dxil_debug_info::DxcPixDxilLiveVariables::GetVariableByIndex(
    _In_ DWORD Index,
    _Outptr_result_z_ IDxcPixVariable **ppVariable)
{
  if (Index >= m_LiveVars.size())
  {
    return E_BOUNDS;
  }

  auto* VarInfo = m_LiveVars[Index];
  return CreateDxcPixVariable(ppVariable, VarInfo);
}

STDMETHODIMP dxil_debug_info::DxcPixDxilLiveVariables::GetVariableByName(
    _In_ LPCWSTR Name,
    _Outptr_result_z_ IDxcPixVariable **ppVariable)
{
  std::string name = CW2A(Name);

  for (auto *VarInfo : m_LiveVars)
  {
    auto *Var = VarInfo->m_Variable;
    if (Var->getName() == name)
    {
      return CreateDxcPixVariable(ppVariable, VarInfo);
    }
  }

  return E_BOUNDS;
}

HRESULT dxil_debug_info::CreateDxilLiveVariables(
    DxcPixDxilDebugInfo *pDxilDebugInfo,
    std::vector<const VariableInfo *> &&LiveVariables,
    IDxcPixDxilLiveVariables **ppResult)
{
  return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixDxilLiveVariables>(
      ppResult,
      pDxilDebugInfo->GetMallocNoRef(),
      pDxilDebugInfo,
      std::move(LiveVariables));
}
