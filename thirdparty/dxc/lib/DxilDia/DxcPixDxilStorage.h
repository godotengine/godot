///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixDxilStorage.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Declares the DxcPixDxilStorage API. This is the API that allows PIX to    //
// find which register holds the value for a particular member of a          //
// variable.                                                                 //
//                                                                           //
// For DxcPixDxilStorage, a member is a DIBasicType (i.e., float, int, etc.) //
// This library provides Storage classes for inspecting structs, arrays, and //
// basic types. This is what's minimally necessary for enable HLSL debugging //
// in PIX.                                                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"
#include "dxc/dxcapi.h"
#include "dxc/dxcpix.h"

#include "dxc/Support/microcom.h"

#include "DxcPixDxilDebugInfo.h"

namespace llvm
{
class DIType;
}  // namespace llvm

namespace dxil_debug_info
{
struct VariableInfo;

HRESULT CreateDxcPixStorage(
    DxcPixDxilDebugInfo *pDxilDebugInfo,
    llvm::DIType *diType,
    const VariableInfo *VarInfo,
    unsigned CurrentOffsetInBits,
    IDxcPixDxilStorage **ppStorage);

class DxcPixDxilArrayStorage : public IDxcPixDxilStorage
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  CComPtr<IDxcPixType> m_pOriginalType;
  CComPtr<IDxcPixArrayType> m_pType;
  VariableInfo const *m_pVarInfo;
  unsigned m_OffsetFromStorageStartInBits;

  DxcPixDxilArrayStorage(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      IDxcPixType* OriginalType,
      IDxcPixArrayType *pType,
      const VariableInfo* pVarInfo,
      unsigned OffsetFromStorageStartInBits)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pOriginalType(OriginalType)
    , m_pType(pType)
    , m_pVarInfo(pVarInfo)
    , m_OffsetFromStorageStartInBits(OffsetFromStorageStartInBits)
  {
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilArrayStorage)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilStorage>(this, iid, ppvObject);
  }

  STDMETHODIMP AccessField(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP Index(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP GetRegisterNumber(
      _Outptr_result_z_ DWORD *pRegisterNumber) override;

  STDMETHODIMP GetIsAlive() override;

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType **ppType) override;
};

class DxcPixDxilStructStorage : public IDxcPixDxilStorage
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  CComPtr<IDxcPixType> m_pOriginalType;
  CComPtr<IDxcPixStructType> m_pType;
  VariableInfo const *m_pVarInfo;
  unsigned m_OffsetFromStorageStartInBits;

  DxcPixDxilStructStorage(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      IDxcPixType* OriginalType,
      IDxcPixStructType *pType,
      VariableInfo const *pVarInfo,
      unsigned OffsetFromStorageStartInBits)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pOriginalType(OriginalType)
    , m_pType(pType)
    , m_pVarInfo(pVarInfo)
    , m_OffsetFromStorageStartInBits(OffsetFromStorageStartInBits)
  {
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilStructStorage)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilStorage>(this, iid, ppvObject);
  }

  STDMETHODIMP AccessField(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP Index(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP GetRegisterNumber(
      _Outptr_result_z_ DWORD *pRegisterNumber) override;

  STDMETHODIMP GetIsAlive() override;

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType **ppType) override;
};

class DxcPixDxilScalarStorage : public IDxcPixDxilStorage
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  CComPtr<IDxcPixType> m_pOriginalType;
  CComPtr<IDxcPixScalarType> m_pType;
  VariableInfo const *m_pVarInfo;
  unsigned m_OffsetFromStorageStartInBits;

  DxcPixDxilScalarStorage(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      IDxcPixType *OriginalType,
      IDxcPixScalarType *pType,
      VariableInfo const *pVarInfo,
      unsigned OffsetFromStorageStartInBits)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pOriginalType(OriginalType)
    , m_pType(pType)
    , m_pVarInfo(pVarInfo)
    , m_OffsetFromStorageStartInBits(OffsetFromStorageStartInBits)
  {
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilScalarStorage)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilStorage>(this, iid, ppvObject);
  }

  STDMETHODIMP AccessField(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP Index(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixDxilStorage **ppResult) override;

  STDMETHODIMP GetRegisterNumber(
      _Outptr_result_z_ DWORD *pRegisterNumber) override;

  STDMETHODIMP GetIsAlive() override;

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType **ppType) override;
};

}  // namespace dxil_debug_info
