///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixTypes.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Declares the classes implementing DxcPixType and its subinterfaces. These //
// classes are used to interpret llvm::DITypes from the debug metadata.      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include "DxcPixTypes.h"
#include "DxcPixDxilDebugInfo.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

namespace dxil_debug_info
{
HRESULT CreateDxcPixType(
    DxcPixDxilDebugInfo *ppDxilDebugInfo,
    llvm::DIType *diType,
    IDxcPixType **ppResult);

class DxcPixConstType : public IDxcPixConstType
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DIDerivedType *m_pType;
  llvm::DIType *m_pBaseType;

  DxcPixConstType(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DIDerivedType *pType)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pType(pType)
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    m_pBaseType = m_pType->getBaseType().resolve(EmptyMap);
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixConstType)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixConstType, IDxcPixType>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD * pSizeInBits) override;

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType **ppType) override;
};

class DxcPixTypedefType : public IDxcPixTypedefType
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DIDerivedType *m_pType;
  llvm::DIType *m_pBaseType;

  DxcPixTypedefType(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DIDerivedType *pType)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pType(pType)
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    m_pBaseType = m_pType->getBaseType().resolve(EmptyMap);
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixTypedefType)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixTypedefType, IDxcPixType>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *pSizeInBits) override;

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType **ppBaseType) override;
};

class DxcPixScalarType : public IDxcPixScalarType
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DIBasicType *m_pType;

  DxcPixScalarType(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DIBasicType *pType)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pType(pType)
  {
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixScalarType)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixScalarType, IDxcPixType>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *pSizeInBits) override;

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType **ppBaseType) override;
};

class DxcPixArrayType : public IDxcPixArrayType
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DICompositeType *m_pArray;
  llvm::DIType *m_pBaseType;
  unsigned m_DimNum;

  DxcPixArrayType(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DICompositeType *pArray,
      unsigned DimNum)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pArray(pArray)
    , m_DimNum(DimNum)
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    m_pBaseType = m_pArray->getBaseType().resolve(EmptyMap);

#ifndef NDEBUG
    assert(m_DimNum < m_pArray->getElements().size());

    for (auto *Dims : m_pArray->getElements())
    {
      assert(llvm::isa<llvm::DISubrange>(Dims));
    }
#endif  // !NDEBUG
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixArrayType)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixArrayType, IDxcPixType>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *pSizeInBits) override;

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType **ppBaseType) override;

  STDMETHODIMP GetNumElements(
      _Outptr_result_z_ DWORD *ppNumElements) override;

  STDMETHODIMP GetIndexedType(
      _Outptr_result_z_ IDxcPixType **ppElementType) override;

  STDMETHODIMP GetElementType(
      _Outptr_result_z_ IDxcPixType **ppElementType) override;
};

class DxcPixStructType : public IDxcPixStructType
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DICompositeType *m_pStruct;

  DxcPixStructType(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DICompositeType *pStruct
  ) : m_pMalloc(pMalloc)
      , m_pDxilDebugInfo(pDxilDebugInfo)
      , m_pStruct(pStruct)
  {
#ifndef NDEBUG
    for (auto *Node : m_pStruct->getElements())
    {
      assert(llvm::isa<llvm::DIDerivedType>(Node) || llvm::isa<llvm::DISubprogram>(Node));
    }
#endif  // !NDEBUG
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixStructType)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixStructType, IDxcPixType>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetSizeInBits(
      _Outptr_result_z_ DWORD *pSizeInBits) override;

  STDMETHODIMP UnAlias(
      _Outptr_result_z_ IDxcPixType **ppBaseType) override;

  STDMETHODIMP GetNumFields(
      _Outptr_result_z_ DWORD* ppNumFields) override;

  STDMETHODIMP GetFieldByIndex(
      DWORD dwIndex,
      _Outptr_result_z_ IDxcPixStructField **ppField) override;

  STDMETHODIMP GetFieldByName(
      _In_ LPCWSTR lpName,
      _Outptr_result_z_ IDxcPixStructField **ppField) override;
};

class DxcPixStructField : public IDxcPixStructField
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<DxcPixDxilDebugInfo> m_pDxilDebugInfo;
  llvm::DIDerivedType *m_pField;
  llvm::DIType *m_pType;

  DxcPixStructField(
      IMalloc *pMalloc,
      DxcPixDxilDebugInfo *pDxilDebugInfo,
      llvm::DIDerivedType *pField)
    : m_pMalloc(pMalloc)
    , m_pDxilDebugInfo(pDxilDebugInfo)
    , m_pField(pField)
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    m_pType = m_pField->getBaseType().resolve(EmptyMap);
  }

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixStructField)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixStructField>(this, iid, ppvObject);
  }

  STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) override;

  STDMETHODIMP GetType(
      _Outptr_result_z_ IDxcPixType **ppType) override;

  STDMETHODIMP GetOffsetInBits(
      _Outptr_result_z_ DWORD *pOffsetInBits) override;
};
}  // namespace dxil_debug_info
