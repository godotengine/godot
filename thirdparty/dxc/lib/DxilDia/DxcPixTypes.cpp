///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixTypes.cpp                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines the implementation for the DxcPixType -- and its subinterfaces.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


#include "DxcPixBase.h"
#include "DxcPixTypes.h"
#include "DxilDiaSession.h"

HRESULT dxil_debug_info::CreateDxcPixType(
    DxcPixDxilDebugInfo *pDxilDebugInfo,
    llvm::DIType *diType,
    IDxcPixType **ppResult)
{
  if (auto *BT = llvm::dyn_cast<llvm::DIBasicType>(diType))
  {
    return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixScalarType>(
        ppResult,
        pDxilDebugInfo->GetMallocNoRef(),
        pDxilDebugInfo,
        BT);
  }
  else if (auto *CT = llvm::dyn_cast<llvm::DICompositeType>(diType))
  {
    switch (CT->getTag())
    {
    default:
      break;

    case llvm::dwarf::DW_TAG_array_type:
    {
      const unsigned FirstDim = 0;
      return NewDxcPixDxilDebugInfoObjectOrThrow< DxcPixArrayType>(
          ppResult,
          pDxilDebugInfo->GetMallocNoRef(),
          pDxilDebugInfo,
          CT,
          FirstDim);
    }

    case llvm::dwarf::DW_TAG_class_type:
    case llvm::dwarf::DW_TAG_structure_type:
      return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixStructType>(
          ppResult,
          pDxilDebugInfo->GetMallocNoRef(),
          pDxilDebugInfo,
          CT);
    }
  }
  else if (auto* DT = llvm::dyn_cast<llvm::DIDerivedType>(diType))
  {
    switch (DT->getTag())
    {
    default:
      break;

    case llvm::dwarf::DW_TAG_const_type:
      return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixConstType>(
          ppResult,
          pDxilDebugInfo->GetMallocNoRef(),
          pDxilDebugInfo,
          DT);

    case llvm::dwarf::DW_TAG_typedef:
      return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixTypedefType>(
          ppResult,
          pDxilDebugInfo->GetMallocNoRef(),
          pDxilDebugInfo,
          DT);
    }
  }

  return E_UNEXPECTED;
}

STDMETHODIMP dxil_debug_info::DxcPixConstType::GetName(
    _Outptr_result_z_ BSTR *Name)
{
  CComPtr<IDxcPixType> BaseType;
  IFR(UnAlias(&BaseType));

  CComBSTR BaseName;
  IFR(BaseType->GetName(&BaseName));

  *Name = CComBSTR((L"const " + std::wstring(BaseName)).c_str()).Detach();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixConstType::GetSizeInBits(
    _Outptr_result_z_ DWORD *pSize)
{
  CComPtr<IDxcPixType> BaseType;
  IFR(UnAlias(&BaseType));

  return BaseType->GetSizeInBits(pSize);
}

STDMETHODIMP dxil_debug_info::DxcPixConstType::UnAlias(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  return CreateDxcPixType(m_pDxilDebugInfo, m_pBaseType, ppType);
}

STDMETHODIMP dxil_debug_info::DxcPixTypedefType::GetName(
    _Outptr_result_z_ BSTR *Name)
{
  *Name = CComBSTR(CA2W(m_pType->getName().data())).Detach();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixTypedefType::GetSizeInBits(
    _Outptr_result_z_ DWORD *pSize)
{
  CComPtr<IDxcPixType> BaseType;
  IFR(UnAlias(&BaseType));

  return BaseType->GetSizeInBits(pSize);
}

STDMETHODIMP dxil_debug_info::DxcPixTypedefType::UnAlias(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  return CreateDxcPixType(m_pDxilDebugInfo, m_pBaseType, ppType);
}

STDMETHODIMP dxil_debug_info::DxcPixScalarType::GetName(
    _Outptr_result_z_ BSTR *Name)
{
  *Name = CComBSTR(CA2W(m_pType->getName().data())).Detach();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixScalarType::GetSizeInBits(
    _Outptr_result_z_ DWORD *pSizeInBits)
{
  *pSizeInBits = m_pType->getSizeInBits();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixScalarType::UnAlias(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  *ppType = this;
  this->AddRef();
  return S_FALSE;
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::GetName(
    _Outptr_result_z_ BSTR *Name)
{
  return E_FAIL;
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::GetSizeInBits(
    _Outptr_result_z_ DWORD *pSizeInBits)
{
  *pSizeInBits = m_pArray->getSizeInBits();
  for (unsigned ContainerDims = 0; ContainerDims < m_DimNum; ++ContainerDims)
  {
    auto *SR = llvm::dyn_cast<llvm::DISubrange>(m_pArray->getElements()[ContainerDims]);
    auto count = SR->getCount();
    if (count == 0)
    {
      return E_FAIL;
    }
    *pSizeInBits /= count;
  }
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::UnAlias(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  *ppType = this;
  this->AddRef();
  return S_FALSE;
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::GetNumElements(
    _Outptr_result_z_ DWORD *ppNumElements) 
{
  auto* SR = llvm::dyn_cast<llvm::DISubrange>(m_pArray->getElements()[m_DimNum]);
  if (SR == nullptr)
  {
    return E_FAIL;
  }

  *ppNumElements = SR->getCount();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::GetIndexedType(
    _Outptr_result_z_ IDxcPixType **ppIndexedElement)
{
  assert(1 + m_DimNum <= m_pArray->getElements().size());
  if (1 + m_DimNum == m_pArray->getElements().size())
  {
    return CreateDxcPixType(m_pDxilDebugInfo, m_pBaseType, ppIndexedElement);
  }

  return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixArrayType>(
      ppIndexedElement,
      m_pMalloc,
      m_pDxilDebugInfo,
      m_pArray,
      1 + m_DimNum);
}

STDMETHODIMP dxil_debug_info::DxcPixArrayType::GetElementType(
    _Outptr_result_z_ IDxcPixType **ppElementType)
{
  return CreateDxcPixType(m_pDxilDebugInfo, m_pBaseType, ppElementType);
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::GetName(
    _Outptr_result_z_ BSTR *Name)
{
  *Name = CComBSTR(CA2W(m_pStruct->getName().data())).Detach();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::GetSizeInBits(
    _Outptr_result_z_ DWORD *pSizeInBits)
{
  *pSizeInBits = m_pStruct->getSizeInBits();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::UnAlias(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  *ppType = this;
  this->AddRef();
  return S_FALSE;
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::GetNumFields(
    _Outptr_result_z_ DWORD *ppNumFields)
{
  *ppNumFields = m_pStruct->getElements()->getNumOperands();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::GetFieldByIndex(
    DWORD dwIndex,
    _Outptr_result_z_ IDxcPixStructField **ppField)
{
  if (dwIndex >= m_pStruct->getElements().size())
  {
    return E_BOUNDS;
  }

  auto* pDIField = llvm::dyn_cast<llvm::DIDerivedType>(
      m_pStruct->getElements()[dwIndex]);
  if (pDIField == nullptr)
  {
    return E_FAIL;
  }

  return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixStructField>(
      ppField,
      m_pMalloc,
      m_pDxilDebugInfo,
      pDIField);
}

STDMETHODIMP dxil_debug_info::DxcPixStructType::GetFieldByName(
    _In_ LPCWSTR lpName,
    _Outptr_result_z_ IDxcPixStructField **ppField)
{
  std::string name = CW2A(lpName);
  for (auto *Node : m_pStruct->getElements())
  {
    auto* pDIField = llvm::dyn_cast<llvm::DIDerivedType>(Node);
    if (pDIField == nullptr)
    {
      return E_FAIL;
    }

    if (pDIField->getName() == name)
    {
      return NewDxcPixDxilDebugInfoObjectOrThrow<DxcPixStructField>(
          ppField,
          m_pMalloc,
          m_pDxilDebugInfo,
          pDIField);
    }
  }

  return E_BOUNDS;
}

STDMETHODIMP dxil_debug_info::DxcPixStructField::GetName(
    _Outptr_result_z_ BSTR *Name) 
{
  *Name = CComBSTR(CA2W(m_pField->getName().data())).Detach();
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixStructField::GetType(
    _Outptr_result_z_ IDxcPixType **ppType)
{
  return CreateDxcPixType(m_pDxilDebugInfo, m_pType, ppType);
}

STDMETHODIMP dxil_debug_info::DxcPixStructField::GetOffsetInBits(
    _Outptr_result_z_ DWORD *pOffsetInBits)
{
  *pOffsetInBits = m_pField->getOffsetInBits();
  return S_OK;
}
