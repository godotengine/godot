///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcpix.h                                                                 //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides declarations for the DirectX Compiler API with pix debugging.    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_PIX__
#define __DXC_PIX__

#include "objidl.h"
#include "dxc/dxcapi.h"

struct __declspec(uuid("199d8c13-d312-4197-a2c1-07a532999727"))
IDxcPixType : public IUnknown
{
  virtual STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) = 0;

  virtual STDMETHODIMP GetSizeInBits(
      _Out_ DWORD *GetSizeInBits) = 0;

  virtual STDMETHODIMP UnAlias(
      _COM_Outptr_ IDxcPixType** ppBaseType) = 0;
};

struct __declspec(uuid("d9df2c8b-2773-466d-9bc2-d848d8496bf6"))
IDxcPixConstType: public IDxcPixType
{
};

struct __declspec(uuid("7bfca9c0-1ed0-429c-9dc2-c75597d821d2"))
IDxcPixTypedefType: public IDxcPixType
{
};

struct __declspec(uuid("246e1652-ed2a-4ffc-a949-43bf63750ee5"))
IDxcPixScalarType : public IDxcPixType
{
};

struct __declspec(uuid("9ba0d9d3-457b-426f-8019-9f3849982aa2"))
IDxcPixArrayType : public IDxcPixType
{
  virtual STDMETHODIMP GetNumElements(
      _Out_ DWORD* ppNumElements) = 0;

  virtual STDMETHODIMP GetIndexedType(
      _COM_Outptr_ IDxcPixType **ppElementType) = 0;

  virtual STDMETHODIMP GetElementType(
      _COM_Outptr_ IDxcPixType** ppElementType) = 0;
};

struct __declspec(uuid("6c707d08-7995-4a84-bae5-e6d8291f3b78"))
IDxcPixStructField : public IUnknown
{
  virtual STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) = 0;

  virtual STDMETHODIMP GetType(
      _COM_Outptr_ IDxcPixType** ppType) = 0;

  virtual STDMETHODIMP GetOffsetInBits(
      _Out_ DWORD *pOffsetInBits) = 0;
};

struct __declspec(uuid("24c08c44-684b-4b1c-b41b-f8772383d074"))
IDxcPixStructType : public IDxcPixType
{
  virtual STDMETHODIMP GetNumFields(
      _Out_ DWORD* ppNumFields) = 0;

  virtual STDMETHODIMP GetFieldByIndex(
      DWORD dwIndex,
      _COM_Outptr_ IDxcPixStructField **ppField) = 0;

  virtual STDMETHODIMP GetFieldByName(
      _In_ LPCWSTR lpName,
      _COM_Outptr_ IDxcPixStructField** ppField) = 0;
};

struct __declspec(uuid("74d522f5-16c4-40cb-867b-4b4149e3db0e"))
IDxcPixDxilStorage : public IUnknown
{
  virtual STDMETHODIMP AccessField(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixDxilStorage** ppResult) = 0;

  virtual STDMETHODIMP Index(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixDxilStorage** ppResult) = 0;

  virtual STDMETHODIMP GetRegisterNumber(
      _Out_ DWORD* pRegNum) = 0;

  virtual STDMETHODIMP GetIsAlive() = 0;

  virtual STDMETHODIMP GetType(
      _COM_Outptr_ IDxcPixType** ppType) = 0;
};

struct __declspec(uuid("2f954b30-61a7-4348-95b1-2db356a75cde"))
IDxcPixVariable : public IUnknown
{
  virtual STDMETHODIMP GetName(
      _Outptr_result_z_ BSTR *Name) = 0;

  virtual STDMETHODIMP GetType(
      _COM_Outptr_ IDxcPixType** ppType) = 0;

  virtual STDMETHODIMP GetStorage(
      _COM_Outptr_ IDxcPixDxilStorage **ppStorage) = 0;
};

struct __declspec(uuid("c59d302f-34a2-4fe5-9646-32ce7a52d03f"))
IDxcPixDxilLiveVariables : public IUnknown
{
  virtual STDMETHODIMP GetCount(
      _Out_ DWORD *dwSize) = 0;

  virtual STDMETHODIMP GetVariableByIndex(
      _In_ DWORD Index,
      _COM_Outptr_ IDxcPixVariable ** ppVariable) = 0;

  virtual STDMETHODIMP GetVariableByName(
      _In_ LPCWSTR Name,
      _COM_Outptr_ IDxcPixVariable** ppVariable) = 0;
};

struct __declspec(uuid("eb71f85e-8542-44b5-87da-9d76045a1910"))
  IDxcPixDxilInstructionOffsets : public IUnknown
{
  virtual STDMETHODIMP_(DWORD) GetCount() = 0;

  virtual STDMETHODIMP_(DWORD) GetOffsetByIndex(_In_ DWORD Index) = 0;
};

struct __declspec(uuid("761c833d-e7b8-4624-80f8-3a3fb4146342"))
  IDxcPixDxilSourceLocations : public IUnknown
{
  virtual STDMETHODIMP_(DWORD) GetCount() = 0;
  virtual STDMETHODIMP_(DWORD) GetLineNumberByIndex(_In_ DWORD Index) = 0;
  virtual STDMETHODIMP_(DWORD) GetColumnByIndex(_In_ DWORD Index) = 0;
  virtual STDMETHODIMP GetFileNameByIndex(_In_ DWORD Index, _Outptr_result_z_ BSTR *Name) = 0;
};

struct __declspec(uuid("b875638e-108a-4d90-a53a-68d63773cb38"))
IDxcPixDxilDebugInfo : public IUnknown
{
  virtual STDMETHODIMP GetLiveVariablesAt(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilLiveVariables **ppLiveVariables) = 0;

  virtual STDMETHODIMP IsVariableInRegister(
      _In_ DWORD InstructionOffset,
      _In_ const wchar_t *VariableName) = 0;

  virtual STDMETHODIMP GetFunctionName(
      _In_ DWORD InstructionOffset,
      _Outptr_result_z_ BSTR *ppFunctionName) = 0;

  virtual STDMETHODIMP GetStackDepth(
      _In_ DWORD InstructionOffset,
      _Out_ DWORD* StackDepth) = 0;

  virtual STDMETHODIMP InstructionOffsetsFromSourceLocation(
      _In_ const wchar_t *FileName,
      _In_ DWORD SourceLine,
      _In_ DWORD SourceColumn,
      _COM_Outptr_ IDxcPixDxilInstructionOffsets** ppOffsets) = 0;

  virtual STDMETHODIMP SourceLocationsFromInstructionOffset(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilSourceLocations**ppSourceLocations) = 0;
};

struct __declspec(uuid("61b16c95-8799-4ed8-bdb0-3b6c08a141b4"))
IDxcPixCompilationInfo : public IUnknown
{
  virtual STDMETHODIMP GetSourceFile(
      _In_ DWORD SourceFileOrdinal,
      _Outptr_result_z_ BSTR *pSourceName,
      _Outptr_result_z_ BSTR *pSourceContents) = 0;
  virtual STDMETHODIMP GetArguments(
      _Outptr_result_z_ BSTR *pArguments) = 0;
  virtual STDMETHODIMP GetMacroDefinitions(
      _Outptr_result_z_ BSTR *pMacroDefinitions) = 0;
  virtual STDMETHODIMP GetEntryPointFile(
    _Outptr_result_z_ BSTR *pEntryPointFile) = 0;
  virtual STDMETHODIMP GetHlslTarget(
    _Outptr_result_z_ BSTR *pHlslTarget) = 0;
  virtual STDMETHODIMP GetEntryPoint(
    _Outptr_result_z_ BSTR *pEntryPoint) = 0;
};

struct __declspec(uuid("9c2a040d-8068-44ec-8c68-8bfef1b43789"))
IDxcPixDxilDebugInfoFactory : public IUnknown
{
  virtual STDMETHODIMP NewDxcPixDxilDebugInfo(
      _COM_Outptr_ IDxcPixDxilDebugInfo **ppDxilDebugInfo) = 0;
  virtual STDMETHODIMP NewDxcPixCompilationInfo(
      _COM_Outptr_ IDxcPixCompilationInfo **ppCompilationInfo) = 0;
};

#ifndef CLSID_SCOPE
#ifdef _MSC_VER
#define CLSID_SCOPE __declspec(selectany) extern
#else
#define CLSID_SCOPE
#endif
#endif  // !CLSID_SCOPE

CLSID_SCOPE const CLSID
  CLSID_DxcPixDxilDebugger = {/* a712b622-5af7-4c77-a965-c83ac1a5d8bc */
      0xa712b622,
      0x5af7,
      0x4c77,
      {0xa9, 0x65, 0xc8, 0x3a, 0xc1, 0xa5, 0xd8, 0xbc}};

#endif
