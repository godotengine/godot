///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixDxilDebugInfo.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Declares the main class for dxcompiler's API for PIX support.             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"
#include "dxc/dxcapi.h"
#include "dxc/dxcpix.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include <memory>
#include <vector>

namespace dxil_dia
{
class Session;
}  // namespace dxil_dia

namespace llvm
{
class Instruction;
class Module;
}  // namespace llvm

namespace dxil_debug_info
{
class LiveVariables;

class DxcPixDxilDebugInfo : public IDxcPixDxilDebugInfo
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<dxil_dia::Session> m_pSession;
  std::unique_ptr<LiveVariables> m_LiveVars;

  DxcPixDxilDebugInfo(
      IMalloc *pMalloc,
      dxil_dia::Session *pSession);

  llvm::Instruction* FindInstruction(
      DWORD InstructionOffset
  ) const;

public:
  ~DxcPixDxilDebugInfo();

  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilDebugInfo)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilDebugInfo>(this, iid, ppvObject);
  }

  STDMETHODIMP GetLiveVariablesAt(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilLiveVariables **ppLiveVariables) override;

  STDMETHODIMP IsVariableInRegister(
      _In_ DWORD InstructionOffset,
      _In_ const wchar_t *VariableName) override;

  STDMETHODIMP GetFunctionName(
      _In_ DWORD InstructionOffset,
      _Outptr_result_z_ BSTR *ppFunctionName) override;

  STDMETHODIMP GetStackDepth(
      _In_ DWORD InstructionOffset,
      _Outptr_ DWORD *StackDepth) override;


  STDMETHODIMP InstructionOffsetsFromSourceLocation(
      _In_ const wchar_t *FileName, 
      _In_ DWORD SourceLine,
      _In_ DWORD SourceColumn, 
      _COM_Outptr_ IDxcPixDxilInstructionOffsets **ppOffsets) override;

  STDMETHODIMP  SourceLocationsFromInstructionOffset(
      _In_ DWORD InstructionOffset,
      _COM_Outptr_ IDxcPixDxilSourceLocations** ppSourceLocations) override;

  llvm::Module *GetModuleRef();

  IMalloc *GetMallocNoRef()
  {
    return m_pMalloc;
  }
};

class DxcPixDxilInstructionOffsets : public IDxcPixDxilInstructionOffsets
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  CComPtr<dxil_dia::Session> m_pSession;

  DxcPixDxilInstructionOffsets(
    IMalloc* pMalloc,
    dxil_dia::Session *pSession,
    const wchar_t *FileName,
    DWORD SourceLine,
    DWORD SourceColumn);

  std::vector<DWORD> m_offsets;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilInstructionOffsets)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilInstructionOffsets>(this, iid, ppvObject);
  }

  virtual STDMETHODIMP_(DWORD) GetCount() override;
  virtual STDMETHODIMP_(DWORD) GetOffsetByIndex(_In_ DWORD Index) override;
};

class DxcPixDxilSourceLocations : public IDxcPixDxilSourceLocations
{
private:
  DXC_MICROCOM_TM_REF_FIELDS()

  DxcPixDxilSourceLocations(
    IMalloc* pMalloc,
    dxil_dia::Session *pSession,
    llvm::Instruction* IP);

  struct Location
  {
      CComBSTR Filename;
      DWORD Line;
      DWORD Column;
  };
  std::vector<Location> m_locations;

public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_ALLOC(DxcPixDxilSourceLocations)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDxcPixDxilSourceLocations>(this, iid, ppvObject);
  }

  virtual STDMETHODIMP_(DWORD) GetCount() override;
  virtual STDMETHODIMP_(DWORD) GetLineNumberByIndex(_In_ DWORD Index) override;
  virtual STDMETHODIMP_(DWORD) GetColumnByIndex(_In_ DWORD Index)override;
  virtual STDMETHODIMP GetFileNameByIndex(_In_ DWORD Index,
                                          _Outptr_result_z_ BSTR *Name) override;
};

}  // namespace dxil_debug_info
