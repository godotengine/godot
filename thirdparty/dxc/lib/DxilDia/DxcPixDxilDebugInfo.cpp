///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixDxilDebugInfo.cpp                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines the main class for dxcompiler's API for PIX support.              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"

#include "dxc/Support/exception.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include "DxcPixLiveVariables.h"
#include "DxcPixDxilDebugInfo.h"
#include "DxcPixBase.h"
#include "dxc/DxilPixPasses/DxilPixVirtualRegisters.h"

STDMETHODIMP dxil_debug_info::DxcPixDxilDebugInfo::GetLiveVariablesAt(
  _In_ DWORD InstructionOffset,
  _COM_Outptr_ IDxcPixDxilLiveVariables **ppLiveVariables)
{
  return m_LiveVars->GetLiveVariablesAtInstruction(
      FindInstruction(InstructionOffset),
      ppLiveVariables);
}

STDMETHODIMP dxil_debug_info::DxcPixDxilDebugInfo::IsVariableInRegister(
    _In_ DWORD InstructionOffset,
    _In_ const wchar_t *VariableName)
{
  return S_OK;
}

STDMETHODIMP dxil_debug_info::DxcPixDxilDebugInfo::GetFunctionName(
    _In_ DWORD InstructionOffset,
    _Outptr_result_z_ BSTR *ppFunctionName)
{
  llvm::Instruction *IP = FindInstruction(InstructionOffset);

  const llvm::DITypeIdentifierMap EmptyMap;

  if (const llvm::DebugLoc &DL = IP->getDebugLoc())
  {
    auto *S = llvm::dyn_cast<llvm::DIScope>(DL.getScope());
    while(S != nullptr && !llvm::isa<llvm::DICompileUnit>(S))
    {
      if (auto *SS = llvm::dyn_cast<llvm::DISubprogram>(S))
      {
        *ppFunctionName = CComBSTR(CA2W(SS->getName().data())).Detach();
        return S_OK;
      }

      S = S->getScope().resolve(EmptyMap);
    }
  }

  *ppFunctionName = CComBSTR(L"<???>").Detach();
  return S_FALSE;
}

STDMETHODIMP dxil_debug_info::DxcPixDxilDebugInfo::GetStackDepth(
    _In_ DWORD InstructionOffset,
    _Outptr_ DWORD *StackDepth
)
{
  llvm::Instruction *IP = FindInstruction(InstructionOffset);

  DWORD Depth = 0;
  llvm::DebugLoc DL = IP->getDebugLoc();
  while (DL && DL.getInlinedAtScope() != nullptr)
  {
    DL = DL.getInlinedAt();
    ++Depth;
  }

  *StackDepth = Depth;
  return S_OK;
}

#include "DxilDiaSession.h"

dxil_debug_info::DxcPixDxilDebugInfo::DxcPixDxilDebugInfo(
    IMalloc *pMalloc,
    dxil_dia::Session *pSession)
    : m_pMalloc(pMalloc)
    , m_pSession(pSession)
    , m_LiveVars(new LiveVariables())
{
  m_LiveVars->Init(this);
}

dxil_debug_info::DxcPixDxilDebugInfo::~DxcPixDxilDebugInfo() = default;

llvm::Module* dxil_debug_info::DxcPixDxilDebugInfo::GetModuleRef()
{
  return &m_pSession->ModuleRef();
}

llvm::Instruction* dxil_debug_info::DxcPixDxilDebugInfo::FindInstruction(
    DWORD InstructionOffset
) const
{
  const auto & Instructions = m_pSession->InstructionsRef();
  auto it = Instructions.find(InstructionOffset);
  if (it == Instructions.end())
  {
    throw hlsl::Exception(E_BOUNDS, "Out-of-bounds: Instruction offset");
  }

  return const_cast<llvm::Instruction *>(it->second);
}

STDMETHODIMP
dxil_debug_info::DxcPixDxilDebugInfo::InstructionOffsetsFromSourceLocation(
    _In_ const wchar_t *FileName, _In_ DWORD SourceLine,
    _In_ DWORD SourceColumn,
    _COM_Outptr_ IDxcPixDxilInstructionOffsets **ppOffsets) 
{
  return dxil_debug_info::NewDxcPixDxilDebugInfoObjectOrThrow<
      dxil_debug_info::DxcPixDxilInstructionOffsets>(
      ppOffsets, m_pMalloc, m_pSession, FileName, SourceLine, SourceColumn);
}

STDMETHODIMP
dxil_debug_info::DxcPixDxilDebugInfo::SourceLocationsFromInstructionOffset(
    _In_ DWORD InstructionOffset,
    _COM_Outptr_ IDxcPixDxilSourceLocations **ppSourceLocations) {

  llvm::Instruction *IP = FindInstruction(InstructionOffset);

  return dxil_debug_info::NewDxcPixDxilDebugInfoObjectOrThrow<
      dxil_debug_info::DxcPixDxilSourceLocations>(
          ppSourceLocations, m_pMalloc, m_pSession, IP);
}

static bool CompareFilenames(const wchar_t * l, const char * r)
{
  while (*l && *r) {
    bool theSame = false;
    if (*l == L'/' && *r == '\\') {
      theSame = true;
    }
    if (*l == L'\\' && *r == '/') {
      theSame = true;
    }
    if (!theSame) {
      if (::tolower(*l) != ::tolower(*r)) {
        return false;
      }
    }
    l++;
    r++;
  }
  if (*l || *r) {
    return false;
  }
  return true;
}

dxil_debug_info::DxcPixDxilInstructionOffsets::DxcPixDxilInstructionOffsets(
  IMalloc *pMalloc,
  dxil_dia::Session *pSession,
  const wchar_t *FileName,
  DWORD SourceLine,
  DWORD SourceColumn) 
{
  assert(SourceColumn == 0);
  (void)SourceColumn;
  auto Fn = pSession->DxilModuleRef().GetEntryFunction();
  auto &Blocks = Fn->getBasicBlockList();
  for (auto& CurrentBlock : Blocks) {
    auto& Is = CurrentBlock.getInstList();
    for (auto& Inst : Is) {
      auto & debugLoc = Inst.getDebugLoc();
      if (debugLoc)
      {
        unsigned line = debugLoc.getLine();
        if (line == SourceLine)
        {
          auto file = debugLoc.get()->getFilename();
          if (CompareFilenames(FileName, file.str().c_str()))
          {
            std::uint32_t InstructionNumber;
            if (pix_dxil::PixDxilInstNum::FromInst(&Inst, &InstructionNumber))
            {
              m_offsets.push_back(InstructionNumber);
            }
          }
        }
      }
    }
  }
}

DWORD dxil_debug_info::DxcPixDxilInstructionOffsets::GetCount()
{
  return static_cast<DWORD>(m_offsets.size());
}

DWORD dxil_debug_info::DxcPixDxilInstructionOffsets::GetOffsetByIndex(DWORD Index) 
{
  if (Index < static_cast<DWORD>(m_offsets.size()))
  {
    return m_offsets[Index];
  }
  return static_cast<DWORD>(-1);
}


dxil_debug_info::DxcPixDxilSourceLocations::DxcPixDxilSourceLocations(
    IMalloc* pMalloc,
    dxil_dia::Session* pSession,
    llvm::Instruction* IP)
{
    const llvm::DITypeIdentifierMap EmptyMap;

    if (const llvm::DebugLoc& DL = IP->getDebugLoc())
    {
        auto* S = llvm::dyn_cast<llvm::DIScope>(DL.getScope());
        while (S != nullptr && !llvm::isa<llvm::DIFile>(S))
        {
            S = S->getScope().resolve(EmptyMap);
        }

        if (S != nullptr)
        {
          Location loc;
          loc.Line = DL->getLine();
          loc.Column = DL->getColumn();
          loc.Filename = CA2W(S->getFilename().data());
          m_locations.emplace_back(std::move(loc));
        }
    }
}

STDMETHODIMP_(DWORD) dxil_debug_info::DxcPixDxilSourceLocations::GetCount()
{
    return static_cast<DWORD>(m_locations.size());
}

STDMETHODIMP dxil_debug_info::DxcPixDxilSourceLocations::GetFileNameByIndex(
    _In_ DWORD Index, _Outptr_result_z_ BSTR *Name)
{
  if (Index >= static_cast<DWORD>(m_locations.size()))
  {
    return E_BOUNDS;
  }
  *Name = m_locations[Index].Filename.Copy();
  return S_OK;
}

STDMETHODIMP_(DWORD) dxil_debug_info::DxcPixDxilSourceLocations::GetColumnByIndex(
    _In_ DWORD Index) 
{
  if (Index >= static_cast<DWORD>(m_locations.size()))
  {
    return E_BOUNDS;
  }
  return m_locations[Index].Column;
}

STDMETHODIMP_(DWORD) dxil_debug_info::DxcPixDxilSourceLocations::GetLineNumberByIndex(
    _In_ DWORD Index) 
{
    if (Index >= static_cast<DWORD>(m_locations.size()))
    {
        return E_BOUNDS;
    }
    return m_locations[Index].Line;
}

