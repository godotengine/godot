///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixLiveVariables.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implements a mapping from the instructions in the Module to the set of    //
// live variables available in that instruction.                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "dxc/Support/WinIncludes.h"

#include <map>
#include <memory>
#include <vector>

#include "DxcPixDxilDebugInfo.h"

namespace llvm
{
class DIVariable;
class Instruction;
class Module;
class Value;
}  // namespace llvm

namespace dxil_debug_info
{

// VariableInfo is the bag with the information about a particular
// DIVariable in the Module.
struct VariableInfo
{
  using OffsetInBits = unsigned;

  // Location is the dxil alloca register where this variable lives.
  struct Location
  {
    llvm::Value *m_V = nullptr;
    unsigned m_FragmentIndex = 0;
  };

  explicit VariableInfo(
      llvm::DIVariable *Variable
  ) : m_Variable(Variable)
  {
  }

  llvm::DIVariable *m_Variable;

  std::map<OffsetInBits, Location> m_ValueLocationMap;

#ifndef NDEBUG
  std::vector<bool> m_DbgDeclareValidation;
#endif // !NDEBUG
};

class LiveVariables {
public:
  LiveVariables();
  ~LiveVariables();

  HRESULT Init(DxcPixDxilDebugInfo *pDxilDebugInfo);
  void Clear();

  HRESULT GetLiveVariablesAtInstruction(
    llvm::Instruction *Instr,
    IDxcPixDxilLiveVariables **Result) const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_pImpl;
};

HRESULT CreateDxilLiveVariables(
    DxcPixDxilDebugInfo *pDxilDebugInfo,
    std::vector<const VariableInfo *> &&LiveVariables,
    IDxcPixDxilLiveVariables **ppResult);

}  // namespace dxil_debug_info
