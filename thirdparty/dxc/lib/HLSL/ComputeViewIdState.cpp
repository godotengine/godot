///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ComputeViewIdState.cpp                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/ComputeViewIdState.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CallGraph.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::legacy;
using namespace hlsl;
using llvm::legacy::PassManager;
using llvm::legacy::FunctionPassManager;
using std::vector;
using std::unordered_set;
using std::unordered_map;

#define DXILVIEWID_DBG   0

#define DEBUG_TYPE "viewid"

DxilViewIdState::DxilViewIdState(DxilModule *pDxilModule)
    : m_pModule(pDxilModule) {}
unsigned DxilViewIdState::getNumInputSigScalars() const                   { return m_NumInputSigScalars; }
unsigned DxilViewIdState::getNumOutputSigScalars(unsigned StreamId) const { return m_NumOutputSigScalars[StreamId]; }
unsigned DxilViewIdState::getNumPCSigScalars() const                      { return m_NumPCOrPrimSigScalars; }
const DxilViewIdState::OutputsDependentOnViewIdType   &DxilViewIdState::getOutputsDependentOnViewId(unsigned StreamId) const    { return m_OutputsDependentOnViewId[StreamId]; }
const DxilViewIdState::OutputsDependentOnViewIdType   &DxilViewIdState::getPCOutputsDependentOnViewId() const                   { return m_PCOrPrimOutputsDependentOnViewId; }
const DxilViewIdState::InputsContributingToOutputType &DxilViewIdState::getInputsContributingToOutputs(unsigned StreamId) const { return m_InputsContributingToOutputs[StreamId]; }
const DxilViewIdState::InputsContributingToOutputType &DxilViewIdState::getInputsContributingToPCOutputs() const                { return m_InputsContributingToPCOrPrimOutputs; }
const DxilViewIdState::InputsContributingToOutputType &DxilViewIdState::getPCInputsContributingToOutputs() const                { return m_PCInputsContributingToOutputs; }

namespace {

void PrintOutputsDependentOnViewId(
    llvm::raw_ostream &OS, llvm::StringRef SetName, unsigned NumOutputs,
    const DxilViewIdState::OutputsDependentOnViewIdType
        &OutputsDependentOnViewId) {
  OS << SetName << " dependent on ViewId: { ";
  bool bFirst = true;
  for (unsigned i = 0; i < NumOutputs; i++) {
    if (OutputsDependentOnViewId[i]) {
      if (!bFirst)
        OS << ", ";
      OS << i;
      bFirst = false;
    }
  }
  OS << " }\n";
}

void PrintInputsContributingToOutputs(
    llvm::raw_ostream &OS, llvm::StringRef InputSetName,
    llvm::StringRef OutputSetName,
    const DxilViewIdState::InputsContributingToOutputType
        &InputsContributingToOutputs) {
  OS << InputSetName << " contributing to computation of " << OutputSetName
     << ":\n";
  for (auto &it : InputsContributingToOutputs) {
    unsigned outIdx = it.first;
    auto &Inputs = it.second;
    OS << "output " << outIdx << " depends on inputs: { ";
    bool bFirst = true;
    for (unsigned i : Inputs) {
      if (!bFirst)
        OS << ", ";
      OS << i;
      bFirst = false;
    }
    OS << " }\n";
  }
}
} // namespace

void DxilViewIdState::PrintSets(llvm::raw_ostream &OS) {
  const ShaderModel *pSM = m_pModule->GetShaderModel();
  OS << "ViewId state: \n";

  if (pSM->IsGS()) {
    OS << "Number of inputs: "   << m_NumInputSigScalars     << 
                 ", outputs: { " << m_NumOutputSigScalars[0] << ", " << m_NumOutputSigScalars[1] << ", " <<
                                    m_NumOutputSigScalars[2] << ", " << m_NumOutputSigScalars[3] << " }" <<
              ", patchconst: "   << m_NumPCOrPrimSigScalars        << "\n";
  } else if (pSM->IsMS()) {
    OS << "Number of inputs: " << m_NumInputSigScalars <<
      ", vertex outputs: " << m_NumOutputSigScalars[0] <<
      ", primitive outputs: " << m_NumPCOrPrimSigScalars << "\n";
  } else {
    OS << "Number of inputs: " << m_NumInputSigScalars <<
      ", outputs: " << m_NumOutputSigScalars[0] <<
      ", patchconst: " << m_NumPCOrPrimSigScalars << "\n";
  }

  if (pSM->IsGS()) {
    PrintOutputsDependentOnViewId(OS, "Outputs for Stream0", m_NumOutputSigScalars[0], m_OutputsDependentOnViewId[0]);
    PrintOutputsDependentOnViewId(OS, "Outputs for Stream1", m_NumOutputSigScalars[1], m_OutputsDependentOnViewId[1]);
    PrintOutputsDependentOnViewId(OS, "Outputs for Stream2", m_NumOutputSigScalars[2], m_OutputsDependentOnViewId[2]);
    PrintOutputsDependentOnViewId(OS, "Outputs for Stream3", m_NumOutputSigScalars[3], m_OutputsDependentOnViewId[3]);
  } else if (pSM->IsMS()) {
    PrintOutputsDependentOnViewId(OS, "Vertex Outputs", m_NumOutputSigScalars[0], m_OutputsDependentOnViewId[0]);
  } else {
    PrintOutputsDependentOnViewId(OS, "Outputs", m_NumOutputSigScalars[0], m_OutputsDependentOnViewId[0]);
  }

  if (pSM->IsHS()) {
    PrintOutputsDependentOnViewId(OS, "PCOutputs", m_NumPCOrPrimSigScalars, m_PCOrPrimOutputsDependentOnViewId);
  } else if (pSM->IsMS()) {
    PrintOutputsDependentOnViewId(OS, "Primitive Outputs", m_NumPCOrPrimSigScalars, m_PCOrPrimOutputsDependentOnViewId);
  }

  if (pSM->IsGS()) {
    PrintInputsContributingToOutputs(OS, "Inputs", "Outputs for Stream0", m_InputsContributingToOutputs[0]);
    PrintInputsContributingToOutputs(OS, "Inputs", "Outputs for Stream1", m_InputsContributingToOutputs[1]);
    PrintInputsContributingToOutputs(OS, "Inputs", "Outputs for Stream2", m_InputsContributingToOutputs[2]);
    PrintInputsContributingToOutputs(OS, "Inputs", "Outputs for Stream3", m_InputsContributingToOutputs[3]);
  } else if (pSM->IsMS()) {
    PrintInputsContributingToOutputs(OS, "Inputs", "Vertex Outputs", m_InputsContributingToOutputs[0]);
  } else {
    PrintInputsContributingToOutputs(OS, "Inputs", "Outputs", m_InputsContributingToOutputs[0]);
  }
  if (pSM->IsHS()) {
    PrintInputsContributingToOutputs(OS, "Inputs", "PCOutputs", m_InputsContributingToPCOrPrimOutputs);
  } else if (pSM->IsMS()) {
    PrintInputsContributingToOutputs(OS, "Inputs", "Primitive Outputs", m_InputsContributingToPCOrPrimOutputs);
  } else if (pSM->IsDS()) {
    PrintInputsContributingToOutputs(OS, "PCInputs", "Outputs", m_PCInputsContributingToOutputs);
  }
  OS << "\n";
}

void DxilViewIdState::Clear() {
  m_NumInputSigScalars  = 0;
  for (unsigned i = 0; i < kNumStreams; i++) {
    m_NumOutputSigScalars[i] = 0;
    m_OutputsDependentOnViewId[i].reset();
    m_InputsContributingToOutputs[i].clear();
  }
  m_NumPCOrPrimSigScalars     = 0;
  m_PCOrPrimOutputsDependentOnViewId.reset();
  m_InputsContributingToPCOrPrimOutputs.clear();
  m_PCInputsContributingToOutputs.clear();
  m_SerializedState.clear();
}

namespace {

unsigned RoundUpToUINT(unsigned x) { return (x + 31) / 32; }

void SerializeOutputsDependentOnViewId(
    unsigned NumOutputs,
    const DxilViewIdState::OutputsDependentOnViewIdType
        &OutputsDependentOnViewId,
    unsigned *&pData) {
  unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);

  // Serialize output dependence on ViewId.
  for (unsigned i = 0; i < NumOutUINTs; i++) {
    unsigned x = 0;
    for (unsigned j = 0; j < std::min(32u, NumOutputs - 32u * i); j++) {
      if (OutputsDependentOnViewId[i * 32 + j]) {
        x |= (1u << j);
      }
    }
    *pData++ = x;
  }
}

void SerializeInputsContributingToOutput(
    unsigned NumInputs, unsigned NumOutputs,
    const DxilViewIdState::InputsContributingToOutputType
        &InputsContributingToOutputs,
    unsigned *&pData) {
  unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);

  // Serialize output dependence on inputs.
  for (unsigned outputIdx = 0; outputIdx < NumOutputs; outputIdx++) {
    auto it = InputsContributingToOutputs.find(outputIdx);
    if (it != InputsContributingToOutputs.end()) {
      for (unsigned inputIdx : it->second) {
        unsigned w = outputIdx / 32;
        unsigned b = outputIdx % 32;
        pData[inputIdx * NumOutUINTs + w] |= (1u << b);
      }
    }
  }

  pData += NumInputs * NumOutUINTs;
}
} // namespace

void DxilViewIdState::Serialize() {
  const ShaderModel *pSM = m_pModule->GetShaderModel();
  m_SerializedState.clear();

  // Compute serialized state size in UINTs.
  unsigned NumInputs = getNumInputSigScalars();
  unsigned NumStreams = pSM->IsGS() ? kNumStreams : 1;
  unsigned Size = 0;
  Size += 1; // #Inputs.
  for (unsigned StreamId = 0; StreamId < NumStreams; StreamId++) {
    Size += 1; // #Outputs for stream StreamId.
    unsigned NumOutputs = getNumOutputSigScalars(StreamId);
    unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);
    if (m_bUsesViewId) {
      Size += NumOutUINTs; // m_OutputsDependentOnViewId[StreamId]
    }
    Size += NumInputs * NumOutUINTs; // m_InputsContributingToOutputs[StreamId]
  }
  if (pSM->IsHS() || pSM->IsDS() || pSM->IsMS()) {
    Size += 1; // #PatchConstant.
    unsigned NumPCs = getNumPCSigScalars();
    unsigned NumPCUINTs = RoundUpToUINT(NumPCs);
    if (pSM->IsHS() || pSM->IsMS()) {
      if (m_bUsesViewId) {
        Size += NumPCUINTs; // m_PCOrPrimOutputsDependentOnViewId
      }
      Size += NumInputs * NumPCUINTs; // m_InputsContributingToPCOrPrimOutputs
    } else {
      unsigned NumOutputs = getNumOutputSigScalars(0);
      unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);
      Size += NumPCs * NumOutUINTs; // m_PCInputsContributingToOutputs
    }
  }

  m_SerializedState.resize(Size);
  std::fill(m_SerializedState.begin(), m_SerializedState.end(), 0u);

  // Serialize ViewId state.
  unsigned *pData = &m_SerializedState[0];
  *pData++ = NumInputs;
  for (unsigned StreamId = 0; StreamId < NumStreams; StreamId++) {
    unsigned NumOutputs = getNumOutputSigScalars(StreamId);
    *pData++ = NumOutputs;
    if (m_bUsesViewId) {
      SerializeOutputsDependentOnViewId(
          NumOutputs, m_OutputsDependentOnViewId[StreamId], pData);
    }
    SerializeInputsContributingToOutput(
        NumInputs, NumOutputs, m_InputsContributingToOutputs[StreamId], pData);
  }
  if (pSM->IsHS() || pSM->IsDS() || pSM->IsMS()) {
    unsigned NumPCs = getNumPCSigScalars();
    *pData++ = NumPCs;
    if (pSM->IsHS() || pSM->IsMS()) {
      if (m_bUsesViewId) {
        SerializeOutputsDependentOnViewId(NumPCs, m_PCOrPrimOutputsDependentOnViewId,
                                          pData);
      }
      SerializeInputsContributingToOutput(
          NumInputs, NumPCs, m_InputsContributingToPCOrPrimOutputs, pData);
    } else {
      unsigned NumOutputs = getNumOutputSigScalars(0);
      SerializeInputsContributingToOutput(
          NumPCs, NumOutputs, m_PCInputsContributingToOutputs, pData);
    }
  }
  DXASSERT_NOMSG(pData == (&m_SerializedState[0] + Size));
}

const vector<unsigned> &DxilViewIdState::GetSerialized() {
  if (m_SerializedState.empty())
    Serialize();
  return m_SerializedState;
}

const vector<unsigned> &DxilViewIdState::GetSerialized() const {
  return m_SerializedState;
}

namespace {
unsigned DeserializeOutputsDependentOnViewId(
    unsigned NumOutputs,
    DxilViewIdState::OutputsDependentOnViewIdType &OutputsDependentOnViewId,
    const unsigned *pData, unsigned DataSize) {
  unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);
  IFTBOOL(NumOutUINTs <= DataSize, DXC_E_GENERAL_INTERNAL_ERROR);

  // Deserialize output dependence on ViewId.
  for (unsigned i = 0; i < NumOutUINTs; i++) {
    unsigned x = *pData++;
    for (unsigned j = 0; j < std::min(32u, NumOutputs - 32u * i); j++) {
      if (x & (1u << j)) {
        OutputsDependentOnViewId[i * 32 + j] = true;
      }
    }
  }

  return NumOutUINTs;
}

unsigned DeserializeInputsContributingToOutput(
    unsigned NumInputs, unsigned NumOutputs,
    DxilViewIdState::InputsContributingToOutputType
        &InputsContributingToOutputs,
    const unsigned *pData, unsigned DataSize) {
  unsigned NumOutUINTs = RoundUpToUINT(NumOutputs);
  unsigned Size = NumInputs * NumOutUINTs;
  IFTBOOL(Size <= DataSize, DXC_E_GENERAL_INTERNAL_ERROR);

  // Deserialize output dependence on inputs.
  for (unsigned inputIdx = 0; inputIdx < NumInputs; inputIdx++) {
    for (unsigned outputIdx = 0; outputIdx < NumOutputs; outputIdx++) {
      unsigned w = outputIdx / 32;
      unsigned b = outputIdx % 32;
      if (pData[inputIdx * NumOutUINTs + w] & (1u << b)) {
        InputsContributingToOutputs[outputIdx].insert(inputIdx);
      }
    }
  }

  return Size;
}
} // namespace

void DxilViewIdState::Deserialize(const unsigned *pData,
                                  unsigned DataSizeInUINTs) {
  Clear();
  m_SerializedState.resize(DataSizeInUINTs);
  memcpy(m_SerializedState.data(), pData, DataSizeInUINTs * sizeof(unsigned));

  const ShaderModel *pSM = m_pModule->GetShaderModel();
  m_bUsesViewId = m_pModule->m_ShaderFlags.GetViewID();
  unsigned ConsumedUINTs = 0;

  IFTBOOL(DataSizeInUINTs - ConsumedUINTs >= 1, DXC_E_GENERAL_INTERNAL_ERROR);
  unsigned NumInputs = pData[ConsumedUINTs++];
  m_NumInputSigScalars = NumInputs;

  unsigned NumStreams = pSM->IsGS() ? kNumStreams : 1;
  for (unsigned StreamId = 0; StreamId < NumStreams; StreamId++) {
    IFTBOOL(DataSizeInUINTs - ConsumedUINTs >= 1, DXC_E_GENERAL_INTERNAL_ERROR);
    unsigned NumOutputs = pData[ConsumedUINTs++];
    m_NumOutputSigScalars[StreamId] = NumOutputs;

    if (m_bUsesViewId) {
      ConsumedUINTs += DeserializeOutputsDependentOnViewId(
          NumOutputs, m_OutputsDependentOnViewId[StreamId],
          &pData[ConsumedUINTs], DataSizeInUINTs - ConsumedUINTs);
    }
    ConsumedUINTs += DeserializeInputsContributingToOutput(
        NumInputs, NumOutputs, m_InputsContributingToOutputs[StreamId],
        &pData[ConsumedUINTs], DataSizeInUINTs - ConsumedUINTs);
  }

  if (pSM->IsHS() || pSM->IsDS() || pSM->IsMS()) {
    IFTBOOL(DataSizeInUINTs - ConsumedUINTs >= 1, DXC_E_GENERAL_INTERNAL_ERROR);
    unsigned NumPCs = pData[ConsumedUINTs++];
    m_NumPCOrPrimSigScalars = NumPCs;
    if (pSM->IsHS() || pSM->IsMS()) {
      if (m_bUsesViewId) {
        ConsumedUINTs += DeserializeOutputsDependentOnViewId(
            NumPCs, m_PCOrPrimOutputsDependentOnViewId, &pData[ConsumedUINTs],
            DataSizeInUINTs - ConsumedUINTs);
      }
      ConsumedUINTs += DeserializeInputsContributingToOutput(
          NumInputs, NumPCs, m_InputsContributingToPCOrPrimOutputs,
          &pData[ConsumedUINTs], DataSizeInUINTs - ConsumedUINTs);
    } else {
      unsigned NumOutputs = getNumOutputSigScalars(0);
      ConsumedUINTs += DeserializeInputsContributingToOutput(
          NumPCs, NumOutputs, m_PCInputsContributingToOutputs,
          &pData[ConsumedUINTs], DataSizeInUINTs - ConsumedUINTs);
    }
  }

  DXASSERT_NOMSG(ConsumedUINTs == DataSizeInUINTs);
}
