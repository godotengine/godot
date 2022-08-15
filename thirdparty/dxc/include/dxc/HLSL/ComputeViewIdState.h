///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ComputeViewIdState.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Computes output registers dependent on ViewID.                            //
// Computes sets of input registers on which output registers depend.        //
// Computes which input/output shapes are dynamically indexed.               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "llvm/Pass.h"
#include "dxc/HLSL/ControlDependence.h"
#include "llvm/Support/GenericDomTree.h"

#include <memory>
#include <bitset>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>

namespace llvm {
  class Module;
  class Function;
  class BasicBlock;
  class Instruction;
  class ReturnInst;
  class Value;
  class PHINode;
  class AnalysisUsage;
  class CallGraph;
  class CallGraphNode;
  class ModulePass;
  class raw_ostream;
}

namespace hlsl {

class DxilModule;
class DxilSignature;
class DxilSignatureElement;

struct DxilViewIdStateData {
  static const unsigned kNumComps = 4;
  static const unsigned kMaxSigScalars = 32*4;

  using OutputsDependentOnViewIdType = std::bitset<kMaxSigScalars>;
  using InputsContributingToOutputType = std::map<unsigned, std::set<unsigned>>;

  static const unsigned kNumStreams = 4;

  unsigned m_NumInputSigScalars  = 0;
  unsigned m_NumOutputSigScalars[kNumStreams] = {0,0,0,0};
  unsigned m_NumPCOrPrimSigScalars     = 0;

  // Set of scalar outputs dependent on ViewID.
  OutputsDependentOnViewIdType m_OutputsDependentOnViewId[kNumStreams];
  OutputsDependentOnViewIdType m_PCOrPrimOutputsDependentOnViewId;

  // Set of scalar inputs contributing to computation of scalar outputs.
  InputsContributingToOutputType m_InputsContributingToOutputs[kNumStreams];
  InputsContributingToOutputType m_InputsContributingToPCOrPrimOutputs; // HS PC and MS Prim only.
  InputsContributingToOutputType m_PCInputsContributingToOutputs; // DS only.

  bool m_bUsesViewId = false;
};

class DxilViewIdState : public DxilViewIdStateData {
  static const unsigned kNumComps = 4;
  static const unsigned kMaxSigScalars = 32*4;
public:
  using OutputsDependentOnViewIdType = std::bitset<kMaxSigScalars>;
  using InputsContributingToOutputType = std::map<unsigned, std::set<unsigned>>;

  DxilViewIdState(DxilModule *pDxilModule);

  unsigned getNumInputSigScalars() const;
  unsigned getNumOutputSigScalars(unsigned StreamId) const;
  unsigned getNumPCSigScalars() const;
  const OutputsDependentOnViewIdType &getOutputsDependentOnViewId(unsigned StreamId) const;
  const OutputsDependentOnViewIdType &getPCOutputsDependentOnViewId() const;
  const InputsContributingToOutputType &getInputsContributingToOutputs(unsigned StreamId) const;
  const InputsContributingToOutputType &getInputsContributingToPCOutputs() const;
  const InputsContributingToOutputType &getPCInputsContributingToOutputs() const;

  void Serialize();
  const std::vector<unsigned> &GetSerialized();
  const std::vector<unsigned> &
  GetSerialized() const; // returns previously serialized data
  void Deserialize(const unsigned *pData, unsigned DataSizeInUINTs);
  void PrintSets(llvm::raw_ostream &OS);

private:
  DxilModule *m_pModule;
  // Serialized form.
  std::vector<unsigned> m_SerializedState;
  void Clear();
};

} // end of hlsl namespace


namespace llvm {

void initializeComputeViewIdStatePass(llvm::PassRegistry &);
llvm::ModulePass *createComputeViewIdStatePass();

} // end of llvm namespace
