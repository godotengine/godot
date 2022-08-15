///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSignatureLower.h                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower signatures of entry function to DXIL LoadInput/StoreOutput.         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <unordered_set>
#include <unordered_map>
#include "dxc/DXIL/DxilConstants.h"

namespace llvm {
class Value;
class Argument;
class Function;
class StringRef;
} // namespace llvm

namespace hlsl {
class HLModule;
struct DxilEntrySignature;
class DxilFunctionAnnotation;
class ShaderModel;
struct DxilFunctionProps;
class DxilSignatureElement;
class DxilParameterAnnotation;
class SigPoint;

class HLSignatureLower {
public:
  HLSignatureLower(llvm::Function *F, HLModule &M, DxilEntrySignature &Sig)
      : Entry(F), HLM(M), EntrySig(Sig) {}
  void Run();

private:
  // Create signatures.
  void ProcessArgument(llvm::Function *func,
                       DxilFunctionAnnotation *EntryAnnotation,
                       llvm::Argument &arg, DxilFunctionProps &props,
                       const ShaderModel *pSM, bool isPatchConstantFunction,
                       bool forceOut, bool &hasClipPlane);
  void CreateDxilSignatures();
  // Allocate DXIL input/output.
  void AllocateDxilInputOutputs();
  // Generate DXIL input load, output store
  void GenerateDxilInputs();
  void GenerateDxilOutputs();
  void GenerateDxilPrimOutputs();
  void GenerateDxilInputsOutputs(DXIL::SignatureKind SK);
  void GenerateDxilCSInputs();
  void GenerateDxilPatchConstantLdSt();
  void GenerateDxilPatchConstantFunctionInputs();
  void GenerateClipPlanesForVS(llvm::Value *outPosition);
  bool HasClipPlanes();
  // Generate DXIL stream output operation.
  void GenerateStreamOutputOperation(llvm::Value *streamVal, unsigned streamID);
  // Generate DXIL stream output operations.
  void GenerateStreamOutputOperations();
  // Generate DXIL EmitIndices operation.
  void GenerateEmitIndicesOperation(llvm::Value *indicesOutput);
  // Generate DXIL EmitIndices operations.
  void GenerateEmitIndicesOperations();
  // Generate DXIL GetMeshPayload operation.
  void GenerateGetMeshPayloadOperation();

private:
  llvm::Function *Entry;
  HLModule &HLM;
  DxilEntrySignature &EntrySig;
  // For validation
  std::unordered_map<unsigned, std::unordered_set<unsigned>>
      m_InputSemanticsUsed, m_OutputSemanticsUsed[4],
      m_PatchConstantSemanticsUsed, m_OtherSemanticsUsed;
  // SignatureElement to Value map for GenerateDxilInputsOutputs.
  std::unordered_map<DxilSignatureElement *, llvm::Value *> m_sigValueMap;
  // Patch constant function inputs to signature element map for
  // GenerateDxilPatchConstantFunctionInputs.
  std::unordered_map<unsigned, DxilSignatureElement *>
      m_patchConstantInputsSigMap;
  // Set to save inout arguments for GenerateDxilInputsOutputs.
  std::unordered_set<llvm::Value *> m_inoutArgSet;
  // SignatureElement which has precise attribute for GenerateDxilInputsOutputs.
  std::unordered_set<DxilSignatureElement *> m_preciseSigSet;
};
} // namespace hlsl