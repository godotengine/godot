///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilModule.h                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// The main class to work with DXIL, similar to LLVM module.                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilMetadataHelper.h"
#include "dxc/DXIL/DxilCBuffer.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilSampler.h"
#include "dxc/DXIL/DxilShaderFlags.h"
#include "dxc/DXIL/DxilSignature.h"
#include "dxc/DXIL/DxilSubobject.h"
#include "dxc/DXIL/DxilTypeSystem.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
class LLVMContext;
class Module;
class Function;
class Instruction;
class MDTuple;
class MDOperand;
class DebugInfoFinder;
}

namespace hlsl {

class ShaderModel;
class OP;
struct DxilFunctionProps;

class DxilEntryProps;

using DxilEntryPropsMap =
    std::unordered_map<const llvm::Function *, std::unique_ptr<DxilEntryProps>>;

/// Use this class to manipulate DXIL of a shader.
class DxilModule {
public:
  DxilModule(llvm::Module *pModule);
  ~DxilModule();

  // Subsystems.
  llvm::LLVMContext &GetCtx() const;
  llvm::Module *GetModule() const;
  OP *GetOP() const;
  void SetShaderModel(const ShaderModel *pSM, bool bUseMinPrecision = true);
  const ShaderModel *GetShaderModel() const;
  void GetDxilVersion(unsigned &DxilMajor, unsigned &DxilMinor) const;
  void SetValidatorVersion(unsigned ValMajor, unsigned ValMinor);
  bool UpgradeValidatorVersion(unsigned ValMajor, unsigned ValMinor);
  void GetValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const;
  void SetForceZeroStoreLifetimes(bool ForceZeroStoreLifetimes);
  bool GetForceZeroStoreLifetimes() const;

  // Return true on success, requires valid shader model and CollectShaderFlags to have been set
  bool GetMinValidatorVersion(unsigned &ValMajor, unsigned &ValMinor) const;
  // Update validator version to minimum if higher than current (ex: after CollectShaderFlags)
  bool UpgradeToMinValidatorVersion();

  // Entry functions.
  llvm::Function *GetEntryFunction();
  const llvm::Function *GetEntryFunction() const;
  void SetEntryFunction(llvm::Function *pEntryFunc);
  const std::string &GetEntryFunctionName() const;
  void SetEntryFunctionName(const std::string &name);
  llvm::Function *GetPatchConstantFunction();
  const llvm::Function *GetPatchConstantFunction() const;
  void SetPatchConstantFunction(llvm::Function *pFunc);
  bool IsEntryOrPatchConstantFunction(const llvm::Function* pFunc) const;
  llvm::SmallVector<llvm::Function *, 64> GetExportedFunctions();

  // Flags.
  unsigned GetGlobalFlags() const;
  void CollectShaderFlagsForModule();

  // Resources.
  unsigned AddCBuffer(std::unique_ptr<DxilCBuffer> pCB);
  DxilCBuffer &GetCBuffer(unsigned idx);
  const DxilCBuffer &GetCBuffer(unsigned idx) const;
  const std::vector<std::unique_ptr<DxilCBuffer> > &GetCBuffers() const;

  unsigned AddSampler(std::unique_ptr<DxilSampler> pSampler);
  DxilSampler &GetSampler(unsigned idx);
  const DxilSampler &GetSampler(unsigned idx) const;
  const std::vector<std::unique_ptr<DxilSampler> > &GetSamplers() const;

  unsigned AddSRV(std::unique_ptr<DxilResource> pSRV);
  DxilResource &GetSRV(unsigned idx);
  const DxilResource &GetSRV(unsigned idx) const;
  const std::vector<std::unique_ptr<DxilResource> > &GetSRVs() const;

  unsigned AddUAV(std::unique_ptr<DxilResource> pUAV);
  DxilResource &GetUAV(unsigned idx);
  const DxilResource &GetUAV(unsigned idx) const;
  const std::vector<std::unique_ptr<DxilResource> > &GetUAVs() const;

  void LoadDxilResourceBaseFromMDNode(llvm::MDNode *MD, DxilResourceBase &R);
  void LoadDxilResourceFromMDNode(llvm::MDNode *MD, DxilResource &R);
  void LoadDxilSamplerFromMDNode(llvm::MDNode *MD, DxilSampler &S);

  void RemoveUnusedResources();
  void RemoveResourcesWithUnusedSymbols();
  void RemoveFunction(llvm::Function *F);

  bool RenameResourcesWithPrefix(const std::string &prefix);
  bool RenameResourceGlobalsWithBinding(bool bKeepName = true);

  // Signatures.
  DxilSignature &GetInputSignature();
  const DxilSignature &GetInputSignature() const;
  DxilSignature &GetOutputSignature();
  const DxilSignature &GetOutputSignature() const;
  DxilSignature &GetPatchConstOrPrimSignature();
  const DxilSignature &GetPatchConstOrPrimSignature() const;
  const std::vector<uint8_t> &GetSerializedRootSignature() const;
  std::vector<uint8_t> &GetSerializedRootSignature();

  bool HasDxilEntrySignature(const llvm::Function *F) const;
  DxilEntrySignature &GetDxilEntrySignature(const llvm::Function *F);
  // Move DxilEntryProps of F to NewF.
  void ReplaceDxilEntryProps(llvm::Function *F, llvm::Function *NewF);
  // Clone DxilEntryProps of F to NewF.
  void CloneDxilEntryProps(llvm::Function *F, llvm::Function *NewF);
  bool HasDxilEntryProps(const llvm::Function *F) const;
  DxilEntryProps &GetDxilEntryProps(const llvm::Function *F);
  const DxilEntryProps &GetDxilEntryProps(const llvm::Function *F) const;

  // DxilFunctionProps.
  bool HasDxilFunctionProps(const llvm::Function *F) const;
  DxilFunctionProps &GetDxilFunctionProps(const llvm::Function *F);
  const DxilFunctionProps &GetDxilFunctionProps(const llvm::Function *F) const;

  // Move DxilFunctionProps of F to NewF.
  void SetPatchConstantFunctionForHS(llvm::Function *hullShaderFunc, llvm::Function *patchConstantFunc);
  bool IsGraphicsShader(const llvm::Function *F) const; // vs,hs,ds,gs,ps
  bool IsPatchConstantShader(const llvm::Function *F) const;
  bool IsComputeShader(const llvm::Function *F) const;

  // Is an entry function that uses input/output signature conventions?
  // Includes: vs/hs/ds/gs/ps/cs as well as the patch constant function.
  bool IsEntryThatUsesSignatures(const llvm::Function *F) const ;
  // Is F an entry?
  // Includes: IsEntryThatUsesSignatures and all ray tracing shaders.
  bool IsEntry(const llvm::Function *F) const;

  // Remove Root Signature from module metadata, return true if changed
  bool StripRootSignatureFromMetadata();
  // Remove Subobjects from module metadata, return true if changed
  bool StripSubobjectsFromMetadata();
  // Update validator version metadata to current setting
  void UpdateValidatorVersionMetadata();

  // DXIL type system.
  DxilTypeSystem &GetTypeSystem();
  const DxilTypeSystem &GetTypeSystem() const;

  /// Emit llvm.used array to make sure that optimizations do not remove unreferenced globals.
  void EmitLLVMUsed();
  std::vector<llvm::GlobalVariable* > &GetLLVMUsed();
  void ClearLLVMUsed();

  // ViewId state.
  std::vector<unsigned> &GetSerializedViewIdState();
  const std::vector<unsigned> &GetSerializedViewIdState() const;

  // DXIL metadata manipulation.
  /// Clear all DXIL data that exists in in-memory form.
  static void ClearDxilMetadata(llvm::Module &M);
  /// Serialize DXIL in-memory form to metadata form.
  void EmitDxilMetadata();
  /// Update resource metadata.
  /// Note: this method not update Metadata for ViewIdState.
  void ReEmitDxilResources();
  /// Deserialize DXIL metadata form into in-memory form.
  void LoadDxilMetadata();
  /// Return true if non-fatal metadata error was detected.
  bool HasMetadataErrors();

  void EmitDxilCounters();
  void LoadDxilCounters(DxilCounters &counters) const;

  /// Check if a Named meta data node is known by dxil module.
  static bool IsKnownNamedMetaData(llvm::NamedMDNode &Node);

  // Reset functions used to transfer ownership.
  void ResetEntrySignature(DxilEntrySignature *pValue);
  void ResetSerializedRootSignature(std::vector<uint8_t> &Value);
  void ResetTypeSystem(DxilTypeSystem *pValue);
  void ResetOP(hlsl::OP *hlslOP);
  void ResetEntryPropsMap(DxilEntryPropsMap &&PropMap);

  bool StripReflection();
  void StripDebugRelatedCode();

  // Helper to remove dx.* metadata with source and compile options.
  // If the parameter `bReplaceWithDummyData` is true, the named metadata
  // are replaced with valid empty data that satisfy tools.
  void StripShaderSourcesAndCompileOptions(bool bReplaceWithDummyData=false);
  llvm::DebugInfoFinder &GetOrCreateDebugInfoFinder();

  static DxilModule *TryGetDxilModule(llvm::Module *pModule);

  // Helpers for working with precise.

  // Return true if the instruction should be considered precise.
  //
  // An instruction can be marked precise in the following ways:
  //
  // 1. Global refactoring is disabled.
  // 2. The instruction has a precise metadata annotation.
  // 3. The instruction has precise fast math flags set.
  //
  bool IsPrecise(const llvm::Instruction *inst) const;

  // Check if the instruction has fast math flags configured to indicate
  // the instruction is precise.
  static bool HasPreciseFastMathFlags(const llvm::Instruction *inst);
  
  // Set fast math flags configured to indicate the instruction is precise.
  static void SetPreciseFastMathFlags(llvm::Instruction *inst);
  
  // True if fast math flags are preserved across serialize/deserialize.
  static bool PreservesFastMathFlags(const llvm::Instruction *inst);

public:
  ShaderFlags m_ShaderFlags;
  void CollectShaderFlagsForModule(ShaderFlags &Flags);

  // Check if DxilModule contains multi component UAV Loads.
  // This funciton must be called after unused resources are removed from DxilModule
  bool ModuleHasMulticomponentUAVLoads();

  // Compute/Mesh/Amplification shader.
  void SetNumThreads(unsigned x, unsigned y, unsigned z);
  unsigned GetNumThreads(unsigned idx) const;

  // Compute shader
  void SetWaveSize(unsigned size);
  unsigned GetWaveSize() const;

  // Geometry shader.
  DXIL::InputPrimitive GetInputPrimitive() const;
  void SetInputPrimitive(DXIL::InputPrimitive IP);
  unsigned GetMaxVertexCount() const;
  void SetMaxVertexCount(unsigned Count);
  DXIL::PrimitiveTopology GetStreamPrimitiveTopology() const;
  void SetStreamPrimitiveTopology(DXIL::PrimitiveTopology Topology);
  bool HasMultipleOutputStreams() const;
  unsigned GetOutputStream() const;
  unsigned GetGSInstanceCount() const;
  void SetGSInstanceCount(unsigned Count);
  bool IsStreamActive(unsigned Stream) const;
  void SetStreamActive(unsigned Stream, bool bActive);
  void SetActiveStreamMask(unsigned Mask);
  unsigned GetActiveStreamMask() const;

  // Language options
  // UseMinPrecision must be set at SetShaderModel time.
  bool GetUseMinPrecision() const;
  void SetDisableOptimization(bool disableOptimization);
  bool GetDisableOptimization() const;
  void SetAllResourcesBound(bool resourcesBound);
  bool GetAllResourcesBound() const;
  void SetResMayAlias(bool resMayAlias);
  bool GetResMayAlias() const;

  // Intermediate options that do not make it to DXIL
  void SetLegacyResourceReservation(bool legacyResourceReservation);
  bool GetLegacyResourceReservation() const;
  void ClearIntermediateOptions();

  // Hull and Domain shaders.
  unsigned GetInputControlPointCount() const;
  void SetInputControlPointCount(unsigned NumICPs);
  DXIL::TessellatorDomain GetTessellatorDomain() const;
  void SetTessellatorDomain(DXIL::TessellatorDomain TessDomain);

  // Hull shader.
  unsigned GetOutputControlPointCount() const;
  void SetOutputControlPointCount(unsigned NumOCPs);
  DXIL::TessellatorPartitioning GetTessellatorPartitioning() const;
  void SetTessellatorPartitioning(DXIL::TessellatorPartitioning TessPartitioning);
  DXIL::TessellatorOutputPrimitive GetTessellatorOutputPrimitive() const;
  void SetTessellatorOutputPrimitive(DXIL::TessellatorOutputPrimitive TessOutputPrimitive);
  float GetMaxTessellationFactor() const;
  void SetMaxTessellationFactor(float MaxTessellationFactor);

  // Mesh shader
  unsigned GetMaxOutputVertices() const;
  void SetMaxOutputVertices(unsigned NumOVs);
  unsigned GetMaxOutputPrimitives() const;
  void SetMaxOutputPrimitives(unsigned NumOPs);
  DXIL::MeshOutputTopology GetMeshOutputTopology() const;
  void SetMeshOutputTopology(DXIL::MeshOutputTopology MeshOutputTopology);
  unsigned GetPayloadSizeInBytes() const;
  void SetPayloadSizeInBytes(unsigned Size);

  // AutoBindingSpace also enables automatic binding for libraries if set.
  // UINT_MAX == unset
  void SetAutoBindingSpace(uint32_t Space);
  uint32_t GetAutoBindingSpace() const;

  void SetShaderProperties(DxilFunctionProps *props);

  DxilSubobjects *GetSubobjects();
  const DxilSubobjects *GetSubobjects() const;
  DxilSubobjects *ReleaseSubobjects();
  void ResetSubobjects(DxilSubobjects *subobjects);

private:
  // Signatures.
  std::vector<uint8_t> m_SerializedRootSignature;

  // Shader resources.
  std::vector<std::unique_ptr<DxilResource> > m_SRVs;
  std::vector<std::unique_ptr<DxilResource> > m_UAVs;
  std::vector<std::unique_ptr<DxilCBuffer> > m_CBuffers;
  std::vector<std::unique_ptr<DxilSampler> > m_Samplers;

  // Geometry shader.
  DXIL::PrimitiveTopology m_StreamPrimitiveTopology;
  unsigned m_ActiveStreamMask;

private:
  enum IntermediateFlags : uint32_t {
    LegacyResourceReservation = 1 << 0,
  };

private:
  llvm::LLVMContext &m_Ctx;
  llvm::Module *m_pModule;
  llvm::Function *m_pEntryFunc;
  std::string m_EntryName;
  std::unique_ptr<DxilMDHelper> m_pMDHelper;
  std::unique_ptr<llvm::DebugInfoFinder> m_pDebugInfoFinder;
  const ShaderModel *m_pSM;
  unsigned m_DxilMajor;
  unsigned m_DxilMinor;
  unsigned m_ValMajor;
  unsigned m_ValMinor;
  bool m_ForceZeroStoreLifetimes;

  std::unique_ptr<OP> m_pOP;
  size_t m_pUnused;

  // LLVM used.
  std::vector<llvm::GlobalVariable*> m_LLVMUsed;

  // Type annotations.
  std::unique_ptr<DxilTypeSystem> m_pTypeSystem;

  // EntryProps for shader functions.
  DxilEntryPropsMap  m_DxilEntryPropsMap;

  // Keeps track of patch constant functions used by hull shaders
  std::unordered_set<const llvm::Function *>  m_PatchConstantFunctions;

  // Serialized ViewId state.
  std::vector<unsigned> m_SerializedState;

  // DXIL metadata serialization/deserialization.
  llvm::MDTuple *EmitDxilResources();
  void LoadDxilResources(const llvm::MDOperand &MDO);

  // Helpers.
  template<typename T> unsigned AddResource(std::vector<std::unique_ptr<T> > &Vec, std::unique_ptr<T> pRes);
  void LoadDxilSignature(const llvm::MDTuple *pSigTuple, DxilSignature &Sig, bool bInput);

  // properties from HLModule preserved as ShaderFlags
  bool m_bDisableOptimizations;
  bool m_bUseMinPrecision;
  bool m_bAllResourcesBound;
  bool m_bResMayAlias;

  // properties from HLModule that should not make it to the final DXIL
  uint32_t m_IntermediateFlags;
  uint32_t m_AutoBindingSpace;

  // porperties infered from the DXILTypeSystem
  bool m_bHasPayloadQualifiers;

  std::unique_ptr<DxilSubobjects> m_pSubobjects;

  // m_bMetadataErrors is true if non-fatal metadata errors were encountered.
  // Validator will fail in this case, but should not block module load.
  bool m_bMetadataErrors;
};

} // namespace hlsl
