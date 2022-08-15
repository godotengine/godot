///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilShaderFlags.h                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Shader flags for a dxil shader function.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace hlsl {
  class DxilModule;
}

namespace llvm {
  class Function;
}

namespace hlsl {
  // Shader properties.
  class ShaderFlags {
  public:
    ShaderFlags();

    static ShaderFlags CollectShaderFlags(const llvm::Function *F, const hlsl::DxilModule *M);
    unsigned GetGlobalFlags() const;
    uint64_t GetFeatureInfo() const;
    static uint64_t GetShaderFlagsRawForCollection(); // some flags are collected (eg use 64-bit), some provided (eg allow refactoring)
    uint64_t GetShaderFlagsRaw() const;
    void SetShaderFlagsRaw(uint64_t data);
    void CombineShaderFlags(const ShaderFlags &other);

    void SetDisableOptimizations(bool flag) { m_bDisableOptimizations = flag; }
    bool GetDisableOptimizations() const { return m_bDisableOptimizations; }

    void SetDisableMathRefactoring(bool flag) { m_bDisableMathRefactoring = flag; }
    bool GetDisableMathRefactoring() const { return m_bDisableMathRefactoring; }

    void SetEnableDoublePrecision(bool flag) { m_bEnableDoublePrecision = flag; }
    bool GetEnableDoublePrecision() const { return m_bEnableDoublePrecision; }

    void SetForceEarlyDepthStencil(bool flag) { m_bForceEarlyDepthStencil = flag; }
    bool GetForceEarlyDepthStencil() const { return m_bForceEarlyDepthStencil; }

    void SetEnableRawAndStructuredBuffers(bool flag) { m_bEnableRawAndStructuredBuffers = flag; }
    bool GetEnableRawAndStructuredBuffers() const { return m_bEnableRawAndStructuredBuffers; }

    void SetLowPrecisionPresent(bool flag) { m_bLowPrecisionPresent = flag; }
    bool GetLowPrecisionPresent() const { return m_bLowPrecisionPresent; }

    void SetEnableDoubleExtensions(bool flag) { m_bEnableDoubleExtensions = flag; }
    bool GetEnableDoubleExtensions() const { return m_bEnableDoubleExtensions; }

    void SetEnableMSAD(bool flag) { m_bEnableMSAD = flag; }
    bool GetEnableMSAD() const { return m_bEnableMSAD; }

    void SetAllResourcesBound(bool flag) { m_bAllResourcesBound = flag; }
    bool GetAllResourcesBound() const { return m_bAllResourcesBound; }

    void SetCSRawAndStructuredViaShader4X(bool flag) { m_bCSRawAndStructuredViaShader4X = flag; }
    bool GetCSRawAndStructuredViaShader4X() const { return m_bCSRawAndStructuredViaShader4X; }

    void SetROVs(bool flag) { m_bROVS = flag; }
    bool GetROVs() const { return m_bROVS; }

    void SetWaveOps(bool flag) { m_bWaveOps = flag; }
    bool GetWaveOps() const { return m_bWaveOps; }

    void SetInt64Ops(bool flag) { m_bInt64Ops = flag; }
    bool GetInt64Ops() const { return m_bInt64Ops; }

    void SetTiledResources(bool flag) { m_bTiledResources = flag; }
    bool GetTiledResources() const { return m_bTiledResources; }

    void SetStencilRef(bool flag) { m_bStencilRef = flag; }
    bool GetStencilRef() const { return m_bStencilRef; }

    void SetInnerCoverage(bool flag) { m_bInnerCoverage = flag; }
    bool GetInnerCoverage() const { return m_bInnerCoverage; }

    void SetViewportAndRTArrayIndex(bool flag) { m_bViewportAndRTArrayIndex = flag; }
    bool GetViewportAndRTArrayIndex() const { return m_bViewportAndRTArrayIndex; }

    void SetUAVLoadAdditionalFormats(bool flag) { m_bUAVLoadAdditionalFormats = flag; }
    bool GetUAVLoadAdditionalFormats() const { return m_bUAVLoadAdditionalFormats; }

    void SetLevel9ComparisonFiltering(bool flag) { m_bLevel9ComparisonFiltering = flag; }
    bool GetLevel9ComparisonFiltering() const { return m_bLevel9ComparisonFiltering; }

    void Set64UAVs(bool flag) { m_b64UAVs = flag; }
    bool Get64UAVs() const { return m_b64UAVs; }

    void SetUAVsAtEveryStage(bool flag) { m_UAVsAtEveryStage = flag; }
    bool GetUAVsAtEveryStage() const { return m_UAVsAtEveryStage; }

    void SetViewID(bool flag) { m_bViewID = flag; }
    bool GetViewID() const { return m_bViewID; }

    void SetBarycentrics(bool flag) { m_bBarycentrics = flag; }
    bool GetBarycentrics() const { return m_bBarycentrics; }

    void SetUseNativeLowPrecision(bool flag) { m_bUseNativeLowPrecision = flag; }
    bool GetUseNativeLowPrecision() const { return m_bUseNativeLowPrecision; }

    void SetShadingRate(bool flag) { m_bShadingRate = flag; }
    bool GetShadingRate() const { return m_bShadingRate; }

    void SetRaytracingTier1_1(bool flag) { m_bRaytracingTier1_1 = flag; }
    bool GetRaytracingTier1_1() const { return m_bRaytracingTier1_1; }

    void SetSamplerFeedback(bool flag) { m_bSamplerFeedback = flag; }
    bool GetSamplerFeedback() const { return m_bSamplerFeedback; }

    void SetAtomicInt64OnTypedResource(bool flag) { m_bAtomicInt64OnTypedResource = flag; }
    bool GetAtomicInt64OnTypedResource() const { return m_bAtomicInt64OnTypedResource; }

    void SetAtomicInt64OnGroupShared(bool flag) { m_bAtomicInt64OnGroupShared = flag; }
    bool GetAtomicInt64OnGroupShared() const { return m_bAtomicInt64OnGroupShared; }

    void SetDerivativesInMeshAndAmpShaders(bool flag) { m_bDerivativesInMeshAndAmpShaders = flag; }
    bool GetDerivativesInMeshAndAmpShaders() { return m_bDerivativesInMeshAndAmpShaders; }

    void SetAtomicInt64OnHeapResource(bool flag) { m_bAtomicInt64OnHeapResource = flag; }
    bool GetAtomicInt64OnHeapResource() const { return m_bAtomicInt64OnHeapResource; }

    void SetResourceDescriptorHeapIndexing(bool flag) { m_bResourceDescriptorHeapIndexing = flag; }
    bool GetResourceDescriptorHeapIndexing() const { return m_bResourceDescriptorHeapIndexing; }

    void SetSamplerDescriptorHeapIndexing(bool flag) { m_bSamplerDescriptorHeapIndexing = flag; }
    bool GetSamplerDescriptorHeapIndexing() const { return m_bSamplerDescriptorHeapIndexing; }

    void SetResMayNotAlias(bool flag) { m_bResMayNotAlias = flag; }
    bool GetResMayNotAlias() const { return m_bResMayNotAlias; }

    void SetAdvancedTextureOps(bool flag) { m_bAdvancedTextureOps = flag; }
    bool GetAdvancedTextureOps() const { return m_bAdvancedTextureOps; }

    void SetWriteableMSAATextures(bool flag) { m_bWriteableMSAATextures = flag; }
    bool GetWriteableMSAATextures() const { return m_bWriteableMSAATextures; }

  private:
    unsigned m_bDisableOptimizations :1;   // D3D11_1_SB_GLOBAL_FLAG_SKIP_OPTIMIZATION
    unsigned m_bDisableMathRefactoring :1; //~D3D10_SB_GLOBAL_FLAG_REFACTORING_ALLOWED
    unsigned m_bEnableDoublePrecision :1; // D3D11_SB_GLOBAL_FLAG_ENABLE_DOUBLE_PRECISION_FLOAT_OPS
    unsigned m_bForceEarlyDepthStencil :1; // D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL
    unsigned m_bEnableRawAndStructuredBuffers :1; // D3D11_SB_GLOBAL_FLAG_ENABLE_RAW_AND_STRUCTURED_BUFFERS
    unsigned m_bLowPrecisionPresent :1; // D3D11_1_SB_GLOBAL_FLAG_ENABLE_MINIMUM_PRECISION
    unsigned m_bEnableDoubleExtensions :1; // D3D11_1_SB_GLOBAL_FLAG_ENABLE_DOUBLE_EXTENSIONS
    unsigned m_bEnableMSAD :1;        // D3D11_1_SB_GLOBAL_FLAG_ENABLE_SHADER_EXTENSIONS
    unsigned m_bAllResourcesBound :1; // D3D12_SB_GLOBAL_FLAG_ALL_RESOURCES_BOUND

    unsigned m_bViewportAndRTArrayIndex :1;   // SHADER_FEATURE_VIEWPORT_AND_RT_ARRAY_INDEX_FROM_ANY_SHADER_FEEDING_RASTERIZER
    unsigned m_bInnerCoverage :1;             // SHADER_FEATURE_INNER_COVERAGE
    unsigned m_bStencilRef  :1;               // SHADER_FEATURE_STENCIL_REF
    unsigned m_bTiledResources  :1;           // SHADER_FEATURE_TILED_RESOURCES
    unsigned m_bUAVLoadAdditionalFormats :1;  // SHADER_FEATURE_TYPED_UAV_LOAD_ADDITIONAL_FORMATS
    unsigned m_bLevel9ComparisonFiltering :1; // SHADER_FEATURE_LEVEL_9_COMPARISON_FILTERING
                                              // SHADER_FEATURE_11_1_SHADER_EXTENSIONS shared with EnableMSAD
    unsigned m_b64UAVs :1;                    // SHADER_FEATURE_64_UAVS
    unsigned m_UAVsAtEveryStage :1;           // SHADER_FEATURE_UAVS_AT_EVERY_STAGE
    unsigned m_bCSRawAndStructuredViaShader4X : 1; // SHADER_FEATURE_COMPUTE_SHADERS_PLUS_RAW_AND_STRUCTURED_BUFFERS_VIA_SHADER_4_X
    
    // SHADER_FEATURE_COMPUTE_SHADERS_PLUS_RAW_AND_STRUCTURED_BUFFERS_VIA_SHADER_4_X is specifically
    // about shader model 4.x.

    unsigned m_bROVS :1;              // SHADER_FEATURE_ROVS
    unsigned m_bWaveOps :1;           // SHADER_FEATURE_WAVE_OPS
    unsigned m_bInt64Ops :1;          // SHADER_FEATURE_INT64_OPS
    unsigned m_bViewID : 1;           // SHADER_FEATURE_VIEWID
    unsigned m_bBarycentrics : 1;     // SHADER_FEATURE_BARYCENTRICS

    unsigned m_bUseNativeLowPrecision : 1;

    unsigned m_bShadingRate : 1;      // SHADER_FEATURE_SHADINGRATE

    unsigned m_bRaytracingTier1_1 : 1; // SHADER_FEATURE_RAYTRACING_TIER_1_1
    unsigned m_bSamplerFeedback : 1; // SHADER_FEATURE_SAMPLER_FEEDBACK

    unsigned m_bAtomicInt64OnTypedResource : 1; // SHADER_FEATURE_ATOMIC_INT64_ON_TYPED_RESOURCE
    unsigned m_bAtomicInt64OnGroupShared : 1; // SHADER_FEATURE_ATOMIC_INT64_ON_GROUP_SHARED

    unsigned m_bDerivativesInMeshAndAmpShaders : 1; //SHADER_FEATURE_DERIVATIVES_IN_MESH_AND_AMPLIFICATION_SHADERS

    unsigned m_bResourceDescriptorHeapIndexing : 1;  // SHADER_FEATURE_RESOURCE_DESCRIPTOR_HEAP_INDEXING
    unsigned m_bSamplerDescriptorHeapIndexing : 1;  // SHADER_FEATURE_SAMPLER_DESCRIPTOR_HEAP_INDEXING

    unsigned m_bAtomicInt64OnHeapResource : 1; // SHADER_FEATURE_ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE

    // Global flag indicating that any UAV may not alias any other UAV.
    // Set if UAVs are used, unless -res-may-alias was specified.
    // For modules compiled against validator version < 1.7, this flag will be
    // cleared, and it must be assumed that UAV resources may alias.
    unsigned m_bResMayNotAlias : 1;

    unsigned m_bAdvancedTextureOps : 1;     // SHADER_FEATURE_ADVANCED_TEXTURE_OPS
    unsigned m_bWriteableMSAATextures : 1;  // SHADER_FEATURE_WRITEABLE_MSAA_TEXTURES

    uint32_t m_align1 : 28;            // align to 64 bit.
  };



}
