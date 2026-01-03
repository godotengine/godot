//*********************************************************
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License (MIT).
//
//*********************************************************

#pragma once

#ifndef __cplusplus
#error D3DX12 requires C++
#endif

#include "d3d12.h"

//================================================================================================
// D3DX12 Check Feature Support
//================================================================================================

#include <vector>

class CD3DX12FeatureSupport
{
public: // Function declaration
    // Default constructor that creates an empty object
    CD3DX12FeatureSupport() noexcept;

    // Initialize data from the given device
    HRESULT Init(ID3D12Device* pDevice);

    // Retreives the status of the object. If an error occurred in the initialization process, the function returns the error code.
    HRESULT GetStatus() const noexcept { return m_hStatus; }

    // Getter functions for each feature class
    // D3D12_OPTIONS
    BOOL DoublePrecisionFloatShaderOps() const noexcept;
    BOOL OutputMergerLogicOp() const noexcept;
    D3D12_SHADER_MIN_PRECISION_SUPPORT MinPrecisionSupport() const noexcept;
    D3D12_TILED_RESOURCES_TIER TiledResourcesTier() const noexcept;
    D3D12_RESOURCE_BINDING_TIER ResourceBindingTier() const noexcept;
    BOOL PSSpecifiedStencilRefSupported() const noexcept;
    BOOL TypedUAVLoadAdditionalFormats() const noexcept;
    BOOL ROVsSupported() const noexcept;
    D3D12_CONSERVATIVE_RASTERIZATION_TIER ConservativeRasterizationTier() const noexcept;
    BOOL StandardSwizzle64KBSupported() const noexcept;
    BOOL CrossAdapterRowMajorTextureSupported() const noexcept;
    BOOL VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation() const noexcept;
    D3D12_RESOURCE_HEAP_TIER ResourceHeapTier() const noexcept;
    D3D12_CROSS_NODE_SHARING_TIER CrossNodeSharingTier() const noexcept;
    UINT MaxGPUVirtualAddressBitsPerResource() const noexcept;

    // FEATURE_LEVELS
    D3D_FEATURE_LEVEL MaxSupportedFeatureLevel() const noexcept;

    // FORMAT_SUPPORT
    HRESULT FormatSupport(DXGI_FORMAT Format, D3D12_FORMAT_SUPPORT1& Support1, D3D12_FORMAT_SUPPORT2& Support2) const;

    // MUTLTISAMPLE_QUALITY_LEVELS
    HRESULT MultisampleQualityLevels(DXGI_FORMAT Format, UINT SampleCount, D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS Flags, UINT& NumQualityLevels) const;

    // FORMAT_INFO
    HRESULT FormatInfo(DXGI_FORMAT Format, UINT8& PlaneCount) const;

    // GPU_VIRTUAL_ADDRESS_SUPPORT
    UINT MaxGPUVirtualAddressBitsPerProcess() const noexcept;

    // SHADER_MODEL
    D3D_SHADER_MODEL HighestShaderModel() const noexcept;

    // D3D12_OPTIONS1
    BOOL WaveOps() const noexcept;
    UINT WaveLaneCountMin() const noexcept;
    UINT WaveLaneCountMax() const noexcept;
    UINT TotalLaneCount() const noexcept;
    BOOL ExpandedComputeResourceStates() const noexcept;
    BOOL Int64ShaderOps() const noexcept;

    // PROTECTED_RESOURCE_SESSION_SUPPORT
    D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAGS ProtectedResourceSessionSupport(UINT NodeIndex = 0) const;

    // ROOT_SIGNATURE
    D3D_ROOT_SIGNATURE_VERSION HighestRootSignatureVersion() const noexcept;

    // ARCHITECTURE1
    BOOL TileBasedRenderer(UINT NodeIndex = 0) const;
    BOOL UMA(UINT NodeIndex = 0) const;
    BOOL CacheCoherentUMA(UINT NodeIndex = 0) const;
    BOOL IsolatedMMU(UINT NodeIndex = 0) const;

    // D3D12_OPTIONS2
    BOOL DepthBoundsTestSupported() const noexcept;
    D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER ProgrammableSamplePositionsTier() const noexcept;

    // SHADER_CACHE
    D3D12_SHADER_CACHE_SUPPORT_FLAGS ShaderCacheSupportFlags() const noexcept;

    // COMMAND_QUEUE_PRIORITY
    BOOL CommandQueuePrioritySupported(D3D12_COMMAND_LIST_TYPE CommandListType, UINT Priority);

    // D3D12_OPTIONS3
    BOOL CopyQueueTimestampQueriesSupported() const noexcept;
    BOOL CastingFullyTypedFormatSupported() const noexcept;
    D3D12_COMMAND_LIST_SUPPORT_FLAGS WriteBufferImmediateSupportFlags() const noexcept;
    D3D12_VIEW_INSTANCING_TIER ViewInstancingTier() const noexcept;
    BOOL BarycentricsSupported() const noexcept;

    // EXISTING_HEAPS
    BOOL ExistingHeapsSupported() const noexcept;

    // D3D12_OPTIONS4
    BOOL MSAA64KBAlignedTextureSupported() const noexcept;
    D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER SharedResourceCompatibilityTier() const noexcept;
    BOOL Native16BitShaderOpsSupported() const noexcept;

    // SERIALIZATION
    D3D12_HEAP_SERIALIZATION_TIER HeapSerializationTier(UINT NodeIndex = 0) const;

    // CROSS_NODE
    // CrossNodeSharingTier handled in D3D12Options
    BOOL CrossNodeAtomicShaderInstructions() const noexcept;

    // D3D12_OPTIONS5
    BOOL SRVOnlyTiledResourceTier3() const noexcept;
    D3D12_RENDER_PASS_TIER RenderPassesTier() const noexcept;
    D3D12_RAYTRACING_TIER RaytracingTier() const noexcept;

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    // DISPLAYABLE
    BOOL DisplayableTexture() const noexcept;
    // SharedResourceCompatibilityTier handled in D3D12Options4
#endif

    // D3D12_OPTIONS6
    BOOL AdditionalShadingRatesSupported() const noexcept;
    BOOL PerPrimitiveShadingRateSupportedWithViewportIndexing() const noexcept;
    D3D12_VARIABLE_SHADING_RATE_TIER VariableShadingRateTier() const noexcept;
    UINT ShadingRateImageTileSize() const noexcept;
    BOOL BackgroundProcessingSupported() const noexcept;

    // QUERY_META_COMMAND
    HRESULT QueryMetaCommand(D3D12_FEATURE_DATA_QUERY_META_COMMAND& dQueryMetaCommand) const;

    // D3D12_OPTIONS7
    D3D12_MESH_SHADER_TIER MeshShaderTier() const noexcept;
    D3D12_SAMPLER_FEEDBACK_TIER SamplerFeedbackTier() const noexcept;

    // PROTECTED_RESOURCE_SESSION_TYPE_COUNT
    UINT ProtectedResourceSessionTypeCount(UINT NodeIndex = 0) const;

    // PROTECTED_RESOURCE_SESSION_TYPES
    std::vector<GUID> ProtectedResourceSessionTypes(UINT NodeIndex = 0) const;

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
    // D3D12_OPTIONS8
    BOOL UnalignedBlockTexturesSupported() const noexcept;

    // D3D12_OPTIONS9
    BOOL MeshShaderPipelineStatsSupported() const noexcept;
    BOOL MeshShaderSupportsFullRangeRenderTargetArrayIndex() const noexcept;
    BOOL AtomicInt64OnTypedResourceSupported() const noexcept;
    BOOL AtomicInt64OnGroupSharedSupported() const noexcept;
    BOOL DerivativesInMeshAndAmplificationShadersSupported() const noexcept;
    D3D12_WAVE_MMA_TIER WaveMMATier() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    // D3D12_OPTIONS10
    BOOL VariableRateShadingSumCombinerSupported() const noexcept;
    BOOL MeshShaderPerPrimitiveShadingRateSupported() const noexcept;

    // D3D12_OPTIONS11
    BOOL AtomicInt64OnDescriptorHeapResourceSupported() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 600)
    // D3D12_OPTIONS12
    D3D12_TRI_STATE MSPrimitivesPipelineStatisticIncludesCulledPrimitives() const noexcept;
    BOOL EnhancedBarriersSupported() const noexcept;
    BOOL RelaxedFormatCastingSupported() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 602)
    // D3D12_OPTIONS13
    BOOL UnrestrictedBufferTextureCopyPitchSupported() const noexcept;
    BOOL UnrestrictedVertexElementAlignmentSupported() const noexcept;
    BOOL InvertedViewportHeightFlipsYSupported() const noexcept;
    BOOL InvertedViewportDepthFlipsZSupported() const noexcept;
    BOOL TextureCopyBetweenDimensionsSupported() const noexcept;
    BOOL AlphaBlendFactorSupported() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
    // D3D12_OPTIONS14
    BOOL AdvancedTextureOpsSupported() const noexcept;
    BOOL WriteableMSAATexturesSupported() const noexcept;
    BOOL IndependentFrontAndBackStencilRefMaskSupported() const noexcept;

    // D3D12_OPTIONS15
    BOOL TriangleFanSupported() const noexcept;
    BOOL DynamicIndexBufferStripCutSupported() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
    // D3D12_OPTIONS16
    BOOL DynamicDepthBiasSupported() const noexcept;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
    BOOL GPUUploadHeapSupported() const noexcept;

    // D3D12_OPTIONS17
    BOOL NonNormalizedCoordinateSamplersSupported() const noexcept;
    BOOL ManualWriteTrackingResourceSupported() const noexcept;

    // D3D12_OPTIONS18
    BOOL RenderPassesValid() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
    BOOL MismatchingOutputDimensionsSupported() const noexcept;
    UINT SupportedSampleCountsWithNoOutputs() const noexcept;
    BOOL PointSamplingAddressesNeverRoundUp() const noexcept;
    BOOL RasterizerDesc2Supported() const noexcept;
    BOOL NarrowQuadrilateralLinesSupported() const noexcept;
    BOOL AnisoFilterWithPointMipSupported() const noexcept;
    UINT MaxSamplerDescriptorHeapSize() const noexcept;
    UINT MaxSamplerDescriptorHeapSizeWithStaticSamplers() const noexcept;
    UINT MaxViewDescriptorHeapSize() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
    BOOL ComputeOnlyWriteWatchSupported() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
    D3D12_EXECUTE_INDIRECT_TIER ExecuteIndirectTier() const noexcept;
    D3D12_WORK_GRAPHS_TIER WorkGraphsTier() const noexcept;
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 617)
    D3D12_TIGHT_ALIGNMENT_TIER TightAlignmentSupportTier() const noexcept;
#endif

private: // Private structs and helpers declaration
    struct ProtectedResourceSessionTypesLocal : D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES
    {
        std::vector<GUID> TypeVec;
    };

    // Helper function to decide the highest shader model supported by the system
    // Stores the result in m_dShaderModel
    // Must be updated whenever a new shader model is added to the d3d12.h header
    HRESULT QueryHighestShaderModel();

    // Helper function to decide the highest root signature supported
    // Must be updated whenever a new root signature version is added to the d3d12.h header
    HRESULT QueryHighestRootSignatureVersion();

    // Helper funcion to decide the highest feature level
    HRESULT QueryHighestFeatureLevel();

    // Helper function to initialize local protected resource session types structs
    HRESULT QueryProtectedResourceSessionTypes(UINT NodeIndex, UINT Count);

private: // Member data
    // Pointer to the underlying device
    ID3D12Device* m_pDevice;

    // Stores the error code from initialization
    HRESULT m_hStatus;

    // Feature support data structs
    D3D12_FEATURE_DATA_D3D12_OPTIONS m_dOptions;
    D3D_FEATURE_LEVEL m_eMaxFeatureLevel;
    D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT m_dGPUVASupport;
    D3D12_FEATURE_DATA_SHADER_MODEL m_dShaderModel;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 m_dOptions1;
    std::vector<D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT> m_dProtectedResourceSessionSupport;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE m_dRootSignature;
    std::vector<D3D12_FEATURE_DATA_ARCHITECTURE1> m_dArchitecture1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS2 m_dOptions2;
    D3D12_FEATURE_DATA_SHADER_CACHE m_dShaderCache;
    D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY m_dCommandQueuePriority;
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 m_dOptions3;
    D3D12_FEATURE_DATA_EXISTING_HEAPS m_dExistingHeaps;
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 m_dOptions4;
    std::vector<D3D12_FEATURE_DATA_SERIALIZATION> m_dSerialization; // Cat2 NodeIndex
    D3D12_FEATURE_DATA_CROSS_NODE m_dCrossNode;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 m_dOptions5;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    D3D12_FEATURE_DATA_DISPLAYABLE m_dDisplayable;
#endif
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 m_dOptions6;
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 m_dOptions7;
    std::vector<D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT> m_dProtectedResourceSessionTypeCount; // Cat2 NodeIndex
    std::vector<ProtectedResourceSessionTypesLocal> m_dProtectedResourceSessionTypes; // Cat3
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
    D3D12_FEATURE_DATA_D3D12_OPTIONS8 m_dOptions8;
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 m_dOptions9;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    D3D12_FEATURE_DATA_D3D12_OPTIONS10 m_dOptions10;
    D3D12_FEATURE_DATA_D3D12_OPTIONS11 m_dOptions11;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 600)
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 m_dOptions12;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 602)
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 m_dOptions13;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 m_dOptions14;
    D3D12_FEATURE_DATA_D3D12_OPTIONS15 m_dOptions15;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 m_dOptions16;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
    D3D12_FEATURE_DATA_D3D12_OPTIONS17 m_dOptions17;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
    D3D12_FEATURE_DATA_D3D12_OPTIONS18 m_dOptions18;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 m_dOptions19;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
    D3D12_FEATURE_DATA_D3D12_OPTIONS20 m_dOptions20;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
    D3D12_FEATURE_DATA_D3D12_OPTIONS21 m_dOptions21;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 617)
    D3D12_FEATURE_DATA_TIGHT_ALIGNMENT m_dTightAlignment;
#endif
};

// Implementations for CD3DX12FeatureSupport functions

// Macro to set up a getter function for each entry in feature support data
// The getter function will have the same name as the feature option name
#define FEATURE_SUPPORT_GET(RETTYPE,FEATURE,OPTION) \
inline RETTYPE CD3DX12FeatureSupport::OPTION() const noexcept \
{ \
    return FEATURE.OPTION; \
}

// Macro to set up a getter function for each entry in feature support data
// Also specifies the name for the function which can be different from the feature name
#define FEATURE_SUPPORT_GET_NAME(RETTYPE,FEATURE,OPTION,NAME) \
inline RETTYPE CD3DX12FeatureSupport::NAME() const noexcept \
{\
    return FEATURE.OPTION; \
}

// Macro to set up a getter function for feature data indexed by the graphics node ID
// The default parameter is 0, or the first availabe graphics device node
#define FEATURE_SUPPORT_GET_NODE_INDEXED(RETTYPE,FEATURE,OPTION) \
inline RETTYPE CD3DX12FeatureSupport::OPTION(UINT NodeIndex) const \
{\
    return FEATURE[NodeIndex].OPTION; \
}

// Macro to set up a getter function for feature data indexed by NodeIndex
// Allows a custom name for the getter function
#define FEATURE_SUPPORT_GET_NODE_INDEXED_NAME(RETTYPE,FEATURE,OPTION,NAME) \
inline RETTYPE CD3DX12FeatureSupport::NAME(UINT NodeIndex) const \
{\
    return FEATURE[NodeIndex].OPTION; \
}

inline CD3DX12FeatureSupport::CD3DX12FeatureSupport() noexcept
: m_pDevice(nullptr)
, m_hStatus(E_INVALIDARG)
, m_dOptions{}
, m_eMaxFeatureLevel{}
, m_dGPUVASupport{}
, m_dShaderModel{}
, m_dOptions1{}
, m_dRootSignature{}
, m_dOptions2{}
, m_dShaderCache{}
, m_dCommandQueuePriority{}
, m_dOptions3{}
, m_dExistingHeaps{}
, m_dOptions4{}
, m_dCrossNode{}
, m_dOptions5{}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
, m_dDisplayable{}
#endif
, m_dOptions6{}
, m_dOptions7{}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
, m_dOptions8{}
, m_dOptions9{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
, m_dOptions10{}
, m_dOptions11{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 600)
, m_dOptions12{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 602)
, m_dOptions13{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
, m_dOptions14{}
, m_dOptions15{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
, m_dOptions16{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
, m_dOptions17{}
#endif
#if defined (D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
, m_dOptions18{}
#endif
#if defined (D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
, m_dOptions19{}
#endif
#if defined (D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
, m_dOptions20{}
#endif
#if defined (D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
, m_dOptions21{}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 617)
, m_dTightAlignment{}
#endif
{}

inline HRESULT CD3DX12FeatureSupport::Init(ID3D12Device* pDevice)
{
    if (!pDevice)
    {
        m_hStatus = E_INVALIDARG;
        return m_hStatus;
    }

    m_pDevice = pDevice;

    // Initialize static feature support data structures
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &m_dOptions, sizeof(m_dOptions))))
    {
        m_dOptions = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &m_dGPUVASupport, sizeof(m_dGPUVASupport))))
    {
        m_dGPUVASupport = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &m_dOptions1, sizeof(m_dOptions1))))
    {
        m_dOptions1 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &m_dOptions2, sizeof(m_dOptions2))))
    {
        m_dOptions2 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_CACHE, &m_dShaderCache, sizeof(m_dShaderCache))))
    {
        m_dShaderCache = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &m_dOptions3, sizeof(m_dOptions3))))
    {
        m_dOptions3 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &m_dExistingHeaps, sizeof(m_dExistingHeaps))))
    {
        m_dExistingHeaps = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &m_dOptions4, sizeof(m_dOptions4))))
    {
        m_dOptions4 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_CROSS_NODE, &m_dCrossNode, sizeof(m_dCrossNode))))
    {
        m_dCrossNode = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &m_dOptions5, sizeof(m_dOptions5))))
    {
        m_dOptions5 = {};
    }

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_DISPLAYABLE, &m_dDisplayable, sizeof(m_dDisplayable))))
    {
        m_dDisplayable = {};
    }
#endif

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &m_dOptions6, sizeof(m_dOptions6))))
    {
        m_dOptions6 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &m_dOptions7, sizeof(m_dOptions7))))
    {
        m_dOptions7 = {};
    }

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &m_dOptions8, sizeof(m_dOptions8))))
    {
        m_dOptions8 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &m_dOptions9, sizeof(m_dOptions9))))
    {
        m_dOptions9 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS10, &m_dOptions10, sizeof(m_dOptions10))))
    {
        m_dOptions10 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &m_dOptions11, sizeof(m_dOptions11))))
    {
        m_dOptions11 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 600)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &m_dOptions12, sizeof(m_dOptions12))))
    {
        m_dOptions12 = {};
        m_dOptions12.MSPrimitivesPipelineStatisticIncludesCulledPrimitives = D3D12_TRI_STATE::D3D12_TRI_STATE_UNKNOWN;
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 602)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS13, &m_dOptions13, sizeof(m_dOptions13))))
    {
        m_dOptions13 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS14, &m_dOptions14, sizeof(m_dOptions14))))
    {
        m_dOptions14 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15, &m_dOptions15, sizeof(m_dOptions15))))
    {
        m_dOptions15 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &m_dOptions16, sizeof(m_dOptions16))))
    {
        m_dOptions16 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &m_dOptions17, sizeof(m_dOptions17))))
    {
        m_dOptions17 = {};
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS18, &m_dOptions18, sizeof(m_dOptions18))))
    {
        m_dOptions18.RenderPassesValid = false;
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &m_dOptions19, sizeof(m_dOptions19))))
    {
        m_dOptions19 = {};
        m_dOptions19.SupportedSampleCountsWithNoOutputs = 1;
        m_dOptions19.MaxSamplerDescriptorHeapSize = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
        m_dOptions19.MaxSamplerDescriptorHeapSizeWithStaticSamplers = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
        m_dOptions19.MaxViewDescriptorHeapSize = D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1;
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS20, &m_dOptions20, sizeof(m_dOptions20))))
    {
        m_dOptions20 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &m_dOptions21, sizeof(m_dOptions21))))
    {
        m_dOptions21 = {};
    }
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 617)
    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_TIGHT_ALIGNMENT, &m_dTightAlignment, sizeof(m_dTightAlignment))))
    {
        m_dTightAlignment = {};
    }
#endif

    // Initialize per-node feature support data structures
    const UINT uNodeCount = m_pDevice->GetNodeCount();
    m_dProtectedResourceSessionSupport.resize(uNodeCount);
    m_dArchitecture1.resize(uNodeCount);
    m_dSerialization.resize(uNodeCount);
    m_dProtectedResourceSessionTypeCount.resize(uNodeCount);
    m_dProtectedResourceSessionTypes.resize(uNodeCount);
    for (UINT NodeIndex = 0; NodeIndex < uNodeCount; NodeIndex++)
    {
        m_dProtectedResourceSessionSupport[NodeIndex].NodeIndex = NodeIndex;
        if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT, &m_dProtectedResourceSessionSupport[NodeIndex], sizeof(m_dProtectedResourceSessionSupport[NodeIndex]))))
        {
            m_dProtectedResourceSessionSupport[NodeIndex].Support = D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_NONE;
        }

        m_dArchitecture1[NodeIndex].NodeIndex = NodeIndex;
        if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &m_dArchitecture1[NodeIndex], sizeof(m_dArchitecture1[NodeIndex]))))
        {
            D3D12_FEATURE_DATA_ARCHITECTURE dArchLocal = {};
            dArchLocal.NodeIndex = NodeIndex;
            if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &dArchLocal, sizeof(dArchLocal))))
            {
                dArchLocal.TileBasedRenderer = false;
                dArchLocal.UMA = false;
                dArchLocal.CacheCoherentUMA = false;
            }

            m_dArchitecture1[NodeIndex].TileBasedRenderer = dArchLocal.TileBasedRenderer;
            m_dArchitecture1[NodeIndex].UMA = dArchLocal.UMA;
            m_dArchitecture1[NodeIndex].CacheCoherentUMA = dArchLocal.CacheCoherentUMA;
            m_dArchitecture1[NodeIndex].IsolatedMMU = false;
        }

        m_dSerialization[NodeIndex].NodeIndex = NodeIndex;
        if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_SERIALIZATION, &m_dSerialization[NodeIndex], sizeof(m_dSerialization[NodeIndex]))))
        {
            m_dSerialization[NodeIndex].HeapSerializationTier = D3D12_HEAP_SERIALIZATION_TIER_0;
        }

        m_dProtectedResourceSessionTypeCount[NodeIndex].NodeIndex = NodeIndex;
        if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT, &m_dProtectedResourceSessionTypeCount[NodeIndex], sizeof(m_dProtectedResourceSessionTypeCount[NodeIndex]))))
        {
            m_dProtectedResourceSessionTypeCount[NodeIndex].Count = 0;
        }

        // Special procedure to initialize local protected resource session types structs
        // Must wait until session type count initialized
        QueryProtectedResourceSessionTypes(NodeIndex, m_dProtectedResourceSessionTypeCount[NodeIndex].Count);
    }

    // Initialize features that requires highest version check
    if (FAILED(m_hStatus = QueryHighestShaderModel()))
    {
        return m_hStatus;
    }

    if (FAILED(m_hStatus = QueryHighestRootSignatureVersion()))
    {
        return m_hStatus;
    }

    // Initialize Feature Levels data
    if (FAILED(m_hStatus = QueryHighestFeatureLevel()))
    {
        return m_hStatus;
    }

    return m_hStatus;
}

// 0: D3D12_OPTIONS
FEATURE_SUPPORT_GET(BOOL, m_dOptions, DoublePrecisionFloatShaderOps);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, OutputMergerLogicOp);
FEATURE_SUPPORT_GET(D3D12_SHADER_MIN_PRECISION_SUPPORT, m_dOptions, MinPrecisionSupport);
FEATURE_SUPPORT_GET(D3D12_TILED_RESOURCES_TIER, m_dOptions, TiledResourcesTier);
FEATURE_SUPPORT_GET(D3D12_RESOURCE_BINDING_TIER, m_dOptions, ResourceBindingTier);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, PSSpecifiedStencilRefSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, TypedUAVLoadAdditionalFormats);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, ROVsSupported);
FEATURE_SUPPORT_GET(D3D12_CONSERVATIVE_RASTERIZATION_TIER, m_dOptions, ConservativeRasterizationTier);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, StandardSwizzle64KBSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, CrossAdapterRowMajorTextureSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions, VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation);
FEATURE_SUPPORT_GET(D3D12_RESOURCE_HEAP_TIER, m_dOptions, ResourceHeapTier);

// Special procedure for handling caps that is also part of other features
inline D3D12_CROSS_NODE_SHARING_TIER CD3DX12FeatureSupport::CrossNodeSharingTier() const noexcept
{
    if (m_dCrossNode.SharingTier > D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED)
    {
        return m_dCrossNode.SharingTier;
    }
    else
    {
        return m_dOptions.CrossNodeSharingTier;
    }
}

inline UINT CD3DX12FeatureSupport::MaxGPUVirtualAddressBitsPerResource() const noexcept
{
    if (m_dOptions.MaxGPUVirtualAddressBitsPerResource > 0)
    {
        return m_dOptions.MaxGPUVirtualAddressBitsPerResource;
    }
    else
    {
        return m_dGPUVASupport.MaxGPUVirtualAddressBitsPerResource;
    }
}

// 1: Architecture
// Combined with Architecture1

// 2: Feature Levels
// Simply returns the highest supported feature level
inline D3D_FEATURE_LEVEL CD3DX12FeatureSupport::MaxSupportedFeatureLevel() const noexcept
{
    return m_eMaxFeatureLevel;
}

// 3: Feature Format Support
inline HRESULT CD3DX12FeatureSupport::FormatSupport(DXGI_FORMAT Format, D3D12_FORMAT_SUPPORT1& Support1, D3D12_FORMAT_SUPPORT2& Support2) const
{
    D3D12_FEATURE_DATA_FORMAT_SUPPORT dFormatSupport;
    dFormatSupport.Format = Format;

    // It is possible that the function call returns an error
    HRESULT result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &dFormatSupport, sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT));

    Support1 = dFormatSupport.Support1;
    Support2 = dFormatSupport.Support2; // Two outputs. Probably better just to take in the struct as an argument?

    return result;
}

// 4: Multisample Quality Levels
inline HRESULT CD3DX12FeatureSupport::MultisampleQualityLevels(DXGI_FORMAT Format, UINT SampleCount, D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS Flags, UINT& NumQualityLevels) const
{
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS dMultisampleQualityLevels;
    dMultisampleQualityLevels.Format = Format;
    dMultisampleQualityLevels.SampleCount = SampleCount;
    dMultisampleQualityLevels.Flags = Flags;

    HRESULT result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &dMultisampleQualityLevels, sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS));

    if (SUCCEEDED(result))
    {
        NumQualityLevels = dMultisampleQualityLevels.NumQualityLevels;
    }
    else
    {
        NumQualityLevels = 0;
    }

    return result;
}

// 5: Format Info
inline HRESULT CD3DX12FeatureSupport::FormatInfo(DXGI_FORMAT Format, UINT8& PlaneCount) const
{
    D3D12_FEATURE_DATA_FORMAT_INFO dFormatInfo;
    dFormatInfo.Format = Format;

    HRESULT result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &dFormatInfo, sizeof(D3D12_FEATURE_DATA_FORMAT_INFO));
    if (FAILED(result))
    {
        PlaneCount = 0;
    }
    else
    {
        PlaneCount = dFormatInfo.PlaneCount;
    }
    return result;
}

// 6: GPU Virtual Address Support
// MaxGPUVirtualAddressBitsPerResource handled in D3D12Options
FEATURE_SUPPORT_GET(UINT, m_dGPUVASupport, MaxGPUVirtualAddressBitsPerProcess);

// 7: Shader Model
inline D3D_SHADER_MODEL CD3DX12FeatureSupport::HighestShaderModel() const noexcept
{
    return m_dShaderModel.HighestShaderModel;
}

// 8: D3D12 Options1
FEATURE_SUPPORT_GET(BOOL, m_dOptions1, WaveOps);
FEATURE_SUPPORT_GET(UINT, m_dOptions1, WaveLaneCountMin);
FEATURE_SUPPORT_GET(UINT, m_dOptions1, WaveLaneCountMax);
FEATURE_SUPPORT_GET(UINT, m_dOptions1, TotalLaneCount);
FEATURE_SUPPORT_GET(BOOL, m_dOptions1, ExpandedComputeResourceStates);
FEATURE_SUPPORT_GET(BOOL, m_dOptions1, Int64ShaderOps);

// 10: Protected Resource Session Support
inline D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAGS CD3DX12FeatureSupport::ProtectedResourceSessionSupport(UINT NodeIndex) const
{
    return m_dProtectedResourceSessionSupport[NodeIndex].Support;
}

// 12: Root Signature
inline D3D_ROOT_SIGNATURE_VERSION CD3DX12FeatureSupport::HighestRootSignatureVersion() const noexcept
{
    return m_dRootSignature.HighestVersion;
}

// 16: Architecture1
// Same data fields can be queried from m_dArchitecture
FEATURE_SUPPORT_GET_NODE_INDEXED(BOOL, m_dArchitecture1, TileBasedRenderer);
FEATURE_SUPPORT_GET_NODE_INDEXED(BOOL, m_dArchitecture1, UMA);
FEATURE_SUPPORT_GET_NODE_INDEXED(BOOL, m_dArchitecture1, CacheCoherentUMA);
FEATURE_SUPPORT_GET_NODE_INDEXED(BOOL, m_dArchitecture1, IsolatedMMU);

// 18: D3D12 Options2
FEATURE_SUPPORT_GET(BOOL, m_dOptions2, DepthBoundsTestSupported);
FEATURE_SUPPORT_GET(D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER, m_dOptions2, ProgrammableSamplePositionsTier);

// 19: Shader Cache
FEATURE_SUPPORT_GET_NAME(D3D12_SHADER_CACHE_SUPPORT_FLAGS, m_dShaderCache, SupportFlags, ShaderCacheSupportFlags);

// 20: Command Queue Priority
inline BOOL CD3DX12FeatureSupport::CommandQueuePrioritySupported(D3D12_COMMAND_LIST_TYPE CommandListType, UINT Priority)
{
    m_dCommandQueuePriority.CommandListType = CommandListType;
    m_dCommandQueuePriority.Priority = Priority;

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &m_dCommandQueuePriority, sizeof(D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY))))
    {
        return false;
    }

    return m_dCommandQueuePriority.PriorityForTypeIsSupported;
}

// 21: D3D12 Options3
FEATURE_SUPPORT_GET(BOOL, m_dOptions3, CopyQueueTimestampQueriesSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions3, CastingFullyTypedFormatSupported);
FEATURE_SUPPORT_GET(D3D12_COMMAND_LIST_SUPPORT_FLAGS, m_dOptions3, WriteBufferImmediateSupportFlags);
FEATURE_SUPPORT_GET(D3D12_VIEW_INSTANCING_TIER, m_dOptions3, ViewInstancingTier);
FEATURE_SUPPORT_GET(BOOL, m_dOptions3, BarycentricsSupported);

// 22: Existing Heaps
FEATURE_SUPPORT_GET_NAME(BOOL, m_dExistingHeaps, Supported, ExistingHeapsSupported);

// 23: D3D12 Options4
FEATURE_SUPPORT_GET(BOOL, m_dOptions4, MSAA64KBAlignedTextureSupported);
FEATURE_SUPPORT_GET(D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER, m_dOptions4, SharedResourceCompatibilityTier);
FEATURE_SUPPORT_GET(BOOL, m_dOptions4, Native16BitShaderOpsSupported);

// 24: Serialization
FEATURE_SUPPORT_GET_NODE_INDEXED(D3D12_HEAP_SERIALIZATION_TIER, m_dSerialization, HeapSerializationTier);

// 25: Cross Node
// CrossNodeSharingTier handled in D3D12Options
FEATURE_SUPPORT_GET_NAME(BOOL, m_dCrossNode, AtomicShaderInstructions, CrossNodeAtomicShaderInstructions);

// 27: D3D12 Options5
FEATURE_SUPPORT_GET(BOOL, m_dOptions5, SRVOnlyTiledResourceTier3);
FEATURE_SUPPORT_GET(D3D12_RENDER_PASS_TIER, m_dOptions5, RenderPassesTier);
FEATURE_SUPPORT_GET(D3D12_RAYTRACING_TIER, m_dOptions5, RaytracingTier);

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
// 28: Displayable
FEATURE_SUPPORT_GET(BOOL, m_dDisplayable, DisplayableTexture);
// SharedResourceCompatibilityTier handled in D3D12Options4
#endif

// 30: D3D12 Options6
FEATURE_SUPPORT_GET(BOOL, m_dOptions6, AdditionalShadingRatesSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions6, PerPrimitiveShadingRateSupportedWithViewportIndexing);
FEATURE_SUPPORT_GET(D3D12_VARIABLE_SHADING_RATE_TIER, m_dOptions6, VariableShadingRateTier);
FEATURE_SUPPORT_GET(UINT, m_dOptions6, ShadingRateImageTileSize);
FEATURE_SUPPORT_GET(BOOL, m_dOptions6, BackgroundProcessingSupported);

// 31: Query Meta Command
// Keep the original call routine
inline HRESULT CD3DX12FeatureSupport::QueryMetaCommand(D3D12_FEATURE_DATA_QUERY_META_COMMAND& dQueryMetaCommand) const
{
    return m_pDevice->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, &dQueryMetaCommand, sizeof(D3D12_FEATURE_DATA_QUERY_META_COMMAND));
}

// 32: D3D12 Options7
FEATURE_SUPPORT_GET(D3D12_MESH_SHADER_TIER, m_dOptions7, MeshShaderTier);
FEATURE_SUPPORT_GET(D3D12_SAMPLER_FEEDBACK_TIER, m_dOptions7, SamplerFeedbackTier);

// 33: Protected Resource Session Type Count
FEATURE_SUPPORT_GET_NODE_INDEXED_NAME(UINT, m_dProtectedResourceSessionTypeCount, Count, ProtectedResourceSessionTypeCount);

// 34: Protected Resource Session Types
FEATURE_SUPPORT_GET_NODE_INDEXED_NAME(std::vector<GUID>, m_dProtectedResourceSessionTypes, TypeVec, ProtectedResourceSessionTypes);

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
// 36: Options8
FEATURE_SUPPORT_GET(BOOL, m_dOptions8, UnalignedBlockTexturesSupported);

// 37: Options9
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, MeshShaderPipelineStatsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, MeshShaderSupportsFullRangeRenderTargetArrayIndex);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, AtomicInt64OnTypedResourceSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, AtomicInt64OnGroupSharedSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, DerivativesInMeshAndAmplificationShadersSupported);
FEATURE_SUPPORT_GET(D3D12_WAVE_MMA_TIER, m_dOptions9, WaveMMATier);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 4)
// 39: Options10
FEATURE_SUPPORT_GET(BOOL, m_dOptions10, VariableRateShadingSumCombinerSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions10, MeshShaderPerPrimitiveShadingRateSupported);

// 40: Options11
FEATURE_SUPPORT_GET(BOOL, m_dOptions11, AtomicInt64OnDescriptorHeapResourceSupported);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 600)
// 41: Options12
FEATURE_SUPPORT_GET(D3D12_TRI_STATE, m_dOptions12, MSPrimitivesPipelineStatisticIncludesCulledPrimitives);
FEATURE_SUPPORT_GET(BOOL, m_dOptions12, EnhancedBarriersSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions12, RelaxedFormatCastingSupported);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 602)
// 42: Options13
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, UnrestrictedBufferTextureCopyPitchSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, UnrestrictedVertexElementAlignmentSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, InvertedViewportHeightFlipsYSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, InvertedViewportDepthFlipsZSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, TextureCopyBetweenDimensionsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, AlphaBlendFactorSupported);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
// 43: Options14
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, AdvancedTextureOpsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, WriteableMSAATexturesSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, IndependentFrontAndBackStencilRefMaskSupported);

// 44: Options15
FEATURE_SUPPORT_GET(BOOL, m_dOptions15, TriangleFanSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions15, DynamicIndexBufferStripCutSupported);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
// 45: Options16
FEATURE_SUPPORT_GET(BOOL, m_dOptions16, DynamicDepthBiasSupported);
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
FEATURE_SUPPORT_GET(BOOL, m_dOptions16, GPUUploadHeapSupported);

// 46: Options17
FEATURE_SUPPORT_GET(BOOL, m_dOptions17, NonNormalizedCoordinateSamplersSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions17, ManualWriteTrackingResourceSupported);

// 47: Option18
FEATURE_SUPPORT_GET(BOOL, m_dOptions18, RenderPassesValid);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
FEATURE_SUPPORT_GET(BOOL, m_dOptions19, MismatchingOutputDimensionsSupported);
FEATURE_SUPPORT_GET(UINT, m_dOptions19, SupportedSampleCountsWithNoOutputs);
FEATURE_SUPPORT_GET(BOOL, m_dOptions19, PointSamplingAddressesNeverRoundUp);
FEATURE_SUPPORT_GET(BOOL, m_dOptions19, RasterizerDesc2Supported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions19, NarrowQuadrilateralLinesSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions19, AnisoFilterWithPointMipSupported);
FEATURE_SUPPORT_GET(UINT, m_dOptions19, MaxSamplerDescriptorHeapSize);
FEATURE_SUPPORT_GET(UINT, m_dOptions19, MaxSamplerDescriptorHeapSizeWithStaticSamplers);
FEATURE_SUPPORT_GET(UINT, m_dOptions19, MaxViewDescriptorHeapSize);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
// 49: Options20
FEATURE_SUPPORT_GET(BOOL, m_dOptions20, ComputeOnlyWriteWatchSupported);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
// 50: Options21
FEATURE_SUPPORT_GET(D3D12_EXECUTE_INDIRECT_TIER, m_dOptions21, ExecuteIndirectTier);
FEATURE_SUPPORT_GET(D3D12_WORK_GRAPHS_TIER, m_dOptions21, WorkGraphsTier);
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 617)
// 51: TightAlignment
FEATURE_SUPPORT_GET_NAME(D3D12_TIGHT_ALIGNMENT_TIER, m_dTightAlignment, SupportTier, TightAlignmentSupportTier);
#endif

// Helper function to decide the highest shader model supported by the system
// Stores the result in m_dShaderModel
// Must be updated whenever a new shader model is added to the d3d12.h header
inline HRESULT CD3DX12FeatureSupport::QueryHighestShaderModel()
{
    // Check support in descending order
    HRESULT result;

    const D3D_SHADER_MODEL allModelVersions[] =
    {
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 612)
        D3D_SHADER_MODEL_6_9,
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
        D3D_SHADER_MODEL_6_8,
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
        D3D_SHADER_MODEL_6_7,
#endif
        D3D_SHADER_MODEL_6_6,
        D3D_SHADER_MODEL_6_5,
        D3D_SHADER_MODEL_6_4,
        D3D_SHADER_MODEL_6_3,
        D3D_SHADER_MODEL_6_2,
        D3D_SHADER_MODEL_6_1,
        D3D_SHADER_MODEL_6_0,
        D3D_SHADER_MODEL_5_1
    };
    constexpr size_t numModelVersions = sizeof(allModelVersions) / sizeof(D3D_SHADER_MODEL);

    for (size_t i = 0; i < numModelVersions; i++)
    {
        m_dShaderModel.HighestShaderModel = allModelVersions[i];
        result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &m_dShaderModel, sizeof(D3D12_FEATURE_DATA_SHADER_MODEL));
        if (result != E_INVALIDARG)
        {
            // Indicates that the version is recognizable by the runtime and stored in the struct
            // Also terminate on unexpected error code
            if (FAILED(result))
            {
                m_dShaderModel.HighestShaderModel = static_cast<D3D_SHADER_MODEL>(0);
            }
            return result;
        }
    }

    // Shader model may not be supported. Continue the rest initializations
    m_dShaderModel.HighestShaderModel = static_cast<D3D_SHADER_MODEL>(0);
    return S_OK;
}

// Helper function to decide the highest root signature supported
// Must be updated whenever a new root signature version is added to the d3d12.h header
inline HRESULT CD3DX12FeatureSupport::QueryHighestRootSignatureVersion()
{
    HRESULT result;

    const D3D_ROOT_SIGNATURE_VERSION allRootSignatureVersions[] =
    {
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
        D3D_ROOT_SIGNATURE_VERSION_1_2,
#endif
        D3D_ROOT_SIGNATURE_VERSION_1_1,
        D3D_ROOT_SIGNATURE_VERSION_1_0,
        D3D_ROOT_SIGNATURE_VERSION_1,
    };
    constexpr size_t numRootSignatureVersions = sizeof(allRootSignatureVersions) / sizeof(D3D_ROOT_SIGNATURE_VERSION);

    for (size_t i = 0; i < numRootSignatureVersions; i++)
    {
        m_dRootSignature.HighestVersion = allRootSignatureVersions[i];
        result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &m_dRootSignature, sizeof(D3D12_FEATURE_DATA_ROOT_SIGNATURE));
        if (result != E_INVALIDARG)
        {
            if (FAILED(result))
            {
                m_dRootSignature.HighestVersion = static_cast<D3D_ROOT_SIGNATURE_VERSION>(0);
            }
            // If succeeded, the highest version is already written into the member struct
            return result;
        }
    }

    // No version left. Set to invalid value and continue.
    m_dRootSignature.HighestVersion = static_cast<D3D_ROOT_SIGNATURE_VERSION>(0);
    return S_OK;
}

// Helper funcion to decide the highest feature level
inline HRESULT CD3DX12FeatureSupport::QueryHighestFeatureLevel()
{
    HRESULT result;

    // Check against a list of all feature levels present in d3dcommon.h
    // Needs to be updated for future feature levels
    const D3D_FEATURE_LEVEL allLevels[] =
    {
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 3)
        D3D_FEATURE_LEVEL_12_2,
#endif
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1,
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 5)
        D3D_FEATURE_LEVEL_1_0_CORE,
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 611)
        D3D_FEATURE_LEVEL_1_0_GENERIC
#endif
    };

    D3D12_FEATURE_DATA_FEATURE_LEVELS dFeatureLevel;
    dFeatureLevel.NumFeatureLevels = static_cast<UINT>(sizeof(allLevels) / sizeof(D3D_FEATURE_LEVEL));
    dFeatureLevel.pFeatureLevelsRequested = allLevels;

    result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &dFeatureLevel, sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS));
    if (SUCCEEDED(result))
    {
        m_eMaxFeatureLevel = dFeatureLevel.MaxSupportedFeatureLevel;
    }
    else
    {
        m_eMaxFeatureLevel = static_cast<D3D_FEATURE_LEVEL>(0);

        if (result == DXGI_ERROR_UNSUPPORTED)
        {
            // Indicates that none supported. Continue initialization
            result = S_OK;
        }
    }
    return result;
}

// Helper function to initialize local protected resource session types structs
inline HRESULT CD3DX12FeatureSupport::QueryProtectedResourceSessionTypes(UINT NodeIndex, UINT Count)
{
    auto& CurrentPRSTypes = m_dProtectedResourceSessionTypes[NodeIndex];
    CurrentPRSTypes.NodeIndex = NodeIndex;
    CurrentPRSTypes.Count = Count;
    CurrentPRSTypes.TypeVec.resize(CurrentPRSTypes.Count);
    CurrentPRSTypes.pTypes = CurrentPRSTypes.TypeVec.data();

    HRESULT result = m_pDevice->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES, &m_dProtectedResourceSessionTypes[NodeIndex], sizeof(D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES));
    if (FAILED(result))
    {
        // Resize TypeVec to empty
        CurrentPRSTypes.TypeVec.clear();
    }

    return result;
}

#undef FEATURE_SUPPORT_GET
#undef FEATURE_SUPPORT_GET_NAME
#undef FEATURE_SUPPORT_GET_NODE_INDEXED
#undef FEATURE_SUPPORT_GET_NODE_INDEXED_NAME

// end CD3DX12FeatureSupport


