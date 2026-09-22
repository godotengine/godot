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

#include "d3dx12_default.h"
#include "d3d12.h"
#include "d3dx12_core.h"

//------------------------------------------------------------------------------------------------
// Pipeline State Stream Helpers
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// Stream Subobjects, i.e. elements of a stream

struct DefaultSampleMask { operator UINT() noexcept { return UINT_MAX; } };
struct DefaultSampleDesc { operator DXGI_SAMPLE_DESC() noexcept { return DXGI_SAMPLE_DESC{1, 0}; } };

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
template <typename InnerStructType, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename DefaultArg = InnerStructType>
class alignas(void*) CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT
{
private:
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE pssType;
    InnerStructType pssInner;
public:
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT() noexcept : pssType(Type), pssInner(DefaultArg()) {}
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT(InnerStructType const& i) noexcept : pssType(Type), pssInner(i) {}
    CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT& operator=(InnerStructType const& i) noexcept { pssType = Type; pssInner = i; return *this; }
    operator InnerStructType const&() const noexcept { return pssInner; }
    operator InnerStructType&() noexcept { return pssInner; }
    InnerStructType* operator&() noexcept { return &pssInner; }
    InnerStructType const* operator&() const noexcept { return &pssInner; }
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_PIPELINE_STATE_FLAGS,         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS>                             CD3DX12_PIPELINE_STATE_STREAM_FLAGS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< UINT,                               D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK>                         CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< ID3D12RootSignature*,               D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE>                    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_INPUT_LAYOUT_DESC,            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT>                      CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_INDEX_BUFFER_STRIP_CUT_VALUE, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE>                CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_PRIMITIVE_TOPOLOGY_TYPE,      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY>                CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>                                CD3DX12_PIPELINE_STATE_STREAM_VS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS>                                CD3DX12_PIPELINE_STATE_STREAM_GS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_STREAM_OUTPUT_DESC,           D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT>                     CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS>                                CD3DX12_PIPELINE_STATE_STREAM_HS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS>                                CD3DX12_PIPELINE_STATE_STREAM_DS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS>                                CD3DX12_PIPELINE_STATE_STREAM_PS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS>                                CD3DX12_PIPELINE_STATE_STREAM_AS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>                                CD3DX12_PIPELINE_STATE_STREAM_MS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_SHADER_BYTECODE,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS>                                CD3DX12_PIPELINE_STATE_STREAM_CS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_BLEND_DESC,                 D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,          CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_DEPTH_STENCIL_DESC,         D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,  CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_DEPTH_STENCIL_DESC1,        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_DEPTH_STENCIL_DESC2,        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2;
#endif
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< DXGI_FORMAT,                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT>              CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_RASTERIZER_DESC,            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,     CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_RASTERIZER_DESC1,           D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1,    CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER1;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_RASTERIZER_DESC2,           D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2,    CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER2;
#endif
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_RT_FORMAT_ARRAY,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS>             CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< DXGI_SAMPLE_DESC,                   D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,    DefaultSampleDesc> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< UINT,                               D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,    DefaultSampleMask> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_CACHED_PIPELINE_STATE,        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO>                        CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_VIEW_INSTANCING_DESC,       D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, CD3DX12_DEFAULT>  CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 618)
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT<CD3DX12_SERIALIZED_ROOT_SIGNATURE_DESC, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SERIALIZED_ROOT_SIGNATURE>      CD3DX12_PIPELINE_STATE_STREAM_SERIALIZED_ROOT_SIGNATURE;
#endif

//------------------------------------------------------------------------------------------------
// Stream Parser Helpers

struct ID3DX12PipelineParserCallbacks
{
    // Subobject Callbacks
    virtual void FlagsCb(D3D12_PIPELINE_STATE_FLAGS) {}
    virtual void NodeMaskCb(UINT) {}
    virtual void RootSignatureCb(ID3D12RootSignature*) {}
    virtual void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC&) {}
    virtual void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE) {}
    virtual void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE) {}
    virtual void VSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void GSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC&) {}
    virtual void HSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void DSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void PSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void CSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void ASCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void MSCb(const D3D12_SHADER_BYTECODE&) {}
    virtual void BlendStateCb(const D3D12_BLEND_DESC&) {}
    virtual void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC&) {}
    virtual void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1&) {}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
    virtual void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2&) {}
#endif
    virtual void DSVFormatCb(DXGI_FORMAT) {}
    virtual void RasterizerStateCb(const D3D12_RASTERIZER_DESC&) {}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
    virtual void RasterizerState1Cb(const D3D12_RASTERIZER_DESC1&) {}
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
    virtual void RasterizerState2Cb(const D3D12_RASTERIZER_DESC2&) {}
#endif
    virtual void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY&) {}
    virtual void SampleDescCb(const DXGI_SAMPLE_DESC&) {}
    virtual void SampleMaskCb(UINT) {}
    virtual void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC&) {}
    virtual void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE&) {}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 618)
    virtual void SerializedRootSignatureCb(const D3D12_SERIALIZED_ROOT_SIGNATURE_DESC&) {}
#endif

    // Error Callbacks
    virtual void ErrorBadInputParameter(UINT /*ParameterIndex*/) {}
    virtual void ErrorDuplicateSubobject(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE /*DuplicateType*/) {}
    virtual void ErrorUnknownSubobject(UINT /*UnknownTypeValue*/) {}
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 613)
    virtual void FinalizeCb() {}
#endif

    virtual ~ID3DX12PipelineParserCallbacks() = default;
};

struct D3DX12_MESH_SHADER_PIPELINE_STATE_DESC
{
    ID3D12RootSignature*          pRootSignature;
    D3D12_SHADER_BYTECODE         AS;
    D3D12_SHADER_BYTECODE         MS;
    D3D12_SHADER_BYTECODE         PS;
    D3D12_BLEND_DESC              BlendState;
    UINT                          SampleMask;
    D3D12_RASTERIZER_DESC         RasterizerState;
    D3D12_DEPTH_STENCIL_DESC      DepthStencilState;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType;
    UINT                          NumRenderTargets;
    DXGI_FORMAT                   RTVFormats[ D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT ];
    DXGI_FORMAT                   DSVFormat;
    DXGI_SAMPLE_DESC              SampleDesc;
    UINT                          NodeMask;
    D3D12_CACHED_PIPELINE_STATE   CachedPSO;
    D3D12_PIPELINE_STATE_FLAGS    Flags;
};

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 618)
struct CD3DX12_PIPELINE_STATE_STREAM6
{
    CD3DX12_PIPELINE_STATE_STREAM6() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM6(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC2(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
        , SerializedRootSignature(CD3DX12_SERIALIZED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT()))
    {
    }
    CD3DX12_PIPELINE_STATE_STREAM6(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC2(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
        , SerializedRootSignature(CD3DX12_SERIALIZED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT()))
    {
    }
    CD3DX12_PIPELINE_STATE_STREAM6(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
        , SerializedRootSignature(CD3DX12_SERIALIZED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT()))
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER2 RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SERIALIZED_ROOT_SIGNATURE SerializedRootSignature;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.InputLayout = this->InputLayout;
        D.IBStripCutValue = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS = this->VS;
        D.GS = this->GS;
        D.StreamOutput = this->StreamOutput;
        D.HS = this->HS;
        D.DS = this->DS;
        D.PS = this->PS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(D3D12_DEPTH_STENCIL_DESC2(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = CD3DX12_RASTERIZER_DESC2(D3D12_RASTERIZER_DESC2(this->RasterizerState));
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.CS = this->CS;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};
#endif

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
// Use CD3DX12_PIPELINE_STATE_STREAM5 for D3D12_RASTERIZER_DESC2 when CheckFeatureSupport returns true for Options19::RasterizerDesc2Supported is true
// Use CD3DX12_PIPELINE_STATE_STREAM4 for D3D12_RASTERIZER_DESC1 when CheckFeatureSupport returns true for Options16::DynamicDepthBiasSupported is true
// Use CD3DX12_PIPELINE_STATE_STREAM3 for D3D12_DEPTH_STENCIL_DESC2 when CheckFeatureSupport returns true for Options14::IndependentFrontAndBackStencilSupported is true
// Use CD3DX12_PIPELINE_STATE_STREAM2 for OS Build 19041+ (where there is a new mesh shader pipeline).
// Use CD3DX12_PIPELINE_STATE_STREAM1 for OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM5
{
    CD3DX12_PIPELINE_STATE_STREAM5() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM5(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC2(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM5(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC2(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM5(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER2 RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC2(D3D12_DEPTH_STENCIL_DESC2(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = CD3DX12_RASTERIZER_DESC2(D3D12_RASTERIZER_DESC2(this->RasterizerState));
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};
#endif // D3D12_SDK_VERSION >= 610

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
// Use CD3DX12_PIPELINE_STATE_STREAM4 for D3D12_RASTERIZER_DESC1 when CheckFeatureSupport returns true for Options16::DynamicDepthBiasSupported is true
// Use CD3DX12_PIPELINE_STATE_STREAM3 for D3D12_DEPTH_STENCIL_DESC2 when CheckFeatureSupport returns true for Options14::IndependentFrontAndBackStencilSupported is true
// Use CD3DX12_PIPELINE_STATE_STREAM2 for OS Build 19041+ (where there is a new mesh shader pipeline).
// Use CD3DX12_PIPELINE_STATE_STREAM1 for OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM4
{
    CD3DX12_PIPELINE_STATE_STREAM4() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM4(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC1(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM4(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC1(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM4(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER1 RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC2(D3D12_DEPTH_STENCIL_DESC2(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = CD3DX12_RASTERIZER_DESC1(D3D12_RASTERIZER_DESC1(this->RasterizerState));
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};
#endif // D3D12_SDK_VERSION >= 608

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
// Use CD3DX12_PIPELINE_STATE_STREAM3 for D3D12_DEPTH_STENCIL_DESC2 when CheckFeatureSupport returns true for Options14::IndependentFrontAndBackStencilSupported is true
// Use CD3DX12_PIPELINE_STATE_STREAM2 for OS Build 19041+ (where there is a new mesh shader pipeline).
// Use CD3DX12_PIPELINE_STATE_STREAM1 for OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM3
{
    CD3DX12_PIPELINE_STATE_STREAM3() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM3(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM3(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC2(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM3(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC2(D3D12_DEPTH_STENCIL_DESC2(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = this->RasterizerState;
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};
#endif // D3D12_SDK_VERSION >= 606

// CD3DX12_PIPELINE_STATE_STREAM2 Works on OS Build 19041+ (where there is a new mesh shader pipeline).
// Use CD3DX12_PIPELINE_STATE_STREAM1 for OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM2
{
    CD3DX12_PIPELINE_STATE_STREAM2() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM2(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM2(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM2(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC1&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = this->RasterizerState;
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};

// CD3DX12_PIPELINE_STATE_STREAM1 Works on OS Build 16299+ (where there is a new view instancing subobject).
// Use CD3DX12_PIPELINE_STATE_STREAM for OS Build 15063+ support.
struct CD3DX12_PIPELINE_STATE_STREAM1
{
    CD3DX12_PIPELINE_STATE_STREAM1() = default;
    // Mesh and amplification shaders must be set manually, since they do not have representation in D3D12_GRAPHICS_PIPELINE_STATE_DESC
    CD3DX12_PIPELINE_STATE_STREAM1(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM1(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM1(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {
        static_cast<D3D12_DEPTH_STENCIL_DESC1&>(DepthStencilState).DepthEnable = false;
    }
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = this->RasterizerState;
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};


struct CD3DX12_PIPELINE_MESH_STATE_STREAM
{
    CD3DX12_PIPELINE_MESH_STATE_STREAM() = default;
    CD3DX12_PIPELINE_MESH_STATE_STREAM(const D3DX12_MESH_SHADER_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , PS(Desc.PS)
        , AS(Desc.AS)
        , MS(Desc.MS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
        , ViewInstancingDesc(CD3DX12_VIEW_INSTANCING_DESC(CD3DX12_DEFAULT()))
    {}
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_AS AS;
    CD3DX12_PIPELINE_STATE_STREAM_MS MS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING ViewInstancingDesc;
    D3DX12_MESH_SHADER_PIPELINE_STATE_DESC MeshShaderDescV0() const noexcept
    {
        D3DX12_MESH_SHADER_PIPELINE_STATE_DESC D;
        D.Flags = this->Flags;
        D.NodeMask = this->NodeMask;
        D.pRootSignature = this->pRootSignature;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.PS = this->PS;
        D.AS = this->AS;
        D.MS = this->MS;
        D.BlendState = this->BlendState;
        D.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat = this->DSVFormat;
        D.RasterizerState = this->RasterizerState;
        D.NumRenderTargets = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc = this->SampleDesc;
        D.SampleMask = this->SampleMask;
        D.CachedPSO = this->CachedPSO;
        return D;
    }
};

// CD3DX12_PIPELINE_STATE_STREAM works on OS Build 15063+ but does not support new subobject(s) added in OS Build 16299+.
// See CD3DX12_PIPELINE_STATE_STREAM1 for instance.
struct CD3DX12_PIPELINE_STATE_STREAM
{
    CD3DX12_PIPELINE_STATE_STREAM() = default;
    CD3DX12_PIPELINE_STATE_STREAM(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , InputLayout(Desc.InputLayout)
        , IBStripCutValue(Desc.IBStripCutValue)
        , PrimitiveTopologyType(Desc.PrimitiveTopologyType)
        , VS(Desc.VS)
        , GS(Desc.GS)
        , StreamOutput(Desc.StreamOutput)
        , HS(Desc.HS)
        , DS(Desc.DS)
        , PS(Desc.PS)
        , BlendState(CD3DX12_BLEND_DESC(Desc.BlendState))
        , DepthStencilState(CD3DX12_DEPTH_STENCIL_DESC1(Desc.DepthStencilState))
        , DSVFormat(Desc.DSVFormat)
        , RasterizerState(CD3DX12_RASTERIZER_DESC(Desc.RasterizerState))
        , RTVFormats(CD3DX12_RT_FORMAT_ARRAY(Desc.RTVFormats, Desc.NumRenderTargets))
        , SampleDesc(Desc.SampleDesc)
        , SampleMask(Desc.SampleMask)
        , CachedPSO(Desc.CachedPSO)
    {}
    CD3DX12_PIPELINE_STATE_STREAM(const D3D12_COMPUTE_PIPELINE_STATE_DESC& Desc) noexcept
        : Flags(Desc.Flags)
        , NodeMask(Desc.NodeMask)
        , pRootSignature(Desc.pRootSignature)
        , CS(CD3DX12_SHADER_BYTECODE(Desc.CS))
        , CachedPSO(Desc.CachedPSO)
    {}
    CD3DX12_PIPELINE_STATE_STREAM_FLAGS Flags;
    CD3DX12_PIPELINE_STATE_STREAM_NODE_MASK NodeMask;
    CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE pRootSignature;
    CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT InputLayout;
    CD3DX12_PIPELINE_STATE_STREAM_IB_STRIP_CUT_VALUE IBStripCutValue;
    CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY PrimitiveTopologyType;
    CD3DX12_PIPELINE_STATE_STREAM_VS VS;
    CD3DX12_PIPELINE_STATE_STREAM_GS GS;
    CD3DX12_PIPELINE_STATE_STREAM_STREAM_OUTPUT StreamOutput;
    CD3DX12_PIPELINE_STATE_STREAM_HS HS;
    CD3DX12_PIPELINE_STATE_STREAM_DS DS;
    CD3DX12_PIPELINE_STATE_STREAM_PS PS;
    CD3DX12_PIPELINE_STATE_STREAM_CS CS;
    CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC BlendState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL1 DepthStencilState;
    CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT DSVFormat;
    CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER RasterizerState;
    CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC SampleDesc;
    CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK SampleMask;
    CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO CachedPSO;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC GraphicsDescV0() const noexcept
    {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.InputLayout           = this->InputLayout;
        D.IBStripCutValue       = this->IBStripCutValue;
        D.PrimitiveTopologyType = this->PrimitiveTopologyType;
        D.VS                    = this->VS;
        D.GS                    = this->GS;
        D.StreamOutput          = this->StreamOutput;
        D.HS                    = this->HS;
        D.DS                    = this->DS;
        D.PS                    = this->PS;
        D.BlendState            = this->BlendState;
        D.DepthStencilState     = CD3DX12_DEPTH_STENCIL_DESC1(D3D12_DEPTH_STENCIL_DESC1(this->DepthStencilState));
        D.DSVFormat             = this->DSVFormat;
        D.RasterizerState       = this->RasterizerState;
        D.NumRenderTargets      = D3D12_RT_FORMAT_ARRAY(this->RTVFormats).NumRenderTargets;
        memcpy(D.RTVFormats, D3D12_RT_FORMAT_ARRAY(this->RTVFormats).RTFormats, sizeof(D.RTVFormats));
        D.SampleDesc            = this->SampleDesc;
        D.SampleMask            = this->SampleMask;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC ComputeDescV0() const noexcept
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC D;
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.CS                    = this->CS;
        D.CachedPSO             = this->CachedPSO;
        return D;
    }
};


struct CD3DX12_PIPELINE_STATE_STREAM2_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM2 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM2_PARSE_HELPER() noexcept
        : SeenDSS(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC1&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override {PipelineStream.Flags = Flags;}
    void NodeMaskCb(UINT NodeMask) override {PipelineStream.NodeMask = NodeMask;}
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override {PipelineStream.pRootSignature = pRootSignature;}
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override {PipelineStream.InputLayout = InputLayout;}
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override {PipelineStream.IBStripCutValue = IBStripCutValue;}
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override {PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType;}
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override {PipelineStream.VS = VS;}
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override {PipelineStream.GS = GS;}
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override {PipelineStream.StreamOutput = StreamOutput;}
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override {PipelineStream.HS = HS;}
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override {PipelineStream.DS = DS;}
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override {PipelineStream.PS = PS;}
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override {PipelineStream.CS = CS;}
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override {PipelineStream.AS = AS;}
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override {PipelineStream.MS = MS;}
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override {PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState);}
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override
    {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC1&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override {PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC(RasterizerState);}
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override {PipelineStream.RTVFormats = RTVFormats;}
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override {PipelineStream.SampleDesc = SampleDesc;}
    void SampleMaskCb(UINT SampleMask) override {PipelineStream.SampleMask = SampleMask;}
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override {PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc);}
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override {PipelineStream.CachedPSO = CachedPSO;}

private:
    bool SeenDSS;
};

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
struct CD3DX12_PIPELINE_STATE_STREAM3_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM3 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM3_PARSE_HELPER() noexcept
        : SeenDSS(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(UINT NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override { PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType; }
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override { PipelineStream.MS = MS; }
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override
    {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(UINT SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }

private:
    bool SeenDSS;
};
#endif // D3D12_SDK_VERSION >= 606

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
struct CD3DX12_PIPELINE_STATE_STREAM4_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM4 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM4_PARSE_HELPER() noexcept
        : SeenDSS(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(UINT NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override { PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType; }
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override { PipelineStream.MS = MS; }
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override
    {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC1(RasterizerState); }
    void RasterizerState1Cb(const D3D12_RASTERIZER_DESC1& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC1(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(UINT SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }

private:
    bool SeenDSS;
};
#endif // D3D12_SDK_VERSION >= 608

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 613)
// This SDK 613 version has better primitive topology default handling than the v610 equivalent below.
struct CD3DX12_PIPELINE_STATE_STREAM5_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM5 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM5_PARSE_HELPER() noexcept
        : SeenDSS(false),
        SeenMS(false),
        SeenTopology(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(UINT NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override
    {
        PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType;
        SeenTopology = true;
    }
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override { PipelineStream.MS = MS; SeenMS = true; }
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override {PipelineStream.DSVFormat = DSVFormat;}
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState1Cb(const D3D12_RASTERIZER_DESC1& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState2Cb(const D3D12_RASTERIZER_DESC2& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(UINT SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }
    void FinalizeCb() override
    {
        if (!SeenDSS && PipelineStream.DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
        if (!SeenTopology && SeenMS)
        {
            PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
        }
    }

private:
    bool SeenDSS;
    bool SeenMS;
    bool SeenTopology;
};
#elif defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
struct CD3DX12_PIPELINE_STATE_STREAM5_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM5 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM5_PARSE_HELPER() noexcept
        : SeenDSS(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(UINT NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override { PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType; }
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override { PipelineStream.MS = MS; }
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override
    {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState1Cb(const D3D12_RASTERIZER_DESC1& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState2Cb(const D3D12_RASTERIZER_DESC2& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(UINT SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }

private:
    bool SeenDSS;
};
#endif // D3D12_SDK_VERSION >= 610

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 618)
struct CD3DX12_PIPELINE_STATE_STREAM6_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM6 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM6_PARSE_HELPER() noexcept
        : SeenDSS(false),
        SeenMS(false),
        SeenTopology(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override { PipelineStream.Flags = Flags; }
    void NodeMaskCb(UINT NodeMask) override { PipelineStream.NodeMask = NodeMask; }
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override { PipelineStream.pRootSignature = pRootSignature; }
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override { PipelineStream.InputLayout = InputLayout; }
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override { PipelineStream.IBStripCutValue = IBStripCutValue; }
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override
    {
        PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType;
        SeenTopology = true;
    }
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override { PipelineStream.VS = VS; }
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override { PipelineStream.GS = GS; }
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override { PipelineStream.StreamOutput = StreamOutput; }
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override { PipelineStream.HS = HS; }
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override { PipelineStream.DS = DS; }
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override { PipelineStream.PS = PS; }
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override { PipelineStream.CS = CS; }
    void ASCb(const D3D12_SHADER_BYTECODE& AS) override { PipelineStream.AS = AS; }
    void MSCb(const D3D12_SHADER_BYTECODE& MS) override { PipelineStream.MS = MS; SeenMS = true; }
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override { PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState); }
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC2(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override { PipelineStream.DSVFormat = DSVFormat; }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState1Cb(const D3D12_RASTERIZER_DESC1& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RasterizerState2Cb(const D3D12_RASTERIZER_DESC2& RasterizerState) override { PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC2(RasterizerState); }
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override { PipelineStream.RTVFormats = RTVFormats; }
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override { PipelineStream.SampleDesc = SampleDesc; }
    void SampleMaskCb(UINT SampleMask) override { PipelineStream.SampleMask = SampleMask; }
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override { PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc); }
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override { PipelineStream.CachedPSO = CachedPSO; }
    void FinalizeCb() override
    {
        if (!SeenDSS && PipelineStream.DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC2&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
        if (!SeenTopology && SeenMS)
        {
            PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
        }
    }
    void SerializedRootSignatureCb(const D3D12_SERIALIZED_ROOT_SIGNATURE_DESC& SerializedRootSignature) override { PipelineStream.SerializedRootSignature = CD3DX12_SERIALIZED_ROOT_SIGNATURE_DESC(SerializedRootSignature); }

private:
    bool SeenDSS;
    bool SeenMS;
    bool SeenTopology;
};
#endif // D3D12_SDK_VERSION >= 618

struct CD3DX12_PIPELINE_STATE_STREAM_PARSE_HELPER : public ID3DX12PipelineParserCallbacks
{
    CD3DX12_PIPELINE_STATE_STREAM1 PipelineStream;
    CD3DX12_PIPELINE_STATE_STREAM_PARSE_HELPER() noexcept
        : SeenDSS(false)
    {
        // Adjust defaults to account for absent members.
        PipelineStream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

        // Depth disabled if no DSV format specified.
        static_cast<D3D12_DEPTH_STENCIL_DESC1&>(PipelineStream.DepthStencilState).DepthEnable = false;
    }

    // ID3DX12PipelineParserCallbacks
    void FlagsCb(D3D12_PIPELINE_STATE_FLAGS Flags) override {PipelineStream.Flags = Flags;}
    void NodeMaskCb(UINT NodeMask) override {PipelineStream.NodeMask = NodeMask;}
    void RootSignatureCb(ID3D12RootSignature* pRootSignature) override {PipelineStream.pRootSignature = pRootSignature;}
    void InputLayoutCb(const D3D12_INPUT_LAYOUT_DESC& InputLayout) override {PipelineStream.InputLayout = InputLayout;}
    void IBStripCutValueCb(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE IBStripCutValue) override {PipelineStream.IBStripCutValue = IBStripCutValue;}
    void PrimitiveTopologyTypeCb(D3D12_PRIMITIVE_TOPOLOGY_TYPE PrimitiveTopologyType) override {PipelineStream.PrimitiveTopologyType = PrimitiveTopologyType;}
    void VSCb(const D3D12_SHADER_BYTECODE& VS) override {PipelineStream.VS = VS;}
    void GSCb(const D3D12_SHADER_BYTECODE& GS) override {PipelineStream.GS = GS;}
    void StreamOutputCb(const D3D12_STREAM_OUTPUT_DESC& StreamOutput) override {PipelineStream.StreamOutput = StreamOutput;}
    void HSCb(const D3D12_SHADER_BYTECODE& HS) override {PipelineStream.HS = HS;}
    void DSCb(const D3D12_SHADER_BYTECODE& DS) override {PipelineStream.DS = DS;}
    void PSCb(const D3D12_SHADER_BYTECODE& PS) override {PipelineStream.PS = PS;}
    void CSCb(const D3D12_SHADER_BYTECODE& CS) override {PipelineStream.CS = CS;}
    void BlendStateCb(const D3D12_BLEND_DESC& BlendState) override {PipelineStream.BlendState = CD3DX12_BLEND_DESC(BlendState);}
    void DepthStencilStateCb(const D3D12_DEPTH_STENCIL_DESC& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DepthStencilState1Cb(const D3D12_DEPTH_STENCIL_DESC1& DepthStencilState) override
    {
        PipelineStream.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC1(DepthStencilState);
        SeenDSS = true;
    }
    void DSVFormatCb(DXGI_FORMAT DSVFormat) override
    {
        PipelineStream.DSVFormat = DSVFormat;
        if (!SeenDSS && DSVFormat != DXGI_FORMAT_UNKNOWN)
        {
            // Re-enable depth for the default state.
            static_cast<D3D12_DEPTH_STENCIL_DESC1&>(PipelineStream.DepthStencilState).DepthEnable = true;
        }
    }
    void RasterizerStateCb(const D3D12_RASTERIZER_DESC& RasterizerState) override {PipelineStream.RasterizerState = CD3DX12_RASTERIZER_DESC(RasterizerState);}
    void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY& RTVFormats) override {PipelineStream.RTVFormats = RTVFormats;}
    void SampleDescCb(const DXGI_SAMPLE_DESC& SampleDesc) override {PipelineStream.SampleDesc = SampleDesc;}
    void SampleMaskCb(UINT SampleMask) override {PipelineStream.SampleMask = SampleMask;}
    void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC& ViewInstancingDesc) override {PipelineStream.ViewInstancingDesc = CD3DX12_VIEW_INSTANCING_DESC(ViewInstancingDesc);}
    void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE& CachedPSO) override {PipelineStream.CachedPSO = CachedPSO;}

private:
    bool SeenDSS;
};


inline D3D12_PIPELINE_STATE_SUBOBJECT_TYPE D3DX12GetBaseSubobjectType(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE SubobjectType) noexcept
{
    switch (SubobjectType)
    {
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
        return D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2:
        return D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1:
        return D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER;
#endif
    default:
        return SubobjectType;
    }
}

inline HRESULT D3DX12ParsePipelineStream(const D3D12_PIPELINE_STATE_STREAM_DESC& Desc, ID3DX12PipelineParserCallbacks* pCallbacks)
{
    if (pCallbacks == nullptr)
    {
        return E_INVALIDARG;
    }

    if (Desc.SizeInBytes == 0 || Desc.pPipelineStateSubobjectStream == nullptr)
    {
        pCallbacks->ErrorBadInputParameter(1); // first parameter issue
        return E_INVALIDARG;
    }

    bool SubobjectSeen[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MAX_VALID] = {};
    for (SIZE_T CurOffset = 0, SizeOfSubobject = 0; CurOffset < Desc.SizeInBytes; CurOffset += SizeOfSubobject)
    {
        BYTE* pStream = static_cast<BYTE*>(Desc.pPipelineStateSubobjectStream)+CurOffset;
        auto SubobjectType = *reinterpret_cast<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE*>(pStream);
        if (SubobjectType < 0 || SubobjectType >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MAX_VALID)
        {
            pCallbacks->ErrorUnknownSubobject(SubobjectType);
            return E_INVALIDARG;
        }
        if (SubobjectSeen[D3DX12GetBaseSubobjectType(SubobjectType)])
        {
            pCallbacks->ErrorDuplicateSubobject(SubobjectType);
            return E_INVALIDARG; // disallow subobject duplicates in a stream
        }
        SubobjectSeen[SubobjectType] = true;
        switch (SubobjectType)
        {
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
            pCallbacks->RootSignatureCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::pRootSignature)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::pRootSignature);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS:
            pCallbacks->VSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::VS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::VS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS:
            pCallbacks->PSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::PS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::PS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS:
            pCallbacks->DSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS:
            pCallbacks->HSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::HS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::HS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS:
            pCallbacks->GSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::GS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::GS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
            pCallbacks->CSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::CS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::CS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS:
            pCallbacks->ASCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM2::AS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM2::AS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS:
            pCallbacks->MSCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM2::MS)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM2::MS);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT:
            pCallbacks->StreamOutputCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::StreamOutput)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::StreamOutput);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND:
            pCallbacks->BlendStateCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::BlendState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::BlendState);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK:
            pCallbacks->SampleMaskCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::SampleMask)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::SampleMask);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER:
            pCallbacks->RasterizerStateCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::RasterizerState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::RasterizerState);
            break;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER1:
            pCallbacks->RasterizerState1Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM4::RasterizerState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM4::RasterizerState);
            break;
#endif
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER2:
            pCallbacks->RasterizerState2Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM5::RasterizerState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM5::RasterizerState);
            break;
#endif
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL:
            pCallbacks->DepthStencilStateCb(*reinterpret_cast<CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
            pCallbacks->DepthStencilState1Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState);
            break;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2:
            pCallbacks->DepthStencilState2Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM3::DepthStencilState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM3::DepthStencilState);
            break;
#endif
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT:
            pCallbacks->InputLayoutCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::InputLayout)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::InputLayout);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE:
            pCallbacks->IBStripCutValueCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::IBStripCutValue)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::IBStripCutValue);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY:
            pCallbacks->PrimitiveTopologyTypeCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::PrimitiveTopologyType)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::PrimitiveTopologyType);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS:
            pCallbacks->RTVFormatsCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::RTVFormats)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::RTVFormats);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT:
            pCallbacks->DSVFormatCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DSVFormat)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DSVFormat);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC:
            pCallbacks->SampleDescCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::SampleDesc)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::SampleDesc);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK:
            pCallbacks->NodeMaskCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::NodeMask)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::NodeMask);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO:
            pCallbacks->CachedPSOCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::CachedPSO)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::CachedPSO);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS:
            pCallbacks->FlagsCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::Flags)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::Flags);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING:
            pCallbacks->ViewInstancingCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM1::ViewInstancingDesc)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM1::ViewInstancingDesc);
            break;
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 618)
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SERIALIZED_ROOT_SIGNATURE:
            pCallbacks->SerializedRootSignatureCb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM6::SerializedRootSignature)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM6::SerializedRootSignature);
            break;
#endif
        default:
            pCallbacks->ErrorUnknownSubobject(SubobjectType);
            return E_INVALIDARG;
        }
    }
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 613)
    pCallbacks->FinalizeCb();
#endif

    return S_OK;
}

