//*********************************************************
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License (MIT).
//
//*********************************************************

#ifndef __D3DX12_H__
#define __D3DX12_H__

#include "d3d12.h"

#if defined( __cplusplus )

struct CD3DX12_DEFAULT {};
extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;

//------------------------------------------------------------------------------------------------
inline bool operator==( const D3D12_VIEWPORT& l, const D3D12_VIEWPORT& r ) noexcept
{
    return l.TopLeftX == r.TopLeftX && l.TopLeftY == r.TopLeftY && l.Width == r.Width &&
        l.Height == r.Height && l.MinDepth == r.MinDepth && l.MaxDepth == r.MaxDepth;
}

//------------------------------------------------------------------------------------------------
inline bool operator!=( const D3D12_VIEWPORT& l, const D3D12_VIEWPORT& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_RECT : public D3D12_RECT
{
    CD3DX12_RECT() = default;
    explicit CD3DX12_RECT( const D3D12_RECT& o ) noexcept :
        D3D12_RECT( o )
    {}
    explicit CD3DX12_RECT(
        LONG Left,
        LONG Top,
        LONG Right,
        LONG Bottom ) noexcept
    {
        left = Left;
        top = Top;
        right = Right;
        bottom = Bottom;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_VIEWPORT : public D3D12_VIEWPORT
{
    CD3DX12_VIEWPORT() = default;
    explicit CD3DX12_VIEWPORT( const D3D12_VIEWPORT& o ) noexcept :
        D3D12_VIEWPORT( o )
    {}
    explicit CD3DX12_VIEWPORT(
        FLOAT topLeftX,
        FLOAT topLeftY,
        FLOAT width,
        FLOAT height,
        FLOAT minDepth = D3D12_MIN_DEPTH,
        FLOAT maxDepth = D3D12_MAX_DEPTH ) noexcept
    {
        TopLeftX = topLeftX;
        TopLeftY = topLeftY;
        Width = width;
        Height = height;
        MinDepth = minDepth;
        MaxDepth = maxDepth;
    }
    explicit CD3DX12_VIEWPORT(
        _In_ ID3D12Resource* pResource,
        UINT mipSlice = 0,
        FLOAT topLeftX = 0.0f,
        FLOAT topLeftY = 0.0f,
        FLOAT minDepth = D3D12_MIN_DEPTH,
        FLOAT maxDepth = D3D12_MAX_DEPTH ) noexcept
    {
        const auto Desc = pResource->GetDesc();
        const UINT64 SubresourceWidth = Desc.Width >> mipSlice;
        const UINT64 SubresourceHeight = Desc.Height >> mipSlice;
        switch (Desc.Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_BUFFER:
            TopLeftX = topLeftX;
            TopLeftY = 0.0f;
            Width = float(Desc.Width) - topLeftX;
            Height = 1.0f;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            TopLeftX = topLeftX;
            TopLeftY = 0.0f;
            Width = (SubresourceWidth ? float(SubresourceWidth) : 1.0f) - topLeftX;
            Height = 1.0f;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            TopLeftX = topLeftX;
            TopLeftY = topLeftY;
            Width = (SubresourceWidth ? float(SubresourceWidth) : 1.0f) - topLeftX;
            Height = (SubresourceHeight ? float(SubresourceHeight) : 1.0f) - topLeftY;
            break;
        default: break;
        }

        MinDepth = minDepth;
        MaxDepth = maxDepth;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_BOX : public D3D12_BOX
{
    CD3DX12_BOX() = default;
    explicit CD3DX12_BOX( const D3D12_BOX& o ) noexcept :
        D3D12_BOX( o )
    {}
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Right ) noexcept
    {
        left = static_cast<UINT>(Left);
        top = 0;
        front = 0;
        right = static_cast<UINT>(Right);
        bottom = 1;
        back = 1;
    }
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Top,
        LONG Right,
        LONG Bottom ) noexcept
    {
        left = static_cast<UINT>(Left);
        top = static_cast<UINT>(Top);
        front = 0;
        right = static_cast<UINT>(Right);
        bottom = static_cast<UINT>(Bottom);
        back = 1;
    }
    explicit CD3DX12_BOX(
        LONG Left,
        LONG Top,
        LONG Front,
        LONG Right,
        LONG Bottom,
        LONG Back ) noexcept
    {
        left = static_cast<UINT>(Left);
        top = static_cast<UINT>(Top);
        front = static_cast<UINT>(Front);
        right = static_cast<UINT>(Right);
        bottom = static_cast<UINT>(Bottom);
        back = static_cast<UINT>(Back);
    }
};
inline bool operator==( const D3D12_BOX& l, const D3D12_BOX& r ) noexcept
{
    return l.left == r.left && l.top == r.top && l.front == r.front &&
        l.right == r.right && l.bottom == r.bottom && l.back == r.back;
}
inline bool operator!=( const D3D12_BOX& l, const D3D12_BOX& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_DEPTH_STENCIL_DESC : public D3D12_DEPTH_STENCIL_DESC
{
    CD3DX12_DEPTH_STENCIL_DESC() = default;
    explicit CD3DX12_DEPTH_STENCIL_DESC( const D3D12_DEPTH_STENCIL_DESC& o ) noexcept :
        D3D12_DEPTH_STENCIL_DESC( o )
    {}
    explicit CD3DX12_DEPTH_STENCIL_DESC( CD3DX12_DEFAULT ) noexcept
    {
        DepthEnable = TRUE;
        DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        StencilEnable = FALSE;
        StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
        StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp =
        { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
        FrontFace = defaultStencilOp;
        BackFace = defaultStencilOp;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC(
        BOOL depthEnable,
        D3D12_DEPTH_WRITE_MASK depthWriteMask,
        D3D12_COMPARISON_FUNC depthFunc,
        BOOL stencilEnable,
        UINT8 stencilReadMask,
        UINT8 stencilWriteMask,
        D3D12_STENCIL_OP frontStencilFailOp,
        D3D12_STENCIL_OP frontStencilDepthFailOp,
        D3D12_STENCIL_OP frontStencilPassOp,
        D3D12_COMPARISON_FUNC frontStencilFunc,
        D3D12_STENCIL_OP backStencilFailOp,
        D3D12_STENCIL_OP backStencilDepthFailOp,
        D3D12_STENCIL_OP backStencilPassOp,
        D3D12_COMPARISON_FUNC backStencilFunc ) noexcept
    {
        DepthEnable = depthEnable;
        DepthWriteMask = depthWriteMask;
        DepthFunc = depthFunc;
        StencilEnable = stencilEnable;
        StencilReadMask = stencilReadMask;
        StencilWriteMask = stencilWriteMask;
        FrontFace.StencilFailOp = frontStencilFailOp;
        FrontFace.StencilDepthFailOp = frontStencilDepthFailOp;
        FrontFace.StencilPassOp = frontStencilPassOp;
        FrontFace.StencilFunc = frontStencilFunc;
        BackFace.StencilFailOp = backStencilFailOp;
        BackFace.StencilDepthFailOp = backStencilDepthFailOp;
        BackFace.StencilPassOp = backStencilPassOp;
        BackFace.StencilFunc = backStencilFunc;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DEPTH_STENCIL_DESC1 : public D3D12_DEPTH_STENCIL_DESC1
{
    CD3DX12_DEPTH_STENCIL_DESC1() = default;
    explicit CD3DX12_DEPTH_STENCIL_DESC1( const D3D12_DEPTH_STENCIL_DESC1& o ) noexcept :
        D3D12_DEPTH_STENCIL_DESC1( o )
    {}
    explicit CD3DX12_DEPTH_STENCIL_DESC1( const D3D12_DEPTH_STENCIL_DESC& o ) noexcept
    {
        DepthEnable                  = o.DepthEnable;
        DepthWriteMask               = o.DepthWriteMask;
        DepthFunc                    = o.DepthFunc;
        StencilEnable                = o.StencilEnable;
        StencilReadMask              = o.StencilReadMask;
        StencilWriteMask             = o.StencilWriteMask;
        FrontFace.StencilFailOp      = o.FrontFace.StencilFailOp;
        FrontFace.StencilDepthFailOp = o.FrontFace.StencilDepthFailOp;
        FrontFace.StencilPassOp      = o.FrontFace.StencilPassOp;
        FrontFace.StencilFunc        = o.FrontFace.StencilFunc;
        BackFace.StencilFailOp       = o.BackFace.StencilFailOp;
        BackFace.StencilDepthFailOp  = o.BackFace.StencilDepthFailOp;
        BackFace.StencilPassOp       = o.BackFace.StencilPassOp;
        BackFace.StencilFunc         = o.BackFace.StencilFunc;
        DepthBoundsTestEnable        = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC1( CD3DX12_DEFAULT ) noexcept
    {
        DepthEnable = TRUE;
        DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        StencilEnable = FALSE;
        StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
        StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
        const D3D12_DEPTH_STENCILOP_DESC defaultStencilOp =
        { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS };
        FrontFace = defaultStencilOp;
        BackFace = defaultStencilOp;
        DepthBoundsTestEnable = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC1(
        BOOL depthEnable,
        D3D12_DEPTH_WRITE_MASK depthWriteMask,
        D3D12_COMPARISON_FUNC depthFunc,
        BOOL stencilEnable,
        UINT8 stencilReadMask,
        UINT8 stencilWriteMask,
        D3D12_STENCIL_OP frontStencilFailOp,
        D3D12_STENCIL_OP frontStencilDepthFailOp,
        D3D12_STENCIL_OP frontStencilPassOp,
        D3D12_COMPARISON_FUNC frontStencilFunc,
        D3D12_STENCIL_OP backStencilFailOp,
        D3D12_STENCIL_OP backStencilDepthFailOp,
        D3D12_STENCIL_OP backStencilPassOp,
        D3D12_COMPARISON_FUNC backStencilFunc,
        BOOL depthBoundsTestEnable ) noexcept
    {
        DepthEnable = depthEnable;
        DepthWriteMask = depthWriteMask;
        DepthFunc = depthFunc;
        StencilEnable = stencilEnable;
        StencilReadMask = stencilReadMask;
        StencilWriteMask = stencilWriteMask;
        FrontFace.StencilFailOp = frontStencilFailOp;
        FrontFace.StencilDepthFailOp = frontStencilDepthFailOp;
        FrontFace.StencilPassOp = frontStencilPassOp;
        FrontFace.StencilFunc = frontStencilFunc;
        BackFace.StencilFailOp = backStencilFailOp;
        BackFace.StencilDepthFailOp = backStencilDepthFailOp;
        BackFace.StencilPassOp = backStencilPassOp;
        BackFace.StencilFunc = backStencilFunc;
        DepthBoundsTestEnable = depthBoundsTestEnable;
    }
    operator D3D12_DEPTH_STENCIL_DESC() const noexcept
    {
        D3D12_DEPTH_STENCIL_DESC D;
        D.DepthEnable                  = DepthEnable;
        D.DepthWriteMask               = DepthWriteMask;
        D.DepthFunc                    = DepthFunc;
        D.StencilEnable                = StencilEnable;
        D.StencilReadMask              = StencilReadMask;
        D.StencilWriteMask             = StencilWriteMask;
        D.FrontFace.StencilFailOp      = FrontFace.StencilFailOp;
        D.FrontFace.StencilDepthFailOp = FrontFace.StencilDepthFailOp;
        D.FrontFace.StencilPassOp      = FrontFace.StencilPassOp;
        D.FrontFace.StencilFunc        = FrontFace.StencilFunc;
        D.BackFace.StencilFailOp       = BackFace.StencilFailOp;
        D.BackFace.StencilDepthFailOp  = BackFace.StencilDepthFailOp;
        D.BackFace.StencilPassOp       = BackFace.StencilPassOp;
        D.BackFace.StencilFunc         = BackFace.StencilFunc;
        return D;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DEPTH_STENCIL_DESC2 : public D3D12_DEPTH_STENCIL_DESC2
{
    CD3DX12_DEPTH_STENCIL_DESC2() = default;
    explicit CD3DX12_DEPTH_STENCIL_DESC2( const D3D12_DEPTH_STENCIL_DESC2& o ) noexcept :
        D3D12_DEPTH_STENCIL_DESC2( o )
    {}
    explicit CD3DX12_DEPTH_STENCIL_DESC2( const D3D12_DEPTH_STENCIL_DESC1& o ) noexcept
    {
        DepthEnable                  = o.DepthEnable;
        DepthWriteMask               = o.DepthWriteMask;
        DepthFunc                    = o.DepthFunc;
        StencilEnable                = o.StencilEnable;
        FrontFace.StencilFailOp      = o.FrontFace.StencilFailOp;
        FrontFace.StencilDepthFailOp = o.FrontFace.StencilDepthFailOp;
        FrontFace.StencilPassOp      = o.FrontFace.StencilPassOp;
        FrontFace.StencilFunc        = o.FrontFace.StencilFunc;
        FrontFace.StencilReadMask    = o.StencilReadMask;
        FrontFace.StencilWriteMask   = o.StencilWriteMask;

        BackFace.StencilFailOp       = o.BackFace.StencilFailOp;
        BackFace.StencilDepthFailOp  = o.BackFace.StencilDepthFailOp;
        BackFace.StencilPassOp       = o.BackFace.StencilPassOp;
        BackFace.StencilFunc         = o.BackFace.StencilFunc;
        BackFace.StencilReadMask     = o.StencilReadMask;
        BackFace.StencilWriteMask    = o.StencilWriteMask;
        DepthBoundsTestEnable        = o.DepthBoundsTestEnable;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC2( const D3D12_DEPTH_STENCIL_DESC& o ) noexcept
    {
        DepthEnable                  = o.DepthEnable;
        DepthWriteMask               = o.DepthWriteMask;
        DepthFunc                    = o.DepthFunc;
        StencilEnable                = o.StencilEnable;

        FrontFace.StencilFailOp      = o.FrontFace.StencilFailOp;
        FrontFace.StencilDepthFailOp = o.FrontFace.StencilDepthFailOp;
        FrontFace.StencilPassOp      = o.FrontFace.StencilPassOp;
        FrontFace.StencilFunc        = o.FrontFace.StencilFunc;
        FrontFace.StencilReadMask    = o.StencilReadMask;
        FrontFace.StencilWriteMask   = o.StencilWriteMask;

        BackFace.StencilFailOp       = o.BackFace.StencilFailOp;
        BackFace.StencilDepthFailOp  = o.BackFace.StencilDepthFailOp;
        BackFace.StencilPassOp       = o.BackFace.StencilPassOp;
        BackFace.StencilFunc         = o.BackFace.StencilFunc;
        BackFace.StencilReadMask     = o.StencilReadMask;
        BackFace.StencilWriteMask    = o.StencilWriteMask;

        DepthBoundsTestEnable        = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC2( CD3DX12_DEFAULT ) noexcept
    {
        DepthEnable = TRUE;
        DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        StencilEnable = FALSE;
        const D3D12_DEPTH_STENCILOP_DESC1 defaultStencilOp =
        { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS, D3D12_DEFAULT_STENCIL_READ_MASK, D3D12_DEFAULT_STENCIL_WRITE_MASK };
        FrontFace = defaultStencilOp;
        BackFace = defaultStencilOp;
        DepthBoundsTestEnable = FALSE;
    }
    explicit CD3DX12_DEPTH_STENCIL_DESC2(
        BOOL depthEnable,
        D3D12_DEPTH_WRITE_MASK depthWriteMask,
        D3D12_COMPARISON_FUNC depthFunc,
        BOOL stencilEnable,
        D3D12_STENCIL_OP frontStencilFailOp,
        D3D12_STENCIL_OP frontStencilDepthFailOp,
        D3D12_STENCIL_OP frontStencilPassOp,
        D3D12_COMPARISON_FUNC frontStencilFunc,
        UINT8 frontStencilReadMask,
        UINT8 frontStencilWriteMask,
        D3D12_STENCIL_OP backStencilFailOp,
        D3D12_STENCIL_OP backStencilDepthFailOp,
        D3D12_STENCIL_OP backStencilPassOp,
        D3D12_COMPARISON_FUNC backStencilFunc,
        UINT8 backStencilReadMask,
        UINT8 backStencilWriteMask,
        BOOL depthBoundsTestEnable ) noexcept
    {
        DepthEnable = depthEnable;
        DepthWriteMask = depthWriteMask;
        DepthFunc = depthFunc;
        StencilEnable = stencilEnable;

        FrontFace.StencilFailOp = frontStencilFailOp;
        FrontFace.StencilDepthFailOp = frontStencilDepthFailOp;
        FrontFace.StencilPassOp = frontStencilPassOp;
        FrontFace.StencilFunc = frontStencilFunc;
        FrontFace.StencilReadMask = frontStencilReadMask;
        FrontFace.StencilWriteMask = frontStencilWriteMask;

        BackFace.StencilFailOp = backStencilFailOp;
        BackFace.StencilDepthFailOp = backStencilDepthFailOp;
        BackFace.StencilPassOp = backStencilPassOp;
        BackFace.StencilFunc = backStencilFunc;
        BackFace.StencilReadMask = backStencilReadMask;
        BackFace.StencilWriteMask = backStencilWriteMask;

        DepthBoundsTestEnable = depthBoundsTestEnable;
    }

    operator D3D12_DEPTH_STENCIL_DESC() const noexcept
    {
        D3D12_DEPTH_STENCIL_DESC D;
        D.DepthEnable = DepthEnable;
        D.DepthWriteMask = DepthWriteMask;
        D.DepthFunc = DepthFunc;
        D.StencilEnable = StencilEnable;
        D.StencilReadMask = FrontFace.StencilReadMask;
        D.StencilWriteMask = FrontFace.StencilWriteMask;
        D.FrontFace.StencilFailOp = FrontFace.StencilFailOp;
        D.FrontFace.StencilDepthFailOp = FrontFace.StencilDepthFailOp;
        D.FrontFace.StencilPassOp = FrontFace.StencilPassOp;
        D.FrontFace.StencilFunc = FrontFace.StencilFunc;
        D.BackFace.StencilFailOp = BackFace.StencilFailOp;
        D.BackFace.StencilDepthFailOp = BackFace.StencilDepthFailOp;
        D.BackFace.StencilPassOp = BackFace.StencilPassOp;
        D.BackFace.StencilFunc = BackFace.StencilFunc;
        return D;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_BLEND_DESC : public D3D12_BLEND_DESC
{
    CD3DX12_BLEND_DESC() = default;
    explicit CD3DX12_BLEND_DESC( const D3D12_BLEND_DESC& o ) noexcept :
        D3D12_BLEND_DESC( o )
    {}
    explicit CD3DX12_BLEND_DESC( CD3DX12_DEFAULT ) noexcept
    {
        AlphaToCoverageEnable = FALSE;
        IndependentBlendEnable = FALSE;
        const D3D12_RENDER_TARGET_BLEND_DESC defaultRenderTargetBlendDesc =
        {
            FALSE,FALSE,
            D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
            D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
            D3D12_LOGIC_OP_NOOP,
            D3D12_COLOR_WRITE_ENABLE_ALL,
        };
        for (UINT i = 0; i < D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i)
            RenderTarget[ i ] = defaultRenderTargetBlendDesc;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RASTERIZER_DESC : public D3D12_RASTERIZER_DESC
{
    CD3DX12_RASTERIZER_DESC() = default;
    explicit CD3DX12_RASTERIZER_DESC( const D3D12_RASTERIZER_DESC& o ) noexcept :
        D3D12_RASTERIZER_DESC( o )
    {}
    explicit CD3DX12_RASTERIZER_DESC( CD3DX12_DEFAULT ) noexcept
    {
        FillMode = D3D12_FILL_MODE_SOLID;
        CullMode = D3D12_CULL_MODE_BACK;
        FrontCounterClockwise = FALSE;
        DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        DepthClipEnable = TRUE;
        MultisampleEnable = FALSE;
        AntialiasedLineEnable = FALSE;
        ForcedSampleCount = 0;
        ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    }
    explicit CD3DX12_RASTERIZER_DESC(
        D3D12_FILL_MODE fillMode,
        D3D12_CULL_MODE cullMode,
        BOOL frontCounterClockwise,
        INT depthBias,
        FLOAT depthBiasClamp,
        FLOAT slopeScaledDepthBias,
        BOOL depthClipEnable,
        BOOL multisampleEnable,
        BOOL antialiasedLineEnable,
        UINT forcedSampleCount,
        D3D12_CONSERVATIVE_RASTERIZATION_MODE conservativeRaster) noexcept
    {
        FillMode = fillMode;
        CullMode = cullMode;
        FrontCounterClockwise = frontCounterClockwise;
        DepthBias = depthBias;
        DepthBiasClamp = depthBiasClamp;
        SlopeScaledDepthBias = slopeScaledDepthBias;
        DepthClipEnable = depthClipEnable;
        MultisampleEnable = multisampleEnable;
        AntialiasedLineEnable = antialiasedLineEnable;
        ForcedSampleCount = forcedSampleCount;
        ConservativeRaster = conservativeRaster;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_ALLOCATION_INFO : public D3D12_RESOURCE_ALLOCATION_INFO
{
    CD3DX12_RESOURCE_ALLOCATION_INFO() = default;
    explicit CD3DX12_RESOURCE_ALLOCATION_INFO( const D3D12_RESOURCE_ALLOCATION_INFO& o ) noexcept :
        D3D12_RESOURCE_ALLOCATION_INFO( o )
    {}
    CD3DX12_RESOURCE_ALLOCATION_INFO(
        UINT64 size,
        UINT64 alignment ) noexcept
    {
        SizeInBytes = size;
        Alignment = alignment;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_HEAP_PROPERTIES : public D3D12_HEAP_PROPERTIES
{
    CD3DX12_HEAP_PROPERTIES() = default;
    explicit CD3DX12_HEAP_PROPERTIES(const D3D12_HEAP_PROPERTIES &o) noexcept :
        D3D12_HEAP_PROPERTIES(o)
    {}
    CD3DX12_HEAP_PROPERTIES(
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        UINT creationNodeMask = 1,
        UINT nodeMask = 1 ) noexcept
    {
        Type = D3D12_HEAP_TYPE_CUSTOM;
        CPUPageProperty = cpuPageProperty;
        MemoryPoolPreference = memoryPoolPreference;
        CreationNodeMask = creationNodeMask;
        VisibleNodeMask = nodeMask;
    }
    explicit CD3DX12_HEAP_PROPERTIES(
        D3D12_HEAP_TYPE type,
        UINT creationNodeMask = 1,
        UINT nodeMask = 1 ) noexcept
    {
        Type = type;
        CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        CreationNodeMask = creationNodeMask;
        VisibleNodeMask = nodeMask;
    }
    bool IsCPUAccessible() const noexcept
    {
        return Type == D3D12_HEAP_TYPE_UPLOAD || Type == D3D12_HEAP_TYPE_READBACK || (Type == D3D12_HEAP_TYPE_CUSTOM &&
            (CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE || CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
    }
};
inline bool operator==( const D3D12_HEAP_PROPERTIES& l, const D3D12_HEAP_PROPERTIES& r ) noexcept
{
    return l.Type == r.Type && l.CPUPageProperty == r.CPUPageProperty &&
        l.MemoryPoolPreference == r.MemoryPoolPreference &&
        l.CreationNodeMask == r.CreationNodeMask &&
        l.VisibleNodeMask == r.VisibleNodeMask;
}
inline bool operator!=( const D3D12_HEAP_PROPERTIES& l, const D3D12_HEAP_PROPERTIES& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_HEAP_DESC : public D3D12_HEAP_DESC
{
    CD3DX12_HEAP_DESC() = default;
    explicit CD3DX12_HEAP_DESC(const D3D12_HEAP_DESC &o) noexcept :
        D3D12_HEAP_DESC(o)
    {}
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_HEAP_PROPERTIES properties,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = size;
        Properties = properties;
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_HEAP_TYPE type,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = size;
        Properties = CD3DX12_HEAP_PROPERTIES( type );
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        UINT64 size,
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        UINT64 alignment = 0,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = size;
        Properties = CD3DX12_HEAP_PROPERTIES( cpuPageProperty, memoryPoolPreference );
        Alignment = alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
        D3D12_HEAP_PROPERTIES properties,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = properties;
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
        D3D12_HEAP_TYPE type,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = CD3DX12_HEAP_PROPERTIES( type );
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    CD3DX12_HEAP_DESC(
        const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
        D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
        D3D12_MEMORY_POOL memoryPoolPreference,
        D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE ) noexcept
    {
        SizeInBytes = resAllocInfo.SizeInBytes;
        Properties = CD3DX12_HEAP_PROPERTIES( cpuPageProperty, memoryPoolPreference );
        Alignment = resAllocInfo.Alignment;
        Flags = flags;
    }
    bool IsCPUAccessible() const noexcept
    { return static_cast< const CD3DX12_HEAP_PROPERTIES* >( &Properties )->IsCPUAccessible(); }
};
inline bool operator==( const D3D12_HEAP_DESC& l, const D3D12_HEAP_DESC& r ) noexcept
{
    return l.SizeInBytes == r.SizeInBytes &&
        l.Properties == r.Properties &&
        l.Alignment == r.Alignment &&
        l.Flags == r.Flags;
}
inline bool operator!=( const D3D12_HEAP_DESC& l, const D3D12_HEAP_DESC& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_CLEAR_VALUE : public D3D12_CLEAR_VALUE
{
    CD3DX12_CLEAR_VALUE() = default;
    explicit CD3DX12_CLEAR_VALUE(const D3D12_CLEAR_VALUE &o) noexcept :
        D3D12_CLEAR_VALUE(o)
    {}
    CD3DX12_CLEAR_VALUE(
        DXGI_FORMAT format,
        const FLOAT color[4] ) noexcept
    {
        Format = format;
        memcpy( Color, color, sizeof( Color ) );
    }
    CD3DX12_CLEAR_VALUE(
        DXGI_FORMAT format,
        FLOAT depth,
        UINT8 stencil ) noexcept
    {
        Format = format;
        memset( &Color, 0, sizeof( Color ) );
        /* Use memcpy to preserve NAN values */
        memcpy( &DepthStencil.Depth, &depth, sizeof( depth ) );
        DepthStencil.Stencil = stencil;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RANGE : public D3D12_RANGE
{
    CD3DX12_RANGE() = default;
    explicit CD3DX12_RANGE(const D3D12_RANGE &o) noexcept :
        D3D12_RANGE(o)
    {}
    CD3DX12_RANGE(
        SIZE_T begin,
        SIZE_T end ) noexcept
    {
        Begin = begin;
        End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RANGE_UINT64 : public D3D12_RANGE_UINT64
{
    CD3DX12_RANGE_UINT64() = default;
    explicit CD3DX12_RANGE_UINT64(const D3D12_RANGE_UINT64 &o) noexcept :
        D3D12_RANGE_UINT64(o)
    {}
    CD3DX12_RANGE_UINT64(
        UINT64 begin,
        UINT64 end ) noexcept
    {
        Begin = begin;
        End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_RANGE_UINT64 : public D3D12_SUBRESOURCE_RANGE_UINT64
{
    CD3DX12_SUBRESOURCE_RANGE_UINT64() = default;
    explicit CD3DX12_SUBRESOURCE_RANGE_UINT64(const D3D12_SUBRESOURCE_RANGE_UINT64 &o) noexcept :
        D3D12_SUBRESOURCE_RANGE_UINT64(o)
    {}
    CD3DX12_SUBRESOURCE_RANGE_UINT64(
        UINT subresource,
        const D3D12_RANGE_UINT64& range ) noexcept
    {
        Subresource = subresource;
        Range = range;
    }
    CD3DX12_SUBRESOURCE_RANGE_UINT64(
        UINT subresource,
        UINT64 begin,
        UINT64 end ) noexcept
    {
        Subresource = subresource;
        Range.Begin = begin;
        Range.End = end;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE
{
    CD3DX12_SHADER_BYTECODE() = default;
    explicit CD3DX12_SHADER_BYTECODE(const D3D12_SHADER_BYTECODE &o) noexcept :
        D3D12_SHADER_BYTECODE(o)
    {}
    CD3DX12_SHADER_BYTECODE(
        _In_ ID3DBlob* pShaderBlob ) noexcept
    {
        pShaderBytecode = pShaderBlob->GetBufferPointer();
        BytecodeLength = pShaderBlob->GetBufferSize();
    }
    CD3DX12_SHADER_BYTECODE(
        const void* _pShaderBytecode,
        SIZE_T bytecodeLength ) noexcept
    {
        pShaderBytecode = _pShaderBytecode;
        BytecodeLength = bytecodeLength;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILED_RESOURCE_COORDINATE : public D3D12_TILED_RESOURCE_COORDINATE
{
    CD3DX12_TILED_RESOURCE_COORDINATE() = default;
    explicit CD3DX12_TILED_RESOURCE_COORDINATE(const D3D12_TILED_RESOURCE_COORDINATE &o) noexcept :
        D3D12_TILED_RESOURCE_COORDINATE(o)
    {}
    CD3DX12_TILED_RESOURCE_COORDINATE(
        UINT x,
        UINT y,
        UINT z,
        UINT subresource ) noexcept
    {
        X = x;
        Y = y;
        Z = z;
        Subresource = subresource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILE_REGION_SIZE : public D3D12_TILE_REGION_SIZE
{
    CD3DX12_TILE_REGION_SIZE() = default;
    explicit CD3DX12_TILE_REGION_SIZE(const D3D12_TILE_REGION_SIZE &o) noexcept :
        D3D12_TILE_REGION_SIZE(o)
    {}
    CD3DX12_TILE_REGION_SIZE(
        UINT numTiles,
        BOOL useBox,
        UINT width,
        UINT16 height,
        UINT16 depth ) noexcept
    {
        NumTiles = numTiles;
        UseBox = useBox;
        Width = width;
        Height = height;
        Depth = depth;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_TILING : public D3D12_SUBRESOURCE_TILING
{
    CD3DX12_SUBRESOURCE_TILING() = default;
    explicit CD3DX12_SUBRESOURCE_TILING(const D3D12_SUBRESOURCE_TILING &o) noexcept :
        D3D12_SUBRESOURCE_TILING(o)
    {}
    CD3DX12_SUBRESOURCE_TILING(
        UINT widthInTiles,
        UINT16 heightInTiles,
        UINT16 depthInTiles,
        UINT startTileIndexInOverallResource ) noexcept
    {
        WidthInTiles = widthInTiles;
        HeightInTiles = heightInTiles;
        DepthInTiles = depthInTiles;
        StartTileIndexInOverallResource = startTileIndexInOverallResource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TILE_SHAPE : public D3D12_TILE_SHAPE
{
    CD3DX12_TILE_SHAPE() = default;
    explicit CD3DX12_TILE_SHAPE(const D3D12_TILE_SHAPE &o) noexcept :
        D3D12_TILE_SHAPE(o)
    {}
    CD3DX12_TILE_SHAPE(
        UINT widthInTexels,
        UINT heightInTexels,
        UINT depthInTexels ) noexcept
    {
        WidthInTexels = widthInTexels;
        HeightInTexels = heightInTexels;
        DepthInTexels = depthInTexels;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_BARRIER : public D3D12_RESOURCE_BARRIER
{
    CD3DX12_RESOURCE_BARRIER() = default;
    explicit CD3DX12_RESOURCE_BARRIER(const D3D12_RESOURCE_BARRIER &o) noexcept :
        D3D12_RESOURCE_BARRIER(o)
    {}
    static inline CD3DX12_RESOURCE_BARRIER Transition(
        _In_ ID3D12Resource* pResource,
        D3D12_RESOURCE_STATES stateBefore,
        D3D12_RESOURCE_STATES stateAfter,
        UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) noexcept
    {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        result.Flags = flags;
        barrier.Transition.pResource = pResource;
        barrier.Transition.StateBefore = stateBefore;
        barrier.Transition.StateAfter = stateAfter;
        barrier.Transition.Subresource = subresource;
        return result;
    }
    static inline CD3DX12_RESOURCE_BARRIER Aliasing(
        _In_ ID3D12Resource* pResourceBefore,
        _In_ ID3D12Resource* pResourceAfter) noexcept
    {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        barrier.Aliasing.pResourceBefore = pResourceBefore;
        barrier.Aliasing.pResourceAfter = pResourceAfter;
        return result;
    }
    static inline CD3DX12_RESOURCE_BARRIER UAV(
        _In_ ID3D12Resource* pResource) noexcept
    {
        CD3DX12_RESOURCE_BARRIER result = {};
        D3D12_RESOURCE_BARRIER &barrier = result;
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = pResource;
        return result;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_PACKED_MIP_INFO : public D3D12_PACKED_MIP_INFO
{
    CD3DX12_PACKED_MIP_INFO() = default;
    explicit CD3DX12_PACKED_MIP_INFO(const D3D12_PACKED_MIP_INFO &o) noexcept :
        D3D12_PACKED_MIP_INFO(o)
    {}
    CD3DX12_PACKED_MIP_INFO(
        UINT8 numStandardMips,
        UINT8 numPackedMips,
        UINT numTilesForPackedMips,
        UINT startTileIndexInOverallResource ) noexcept
    {
        NumStandardMips = numStandardMips;
        NumPackedMips = numPackedMips;
        NumTilesForPackedMips = numTilesForPackedMips;
        StartTileIndexInOverallResource = startTileIndexInOverallResource;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_SUBRESOURCE_FOOTPRINT : public D3D12_SUBRESOURCE_FOOTPRINT
{
    CD3DX12_SUBRESOURCE_FOOTPRINT() = default;
    explicit CD3DX12_SUBRESOURCE_FOOTPRINT(const D3D12_SUBRESOURCE_FOOTPRINT &o) noexcept :
        D3D12_SUBRESOURCE_FOOTPRINT(o)
    {}
    CD3DX12_SUBRESOURCE_FOOTPRINT(
        DXGI_FORMAT format,
        UINT width,
        UINT height,
        UINT depth,
        UINT rowPitch ) noexcept
    {
        Format = format;
        Width = width;
        Height = height;
        Depth = depth;
        RowPitch = rowPitch;
    }
    explicit CD3DX12_SUBRESOURCE_FOOTPRINT(
        const D3D12_RESOURCE_DESC& resDesc,
        UINT rowPitch ) noexcept
    {
        Format = resDesc.Format;
        Width = UINT( resDesc.Width );
        Height = resDesc.Height;
        Depth = (resDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? resDesc.DepthOrArraySize : 1u);
        RowPitch = rowPitch;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_TEXTURE_COPY_LOCATION : public D3D12_TEXTURE_COPY_LOCATION
{
    CD3DX12_TEXTURE_COPY_LOCATION() = default;
    explicit CD3DX12_TEXTURE_COPY_LOCATION(const D3D12_TEXTURE_COPY_LOCATION &o) noexcept :
        D3D12_TEXTURE_COPY_LOCATION(o)
    {}
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource* pRes) noexcept
    {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        PlacedFootprint = {};
    }
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource* pRes, D3D12_PLACED_SUBRESOURCE_FOOTPRINT const& Footprint) noexcept
    {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        PlacedFootprint = Footprint;
    }
    CD3DX12_TEXTURE_COPY_LOCATION(_In_ ID3D12Resource* pRes, UINT Sub) noexcept
    {
        pResource = pRes;
        Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        PlacedFootprint = {};
        SubresourceIndex = Sub;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DESCRIPTOR_RANGE : public D3D12_DESCRIPTOR_RANGE
{
    CD3DX12_DESCRIPTOR_RANGE() = default;
    explicit CD3DX12_DESCRIPTOR_RANGE(const D3D12_DESCRIPTOR_RANGE &o) noexcept :
        D3D12_DESCRIPTOR_RANGE(o)
    {}
    CD3DX12_DESCRIPTOR_RANGE(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, offsetInDescriptorsFromTableStart);
    }

    inline void Init(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, offsetInDescriptorsFromTableStart);
    }

    static inline void Init(
        _Out_ D3D12_DESCRIPTOR_RANGE &range,
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        range.RangeType = rangeType;
        range.NumDescriptors = numDescriptors;
        range.BaseShaderRegister = baseShaderRegister;
        range.RegisterSpace = registerSpace;
        range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR_TABLE : public D3D12_ROOT_DESCRIPTOR_TABLE
{
    CD3DX12_ROOT_DESCRIPTOR_TABLE() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR_TABLE(const D3D12_ROOT_DESCRIPTOR_TABLE &o) noexcept :
        D3D12_ROOT_DESCRIPTOR_TABLE(o)
    {}
    CD3DX12_ROOT_DESCRIPTOR_TABLE(
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* _pDescriptorRanges) noexcept
    {
        Init(numDescriptorRanges, _pDescriptorRanges);
    }

    inline void Init(
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* _pDescriptorRanges) noexcept
    {
        Init(*this, numDescriptorRanges, _pDescriptorRanges);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR_TABLE &rootDescriptorTable,
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* _pDescriptorRanges) noexcept
    {
        rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
        rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_CONSTANTS : public D3D12_ROOT_CONSTANTS
{
    CD3DX12_ROOT_CONSTANTS() = default;
    explicit CD3DX12_ROOT_CONSTANTS(const D3D12_ROOT_CONSTANTS &o) noexcept :
        D3D12_ROOT_CONSTANTS(o)
    {}
    CD3DX12_ROOT_CONSTANTS(
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0) noexcept
    {
        Init(num32BitValues, shaderRegister, registerSpace);
    }

    inline void Init(
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0) noexcept
    {
        Init(*this, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_CONSTANTS &rootConstants,
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0) noexcept
    {
        rootConstants.Num32BitValues = num32BitValues;
        rootConstants.ShaderRegister = shaderRegister;
        rootConstants.RegisterSpace = registerSpace;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR : public D3D12_ROOT_DESCRIPTOR
{
    CD3DX12_ROOT_DESCRIPTOR() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR(const D3D12_ROOT_DESCRIPTOR &o) noexcept :
        D3D12_ROOT_DESCRIPTOR(o)
    {}
    CD3DX12_ROOT_DESCRIPTOR(
        UINT shaderRegister,
        UINT registerSpace = 0) noexcept
    {
        Init(shaderRegister, registerSpace);
    }

    inline void Init(
        UINT shaderRegister,
        UINT registerSpace = 0) noexcept
    {
        Init(*this, shaderRegister, registerSpace);
    }

    static inline void Init(_Out_ D3D12_ROOT_DESCRIPTOR &table, UINT shaderRegister, UINT registerSpace = 0) noexcept
    {
        table.ShaderRegister = shaderRegister;
        table.RegisterSpace = registerSpace;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_PARAMETER : public D3D12_ROOT_PARAMETER
{
    CD3DX12_ROOT_PARAMETER() = default;
    explicit CD3DX12_ROOT_PARAMETER(const D3D12_ROOT_PARAMETER &o) noexcept :
        D3D12_ROOT_PARAMETER(o)
    {}

    static inline void InitAsDescriptorTable(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        UINT numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR_TABLE::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
    }

    static inline void InitAsConstants(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void InitAsConstantBufferView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsShaderResourceView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    static inline void InitAsUnorderedAccessView(
        _Out_ D3D12_ROOT_PARAMETER &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR::Init(rootParam.Descriptor, shaderRegister, registerSpace);
    }

    inline void InitAsDescriptorTable(
        UINT numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
    }

    inline void InitAsConstants(
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsConstantBufferView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsConstantBufferView(*this, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsShaderResourceView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsShaderResourceView(*this, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsUnorderedAccessView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, visibility);
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_STATIC_SAMPLER_DESC : public D3D12_STATIC_SAMPLER_DESC
{
    CD3DX12_STATIC_SAMPLER_DESC() = default;
    explicit CD3DX12_STATIC_SAMPLER_DESC(const D3D12_STATIC_SAMPLER_DESC &o) noexcept :
        D3D12_STATIC_SAMPLER_DESC(o)
    {}
    CD3DX12_STATIC_SAMPLER_DESC(
         UINT shaderRegister,
         D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
         D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         FLOAT mipLODBias = 0,
         UINT maxAnisotropy = 16,
         D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
         D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
         FLOAT minLOD = 0.f,
         FLOAT maxLOD = D3D12_FLOAT32_MAX,
         D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
         UINT registerSpace = 0) noexcept
    {
        Init(
            shaderRegister,
            filter,
            addressU,
            addressV,
            addressW,
            mipLODBias,
            maxAnisotropy,
            comparisonFunc,
            borderColor,
            minLOD,
            maxLOD,
            shaderVisibility,
            registerSpace);
    }

    static inline void Init(
        _Out_ D3D12_STATIC_SAMPLER_DESC &samplerDesc,
         UINT shaderRegister,
         D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
         D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         FLOAT mipLODBias = 0,
         UINT maxAnisotropy = 16,
         D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
         D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
         FLOAT minLOD = 0.f,
         FLOAT maxLOD = D3D12_FLOAT32_MAX,
         D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
         UINT registerSpace = 0) noexcept
    {
        samplerDesc.ShaderRegister = shaderRegister;
        samplerDesc.Filter = filter;
        samplerDesc.AddressU = addressU;
        samplerDesc.AddressV = addressV;
        samplerDesc.AddressW = addressW;
        samplerDesc.MipLODBias = mipLODBias;
        samplerDesc.MaxAnisotropy = maxAnisotropy;
        samplerDesc.ComparisonFunc = comparisonFunc;
        samplerDesc.BorderColor = borderColor;
        samplerDesc.MinLOD = minLOD;
        samplerDesc.MaxLOD = maxLOD;
        samplerDesc.ShaderVisibility = shaderVisibility;
        samplerDesc.RegisterSpace = registerSpace;
    }
    inline void Init(
         UINT shaderRegister,
         D3D12_FILTER filter = D3D12_FILTER_ANISOTROPIC,
         D3D12_TEXTURE_ADDRESS_MODE addressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         D3D12_TEXTURE_ADDRESS_MODE addressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
         FLOAT mipLODBias = 0,
         UINT maxAnisotropy = 16,
         D3D12_COMPARISON_FUNC comparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL,
         D3D12_STATIC_BORDER_COLOR borderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
         FLOAT minLOD = 0.f,
         FLOAT maxLOD = D3D12_FLOAT32_MAX,
         D3D12_SHADER_VISIBILITY shaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
         UINT registerSpace = 0) noexcept
    {
        Init(
            *this,
            shaderRegister,
            filter,
            addressU,
            addressV,
            addressW,
            mipLODBias,
            maxAnisotropy,
            comparisonFunc,
            borderColor,
            minLOD,
            maxLOD,
            shaderVisibility,
            registerSpace);
    }

};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_SIGNATURE_DESC : public D3D12_ROOT_SIGNATURE_DESC
{
    CD3DX12_ROOT_SIGNATURE_DESC() = default;
    explicit CD3DX12_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC &o) noexcept :
        D3D12_ROOT_SIGNATURE_DESC(o)
    {}
    CD3DX12_ROOT_SIGNATURE_DESC(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT) noexcept
    {
        Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    }

    inline void Init(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_SIGNATURE_DESC &desc,
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        desc.NumParameters = numParameters;
        desc.pParameters = _pParameters;
        desc.NumStaticSamplers = numStaticSamplers;
        desc.pStaticSamplers = _pStaticSamplers;
        desc.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_DESCRIPTOR_RANGE1 : public D3D12_DESCRIPTOR_RANGE1
{
    CD3DX12_DESCRIPTOR_RANGE1() = default;
    explicit CD3DX12_DESCRIPTOR_RANGE1(const D3D12_DESCRIPTOR_RANGE1 &o) noexcept :
        D3D12_DESCRIPTOR_RANGE1(o)
    {}
    CD3DX12_DESCRIPTOR_RANGE1(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
    }

    inline void Init(
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
    }

    static inline void Init(
        _Out_ D3D12_DESCRIPTOR_RANGE1 &range,
        D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
        UINT numDescriptors,
        UINT baseShaderRegister,
        UINT registerSpace = 0,
        D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
        UINT offsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) noexcept
    {
        range.RangeType = rangeType;
        range.NumDescriptors = numDescriptors;
        range.BaseShaderRegister = baseShaderRegister;
        range.RegisterSpace = registerSpace;
        range.Flags = flags;
        range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR_TABLE1 : public D3D12_ROOT_DESCRIPTOR_TABLE1
{
    CD3DX12_ROOT_DESCRIPTOR_TABLE1() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR_TABLE1(const D3D12_ROOT_DESCRIPTOR_TABLE1 &o) noexcept :
        D3D12_ROOT_DESCRIPTOR_TABLE1(o)
    {}
    CD3DX12_ROOT_DESCRIPTOR_TABLE1(
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges) noexcept
    {
        Init(numDescriptorRanges, _pDescriptorRanges);
    }

    inline void Init(
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges) noexcept
    {
        Init(*this, numDescriptorRanges, _pDescriptorRanges);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR_TABLE1 &rootDescriptorTable,
        UINT numDescriptorRanges,
        _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges) noexcept
    {
        rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
        rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_DESCRIPTOR1 : public D3D12_ROOT_DESCRIPTOR1
{
    CD3DX12_ROOT_DESCRIPTOR1() = default;
    explicit CD3DX12_ROOT_DESCRIPTOR1(const D3D12_ROOT_DESCRIPTOR1 &o) noexcept :
        D3D12_ROOT_DESCRIPTOR1(o)
    {}
    CD3DX12_ROOT_DESCRIPTOR1(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept
    {
        Init(shaderRegister, registerSpace, flags);
    }

    inline void Init(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept
    {
        Init(*this, shaderRegister, registerSpace, flags);
    }

    static inline void Init(
        _Out_ D3D12_ROOT_DESCRIPTOR1 &table,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE) noexcept
    {
        table.ShaderRegister = shaderRegister;
        table.RegisterSpace = registerSpace;
        table.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_ROOT_PARAMETER1 : public D3D12_ROOT_PARAMETER1
{
    CD3DX12_ROOT_PARAMETER1() = default;
    explicit CD3DX12_ROOT_PARAMETER1(const D3D12_ROOT_PARAMETER1 &o) noexcept :
        D3D12_ROOT_PARAMETER1(o)
    {}

    static inline void InitAsDescriptorTable(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        UINT numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR_TABLE1::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
    }

    static inline void InitAsConstants(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
    }

    static inline void InitAsConstantBufferView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    static inline void InitAsShaderResourceView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    static inline void InitAsUnorderedAccessView(
        _Out_ D3D12_ROOT_PARAMETER1 &rootParam,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParam.ShaderVisibility = visibility;
        CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
    }

    inline void InitAsDescriptorTable(
        UINT numDescriptorRanges,
        _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
    }

    inline void InitAsConstants(
        UINT num32BitValues,
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
    }

    inline void InitAsConstantBufferView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsConstantBufferView(*this, shaderRegister, registerSpace, flags, visibility);
    }

    inline void InitAsShaderResourceView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsShaderResourceView(*this, shaderRegister, registerSpace, flags, visibility);
    }

    inline void InitAsUnorderedAccessView(
        UINT shaderRegister,
        UINT registerSpace = 0,
        D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL) noexcept
    {
        InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, flags, visibility);
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC : public D3D12_VERSIONED_ROOT_SIGNATURE_DESC
{
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC() = default;
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC &o) noexcept :
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC(o)
    {}
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC &o) noexcept
    {
        Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        Desc_1_0 = o;
    }
    explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC1 &o) noexcept
    {
        Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        Desc_1_1 = o;
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init_1_0(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init_1_1(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT) noexcept
    {
        Init_1_1(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    }

    inline void Init_1_0(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init_1_0(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init_1_0(
        _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0.NumParameters = numParameters;
        desc.Desc_1_0.pParameters = _pParameters;
        desc.Desc_1_0.NumStaticSamplers = numStaticSamplers;
        desc.Desc_1_0.pStaticSamplers = _pStaticSamplers;
        desc.Desc_1_0.Flags = flags;
    }

    inline void Init_1_1(
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        Init_1_1(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
    }

    static inline void Init_1_1(
        _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC &desc,
        UINT numParameters,
        _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
        UINT numStaticSamplers = 0,
        _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
        D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) noexcept
    {
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc.Desc_1_1.NumParameters = numParameters;
        desc.Desc_1_1.pParameters = _pParameters;
        desc.Desc_1_1.NumStaticSamplers = numStaticSamplers;
        desc.Desc_1_1.pStaticSamplers = _pStaticSamplers;
        desc.Desc_1_1.Flags = flags;
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_CPU_DESCRIPTOR_HANDLE : public D3D12_CPU_DESCRIPTOR_HANDLE
{
    CD3DX12_CPU_DESCRIPTOR_HANDLE() = default;
    explicit CD3DX12_CPU_DESCRIPTOR_HANDLE(const D3D12_CPU_DESCRIPTOR_HANDLE &o) noexcept :
        D3D12_CPU_DESCRIPTOR_HANDLE(o)
    {}
    CD3DX12_CPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
    CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(other, offsetScaledByIncrementSize);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &other, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE& Offset(INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        ptr = SIZE_T(INT64(ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
        return *this;
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE& Offset(INT offsetScaledByIncrementSize) noexcept
    {
        ptr = SIZE_T(INT64(ptr) + INT64(offsetScaledByIncrementSize));
        return *this;
    }
    bool operator==(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr == other.ptr);
    }
    bool operator!=(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr != other.ptr);
    }
    CD3DX12_CPU_DESCRIPTOR_HANDLE &operator=(const D3D12_CPU_DESCRIPTOR_HANDLE &other) noexcept
    {
        ptr = other.ptr;
        return *this;
    }

    inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetScaledByIncrementSize);
    }

    inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
    }

    static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
    }

    static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_CPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    }
};

//------------------------------------------------------------------------------------------------
struct CD3DX12_GPU_DESCRIPTOR_HANDLE : public D3D12_GPU_DESCRIPTOR_HANDLE
{
    CD3DX12_GPU_DESCRIPTOR_HANDLE() = default;
    explicit CD3DX12_GPU_DESCRIPTOR_HANDLE(const D3D12_GPU_DESCRIPTOR_HANDLE &o) noexcept :
        D3D12_GPU_DESCRIPTOR_HANDLE(o)
    {}
    CD3DX12_GPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(other, offsetScaledByIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE& Offset(INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        ptr = UINT64(INT64(ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
        return *this;
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE& Offset(INT offsetScaledByIncrementSize) noexcept
    {
        ptr = UINT64(INT64(ptr) + INT64(offsetScaledByIncrementSize));
        return *this;
    }
    inline bool operator==(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr == other.ptr);
    }
    inline bool operator!=(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr != other.ptr);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE &operator=(const D3D12_GPU_DESCRIPTOR_HANDLE &other) noexcept
    {
        ptr = other.ptr;
        return *this;
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetScaledByIncrementSize);
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    }
};

//------------------------------------------------------------------------------------------------
constexpr UINT D3D12CalcSubresource( UINT MipSlice, UINT ArraySlice, UINT PlaneSlice, UINT MipLevels, UINT ArraySize ) noexcept
{
    return MipSlice + ArraySlice * MipLevels + PlaneSlice * MipLevels * ArraySize;
}

//------------------------------------------------------------------------------------------------
template <typename T, typename U, typename V>
inline void D3D12DecomposeSubresource( UINT Subresource, UINT MipLevels, UINT ArraySize, _Out_ T& MipSlice, _Out_ U& ArraySlice, _Out_ V& PlaneSlice ) noexcept
{
    MipSlice = static_cast<T>(Subresource % MipLevels);
    ArraySlice = static_cast<U>((Subresource / MipLevels) % ArraySize);
    PlaneSlice = static_cast<V>(Subresource / (MipLevels * ArraySize));
}

//------------------------------------------------------------------------------------------------
inline UINT8 D3D12GetFormatPlaneCount(
    _In_ ID3D12Device* pDevice,
    DXGI_FORMAT Format
    ) noexcept
{
    D3D12_FEATURE_DATA_FORMAT_INFO formatInfo = { Format, 0 };
    if (FAILED(pDevice->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &formatInfo, sizeof(formatInfo))))
    {
        return 0;
    }
    return formatInfo.PlaneCount;
}

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_DESC : public D3D12_RESOURCE_DESC
{
    CD3DX12_RESOURCE_DESC() = default;
    explicit CD3DX12_RESOURCE_DESC( const D3D12_RESOURCE_DESC& o ) noexcept :
        D3D12_RESOURCE_DESC( o )
    {}
    CD3DX12_RESOURCE_DESC(
        D3D12_RESOURCE_DIMENSION dimension,
        UINT64 alignment,
        UINT64 width,
        UINT height,
        UINT16 depthOrArraySize,
        UINT16 mipLevels,
        DXGI_FORMAT format,
        UINT sampleCount,
        UINT sampleQuality,
        D3D12_TEXTURE_LAYOUT layout,
        D3D12_RESOURCE_FLAGS flags ) noexcept
    {
        Dimension = dimension;
        Alignment = alignment;
        Width = width;
        Height = height;
        DepthOrArraySize = depthOrArraySize;
        MipLevels = mipLevels;
        Format = format;
        SampleDesc.Count = sampleCount;
        SampleDesc.Quality = sampleQuality;
        Layout = layout;
        Flags = flags;
    }
    static inline CD3DX12_RESOURCE_DESC Buffer(
        const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE ) noexcept
    {
        return CD3DX12_RESOURCE_DESC( D3D12_RESOURCE_DIMENSION_BUFFER, resAllocInfo.Alignment, resAllocInfo.SizeInBytes,
            1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags );
    }
    static inline CD3DX12_RESOURCE_DESC Buffer(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC( D3D12_RESOURCE_DIMENSION_BUFFER, alignment, width, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags );
    }
    static inline CD3DX12_RESOURCE_DESC Tex1D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC( D3D12_RESOURCE_DIMENSION_TEXTURE1D, alignment, width, 1, arraySize,
            mipLevels, format, 1, 0, layout, flags );
    }
    static inline CD3DX12_RESOURCE_DESC Tex2D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT height,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        UINT sampleCount = 1,
        UINT sampleQuality = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC( D3D12_RESOURCE_DIMENSION_TEXTURE2D, alignment, width, height, arraySize,
            mipLevels, format, sampleCount, sampleQuality, layout, flags );
    }
    static inline CD3DX12_RESOURCE_DESC Tex3D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT height,
        UINT16 depth,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC( D3D12_RESOURCE_DIMENSION_TEXTURE3D, alignment, width, height, depth,
            mipLevels, format, 1, 0, layout, flags );
    }
    inline UINT16 Depth() const noexcept
    { return (Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1u); }
    inline UINT16 ArraySize() const noexcept
    { return (Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1u); }
    inline UINT8 PlaneCount(_In_ ID3D12Device* pDevice) const noexcept
    { return D3D12GetFormatPlaneCount(pDevice, Format); }
    inline UINT Subresources(_In_ ID3D12Device* pDevice) const noexcept
    { return static_cast<UINT>(MipLevels) * ArraySize() * PlaneCount(pDevice); }
    inline UINT CalcSubresource(UINT MipSlice, UINT ArraySlice, UINT PlaneSlice) noexcept
    { return D3D12CalcSubresource(MipSlice, ArraySlice, PlaneSlice, MipLevels, ArraySize()); }
};
inline bool operator==( const D3D12_RESOURCE_DESC& l, const D3D12_RESOURCE_DESC& r ) noexcept
{
    return l.Dimension == r.Dimension &&
        l.Alignment == r.Alignment &&
        l.Width == r.Width &&
        l.Height == r.Height &&
        l.DepthOrArraySize == r.DepthOrArraySize &&
        l.MipLevels == r.MipLevels &&
        l.Format == r.Format &&
        l.SampleDesc.Count == r.SampleDesc.Count &&
        l.SampleDesc.Quality == r.SampleDesc.Quality &&
        l.Layout == r.Layout &&
        l.Flags == r.Flags;
}
inline bool operator!=( const D3D12_RESOURCE_DESC& l, const D3D12_RESOURCE_DESC& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_RESOURCE_DESC1 : public D3D12_RESOURCE_DESC1
{
    CD3DX12_RESOURCE_DESC1() = default;
    explicit CD3DX12_RESOURCE_DESC1( const D3D12_RESOURCE_DESC1& o ) noexcept :
        D3D12_RESOURCE_DESC1( o )
    {}
    explicit CD3DX12_RESOURCE_DESC1( const D3D12_RESOURCE_DESC& o ) noexcept
    {
        Dimension = o.Dimension;
        Alignment = o.Alignment;
        Width = o.Width;
        Height = o.Height;
        DepthOrArraySize = o.DepthOrArraySize;
        MipLevels = o.MipLevels;
        Format = o.Format;
        SampleDesc = o.SampleDesc;
        Layout = o.Layout;
        Flags = o.Flags;
        SamplerFeedbackMipRegion = {};
    }
    CD3DX12_RESOURCE_DESC1(
        D3D12_RESOURCE_DIMENSION dimension,
        UINT64 alignment,
        UINT64 width,
        UINT height,
        UINT16 depthOrArraySize,
        UINT16 mipLevels,
        DXGI_FORMAT format,
        UINT sampleCount,
        UINT sampleQuality,
        D3D12_TEXTURE_LAYOUT layout,
        D3D12_RESOURCE_FLAGS flags,
        UINT samplerFeedbackMipRegionWidth = 0,
        UINT samplerFeedbackMipRegionHeight = 0,
        UINT samplerFeedbackMipRegionDepth = 0) noexcept
    {
        Dimension = dimension;
        Alignment = alignment;
        Width = width;
        Height = height;
        DepthOrArraySize = depthOrArraySize;
        MipLevels = mipLevels;
        Format = format;
        SampleDesc.Count = sampleCount;
        SampleDesc.Quality = sampleQuality;
        Layout = layout;
        Flags = flags;
        SamplerFeedbackMipRegion.Width = samplerFeedbackMipRegionWidth;
        SamplerFeedbackMipRegion.Height = samplerFeedbackMipRegionHeight;
        SamplerFeedbackMipRegion.Depth = samplerFeedbackMipRegionDepth;
    }
    static inline CD3DX12_RESOURCE_DESC1 Buffer(
        const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE ) noexcept
    {
        return CD3DX12_RESOURCE_DESC1( D3D12_RESOURCE_DIMENSION_BUFFER, resAllocInfo.Alignment, resAllocInfo.SizeInBytes,
            1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags, 0, 0, 0 );
    }
    static inline CD3DX12_RESOURCE_DESC1 Buffer(
        UINT64 width,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC1( D3D12_RESOURCE_DIMENSION_BUFFER, alignment, width, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags, 0, 0, 0 );
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex1D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC1( D3D12_RESOURCE_DIMENSION_TEXTURE1D, alignment, width, 1, arraySize,
            mipLevels, format, 1, 0, layout, flags, 0, 0, 0 );
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex2D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT height,
        UINT16 arraySize = 1,
        UINT16 mipLevels = 0,
        UINT sampleCount = 1,
        UINT sampleQuality = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0,
        UINT samplerFeedbackMipRegionWidth = 0,
        UINT samplerFeedbackMipRegionHeight = 0,
        UINT samplerFeedbackMipRegionDepth = 0) noexcept
    {
        return CD3DX12_RESOURCE_DESC1( D3D12_RESOURCE_DIMENSION_TEXTURE2D, alignment, width, height, arraySize,
            mipLevels, format, sampleCount, sampleQuality, layout, flags, samplerFeedbackMipRegionWidth,
            samplerFeedbackMipRegionHeight, samplerFeedbackMipRegionDepth );
    }
    static inline CD3DX12_RESOURCE_DESC1 Tex3D(
        DXGI_FORMAT format,
        UINT64 width,
        UINT height,
        UINT16 depth,
        UINT16 mipLevels = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
        UINT64 alignment = 0 ) noexcept
    {
        return CD3DX12_RESOURCE_DESC1( D3D12_RESOURCE_DIMENSION_TEXTURE3D, alignment, width, height, depth,
            mipLevels, format, 1, 0, layout, flags, 0, 0, 0 );
    }
    inline UINT16 Depth() const noexcept
    { return (Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1u); }
    inline UINT16 ArraySize() const noexcept
    { return (Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize : 1u); }
    inline UINT8 PlaneCount(_In_ ID3D12Device* pDevice) const noexcept
    { return D3D12GetFormatPlaneCount(pDevice, Format); }
    inline UINT Subresources(_In_ ID3D12Device* pDevice) const noexcept
    { return static_cast<UINT>(MipLevels) * ArraySize() * PlaneCount(pDevice); }
    inline UINT CalcSubresource(UINT MipSlice, UINT ArraySlice, UINT PlaneSlice) noexcept
    { return D3D12CalcSubresource(MipSlice, ArraySlice, PlaneSlice, MipLevels, ArraySize()); }
};
inline bool operator==( const D3D12_RESOURCE_DESC1& l, const D3D12_RESOURCE_DESC1& r ) noexcept
{
    return l.Dimension == r.Dimension &&
        l.Alignment == r.Alignment &&
        l.Width == r.Width &&
        l.Height == r.Height &&
        l.DepthOrArraySize == r.DepthOrArraySize &&
        l.MipLevels == r.MipLevels &&
        l.Format == r.Format &&
        l.SampleDesc.Count == r.SampleDesc.Count &&
        l.SampleDesc.Quality == r.SampleDesc.Quality &&
        l.Layout == r.Layout &&
        l.Flags == r.Flags &&
        l.SamplerFeedbackMipRegion.Width == r.SamplerFeedbackMipRegion.Width &&
        l.SamplerFeedbackMipRegion.Height == r.SamplerFeedbackMipRegion.Height &&
        l.SamplerFeedbackMipRegion.Depth == r.SamplerFeedbackMipRegion.Depth;
}
inline bool operator!=( const D3D12_RESOURCE_DESC1& l, const D3D12_RESOURCE_DESC1& r ) noexcept
{ return !( l == r ); }

//------------------------------------------------------------------------------------------------
struct CD3DX12_VIEW_INSTANCING_DESC : public D3D12_VIEW_INSTANCING_DESC
{
    CD3DX12_VIEW_INSTANCING_DESC() = default;
    explicit CD3DX12_VIEW_INSTANCING_DESC( const D3D12_VIEW_INSTANCING_DESC& o ) noexcept :
        D3D12_VIEW_INSTANCING_DESC( o )
    {}
    explicit CD3DX12_VIEW_INSTANCING_DESC( CD3DX12_DEFAULT ) noexcept
    {
        ViewInstanceCount = 0;
        pViewInstanceLocations = nullptr;
        Flags = D3D12_VIEW_INSTANCING_FLAG_NONE;
    }
    explicit CD3DX12_VIEW_INSTANCING_DESC(
        UINT InViewInstanceCount,
        const D3D12_VIEW_INSTANCE_LOCATION* InViewInstanceLocations,
        D3D12_VIEW_INSTANCING_FLAGS InFlags) noexcept
    {
        ViewInstanceCount = InViewInstanceCount;
        pViewInstanceLocations = InViewInstanceLocations;
        Flags = InFlags;
    }
};

//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
inline void MemcpySubresource(
    _In_ const D3D12_MEMCPY_DEST* pDest,
    _In_ const D3D12_SUBRESOURCE_DATA* pSrc,
    SIZE_T RowSizeInBytes,
    UINT NumRows,
    UINT NumSlices) noexcept
{
    for (UINT z = 0; z < NumSlices; ++z)
    {
        auto pDestSlice = static_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
        auto pSrcSlice = static_cast<const BYTE*>(pSrc->pData) + pSrc->SlicePitch * LONG_PTR(z);
        for (UINT y = 0; y < NumRows; ++y)
        {
            memcpy(pDestSlice + pDest->RowPitch * y,
                   pSrcSlice + pSrc->RowPitch * LONG_PTR(y),
                   RowSizeInBytes);
        }
    }
}

//------------------------------------------------------------------------------------------------
// Row-by-row memcpy
inline void MemcpySubresource(
    _In_ const D3D12_MEMCPY_DEST* pDest,
    _In_ const void* pResourceData,
    _In_ const D3D12_SUBRESOURCE_INFO* pSrc,
    SIZE_T RowSizeInBytes,
    UINT NumRows,
    UINT NumSlices) noexcept
{
    for (UINT z = 0; z < NumSlices; ++z)
    {
        auto pDestSlice = static_cast<BYTE*>(pDest->pData) + pDest->SlicePitch * z;
        auto pSrcSlice = (static_cast<const BYTE*>(pResourceData) + pSrc->Offset) + pSrc->DepthPitch * ULONG_PTR(z);
        for (UINT y = 0; y < NumRows; ++y)
        {
            memcpy(pDestSlice + pDest->RowPitch * y,
                pSrcSlice + pSrc->RowPitch * ULONG_PTR(y),
                RowSizeInBytes);
        }
    }
}

//------------------------------------------------------------------------------------------------
// Returns required size of a buffer to be used for data upload
inline UINT64 GetRequiredIntermediateSize(
    _In_ ID3D12Resource* pDestinationResource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource) UINT NumSubresources) noexcept
{
    const auto Desc = pDestinationResource->GetDesc();
    UINT64 RequiredSize = 0;

    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_ID3D12Device, reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, 0, nullptr, nullptr, nullptr, &RequiredSize);
    pDevice->Release();

    return RequiredSize;
}

//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    _In_range_(0,D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource) UINT NumSubresources,
    UINT64 RequiredSize,
    _In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    _In_reads_(NumSubresources) const UINT* pNumRows,
    _In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) noexcept
{
    // Minor validation
    const auto IntermediateDesc = pIntermediate->GetDesc();
    const auto DestinationDesc = pDestinationResource->GetDesc();
    if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset ||
        RequiredSize > SIZE_T(-1) ||
        (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
            (FirstSubresource != 0 || NumSubresources != 1)))
    {
        return 0;
    }

    BYTE* pData;
    HRESULT hr = pIntermediate->Map(0, nullptr, reinterpret_cast<void**>(&pData));
    if (FAILED(hr))
    {
        return 0;
    }

    for (UINT i = 0; i < NumSubresources; ++i)
    {
        if (pRowSizesInBytes[i] > SIZE_T(-1)) return 0;
        D3D12_MEMCPY_DEST DestData = { pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, SIZE_T(pLayouts[i].Footprint.RowPitch) * SIZE_T(pNumRows[i]) };
        MemcpySubresource(&DestData, &pSrcData[i], static_cast<SIZE_T>(pRowSizesInBytes[i]), pNumRows[i], pLayouts[i].Footprint.Depth);
    }
    pIntermediate->Unmap(0, nullptr);

    if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        pCmdList->CopyBufferRegion(
            pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
    }
    else
    {
        for (UINT i = 0; i < NumSubresources; ++i)
        {
            const CD3DX12_TEXTURE_COPY_LOCATION Dst(pDestinationResource, i + FirstSubresource);
            const CD3DX12_TEXTURE_COPY_LOCATION Src(pIntermediate, pLayouts[i]);
            pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
        }
    }
    return RequiredSize;
}

//------------------------------------------------------------------------------------------------
// All arrays must be populated (e.g. by calling GetCopyableFootprints)
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    _In_range_(0,D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource) UINT NumSubresources,
    UINT64 RequiredSize,
    _In_reads_(NumSubresources) const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    _In_reads_(NumSubresources) const UINT* pNumRows,
    _In_reads_(NumSubresources) const UINT64* pRowSizesInBytes,
    _In_ const void* pResourceData,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_INFO* pSrcData) noexcept
{
    // Minor validation
    const auto IntermediateDesc = pIntermediate->GetDesc();
    const auto DestinationDesc = pDestinationResource->GetDesc();
    if (IntermediateDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        IntermediateDesc.Width < RequiredSize + pLayouts[0].Offset ||
        RequiredSize > SIZE_T(-1) ||
        (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
            (FirstSubresource != 0 || NumSubresources != 1)))
    {
        return 0;
    }

    BYTE* pData;
    HRESULT hr = pIntermediate->Map(0, nullptr, reinterpret_cast<void**>(&pData));
    if (FAILED(hr))
    {
        return 0;
    }

    for (UINT i = 0; i < NumSubresources; ++i)
    {
        if (pRowSizesInBytes[i] > SIZE_T(-1)) return 0;
        D3D12_MEMCPY_DEST DestData = { pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, SIZE_T(pLayouts[i].Footprint.RowPitch) * SIZE_T(pNumRows[i]) };
        MemcpySubresource(&DestData, pResourceData, &pSrcData[i], static_cast<SIZE_T>(pRowSizesInBytes[i]), pNumRows[i], pLayouts[i].Footprint.Depth);
    }
    pIntermediate->Unmap(0, nullptr);

    if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        pCmdList->CopyBufferRegion(
            pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
    }
    else
    {
        for (UINT i = 0; i < NumSubresources; ++i)
        {
            const CD3DX12_TEXTURE_COPY_LOCATION Dst(pDestinationResource, i + FirstSubresource);
            const CD3DX12_TEXTURE_COPY_LOCATION Src(pIntermediate, pLayouts[i]);
            pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, nullptr);
        }
    }
    return RequiredSize;
}

//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0,D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource) UINT NumSubresources,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) noexcept
{
    UINT64 RequiredSize = 0;
    const auto MemToAlloc = static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)) * NumSubresources;
    if (MemToAlloc > SIZE_MAX)
    {
       return 0;
    }
    void* pMem = HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
    if (pMem == nullptr)
    {
       return 0;
    }
    auto pLayouts = static_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
    auto pRowSizesInBytes = reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
    auto pNumRows = reinterpret_cast<UINT*>(pRowSizesInBytes + NumSubresources);

    const auto Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_ID3D12Device, reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
    pDevice->Release();

    const UINT64 Result = UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, pLayouts, pNumRows, pRowSizesInBytes, pSrcData);
    HeapFree(GetProcessHeap(), 0, pMem);
    return Result;
}

//------------------------------------------------------------------------------------------------
// Heap-allocating UpdateSubresources implementation
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0,D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0,D3D12_REQ_SUBRESOURCES-FirstSubresource) UINT NumSubresources,
    _In_ const void* pResourceData,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_INFO* pSrcData) noexcept
{
    UINT64 RequiredSize = 0;
    const auto MemToAlloc = static_cast<UINT64>(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)) * NumSubresources;
    if (MemToAlloc > SIZE_MAX)
    {
        return 0;
    }
    void* pMem = HeapAlloc(GetProcessHeap(), 0, static_cast<SIZE_T>(MemToAlloc));
    if (pMem == nullptr)
    {
        return 0;
    }
    auto pLayouts = static_cast<D3D12_PLACED_SUBRESOURCE_FOOTPRINT*>(pMem);
    auto pRowSizesInBytes = reinterpret_cast<UINT64*>(pLayouts + NumSubresources);
    auto pNumRows = reinterpret_cast<UINT*>(pRowSizesInBytes + NumSubresources);

    const auto Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_ID3D12Device, reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
    pDevice->Release();

    const UINT64 Result = UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, pLayouts, pNumRows, pRowSizesInBytes, pResourceData, pSrcData);
    HeapFree(GetProcessHeap(), 0, pMem);
    return Result;
}

//------------------------------------------------------------------------------------------------
// Stack-allocating UpdateSubresources implementation
template <UINT MaxSubresources>
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0,MaxSubresources) UINT FirstSubresource,
    _In_range_(1,MaxSubresources-FirstSubresource) UINT NumSubresources,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_DATA* pSrcData) noexcept
{
    UINT64 RequiredSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
    UINT NumRows[MaxSubresources];
    UINT64 RowSizesInBytes[MaxSubresources];

    const auto Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_ID3D12Device, reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts, NumRows, RowSizesInBytes, &RequiredSize);
    pDevice->Release();

    return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, Layouts, NumRows, RowSizesInBytes, pSrcData);
}

//------------------------------------------------------------------------------------------------
// Stack-allocating UpdateSubresources implementation
template <UINT MaxSubresources>
inline UINT64 UpdateSubresources(
    _In_ ID3D12GraphicsCommandList* pCmdList,
    _In_ ID3D12Resource* pDestinationResource,
    _In_ ID3D12Resource* pIntermediate,
    UINT64 IntermediateOffset,
    _In_range_(0,MaxSubresources) UINT FirstSubresource,
    _In_range_(1,MaxSubresources-FirstSubresource) UINT NumSubresources,
    _In_ const void* pResourceData,
    _In_reads_(NumSubresources) const D3D12_SUBRESOURCE_INFO* pSrcData) noexcept
{
    UINT64 RequiredSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT Layouts[MaxSubresources];
    UINT NumRows[MaxSubresources];
    UINT64 RowSizesInBytes[MaxSubresources];

    const auto Desc = pDestinationResource->GetDesc();
    ID3D12Device* pDevice = nullptr;
    pDestinationResource->GetDevice(IID_ID3D12Device, reinterpret_cast<void**>(&pDevice));
    pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, IntermediateOffset, Layouts, NumRows, RowSizesInBytes, &RequiredSize);
    pDevice->Release();

    return UpdateSubresources(pCmdList, pDestinationResource, pIntermediate, FirstSubresource, NumSubresources, RequiredSize, Layouts, NumRows, RowSizesInBytes, pResourceData, pSrcData);
}

//------------------------------------------------------------------------------------------------
constexpr bool D3D12IsLayoutOpaque( D3D12_TEXTURE_LAYOUT Layout ) noexcept
{ return Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN || Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE; }

//------------------------------------------------------------------------------------------------
template <typename t_CommandListType>
inline ID3D12CommandList * const * CommandListCast(t_CommandListType * const * pp) noexcept
{
    // This cast is useful for passing strongly typed command list pointers into
    // ExecuteCommandLists.
    // This cast is valid as long as the const-ness is respected. D3D12 APIs do
    // respect the const-ness of their arguments.
    return reinterpret_cast<ID3D12CommandList * const *>(pp);
}

//------------------------------------------------------------------------------------------------
// D3D12 exports a new method for serializing root signatures in the Windows 10 Anniversary Update.
// To help enable root signature 1.1 features when they are available and not require maintaining
// two code paths for building root signatures, this helper method reconstructs a 1.0 signature when
// 1.1 is not supported.
inline HRESULT D3DX12SerializeVersionedRootSignature(
    _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignatureDesc,
    D3D_ROOT_SIGNATURE_VERSION MaxVersion,
    _Outptr_ ID3DBlob** ppBlob,
    _Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob) noexcept
{
    if (ppErrorBlob != nullptr)
    {
        *ppErrorBlob = nullptr;
    }

    switch (MaxVersion)
    {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            switch (pRootSignatureDesc->Version)
            {
                case D3D_ROOT_SIGNATURE_VERSION_1_0:
                    return D3D12SerializeRootSignature(&pRootSignatureDesc->Desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);

                case D3D_ROOT_SIGNATURE_VERSION_1_1:
                {
                    HRESULT hr = S_OK;
                    const D3D12_ROOT_SIGNATURE_DESC1& desc_1_1 = pRootSignatureDesc->Desc_1_1;

                    const SIZE_T ParametersSize = sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters;
                    void* pParameters = (ParametersSize > 0) ? HeapAlloc(GetProcessHeap(), 0, ParametersSize) : nullptr;
                    if (ParametersSize > 0 && pParameters == nullptr)
                    {
                        hr = E_OUTOFMEMORY;
                    }
                    auto pParameters_1_0 = static_cast<D3D12_ROOT_PARAMETER*>(pParameters);

                    if (SUCCEEDED(hr))
                    {
                        for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                        {
                            __analysis_assume(ParametersSize == sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters);
                            pParameters_1_0[n].ParameterType = desc_1_1.pParameters[n].ParameterType;
                            pParameters_1_0[n].ShaderVisibility = desc_1_1.pParameters[n].ShaderVisibility;

                            switch (desc_1_1.pParameters[n].ParameterType)
                            {
                            case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
                                pParameters_1_0[n].Constants.Num32BitValues = desc_1_1.pParameters[n].Constants.Num32BitValues;
                                pParameters_1_0[n].Constants.RegisterSpace = desc_1_1.pParameters[n].Constants.RegisterSpace;
                                pParameters_1_0[n].Constants.ShaderRegister = desc_1_1.pParameters[n].Constants.ShaderRegister;
                                break;

                            case D3D12_ROOT_PARAMETER_TYPE_CBV:
                            case D3D12_ROOT_PARAMETER_TYPE_SRV:
                            case D3D12_ROOT_PARAMETER_TYPE_UAV:
                                pParameters_1_0[n].Descriptor.RegisterSpace = desc_1_1.pParameters[n].Descriptor.RegisterSpace;
                                pParameters_1_0[n].Descriptor.ShaderRegister = desc_1_1.pParameters[n].Descriptor.ShaderRegister;
                                break;

                            case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
                                const D3D12_ROOT_DESCRIPTOR_TABLE1& table_1_1 = desc_1_1.pParameters[n].DescriptorTable;

                                const SIZE_T DescriptorRangesSize = sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges;
                                void* pDescriptorRanges = (DescriptorRangesSize > 0 && SUCCEEDED(hr)) ? HeapAlloc(GetProcessHeap(), 0, DescriptorRangesSize) : nullptr;
                                if (DescriptorRangesSize > 0 && pDescriptorRanges == nullptr)
                                {
                                    hr = E_OUTOFMEMORY;
                                }
                                auto pDescriptorRanges_1_0 = static_cast<D3D12_DESCRIPTOR_RANGE*>(pDescriptorRanges);

                                if (SUCCEEDED(hr))
                                {
                                    for (UINT x = 0; x < table_1_1.NumDescriptorRanges; x++)
                                    {
                                        __analysis_assume(DescriptorRangesSize == sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges);
                                        pDescriptorRanges_1_0[x].BaseShaderRegister = table_1_1.pDescriptorRanges[x].BaseShaderRegister;
                                        pDescriptorRanges_1_0[x].NumDescriptors = table_1_1.pDescriptorRanges[x].NumDescriptors;
                                        pDescriptorRanges_1_0[x].OffsetInDescriptorsFromTableStart = table_1_1.pDescriptorRanges[x].OffsetInDescriptorsFromTableStart;
                                        pDescriptorRanges_1_0[x].RangeType = table_1_1.pDescriptorRanges[x].RangeType;
                                        pDescriptorRanges_1_0[x].RegisterSpace = table_1_1.pDescriptorRanges[x].RegisterSpace;
                                    }
                                }

                                D3D12_ROOT_DESCRIPTOR_TABLE& table_1_0 = pParameters_1_0[n].DescriptorTable;
                                table_1_0.NumDescriptorRanges = table_1_1.NumDescriptorRanges;
                                table_1_0.pDescriptorRanges = pDescriptorRanges_1_0;
                            }
                        }
                    }

                    if (SUCCEEDED(hr))
                    {
                        const CD3DX12_ROOT_SIGNATURE_DESC desc_1_0(desc_1_1.NumParameters, pParameters_1_0, desc_1_1.NumStaticSamplers, desc_1_1.pStaticSamplers, desc_1_1.Flags);
                        hr = D3D12SerializeRootSignature(&desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
                    }

                    if (pParameters)
                    {
                        for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                        {
                            if (desc_1_1.pParameters[n].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                            {
                                auto pDescriptorRanges_1_0 = pParameters_1_0[n].DescriptorTable.pDescriptorRanges;
                                HeapFree(GetProcessHeap(), 0, reinterpret_cast<void*>(const_cast<D3D12_DESCRIPTOR_RANGE*>(pDescriptorRanges_1_0)));
                            }
                        }
                        HeapFree(GetProcessHeap(), 0, pParameters);
                    }
                    return hr;
                }
            }
            break;

        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            return D3D12SerializeVersionedRootSignature(pRootSignatureDesc, ppBlob, ppErrorBlob);
    }

    return E_INVALIDARG;
}

//------------------------------------------------------------------------------------------------
struct CD3DX12_RT_FORMAT_ARRAY : public D3D12_RT_FORMAT_ARRAY
{
    CD3DX12_RT_FORMAT_ARRAY() = default;
    explicit CD3DX12_RT_FORMAT_ARRAY(const D3D12_RT_FORMAT_ARRAY& o) noexcept
        : D3D12_RT_FORMAT_ARRAY(o)
    {}
    explicit CD3DX12_RT_FORMAT_ARRAY(_In_reads_(NumFormats) const DXGI_FORMAT* pFormats, UINT NumFormats) noexcept
    {
        NumRenderTargets = NumFormats;
        memcpy(RTFormats, pFormats, sizeof(RTFormats));
        // assumes ARRAY_SIZE(pFormats) == ARRAY_SIZE(RTFormats)
    }
};

//------------------------------------------------------------------------------------------------
// Pipeline State Stream Helpers
//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
// Stream Subobjects, i.e. elements of a stream

struct DefaultSampleMask { operator UINT() noexcept { return UINT_MAX; } };
struct DefaultSampleDesc { operator DXGI_SAMPLE_DESC() noexcept { return DXGI_SAMPLE_DESC{1, 0}; } };

#pragma warning(push)
#pragma warning(disable : 4324)
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
#pragma warning(pop)
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
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_DEPTH_STENCIL_DESC2,        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL2;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< DXGI_FORMAT,                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT>              CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_RASTERIZER_DESC,            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,     CD3DX12_DEFAULT>   CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_RT_FORMAT_ARRAY,              D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS>             CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< DXGI_SAMPLE_DESC,                   D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,    DefaultSampleDesc> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< UINT,                               D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,    DefaultSampleMask> CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_MASK;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< D3D12_CACHED_PIPELINE_STATE,        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO>                        CD3DX12_PIPELINE_STATE_STREAM_CACHED_PSO;
typedef CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT< CD3DX12_VIEW_INSTANCING_DESC,       D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, CD3DX12_DEFAULT>  CD3DX12_PIPELINE_STATE_STREAM_VIEW_INSTANCING;

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
    virtual void DepthStencilState2Cb(const D3D12_DEPTH_STENCIL_DESC2&) {}
    virtual void DSVFormatCb(DXGI_FORMAT) {}
    virtual void RasterizerStateCb(const D3D12_RASTERIZER_DESC&) {}
    virtual void RTVFormatsCb(const D3D12_RT_FORMAT_ARRAY&) {}
    virtual void SampleDescCb(const DXGI_SAMPLE_DESC&) {}
    virtual void SampleMaskCb(UINT) {}
    virtual void ViewInstancingCb(const D3D12_VIEW_INSTANCING_DESC&) {}
    virtual void CachedPSOCb(const D3D12_CACHED_PIPELINE_STATE&) {}

    // Error Callbacks
    virtual void ErrorBadInputParameter(UINT /*ParameterIndex*/) {}
    virtual void ErrorDuplicateSubobject(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE /*DuplicateType*/) {}
    virtual void ErrorUnknownSubobject(UINT /*UnknownTypeValue*/) {}

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
        D.Flags                 = this->Flags;
        D.NodeMask              = this->NodeMask;
        D.pRootSignature        = this->pRootSignature;
        D.PS                    = this->PS;
        D.AS                    = this->AS;
        D.MS                    = this->MS;
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
    case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2:
        return D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
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
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL:
            pCallbacks->DepthStencilStateCb(*reinterpret_cast<CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
            pCallbacks->DepthStencilState1Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM::DepthStencilState);
            break;
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2:
            pCallbacks->DepthStencilState2Cb(*reinterpret_cast<decltype(CD3DX12_PIPELINE_STATE_STREAM3::DepthStencilState)*>(pStream));
            SizeOfSubobject = sizeof(CD3DX12_PIPELINE_STATE_STREAM3::DepthStencilState);
            break;
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
        default:
            pCallbacks->ErrorUnknownSubobject(SubobjectType);
            return E_INVALIDARG;
        }
    }

    return S_OK;
}

//------------------------------------------------------------------------------------------------
inline bool operator==( const D3D12_CLEAR_VALUE &a, const D3D12_CLEAR_VALUE &b) noexcept
{
    if (a.Format != b.Format) return false;
    if (a.Format == DXGI_FORMAT_D24_UNORM_S8_UINT
     || a.Format == DXGI_FORMAT_D16_UNORM
     || a.Format == DXGI_FORMAT_D32_FLOAT
     || a.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
    {
        return (a.DepthStencil.Depth == b.DepthStencil.Depth) &&
          (a.DepthStencil.Stencil == b.DepthStencil.Stencil);
    } else {
        return (a.Color[0] == b.Color[0]) &&
               (a.Color[1] == b.Color[1]) &&
               (a.Color[2] == b.Color[2]) &&
               (a.Color[3] == b.Color[3]);
    }
}
inline bool operator==( const D3D12_RENDER_PASS_BEGINNING_ACCESS_CLEAR_PARAMETERS &a, const D3D12_RENDER_PASS_BEGINNING_ACCESS_CLEAR_PARAMETERS &b) noexcept
{
    return a.ClearValue == b.ClearValue;
}
inline bool operator==( const D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_PARAMETERS &a, const D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_PARAMETERS &b) noexcept
{
    if (a.pSrcResource != b.pSrcResource) return false;
    if (a.pDstResource != b.pDstResource) return false;
    if (a.SubresourceCount != b.SubresourceCount) return false;
    if (a.Format != b.Format) return false;
    if (a.ResolveMode != b.ResolveMode) return false;
    if (a.PreserveResolveSource != b.PreserveResolveSource) return false;
    return true;
}
inline bool operator==( const D3D12_RENDER_PASS_BEGINNING_ACCESS &a, const D3D12_RENDER_PASS_BEGINNING_ACCESS &b) noexcept
{
    if (a.Type != b.Type) return false;
    if (a.Type == D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR && !(a.Clear == b.Clear)) return false;
    return true;
}
inline bool operator==( const D3D12_RENDER_PASS_ENDING_ACCESS &a, const D3D12_RENDER_PASS_ENDING_ACCESS &b) noexcept
{
    if (a.Type != b.Type) return false;
    if (a.Type == D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE && !(a.Resolve == b.Resolve)) return false;
    return true;
}
inline bool operator==( const D3D12_RENDER_PASS_RENDER_TARGET_DESC &a, const D3D12_RENDER_PASS_RENDER_TARGET_DESC &b) noexcept
{
    if (a.cpuDescriptor.ptr != b.cpuDescriptor.ptr) return false;
    if (!(a.BeginningAccess == b.BeginningAccess)) return false;
    if (!(a.EndingAccess == b.EndingAccess)) return false;
    return true;
}
inline bool operator==( const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC &a, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC &b) noexcept
{
    if (a.cpuDescriptor.ptr != b.cpuDescriptor.ptr) return false;
    if (!(a.DepthBeginningAccess == b.DepthBeginningAccess)) return false;
    if (!(a.StencilBeginningAccess == b.StencilBeginningAccess)) return false;
    if (!(a.DepthEndingAccess == b.DepthEndingAccess)) return false;
    if (!(a.StencilEndingAccess == b.StencilEndingAccess)) return false;
    return true;
}


#ifndef D3DX12_NO_STATE_OBJECT_HELPERS

//================================================================================================
// D3DX12 State Object Creation Helpers
//
// Helper classes for creating new style state objects out of an arbitrary set of subobjects.
// Uses STL
//
// Start by instantiating CD3DX12_STATE_OBJECT_DESC (see its public methods).
// One of its methods is CreateSubobject(), which has a comment showing a couple of options for
// defining subobjects using the helper classes for each subobject (CD3DX12_DXIL_LIBRARY_SUBOBJECT
// etc.). The subobject helpers each have methods specific to the subobject for configuring its
// contents.
//
//================================================================================================
#include <list>
#include <memory>
#include <string>
#include <vector>
#ifndef D3DX12_USE_ATL
#include <wrl/client.h>
#define D3DX12_COM_PTR Microsoft::WRL::ComPtr
#define D3DX12_COM_PTR_GET(x) x.Get()
#define D3DX12_COM_PTR_ADDRESSOF(x) x.GetAddressOf()
#else
#include <atlbase.h>
#define D3DX12_COM_PTR ATL::CComPtr
#define D3DX12_COM_PTR_GET(x) x.p
#define D3DX12_COM_PTR_ADDRESSOF(x) &x.p
#endif

//------------------------------------------------------------------------------------------------
class CD3DX12_STATE_OBJECT_DESC
{
public:
    CD3DX12_STATE_OBJECT_DESC() noexcept
    {
        Init(D3D12_STATE_OBJECT_TYPE_COLLECTION);
    }
    CD3DX12_STATE_OBJECT_DESC(D3D12_STATE_OBJECT_TYPE Type) noexcept
    {
        Init(Type);
    }
    void SetStateObjectType(D3D12_STATE_OBJECT_TYPE Type) noexcept { m_Desc.Type = Type; }
    operator const D3D12_STATE_OBJECT_DESC&()
    {
        // Do final preparation work
        m_RepointedAssociations.clear();
        m_SubobjectArray.clear();
        m_SubobjectArray.reserve(m_Desc.NumSubobjects);
        // Flatten subobjects into an array (each flattened subobject still has a
        // member that's a pointer to its desc that's not flattened)
        for (auto Iter = m_SubobjectList.begin();
            Iter != m_SubobjectList.end(); Iter++)
        {
            m_SubobjectArray.push_back(*Iter);
            // Store new location in array so we can redirect pointers contained in subobjects
            Iter->pSubobjectArrayLocation = &m_SubobjectArray.back();
        }
        // For subobjects with pointer fields, create a new copy of those subobject definitions
        // with fixed pointers
        for (UINT i = 0; i < m_Desc.NumSubobjects; i++)
        {
            if (m_SubobjectArray[i].Type == D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION)
            {
                auto pOriginalSubobjectAssociation =
                    static_cast<const D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION*>(m_SubobjectArray[i].pDesc);
                D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION Repointed = *pOriginalSubobjectAssociation;
                auto pWrapper =
                    static_cast<const SUBOBJECT_WRAPPER*>(pOriginalSubobjectAssociation->pSubobjectToAssociate);
                Repointed.pSubobjectToAssociate = pWrapper->pSubobjectArrayLocation;
                m_RepointedAssociations.push_back(Repointed);
                m_SubobjectArray[i].pDesc = &m_RepointedAssociations.back();
            }
        }
        // Below: using ugly way to get pointer in case .data() is not defined
        m_Desc.pSubobjects = m_Desc.NumSubobjects ? &m_SubobjectArray[0] : nullptr;
        return m_Desc;
    }
    operator const D3D12_STATE_OBJECT_DESC*()
    {
        // Cast calls the above final preparation work
        return &static_cast<const D3D12_STATE_OBJECT_DESC&>(*this);
    }

    // CreateSubobject creates a sububject helper (e.g. CD3DX12_HIT_GROUP_SUBOBJECT)
    // whose lifetime is owned by this class.
    // e.g.
    //
    //    CD3DX12_STATE_OBJECT_DESC Collection1(D3D12_STATE_OBJECT_TYPE_COLLECTION);
    //    auto Lib0 = Collection1.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    //    Lib0->SetDXILLibrary(&pMyAppDxilLibs[0]);
    //    Lib0->DefineExport(L"rayGenShader0"); // in practice these export listings might be
    //                                          // data/engine driven
    //    etc.
    //
    // Alternatively, users can instantiate sububject helpers explicitly, such as via local
    // variables instead, passing the state object desc that should point to it into the helper
    // constructor (or call mySubobjectHelper.AddToStateObject(Collection1)).
    // In this alternative scenario, the user must keep the subobject alive as long as the state
    // object it is associated with is alive, else its pointer references will be stale.
    // e.g.
    //
    //    CD3DX12_STATE_OBJECT_DESC RaytracingState2(D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE);
    //    CD3DX12_DXIL_LIBRARY_SUBOBJECT LibA(RaytracingState2);
    //    LibA.SetDXILLibrary(&pMyAppDxilLibs[4]); // not manually specifying exports
    //                                             // - meaning all exports in the libraries
    //                                             // are exported
    //    etc.

    template<typename T>
    T* CreateSubobject()
    {
        T* pSubobject = new T(*this);
        m_OwnedSubobjectHelpers.emplace_back(pSubobject);
        return pSubobject;
    }

private:
    D3D12_STATE_SUBOBJECT* TrackSubobject(D3D12_STATE_SUBOBJECT_TYPE Type, void* pDesc)
    {
        SUBOBJECT_WRAPPER Subobject;
        Subobject.pSubobjectArrayLocation = nullptr;
        Subobject.Type = Type;
        Subobject.pDesc = pDesc;
        m_SubobjectList.push_back(Subobject);
        m_Desc.NumSubobjects++;
        return &m_SubobjectList.back();
    }
    void Init(D3D12_STATE_OBJECT_TYPE Type) noexcept
    {
        SetStateObjectType(Type);
        m_Desc.pSubobjects = nullptr;
        m_Desc.NumSubobjects = 0;
        m_SubobjectList.clear();
        m_SubobjectArray.clear();
        m_RepointedAssociations.clear();
    }
    typedef struct SUBOBJECT_WRAPPER : public D3D12_STATE_SUBOBJECT
    {
        D3D12_STATE_SUBOBJECT* pSubobjectArrayLocation; // new location when flattened into array
                                                        // for repointing pointers in subobjects
    } SUBOBJECT_WRAPPER;
    D3D12_STATE_OBJECT_DESC m_Desc;
    std::list<SUBOBJECT_WRAPPER>   m_SubobjectList; // Pointers to list nodes handed out so
                                                    // these can be edited live
    std::vector<D3D12_STATE_SUBOBJECT> m_SubobjectArray; // Built at the end, copying list contents

    std::list<D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION>
            m_RepointedAssociations; // subobject type that contains pointers to other subobjects,
                                     // repointed to flattened array

    class StringContainer
    {
    public:
        LPCWSTR LocalCopy(LPCWSTR string, bool bSingleString = false)
        {
            if (string)
            {
                if (bSingleString)
                {
                    m_Strings.clear();
                    m_Strings.push_back(string);
                }
                else
                {
                    m_Strings.push_back(string);
                }
                return m_Strings.back().c_str();
            }
            else
            {
                return nullptr;
            }
        }
        void clear() noexcept { m_Strings.clear(); }
    private:
        std::list<std::wstring> m_Strings;
    };

    class SUBOBJECT_HELPER_BASE
    {
    public:
        SUBOBJECT_HELPER_BASE() noexcept { Init(); }
        virtual ~SUBOBJECT_HELPER_BASE() = default;
        virtual D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept = 0;
        void AddToStateObject(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
        {
            m_pSubobject = ContainingStateObject.TrackSubobject(Type(), Data());
        }
    protected:
        virtual void* Data() noexcept = 0;
        void Init() noexcept { m_pSubobject = nullptr; }
        D3D12_STATE_SUBOBJECT* m_pSubobject;
    };

#if(__cplusplus >= 201103L)
    std::list<std::unique_ptr<const SUBOBJECT_HELPER_BASE>> m_OwnedSubobjectHelpers;
#else
    class OWNED_HELPER
    {
    public:
        OWNED_HELPER(const SUBOBJECT_HELPER_BASE* pHelper) noexcept { m_pHelper = pHelper; }
        ~OWNED_HELPER() { delete m_pHelper; }
        const SUBOBJECT_HELPER_BASE* m_pHelper;
    };

    std::list<OWNED_HELPER> m_OwnedSubobjectHelpers;
#endif

    friend class CD3DX12_DXIL_LIBRARY_SUBOBJECT;
    friend class CD3DX12_EXISTING_COLLECTION_SUBOBJECT;
    friend class CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT;
    friend class CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    friend class CD3DX12_HIT_GROUP_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT;
    friend class CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT;
    friend class CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT;
    friend class CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT;
    friend class CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT;
    friend class CD3DX12_NODE_MASK_SUBOBJECT;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_DXIL_LIBRARY_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_DXIL_LIBRARY_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_DXIL_LIBRARY_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetDXILLibrary(const D3D12_SHADER_BYTECODE* pCode) noexcept
    {
        static const D3D12_SHADER_BYTECODE Default = {};
        m_Desc.DXILLibrary = pCode ? *pCode : Default;
    }
    void DefineExport(
        LPCWSTR Name,
        LPCWSTR ExportToRename = nullptr,
        D3D12_EXPORT_FLAGS Flags = D3D12_EXPORT_FLAG_NONE)
    {
        D3D12_EXPORT_DESC Export;
        Export.Name = m_Strings.LocalCopy(Name);
        Export.ExportToRename = m_Strings.LocalCopy(ExportToRename);
        Export.Flags = Flags;
        m_Exports.push_back(Export);
        m_Desc.pExports = &m_Exports[0];  // using ugly way to get pointer in case .data() is not defined
        m_Desc.NumExports = static_cast<UINT>(m_Exports.size());
    }
    template<size_t N>
    void DefineExports(LPCWSTR(&Exports)[N])
    {
        for (UINT i = 0; i < N; i++)
        {
            DefineExport(Exports[i]);
        }
    }
    void DefineExports(const LPCWSTR* Exports, UINT N)
    {
        for (UINT i = 0; i < N; i++)
        {
            DefineExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_DXIL_LIBRARY_DESC&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_Exports.clear();
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_DXIL_LIBRARY_DESC m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    std::vector<D3D12_EXPORT_DESC> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_EXISTING_COLLECTION_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_EXISTING_COLLECTION_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_EXISTING_COLLECTION_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetExistingCollection(ID3D12StateObject*pExistingCollection) noexcept
    {
        m_Desc.pExistingCollection = pExistingCollection;
        m_CollectionRef = pExistingCollection;
    }
    void DefineExport(
        LPCWSTR Name,
        LPCWSTR ExportToRename = nullptr,
        D3D12_EXPORT_FLAGS Flags = D3D12_EXPORT_FLAG_NONE)
    {
        D3D12_EXPORT_DESC Export;
        Export.Name = m_Strings.LocalCopy(Name);
        Export.ExportToRename = m_Strings.LocalCopy(ExportToRename);
        Export.Flags = Flags;
        m_Exports.push_back(Export);
        m_Desc.pExports = &m_Exports[0]; // using ugly way to get pointer in case .data() is not defined
        m_Desc.NumExports = static_cast<UINT>(m_Exports.size());
    }
    template<size_t N>
    void DefineExports(LPCWSTR(&Exports)[N])
    {
        for (UINT i = 0; i < N; i++)
        {
            DefineExport(Exports[i]);
        }
    }
    void DefineExports(const LPCWSTR* Exports, UINT N)
    {
        for (UINT i = 0; i < N; i++)
        {
            DefineExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_EXISTING_COLLECTION_DESC&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_CollectionRef = nullptr;
        m_Strings.clear();
        m_Exports.clear();
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_EXISTING_COLLECTION_DESC m_Desc;
    D3DX12_COM_PTR<ID3D12StateObject> m_CollectionRef;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    std::vector<D3D12_EXPORT_DESC> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetSubobjectToAssociate(const D3D12_STATE_SUBOBJECT& SubobjectToAssociate) noexcept
    {
        m_Desc.pSubobjectToAssociate = &SubobjectToAssociate;
    }
    void AddExport(LPCWSTR Export)
    {
        m_Desc.NumExports++;
        m_Exports.push_back(m_Strings.LocalCopy(Export));
        m_Desc.pExports = &m_Exports[0];  // using ugly way to get pointer in case .data() is not defined
    }
    template<size_t N>
    void AddExports(LPCWSTR (&Exports)[N])
    {
        for (UINT i = 0; i < N; i++)
        {
            AddExport(Exports[i]);
        }
    }
    void AddExports(const LPCWSTR* Exports, UINT N)
    {
        for (UINT i = 0; i < N; i++)
        {
            AddExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_Exports.clear();
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    std::vector<LPCWSTR> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION() noexcept
    {
        Init();
    }
    CD3DX12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetSubobjectNameToAssociate(LPCWSTR SubobjectToAssociate)
    {
        m_Desc.SubobjectToAssociate = m_SubobjectName.LocalCopy(SubobjectToAssociate, true);
    }
    void AddExport(LPCWSTR Export)
    {
        m_Desc.NumExports++;
        m_Exports.push_back(m_Strings.LocalCopy(Export));
        m_Desc.pExports = &m_Exports[0];  // using ugly way to get pointer in case .data() is not defined
    }
    template<size_t N>
    void AddExports(LPCWSTR (&Exports)[N])
    {
        for (UINT i = 0; i < N; i++)
        {
            AddExport(Exports[i]);
        }
    }
    void AddExports(const LPCWSTR* Exports, UINT N)
    {
        for (UINT i = 0; i < N; i++)
        {
            AddExport(Exports[i]);
        }
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        m_Strings.clear();
        m_SubobjectName.clear();
        m_Exports.clear();
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION m_Desc;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_Strings;
    CD3DX12_STATE_OBJECT_DESC::StringContainer m_SubobjectName;
    std::vector<LPCWSTR> m_Exports;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_HIT_GROUP_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_HIT_GROUP_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_HIT_GROUP_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetHitGroupExport(LPCWSTR exportName)
    {
        m_Desc.HitGroupExport = m_Strings[0].LocalCopy(exportName, true);
    }
    void SetHitGroupType(D3D12_HIT_GROUP_TYPE Type) noexcept { m_Desc.Type = Type; }
    void SetAnyHitShaderImport(LPCWSTR importName)
    {
        m_Desc.AnyHitShaderImport = m_Strings[1].LocalCopy(importName, true);
    }
    void SetClosestHitShaderImport(LPCWSTR importName)
    {
        m_Desc.ClosestHitShaderImport = m_Strings[2].LocalCopy(importName, true);
    }
    void SetIntersectionShaderImport(LPCWSTR importName)
    {
        m_Desc.IntersectionShaderImport = m_Strings[3].LocalCopy(importName, true);
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_HIT_GROUP_DESC&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
        for (UINT i = 0; i < m_NumStrings; i++)
        {
            m_Strings[i].clear();
        }
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_HIT_GROUP_DESC m_Desc;
    static constexpr UINT m_NumStrings = 4;
    CD3DX12_STATE_OBJECT_DESC::StringContainer
        m_Strings[m_NumStrings]; // one string for every entrypoint name
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(UINT MaxPayloadSizeInBytes, UINT MaxAttributeSizeInBytes) noexcept
    {
        m_Desc.MaxPayloadSizeInBytes = MaxPayloadSizeInBytes;
        m_Desc.MaxAttributeSizeInBytes = MaxAttributeSizeInBytes;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_SHADER_CONFIG&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_SHADER_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(UINT MaxTraceRecursionDepth) noexcept
    {
        m_Desc.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_PIPELINE_CONFIG&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_PIPELINE_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_RAYTRACING_PIPELINE_CONFIG1_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void Config(UINT MaxTraceRecursionDepth, D3D12_RAYTRACING_PIPELINE_FLAGS Flags) noexcept
    {
        m_Desc.MaxTraceRecursionDepth = MaxTraceRecursionDepth;
        m_Desc.Flags = Flags;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_RAYTRACING_PIPELINE_CONFIG1&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_RAYTRACING_PIPELINE_CONFIG1 m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetRootSignature(ID3D12RootSignature* pRootSig) noexcept
    {
        m_pRootSig = pRootSig;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator ID3D12RootSignature*() const noexcept { return D3DX12_COM_PTR_GET(m_pRootSig); }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_pRootSig = nullptr;
    }
    void* Data() noexcept override { return D3DX12_COM_PTR_ADDRESSOF(m_pRootSig); }
    D3DX12_COM_PTR<ID3D12RootSignature> m_pRootSig;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetRootSignature(ID3D12RootSignature* pRootSig) noexcept
    {
        m_pRootSig = pRootSig;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator ID3D12RootSignature*() const noexcept { return D3DX12_COM_PTR_GET(m_pRootSig); }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_pRootSig = nullptr;
    }
    void* Data() noexcept override { return D3DX12_COM_PTR_ADDRESSOF(m_pRootSig); }
    D3DX12_COM_PTR<ID3D12RootSignature> m_pRootSig;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_STATE_OBJECT_CONFIG_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetFlags(D3D12_STATE_OBJECT_FLAGS Flags) noexcept
    {
        m_Desc.Flags = Flags;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_STATE_OBJECT_CONFIG;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_STATE_OBJECT_CONFIG&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_STATE_OBJECT_CONFIG m_Desc;
};

//------------------------------------------------------------------------------------------------
class CD3DX12_NODE_MASK_SUBOBJECT
    : public CD3DX12_STATE_OBJECT_DESC::SUBOBJECT_HELPER_BASE
{
public:
    CD3DX12_NODE_MASK_SUBOBJECT() noexcept
    {
        Init();
    }
    CD3DX12_NODE_MASK_SUBOBJECT(CD3DX12_STATE_OBJECT_DESC& ContainingStateObject)
    {
        Init();
        AddToStateObject(ContainingStateObject);
    }
    void SetNodeMask(UINT NodeMask) noexcept
    {
        m_Desc.NodeMask = NodeMask;
    }
    D3D12_STATE_SUBOBJECT_TYPE Type() const noexcept override
    {
        return D3D12_STATE_SUBOBJECT_TYPE_NODE_MASK;
    }
    operator const D3D12_STATE_SUBOBJECT&() const noexcept { return *m_pSubobject; }
    operator const D3D12_NODE_MASK&() const noexcept { return m_Desc; }
private:
    void Init() noexcept
    {
        SUBOBJECT_HELPER_BASE::Init();
        m_Desc = {};
    }
    void* Data() noexcept override { return &m_Desc; }
    D3D12_NODE_MASK m_Desc;
};

#endif // !D3DX12_NO_STATE_OBJECT_HELPERS


//================================================================================================
// D3DX12 Enhanced Barrier Helpers
//================================================================================================

class CD3DX12_BARRIER_SUBRESOURCE_RANGE : public D3D12_BARRIER_SUBRESOURCE_RANGE
{
public:
    CD3DX12_BARRIER_SUBRESOURCE_RANGE() = default;
    CD3DX12_BARRIER_SUBRESOURCE_RANGE(const D3D12_BARRIER_SUBRESOURCE_RANGE &o) noexcept :
        D3D12_BARRIER_SUBRESOURCE_RANGE(o)
    {}
    explicit CD3DX12_BARRIER_SUBRESOURCE_RANGE(UINT Subresource) noexcept :
        D3D12_BARRIER_SUBRESOURCE_RANGE{ Subresource, 0, 0, 0, 0, 0 }
    {}
    CD3DX12_BARRIER_SUBRESOURCE_RANGE(
        UINT FirstMipLevel,
        UINT NumMips,
        UINT FirstArraySlice,
        UINT NumArraySlices,
        UINT FirstPlane = 0,
        UINT NumPlanes = 1) noexcept :
        D3D12_BARRIER_SUBRESOURCE_RANGE
        {
            FirstMipLevel,
            NumMips,
            FirstArraySlice,
            NumArraySlices,
            FirstPlane,
            NumPlanes
        }
    {}
};

class CD3DX12_GLOBAL_BARRIER : public D3D12_GLOBAL_BARRIER
{
public:
    CD3DX12_GLOBAL_BARRIER() = default;
    CD3DX12_GLOBAL_BARRIER(const D3D12_GLOBAL_BARRIER &o) noexcept : D3D12_GLOBAL_BARRIER(o){}
    CD3DX12_GLOBAL_BARRIER(
        D3D12_BARRIER_SYNC syncBefore,
        D3D12_BARRIER_SYNC syncAfter,
        D3D12_BARRIER_ACCESS accessBefore,
        D3D12_BARRIER_ACCESS accessAfter) noexcept : D3D12_GLOBAL_BARRIER {
            syncBefore,
            syncAfter,
            accessBefore,
            accessAfter
        }
    {}
};

class CD3DX12_BUFFER_BARRIER : public D3D12_BUFFER_BARRIER
{
public:
    CD3DX12_BUFFER_BARRIER() = default;
    CD3DX12_BUFFER_BARRIER(const D3D12_BUFFER_BARRIER &o) noexcept : D3D12_BUFFER_BARRIER(o){}
    CD3DX12_BUFFER_BARRIER(
        D3D12_BARRIER_SYNC syncBefore,
        D3D12_BARRIER_SYNC syncAfter,
        D3D12_BARRIER_ACCESS accessBefore,
        D3D12_BARRIER_ACCESS accessAfter,
        ID3D12Resource *pRes) noexcept : D3D12_BUFFER_BARRIER {
            syncBefore,
            syncAfter,
            accessBefore,
            accessAfter,
            pRes,
            0, ULLONG_MAX
        }
    {}
};

class CD3DX12_TEXTURE_BARRIER : public D3D12_TEXTURE_BARRIER
{
public:
    CD3DX12_TEXTURE_BARRIER() = default;
    CD3DX12_TEXTURE_BARRIER(const D3D12_TEXTURE_BARRIER &o) noexcept : D3D12_TEXTURE_BARRIER(o){}
    CD3DX12_TEXTURE_BARRIER(
        D3D12_BARRIER_SYNC syncBefore,
        D3D12_BARRIER_SYNC syncAfter,
        D3D12_BARRIER_ACCESS accessBefore,
        D3D12_BARRIER_ACCESS accessAfter,
        D3D12_BARRIER_LAYOUT layoutBefore,
        D3D12_BARRIER_LAYOUT layoutAfter,
        ID3D12Resource *pRes,
        const D3D12_BARRIER_SUBRESOURCE_RANGE &subresources,
        D3D12_TEXTURE_BARRIER_FLAGS flag = D3D12_TEXTURE_BARRIER_FLAG_NONE) noexcept : D3D12_TEXTURE_BARRIER {
            syncBefore,
            syncAfter,
            accessBefore,
            accessAfter,
            layoutBefore,
            layoutAfter,
            pRes,
            subresources,
            flag
        }
    {}
};

class CD3DX12_BARRIER_GROUP : public D3D12_BARRIER_GROUP
{
public:
    CD3DX12_BARRIER_GROUP() = default;
    CD3DX12_BARRIER_GROUP(const D3D12_BARRIER_GROUP &o) noexcept : D3D12_BARRIER_GROUP(o){}
    CD3DX12_BARRIER_GROUP(UINT32 numBarriers, const D3D12_BUFFER_BARRIER *pBarriers) noexcept
    {
        Type = D3D12_BARRIER_TYPE_BUFFER;
        NumBarriers = numBarriers;
        pBufferBarriers = pBarriers;
    }
    CD3DX12_BARRIER_GROUP(UINT32 numBarriers, const D3D12_TEXTURE_BARRIER *pBarriers) noexcept
    {
        Type = D3D12_BARRIER_TYPE_TEXTURE;
        NumBarriers = numBarriers;
        pTextureBarriers = pBarriers;
    }
    CD3DX12_BARRIER_GROUP(UINT32 numBarriers, const D3D12_GLOBAL_BARRIER *pBarriers) noexcept
    {
        Type = D3D12_BARRIER_TYPE_GLOBAL;
        NumBarriers = numBarriers;
        pGlobalBarriers = pBarriers;
    }
};


#ifndef D3DX12_NO_CHECK_FEATURE_SUPPORT_CLASS

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

    // DISPLAYABLE
    BOOL DisplayableTexture() const noexcept;
    // SharedResourceCompatibilityTier handled in D3D12Options4

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

    // D3D12_OPTIONS8
    BOOL UnalignedBlockTexturesSupported() const noexcept;

    // D3D12_OPTIONS9
    BOOL MeshShaderPipelineStatsSupported() const noexcept;
    BOOL MeshShaderSupportsFullRangeRenderTargetArrayIndex() const noexcept;
    BOOL AtomicInt64OnTypedResourceSupported() const noexcept;
    BOOL AtomicInt64OnGroupSharedSupported() const noexcept;
    BOOL DerivativesInMeshAndAmplificationShadersSupported() const noexcept;
    D3D12_WAVE_MMA_TIER WaveMMATier() const noexcept;

    // D3D12_OPTIONS10
    BOOL VariableRateShadingSumCombinerSupported() const noexcept;
    BOOL MeshShaderPerPrimitiveShadingRateSupported() const noexcept;

    // D3D12_OPTIONS11
    BOOL AtomicInt64OnDescriptorHeapResourceSupported() const noexcept;

    // D3D12_OPTIONS12
    D3D12_TRI_STATE MSPrimitivesPipelineStatisticIncludesCulledPrimitives() const noexcept;
    BOOL EnhancedBarriersSupported() const noexcept;
    BOOL RelaxedFormatCastingSupported() const noexcept;

    // D3D12_OPTIONS13
    BOOL UnrestrictedBufferTextureCopyPitchSupported() const noexcept;
    BOOL UnrestrictedVertexElementAlignmentSupported() const noexcept;
    BOOL InvertedViewportHeightFlipsYSupported() const noexcept;
    BOOL InvertedViewportDepthFlipsZSupported() const noexcept;
    BOOL TextureCopyBetweenDimensionsSupported() const noexcept;
    BOOL AlphaBlendFactorSupported() const noexcept;

    // D3D12_OPTIONS14
    BOOL AdvancedTextureOpsSupported() const noexcept;
    BOOL WriteableMSAATexturesSupported() const noexcept;
    BOOL IndependentFrontAndBackStencilRefMaskSupported() const noexcept;

    // D3D12_OPTIONS15
    BOOL TriangleFanSupported() const noexcept;
    BOOL DynamicIndexBufferStripCutSupported() const noexcept;

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
    D3D12_FEATURE_DATA_DISPLAYABLE m_dDisplayable;
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 m_dOptions6;
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 m_dOptions7;
    std::vector<D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT> m_dProtectedResourceSessionTypeCount; // Cat2 NodeIndex
    std::vector<ProtectedResourceSessionTypesLocal> m_dProtectedResourceSessionTypes; // Cat3
    D3D12_FEATURE_DATA_D3D12_OPTIONS8 m_dOptions8;
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 m_dOptions9;
    D3D12_FEATURE_DATA_D3D12_OPTIONS10 m_dOptions10;
    D3D12_FEATURE_DATA_D3D12_OPTIONS11 m_dOptions11;
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 m_dOptions12;
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 m_dOptions13;
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 m_dOptions14;
    D3D12_FEATURE_DATA_D3D12_OPTIONS15 m_dOptions15;
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
, m_dDisplayable{}
, m_dOptions6{}
, m_dOptions7{}
, m_dOptions8{}
, m_dOptions9{}
, m_dOptions10{}
, m_dOptions11{}
, m_dOptions12{}
, m_dOptions13{}
, m_dOptions14{}
, m_dOptions15{}
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
        m_dOptions.DoublePrecisionFloatShaderOps = false;
        m_dOptions.OutputMergerLogicOp = false;
        m_dOptions.MinPrecisionSupport = D3D12_SHADER_MIN_PRECISION_SUPPORT_NONE;
        m_dOptions.TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
        m_dOptions.ResourceBindingTier = static_cast<D3D12_RESOURCE_BINDING_TIER>(0);
        m_dOptions.PSSpecifiedStencilRefSupported = false;
        m_dOptions.TypedUAVLoadAdditionalFormats = false;
        m_dOptions.ROVsSupported = false;
        m_dOptions.ConservativeRasterizationTier = D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
        m_dOptions.MaxGPUVirtualAddressBitsPerResource = 0;
        m_dOptions.StandardSwizzle64KBSupported = false;
        m_dOptions.CrossNodeSharingTier = D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED;
        m_dOptions.CrossAdapterRowMajorTextureSupported = false;
        m_dOptions.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation = false;
        m_dOptions.ResourceHeapTier = static_cast<D3D12_RESOURCE_HEAP_TIER>(0);
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &m_dGPUVASupport, sizeof(m_dGPUVASupport))))
    {
        m_dGPUVASupport.MaxGPUVirtualAddressBitsPerProcess = 0;
        m_dGPUVASupport.MaxGPUVirtualAddressBitsPerResource = 0;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &m_dOptions1, sizeof(m_dOptions1))))
    {
        m_dOptions1.WaveOps = false;
        m_dOptions1.WaveLaneCountMax = 0;
        m_dOptions1.WaveLaneCountMin = 0;
        m_dOptions1.TotalLaneCount = 0;
        m_dOptions1.ExpandedComputeResourceStates = 0;
        m_dOptions1.Int64ShaderOps = 0;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &m_dOptions2, sizeof(m_dOptions2))))
    {
        m_dOptions2.DepthBoundsTestSupported = false;
        m_dOptions2.ProgrammableSamplePositionsTier = D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_CACHE, &m_dShaderCache, sizeof(m_dShaderCache))))
    {
        m_dShaderCache.SupportFlags = D3D12_SHADER_CACHE_SUPPORT_NONE;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &m_dOptions3, sizeof(m_dOptions3))))
    {
        m_dOptions3.CopyQueueTimestampQueriesSupported = false;
        m_dOptions3.CastingFullyTypedFormatSupported = false;
        m_dOptions3.WriteBufferImmediateSupportFlags = D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE;
        m_dOptions3.ViewInstancingTier = D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED;
        m_dOptions3.BarycentricsSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &m_dExistingHeaps, sizeof(m_dExistingHeaps))))
    {
        m_dExistingHeaps.Supported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &m_dOptions4, sizeof(m_dOptions4))))
    {
        m_dOptions4.MSAA64KBAlignedTextureSupported = false;
        m_dOptions4.Native16BitShaderOpsSupported = false;
        m_dOptions4.SharedResourceCompatibilityTier = D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_CROSS_NODE, &m_dCrossNode, sizeof(m_dCrossNode))))
    {
        m_dCrossNode.SharingTier = D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED;
        m_dCrossNode.AtomicShaderInstructions = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &m_dOptions5, sizeof(m_dOptions5))))
    {
        m_dOptions5.SRVOnlyTiledResourceTier3 = false;
        m_dOptions5.RenderPassesTier = D3D12_RENDER_PASS_TIER_0;
        m_dOptions5.RaytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_DISPLAYABLE, &m_dDisplayable, sizeof(m_dDisplayable))))
    {
        m_dDisplayable.DisplayableTexture = false;
        m_dDisplayable.SharedResourceCompatibilityTier = D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &m_dOptions6, sizeof(m_dOptions6))))
    {
        m_dOptions6.AdditionalShadingRatesSupported = false;
        m_dOptions6.PerPrimitiveShadingRateSupportedWithViewportIndexing = false;
        m_dOptions6.VariableShadingRateTier = D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED;
        m_dOptions6.ShadingRateImageTileSize = 0;
        m_dOptions6.BackgroundProcessingSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &m_dOptions7, sizeof(m_dOptions7))))
    {
        m_dOptions7.MeshShaderTier = D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
        m_dOptions7.SamplerFeedbackTier = D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &m_dOptions8, sizeof(m_dOptions8))))
    {
        m_dOptions8.UnalignedBlockTexturesSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &m_dOptions9, sizeof(m_dOptions9))))
    {
        m_dOptions9.MeshShaderPipelineStatsSupported = false;
        m_dOptions9.MeshShaderSupportsFullRangeRenderTargetArrayIndex = false;
        m_dOptions9.AtomicInt64OnGroupSharedSupported = false;
        m_dOptions9.AtomicInt64OnTypedResourceSupported = false;
        m_dOptions9.DerivativesInMeshAndAmplificationShadersSupported = false;
        m_dOptions9.WaveMMATier = D3D12_WAVE_MMA_TIER_NOT_SUPPORTED;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS10, &m_dOptions10, sizeof(m_dOptions10))))
    {
        m_dOptions10.MeshShaderPerPrimitiveShadingRateSupported = false;
        m_dOptions10.VariableRateShadingSumCombinerSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &m_dOptions11, sizeof(m_dOptions11))))
    {
        m_dOptions11.AtomicInt64OnDescriptorHeapResourceSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &m_dOptions12, sizeof(m_dOptions12))))
    {
        m_dOptions12.MSPrimitivesPipelineStatisticIncludesCulledPrimitives = D3D12_TRI_STATE::D3D12_TRI_STATE_UNKNOWN;
        m_dOptions12.EnhancedBarriersSupported = false;
        m_dOptions12.RelaxedFormatCastingSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS13, &m_dOptions13, sizeof(m_dOptions13))))
    {
        m_dOptions13.UnrestrictedBufferTextureCopyPitchSupported = false;
        m_dOptions13.UnrestrictedVertexElementAlignmentSupported = false;
        m_dOptions13.InvertedViewportHeightFlipsYSupported = false;
        m_dOptions13.InvertedViewportDepthFlipsZSupported = false;
        m_dOptions13.TextureCopyBetweenDimensionsSupported = false;
        m_dOptions13.AlphaBlendFactorSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS14, &m_dOptions14, sizeof(m_dOptions14))))
    {
        m_dOptions14.AdvancedTextureOpsSupported = false;
        m_dOptions14.WriteableMSAATexturesSupported = false;
        m_dOptions14.IndependentFrontAndBackStencilRefMaskSupported = false;
    }

    if (FAILED(m_pDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15, &m_dOptions15, sizeof(m_dOptions15))))
    {
        m_dOptions15.TriangleFanSupported = false;
        m_dOptions15.DynamicIndexBufferStripCutSupported = false;
    }

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

// 28: Displayable
FEATURE_SUPPORT_GET(BOOL, m_dDisplayable, DisplayableTexture);
// SharedResourceCompatibilityTier handled in D3D12Options4

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

// 36: Options8
FEATURE_SUPPORT_GET(BOOL, m_dOptions8, UnalignedBlockTexturesSupported);

// 37: Options9
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, MeshShaderPipelineStatsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, MeshShaderSupportsFullRangeRenderTargetArrayIndex);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, AtomicInt64OnTypedResourceSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, AtomicInt64OnGroupSharedSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions9, DerivativesInMeshAndAmplificationShadersSupported);
FEATURE_SUPPORT_GET(D3D12_WAVE_MMA_TIER, m_dOptions9, WaveMMATier);

// 39: Options10
FEATURE_SUPPORT_GET(BOOL, m_dOptions10, VariableRateShadingSumCombinerSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions10, MeshShaderPerPrimitiveShadingRateSupported);

// 40: Options11
FEATURE_SUPPORT_GET(BOOL, m_dOptions11, AtomicInt64OnDescriptorHeapResourceSupported);

// 41: Options12
FEATURE_SUPPORT_GET(D3D12_TRI_STATE, m_dOptions12, MSPrimitivesPipelineStatisticIncludesCulledPrimitives);
FEATURE_SUPPORT_GET(BOOL, m_dOptions12, EnhancedBarriersSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions12, RelaxedFormatCastingSupported);

// 42: Options13
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, UnrestrictedBufferTextureCopyPitchSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, UnrestrictedVertexElementAlignmentSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, InvertedViewportHeightFlipsYSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, InvertedViewportDepthFlipsZSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, TextureCopyBetweenDimensionsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions13, AlphaBlendFactorSupported);

// 43: Options14
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, AdvancedTextureOpsSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, WriteableMSAATexturesSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions14, IndependentFrontAndBackStencilRefMaskSupported);

// 44: Options15
FEATURE_SUPPORT_GET(BOOL, m_dOptions15, TriangleFanSupported);
FEATURE_SUPPORT_GET(BOOL, m_dOptions15, DynamicIndexBufferStripCutSupported);

// Helper function to decide the highest shader model supported by the system
// Stores the result in m_dShaderModel
// Must be updated whenever a new shader model is added to the d3d12.h header
inline HRESULT CD3DX12FeatureSupport::QueryHighestShaderModel()
{
    // Check support in descending order
    HRESULT result;

    const D3D_SHADER_MODEL allModelVersions[] =
    {
        D3D_SHADER_MODEL_6_8,
        D3D_SHADER_MODEL_6_7,
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
        D3D_FEATURE_LEVEL_12_2,
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1,
        D3D_FEATURE_LEVEL_1_0_CORE
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

#endif // !D3DX12_NO_CHECK_FEATURE_SUPPORT_CLASS

#undef D3DX12_COM_PTR
#undef D3DX12_COM_PTR_GET
#undef D3DX12_COM_PTR_ADDRESSOF

#endif // defined( __cplusplus )

#endif //__D3DX12_H__

