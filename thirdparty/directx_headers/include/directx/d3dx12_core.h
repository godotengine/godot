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

#include <string.h>
#include "d3d12.h"
#include "d3dx12_default.h"

//------------------------------------------------------------------------------------------------
#ifndef D3DX12_ASSERT
  #ifdef assert
    #define D3DX12_ASSERT(x) assert(x)
  #else
    #define D3DX12_ASSERT(x)
  #endif
#endif

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
#if defined(_MSC_VER) || !defined(_WIN32)
        const auto Desc = pResource->GetDesc();
#else
        D3D12_RESOURCE_DESC tmpDesc;
        const auto& Desc = *pResource->GetDesc(&tmpDesc);
#endif
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
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
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
#endif // D3D12_SDK_VERSION >= 606

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
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 608)
struct CD3DX12_RASTERIZER_DESC1 : public D3D12_RASTERIZER_DESC1
{
    CD3DX12_RASTERIZER_DESC1() = default;
    explicit CD3DX12_RASTERIZER_DESC1(const D3D12_RASTERIZER_DESC1& o) noexcept :
        D3D12_RASTERIZER_DESC1(o)

    {
    }
    explicit CD3DX12_RASTERIZER_DESC1(const D3D12_RASTERIZER_DESC& o) noexcept
    {
        FillMode = o.FillMode;
        CullMode = o.CullMode;
        FrontCounterClockwise = o.FrontCounterClockwise;
        DepthBias = static_cast<FLOAT>(o.DepthBias);
        DepthBiasClamp = o.DepthBiasClamp;
        SlopeScaledDepthBias = o.SlopeScaledDepthBias;
        DepthClipEnable = o.DepthClipEnable;
        MultisampleEnable = o.MultisampleEnable;
        AntialiasedLineEnable = o.AntialiasedLineEnable;
        ForcedSampleCount = o.ForcedSampleCount;
        ConservativeRaster = o.ConservativeRaster;
    }
    explicit CD3DX12_RASTERIZER_DESC1(CD3DX12_DEFAULT) noexcept
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
    explicit CD3DX12_RASTERIZER_DESC1(
        D3D12_FILL_MODE fillMode,
        D3D12_CULL_MODE cullMode,
        BOOL frontCounterClockwise,
        FLOAT depthBias,
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


    operator D3D12_RASTERIZER_DESC() const noexcept
    {
        D3D12_RASTERIZER_DESC o;

        o.FillMode = FillMode;
        o.CullMode = CullMode;
        o.FrontCounterClockwise = FrontCounterClockwise;
        o.DepthBias = static_cast<INT>(DepthBias);
        o.DepthBiasClamp = DepthBiasClamp;
        o.SlopeScaledDepthBias = SlopeScaledDepthBias;
        o.DepthClipEnable = DepthClipEnable;
        o.MultisampleEnable = MultisampleEnable;
        o.AntialiasedLineEnable = AntialiasedLineEnable;
        o.ForcedSampleCount = ForcedSampleCount;
        o.ConservativeRaster = ConservativeRaster;

        return o;
    }
};
#endif // D3D12_SDK_VERSION >= 608

//------------------------------------------------------------------------------------------------
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 610)
struct CD3DX12_RASTERIZER_DESC2 : public D3D12_RASTERIZER_DESC2
{
    CD3DX12_RASTERIZER_DESC2() = default;
    explicit CD3DX12_RASTERIZER_DESC2(const D3D12_RASTERIZER_DESC2& o) noexcept :
        D3D12_RASTERIZER_DESC2(o)

    {
    }
    explicit CD3DX12_RASTERIZER_DESC2(const D3D12_RASTERIZER_DESC1& o) noexcept
    {
        FillMode = o.FillMode;
        CullMode = o.CullMode;
        FrontCounterClockwise = o.FrontCounterClockwise;
        DepthBias = o.DepthBias;
        DepthBiasClamp = o.DepthBiasClamp;
        SlopeScaledDepthBias = o.SlopeScaledDepthBias;
        DepthClipEnable = o.DepthClipEnable;
        LineRasterizationMode = D3D12_LINE_RASTERIZATION_MODE_ALIASED;
        if (o.MultisampleEnable)
        {
            LineRasterizationMode = D3D12_LINE_RASTERIZATION_MODE_QUADRILATERAL_WIDE;
        }
        else if (o.AntialiasedLineEnable)
        {
            LineRasterizationMode = D3D12_LINE_RASTERIZATION_MODE_ALPHA_ANTIALIASED;
        }
        ForcedSampleCount = o.ForcedSampleCount;
        ConservativeRaster = o.ConservativeRaster;
    }
    explicit CD3DX12_RASTERIZER_DESC2(const D3D12_RASTERIZER_DESC& o) noexcept
        : CD3DX12_RASTERIZER_DESC2(CD3DX12_RASTERIZER_DESC1(o))
    {
    }
    explicit CD3DX12_RASTERIZER_DESC2(CD3DX12_DEFAULT) noexcept
    {
        FillMode = D3D12_FILL_MODE_SOLID;
        CullMode = D3D12_CULL_MODE_BACK;
        FrontCounterClockwise = FALSE;
        DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        DepthClipEnable = TRUE;
        LineRasterizationMode = D3D12_LINE_RASTERIZATION_MODE_ALIASED;
        ForcedSampleCount = 0;
        ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    }
    explicit CD3DX12_RASTERIZER_DESC2(
        D3D12_FILL_MODE fillMode,
        D3D12_CULL_MODE cullMode,
        BOOL frontCounterClockwise,
        FLOAT depthBias,
        FLOAT depthBiasClamp,
        FLOAT slopeScaledDepthBias,
        BOOL depthClipEnable,
        D3D12_LINE_RASTERIZATION_MODE lineRasterizationMode,
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
        LineRasterizationMode = lineRasterizationMode;
        ForcedSampleCount = forcedSampleCount;
        ConservativeRaster = conservativeRaster;
    }


    operator D3D12_RASTERIZER_DESC1() const noexcept
    {
        D3D12_RASTERIZER_DESC1 o;

        o.FillMode = FillMode;
        o.CullMode = CullMode;
        o.FrontCounterClockwise = FrontCounterClockwise;
        o.DepthBias = DepthBias;
        o.DepthBiasClamp = DepthBiasClamp;
        o.SlopeScaledDepthBias = SlopeScaledDepthBias;
        o.DepthClipEnable = DepthClipEnable;
        o.MultisampleEnable = FALSE;
        o.AntialiasedLineEnable = FALSE;
        if (LineRasterizationMode == D3D12_LINE_RASTERIZATION_MODE_ALPHA_ANTIALIASED)
        {
            o.AntialiasedLineEnable = TRUE;
        }
        else if (LineRasterizationMode != D3D12_LINE_RASTERIZATION_MODE_ALIASED)
        {
            o.MultisampleEnable = TRUE;
        }
        o.ForcedSampleCount = ForcedSampleCount;
        o.ConservativeRaster = ConservativeRaster;

        return o;
    }
    operator D3D12_RASTERIZER_DESC() const noexcept
    {
        return (D3D12_RASTERIZER_DESC)CD3DX12_RASTERIZER_DESC1((D3D12_RASTERIZER_DESC1)*this);
    }
};
#endif // D3D12_SDK_VERSION >= 610

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
        return Type == D3D12_HEAP_TYPE_UPLOAD || Type == D3D12_HEAP_TYPE_READBACK
#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 609)
            || Type == D3D12_HEAP_TYPE_GPU_UPLOAD
#endif
            || (Type == D3D12_HEAP_TYPE_CUSTOM &&
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
constexpr UINT D3D12CalcSubresource( UINT MipSlice, UINT ArraySlice, UINT PlaneSlice, UINT MipLevels, UINT ArraySize ) noexcept
{
    return MipSlice + ArraySlice * MipLevels + PlaneSlice * MipLevels * ArraySize;
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
// Fills in the mipmap and alignment values of pDesc when either members are zero
// Used to replace an implicit field to an explicit (0 mip map = max mip map level)
// If expansion has occured, returns LclDesc, else returns the original pDesc
inline const CD3DX12_RESOURCE_DESC1* D3DX12ConditionallyExpandAPIDesc(
    CD3DX12_RESOURCE_DESC1& LclDesc,
    const CD3DX12_RESOURCE_DESC1* pDesc)
{
    // Expand mip levels:
    if (pDesc->MipLevels == 0 || pDesc->Alignment == 0)
    {
        LclDesc = *pDesc;
        if (pDesc->MipLevels == 0)
        {
            auto MaxMipLevels = [](UINT64 uiMaxDimension) -> UINT16
            {
                UINT16 uiRet = 0;
                while (uiMaxDimension > 0)
                {
                    uiRet++;
                    uiMaxDimension >>= 1;
                }
                return uiRet;
            };
            auto Max = [](UINT64 const & a, UINT64 const & b)
            {
                return (a < b) ? b : a;
            };

            LclDesc.MipLevels = MaxMipLevels(
                Max(LclDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? LclDesc.DepthOrArraySize : 1,
                    Max(LclDesc.Width, LclDesc.Height)));
        }
        if (pDesc->Alignment == 0)
        {
            if (pDesc->Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE
                || pDesc->Layout == D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE
                )
            {
                LclDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            }
            else
            {
                LclDesc.Alignment =
                    (pDesc->SampleDesc.Count > 1 ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            }
        }
        return &LclDesc;
    }
    else
    {
        return pDesc;
    }
}

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