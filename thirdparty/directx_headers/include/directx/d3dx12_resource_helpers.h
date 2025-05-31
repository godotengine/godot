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
#include "d3dx12_core.h"
#include "d3dx12_property_format_table.h"
//------------------------------------------------------------------------------------------------
template <typename T, typename U, typename V>
inline void D3D12DecomposeSubresource( UINT Subresource, UINT MipLevels, UINT ArraySize, _Out_ T& MipSlice, _Out_ U& ArraySlice, _Out_ V& PlaneSlice ) noexcept
{
    MipSlice = static_cast<T>(Subresource % MipLevels);
    ArraySlice = static_cast<U>((Subresource / MipLevels) % ArraySize);
    PlaneSlice = static_cast<V>(Subresource / (MipLevels * ArraySize));
}

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
#if defined(_MSC_VER) || !defined(_WIN32)
    const auto Desc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc;
    const auto& Desc = *pDestinationResource->GetDesc(&tmpDesc);
#endif
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
#if defined(_MSC_VER) || !defined(_WIN32)
    const auto IntermediateDesc = pIntermediate->GetDesc();
    const auto DestinationDesc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc1, tmpDesc2;
    const auto& IntermediateDesc = *pIntermediate->GetDesc(&tmpDesc1);
    const auto& DestinationDesc = *pDestinationResource->GetDesc(&tmpDesc2);
#endif
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
#if defined(_MSC_VER) || !defined(_WIN32)
    const auto IntermediateDesc = pIntermediate->GetDesc();
    const auto DestinationDesc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc1, tmpDesc2;
    const auto& IntermediateDesc = *pIntermediate->GetDesc(&tmpDesc1);
    const auto& DestinationDesc = *pDestinationResource->GetDesc(&tmpDesc2);
#endif
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

#if defined(_MSC_VER) || !defined(_WIN32)
    const auto Desc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc;
    const auto& Desc = *pDestinationResource->GetDesc(&tmpDesc);
#endif
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

#if defined(_MSC_VER) || !defined(_WIN32)
    const auto Desc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc;
    const auto& Desc = *pDestinationResource->GetDesc(&tmpDesc);
#endif
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

#if defined(_MSC_VER) || !defined(_WIN32)
    const auto Desc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc;
    const auto& Desc = *pDestinationResource->GetDesc(&tmpDesc);
#endif
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

#if defined(_MSC_VER) || !defined(_WIN32)
    const auto Desc = pDestinationResource->GetDesc();
#else
    D3D12_RESOURCE_DESC tmpDesc;
    const auto& Desc = *pDestinationResource->GetDesc(&tmpDesc);
#endif
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
template< typename T >
inline T D3DX12Align(T uValue, T uAlign)
{
    // Assert power of 2 alignment
    D3DX12_ASSERT(0 == (uAlign & (uAlign - 1)));
    T uMask = uAlign - 1;
    T uResult = (uValue + uMask) & ~uMask;
    D3DX12_ASSERT(uResult >= uValue);
    D3DX12_ASSERT(0 == (uResult % uAlign));
    return uResult;
}

//------------------------------------------------------------------------------------------------
template< typename T >
inline T D3DX12AlignAtLeast(T uValue, T uAlign)
{
    T aligned = D3DX12Align(uValue, uAlign);
    return aligned > uAlign ? aligned : uAlign;
}

inline const CD3DX12_RESOURCE_DESC1* D3DX12ConditionallyExpandAPIDesc(
    D3D12_RESOURCE_DESC1& LclDesc,
    const D3D12_RESOURCE_DESC1* pDesc)
{
    return D3DX12ConditionallyExpandAPIDesc(static_cast<CD3DX12_RESOURCE_DESC1&>(LclDesc), static_cast<const CD3DX12_RESOURCE_DESC1*>(pDesc));
}

#if defined(D3D12_SDK_VERSION) && (D3D12_SDK_VERSION >= 606)
//------------------------------------------------------------------------------------------------
// The difference between D3DX12GetCopyableFootprints and ID3D12Device::GetCopyableFootprints
// is that this one loses a lot of error checking by assuming the arguments are correct
inline bool D3DX12GetCopyableFootprints(
    _In_  const D3D12_RESOURCE_DESC1& ResourceDesc,
    _In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
    _In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) UINT NumSubresources,
    UINT64 BaseOffset,
    _Out_writes_opt_(NumSubresources) D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    _Out_writes_opt_(NumSubresources) UINT* pNumRows,
    _Out_writes_opt_(NumSubresources) UINT64* pRowSizeInBytes,
    _Out_opt_ UINT64* pTotalBytes)
{
    constexpr UINT64 uint64_max = ~0ull;
    UINT64 TotalBytes = uint64_max;
    UINT uSubRes = 0;

    bool bResourceOverflow = false;
    TotalBytes = 0;

    const DXGI_FORMAT Format = ResourceDesc.Format;

    CD3DX12_RESOURCE_DESC1 LresourceDesc;
    const CD3DX12_RESOURCE_DESC1& resourceDesc = *D3DX12ConditionallyExpandAPIDesc(LresourceDesc, &ResourceDesc);

    // Check if its a valid format
    D3DX12_ASSERT(D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::FormatExists(Format));

    const UINT WidthAlignment = D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::GetWidthAlignment( Format );
    const UINT HeightAlignment = D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::GetHeightAlignment( Format );
    const UINT16 DepthAlignment = UINT16( D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::GetDepthAlignment( Format ) );

    for (; uSubRes < NumSubresources; ++uSubRes)
    {
        bool bOverflow = false;
        UINT Subresource = FirstSubresource + uSubRes;

        D3DX12_ASSERT(resourceDesc.MipLevels != 0);
        UINT subresourceCount = resourceDesc.MipLevels * resourceDesc.ArraySize() * D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::GetPlaneCount(resourceDesc.Format);

        if (Subresource > subresourceCount)
        {
            break;
        }

        TotalBytes = D3DX12Align< UINT64 >( TotalBytes, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT );

        UINT MipLevel, ArraySlice, PlaneSlice;
        D3D12DecomposeSubresource(Subresource, resourceDesc.MipLevels, resourceDesc.ArraySize(), /*_Out_*/MipLevel, /*_Out_*/ArraySlice, /*_Out_*/PlaneSlice);

        const UINT64 Width = D3DX12AlignAtLeast<UINT64>(resourceDesc.Width >> MipLevel, WidthAlignment);
        const UINT Height =  D3DX12AlignAtLeast(resourceDesc.Height >> MipLevel, HeightAlignment);
        const UINT16 Depth = D3DX12AlignAtLeast<UINT16>(resourceDesc.Depth() >> MipLevel, DepthAlignment);

        // Adjust for the current PlaneSlice.  Most formats have only one plane.
        DXGI_FORMAT PlaneFormat;
        UINT32 MinPlanePitchWidth, PlaneWidth, PlaneHeight;
        D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::GetPlaneSubsampledSizeAndFormatForCopyableLayout(PlaneSlice, Format, (UINT)Width, Height, /*_Out_*/ PlaneFormat, /*_Out_*/ MinPlanePitchWidth, /* _Out_ */ PlaneWidth, /*_Out_*/ PlaneHeight);

        D3D12_SUBRESOURCE_FOOTPRINT LocalPlacement;
        auto& Placement = pLayouts ? pLayouts[uSubRes].Footprint : LocalPlacement;
        Placement.Format = PlaneFormat;
        Placement.Width = PlaneWidth;
        Placement.Height = PlaneHeight;
        Placement.Depth = Depth;

        // Calculate row pitch
        UINT MinPlaneRowPitch = 0;
        D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::CalculateMinimumRowMajorRowPitch(PlaneFormat, MinPlanePitchWidth, MinPlaneRowPitch);

        // Formats with more than one plane choose a larger pitch alignment to ensure that each plane begins on the row
        // immediately following the previous plane while still adhering to subresource alignment restrictions.
        static_assert(   D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT >= D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
                        && ((D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) == 0),
                        "D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT  must be >= and evenly divisible by D3D12_TEXTURE_DATA_PITCH_ALIGNMENT." );

        Placement.RowPitch = D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::Planar(Format)
            ? D3DX12Align< UINT >( MinPlaneRowPitch, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT )
            : D3DX12Align< UINT >( MinPlaneRowPitch, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT );

        if (pRowSizeInBytes)
        {
            UINT PlaneRowSize = 0;
            D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::CalculateMinimumRowMajorRowPitch(PlaneFormat, PlaneWidth, PlaneRowSize);

            pRowSizeInBytes[uSubRes] = PlaneRowSize;
        }

        // Number of rows (accounting for block compression and additional planes)
        UINT NumRows = 0;
        if (D3D12_PROPERTY_LAYOUT_FORMAT_TABLE::Planar(Format))
        {
            NumRows = PlaneHeight;
        }
        else
        {
            D3DX12_ASSERT(Height % HeightAlignment == 0);
            NumRows = Height / HeightAlignment;
        }

        if (pNumRows)
        {
            pNumRows[uSubRes] = NumRows;
        }

            // Offsetting
            if (pLayouts)
            {
                pLayouts[uSubRes].Offset = (bOverflow ? uint64_max : TotalBytes + BaseOffset);
            }

        const UINT16 NumSlices = Depth;
        const UINT64 SubresourceSize = (NumRows * NumSlices - 1) * Placement.RowPitch + MinPlaneRowPitch;

        // uint64 addition with overflow checking
        TotalBytes = TotalBytes + SubresourceSize;
        if(TotalBytes < SubresourceSize)
        {
            TotalBytes = uint64_max;
        }
        bResourceOverflow  = bResourceOverflow  || bOverflow;
    }

    // Overflow error
    if (bResourceOverflow)
    {
        TotalBytes = uint64_max;
    }


    if (pLayouts)
    {
        memset( pLayouts + uSubRes, -1, sizeof( *pLayouts ) * (NumSubresources - uSubRes) );
    }
    if (pNumRows)
    {
        memset(pNumRows + uSubRes, -1, sizeof(*pNumRows) * (NumSubresources - uSubRes));
    }
    if (pRowSizeInBytes)
    {
        memset(pRowSizeInBytes + uSubRes, -1, sizeof(*pRowSizeInBytes) * (NumSubresources - uSubRes));
    }
    if (pTotalBytes)
    {
        *pTotalBytes = TotalBytes;
    }
    if(TotalBytes == uint64_max)
    {
        return false;
    }
    return true;
}

//------------------------------------------------------------------------------------------------
inline D3D12_RESOURCE_DESC1 D3DX12ResourceDesc0ToDesc1(D3D12_RESOURCE_DESC const& desc0)
{
    D3D12_RESOURCE_DESC1       desc1;
    desc1.Dimension          = desc0.Dimension;
    desc1.Alignment          = desc0.Alignment;
    desc1.Width              = desc0.Width;
    desc1.Height             = desc0.Height;
    desc1.DepthOrArraySize   = desc0.DepthOrArraySize;
    desc1.MipLevels          = desc0.MipLevels;
    desc1.Format             = desc0.Format;
    desc1.SampleDesc.Count   = desc0.SampleDesc.Count;
    desc1.SampleDesc.Quality = desc0.SampleDesc.Quality;
    desc1.Layout             = desc0.Layout;
    desc1.Flags              = desc0.Flags;
    desc1.SamplerFeedbackMipRegion.Width = 0;
    desc1.SamplerFeedbackMipRegion.Height = 0;
    desc1.SamplerFeedbackMipRegion.Depth = 0;
    return desc1;
}

//------------------------------------------------------------------------------------------------
inline bool D3DX12GetCopyableFootprints(
	_In_  const D3D12_RESOURCE_DESC& pResourceDesc,
	_In_range_(0, D3D12_REQ_SUBRESOURCES) UINT FirstSubresource,
	_In_range_(0, D3D12_REQ_SUBRESOURCES - FirstSubresource) UINT NumSubresources,
	UINT64 BaseOffset,
	_Out_writes_opt_(NumSubresources) D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
	_Out_writes_opt_(NumSubresources) UINT* pNumRows,
	_Out_writes_opt_(NumSubresources) UINT64* pRowSizeInBytes,
	_Out_opt_ UINT64* pTotalBytes)
{
    // From D3D12_RESOURCE_DESC to D3D12_RESOURCE_DESC1
    D3D12_RESOURCE_DESC1 desc = D3DX12ResourceDesc0ToDesc1(pResourceDesc);
	return D3DX12GetCopyableFootprints(
		*static_cast<CD3DX12_RESOURCE_DESC1*>(&desc),// From D3D12_RESOURCE_DESC1 to CD3DX12_RESOURCE_DESC1
		FirstSubresource,
		NumSubresources,
		BaseOffset,
		pLayouts,
		pNumRows,
		pRowSizeInBytes,
		pTotalBytes);
}

#endif // D3D12_SDK_VERSION >= 606
