//*********************************************************
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License (MIT).
//
//*********************************************************
#ifndef __D3D12_PROPERTY_LAYOUT_FORMAT_TABLE_H__
#define __D3D12_PROPERTY_LAYOUT_FORMAT_TABLE_H__
#include "d3d12.h"
#define MAP_ALIGN_REQUIREMENT 16 // Map is required to return 16-byte aligned addresses

struct D3D12_PROPERTY_LAYOUT_FORMAT_TABLE
{
public:
    // ----------------------------------------------------------------------------
    // Information describing everything about a D3D Resource Format
    // ----------------------------------------------------------------------------
    typedef struct FORMAT_DETAIL
    {
        DXGI_FORMAT                 DXGIFormat;
        DXGI_FORMAT                 ParentFormat;
        const DXGI_FORMAT*          pDefaultFormatCastSet;  // This is dependent on FL/driver version, but is here to save a lot of space
        UINT8                       BitsPerComponent[4]; // only used for D3DFTL_PARTIAL_TYPE or FULL_TYPE
        UINT8                       BitsPerUnit;
        BYTE                        SRGBFormat : 1;
        UINT                        WidthAlignment : 4;      // number of texels to align to in a mip level.
        UINT                        HeightAlignment : 4;     // Top level dimensions must be a multiple of these
        UINT                        DepthAlignment : 1;      // values.
        D3D_FORMAT_LAYOUT           Layout : 1;
        D3D_FORMAT_TYPE_LEVEL       TypeLevel : 2;
        D3D_FORMAT_COMPONENT_NAME   ComponentName0 : 3; // RED    ... only used for D3DFTL_PARTIAL_TYPE or FULL_TYPE
        D3D_FORMAT_COMPONENT_NAME   ComponentName1 : 3; // GREEN  ... only used for D3DFTL_PARTIAL_TYPE or FULL_TYPE
        D3D_FORMAT_COMPONENT_NAME   ComponentName2 : 3; // BLUE   ... only used for D3DFTL_PARTIAL_TYPE or FULL_TYPE
        D3D_FORMAT_COMPONENT_NAME   ComponentName3 : 3; // ALPHA  ... only used for D3DFTL_PARTIAL_TYPE or FULL_TYPE
        D3D_FORMAT_COMPONENT_INTERPRETATION ComponentInterpretation0 : 3; // only used for D3DFTL_FULL_TYPE
        D3D_FORMAT_COMPONENT_INTERPRETATION ComponentInterpretation1 : 3; // only used for D3DFTL_FULL_TYPE
        D3D_FORMAT_COMPONENT_INTERPRETATION ComponentInterpretation2 : 3; // only used for D3DFTL_FULL_TYPE
        D3D_FORMAT_COMPONENT_INTERPRETATION ComponentInterpretation3 : 3; // only used for D3DFTL_FULL_TYPE
        bool                        bDX9VertexOrIndexFormat : 1;
        bool                        bDX9TextureFormat : 1;
        bool                        bFloatNormFormat : 1;
        bool                        bPlanar : 1;
        bool                        bYUV : 1;
        bool                        bDependantFormatCastSet : 1;  // This indicates that the format cast set is dependent on FL/driver version
        bool                        bInternal : 1;
    } FORMAT_DETAIL;

private:
    static const FORMAT_DETAIL      s_FormatDetail[];
    static const UINT               s_NumFormats;
    static const LPCSTR             s_FormatNames[]; // separate from above structure so it can be compiled out of runtime.
public:
    static UINT                 GetNumFormats();
    static const FORMAT_DETAIL* GetFormatTable();
    static D3D_FEATURE_LEVEL    GetHighestDefinedFeatureLevel();

    static DXGI_FORMAT          GetFormat               (SIZE_T Index);
    static bool                 FormatExists            (DXGI_FORMAT Format);
    static bool                 FormatExistsInHeader    (DXGI_FORMAT Format, bool bExternalHeader = true);
    static UINT                 GetByteAlignment        (DXGI_FORMAT Format);
    static bool                 IsBlockCompressFormat   (DXGI_FORMAT Format);
    static LPCSTR               GetName                 (DXGI_FORMAT Format, bool bHideInternalFormats = true);
    static bool                 IsSRGBFormat            (DXGI_FORMAT Format);
    static UINT                 GetBitsPerStencil       (DXGI_FORMAT Format);
    static void                 GetFormatReturnTypes    (DXGI_FORMAT Format, D3D_FORMAT_COMPONENT_INTERPRETATION* pInterpretations); // return array of 4 components
    static UINT                 GetNumComponentsInFormat(DXGI_FORMAT Format);

    // Converts the sequential component index (range from 0 to GetNumComponentsInFormat()) to
    // the absolute component index (range 0 to 3).
    static UINT                                 Sequential2AbsoluteComponentIndex           (DXGI_FORMAT Format, UINT SequentialComponentIndex);
    static bool                                 CanBeCastEvenFullyTyped                     (DXGI_FORMAT Format, D3D_FEATURE_LEVEL fl);
    static UINT8                                GetAddressingBitsPerAlignedSize             (DXGI_FORMAT Format);
    static DXGI_FORMAT                          GetParentFormat                             (DXGI_FORMAT Format);
    static const DXGI_FORMAT*                   GetFormatCastSet                            (DXGI_FORMAT Format);
    static D3D_FORMAT_LAYOUT                    GetLayout                                   (DXGI_FORMAT Format);
    static D3D_FORMAT_TYPE_LEVEL                GetTypeLevel                                (DXGI_FORMAT Format);
    static UINT                                 GetBitsPerUnit                              (DXGI_FORMAT Format);
    static UINT                                 GetBitsPerUnitThrow                         (DXGI_FORMAT Format);
    static UINT                                 GetBitsPerElement                           (DXGI_FORMAT Format); // Legacy function used to support D3D10on9 only. Do not use.
    static UINT                                 GetWidthAlignment                           (DXGI_FORMAT Format);
    static UINT                                 GetHeightAlignment                          (DXGI_FORMAT Format);
    static UINT                                 GetDepthAlignment                           (DXGI_FORMAT Format);
    static BOOL                                 Planar                                      (DXGI_FORMAT Format);
    static BOOL                                 NonOpaquePlanar                             (DXGI_FORMAT Format);
    static BOOL                                 YUV                                         (DXGI_FORMAT Format);
    static BOOL                                 Opaque                                      (DXGI_FORMAT Format);
    static bool                                 FamilySupportsStencil                       (DXGI_FORMAT Format);
    static UINT                                 NonOpaquePlaneCount                         (DXGI_FORMAT Format);
    static BOOL                                 DX9VertexOrIndexFormat                      (DXGI_FORMAT Format);
    static BOOL                                 DX9TextureFormat                            (DXGI_FORMAT Format);
    static BOOL                                 FloatNormTextureFormat                      (DXGI_FORMAT Format);
    static bool                                 DepthOnlyFormat                             (DXGI_FORMAT format);
    static UINT8                                GetPlaneCount                               (DXGI_FORMAT Format);
    static bool                                 MotionEstimatorAllowedInputFormat           (DXGI_FORMAT Format);
    static bool                                 SupportsSamplerFeedback                     (DXGI_FORMAT Format);
    static bool                                 DecodeHistogramAllowedForOutputFormatSupport(DXGI_FORMAT Format);
    static UINT8                                GetPlaneSliceFromViewFormat                 (DXGI_FORMAT ResourceFormat, DXGI_FORMAT ViewFormat);
    static bool                                 FloatAndNotFloatFormats                     (DXGI_FORMAT FormatA,        DXGI_FORMAT FormatB);
    static bool                                 SNORMAndUNORMFormats                        (DXGI_FORMAT FormatA,        DXGI_FORMAT FormatB);
    static bool                                 ValidCastToR32UAV                           (DXGI_FORMAT from,           DXGI_FORMAT to);
    static bool                                 IsSupportedTextureDisplayableFormat         (DXGI_FORMAT,                bool bMediaFormatOnly);
    static D3D_FORMAT_COMPONENT_INTERPRETATION  GetFormatComponentInterpretation            (DXGI_FORMAT Format,         UINT AbsoluteComponentIndex);
    static UINT                                 GetBitsPerComponent                         (DXGI_FORMAT Format,         UINT AbsoluteComponentIndex);
    static D3D_FORMAT_COMPONENT_NAME            GetComponentName                            (DXGI_FORMAT Format,         UINT AbsoluteComponentIndex);
    static HRESULT                              CalculateExtraPlanarRows                    (DXGI_FORMAT format,         UINT plane0Height, _Out_ UINT& totalHeight);
    static HRESULT                              CalculateMinimumRowMajorRowPitch            (DXGI_FORMAT Format,         UINT Width, _Out_ UINT& RowPitch);
    static HRESULT                              CalculateMinimumRowMajorSlicePitch          (DXGI_FORMAT Format,         UINT ContextBasedRowPitch, UINT Height, _Out_ UINT& SlicePitch);
    static void                                 GetYCbCrChromaSubsampling                   (DXGI_FORMAT Format,         _Out_ UINT& HorizontalSubsampling, _Out_ UINT& VerticalSubsampling);

    static HRESULT                              CalculateResourceSize               (UINT width, UINT height, UINT depth, DXGI_FORMAT format, UINT mipLevels, UINT subresources, _Out_ SIZE_T& totalByteSize, _Out_writes_opt_(subresources) D3D12_MEMCPY_DEST* pDst = nullptr);
    static void                                 GetTileShape                        (D3D12_TILE_SHAPE* pTileShape, DXGI_FORMAT Format, D3D12_RESOURCE_DIMENSION Dimension, UINT SampleCount);
    static void                                 Get4KTileShape                      (D3D12_TILE_SHAPE* pTileShape, DXGI_FORMAT Format, D3D12_RESOURCE_DIMENSION Dimension, UINT SampleCount);
    static void                                 GetMipDimensions                    (UINT8 mipSlice, _Inout_ UINT64* pWidth, _Inout_opt_ UINT64* pHeight = nullptr, _Inout_opt_ UINT64* pDepth = nullptr);
    static void                                 GetPlaneSubsampledSizeAndFormatForCopyableLayout(UINT PlaneSlice, DXGI_FORMAT Format, UINT Width, UINT Height, _Out_ DXGI_FORMAT& PlaneFormat, _Out_ UINT& MinPlanePitchWidth, _Out_ UINT& PlaneWidth, _Out_ UINT& PlaneHeight);

    static UINT                                 GetDetailTableIndex         (DXGI_FORMAT  Format);
    static UINT                                 GetDetailTableIndexNoThrow  (DXGI_FORMAT  Format);
    static UINT                                 GetDetailTableIndexThrow    (DXGI_FORMAT  Format);
private:
    static const FORMAT_DETAIL*                 GetFormatDetail             (DXGI_FORMAT  Format);

};

#endif