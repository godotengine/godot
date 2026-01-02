/*-------------------------------------------------------------------------------------
 *
 * Copyright (c) Microsoft Corporation
 * Licensed under the MIT license
 *
 *-------------------------------------------------------------------------------------*/


/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.01.0628 */



/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 500
#endif

/* verify that the <rpcsal.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCSAL_H_VERSION__
#define __REQUIRED_RPCSAL_H_VERSION__ 100
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif /* __RPCNDR_H_VERSION__ */

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __d3d12video_h__
#define __d3d12video_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#ifndef DECLSPEC_XFGVIRT
#if defined(_CONTROL_FLOW_GUARD_XFG)
#define DECLSPEC_XFGVIRT(base, func) __declspec(xfg_virtual(base, func))
#else
#define DECLSPEC_XFGVIRT(base, func)
#endif
#endif

/* Forward Declarations */ 

#ifndef __ID3D12VideoDecoderHeap_FWD_DEFINED__
#define __ID3D12VideoDecoderHeap_FWD_DEFINED__
typedef interface ID3D12VideoDecoderHeap ID3D12VideoDecoderHeap;

#endif 	/* __ID3D12VideoDecoderHeap_FWD_DEFINED__ */


#ifndef __ID3D12VideoDevice_FWD_DEFINED__
#define __ID3D12VideoDevice_FWD_DEFINED__
typedef interface ID3D12VideoDevice ID3D12VideoDevice;

#endif 	/* __ID3D12VideoDevice_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecoder_FWD_DEFINED__
#define __ID3D12VideoDecoder_FWD_DEFINED__
typedef interface ID3D12VideoDecoder ID3D12VideoDecoder;

#endif 	/* __ID3D12VideoDecoder_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessor_FWD_DEFINED__
#define __ID3D12VideoProcessor_FWD_DEFINED__
typedef interface ID3D12VideoProcessor ID3D12VideoProcessor;

#endif 	/* __ID3D12VideoProcessor_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList_FWD_DEFINED__
#define __ID3D12VideoDecodeCommandList_FWD_DEFINED__
typedef interface ID3D12VideoDecodeCommandList ID3D12VideoDecodeCommandList;

#endif 	/* __ID3D12VideoDecodeCommandList_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList_FWD_DEFINED__
#define __ID3D12VideoProcessCommandList_FWD_DEFINED__
typedef interface ID3D12VideoProcessCommandList ID3D12VideoProcessCommandList;

#endif 	/* __ID3D12VideoProcessCommandList_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList1_FWD_DEFINED__
#define __ID3D12VideoDecodeCommandList1_FWD_DEFINED__
typedef interface ID3D12VideoDecodeCommandList1 ID3D12VideoDecodeCommandList1;

#endif 	/* __ID3D12VideoDecodeCommandList1_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList1_FWD_DEFINED__
#define __ID3D12VideoProcessCommandList1_FWD_DEFINED__
typedef interface ID3D12VideoProcessCommandList1 ID3D12VideoProcessCommandList1;

#endif 	/* __ID3D12VideoProcessCommandList1_FWD_DEFINED__ */


#ifndef __ID3D12VideoMotionEstimator_FWD_DEFINED__
#define __ID3D12VideoMotionEstimator_FWD_DEFINED__
typedef interface ID3D12VideoMotionEstimator ID3D12VideoMotionEstimator;

#endif 	/* __ID3D12VideoMotionEstimator_FWD_DEFINED__ */


#ifndef __ID3D12VideoMotionVectorHeap_FWD_DEFINED__
#define __ID3D12VideoMotionVectorHeap_FWD_DEFINED__
typedef interface ID3D12VideoMotionVectorHeap ID3D12VideoMotionVectorHeap;

#endif 	/* __ID3D12VideoMotionVectorHeap_FWD_DEFINED__ */


#ifndef __ID3D12VideoDevice1_FWD_DEFINED__
#define __ID3D12VideoDevice1_FWD_DEFINED__
typedef interface ID3D12VideoDevice1 ID3D12VideoDevice1;

#endif 	/* __ID3D12VideoDevice1_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList_FWD_DEFINED__
#define __ID3D12VideoEncodeCommandList_FWD_DEFINED__
typedef interface ID3D12VideoEncodeCommandList ID3D12VideoEncodeCommandList;

#endif 	/* __ID3D12VideoEncodeCommandList_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecoder1_FWD_DEFINED__
#define __ID3D12VideoDecoder1_FWD_DEFINED__
typedef interface ID3D12VideoDecoder1 ID3D12VideoDecoder1;

#endif 	/* __ID3D12VideoDecoder1_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecoderHeap1_FWD_DEFINED__
#define __ID3D12VideoDecoderHeap1_FWD_DEFINED__
typedef interface ID3D12VideoDecoderHeap1 ID3D12VideoDecoderHeap1;

#endif 	/* __ID3D12VideoDecoderHeap1_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessor1_FWD_DEFINED__
#define __ID3D12VideoProcessor1_FWD_DEFINED__
typedef interface ID3D12VideoProcessor1 ID3D12VideoProcessor1;

#endif 	/* __ID3D12VideoProcessor1_FWD_DEFINED__ */


#ifndef __ID3D12VideoExtensionCommand_FWD_DEFINED__
#define __ID3D12VideoExtensionCommand_FWD_DEFINED__
typedef interface ID3D12VideoExtensionCommand ID3D12VideoExtensionCommand;

#endif 	/* __ID3D12VideoExtensionCommand_FWD_DEFINED__ */


#ifndef __ID3D12VideoDevice2_FWD_DEFINED__
#define __ID3D12VideoDevice2_FWD_DEFINED__
typedef interface ID3D12VideoDevice2 ID3D12VideoDevice2;

#endif 	/* __ID3D12VideoDevice2_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList2_FWD_DEFINED__
#define __ID3D12VideoDecodeCommandList2_FWD_DEFINED__
typedef interface ID3D12VideoDecodeCommandList2 ID3D12VideoDecodeCommandList2;

#endif 	/* __ID3D12VideoDecodeCommandList2_FWD_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList3_FWD_DEFINED__
#define __ID3D12VideoDecodeCommandList3_FWD_DEFINED__
typedef interface ID3D12VideoDecodeCommandList3 ID3D12VideoDecodeCommandList3;

#endif 	/* __ID3D12VideoDecodeCommandList3_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList2_FWD_DEFINED__
#define __ID3D12VideoProcessCommandList2_FWD_DEFINED__
typedef interface ID3D12VideoProcessCommandList2 ID3D12VideoProcessCommandList2;

#endif 	/* __ID3D12VideoProcessCommandList2_FWD_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList3_FWD_DEFINED__
#define __ID3D12VideoProcessCommandList3_FWD_DEFINED__
typedef interface ID3D12VideoProcessCommandList3 ID3D12VideoProcessCommandList3;

#endif 	/* __ID3D12VideoProcessCommandList3_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList1_FWD_DEFINED__
#define __ID3D12VideoEncodeCommandList1_FWD_DEFINED__
typedef interface ID3D12VideoEncodeCommandList1 ID3D12VideoEncodeCommandList1;

#endif 	/* __ID3D12VideoEncodeCommandList1_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncoder_FWD_DEFINED__
#define __ID3D12VideoEncoder_FWD_DEFINED__
typedef interface ID3D12VideoEncoder ID3D12VideoEncoder;

#endif 	/* __ID3D12VideoEncoder_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncoderHeap_FWD_DEFINED__
#define __ID3D12VideoEncoderHeap_FWD_DEFINED__
typedef interface ID3D12VideoEncoderHeap ID3D12VideoEncoderHeap;

#endif 	/* __ID3D12VideoEncoderHeap_FWD_DEFINED__ */


#ifndef __ID3D12VideoDevice3_FWD_DEFINED__
#define __ID3D12VideoDevice3_FWD_DEFINED__
typedef interface ID3D12VideoDevice3 ID3D12VideoDevice3;

#endif 	/* __ID3D12VideoDevice3_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList2_FWD_DEFINED__
#define __ID3D12VideoEncodeCommandList2_FWD_DEFINED__
typedef interface ID3D12VideoEncodeCommandList2 ID3D12VideoEncodeCommandList2;

#endif 	/* __ID3D12VideoEncodeCommandList2_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList3_FWD_DEFINED__
#define __ID3D12VideoEncodeCommandList3_FWD_DEFINED__
typedef interface ID3D12VideoEncodeCommandList3 ID3D12VideoEncodeCommandList3;

#endif 	/* __ID3D12VideoEncodeCommandList3_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncoderHeap1_FWD_DEFINED__
#define __ID3D12VideoEncoderHeap1_FWD_DEFINED__
typedef interface ID3D12VideoEncoderHeap1 ID3D12VideoEncoderHeap1;

#endif 	/* __ID3D12VideoEncoderHeap1_FWD_DEFINED__ */


#ifndef __ID3D12VideoDevice4_FWD_DEFINED__
#define __ID3D12VideoDevice4_FWD_DEFINED__
typedef interface ID3D12VideoDevice4 ID3D12VideoDevice4;

#endif 	/* __ID3D12VideoDevice4_FWD_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList4_FWD_DEFINED__
#define __ID3D12VideoEncodeCommandList4_FWD_DEFINED__
typedef interface ID3D12VideoEncodeCommandList4 ID3D12VideoEncodeCommandList4;

#endif 	/* __ID3D12VideoEncodeCommandList4_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "dxgicommon.h"
#include "d3d12.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_d3d12video_0000_0000 */
/* [local] */ 

#include <winapifamily.h>
#pragma region App Family
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES)
typedef 
enum D3D12_VIDEO_FIELD_TYPE
    {
        D3D12_VIDEO_FIELD_TYPE_NONE	= 0,
        D3D12_VIDEO_FIELD_TYPE_INTERLACED_TOP_FIELD_FIRST	= 1,
        D3D12_VIDEO_FIELD_TYPE_INTERLACED_BOTTOM_FIELD_FIRST	= 2
    } 	D3D12_VIDEO_FIELD_TYPE;

typedef 
enum D3D12_VIDEO_FRAME_STEREO_FORMAT
    {
        D3D12_VIDEO_FRAME_STEREO_FORMAT_NONE	= 0,
        D3D12_VIDEO_FRAME_STEREO_FORMAT_MONO	= 1,
        D3D12_VIDEO_FRAME_STEREO_FORMAT_HORIZONTAL	= 2,
        D3D12_VIDEO_FRAME_STEREO_FORMAT_VERTICAL	= 3,
        D3D12_VIDEO_FRAME_STEREO_FORMAT_SEPARATE	= 4
    } 	D3D12_VIDEO_FRAME_STEREO_FORMAT;

typedef struct D3D12_VIDEO_FORMAT
    {
    DXGI_FORMAT Format;
    DXGI_COLOR_SPACE_TYPE ColorSpace;
    } 	D3D12_VIDEO_FORMAT;

typedef struct D3D12_VIDEO_SAMPLE
    {
    UINT Width;
    UINT Height;
    D3D12_VIDEO_FORMAT Format;
    } 	D3D12_VIDEO_SAMPLE;

typedef 
enum D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE
    {
        D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE_NONE	= 0,
        D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE_FIELD_BASED	= 1
    } 	D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE;

typedef 
enum D3D12_FEATURE_VIDEO
    {
        D3D12_FEATURE_VIDEO_DECODE_SUPPORT	= 0,
        D3D12_FEATURE_VIDEO_DECODE_PROFILES	= 1,
        D3D12_FEATURE_VIDEO_DECODE_FORMATS	= 2,
        D3D12_FEATURE_VIDEO_DECODE_CONVERSION_SUPPORT	= 3,
        D3D12_FEATURE_VIDEO_PROCESS_SUPPORT	= 5,
        D3D12_FEATURE_VIDEO_PROCESS_MAX_INPUT_STREAMS	= 6,
        D3D12_FEATURE_VIDEO_PROCESS_REFERENCE_INFO	= 7,
        D3D12_FEATURE_VIDEO_DECODER_HEAP_SIZE	= 8,
        D3D12_FEATURE_VIDEO_PROCESSOR_SIZE	= 9,
        D3D12_FEATURE_VIDEO_DECODE_PROFILE_COUNT	= 10,
        D3D12_FEATURE_VIDEO_DECODE_FORMAT_COUNT	= 11,
        D3D12_FEATURE_VIDEO_ARCHITECTURE	= 17,
        D3D12_FEATURE_VIDEO_DECODE_HISTOGRAM	= 18,
        D3D12_FEATURE_VIDEO_FEATURE_AREA_SUPPORT	= 19,
        D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR	= 20,
        D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR_SIZE	= 21,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMAND_COUNT	= 22,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMANDS	= 23,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMAND_PARAMETER_COUNT	= 24,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMAND_PARAMETERS	= 25,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMAND_SUPPORT	= 26,
        D3D12_FEATURE_VIDEO_EXTENSION_COMMAND_SIZE	= 27,
        D3D12_FEATURE_VIDEO_DECODE_PROTECTED_RESOURCES	= 28,
        D3D12_FEATURE_VIDEO_PROCESS_PROTECTED_RESOURCES	= 29,
        D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR_PROTECTED_RESOURCES	= 30,
        D3D12_FEATURE_VIDEO_DECODER_HEAP_SIZE1	= 31,
        D3D12_FEATURE_VIDEO_PROCESSOR_SIZE1	= 32,
        D3D12_FEATURE_VIDEO_ENCODER_CODEC	= 33,
        D3D12_FEATURE_VIDEO_ENCODER_PROFILE_LEVEL	= 34,
        D3D12_FEATURE_VIDEO_ENCODER_OUTPUT_RESOLUTION_RATIOS_COUNT	= 35,
        D3D12_FEATURE_VIDEO_ENCODER_OUTPUT_RESOLUTION	= 36,
        D3D12_FEATURE_VIDEO_ENCODER_INPUT_FORMAT	= 37,
        D3D12_FEATURE_VIDEO_ENCODER_RATE_CONTROL_MODE	= 38,
        D3D12_FEATURE_VIDEO_ENCODER_INTRA_REFRESH_MODE	= 39,
        D3D12_FEATURE_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE	= 40,
        D3D12_FEATURE_VIDEO_ENCODER_HEAP_SIZE	= 41,
        D3D12_FEATURE_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT	= 42,
        D3D12_FEATURE_VIDEO_ENCODER_SUPPORT	= 43,
        D3D12_FEATURE_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT	= 44,
        D3D12_FEATURE_VIDEO_ENCODER_RESOURCE_REQUIREMENTS	= 45,
        D3D12_FEATURE_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG	= 46,
        D3D12_FEATURE_VIDEO_ENCODER_SUPPORT1	= 47,
        D3D12_FEATURE_VIDEO_ENCODER_RESOURCE_REQUIREMENTS1	= 48,
        D3D12_FEATURE_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT	= 49,
        D3D12_FEATURE_VIDEO_ENCODER_QPMAP_INPUT	= 50,
        D3D12_FEATURE_VIDEO_ENCODER_DIRTY_REGIONS	= 51,
        D3D12_FEATURE_VIDEO_ENCODER_MOTION_SEARCH	= 52,
        D3D12_FEATURE_VIDEO_ENCODER_SUPPORT2	= 55,
        D3D12_FEATURE_VIDEO_ENCODER_HEAP_SIZE1	= 56,
        D3D12_FEATURE_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS	= 57
    } 	D3D12_FEATURE_VIDEO;

typedef 
enum D3D12_BITSTREAM_ENCRYPTION_TYPE
    {
        D3D12_BITSTREAM_ENCRYPTION_TYPE_NONE	= 0
    } 	D3D12_BITSTREAM_ENCRYPTION_TYPE;

typedef struct D3D12_VIDEO_DECODE_CONFIGURATION
    {
    GUID DecodeProfile;
    D3D12_BITSTREAM_ENCRYPTION_TYPE BitstreamEncryption;
    D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE InterlaceType;
    } 	D3D12_VIDEO_DECODE_CONFIGURATION;

typedef struct D3D12_VIDEO_DECODER_DESC
    {
    UINT NodeMask;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    } 	D3D12_VIDEO_DECODER_DESC;

typedef struct D3D12_VIDEO_DECODER_HEAP_DESC
    {
    UINT NodeMask;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    UINT DecodeWidth;
    UINT DecodeHeight;
    DXGI_FORMAT Format;
    DXGI_RATIONAL FrameRate;
    UINT BitRate;
    UINT MaxDecodePictureBufferCount;
    } 	D3D12_VIDEO_DECODER_HEAP_DESC;

typedef struct D3D12_VIDEO_SIZE_RANGE
    {
    UINT MaxWidth;
    UINT MaxHeight;
    UINT MinWidth;
    UINT MinHeight;
    } 	D3D12_VIDEO_SIZE_RANGE;

typedef 
enum D3D12_VIDEO_PROCESS_FILTER
    {
        D3D12_VIDEO_PROCESS_FILTER_BRIGHTNESS	= 0,
        D3D12_VIDEO_PROCESS_FILTER_CONTRAST	= 1,
        D3D12_VIDEO_PROCESS_FILTER_HUE	= 2,
        D3D12_VIDEO_PROCESS_FILTER_SATURATION	= 3,
        D3D12_VIDEO_PROCESS_FILTER_NOISE_REDUCTION	= 4,
        D3D12_VIDEO_PROCESS_FILTER_EDGE_ENHANCEMENT	= 5,
        D3D12_VIDEO_PROCESS_FILTER_ANAMORPHIC_SCALING	= 6,
        D3D12_VIDEO_PROCESS_FILTER_STEREO_ADJUSTMENT	= 7
    } 	D3D12_VIDEO_PROCESS_FILTER;

typedef 
enum D3D12_VIDEO_PROCESS_FILTER_FLAGS
    {
        D3D12_VIDEO_PROCESS_FILTER_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_BRIGHTNESS	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_BRIGHTNESS ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_CONTRAST	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_CONTRAST ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_HUE	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_HUE ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_SATURATION	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_SATURATION ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_NOISE_REDUCTION	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_NOISE_REDUCTION ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_EDGE_ENHANCEMENT	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_EDGE_ENHANCEMENT ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_ANAMORPHIC_SCALING	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_ANAMORPHIC_SCALING ) ,
        D3D12_VIDEO_PROCESS_FILTER_FLAG_STEREO_ADJUSTMENT	= ( 1 << D3D12_VIDEO_PROCESS_FILTER_STEREO_ADJUSTMENT ) 
    } 	D3D12_VIDEO_PROCESS_FILTER_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_FILTER_FLAGS )
typedef 
enum D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS
    {
        D3D12_VIDEO_PROCESS_DEINTERLACE_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_DEINTERLACE_FLAG_BOB	= 0x1,
        D3D12_VIDEO_PROCESS_DEINTERLACE_FLAG_CUSTOM	= 0x80000000
    } 	D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS )
typedef struct D3D12_VIDEO_PROCESS_ALPHA_BLENDING
    {
    BOOL Enable;
    FLOAT Alpha;
    } 	D3D12_VIDEO_PROCESS_ALPHA_BLENDING;

typedef struct D3D12_VIDEO_PROCESS_LUMA_KEY
    {
    BOOL Enable;
    FLOAT Lower;
    FLOAT Upper;
    } 	D3D12_VIDEO_PROCESS_LUMA_KEY;

typedef struct D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC
    {
    DXGI_FORMAT Format;
    DXGI_COLOR_SPACE_TYPE ColorSpace;
    DXGI_RATIONAL SourceAspectRatio;
    DXGI_RATIONAL DestinationAspectRatio;
    DXGI_RATIONAL FrameRate;
    D3D12_VIDEO_SIZE_RANGE SourceSizeRange;
    D3D12_VIDEO_SIZE_RANGE DestinationSizeRange;
    BOOL EnableOrientation;
    D3D12_VIDEO_PROCESS_FILTER_FLAGS FilterFlags;
    D3D12_VIDEO_FRAME_STEREO_FORMAT StereoFormat;
    D3D12_VIDEO_FIELD_TYPE FieldType;
    D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS DeinterlaceMode;
    BOOL EnableAlphaBlending;
    D3D12_VIDEO_PROCESS_LUMA_KEY LumaKey;
    UINT NumPastFrames;
    UINT NumFutureFrames;
    BOOL EnableAutoProcessing;
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC;

typedef 
enum D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE
    {
        D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_OPAQUE	= 0,
        D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_BACKGROUND	= 1,
        D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_DESTINATION	= 2,
        D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_SOURCE_STREAM	= 3
    } 	D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE;

typedef struct D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC
    {
    DXGI_FORMAT Format;
    DXGI_COLOR_SPACE_TYPE ColorSpace;
    D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE AlphaFillMode;
    UINT AlphaFillModeSourceStreamIndex;
    FLOAT BackgroundColor[ 4 ];
    DXGI_RATIONAL FrameRate;
    BOOL EnableStereo;
    } 	D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0000_v0_0_s_ifspec;

#ifndef __ID3D12VideoDecoderHeap_INTERFACE_DEFINED__
#define __ID3D12VideoDecoderHeap_INTERFACE_DEFINED__

/* interface ID3D12VideoDecoderHeap */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecoderHeap;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("0946B7C9-EBF6-4047-BB73-8683E27DBB1F")
    ID3D12VideoDecoderHeap : public ID3D12Pageable
    {
    public:
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_DECODER_HEAP_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
#else
        virtual D3D12_VIDEO_DECODER_HEAP_DESC *STDMETHODCALLTYPE GetDesc( 
            D3D12_VIDEO_DECODER_HEAP_DESC * RetVal) = 0;
#endif
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecoderHeapVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecoderHeap * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecoderHeap * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecoderHeap * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecoderHeap * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecoderHeap * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecoderHeap * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecoderHeap * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoderHeap, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_DECODER_HEAP_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoderHeap * This);
        
#else
        D3D12_VIDEO_DECODER_HEAP_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoderHeap * This,
            D3D12_VIDEO_DECODER_HEAP_DESC * RetVal);
        
#endif
        
        END_INTERFACE
    } ID3D12VideoDecoderHeapVtbl;

    interface ID3D12VideoDecoderHeap
    {
        CONST_VTBL struct ID3D12VideoDecoderHeapVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecoderHeap_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecoderHeap_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecoderHeap_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecoderHeap_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecoderHeap_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecoderHeap_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecoderHeap_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecoderHeap_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoDecoderHeap_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoDecoderHeap_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecoderHeap_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDevice_INTERFACE_DEFINED__
#define __ID3D12VideoDevice_INTERFACE_DEFINED__

/* interface ID3D12VideoDevice */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDevice;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1F052807-0B46-4ACC-8A89-364F793718A4")
    ID3D12VideoDevice : public IUnknown
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CheckFeatureSupport( 
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoDecoder( 
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoDecoderHeap( 
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoProcessor( 
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDeviceVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDevice * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDevice * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDevice * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CheckFeatureSupport)
        HRESULT ( STDMETHODCALLTYPE *CheckFeatureSupport )( 
            ID3D12VideoDevice * This,
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder )( 
            ID3D12VideoDevice * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap )( 
            ID3D12VideoDevice * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoProcessor)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor )( 
            ID3D12VideoDevice * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        END_INTERFACE
    } ID3D12VideoDeviceVtbl;

    interface ID3D12VideoDevice
    {
        CONST_VTBL struct ID3D12VideoDeviceVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDevice_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDevice_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDevice_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDevice_CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize)	\
    ( (This)->lpVtbl -> CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize) ) 

#define ID3D12VideoDevice_CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice_CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice_CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDevice_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDecoder_INTERFACE_DEFINED__
#define __ID3D12VideoDecoder_INTERFACE_DEFINED__

/* interface ID3D12VideoDecoder */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("C59B6BDC-7720-4074-A136-17A156037470")
    ID3D12VideoDecoder : public ID3D12Pageable
    {
    public:
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_DECODER_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
#else
        virtual D3D12_VIDEO_DECODER_DESC *STDMETHODCALLTYPE GetDesc( 
            D3D12_VIDEO_DECODER_DESC * RetVal) = 0;
#endif
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecoderVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecoder * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecoder * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecoder * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecoder * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecoder * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecoder * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecoder * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoder, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_DECODER_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoder * This);
        
#else
        D3D12_VIDEO_DECODER_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoder * This,
            D3D12_VIDEO_DECODER_DESC * RetVal);
        
#endif
        
        END_INTERFACE
    } ID3D12VideoDecoderVtbl;

    interface ID3D12VideoDecoder
    {
        CONST_VTBL struct ID3D12VideoDecoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecoder_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecoder_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecoder_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecoder_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecoder_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoDecoder_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoDecoder_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecoder_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0003 */
/* [local] */ 

typedef 
enum D3D12_VIDEO_DECODE_TIER
    {
        D3D12_VIDEO_DECODE_TIER_NOT_SUPPORTED	= 0,
        D3D12_VIDEO_DECODE_TIER_1	= 1,
        D3D12_VIDEO_DECODE_TIER_2	= 2,
        D3D12_VIDEO_DECODE_TIER_3	= 3
    } 	D3D12_VIDEO_DECODE_TIER;

typedef 
enum D3D12_VIDEO_DECODE_SUPPORT_FLAGS
    {
        D3D12_VIDEO_DECODE_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_DECODE_SUPPORT_FLAG_SUPPORTED	= 0x1
    } 	D3D12_VIDEO_DECODE_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_DECODE_SUPPORT_FLAGS )
typedef 
enum D3D12_VIDEO_DECODE_CONFIGURATION_FLAGS
    {
        D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_NONE	= 0,
        D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED	= 0x1,
        D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_POST_PROCESSING_SUPPORTED	= 0x2,
        D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_REFERENCE_ONLY_ALLOCATIONS_REQUIRED	= 0x4,
        D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_ALLOW_RESOLUTION_CHANGE_ON_NON_KEY_FRAME	= 0x8
    } 	D3D12_VIDEO_DECODE_CONFIGURATION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_DECODE_CONFIGURATION_FLAGS )
typedef 
enum D3D12_VIDEO_DECODE_STATUS
    {
        D3D12_VIDEO_DECODE_STATUS_OK	= 0,
        D3D12_VIDEO_DECODE_STATUS_CONTINUE	= 1,
        D3D12_VIDEO_DECODE_STATUS_CONTINUE_SKIP_DISPLAY	= 2,
        D3D12_VIDEO_DECODE_STATUS_RESTART	= 3,
        D3D12_VIDEO_DECODE_STATUS_RATE_EXCEEDED	= 4
    } 	D3D12_VIDEO_DECODE_STATUS;

typedef 
enum D3D12_VIDEO_DECODE_ARGUMENT_TYPE
    {
        D3D12_VIDEO_DECODE_ARGUMENT_TYPE_PICTURE_PARAMETERS	= 0,
        D3D12_VIDEO_DECODE_ARGUMENT_TYPE_INVERSE_QUANTIZATION_MATRIX	= 1,
        D3D12_VIDEO_DECODE_ARGUMENT_TYPE_SLICE_CONTROL	= 2,
        D3D12_VIDEO_DECODE_ARGUMENT_TYPE_HUFFMAN_TABLE	= 3
    } 	D3D12_VIDEO_DECODE_ARGUMENT_TYPE;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    UINT Width;
    UINT Height;
    DXGI_FORMAT DecodeFormat;
    DXGI_RATIONAL FrameRate;
    UINT BitRate;
    D3D12_VIDEO_DECODE_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_DECODE_CONFIGURATION_FLAGS ConfigurationFlags;
    D3D12_VIDEO_DECODE_TIER DecodeTier;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_PROFILE_COUNT
    {
    UINT NodeIndex;
    UINT ProfileCount;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_PROFILE_COUNT;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_PROFILES
    {
    UINT NodeIndex;
    UINT ProfileCount;
    _Field_size_full_(ProfileCount)  GUID *pProfiles;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_PROFILES;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_FORMAT_COUNT
    {
    UINT NodeIndex;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    UINT FormatCount;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_FORMAT_COUNT;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_FORMATS
    {
    UINT NodeIndex;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    UINT FormatCount;
    _Field_size_full_(FormatCount)  DXGI_FORMAT *pOutputFormats;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_FORMATS;

typedef struct D3D12_FEATURE_DATA_VIDEO_ARCHITECTURE
    {
    BOOL IOCoherent;
    } 	D3D12_FEATURE_DATA_VIDEO_ARCHITECTURE;

typedef 
enum D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT
    {
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_Y	= 0,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_U	= 1,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_V	= 2,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_R	= 0,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_G	= 1,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_B	= 2,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_A	= 3
    } 	D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT;

typedef 
enum D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAGS
    {
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_NONE	= 0,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_Y	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_Y ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_U	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_U ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_V	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_V ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_R	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_R ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_G	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_G ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_B	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_B ) ,
        D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAG_A	= ( 1 << D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_A ) 
    } 	D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_HISTOGRAM
    {
    UINT NodeIndex;
    GUID DecodeProfile;
    UINT Width;
    UINT Height;
    DXGI_FORMAT DecodeFormat;
    D3D12_VIDEO_DECODE_HISTOGRAM_COMPONENT_FLAGS Components;
    UINT BinCount;
    UINT CounterBitDepth;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_HISTOGRAM;

typedef 
enum D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAGS
    {
        D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAG_SUPPORTED	= 0x1
    } 	D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAGS )
typedef 
enum D3D12_VIDEO_SCALE_SUPPORT_FLAGS
    {
        D3D12_VIDEO_SCALE_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_SCALE_SUPPORT_FLAG_POW2_ONLY	= 0x1,
        D3D12_VIDEO_SCALE_SUPPORT_FLAG_EVEN_DIMENSIONS_ONLY	= 0x2,
        D3D12_VIDEO_SCALE_SUPPORT_FLAG_DPB_ENCODER_RESOURCES	= 0x4
    } 	D3D12_VIDEO_SCALE_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_SCALE_SUPPORT_FLAGS )
typedef struct D3D12_VIDEO_SCALE_SUPPORT
    {
    D3D12_VIDEO_SIZE_RANGE OutputSizeRange;
    D3D12_VIDEO_SCALE_SUPPORT_FLAGS Flags;
    } 	D3D12_VIDEO_SCALE_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_CONVERSION_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    D3D12_VIDEO_SAMPLE DecodeSample;
    D3D12_VIDEO_FORMAT OutputFormat;
    DXGI_RATIONAL FrameRate;
    UINT BitRate;
    D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_SCALE_SUPPORT ScaleSupport;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_CONVERSION_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODER_HEAP_SIZE
    {
    D3D12_VIDEO_DECODER_HEAP_DESC VideoDecoderHeapDesc;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODER_HEAP_SIZE;

typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESSOR_SIZE
    {
    UINT NodeMask;
    const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc;
    UINT NumInputStreamDescs;
    const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESSOR_SIZE;

typedef struct D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS
    {
    UINT64 Status;
    UINT64 NumMacroblocksAffected;
    DXGI_RATIONAL FrameRate;
    UINT BitRate;
    } 	D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS;

typedef struct D3D12_VIDEO_DECODE_FRAME_ARGUMENT
    {
    D3D12_VIDEO_DECODE_ARGUMENT_TYPE Type;
    UINT Size;
    _Field_size_bytes_full_(Size)  void *pData;
    } 	D3D12_VIDEO_DECODE_FRAME_ARGUMENT;

typedef struct D3D12_VIDEO_DECODE_REFERENCE_FRAMES
    {
    UINT NumTexture2Ds;
    _Field_size_full_(NumTexture2Ds)  ID3D12Resource **ppTexture2Ds;
    _Field_size_full_(NumTexture2Ds)  UINT *pSubresources;
    _Field_size_full_opt_(NumTexture2Ds)  ID3D12VideoDecoderHeap **ppHeaps;
    } 	D3D12_VIDEO_DECODE_REFERENCE_FRAMES;

typedef struct D3D12_VIDEO_DECODE_COMPRESSED_BITSTREAM
    {
    ID3D12Resource *pBuffer;
    UINT64 Offset;
    UINT64 Size;
    } 	D3D12_VIDEO_DECODE_COMPRESSED_BITSTREAM;

typedef struct D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS
    {
    BOOL Enable;
    ID3D12Resource *pReferenceTexture2D;
    UINT ReferenceSubresource;
    DXGI_COLOR_SPACE_TYPE OutputColorSpace;
    DXGI_COLOR_SPACE_TYPE DecodeColorSpace;
    } 	D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS;

typedef struct D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS
    {
    UINT NumFrameArguments;
    D3D12_VIDEO_DECODE_FRAME_ARGUMENT FrameArguments[ 10 ];
    D3D12_VIDEO_DECODE_REFERENCE_FRAMES ReferenceFrames;
    D3D12_VIDEO_DECODE_COMPRESSED_BITSTREAM CompressedBitstream;
    ID3D12VideoDecoderHeap *pHeap;
    } 	D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS;

typedef struct D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS
    {
    ID3D12Resource *pOutputTexture2D;
    UINT OutputSubresource;
    D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS ConversionArguments;
    } 	D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0003_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0003_v0_0_s_ifspec;

#ifndef __ID3D12VideoProcessor_INTERFACE_DEFINED__
#define __ID3D12VideoProcessor_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessor */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessor;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("304FDB32-BEDE-410A-8545-943AC6A46138")
    ID3D12VideoProcessor : public ID3D12Pageable
    {
    public:
        virtual UINT STDMETHODCALLTYPE GetNodeMask( void) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetNumInputStreamDescs( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetInputStreamDescs( 
            UINT NumInputStreamDescs,
            _Out_writes_(NumInputStreamDescs)  D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs) = 0;
        
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC STDMETHODCALLTYPE GetOutputStreamDesc( void) = 0;
#else
        virtual D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *STDMETHODCALLTYPE GetOutputStreamDesc( 
            D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC * RetVal) = 0;
#endif
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessorVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessor * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessor * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessor * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessor * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessor * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessor * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessor * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessor * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetNodeMask)
        UINT ( STDMETHODCALLTYPE *GetNodeMask )( 
            ID3D12VideoProcessor * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetNumInputStreamDescs)
        UINT ( STDMETHODCALLTYPE *GetNumInputStreamDescs )( 
            ID3D12VideoProcessor * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetInputStreamDescs)
        HRESULT ( STDMETHODCALLTYPE *GetInputStreamDescs )( 
            ID3D12VideoProcessor * This,
            UINT NumInputStreamDescs,
            _Out_writes_(NumInputStreamDescs)  D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetOutputStreamDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC ( STDMETHODCALLTYPE *GetOutputStreamDesc )( 
            ID3D12VideoProcessor * This);
        
#else
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *( STDMETHODCALLTYPE *GetOutputStreamDesc )( 
            ID3D12VideoProcessor * This,
            D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC * RetVal);
        
#endif
        
        END_INTERFACE
    } ID3D12VideoProcessorVtbl;

    interface ID3D12VideoProcessor
    {
        CONST_VTBL struct ID3D12VideoProcessorVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessor_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessor_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessor_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessor_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessor_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessor_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessor_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessor_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 



#define ID3D12VideoProcessor_GetNodeMask(This)	\
    ( (This)->lpVtbl -> GetNodeMask(This) ) 

#define ID3D12VideoProcessor_GetNumInputStreamDescs(This)	\
    ( (This)->lpVtbl -> GetNumInputStreamDescs(This) ) 

#define ID3D12VideoProcessor_GetInputStreamDescs(This,NumInputStreamDescs,pInputStreamDescs)	\
    ( (This)->lpVtbl -> GetInputStreamDescs(This,NumInputStreamDescs,pInputStreamDescs) ) 
#if !defined(_WIN32)

#define ID3D12VideoProcessor_GetOutputStreamDesc(This)	\
    ( (This)->lpVtbl -> GetOutputStreamDesc(This) ) 
#else
#define ID3D12VideoProcessor_GetOutputStreamDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetOutputStreamDesc(This,RetVal) ) 
#endif

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessor_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0004 */
/* [local] */ 

typedef 
enum D3D12_VIDEO_PROCESS_FEATURE_FLAGS
    {
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_ALPHA_FILL	= 0x1,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_LUMA_KEY	= 0x2,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_STEREO	= 0x4,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_ROTATION	= 0x8,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_FLIP	= 0x10,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_ALPHA_BLENDING	= 0x20,
        D3D12_VIDEO_PROCESS_FEATURE_FLAG_PIXEL_ASPECT_RATIO	= 0x40
    } 	D3D12_VIDEO_PROCESS_FEATURE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_FEATURE_FLAGS )
typedef 
enum D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAGS
    {
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_DENOISE	= 0x1,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_DERINGING	= 0x2,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_EDGE_ENHANCEMENT	= 0x4,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_COLOR_CORRECTION	= 0x8,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_FLESH_TONE_MAPPING	= 0x10,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_IMAGE_STABILIZATION	= 0x20,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_SUPER_RESOLUTION	= 0x40,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_ANAMORPHIC_SCALING	= 0x80,
        D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAG_CUSTOM	= 0x80000000
    } 	D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAGS )
typedef 
enum D3D12_VIDEO_PROCESS_ORIENTATION
    {
        D3D12_VIDEO_PROCESS_ORIENTATION_DEFAULT	= 0,
        D3D12_VIDEO_PROCESS_ORIENTATION_FLIP_HORIZONTAL	= 1,
        D3D12_VIDEO_PROCESS_ORIENTATION_CLOCKWISE_90	= 2,
        D3D12_VIDEO_PROCESS_ORIENTATION_CLOCKWISE_90_FLIP_HORIZONTAL	= 3,
        D3D12_VIDEO_PROCESS_ORIENTATION_CLOCKWISE_180	= 4,
        D3D12_VIDEO_PROCESS_ORIENTATION_FLIP_VERTICAL	= 5,
        D3D12_VIDEO_PROCESS_ORIENTATION_CLOCKWISE_270	= 6,
        D3D12_VIDEO_PROCESS_ORIENTATION_CLOCKWISE_270_FLIP_HORIZONTAL	= 7
    } 	D3D12_VIDEO_PROCESS_ORIENTATION;

typedef 
enum D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAGS
    {
        D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_FRAME_DISCONTINUITY	= 0x1,
        D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_FRAME_REPEAT	= 0x2
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAGS )
typedef struct D3D12_VIDEO_PROCESS_FILTER_RANGE
    {
    INT Minimum;
    INT Maximum;
    INT Default;
    FLOAT Multiplier;
    } 	D3D12_VIDEO_PROCESS_FILTER_RANGE;

typedef 
enum D3D12_VIDEO_PROCESS_SUPPORT_FLAGS
    {
        D3D12_VIDEO_PROCESS_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_PROCESS_SUPPORT_FLAG_SUPPORTED	= 0x1
    } 	D3D12_VIDEO_PROCESS_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROCESS_SUPPORT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_SAMPLE InputSample;
    D3D12_VIDEO_FIELD_TYPE InputFieldType;
    D3D12_VIDEO_FRAME_STEREO_FORMAT InputStereoFormat;
    DXGI_RATIONAL InputFrameRate;
    D3D12_VIDEO_FORMAT OutputFormat;
    D3D12_VIDEO_FRAME_STEREO_FORMAT OutputStereoFormat;
    DXGI_RATIONAL OutputFrameRate;
    D3D12_VIDEO_PROCESS_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_SCALE_SUPPORT ScaleSupport;
    D3D12_VIDEO_PROCESS_FEATURE_FLAGS FeatureSupport;
    D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS DeinterlaceSupport;
    D3D12_VIDEO_PROCESS_AUTO_PROCESSING_FLAGS AutoProcessingSupport;
    D3D12_VIDEO_PROCESS_FILTER_FLAGS FilterSupport;
    D3D12_VIDEO_PROCESS_FILTER_RANGE FilterRangeSupport[ 32 ];
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESS_MAX_INPUT_STREAMS
    {
    UINT NodeIndex;
    UINT MaxInputStreams;
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESS_MAX_INPUT_STREAMS;

typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESS_REFERENCE_INFO
    {
    UINT NodeIndex;
    D3D12_VIDEO_PROCESS_DEINTERLACE_FLAGS DeinterlaceMode;
    D3D12_VIDEO_PROCESS_FILTER_FLAGS Filters;
    D3D12_VIDEO_PROCESS_FEATURE_FLAGS FeatureSupport;
    DXGI_RATIONAL InputFrameRate;
    DXGI_RATIONAL OutputFrameRate;
    BOOL EnableAutoProcessing;
    UINT PastFrames;
    UINT FutureFrames;
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESS_REFERENCE_INFO;

typedef struct D3D12_VIDEO_PROCESS_REFERENCE_SET
    {
    UINT NumPastFrames;
    ID3D12Resource **ppPastFrames;
    UINT *pPastSubresources;
    UINT NumFutureFrames;
    ID3D12Resource **ppFutureFrames;
    UINT *pFutureSubresources;
    } 	D3D12_VIDEO_PROCESS_REFERENCE_SET;

typedef struct D3D12_VIDEO_PROCESS_TRANSFORM
    {
    D3D12_RECT SourceRectangle;
    D3D12_RECT DestinationRectangle;
    D3D12_VIDEO_PROCESS_ORIENTATION Orientation;
    } 	D3D12_VIDEO_PROCESS_TRANSFORM;

typedef struct D3D12_VIDEO_PROCESS_INPUT_STREAM_RATE
    {
    UINT OutputIndex;
    UINT InputFrameOrField;
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM_RATE;

typedef struct D3D12_VIDEO_PROCESS_INPUT_STREAM
    {
    ID3D12Resource *pTexture2D;
    UINT Subresource;
    D3D12_VIDEO_PROCESS_REFERENCE_SET ReferenceSet;
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM;

typedef struct D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS
    {
    D3D12_VIDEO_PROCESS_INPUT_STREAM InputStream[ 2 ];
    D3D12_VIDEO_PROCESS_TRANSFORM Transform;
    D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAGS Flags;
    D3D12_VIDEO_PROCESS_INPUT_STREAM_RATE RateInfo;
    INT FilterLevels[ 32 ];
    D3D12_VIDEO_PROCESS_ALPHA_BLENDING AlphaBlending;
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS;

typedef struct D3D12_VIDEO_PROCESS_OUTPUT_STREAM
    {
    ID3D12Resource *pTexture2D;
    UINT Subresource;
    } 	D3D12_VIDEO_PROCESS_OUTPUT_STREAM;

typedef struct D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS
    {
    D3D12_VIDEO_PROCESS_OUTPUT_STREAM OutputStream[ 2 ];
    D3D12_RECT TargetRectangle;
    } 	D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0004_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0004_v0_0_s_ifspec;

#ifndef __ID3D12VideoDecodeCommandList_INTERFACE_DEFINED__
#define __ID3D12VideoDecodeCommandList_INTERFACE_DEFINED__

/* interface ID3D12VideoDecodeCommandList */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecodeCommandList;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("3B60536E-AD29-4E64-A269-F853837E5E53")
    ID3D12VideoDecodeCommandList : public ID3D12CommandList
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE Close( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE Reset( 
            _In_  ID3D12CommandAllocator *pAllocator) = 0;
        
        virtual void STDMETHODCALLTYPE ClearState( void) = 0;
        
        virtual void STDMETHODCALLTYPE ResourceBarrier( 
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers) = 0;
        
        virtual void STDMETHODCALLTYPE DiscardResource( 
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion) = 0;
        
        virtual void STDMETHODCALLTYPE BeginQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE EndQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveQueryData( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset) = 0;
        
        virtual void STDMETHODCALLTYPE SetPredication( 
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation) = 0;
        
        virtual void STDMETHODCALLTYPE SetMarker( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE BeginEvent( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE EndEvent( void) = 0;
        
        virtual void STDMETHODCALLTYPE DecodeFrame( 
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE WriteBufferImmediate( 
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecodeCommandListVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecodeCommandList * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecodeCommandList * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecodeCommandList * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoDecodeCommandList * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoDecodeCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoDecodeCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoDecodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DecodeFrame)
        void ( STDMETHODCALLTYPE *DecodeFrame )( 
            ID3D12VideoDecodeCommandList * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoDecodeCommandList * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        END_INTERFACE
    } ID3D12VideoDecodeCommandListVtbl;

    interface ID3D12VideoDecodeCommandList
    {
        CONST_VTBL struct ID3D12VideoDecodeCommandListVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecodeCommandList_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecodeCommandList_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecodeCommandList_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecodeCommandList_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecodeCommandList_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecodeCommandList_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecodeCommandList_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecodeCommandList_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoDecodeCommandList_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoDecodeCommandList_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoDecodeCommandList_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoDecodeCommandList_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoDecodeCommandList_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoDecodeCommandList_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoDecodeCommandList_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoDecodeCommandList_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoDecodeCommandList_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoDecodeCommandList_DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoDecodeCommandList_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecodeCommandList_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList_INTERFACE_DEFINED__
#define __ID3D12VideoProcessCommandList_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessCommandList */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessCommandList;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("AEB2543A-167F-4682-ACC8-D159ED4A6209")
    ID3D12VideoProcessCommandList : public ID3D12CommandList
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE Close( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE Reset( 
            _In_  ID3D12CommandAllocator *pAllocator) = 0;
        
        virtual void STDMETHODCALLTYPE ClearState( void) = 0;
        
        virtual void STDMETHODCALLTYPE ResourceBarrier( 
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers) = 0;
        
        virtual void STDMETHODCALLTYPE DiscardResource( 
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion) = 0;
        
        virtual void STDMETHODCALLTYPE BeginQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE EndQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveQueryData( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset) = 0;
        
        virtual void STDMETHODCALLTYPE SetPredication( 
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation) = 0;
        
        virtual void STDMETHODCALLTYPE SetMarker( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE BeginEvent( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE EndEvent( void) = 0;
        
        virtual void STDMETHODCALLTYPE ProcessFrames( 
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS *pInputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE WriteBufferImmediate( 
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessCommandListVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessCommandList * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessCommandList * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessCommandList * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessCommandList * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessCommandList * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessCommandList * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoProcessCommandList * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoProcessCommandList * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoProcessCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoProcessCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoProcessCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ProcessFrames)
        void ( STDMETHODCALLTYPE *ProcessFrames )( 
            ID3D12VideoProcessCommandList * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoProcessCommandList * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        END_INTERFACE
    } ID3D12VideoProcessCommandListVtbl;

    interface ID3D12VideoProcessCommandList
    {
        CONST_VTBL struct ID3D12VideoProcessCommandListVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessCommandList_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessCommandList_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessCommandList_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessCommandList_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessCommandList_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessCommandList_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessCommandList_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessCommandList_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoProcessCommandList_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoProcessCommandList_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoProcessCommandList_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoProcessCommandList_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoProcessCommandList_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoProcessCommandList_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoProcessCommandList_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoProcessCommandList_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoProcessCommandList_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoProcessCommandList_ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 

#define ID3D12VideoProcessCommandList_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessCommandList_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0006 */
/* [local] */ 

typedef struct D3D12_VIDEO_DECODE_OUTPUT_HISTOGRAM
    {
    UINT64 Offset;
    ID3D12Resource *pBuffer;
    } 	D3D12_VIDEO_DECODE_OUTPUT_HISTOGRAM;

typedef struct D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS1
    {
    BOOL Enable;
    ID3D12Resource *pReferenceTexture2D;
    UINT ReferenceSubresource;
    DXGI_COLOR_SPACE_TYPE OutputColorSpace;
    DXGI_COLOR_SPACE_TYPE DecodeColorSpace;
    UINT OutputWidth;
    UINT OutputHeight;
    } 	D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS1;

typedef struct D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1
    {
    ID3D12Resource *pOutputTexture2D;
    UINT OutputSubresource;
    D3D12_VIDEO_DECODE_CONVERSION_ARGUMENTS1 ConversionArguments;
    D3D12_VIDEO_DECODE_OUTPUT_HISTOGRAM Histograms[ 4 ];
    } 	D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0006_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0006_v0_0_s_ifspec;

#ifndef __ID3D12VideoDecodeCommandList1_INTERFACE_DEFINED__
#define __ID3D12VideoDecodeCommandList1_INTERFACE_DEFINED__

/* interface ID3D12VideoDecodeCommandList1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecodeCommandList1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("D52F011B-B56E-453C-A05A-A7F311C8F472")
    ID3D12VideoDecodeCommandList1 : public ID3D12VideoDecodeCommandList
    {
    public:
        virtual void STDMETHODCALLTYPE DecodeFrame1( 
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1 *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecodeCommandList1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecodeCommandList1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecodeCommandList1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoDecodeCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoDecodeCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoDecodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DecodeFrame)
        void ( STDMETHODCALLTYPE *DecodeFrame )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoDecodeCommandList1 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList1, DecodeFrame1)
        void ( STDMETHODCALLTYPE *DecodeFrame1 )( 
            ID3D12VideoDecodeCommandList1 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1 *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        END_INTERFACE
    } ID3D12VideoDecodeCommandList1Vtbl;

    interface ID3D12VideoDecodeCommandList1
    {
        CONST_VTBL struct ID3D12VideoDecodeCommandList1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecodeCommandList1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecodeCommandList1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecodeCommandList1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecodeCommandList1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecodeCommandList1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecodeCommandList1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecodeCommandList1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecodeCommandList1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoDecodeCommandList1_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoDecodeCommandList1_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoDecodeCommandList1_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoDecodeCommandList1_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoDecodeCommandList1_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoDecodeCommandList1_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoDecodeCommandList1_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList1_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList1_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoDecodeCommandList1_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoDecodeCommandList1_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList1_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList1_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoDecodeCommandList1_DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoDecodeCommandList1_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoDecodeCommandList1_DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecodeCommandList1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0007 */
/* [local] */ 

typedef struct D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1
    {
    D3D12_VIDEO_PROCESS_INPUT_STREAM InputStream[ 2 ];
    D3D12_VIDEO_PROCESS_TRANSFORM Transform;
    D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAGS Flags;
    D3D12_VIDEO_PROCESS_INPUT_STREAM_RATE RateInfo;
    INT FilterLevels[ 32 ];
    D3D12_VIDEO_PROCESS_ALPHA_BLENDING AlphaBlending;
    D3D12_VIDEO_FIELD_TYPE FieldType;
    } 	D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0007_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0007_v0_0_s_ifspec;

#ifndef __ID3D12VideoProcessCommandList1_INTERFACE_DEFINED__
#define __ID3D12VideoProcessCommandList1_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessCommandList1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessCommandList1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("542C5C4D-7596-434F-8C93-4EFA6766F267")
    ID3D12VideoProcessCommandList1 : public ID3D12VideoProcessCommandList
    {
    public:
        virtual void STDMETHODCALLTYPE ProcessFrames1( 
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1 *pInputArguments) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessCommandList1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessCommandList1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessCommandList1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoProcessCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoProcessCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoProcessCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ProcessFrames)
        void ( STDMETHODCALLTYPE *ProcessFrames )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoProcessCommandList1 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList1, ProcessFrames1)
        void ( STDMETHODCALLTYPE *ProcessFrames1 )( 
            ID3D12VideoProcessCommandList1 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1 *pInputArguments);
        
        END_INTERFACE
    } ID3D12VideoProcessCommandList1Vtbl;

    interface ID3D12VideoProcessCommandList1
    {
        CONST_VTBL struct ID3D12VideoProcessCommandList1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessCommandList1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessCommandList1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessCommandList1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessCommandList1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessCommandList1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessCommandList1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessCommandList1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessCommandList1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoProcessCommandList1_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoProcessCommandList1_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoProcessCommandList1_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoProcessCommandList1_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoProcessCommandList1_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoProcessCommandList1_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoProcessCommandList1_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList1_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList1_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoProcessCommandList1_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoProcessCommandList1_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList1_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList1_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoProcessCommandList1_ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 

#define ID3D12VideoProcessCommandList1_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoProcessCommandList1_ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessCommandList1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0008 */
/* [local] */ 

typedef 
enum D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE
    {
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8	= 0,
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16	= 1
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE;

typedef 
enum D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAGS
    {
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_NONE	= 0,
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_8X8	= ( 1 << D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8 ) ,
        D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_16X16	= ( 1 << D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16 ) 
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAGS )
typedef 
enum D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION
    {
        D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL	= 0
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION;

typedef 
enum D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAGS
    {
        D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAG_NONE	= 0,
        D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAG_QUARTER_PEL	= ( 1 << D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL ) 
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_FEATURE_AREA_SUPPORT
    {
    UINT NodeIndex;
    BOOL VideoDecodeSupport;
    BOOL VideoProcessSupport;
    BOOL VideoEncodeSupport;
    } 	D3D12_FEATURE_DATA_VIDEO_FEATURE_AREA_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR
    {
    UINT NodeIndex;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAGS BlockSizeFlags;
    D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAGS PrecisionFlags;
    D3D12_VIDEO_SIZE_RANGE SizeRange;
    } 	D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR;

typedef struct D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR_SIZE
    {
    UINT NodeIndex;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE BlockSize;
    D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION Precision;
    D3D12_VIDEO_SIZE_RANGE SizeRange;
    BOOL Protected;
    UINT64 MotionVectorHeapMemoryPoolL0Size;
    UINT64 MotionVectorHeapMemoryPoolL1Size;
    UINT64 MotionEstimatorMemoryPoolL0Size;
    UINT64 MotionEstimatorMemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR_SIZE;

typedef struct D3D12_VIDEO_MOTION_ESTIMATOR_DESC
    {
    UINT NodeMask;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE BlockSize;
    D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION Precision;
    D3D12_VIDEO_SIZE_RANGE SizeRange;
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0008_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0008_v0_0_s_ifspec;

#ifndef __ID3D12VideoMotionEstimator_INTERFACE_DEFINED__
#define __ID3D12VideoMotionEstimator_INTERFACE_DEFINED__

/* interface ID3D12VideoMotionEstimator */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoMotionEstimator;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("33FDAE0E-098B-428F-87BB-34B695DE08F8")
    ID3D12VideoMotionEstimator : public ID3D12Pageable
    {
    public:
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_MOTION_ESTIMATOR_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
#else
        virtual D3D12_VIDEO_MOTION_ESTIMATOR_DESC *STDMETHODCALLTYPE GetDesc( 
            D3D12_VIDEO_MOTION_ESTIMATOR_DESC * RetVal) = 0;
#endif
        
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoMotionEstimatorVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoMotionEstimator * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoMotionEstimator * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoMotionEstimator * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoMotionEstimator * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoMotionEstimator * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoMotionEstimator * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoMotionEstimator * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoMotionEstimator * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoMotionEstimator, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_MOTION_ESTIMATOR_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoMotionEstimator * This);
        
#else
        D3D12_VIDEO_MOTION_ESTIMATOR_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoMotionEstimator * This,
            D3D12_VIDEO_MOTION_ESTIMATOR_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoMotionEstimator, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoMotionEstimator * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoMotionEstimatorVtbl;

    interface ID3D12VideoMotionEstimator
    {
        CONST_VTBL struct ID3D12VideoMotionEstimatorVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoMotionEstimator_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoMotionEstimator_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoMotionEstimator_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoMotionEstimator_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoMotionEstimator_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoMotionEstimator_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoMotionEstimator_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoMotionEstimator_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoMotionEstimator_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoMotionEstimator_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif

#define ID3D12VideoMotionEstimator_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoMotionEstimator_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0009 */
/* [local] */ 

typedef struct D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC
    {
    UINT NodeMask;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE BlockSize;
    D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION Precision;
    D3D12_VIDEO_SIZE_RANGE SizeRange;
    } 	D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0009_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0009_v0_0_s_ifspec;

#ifndef __ID3D12VideoMotionVectorHeap_INTERFACE_DEFINED__
#define __ID3D12VideoMotionVectorHeap_INTERFACE_DEFINED__

/* interface ID3D12VideoMotionVectorHeap */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoMotionVectorHeap;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("5BE17987-743A-4061-834B-23D22DAEA505")
    ID3D12VideoMotionVectorHeap : public ID3D12Pageable
    {
    public:
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
#else
        virtual D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *STDMETHODCALLTYPE GetDesc( 
            D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC * RetVal) = 0;
#endif
        
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoMotionVectorHeapVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoMotionVectorHeap * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoMotionVectorHeap * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoMotionVectorHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoMotionVectorHeap * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoMotionVectorHeap * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoMotionVectorHeap * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoMotionVectorHeap * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoMotionVectorHeap * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoMotionVectorHeap, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoMotionVectorHeap * This);
        
#else
        D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoMotionVectorHeap * This,
            D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoMotionVectorHeap, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoMotionVectorHeap * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoMotionVectorHeapVtbl;

    interface ID3D12VideoMotionVectorHeap
    {
        CONST_VTBL struct ID3D12VideoMotionVectorHeapVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoMotionVectorHeap_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoMotionVectorHeap_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoMotionVectorHeap_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoMotionVectorHeap_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoMotionVectorHeap_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoMotionVectorHeap_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoMotionVectorHeap_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoMotionVectorHeap_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoMotionVectorHeap_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoMotionVectorHeap_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif

#define ID3D12VideoMotionVectorHeap_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoMotionVectorHeap_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDevice1_INTERFACE_DEFINED__
#define __ID3D12VideoDevice1_INTERFACE_DEFINED__

/* interface ID3D12VideoDevice1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDevice1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("981611AD-A144-4C83-9890-F30E26D658AB")
    ID3D12VideoDevice1 : public ID3D12VideoDevice
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateVideoMotionEstimator( 
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionEstimator) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoMotionVectorHeap( 
            _In_  const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionVectorHeap) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDevice1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDevice1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDevice1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDevice1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CheckFeatureSupport)
        HRESULT ( STDMETHODCALLTYPE *CheckFeatureSupport )( 
            ID3D12VideoDevice1 * This,
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder )( 
            ID3D12VideoDevice1 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap )( 
            ID3D12VideoDevice1 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoProcessor)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor )( 
            ID3D12VideoDevice1 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionEstimator)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionEstimator )( 
            ID3D12VideoDevice1 * This,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionEstimator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionVectorHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionVectorHeap )( 
            ID3D12VideoDevice1 * This,
            _In_  const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionVectorHeap);
        
        END_INTERFACE
    } ID3D12VideoDevice1Vtbl;

    interface ID3D12VideoDevice1
    {
        CONST_VTBL struct ID3D12VideoDevice1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDevice1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDevice1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDevice1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDevice1_CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize)	\
    ( (This)->lpVtbl -> CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize) ) 

#define ID3D12VideoDevice1_CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice1_CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice1_CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor) ) 


#define ID3D12VideoDevice1_CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator)	\
    ( (This)->lpVtbl -> CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator) ) 

#define ID3D12VideoDevice1_CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap)	\
    ( (This)->lpVtbl -> CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDevice1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0011 */
/* [local] */ 

typedef struct D3D12_RESOURCE_COORDINATE
    {
    UINT64 X;
    UINT Y;
    UINT Z;
    UINT SubresourceIndex;
    } 	D3D12_RESOURCE_COORDINATE;

typedef struct D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT
    {
    ID3D12VideoMotionVectorHeap *pMotionVectorHeap;
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT;

typedef struct D3D12_VIDEO_MOTION_ESTIMATOR_INPUT
    {
    ID3D12Resource *pInputTexture2D;
    UINT InputSubresourceIndex;
    ID3D12Resource *pReferenceTexture2D;
    UINT ReferenceSubresourceIndex;
    ID3D12VideoMotionVectorHeap *pHintMotionVectorHeap;
    } 	D3D12_VIDEO_MOTION_ESTIMATOR_INPUT;

typedef struct D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT
    {
    ID3D12Resource *pMotionVectorTexture2D;
    D3D12_RESOURCE_COORDINATE MotionVectorCoordinate;
    } 	D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT;

typedef struct D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT
    {
    ID3D12VideoMotionVectorHeap *pMotionVectorHeap;
    UINT PixelWidth;
    UINT PixelHeight;
    } 	D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0011_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0011_v0_0_s_ifspec;

#ifndef __ID3D12VideoEncodeCommandList_INTERFACE_DEFINED__
#define __ID3D12VideoEncodeCommandList_INTERFACE_DEFINED__

/* interface ID3D12VideoEncodeCommandList */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncodeCommandList;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8455293A-0CBD-4831-9B39-FBDBAB724723")
    ID3D12VideoEncodeCommandList : public ID3D12CommandList
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE Close( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE Reset( 
            _In_  ID3D12CommandAllocator *pAllocator) = 0;
        
        virtual void STDMETHODCALLTYPE ClearState( void) = 0;
        
        virtual void STDMETHODCALLTYPE ResourceBarrier( 
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers) = 0;
        
        virtual void STDMETHODCALLTYPE DiscardResource( 
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion) = 0;
        
        virtual void STDMETHODCALLTYPE BeginQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE EndQuery( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveQueryData( 
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset) = 0;
        
        virtual void STDMETHODCALLTYPE SetPredication( 
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation) = 0;
        
        virtual void STDMETHODCALLTYPE SetMarker( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE BeginEvent( 
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size) = 0;
        
        virtual void STDMETHODCALLTYPE EndEvent( void) = 0;
        
        virtual void STDMETHODCALLTYPE EstimateMotion( 
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveMotionVectorHeap( 
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE WriteBufferImmediate( 
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes) = 0;
        
        virtual void STDMETHODCALLTYPE SetProtectedResourceSession( 
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncodeCommandListVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncodeCommandList * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncodeCommandList * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncodeCommandList * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoEncodeCommandList * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoEncodeCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoEncodeCommandList * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoEncodeCommandList * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EstimateMotion)
        void ( STDMETHODCALLTYPE *EstimateMotion )( 
            ID3D12VideoEncodeCommandList * This,
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveMotionVectorHeap)
        void ( STDMETHODCALLTYPE *ResolveMotionVectorHeap )( 
            ID3D12VideoEncodeCommandList * This,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoEncodeCommandList * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoEncodeCommandList * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        END_INTERFACE
    } ID3D12VideoEncodeCommandListVtbl;

    interface ID3D12VideoEncodeCommandList
    {
        CONST_VTBL struct ID3D12VideoEncodeCommandListVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncodeCommandList_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncodeCommandList_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncodeCommandList_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncodeCommandList_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncodeCommandList_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncodeCommandList_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncodeCommandList_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncodeCommandList_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoEncodeCommandList_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoEncodeCommandList_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoEncodeCommandList_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoEncodeCommandList_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoEncodeCommandList_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoEncodeCommandList_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoEncodeCommandList_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoEncodeCommandList_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoEncodeCommandList_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoEncodeCommandList_EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList_ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#define ID3D12VideoEncodeCommandList_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncodeCommandList_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0012 */
/* [local] */ 

typedef 
enum D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS
    {
        D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAG_SUPPORTED	= 0x1
    } 	D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_DECODE_PROTECTED_RESOURCES
    {
    UINT NodeIndex;
    D3D12_VIDEO_DECODE_CONFIGURATION Configuration;
    D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODE_PROTECTED_RESOURCES;

typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESS_PROTECTED_RESOURCES
    {
    UINT NodeIndex;
    D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESS_PROTECTED_RESOURCES;

typedef struct D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR_PROTECTED_RESOURCES
    {
    UINT NodeIndex;
    D3D12_VIDEO_PROTECTED_RESOURCE_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR_PROTECTED_RESOURCES;

typedef struct D3D12_FEATURE_DATA_VIDEO_DECODER_HEAP_SIZE1
    {
    D3D12_VIDEO_DECODER_HEAP_DESC VideoDecoderHeapDesc;
    BOOL Protected;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_DECODER_HEAP_SIZE1;

typedef struct D3D12_FEATURE_DATA_VIDEO_PROCESSOR_SIZE1
    {
    UINT NodeMask;
    const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc;
    UINT NumInputStreamDescs;
    const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs;
    BOOL Protected;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_PROCESSOR_SIZE1;

typedef 
enum D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE
    {
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_CREATION	= 0,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_INITIALIZATION	= 1,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_EXECUTION	= 2,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_CAPS_INPUT	= 3,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_CAPS_OUTPUT	= 4,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_DEVICE_EXECUTE_INPUT	= 5,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE_DEVICE_EXECUTE_OUTPUT	= 6
    } 	D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE;

typedef 
enum D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE
    {
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_UINT8	= 0,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_UINT16	= 1,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_UINT32	= 2,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_UINT64	= 3,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_SINT8	= 4,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_SINT16	= 5,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_SINT32	= 6,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_SINT64	= 7,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_FLOAT	= 8,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_DOUBLE	= 9,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE_RESOURCE	= 10
    } 	D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE;

typedef 
enum D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAGS
    {
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAG_NONE	= 0,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAG_READ	= 0x1,
        D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAG_WRITE	= 0x2
    } 	D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_COUNT
    {
    UINT NodeIndex;
    UINT CommandCount;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_COUNT;

typedef struct D3D12_VIDEO_EXTENSION_COMMAND_INFO
    {
    GUID CommandId;
    LPCWSTR Name;
    D3D12_COMMAND_LIST_SUPPORT_FLAGS CommandListSupportFlags;
    } 	D3D12_VIDEO_EXTENSION_COMMAND_INFO;

typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMANDS
    {
    UINT NodeIndex;
    UINT CommandCount;
    _Field_size_full_(CommandCount)  D3D12_VIDEO_EXTENSION_COMMAND_INFO *pCommandInfos;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMANDS;

typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_PARAMETER_COUNT
    {
    GUID CommandId;
    D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE Stage;
    UINT ParameterCount;
    UINT ParameterPacking;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_PARAMETER_COUNT;

typedef struct D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_INFO
    {
    LPCWSTR Name;
    D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_TYPE Type;
    D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_FLAGS Flags;
    } 	D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_INFO;

typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_PARAMETERS
    {
    GUID CommandId;
    D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_STAGE Stage;
    UINT ParameterCount;
    _Field_size_full_(ParameterCount)  D3D12_VIDEO_EXTENSION_COMMAND_PARAMETER_INFO *pParameterInfos;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_PARAMETERS;

typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_SUPPORT
    {
    UINT NodeIndex;
    GUID CommandId;
    _Field_size_bytes_full_opt_(InputDataSizeInBytes)  const void *pInputData;
    SIZE_T InputDataSizeInBytes;
    _Field_size_bytes_full_opt_(OutputDataSizeInBytes)  void *pOutputData;
    SIZE_T OutputDataSizeInBytes;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_SIZE
    {
    UINT NodeIndex;
    GUID CommandId;
    _Field_size_bytes_full_(CreationParametersDataSizeInBytes)  const void *pCreationParameters;
    SIZE_T CreationParametersSizeInBytes;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_EXTENSION_COMMAND_SIZE;

typedef struct D3D12_VIDEO_EXTENSION_COMMAND_DESC
    {
    UINT NodeMask;
    GUID CommandId;
    } 	D3D12_VIDEO_EXTENSION_COMMAND_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0012_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0012_v0_0_s_ifspec;

#ifndef __ID3D12VideoDecoder1_INTERFACE_DEFINED__
#define __ID3D12VideoDecoder1_INTERFACE_DEFINED__

/* interface ID3D12VideoDecoder1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecoder1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("79A2E5FB-CCD2-469A-9FDE-195D10951F7E")
    ID3D12VideoDecoder1 : public ID3D12VideoDecoder
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecoder1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecoder1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecoder1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecoder1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecoder1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecoder1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecoder1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecoder1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecoder1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoder, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_DECODER_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoder1 * This);
        
#else
        D3D12_VIDEO_DECODER_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoder1 * This,
            D3D12_VIDEO_DECODER_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoder1, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoDecoder1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoDecoder1Vtbl;

    interface ID3D12VideoDecoder1
    {
        CONST_VTBL struct ID3D12VideoDecoder1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecoder1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecoder1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecoder1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecoder1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecoder1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecoder1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecoder1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecoder1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoDecoder1_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoDecoder1_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif


#define ID3D12VideoDecoder1_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecoder1_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDecoderHeap1_INTERFACE_DEFINED__
#define __ID3D12VideoDecoderHeap1_INTERFACE_DEFINED__

/* interface ID3D12VideoDecoderHeap1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecoderHeap1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("DA1D98C5-539F-41B2-BF6B-1198A03B6D26")
    ID3D12VideoDecoderHeap1 : public ID3D12VideoDecoderHeap
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecoderHeap1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecoderHeap1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecoderHeap1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecoderHeap1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecoderHeap1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecoderHeap1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecoderHeap1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoderHeap, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_DECODER_HEAP_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoderHeap1 * This);
        
#else
        D3D12_VIDEO_DECODER_HEAP_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoDecoderHeap1 * This,
            D3D12_VIDEO_DECODER_HEAP_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecoderHeap1, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoDecoderHeap1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoDecoderHeap1Vtbl;

    interface ID3D12VideoDecoderHeap1
    {
        CONST_VTBL struct ID3D12VideoDecoderHeap1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecoderHeap1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecoderHeap1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecoderHeap1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecoderHeap1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecoderHeap1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecoderHeap1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecoderHeap1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecoderHeap1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoDecoderHeap1_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoDecoderHeap1_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif


#define ID3D12VideoDecoderHeap1_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecoderHeap1_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoProcessor1_INTERFACE_DEFINED__
#define __ID3D12VideoProcessor1_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessor1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessor1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("F3CFE615-553F-425C-86D8-EE8C1B1FB01C")
    ID3D12VideoProcessor1 : public ID3D12VideoProcessor
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessor1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessor1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessor1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessor1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessor1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessor1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessor1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessor1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessor1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetNodeMask)
        UINT ( STDMETHODCALLTYPE *GetNodeMask )( 
            ID3D12VideoProcessor1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetNumInputStreamDescs)
        UINT ( STDMETHODCALLTYPE *GetNumInputStreamDescs )( 
            ID3D12VideoProcessor1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetInputStreamDescs)
        HRESULT ( STDMETHODCALLTYPE *GetInputStreamDescs )( 
            ID3D12VideoProcessor1 * This,
            UINT NumInputStreamDescs,
            _Out_writes_(NumInputStreamDescs)  D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor, GetOutputStreamDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC ( STDMETHODCALLTYPE *GetOutputStreamDesc )( 
            ID3D12VideoProcessor1 * This);
        
#else
        D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *( STDMETHODCALLTYPE *GetOutputStreamDesc )( 
            ID3D12VideoProcessor1 * This,
            D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessor1, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoProcessor1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoProcessor1Vtbl;

    interface ID3D12VideoProcessor1
    {
        CONST_VTBL struct ID3D12VideoProcessor1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessor1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessor1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessor1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessor1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessor1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessor1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessor1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessor1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 



#define ID3D12VideoProcessor1_GetNodeMask(This)	\
    ( (This)->lpVtbl -> GetNodeMask(This) ) 

#define ID3D12VideoProcessor1_GetNumInputStreamDescs(This)	\
    ( (This)->lpVtbl -> GetNumInputStreamDescs(This) ) 

#define ID3D12VideoProcessor1_GetInputStreamDescs(This,NumInputStreamDescs,pInputStreamDescs)	\
    ( (This)->lpVtbl -> GetInputStreamDescs(This,NumInputStreamDescs,pInputStreamDescs) ) 
#if !defined(_WIN32)

#define ID3D12VideoProcessor1_GetOutputStreamDesc(This)	\
    ( (This)->lpVtbl -> GetOutputStreamDesc(This) ) 
#else
#define ID3D12VideoProcessor1_GetOutputStreamDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetOutputStreamDesc(This,RetVal) ) 
#endif


#define ID3D12VideoProcessor1_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessor1_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoExtensionCommand_INTERFACE_DEFINED__
#define __ID3D12VideoExtensionCommand_INTERFACE_DEFINED__

/* interface ID3D12VideoExtensionCommand */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoExtensionCommand;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("554E41E8-AE8E-4A8C-B7D2-5B4F274A30E4")
    ID3D12VideoExtensionCommand : public ID3D12Pageable
    {
    public:
#if defined(_MSC_VER) || !defined(_WIN32)
        virtual D3D12_VIDEO_EXTENSION_COMMAND_DESC STDMETHODCALLTYPE GetDesc( void) = 0;
#else
        virtual D3D12_VIDEO_EXTENSION_COMMAND_DESC *STDMETHODCALLTYPE GetDesc( 
            D3D12_VIDEO_EXTENSION_COMMAND_DESC * RetVal) = 0;
#endif
        
        virtual HRESULT STDMETHODCALLTYPE GetProtectedResourceSession( 
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoExtensionCommandVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoExtensionCommand * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoExtensionCommand * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoExtensionCommand * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoExtensionCommand * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoExtensionCommand * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoExtensionCommand * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoExtensionCommand * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoExtensionCommand * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoExtensionCommand, GetDesc)
#if !defined(_WIN32)
        D3D12_VIDEO_EXTENSION_COMMAND_DESC ( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoExtensionCommand * This);
        
#else
        D3D12_VIDEO_EXTENSION_COMMAND_DESC *( STDMETHODCALLTYPE *GetDesc )( 
            ID3D12VideoExtensionCommand * This,
            D3D12_VIDEO_EXTENSION_COMMAND_DESC * RetVal);
        
#endif
        
        DECLSPEC_XFGVIRT(ID3D12VideoExtensionCommand, GetProtectedResourceSession)
        HRESULT ( STDMETHODCALLTYPE *GetProtectedResourceSession )( 
            ID3D12VideoExtensionCommand * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppProtectedSession);
        
        END_INTERFACE
    } ID3D12VideoExtensionCommandVtbl;

    interface ID3D12VideoExtensionCommand
    {
        CONST_VTBL struct ID3D12VideoExtensionCommandVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoExtensionCommand_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoExtensionCommand_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoExtensionCommand_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoExtensionCommand_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoExtensionCommand_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoExtensionCommand_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoExtensionCommand_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoExtensionCommand_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#if !defined(_WIN32)

#define ID3D12VideoExtensionCommand_GetDesc(This)	\
    ( (This)->lpVtbl -> GetDesc(This) ) 
#else
#define ID3D12VideoExtensionCommand_GetDesc(This,RetVal)	\
    ( (This)->lpVtbl -> GetDesc(This,RetVal) ) 
#endif

#define ID3D12VideoExtensionCommand_GetProtectedResourceSession(This,riid,ppProtectedSession)	\
    ( (This)->lpVtbl -> GetProtectedResourceSession(This,riid,ppProtectedSession) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoExtensionCommand_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDevice2_INTERFACE_DEFINED__
#define __ID3D12VideoDevice2_INTERFACE_DEFINED__

/* interface ID3D12VideoDevice2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDevice2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("F019AC49-F838-4A95-9B17-579437C8F513")
    ID3D12VideoDevice2 : public ID3D12VideoDevice1
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateVideoDecoder1( 
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoDecoderHeap1( 
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoProcessor1( 
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoExtensionCommand( 
            _In_  const D3D12_VIDEO_EXTENSION_COMMAND_DESC *pDesc,
            _In_reads_bytes_(CreationParametersDataSizeInBytes)  const void *pCreationParameters,
            SIZE_T CreationParametersDataSizeInBytes,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoExtensionCommand) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE ExecuteExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes,
            _Out_writes_bytes_(OutputDataSizeInBytes)  void *pOutputData,
            SIZE_T OutputDataSizeInBytes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDevice2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDevice2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDevice2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDevice2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CheckFeatureSupport)
        HRESULT ( STDMETHODCALLTYPE *CheckFeatureSupport )( 
            ID3D12VideoDevice2 * This,
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoProcessor)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor )( 
            ID3D12VideoDevice2 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionEstimator)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionEstimator )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionEstimator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionVectorHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionVectorHeap )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionVectorHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoder1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder1 )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoderHeap1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap1 )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoProcessor1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor1 )( 
            ID3D12VideoDevice2 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoExtensionCommand )( 
            ID3D12VideoDevice2 * This,
            _In_  const D3D12_VIDEO_EXTENSION_COMMAND_DESC *pDesc,
            _In_reads_bytes_(CreationParametersDataSizeInBytes)  const void *pCreationParameters,
            SIZE_T CreationParametersDataSizeInBytes,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoExtensionCommand);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, ExecuteExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoDevice2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes,
            _Out_writes_bytes_(OutputDataSizeInBytes)  void *pOutputData,
            SIZE_T OutputDataSizeInBytes);
        
        END_INTERFACE
    } ID3D12VideoDevice2Vtbl;

    interface ID3D12VideoDevice2
    {
        CONST_VTBL struct ID3D12VideoDevice2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDevice2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDevice2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDevice2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDevice2_CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize)	\
    ( (This)->lpVtbl -> CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize) ) 

#define ID3D12VideoDevice2_CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice2_CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice2_CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor) ) 


#define ID3D12VideoDevice2_CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator)	\
    ( (This)->lpVtbl -> CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator) ) 

#define ID3D12VideoDevice2_CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap)	\
    ( (This)->lpVtbl -> CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap) ) 


#define ID3D12VideoDevice2_CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice2_CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice2_CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor) ) 

#define ID3D12VideoDevice2_CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand)	\
    ( (This)->lpVtbl -> CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand) ) 

#define ID3D12VideoDevice2_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDevice2_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList2_INTERFACE_DEFINED__
#define __ID3D12VideoDecodeCommandList2_INTERFACE_DEFINED__

/* interface ID3D12VideoDecodeCommandList2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecodeCommandList2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("6e120880-c114-4153-8036-d247051e1729")
    ID3D12VideoDecodeCommandList2 : public ID3D12VideoDecodeCommandList1
    {
    public:
        virtual void STDMETHODCALLTYPE SetProtectedResourceSession( 
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession) = 0;
        
        virtual void STDMETHODCALLTYPE InitializeExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes) = 0;
        
        virtual void STDMETHODCALLTYPE ExecuteExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecodeCommandList2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecodeCommandList2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecodeCommandList2 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoDecodeCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoDecodeCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoDecodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DecodeFrame)
        void ( STDMETHODCALLTYPE *DecodeFrame )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoDecodeCommandList2 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList1, DecodeFrame1)
        void ( STDMETHODCALLTYPE *DecodeFrame1 )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1 *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoDecodeCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        END_INTERFACE
    } ID3D12VideoDecodeCommandList2Vtbl;

    interface ID3D12VideoDecodeCommandList2
    {
        CONST_VTBL struct ID3D12VideoDecodeCommandList2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecodeCommandList2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecodeCommandList2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecodeCommandList2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecodeCommandList2_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecodeCommandList2_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecodeCommandList2_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecodeCommandList2_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecodeCommandList2_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoDecodeCommandList2_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoDecodeCommandList2_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoDecodeCommandList2_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoDecodeCommandList2_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoDecodeCommandList2_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoDecodeCommandList2_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoDecodeCommandList2_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList2_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList2_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoDecodeCommandList2_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoDecodeCommandList2_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList2_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList2_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoDecodeCommandList2_DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoDecodeCommandList2_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoDecodeCommandList2_DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments) ) 


#define ID3D12VideoDecodeCommandList2_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 

#define ID3D12VideoDecodeCommandList2_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoDecodeCommandList2_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecodeCommandList2_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDecodeCommandList3_INTERFACE_DEFINED__
#define __ID3D12VideoDecodeCommandList3_INTERFACE_DEFINED__

/* interface ID3D12VideoDecodeCommandList3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDecodeCommandList3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("2aee8c37-9562-42da-8abf-61efeb2e4513")
    ID3D12VideoDecodeCommandList3 : public ID3D12VideoDecodeCommandList2
    {
    public:
        virtual void STDMETHODCALLTYPE Barrier( 
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDecodeCommandList3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDecodeCommandList3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoDecodeCommandList3 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoDecodeCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoDecodeCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoDecodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, DecodeFrame)
        void ( STDMETHODCALLTYPE *DecodeFrame )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoDecodeCommandList3 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList1, DecodeFrame1)
        void ( STDMETHODCALLTYPE *DecodeFrame1 )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12VideoDecoder *pDecoder,
            _In_  const D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS1 *pOutputArguments,
            _In_  const D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList2, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoDecodeCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDecodeCommandList3, Barrier)
        void ( STDMETHODCALLTYPE *Barrier )( 
            ID3D12VideoDecodeCommandList3 * This,
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups);
        
        END_INTERFACE
    } ID3D12VideoDecodeCommandList3Vtbl;

    interface ID3D12VideoDecodeCommandList3
    {
        CONST_VTBL struct ID3D12VideoDecodeCommandList3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDecodeCommandList3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDecodeCommandList3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDecodeCommandList3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDecodeCommandList3_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoDecodeCommandList3_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoDecodeCommandList3_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoDecodeCommandList3_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoDecodeCommandList3_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoDecodeCommandList3_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoDecodeCommandList3_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoDecodeCommandList3_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoDecodeCommandList3_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoDecodeCommandList3_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoDecodeCommandList3_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoDecodeCommandList3_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList3_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoDecodeCommandList3_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoDecodeCommandList3_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoDecodeCommandList3_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList3_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoDecodeCommandList3_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoDecodeCommandList3_DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame(This,pDecoder,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoDecodeCommandList3_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoDecodeCommandList3_DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> DecodeFrame1(This,pDecoder,pOutputArguments,pInputArguments) ) 


#define ID3D12VideoDecodeCommandList3_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 

#define ID3D12VideoDecodeCommandList3_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoDecodeCommandList3_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 


#define ID3D12VideoDecodeCommandList3_Barrier(This,NumBarrierGroups,pBarrierGroups)	\
    ( (This)->lpVtbl -> Barrier(This,NumBarrierGroups,pBarrierGroups) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDecodeCommandList3_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList2_INTERFACE_DEFINED__
#define __ID3D12VideoProcessCommandList2_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessCommandList2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessCommandList2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("db525ae4-6ad6-473c-baa7-59b2e37082e4")
    ID3D12VideoProcessCommandList2 : public ID3D12VideoProcessCommandList1
    {
    public:
        virtual void STDMETHODCALLTYPE SetProtectedResourceSession( 
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession) = 0;
        
        virtual void STDMETHODCALLTYPE InitializeExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes) = 0;
        
        virtual void STDMETHODCALLTYPE ExecuteExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessCommandList2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessCommandList2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessCommandList2 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoProcessCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoProcessCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoProcessCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ProcessFrames)
        void ( STDMETHODCALLTYPE *ProcessFrames )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoProcessCommandList2 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList1, ProcessFrames1)
        void ( STDMETHODCALLTYPE *ProcessFrames1 )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1 *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoProcessCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        END_INTERFACE
    } ID3D12VideoProcessCommandList2Vtbl;

    interface ID3D12VideoProcessCommandList2
    {
        CONST_VTBL struct ID3D12VideoProcessCommandList2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessCommandList2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessCommandList2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessCommandList2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessCommandList2_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessCommandList2_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessCommandList2_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessCommandList2_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessCommandList2_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoProcessCommandList2_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoProcessCommandList2_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoProcessCommandList2_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoProcessCommandList2_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoProcessCommandList2_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoProcessCommandList2_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoProcessCommandList2_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList2_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList2_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoProcessCommandList2_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoProcessCommandList2_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList2_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList2_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoProcessCommandList2_ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 

#define ID3D12VideoProcessCommandList2_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoProcessCommandList2_ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 


#define ID3D12VideoProcessCommandList2_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 

#define ID3D12VideoProcessCommandList2_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoProcessCommandList2_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessCommandList2_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoProcessCommandList3_INTERFACE_DEFINED__
#define __ID3D12VideoProcessCommandList3_INTERFACE_DEFINED__

/* interface ID3D12VideoProcessCommandList3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoProcessCommandList3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1a0a4ca4-9f08-40ce-9558-b411fd2666ff")
    ID3D12VideoProcessCommandList3 : public ID3D12VideoProcessCommandList2
    {
    public:
        virtual void STDMETHODCALLTYPE Barrier( 
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoProcessCommandList3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoProcessCommandList3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoProcessCommandList3 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoProcessCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoProcessCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoProcessCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, ProcessFrames)
        void ( STDMETHODCALLTYPE *ProcessFrames )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoProcessCommandList3 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList1, ProcessFrames1)
        void ( STDMETHODCALLTYPE *ProcessFrames1 )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12VideoProcessor *pVideoProcessor,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS *pOutputArguments,
            UINT NumInputStreams,
            _In_reads_(NumInputStreams)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS1 *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList2, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoProcessCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoProcessCommandList3, Barrier)
        void ( STDMETHODCALLTYPE *Barrier )( 
            ID3D12VideoProcessCommandList3 * This,
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups);
        
        END_INTERFACE
    } ID3D12VideoProcessCommandList3Vtbl;

    interface ID3D12VideoProcessCommandList3
    {
        CONST_VTBL struct ID3D12VideoProcessCommandList3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoProcessCommandList3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoProcessCommandList3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoProcessCommandList3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoProcessCommandList3_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoProcessCommandList3_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoProcessCommandList3_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoProcessCommandList3_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoProcessCommandList3_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoProcessCommandList3_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoProcessCommandList3_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoProcessCommandList3_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoProcessCommandList3_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoProcessCommandList3_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoProcessCommandList3_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoProcessCommandList3_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList3_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoProcessCommandList3_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoProcessCommandList3_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoProcessCommandList3_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList3_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoProcessCommandList3_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoProcessCommandList3_ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 

#define ID3D12VideoProcessCommandList3_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 


#define ID3D12VideoProcessCommandList3_ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments)	\
    ( (This)->lpVtbl -> ProcessFrames1(This,pVideoProcessor,pOutputArguments,NumInputStreams,pInputArguments) ) 


#define ID3D12VideoProcessCommandList3_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 

#define ID3D12VideoProcessCommandList3_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoProcessCommandList3_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 


#define ID3D12VideoProcessCommandList3_Barrier(This,NumBarrierGroups,pBarrierGroups)	\
    ( (This)->lpVtbl -> Barrier(This,NumBarrierGroups,pBarrierGroups) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoProcessCommandList3_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList1_INTERFACE_DEFINED__
#define __ID3D12VideoEncodeCommandList1_INTERFACE_DEFINED__

/* interface ID3D12VideoEncodeCommandList1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncodeCommandList1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("94971eca-2bdb-4769-88cf-3675ea757ebc")
    ID3D12VideoEncodeCommandList1 : public ID3D12VideoEncodeCommandList
    {
    public:
        virtual void STDMETHODCALLTYPE InitializeExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes) = 0;
        
        virtual void STDMETHODCALLTYPE ExecuteExtensionCommand( 
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncodeCommandList1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncodeCommandList1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncodeCommandList1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoEncodeCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoEncodeCommandList1 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoEncodeCommandList1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EstimateMotion)
        void ( STDMETHODCALLTYPE *EstimateMotion )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveMotionVectorHeap)
        void ( STDMETHODCALLTYPE *ResolveMotionVectorHeap )( 
            ID3D12VideoEncodeCommandList1 * This,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoEncodeCommandList1 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoEncodeCommandList1 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        END_INTERFACE
    } ID3D12VideoEncodeCommandList1Vtbl;

    interface ID3D12VideoEncodeCommandList1
    {
        CONST_VTBL struct ID3D12VideoEncodeCommandList1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncodeCommandList1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncodeCommandList1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncodeCommandList1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncodeCommandList1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncodeCommandList1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncodeCommandList1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncodeCommandList1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncodeCommandList1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoEncodeCommandList1_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoEncodeCommandList1_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoEncodeCommandList1_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoEncodeCommandList1_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoEncodeCommandList1_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoEncodeCommandList1_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoEncodeCommandList1_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList1_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList1_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoEncodeCommandList1_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoEncodeCommandList1_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList1_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList1_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoEncodeCommandList1_EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList1_ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList1_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#define ID3D12VideoEncodeCommandList1_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 


#define ID3D12VideoEncodeCommandList1_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoEncodeCommandList1_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncodeCommandList1_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0022 */
/* [local] */ 

DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MPEG2, 0xee27417f, 0x5e28, 0x4e65, 0xbe, 0xea, 0x1d, 0x26, 0xb5, 0x08, 0xad, 0xc9); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MPEG1_AND_MPEG2, 0x86695f12, 0x340e, 0x4f04, 0x9f, 0xd3, 0x92, 0x53, 0xdd, 0x32, 0x74, 0x60); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_H264, 0x1b81be68, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_H264_STEREO_PROGRESSIVE, 0xd79be8da, 0x0cf1, 0x4c81, 0xb8, 0x2a, 0x69, 0xa4, 0xe2, 0x36, 0xf4, 0x3d);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_H264_STEREO, 0xf9aaccbb, 0xc2b6, 0x4cfc, 0x87, 0x79, 0x57, 0x07, 0xb1, 0x76, 0x05, 0x52);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_H264_MULTIVIEW, 0x705b9d82, 0x76cf, 0x49d6, 0xb7, 0xe6, 0xac, 0x88, 0x72, 0xdb, 0x01, 0x3c);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_VC1, 0x1b81beA3, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_VC1_D2010, 0x1b81beA4, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MPEG4PT2_SIMPLE, 0xefd64d74, 0xc9e8,0x41d7,0xa5,0xe9,0xe9,0xb0,0xe3,0x9f,0xa3,0x19);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MPEG4PT2_ADVSIMPLE_NOGMC, 0xed418a9f, 0x010d, 0x4eda, 0x9a, 0xe3, 0x9a, 0x65, 0x35, 0x8d, 0x8d, 0x2e);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN, 0x5b11d51b, 0x2f4c, 0x4452, 0xbc, 0xc3, 0x09, 0xf2, 0xa1, 0x16, 0x0c, 0xc0);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN10, 0x107af0e0, 0xef1a, 0x4d19, 0xab, 0xa8, 0x67, 0xa1, 0x63, 0x07, 0x3d, 0x13);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MONOCHROME, 0x0685b993, 0x3d8c, 0x43a0, 0x8b, 0x28, 0xd7, 0x4c, 0x2d, 0x68, 0x99, 0xa4);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MONOCHROME10, 0x142a1d0f, 0x69dd, 0x4ec9, 0x85, 0x91, 0xb1, 0x2f, 0xfc, 0xb9, 0x1a, 0x29);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN12, 0x1a72925f, 0x0c2c, 0x4f15, 0x96, 0xfb, 0xb1, 0x7d, 0x14, 0x73, 0x60, 0x3f);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN10_422, 0x0bac4fe5, 0x1532, 0x4429, 0xa8, 0x54, 0xf8, 0x4d, 0xe0, 0x49, 0x53, 0xdb);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN12_422, 0x55bcac81, 0xf311, 0x4093, 0xa7, 0xd0, 0x1c, 0xbc, 0x0b, 0x84, 0x9b, 0xee);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN_444, 0x4008018f, 0xf537, 0x4b36, 0x98, 0xcf, 0x61, 0xaf, 0x8a, 0x2c, 0x1a, 0x33);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN10_EXT, 0x9cc55490, 0xe37c, 0x4932, 0x86, 0x84, 0x49, 0x20, 0xf9, 0xf6, 0x40, 0x9c);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN10_444, 0x0dabeffa, 0x4458, 0x4602, 0xbc, 0x03, 0x07, 0x95, 0x65, 0x9d, 0x61, 0x7c);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN12_444, 0x9798634d, 0xfe9d, 0x48e5, 0xb4, 0xda, 0xdb, 0xec, 0x45, 0xb3, 0xdf, 0x01);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_HEVC_MAIN16, 0xa4fbdbb0, 0xa113, 0x482b, 0xa2, 0x32, 0x63, 0x5c, 0xc0, 0x69, 0x7f, 0x6d);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_VP9, 0x463707f8, 0xa1d0, 0x4585, 0x87, 0x6d, 0x83, 0xaa, 0x6d, 0x60, 0xb8, 0x9e);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_VP9_10BIT_PROFILE2, 0xa4c749ef, 0x6ecf, 0x48aa, 0x84, 0x48, 0x50, 0xa7, 0xa1, 0x16, 0x5f, 0xf7);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_VP8, 0x90b899ea, 0x3a62, 0x4705, 0x88, 0xb3, 0x8d, 0xf0, 0x4b, 0x27, 0x44, 0xe7);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_AV1_PROFILE0, 0xb8be4ccb, 0xcf53, 0x46ba, 0x8d, 0x59, 0xd6, 0xb8, 0xa6, 0xda, 0x5d, 0x2a);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_AV1_PROFILE1, 0x6936ff0f, 0x45b1, 0x4163, 0x9c, 0xc1, 0x64, 0x6e, 0xf6, 0x94, 0x61, 0x08);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_AV1_PROFILE2, 0x0c5f2aa1, 0xe541, 0x4089, 0xbb, 0x7b, 0x98, 0x11, 0x0a, 0x19, 0xd7, 0xc8);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_AV1_12BIT_PROFILE2, 0x17127009, 0xa00f, 0x4ce1, 0x99, 0x4e, 0xbf, 0x40, 0x81, 0xf6, 0xf3, 0xf0);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_AV1_12BIT_PROFILE2_420, 0x2d80bed6, 0x9cac, 0x4835, 0x9e, 0x91, 0x32, 0x7b, 0xbc, 0x4f, 0x9e, 0xe8);
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MJPEG_VLD_420, 0x725cb506, 0xc29, 0x43c4, 0x94, 0x40, 0x8e, 0x93, 0x97, 0x90, 0x3a, 0x4); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MJPEG_VLD_422, 0x5b77b9cd, 0x1a35, 0x4c30, 0x9f, 0xd8, 0xef, 0x4b, 0x60, 0xc0, 0x35, 0xdd); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MJPEG_VLD_444, 0xd95161f9, 0xd44, 0x47e6, 0xbc, 0xf5, 0x1b, 0xfb, 0xfb, 0x26, 0x8f, 0x97); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_MJPEG_VLD_4444, 0xc91748d5, 0xfd18, 0x4aca, 0x9d, 0xb3, 0x3a, 0x66, 0x34, 0xab, 0x54, 0x7d); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_JPEG_VLD_420, 0xcf782c83, 0xbef5, 0x4a2c, 0x87, 0xcb, 0x60, 0x19, 0xe7, 0xb1, 0x75, 0xac); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_JPEG_VLD_422, 0xf04df417, 0xeee2, 0x4067, 0xa7, 0x78, 0xf3, 0x5c, 0x15, 0xab, 0x97, 0x21); 
DEFINE_GUID(D3D12_VIDEO_DECODE_PROFILE_JPEG_VLD_444, 0x4cd00e17, 0x89ba, 0x48ef, 0xb9, 0xf9, 0xed, 0xcb, 0x82, 0x71, 0x3f, 0x65);
typedef 
enum D3D12_VIDEO_ENCODER_AV1_PROFILE
    {
        D3D12_VIDEO_ENCODER_AV1_PROFILE_MAIN	= 0,
        D3D12_VIDEO_ENCODER_AV1_PROFILE_HIGH	= 1,
        D3D12_VIDEO_ENCODER_AV1_PROFILE_PROFESSIONAL	= 2
    } 	D3D12_VIDEO_ENCODER_AV1_PROFILE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_LEVELS
    {
        D3D12_VIDEO_ENCODER_AV1_LEVELS_2_0	= 0,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_2_1	= 1,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_2_2	= 2,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_2_3	= 3,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_3_0	= 4,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_3_1	= 5,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_3_2	= 6,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_3_3	= 7,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_4_0	= 8,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_4_1	= 9,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_4_2	= 10,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_4_3	= 11,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_5_0	= 12,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_5_1	= 13,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_5_2	= 14,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_5_3	= 15,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_6_0	= 16,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_6_1	= 17,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_6_2	= 18,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_6_3	= 19,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_7_0	= 20,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_7_1	= 21,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_7_2	= 22,
        D3D12_VIDEO_ENCODER_AV1_LEVELS_7_3	= 23
    } 	D3D12_VIDEO_ENCODER_AV1_LEVELS;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_TIER
    {
        D3D12_VIDEO_ENCODER_AV1_TIER_MAIN	= 0,
        D3D12_VIDEO_ENCODER_AV1_TIER_HIGH	= 1
    } 	D3D12_VIDEO_ENCODER_AV1_TIER;

typedef struct D3D12_VIDEO_ENCODER_AV1_LEVEL_TIER_CONSTRAINTS
    {
    D3D12_VIDEO_ENCODER_AV1_LEVELS Level;
    D3D12_VIDEO_ENCODER_AV1_TIER Tier;
    } 	D3D12_VIDEO_ENCODER_AV1_LEVEL_TIER_CONSTRAINTS;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_128x128_SUPERBLOCK	= 0x1,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_FILTER_INTRA	= 0x2,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_INTRA_EDGE_FILTER	= 0x4,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_INTERINTRA_COMPOUND	= 0x8,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_MASKED_COMPOUND	= 0x10,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_WARPED_MOTION	= 0x20,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_DUAL_FILTER	= 0x40,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_JNT_COMP	= 0x80,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_FORCED_INTEGER_MOTION_VECTORS	= 0x100,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_SUPER_RESOLUTION	= 0x200,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_LOOP_RESTORATION_FILTER	= 0x400,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_PALETTE_ENCODING	= 0x800,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_CDEF_FILTERING	= 0x1000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_INTRA_BLOCK_COPY	= 0x2000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_FRAME_REFERENCE_MOTION_VECTORS	= 0x4000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_ORDER_HINT_TOOLS	= 0x8000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_AUTO_SEGMENTATION	= 0x10000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_CUSTOM_SEGMENTATION	= 0x20000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_LOOP_FILTER_DELTAS	= 0x40000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_QUANTIZATION_DELTAS	= 0x80000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_QUANTIZATION_MATRIX	= 0x100000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_REDUCED_TX_SET	= 0x200000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_MOTION_MODE_SWITCHABLE	= 0x400000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_ALLOW_HIGH_PRECISION_MV	= 0x800000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_SKIP_MODE_PRESENT	= 0x1000000,
        D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_DELTA_LF_PARAMS	= 0x2000000
    } 	D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_TX_MODE
    {
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_ONLY4x4	= 0,
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_LARGEST	= 1,
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_SELECT	= 2
    } 	D3D12_VIDEO_ENCODER_AV1_TX_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAG_ONLY4x4	= ( 1 << D3D12_VIDEO_ENCODER_AV1_TX_MODE_ONLY4x4 ) ,
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAG_LARGEST	= ( 1 << D3D12_VIDEO_ENCODER_AV1_TX_MODE_LARGEST ) ,
        D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAG_SELECT	= ( 1 << D3D12_VIDEO_ENCODER_AV1_TX_MODE_SELECT ) 
    } 	D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS
    {
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP	= 0,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP_SMOOTH	= 1,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP_SHARP	= 2,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_BILINEAR	= 3,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_SWITCHABLE	= 4
    } 	D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_EIGHTTAP	= ( 1 << D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP ) ,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_EIGHTTAP_SMOOTH	= ( 1 << D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP_SMOOTH ) ,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_EIGHTTAP_SHARP	= ( 1 << D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_EIGHTTAP_SHARP ) ,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_BILINEAR	= ( 1 << D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_BILINEAR ) ,
        D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAG_SWITCHABLE	= ( 1 << D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_SWITCHABLE ) 
    } 	D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE
    {
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE_4x4	= 0,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE_8x8	= 1,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE_16x16	= 2,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE_32x32	= 3,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE_64x64	= 4
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE
    {
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_DISABLED	= 0,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_Q	= 1,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_Y_V	= 2,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_Y_H	= 3,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_U	= 4,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_V	= 5,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_REF_FRAME	= 6,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_SKIP	= 7,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_GLOBALMV	= 8
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_DISABLED	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_DISABLED ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_Q	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_Q ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_LF_Y_V	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_Y_V ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_LF_Y_H	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_Y_H ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_LF_U	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_U ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_LF_V	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_LF_V ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_REF_FRAME	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_REF_FRAME ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_SKIP	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_SKIP ) ,
        D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAG_ALT_GLOBALMV	= ( 1 << D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_ALT_GLOBALMV ) 
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE
    {
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE_DISABLED	= 0,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE_SWITCHABLE	= 1,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE_WIENER	= 2,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE_SGRPROJ	= 3
    } 	D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE
    {
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE_DISABLED	= 0,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE_32x32	= 1,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE_64x64	= 2,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE_128x128	= 3,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE_256x256	= 4
    } 	D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAG_NOT_SUPPORTED	= 0,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAG_32x32	= 0x1,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAG_64x64	= 0x2,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAG_128x128	= 0x4,
        D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAG_256x256	= 0x8
    } 	D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION
    {
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_IDENTITY	= 0,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_TRANSLATION	= 1,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_ROTZOOM	= 2,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_AFFINE	= 3
    } 	D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAG_IDENTITY	= ( 1 << D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_IDENTITY ) ,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAG_TRANSLATION	= ( 1 << D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_TRANSLATION ) ,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAG_ROTZOOM	= ( 1 << D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_ROTZOOM ) ,
        D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAG_AFFINE	= ( 1 << D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_AFFINE ) 
    } 	D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_QUANTIZATION	= 0x1,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_QUANTIZATION_DELTA	= 0x2,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_LOOP_FILTER	= 0x4,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_LOOP_FILTER_DELTA	= 0x8,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_CDEF_DATA	= 0x10,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_CONTEXT_UPDATE_TILE_ID	= 0x20,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_COMPOUND_PREDICTION_MODE	= 0x40,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_PRIMARY_REF_FRAME	= 0x80,
        D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAG_REFERENCE_INDICES	= 0x100
    } 	D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION_SUPPORT
    {
    D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS SupportedFeatureFlags;
    D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS RequiredFeatureFlags;
    D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS_FLAGS SupportedInterpolationFilters;
    D3D12_VIDEO_ENCODER_AV1_RESTORATION_SUPPORT_FLAGS SupportedRestorationParams[ 3 ][ 3 ];
    D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MODE_FLAGS SupportedSegmentationModes;
    D3D12_VIDEO_ENCODER_AV1_TX_MODE_FLAGS SupportedTxModes[ 4 ];
    D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_BLOCK_SIZE SegmentationBlockSize;
    D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES_FLAGS PostEncodeValuesFlags;
    UINT MaxTemporalLayers;
    UINT MaxSpatialLayers;
    } 	D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION_SUPPORT;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE
    {
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_KEY_FRAME	= 0,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_INTER_FRAME	= 1,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_INTRA_ONLY_FRAME	= 2,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_SWITCH_FRAME	= 3
    } 	D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAG_KEY_FRAME	= ( 1 << D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_KEY_FRAME ) ,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAG_INTER_FRAME	= ( 1 << D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_INTER_FRAME ) ,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAG_INTRA_ONLY_FRAME	= ( 1 << D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_INTRA_ONLY_FRAME ) ,
        D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAG_SWITCH_FRAME	= ( 1 << D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_SWITCH_FRAME ) 
    } 	D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE
    {
        D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE_SINGLE_REFERENCE	= 0,
        D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE_COMPOUND_REFERENCE	= 1
    } 	D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE;

typedef struct D3D12_VIDEO_ENCODER_CODEC_AV1_PICTURE_CONTROL_SUPPORT
    {
    D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE PredictionMode;
    UINT MaxUniqueReferencesPerFrame;
    D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE_FLAGS SupportedFrameTypes;
    D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION_FLAGS SupportedReferenceWarpedMotionFlags;
    } 	D3D12_VIDEO_ENCODER_CODEC_AV1_PICTURE_CONTROL_SUPPORT;

typedef struct D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION
    {
    D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS FeatureFlags;
    UINT OrderHintBitsMinus1;
    } 	D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION;

typedef struct D3D12_VIDEO_ENCODER_AV1_SEQUENCE_STRUCTURE
    {
    UINT IntraDistance;
    UINT InterFramePeriod;
    } 	D3D12_VIDEO_ENCODER_AV1_SEQUENCE_STRUCTURE;

typedef struct D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_WARPED_MOTION_INFO
    {
    D3D12_VIDEO_ENCODER_AV1_REFERENCE_WARPED_MOTION_TRANSFORMATION TransformationType;
    INT TransformationMatrix[ 8 ];
    BOOL InvalidAffineSet;
    } 	D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_WARPED_MOTION_INFO;

typedef struct D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_DESCRIPTOR
    {
    UINT ReconstructedPictureResourceIndex;
    UINT TemporalLayerIndexPlus1;
    UINT SpatialLayerIndexPlus1;
    D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE FrameType;
    D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_WARPED_MOTION_INFO WarpedMotionInfo;
    UINT OrderHint;
    UINT PictureIndex;
    } 	D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_DESCRIPTOR;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_ERROR_RESILIENT_MODE	= 0x1,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_DISABLE_CDF_UPDATE	= 0x2,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_PALETTE_ENCODING	= 0x4,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_SKIP_MODE	= 0x8,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_FRAME_REFERENCE_MOTION_VECTORS	= 0x10,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_FORCE_INTEGER_MOTION_VECTORS	= 0x20,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ALLOW_INTRA_BLOCK_COPY	= 0x40,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_USE_SUPER_RESOLUTION	= 0x80,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_DISABLE_FRAME_END_UPDATE_CDF	= 0x100,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_FRAME_SEGMENTATION_AUTO	= 0x200,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_FRAME_SEGMENTATION_CUSTOM	= 0x400,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ENABLE_WARPED_MOTION	= 0x800,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_REDUCED_TX_SET	= 0x1000,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_MOTION_MODE_SWITCHABLE	= 0x2000,
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAG_ALLOW_HIGH_PRECISION_MV	= 0x4000
    } 	D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_AV1_RESTORATION_CONFIG
    {
    D3D12_VIDEO_ENCODER_AV1_RESTORATION_TYPE FrameRestorationType[ 3 ];
    D3D12_VIDEO_ENCODER_AV1_RESTORATION_TILESIZE LoopRestorationPixelSize[ 3 ];
    } 	D3D12_VIDEO_ENCODER_AV1_RESTORATION_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_AV1_SEGMENT_DATA
    {
    UINT64 EnabledFeatures;
    INT64 FeatureValue[ 8 ];
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENT_DATA;

typedef struct D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_CONFIG
    {
    UINT64 UpdateMap;
    UINT64 TemporalUpdate;
    UINT64 UpdateData;
    UINT64 NumSegments;
    D3D12_VIDEO_ENCODER_AV1_SEGMENT_DATA SegmentsData[ 8 ];
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MAP
    {
    UINT SegmentsMapByteSize;
    UINT8 *pSegmentsMap;
    } 	D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MAP;

typedef struct D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_CONFIG
    {
    UINT64 LoopFilterLevel[ 2 ];
    UINT64 LoopFilterLevelU;
    UINT64 LoopFilterLevelV;
    UINT64 LoopFilterSharpnessLevel;
    UINT64 LoopFilterDeltaEnabled;
    UINT64 UpdateRefDelta;
    INT64 RefDeltas[ 8 ];
    UINT64 UpdateModeDelta;
    INT64 ModeDeltas[ 2 ];
    } 	D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_DELTA_CONFIG
    {
    UINT64 DeltaLFPresent;
    UINT64 DeltaLFMulti;
    UINT64 DeltaLFRes;
    } 	D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_DELTA_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_CONFIG
    {
    UINT64 BaseQIndex;
    INT64 YDCDeltaQ;
    INT64 UDCDeltaQ;
    INT64 UACDeltaQ;
    INT64 VDCDeltaQ;
    INT64 VACDeltaQ;
    UINT64 UsingQMatrix;
    UINT64 QMY;
    UINT64 QMU;
    UINT64 QMV;
    } 	D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_DELTA_CONFIG
    {
    UINT64 DeltaQPresent;
    UINT64 DeltaQRes;
    } 	D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_DELTA_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_AV1_CDEF_CONFIG
    {
    UINT64 CdefBits;
    UINT64 CdefDampingMinus3;
    UINT64 CdefYPriStrength[ 8 ];
    UINT64 CdefUVPriStrength[ 8 ];
    UINT64 CdefYSecStrength[ 8 ];
    UINT64 CdefUVSecStrength[ 8 ];
    } 	D3D12_VIDEO_ENCODER_AV1_CDEF_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_CODEC_DATA
    {
    D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_FLAGS Flags;
    D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE FrameType;
    D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE CompoundPredictionType;
    D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS InterpolationFilter;
    D3D12_VIDEO_ENCODER_AV1_RESTORATION_CONFIG FrameRestorationConfig;
    D3D12_VIDEO_ENCODER_AV1_TX_MODE TxMode;
    UINT SuperResDenominator;
    UINT OrderHint;
    UINT PictureIndex;
    UINT TemporalLayerIndexPlus1;
    UINT SpatialLayerIndexPlus1;
    D3D12_VIDEO_ENCODER_AV1_REFERENCE_PICTURE_DESCRIPTOR ReferenceFramesReconPictureDescriptors[ 8 ];
    UINT ReferenceIndices[ 7 ];
    UINT PrimaryRefFrame;
    UINT RefreshFrameFlags;
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_CONFIG LoopFilter;
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_DELTA_CONFIG LoopFilterDelta;
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_CONFIG Quantization;
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_DELTA_CONFIG QuantizationDelta;
    D3D12_VIDEO_ENCODER_AV1_CDEF_CONFIG CDEF;
    UINT QPMapValuesCount;
    _Field_size_full_(QPMapValuesCount)  INT16 *pRateControlQPMap;
    D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_CONFIG CustomSegmentation;
    D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_MAP CustomSegmentsMap;
    } 	D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_CODEC_DATA;

typedef struct D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES
    {
    UINT64 RowCount;
    UINT64 ColCount;
    UINT64 RowHeights[ 64 ];
    UINT64 ColWidths[ 64 ];
    UINT64 ContextUpdateTileId;
    } 	D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES;

typedef struct D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES
    {
    UINT64 CompoundPredictionType;
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_CONFIG LoopFilter;
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_DELTA_CONFIG LoopFilterDelta;
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_CONFIG Quantization;
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_DELTA_CONFIG QuantizationDelta;
    D3D12_VIDEO_ENCODER_AV1_CDEF_CONFIG CDEF;
    D3D12_VIDEO_ENCODER_AV1_SEGMENTATION_CONFIG SegmentationConfig;
    UINT64 PrimaryRefFrame;
    UINT64 ReferenceIndices[ 7 ];
    } 	D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES;

typedef 
enum D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE
    {
        D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_ABSOLUTE_QP_MAP	= 0,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CQP	= 1,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CBR	= 2,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_VBR	= 3,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_QVBR	= 4
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAGS
    {
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_DELTA_QP	= 0x1,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_FRAME_ANALYSIS	= 0x2,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_QP_RANGE	= 0x4,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_INITIAL_QP	= 0x8,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_MAX_FRAME_SIZE	= 0x10,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_VBV_SIZES	= 0x20,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_EXTENSION1_SUPPORT	= 0x40,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_QUALITY_VS_SPEED	= 0x80,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_SPATIAL_ADAPTIVE_QP	= 0x100
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP
    {
    UINT ConstantQP_FullIntracodedFrame;
    UINT ConstantQP_InterPredictedFrame_PrevRefOnly;
    UINT ConstantQP_InterPredictedFrame_BiDirectionalRef;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP1
    {
    UINT ConstantQP_FullIntracodedFrame;
    UINT ConstantQP_InterPredictedFrame_PrevRefOnly;
    UINT ConstantQP_InterPredictedFrame_BiDirectionalRef;
    UINT QualityVsSpeed;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP1;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetBitRate;
    UINT64 VBVCapacity;
    UINT64 InitialVBVFullness;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR1
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetBitRate;
    UINT64 VBVCapacity;
    UINT64 InitialVBVFullness;
    UINT QualityVsSpeed;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR1;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetAvgBitRate;
    UINT64 PeakBitRate;
    UINT64 VBVCapacity;
    UINT64 InitialVBVFullness;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR1
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetAvgBitRate;
    UINT64 PeakBitRate;
    UINT64 VBVCapacity;
    UINT64 InitialVBVFullness;
    UINT QualityVsSpeed;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR1;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetAvgBitRate;
    UINT64 PeakBitRate;
    UINT ConstantQualityTarget;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR1
    {
    UINT InitialQP;
    UINT MinQP;
    UINT MaxQP;
    UINT64 MaxFrameBitSize;
    UINT64 TargetAvgBitRate;
    UINT64 PeakBitRate;
    UINT ConstantQualityTarget;
    UINT64 VBVCapacity;
    UINT64 InitialVBVFullness;
    UINT QualityVsSpeed;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR1;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_ABSOLUTE_QP_MAP
    {
    UINT QualityVsSpeed;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_ABSOLUTE_QP_MAP;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL_CONFIGURATION_PARAMS
    {
    UINT DataSize;
    union 
        {
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP *pConfiguration_CQP;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR *pConfiguration_CBR;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR *pConfiguration_VBR;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR *pConfiguration_QVBR;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP1 *pConfiguration_CQP1;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR1 *pConfiguration_CBR1;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR1 *pConfiguration_VBR1;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_QVBR1 *pConfiguration_QVBR1;
        const D3D12_VIDEO_ENCODER_RATE_CONTROL_ABSOLUTE_QP_MAP *pConfiguration_AbsoluteQPMap;
        } 	;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_CONFIGURATION_PARAMS;

typedef struct D3D12_VIDEO_ENCODER_RATE_CONTROL
    {
    D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE Mode;
    D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAGS Flags;
    D3D12_VIDEO_ENCODER_RATE_CONTROL_CONFIGURATION_PARAMS ConfigParams;
    DXGI_RATIONAL TargetFrameRate;
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC
    {
        D3D12_VIDEO_ENCODER_CODEC_H264	= 0,
        D3D12_VIDEO_ENCODER_CODEC_HEVC	= 1,
        D3D12_VIDEO_ENCODER_CODEC_AV1	= 2
    } 	D3D12_VIDEO_ENCODER_CODEC;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC;

typedef 
enum D3D12_VIDEO_ENCODER_PROFILE_H264
    {
        D3D12_VIDEO_ENCODER_PROFILE_H264_MAIN	= 0,
        D3D12_VIDEO_ENCODER_PROFILE_H264_HIGH	= 1,
        D3D12_VIDEO_ENCODER_PROFILE_H264_HIGH_10	= 2
    } 	D3D12_VIDEO_ENCODER_PROFILE_H264;

typedef 
enum D3D12_VIDEO_ENCODER_PROFILE_HEVC
    {
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN	= 0,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN10	= 1,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN12	= 2,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN10_422	= 3,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN12_422	= 4,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN_444	= 5,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN10_444	= 6,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN12_444	= 7,
        D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN16_444	= 8
    } 	D3D12_VIDEO_ENCODER_PROFILE_HEVC;

typedef struct D3D12_VIDEO_ENCODER_PROFILE_DESC
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_PROFILE_H264 *pH264Profile;
        D3D12_VIDEO_ENCODER_PROFILE_HEVC *pHEVCProfile;
        D3D12_VIDEO_ENCODER_AV1_PROFILE *pAV1Profile;
        } 	;
    } 	D3D12_VIDEO_ENCODER_PROFILE_DESC;

typedef 
enum D3D12_VIDEO_ENCODER_LEVELS_H264
    {
        D3D12_VIDEO_ENCODER_LEVELS_H264_1	= 0,
        D3D12_VIDEO_ENCODER_LEVELS_H264_1b	= 1,
        D3D12_VIDEO_ENCODER_LEVELS_H264_11	= 2,
        D3D12_VIDEO_ENCODER_LEVELS_H264_12	= 3,
        D3D12_VIDEO_ENCODER_LEVELS_H264_13	= 4,
        D3D12_VIDEO_ENCODER_LEVELS_H264_2	= 5,
        D3D12_VIDEO_ENCODER_LEVELS_H264_21	= 6,
        D3D12_VIDEO_ENCODER_LEVELS_H264_22	= 7,
        D3D12_VIDEO_ENCODER_LEVELS_H264_3	= 8,
        D3D12_VIDEO_ENCODER_LEVELS_H264_31	= 9,
        D3D12_VIDEO_ENCODER_LEVELS_H264_32	= 10,
        D3D12_VIDEO_ENCODER_LEVELS_H264_4	= 11,
        D3D12_VIDEO_ENCODER_LEVELS_H264_41	= 12,
        D3D12_VIDEO_ENCODER_LEVELS_H264_42	= 13,
        D3D12_VIDEO_ENCODER_LEVELS_H264_5	= 14,
        D3D12_VIDEO_ENCODER_LEVELS_H264_51	= 15,
        D3D12_VIDEO_ENCODER_LEVELS_H264_52	= 16,
        D3D12_VIDEO_ENCODER_LEVELS_H264_6	= 17,
        D3D12_VIDEO_ENCODER_LEVELS_H264_61	= 18,
        D3D12_VIDEO_ENCODER_LEVELS_H264_62	= 19
    } 	D3D12_VIDEO_ENCODER_LEVELS_H264;

typedef 
enum D3D12_VIDEO_ENCODER_TIER_HEVC
    {
        D3D12_VIDEO_ENCODER_TIER_HEVC_MAIN	= 0,
        D3D12_VIDEO_ENCODER_TIER_HEVC_HIGH	= 1
    } 	D3D12_VIDEO_ENCODER_TIER_HEVC;

typedef 
enum D3D12_VIDEO_ENCODER_LEVELS_HEVC
    {
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_1	= 0,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_2	= 1,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_21	= 2,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_3	= 3,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_31	= 4,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_4	= 5,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_41	= 6,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_5	= 7,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_51	= 8,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_52	= 9,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_6	= 10,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_61	= 11,
        D3D12_VIDEO_ENCODER_LEVELS_HEVC_62	= 12
    } 	D3D12_VIDEO_ENCODER_LEVELS_HEVC;

typedef struct D3D12_VIDEO_ENCODER_LEVEL_TIER_CONSTRAINTS_HEVC
    {
    D3D12_VIDEO_ENCODER_LEVELS_HEVC Level;
    D3D12_VIDEO_ENCODER_TIER_HEVC Tier;
    } 	D3D12_VIDEO_ENCODER_LEVEL_TIER_CONSTRAINTS_HEVC;

typedef struct D3D12_VIDEO_ENCODER_LEVEL_SETTING
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_LEVELS_H264 *pH264LevelSetting;
        D3D12_VIDEO_ENCODER_LEVEL_TIER_CONSTRAINTS_HEVC *pHEVCLevelSetting;
        D3D12_VIDEO_ENCODER_AV1_LEVEL_TIER_CONSTRAINTS *pAV1LevelSetting;
        } 	;
    } 	D3D12_VIDEO_ENCODER_LEVEL_SETTING;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_PROFILE_LEVEL
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    BOOL IsSupported;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING MinSupportedLevel;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING MaxSupportedLevel;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_PROFILE_LEVEL;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC
    {
    UINT Width;
    UINT Height;
    } 	D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_RATIO_DESC
    {
    UINT WidthRatio;
    UINT HeightRatio;
    } 	D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_RATIO_DESC;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_OUTPUT_RESOLUTION_RATIOS_COUNT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    UINT ResolutionRatiosCount;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_OUTPUT_RESOLUTION_RATIOS_COUNT;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_OUTPUT_RESOLUTION
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    UINT ResolutionRatiosCount;
    BOOL IsSupported;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC MinResolutionSupported;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC MaxResolutionSupported;
    UINT ResolutionWidthMultipleRequirement;
    UINT ResolutionHeightMultipleRequirement;
    _Field_size_full_(ResolutionRatiosCount)  D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_RATIO_DESC *pResolutionRatios;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_OUTPUT_RESOLUTION;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_INPUT_FORMAT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    DXGI_FORMAT Format;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_INPUT_FORMAT;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RATE_CONTROL_MODE
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE RateControlMode;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RATE_CONTROL_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE
    {
        D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE_NONE	= 0,
        D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE_ROW_BASED	= 1
    } 	D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_INTRA_REFRESH_MODE
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING Level;
    D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE IntraRefreshMode;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_INTRA_REFRESH_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE
    {
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_FULL_FRAME	= 0,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_BYTES_PER_SUBREGION	= 1,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_SQUARE_UNITS_PER_SUBREGION_ROW_UNALIGNED	= 2,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_UNIFORM_PARTITIONING_ROWS_PER_SUBREGION	= 3,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_UNIFORM_PARTITIONING_SUBREGIONS_PER_FRAME	= 4,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_UNIFORM_GRID_PARTITION	= 5,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_CONFIGURABLE_GRID_PARTITION	= 6,
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_AUTO	= 7
    } 	D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING Level;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionMode;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE;

typedef 
enum D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAGS
    {
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_NOT_SPECIFIED	= 0x1,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_CODEC_CONSTRAINT	= 0x2,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_HARDWARE_CONSTRAINT	= 0x4,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_ROWS_COUNT	= 0x8,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_COLS_COUNT	= 0x10,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_WIDTH	= 0x20,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_AREA	= 0x40,
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAG_TOTAL_TILES	= 0x80
    } 	D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT
    {
    BOOL Use128SuperBlocks;
    D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES TilesConfiguration;
    D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_VALIDATION_FLAGS ValidationFlags;
    UINT MinTileRows;
    UINT MaxTileRows;
    UINT MinTileCols;
    UINT MaxTileCols;
    UINT MinTileWidth;
    UINT MaxTileWidth;
    UINT MinTileArea;
    UINT MaxTileArea;
    UINT TileSizeBytesMinus1;
    } 	D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT;

typedef struct D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_AV1_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT *pAV1Support;
        } 	;
    } 	D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING Level;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionMode;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC FrameResolution;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG_SUPPORT CodecSupport;
    BOOL IsSupported;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_CONFIG;

typedef 
enum D3D12_VIDEO_ENCODER_HEAP_FLAGS
    {
        D3D12_VIDEO_ENCODER_HEAP_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_HEAP_FLAG_ALLOW_SUBREGION_NOTIFICATION_ARRAY_OF_BUFFERS	= 0x1,
        D3D12_VIDEO_ENCODER_HEAP_FLAG_ALLOW_SUBREGION_NOTIFICATION_SINGLE_BUFFER	= 0x2,
        D3D12_VIDEO_ENCODER_HEAP_FLAG_ALLOW_DIRTY_REGIONS	= 0x4,
        D3D12_VIDEO_ENCODER_HEAP_FLAG_ALLOW_RATE_CONTROL_FRAME_ANALYSIS	= 0x8
    } 	D3D12_VIDEO_ENCODER_HEAP_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_HEAP_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_HEAP_DESC
    {
    UINT NodeMask;
    D3D12_VIDEO_ENCODER_HEAP_FLAGS Flags;
    D3D12_VIDEO_ENCODER_CODEC EncodeCodec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC EncodeProfile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING EncodeLevel;
    UINT ResolutionsListCount;
    _Field_size_full_(ResolutionsListCount)  const D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList;
    } 	D3D12_VIDEO_ENCODER_HEAP_DESC;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_HEAP_SIZE
    {
    D3D12_VIDEO_ENCODER_HEAP_DESC HeapDesc;
    BOOL IsSupported;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_HEAP_SIZE;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAGS
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_CABAC_ENCODING_SUPPORT	= 0x1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_INTRA_SLICE_CONSTRAINED_ENCODING_SUPPORT	= 0x2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_BFRAME_LTR_COMBINED_SUPPORT	= 0x4,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_ADAPTIVE_8x8_TRANSFORM_ENCODING_SUPPORT	= 0x8,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_DIRECT_SPATIAL_ENCODING_SUPPORT	= 0x10,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_DIRECT_TEMPORAL_ENCODING_SUPPORT	= 0x20,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_CONSTRAINED_INTRAPREDICTION_SUPPORT	= 0x40,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAG_NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG_SLICE_SUPPORT	= 0x80
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODES
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_0_ALL_LUMA_CHROMA_SLICE_BLOCK_EDGES_ALWAYS_FILTERED	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_1_DISABLE_ALL_SLICE_BLOCK_EDGES	= 1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_2_DISABLE_SLICE_BOUNDARIES_BLOCKS	= 2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_3_USE_TWO_STAGE_DEBLOCKING	= 3,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_4_DISABLE_CHROMA_BLOCK_EDGES	= 4,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_5_DISABLE_CHROMA_BLOCK_EDGES_AND_LUMA_BOUNDARIES	= 5,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_6_DISABLE_CHROMA_BLOCK_EDGES_AND_USE_LUMA_TWO_STAGE_DEBLOCKING	= 6
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODES;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAGS
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_0_ALL_LUMA_CHROMA_SLICE_BLOCK_EDGES_ALWAYS_FILTERED	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_0_ALL_LUMA_CHROMA_SLICE_BLOCK_EDGES_ALWAYS_FILTERED ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_1_DISABLE_ALL_SLICE_BLOCK_EDGES	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_1_DISABLE_ALL_SLICE_BLOCK_EDGES ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_2_DISABLE_SLICE_BOUNDARIES_BLOCKS	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_2_DISABLE_SLICE_BOUNDARIES_BLOCKS ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_3_USE_TWO_STAGE_DEBLOCKING	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_3_USE_TWO_STAGE_DEBLOCKING ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_4_DISABLE_CHROMA_BLOCK_EDGES	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_4_DISABLE_CHROMA_BLOCK_EDGES ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_5_DISABLE_CHROMA_BLOCK_EDGES_AND_LUMA_BOUNDARIES	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_5_DISABLE_CHROMA_BLOCK_EDGES_AND_LUMA_BOUNDARIES ) ,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAG_6_DISABLE_CHROMA_BLOCK_EDGES_AND_USE_LUMA_TWO_STAGE_DEBLOCKING	= ( 1 << D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_6_DISABLE_CHROMA_BLOCK_EDGES_AND_USE_LUMA_TWO_STAGE_DEBLOCKING ) 
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264
    {
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODE_FLAGS DisableDeblockingFilterSupportedModes;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_BFRAME_LTR_COMBINED_SUPPORT	= 0x1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_INTRA_SLICE_CONSTRAINED_ENCODING_SUPPORT	= 0x2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CONSTRAINED_INTRAPREDICTION_SUPPORT	= 0x4,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_SAO_FILTER_SUPPORT	= 0x8,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_ASYMETRIC_MOTION_PARTITION_SUPPORT	= 0x10,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_ASYMETRIC_MOTION_PARTITION_REQUIRED	= 0x20,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_SUPPORT	= 0x40,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_DISABLING_LOOP_FILTER_ACROSS_SLICES_SUPPORT	= 0x80,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_P_FRAMES_IMPLEMENTED_AS_LOW_DELAY_B_FRAMES	= 0x100,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG_SLICE_SUPPORT	= 0x200,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_ROTATION_ENABLED_SUPPORT	= 0x400,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_ROTATION_ENABLED_REQUIRED	= 0x800,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_CONTEXT_ENABLED_SUPPORT	= 0x1000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_CONTEXT_ENABLED_REQUIRED	= 0x2000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_IMPLICIT_RDPCM_ENABLED_SUPPORT	= 0x4000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_IMPLICIT_RDPCM_ENABLED_REQUIRED	= 0x8000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXPLICIT_RDPCM_ENABLED_SUPPORT	= 0x10000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXPLICIT_RDPCM_ENABLED_REQUIRED	= 0x20000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXTENDED_PRECISION_PROCESSING_SUPPORT	= 0x40000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXTENDED_PRECISION_PROCESSING_REQUIRED	= 0x80000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_INTRA_SMOOTHING_DISABLED_SUPPORT	= 0x100000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_INTRA_SMOOTHING_DISABLED_REQUIRED	= 0x200000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_HIGH_PRECISION_OFFSETS_ENABLED_SUPPORT	= 0x400000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_HIGH_PRECISION_OFFSETS_ENABLED_REQUIRED	= 0x800000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_PERSISTENT_RICE_ADAPTATION_ENABLED_SUPPORT	= 0x1000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_PERSISTENT_RICE_ADAPTATION_ENABLED_REQUIRED	= 0x2000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CABAC_BYPASS_ALIGNMENT_ENABLED_SUPPORT	= 0x4000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CABAC_BYPASS_ALIGNMENT_ENABLED_REQUIRED	= 0x8000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CROSS_COMPONENT_PREDICTION_ENABLED_FLAG_SUPPORT	= 0x10000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CROSS_COMPONENT_PREDICTION_ENABLED_FLAG_REQUIRED	= 0x20000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CHROMA_QP_OFFSET_LIST_ENABLED_FLAG_SUPPORT	= 0x40000000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CHROMA_QP_OFFSET_LIST_ENABLED_FLAG_REQUIRED	= 0x80000000
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE_8x8	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE_16x16	= 1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE_32x32	= 2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE_64x64	= 3
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE_4x4	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE_8x8	= 1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE_16x16	= 2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE_32x32	= 3
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE;

typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC
    {
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MinLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MaxLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MinLumaTransformUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MaxLumaTransformUnitSize;
    UCHAR max_transform_hierarchy_depth_inter;
    UCHAR max_transform_hierarchy_depth_intra;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS1
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG1_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG1_SEPARATE_COLOUR_PLANE_SUPPORT	= 0x1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG1_SEPARATE_COLOUR_PLANE_REQUIRED	= 0x2
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS1;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS1)
typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC1
    {
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MinLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MaxLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MinLumaTransformUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MaxLumaTransformUnitSize;
    UCHAR max_transform_hierarchy_depth_inter;
    UCHAR max_transform_hierarchy_depth_intra;
    UINT allowed_diff_cu_chroma_qp_offset_depth_values;
    UINT allowed_log2_sao_offset_scale_luma_values;
    UINT allowed_log2_sao_offset_scale_chroma_values;
    UINT allowed_log2_max_transform_skip_block_size_minus2_values;
    UINT allowed_chroma_qp_offset_list_len_minus1_values;
    UINT allowed_cb_qp_offset_list_values[ 6 ];
    UINT allowed_cr_qp_offset_list_values[ 6 ];
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAGS1 SupportFlags1;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC1;

typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264 *pH264Support;
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC *pHEVCSupport;
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC1 *pHEVCSupport1;
        D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION_SUPPORT *pAV1Support;
        } 	;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    BOOL IsSupported;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT CodecSupportLimits;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT;

typedef struct D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_H264
    {
    UINT MaxL0ReferencesForP;
    UINT MaxL0ReferencesForB;
    UINT MaxL1ReferencesForB;
    UINT MaxLongTermReferences;
    UINT MaxDPBCapacity;
    } 	D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_H264;

typedef struct D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_HEVC
    {
    UINT MaxL0ReferencesForP;
    UINT MaxL0ReferencesForB;
    UINT MaxL1ReferencesForB;
    UINT MaxLongTermReferences;
    UINT MaxDPBCapacity;
    } 	D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_HEVC;

typedef struct D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_H264 *pH264Support;
        D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT_HEVC *pHEVCSupport;
        D3D12_VIDEO_ENCODER_CODEC_AV1_PICTURE_CONTROL_SUPPORT *pAV1Support;
        } 	;
    } 	D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    BOOL IsSupported;
    D3D12_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT PictureSupport;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC_PICTURE_CONTROL_SUPPORT;

typedef 
enum D3D12_VIDEO_ENCODER_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_GENERAL_SUPPORT_OK	= 0x1,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_RECONFIGURATION_AVAILABLE	= 0x2,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RESOLUTION_RECONFIGURATION_AVAILABLE	= 0x4,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_VBV_SIZE_CONFIG_AVAILABLE	= 0x8,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_FRAME_ANALYSIS_AVAILABLE	= 0x10,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RECONSTRUCTED_FRAMES_REQUIRE_TEXTURE_ARRAYS	= 0x20,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_DELTA_QP_AVAILABLE	= 0x40,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_SUBREGION_LAYOUT_RECONFIGURATION_AVAILABLE	= 0x80,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_ADJUSTABLE_QP_RANGE_AVAILABLE	= 0x100,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_INITIAL_QP_AVAILABLE	= 0x200,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_MAX_FRAME_SIZE_AVAILABLE	= 0x400,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_SEQUENCE_GOP_RECONFIGURATION_AVAILABLE	= 0x800,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_MOTION_ESTIMATION_PRECISION_MODE_LIMIT_AVAILABLE	= 0x1000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_EXTENSION1_SUPPORT	= 0x2000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_QUALITY_VS_SPEED_AVAILABLE	= 0x4000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_READABLE_RECONSTRUCTED_PICTURE_LAYOUT_AVAILABLE	= 0x8000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_PER_BLOCK_QP_MAP_METADATA_AVAILABLE	= 0x10000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_PER_BLOCK_SATD_MAP_METADATA_AVAILABLE	= 0x20000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_PER_BLOCK_RC_BIT_ALLOCATION_MAP_METADATA_AVAILABLE	= 0x40000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_SUBREGION_NOTIFICATION_ARRAY_OF_BUFFERS_AVAILABLE	= 0x80000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_SUBREGION_NOTIFICATION_SINGLE_BUFFER_AVAILABLE	= 0x100000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_FRAME_PSNR_METADATA_AVAILABLE	= 0x200000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_SUBREGIONS_PSNR_METADATA_AVAILABLE	= 0x400000,
        D3D12_VIDEO_ENCODER_SUPPORT_FLAG_RATE_CONTROL_SPATIAL_ADAPTIVE_QP_AVAILABLE	= 0x800000
    } 	D3D12_VIDEO_ENCODER_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_SUPPORT_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAGS
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_USE_CONSTRAINED_INTRAPREDICTION	= 0x1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_USE_ADAPTIVE_8x8_TRANSFORM	= 0x2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_ENABLE_CABAC_ENCODING	= 0x4,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAG_ALLOW_REQUEST_INTRA_CONSTRAINED_SLICES	= 0x8
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAGS)
typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES_DISABLED	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES_TEMPORAL	= 1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES_SPATIAL	= 2
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES;

typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264
    {
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_FLAGS ConfigurationFlags;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_DIRECT_MODES DirectModeConfig;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264_SLICES_DEBLOCKING_MODES DisableDeblockingFilterConfig;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264;

typedef 
enum D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAGS
    {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_DISABLE_LOOP_FILTER_ACROSS_SLICES	= 0x1,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_ALLOW_REQUEST_INTRA_CONSTRAINED_SLICES	= 0x2,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_ENABLE_SAO_FILTER	= 0x4,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_ENABLE_LONG_TERM_REFERENCES	= 0x8,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_USE_ASYMETRIC_MOTION_PARTITION	= 0x10,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_ENABLE_TRANSFORM_SKIPPING	= 0x20,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_USE_CONSTRAINED_INTRAPREDICTION	= 0x40,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_TRANSFORM_SKIP_ROTATION	= 0x80,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_TRANSFORM_SKIP_CONTEXT	= 0x100,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_IMPLICIT_RDPCM	= 0x200,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_EXPLICIT_RDPCM	= 0x400,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_EXTENDED_PRECISION_PROCESSING	= 0x800,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_INTRA_SMOOTHING_DISABLED	= 0x1000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_HIGH_PRECISION_OFFSETS	= 0x2000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_PERSISTENT_RICE_ADAPTATION	= 0x4000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_CABAC_BYPASS_ALIGNMENT	= 0x8000,
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAG_SEPARATE_COLOUR_PLANE	= 0x10000
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC
    {
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_FLAGS ConfigurationFlags;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MinLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE MaxLumaCodingUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MinLumaTransformUnitSize;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE MaxLumaTransformUnitSize;
    UCHAR max_transform_hierarchy_depth_inter;
    UCHAR max_transform_hierarchy_depth_intra;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC;

typedef struct D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264 *pH264Config;
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC *pHEVCConfig;
        D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION *pAV1Config;
        } 	;
    } 	D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION;

typedef struct D3D12_VIDEO_ENCODER_INTRA_REFRESH
    {
    D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE Mode;
    UINT IntraRefreshDuration;
    } 	D3D12_VIDEO_ENCODER_INTRA_REFRESH;

typedef 
enum D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE
    {
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_MAXIMUM	= 0,
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_FULL_PIXEL	= 1,
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_HALF_PIXEL	= 2,
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_QUARTER_PIXEL	= 3,
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE_EIGHTH_PIXEL	= 4
    } 	D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS
    {
    UINT MaxSubregionsNumber;
    UINT MaxIntraRefreshFrameDuration;
    UINT SubregionBlockPixelsSize;
    UINT QPMapRegionPixelsSize;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS;

typedef 
enum D3D12_VIDEO_ENCODER_VALIDATION_FLAGS
    {
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_CODEC_NOT_SUPPORTED	= 0x1,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_INPUT_FORMAT_NOT_SUPPORTED	= 0x8,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_CODEC_CONFIGURATION_NOT_SUPPORTED	= 0x10,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_RATE_CONTROL_MODE_NOT_SUPPORTED	= 0x20,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_RATE_CONTROL_CONFIGURATION_NOT_SUPPORTED	= 0x40,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_INTRA_REFRESH_MODE_NOT_SUPPORTED	= 0x80,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_SUBREGION_LAYOUT_MODE_NOT_SUPPORTED	= 0x100,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_RESOLUTION_NOT_SUPPORTED_IN_LIST	= 0x200,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_GOP_STRUCTURE_NOT_SUPPORTED	= 0x800,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_SUBREGION_LAYOUT_DATA_NOT_SUPPORTED	= 0x1000,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_QPMAP_NOT_SUPPORTED	= 0x2000,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_DIRTY_REGIONS_NOT_SUPPORTED	= 0x4000,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_MOTION_SEARCH_NOT_SUPPORTED	= 0x8000,
        D3D12_VIDEO_ENCODER_VALIDATION_FLAG_FRAME_ANALYSIS_NOT_SUPPORTED	= 0x10000
    } 	D3D12_VIDEO_ENCODER_VALIDATION_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_VALIDATION_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264
    {
    UINT GOPLength;
    UINT PPicturePeriod;
    UCHAR pic_order_cnt_type;
    UCHAR log2_max_frame_num_minus4;
    UCHAR log2_max_pic_order_cnt_lsb_minus4;
    } 	D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264;

typedef struct D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_HEVC
    {
    UINT GOPLength;
    UINT PPicturePeriod;
    UCHAR log2_max_pic_order_cnt_lsb_minus4;
    } 	D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_HEVC;

typedef struct D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264 *pH264GroupOfPictures;
        D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_HEVC *pHEVCGroupOfPictures;
        D3D12_VIDEO_ENCODER_AV1_SEQUENCE_STRUCTURE *pAV1SequenceStructure;
        } 	;
    } 	D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE CodecGopSequence;
    D3D12_VIDEO_ENCODER_RATE_CONTROL RateControl;
    D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE IntraRefresh;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionFrameEncoding;
    UINT ResolutionsListCount;
    const D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList;
    UINT MaxReferenceFramesInDPB;
    D3D12_VIDEO_ENCODER_VALIDATION_FLAGS ValidationFlags;
    D3D12_VIDEO_ENCODER_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_PROFILE_DESC SuggestedProfile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING SuggestedLevel;
    _Field_size_full_(ResolutionsListCount)  D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS *pResolutionDependentSupport;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_SLICES
    {
    union 
        {
        UINT MaxBytesPerSlice;
        UINT NumberOfCodingUnitsPerSlice;
        UINT NumberOfRowsPerSlice;
        UINT NumberOfSlicesPerFrame;
        } 	;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_SLICES;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA
    {
    UINT DataSize;
    union 
        {
        const D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_SLICES *pSlicesPartition_H264;
        const D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_SLICES *pSlicesPartition_HEVC;
        const D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES *pTilesPartition_AV1;
        } 	;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT1
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE CodecGopSequence;
    D3D12_VIDEO_ENCODER_RATE_CONTROL RateControl;
    D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE IntraRefresh;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionFrameEncoding;
    UINT ResolutionsListCount;
    const D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList;
    UINT MaxReferenceFramesInDPB;
    D3D12_VIDEO_ENCODER_VALIDATION_FLAGS ValidationFlags;
    D3D12_VIDEO_ENCODER_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_PROFILE_DESC SuggestedProfile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING SuggestedLevel;
    _Field_size_full_(ResolutionsListCount)  D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS *pResolutionDependentSupport;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA SubregionFrameEncodingData;
    UINT MaxQualityVsSpeed;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT1;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC PictureTargetResolution;
    BOOL IsSupported;
    UINT CompressedBitstreamBufferAccessAlignment;
    UINT EncoderMetadataBufferAccessAlignment;
    UINT MaxEncoderOutputMetadataBufferSize;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS;

typedef 
enum D3D12_VIDEO_ENCODER_FLAGS
    {
        D3D12_VIDEO_ENCODER_FLAG_NONE	= 0
    } 	D3D12_VIDEO_ENCODER_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_DESC
    {
    UINT NodeMask;
    D3D12_VIDEO_ENCODER_FLAGS Flags;
    D3D12_VIDEO_ENCODER_CODEC EncodeCodec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC EncodeProfile;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE MaxMotionEstimationPrecision;
    } 	D3D12_VIDEO_ENCODER_DESC;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0022_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0022_v0_0_s_ifspec;

#ifndef __ID3D12VideoEncoder_INTERFACE_DEFINED__
#define __ID3D12VideoEncoder_INTERFACE_DEFINED__

/* interface ID3D12VideoEncoder */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncoder;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("2E0D212D-8DF9-44A6-A770-BB289B182737")
    ID3D12VideoEncoder : public ID3D12Pageable
    {
    public:
        virtual UINT STDMETHODCALLTYPE GetNodeMask( void) = 0;
        
        virtual D3D12_VIDEO_ENCODER_FLAGS STDMETHODCALLTYPE GetEncoderFlags( void) = 0;
        
        virtual D3D12_VIDEO_ENCODER_CODEC STDMETHODCALLTYPE GetCodec( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetCodecProfile( 
            _Inout_  D3D12_VIDEO_ENCODER_PROFILE_DESC dstProfile) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetCodecConfiguration( 
            _Inout_  D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION dstCodecConfig) = 0;
        
        virtual DXGI_FORMAT STDMETHODCALLTYPE GetInputFormat( void) = 0;
        
        virtual D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE STDMETHODCALLTYPE GetMaxMotionEstimationPrecision( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncoderVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncoder * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncoder * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncoder * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncoder * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncoder * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncoder * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetNodeMask)
        UINT ( STDMETHODCALLTYPE *GetNodeMask )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetEncoderFlags)
        D3D12_VIDEO_ENCODER_FLAGS ( STDMETHODCALLTYPE *GetEncoderFlags )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetCodec)
        D3D12_VIDEO_ENCODER_CODEC ( STDMETHODCALLTYPE *GetCodec )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetCodecProfile)
        HRESULT ( STDMETHODCALLTYPE *GetCodecProfile )( 
            ID3D12VideoEncoder * This,
            _Inout_  D3D12_VIDEO_ENCODER_PROFILE_DESC dstProfile);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetCodecConfiguration)
        HRESULT ( STDMETHODCALLTYPE *GetCodecConfiguration )( 
            ID3D12VideoEncoder * This,
            _Inout_  D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION dstCodecConfig);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetInputFormat)
        DXGI_FORMAT ( STDMETHODCALLTYPE *GetInputFormat )( 
            ID3D12VideoEncoder * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoder, GetMaxMotionEstimationPrecision)
        D3D12_VIDEO_ENCODER_MOTION_ESTIMATION_PRECISION_MODE ( STDMETHODCALLTYPE *GetMaxMotionEstimationPrecision )( 
            ID3D12VideoEncoder * This);
        
        END_INTERFACE
    } ID3D12VideoEncoderVtbl;

    interface ID3D12VideoEncoder
    {
        CONST_VTBL struct ID3D12VideoEncoderVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncoder_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncoder_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncoder_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncoder_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncoder_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncoder_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncoder_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncoder_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 



#define ID3D12VideoEncoder_GetNodeMask(This)	\
    ( (This)->lpVtbl -> GetNodeMask(This) ) 

#define ID3D12VideoEncoder_GetEncoderFlags(This)	\
    ( (This)->lpVtbl -> GetEncoderFlags(This) ) 

#define ID3D12VideoEncoder_GetCodec(This)	\
    ( (This)->lpVtbl -> GetCodec(This) ) 

#define ID3D12VideoEncoder_GetCodecProfile(This,dstProfile)	\
    ( (This)->lpVtbl -> GetCodecProfile(This,dstProfile) ) 

#define ID3D12VideoEncoder_GetCodecConfiguration(This,dstCodecConfig)	\
    ( (This)->lpVtbl -> GetCodecConfiguration(This,dstCodecConfig) ) 

#define ID3D12VideoEncoder_GetInputFormat(This)	\
    ( (This)->lpVtbl -> GetInputFormat(This) ) 

#define ID3D12VideoEncoder_GetMaxMotionEstimationPrecision(This)	\
    ( (This)->lpVtbl -> GetMaxMotionEstimationPrecision(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncoder_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoEncoderHeap_INTERFACE_DEFINED__
#define __ID3D12VideoEncoderHeap_INTERFACE_DEFINED__

/* interface ID3D12VideoEncoderHeap */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncoderHeap;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("22B35D96-876A-44C0-B25E-FB8C9C7F1C4A")
    ID3D12VideoEncoderHeap : public ID3D12Pageable
    {
    public:
        virtual UINT STDMETHODCALLTYPE GetNodeMask( void) = 0;
        
        virtual D3D12_VIDEO_ENCODER_HEAP_FLAGS STDMETHODCALLTYPE GetEncoderHeapFlags( void) = 0;
        
        virtual D3D12_VIDEO_ENCODER_CODEC STDMETHODCALLTYPE GetCodec( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetCodecProfile( 
            _Inout_  D3D12_VIDEO_ENCODER_PROFILE_DESC dstProfile) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetCodecLevel( 
            _Inout_  D3D12_VIDEO_ENCODER_LEVEL_SETTING dstLevel) = 0;
        
        virtual UINT STDMETHODCALLTYPE GetResolutionListCount( void) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE GetResolutionList( 
            const UINT ResolutionsListCount,
            _Out_writes_(ResolutionsListCount)  D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncoderHeapVtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncoderHeap * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncoderHeap * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncoderHeap * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncoderHeap * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncoderHeap * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncoderHeap * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetNodeMask)
        UINT ( STDMETHODCALLTYPE *GetNodeMask )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetEncoderHeapFlags)
        D3D12_VIDEO_ENCODER_HEAP_FLAGS ( STDMETHODCALLTYPE *GetEncoderHeapFlags )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodec)
        D3D12_VIDEO_ENCODER_CODEC ( STDMETHODCALLTYPE *GetCodec )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodecProfile)
        HRESULT ( STDMETHODCALLTYPE *GetCodecProfile )( 
            ID3D12VideoEncoderHeap * This,
            _Inout_  D3D12_VIDEO_ENCODER_PROFILE_DESC dstProfile);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodecLevel)
        HRESULT ( STDMETHODCALLTYPE *GetCodecLevel )( 
            ID3D12VideoEncoderHeap * This,
            _Inout_  D3D12_VIDEO_ENCODER_LEVEL_SETTING dstLevel);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetResolutionListCount)
        UINT ( STDMETHODCALLTYPE *GetResolutionListCount )( 
            ID3D12VideoEncoderHeap * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetResolutionList)
        HRESULT ( STDMETHODCALLTYPE *GetResolutionList )( 
            ID3D12VideoEncoderHeap * This,
            const UINT ResolutionsListCount,
            _Out_writes_(ResolutionsListCount)  D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList);
        
        END_INTERFACE
    } ID3D12VideoEncoderHeapVtbl;

    interface ID3D12VideoEncoderHeap
    {
        CONST_VTBL struct ID3D12VideoEncoderHeapVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncoderHeap_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncoderHeap_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncoderHeap_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncoderHeap_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncoderHeap_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncoderHeap_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncoderHeap_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncoderHeap_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 



#define ID3D12VideoEncoderHeap_GetNodeMask(This)	\
    ( (This)->lpVtbl -> GetNodeMask(This) ) 

#define ID3D12VideoEncoderHeap_GetEncoderHeapFlags(This)	\
    ( (This)->lpVtbl -> GetEncoderHeapFlags(This) ) 

#define ID3D12VideoEncoderHeap_GetCodec(This)	\
    ( (This)->lpVtbl -> GetCodec(This) ) 

#define ID3D12VideoEncoderHeap_GetCodecProfile(This,dstProfile)	\
    ( (This)->lpVtbl -> GetCodecProfile(This,dstProfile) ) 

#define ID3D12VideoEncoderHeap_GetCodecLevel(This,dstLevel)	\
    ( (This)->lpVtbl -> GetCodecLevel(This,dstLevel) ) 

#define ID3D12VideoEncoderHeap_GetResolutionListCount(This)	\
    ( (This)->lpVtbl -> GetResolutionListCount(This) ) 

#define ID3D12VideoEncoderHeap_GetResolutionList(This,ResolutionsListCount,pResolutionList)	\
    ( (This)->lpVtbl -> GetResolutionList(This,ResolutionsListCount,pResolutionList) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncoderHeap_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDevice3_INTERFACE_DEFINED__
#define __ID3D12VideoDevice3_INTERFACE_DEFINED__

/* interface ID3D12VideoDevice3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDevice3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("4243ADB4-3A32-4666-973C-0CCC5625DC44")
    ID3D12VideoDevice3 : public ID3D12VideoDevice2
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateVideoEncoder( 
            _In_  const D3D12_VIDEO_ENCODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoder) = 0;
        
        virtual HRESULT STDMETHODCALLTYPE CreateVideoEncoderHeap( 
            _In_  const D3D12_VIDEO_ENCODER_HEAP_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoderHeap) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDevice3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDevice3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDevice3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDevice3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CheckFeatureSupport)
        HRESULT ( STDMETHODCALLTYPE *CheckFeatureSupport )( 
            ID3D12VideoDevice3 * This,
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoProcessor)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor )( 
            ID3D12VideoDevice3 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionEstimator)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionEstimator )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionEstimator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionVectorHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionVectorHeap )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionVectorHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoder1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder1 )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoderHeap1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap1 )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoProcessor1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor1 )( 
            ID3D12VideoDevice3 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoExtensionCommand )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_EXTENSION_COMMAND_DESC *pDesc,
            _In_reads_bytes_(CreationParametersDataSizeInBytes)  const void *pCreationParameters,
            SIZE_T CreationParametersDataSizeInBytes,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoExtensionCommand);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, ExecuteExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoDevice3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes,
            _Out_writes_bytes_(OutputDataSizeInBytes)  void *pOutputData,
            SIZE_T OutputDataSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice3, CreateVideoEncoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoEncoder )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_ENCODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice3, CreateVideoEncoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoEncoderHeap )( 
            ID3D12VideoDevice3 * This,
            _In_  const D3D12_VIDEO_ENCODER_HEAP_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoderHeap);
        
        END_INTERFACE
    } ID3D12VideoDevice3Vtbl;

    interface ID3D12VideoDevice3
    {
        CONST_VTBL struct ID3D12VideoDevice3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDevice3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDevice3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDevice3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDevice3_CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize)	\
    ( (This)->lpVtbl -> CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize) ) 

#define ID3D12VideoDevice3_CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice3_CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice3_CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor) ) 


#define ID3D12VideoDevice3_CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator)	\
    ( (This)->lpVtbl -> CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator) ) 

#define ID3D12VideoDevice3_CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap)	\
    ( (This)->lpVtbl -> CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap) ) 


#define ID3D12VideoDevice3_CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice3_CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice3_CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor) ) 

#define ID3D12VideoDevice3_CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand)	\
    ( (This)->lpVtbl -> CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand) ) 

#define ID3D12VideoDevice3_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes) ) 


#define ID3D12VideoDevice3_CreateVideoEncoder(This,pDesc,riid,ppVideoEncoder)	\
    ( (This)->lpVtbl -> CreateVideoEncoder(This,pDesc,riid,ppVideoEncoder) ) 

#define ID3D12VideoDevice3_CreateVideoEncoderHeap(This,pDesc,riid,ppVideoEncoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoEncoderHeap(This,pDesc,riid,ppVideoEncoderHeap) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDevice3_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0025 */
/* [local] */ 

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_TYPE_H264
    {
        D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_I_FRAME	= 0,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_P_FRAME	= 1,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_B_FRAME	= 2,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_IDR_FRAME	= 3
    } 	D3D12_VIDEO_ENCODER_FRAME_TYPE_H264;

typedef struct D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_H264
    {
    UINT ReconstructedPictureResourceIndex;
    BOOL IsLongTermReference;
    UINT LongTermPictureIdx;
    UINT PictureOrderCountNumber;
    UINT FrameDecodingOrderNumber;
    UINT TemporalLayerIndex;
    } 	D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_H264;

typedef 
enum D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAGS
    {
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAG_REQUEST_INTRA_CONSTRAINED_SLICES	= 0x1,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAG_REQUEST_NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG_SLICE	= 0x2
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_MARKING_OPERATION
    {
    UCHAR memory_management_control_operation;
    UINT difference_of_pic_nums_minus1;
    UINT long_term_pic_num;
    UINT long_term_frame_idx;
    UINT max_long_term_frame_idx_plus1;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_MARKING_OPERATION;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_LIST_MODIFICATION_OPERATION
    {
    UCHAR modification_of_pic_nums_idc;
    UINT abs_diff_pic_num_minus1;
    UINT long_term_pic_num;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_LIST_MODIFICATION_OPERATION;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264
    {
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_FLAGS Flags;
    D3D12_VIDEO_ENCODER_FRAME_TYPE_H264 FrameType;
    UINT pic_parameter_set_id;
    UINT idr_pic_id;
    UINT PictureOrderCountNumber;
    UINT FrameDecodingOrderNumber;
    UINT TemporalLayerIndex;
    UINT List0ReferenceFramesCount;
    _Field_size_full_(List0ReferenceFramesCount)  UINT *pList0ReferenceFrames;
    UINT List1ReferenceFramesCount;
    _Field_size_full_(List1ReferenceFramesCount)  UINT *pList1ReferenceFrames;
    UINT ReferenceFramesReconPictureDescriptorsCount;
    _Field_size_full_(ReferenceFramesReconPictureDescriptorsCount)  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_H264 *pReferenceFramesReconPictureDescriptors;
    UCHAR adaptive_ref_pic_marking_mode_flag;
    UINT RefPicMarkingOperationsCommandsCount;
    _Field_size_full_(RefPicMarkingOperationsCommandsCount)  D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_MARKING_OPERATION *pRefPicMarkingOperationsCommands;
    UINT List0RefPicModificationsCount;
    _Field_size_full_(List0RefPicModificationsCount)  D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_LIST_MODIFICATION_OPERATION *pList0RefPicModifications;
    UINT List1RefPicModificationsCount;
    _Field_size_full_(List1RefPicModificationsCount)  D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264_REFERENCE_PICTURE_LIST_MODIFICATION_OPERATION *pList1RefPicModifications;
    UINT QPMapValuesCount;
    _Field_size_full_(QPMapValuesCount)  INT8 *pRateControlQPMap;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264;

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC
    {
        D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_I_FRAME	= 0,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_P_FRAME	= 1,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_B_FRAME	= 2,
        D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_IDR_FRAME	= 3
    } 	D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC;

typedef struct D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC
    {
    UINT ReconstructedPictureResourceIndex;
    BOOL IsRefUsedByCurrentPic;
    BOOL IsLongTermReference;
    UINT PictureOrderCountNumber;
    UINT TemporalLayerIndex;
    } 	D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC;

typedef 
enum D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS
    {
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAG_REQUEST_INTRA_CONSTRAINED_SLICES	= 0x1,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAG_REQUEST_NUM_REF_IDX_ACTIVE_OVERRIDE_FLAG_SLICE	= 0x2,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAG_CROSS_COMPONENT_PREDICTION	= 0x4,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAG_CHROMA_QP_OFFSET_LIST	= 0x8
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC
    {
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS Flags;
    D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC FrameType;
    UINT slice_pic_parameter_set_id;
    UINT PictureOrderCountNumber;
    UINT TemporalLayerIndex;
    UINT List0ReferenceFramesCount;
    _Field_size_full_(List0ReferenceFramesCount)  UINT *pList0ReferenceFrames;
    UINT List1ReferenceFramesCount;
    _Field_size_full_(List1ReferenceFramesCount)  UINT *pList1ReferenceFrames;
    UINT ReferenceFramesReconPictureDescriptorsCount;
    _Field_size_full_(ReferenceFramesReconPictureDescriptorsCount)  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC *pReferenceFramesReconPictureDescriptors;
    UINT List0RefPicModificationsCount;
    _Field_size_full_(List0RefPicModificationsCount)  UINT *pList0RefPicModifications;
    UINT List1RefPicModificationsCount;
    _Field_size_full_(List1RefPicModificationsCount)  UINT *pList1RefPicModifications;
    UINT QPMapValuesCount;
    _Field_size_full_(QPMapValuesCount)  INT8 *pRateControlQPMap;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC1
    {
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS Flags;
    D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC FrameType;
    UINT slice_pic_parameter_set_id;
    UINT PictureOrderCountNumber;
    UINT TemporalLayerIndex;
    UINT List0ReferenceFramesCount;
    _Field_size_full_(List0ReferenceFramesCount)  UINT *pList0ReferenceFrames;
    UINT List1ReferenceFramesCount;
    _Field_size_full_(List1ReferenceFramesCount)  UINT *pList1ReferenceFrames;
    UINT ReferenceFramesReconPictureDescriptorsCount;
    _Field_size_full_(ReferenceFramesReconPictureDescriptorsCount)  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC *pReferenceFramesReconPictureDescriptors;
    UINT List0RefPicModificationsCount;
    _Field_size_full_(List0RefPicModificationsCount)  UINT *pList0RefPicModifications;
    UINT List1RefPicModificationsCount;
    _Field_size_full_(List1RefPicModificationsCount)  UINT *pList1RefPicModifications;
    UINT QPMapValuesCount;
    _Field_size_full_(QPMapValuesCount)  INT8 *pRateControlQPMap;
    UCHAR diff_cu_chroma_qp_offset_depth;
    UCHAR log2_sao_offset_scale_luma;
    UCHAR log2_sao_offset_scale_chroma;
    UCHAR log2_max_transform_skip_block_size_minus2;
    UCHAR chroma_qp_offset_list_len_minus1;
    CHAR cb_qp_offset_list[ 6 ];
    CHAR cr_qp_offset_list[ 6 ];
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC1;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264 *pH264PicData;
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC *pHEVCPicData;
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC1 *pHEVCPicData1;
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_CODEC_DATA *pAV1PicData;
        } 	;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA;

typedef struct D3D12_VIDEO_ENCODE_REFERENCE_FRAMES
    {
    UINT NumTexture2Ds;
    _Field_size_full_(NumTexture2Ds)  ID3D12Resource **ppTexture2Ds;
    _Field_size_full_(NumTexture2Ds)  UINT *pSubresources;
    } 	D3D12_VIDEO_ENCODE_REFERENCE_FRAMES;

typedef 
enum D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAGS
    {
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_USED_AS_REFERENCE_PICTURE	= 0x1,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_ENABLE_QUANTIZATION_MATRIX_INPUT	= 0x2,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_ENABLE_DIRTY_REGIONS_INPUT	= 0x4,
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_ENABLE_MOTION_VECTORS_INPUT	= 0x8
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC
    {
    UINT IntraRefreshFrameIndex;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAGS Flags;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA PictureControlCodecData;
    D3D12_VIDEO_ENCODE_REFERENCE_FRAMES ReferenceFrames;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC;

typedef 
enum D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAGS
    {
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_RESOLUTION_CHANGE	= 0x1,
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_RATE_CONTROL_CHANGE	= 0x2,
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_SUBREGION_LAYOUT_CHANGE	= 0x4,
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_REQUEST_INTRA_REFRESH	= 0x8,
        D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAG_GOP_SEQUENCE_CHANGE	= 0x10
    } 	D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_DESC
    {
    D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_FLAGS Flags;
    D3D12_VIDEO_ENCODER_INTRA_REFRESH IntraRefreshConfig;
    D3D12_VIDEO_ENCODER_RATE_CONTROL RateControl;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC PictureTargetResolution;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SelectedLayoutMode;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA FrameSubregionsLayoutData;
    D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE CodecGopSequence;
    } 	D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_DESC;

typedef struct D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS
    {
    D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_DESC SequenceControlDesc;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC PictureControlDesc;
    ID3D12Resource *pInputFrame;
    UINT InputFrameSubresource;
    UINT CurrentFrameBitstreamMetadataSize;
    } 	D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS;

typedef struct D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM
    {
    ID3D12Resource *pBuffer;
    UINT64 FrameStartOffset;
    } 	D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM;

typedef struct D3D12_VIDEO_ENCODER_RECONSTRUCTED_PICTURE
    {
    ID3D12Resource *pReconstructedPicture;
    UINT ReconstructedPictureSubresource;
    } 	D3D12_VIDEO_ENCODER_RECONSTRUCTED_PICTURE;

typedef struct D3D12_VIDEO_ENCODER_FRAME_SUBREGION_METADATA
    {
    UINT64 bSize;
    UINT64 bStartOffset;
    UINT64 bHeaderSize;
    } 	D3D12_VIDEO_ENCODER_FRAME_SUBREGION_METADATA;

typedef 
enum D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAGS
    {
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_NO_ERROR	= 0,
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_CODEC_PICTURE_CONTROL_NOT_SUPPORTED	= 0x1,
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_SUBREGION_LAYOUT_CONFIGURATION_NOT_SUPPORTED	= 0x2,
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_INVALID_REFERENCE_PICTURES	= 0x4,
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_RECONFIGURATION_REQUEST_NOT_SUPPORTED	= 0x8,
        D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_INVALID_METADATA_BUFFER_SOURCE	= 0x10
    } 	D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAGS)
typedef struct D3D12_VIDEO_ENCODER_OUTPUT_METADATA_STATISTICS
    {
    UINT64 AverageQP;
    UINT64 IntraCodingUnitsCount;
    UINT64 InterCodingUnitsCount;
    UINT64 SkipCodingUnitsCount;
    UINT64 AverageMotionEstimationXDirection;
    UINT64 AverageMotionEstimationYDirection;
    } 	D3D12_VIDEO_ENCODER_OUTPUT_METADATA_STATISTICS;

typedef struct D3D12_VIDEO_ENCODER_OUTPUT_METADATA
    {
    UINT64 EncodeErrorFlags;
    D3D12_VIDEO_ENCODER_OUTPUT_METADATA_STATISTICS EncodeStats;
    UINT64 EncodedBitstreamWrittenBytesCount;
    UINT64 WrittenSubregionsCount;
    } 	D3D12_VIDEO_ENCODER_OUTPUT_METADATA;

typedef struct D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER
    {
    ID3D12Resource *pBuffer;
    UINT64 Offset;
    } 	D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS
    {
    D3D12_VIDEO_ENCODER_CODEC EncoderCodec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC EncoderProfile;
    DXGI_FORMAT EncoderInputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC EncodedPictureEffectiveResolution;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER HWLayoutMetadata;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS
    {
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER ResolvedLayoutMetadata;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS;

typedef struct D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS
    {
    D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM Bitstream;
    D3D12_VIDEO_ENCODER_RECONSTRUCTED_PICTURE ReconstructedPicture;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER EncoderOutputMetadata;
    } 	D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0025_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0025_v0_0_s_ifspec;

#ifndef __ID3D12VideoEncodeCommandList2_INTERFACE_DEFINED__
#define __ID3D12VideoEncodeCommandList2_INTERFACE_DEFINED__

/* interface ID3D12VideoEncodeCommandList2 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncodeCommandList2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("895491e2-e701-46a9-9a1f-8d3480ed867a")
    ID3D12VideoEncodeCommandList2 : public ID3D12VideoEncodeCommandList1
    {
    public:
        virtual void STDMETHODCALLTYPE EncodeFrame( 
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS *pOutputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveEncoderOutputMetadata( 
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS *pOutputArguments) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncodeCommandList2Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncodeCommandList2 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncodeCommandList2 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoEncodeCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoEncodeCommandList2 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoEncodeCommandList2 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EstimateMotion)
        void ( STDMETHODCALLTYPE *EstimateMotion )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveMotionVectorHeap)
        void ( STDMETHODCALLTYPE *ResolveMotionVectorHeap )( 
            ID3D12VideoEncodeCommandList2 * This,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoEncodeCommandList2 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, EncodeFrame)
        void ( STDMETHODCALLTYPE *EncodeFrame )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, ResolveEncoderOutputMetadata)
        void ( STDMETHODCALLTYPE *ResolveEncoderOutputMetadata )( 
            ID3D12VideoEncodeCommandList2 * This,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS *pOutputArguments);
        
        END_INTERFACE
    } ID3D12VideoEncodeCommandList2Vtbl;

    interface ID3D12VideoEncodeCommandList2
    {
        CONST_VTBL struct ID3D12VideoEncodeCommandList2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncodeCommandList2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncodeCommandList2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncodeCommandList2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncodeCommandList2_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncodeCommandList2_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncodeCommandList2_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncodeCommandList2_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncodeCommandList2_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoEncodeCommandList2_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoEncodeCommandList2_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoEncodeCommandList2_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoEncodeCommandList2_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoEncodeCommandList2_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoEncodeCommandList2_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoEncodeCommandList2_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList2_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList2_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoEncodeCommandList2_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoEncodeCommandList2_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList2_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList2_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoEncodeCommandList2_EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList2_ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList2_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#define ID3D12VideoEncodeCommandList2_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 


#define ID3D12VideoEncodeCommandList2_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoEncodeCommandList2_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 


#define ID3D12VideoEncodeCommandList2_EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments) ) 

#define ID3D12VideoEncodeCommandList2_ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncodeCommandList2_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoEncodeCommandList3_INTERFACE_DEFINED__
#define __ID3D12VideoEncodeCommandList3_INTERFACE_DEFINED__

/* interface ID3D12VideoEncodeCommandList3 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncodeCommandList3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("7f027b22-1515-4e85-aa0d-026486580576")
    ID3D12VideoEncodeCommandList3 : public ID3D12VideoEncodeCommandList2
    {
    public:
        virtual void STDMETHODCALLTYPE Barrier( 
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncodeCommandList3Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncodeCommandList3 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncodeCommandList3 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoEncodeCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoEncodeCommandList3 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoEncodeCommandList3 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EstimateMotion)
        void ( STDMETHODCALLTYPE *EstimateMotion )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveMotionVectorHeap)
        void ( STDMETHODCALLTYPE *ResolveMotionVectorHeap )( 
            ID3D12VideoEncodeCommandList3 * This,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoEncodeCommandList3 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, EncodeFrame)
        void ( STDMETHODCALLTYPE *EncodeFrame )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, ResolveEncoderOutputMetadata)
        void ( STDMETHODCALLTYPE *ResolveEncoderOutputMetadata )( 
            ID3D12VideoEncodeCommandList3 * This,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList3, Barrier)
        void ( STDMETHODCALLTYPE *Barrier )( 
            ID3D12VideoEncodeCommandList3 * This,
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups);
        
        END_INTERFACE
    } ID3D12VideoEncodeCommandList3Vtbl;

    interface ID3D12VideoEncodeCommandList3
    {
        CONST_VTBL struct ID3D12VideoEncodeCommandList3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncodeCommandList3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncodeCommandList3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncodeCommandList3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncodeCommandList3_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncodeCommandList3_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncodeCommandList3_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncodeCommandList3_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncodeCommandList3_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoEncodeCommandList3_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoEncodeCommandList3_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoEncodeCommandList3_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoEncodeCommandList3_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoEncodeCommandList3_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoEncodeCommandList3_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoEncodeCommandList3_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList3_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList3_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoEncodeCommandList3_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoEncodeCommandList3_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList3_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList3_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoEncodeCommandList3_EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList3_ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList3_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#define ID3D12VideoEncodeCommandList3_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 


#define ID3D12VideoEncodeCommandList3_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoEncodeCommandList3_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 


#define ID3D12VideoEncodeCommandList3_EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments) ) 

#define ID3D12VideoEncodeCommandList3_ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments) ) 


#define ID3D12VideoEncodeCommandList3_Barrier(This,NumBarrierGroups,pBarrierGroups)	\
    ( (This)->lpVtbl -> Barrier(This,NumBarrierGroups,pBarrierGroups) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncodeCommandList3_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0027 */
/* [local] */ 

typedef struct D3D12_VIDEO_ENCODER_HEAP_DESC1
    {
    UINT NodeMask;
    D3D12_VIDEO_ENCODER_HEAP_FLAGS Flags;
    D3D12_VIDEO_ENCODER_CODEC EncodeCodec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC EncodeProfile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING EncodeLevel;
    UINT ResolutionsListCount;
    _Field_size_full_(ResolutionsListCount)  const D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList;
    UINT Pow2DownscaleFactor;
    } 	D3D12_VIDEO_ENCODER_HEAP_DESC1;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0027_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0027_v0_0_s_ifspec;

#ifndef __ID3D12VideoEncoderHeap1_INTERFACE_DEFINED__
#define __ID3D12VideoEncoderHeap1_INTERFACE_DEFINED__

/* interface ID3D12VideoEncoderHeap1 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncoderHeap1;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("ea8f1968-4aa0-43a4-9d30-ba86ec84d4f9")
    ID3D12VideoEncoderHeap1 : public ID3D12VideoEncoderHeap
    {
    public:
        virtual UINT STDMETHODCALLTYPE GetPow2DownscaleFactor( void) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncoderHeap1Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncoderHeap1 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncoderHeap1 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncoderHeap1 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncoderHeap1 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncoderHeap1 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncoderHeap1 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetNodeMask)
        UINT ( STDMETHODCALLTYPE *GetNodeMask )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetEncoderHeapFlags)
        D3D12_VIDEO_ENCODER_HEAP_FLAGS ( STDMETHODCALLTYPE *GetEncoderHeapFlags )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodec)
        D3D12_VIDEO_ENCODER_CODEC ( STDMETHODCALLTYPE *GetCodec )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodecProfile)
        HRESULT ( STDMETHODCALLTYPE *GetCodecProfile )( 
            ID3D12VideoEncoderHeap1 * This,
            _Inout_  D3D12_VIDEO_ENCODER_PROFILE_DESC dstProfile);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetCodecLevel)
        HRESULT ( STDMETHODCALLTYPE *GetCodecLevel )( 
            ID3D12VideoEncoderHeap1 * This,
            _Inout_  D3D12_VIDEO_ENCODER_LEVEL_SETTING dstLevel);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetResolutionListCount)
        UINT ( STDMETHODCALLTYPE *GetResolutionListCount )( 
            ID3D12VideoEncoderHeap1 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap, GetResolutionList)
        HRESULT ( STDMETHODCALLTYPE *GetResolutionList )( 
            ID3D12VideoEncoderHeap1 * This,
            const UINT ResolutionsListCount,
            _Out_writes_(ResolutionsListCount)  D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncoderHeap1, GetPow2DownscaleFactor)
        UINT ( STDMETHODCALLTYPE *GetPow2DownscaleFactor )( 
            ID3D12VideoEncoderHeap1 * This);
        
        END_INTERFACE
    } ID3D12VideoEncoderHeap1Vtbl;

    interface ID3D12VideoEncoderHeap1
    {
        CONST_VTBL struct ID3D12VideoEncoderHeap1Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncoderHeap1_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncoderHeap1_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncoderHeap1_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncoderHeap1_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncoderHeap1_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncoderHeap1_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncoderHeap1_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncoderHeap1_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 



#define ID3D12VideoEncoderHeap1_GetNodeMask(This)	\
    ( (This)->lpVtbl -> GetNodeMask(This) ) 

#define ID3D12VideoEncoderHeap1_GetEncoderHeapFlags(This)	\
    ( (This)->lpVtbl -> GetEncoderHeapFlags(This) ) 

#define ID3D12VideoEncoderHeap1_GetCodec(This)	\
    ( (This)->lpVtbl -> GetCodec(This) ) 

#define ID3D12VideoEncoderHeap1_GetCodecProfile(This,dstProfile)	\
    ( (This)->lpVtbl -> GetCodecProfile(This,dstProfile) ) 

#define ID3D12VideoEncoderHeap1_GetCodecLevel(This,dstLevel)	\
    ( (This)->lpVtbl -> GetCodecLevel(This,dstLevel) ) 

#define ID3D12VideoEncoderHeap1_GetResolutionListCount(This)	\
    ( (This)->lpVtbl -> GetResolutionListCount(This) ) 

#define ID3D12VideoEncoderHeap1_GetResolutionList(This,ResolutionsListCount,pResolutionList)	\
    ( (This)->lpVtbl -> GetResolutionList(This,ResolutionsListCount,pResolutionList) ) 


#define ID3D12VideoEncoderHeap1_GetPow2DownscaleFactor(This)	\
    ( (This)->lpVtbl -> GetPow2DownscaleFactor(This) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncoderHeap1_INTERFACE_DEFINED__ */


#ifndef __ID3D12VideoDevice4_INTERFACE_DEFINED__
#define __ID3D12VideoDevice4_INTERFACE_DEFINED__

/* interface ID3D12VideoDevice4 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoDevice4;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("e59ad09e-f1ae-42bb-8983-9f6e5586c4eb")
    ID3D12VideoDevice4 : public ID3D12VideoDevice3
    {
    public:
        virtual HRESULT STDMETHODCALLTYPE CreateVideoEncoderHeap1( 
            _In_  const D3D12_VIDEO_ENCODER_HEAP_DESC1 *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoderHeap) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoDevice4Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoDevice4 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoDevice4 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoDevice4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CheckFeatureSupport)
        HRESULT ( STDMETHODCALLTYPE *CheckFeatureSupport )( 
            ID3D12VideoDevice4 * This,
            D3D12_FEATURE_VIDEO FeatureVideo,
            _Inout_updates_bytes_(FeatureSupportDataSize)  void *pFeatureSupportData,
            UINT FeatureSupportDataSize);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoDecoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice, CreateVideoProcessor)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor )( 
            ID3D12VideoDevice4 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionEstimator)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionEstimator )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionEstimator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice1, CreateVideoMotionVectorHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoMotionVectorHeap )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoMotionVectorHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoder1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoder1 )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_DECODER_DESC *pDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoDecoderHeap1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoDecoderHeap1 )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_DECODER_HEAP_DESC *pVideoDecoderHeapDesc,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoDecoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoProcessor1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoProcessor1 )( 
            ID3D12VideoDevice4 * This,
            UINT NodeMask,
            _In_  const D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC *pOutputStreamDesc,
            UINT NumInputStreamDescs,
            _In_reads_(NumInputStreamDescs)  const D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC *pInputStreamDescs,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoProcessor);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, CreateVideoExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoExtensionCommand )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_EXTENSION_COMMAND_DESC *pDesc,
            _In_reads_bytes_(CreationParametersDataSizeInBytes)  const void *pCreationParameters,
            SIZE_T CreationParametersDataSizeInBytes,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoExtensionCommand);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice2, ExecuteExtensionCommand)
        HRESULT ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoDevice4 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes,
            _Out_writes_bytes_(OutputDataSizeInBytes)  void *pOutputData,
            SIZE_T OutputDataSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice3, CreateVideoEncoder)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoEncoder )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_ENCODER_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoder);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice3, CreateVideoEncoderHeap)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoEncoderHeap )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_ENCODER_HEAP_DESC *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoderHeap);
        
        DECLSPEC_XFGVIRT(ID3D12VideoDevice4, CreateVideoEncoderHeap1)
        HRESULT ( STDMETHODCALLTYPE *CreateVideoEncoderHeap1 )( 
            ID3D12VideoDevice4 * This,
            _In_  const D3D12_VIDEO_ENCODER_HEAP_DESC1 *pDesc,
            _In_  REFIID riid,
            _COM_Outptr_  void **ppVideoEncoderHeap);
        
        END_INTERFACE
    } ID3D12VideoDevice4Vtbl;

    interface ID3D12VideoDevice4
    {
        CONST_VTBL struct ID3D12VideoDevice4Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoDevice4_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoDevice4_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoDevice4_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoDevice4_CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize)	\
    ( (This)->lpVtbl -> CheckFeatureSupport(This,FeatureVideo,pFeatureSupportData,FeatureSupportDataSize) ) 

#define ID3D12VideoDevice4_CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder(This,pDesc,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice4_CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap(This,pVideoDecoderHeapDesc,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice4_CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,riid,ppVideoProcessor) ) 


#define ID3D12VideoDevice4_CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator)	\
    ( (This)->lpVtbl -> CreateVideoMotionEstimator(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionEstimator) ) 

#define ID3D12VideoDevice4_CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap)	\
    ( (This)->lpVtbl -> CreateVideoMotionVectorHeap(This,pDesc,pProtectedResourceSession,riid,ppVideoMotionVectorHeap) ) 


#define ID3D12VideoDevice4_CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder)	\
    ( (This)->lpVtbl -> CreateVideoDecoder1(This,pDesc,pProtectedResourceSession,riid,ppVideoDecoder) ) 

#define ID3D12VideoDevice4_CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoDecoderHeap1(This,pVideoDecoderHeapDesc,pProtectedResourceSession,riid,ppVideoDecoderHeap) ) 

#define ID3D12VideoDevice4_CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor)	\
    ( (This)->lpVtbl -> CreateVideoProcessor1(This,NodeMask,pOutputStreamDesc,NumInputStreamDescs,pInputStreamDescs,pProtectedResourceSession,riid,ppVideoProcessor) ) 

#define ID3D12VideoDevice4_CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand)	\
    ( (This)->lpVtbl -> CreateVideoExtensionCommand(This,pDesc,pCreationParameters,CreationParametersDataSizeInBytes,pProtectedResourceSession,riid,ppVideoExtensionCommand) ) 

#define ID3D12VideoDevice4_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes,pOutputData,OutputDataSizeInBytes) ) 


#define ID3D12VideoDevice4_CreateVideoEncoder(This,pDesc,riid,ppVideoEncoder)	\
    ( (This)->lpVtbl -> CreateVideoEncoder(This,pDesc,riid,ppVideoEncoder) ) 

#define ID3D12VideoDevice4_CreateVideoEncoderHeap(This,pDesc,riid,ppVideoEncoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoEncoderHeap(This,pDesc,riid,ppVideoEncoderHeap) ) 


#define ID3D12VideoDevice4_CreateVideoEncoderHeap1(This,pDesc,riid,ppVideoEncoderHeap)	\
    ( (This)->lpVtbl -> CreateVideoEncoderHeap1(This,pDesc,riid,ppVideoEncoderHeap) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoDevice4_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0029 */
/* [local] */ 

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_HEAP_SIZE1
    {
    D3D12_VIDEO_ENCODER_HEAP_DESC1 HeapDesc;
    BOOL IsSupported;
    UINT64 MemoryPoolL0Size;
    UINT64 MemoryPoolL1Size;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_HEAP_SIZE1;

typedef 
enum D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS
    {
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_QP_MAP	= 0x1,
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_SATD_MAP	= 0x2,
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_RC_BIT_ALLOCATION_MAP	= 0x4,
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_FRAME_PSNR	= 0x8,
        D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAG_SUBREGIONS_PSNR	= 0x10
    } 	D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS(D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS1
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC PictureTargetResolution;
    BOOL IsSupported;
    UINT CompressedBitstreamBufferAccessAlignment;
    UINT EncoderMetadataBufferAccessAlignment;
    UINT MaxEncoderOutputMetadataBufferSize;
    D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS OptionalMetadata;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC EncoderOutputMetadataQPMapTextureDimensions;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC EncoderOutputMetadataSATDMapTextureDimensions;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC EncoderOutputMetadataBitAllocationMapTextureDimensions;
    UINT EncoderOutputMetadataFramePSNRComponentsNumber;
    UINT EncoderOutputMetadataSubregionsPSNRComponentsNumber;
    UINT EncoderOutputMetadataSubregionsPSNRResolvedMetadataBufferSize;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS1;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_PSNR_RESOLVED_LAYOUT
    {
    float PSNRY;
    float PSNRU;
    float PSNRV;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_PSNR_RESOLVED_LAYOUT;

typedef 
enum D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE
    {
        D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE_CPU_BUFFER	= 0,
        D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE_GPU_TEXTURE	= 1
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE;

typedef 
enum D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE
    {
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE_DIRTY	= 0,
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE_SKIP	= 1
    } 	D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE;

typedef struct D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO
    {
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING Level;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC InputResolution;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionFrameEncoding;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA SubregionFrameEncodingData;
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO;

typedef 
enum D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE
    {
        D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE_QUANTIZATION_MATRIX	= 0,
        D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE_DIRTY_REGIONS	= 1,
        D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE_MOTION_VECTORS	= 2
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE;

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE
    {
        D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_FULL_SEARCH	= 0,
        D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_START_HINT	= 1,
        D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_START_HINT_LIMITED_DISTANCE	= 2
    } 	D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_QPMAP_INPUT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO SessionInfo;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    BOOL IsSupported;
    UINT MapSourcePreferenceRanking;
    UINT BlockSize;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_QPMAP_INPUT;

typedef 
enum D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAG_REPEAT_FRAME	= 0x1,
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAG_DIRTY_REGIONS	= 0x2,
        D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAG_DIRTY_REGIONS_REQUIRE_FULL_ROW	= 0x4
    } 	D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_DIRTY_REGIONS
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO SessionInfo;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE MapValuesType;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAGS SupportFlags;
    UINT MapSourcePreferenceRanking;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_DIRTY_REGIONS;

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION
    {
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_FULL_PIXEL	= 0,
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_HALF_PIXEL	= 1,
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_QUARTER_PIXEL	= 2
    } 	D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION;

typedef 
enum D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAG_FULL_PIXEL	= ( 1 << D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_FULL_PIXEL ) ,
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAG_HALF_PIXEL	= ( 1 << D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_HALF_PIXEL ) ,
        D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAG_QUARTER_PIXEL	= ( 1 << D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_QUARTER_PIXEL ) 
    } 	D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAGS )
typedef 
enum D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAG_SUPPORTED	= 0x1,
        D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAG_MULTIPLE_HINTS	= 0x2,
        D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAG_GPU_TEXTURE_MULTIPLE_REFERENCES	= 0x4
    } 	D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_MOTION_SEARCH
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO SessionInfo;
    D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE MotionSearchMode;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    BOOL BidirectionalRefFrameEnabled;
    D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAGS SupportFlags;
    UINT MaxMotionHints;
    UINT MinDeviation;
    UINT MaxDeviation;
    UINT MapSourcePreferenceRanking;
    D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAGS MotionUnitPrecisionSupport;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_MOTION_SEARCH;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO SessionInfo;
    D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE MapType;
    BOOL IsSupported;
    UINT64 MaxResolvedBufferAllocationSize;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT;

typedef struct D3D12_VIDEO_ENCODER_QPMAP_CONFIGURATION
    {
    BOOL Enabled;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    } 	D3D12_VIDEO_ENCODER_QPMAP_CONFIGURATION;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_QPMAP
    {
    UINT MapSourcePreferenceRanking;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_QPMAP;

typedef struct D3D12_VIDEO_ENCODER_DIRTY_REGIONS_CONFIGURATION
    {
    BOOL Enabled;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE MapValuesType;
    } 	D3D12_VIDEO_ENCODER_DIRTY_REGIONS_CONFIGURATION;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_DIRTY_REGIONS
    {
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_SUPPORT_FLAGS DirtyRegionsSupportFlags;
    UINT MapSourcePreferenceRanking;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_DIRTY_REGIONS;

typedef struct D3D12_VIDEO_ENCODER_MOTION_SEARCH_CONFIGURATION
    {
    BOOL Enabled;
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE MotionSearchMode;
    BOOL BidirectionalRefFrameEnabled;
    } 	D3D12_VIDEO_ENCODER_MOTION_SEARCH_CONFIGURATION;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_MOTION_SEARCH
    {
    UINT MaxMotionHints;
    UINT MinDeviation;
    UINT MaxDeviation;
    UINT MapSourcePreferenceRanking;
    D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION_SUPPORT_FLAGS MotionUnitPrecisionSupportFlags;
    D3D12_VIDEO_ENCODER_MOTION_SEARCH_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_MOTION_SEARCH;

typedef 
enum D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAGS
    {
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_INTRACODED_FRAME_SUPPORTED	= 0x1,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_UNIDIR_INTER_FRAME_SUPPORTED	= 0x2,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_BIDIR_INTER_FRAME_SUPPORTED	= 0x4,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_EXTERNAL_DPB_DOWNSCALING	= 0x8,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_DYNAMIC_1ST_PASS_SKIP	= 0x10,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_DYNAMIC_DOWNSCALE_FACTOR_CHANGE_KEY_FRAME	= 0x20,
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_SUPPORTED	= ( ( D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_INTRACODED_FRAME_SUPPORTED | D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_UNIDIR_INTER_FRAME_SUPPORTED )  | D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAG_BIDIR_INTER_FRAME_SUPPORTED ) 
    } 	D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAGS;

DEFINE_ENUM_FLAG_OPERATORS( D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAGS )
typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC Profile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING Level;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC InputResolution;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionFrameEncoding;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA SubregionFrameEncodingData;
    D3D12_VIDEO_ENCODER_QPMAP_CONFIGURATION QPMap;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_CONFIGURATION DirtyRegions;
    D3D12_VIDEO_ENCODER_MOTION_SEARCH_CONFIGURATION MotionSearch;
    UINT Pow2DownscaleFactor;
    D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_FRAME_ANALYSIS
    {
    D3D12_VIDEO_ENCODER_RATE_CONTROL_FRAME_ANALYSIS_SUPPORT_FLAGS SupportFlags;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_FRAME_ANALYSIS;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS1
    {
    UINT MaxSubregionsNumber;
    UINT MaxIntraRefreshFrameDuration;
    UINT SubregionBlockPixelsSize;
    UINT QPMapRegionPixelsSize;
    D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_QPMAP QPMap;
    D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_DIRTY_REGIONS DirtyRegions;
    D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_MOTION_SEARCH MotionSearch;
    D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_FRAME_ANALYSIS FrameAnalysis;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS1;

typedef struct D3D12_VIDEO_ENCODER_FRAME_ANALYSIS_CONFIGURATION
    {
    BOOL Enabled;
    UINT Pow2DownscaleFactor;
    } 	D3D12_VIDEO_ENCODER_FRAME_ANALYSIS_CONFIGURATION;

typedef struct D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT2
    {
    UINT NodeIndex;
    D3D12_VIDEO_ENCODER_CODEC Codec;
    DXGI_FORMAT InputFormat;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE CodecGopSequence;
    D3D12_VIDEO_ENCODER_RATE_CONTROL RateControl;
    D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE IntraRefresh;
    D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE SubregionFrameEncoding;
    UINT ResolutionsListCount;
    const D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC *pResolutionList;
    UINT MaxReferenceFramesInDPB;
    D3D12_VIDEO_ENCODER_VALIDATION_FLAGS ValidationFlags;
    D3D12_VIDEO_ENCODER_SUPPORT_FLAGS SupportFlags;
    D3D12_VIDEO_ENCODER_PROFILE_DESC SuggestedProfile;
    D3D12_VIDEO_ENCODER_LEVEL_SETTING SuggestedLevel;
    _Field_size_full_(ResolutionsListCount)  D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS1 *pResolutionDependentSupport;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA SubregionFrameEncodingData;
    UINT MaxQualityVsSpeed;
    D3D12_VIDEO_ENCODER_QPMAP_CONFIGURATION QPMap;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_CONFIGURATION DirtyRegions;
    D3D12_VIDEO_ENCODER_MOTION_SEARCH_CONFIGURATION MotionSearch;
    D3D12_VIDEO_ENCODER_FRAME_ANALYSIS_CONFIGURATION FrameAnalysis;
    } 	D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT2;

typedef struct D3D12_VIDEO_ENCODER_DIRTY_RECT_INFO
    {
    BOOL FullFrameIdentical;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE MapValuesType;
    UINT NumDirtyRects;
    _Field_size_full_(NumDirtyRects)  RECT *pDirtyRects;
    UINT SourceDPBFrameReference;
    } 	D3D12_VIDEO_ENCODER_DIRTY_RECT_INFO;

typedef struct D3D12_VIDEO_ENCODER_DIRTY_REGIONS
    {
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    union 
        {
        ID3D12Resource *pOpaqueLayoutBuffer;
        D3D12_VIDEO_ENCODER_DIRTY_RECT_INFO *pCPUBuffer;
        } 	;
    } 	D3D12_VIDEO_ENCODER_DIRTY_REGIONS;

typedef struct D3D12_VIDEO_ENCODER_QUANTIZATION_OPAQUE_MAP
    {
    ID3D12Resource *pOpaqueQuantizationMap;
    } 	D3D12_VIDEO_ENCODER_QUANTIZATION_OPAQUE_MAP;

typedef struct D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_CONFIG
    {
    D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE MotionSearchMode;
    UINT SearchDeviationLimit;
    } 	D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_CONFIG;

typedef struct D3D12_VIDEO_ENCODER_MOVE_RECT
    {
    POINT SourcePoint;
    RECT DestRect;
    } 	D3D12_VIDEO_ENCODER_MOVE_RECT;

typedef 
enum D3D12_VIDEO_ENCODER_MOVEREGION_INFO_FLAGS
    {
        D3D12_VIDEO_ENCODER_MOVEREGION_INFO_FLAG_NONE	= 0,
        D3D12_VIDEO_ENCODER_MOVEREGION_INFO_FLAG_MULTIPLE_HINTS	= 0x1
    } 	D3D12_VIDEO_ENCODER_MOVEREGION_INFO_FLAGS;

typedef struct D3D12_VIDEO_ENCODER_MOVEREGION_INFO
    {
    UINT NumMoveRegions;
    _Field_size_full_(NumMoveRegions)  D3D12_VIDEO_ENCODER_MOVE_RECT *pMoveRegions;
    D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_CONFIG MotionSearchModeConfiguration;
    UINT SourceDPBFrameReference;
    D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION MotionUnitPrecision;
    D3D12_VIDEO_ENCODER_MOVEREGION_INFO_FLAGS Flags;
    } 	D3D12_VIDEO_ENCODER_MOVEREGION_INFO;

typedef struct D3D12_VIDEO_ENCODER_FRAME_MOTION_VECTORS
    {
    D3D12_VIDEO_ENCODER_INPUT_MAP_SOURCE MapSource;
    union 
        {
        ID3D12Resource *pOpaqueLayoutBuffer;
        D3D12_VIDEO_ENCODER_MOVEREGION_INFO *pCPUBuffer;
        } 	;
    } 	D3D12_VIDEO_ENCODER_FRAME_MOTION_VECTORS;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC2
    {
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC_FLAGS Flags;
    D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC FrameType;
    UINT slice_pic_parameter_set_id;
    UINT PictureOrderCountNumber;
    UINT TemporalLayerIndex;
    UINT List0ReferenceFramesCount;
    _Field_size_full_(List0ReferenceFramesCount)  UINT *pList0ReferenceFrames;
    UINT List1ReferenceFramesCount;
    _Field_size_full_(List1ReferenceFramesCount)  UINT *pList1ReferenceFrames;
    UINT ReferenceFramesReconPictureDescriptorsCount;
    _Field_size_full_(ReferenceFramesReconPictureDescriptorsCount)  D3D12_VIDEO_ENCODER_REFERENCE_PICTURE_DESCRIPTOR_HEVC *pReferenceFramesReconPictureDescriptors;
    UINT List0RefPicModificationsCount;
    _Field_size_full_(List0RefPicModificationsCount)  UINT *pList0RefPicModifications;
    UINT List1RefPicModificationsCount;
    _Field_size_full_(List1RefPicModificationsCount)  UINT *pList1RefPicModifications;
    UINT QPMapValuesCount;
    _Field_size_full_(QPMapValuesCount)  INT8 *pRateControlQPMap;
    UCHAR diff_cu_chroma_qp_offset_depth;
    UCHAR log2_sao_offset_scale_luma;
    UCHAR log2_sao_offset_scale_chroma;
    UCHAR log2_max_transform_skip_block_size_minus2;
    UCHAR chroma_qp_offset_list_len_minus1;
    CHAR cb_qp_offset_list[ 6 ];
    CHAR cr_qp_offset_list[ 6 ];
    UINT num_ref_idx_l0_active_minus1;
    UINT num_ref_idx_l1_active_minus1;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC2;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA1
    {
    UINT DataSize;
    union 
        {
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_H264 *pH264PicData;
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA_HEVC2 *pHEVCPicData;
        D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_CODEC_DATA *pAV1PicData;
        } 	;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA1;

typedef struct D3D12_VIDEO_ENCODER_FRAME_ANALYSIS
    {
    ID3D12Resource *pDownscaledFrame;
    UINT64 Subresource;
    D3D12_VIDEO_ENCODE_REFERENCE_FRAMES DownscaledReferences;
    } 	D3D12_VIDEO_ENCODER_FRAME_ANALYSIS;

typedef struct D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC1
    {
    UINT IntraRefreshFrameIndex;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAGS Flags;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA1 PictureControlCodecData;
    D3D12_VIDEO_ENCODE_REFERENCE_FRAMES ReferenceFrames;
    D3D12_VIDEO_ENCODER_FRAME_MOTION_VECTORS MotionVectors;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS DirtyRects;
    D3D12_VIDEO_ENCODER_QUANTIZATION_OPAQUE_MAP QuantizationTextureMap;
    D3D12_VIDEO_ENCODER_FRAME_ANALYSIS FrameAnalysis;
    } 	D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC1;

typedef struct D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS1
    {
    D3D12_VIDEO_ENCODER_SEQUENCE_CONTROL_DESC SequenceControlDesc;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_DESC1 PictureControlDesc;
    ID3D12Resource *pInputFrame;
    UINT InputFrameSubresource;
    UINT CurrentFrameBitstreamMetadataSize;
    D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS OptionalMetadata;
    } 	D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS1;

typedef 
enum D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM_BUFFER_MODE
    {
        D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM_BUFFER_MODE_ARRAY_OF_BUFFERS	= 0,
        D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM_BUFFER_MODE_SINGLE_BUFFER	= 1
    } 	D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM_BUFFER_MODE;

typedef struct D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM
    {
    D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM_BUFFER_MODE BufferMode;
    UINT ExpectedSubregionCount;
    UINT64 *pSubregionBitstreamsBaseOffsets;
    ID3D12Resource **ppSubregionBitstreams;
    ID3D12Resource **ppSubregionSizes;
    ID3D12Resource **ppSubregionOffsets;
    ID3D12Fence **ppSubregionFences;
    UINT64 *pSubregionFenceValues;
    } 	D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM;

typedef 
enum D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM_NOTIFICATION_MODE
    {
        D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM_NOTIFICATION_MODE_FULL_FRAME	= 0,
        D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM_NOTIFICATION_MODE_SUBREGIONS	= 1
    } 	D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM_NOTIFICATION_MODE;

typedef struct D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM1
    {
    D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM_NOTIFICATION_MODE NotificationMode;
    union 
        {
        D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM FrameOutputBuffer;
        D3D12_VIDEO_ENCODER_SUBREGION_COMPRESSED_BITSTREAM SubregionOutputBuffers;
        } 	;
    } 	D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM1;

typedef struct D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS1
    {
    D3D12_VIDEO_ENCODER_COMPRESSED_BITSTREAM1 Bitstream;
    D3D12_VIDEO_ENCODER_RECONSTRUCTED_PICTURE ReconstructedPicture;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER EncoderOutputMetadata;
    D3D12_VIDEO_ENCODER_RECONSTRUCTED_PICTURE FrameAnalysisReconstructedPicture;
    } 	D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS1;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS1
    {
    D3D12_VIDEO_ENCODER_CODEC EncoderCodec;
    D3D12_VIDEO_ENCODER_PROFILE_DESC EncoderProfile;
    DXGI_FORMAT EncoderInputFormat;
    D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC EncodedPictureEffectiveResolution;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER HWLayoutMetadata;
    D3D12_VIDEO_ENCODER_OPTIONAL_METADATA_ENABLE_FLAGS OptionalMetadata;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION CodecConfiguration;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS1;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS1
    {
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER ResolvedLayoutMetadata;
    ID3D12Resource *pOutputQPMap;
    ID3D12Resource *pOutputSATDMap;
    ID3D12Resource *pOutputBitAllocationMap;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER ResolvedFramePSNRData;
    D3D12_VIDEO_ENCODER_ENCODE_OPERATION_METADATA_BUFFER ResolvedSubregionsPSNRData;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS1;

typedef struct D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_QUANTIZATION_MATRIX
    {
    ID3D12Resource *pQuantizationMap;
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_QUANTIZATION_MATRIX;

typedef struct D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_DIRTY_REGIONS
    {
    BOOL FullFrameIdentical;
    D3D12_VIDEO_ENCODER_DIRTY_REGIONS_MAP_VALUES_MODE MapValuesType;
    ID3D12Resource *pDirtyRegionsMap;
    UINT SourceDPBFrameReference;
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_DIRTY_REGIONS;

typedef struct D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_MOTION_VECTORS
    {
    D3D12_VIDEO_ENCODER_FRAME_MOTION_SEARCH_MODE_CONFIG MotionSearchModeConfiguration;
    UINT NumHintsPerPixel;
    _Field_size_full_(NumHintsPerPixel)  ID3D12Resource **ppMotionVectorMaps;
    _Field_size_full_(NumHintsPerPixel)  UINT *pMotionVectorMapsSubresources;
    _Field_size_full_(NumHintsPerPixel)  ID3D12Resource **ppMotionVectorMapsMetadata;
    _Field_size_full_(NumHintsPerPixel)  UINT *pMotionVectorMapsMetadataSubresources;
    D3D12_VIDEO_ENCODER_FRAME_INPUT_MOTION_UNIT_PRECISION MotionUnitPrecision;
    D3D12_VIDEO_ENCODER_PICTURE_CONTROL_CODEC_DATA1 PictureControlConfiguration;
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_MOTION_VECTORS;

typedef struct D3D12_VIDEO_ENCODER_INPUT_MAP_DATA
    {
    D3D12_VIDEO_ENCODER_INPUT_MAP_TYPE MapType;
    union 
        {
        D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_QUANTIZATION_MATRIX Quantization;
        D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_DIRTY_REGIONS DirtyRegions;
        D3D12_VIDEO_ENCODER_INPUT_MAP_DATA_MOTION_VECTORS MotionVectors;
        } 	;
    } 	D3D12_VIDEO_ENCODER_INPUT_MAP_DATA;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_INPUT_ARGUMENTS
    {
    D3D12_VIDEO_ENCODER_INPUT_MAP_SESSION_INFO SessionInfo;
    D3D12_VIDEO_ENCODER_INPUT_MAP_DATA InputData;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_INPUT_ARGUMENTS;

typedef struct D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_OUTPUT_ARGUMENTS
    {
    ID3D12Resource *pOpaqueLayoutBuffer;
    } 	D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_OUTPUT_ARGUMENTS;



extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0029_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0029_v0_0_s_ifspec;

#ifndef __ID3D12VideoEncodeCommandList4_INTERFACE_DEFINED__
#define __ID3D12VideoEncodeCommandList4_INTERFACE_DEFINED__

/* interface ID3D12VideoEncodeCommandList4 */
/* [unique][local][object][uuid] */ 


EXTERN_C const IID IID_ID3D12VideoEncodeCommandList4;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("69aeb5b7-55f2-4012-8b73-3a88d65a204c")
    ID3D12VideoEncodeCommandList4 : public ID3D12VideoEncodeCommandList3
    {
    public:
        virtual void STDMETHODCALLTYPE EncodeFrame1( 
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap1 *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS1 *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS1 *pOutputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveEncoderOutputMetadata1( 
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS1 *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS1 *pOutputArguments) = 0;
        
        virtual void STDMETHODCALLTYPE ResolveInputParamLayout( 
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_OUTPUT_ARGUMENTS *pOutputArguments) = 0;
        
    };
    
    
#else 	/* C style interface */

    typedef struct ID3D12VideoEncodeCommandList4Vtbl
    {
        BEGIN_INTERFACE
        
        DECLSPEC_XFGVIRT(IUnknown, QueryInterface)
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            ID3D12VideoEncodeCommandList4 * This,
            REFIID riid,
            _COM_Outptr_  void **ppvObject);
        
        DECLSPEC_XFGVIRT(IUnknown, AddRef)
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(IUnknown, Release)
        ULONG ( STDMETHODCALLTYPE *Release )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12Object, GetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *GetPrivateData )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  REFGUID guid,
            _Inout_  UINT *pDataSize,
            _Out_writes_bytes_opt_( *pDataSize )  void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateData)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateData )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  REFGUID guid,
            _In_  UINT DataSize,
            _In_reads_bytes_opt_( DataSize )  const void *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetPrivateDataInterface)
        HRESULT ( STDMETHODCALLTYPE *SetPrivateDataInterface )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  REFGUID guid,
            _In_opt_  const IUnknown *pData);
        
        DECLSPEC_XFGVIRT(ID3D12Object, SetName)
        HRESULT ( STDMETHODCALLTYPE *SetName )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_z_  LPCWSTR Name);
        
        DECLSPEC_XFGVIRT(ID3D12DeviceChild, GetDevice)
        HRESULT ( STDMETHODCALLTYPE *GetDevice )( 
            ID3D12VideoEncodeCommandList4 * This,
            REFIID riid,
            _COM_Outptr_opt_  void **ppvDevice);
        
        DECLSPEC_XFGVIRT(ID3D12CommandList, GetType)
        D3D12_COMMAND_LIST_TYPE ( STDMETHODCALLTYPE *GetType )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Close)
        HRESULT ( STDMETHODCALLTYPE *Close )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, Reset)
        HRESULT ( STDMETHODCALLTYPE *Reset )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12CommandAllocator *pAllocator);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ClearState)
        void ( STDMETHODCALLTYPE *ClearState )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResourceBarrier)
        void ( STDMETHODCALLTYPE *ResourceBarrier )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  UINT NumBarriers,
            _In_reads_(NumBarriers)  const D3D12_RESOURCE_BARRIER *pBarriers);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, DiscardResource)
        void ( STDMETHODCALLTYPE *DiscardResource )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12Resource *pResource,
            _In_opt_  const D3D12_DISCARD_REGION *pRegion);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginQuery)
        void ( STDMETHODCALLTYPE *BeginQuery )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndQuery)
        void ( STDMETHODCALLTYPE *EndQuery )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT Index);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveQueryData)
        void ( STDMETHODCALLTYPE *ResolveQueryData )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12QueryHeap *pQueryHeap,
            _In_  D3D12_QUERY_TYPE Type,
            _In_  UINT StartIndex,
            _In_  UINT NumQueries,
            _In_  ID3D12Resource *pDestinationBuffer,
            _In_  UINT64 AlignedDestinationBufferOffset);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetPredication)
        void ( STDMETHODCALLTYPE *SetPredication )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_opt_  ID3D12Resource *pBuffer,
            _In_  UINT64 AlignedBufferOffset,
            _In_  D3D12_PREDICATION_OP Operation);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetMarker)
        void ( STDMETHODCALLTYPE *SetMarker )( 
            ID3D12VideoEncodeCommandList4 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, BeginEvent)
        void ( STDMETHODCALLTYPE *BeginEvent )( 
            ID3D12VideoEncodeCommandList4 * This,
            UINT Metadata,
            _In_reads_bytes_opt_(Size)  const void *pData,
            UINT Size);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EndEvent)
        void ( STDMETHODCALLTYPE *EndEvent )( 
            ID3D12VideoEncodeCommandList4 * This);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, EstimateMotion)
        void ( STDMETHODCALLTYPE *EstimateMotion )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12VideoMotionEstimator *pMotionEstimator,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT *pOutputArguments,
            _In_  const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, ResolveMotionVectorHeap)
        void ( STDMETHODCALLTYPE *ResolveMotionVectorHeap )( 
            ID3D12VideoEncodeCommandList4 * This,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT *pOutputArguments,
            const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT *pInputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, WriteBufferImmediate)
        void ( STDMETHODCALLTYPE *WriteBufferImmediate )( 
            ID3D12VideoEncodeCommandList4 * This,
            UINT Count,
            _In_reads_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *pParams,
            _In_reads_opt_(Count)  const D3D12_WRITEBUFFERIMMEDIATE_MODE *pModes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList, SetProtectedResourceSession)
        void ( STDMETHODCALLTYPE *SetProtectedResourceSession )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_opt_  ID3D12ProtectedResourceSession *pProtectedResourceSession);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, InitializeExtensionCommand)
        void ( STDMETHODCALLTYPE *InitializeExtensionCommand )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(InitializationParametersSizeInBytes)  const void *pInitializationParameters,
            SIZE_T InitializationParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList1, ExecuteExtensionCommand)
        void ( STDMETHODCALLTYPE *ExecuteExtensionCommand )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12VideoExtensionCommand *pExtensionCommand,
            _In_reads_bytes_(ExecutionParametersSizeInBytes)  const void *pExecutionParameters,
            SIZE_T ExecutionParametersSizeInBytes);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, EncodeFrame)
        void ( STDMETHODCALLTYPE *EncodeFrame )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList2, ResolveEncoderOutputMetadata)
        void ( STDMETHODCALLTYPE *ResolveEncoderOutputMetadata )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList3, Barrier)
        void ( STDMETHODCALLTYPE *Barrier )( 
            ID3D12VideoEncodeCommandList4 * This,
            UINT32 NumBarrierGroups,
            _In_reads_(NumBarrierGroups)  const D3D12_BARRIER_GROUP *pBarrierGroups);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList4, EncodeFrame1)
        void ( STDMETHODCALLTYPE *EncodeFrame1 )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  ID3D12VideoEncoder *pEncoder,
            _In_  ID3D12VideoEncoderHeap1 *pHeap,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_INPUT_ARGUMENTS1 *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS1 *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList4, ResolveEncoderOutputMetadata1)
        void ( STDMETHODCALLTYPE *ResolveEncoderOutputMetadata1 )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS1 *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS1 *pOutputArguments);
        
        DECLSPEC_XFGVIRT(ID3D12VideoEncodeCommandList4, ResolveInputParamLayout)
        void ( STDMETHODCALLTYPE *ResolveInputParamLayout )( 
            ID3D12VideoEncodeCommandList4 * This,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_INPUT_ARGUMENTS *pInputArguments,
            _In_  const D3D12_VIDEO_ENCODER_RESOLVE_INPUT_PARAM_LAYOUT_OUTPUT_ARGUMENTS *pOutputArguments);
        
        END_INTERFACE
    } ID3D12VideoEncodeCommandList4Vtbl;

    interface ID3D12VideoEncodeCommandList4
    {
        CONST_VTBL struct ID3D12VideoEncodeCommandList4Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define ID3D12VideoEncodeCommandList4_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define ID3D12VideoEncodeCommandList4_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define ID3D12VideoEncodeCommandList4_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define ID3D12VideoEncodeCommandList4_GetPrivateData(This,guid,pDataSize,pData)	\
    ( (This)->lpVtbl -> GetPrivateData(This,guid,pDataSize,pData) ) 

#define ID3D12VideoEncodeCommandList4_SetPrivateData(This,guid,DataSize,pData)	\
    ( (This)->lpVtbl -> SetPrivateData(This,guid,DataSize,pData) ) 

#define ID3D12VideoEncodeCommandList4_SetPrivateDataInterface(This,guid,pData)	\
    ( (This)->lpVtbl -> SetPrivateDataInterface(This,guid,pData) ) 

#define ID3D12VideoEncodeCommandList4_SetName(This,Name)	\
    ( (This)->lpVtbl -> SetName(This,Name) ) 


#define ID3D12VideoEncodeCommandList4_GetDevice(This,riid,ppvDevice)	\
    ( (This)->lpVtbl -> GetDevice(This,riid,ppvDevice) ) 


#define ID3D12VideoEncodeCommandList4_GetType(This)	\
    ( (This)->lpVtbl -> GetType(This) ) 


#define ID3D12VideoEncodeCommandList4_Close(This)	\
    ( (This)->lpVtbl -> Close(This) ) 

#define ID3D12VideoEncodeCommandList4_Reset(This,pAllocator)	\
    ( (This)->lpVtbl -> Reset(This,pAllocator) ) 

#define ID3D12VideoEncodeCommandList4_ClearState(This)	\
    ( (This)->lpVtbl -> ClearState(This) ) 

#define ID3D12VideoEncodeCommandList4_ResourceBarrier(This,NumBarriers,pBarriers)	\
    ( (This)->lpVtbl -> ResourceBarrier(This,NumBarriers,pBarriers) ) 

#define ID3D12VideoEncodeCommandList4_DiscardResource(This,pResource,pRegion)	\
    ( (This)->lpVtbl -> DiscardResource(This,pResource,pRegion) ) 

#define ID3D12VideoEncodeCommandList4_BeginQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> BeginQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList4_EndQuery(This,pQueryHeap,Type,Index)	\
    ( (This)->lpVtbl -> EndQuery(This,pQueryHeap,Type,Index) ) 

#define ID3D12VideoEncodeCommandList4_ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset)	\
    ( (This)->lpVtbl -> ResolveQueryData(This,pQueryHeap,Type,StartIndex,NumQueries,pDestinationBuffer,AlignedDestinationBufferOffset) ) 

#define ID3D12VideoEncodeCommandList4_SetPredication(This,pBuffer,AlignedBufferOffset,Operation)	\
    ( (This)->lpVtbl -> SetPredication(This,pBuffer,AlignedBufferOffset,Operation) ) 

#define ID3D12VideoEncodeCommandList4_SetMarker(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> SetMarker(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList4_BeginEvent(This,Metadata,pData,Size)	\
    ( (This)->lpVtbl -> BeginEvent(This,Metadata,pData,Size) ) 

#define ID3D12VideoEncodeCommandList4_EndEvent(This)	\
    ( (This)->lpVtbl -> EndEvent(This) ) 

#define ID3D12VideoEncodeCommandList4_EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> EstimateMotion(This,pMotionEstimator,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList4_ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments)	\
    ( (This)->lpVtbl -> ResolveMotionVectorHeap(This,pOutputArguments,pInputArguments) ) 

#define ID3D12VideoEncodeCommandList4_WriteBufferImmediate(This,Count,pParams,pModes)	\
    ( (This)->lpVtbl -> WriteBufferImmediate(This,Count,pParams,pModes) ) 

#define ID3D12VideoEncodeCommandList4_SetProtectedResourceSession(This,pProtectedResourceSession)	\
    ( (This)->lpVtbl -> SetProtectedResourceSession(This,pProtectedResourceSession) ) 


#define ID3D12VideoEncodeCommandList4_InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes)	\
    ( (This)->lpVtbl -> InitializeExtensionCommand(This,pExtensionCommand,pInitializationParameters,InitializationParametersSizeInBytes) ) 

#define ID3D12VideoEncodeCommandList4_ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes)	\
    ( (This)->lpVtbl -> ExecuteExtensionCommand(This,pExtensionCommand,pExecutionParameters,ExecutionParametersSizeInBytes) ) 


#define ID3D12VideoEncodeCommandList4_EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> EncodeFrame(This,pEncoder,pHeap,pInputArguments,pOutputArguments) ) 

#define ID3D12VideoEncodeCommandList4_ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> ResolveEncoderOutputMetadata(This,pInputArguments,pOutputArguments) ) 


#define ID3D12VideoEncodeCommandList4_Barrier(This,NumBarrierGroups,pBarrierGroups)	\
    ( (This)->lpVtbl -> Barrier(This,NumBarrierGroups,pBarrierGroups) ) 


#define ID3D12VideoEncodeCommandList4_EncodeFrame1(This,pEncoder,pHeap,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> EncodeFrame1(This,pEncoder,pHeap,pInputArguments,pOutputArguments) ) 

#define ID3D12VideoEncodeCommandList4_ResolveEncoderOutputMetadata1(This,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> ResolveEncoderOutputMetadata1(This,pInputArguments,pOutputArguments) ) 

#define ID3D12VideoEncodeCommandList4_ResolveInputParamLayout(This,pInputArguments,pOutputArguments)	\
    ( (This)->lpVtbl -> ResolveInputParamLayout(This,pInputArguments,pOutputArguments) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __ID3D12VideoEncodeCommandList4_INTERFACE_DEFINED__ */


/* interface __MIDL_itf_d3d12video_0000_0030 */
/* [local] */ 

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP | WINAPI_PARTITION_GAMES) */
#pragma endregion
DEFINE_GUID(IID_ID3D12VideoDecoderHeap,0x0946B7C9,0xEBF6,0x4047,0xBB,0x73,0x86,0x83,0xE2,0x7D,0xBB,0x1F);
DEFINE_GUID(IID_ID3D12VideoDevice,0x1F052807,0x0B46,0x4ACC,0x8A,0x89,0x36,0x4F,0x79,0x37,0x18,0xA4);
DEFINE_GUID(IID_ID3D12VideoDecoder,0xC59B6BDC,0x7720,0x4074,0xA1,0x36,0x17,0xA1,0x56,0x03,0x74,0x70);
DEFINE_GUID(IID_ID3D12VideoProcessor,0x304FDB32,0xBEDE,0x410A,0x85,0x45,0x94,0x3A,0xC6,0xA4,0x61,0x38);
DEFINE_GUID(IID_ID3D12VideoDecodeCommandList,0x3B60536E,0xAD29,0x4E64,0xA2,0x69,0xF8,0x53,0x83,0x7E,0x5E,0x53);
DEFINE_GUID(IID_ID3D12VideoProcessCommandList,0xAEB2543A,0x167F,0x4682,0xAC,0xC8,0xD1,0x59,0xED,0x4A,0x62,0x09);
DEFINE_GUID(IID_ID3D12VideoDecodeCommandList1,0xD52F011B,0xB56E,0x453C,0xA0,0x5A,0xA7,0xF3,0x11,0xC8,0xF4,0x72);
DEFINE_GUID(IID_ID3D12VideoProcessCommandList1,0x542C5C4D,0x7596,0x434F,0x8C,0x93,0x4E,0xFA,0x67,0x66,0xF2,0x67);
DEFINE_GUID(IID_ID3D12VideoMotionEstimator,0x33FDAE0E,0x098B,0x428F,0x87,0xBB,0x34,0xB6,0x95,0xDE,0x08,0xF8);
DEFINE_GUID(IID_ID3D12VideoMotionVectorHeap,0x5BE17987,0x743A,0x4061,0x83,0x4B,0x23,0xD2,0x2D,0xAE,0xA5,0x05);
DEFINE_GUID(IID_ID3D12VideoDevice1,0x981611AD,0xA144,0x4C83,0x98,0x90,0xF3,0x0E,0x26,0xD6,0x58,0xAB);
DEFINE_GUID(IID_ID3D12VideoEncodeCommandList,0x8455293A,0x0CBD,0x4831,0x9B,0x39,0xFB,0xDB,0xAB,0x72,0x47,0x23);
DEFINE_GUID(IID_ID3D12VideoDecoder1,0x79A2E5FB,0xCCD2,0x469A,0x9F,0xDE,0x19,0x5D,0x10,0x95,0x1F,0x7E);
DEFINE_GUID(IID_ID3D12VideoDecoderHeap1,0xDA1D98C5,0x539F,0x41B2,0xBF,0x6B,0x11,0x98,0xA0,0x3B,0x6D,0x26);
DEFINE_GUID(IID_ID3D12VideoProcessor1,0xF3CFE615,0x553F,0x425C,0x86,0xD8,0xEE,0x8C,0x1B,0x1F,0xB0,0x1C);
DEFINE_GUID(IID_ID3D12VideoExtensionCommand,0x554E41E8,0xAE8E,0x4A8C,0xB7,0xD2,0x5B,0x4F,0x27,0x4A,0x30,0xE4);
DEFINE_GUID(IID_ID3D12VideoDevice2,0xF019AC49,0xF838,0x4A95,0x9B,0x17,0x57,0x94,0x37,0xC8,0xF5,0x13);
DEFINE_GUID(IID_ID3D12VideoDecodeCommandList2,0x6e120880,0xc114,0x4153,0x80,0x36,0xd2,0x47,0x05,0x1e,0x17,0x29);
DEFINE_GUID(IID_ID3D12VideoDecodeCommandList3,0x2aee8c37,0x9562,0x42da,0x8a,0xbf,0x61,0xef,0xeb,0x2e,0x45,0x13);
DEFINE_GUID(IID_ID3D12VideoProcessCommandList2,0xdb525ae4,0x6ad6,0x473c,0xba,0xa7,0x59,0xb2,0xe3,0x70,0x82,0xe4);
DEFINE_GUID(IID_ID3D12VideoProcessCommandList3,0x1a0a4ca4,0x9f08,0x40ce,0x95,0x58,0xb4,0x11,0xfd,0x26,0x66,0xff);
DEFINE_GUID(IID_ID3D12VideoEncodeCommandList1,0x94971eca,0x2bdb,0x4769,0x88,0xcf,0x36,0x75,0xea,0x75,0x7e,0xbc);
DEFINE_GUID(IID_ID3D12VideoEncoder,0x2E0D212D,0x8DF9,0x44A6,0xA7,0x70,0xBB,0x28,0x9B,0x18,0x27,0x37);
DEFINE_GUID(IID_ID3D12VideoEncoderHeap,0x22B35D96,0x876A,0x44C0,0xB2,0x5E,0xFB,0x8C,0x9C,0x7F,0x1C,0x4A);
DEFINE_GUID(IID_ID3D12VideoDevice3,0x4243ADB4,0x3A32,0x4666,0x97,0x3C,0x0C,0xCC,0x56,0x25,0xDC,0x44);
DEFINE_GUID(IID_ID3D12VideoEncodeCommandList2,0x895491e2,0xe701,0x46a9,0x9a,0x1f,0x8d,0x34,0x80,0xed,0x86,0x7a);
DEFINE_GUID(IID_ID3D12VideoEncodeCommandList3,0x7f027b22,0x1515,0x4e85,0xaa,0x0d,0x02,0x64,0x86,0x58,0x05,0x76);
DEFINE_GUID(IID_ID3D12VideoEncoderHeap1,0xea8f1968,0x4aa0,0x43a4,0x9d,0x30,0xba,0x86,0xec,0x84,0xd4,0xf9);
DEFINE_GUID(IID_ID3D12VideoDevice4,0xe59ad09e,0xf1ae,0x42bb,0x89,0x83,0x9f,0x6e,0x55,0x86,0xc4,0xeb);
DEFINE_GUID(IID_ID3D12VideoEncodeCommandList4,0x69aeb5b7,0x55f2,0x4012,0x8b,0x73,0x3a,0x88,0xd6,0x5a,0x20,0x4c);


extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0030_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_d3d12video_0000_0030_v0_0_s_ifspec;

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


