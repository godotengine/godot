// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_OPTICALFLOW_CALLBACKS_HLSL_H
#define FFX_OPTICALFLOW_CALLBACKS_HLSL_H

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif //__hlsl_dx_compiler
#include "ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif //__hlsl_dx_compiler

#define FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION 1
#define FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS 1
#define FFX_OPTICALFLOW_USE_HEURISTICS 1
#define FFX_OPTICALFLOW_BLOCK_SIZE 8
#define FFX_LOCAL_SEARCH_FALLBACK 1

// perf optimization for h/w not supporting accelerated msad4()
#if !defined(FFX_PREFER_WAVE64) && defined(FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION)
#undef FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION
#endif

#include "opticalflow/ffx_opticalflow_common.h"

#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif


#pragma warning(disable: 3205)  // conversion from larger type to smaller

#define DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_OPTICALFLOW_DECLARE_SRV(regIndex)  register(DECLARE_SRV_REGISTER(regIndex))
#define FFX_OPTICALFLOW_DECLARE_UAV(regIndex)  register(DECLARE_UAV_REGISTER(regIndex))
#define FFX_OPTICALFLOW_DECLARE_CB(regIndex)   register(DECLARE_CB_REGISTER(regIndex))

#if defined(FFX_OPTICALFLOW_BIND_CB_COMMON)
    cbuffer cbOF : FFX_OPTICALFLOW_DECLARE_CB(FFX_OPTICALFLOW_BIND_CB_COMMON)
    {
        FfxInt32x2 iInputLumaResolution;
        FfxUInt32 uOpticalFlowPyramidLevel;
        FfxUInt32 uOpticalFlowPyramidLevelCount;

        FfxUInt32 iFrameIndex;
        FfxUInt32 backbufferTransferFunction;
        FfxFloat32x2 minMaxLuminance;
    };
#define FFX_OPTICALFLOW_CONSTANT_BUFFER_1_SIZE 8

#endif //FFX_OPTICALFLOW_BIND_CB_COMMON

#if defined(FFX_OPTICALFLOW_BIND_CB_SPD)
cbuffer cbOF_SPD : FFX_OPTICALFLOW_DECLARE_CB(FFX_OPTICALFLOW_BIND_CB_SPD) {

    FfxUInt32     mips;
    FfxUInt32     numWorkGroups;
    FfxUInt32x2   workGroupOffset;
    FfxUInt32     numWorkGroupOpticalFlowInputPyramid;
    FfxUInt32     pad0_;
    FfxUInt32     pad1_;
    FfxUInt32     pad2_;
};

FfxUInt32 NumWorkGroups()
{
    return numWorkGroupOpticalFlowInputPyramid;
}
#endif //FFX_OPTICALFLOW_BIND_CB_SPD

#define FFX_OPTICALFLOW_CONSTANT_BUFFER_2_SIZE 8

#define FFX_OPTICALFLOW_DESCRIPTOR_COUNT 32

#define FFX_OPTICALFLOW_ROOTSIG_STRINGIFY(p) FFX_OPTICALFLOW_ROOTSIG_STR(p)
#define FFX_OPTICALFLOW_ROOTSIG_STR(p) #p
#define FFX_OPTICALFLOW_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_OPTICALFLOW_ROOTSIG_STRINGIFY(FFX_OPTICALFLOW_DESCRIPTOR_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_OPTICALFLOW_ROOTSIG_STRINGIFY(FFX_OPTICALFLOW_DESCRIPTOR_COUNT) ")), " \
                                    "CBV(b0), " \
                                    "StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK), " \
                                    "StaticSampler(s1, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)" )]

#define FFX_OPTICALFLOW_CB2_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_OPTICALFLOW_ROOTSIG_STRINGIFY(FFX_OPTICALFLOW_DESCRIPTOR_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_OPTICALFLOW_ROOTSIG_STRINGIFY(FFX_OPTICALFLOW_DESCRIPTOR_COUNT) ")), " \
                                    "CBV(b0), " \
                                    "StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK), " \
                                    "StaticSampler(s1, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                                                      "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressV = TEXTURE_ADDRESS_CLAMP, " \
                                                      "addressW = TEXTURE_ADDRESS_CLAMP, " \
                                                      "comparisonFunc = COMPARISON_NEVER, " \
                                                      "borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK)" )]
#if defined(FFX_OPTICALFLOW_EMBED_ROOTSIG)
#define FFX_OPTICALFLOW_EMBED_ROOTSIG_CONTENT FFX_OPTICALFLOW_ROOTSIG
#define FFX_OPTICALFLOW_EMBED_CB2_ROOTSIG_CONTENT FFX_OPTICALFLOW_CB2_ROOTSIG
#else
#define FFX_OPTICALFLOW_EMBED_ROOTSIG_CONTENT
#define FFX_OPTICALFLOW_EMBED_CB2_ROOTSIG_CONTENT
#endif // #if FFX_OPTICALFLOW_EMBED_ROOTSIG

FfxInt32x2 DisplaySize()
{
    return iInputLumaResolution;
}

FfxUInt32 FrameIndex()
{
    return iFrameIndex;
}

FfxUInt32 BackbufferTransferFunction()
{
    return backbufferTransferFunction;
}

FfxFloat32x2 MinMaxLuminance()
{
    return minMaxLuminance;
}

FfxBoolean CrossedSceneChangeThreshold(FfxFloat32 sceneChangeValue)
{
    return sceneChangeValue > 0.45f;
}

FfxUInt32 OpticalFlowPyramidLevel()
{
    return uOpticalFlowPyramidLevel;
}

FfxUInt32 OpticalFlowPyramidLevelCount()
{
    return uOpticalFlowPyramidLevelCount;
}

FfxInt32x2 OpticalFlowHistogramMaxVelocity()
{
    const FfxInt32 searchRadius = 8;
    FfxInt32 scale = FfxInt32(1) << (OpticalFlowPyramidLevelCount() - 1 - OpticalFlowPyramidLevel());
    FfxInt32 maxVelocity = searchRadius * scale;
    return FfxInt32x2(maxVelocity, maxVelocity);
}

    #if defined FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR
        Texture2D<FfxFloat32x4>                   r_input_color                       : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS
        Texture2D<FfxFloat32x2>                   r_input_motion_vectors              : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT
        Texture2D<FfxUInt32>                      r_optical_flow_input                : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT
        Texture2D<FfxUInt32>                      r_optical_flow_previous_input       : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW
        Texture2D<FfxInt32x2>                     r_optical_flow                      : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS
        Texture2D<FfxInt32x2>                     r_optical_flow_previous             : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO
        Texture2D<FfxUInt32x2>                    r_optical_flow_additional_info      : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS
        Texture2D<FfxUInt32x2>                    r_optical_flow_additional_info_previous : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM
        Texture2D<FfxUInt32>                      r_optical_flow_histogram            : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH
        Texture2D<FfxUInt32>                      r_optical_flow_global_motion_search : FFX_OPTICALFLOW_DECLARE_SRV(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH);
    #endif

    // UAV declarations
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT
        RWTexture2D<FfxUInt32>                   rw_optical_flow_input               : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_1       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_2       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_3       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_4       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_5       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        globallycoherent RWTexture2D<FfxUInt32>  rw_optical_flow_input_level_6       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW
        RWTexture2D<FfxInt32x2>                   rw_optical_flow                     : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL
        RWTexture2D<FfxInt32x2>                   rw_optical_flow_next_level          : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO
        RWTexture2D<FfxUInt32x2>                  rw_optical_flow_additional_info     : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL
        RWTexture2D<FfxUInt32x2>                  rw_optical_flow_additional_info_next_level : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM
        RWTexture2D<FfxUInt32>                    rw_optical_flow_histogram           : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH
        globallycoherent RWTexture2D<FfxUInt32>   rw_optical_flow_global_motion_search: FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM
        RWTexture2D<FfxUInt32>                    rw_optical_flow_scd_histogram       : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM
        RWTexture2D<FfxFloat32>                   rw_optical_flow_scd_previous_histogram : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP
        RWTexture2D<FfxUInt32>                    rw_optical_flow_scd_temp            : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP);
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT
        RWTexture2D<FfxUInt32>                    rw_optical_flow_scd_output          : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT);
    #endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR)
FfxFloat32x4 LoadInputColor(FfxUInt32x2 iPxHistory)
{
    return r_input_color[iPxHistory];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS)
FfxFloat32x2 LoadGameMotionVector(FfxInt32x2 iPxPos)
{
    FfxFloat32x2 positionScale = FfxFloat32x2(RenderSize()) / DisplaySize();
    return r_input_motion_vectors[iPxPos * positionScale] * motionVectorScale / positionScale;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT)
void StoreOpticalFlowInput(FfxInt32x2 iPxPos, FfxUInt32 fLuma)
{
    rw_optical_flow_input[iPxPos] = fLuma;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT)
FfxUInt32 LoadOpticalFlowInput(FfxInt32x2 iPxPos)
{
#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1
    return max(1, r_optical_flow_input[iPxPos]);
#else
    return r_optical_flow_input[iPxPos];
#endif
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT)
FfxUInt32 LoadRwOpticalFlowInput(FfxInt32x2 iPxPos)
{
    return rw_optical_flow_input[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT)
FfxUInt32 LoadOpticalFlowPreviousInput(FfxInt32x2 iPxPos)
{
#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1
    return max(1, r_optical_flow_previous_input[iPxPos]);
#else
    return r_optical_flow_previous_input[iPxPos];
#endif
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW)
FfxInt32x2 LoadOpticalFlow(FfxInt32x2 iPxPos)
{
    return r_optical_flow[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW)
FfxInt32x2 LoadRwOpticalFlow(FfxInt32x2 iPxPos)
{
    return rw_optical_flow[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS)
FfxInt32x2 LoadPreviousOpticalFlow(FfxInt32x2 iPxPos)
{
    return r_optical_flow_previous[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW)
void StoreOpticalFlow(FfxInt32x2 iPxPos, FfxInt32x2 motionVector)
{
    rw_optical_flow[iPxPos] = motionVector;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL)
void StoreOpticalFlowNextLevel(FfxInt32x2 iPxPos, FfxInt32x2 motionVector)
{
    rw_optical_flow_next_level[iPxPos] = motionVector;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO)
FfxUInt32x2 LoadOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return r_optical_flow_additional_info[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO)
FfxUInt32x2 LoadRwOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return rw_optical_flow_additional_info[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS)
FfxUInt32x2 LoadPreviousOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return r_optical_flow_additional_info_previous[iPxPos];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO)
void StoreOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos, FfxUInt32x2 additionalInfo)
{
    rw_optical_flow_additional_info[iPxPos] = additionalInfo;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL)
void StoreOpticalFlowNextLevelAdditionalInfo(FfxInt32x2 iPxPos, FfxUInt32x2 additionalInfo)
{
    rw_optical_flow_additional_info_next_level[iPxPos] = additionalInfo;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM)
FfxUInt32 LoadOpticalFlowHistogram(FfxInt32x2 iBucketId)
{
    return r_optical_flow_histogram[iBucketId];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM)
void AtomicIncrementOpticalFlowHistogram(FfxInt32x2 iBucketId)
{
    InterlockedAdd(rw_optical_flow_histogram[iBucketId], 1);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxInt32x2 LoadGlobalMotionVector()
{
    FfxInt32 vx = FfxInt32(r_optical_flow_global_motion_search[FfxInt32x2(0, 0)]);
    FfxInt32 vy = FfxInt32(r_optical_flow_global_motion_search[FfxInt32x2(1, 0)]);
    return FfxInt32x2(vx, vy);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxInt32x2 LoadRwGlobalMotionVector()
{
    FfxInt32 vx = FfxInt32(rw_optical_flow_global_motion_search[FfxInt32x2(0, 0)]);
    FfxInt32 vy = FfxInt32(rw_optical_flow_global_motion_search[FfxInt32x2(1, 0)]);
    return FfxInt32x2(vx, vy);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxUInt32 LoadGlobalMotionValue(FfxInt32 index)
{
    return rw_optical_flow_global_motion_search[FfxInt32x2(index, 0)];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
void StoreGlobalMotionValue(FfxInt32 index, FfxUInt32 value)
{
    rw_optical_flow_global_motion_search[FfxInt32x2(index, 0)] = value;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxUInt32 AtomicIncrementGlobalMotionValue(FfxInt32 index)
{
    FfxUInt32 initialValue;
    InterlockedAdd(rw_optical_flow_global_motion_search[FfxInt32x2(index, 0)], 1, initialValue);
    return initialValue;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
FfxUInt32 LoadRwSCDHistogram(FfxInt32 iIndex)
{
    return rw_optical_flow_scd_histogram[FfxInt32x2(iIndex, 0)];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
void StoreSCDHistogram(FfxInt32 iIndex, FfxUInt32 value)
{
    rw_optical_flow_scd_histogram[FfxInt32x2(iIndex, 0)] = value;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
void AtomicIncrementSCDHistogram(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    InterlockedAdd(rw_optical_flow_scd_histogram[FfxInt32x2(iIndex, 0)], valueToAdd);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM)
FfxFloat32 LoadRwSCDPreviousHistogram(FfxInt32 iIndex)
{
    return rw_optical_flow_scd_previous_histogram[FfxInt32x2(iIndex, 0)];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM)
void StoreSCDPreviousHistogram(FfxInt32 iIndex, FfxFloat32 value)
{
    rw_optical_flow_scd_previous_histogram[FfxInt32x2(iIndex, 0)] = value;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
FfxUInt32 LoadRwSCDTemp(FfxInt32 iIndex)
{
    return rw_optical_flow_scd_temp[FfxInt32x2(iIndex, 0)];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
void AtomicIncrementSCDTemp(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    InterlockedAdd(rw_optical_flow_scd_temp[FfxInt32x2(iIndex, 0)], valueToAdd);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
void ResetSCDTemp()
{
    rw_optical_flow_scd_temp[FfxInt32x2(0, 0)] = 0;
    rw_optical_flow_scd_temp[FfxInt32x2(1, 0)] = 0;
    rw_optical_flow_scd_temp[FfxInt32x2(2, 0)] = 0;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
FfxUInt32 LoadRwSCDOutput(FfxInt32 iIndex)
{
    return rw_optical_flow_scd_output[FfxInt32x2(iIndex, 0)];
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
void StoreSCDOutput(FfxInt32 iIndex, FfxUInt32 value)
{
    rw_optical_flow_scd_output[FfxInt32x2(iIndex, 0)] = value;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
FfxUInt32 AtomicIncrementSCDOutput(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    FfxUInt32 initialValue;
    InterlockedAdd(rw_optical_flow_scd_output[FfxInt32x2(iIndex, 0)], valueToAdd, initialValue);
    return initialValue;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_DEBUG_VISUALIZATION)
void StoreDebugVisualization(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
{
    rw_debug_visualization[iPxPos] = FfxFloat32x4(fColor, 1.f);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
FfxFloat32 GetSceneChangeValue()
{
    if (FrameIndex() <= 5)
        return 1.0;
    else
        return ffxAsFloat(LoadRwSCDOutput(SCD_OUTPUT_SCENE_CHANGE_SLOT));
}

FfxBoolean IsSceneChanged()
{
    if (FrameIndex() <= 5)
    {
        return 1.0;
    }
    else
    {
        return (LoadRwSCDOutput(SCD_OUTPUT_HISTORY_BITS_SLOT) & 0xfu) != 0;
    }
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT)
FfxUInt32 LoadFirstImagePackedLuma(FfxInt32x2 iPxPos)
{
    const FfxInt32 lumaTextureWidth = DisplaySize().x >> OpticalFlowPyramidLevel();
    const FfxInt32 lumaTextureHeight = DisplaySize().y >> OpticalFlowPyramidLevel();

    FfxInt32x2 adjustedPos = FfxInt32x2(
        ffxClamp(iPxPos.x, 0, lumaTextureWidth - 4),
        ffxClamp(iPxPos.y, 0, lumaTextureHeight - 1)
    );

    FfxUInt32 luma0 = LoadOpticalFlowInput(adjustedPos + FfxInt32x2(0, 0));
    FfxUInt32 luma1 = LoadOpticalFlowInput(adjustedPos + FfxInt32x2(1, 0));
    FfxUInt32 luma2 = LoadOpticalFlowInput(adjustedPos + FfxInt32x2(2, 0));
    FfxUInt32 luma3 = LoadOpticalFlowInput(adjustedPos + FfxInt32x2(3, 0));

    return GetPackedLuma(lumaTextureWidth, iPxPos.x, luma0, luma1, luma2, luma3);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT)
FfxUInt32 LoadSecondImagePackedLuma(FfxInt32x2 iPxPos)
{
    const FfxInt32 lumaTextureWidth = DisplaySize().x >> OpticalFlowPyramidLevel();
    const FfxInt32 lumaTextureHeight = DisplaySize().y >> OpticalFlowPyramidLevel();

    FfxInt32x2 adjustedPos = FfxInt32x2(
        ffxClamp(iPxPos.x, 0, lumaTextureWidth - 4),
        ffxClamp(iPxPos.y, 0, lumaTextureHeight - 1)
    );

    FfxUInt32 luma0 = LoadOpticalFlowPreviousInput(adjustedPos + FfxInt32x2(0, 0));
    FfxUInt32 luma1 = LoadOpticalFlowPreviousInput(adjustedPos + FfxInt32x2(1, 0));
    FfxUInt32 luma2 = LoadOpticalFlowPreviousInput(adjustedPos + FfxInt32x2(2, 0));
    FfxUInt32 luma3 = LoadOpticalFlowPreviousInput(adjustedPos + FfxInt32x2(3, 0));

    return GetPackedLuma(lumaTextureWidth, iPxPos.x, luma0, luma1, luma2, luma3);
}
#endif


void SPD_SetMipmap(int2 iPxPos, int index, float value)
{
    switch (index)
    {
    case 0:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        rw_optical_flow_input_level_1[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        break;
    case 1:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        rw_optical_flow_input_level_2[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        break;
    case 2:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        rw_optical_flow_input_level_3[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        break;
    case 3:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        rw_optical_flow_input_level_4[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        break;
    case 4:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        rw_optical_flow_input_level_5[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        break;
    case 5:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        rw_optical_flow_input_level_6[iPxPos] = value;
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        break;
    }
}

#endif // #if defined(FFX_GPU)

#endif // FFX_OPTICALFLOW_CALLBACKS_HLSL_H
