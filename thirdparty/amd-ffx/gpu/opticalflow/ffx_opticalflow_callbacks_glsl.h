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

#include "ffx_opticalflow_resources.h"

// no msad4 in glsl
#define FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION 0
#define FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS 1
#define FFX_OPTICALFLOW_USE_HEURISTICS 1
#define FFX_OPTICALFLOW_BLOCK_SIZE 8
#define FFX_LOCAL_SEARCH_FALLBACK 1

// perf optimization for h/w not supporting accelerated msad4()
//#if !defined(FFX_PREFER_WAVE64) && defined(FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION)
//#undef FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION
//#endif

#if defined(FFX_GPU)
#include "ffx_core.h"
#endif // #if defined(FFX_GPU)

#if defined(FFX_GPU)
#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // #if defined(FFX_GPU)


#include "opticalflow/ffx_opticalflow_common.h"


#if defined(FFX_OPTICALFLOW_BIND_CB_COMMON)
    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_CB_COMMON, std140) uniform cbOF_t
    {
        FfxInt32x2 iInputLumaResolution;
        FfxUInt32 uOpticalFlowPyramidLevel;
        FfxUInt32 uOpticalFlowPyramidLevelCount;

        FfxUInt32 iFrameIndex;
        FfxUInt32 backbufferTransferFunction;
        FfxFloat32x2 minMaxLuminance;
    } cbOF;

FfxInt32x2 DisplaySize()
{
    return cbOF.iInputLumaResolution;
}

FfxUInt32 FrameIndex()
{
    return cbOF.iFrameIndex;
}

FfxUInt32 BackbufferTransferFunction()
{
    return cbOF.backbufferTransferFunction;
}

FfxFloat32x2 MinMaxLuminance()
{
    return cbOF.minMaxLuminance;
}

FfxBoolean CrossedSceneChangeThreshold(FfxFloat32 sceneChangeValue)
{
    return sceneChangeValue > 0.45f;
}

FfxUInt32 OpticalFlowPyramidLevel()
{
    return cbOF.uOpticalFlowPyramidLevel;
}

FfxUInt32 OpticalFlowPyramidLevelCount()
{
    return cbOF.uOpticalFlowPyramidLevelCount;
}

FfxInt32x2 OpticalFlowHistogramMaxVelocity()
{
    const FfxInt32 searchRadius = 8;
    FfxInt32 scale = FfxInt32(1) << (OpticalFlowPyramidLevelCount() - 1 - OpticalFlowPyramidLevel());
    FfxInt32 maxVelocity = searchRadius * scale;
    return FfxInt32x2(maxVelocity, maxVelocity);
}
#endif //FFX_OPTICALFLOW_BIND_CB_COMMON

#if defined(FFX_OPTICALFLOW_BIND_CB_SPD)
    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_CB_SPD, std140) uniform cbOF_SPD_t
    {
        FfxUInt32     mips;
        FfxUInt32     numWorkGroups;
        FfxUInt32x2   workGroupOffset;
        FfxUInt32     numWorkGroupOpticalFlowInputPyramid;
        FfxUInt32     pad0_;
        FfxUInt32     pad1_;
        FfxUInt32     pad2_;
    } cbOF_SPD;

uint NumWorkGroups()
{
    return cbOF_SPD.numWorkGroupOpticalFlowInputPyramid;
}
#endif // defined(FFX_OPTICALFLOW_BIND_CB_SPD)

    #if defined FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR)                            uniform texture2D   r_input_color;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS)                   uniform texture2D   r_input_motion_vectors;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT)                     uniform utexture2D  r_optical_flow_input;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT)            uniform utexture2D  r_optical_flow_previous_input;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW)                           uniform itexture2D  r_optical_flow;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS)                  uniform itexture2D  r_optical_flow_previous;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO)           uniform utexture2D  r_optical_flow_additional_info;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS)  uniform utexture2D  r_optical_flow_additional_info_previous;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM)                 uniform utexture2D  r_optical_flow_histogram;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)      uniform utexture2D  r_optical_flow_global_motion_search;
    #endif

    // UAV declarations
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT, r8ui)                  uniform uimage2D   rw_optical_flow_input;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1, r8ui)          uniform uimage2D   rw_optical_flow_input_level_1;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2, r8ui)          uniform uimage2D   rw_optical_flow_input_level_2;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3, r8ui)          uniform uimage2D   rw_optical_flow_input_level_3;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4, r8ui)          uniform uimage2D   rw_optical_flow_input_level_4;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5, r8ui)          uniform uimage2D   rw_optical_flow_input_level_5;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6, r8ui)          uniform uimage2D   rw_optical_flow_input_level_6;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW, rg16i)                       uniform iimage2D   rw_optical_flow;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL, rg16i)            uniform iimage2D   rw_optical_flow_next_level;
    #endif
    //#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO
    //    RWTexture2D<FfxUInt32x2>                  rw_optical_flow_additional_info     : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO);
    //    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO, r32ui) uniform uimage2D   rw_optical_flow_additional_info;
    //#endif
    //#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL
    //    RWTexture2D<FfxUInt32x2>                  rw_optical_flow_additional_info_next_level : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL);
    //    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL, r32ui) uniform uimage2D   rw_optical_flow_additional_info_next_level;
    //#endif
    //#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM
    //    RWTexture2D<FfxUInt32>                    rw_optical_flow_histogram           : FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM);
    //    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM, r32ui) uniform uimage2D   rw_optical_flow_histogram;
    //#endif
    //#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH
    //    globallycoherent RWTexture2D<FfxUInt32>   rw_optical_flow_global_motion_search: FFX_OPTICALFLOW_DECLARE_UAV(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH);
    //    layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH, r32ui) coherent uniform uimage2D   rw_optical_flow_global_motion_search;
    //#endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM, r32ui)         uniform uimage2D   rw_optical_flow_scd_histogram;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM, r32f) uniform image2D   rw_optical_flow_scd_previous_histogram;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP, r32ui)              uniform uimage2D   rw_optical_flow_scd_temp;
    #endif
    #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT
        layout (set = 0, binding = FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT, r32ui)            uniform uimage2D   rw_optical_flow_scd_output;
    #endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_INPUT_COLOR)
FfxFloat32x4 LoadInputColor(FfxUInt32x2 iPxHistory)
{
    return texelFetch(r_input_color, FfxInt32x2(iPxHistory), 0);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_INPUT_MOTION_VECTORS)
FfxFloat32x2 LoadGameMotionVector(FfxInt32x2 iPxPos)
{
    FfxFloat32x2 positionScale = FfxFloat32x2(RenderSize()) / DisplaySize();
    return texelFetch(r_input_motion_vectors, FfxInt32x2(iPxPos * positionScale), 0).xy * motionVectorScale / positionScale;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT)
void StoreOpticalFlowInput(FfxInt32x2 iPxPos, FfxUInt32 fLuma)
{
    imageStore(rw_optical_flow_input, iPxPos, FfxUInt32x4(fLuma, 0, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_INPUT)
FfxUInt32 LoadOpticalFlowInput(FfxInt32x2 iPxPos)
{
#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1
    return max(1, texelFetch(r_optical_flow_input, iPxPos, 0).x);
#else
    return texelFetch(r_optical_flow_input, iPxPos, 0).x;
#endif
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT)
FfxUInt32 LoadRwOpticalFlowInput(FfxInt32x2 iPxPos)
{
    return imageLoad(rw_optical_flow_input, iPxPos).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS_INPUT)
FfxUInt32 LoadOpticalFlowPreviousInput(FfxInt32x2 iPxPos)
{
#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1
    return max(1, texelFetch(r_optical_flow_previous_input, iPxPos, 0).x);
#else
    return texelFetch(r_optical_flow_previous_input, iPxPos, 0).x;
#endif
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW)
FfxInt32x2 LoadOpticalFlow(FfxInt32x2 iPxPos)
{
    return texelFetch(r_optical_flow, iPxPos, 0).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW)
FfxInt32x2 LoadRwOpticalFlow(FfxInt32x2 iPxPos)
{
    return imageLoad(rw_optical_flow, iPxPos).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_PREVIOUS)
FfxInt32x2 LoadPreviousOpticalFlow(FfxInt32x2 iPxPos)
{
    return texelFetch(r_optical_flow_previous, iPxPos, 0).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW)
void StoreOpticalFlow(FfxInt32x2 iPxPos, FfxInt32x2 motionVector)
{
    imageStore(rw_optical_flow, iPxPos, FfxInt32x4(motionVector, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_NEXT_LEVEL)
void StoreOpticalFlowNextLevel(FfxInt32x2 iPxPos, FfxInt32x2 motionVector)
{
    imageStore(rw_optical_flow_next_level, iPxPos, FfxInt32x4(motionVector, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO)
FfxUInt32x2 LoadOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return texelFetch(r_optical_flow_additional_info, iPxPos, 0).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO)
FfxUInt32x2 LoadRwOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return imageLoad(rw_optical_flow_additional_info, iPxPos).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_ADDITIONAL_INFO_PREVIOUS)
FfxUInt32x2 LoadPreviousOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos)
{
    return texelFetch(r_optical_flow_additional_info_previous, iPxPos, 0).xy;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO)
void StoreOpticalFlowAdditionalInfo(FfxInt32x2 iPxPos, FfxUInt32x2 additionalInfo)
{
    imageStore(rw_optical_flow_additional_info, iPxPos, FfxUInt32x4(additionalInfo, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_ADDITIONAL_INFO_NEXT_LEVEL)
void StoreOpticalFlowNextLevelAdditionalInfo(FfxInt32x2 iPxPos, FfxUInt32x2 additionalInfo)
{
    imageStore(rw_optical_flow_additional_info_next_level, iPxPos, FfxUInt32x4(additionalInfo, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_HISTOGRAM)
FfxUInt32 LoadOpticalFlowHistogram(FfxInt32x2 iBucketId)
{
    return texelFetch(r_optical_flow_histogram, iBucketId, 0).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_HISTOGRAM)
void AtomicIncrementOpticalFlowHistogram(FfxInt32x2 iBucketId)
{
    imageAtomicAdd(rw_optical_flow_histogram, iBucketId, 1);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxInt32x2 LoadGlobalMotionVector()
{
    FfxInt32 vx = FfxInt32(texelFetch(r_optical_flow_global_motion_search, FfxInt32x2(0, 0), 0).x);
    FfxInt32 vy = FfxInt32(texelFetch(r_optical_flow_global_motion_search, FfxInt32x2(1, 0), 0).x);
    return FfxInt32x2(vx, vy);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxInt32x2 LoadRwGlobalMotionVector()
{
    FfxInt32 vx = FfxInt32(imageLoad(rw_optical_flow_global_motion_search, FfxInt32x2(0, 0)).x);
    FfxInt32 vy = FfxInt32(imageLoad(rw_optical_flow_global_motion_search, FfxInt32x2(1, 0)).x);
    return FfxInt32x2(vx, vy);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxUInt32 LoadGlobalMotionValue(FfxInt32 index)
{
    return imageLoad(rw_optical_flow_global_motion_search, FfxInt32x2(index, 0)).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
void StoreGlobalMotionValue(FfxInt32 index, FfxUInt32 value)
{
    imageStore(rw_optical_flow_global_motion_search, FfxInt32x2(index, 0), FfxUInt32x4(value, 0, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_GLOBAL_MOTION_SEARCH)
FfxUInt32 AtomicIncrementGlobalMotionValue(FfxInt32 index)
{
    return imageAtomicAdd(rw_optical_flow_global_motion_search, FfxInt32x2(index, 0), 1);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
FfxUInt32 LoadRwSCDHistogram(FfxInt32 iIndex)
{
    return imageLoad(rw_optical_flow_scd_histogram, FfxInt32x2(iIndex, 0)).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
void StoreSCDHistogram(FfxInt32 iIndex, FfxUInt32 value)
{
    imageStore(rw_optical_flow_scd_histogram, FfxInt32x2(iIndex, 0), FfxUInt32x4(value, 0, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_HISTOGRAM)
void AtomicIncrementSCDHistogram(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    imageAtomicAdd(rw_optical_flow_scd_histogram, FfxInt32x2(iIndex, 0), valueToAdd);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM)
FfxFloat32 LoadRwSCDPreviousHistogram(FfxInt32 iIndex)
{
    return imageLoad(rw_optical_flow_scd_previous_histogram, FfxInt32x2(iIndex, 0)).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM)
void StoreSCDPreviousHistogram(FfxInt32 iIndex, FfxFloat32 value)
{
    imageStore(rw_optical_flow_scd_previous_histogram, FfxInt32x2(iIndex, 0), FfxFloat32x4(value, 0.0, 0.0, 0.0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
FfxUInt32 LoadRwSCDTemp(FfxInt32 iIndex)
{
    return imageLoad(rw_optical_flow_scd_temp, FfxInt32x2(iIndex, 0)).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
void AtomicIncrementSCDTemp(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    imageAtomicAdd(rw_optical_flow_scd_temp, FfxInt32x2(iIndex, 0), valueToAdd);
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_TEMP)
void ResetSCDTemp()
{
    imageStore(rw_optical_flow_scd_temp, FfxInt32x2(0, 0), FfxUInt32x4(0, 0, 0, 0));
    imageStore(rw_optical_flow_scd_temp, FfxInt32x2(1, 0), FfxUInt32x4(0, 0, 0, 0));
    imageStore(rw_optical_flow_scd_temp, FfxInt32x2(2, 0), FfxUInt32x4(0, 0, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
FfxUInt32 LoadRwSCDOutput(FfxInt32 iIndex)
{
    return imageLoad(rw_optical_flow_scd_output, FfxInt32x2(iIndex, 0)).x;
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
void StoreSCDOutput(FfxInt32 iIndex, FfxUInt32 value)
{
    imageStore(rw_optical_flow_scd_output, FfxInt32x2(iIndex, 0), FfxUInt32x4(value, 0, 0, 0));
}
#endif

#if defined(FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_SCD_OUTPUT)
FfxUInt32 AtomicIncrementSCDOutput(FfxInt32 iIndex, FfxUInt32 valueToAdd)
{
    return imageAtomicAdd(rw_optical_flow_scd_output, FfxInt32x2(iIndex, 0), valueToAdd);
}
#endif

//#if defined(FFX_OPTICALFLOW_BIND_UAV_DEBUG_VISUALIZATION)
//void StoreDebugVisualization(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
//{
//    imageStore(rw_debug_visualization, iPxPos, FfxFloat32x4(fColor, 1.f));
//}
//#endif

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
        return true;
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


void SPD_SetMipmap(FfxInt32x2 iPxPos, FfxUInt32 index, FfxFloat32 value)
{
    FfxUInt32x4 value4 = FfxUInt32x4(value, 0.0, 0.0, 0.0);
    switch (index)
    {
    case 0:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        imageStore(rw_optical_flow_input_level_1, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_1
        break;
    case 1:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        imageStore(rw_optical_flow_input_level_2, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_2
        break;
    case 2:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        imageStore(rw_optical_flow_input_level_3, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_3
        break;
    case 3:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        imageStore(rw_optical_flow_input_level_4, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_4
        break;
    case 4:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        imageStore(rw_optical_flow_input_level_5, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_5
        break;
    case 5:
#if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        imageStore(rw_optical_flow_input_level_6, iPxPos, value4);
#endif // #if defined FFX_OPTICALFLOW_BIND_UAV_OPTICAL_FLOW_INPUT_LEVEL_6
        break;
    }
}

#endif // #if defined(FFX_GPU)

#endif // FFX_OPTICALFLOW_CALLBACKS_HLSL_H
