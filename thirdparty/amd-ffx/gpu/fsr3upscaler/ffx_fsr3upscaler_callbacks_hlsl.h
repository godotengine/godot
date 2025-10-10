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

#include "ffx_fsr3upscaler_resources.h"

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif //__hlsl_dx_compiler
#include "../ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif //__hlsl_dx_compiler
#endif // #if defined(FFX_GPU)

#if defined(FFX_GPU)
#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // FFX_PREFER_WAVE64

#pragma warning(disable: 3205)  // conversion from larger type to smaller

#define DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_FSR3UPSCALER_DECLARE_SRV(regIndex)  register(DECLARE_SRV_REGISTER(regIndex))
#define FFX_FSR3UPSCALER_DECLARE_UAV(regIndex)  register(DECLARE_UAV_REGISTER(regIndex))
#define FFX_FSR3UPSCALER_DECLARE_CB(regIndex)   register(DECLARE_CB_REGISTER(regIndex))

#if defined(FSR3UPSCALER_BIND_CB_FSR3UPSCALER)
cbuffer cbFSR3Upscaler : FFX_FSR3UPSCALER_DECLARE_CB(FSR3UPSCALER_BIND_CB_FSR3UPSCALER)
{
    FfxInt32x2    iRenderSize;
    FfxInt32x2    iPreviousFrameRenderSize;

    FfxInt32x2    iUpscaleSize;
    FfxInt32x2    iPreviousFrameUpscaleSize;

    FfxInt32x2    iMaxRenderSize;
    FfxInt32x2    iMaxUpscaleSize;

    FfxFloat32x4  fDeviceToViewDepth;

    FfxFloat32x2  fJitter;
    FfxFloat32x2  fPreviousFrameJitter;

    FfxFloat32x2  fMotionVectorScale;
    FfxFloat32x2  fDownscaleFactor;

    FfxFloat32x2  fMotionVectorJitterCancellation;
    FfxFloat32    fTanHalfFOV;
    FfxFloat32    fJitterSequenceLength;

    FfxFloat32    fDeltaTime;
    FfxFloat32    fDeltaPreExposure;
    FfxFloat32    fViewSpaceToMetersFactor;
    FfxFloat32    fFrameIndex;

    FfxFloat32    fVelocityFactor;
    FfxFloat32    fReactivenessScale;
    FfxFloat32    fShadingChangeScale;
    FfxFloat32    fAccumulationAddedPerFrame;
    FfxFloat32    fMinDisocclusionAccumulation;
};

#define FFX_FSR3UPSCALER_CONSTANT_BUFFER_1_SIZE (sizeof(cbFSR3Upscaler) / 4)  // Number of 32-bit values. This must be kept in sync with the cbFSR3Upscaler size.

/* Define getter functions in the order they are defined in the CB! */
FfxInt32x2 RenderSize()
{
    return iRenderSize;
}

FfxInt32x2 PreviousFrameRenderSize()
{
    return iPreviousFrameRenderSize;
}

FfxInt32x2 MaxRenderSize()
{
    return iMaxRenderSize;
}

FfxInt32x2 UpscaleSize()
{
    return iUpscaleSize;
}

FfxInt32x2 PreviousFrameUpscaleSize()
{
    return iPreviousFrameUpscaleSize;
}

FfxInt32x2 MaxUpscaleSize()
{
    return iMaxUpscaleSize;
}

FfxFloat32x2 Jitter()
{
    return fJitter;
}

FfxFloat32x2 PreviousFrameJitter()
{
    return fPreviousFrameJitter;
}

FfxFloat32x4 DeviceToViewSpaceTransformFactors()
{
    return fDeviceToViewDepth;
}

FfxFloat32x2 MotionVectorScale()
{
    return fMotionVectorScale;
}

FfxFloat32x2 DownscaleFactor()
{
    return fDownscaleFactor;
}

FfxFloat32x2 MotionVectorJitterCancellation()
{
    return fMotionVectorJitterCancellation;
}

FfxFloat32 TanHalfFoV()
{
    return fTanHalfFOV;
}

FfxFloat32 JitterSequenceLength()
{
    return fJitterSequenceLength;
}

FfxFloat32 DeltaTime()
{
    return fDeltaTime;
}

FfxFloat32 DeltaPreExposure()
{
    return fDeltaPreExposure;
}

FfxFloat32 ViewSpaceToMetersFactor()
{
    return fViewSpaceToMetersFactor;
}

FfxFloat32 FrameIndex()
{
    return fFrameIndex;
}

FfxFloat32 VelocityFactor()
{
    return fVelocityFactor;
}

FfxFloat32 AccumulationAddedPerFrame()
{
    return fAccumulationAddedPerFrame;
}

FfxFloat32 MinDisocclusionAccumulation()
{
    return fMinDisocclusionAccumulation;
}

#endif // #if defined(FSR3UPSCALER_BIND_CB_FSR3UPSCALER)

#define FFX_FSR3UPSCALER_ROOTSIG_STRINGIFY(p) FFX_FSR3UPSCALER_ROOTSIG_STR(p)
#define FFX_FSR3UPSCALER_ROOTSIG_STR(p) #p
#define FFX_FSR3UPSCALER_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR3UPSCALER_ROOTSIG_STRINGIFY(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR3UPSCALER_ROOTSIG_STRINGIFY(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT) ")), " \
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

#define FFX_FSR3UPSCALER_CONSTANT_BUFFER_2_SIZE 6  // Number of 32-bit values. This must be kept in sync with max( cbRCAS , cbSPD) size.

#define FFX_FSR3UPSCALER_CB2_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR3UPSCALER_ROOTSIG_STRINGIFY(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR3UPSCALER_ROOTSIG_STRINGIFY(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "CBV(b0), " \
                                    "CBV(b1), " \
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
#if defined(FFX_FSR3UPSCALER_EMBED_ROOTSIG)
#define FFX_FSR3UPSCALER_EMBED_ROOTSIG_CONTENT FFX_FSR3UPSCALER_ROOTSIG
#define FFX_FSR3UPSCALER_EMBED_CB2_ROOTSIG_CONTENT FFX_FSR3UPSCALER_CB2_ROOTSIG
#else
#define FFX_FSR3UPSCALER_EMBED_ROOTSIG_CONTENT
#define FFX_FSR3UPSCALER_EMBED_CB2_ROOTSIG_CONTENT
#endif // #if FFX_FSR3UPSCALER_EMBED_ROOTSIG

#if defined(FSR3UPSCALER_BIND_CB_AUTOREACTIVE)
cbuffer cbGenerateReactive : FFX_FSR3UPSCALER_DECLARE_CB(FSR3UPSCALER_BIND_CB_AUTOREACTIVE)
{
    FfxFloat32   fTcThreshold; // 0.1 is a good starting value, lower will result in more TC pixels
    FfxFloat32   fTcScale;
    FfxFloat32   fReactiveScale;
    FfxFloat32   fReactiveMax;
};

FfxFloat32 TcThreshold()
{
    return fTcThreshold;
}

FfxFloat32 TcScale()
{
    return fTcScale;
}

FfxFloat32 ReactiveScale()
{
    return fReactiveScale;
}

FfxFloat32 ReactiveMax()
{
    return fReactiveMax;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_AUTOREACTIVE)

#if defined(FSR3UPSCALER_BIND_CB_RCAS)
cbuffer cbRCAS : FFX_FSR3UPSCALER_DECLARE_CB(FSR3UPSCALER_BIND_CB_RCAS)
{
    FfxUInt32x4 rcasConfig;
};

FfxUInt32x4 RCASConfig()
{
    return rcasConfig;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_RCAS)


#if defined(FSR3UPSCALER_BIND_CB_REACTIVE)
cbuffer cbGenerateReactive : FFX_FSR3UPSCALER_DECLARE_CB(FSR3UPSCALER_BIND_CB_REACTIVE)
{
    FfxFloat32   gen_reactive_scale;
    FfxFloat32   gen_reactive_threshold;
    FfxFloat32   gen_reactive_binaryValue;
    FfxUInt32    gen_reactive_flags;
};

FfxFloat32 GenReactiveScale()
{
    return gen_reactive_scale;
}

FfxFloat32 GenReactiveThreshold()
{
    return gen_reactive_threshold;
}

FfxFloat32 GenReactiveBinaryValue()
{
    return gen_reactive_binaryValue;
}

FfxUInt32 GenReactiveFlags()
{
    return gen_reactive_flags;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_REACTIVE)

#if defined(FSR3UPSCALER_BIND_CB_SPD)
cbuffer cbSPD : FFX_FSR3UPSCALER_DECLARE_CB(FSR3UPSCALER_BIND_CB_SPD) {

    FfxUInt32   mips;
    FfxUInt32   numWorkGroups;
    FfxUInt32x2 workGroupOffset;
    FfxUInt32x2 renderSize;
};

FfxUInt32 MipCount()
{
    return mips;
}

FfxUInt32 NumWorkGroups()
{
    return numWorkGroups;
}

FfxUInt32x2 WorkGroupOffset()
{
    return workGroupOffset;
}

FfxUInt32x2 SPD_RenderSize()
{
    return renderSize;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_SPD)

SamplerState s_PointClamp : register(s0);
SamplerState s_LinearClamp : register(s1);

#if defined(FSR3UPSCALER_BIND_SRV_SPD_MIPS)
Texture2D<FfxFloat32x2> r_spd_mips : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_SPD_MIPS);

FfxInt32x2 GetSPDMipDimensions(FfxUInt32 uMipLevel)
{
    FfxUInt32 uWidth;
    FfxUInt32 uHeight;
    FfxUInt32 uLevels;
    r_spd_mips.GetDimensions(uMipLevel, uWidth, uHeight, uLevels);

    return FfxInt32x2(uWidth, uHeight);
}

FfxFloat32x2 SampleSPDMipLevel(FfxFloat32x2 fUV, FfxUInt32 mipLevel)
{
    return r_spd_mips.SampleLevel(s_LinearClamp, fUV, mipLevel);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_DEPTH)
Texture2D<FfxFloat32> r_input_depth : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INPUT_DEPTH);

FfxFloat32 LoadInputDepth(FfxUInt32x2 iPxPos)
{
    return r_input_depth[iPxPos];
}

FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV)
{
    return r_input_depth.SampleLevel(s_LinearClamp, fUV, 0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_REACTIVE_MASK)
Texture2D<FfxFloat32> r_reactive_mask : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_REACTIVE_MASK);

FfxFloat32 LoadReactiveMask(FfxUInt32x2 iPxPos)
{
    return r_reactive_mask[iPxPos] * fReactivenessScale;
}

FfxInt32x2 GetReactiveMaskResourceDimensions()
{
    FfxUInt32 uWidth;
    FfxUInt32 uHeight;
    r_reactive_mask.GetDimensions(uWidth, uHeight);

    return FfxInt32x2(uWidth, uHeight);
}

FfxFloat32 SampleReactiveMask(FfxFloat32x2 fUV)
{
    return r_reactive_mask.SampleLevel(s_LinearClamp, fUV, 0).x * fReactivenessScale;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK)
Texture2D<FfxFloat32> r_transparency_and_composition_mask : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK);

FfxFloat32 LoadTransparencyAndCompositionMask(FfxUInt32x2 iPxPos)
{
    return r_transparency_and_composition_mask[iPxPos];
}

FfxInt32x2 GetTransparencyAndCompositionMaskResourceDimensions()
{
    FfxUInt32 uWidth;
    FfxUInt32 uHeight;
    r_transparency_and_composition_mask.GetDimensions(uWidth, uHeight);

    return FfxInt32x2(uWidth, uHeight);
}

FfxFloat32 SampleTransparencyAndCompositionMask(FfxFloat32x2 fUV)
{
    return r_transparency_and_composition_mask.SampleLevel(s_LinearClamp, fUV, 0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_COLOR)
Texture2D<FfxFloat32x4> r_input_color_jittered : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INPUT_COLOR);

FfxFloat32x3 LoadInputColor(FfxUInt32x2 iPxPos)
{
    return r_input_color_jittered[iPxPos].rgb;
}

FfxFloat32x3 SampleInputColor(FfxFloat32x2 fUV)
{
    return r_input_color_jittered.SampleLevel(s_LinearClamp, fUV, 0).rgb;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_MOTION_VECTORS)
Texture2D<FfxFloat32x4> r_input_motion_vectors : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INPUT_MOTION_VECTORS);

FfxFloat32x2 LoadInputMotionVector(FfxUInt32x2 iPxDilatedMotionVectorPos)
{
    FfxFloat32x2 fSrcMotionVector = r_input_motion_vectors[iPxDilatedMotionVectorPos].xy;

    FfxFloat32x2 fUvMotionVector = fSrcMotionVector * MotionVectorScale();

#if FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS
    fUvMotionVector -= MotionVectorJitterCancellation();
#endif

    return fUvMotionVector;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INTERNAL_UPSCALED)
Texture2D<FfxFloat32x4> r_internal_upscaled_color : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INTERNAL_UPSCALED);

FfxFloat32x4 LoadHistory(FfxUInt32x2 iPxHistory)
{
    return r_internal_upscaled_color[iPxHistory];
}

FfxFloat32x4 SampleHistory(FfxFloat32x2 fUV)
{
    return r_internal_upscaled_color.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_LUMA_HISTORY)
RWTexture2D<FfxFloat32x4> rw_luma_history : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_LUMA_HISTORY);

void StoreLumaHistory(FfxUInt32x2 iPxPos, FfxFloat32x4 fLumaHistory)
{
    rw_luma_history[iPxPos] = fLumaHistory;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_LUMA_HISTORY)
Texture2D<FfxFloat32x4> r_luma_history : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_LUMA_HISTORY);

FfxFloat32x4 LoadLumaHistory(FfxInt32x2 iPxPos)
{
    return r_luma_history[iPxPos];
}

FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 fUV)
{
    return r_luma_history.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_RCAS_INPUT) 
Texture2D<FfxFloat32x4> r_rcas_input : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_RCAS_INPUT);

FfxFloat32x4 LoadRCAS_Input(FfxInt32x2 iPxPos)
{
    return r_rcas_input[iPxPos];
}

FfxFloat32x3 SampleRCAS_Input(FfxFloat32x2 fUV)
{
    return r_rcas_input.SampleLevel(s_LinearClamp, fUV, 0).rgb;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_INTERNAL_UPSCALED)
RWTexture2D<FfxFloat32x4> rw_internal_upscaled_color : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_INTERNAL_UPSCALED);

void StoreReprojectedHistory(FfxUInt32x2 iPxHistory, FfxFloat32x4 fHistory)
{
    rw_internal_upscaled_color[iPxHistory] = fHistory;
}

void StoreInternalColorAndWeight(FfxUInt32x2 iPxPos, FfxFloat32x4 fColorAndWeight)
{
    rw_internal_upscaled_color[iPxPos] = fColorAndWeight;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_UPSCALED_OUTPUT)
RWTexture2D<FfxFloat32x4> rw_upscaled_output : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_UPSCALED_OUTPUT);

void StoreUpscaledOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
{
    rw_upscaled_output[iPxPos] = FfxFloat32x4(fColor, 1.f);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_ACCUMULATION)
Texture2D<FfxFloat32> r_accumulation : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_ACCUMULATION);

FfxFloat32 SampleAccumulation(FfxFloat32x2 fUV)
{
    return r_accumulation.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_ACCUMULATION)
RWTexture2D<FfxFloat32> rw_accumulation : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_ACCUMULATION);

void StoreAccumulation(FfxUInt32x2 iPxPos, FfxFloat32 fAccumulation)
{
    rw_accumulation[iPxPos] = fAccumulation;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_SHADING_CHANGE)
Texture2D<FfxFloat32> r_shading_change : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_SHADING_CHANGE);

FfxFloat32 LoadShadingChange(FfxUInt32x2 iPxPos)
{
    return r_shading_change[iPxPos] * fShadingChangeScale;
}

FfxFloat32 SampleShadingChange(FfxFloat32x2 fUV)
{
    return r_shading_change.SampleLevel(s_LinearClamp, fUV, 0) * fShadingChangeScale;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_SHADING_CHANGE)
RWTexture2D<FfxFloat32> rw_shading_change : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SHADING_CHANGE);

void StoreShadingChange(FfxUInt32x2 iPxPos, FfxFloat32 fShadingChange)
{
    rw_shading_change[iPxPos] = fShadingChange;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH)
Texture2D<FfxFloat32> r_farthest_depth : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH);

FfxInt32x2 GetFarthestDepthResourceDimensions()
{
    FfxUInt32 uWidth;
    FfxUInt32 uHeight;
    r_farthest_depth.GetDimensions(uWidth, uHeight);

    return FfxInt32x2(uWidth, uHeight);
}

FfxFloat32 LoadFarthestDepth(FfxUInt32x2 iPxPos)
{
    return r_farthest_depth[iPxPos];
}

FfxFloat32 SampleFarthestDepth(FfxFloat32x2 fUV)
{
    return r_farthest_depth.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH)
RWTexture2D<FfxFloat32> rw_farthest_depth : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH);

void StoreFarthestDepth(FfxUInt32x2 iPxPos, FfxFloat32 fDepth)
{
    rw_farthest_depth[iPxPos] = fDepth;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH_MIP1)
Texture2D<FfxFloat32> r_farthest_depth_mip1 : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH_MIP1);

FfxInt32x2 GetFarthestDepthMip1ResourceDimensions()
{
    FfxUInt32 uWidth;
    FfxUInt32 uHeight;
    r_farthest_depth_mip1.GetDimensions(uWidth, uHeight);

    return FfxInt32x2(uWidth, uHeight);
}

FfxFloat32 LoadFarthestDepthMip1(FfxUInt32x2 iPxPos)
{
    return r_farthest_depth_mip1[iPxPos];
}

FfxFloat32 SampleFarthestDepthMip1(FfxFloat32x2 fUV)
{
    return r_farthest_depth_mip1.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1)
RWTexture2D<FfxFloat32> rw_farthest_depth_mip1 : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1);

void StoreFarthestDepthMip1(FfxUInt32x2 iPxPos, FfxFloat32 fDepth)
{
    rw_farthest_depth_mip1[iPxPos] = fDepth;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_CURRENT_LUMA)
Texture2D<FfxFloat32> r_current_luma : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_CURRENT_LUMA);

FfxFloat32 LoadCurrentLuma(FfxUInt32x2 iPxPos)
{
    return r_current_luma[iPxPos];
}

FfxFloat32 SampleCurrentLuma(FfxFloat32x2 uv)
{
    return r_current_luma.SampleLevel(s_LinearClamp, uv, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_CURRENT_LUMA)
RWTexture2D<FfxFloat32> rw_current_luma : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_CURRENT_LUMA);

void StoreCurrentLuma(FfxUInt32x2 iPxPos, FfxFloat32 fLuma)
{
    rw_current_luma[iPxPos] = fLuma;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_LUMA_INSTABILITY)
Texture2D<FfxFloat32> r_luma_instability : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_LUMA_INSTABILITY);

FfxFloat32 SampleLumaInstability(FfxFloat32x2 uv)
{
    return r_luma_instability.SampleLevel(s_LinearClamp, uv, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_LUMA_INSTABILITY)
RWTexture2D<FfxFloat32> rw_luma_instability : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_LUMA_INSTABILITY);

void StoreLumaInstability(FfxUInt32x2 iPxPos, FfxFloat32 fLumaInstability)
{
    rw_luma_instability[iPxPos] = fLumaInstability;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREVIOUS_LUMA)
Texture2D<FfxFloat32> r_previous_luma : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_PREVIOUS_LUMA);

FfxFloat32 LoadPreviousLuma(FfxUInt32x2 iPxPos)
{
    return r_previous_luma[iPxPos];
}

FfxFloat32 SamplePreviousLuma(FfxFloat32x2 uv)
{
    return r_previous_luma.SampleLevel(s_LinearClamp, uv, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_NEW_LOCKS)
Texture2D<unorm FfxFloat32> r_new_locks : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_NEW_LOCKS);

FfxFloat32 LoadNewLocks(FfxUInt32x2 iPxPos)
{
    return r_new_locks[iPxPos];
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_NEW_LOCKS)
RWTexture2D<unorm FfxFloat32> rw_new_locks : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_NEW_LOCKS);

FfxFloat32 LoadRwNewLocks(FfxUInt32x2 iPxPos)
{
    return rw_new_locks[iPxPos];
}

void StoreNewLocks(FfxUInt32x2 iPxPos, FfxFloat32 newLock)
{
    rw_new_locks[iPxPos] = newLock;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
Texture2D<FfxUInt32> r_reconstructed_previous_nearest_depth : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH);

FfxFloat32 LoadReconstructedPrevDepth(FfxUInt32x2 iPxPos)
{
    return asfloat(r_reconstructed_previous_nearest_depth[iPxPos]);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
RWTexture2D<FfxUInt32> rw_reconstructed_previous_nearest_depth : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH);

void StoreReconstructedDepth(FfxUInt32x2 iPxSample, FfxFloat32 fDepth)
{
    FfxUInt32 uDepth = asuint(fDepth);

#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
    InterlockedMax(rw_reconstructed_previous_nearest_depth[iPxSample], uDepth);
#else
    InterlockedMin(rw_reconstructed_previous_nearest_depth[iPxSample], uDepth); // min for standard, max for inverted depth
#endif
}

void SetReconstructedDepth(FfxUInt32x2 iPxSample, const FfxUInt32 uValue)
{
    rw_reconstructed_previous_nearest_depth[iPxSample] = uValue;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_DEPTH)
RWTexture2D<FfxFloat32> rw_dilated_depth : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_DILATED_DEPTH);

void StoreDilatedDepth(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 fDepth)
{
    rw_dilated_depth[iPxPos] = fDepth;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_MOTION_VECTORS)
RWTexture2D<FfxFloat32x2> rw_dilated_motion_vectors : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_DILATED_MOTION_VECTORS);

void StoreDilatedMotionVector(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fMotionVector)
{
    rw_dilated_motion_vectors[iPxPos] = fMotionVector;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS)
Texture2D<FfxFloat32x2> r_dilated_motion_vectors : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS);

FfxFloat32x2 LoadDilatedMotionVector(FfxUInt32x2 iPxInput)
{
    return r_dilated_motion_vectors[iPxInput];
}

FfxFloat32x2 SampleDilatedMotionVector(FfxFloat32x2 fUV)
{
    return r_dilated_motion_vectors.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_DEPTH)
Texture2D<FfxFloat32> r_dilated_depth : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_DILATED_DEPTH);

FfxFloat32 LoadDilatedDepth(FfxUInt32x2 iPxInput)
{
    return r_dilated_depth[iPxInput];
}

FfxFloat32 SampleDilatedDepth(FfxFloat32x2 fUV)
{
    return r_dilated_depth.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_EXPOSURE)
Texture2D<FfxFloat32x2> r_input_exposure : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INPUT_EXPOSURE);

FfxFloat32 Exposure()
{
    FfxFloat32 exposure = r_input_exposure[FfxUInt32x2(0, 0)].x;

#if defined(__XBOX_SCARLETT)
    if (exposure < 0.000030517578/** 2^-15 */) {
        exposure = 1.0f;
    }
#else
    if (exposure == 0.0f) {
        exposure = 1.0f;
    }
#endif // #if defined(__XBOX_SCARLETT)

    return exposure;
}
#endif

// BEGIN: FSR3UPSCALER_BIND_SRV_LANCZOS_LUT
#if defined(FSR3UPSCALER_BIND_SRV_LANCZOS_LUT)
Texture2D<FfxFloat32> r_lanczos_lut : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_LANCZOS_LUT);
#endif

FfxFloat32 SampleLanczos2Weight(FfxFloat32 x)
{
#if defined(FSR3UPSCALER_BIND_SRV_LANCZOS_LUT)
    return r_lanczos_lut.SampleLevel(s_LinearClamp, FfxFloat32x2(x / 2, 0.5f), 0);
#else
    return 0.f;
#endif
}
// END: FSR3UPSCALER_BIND_SRV_LANCZOS_LUT

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_REACTIVE_MASKS)
Texture2D<unorm FfxFloat32x4> r_dilated_reactive_masks : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_DILATED_REACTIVE_MASKS);

FfxFloat32x4 SampleDilatedReactiveMasks(FfxFloat32x2 fUV)
{
    return r_dilated_reactive_masks.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_REACTIVE_MASKS)
RWTexture2D<unorm FfxFloat32x4> rw_dilated_reactive_masks : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_DILATED_REACTIVE_MASKS);

void StoreDilatedReactiveMasks(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 fDilatedReactiveMasks)
{
    rw_dilated_reactive_masks[iPxPos] = fDilatedReactiveMasks;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_OPAQUE_ONLY)
Texture2D<FfxFloat32x4> r_input_opaque_only : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_INPUT_OPAQUE_ONLY);

FfxFloat32x3 LoadOpaqueOnly(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_opaque_only[iPxPos].xyz;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREV_PRE_ALPHA_COLOR)
Texture2D<float3> r_input_prev_color_pre_alpha : FFX_FSR3UPSCALER_DECLARE_SRV(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR);

FfxFloat32x3 LoadPrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_pre_alpha[iPxPos];
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREV_POST_ALPHA_COLOR)
Texture2D<float3> r_input_prev_color_post_alpha : FFX_FSR3UPSCALER_DECLARE_SRV(FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR);

FfxFloat32x3 LoadPrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_post_alpha[iPxPos];
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_AUTOREACTIVE) && \
    defined(FSR3UPSCALER_BIND_UAV_AUTOCOMPOSITION)

RWTexture2D<float> rw_output_autoreactive       : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_AUTOREACTIVE);
RWTexture2D<float> rw_output_autocomposition    : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_AUTOCOMPOSITION);

void StoreAutoReactive(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 fReactive)
{
    rw_output_autoreactive[iPxPos] = fReactive.x;

    rw_output_autocomposition[iPxPos] = fReactive.y;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_PREV_PRE_ALPHA_COLOR)
RWTexture2D<float3> rw_output_prev_color_pre_alpha : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_PREV_PRE_ALPHA_COLOR);

void StorePrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_pre_alpha[iPxPos] = color;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_PREV_POST_ALPHA_COLOR)
RWTexture2D<float3> rw_output_prev_color_post_alpha : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_PREV_POST_ALPHA_COLOR);

void StorePrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_post_alpha[iPxPos] = color;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FRAME_INFO)
RWTexture2D<FfxFloat32x4> rw_frame_info : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_FRAME_INFO);

FfxFloat32x4 LoadFrameInfo()
{
    return rw_frame_info[FfxInt32x2(0, 0)];
}

void StoreFrameInfo(FfxFloat32x4 fInfo)
{
    rw_frame_info[FfxInt32x2(0, 0)] = fInfo;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FRAME_INFO)
Texture2D<FfxFloat32x4> r_frame_info : FFX_FSR3UPSCALER_DECLARE_SRV(FSR3UPSCALER_BIND_SRV_FRAME_INFO);

FfxFloat32x4 FrameInfo()
{
    return r_frame_info[FfxInt32x2(0, 0)];
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_0)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_1)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_2)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_3)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_4)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5)

RWTexture2D<FfxFloat32x2>                   rw_spd_mip0   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_0);
RWTexture2D<FfxFloat32x2>                   rw_spd_mip1   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_1);
RWTexture2D<FfxFloat32x2>                   rw_spd_mip2   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_2);
RWTexture2D<FfxFloat32x2>                   rw_spd_mip3   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_3);
RWTexture2D<FfxFloat32x2>                   rw_spd_mip4   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_4);
globallycoherent RWTexture2D<FfxFloat32x2>  rw_spd_mip5   : FFX_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5);

FfxFloat32x2 RWLoadPyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 index)
{
#define LOAD(idx)                                 \
            if (index == idx)                             \
            {                                             \
                return rw_spd_mip##idx[iPxPos]; \
            }
    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);

    return 0;

#undef LOAD
}

void StorePyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 outValue, FFX_PARAMETER_IN FfxUInt32 index)
{
#define STORE(idx)                   \
            if (index == idx)                \
            {                                \
                rw_spd_mip##idx[iPxPos] = outValue; \
            }

    STORE(0);
    STORE(1);
    STORE(2);
    STORE(3);
    STORE(4);
    STORE(5);

#undef STORE
}
#endif

#if defined FSR3UPSCALER_BIND_UAV_SPD_GLOBAL_ATOMIC
globallycoherent RWTexture2D<FfxUInt32>   rw_spd_global_atomic : FFX_FSR3UPSCALER_DECLARE_UAV(FSR3UPSCALER_BIND_UAV_SPD_GLOBAL_ATOMIC);

void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter)
{
    InterlockedAdd(rw_spd_global_atomic[FfxInt32x2(0, 0)], 1, spdCounter);
}

void SPD_ResetAtomicCounter()
{
    rw_spd_global_atomic[FfxInt32x2(0, 0)] = 0;
}
#endif

#endif // #if defined(FFX_GPU)
