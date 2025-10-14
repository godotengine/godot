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
#include "../ffx_core.h"
#endif // #if defined(FFX_GPU)

#if defined(FFX_GPU)
#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // FFX_PREFER_WAVE64

#if defined(FSR3UPSCALER_BIND_CB_FSR3UPSCALER)
    layout (set = 0, binding = FSR3UPSCALER_BIND_CB_FSR3UPSCALER, std140) uniform cbFSR3UPSCALER_t
    {
        FfxInt32x2 iRenderSize;
        FfxInt32x2 iPreviousFrameRenderSize;

        FfxInt32x2 iUpscaleSize;
        FfxInt32x2 iPreviousFrameUpscaleSize;

        FfxInt32x2 iMaxRenderSize;
        FfxInt32x2 iMaxUpscaleSize;

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

    	// GODOT BEGINS
    	// Do not change this to an array, as arrays have different alignment requirements
    	FfxFloat32    fPad1;
    	FfxFloat32    fPad2;
    	FfxFloat32    fPad3;
    	mat4          mReprojectionMatrix;
    	// GODOT ENDS
	} cbFSR3Upscaler;


FfxInt32x2 RenderSize()
{
    return cbFSR3Upscaler.iRenderSize;
}

FfxInt32x2 PreviousFrameRenderSize()
{
    return cbFSR3Upscaler.iPreviousFrameRenderSize;
}

FfxInt32x2 MaxRenderSize()
{
    return cbFSR3Upscaler.iMaxRenderSize;
}

FfxInt32x2 UpscaleSize()
{
    return cbFSR3Upscaler.iUpscaleSize;
}

FfxInt32x2 PreviousFrameUpscaleSize()
{
    return cbFSR3Upscaler.iPreviousFrameUpscaleSize;
}

FfxInt32x2 MaxUpscaleSize()
{
    return cbFSR3Upscaler.iMaxUpscaleSize;
}

FfxFloat32x2 Jitter()
{
    return cbFSR3Upscaler.fJitter;
}

FfxFloat32x2 PreviousFrameJitter()
{
    return cbFSR3Upscaler.fPreviousFrameJitter;
}

FfxFloat32x4 DeviceToViewSpaceTransformFactors()
{
    return cbFSR3Upscaler.fDeviceToViewDepth;
}

FfxFloat32x2 MotionVectorScale()
{
    return cbFSR3Upscaler.fMotionVectorScale;
}

FfxFloat32x2 DownscaleFactor()
{
    return cbFSR3Upscaler.fDownscaleFactor;
}

FfxFloat32x2 MotionVectorJitterCancellation()
{
    return cbFSR3Upscaler.fMotionVectorJitterCancellation;
}

FfxFloat32 TanHalfFoV()
{
    return cbFSR3Upscaler.fTanHalfFOV;
}

FfxFloat32 JitterSequenceLength()
{
    return cbFSR3Upscaler.fJitterSequenceLength;
}

FfxFloat32 DeltaTime()
{
    return cbFSR3Upscaler.fDeltaTime;
}

FfxFloat32 DeltaPreExposure()
{
    return cbFSR3Upscaler.fDeltaPreExposure;
}

FfxFloat32 ViewSpaceToMetersFactor()
{
    return cbFSR3Upscaler.fViewSpaceToMetersFactor;
}

FfxFloat32 FrameIndex()
{
    return cbFSR3Upscaler.fFrameIndex;
}

FfxFloat32 VelocityFactor()
{
    return cbFSR3Upscaler.fVelocityFactor;
}

FfxFloat32 AccumulationAddedPerFrame()
{
    return cbFSR3Upscaler.fAccumulationAddedPerFrame;
}

FfxFloat32 MinDisocclusionAccumulation()
{
    return cbFSR3Upscaler.fMinDisocclusionAccumulation;
}

#endif // #if defined(FSR3UPSCALER_BIND_CB_FSR3UPSCALER)


#if defined(FSR3UPSCALER_BIND_CB_AUTOREACTIVE)
layout(set = 0, binding = FSR3UPSCALER_BIND_CB_AUTOREACTIVE, std140) uniform cbGenerateReactive_t
{
    FfxFloat32   fTcThreshold; // 0.1 is a good starting value, lower will result in more TC pixels
    FfxFloat32   fTcScale;
    FfxFloat32   fReactiveScale;
    FfxFloat32   fReactiveMax;
} cbGenerateReactive;

FfxFloat32 TcThreshold()
{
    return cbGenerateReactive.fTcThreshold;
}

FfxFloat32 TcScale()
{
    return cbGenerateReactive.fTcScale;
}

FfxFloat32 ReactiveScale()
{
    return cbGenerateReactive.fReactiveScale;
}

FfxFloat32 ReactiveMax()
{
    return cbGenerateReactive.fReactiveMax;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_AUTOREACTIVE)

#if defined(FSR3UPSCALER_BIND_CB_RCAS)
layout(set = 0, binding = FSR3UPSCALER_BIND_CB_RCAS, std140) uniform cbRCAS_t
{
    FfxUInt32x4 rcasConfig;
} cbRCAS;

FfxUInt32x4 RCASConfig()
{
    return cbRCAS.rcasConfig;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_RCAS)


#if defined(FSR3UPSCALER_BIND_CB_REACTIVE)
layout(set = 0, binding = FSR3UPSCALER_BIND_CB_REACTIVE, std140) uniform cbGenerateReactive_t
{
    FfxFloat32   gen_reactive_scale;
    FfxFloat32   gen_reactive_threshold;
    FfxFloat32   gen_reactive_binaryValue;
    FfxUInt32    gen_reactive_flags;
} cbGenerateReactive;

FfxFloat32 GenReactiveScale()
{
    return cbGenerateReactive.gen_reactive_scale;
}

FfxFloat32 GenReactiveThreshold()
{
    return cbGenerateReactive.gen_reactive_threshold;
}

FfxFloat32 GenReactiveBinaryValue()
{
    return cbGenerateReactive.gen_reactive_binaryValue;
}

FfxUInt32 GenReactiveFlags()
{
    return cbGenerateReactive.gen_reactive_flags;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_REACTIVE)


#if defined(FSR3UPSCALER_BIND_CB_SPD)
layout(set = 0, binding = FSR3UPSCALER_BIND_CB_SPD, std140) uniform cbSPD_t
{
    FfxUInt32   mips;
    FfxUInt32   numWorkGroups;
    FfxUInt32x2 workGroupOffset;
    FfxUInt32x2 renderSize;
} cbSPD;

FfxUInt32 MipCount()
{
    return cbSPD.mips;
}

FfxUInt32 NumWorkGroups()
{
    return cbSPD.numWorkGroups;
}

FfxUInt32x2 WorkGroupOffset()
{
    return cbSPD.workGroupOffset;
}

FfxUInt32x2 SPD_RenderSize()
{
    return cbSPD.renderSize;
}
#endif // #if defined(FSR3UPSCALER_BIND_CB_SPD)

// GODOT BEGINS
// Godot DX12 backend doesn't support binding numbers larger than 1000, so we have to remap them.
layout (set = 0, binding = 100 /*1000*/) uniform sampler s_PointClamp;
layout (set = 0, binding = 101 /*1001*/) uniform sampler s_LinearClamp;
// GODOT ENDS

#if defined(FSR3UPSCALER_BIND_SRV_SPD_MIPS)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_SPD_MIPS) uniform texture2D  r_spd_mips;

FfxInt32x2 GetSPDMipDimensions(FfxUInt32 uMipLevel)
{
	return textureSize(r_spd_mips, int(uMipLevel)).xy;
}

FfxFloat32x2 SampleSPDMipLevel(FfxFloat32x2 fUV, FfxUInt32 mipLevel)
{
	return textureLod(sampler2D(r_spd_mips, s_LinearClamp), fUV, float(mipLevel)).rg;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INPUT_DEPTH) uniform texture2D  r_input_depth;

FfxFloat32 LoadInputDepth(FfxInt32x2 iPxPos)
{
	return texelFetch(r_input_depth, iPxPos, 0).r;
}

FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_input_depth, s_LinearClamp), fUV, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_REACTIVE_MASK)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_REACTIVE_MASK) uniform texture2D  r_reactive_mask;

FfxFloat32 LoadReactiveMask(FfxInt32x2 iPxPos)
{
    // GODOT BEGINS
#if FFX_FSR3UPSCALER_OPTION_GODOT_REACTIVE_MASK_CLAMP
	return min(texelFetch(r_reactive_mask, FfxInt32x2(iPxPos), 0).r * cbFSR3Upscaler.fReactivenessScale, 0.9f);
#else
 	return texelFetch(r_reactive_mask, FfxInt32x2(iPxPos), 0).r * cbFSR3Upscaler.fReactivenessScale;
#endif
	// GODOT ENDS
}

FfxInt32x2 GetReactiveMaskResourceDimensions()
{
    return textureSize(r_reactive_mask, 0).xy;
}

FfxFloat32 SampleReactiveMask(FfxFloat32x2 fUV)
{
	return textureLod(sampler2D(r_reactive_mask, s_LinearClamp), fUV, 0.0).x * cbFSR3Upscaler.fReactivenessScale;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK) uniform texture2D  r_transparency_and_composition_mask;

FfxFloat32 LoadTransparencyAndCompositionMask(FfxUInt32x2 iPxPos)
{
	return texelFetch(r_transparency_and_composition_mask, FfxInt32x2(iPxPos), 0).r;
}

FfxInt32x2 GetTransparencyAndCompositionMaskResourceDimensions()
{
    return textureSize(r_transparency_and_composition_mask, 0).xy;
}

FfxFloat32 SampleTransparencyAndCompositionMask(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_transparency_and_composition_mask, s_LinearClamp), fUV, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_COLOR)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INPUT_COLOR) uniform texture2D  r_input_color_jittered;

FfxFloat32x3 LoadInputColor(FfxInt32x2 iPxPos)
{
	return texelFetch(r_input_color_jittered, iPxPos, 0).rgb;
}

FfxFloat32x3 SampleInputColor(FfxFloat32x2 fUV)
{
	return textureLod(sampler2D(r_input_color_jittered, s_LinearClamp), fUV, 0.0).rgb;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_MOTION_VECTORS)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INPUT_MOTION_VECTORS) uniform texture2D  r_input_motion_vectors;

FfxFloat32x2 LoadInputMotionVector(FfxInt32x2 iPxDilatedMotionVectorPos)
{
	FfxFloat32x2 fSrcMotionVector = texelFetch(r_input_motion_vectors, iPxDilatedMotionVectorPos, 0).xy;

    // GODOT BEGINS
#if FFX_FSR3UPSCALER_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS
	bool bInvalidMotionVector = all(lessThanEqual(fSrcMotionVector, vec2(-1.0f, -1.0f)));
	if (bInvalidMotionVector)
	{
		FfxFloat32 fSrcDepth = LoadInputDepth(iPxDilatedMotionVectorPos);
		FfxFloat32x2 fUv = (iPxDilatedMotionVectorPos + FfxFloat32(0.5)) / RenderSize();
		fSrcMotionVector = FFX_FSR_OPTION_GODOT_DERIVE_INVALID_MOTION_VECTORS_FUNCTION(fUv, fSrcDepth, cbFSR3Upscaler.mReprojectionMatrix);
	}
#endif
	// GODOT ENDS

	FfxFloat32x2 fUvMotionVector = fSrcMotionVector * MotionVectorScale();

#if FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS
	fUvMotionVector -= MotionVectorJitterCancellation();
#endif

	return fUvMotionVector;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INTERNAL_UPSCALED)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INTERNAL_UPSCALED) uniform texture2D  r_internal_upscaled_color;

FfxFloat32x4 LoadHistory(FfxInt32x2 iPxHistory)
{
	return texelFetch(r_internal_upscaled_color, iPxHistory, 0);
}

FfxFloat32x4 SampleHistory(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_internal_upscaled_color, s_LinearClamp), fUV, 0.0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_LUMA_HISTORY)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_LUMA_HISTORY, rgba8) uniform image2D  rw_luma_history;

void StoreLumaHistory(FfxInt32x2 iPxPos, FfxFloat32x4 fLumaHistory)
{
	imageStore(rw_luma_history, iPxPos, fLumaHistory);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_LUMA_HISTORY)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_LUMA_HISTORY) uniform texture2D  r_luma_history;

FfxFloat32x4 LoadLumaHistory(FfxInt32x2 iPxPos)
{
    return texelFetch(r_luma_history, iPxPos, 0);
}

FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 fUV)
{
	return textureLod(sampler2D(r_luma_history, s_LinearClamp), fUV, 0.0);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_RCAS_INPUT)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_RCAS_INPUT) uniform texture2D  r_rcas_input;

FfxFloat32x4 LoadRCAS_Input(FfxInt32x2 iPxPos)
{
    return texelFetch(r_rcas_input, iPxPos, 0);
}

FfxFloat32x3 SampleRCAS_Input(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_rcas_input, s_LinearClamp), fUV, 0.0).rgb;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_INTERNAL_UPSCALED)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_INTERNAL_UPSCALED, rgba16f) writeonly uniform image2D  rw_internal_upscaled_color;

void StoreReprojectedHistory(FfxInt32x2 iPxHistory, FfxFloat32x4 fHistory)
{
	imageStore(rw_internal_upscaled_color, iPxHistory, fHistory);
}

void StoreInternalColorAndWeight(FfxInt32x2 iPxPos, FfxFloat32x4 fColorAndWeight)
{
	imageStore(rw_internal_upscaled_color, iPxPos, fColorAndWeight);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_UPSCALED_OUTPUT)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_UPSCALED_OUTPUT /* app controlled format */) writeonly uniform image2D  rw_upscaled_output;

void StoreUpscaledOutput(FfxInt32x2 iPxPos, FfxFloat32x3 fColor)
{
    imageStore(rw_upscaled_output, iPxPos, FfxFloat32x4(fColor, 1.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_ACCUMULATION)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_ACCUMULATION) uniform texture2D  r_accumulation;

FfxFloat32 SampleAccumulation(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_accumulation, s_LinearClamp), fUV, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_ACCUMULATION)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_ACCUMULATION, r8) uniform image2D  rw_accumulation;

void StoreAccumulation(FfxInt32x2 iPxPos, FfxFloat32 fAccumulation)
{
    imageStore(rw_accumulation, iPxPos, vec4(fAccumulation, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_SHADING_CHANGE)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_SHADING_CHANGE) uniform texture2D  r_shading_change;

FfxFloat32 LoadShadingChange(FfxInt32x2 iPxPos)
{
    return texelFetch(r_shading_change, iPxPos, 0).x * cbFSR3Upscaler.fShadingChangeScale;
}

FfxFloat32 SampleShadingChange(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_shading_change, s_LinearClamp), fUV, 0.0).x * cbFSR3Upscaler.fShadingChangeScale;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_SHADING_CHANGE)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SHADING_CHANGE, r8) uniform image2D  rw_shading_change;

void StoreShadingChange(FfxInt32x2 iPxPos, FfxFloat32 fShadingChange)
{
    imageStore(rw_shading_change, iPxPos, vec4(fShadingChange, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH) uniform texture2D  r_farthest_depth;

FfxInt32x2 GetFarthestDepthResourceDimensions()
{
	return textureSize(r_farthest_depth, 0).xy;
}

FfxFloat32 LoadFarthestDepth(FfxInt32x2 iPxPos)
{
    return texelFetch(r_farthest_depth, iPxPos, 0).x;
}

FfxFloat32 SampleFarthestDepth(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_farthest_depth, s_LinearClamp), fUV, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH, r16f) uniform image2D  rw_farthest_depth;

void StoreFarthestDepth(FfxInt32x2 iPxPos, FfxFloat32 fDepth)
{
    imageStore(rw_farthest_depth, iPxPos, vec4(fDepth, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH_MIP1)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_FARTHEST_DEPTH_MIP1) uniform texture2D  r_farthest_depth_mip1;

FfxInt32x2 GetFarthestDepthMip1ResourceDimensions()
{
	return textureSize(r_farthest_depth_mip1, 0).xy;
}

FfxFloat32 LoadFarthestDepthMip1(FfxInt32x2 iPxPos)
{
    return texelFetch(r_farthest_depth_mip1, iPxPos, 0).x;
}

FfxFloat32 SampleFarthestDepthMip1(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_farthest_depth_mip1, s_LinearClamp), fUV, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_FARTHEST_DEPTH_MIP1, r16f) uniform image2D  rw_farthest_depth_mip1;

void StoreFarthestDepthMip1(FfxInt32x2 iPxPos, FfxFloat32 fDepth)
{
    imageStore(rw_farthest_depth_mip1, iPxPos, vec4(fDepth, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_CURRENT_LUMA)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_CURRENT_LUMA) uniform texture2D  r_current_luma;

FfxFloat32 LoadCurrentLuma(FfxInt32x2 iPxPos)
{
    return texelFetch(r_current_luma, iPxPos, 0).r;
}

FfxFloat32 SampleCurrentLuma(FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_current_luma, s_LinearClamp), uv, 0.0).r;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_CURRENT_LUMA)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_CURRENT_LUMA, r16f) uniform image2D  rw_current_luma;

void StoreCurrentLuma(FfxInt32x2 iPxPos, FfxFloat32 fLuma)
{
    imageStore(rw_current_luma, iPxPos, vec4(fLuma, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_LUMA_INSTABILITY)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_LUMA_INSTABILITY) uniform texture2D  r_luma_instability;

FfxFloat32 SampleLumaInstability(FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_luma_instability, s_LinearClamp), uv, 0.0).x;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_LUMA_INSTABILITY)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_LUMA_INSTABILITY, r16f) uniform image2D  rw_luma_instability;

void StoreLumaInstability(FfxInt32x2 iPxPos, FfxFloat32 fLumaInstability)
{
    imageStore(rw_luma_instability, iPxPos, vec4(fLumaInstability, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREVIOUS_LUMA)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_PREVIOUS_LUMA) uniform texture2D  r_previous_luma;

FfxFloat32 LoadPreviousLuma(FfxInt32x2 iPxPos)
{
    return texelFetch(r_previous_luma, iPxPos, 0).r;
}

FfxFloat32 SamplePreviousLuma(FfxFloat32x2 uv)
{
    return textureLod(sampler2D(r_previous_luma, s_LinearClamp), uv, 0.0).r;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_NEW_LOCKS)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_NEW_LOCKS) uniform texture2D  r_new_locks;

FfxFloat32 LoadNewLocks(FfxInt32x2 iPxPos)
{
	return texelFetch(r_new_locks, iPxPos, 0).r;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_NEW_LOCKS)
layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_NEW_LOCKS, r8) uniform image2D  rw_new_locks;

FfxFloat32 LoadRwNewLocks(FfxInt32x2 iPxPos)
{
	return imageLoad(rw_new_locks, iPxPos).r;
}

void StoreNewLocks(FfxInt32x2 iPxPos, FfxFloat32 newLock)
{
	imageStore(rw_new_locks, iPxPos, vec4(newLock, 0, 0, 0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH) uniform utexture2D r_reconstructed_previous_nearest_depth;

FfxFloat32 LoadReconstructedPrevDepth(FfxInt32x2 iPxPos)
{
	return uintBitsToFloat(texelFetch(r_reconstructed_previous_nearest_depth, iPxPos, 0).r);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH, r32ui) uniform uimage2D  rw_reconstructed_previous_nearest_depth;

void StoreReconstructedDepth(FfxInt32x2 iPxSample, FfxFloat32 fDepth)
{
	FfxUInt32 uDepth = floatBitsToUint(fDepth);

	#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
		imageAtomicMax(rw_reconstructed_previous_nearest_depth, iPxSample, uDepth);
	#else
		imageAtomicMin(rw_reconstructed_previous_nearest_depth, iPxSample, uDepth); // min for standard, max for inverted depth
	#endif
}

void SetReconstructedDepth(FfxInt32x2 iPxSample, FfxUInt32 uValue)
{
	imageStore(rw_reconstructed_previous_nearest_depth, iPxSample, uvec4(uValue, 0, 0, 0));
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_DILATED_DEPTH, r32f) writeonly uniform image2D  rw_dilated_depth;

void StoreDilatedDepth(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 fDepth)
{
	imageStore(rw_dilated_depth, iPxPos, vec4(fDepth, 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_MOTION_VECTORS)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_DILATED_MOTION_VECTORS, rg16f) writeonly uniform image2D  rw_dilated_motion_vectors;

void StoreDilatedMotionVector(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fMotionVector)
{
	imageStore(rw_dilated_motion_vectors, iPxPos, vec4(fMotionVector, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_DILATED_MOTION_VECTORS) uniform texture2D  r_dilated_motion_vectors;

FfxFloat32x2 LoadDilatedMotionVector(FfxInt32x2 iPxInput)
{
	return texelFetch(r_dilated_motion_vectors, iPxInput, 0).xy;
}

FfxFloat32x2 SampleDilatedMotionVector(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_dilated_motion_vectors, s_LinearClamp), fUV, 0.0).xy;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_DEPTH)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_DILATED_DEPTH) uniform texture2D  r_dilated_depth;

FfxFloat32 LoadDilatedDepth(FfxInt32x2 iPxInput)
{
	return texelFetch(r_dilated_depth, iPxInput, 0).r;
}

FfxFloat32 SampleDilatedDepth(FfxFloat32x2 fUV)
{
    return textureLod(sampler2D(r_dilated_depth, s_LinearClamp), fUV, 0.0).r;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_EXPOSURE)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INPUT_EXPOSURE) uniform texture2D  r_input_exposure;

FfxFloat32 Exposure()
{
	FfxFloat32 exposure = texelFetch(r_input_exposure, FfxInt32x2(0, 0), 0).x;

	if (exposure == 0.0) {
		exposure = 1.0;
	}

	return exposure;
}
#endif

// BEGIN: FSR3UPSCALER_BIND_SRV_LANCZOS_LUT
#if defined(FSR3UPSCALER_BIND_SRV_LANCZOS_LUT)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_LANCZOS_LUT) uniform texture2D  r_lanczos_lut;
#endif

FfxFloat32 SampleLanczos2Weight(FfxFloat32 x)
{
#if defined(FSR3UPSCALER_BIND_SRV_LANCZOS_LUT)
	return textureLod(sampler2D(r_lanczos_lut, s_LinearClamp), FfxFloat32x2(x / 2.0, 0.5), 0.0).x;
#else
    return 0.f;
#endif
}
// END: FSR3UPSCALER_BIND_SRV_LANCZOS_LUT

#if defined(FSR3UPSCALER_BIND_SRV_DILATED_REACTIVE_MASKS)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_DILATED_REACTIVE_MASKS) uniform texture2D  r_dilated_reactive_masks;

FfxFloat32x4 SampleDilatedReactiveMasks(FfxFloat32x2 fUV)
{
	return textureLod(sampler2D(r_dilated_reactive_masks, s_LinearClamp), fUV, 0.0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_DILATED_REACTIVE_MASKS)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_DILATED_REACTIVE_MASKS, rgba8) writeonly uniform image2D  rw_dilated_reactive_masks;

void StoreDilatedReactiveMasks(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 fDilatedReactiveMasks)
{
    imageStore(rw_dilated_reactive_masks, iPxPos, fDilatedReactiveMasks);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_INPUT_OPAQUE_ONLY)
layout (set = 0, binding = FSR3UPSCALER_BIND_SRV_INPUT_OPAQUE_ONLY) uniform texture2D  r_input_opaque_only;

FfxFloat32x3 LoadOpaqueOnly(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
	return texelFetch(r_input_opaque_only, iPxPos, 0).xyz;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREV_PRE_ALPHA_COLOR)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_PREV_PRE_ALPHA_COLOR) uniform texture2D  r_input_prev_color_pre_alpha;

FfxFloat32x3 LoadPrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
	return texelFetch(r_input_prev_color_pre_alpha, iPxPos, 0).xyz;
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_PREV_POST_ALPHA_COLOR)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_PREV_POST_ALPHA_COLOR) uniform texture2D  r_input_prev_color_post_alpha;

FfxFloat32x3 LoadPrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
	return texelFetch(r_input_prev_color_post_alpha, iPxPos, 0).xyz;
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_AUTOREACTIVE) && \
    defined(FSR3UPSCALER_BIND_UAV_AUTOCOMPOSITION)

layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_AUTOREACTIVE, r32f)    uniform image2D  rw_output_autoreactive;
layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_AUTOCOMPOSITION, r32f) uniform image2D  rw_output_autocomposition;

void StoreAutoReactive(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 fReactive)
{
	imageStore(rw_output_autoreactive, iPxPos, FfxFloat32x4(FfxFloat32(fReactive.x), 0.0, 0.0, 0.0));

	imageStore(rw_output_autocomposition, iPxPos, FfxFloat32x4(FfxFloat32(fReactive.y), 0.0, 0.0, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_PREV_PRE_ALPHA_COLOR)
layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_PREV_PRE_ALPHA_COLOR, r11f_g11f_b10f) uniform image2D  rw_output_prev_color_pre_alpha;

void StorePrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
	imageStore(rw_output_prev_color_pre_alpha, iPxPos, FfxFloat32x4(color, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_PREV_POST_ALPHA_COLOR)
layout(set = 0, binding = FSR3UPSCALER_BIND_UAV_PREV_POST_ALPHA_COLOR, r11f_g11f_b10f) uniform image2D  rw_output_prev_color_post_alpha;

void StorePrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
	imageStore(rw_output_prev_color_post_alpha, iPxPos, FfxFloat32x4(color, 0.0));
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_FRAME_INFO)
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_FRAME_INFO, rgba32f) uniform image2D  rw_frame_info;

FfxFloat32x4 LoadFrameInfo()
{
    return imageLoad(rw_frame_info, ivec2(0, 0));
}

void StoreFrameInfo(FfxFloat32x4 fInfo)
{
    imageStore(rw_frame_info, ivec2(0, 0), fInfo);
}
#endif

#if defined(FSR3UPSCALER_BIND_SRV_FRAME_INFO)
layout(set = 0, binding = FSR3UPSCALER_BIND_SRV_FRAME_INFO) uniform texture2D  r_frame_info;

FfxFloat32x4 FrameInfo()
{
    return texelFetch(r_frame_info, ivec2(0, 0), 0);
}
#endif

#if defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_0)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_1)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_2)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_3)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_4)    && \
    defined(FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5)

layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_0, rg16f)          uniform image2D  rw_spd_mip0;
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_1, rg16f)          uniform image2D  rw_spd_mip1;
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_2, rg16f)          uniform image2D  rw_spd_mip2;
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_3, rg16f)          uniform image2D  rw_spd_mip3;
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_4, rg16f)          uniform image2D  rw_spd_mip4;
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_MIPS_LEVEL_5, rg16f) coherent uniform image2D  rw_spd_mip5;

FfxFloat32x2 RWLoadPyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 index)
{
#define LOAD(idx)                                 \
            if (index == idx)                             \
            {                                             \
                return imageLoad(rw_spd_mip##idx, iPxPos).xy; \
            }
    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);

    return FfxFloat32x2(0.0, 0.0);

#undef LOAD
}

void StorePyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 outValue, FFX_PARAMETER_IN FfxUInt32 index)
{
#define STORE(idx)                   \
            if (index == idx)                \
            {                                \
                imageStore(rw_spd_mip##idx, iPxPos, vec4(outValue, 0.0, 0.0)); \
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
layout (set = 0, binding = FSR3UPSCALER_BIND_UAV_SPD_GLOBAL_ATOMIC, r32ui) coherent uniform uimage2D  rw_spd_global_atomic;

void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter)
{
    spdCounter = imageAtomicAdd(rw_spd_global_atomic, ivec2(0, 0), 1);
}

void SPD_ResetAtomicCounter()
{
    imageStore(rw_spd_global_atomic, ivec2(0, 0), uvec4(0));
}
#endif

#endif // #if defined(FFX_GPU)
