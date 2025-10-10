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

#include "ffx_frameinterpolation_resources.h"
#include "ffx_core.h"

#define COUNTER_SPD                          0
#define COUNTER_FRAME_INDEX_SINCE_LAST_RESET 1

  ///////////////////////////////////////////////
 // declare CBs and CB accessors
///////////////////////////////////////////////
#if defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION, std140) uniform cbFI_t
    {
        FfxInt32x2      renderSize;
        FfxInt32x2      displaySize;

        FfxFloat32x2    displaySizeRcp;
        FfxFloat32      cameraNear;
        FfxFloat32      cameraFar;

        FfxInt32x2      upscalerTargetSize;
        FfxInt32        Mode;
        FfxInt32        reset;

        FfxFloat32x4    fDeviceToViewDepth;

        FfxFloat32      deltaTime;
        FfxInt32        HUDLessAttachedFactor;
        FfxInt32x2      distortionFieldSize;

        FfxFloat32x2    opticalFlowScale;
        FfxInt32        opticalFlowBlockSize;
        FfxUInt32       dispatchFlags;

        FfxInt32x2      maxRenderSize;
        FfxInt32        opticalFlowHalfResMode;
        FfxInt32        NumInstances;

        FfxInt32x2      interpolationRectBase;
        FfxInt32x2      interpolationRectSize;

        FfxFloat32x3    debugBarColor;
        FfxUInt32       backBufferTransferFunction;

        FfxFloat32x2    minMaxLuminance;
        FfxFloat32      fTanHalfFOV;
        FfxInt32        _pad1;

        FfxFloat32x2    fJitter;
        FfxFloat32x2    fMotionVectorScale;
    } cbFI;

    FfxFloat32x2 Jitter()
    {
        return cbFI.fJitter;
    }

    FfxInt32x2 InterpolationRectBase()
    {
        return cbFI.interpolationRectBase;
    }

    FfxInt32x2 InterpolationRectSize()
    {
        return cbFI.interpolationRectSize;
    }

    FfxFloat32x2 MotionVectorScale()
    {
        return cbFI.fMotionVectorScale;
    }

    FfxInt32x2 RenderSize()
    {
        return cbFI.renderSize;
    }

    FfxInt32x2 DisplaySize()
    {
        return cbFI.displaySize;
    }

    FfxBoolean Reset()
    {
        return cbFI.reset == 1;
    }

    FfxFloat32x4 DeviceToViewSpaceTransformFactors()
    {
        return cbFI.fDeviceToViewDepth;
    }

    FfxInt32x2 GetOpticalFlowSize()
    {
        FfxInt32x2 iOpticalFlowSize = FfxInt32x2((1.0 / cbFI.opticalFlowScale) / FfxFloat32x2(cbFI.opticalFlowBlockSize.xx));

        return iOpticalFlowSize;
    }

    FfxInt32x2 GetOpticalFlowSize2()
    {
        return GetOpticalFlowSize() * 1;
    }

    FfxFloat32x2 GetOpticalFlowScale()
    {
        return cbFI.opticalFlowScale;
    }

    FfxInt32 GetOpticalFlowBlockSize()
    {
        return cbFI.opticalFlowBlockSize;
    }
    
    FfxInt32 GetHUDLessAttachedFactor()
    {
        return cbFI.HUDLessAttachedFactor;
    }

    FfxInt32x2 GetDistortionFieldSize()
    {
        return cbFI.distortionFieldSize;
    }

    FfxUInt32 GetDispatchFlags()
    {
        return cbFI.dispatchFlags;
    }

    FfxInt32x2 GetMaxRenderSize()
    {
        return cbFI.maxRenderSize;
    }

    FfxInt32 GetOpticalFlowHalfResMode()
    {
        return cbFI.opticalFlowHalfResMode;
    }

    FfxFloat32x3 GetDebugBarColor()
    {
        return cbFI.debugBarColor;
    }

    FfxFloat32 TanHalfFoV()
    {
        return cbFI.fTanHalfFOV;
    }

    FfxUInt32 BackBufferTransferFunction()
    {
        return cbFI.backBufferTransferFunction;
    }

    FfxFloat32 MinLuminance()
    {
        return cbFI.minMaxLuminance[0];
    }

    FfxFloat32 MaxLuminance()
    {
        return cbFI.minMaxLuminance[1];
    }

#endif // defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)


#if defined(FFX_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID, std140) uniform cbInpaintingPyramid_t
    {
        FfxUInt32 mips;
        FfxUInt32 numWorkGroups;
        FfxUInt32x2 workGroupOffset;
    } cbInpaintingPyramid;

    FfxUInt32 NumMips()
    {
        return cbInpaintingPyramid.mips;
    }
    FfxUInt32 NumWorkGroups()
    {
        return cbInpaintingPyramid.numWorkGroups;
    }
    FfxUInt32x2 WorkGroupOffset()
    {
        return cbInpaintingPyramid.workGroupOffset;
    }

#endif // defined(FFX_FRAMEINTERPOLATION_BIND_CB_INPAINTING_PYRAMID)


  ///////////////////////////////////////////////
 // declare samplers
///////////////////////////////////////////////


layout (set = 0, binding = 1000) uniform sampler s_LinearClamp;

  ///////////////////////////////////////////////
 // declare SRVs and SRV accessors
///////////////////////////////////////////////

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_PREVIOUS_INTERPOLATION_SOURCE)  uniform texture2D  r_previous_interpolation_source;

    FfxFloat32x3 LoadPreviousBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_previous_interpolation_source, iPxPos, 0).rgb;
    }
    FfxFloat32x3 SamplePreviousBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return textureLod(sampler2D(r_previous_interpolation_source, s_LinearClamp), fUv, 0.0).xyz;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_CURRENT_INTERPOLATION_SOURCE)  uniform texture2D  r_current_interpolation_source;

    FfxFloat32x3 LoadCurrentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_current_interpolation_source, iPxPos, 0).rgb;
    }
    FfxFloat32x3 SampleCurrentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return textureLod(sampler2D(r_current_interpolation_source, s_LinearClamp), fUv, 0.0).xyz;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS)  uniform texture2D  r_dilated_motion_vectors;

    FfxFloat32x2 LoadDilatedMotionVector(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_dilated_motion_vectors, iPxPos, 0).xy;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_DEPTH
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_DEPTH)  uniform texture2D  r_dilated_depth;

    FfxFloat32 LoadDilatedDepth(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_dilated_depth, iPxPos, 0).x;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME)  uniform utexture2D  r_reconstructed_depth_previous_frame;

    FfxFloat32 LoadReconstructedDepthPreviousFrame(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return ffxAsFloat(texelFetch(r_reconstructed_depth_previous_frame, iPxInput, 0).x);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME)  uniform utexture2D  r_reconstructed_depth_interpolated_frame;

    FfxFloat32 LoadEstimatedInterpolationFrameDepth(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return ffxAsFloat(texelFetch(r_reconstructed_depth_interpolated_frame, iPxInput, 0).x);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DISOCCLUSION_MASK
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_DISOCCLUSION_MASK)  uniform texture2D  r_disocclusion_mask;

    FfxFloat32x4 LoadDisocclusionMask(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_disocclusion_mask, iPxPos, 0);
    }
    FfxFloat32x4 SampleDisocclusionMask(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return textureLod(sampler2D(r_disocclusion_mask, s_LinearClamp), fUv, 0);
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_Y)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_X)  uniform utexture2D  r_game_motion_vector_field_x;
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_Y)  uniform utexture2D  r_game_motion_vector_field_y;

    FfxUInt32x2 LoadGameFieldMv(FFX_PARAMETER_IN FfxInt32x2 iPxSample)
    {
        FfxUInt32 packedX = texelFetch(r_game_motion_vector_field_x, iPxSample, 0).x;
        FfxUInt32 packedY = texelFetch(r_game_motion_vector_field_y, iPxSample, 0).x;

        return FfxUInt32x2(packedX, packedY);
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X)  uniform utexture2D  r_optical_flow_motion_vector_field_x;
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y)  uniform utexture2D  r_optical_flow_motion_vector_field_y;

    FfxUInt32x2 LoadOpticalFlowFieldMv(FFX_PARAMETER_IN FfxInt32x2 iPxSample)
    {
        FfxUInt32 packedX = texelFetch(r_optical_flow_motion_vector_field_x, iPxSample, 0).x;
        FfxUInt32 packedY = texelFetch(r_optical_flow_motion_vector_field_y, iPxSample, 0).x;

        return FfxUInt32x2(packedX, packedY);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW)  uniform itexture2D  r_optical_flow;
    
    #if defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
        FfxFloat32x2 LoadOpticalFlow(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
        {
            return texelFetch(r_optical_flow, iPxPos, 0).xy * GetOpticalFlowScale();
        }
    #endif
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_UPSAMPLED
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_UPSAMPLED)  uniform texture2D  r_optical_flow_upsampled;

    FfxFloat32x2 LoadOpticalFlowUpsampled(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_optical_flow_upsampled, iPxPos, 0).xy;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_CONFIDENCE
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_CONFIDENCE)  uniform utexture2D  r_optical_flow_confidence;
    
    FfxFloat32 LoadOpticalFlowConfidence(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_optical_flow_confidence, iPxPos, 0).y;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_GLOBAL_MOTION)  uniform utexture2D  r_optical_flow_global_motion;

    FfxUInt32 LoadOpticalFlowGlobalMotion(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_optical_flow_global_motion, iPxPos, 0).x;
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_SCENE_CHANGE_DETECTION
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_SCENE_CHANGE_DETECTION)  uniform utexture2D  r_optical_flow_scd;

    FfxUInt32 LoadOpticalFlowSceneChangeDetection(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_optical_flow_scd, iPxPos, 0).x;
    }

    FfxBoolean HasSceneChanged()
    {
        #define SCD_OUTPUT_HISTORY_BITS_SLOT 1
        //if (FrameIndex() <= 5) // threshold according to original OpenCL code
        //{
        //    return 1.0;
        //}
        //else
        {
            // Report that the scene is changed if the change was detected in any of the
            // 4 previous frames (0xfu - covers 4 history bits).
            return ((texelFetch(r_optical_flow_scd, FfxInt32x2(SCD_OUTPUT_HISTORY_BITS_SLOT, 0), 0).x) & 0xfu) != 0;
        }
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_DEBUG
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_DEBUG)  uniform texture2D  r_optical_flow_debug;

    FfxFloat32x4 LoadOpticalFlowDebug(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_optical_flow_debug, iPxPos, 0);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_OUTPUT
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_OUTPUT)  uniform texture2D  r_output;

    FfxFloat32x4 LoadFrameInterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return texelFetch(r_output, iPxInput, 0);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_INPAINTING_PYRAMID
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_INPAINTING_PYRAMID)  uniform texture2D  r_inpainting_pyramid;

    FfxFloat32x4 LoadInpaintingPyramid(FFX_PARAMETER_IN FfxInt32 mipLevel, FFX_PARAMETER_IN FfxUInt32x2 iPxInput)
    {
        return texelFetch(r_inpainting_pyramid, FfxInt32x2(iPxInput), mipLevel);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_PRESENT_BACKBUFFER)  uniform texture2D  r_present_backbuffer;

    FfxFloat32x4 LoadPresentBackbuffer(FFX_PARAMETER_IN FfxInt32x2 iPxInput)
    {
        return texelFetch(r_present_backbuffer, iPxInput, 0);
    }
    FfxFloat32x4 SamplePresentBackbuffer(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return textureLod(sampler2D(r_present_backbuffer, s_LinearClamp), fUv, 0.0);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_COUNTERS
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_COUNTERS) readonly buffer FrameInterpolationCounters_t
    {
        FfxUInt32 data[];
    } r_counters;

    FfxUInt32 LoadCounter(FFX_PARAMETER_IN FfxInt32 iPxPos)
    {
        return r_counters.data[iPxPos];
    }

    FfxUInt32 FrameIndexSinceLastReset()
    {
        return LoadCounter(COUNTER_FRAME_INDEX_SINCE_LAST_RESET);
    }
#endif


#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_INPUT_DEPTH) uniform texture2D  r_input_depth;

    FfxFloat32 LoadInputDepth(FfxInt32x2 iPxPos)
    {
        return texelFetch(r_input_depth, iPxPos, 0).x;
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_VECTORS)
    layout (set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_INPUT_MOTION_VECTORS) uniform texture2D  r_input_motion_vectors;

    FfxFloat32x2 LoadInputMotionVector(FfxInt32x2 iPxDilatedMotionVectorPos)
    {
        FfxFloat32x2 fSrcMotionVector = texelFetch(r_input_motion_vectors, iPxDilatedMotionVectorPos, 0).xy;

        FfxFloat32x2 fUvMotionVector = fSrcMotionVector * MotionVectorScale();

    #if FFX_FRAMEINTERPOLATION_OPTION_JITTERED_MOTION_VECTORS
        fUvMotionVector -= MotionVectorJitterCancellation();
    #endif

        return fUvMotionVector;
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_DISTORTION_FIELD)
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_SRV_DISTORTION_FIELD) uniform texture2D  r_input_distortion_field;
    FfxFloat32x2 SampleDistortionField(FFX_PARAMETER_IN FfxFloat32x2 fUv)
    {
        return textureLod(sampler2D(r_input_distortion_field, s_LinearClamp), fUv, 0.0).xy;
    }
#endif

///////////////////////////////////////////////
// declare UAVs and UAV accessors
///////////////////////////////////////////////
#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_OUTPUT
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_OUTPUT /* app controlled format */)  uniform image2D    rw_output;

    FfxFloat32x4 RWLoadFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_output, iPxPos);
    }

    void StoreFrameinterpolationOutput(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 val)
    {
        imageStore(rw_output, iPxPos, val);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_DILATED_MOTION_VECTORS
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_DILATED_MOTION_VECTORS, rg16f)  uniform image2D    rw_dilated_motion_vectors;

    FfxFloat32x2 RWLoadDilatedMotionVectors(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_dilated_motion_vectors, iPxPos).xy;
    }

    void StoreDilatedMotionVectors(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 val)
    {
        imageStore(rw_dilated_motion_vectors, iPxPos, FfxFloat32x4(val, 0.0, 0.0));
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_DILATED_DEPTH
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_DILATED_DEPTH, r32f)  uniform image2D    rw_dilated_depth;

    FfxFloat32 RWLoadDilatedDepth(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_dilated_depth, iPxPos).x;
    }

    void StoreDilatedDepth(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 val)
    {
        imageStore(rw_dilated_depth, iPxPos, FfxFloat32x4(val, 0.0, 0.0, 0.0));
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME, r32ui)  uniform uimage2D    rw_reconstructed_depth_previous_frame;

    FfxFloat32 RWLoadReconstructedDepthPreviousFrame(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return ffxAsFloat(imageLoad(rw_reconstructed_depth_previous_frame, iPxPos).x);
    }

    void UpdateReconstructedDepthPreviousFrame(FfxInt32x2 iPxSample, FfxFloat32 fDepth)
    {
        FfxUInt32 uDepth = ffxAsUInt32(fDepth);

#if FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        imageAtomicMax(rw_reconstructed_depth_previous_frame, iPxSample, uDepth);
#else
        imageAtomicMin(rw_reconstructed_depth_previous_frame, iPxSample, uDepth);  // min for standard, max for inverted depth
#endif
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME, r32ui)  uniform uimage2D    rw_reconstructed_depth_interpolated_frame;
    
    FfxFloat32 RWLoadReconstructedDepthInterpolatedFrame(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return ffxAsFloat(imageLoad(rw_reconstructed_depth_interpolated_frame, iPxPos).x);
    }

    void StoreReconstructedDepthInterpolatedFrame(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 value)
    {
        FfxUInt32 uDepth                                  = ffxAsUInt32(value);
        imageStore(rw_reconstructed_depth_interpolated_frame, iPxPos, FfxUInt32x4(uDepth, 0, 0, 0));
    }

    void UpdateReconstructedDepthInterpolatedFrame(FfxInt32x2 iPxSample, FfxFloat32 fDepth)
    {
        FfxUInt32 uDepth = ffxAsUInt32(fDepth);

#if FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
        imageAtomicMax(rw_reconstructed_depth_interpolated_frame, iPxSample, uDepth);
#else
        imageAtomicMin(rw_reconstructed_depth_interpolated_frame, iPxSample, uDepth);  // min for standard, max for inverted depth
#endif
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_DISOCCLUSION_MASK
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_DISOCCLUSION_MASK, rg8)  uniform image2D    rw_disocclusion_mask;

    FfxFloat32x2 RWLoadDisocclusionMask(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_disocclusion_mask, iPxPos).xy;
    }

    void StoreDisocclusionMask(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 val)
    {
        imageStore(rw_disocclusion_mask, iPxPos, FfxFloat32x4(val, 0.0, 0.0));
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_UAV_GAME_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_GAME_MOTION_VECTOR_FIELD_Y)

	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_GAME_MOTION_VECTOR_FIELD_X, r32ui)  uniform uimage2D    rw_game_motion_vector_field_x;
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_GAME_MOTION_VECTOR_FIELD_Y, r32ui)  uniform uimage2D    rw_game_motion_vector_field_y;

    FfxUInt32 RWLoadGameMotionVectorFieldX(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_game_motion_vector_field_x, iPxPos).x;
    }

    void StoreGameMotionVectorFieldX(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 val)
    {
        imageStore(rw_game_motion_vector_field_x, iPxPos, FfxUInt32x4(val, 0, 0, 0));
    }

    FfxUInt32 RWLoadGameMotionVectorFieldY(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_game_motion_vector_field_y, iPxPos).x;
    }

    void StoreGameMotionVectorFieldY(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 val)
    {
        imageStore(rw_game_motion_vector_field_y, iPxPos, FfxUInt32x4(val, 0, 0, 0));
    }

    void UpdateGameMotionVectorField(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32x2 packedVector)
    {
        imageAtomicMax(rw_game_motion_vector_field_x, iPxPos, packedVector.x);
        imageAtomicMax(rw_game_motion_vector_field_y, iPxPos, packedVector.y);
    }

    FfxUInt32 UpdateGameMotionVectorFieldEx(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32x2 packedVector)
    {
        FfxUInt32 uPreviousValueX = imageAtomicMax(rw_game_motion_vector_field_x, iPxPos, packedVector.x);
        FfxUInt32 uPreviousValueY = imageAtomicMax(rw_game_motion_vector_field_y, iPxPos, packedVector.y);

        const FfxUInt32 uExistingVectorFieldEntry = ffxMax(uPreviousValueX, uPreviousValueY);

        return uExistingVectorFieldEntry;
    }
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_UAV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y)

	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X, r32ui)  uniform uimage2D    rw_optical_flow_motion_vector_field_x;
	layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y, r32ui)  uniform uimage2D    rw_optical_flow_motion_vector_field_y;

    FfxUInt32 RWLoadOpticalflowMotionVectorFieldX(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_optical_flow_motion_vector_field_x, iPxPos).x;
    }
    void StoreOpticalflowMotionVectorFieldX(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 val)
    {
        imageStore(rw_optical_flow_motion_vector_field_x, iPxPos, FfxUInt32x4(val, 0, 0, 0));
    }
    FfxUInt32 RWLoadOpticalflowMotionVectorFieldY(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return imageLoad(rw_optical_flow_motion_vector_field_y, iPxPos).x;
    }
    void StoreOpticalflowMotionVectorFieldY(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 val)
    {
        imageStore(rw_optical_flow_motion_vector_field_y, iPxPos, FfxUInt32x4(val, 0, 0, 0));
    }
    void UpdateOpticalflowMotionVectorField(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32x2 packedVector)
    {
        imageAtomicMax(rw_optical_flow_motion_vector_field_x, iPxPos, packedVector.x);
        imageAtomicMax(rw_optical_flow_motion_vector_field_y, iPxPos, packedVector.y);
    }
#endif

#ifdef FFX_FRAMEINTERPOLATION_BIND_UAV_COUNTERS
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_COUNTERS) coherent buffer FrameInterpolationRWCounters_t
    {
        FfxUInt32 data[];
    } rw_counters;
    
    FfxUInt32 RWLoadCounter(FFX_PARAMETER_IN FfxInt32 iPxPos)
    {
        return rw_counters.data[iPxPos];
    }

    void StoreCounter(FFX_PARAMETER_IN FfxInt32 iPxPos, FFX_PARAMETER_IN FfxUInt32 counter)
    {
        rw_counters.data[iPxPos] = counter;
    }
    void AtomicIncreaseCounter(FFX_PARAMETER_IN FfxInt32 iPxPos, FFX_PARAMETER_OUT FfxUInt32 oldVal)
    {
        oldVal = atomicAdd(rw_counters.data[iPxPos], 1);
    }
#endif


#if defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_0)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_1)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_2)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_3)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_4)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_5)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_6)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_7)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_8)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_9)    && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_10)   && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_11)   && \
    defined(FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_12)

    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_0,  rgba16f)           uniform image2D  rw_inpainting_pyramid0;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_1,  rgba16f)           uniform image2D  rw_inpainting_pyramid1;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_2,  rgba16f)           uniform image2D  rw_inpainting_pyramid2;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_3,  rgba16f)           uniform image2D  rw_inpainting_pyramid3;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_4,  rgba16f)           uniform image2D  rw_inpainting_pyramid4;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_5,  rgba16f)  coherent uniform image2D  rw_inpainting_pyramid5;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_6,  rgba16f)           uniform image2D  rw_inpainting_pyramid6;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_7,  rgba16f)           uniform image2D  rw_inpainting_pyramid7;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_8,  rgba16f)           uniform image2D  rw_inpainting_pyramid8;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_9,  rgba16f)           uniform image2D  rw_inpainting_pyramid9;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_10, rgba16f)           uniform image2D  rw_inpainting_pyramid10;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_11, rgba16f)           uniform image2D  rw_inpainting_pyramid11;
    layout(set = 0, binding = FFX_FRAMEINTERPOLATION_BIND_UAV_INPAINTING_PYRAMID_MIPMAP_12, rgba16f)           uniform image2D  rw_inpainting_pyramid12;


    FfxFloat32x4 RWLoadInpaintingPyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxUInt32 index)
    {
        #define LOAD(idx)                                 \
            if (index == idx)                             \
            {                                             \
                return imageLoad(rw_inpainting_pyramid##idx, iPxPos); \
            }
        LOAD(0);
        LOAD(1);
        LOAD(2);
        LOAD(3);
        LOAD(4);
        LOAD(5);
        LOAD(6);
        LOAD(7);
        LOAD(8);
        LOAD(9);
        LOAD(10);
        LOAD(11);
        LOAD(12);
        return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);

        #undef LOAD
    }

    void StoreInpaintingPyramid(FFX_PARAMETER_IN FfxInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 outValue, FFX_PARAMETER_IN FfxUInt32 index)
    {
        #define STORE(idx)                   \
            if (index == idx)                \
            {                                \
                imageStore(rw_inpainting_pyramid##idx, iPxPos, outValue); \
            }

        STORE(0);
        STORE(1);
        STORE(2);
        STORE(3);
        STORE(4);
        STORE(5);
        STORE(6);
        STORE(7);
        STORE(8);
        STORE(9);
        STORE(10);
        STORE(11);
        STORE(12);

        #undef STORE
    }
#endif
