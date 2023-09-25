// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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

#include "ffx_fsr2_resources.h"

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif //__hlsl_dx_compiler
#include "ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif //__hlsl_dx_compiler
#endif // #if defined(FFX_GPU)

#if defined(FFX_GPU)
#ifndef FFX_FSR2_PREFER_WAVE64
#define FFX_FSR2_PREFER_WAVE64
#endif // #if defined(FFX_GPU)

#if defined(FFX_GPU)
#pragma warning(disable: 3205)  // conversion from larger type to smaller
#endif // #if defined(FFX_GPU)

#define DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_FSR2_DECLARE_SRV(regIndex)  register(DECLARE_SRV_REGISTER(regIndex))
#define FFX_FSR2_DECLARE_UAV(regIndex)  register(DECLARE_UAV_REGISTER(regIndex))
#define FFX_FSR2_DECLARE_CB(regIndex)   register(DECLARE_CB_REGISTER(regIndex))

#if defined(FSR2_BIND_CB_FSR2) || defined(FFX_INTERNAL)
    cbuffer cbFSR2 : FFX_FSR2_DECLARE_CB(FSR2_BIND_CB_FSR2)
    {
        FfxInt32x2    iRenderSize;
        FfxInt32x2    iMaxRenderSize;
        FfxInt32x2    iDisplaySize;
        FfxInt32x2    iInputColorResourceDimensions;
        FfxInt32x2    iLumaMipDimensions;
        FfxInt32      iLumaMipLevelToUse;
        FfxInt32      iFrameIndex;

        FfxFloat32x4  fDeviceToViewDepth;
        FfxFloat32x2  fJitter;
        FfxFloat32x2  fMotionVectorScale;
        FfxFloat32x2  fDownscaleFactor;
        FfxFloat32x2  fMotionVectorJitterCancellation;
        FfxFloat32    fPreExposure;
        FfxFloat32    fPreviousFramePreExposure;
        FfxFloat32    fTanHalfFOV;
        FfxFloat32    fJitterSequenceLength;
        FfxFloat32    fDeltaTime;
        FfxFloat32    fDynamicResChangeFactor;
        FfxFloat32    fViewSpaceToMetersFactor;
    };

#define FFX_FSR2_CONSTANT_BUFFER_1_SIZE (sizeof(cbFSR2) / 4)  // Number of 32-bit values. This must be kept in sync with the cbFSR2 size.
#endif

#if defined(FFX_GPU)
#define FFX_FSR2_ROOTSIG_STRINGIFY(p) FFX_FSR2_ROOTSIG_STR(p)
#define FFX_FSR2_ROOTSIG_STR(p) #p
#define FFX_FSR2_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "RootConstants(num32BitConstants=" FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_CONSTANT_BUFFER_1_SIZE) ", b0), " \
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

#define FFX_FSR2_CONSTANT_BUFFER_2_SIZE 6  // Number of 32-bit values. This must be kept in sync with max( cbRCAS , cbSPD) size.

#define FFX_FSR2_CB2_ROOTSIG [RootSignature( "DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "RootConstants(num32BitConstants=" FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_CONSTANT_BUFFER_1_SIZE) ", b0), " \
                                    "RootConstants(num32BitConstants=" FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_CONSTANT_BUFFER_2_SIZE) ", b1), " \
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
#if defined(FFX_FSR2_EMBED_ROOTSIG)
#define FFX_FSR2_EMBED_ROOTSIG_CONTENT FFX_FSR2_ROOTSIG
#define FFX_FSR2_EMBED_CB2_ROOTSIG_CONTENT FFX_FSR2_CB2_ROOTSIG
#else
#define FFX_FSR2_EMBED_ROOTSIG_CONTENT
#define FFX_FSR2_EMBED_CB2_ROOTSIG_CONTENT
#endif // #if FFX_FSR2_EMBED_ROOTSIG
#endif // #if defined(FFX_GPU)

/* Define getter functions in the order they are defined in the CB! */
FfxInt32x2 RenderSize()
{
    return iRenderSize;
}

FfxInt32x2 MaxRenderSize()
{
    return iMaxRenderSize;
}

FfxInt32x2 DisplaySize()
{
    return iDisplaySize;
}

FfxInt32x2 InputColorResourceDimensions()
{
    return iInputColorResourceDimensions;
}

FfxInt32x2 LumaMipDimensions()
{
    return iLumaMipDimensions;
}

FfxInt32  LumaMipLevelToUse()
{
    return iLumaMipLevelToUse;
}

FfxInt32 FrameIndex()
{
    return iFrameIndex;
}

FfxFloat32x2 Jitter()
{
    return fJitter;
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

FfxFloat32 PreExposure()
{
    return fPreExposure;
}

FfxFloat32 PreviousFramePreExposure()
{
    return fPreviousFramePreExposure;
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

FfxFloat32 DynamicResChangeFactor()
{
    return fDynamicResChangeFactor;
}

FfxFloat32 ViewSpaceToMetersFactor()
{
    return fViewSpaceToMetersFactor;
}


SamplerState s_PointClamp : register(s0);
SamplerState s_LinearClamp : register(s1);

// SRVs
#if defined(FFX_INTERNAL)
    Texture2D<FfxFloat32x4>                       r_input_opaque_only                       : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY);
    Texture2D<FfxFloat32x4>                       r_input_color_jittered                    : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR);
    Texture2D<FfxFloat32x4>                       r_input_motion_vectors                    : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS);
    Texture2D<FfxFloat32>                         r_input_depth                             : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_DEPTH);
    Texture2D<FfxFloat32x2>                       r_input_exposure                          : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE);
    Texture2D<FfxFloat32x2>                       r_auto_exposure                           : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE);
    Texture2D<FfxFloat32>                         r_reactive_mask                           : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK);
    Texture2D<FfxFloat32>                         r_transparency_and_composition_mask       : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK);
    Texture2D<FfxUInt32>                          r_reconstructed_previous_nearest_depth    : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH);
    Texture2D<FfxFloat32x2>                       r_dilated_motion_vectors                  : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS);
    Texture2D<FfxFloat32x2>                       r_previous_dilated_motion_vectors         : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREVIOUS_DILATED_MOTION_VECTORS);
    Texture2D<FfxFloat32>                         r_dilatedDepth                            : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH);
    Texture2D<FfxFloat32x4>                       r_internal_upscaled_color                 : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR);
    Texture2D<unorm FfxFloat32x2>                 r_lock_status                             : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS);
    Texture2D<FfxFloat32>                         r_lock_input_luma                         : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA);
    Texture2D<unorm FfxFloat32>                   r_new_locks                               : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS);
    Texture2D<FfxFloat32x4>                       r_prepared_input_color                    : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR);
    Texture2D<FfxFloat32x4>                       r_luma_history                            : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY);
    Texture2D<FfxFloat32x4>                       r_rcas_input                              : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_RCAS_INPUT);
    Texture2D<FfxFloat32>                         r_lanczos_lut                             : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_LANCZOS_LUT);
    Texture2D<FfxFloat32>                         r_imgMips                                 : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE);
    Texture2D<FfxFloat32>                         r_upsample_maximum_bias_lut               : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTITIER_UPSAMPLE_MAXIMUM_BIAS_LUT);
    Texture2D<unorm FfxFloat32x2>                 r_dilated_reactive_masks                  : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS);
    Texture2D<float3>                             r_input_prev_color_pre_alpha              : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR);
    Texture2D<float3>                             r_input_prev_color_post_alpha             : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR);

    Texture2D<FfxFloat32x4>                       r_debug_out                               : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_DEBUG_OUTPUT);

    // UAV declarations
    RWTexture2D<FfxUInt32>                        rw_reconstructed_previous_nearest_depth   : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH);
    RWTexture2D<FfxFloat32x2>                     rw_dilated_motion_vectors                 : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS);
    RWTexture2D<FfxFloat32>                       rw_dilatedDepth                           : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH);
    RWTexture2D<FfxFloat32x4>                     rw_internal_upscaled_color                : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR);
    RWTexture2D<unorm FfxFloat32x2>               rw_lock_status                            : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS);
    RWTexture2D<FfxFloat32>                       rw_lock_input_luma                        : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA);
    RWTexture2D<unorm FfxFloat32>                 rw_new_locks                              : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS);
    RWTexture2D<FfxFloat32x4>                     rw_prepared_input_color                   : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR);
    RWTexture2D<FfxFloat32x4>                     rw_luma_history                           : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY);
    RWTexture2D<FfxFloat32x4>                     rw_upscaled_output                        : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT);

    globallycoherent RWTexture2D<FfxFloat32>      rw_img_mip_shading_change                 : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_SHADING_CHANGE);
    globallycoherent RWTexture2D<FfxFloat32>      rw_img_mip_5                              : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_5);
    RWTexture2D<unorm FfxFloat32x2>               rw_dilated_reactive_masks                 : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS);
    RWTexture2D<FfxFloat32x2>                     rw_auto_exposure                          : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE);
    globallycoherent RWTexture2D<FfxUInt32>       rw_spd_global_atomic                      : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT);
    RWTexture2D<FfxFloat32x4>                     rw_debug_out                              : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_DEBUG_OUTPUT);
    
    RWTexture2D<float>                            rw_output_autoreactive                    : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE);
    RWTexture2D<float>                            rw_output_autocomposition                 : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION);
    RWTexture2D<float3>                           rw_output_prev_color_pre_alpha            : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR);
    RWTexture2D<float3>                           rw_output_prev_color_post_alpha           : FFX_FSR2_DECLARE_UAV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR);  

#else // #if defined(FFX_INTERNAL)
    #if defined FSR2_BIND_SRV_INPUT_COLOR
        Texture2D<FfxFloat32x4>                   r_input_color_jittered                    : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INPUT_COLOR);
    #endif
    #if defined FSR2_BIND_SRV_INPUT_OPAQUE_ONLY
        Texture2D<FfxFloat32x4>                   r_input_opaque_only                       : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INPUT_OPAQUE_ONLY);
    #endif
    #if defined FSR2_BIND_SRV_INPUT_MOTION_VECTORS
        Texture2D<FfxFloat32x4>                   r_input_motion_vectors                    : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INPUT_MOTION_VECTORS);
    #endif
    #if defined FSR2_BIND_SRV_INPUT_DEPTH
        Texture2D<FfxFloat32>                     r_input_depth                             : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INPUT_DEPTH);
    #endif 
    #if defined FSR2_BIND_SRV_INPUT_EXPOSURE
        Texture2D<FfxFloat32x2>                   r_input_exposure                          : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INPUT_EXPOSURE);
    #endif
    #if defined FSR2_BIND_SRV_AUTO_EXPOSURE
        Texture2D<FfxFloat32x2>                   r_auto_exposure                           : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_AUTO_EXPOSURE);
    #endif
    #if defined FSR2_BIND_SRV_REACTIVE_MASK
        Texture2D<FfxFloat32>                     r_reactive_mask                           : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_REACTIVE_MASK);
    #endif 
    #if defined FSR2_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK
        Texture2D<FfxFloat32>                     r_transparency_and_composition_mask       : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK);
    #endif
    #if defined FSR2_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH
        Texture2D<FfxUInt32>                      r_reconstructed_previous_nearest_depth    : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH);
    #endif 
    #if defined FSR2_BIND_SRV_DILATED_MOTION_VECTORS
       Texture2D<FfxFloat32x2>                    r_dilated_motion_vectors                  : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_DILATED_MOTION_VECTORS);
    #endif
    #if defined FSR2_BIND_SRV_PREVIOUS_DILATED_MOTION_VECTORS
           Texture2D<FfxFloat32x2>                r_previous_dilated_motion_vectors         : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_PREVIOUS_DILATED_MOTION_VECTORS);
    #endif
    #if defined FSR2_BIND_SRV_DILATED_DEPTH
        Texture2D<FfxFloat32>                     r_dilatedDepth                            : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_DILATED_DEPTH);
    #endif
    #if defined FSR2_BIND_SRV_INTERNAL_UPSCALED
        Texture2D<FfxFloat32x4>                   r_internal_upscaled_color                 : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_INTERNAL_UPSCALED);
    #endif
    #if defined FSR2_BIND_SRV_LOCK_STATUS
        Texture2D<unorm FfxFloat32x2>             r_lock_status                             : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_LOCK_STATUS);
    #endif
    #if defined FSR2_BIND_SRV_LOCK_INPUT_LUMA
        Texture2D<FfxFloat32>                     r_lock_input_luma                         : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_LOCK_INPUT_LUMA);
    #endif
    #if defined FSR2_BIND_SRV_NEW_LOCKS
        Texture2D<unorm FfxFloat32>               r_new_locks                               : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_NEW_LOCKS);
    #endif
    #if defined FSR2_BIND_SRV_PREPARED_INPUT_COLOR
        Texture2D<FfxFloat32x4>                  r_prepared_input_color                    : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_PREPARED_INPUT_COLOR);
    #endif
    #if defined FSR2_BIND_SRV_LUMA_HISTORY
        Texture2D<unorm FfxFloat32x4>             r_luma_history                            : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_LUMA_HISTORY);
    #endif
    #if defined FSR2_BIND_SRV_RCAS_INPUT
        Texture2D<FfxFloat32x4>                   r_rcas_input                              : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_RCAS_INPUT);
    #endif
    #if defined FSR2_BIND_SRV_LANCZOS_LUT
        Texture2D<FfxFloat32>                     r_lanczos_lut                             : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_LANCZOS_LUT);
    #endif
    #if defined FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS
        Texture2D<FfxFloat32>                     r_imgMips                                 : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS);
    #endif
    #if defined FSR2_BIND_SRV_UPSCALE_MAXIMUM_BIAS_LUT
        Texture2D<FfxFloat32>                     r_upsample_maximum_bias_lut               : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_UPSCALE_MAXIMUM_BIAS_LUT);
    #endif
    #if defined FSR2_BIND_SRV_DILATED_REACTIVE_MASKS
        Texture2D<unorm FfxFloat32x2>             r_dilated_reactive_masks                  : FFX_FSR2_DECLARE_SRV(FSR2_BIND_SRV_DILATED_REACTIVE_MASKS);
    #endif

    #if defined FSR2_BIND_SRV_PREV_PRE_ALPHA_COLOR
        Texture2D<float3>                         r_input_prev_color_pre_alpha              : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR);
    #endif
    #if defined FSR2_BIND_SRV_PREV_POST_ALPHA_COLOR
        Texture2D<float3>                         r_input_prev_color_post_alpha             : FFX_FSR2_DECLARE_SRV(FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR);
    #endif
   
    // UAV declarations
    #if defined FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH
        RWTexture2D<FfxUInt32>                    rw_reconstructed_previous_nearest_depth   : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH);
    #endif
    #if defined FSR2_BIND_UAV_DILATED_MOTION_VECTORS
        RWTexture2D<FfxFloat32x2>                 rw_dilated_motion_vectors                 : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_DILATED_MOTION_VECTORS);
    #endif
    #if defined FSR2_BIND_UAV_DILATED_DEPTH
        RWTexture2D<FfxFloat32>                   rw_dilatedDepth                           : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_DILATED_DEPTH);
    #endif
    #if defined FSR2_BIND_UAV_INTERNAL_UPSCALED
        RWTexture2D<FfxFloat32x4>                 rw_internal_upscaled_color                : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_INTERNAL_UPSCALED);
    #endif
    #if defined FSR2_BIND_UAV_LOCK_STATUS
        RWTexture2D<unorm FfxFloat32x2>           rw_lock_status                            : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_LOCK_STATUS);
    #endif
    #if defined FSR2_BIND_UAV_LOCK_INPUT_LUMA
        RWTexture2D<FfxFloat32>                   rw_lock_input_luma                        : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_LOCK_INPUT_LUMA);
    #endif
    #if defined FSR2_BIND_UAV_NEW_LOCKS
        RWTexture2D<unorm FfxFloat32>             rw_new_locks                              : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_NEW_LOCKS);
    #endif
    #if defined FSR2_BIND_UAV_PREPARED_INPUT_COLOR
        RWTexture2D<FfxFloat32x4>                 rw_prepared_input_color                   : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_PREPARED_INPUT_COLOR);
    #endif
    #if defined FSR2_BIND_UAV_LUMA_HISTORY
        RWTexture2D<FfxFloat32x4>                 rw_luma_history                           : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_LUMA_HISTORY);
    #endif
    #if defined FSR2_BIND_UAV_UPSCALED_OUTPUT
        RWTexture2D<FfxFloat32x4>                 rw_upscaled_output                        : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_UPSCALED_OUTPUT);
    #endif
    #if defined FSR2_BIND_UAV_EXPOSURE_MIP_LUMA_CHANGE
        globallycoherent RWTexture2D<FfxFloat32>  rw_img_mip_shading_change                 : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_EXPOSURE_MIP_LUMA_CHANGE);
    #endif
    #if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
        globallycoherent RWTexture2D<FfxFloat32>  rw_img_mip_5                              : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_EXPOSURE_MIP_5);
    #endif
    #if defined FSR2_BIND_UAV_DILATED_REACTIVE_MASKS
        RWTexture2D<unorm FfxFloat32x2>           rw_dilated_reactive_masks                 : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_DILATED_REACTIVE_MASKS);
    #endif
    #if defined FSR2_BIND_UAV_EXPOSURE
        RWTexture2D<FfxFloat32x2>                 rw_exposure                               : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_EXPOSURE);
    #endif
    #if defined FSR2_BIND_UAV_AUTO_EXPOSURE
        RWTexture2D<FfxFloat32x2>                 rw_auto_exposure                          : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_AUTO_EXPOSURE);
    #endif
    #if defined FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC
        globallycoherent RWTexture2D<FfxUInt32>   rw_spd_global_atomic                      : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC);
    #endif

    #if defined FSR2_BIND_UAV_AUTOREACTIVE
        RWTexture2D<float>                        rw_output_autoreactive                    : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_AUTOREACTIVE);
    #endif
    #if defined FSR2_BIND_UAV_AUTOCOMPOSITION
        RWTexture2D<float>                        rw_output_autocomposition                 : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_AUTOCOMPOSITION);
    #endif
    #if defined FSR2_BIND_UAV_PREV_PRE_ALPHA_COLOR
        RWTexture2D<float3>                       rw_output_prev_color_pre_alpha            : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_PREV_PRE_ALPHA_COLOR);
    #endif
    #if defined FSR2_BIND_UAV_PREV_POST_ALPHA_COLOR
        RWTexture2D<float3>                       rw_output_prev_color_post_alpha           : FFX_FSR2_DECLARE_UAV(FSR2_BIND_UAV_PREV_POST_ALPHA_COLOR);
    #endif
#endif // #if defined(FFX_INTERNAL)

#if defined(FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS) || defined(FFX_INTERNAL)
FfxFloat32 LoadMipLuma(FfxUInt32x2 iPxPos, FfxUInt32 mipLevel)
{
    return r_imgMips.mips[mipLevel][iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS) || defined(FFX_INTERNAL)
FfxFloat32 SampleMipLuma(FfxFloat32x2 fUV, FfxUInt32 mipLevel)
{
    return r_imgMips.SampleLevel(s_LinearClamp, fUV, mipLevel);
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_DEPTH) || defined(FFX_INTERNAL)
FfxFloat32 LoadInputDepth(FfxUInt32x2 iPxPos)
{
    return r_input_depth[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_DEPTH) || defined(FFX_INTERNAL)
FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV)
{
    return r_input_depth.SampleLevel(s_LinearClamp, fUV, 0).x;
}
#endif

#if defined(FSR2_BIND_SRV_REACTIVE_MASK) || defined(FFX_INTERNAL)
FfxFloat32 LoadReactiveMask(FfxUInt32x2 iPxPos)
{
    return r_reactive_mask[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK) || defined(FFX_INTERNAL)
FfxFloat32 LoadTransparencyAndCompositionMask(FfxUInt32x2 iPxPos)
{
    return r_transparency_and_composition_mask[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_COLOR) || defined(FFX_INTERNAL)
FfxFloat32x3 LoadInputColor(FfxUInt32x2 iPxPos)
{
    return r_input_color_jittered[iPxPos].rgb;
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_COLOR) || defined(FFX_INTERNAL)
FfxFloat32x3 SampleInputColor(FfxFloat32x2 fUV)
{
    return r_input_color_jittered.SampleLevel(s_LinearClamp, fUV, 0).rgb;
}
#endif

#if defined(FSR2_BIND_SRV_PREPARED_INPUT_COLOR) || defined(FFX_INTERNAL)
FfxFloat32x3 LoadPreparedInputColor(FfxUInt32x2 iPxPos)
{
    return r_prepared_input_color[iPxPos].xyz;
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_MOTION_VECTORS) || defined(FFX_INTERNAL)
FfxFloat32x2 LoadInputMotionVector(FfxUInt32x2 iPxDilatedMotionVectorPos)
{
    FfxFloat32x2 fSrcMotionVector = r_input_motion_vectors[iPxDilatedMotionVectorPos].xy;

    FfxFloat32x2 fUvMotionVector = fSrcMotionVector * MotionVectorScale();

#if FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS
    fUvMotionVector -= MotionVectorJitterCancellation();
#endif

    return fUvMotionVector;
}
#endif

#if defined(FSR2_BIND_SRV_INTERNAL_UPSCALED) || defined(FFX_INTERNAL)
FfxFloat32x4 LoadHistory(FfxUInt32x2 iPxHistory)
{
    return r_internal_upscaled_color[iPxHistory];
}
#endif

#if defined(FSR2_BIND_UAV_LUMA_HISTORY) || defined(FFX_INTERNAL)
void StoreLumaHistory(FfxUInt32x2 iPxPos, FfxFloat32x4 fLumaHistory)
{
    rw_luma_history[iPxPos] = fLumaHistory;
}
#endif

#if defined(FSR2_BIND_SRV_LUMA_HISTORY) || defined(FFX_INTERNAL)
FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 fUV)
{
    return r_luma_history.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FFX_INTERNAL)
FfxFloat32x4 SampleDebug(FfxFloat32x2 fUV)
{
    return r_debug_out.SampleLevel(s_LinearClamp, fUV, 0).w;
}
#endif

#if defined(FSR2_BIND_UAV_INTERNAL_UPSCALED) || defined(FFX_INTERNAL)
void StoreReprojectedHistory(FfxUInt32x2 iPxHistory, FfxFloat32x4 fHistory)
{
    rw_internal_upscaled_color[iPxHistory] = fHistory;
}
#endif

#if defined(FSR2_BIND_UAV_INTERNAL_UPSCALED) || defined(FFX_INTERNAL)
void StoreInternalColorAndWeight(FfxUInt32x2 iPxPos, FfxFloat32x4 fColorAndWeight)
{
    rw_internal_upscaled_color[iPxPos] = fColorAndWeight;
}
#endif

#if defined(FSR2_BIND_UAV_UPSCALED_OUTPUT) || defined(FFX_INTERNAL)
void StoreUpscaledOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
{
    rw_upscaled_output[iPxPos] = FfxFloat32x4(fColor, 1.f);
}
#endif

//LOCK_LIFETIME_REMAINING == 0
//Should make LockInitialLifetime() return a const 1.0f later
#if defined(FSR2_BIND_SRV_LOCK_STATUS) || defined(FFX_INTERNAL)
FfxFloat32x2 LoadLockStatus(FfxUInt32x2 iPxPos)
{
    return r_lock_status[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_LOCK_STATUS) || defined(FFX_INTERNAL)
void StoreLockStatus(FfxUInt32x2 iPxPos, FfxFloat32x2 fLockStatus)
{
    rw_lock_status[iPxPos] = fLockStatus;
}
#endif

#if defined(FSR2_BIND_SRV_LOCK_INPUT_LUMA) || defined(FFX_INTERNAL)
FfxFloat32 LoadLockInputLuma(FfxUInt32x2 iPxPos)
{
    return r_lock_input_luma[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_LOCK_INPUT_LUMA) || defined(FFX_INTERNAL)
void StoreLockInputLuma(FfxUInt32x2 iPxPos, FfxFloat32 fLuma)
{
    rw_lock_input_luma[iPxPos] = fLuma;
}
#endif

#if defined(FSR2_BIND_SRV_NEW_LOCKS) || defined(FFX_INTERNAL)
FfxFloat32 LoadNewLocks(FfxUInt32x2 iPxPos)
{
    return r_new_locks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_NEW_LOCKS) || defined(FFX_INTERNAL)
FfxFloat32 LoadRwNewLocks(FfxUInt32x2 iPxPos)
{
    return rw_new_locks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_NEW_LOCKS) || defined(FFX_INTERNAL)
void StoreNewLocks(FfxUInt32x2 iPxPos, FfxFloat32 newLock)
{
    rw_new_locks[iPxPos] = newLock;
}
#endif

#if defined(FSR2_BIND_UAV_PREPARED_INPUT_COLOR) || defined(FFX_INTERNAL)
void StorePreparedInputColor(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 fTonemapped)
{
    rw_prepared_input_color[iPxPos] = fTonemapped;
}
#endif

#if defined(FSR2_BIND_SRV_PREPARED_INPUT_COLOR) || defined(FFX_INTERNAL)
FfxFloat32 SampleDepthClip(FfxFloat32x2 fUV)
{
    return r_prepared_input_color.SampleLevel(s_LinearClamp, fUV, 0).w;
}
#endif

#if defined(FSR2_BIND_SRV_LOCK_STATUS) || defined(FFX_INTERNAL)
FfxFloat32x2 SampleLockStatus(FfxFloat32x2 fUV)
{
    FfxFloat32x2 fLockStatus = r_lock_status.SampleLevel(s_LinearClamp, fUV, 0);
    return fLockStatus;
}
#endif

#if defined(FSR2_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH) || defined(FFX_INTERNAL)
FfxFloat32 LoadReconstructedPrevDepth(FfxUInt32x2 iPxPos)
{
    return asfloat(r_reconstructed_previous_nearest_depth[iPxPos]);
}
#endif

#if defined(FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH) || defined(FFX_INTERNAL)
void StoreReconstructedDepth(FfxUInt32x2 iPxSample, FfxFloat32 fDepth)
{
    FfxUInt32 uDepth = asuint(fDepth);

    #if FFX_FSR2_OPTION_INVERTED_DEPTH
        InterlockedMax(rw_reconstructed_previous_nearest_depth[iPxSample], uDepth);
    #else
        InterlockedMin(rw_reconstructed_previous_nearest_depth[iPxSample], uDepth); // min for standard, max for inverted depth
    #endif
}
#endif

#if defined(FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH) || defined(FFX_INTERNAL)
void SetReconstructedDepth(FfxUInt32x2 iPxSample, const FfxUInt32 uValue)
{
    rw_reconstructed_previous_nearest_depth[iPxSample] = uValue;
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_DEPTH) || defined(FFX_INTERNAL)
void StoreDilatedDepth(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 fDepth)
{
    rw_dilatedDepth[iPxPos] = fDepth;
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_MOTION_VECTORS) || defined(FFX_INTERNAL)
void StoreDilatedMotionVector(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fMotionVector)
{
    rw_dilated_motion_vectors[iPxPos] = fMotionVector;
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_MOTION_VECTORS) || defined(FFX_INTERNAL)
FfxFloat32x2 LoadDilatedMotionVector(FfxUInt32x2 iPxInput)
{
    return r_dilated_motion_vectors[iPxInput].xy;
}
#endif

#if defined(FSR2_BIND_SRV_PREVIOUS_DILATED_MOTION_VECTORS) || defined(FFX_INTERNAL)
FfxFloat32x2 LoadPreviousDilatedMotionVector(FfxUInt32x2 iPxInput)
{
    return r_previous_dilated_motion_vectors[iPxInput].xy;
}

FfxFloat32x2 SamplePreviousDilatedMotionVector(FfxFloat32x2 uv)
{
    return r_previous_dilated_motion_vectors.SampleLevel(s_LinearClamp, uv, 0).xy;
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_DEPTH) || defined(FFX_INTERNAL)
FfxFloat32 LoadDilatedDepth(FfxUInt32x2 iPxInput)
{
    return r_dilatedDepth[iPxInput];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_EXPOSURE) || defined(FFX_INTERNAL)
FfxFloat32 Exposure()
{
    FfxFloat32 exposure = r_input_exposure[FfxUInt32x2(0, 0)].x;

    if (exposure == 0.0f) {
        exposure = 1.0f;
    }

    return exposure;
}
#endif

#if defined(FSR2_BIND_SRV_AUTO_EXPOSURE) || defined(FFX_INTERNAL)
FfxFloat32 AutoExposure()
{
    FfxFloat32 exposure = r_auto_exposure[FfxUInt32x2(0, 0)].x;

    if (exposure == 0.0f) {
        exposure = 1.0f;
    }

    return exposure;
}
#endif

FfxFloat32 SampleLanczos2Weight(FfxFloat32 x)
{
#if defined(FSR2_BIND_SRV_LANCZOS_LUT) || defined(FFX_INTERNAL)
    return r_lanczos_lut.SampleLevel(s_LinearClamp, FfxFloat32x2(x / 2, 0.5f), 0);
#else
    return 0.f;
#endif
}

#if defined(FSR2_BIND_SRV_UPSCALE_MAXIMUM_BIAS_LUT) || defined(FFX_INTERNAL)
FfxFloat32 SampleUpsampleMaximumBias(FfxFloat32x2 uv)
{
    // Stored as a SNORM, so make sure to multiply by 2 to retrieve the actual expected range.
    return FfxFloat32(2.0) * r_upsample_maximum_bias_lut.SampleLevel(s_LinearClamp, abs(uv) * 2.0, 0);
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_REACTIVE_MASKS) || defined(FFX_INTERNAL)
FfxFloat32x2 SampleDilatedReactiveMasks(FfxFloat32x2 fUV)
{
	return r_dilated_reactive_masks.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_REACTIVE_MASKS) || defined(FFX_INTERNAL)
FfxFloat32x2 LoadDilatedReactiveMasks(FFX_PARAMETER_IN FfxUInt32x2 iPxPos)
{
    return r_dilated_reactive_masks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_REACTIVE_MASKS) || defined(FFX_INTERNAL)
void StoreDilatedReactiveMasks(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fDilatedReactiveMasks)
{
    rw_dilated_reactive_masks[iPxPos] = fDilatedReactiveMasks;
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_OPAQUE_ONLY) || defined(FFX_INTERNAL)
FfxFloat32x3 LoadOpaqueOnly(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_opaque_only[iPxPos].xyz;
}
#endif

#if defined(FSR2_BIND_SRV_PREV_PRE_ALPHA_COLOR) || defined(FFX_INTERNAL)
FfxFloat32x3 LoadPrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_pre_alpha[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_PREV_POST_ALPHA_COLOR) || defined(FFX_INTERNAL)
FfxFloat32x3 LoadPrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_post_alpha[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_AUTOREACTIVE) || defined(FFX_INTERNAL)
#if defined(FSR2_BIND_UAV_AUTOCOMPOSITION) || defined(FFX_INTERNAL)
void StoreAutoReactive(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 fReactive)
{
    rw_output_autoreactive[iPxPos] = fReactive.x;

    rw_output_autocomposition[iPxPos] = fReactive.y;
}
#endif
#endif

#if defined(FSR2_BIND_UAV_PREV_PRE_ALPHA_COLOR) || defined(FFX_INTERNAL)
void StorePrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_pre_alpha[iPxPos] = color;

}
#endif

#if defined(FSR2_BIND_UAV_PREV_POST_ALPHA_COLOR) || defined(FFX_INTERNAL)
void StorePrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_post_alpha[iPxPos] = color;
}
#endif

#endif // #if defined(FFX_GPU)
