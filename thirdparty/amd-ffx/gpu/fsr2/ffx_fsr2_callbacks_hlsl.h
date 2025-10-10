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

#include "ffx_fsr2_resources.h"

#if defined(FFX_GPU)
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic push
#pragma dxc diagnostic ignored "-Wambig-lit-shift"
#endif //__hlsl_dx_compiler
#include "../ffx_core.h"
#ifdef __hlsl_dx_compiler
#pragma dxc diagnostic pop
#endif //__hlsl_dx_compiler

#ifndef FFX_PREFER_WAVE64
#define FFX_PREFER_WAVE64
#endif // #ifndef FFX_PREFER_WAVE64

#pragma warning(disable: 3205)  // conversion from larger type to smaller

#define DECLARE_SRV_REGISTER(regIndex)  t##regIndex
#define DECLARE_UAV_REGISTER(regIndex)  u##regIndex
#define DECLARE_CB_REGISTER(regIndex)   b##regIndex
#define FFX_FSR2_DECLARE_SRV(regIndex)  register(DECLARE_SRV_REGISTER(regIndex))
#define FFX_FSR2_DECLARE_UAV(regIndex)  register(DECLARE_UAV_REGISTER(regIndex))
#define FFX_FSR2_DECLARE_CB(regIndex)   register(DECLARE_CB_REGISTER(regIndex))

#if defined(FSR2_BIND_CB_FSR2)
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
        FfxFloat32    fPadding;
    };

#define FFX_FSR2_CONSTANT_BUFFER_1_SIZE 32

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
#endif // #if defined(FSR2_BIND_CB_FSR2)

#define FFX_FSR2_ROOTSIG_STRINGIFY(p) FFX_FSR2_ROOTSIG_STR(p)
#define FFX_FSR2_ROOTSIG_STR(p) #p
#define FFX_FSR2_ROOTSIG [RootSignature("DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
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

#define FFX_FSR2_CONSTANT_BUFFER_2_SIZE 6           // Number of 32-bit values. This must be kept in sync with max( cbRCAS , cbSPD) size.

#define FFX_FSR2_CB2_ROOTSIG [RootSignature("DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
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

#define FFX_FSR2_CONSTANT_BUFFER_3_SIZE 4           // Number of 32-bit values. This must be kept in sync with cbGenerateReactive size.

#define FFX_FSR2_REACTIVE_ROOTSIG [RootSignature("DescriptorTable(UAV(u0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
                                    "DescriptorTable(SRV(t0, numDescriptors = " FFX_FSR2_ROOTSIG_STRINGIFY(FFX_FSR2_RESOURCE_IDENTIFIER_COUNT) ")), " \
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

#if defined(FFX_FSR2_EMBED_ROOTSIG)
#define FFX_FSR2_EMBED_ROOTSIG_CONTENT FFX_FSR2_ROOTSIG
#define FFX_FSR2_EMBED_CB2_ROOTSIG_CONTENT FFX_FSR2_CB2_ROOTSIG
#define FFX_FSR2_EMBED_ROOTSIG_REACTIVE_CONTENT FFX_FSR2_REACTIVE_ROOTSIG
#else
#define FFX_FSR2_EMBED_ROOTSIG_CONTENT
#define FFX_FSR2_EMBED_CB2_ROOTSIG_CONTENT
#define FFX_FSR2_EMBED_ROOTSIG_REACTIVE_CONTENT
#endif // #if FFX_FSR2_EMBED_ROOTSIG

#if defined(FSR2_BIND_CB_AUTOREACTIVE)
cbuffer cbGenerateReactive : FFX_FSR2_DECLARE_CB(FSR2_BIND_CB_AUTOREACTIVE)
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
#endif // #if defined(FSR2_BIND_CB_AUTOREACTIVE)

#if defined(FSR2_BIND_CB_RCAS)
cbuffer cbRCAS : FFX_FSR2_DECLARE_CB(FSR2_BIND_CB_RCAS)
{
    FfxUInt32x4 rcasConfig;
};

FfxUInt32x4 RCASConfig()
{
    return rcasConfig;
}
#endif // #if defined(FSR2_BIND_CB_RCAS)


#if defined(FSR2_BIND_CB_REACTIVE)
cbuffer cbGenerateReactive : FFX_FSR2_DECLARE_CB(FSR2_BIND_CB_REACTIVE)
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
#endif // #if defined(FSR2_BIND_CB_REACTIVE)

#if defined(FSR2_BIND_CB_SPD)
cbuffer cbSPD : FFX_FSR2_DECLARE_CB(FSR2_BIND_CB_SPD) {

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
#endif // #if defined(FSR2_BIND_CB_SPD)

SamplerState s_PointClamp : register(s0);
SamplerState s_LinearClamp : register(s1);

    // SRVs
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

#if defined(FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS)
FfxFloat32 LoadMipLuma(FfxUInt32x2 iPxPos, FfxUInt32 mipLevel)
{
    return r_imgMips.mips[mipLevel][iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_SCENE_LUMINANCE_MIPS)
FfxFloat32 SampleMipLuma(FfxFloat32x2 fUV, FfxUInt32 mipLevel)
{
    return r_imgMips.SampleLevel(s_LinearClamp, fUV, mipLevel);
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_DEPTH)
FfxFloat32 LoadInputDepth(FfxUInt32x2 iPxPos)
{
    return r_input_depth[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_DEPTH)
FfxFloat32 SampleInputDepth(FfxFloat32x2 fUV)
{
    return r_input_depth.SampleLevel(s_LinearClamp, fUV, 0).x;
}
#endif

#if defined(FSR2_BIND_SRV_REACTIVE_MASK)
FfxFloat32 LoadReactiveMask(FfxUInt32x2 iPxPos)
{
    return r_reactive_mask[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_TRANSPARENCY_AND_COMPOSITION_MASK)
FfxFloat32 LoadTransparencyAndCompositionMask(FfxUInt32x2 iPxPos)
{
    return r_transparency_and_composition_mask[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_COLOR)
FfxFloat32x3 LoadInputColor(FfxUInt32x2 iPxPos)
{
    return r_input_color_jittered[iPxPos].rgb;
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_COLOR)
FfxFloat32x3 SampleInputColor(FfxFloat32x2 fUV)
{
    return r_input_color_jittered.SampleLevel(s_LinearClamp, fUV, 0).rgb;
}
#endif

#if defined(FSR2_BIND_SRV_PREPARED_INPUT_COLOR)
FfxFloat32x3 LoadPreparedInputColor(FfxUInt32x2 iPxPos)
{
    return r_prepared_input_color[iPxPos].xyz;
}

#if FFX_HALF && defined(__XBOX_SCARLETT) && defined(__XBATG_EXTRA_16_BIT_OPTIMISATION) && (__XBATG_EXTRA_16_BIT_OPTIMISATION == 1)
FFX_MIN16_F3 LoadPreparedInputColorHalf(FfxUInt32x2 iPxPos)
{
    return FFX_MIN16_F3(r_prepared_input_color[iPxPos].xyz);
}
#endif

#endif

#if defined(FSR2_BIND_SRV_INPUT_MOTION_VECTORS)
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

#if defined(FSR2_BIND_SRV_INTERNAL_UPSCALED)
FfxFloat32x4 LoadHistory(FfxUInt32x2 iPxHistory)
{
    return r_internal_upscaled_color[iPxHistory];
}
#endif

#if defined(FSR2_BIND_UAV_LUMA_HISTORY)
void StoreLumaHistory(FfxUInt32x2 iPxPos, FfxFloat32x4 fLumaHistory)
{
    rw_luma_history[iPxPos] = fLumaHistory;
}
#endif

#if defined(FSR2_BIND_SRV_LUMA_HISTORY)
FfxFloat32x4 SampleLumaHistory(FfxFloat32x2 fUV)
{
    return r_luma_history.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

FfxFloat32x4 LoadRCAS_Input(FfxInt32x2 iPxPos)
{
#if defined(FSR2_BIND_SRV_RCAS_INPUT) 
    return r_rcas_input[iPxPos];
#else
    return 0.0;
#endif
}

#if defined(FSR2_BIND_UAV_INTERNAL_UPSCALED)
void StoreReprojectedHistory(FfxUInt32x2 iPxHistory, FfxFloat32x4 fHistory)
{
    rw_internal_upscaled_color[iPxHistory] = fHistory;
}
#endif

#if defined(FSR2_BIND_UAV_INTERNAL_UPSCALED)
void StoreInternalColorAndWeight(FfxUInt32x2 iPxPos, FfxFloat32x4 fColorAndWeight)
{
    rw_internal_upscaled_color[iPxPos] = fColorAndWeight;
}
#endif

#if defined(FSR2_BIND_UAV_UPSCALED_OUTPUT)
void StoreUpscaledOutput(FfxUInt32x2 iPxPos, FfxFloat32x3 fColor)
{
    rw_upscaled_output[iPxPos] = FfxFloat32x4(fColor, 1.f);
}
#endif

//LOCK_LIFETIME_REMAINING == 0
//Should make LockInitialLifetime() return a const 1.0f later
#if defined(FSR2_BIND_SRV_LOCK_STATUS)
FfxFloat32x2 LoadLockStatus(FfxUInt32x2 iPxPos)
{
    return r_lock_status[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_LOCK_STATUS)
void StoreLockStatus(FfxUInt32x2 iPxPos, FfxFloat32x2 fLockStatus)
{
    rw_lock_status[iPxPos] = fLockStatus;
}
#endif

#if defined(FSR2_BIND_SRV_LOCK_INPUT_LUMA)
FfxFloat32 LoadLockInputLuma(FfxUInt32x2 iPxPos)
{
    return r_lock_input_luma[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_LOCK_INPUT_LUMA)
void StoreLockInputLuma(FfxUInt32x2 iPxPos, FfxFloat32 fLuma)
{
    rw_lock_input_luma[iPxPos] = fLuma;
}
#endif

#if defined(FSR2_BIND_SRV_NEW_LOCKS)
FfxFloat32 LoadNewLocks(FfxUInt32x2 iPxPos)
{
    return r_new_locks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_NEW_LOCKS)
FfxFloat32 LoadRwNewLocks(FfxUInt32x2 iPxPos)
{
    return rw_new_locks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_NEW_LOCKS)
void StoreNewLocks(FfxUInt32x2 iPxPos, FfxFloat32 newLock)
{
    rw_new_locks[iPxPos] = newLock;
}
#endif

#if defined(FSR2_BIND_UAV_PREPARED_INPUT_COLOR)
void StorePreparedInputColor(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x4 fTonemapped)
{
    rw_prepared_input_color[iPxPos] = fTonemapped;
}
#endif

#if defined(FSR2_BIND_SRV_PREPARED_INPUT_COLOR)
FfxFloat32 SampleDepthClip(FfxFloat32x2 fUV)
{
    return r_prepared_input_color.SampleLevel(s_LinearClamp, fUV, 0).w;
}
#endif

#if defined(FSR2_BIND_SRV_LOCK_STATUS)
FfxFloat32x2 SampleLockStatus(FfxFloat32x2 fUV)
{
    FfxFloat32x2 fLockStatus = r_lock_status.SampleLevel(s_LinearClamp, fUV, 0);
    return fLockStatus;
}
#endif

#if defined(FSR2_BIND_SRV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
FfxFloat32 LoadReconstructedPrevDepth(FfxUInt32x2 iPxPos)
{
    return asfloat(r_reconstructed_previous_nearest_depth[iPxPos]);
}
#endif

#if defined(FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
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

#if defined(FSR2_BIND_UAV_RECONSTRUCTED_PREV_NEAREST_DEPTH)
void SetReconstructedDepth(FfxUInt32x2 iPxSample, const FfxUInt32 uValue)
{
    rw_reconstructed_previous_nearest_depth[iPxSample] = uValue;
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_DEPTH)
void StoreDilatedDepth(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32 fDepth)
{
    rw_dilatedDepth[iPxPos] = fDepth;
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_MOTION_VECTORS)
void StoreDilatedMotionVector(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fMotionVector)
{
    rw_dilated_motion_vectors[iPxPos] = fMotionVector;
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_MOTION_VECTORS)
FfxFloat32x2 LoadDilatedMotionVector(FfxUInt32x2 iPxInput)
{
    return r_dilated_motion_vectors[iPxInput].xy;
}
#endif

#if defined(FSR2_BIND_SRV_PREVIOUS_DILATED_MOTION_VECTORS)
FfxFloat32x2 LoadPreviousDilatedMotionVector(FfxUInt32x2 iPxInput)
{
    return r_previous_dilated_motion_vectors[iPxInput].xy;
}

FfxFloat32x2 SamplePreviousDilatedMotionVector(FfxFloat32x2 uv)
{
    return r_previous_dilated_motion_vectors.SampleLevel(s_LinearClamp, uv, 0).xy;
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_DEPTH)
FfxFloat32 LoadDilatedDepth(FfxUInt32x2 iPxInput)
{
    return r_dilatedDepth[iPxInput];
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_EXPOSURE)
FfxFloat32 Exposure()
{
    FfxFloat32 exposure = r_input_exposure[FfxUInt32x2(0, 0)].x;

    if (exposure == 0.0f) {
        exposure = 1.0f;
    }

    return exposure;
}
#endif

#if defined(FSR2_BIND_SRV_AUTO_EXPOSURE)
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
#if defined(FSR2_BIND_SRV_LANCZOS_LUT)
    return r_lanczos_lut.SampleLevel(s_LinearClamp, FfxFloat32x2(x / 2, 0.5f), 0);
#else
    return 0.f;
#endif
}

#if FFX_HALF && defined(__XBOX_SCARLETT) && defined(__XBATG_EXTRA_16_BIT_OPTIMISATION) && (__XBATG_EXTRA_16_BIT_OPTIMISATION == 1)

FFX_MIN16_F SampleLanczos2Weight_NoValu(FFX_MIN16_F x)
{
#if defined(FSR2_BIND_SRV_LANCZOS_LUT)
    return FFX_MIN16_F(r_lanczos_lut.SampleLevel(s_LinearClamp, __XB_AsHalf(__XB_V_PACK_B32_F16(x, 0.5)), 0));
#else
    return 0.0;
#endif
}

FFX_MIN16_F SampleLanczos2Weight_NoValuNoA16(FfxFloat32 x)
{
#if defined(FSR2_BIND_SRV_LANCZOS_LUT)
    return FFX_MIN16_F(r_lanczos_lut.SampleLevel(s_LinearClamp, FfxFloat32x2(x, 0.5), 0));
#else
    return 0.0;
#endif
}
#endif

#if defined(FSR2_BIND_SRV_UPSCALE_MAXIMUM_BIAS_LUT)
FfxFloat32 SampleUpsampleMaximumBias(FfxFloat32x2 uv)
{
    // Stored as a SNORM, so make sure to multiply by 2 to retrieve the actual expected range.
    return FfxFloat32(2.0) * r_upsample_maximum_bias_lut.SampleLevel(s_LinearClamp, abs(uv) * 2.0, 0);
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_REACTIVE_MASKS)
FfxFloat32x2 SampleDilatedReactiveMasks(FfxFloat32x2 fUV)
{
	return r_dilated_reactive_masks.SampleLevel(s_LinearClamp, fUV, 0);
}
#endif

#if defined(FSR2_BIND_SRV_DILATED_REACTIVE_MASKS)
FfxFloat32x2 LoadDilatedReactiveMasks(FFX_PARAMETER_IN FfxUInt32x2 iPxPos)
{
    return r_dilated_reactive_masks[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_DILATED_REACTIVE_MASKS)
void StoreDilatedReactiveMasks(FFX_PARAMETER_IN FfxUInt32x2 iPxPos, FFX_PARAMETER_IN FfxFloat32x2 fDilatedReactiveMasks)
{
    rw_dilated_reactive_masks[iPxPos] = fDilatedReactiveMasks;
}
#endif

#if defined(FSR2_BIND_SRV_INPUT_OPAQUE_ONLY)
FfxFloat32x3 LoadOpaqueOnly(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_opaque_only[iPxPos].xyz;
}
#endif

#if defined(FSR2_BIND_SRV_PREV_PRE_ALPHA_COLOR)
FfxFloat32x3 LoadPrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_pre_alpha[iPxPos];
}
#endif

#if defined(FSR2_BIND_SRV_PREV_POST_ALPHA_COLOR)
FfxFloat32x3 LoadPrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos)
{
    return r_input_prev_color_post_alpha[iPxPos];
}
#endif

#if defined(FSR2_BIND_UAV_AUTOREACTIVE)
#if defined(FSR2_BIND_UAV_AUTOCOMPOSITION)
void StoreAutoReactive(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F2 fReactive)
{
    rw_output_autoreactive[iPxPos] = fReactive.x;

    rw_output_autocomposition[iPxPos] = fReactive.y;
}
#endif
#endif

#if defined(FSR2_BIND_UAV_PREV_PRE_ALPHA_COLOR)
void StorePrevPreAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_pre_alpha[iPxPos] = color;

}
#endif

#if defined(FSR2_BIND_UAV_PREV_POST_ALPHA_COLOR)
void StorePrevPostAlpha(FFX_PARAMETER_IN FFX_MIN16_I2 iPxPos, FFX_PARAMETER_IN FFX_MIN16_F3 color)
{
    rw_output_prev_color_post_alpha[iPxPos] = color;
}
#endif

FfxFloat32x2 SPD_LoadExposureBuffer()
{
#if defined FSR2_BIND_UAV_AUTO_EXPOSURE
    return rw_auto_exposure[FfxInt32x2(0, 0)];
#else
    return FfxFloat32x2(0.f, 0.f);
#endif // #if defined FSR2_BIND_UAV_AUTO_EXPOSURE
}

void SPD_SetExposureBuffer(FfxFloat32x2 value)
{
#if defined FSR2_BIND_UAV_AUTO_EXPOSURE
    rw_auto_exposure[FfxInt32x2(0, 0)] = value;
#endif // #if defined FSR2_BIND_UAV_AUTO_EXPOSURE
}

FfxFloat32x4 SPD_LoadMipmap5(FfxInt32x2 iPxPos)
{
#if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
    return FfxFloat32x4(rw_img_mip_5[iPxPos], 0, 0, 0);
#else
    return FfxFloat32x4(0.f, 0.f, 0.f, 0.f);
#endif // #if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
}

void SPD_SetMipmap(FfxInt32x2 iPxPos, FfxUInt32 slice, FfxFloat32 value)
{
    switch (slice)
    {
    case FFX_FSR2_SHADING_CHANGE_MIP_LEVEL:
#if defined FSR2_BIND_UAV_EXPOSURE_MIP_LUMA_CHANGE
        rw_img_mip_shading_change[iPxPos] = value;
#endif // #if defined FSR2_BIND_UAV_EXPOSURE_MIP_LUMA_CHANGE
        break;
    case 5:
#if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
        rw_img_mip_5[iPxPos] = value;
#endif // #if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
        break;
    default:

        // avoid flattened side effect
#if defined(FSR2_BIND_UAV_EXPOSURE_MIP_LUMA_CHANGE)
        rw_img_mip_shading_change[iPxPos] = rw_img_mip_shading_change[iPxPos];
#elif defined(FSR2_BIND_UAV_EXPOSURE_MIP_5)
        rw_img_mip_5[iPxPos] = rw_img_mip_5[iPxPos];
#endif // #if defined FSR2_BIND_UAV_EXPOSURE_MIP_5
        break;
    }
}

void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter)
{
#if defined FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC
    InterlockedAdd(rw_spd_global_atomic[FfxInt32x2(0, 0)], 1, spdCounter);
#endif // #if defined FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC
}

void SPD_ResetAtomicCounter()
{
#if defined FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC
    rw_spd_global_atomic[FfxInt32x2(0, 0)] = 0;
#endif // #if defined FSR2_BIND_UAV_SPD_GLOBAL_ATOMIC
}

#endif // #if defined(FFX_GPU)
