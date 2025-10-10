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

#pragma once
#include "gpu/fsr3upscaler/ffx_fsr3upscaler_resources.h"

/// An enumeration of all the permutations that can be passed to the FSR3 Upscaler algorithm.
///
/// FSR3 Upscaler features are organized through a set of pre-defined compile
/// permutation options that need to be specified. Which shader blob
/// is returned for pipeline creation will be determined by what combination
/// of shader permutations are enabled.
///
/// @ingroup FSR3Upscaler
typedef enum Fs3UpscalerShaderPermutationOptions
{
    FSR3UPSCALER_SHADER_PERMUTATION_USE_LANCZOS_TYPE       = (1 << 0),  ///< Off means reference, On means LUT
    FSR3UPSCALER_SHADER_PERMUTATION_HDR_COLOR_INPUT        = (1 << 1),  ///< Enables the HDR code path
    FSR3UPSCALER_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS = (1 << 2),  ///< Indicates low resolution motion vectors provided
    FSR3UPSCALER_SHADER_PERMUTATION_JITTER_MOTION_VECTORS  = (1 << 3),  ///< Indicates motion vectors were generated with jitter
    FSR3UPSCALER_SHADER_PERMUTATION_DEPTH_INVERTED         = (1 << 4),  ///< Indicates input resources were generated with inverted depth
    FSR3UPSCALER_SHADER_PERMUTATION_ENABLE_SHARPENING      = (1 << 5),  ///< Enables a supplementary sharpening pass
    FSR3UPSCALER_SHADER_PERMUTATION_FORCE_WAVE64           = (1 << 6),  ///< doesn't map to a define, selects different table
    FSR3UPSCALER_SHADER_PERMUTATION_ALLOW_FP16             = (1 << 7),  ///< Enables fast math computations where possible
} Fs3UpscalerShaderPermutationOptions;

// Constants for FSR3 Upscaler dispatches. Must be kept in sync with cbFSR3Upscaler in ffx_fsr2_callbacks_hlsl.h
typedef struct Fsr3UpscalerConstants {

    int32_t                     renderSize[2];
    int32_t                     previousFrameRenderSize[2];

    int32_t                     upscaleSize[2];
    int32_t                     previousFrameUpscaleSize[2];

    int32_t                     maxRenderSize[2];
    int32_t                     maxUpscaleSize[2];

    float                       deviceToViewDepth[4];

    float                       jitterOffset[2];
    float                       previousFrameJitterOffset[2];

    float                       motionVectorScale[2];
    float                       downscaleFactor[2];

    float                       motionVectorJitterCancellation[2];
    float                       tanHalfFOV;
    float                       jitterPhaseCount;

    float                       deltaTime;
    float                       deltaPreExposure;
    float                       viewSpaceToMetersFactor;
    float                       frameIndex;

    float                       velocityFactor;
    float                       reactivenessScale;
    float                       shadingChangeScale;
    float                       accumulationAddedPerFrame;
    float                       minDisocclusionAccumulation;
} Fsr3UpscalerConstants;

struct FfxFsr3UpscalerContextDescription;
struct FfxDeviceCapabilities;
struct FfxPipelineState;

// FfxFsr3UpscalerContext_Private
// The private implementation of the FSR3 Upscaler context.
typedef struct FfxFsr3UpscalerContext_Private {

    FfxFsr3UpscalerContextDescription   contextDescription;
    FfxUInt32                           effectContextId;
    Fsr3UpscalerConstants               constants;
    FfxDevice                           device;
    FfxDeviceCapabilities               deviceCapabilities;
    FfxPipelineState                    pipelinePrepareInputs;
    FfxPipelineState                    pipelinePrepareReactivity;
    FfxPipelineState                    pipelineShadingChange;
    FfxPipelineState                    pipelineAccumulate;
    FfxPipelineState                    pipelineAccumulateSharpen;
    FfxPipelineState                    pipelineRCAS;
    FfxPipelineState                    pipelineLumaPyramid;
    FfxPipelineState                    pipelineGenerateReactive;
    FfxPipelineState                    pipelineTcrAutogenerate;
    FfxPipelineState                    pipelineShadingChangePyramid;
    FfxPipelineState                    pipelineLumaInstability;
    FfxPipelineState                    pipelineDebugView;
    FfxConstantBuffer                   constantBuffers[FFX_FSR3UPSCALER_CONSTANTBUFFER_COUNT];

    // 2 arrays of resources, as e.g. FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_LOCK_STATUS will use different resources when bound as SRV vs when bound as UAV
    FfxResourceInternal                 srvResources[FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal                 uavResources[FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_COUNT];

    bool                                firstExecution;
    uint32_t                            resourceFrameIndex;
    float                               previousJitterOffset[2];
    float                               preExposure;
    float                               previousFramePreExposure;

} FfxFsr3UpscalerContext_Private;

// declare fsr3UpscalerCreate so it can be used from fsr3
FFX_API FfxErrorCode fsr3UpscalerCreate(FfxFsr3UpscalerContext_Private* context, const FfxFsr3UpscalerContextDescription* contextDescription);
