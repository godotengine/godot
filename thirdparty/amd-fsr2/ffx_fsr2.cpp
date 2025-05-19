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

#include <algorithm>    // for max used inside SPD CPU code.
#include <cmath>        // for fabs, abs, sinf, sqrt, etc.
#include <string.h>     // for memset
#include <cfloat>       // for FLT_EPSILON
#include "ffx_fsr2.h"
#define FFX_CPU
#include "shaders/ffx_core.h"
#include "shaders/ffx_fsr1.h"
#include "shaders/ffx_spd.h"
#include "shaders/ffx_fsr2_callbacks_hlsl.h"

#include "ffx_fsr2_maximum_bias.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof(array[0]))
#endif

#ifndef _MSC_VER
#include <wchar.h>
#define wcscpy_s wcscpy
#endif

// max queued frames for descriptor management
static const uint32_t FSR2_MAX_QUEUED_FRAMES = 16;

#include "ffx_fsr2_private.h"

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t    index;
    wchar_t     name[64];
}ResourceBinding;

static const ResourceBinding srvResourceBindingTable[] =
{
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR,                              L"r_input_color_jittered"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY,                        L"r_input_opaque_only"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS,                     L"r_input_motion_vectors"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_DEPTH,                              L"r_input_depth" },
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE,                           L"r_input_exposure"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE,                            L"r_auto_exposure"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK,                      L"r_reactive_mask"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK,  L"r_transparency_and_composition_mask"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH,     L"r_reconstructed_previous_nearest_depth"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS,                   L"r_dilated_motion_vectors"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREVIOUS_DILATED_MOTION_VECTORS,          L"r_previous_dilated_motion_vectors"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH,                            L"r_dilatedDepth"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,                  L"r_internal_upscaled_color"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS,                              L"r_lock_status"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR,                     L"r_prepared_input_color"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY,                             L"r_luma_history" },
    {FFX_FSR2_RESOURCE_IDENTIFIER_RCAS_INPUT,                               L"r_rcas_input"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LANCZOS_LUT,                              L"r_lanczos_lut"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE,                          L"r_imgMips"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_SHADING_CHANGE,    L"r_img_mip_shading_change"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_5,                 L"r_img_mip_5"},
    {FFX_FSR2_RESOURCE_IDENTITIER_UPSAMPLE_MAXIMUM_BIAS_LUT,                L"r_upsample_maximum_bias_lut"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS,                   L"r_dilated_reactive_masks"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS,                                L"r_new_locks"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA,                          L"r_lock_input_luma"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR,                     L"r_input_prev_color_pre_alpha"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR,                    L"r_input_prev_color_post_alpha"},
};

static const ResourceBinding uavResourceBindingTable[] =
{
    {FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH,    L"rw_reconstructed_previous_nearest_depth"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS,                  L"rw_dilated_motion_vectors"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH,                           L"rw_dilatedDepth"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,                 L"rw_internal_upscaled_color"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS,                             L"rw_lock_status"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR,                    L"rw_prepared_input_color"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY,                            L"rw_luma_history"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT,                         L"rw_upscaled_output"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_SHADING_CHANGE,   L"rw_img_mip_shading_change"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_5,                L"rw_img_mip_5"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS,                  L"rw_dilated_reactive_masks"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE,                           L"rw_auto_exposure"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT,                        L"rw_spd_global_atomic"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS,                               L"rw_new_locks"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA,                         L"rw_lock_input_luma"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE,                            L"rw_output_autoreactive"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION,                         L"rw_output_autocomposition"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR,                    L"rw_output_prev_color_pre_alpha"},
    {FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR,                   L"rw_output_prev_color_post_alpha"},
};

static const ResourceBinding cbResourceBindingTable[] =
{
    {FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_FSR2,           L"cbFSR2"},
    {FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_SPD,            L"cbSPD"},
    {FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_RCAS,           L"cbRCAS"},
    {FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_GENREACTIVE,    L"cbGenerateReactive"},
};

// Broad structure of the root signature.
typedef enum Fsr2RootSignatureLayout {

    FSR2_ROOT_SIGNATURE_LAYOUT_UAVS,
    FSR2_ROOT_SIGNATURE_LAYOUT_SRVS,
    FSR2_ROOT_SIGNATURE_LAYOUT_CONSTANTS,
    FSR2_ROOT_SIGNATURE_LAYOUT_CONSTANTS_REGISTER_1,
    FSR2_ROOT_SIGNATURE_LAYOUT_PARAMETER_COUNT
} Fsr2RootSignatureLayout;

typedef struct Fsr2RcasConstants {

    uint32_t                    rcasConfig[4];
} FfxRcasConstants;

typedef struct Fsr2SpdConstants {

    uint32_t                    mips;
    uint32_t                    numworkGroups;
    uint32_t                    workGroupOffset[2];
    uint32_t                    renderSize[2];
} Fsr2SpdConstants;

typedef struct Fsr2GenerateReactiveConstants
{
    float       scale;
    float       threshold;
    float       binaryValue;
    uint32_t    flags;

} Fsr2GenerateReactiveConstants;

typedef struct Fsr2GenerateReactiveConstants2
{
    float       autoTcThreshold;
    float       autoTcScale;
    float       autoReactiveScale;
    float       autoReactiveMax;

} Fsr2GenerateReactiveConstants2;

typedef union Fsr2SecondaryUnion {

    Fsr2RcasConstants               rcas;
    Fsr2SpdConstants                spd;
    Fsr2GenerateReactiveConstants2  autogenReactive;
} Fsr2SecondaryUnion;

typedef struct Fsr2ResourceDescription {

    uint32_t                    id;
    const wchar_t*              name;
    FfxResourceUsage            usage;
    FfxSurfaceFormat            format;
    uint32_t                    width;
    uint32_t                    height;
    uint32_t                    mipCount;
    FfxResourceFlags            flags;
    uint32_t                    initDataSize;
    void*                       initData;
} Fsr2ResourceDescription;

FfxConstantBuffer globalFsr2ConstantBuffers[4] = {
    { sizeof(Fsr2Constants) / sizeof(uint32_t) },
    { sizeof(Fsr2SpdConstants) / sizeof(uint32_t) },
    { sizeof(Fsr2RcasConstants) / sizeof(uint32_t) },
    { sizeof(Fsr2GenerateReactiveConstants) / sizeof(uint32_t) }
};

// Lanczos
static float lanczos2(float value)
{
    return abs(value) < FFX_EPSILON ? 1.f : (sinf(FFX_PI * value) / (FFX_PI * value)) * (sinf(0.5f * FFX_PI * value) / (0.5f * FFX_PI * value));
}

// Calculate halton number for index and base.
static float halton(int32_t index, int32_t base)
{
    float f = 1.0f, result = 0.0f;

    for (int32_t currentIndex = index; currentIndex > 0;) {

        f /= (float)base;
        result = result + f * (float)(currentIndex % base);
        currentIndex = (uint32_t)(floorf((float)(currentIndex) / (float)(base)));
    }

    return result;
}

static void fsr2DebugCheckDispatch(FfxFsr2Context_Private* context, const FfxFsr2DispatchDescription* params)
{
    if (params->commandList == nullptr)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"commandList is null");
    }

    if (params->color.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"color resource is null");
    }

    if (params->depth.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"depth resource is null");
    }

    if (params->motionVectors.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"motionVectors resource is null");
    }

    if (params->exposure.resource != nullptr)
    {
        if ((context->contextDescription.flags & FFX_FSR2_ENABLE_AUTO_EXPOSURE) == FFX_FSR2_ENABLE_AUTO_EXPOSURE)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"exposure resource provided, however auto exposure flag is present");
        }
    }

    if (params->output.resource == nullptr)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"output resource is null");
    }

    if (fabs(params->jitterOffset.x) > 1.0f || fabs(params->jitterOffset.y) > 1.0f)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"jitterOffset contains value outside of expected range [-1.0, 1.0]");
    }

    if ((params->motionVectorScale.x > (float)context->contextDescription.maxRenderSize.width) ||
        (params->motionVectorScale.y > (float)context->contextDescription.maxRenderSize.height))
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"motionVectorScale contains scale value greater than maxRenderSize");
    }
    if ((params->motionVectorScale.x == 0.0f) ||
        (params->motionVectorScale.y == 0.0f))
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"motionVectorScale contains zero scale value");
    }

    if ((params->renderSize.width > context->contextDescription.maxRenderSize.width) ||
        (params->renderSize.height > context->contextDescription.maxRenderSize.height))
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"renderSize is greater than context maxRenderSize");
    }
    if ((params->renderSize.width == 0) ||
        (params->renderSize.height == 0))
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"renderSize contains zero dimension");
    }

    if (params->sharpness < 0.0f || params->sharpness > 1.0f)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"sharpness contains value outside of expected range [0.0, 1.0]");
    }

    if (params->frameTimeDelta < 1.0f)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING, L"frameTimeDelta is less than 1.0f - this value should be milliseconds (~16.6f for 60fps)");
    }

    if (params->preExposure == 0.0f)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"preExposure provided as 0.0f which is invalid");
    }

    bool infiniteDepth = (context->contextDescription.flags & FFX_FSR2_ENABLE_DEPTH_INFINITE) == FFX_FSR2_ENABLE_DEPTH_INFINITE;
    bool inverseDepth = (context->contextDescription.flags & FFX_FSR2_ENABLE_DEPTH_INVERTED) == FFX_FSR2_ENABLE_DEPTH_INVERTED;

    if (inverseDepth)
    {
        if (params->cameraNear < params->cameraFar)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INVERTED flag is present yet cameraNear is less than cameraFar");
        }
        if (infiniteDepth)
        {
            if (params->cameraNear != FLT_MAX)
            {
                context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                    L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, yet cameraNear != FLT_MAX");
            }
        }
        if (params->cameraFar < 0.075f)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, cameraFar value is very low which may result in depth separation artefacting");
        }
    }
    else
    {
        if (params->cameraNear > params->cameraFar)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"cameraNear is greater than cameraFar in non-inverted-depth context");
        }
        if (infiniteDepth)
        {
            if (params->cameraFar != FLT_MAX)
            {
                context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                    L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, yet cameraFar != FLT_MAX");
            }
        }
        if (params->cameraNear < 0.075f)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, cameraNear value is very low which may result in depth separation artefacting");
        }
    }

    if (params->cameraFovAngleVertical <= 0.0f)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"cameraFovAngleVertical is 0.0f - this value should be > 0.0f");
    }
    if (params->cameraFovAngleVertical > FFX_PI)
    {
        context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_ERROR, L"cameraFovAngleVertical is greater than 180 degrees/PI");
    }
}

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvResourceBindings[srvIndex].name))
                break;
        }
        if (mapIndex == _countof(srvResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvResourceBindings[srvIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavResourceBindings[uavIndex].name))
                break;
        }
        if (mapIndex == _countof(uavResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavResourceBindings[uavIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    for (uint32_t cbIndex = 0; cbIndex < inoutPipeline->constCount; ++cbIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(cbResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(cbResourceBindingTable[mapIndex].name, inoutPipeline->cbResourceBindings[cbIndex].name))
                break;
        }
        if (mapIndex == _countof(cbResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->cbResourceBindings[cbIndex].resourceIdentifier = cbResourceBindingTable[mapIndex].index;
    }

    return FFX_OK;
}


static FfxErrorCode createPipelineStates(FfxFsr2Context_Private* context)
{
    FFX_ASSERT(context);

    const size_t samplerCount = 2;
    FfxFilterType samplers[samplerCount];
    samplers[0] = FFX_FILTER_TYPE_POINT;
    samplers[1] = FFX_FILTER_TYPE_LINEAR;

    const size_t rootConstantCount = 2;
    uint32_t rootConstants[rootConstantCount];
    rootConstants[0] = sizeof(Fsr2Constants) / sizeof(uint32_t);
    rootConstants[1] = sizeof(Fsr2SecondaryUnion) / sizeof(uint32_t);

    FfxPipelineDescription pipelineDescription;
    pipelineDescription.contextFlags = context->contextDescription.flags;
    pipelineDescription.samplerCount = samplerCount;
    pipelineDescription.samplers = samplers;
    pipelineDescription.rootConstantBufferCount = rootConstantCount;
    pipelineDescription.rootConstantBufferSizes = rootConstants;

    // New interface: will handle RootSignature in backend
    // set up pipeline descriptor (basically RootSignature and binding)
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID, &pipelineDescription, &context->pipelineComputeLuminancePyramid));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_RCAS, &pipelineDescription, &context->pipelineRCAS));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_GENERATE_REACTIVE, &pipelineDescription, &context->pipelineGenerateReactive));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_TCR_AUTOGENERATE, &pipelineDescription, &context->pipelineTcrAutogenerate));

    pipelineDescription.rootConstantBufferCount = 1;
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_DEPTH_CLIP, &pipelineDescription, &context->pipelineDepthClip));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH, &pipelineDescription, &context->pipelineReconstructPreviousDepth));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_LOCK, &pipelineDescription, &context->pipelineLock));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_ACCUMULATE, &pipelineDescription, &context->pipelineAccumulate));
    FFX_VALIDATE(context->contextDescription.callbacks.fpCreatePipeline(&context->contextDescription.callbacks, FFX_FSR2_PASS_ACCUMULATE_SHARPEN, &pipelineDescription, &context->pipelineAccumulateSharpen));
    
    // for each pipeline: re-route/fix-up IDs based on names
    patchResourceBindings(&context->pipelineDepthClip);
    patchResourceBindings(&context->pipelineReconstructPreviousDepth);
    patchResourceBindings(&context->pipelineLock);
    patchResourceBindings(&context->pipelineAccumulate);
    patchResourceBindings(&context->pipelineComputeLuminancePyramid);
    patchResourceBindings(&context->pipelineAccumulateSharpen);
    patchResourceBindings(&context->pipelineRCAS);
    patchResourceBindings(&context->pipelineGenerateReactive);
    patchResourceBindings(&context->pipelineTcrAutogenerate);

    return FFX_OK;
}

static FfxErrorCode generateReactiveMaskInternal(FfxFsr2Context_Private* contextPrivate, const FfxFsr2DispatchDescription* params);

static FfxErrorCode fsr2Create(FfxFsr2Context_Private* context, const FfxFsr2ContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxFsr2Context_Private));
    context->device = contextDescription->device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxFsr2ContextDescription));

    if ((context->contextDescription.flags & FFX_FSR2_ENABLE_DEBUG_CHECKING) == FFX_FSR2_ENABLE_DEBUG_CHECKING)
    {
        if (context->contextDescription.fpMessage == nullptr)
        {
            FFX_ASSERT(context->contextDescription.fpMessage != nullptr);
            // remove the debug checking flag - we have no message function
            context->contextDescription.flags &= ~FFX_FSR2_ENABLE_DEBUG_CHECKING;
        }
    }

    // Create the device.
    FfxErrorCode errorCode = context->contextDescription.callbacks.fpCreateBackendContext(&context->contextDescription.callbacks, context->device);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // call out for device caps.
    errorCode = context->contextDescription.callbacks.fpGetDeviceCapabilities(&context->contextDescription.callbacks, &context->deviceCapabilities, context->device);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // set defaults
    context->firstExecution = true;
    context->resourceFrameIndex = 0;

    context->constants.displaySize[0] = contextDescription->displaySize.width;
    context->constants.displaySize[1] = contextDescription->displaySize.height;

    // generate the data for the LUT.
    const uint32_t lanczos2LutWidth = 128;
    int16_t lanczos2Weights[lanczos2LutWidth] = { };

    for (uint32_t currentLanczosWidthIndex = 0; currentLanczosWidthIndex < lanczos2LutWidth; currentLanczosWidthIndex++) {

        const float x = 2.0f * currentLanczosWidthIndex / float(lanczos2LutWidth - 1);
        const float y = lanczos2(x);
        lanczos2Weights[currentLanczosWidthIndex] = int16_t(roundf(y * 32767.0f));
    }

    // upload path only supports R16_SNORM, let's go and convert
    int16_t maximumBias[FFX_FSR2_MAXIMUM_BIAS_TEXTURE_WIDTH * FFX_FSR2_MAXIMUM_BIAS_TEXTURE_HEIGHT];
    for (uint32_t i = 0; i < FFX_FSR2_MAXIMUM_BIAS_TEXTURE_WIDTH * FFX_FSR2_MAXIMUM_BIAS_TEXTURE_HEIGHT; ++i) {

        maximumBias[i] = int16_t(roundf(ffxFsr2MaximumBias[i] / 2.0f * 32767.0f));
    }

    uint8_t defaultReactiveMaskData = 0U;
    uint32_t atomicInitData = 0U;
    float defaultExposure[] = { 0.0f, 0.0f };
    const FfxResourceType texture1dResourceType = (context->contextDescription.flags & FFX_FSR2_ENABLE_TEXTURE1D_USAGE) ? FFX_RESOURCE_TYPE_TEXTURE1D : FFX_RESOURCE_TYPE_TEXTURE2D;

    // declare internal resources needed
    const Fsr2ResourceDescription internalSurfaceDesc[] = {

        {   FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR, L"FSR2_PreparedInputColor", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH, L"FSR2_ReconstructedPrevNearestDepth", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_1, L"FSR2_InternalDilatedVelocity1", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_2, L"FSR2_InternalDilatedVelocity2", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_DEPTH, L"FSR2_DilatedDepth", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R32_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },
            
        {   FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_1, L"FSR2_LockStatus1", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16_FLOAT, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_2, L"FSR2_LockStatus2", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16_FLOAT, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_INPUT_LUMA, L"FSR2_LockInputLuma", (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_NEW_LOCKS, L"FSR2_NewLocks", (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R8_UNORM, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1, L"FSR2_InternalUpscaled1", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2, L"FSR2_InternalUpscaled2", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE, L"FSR2_ExposureMips", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16_FLOAT, contextDescription->maxRenderSize.width / 2, contextDescription->maxRenderSize.height / 2, 0, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_1, L"FSR2_LumaHistory1", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_2, L"FSR2_LumaHistory2", (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R8G8B8A8_UNORM, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_SPD_ATOMIC_COUNT, L"FSR2_SpdAtomicCounter", (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV),
            FFX_SURFACE_FORMAT_R32_UINT, 1, 1, 1, FFX_RESOURCE_FLAGS_ALIASABLE, sizeof(atomicInitData), &atomicInitData },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_REACTIVE_MASKS, L"FSR2_DilatedReactiveMasks", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8G8_UNORM, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_ALIASABLE },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_LANCZOS_LUT, L"FSR2_LanczosLutData", FFX_RESOURCE_USAGE_READ_ONLY,
            FFX_SURFACE_FORMAT_R16_SNORM, lanczos2LutWidth, 1, 1, FFX_RESOURCE_FLAGS_NONE, sizeof(lanczos2Weights), lanczos2Weights },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_REACTIVITY, L"FSR2_DefaultReactiviyMask", FFX_RESOURCE_USAGE_READ_ONLY,
            FFX_SURFACE_FORMAT_R8_UNORM, 1, 1, 1, FFX_RESOURCE_FLAGS_NONE, sizeof(defaultReactiveMaskData), &defaultReactiveMaskData },

        {   FFX_FSR2_RESOURCE_IDENTITIER_UPSAMPLE_MAXIMUM_BIAS_LUT, L"FSR2_MaximumUpsampleBias", FFX_RESOURCE_USAGE_READ_ONLY,
            FFX_SURFACE_FORMAT_R16_SNORM, FFX_FSR2_MAXIMUM_BIAS_TEXTURE_WIDTH, FFX_FSR2_MAXIMUM_BIAS_TEXTURE_HEIGHT, 1, FFX_RESOURCE_FLAGS_NONE, sizeof(maximumBias), maximumBias },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_EXPOSURE, L"FSR2_DefaultExposure", FFX_RESOURCE_USAGE_READ_ONLY,
            FFX_SURFACE_FORMAT_R32G32_FLOAT, 1, 1, 1, FFX_RESOURCE_FLAGS_NONE, sizeof(defaultExposure), defaultExposure },

        {   FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE, L"FSR2_AutoExposure", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32G32_FLOAT, 1, 1, 1, FFX_RESOURCE_FLAGS_NONE },


        // only one for now, will need pingpont to respect the motion vectors
        {   FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE, L"FSR2_AutoReactive", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UNORM, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },
        {   FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION, L"FSR2_AutoComposition", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UNORM, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },
        {   FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_1, L"FSR2_PrevPreAlpha0", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R11G11B10_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },
        {   FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_1, L"FSR2_PrevPostAlpha0", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R11G11B10_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },
        {   FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_2, L"FSR2_PrevPreAlpha1", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R11G11B10_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },
        {   FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_2, L"FSR2_PrevPostAlpha1", FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R11G11B10_FLOAT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1, FFX_RESOURCE_FLAGS_NONE },

    };

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex) {

        const Fsr2ResourceDescription* currentSurfaceDescription = &internalSurfaceDesc[currentSurfaceIndex];
        const FfxResourceType resourceType = currentSurfaceDescription->height > 1 ? FFX_RESOURCE_TYPE_TEXTURE2D : texture1dResourceType;
        const FfxResourceDescription resourceDescription = { resourceType, currentSurfaceDescription->format, currentSurfaceDescription->width, currentSurfaceDescription->height, 1, currentSurfaceDescription->mipCount };
        const FfxResourceStates initialState = (currentSurfaceDescription->usage == FFX_RESOURCE_USAGE_READ_ONLY) ? FFX_RESOURCE_STATE_COMPUTE_READ : FFX_RESOURCE_STATE_UNORDERED_ACCESS;
        const FfxCreateResourceDescription createResourceDescription = { FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, currentSurfaceDescription->initDataSize, currentSurfaceDescription->initData, currentSurfaceDescription->name, currentSurfaceDescription->usage, currentSurfaceDescription->id };

        FFX_VALIDATE(context->contextDescription.callbacks.fpCreateResource(&context->contextDescription.callbacks, &createResourceDescription, &context->srvResources[currentSurfaceDescription->id]));
    }

    // copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));

    // avoid compiling pipelines on first render
    {
        context->refreshPipelineStates = false;
        errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }
    return FFX_OK;
}

static void fsr2SafeReleasePipeline(FfxFsr2Context_Private* context, FfxPipelineState* pipeline)
{
    FFX_ASSERT(pipeline);

    context->contextDescription.callbacks.fpDestroyPipeline(&context->contextDescription.callbacks, pipeline);
}

static void fsr2SafeReleaseResource(FfxFsr2Context_Private* context, FfxResourceInternal resource)
{
    context->contextDescription.callbacks.fpDestroyResource(&context->contextDescription.callbacks, resource);
}

static void fsr2SafeReleaseDevice(FfxFsr2Context_Private* context, FfxDevice* device)
{
    if (*device == nullptr) {
        return;
    }

    context->contextDescription.callbacks.fpDestroyBackendContext(&context->contextDescription.callbacks);
    *device = nullptr;
}

static FfxErrorCode fsr2Release(FfxFsr2Context_Private* context)
{
    FFX_ASSERT(context);

    fsr2SafeReleasePipeline(context, &context->pipelineDepthClip);
    fsr2SafeReleasePipeline(context, &context->pipelineReconstructPreviousDepth);
    fsr2SafeReleasePipeline(context, &context->pipelineLock);
    fsr2SafeReleasePipeline(context, &context->pipelineAccumulate);
    fsr2SafeReleasePipeline(context, &context->pipelineAccumulateSharpen);
    fsr2SafeReleasePipeline(context, &context->pipelineRCAS);
    fsr2SafeReleasePipeline(context, &context->pipelineComputeLuminancePyramid);
    fsr2SafeReleasePipeline(context, &context->pipelineGenerateReactive);
    fsr2SafeReleasePipeline(context, &context->pipelineTcrAutogenerate);

    // unregister resources not created internally
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_DEPTH] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_RCAS_INPUT] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT] = { FFX_FSR2_RESOURCE_IDENTIFIER_NULL };

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_FSR2_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex) {

        fsr2SafeReleaseResource(context, context->srvResources[currentResourceIndex]);
    }

    fsr2SafeReleaseDevice(context, &context->device);

    return FFX_OK;
}

static void setupDeviceDepthToViewSpaceDepthParams(FfxFsr2Context_Private* context, const FfxFsr2DispatchDescription* params)
{
    const bool bInverted = (context->contextDescription.flags & FFX_FSR2_ENABLE_DEPTH_INVERTED) == FFX_FSR2_ENABLE_DEPTH_INVERTED;
    const bool bInfinite = (context->contextDescription.flags & FFX_FSR2_ENABLE_DEPTH_INFINITE) == FFX_FSR2_ENABLE_DEPTH_INFINITE;

    // make sure it has no impact if near and far plane values are swapped in dispatch params
    // the flags "inverted" and "infinite" will decide what transform to use
    float fMin = FFX_MINIMUM(params->cameraNear, params->cameraFar);
    float fMax = FFX_MAXIMUM(params->cameraNear, params->cameraFar);

    if (bInverted) {
        float tmp = fMin;
        fMin = fMax;
        fMax = tmp;
    }

    // a 0 0 0   x
    // 0 b 0 0   y
    // 0 0 c d   z
    // 0 0 e 0   1

    const float fQ = fMax / (fMin - fMax);
    const float d = -1.0f; // for clarity

    const float matrix_elem_c[2][2] = {
        fQ,                     // non reversed, non infinite
        -1.0f - FLT_EPSILON,    // non reversed, infinite
        fQ,                     // reversed, non infinite
        0.0f + FLT_EPSILON      // reversed, infinite
    };

    const float matrix_elem_e[2][2] = {
        fQ * fMin,             // non reversed, non infinite
        -fMin - FLT_EPSILON,    // non reversed, infinite
        fQ * fMin,             // reversed, non infinite
        fMax,                  // reversed, infinite
    };

    context->constants.deviceToViewDepth[0] = d * matrix_elem_c[bInverted][bInfinite];
    context->constants.deviceToViewDepth[1] = matrix_elem_e[bInverted][bInfinite];

    // revert x and y coords
    const float aspect = params->renderSize.width / float(params->renderSize.height);
    const float cotHalfFovY = cosf(0.5f * params->cameraFovAngleVertical) / sinf(0.5f * params->cameraFovAngleVertical);
    const float a = cotHalfFovY / aspect;
    const float b = cotHalfFovY;

    context->constants.deviceToViewDepth[2] = (1.0f / a);
    context->constants.deviceToViewDepth[3] = (1.0f / b);
}

static void scheduleDispatch(FfxFsr2Context_Private* context, const FfxFsr2DispatchDescription* params, const FfxPipelineState* pipeline, uint32_t dispatchX, uint32_t dispatchY)
{
    FfxComputeJobDescription jobDescriptor = {};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvCount; ++currentShaderResourceViewIndex) {

        const uint32_t currentResourceId = pipeline->srvResourceBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = context->srvResources[currentResourceId];
        jobDescriptor.srvs[currentShaderResourceViewIndex] = currentResource;
        wcscpy_s(jobDescriptor.srvNames[currentShaderResourceViewIndex], pipeline->srvResourceBindings[currentShaderResourceViewIndex].name);
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavCount; ++currentUnorderedAccessViewIndex) {

        const uint32_t currentResourceId = pipeline->uavResourceBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        wcscpy_s(jobDescriptor.uavNames[currentUnorderedAccessViewIndex], pipeline->uavResourceBindings[currentUnorderedAccessViewIndex].name);

        if (currentResourceId >= FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_0 && currentResourceId <= FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_12)
        {
            const FfxResourceInternal currentResource = context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE];
            jobDescriptor.uavs[currentUnorderedAccessViewIndex] = currentResource;
            jobDescriptor.uavMip[currentUnorderedAccessViewIndex] = currentResourceId - FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE_MIPMAP_0;
        }
        else
        {
            const FfxResourceInternal currentResource = context->uavResources[currentResourceId];
            jobDescriptor.uavs[currentUnorderedAccessViewIndex] = currentResource;
            jobDescriptor.uavMip[currentUnorderedAccessViewIndex] = 0;
        }
    }
    
    jobDescriptor.dimensions[0] = dispatchX;
    jobDescriptor.dimensions[1] = dispatchY;
    jobDescriptor.dimensions[2] = 1;
    jobDescriptor.pipeline = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex) {
        wcscpy_s( jobDescriptor.cbNames[currentRootConstantIndex], pipeline->cbResourceBindings[currentRootConstantIndex].name);
        jobDescriptor.cbs[currentRootConstantIndex] = globalFsr2ConstantBuffers[pipeline->cbResourceBindings[currentRootConstantIndex].resourceIdentifier];
        jobDescriptor.cbSlotIndex[currentRootConstantIndex] = pipeline->cbResourceBindings[currentRootConstantIndex].slotIndex;
    }

    FfxGpuJobDescription dispatchJob = { FFX_GPU_JOB_COMPUTE };
    dispatchJob.computeJobDescriptor = jobDescriptor;

    context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &dispatchJob);
}

static FfxErrorCode fsr2Dispatch(FfxFsr2Context_Private* context, const FfxFsr2DispatchDescription* params)
{
    if ((context->contextDescription.flags & FFX_FSR2_ENABLE_DEBUG_CHECKING) == FFX_FSR2_ENABLE_DEBUG_CHECKING)
    {
        fsr2DebugCheckDispatch(context, params);
    }
    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    // try and refresh shaders first. Early exit in case of error.
    if (context->refreshPipelineStates) {

        context->refreshPipelineStates = false;

        const FfxErrorCode errorCode = createPipelineStates(context);
        FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);
    }

    if (context->firstExecution)
    {
        FfxGpuJobDescription clearJob = { FFX_GPU_JOB_CLEAR_FLOAT };

        const float clearValuesToZeroFloat[]{ 0.f, 0.f, 0.f, 0.f };
        memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));

        clearJob.clearJobDescriptor.target = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_1];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);
        clearJob.clearJobDescriptor.target = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_2];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);
        clearJob.clearJobDescriptor.target = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREPARED_INPUT_COLOR];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);
    }

    // Prepare per frame descriptor tables
    const bool isOddFrame = !!(context->resourceFrameIndex & 1);
    const uint32_t currentCpuOnlyTableBase = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_COUNT : 0;
    const uint32_t currentGpuTableBase = 2 * FFX_FSR2_RESOURCE_IDENTIFIER_COUNT * context->resourceFrameIndex;
    const uint32_t lockStatusSrvResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_2 : FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_1;
    const uint32_t lockStatusUavResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_1 : FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS_2;
    const uint32_t upscaledColorSrvResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2 : FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1;
    const uint32_t upscaledColorUavResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_1 : FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR_2;
    const uint32_t dilatedMotionVectorsResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_2 : FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_1;
    const uint32_t previousDilatedMotionVectorsResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_1 : FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DILATED_MOTION_VECTORS_2;
    const uint32_t lumaHistorySrvResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_2 : FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_1;
    const uint32_t lumaHistoryUavResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_1 : FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY_2;

    const uint32_t prevPreAlphaColorSrvResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_2 : FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_1;
    const uint32_t prevPreAlphaColorUavResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_1 : FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR_2;
    const uint32_t prevPostAlphaColorSrvResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_2 : FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_1;
    const uint32_t prevPostAlphaColorUavResourceIndex = isOddFrame ? FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_1 : FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR_2;

    const bool resetAccumulation = params->reset || context->firstExecution;
    context->firstExecution = false;

    context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->color, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR]);
    context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->depth, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_DEPTH]);
    context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->motionVectors, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_MOTION_VECTORS]);

    // if auto exposure is enabled use the auto exposure SRV, otherwise what the app sends.
    if (context->contextDescription.flags & FFX_FSR2_ENABLE_AUTO_EXPOSURE) {
        context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE];
    } else {
        if (ffxFsr2ResourceIsNull(params->exposure)) {
            context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_EXPOSURE];
        } else {
            context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->exposure, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_EXPOSURE]);
        }
    }
 
    if (params->enableAutoReactive)
    {
        context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->colorOpaqueOnly, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR]);
    }
    
    if (ffxFsr2ResourceIsNull(params->reactive)) {
        context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_REACTIVITY];
    }
    else {
        context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->reactive, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK]);
    }
    
    if (ffxFsr2ResourceIsNull(params->transparencyAndComposition)) {
        context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_DEFAULT_REACTIVITY];
    } else {
        context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->transparencyAndComposition, &context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK]);
    }

    context->contextDescription.callbacks.fpRegisterResource(&context->contextDescription.callbacks, &params->output, &context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT]);
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS] = context->srvResources[lockStatusSrvResourceIndex];
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR] = context->srvResources[upscaledColorSrvResourceIndex];
    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS] = context->uavResources[lockStatusUavResourceIndex];
    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR] = context->uavResources[upscaledColorUavResourceIndex];
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_RCAS_INPUT] = context->uavResources[upscaledColorUavResourceIndex];

    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS] = context->srvResources[dilatedMotionVectorsResourceIndex];
    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS] = context->uavResources[dilatedMotionVectorsResourceIndex];
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREVIOUS_DILATED_MOTION_VECTORS] = context->srvResources[previousDilatedMotionVectorsResourceIndex];

    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY] = context->uavResources[lumaHistoryUavResourceIndex];
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_LUMA_HISTORY] = context->srvResources[lumaHistorySrvResourceIndex];

    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR]  = context->srvResources[prevPreAlphaColorSrvResourceIndex];
    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR]  = context->uavResources[prevPreAlphaColorUavResourceIndex];
    context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR] = context->srvResources[prevPostAlphaColorSrvResourceIndex];
    context->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR] = context->uavResources[prevPostAlphaColorUavResourceIndex];

    // actual resource size may differ from render/display resolution (e.g. due to Hw/API restrictions), so query the descriptor for UVs adjustment
    const FfxResourceDescription resourceDescInputColor = context->contextDescription.callbacks.fpGetResourceDescription(&context->contextDescription.callbacks, context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR]);
    const FfxResourceDescription resourceDescLockStatus = context->contextDescription.callbacks.fpGetResourceDescription(&context->contextDescription.callbacks, context->srvResources[lockStatusSrvResourceIndex]);
    const FfxResourceDescription resourceDescReactiveMask = context->contextDescription.callbacks.fpGetResourceDescription(&context->contextDescription.callbacks, context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK]);
    FFX_ASSERT(resourceDescInputColor.type == FFX_RESOURCE_TYPE_TEXTURE2D);
    FFX_ASSERT(resourceDescLockStatus.type == FFX_RESOURCE_TYPE_TEXTURE2D);

    context->constants.jitterOffset[0] = params->jitterOffset.x;
    context->constants.jitterOffset[1] = params->jitterOffset.y;
    context->constants.renderSize[0] = int32_t(params->renderSize.width ? params->renderSize.width   : resourceDescInputColor.width);
    context->constants.renderSize[1] = int32_t(params->renderSize.height ? params->renderSize.height : resourceDescInputColor.height);
    context->constants.maxRenderSize[0] = int32_t(context->contextDescription.maxRenderSize.width);
    context->constants.maxRenderSize[1] = int32_t(context->contextDescription.maxRenderSize.height);
    context->constants.inputColorResourceDimensions[0] = resourceDescInputColor.width;
    context->constants.inputColorResourceDimensions[1] = resourceDescInputColor.height;

    // compute the horizontal FOV for the shader from the vertical one.
    const float aspectRatio = (float)params->renderSize.width / (float)params->renderSize.height;
    const float cameraAngleHorizontal = atan(tan(params->cameraFovAngleVertical / 2) * aspectRatio) * 2;
    context->constants.tanHalfFOV = tanf(cameraAngleHorizontal * 0.5f);
    context->constants.viewSpaceToMetersFactor = (params->viewSpaceToMetersFactor > 0.0f) ? params->viewSpaceToMetersFactor : 1.0f;

    // compute params to enable device depth to view space depth computation in shader
    setupDeviceDepthToViewSpaceDepthParams(context, params);

    // To be updated if resource is larger than the actual image size
    context->constants.downscaleFactor[0] = float(context->constants.renderSize[0]) / context->contextDescription.displaySize.width;
    context->constants.downscaleFactor[1] = float(context->constants.renderSize[1]) / context->contextDescription.displaySize.height;
    context->constants.previousFramePreExposure = context->constants.preExposure;
    context->constants.preExposure = (params->preExposure != 0) ? params->preExposure : 1.0f;

    // motion vector data
    const int32_t* motionVectorsTargetSize = (context->contextDescription.flags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? context->constants.displaySize : context->constants.renderSize;

    context->constants.motionVectorScale[0] = (params->motionVectorScale.x / motionVectorsTargetSize[0]);
    context->constants.motionVectorScale[1] = (params->motionVectorScale.y / motionVectorsTargetSize[1]);

    // compute jitter cancellation
    if (context->contextDescription.flags & FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) {

        context->constants.motionVectorJitterCancellation[0] = (context->previousJitterOffset[0] - context->constants.jitterOffset[0]) / motionVectorsTargetSize[0];
        context->constants.motionVectorJitterCancellation[1] = (context->previousJitterOffset[1] - context->constants.jitterOffset[1]) / motionVectorsTargetSize[1];

        context->previousJitterOffset[0] = context->constants.jitterOffset[0];
        context->previousJitterOffset[1] = context->constants.jitterOffset[1];
    }

    // lock data, assuming jitter sequence length computation for now
    const int32_t jitterPhaseCount = ffxFsr2GetJitterPhaseCount(params->renderSize.width, context->contextDescription.displaySize.width);

    // init on first frame
    if (resetAccumulation || context->constants.jitterPhaseCount == 0) {
        context->constants.jitterPhaseCount = (float)jitterPhaseCount;
    } else {
        const int32_t jitterPhaseCountDelta = (int32_t)(jitterPhaseCount - context->constants.jitterPhaseCount);
        if (jitterPhaseCountDelta > 0) {
            context->constants.jitterPhaseCount++;
        } else if (jitterPhaseCountDelta < 0) {
            context->constants.jitterPhaseCount--;
        }
    }

    // convert delta time to seconds and clamp to [0, 1].
    context->constants.deltaTime = FFX_MAXIMUM(0.0f, FFX_MINIMUM(1.0f, params->frameTimeDelta / 1000.0f));

    if (resetAccumulation) {
        context->constants.frameIndex = 0;
    } else {
        context->constants.frameIndex++;
    }

    // shading change usage of the SPD mip levels.
    context->constants.lumaMipLevelToUse = uint32_t(FFX_FSR2_SHADING_CHANGE_MIP_LEVEL);

    const float mipDiv = float(2 << context->constants.lumaMipLevelToUse);
    context->constants.lumaMipDimensions[0] = uint32_t(context->constants.maxRenderSize[0] / mipDiv);
    context->constants.lumaMipDimensions[1] = uint32_t(context->constants.maxRenderSize[1] / mipDiv);

    memcpy(context->constants.reprojectionMatrix, params->reprojectionMatrix, sizeof(context->constants.reprojectionMatrix));

    // reactive mask bias
    const int32_t threadGroupWorkRegionDim = 8;
    const int32_t dispatchSrcX = (context->constants.renderSize[0] + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    const int32_t dispatchSrcY = (context->constants.renderSize[1] + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    const int32_t dispatchDstX = (context->contextDescription.displaySize.width + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    const int32_t dispatchDstY = (context->contextDescription.displaySize.height + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;

    // Clear reconstructed depth for max depth store.
    if (resetAccumulation) {

        FfxGpuJobDescription clearJob = { FFX_GPU_JOB_CLEAR_FLOAT };

        // LockStatus resource has no sign bit, callback functions are compensating for this.
        // Clearing the resource must follow the same logic.
        float clearValuesLockStatus[4]{};
        clearValuesLockStatus[LOCK_LIFETIME_REMAINING] = 0.0f;
        clearValuesLockStatus[LOCK_TEMPORAL_LUMA] = 0.0f;

        memcpy(clearJob.clearJobDescriptor.color, clearValuesLockStatus, 4 * sizeof(float));
        clearJob.clearJobDescriptor.target = context->srvResources[lockStatusSrvResourceIndex];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);

        const float clearValuesToZeroFloat[]{ 0.f, 0.f, 0.f, 0.f };
        memcpy(clearJob.clearJobDescriptor.color, clearValuesToZeroFloat, 4 * sizeof(float));
        clearJob.clearJobDescriptor.target = context->srvResources[upscaledColorSrvResourceIndex];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);

        clearJob.clearJobDescriptor.target = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_SCENE_LUMINANCE];
        context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);

        //if (context->contextDescription.flags & FFX_FSR2_ENABLE_AUTO_EXPOSURE)
        // Auto exposure always used to track luma changes in locking logic
        {
            const float clearValuesExposure[]{ -1.f, 1e8f, 0.f, 0.f };
            memcpy(clearJob.clearJobDescriptor.color, clearValuesExposure, 4 * sizeof(float));
            clearJob.clearJobDescriptor.target = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTO_EXPOSURE];
            context->contextDescription.callbacks.fpScheduleGpuJob(&context->contextDescription.callbacks, &clearJob);
        }
    }

    // Auto exposure
    uint32_t dispatchThreadGroupCountXY[2];
    uint32_t workGroupOffset[2];
    uint32_t numWorkGroupsAndMips[2];
    uint32_t rectInfo[4] = { 0, 0, params->renderSize.width, params->renderSize.height };
    SpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo);

    // downsample
    Fsr2SpdConstants luminancePyramidConstants;
    luminancePyramidConstants.numworkGroups = numWorkGroupsAndMips[0];
    luminancePyramidConstants.mips = numWorkGroupsAndMips[1];
    luminancePyramidConstants.workGroupOffset[0] = workGroupOffset[0];
    luminancePyramidConstants.workGroupOffset[1] = workGroupOffset[1];
    luminancePyramidConstants.renderSize[0] = params->renderSize.width;
    luminancePyramidConstants.renderSize[1] = params->renderSize.height;

    // compute the constants.
    Fsr2RcasConstants rcasConsts = {};
    const float sharpenessRemapped = (-2.0f * params->sharpness) + 2.0f;
    FsrRcasCon(rcasConsts.rcasConfig, sharpenessRemapped);

    Fsr2GenerateReactiveConstants2 genReactiveConsts = {};
    genReactiveConsts.autoTcThreshold = params->autoTcThreshold;
    genReactiveConsts.autoTcScale = params->autoTcScale;
    genReactiveConsts.autoReactiveScale = params->autoReactiveScale;
    genReactiveConsts.autoReactiveMax = params->autoReactiveMax;

    // initialize constantBuffers data
    memcpy(&globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_FSR2].data,        &context->constants,        globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_FSR2].uint32Size * sizeof(uint32_t));
    memcpy(&globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_SPD].data,         &luminancePyramidConstants, globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_SPD].uint32Size  * sizeof(uint32_t));
    memcpy(&globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_RCAS].data,        &rcasConsts,                globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_RCAS].uint32Size * sizeof(uint32_t));
    memcpy(&globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_GENREACTIVE].data, &genReactiveConsts,         globalFsr2ConstantBuffers[FFX_FSR2_CONSTANTBUFFER_IDENTIFIER_GENREACTIVE].uint32Size * sizeof(uint32_t));

    // Auto reactive
    if (params->enableAutoReactive)
    {
        generateReactiveMaskInternal(context, params);
        context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_REACTIVE_MASK] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE];
        context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_TRANSPARENCY_AND_COMPOSITION_MASK] = context->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION];
    }
    scheduleDispatch(context, params, &context->pipelineComputeLuminancePyramid, dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1]);
    scheduleDispatch(context, params, &context->pipelineReconstructPreviousDepth, dispatchSrcX, dispatchSrcY);
    scheduleDispatch(context, params, &context->pipelineDepthClip, dispatchSrcX, dispatchSrcY);

    const bool sharpenEnabled = params->enableSharpening;

    scheduleDispatch(context, params, &context->pipelineLock, dispatchSrcX, dispatchSrcY);
    scheduleDispatch(context, params, sharpenEnabled ? &context->pipelineAccumulateSharpen : &context->pipelineAccumulate, dispatchDstX, dispatchDstY);

    // RCAS
    if (sharpenEnabled) {

        // dispatch RCAS
        const int32_t threadGroupWorkRegionDimRCAS = 16;
        const int32_t dispatchX = (context->contextDescription.displaySize.width + (threadGroupWorkRegionDimRCAS - 1)) / threadGroupWorkRegionDimRCAS;
        const int32_t dispatchY = (context->contextDescription.displaySize.height + (threadGroupWorkRegionDimRCAS - 1)) / threadGroupWorkRegionDimRCAS;
        scheduleDispatch(context, params, &context->pipelineRCAS, dispatchX, dispatchY);
    }

    context->resourceFrameIndex = (context->resourceFrameIndex + 1) % FSR2_MAX_QUEUED_FRAMES;

    // Fsr2MaxQueuedFrames must be an even number.
    FFX_STATIC_ASSERT((FSR2_MAX_QUEUED_FRAMES & 1) == 0);

    context->contextDescription.callbacks.fpExecuteGpuJobs(&context->contextDescription.callbacks, commandList);

    // release dynamic resources
    context->contextDescription.callbacks.fpUnregisterResources(&context->contextDescription.callbacks);

    return FFX_OK;
}

FfxErrorCode ffxFsr2ContextCreate(FfxFsr2Context* context, const FfxFsr2ContextDescription* contextDescription)
{
    // zero context memory
    memset(context, 0, sizeof(FfxFsr2Context));

    // check pointers are valid.
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        contextDescription,
        FFX_ERROR_INVALID_POINTER);

    // validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->callbacks.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->callbacks.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->callbacks.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->callbacks.scratchBuffer) {

        FFX_RETURN_ON_ERROR(contextDescription->callbacks.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxFsr2Context) >= sizeof(FfxFsr2Context_Private));

    // create the context.
    FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)(context);
    const FfxErrorCode errorCode = fsr2Create(contextPrivate, contextDescription);

    return errorCode;
}

FfxErrorCode ffxFsr2ContextDestroy(FfxFsr2Context* context)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);

    // destroy the context.
    FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)(context);
    const FfxErrorCode errorCode = fsr2Release(contextPrivate);
    return errorCode;
}

FfxErrorCode ffxFsr2ContextDispatch(FfxFsr2Context* context, const FfxFsr2DispatchDescription* dispatchParams)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        dispatchParams,
        FFX_ERROR_INVALID_POINTER);

    FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)(context);

    // validate that renderSize is within the maximum.
    FFX_RETURN_ON_ERROR(
        dispatchParams->renderSize.width <= contextPrivate->contextDescription.maxRenderSize.width,
        FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(
        dispatchParams->renderSize.height <= contextPrivate->contextDescription.maxRenderSize.height,
        FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(
        contextPrivate->device,
        FFX_ERROR_NULL_DEVICE);

    // dispatch the FSR2 passes.
    const FfxErrorCode errorCode = fsr2Dispatch(contextPrivate, dispatchParams);
    return errorCode;
}

float ffxFsr2GetUpscaleRatioFromQualityMode(FfxFsr2QualityMode qualityMode)
{
    switch (qualityMode) {

    case FFX_FSR2_QUALITY_MODE_QUALITY:
        return 1.5f;
    case FFX_FSR2_QUALITY_MODE_BALANCED:
        return 1.7f;
    case FFX_FSR2_QUALITY_MODE_PERFORMANCE:
        return 2.0f;
    case FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE:
        return 3.0f;
    default:
        return 0.0f;
    }
}

FfxErrorCode ffxFsr2GetRenderResolutionFromQualityMode(
    uint32_t* renderWidth,
    uint32_t* renderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    FfxFsr2QualityMode qualityMode)
{
    FFX_RETURN_ON_ERROR(
        renderWidth,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        renderHeight,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        FFX_FSR2_QUALITY_MODE_QUALITY <= qualityMode && qualityMode <= FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE,
        FFX_ERROR_INVALID_ENUM);

    // scale by the predefined ratios in each dimension.
    const float ratio = ffxFsr2GetUpscaleRatioFromQualityMode(qualityMode);
    const uint32_t scaledDisplayWidth = (uint32_t)((float)displayWidth / ratio);
    const uint32_t scaledDisplayHeight = (uint32_t)((float)displayHeight / ratio);
    *renderWidth = scaledDisplayWidth;
    *renderHeight = scaledDisplayHeight;

    return FFX_OK;
}

FfxErrorCode ffxFsr2ContextEnqueueRefreshPipelineRequest(FfxFsr2Context* context)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);

    FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)context;
    contextPrivate->refreshPipelineStates = true;

    return FFX_OK;
}

int32_t ffxFsr2GetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth)
{
    const float basePhaseCount = 8.0f;
    const int32_t jitterPhaseCount = int32_t(basePhaseCount * pow((float(displayWidth) / renderWidth), 2.0f));
    return jitterPhaseCount;
}

FfxErrorCode ffxFsr2GetJitterOffset(float* outX, float* outY, int32_t index, int32_t phaseCount)
{
    FFX_RETURN_ON_ERROR(
        outX,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        outY,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        phaseCount > 0,
        FFX_ERROR_INVALID_ARGUMENT);

    const float x = halton((index % phaseCount) + 1, 2) - 0.5f;
    const float y = halton((index % phaseCount) + 1, 3) - 0.5f;

    *outX = x;
    *outY = y;
    return FFX_OK;
}

FFX_API bool ffxFsr2ResourceIsNull(FfxResource resource)
{
    return resource.resource == NULL;
}

FfxErrorCode ffxFsr2ContextGenerateReactiveMask(FfxFsr2Context* context, const FfxFsr2GenerateReactiveDescription* params)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        params,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        params->commandList,
        FFX_ERROR_INVALID_POINTER);

    FfxFsr2Context_Private* contextPrivate = (FfxFsr2Context_Private*)(context);

    FFX_RETURN_ON_ERROR(
        contextPrivate->device,
        FFX_ERROR_NULL_DEVICE);

    if (contextPrivate->refreshPipelineStates) {

        createPipelineStates(contextPrivate);
        contextPrivate->refreshPipelineStates = false;
    }

    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    FfxPipelineState* pipeline = &contextPrivate->pipelineGenerateReactive;

    const int32_t threadGroupWorkRegionDim = 8;
    const int32_t dispatchSrcX = (params->renderSize.width  + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    const int32_t dispatchSrcY = (params->renderSize.height + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;

    // save internal reactive resource
    FfxResourceInternal internalReactive = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE];

    FfxComputeJobDescription jobDescriptor = {};
    contextPrivate->contextDescription.callbacks.fpRegisterResource(&contextPrivate->contextDescription.callbacks, &params->colorOpaqueOnly, &contextPrivate->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY]);
    contextPrivate->contextDescription.callbacks.fpRegisterResource(&contextPrivate->contextDescription.callbacks, &params->colorPreUpscale, &contextPrivate->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR]);
    contextPrivate->contextDescription.callbacks.fpRegisterResource(&contextPrivate->contextDescription.callbacks, &params->outReactive, &contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE]);
    
    jobDescriptor.uavs[0] = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE];

    wcscpy_s(jobDescriptor.srvNames[0], pipeline->srvResourceBindings[0].name);
    wcscpy_s(jobDescriptor.srvNames[1], pipeline->srvResourceBindings[1].name);
    wcscpy_s(jobDescriptor.uavNames[0], pipeline->uavResourceBindings[0].name);

    jobDescriptor.dimensions[0] = dispatchSrcX;
    jobDescriptor.dimensions[1] = dispatchSrcY;
    jobDescriptor.dimensions[2] = 1;
    jobDescriptor.pipeline = *pipeline;

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvCount; ++currentShaderResourceViewIndex) {

        const uint32_t currentResourceId = pipeline->srvResourceBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = contextPrivate->srvResources[currentResourceId];
        jobDescriptor.srvs[currentShaderResourceViewIndex] = currentResource;
        wcscpy_s(jobDescriptor.srvNames[currentShaderResourceViewIndex], pipeline->srvResourceBindings[currentShaderResourceViewIndex].name);
    }

    Fsr2GenerateReactiveConstants constants = {};
    constants.scale = params->scale;
    constants.threshold = params->cutoffThreshold;
    constants.binaryValue = params->binaryValue;
    constants.flags = params->flags;

    jobDescriptor.cbs[0].uint32Size = sizeof(constants);
    memcpy(&jobDescriptor.cbs[0].data, &constants, sizeof(constants));
    wcscpy_s(jobDescriptor.cbNames[0], pipeline->cbResourceBindings[0].name);

    FfxGpuJobDescription dispatchJob = { FFX_GPU_JOB_COMPUTE };
    dispatchJob.computeJobDescriptor = jobDescriptor;

    contextPrivate->contextDescription.callbacks.fpScheduleGpuJob(&contextPrivate->contextDescription.callbacks, &dispatchJob);

    contextPrivate->contextDescription.callbacks.fpExecuteGpuJobs(&contextPrivate->contextDescription.callbacks, commandList);

    // restore internal reactive
    contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE] = internalReactive;

    return FFX_OK;
}

static FfxErrorCode generateReactiveMaskInternal(FfxFsr2Context_Private* contextPrivate, const FfxFsr2DispatchDescription* params)
{
    if (contextPrivate->refreshPipelineStates) {

        createPipelineStates(contextPrivate);
        contextPrivate->refreshPipelineStates = false;
    }

    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    FfxPipelineState* pipeline = &contextPrivate->pipelineTcrAutogenerate;

    const int32_t threadGroupWorkRegionDim = 8;
    const int32_t dispatchSrcX = (params->renderSize.width + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;
    const int32_t dispatchSrcY = (params->renderSize.height + (threadGroupWorkRegionDim - 1)) / threadGroupWorkRegionDim;

    FfxComputeJobDescription jobDescriptor = {};
    contextPrivate->contextDescription.callbacks.fpRegisterResource(&contextPrivate->contextDescription.callbacks, &params->colorOpaqueOnly, &contextPrivate->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_OPAQUE_ONLY]);
    contextPrivate->contextDescription.callbacks.fpRegisterResource(&contextPrivate->contextDescription.callbacks, &params->color, &contextPrivate->srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_INPUT_COLOR]);

    jobDescriptor.uavs[0] = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOREACTIVE];
    jobDescriptor.uavs[1] = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_AUTOCOMPOSITION];
    jobDescriptor.uavs[2] = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_PRE_ALPHA_COLOR];
    jobDescriptor.uavs[3] = contextPrivate->uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_PREV_POST_ALPHA_COLOR];

    wcscpy_s(jobDescriptor.uavNames[0], pipeline->uavResourceBindings[0].name);
    wcscpy_s(jobDescriptor.uavNames[1], pipeline->uavResourceBindings[1].name);
    wcscpy_s(jobDescriptor.uavNames[2], pipeline->uavResourceBindings[2].name);
    wcscpy_s(jobDescriptor.uavNames[3], pipeline->uavResourceBindings[3].name);

    jobDescriptor.dimensions[0] = dispatchSrcX;
    jobDescriptor.dimensions[1] = dispatchSrcY;
    jobDescriptor.dimensions[2] = 1;
    jobDescriptor.pipeline = *pipeline;

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvCount; ++currentShaderResourceViewIndex) {

        const uint32_t currentResourceId = pipeline->srvResourceBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = contextPrivate->srvResources[currentResourceId];
        jobDescriptor.srvs[currentShaderResourceViewIndex] = currentResource;
        wcscpy_s(jobDescriptor.srvNames[currentShaderResourceViewIndex], pipeline->srvResourceBindings[currentShaderResourceViewIndex].name);
    }

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex) {
        wcscpy_s(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->cbResourceBindings[currentRootConstantIndex].name);
        jobDescriptor.cbs[currentRootConstantIndex] = globalFsr2ConstantBuffers[pipeline->cbResourceBindings[currentRootConstantIndex].resourceIdentifier];
        jobDescriptor.cbSlotIndex[currentRootConstantIndex] = pipeline->cbResourceBindings[currentRootConstantIndex].slotIndex;
    }

    FfxGpuJobDescription dispatchJob = { FFX_GPU_JOB_COMPUTE };
    dispatchJob.computeJobDescriptor = jobDescriptor;

    contextPrivate->contextDescription.callbacks.fpScheduleGpuJob(&contextPrivate->contextDescription.callbacks, &dispatchJob);

    return FFX_OK;
}
