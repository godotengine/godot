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

#include <algorithm>    // for max used inside SPD CPU code.
#include <cmath>        // for fabs, abs, sinf, sqrt, etc.
#include <string>       // for memset
#include <cfloat>       // for FLT_EPSILON
#include "ffx_frameinterpolation.h"

#define FFX_CPU

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#endif

#include "gpu/ffx_core.h"
#include "gpu/spd/ffx_spd.h"
#include "ffx_object_management.h"

#include "ffx_frameinterpolation_private.h"

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t    index;
    wchar_t     name[64];
}ResourceBinding;

static const ResourceBinding srvResourceBindingTable[] =
{
    // Frame Interpolation textures
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH,                                      L"r_input_depth"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS,                             L"r_input_motion_vectors"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISTORTION_FIELD,                           L"r_input_distortion_field"},

    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH,                              L"r_dilated_depth"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS,                     L"r_dilated_motion_vectors"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME,         L"r_reconstructed_depth_previous_frame"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME,     L"r_reconstructed_depth_interpolated_frame"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE,              L"r_previous_interpolation_source"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE,               L"r_current_interpolation_source"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK,                          L"r_disocclusion_mask"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_X,                 L"r_game_motion_vector_field_x"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_Y,                 L"r_game_motion_vector_field_y"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X,         L"r_optical_flow_motion_vector_field_x"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y,         L"r_optical_flow_motion_vector_field_y"},

    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR,                        L"r_optical_flow"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_CONFIDENCE,                    L"r_optical_flow_confidence"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_GLOBAL_MOTION,                 L"r_optical_flow_global_motion"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCENE_CHANGE_DETECTION,        L"r_optical_flow_scd"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT,                                     L"r_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_MASK,                            L"r_inpainting_mask"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID,                         L"r_inpainting_pyramid"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PRESENT_BACKBUFFER,                         L"r_present_backbuffer"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS,                                   L"r_counters"},
};

static const ResourceBinding uavResourceBindingTable[] =
{
    // Frame Interpolation textures
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH,                              L"rw_dilated_depth"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS,                     L"rw_dilated_motion_vectors"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME,         L"rw_reconstructed_depth_previous_frame"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME,     L"rw_reconstructed_depth_interpolated_frame"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT,                                     L"rw_output"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK,                          L"rw_disocclusion_mask"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_X,                 L"rw_game_motion_vector_field_x"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_Y,                 L"rw_game_motion_vector_field_y"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X,         L"rw_optical_flow_motion_vector_field_x"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y,         L"rw_optical_flow_motion_vector_field_y"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_MASK,                            L"rw_inpainting_mask"},

    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS,                                   L"rw_counters"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_0,                L"rw_inpainting_pyramid0"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_1,                L"rw_inpainting_pyramid1"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_2,                L"rw_inpainting_pyramid2"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_3,                L"rw_inpainting_pyramid3"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_4,                L"rw_inpainting_pyramid4"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_5,                L"rw_inpainting_pyramid5"}, // extra declaration, as this is globallycoherent
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_6,                L"rw_inpainting_pyramid6"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_7,                L"rw_inpainting_pyramid7"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_8,                L"rw_inpainting_pyramid8"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_9,                L"rw_inpainting_pyramid9"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_10,               L"rw_inpainting_pyramid10"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_11,               L"rw_inpainting_pyramid11"},
    {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_12,               L"rw_inpainting_pyramid12"},
};

static const ResourceBinding cbResourceBindingTable[] =
{
    {FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER,                                      L"cbFI"},
    {FFX_FRAMEINTERPOLATION_INPAINTING_PYRAMID_CONSTANTBUFFER_IDENTIFIER,                   L"cbInpaintingPyramid"},
};

// Broad structure of the root signature.
typedef enum FrameInterpolationRootSignatureLayout {

    FRAMEINTERPOLATION_ROOT_SIGNATURE_LAYOUT_UAVS,
    FRAMEINTERPOLATION_ROOT_SIGNATURE_LAYOUT_SRVS,
    FRAMEINTERPOLATION_ROOT_SIGNATURE_LAYOUT_CONSTANTS,
    FRAMEINTERPOLATION_ROOT_SIGNATURE_LAYOUT_CONSTANTS_REGISTER_1,
    FRAMEINTERPOLATION_ROOT_SIGNATURE_LAYOUT_PARAMETER_COUNT
} FrameInterpolationRootSignatureLayout;

typedef union FrameInterpolationSecondaryUnion
{
    InpaintingPyramidConstants inpaintingPyramidConstants;
} FrameInterpolationSecondaryUnion;

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

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == _countof(srvResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
    }

    // check for UAVs where mip chains are to be bound
    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == _countof(uavResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    for (uint32_t cbIndex = 0; cbIndex < inoutPipeline->constCount; ++cbIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(cbResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(cbResourceBindingTable[mapIndex].name, inoutPipeline->constantBufferBindings[cbIndex].name))
                break;
        }
        if (mapIndex == _countof(cbResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->constantBufferBindings[cbIndex].resourceIdentifier = cbResourceBindingTable[mapIndex].index;
    }

    for (uint32_t uavBufferIndex = 0; uavBufferIndex < inoutPipeline->uavBufferCount; ++uavBufferIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(uavResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(uavResourceBindingTable[mapIndex].name, inoutPipeline->uavBufferBindings[uavBufferIndex].name))
                break;
        }
        if (mapIndex == _countof(uavResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavBufferBindings[uavBufferIndex].resourceIdentifier = uavResourceBindingTable[mapIndex].index;
    }

    for (uint32_t srvBufferIndex = 0; srvBufferIndex < inoutPipeline->srvBufferCount; ++srvBufferIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(srvResourceBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvResourceBindingTable[mapIndex].name, inoutPipeline->srvBufferBindings[srvBufferIndex].name))
                break;
        }
        if (mapIndex == _countof(srvResourceBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvBufferBindings[srvBufferIndex].resourceIdentifier = srvResourceBindingTable[mapIndex].index;
    }


    return FFX_OK;
}

static uint32_t getPipelinePermutationFlags(uint32_t contextFlags, FfxPass, bool fp16, bool force64, bool)
{
    // work out what permutation to load.
    uint32_t flags = 0;
    flags |= (contextFlags & FFX_FRAMEINTERPOLATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? 0 : FRAMEINTERPOLATION_SHADER_PERMUTATION_LOW_RES_MOTION_VECTORS;
    flags |= (contextFlags & FFX_FRAMEINTERPOLATION_ENABLE_JITTER_MOTION_VECTORS) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_JITTER_MOTION_VECTORS : 0;
    flags |= (contextFlags & FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_DEPTH_INVERTED : 0;
    flags |= (force64) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_FORCE_WAVE64 : 0;
    flags |= (fp16) ? FRAMEINTERPOLATION_SHADER_PERMUTATION_ALLOW_FP16 : 0;
    return flags;
}

static FfxErrorCode createPipelineStates(FfxFrameInterpolationContext_Private* context)
{
    FFX_ASSERT(context);

    FfxPipelineDescription pipelineDescription = {};
    pipelineDescription.contextFlags           = context->contextDescription.flags;
    pipelineDescription.stage                  = FFX_BIND_COMPUTE_SHADER_STAGE;

    // Samplers
    pipelineDescription.samplerCount      = 2;
    FfxSamplerDescription samplerDescs[2] = {
        {FFX_FILTER_TYPE_MINMAGMIP_POINT, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE},
        {FFX_FILTER_TYPE_MINMAGMIP_LINEAR, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE}};
    pipelineDescription.samplers = samplerDescs;

    // Root constants
    pipelineDescription.rootConstantBufferCount     = 2;
    FfxRootConstantDescription rootConstantDescs[2] = 
    {
        {sizeof(FrameInterpolationConstants) / sizeof(uint32_t), FFX_BIND_COMPUTE_SHADER_STAGE},
        {sizeof(InpaintingPyramidConstants) / sizeof(uint32_t), FFX_BIND_COMPUTE_SHADER_STAGE}
    };
    pipelineDescription.rootConstants               = rootConstantDescs;

    // Query device capabilities
    FfxDeviceCapabilities capabilities;
    context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities);

    // Setup a few options used to determine permutation flags
    bool haveShaderModel66 = capabilities.maximumSupportedShaderModel >= FFX_SHADER_MODEL_6_6;
    bool supportedFP16     = capabilities.fp16Supported;
    bool canForceWave64    = false;
    bool useLut            = false;

    const uint32_t waveLaneCountMin = capabilities.waveLaneCountMin;
    const uint32_t waveLaneCountMax = capabilities.waveLaneCountMax;
    if (waveLaneCountMin == 32 && waveLaneCountMax == 64)
    {
        useLut         = true;
        canForceWave64 = haveShaderModel66;
    }
    else
        canForceWave64 = false;

    // Work out what permutation to load.
    uint32_t contextFlags = context->contextDescription.flags;

    // Set up pipeline descriptor (basically RootSignature and binding)
    auto CreateComputePipeline = [&](FfxPass pass, const wchar_t* name, FfxPipelineState* pipeline) -> FfxErrorCode {
        ffxSafeReleasePipeline(&context->contextDescription.backendInterface, pipeline, context->effectContextId);
        wcscpy_s(pipelineDescription.name, name);
        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(
            &context->contextDescription.backendInterface,
            FFX_EFFECT_FRAMEINTERPOLATION,
            pass,
            getPipelinePermutationFlags(contextFlags, pass, supportedFP16, canForceWave64, useLut),
            &pipelineDescription,
            context->effectContextId,
            pipeline));
        patchResourceBindings(pipeline);

        return FFX_OK;
    };

    auto CreateRasterPipeline = [&](FfxPass pass, const wchar_t* name, FfxPipelineState* pipeline) -> FfxErrorCode {
        wcscpy_s(pipelineDescription.name, name);
        pipelineDescription.stage            = (FfxBindStage)(FFX_BIND_VERTEX_SHADER_STAGE | FFX_BIND_PIXEL_SHADER_STAGE);
        pipelineDescription.backbufferFormat = context->contextDescription.backBufferFormat;
        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(
            &context->contextDescription.backendInterface,
            FFX_EFFECT_FRAMEINTERPOLATION,
            pass,
            getPipelinePermutationFlags(contextFlags, pass, supportedFP16, canForceWave64, useLut),
            &pipelineDescription,
            context->effectContextId,
            pipeline));

        return FFX_OK;
    };

    // Frame Interpolation Pipelines
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_AND_DILATE,               L"RECONSTRUCT_AND_DILATE", &context->pipelineFiReconstructAndDilate);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_SETUP,                                L"SETUP", &context->pipelineFiSetup);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_PREV_DEPTH,               L"RECONSTRUCT_PREV_DEPTH", &context->pipelineFiReconstructPreviousDepth);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_GAME_MOTION_VECTOR_FIELD,             L"GAME_MOTION_VECTOR_FIELD", &context->pipelineFiGameMotionVectorField);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_OPTICAL_FLOW_VECTOR_FIELD,            L"OPTICAL_FLOW_VECTOR_FIELD", &context->pipelineFiOpticalFlowVectorField);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_DISOCCLUSION_MASK,                    L"DISOCCLUSION_MASK", &context->pipelineFiDisocclusionMask);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_INTERPOLATION,                        L"INTERPOLATION", &context->pipelineFiScfi);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_INPAINTING_PYRAMID,                   L"INPAINTING_PYRAMID", &context->pipelineInpaintingPyramid);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_INPAINTING,                           L"INPAINTING", &context->pipelineInpainting);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_GAME_VECTOR_FIELD_INPAINTING_PYRAMID, L"GAME_VECTOR_FIELD_INPAINTING_PYRAMID", & context->pipelineGameVectorFieldInpaintingPyramid);
    CreateComputePipeline(FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW,                           L"DEBUG_VIEW", &context->pipelineDebugView);

    return FFX_OK;
}


// Format precision group for HUDless.
// Also format needs at least the 3 RGB channels to be valid
// int formats aren't accepted.
int GetFormatPrecisionGroup(FfxSurfaceFormat format)
{
    switch (format)
    {
    case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
    case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
    case FFX_SURFACE_FORMAT_R32G32B32_FLOAT:
        return 0;

    case FFX_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
    case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        return 1;

    case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
    case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
    case FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
    case FFX_SURFACE_FORMAT_B8G8R8A8_UNORM:
        return 2;

    case FFX_SURFACE_FORMAT_R8G8B8A8_SNORM:
        return 3;

    case FFX_SURFACE_FORMAT_R8G8B8A8_SRGB:
    case FFX_SURFACE_FORMAT_B8G8R8A8_SRGB:
        return 4;

    case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
        return 5;

    case FFX_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
    case FFX_SURFACE_FORMAT_R10G10B10A2_UNORM:
        return 6;

    case FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP:
        return 7;

    // we don't accept the following formats
    case FFX_SURFACE_FORMAT_R32G32B32A32_UINT:
    case FFX_SURFACE_FORMAT_R32G32_FLOAT:
    case FFX_SURFACE_FORMAT_R8_UINT:
    case FFX_SURFACE_FORMAT_R32_UINT:
    case FFX_SURFACE_FORMAT_R16G16_UINT:
    case FFX_SURFACE_FORMAT_R16G16_SINT:
    case FFX_SURFACE_FORMAT_R16G16_FLOAT:
    case FFX_SURFACE_FORMAT_R16_FLOAT:
    case FFX_SURFACE_FORMAT_R16_UINT:
    case FFX_SURFACE_FORMAT_R16_UNORM:
    case FFX_SURFACE_FORMAT_R16_SNORM:
    case FFX_SURFACE_FORMAT_R8_UNORM:
    case FFX_SURFACE_FORMAT_R8G8_UNORM:
    case FFX_SURFACE_FORMAT_R8G8_UINT:
    case FFX_SURFACE_FORMAT_R32_FLOAT:
    case FFX_SURFACE_FORMAT_UNKNOWN:
    default:
        return -1;
    }
}

static FfxErrorCode frameinterpolationCreate(FfxFrameInterpolationContext_Private* context, const FfxFrameInterpolationContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // validate compatibility between backbuffer and hudless formats
    int backBufferGroup = GetFormatPrecisionGroup(contextDescription->backBufferFormat);
    int previousInterpolationSourceGroup = GetFormatPrecisionGroup(contextDescription->previousInterpolationSourceFormat);
    FFX_RETURN_ON_ERROR(backBufferGroup >= 0 && previousInterpolationSourceGroup >= 0 && backBufferGroup == previousInterpolationSourceGroup, FFX_ERROR_INVALID_ARGUMENT);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxFrameInterpolationContext_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxFrameInterpolationContextDescription));

    // Check version info - make sure we are linked with the right backend version
    FfxVersionNumber version = context->contextDescription.backendInterface.fpGetSDKVersion(&context->contextDescription.backendInterface);
    FFX_RETURN_ON_ERROR(version == FFX_SDK_MAKE_VERSION(1, 1, 4), FFX_ERROR_INVALID_VERSION);

    // Create the context.
    FfxErrorCode errorCode = context->contextDescription.backendInterface.fpCreateBackendContext(&context->contextDescription.backendInterface, FFX_EFFECT_FRAMEINTERPOLATION, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // call out for device caps.
    errorCode = context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &context->deviceCapabilities);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // set defaults
    context->firstExecution = true;

    context->asyncSupported                     = (contextDescription->flags & FFX_FRAMEINTERPOLATION_ENABLE_ASYNC_SUPPORT) == FFX_FRAMEINTERPOLATION_ENABLE_ASYNC_SUPPORT;
    context->constants.maxRenderSize[0]         = contextDescription->maxRenderSize.width;
    context->constants.maxRenderSize[1]         = contextDescription->maxRenderSize.height;
    context->constants.displaySize[0]           = contextDescription->displaySize.width;
    context->constants.displaySize[1]           = contextDescription->displaySize.height;
    context->constants.displaySizeRcp[0]        = 1.0f / contextDescription->displaySize.width;
    context->constants.displaySizeRcp[1]        = 1.0f / contextDescription->displaySize.height;
    context->constants.interpolationRectBase[0] = 0;
    context->constants.interpolationRectBase[1] = 0;
    context->constants.interpolationRectSize[0] = contextDescription->displaySize.width;
    context->constants.interpolationRectSize[1] = contextDescription->displaySize.height;

    // generate the data for the LUT.
    const uint32_t lanczos2LutWidth = 128;
    int16_t lanczos2Weights[lanczos2LutWidth] = { };

    for (uint32_t currentLanczosWidthIndex = 0; currentLanczosWidthIndex < lanczos2LutWidth; currentLanczosWidthIndex++) {

        const float x = 2.0f * currentLanczosWidthIndex / float(lanczos2LutWidth - 1);
        const float y = lanczos2(x);
        lanczos2Weights[currentLanczosWidthIndex] = int16_t(roundf(y * 32767.0f));
    }

    uint8_t defaultDistortionFieldData[2] = { 0, 0 };
    uint32_t atomicInitData[2] = { 0, 0 };
    float defaultExposure[] = { 0.0f, 0.0f };
    const FfxResourceType texture1dResourceType = (context->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_TEXTURE1D_USAGE) ? FFX_RESOURCE_TYPE_TEXTURE1D : FFX_RESOURCE_TYPE_TEXTURE2D;

    // declare internal resources needed
    const FfxInternalResourceDescription internalSurfaceDesc[] = {

        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME, L"FI_ReconstructedDepthInterpolatedFrame",  FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,      FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_X,             L"FI_GameMotionVectorFieldX",               FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV, 
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,      FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_Y,             L"FI_GameMotionVectorFieldY",               FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,      FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID,                     L"FI_InpaintingPyramid",                    FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, contextDescription->displaySize.width / 2, contextDescription->displaySize.height / 2, 0, FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS,                               L"FI_Counters",                             FFX_RESOURCE_TYPE_BUFFER, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_UNKNOWN, 8, 4, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}}, // structured buffer contraining 2 UINT values
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X,     L"FI_OpticalFlowMotionVectorFieldX",        FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,      FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y,     L"FI_OpticalFlowMotionVectorFieldY",        FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R32_UINT, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,      FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE,          L"FI_PreviousInterpolationSouce",           FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            contextDescription->previousInterpolationSourceFormat, contextDescription->displaySize.width, contextDescription->displaySize.height, 1, FFX_RESOURCE_FLAGS_NONE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_MASK,                        L"FI_InpaintingMask",                       FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV,
            FFX_SURFACE_FORMAT_R8_UNORM, contextDescription->displaySize.width, contextDescription->displaySize.height, 1,          FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK,                      L"FI_DisocclusionMask",                     FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_UAV, 
            FFX_SURFACE_FORMAT_R8G8_UNORM, contextDescription->maxRenderSize.width, contextDescription->maxRenderSize.height, 1,    FFX_RESOURCE_FLAGS_ALIASABLE, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}},
        {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEFAULT_DISTORTION_FIELD, L"FI_DefaultDistortionField", FFX_RESOURCE_TYPE_TEXTURE2D, FFX_RESOURCE_USAGE_READ_ONLY,
            FFX_SURFACE_FORMAT_R8G8_UNORM, 1, 1, 1, FFX_RESOURCE_FLAGS_NONE, FfxResourceInitData::FfxResourceInitBuffer(sizeof(defaultDistortionFieldData), defaultDistortionFieldData) },

    };

    // clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex) {

        const FfxInternalResourceDescription* currentSurfaceDescription = &internalSurfaceDesc[currentSurfaceIndex];
        const FfxResourceDescription          resourceDescription       = {currentSurfaceDescription->type,
                                                                           currentSurfaceDescription->format,
                                                                           currentSurfaceDescription->width,
                                                                           currentSurfaceDescription->height,
                                                                           1,
                                                                           currentSurfaceDescription->mipCount,
                                                                           currentSurfaceDescription->flags,
                                                                           currentSurfaceDescription->usage};
        FfxResourceStates initialState = FFX_RESOURCE_STATE_UNORDERED_ACCESS;
        if (currentSurfaceDescription->usage == FFX_RESOURCE_USAGE_READ_ONLY) initialState = FFX_RESOURCE_STATE_COMPUTE_READ;
        if (currentSurfaceDescription->usage == FFX_RESOURCE_USAGE_RENDERTARGET) initialState = FFX_RESOURCE_STATE_RENDER_TARGET;

        const FfxCreateResourceDescription createResourceDescription = { FFX_HEAP_TYPE_DEFAULT, resourceDescription, initialState, currentSurfaceDescription->name, currentSurfaceDescription->id, currentSurfaceDescription->initData };

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateResource(&context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[currentSurfaceDescription->id]));
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

static FfxErrorCode frameinterpolationRelease(FfxFrameInterpolationContext_Private* context)
{
    FFX_ASSERT(context);

    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiReconstructAndDilate, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiSetup, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiReconstructPreviousDepth, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiGameMotionVectorField, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiOpticalFlowVectorField, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiDisocclusionMask, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineFiScfi, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineInpaintingPyramid, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineInpainting, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineGameVectorFieldInpaintingPyramid, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineDebugView, context->effectContextId);

    // unregister resources not created internally
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]          = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]                   = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_CONFIDENCE]               = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_GLOBAL_MOTION]            = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCENE_CHANGE_DETECTION]   = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]                                = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]                                = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH]                         = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH]                         = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS]                = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS]                = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME]    = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME]    = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISTORTION_FIELD]                      = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};

    // Release the copy resources for those that had init data
    ffxSafeReleaseCopyResource(&context->contextDescription.backendInterface, context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS], context->effectContextId);
    ffxSafeReleaseCopyResource(&context->contextDescription.backendInterface, context->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEFAULT_DISTORTION_FIELD], context->effectContextId);

    // release internal resources
    for (int32_t currentResourceIndex = 0; currentResourceIndex < FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNT; ++currentResourceIndex) {

        ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[currentResourceIndex], context->effectContextId);
    }

    // Destroy the context
    context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId);

    return FFX_OK;
}

static void scheduleDispatch(FfxFrameInterpolationContext_Private* context, const FfxPipelineState* pipeline, uint32_t dispatchX, uint32_t dispatchY)
{
    FfxComputeJobDescription jobDescriptor = {};

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t            currentResourceId               = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = context->srvResources[currentResourceId];
        jobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.srvTextures[currentShaderResourceViewIndex].name, pipeline->srvTextureBindings[currentShaderResourceViewIndex].name);
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex) {

        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name, pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name);
#endif

        if (currentResourceId >= FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_0 && currentResourceId <= FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_12)
        {
            const FfxResourceInternal currentResource = context->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID];
            jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip = currentResourceId - FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID_MIPMAP_0;
        }
        else
        {
            const FfxResourceInternal currentResource = context->uavResources[currentResourceId];
            jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
            jobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip = 0;
        }
    }

    jobDescriptor.dimensions[0] = dispatchX;
    jobDescriptor.dimensions[1] = dispatchY;
    jobDescriptor.dimensions[2] = 1;
    jobDescriptor.pipeline = *pipeline;

    for (uint32_t currentRootConstantIndex = 0; currentRootConstantIndex < pipeline->constCount; ++currentRootConstantIndex) {
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.cbNames[currentRootConstantIndex], pipeline->constantBufferBindings[currentRootConstantIndex].name);
#endif
        jobDescriptor.cbs[currentRootConstantIndex] = context->constantBuffers[pipeline->constantBufferBindings[currentRootConstantIndex].resourceIdentifier];
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavBufferCount; ++currentUnorderedAccessViewIndex)
    {
        const uint32_t currentResourceId = pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
        jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].resource = context->uavResources[currentResourceId];        
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.uavBuffers[currentUnorderedAccessViewIndex].name, pipeline->uavBufferBindings[currentUnorderedAccessViewIndex].name);
#endif
    }

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvBufferCount; ++currentShaderResourceViewIndex)
    {
        const uint32_t currentResourceId = pipeline->srvBufferBindings[currentShaderResourceViewIndex].resourceIdentifier;
        jobDescriptor.srvBuffers[currentShaderResourceViewIndex].resource = context->srvResources[currentResourceId];
#ifdef FFX_DEBUG
        wcscpy_s(jobDescriptor.srvBuffers[currentShaderResourceViewIndex].name, pipeline->srvBufferBindings[currentShaderResourceViewIndex].name);
#endif
    }

    FfxGpuJobDescription dispatchJob = { FFX_GPU_JOB_COMPUTE };
    wcscpy_s(dispatchJob.jobLabel, pipeline->name);
    dispatchJob.computeJobDescriptor = jobDescriptor;

    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

FFX_API FfxErrorCode ffxFrameInterpolationGetSharedResourceDescriptions(FfxFrameInterpolationContext* context, FfxFrameInterpolationSharedResourceDescriptions* SharedResources)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        SharedResources,
        FFX_ERROR_INVALID_POINTER);

    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)(context);
    SharedResources->dilatedDepth = { FFX_HEAP_TYPE_DEFAULT, { FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R32_FLOAT, contextPrivate->contextDescription.maxRenderSize.width, contextPrivate->contextDescription.maxRenderSize.height, 1, 1, FFX_RESOURCE_FLAGS_NONE, (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV | FFX_RESOURCE_USAGE_DCC_RENDERTARGET) },
        FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"FISHARED_DilatedDepth", FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} };
    SharedResources->dilatedMotionVectors = { FFX_HEAP_TYPE_DEFAULT, { FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R16G16_FLOAT, contextPrivate->contextDescription.maxRenderSize.width, contextPrivate->contextDescription.maxRenderSize.height, 1, 1, FFX_RESOURCE_FLAGS_NONE, (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV | FFX_RESOURCE_USAGE_DCC_RENDERTARGET) },
            FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"FISHARED_DilatedVelocity", FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} };
    SharedResources->reconstructedPrevNearestDepth = { FFX_HEAP_TYPE_DEFAULT, { FFX_RESOURCE_TYPE_TEXTURE2D, FFX_SURFACE_FORMAT_R32_UINT, contextPrivate->contextDescription.maxRenderSize.width, contextPrivate->contextDescription.maxRenderSize.height, 1, 1, FFX_RESOURCE_FLAGS_NONE, (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV) },
            FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"FISHARED_ReconstructedPrevNearestDepth", FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME, {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED} };

    return FFX_OK;
}

FfxErrorCode ffxFrameInterpolationContextCreate(FfxFrameInterpolationContext* context, FfxFrameInterpolationContextDescription* contextDescription)
{
    // zero context memory
    //memset(context, 0, sizeof(FfxFrameinterpolationContext));

    // check pointers are valid.
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        contextDescription,
        FFX_ERROR_INVALID_POINTER);

    // validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetSDKVersion, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    // if a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer) {

        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    // ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxFrameInterpolationContext) >= sizeof(FfxFrameInterpolationContext_Private));

    // create the context.
    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)(context);
    FfxErrorCode errorCode = frameinterpolationCreate(contextPrivate, contextDescription);

    return errorCode;
}

FFX_API FfxErrorCode ffxFrameInterpolationContextGetGpuMemoryUsage(FfxFrameInterpolationContext* context, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);
    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);

    FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

FFX_API FfxErrorCode ffxSharedContextGetGpuMemoryUsage(FfxInterface* backendInterfaceShared, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(backendInterfaceShared, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);

    FfxErrorCode errorCode = backendInterfaceShared->fpGetEffectGpuMemoryUsage(
        backendInterfaceShared, 0, vramUsage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

FfxErrorCode ffxFrameInterpolationContextDestroy(FfxFrameInterpolationContext* context)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);

    // destroy the context.
    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)(context);
    const FfxErrorCode      errorCode      = frameinterpolationRelease(contextPrivate);

    return errorCode;
}

FfxErrorCode ffxFrameInterpolationContextEnqueueRefreshPipelineRequest(FfxFrameInterpolationContext* context)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);

    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)context;
    contextPrivate->refreshPipelineStates = true;

    return FFX_OK;
}

static void setupDeviceDepthToViewSpaceDepthParams(FfxFrameInterpolationContext_Private* context, const FfxFrameInterpolationRenderDescription* params, FrameInterpolationConstants* constants)
{
    const bool bInverted = (context->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED) == FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
    const bool bInfinite = (context->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE) == FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE;

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
        fQ * fMin,              // non reversed, non infinite
        -fMin - FLT_EPSILON,    // non reversed, infinite
        fQ * fMin,              // reversed, non infinite
        fMax,                   // reversed, infinite
    };

    constants->deviceToViewDepth[0] = d * matrix_elem_c[bInverted][bInfinite];
    constants->deviceToViewDepth[1] = matrix_elem_e[bInverted][bInfinite] * params->viewSpaceToMetersFactor;

    // revert x and y coords
    const float aspect = params->renderSize.width / float(params->renderSize.height);
    const float cotHalfFovY = cosf(0.5f * params->cameraFovAngleVertical) / sinf(0.5f * params->cameraFovAngleVertical);
    const float a = cotHalfFovY / aspect;
    const float b = cotHalfFovY;

    constants->deviceToViewDepth[2] = (1.0f / a);
    constants->deviceToViewDepth[3] = (1.0f / b);
    
}

FFX_API bool ffxFrameInterpolationResourceIsNull(FfxResource resource)
{
    return resource.resource == NULL;
}

static const float debugBarColorSequence[] = {
    0.0f, 1.0f, 1.0f,   // teal
    1.0f, 0.42f, 0.0f,  // orange
    0.0f, 0.16f, 1.0f,  // blue
    0.74f, 1.0f, 0.0f,  // lime
    0.68f, 0.0f, 1.0f,  // purple
    0.0f, 1.0f, 0.1f,   // green
    1.0f, 1.0f, 0.48f   // bright yellow
};
const size_t debugBarColorSequenceLength = 7;

static void fsr3FrameInterpolationDebugCheckPrepare(FfxFrameInterpolationContext_Private* context, const FfxFrameInterpolationPrepareDescription* params)
{
    
    static const FfxFloat32x3 zeroVector3D = { 0.f,0.f,0.f };
    if ((memcmp(params->cameraPosition, zeroVector3D, sizeof(FfxFloat32x3)) == 0) &&
        (memcmp(params->cameraUp, zeroVector3D, sizeof(FfxFloat32x3)) == 0) &&
        (memcmp(params->cameraRight, zeroVector3D, sizeof(FfxFloat32x3)) == 0) &&
        (memcmp(params->cameraForward, zeroVector3D, sizeof(FfxFloat32x3)) == 0))
    {
        FFX_PRINT_MESSAGE(FFX_MESSAGE_TYPE_WARNING, L"ffxDispatchDescFrameGenerationPrepareCameraInfo needs to be passed as linked struct. This is a required input to FSR3.1.4 and onwards for best quality.");
    }
}

FFX_API FfxErrorCode ffxFrameInterpolationPrepare(FfxFrameInterpolationContext* context,
    const FfxFrameInterpolationPrepareDescription* params)
{
    FfxFrameInterpolationContext_Private* contextPrivate = (FfxFrameInterpolationContext_Private*)(context);

    if ((contextPrivate->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DEBUG_CHECKING) == FFX_FRAMEINTERPOLATION_ENABLE_DEBUG_CHECKING)
    {
        fsr3FrameInterpolationDebugCheckPrepare(contextPrivate, params);
    }
    
    contextPrivate->constants.renderSize[0]         = params->renderSize.width;
    contextPrivate->constants.renderSize[1]         = params->renderSize.height;
    contextPrivate->constants.jitter[0]             = params->jitterOffset.x;
    contextPrivate->constants.jitter[1]             = params->jitterOffset.y;

    const int32_t* motionVectorsTargetSize          = (contextPrivate->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS)
                                                        ? contextPrivate->constants.displaySize
                                                        : contextPrivate->constants.renderSize;
    contextPrivate->constants.motionVectorScale[0]  = (params->motionVectorScale.x / motionVectorsTargetSize[0]);
    contextPrivate->constants.motionVectorScale[1]  = (params->motionVectorScale.y / motionVectorsTargetSize[1]);

    contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &contextPrivate->contextDescription.backendInterface,
        &contextPrivate->constants,
        sizeof(contextPrivate->constants),
        &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER]);

    FFX_ASSERT(!ffxFrameInterpolationResourceIsNull(params->depth));
    FFX_ASSERT(!ffxFrameInterpolationResourceIsNull(params->motionVectors));

    contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->depth,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH]);

    contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->motionVectors,
        contextPrivate->effectContextId,
        &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS]);

    contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->dilatedDepth,
        contextPrivate->effectContextId,
        &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH]);
    contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->dilatedMotionVectors,
        contextPrivate->effectContextId,
        &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS]);
    contextPrivate->contextDescription.backendInterface.fpRegisterResource(
        &contextPrivate->contextDescription.backendInterface,
        &params->reconstructedPrevDepth,
        contextPrivate->effectContextId,
        &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME]);

    // clear estimated depth resources
    {
        FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};
        const bool bInverted = (contextPrivate->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED) == FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
        const float clearDepthValue[]{bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f};
        memcpy(clearJob.clearJobDescriptor.color, clearDepthValue, 4 * sizeof(float));
        wcscpy_s(clearJob.jobLabel, L"Clear Reconstructed Previous Nearest Depth");
        clearJob.clearJobDescriptor.target = contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME];
        contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &clearJob);
    }

    uint32_t                              renderDispatchSizeX = uint32_t(params->renderSize.width + 7) / 8;
    uint32_t                              renderDispatchSizeY = uint32_t(params->renderSize.height + 7) / 8;

    scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiReconstructAndDilate, renderDispatchSizeX, renderDispatchSizeY);

    contextPrivate->contextDescription.backendInterface.fpExecuteGpuJobs(&contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId);

    // release dynamic resources
    contextPrivate->contextDescription.backendInterface.fpUnregisterResources(&contextPrivate->contextDescription.backendInterface,
                                                                              params->commandList,
                                                                              contextPrivate->effectContextId);

    contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEPTH]                              = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_MOTION_VECTORS]                     = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH]                      = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS]             = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};
    contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME] = {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_NULL};

    return FFX_OK;
}

FFX_API FfxErrorCode ffxFrameInterpolationDispatch(FfxFrameInterpolationContext* context, const FfxFrameInterpolationDispatchDescription* params)
{
    FfxFrameInterpolationContext_Private*         contextPrivate = (FfxFrameInterpolationContext_Private*)(context);
    const FfxFrameInterpolationRenderDescription* renderDesc     = &contextPrivate->renderDescription;

    if (contextPrivate->refreshPipelineStates) {

        createPipelineStates(contextPrivate);
        contextPrivate->refreshPipelineStates = false;
    }

    const bool bReset = (contextPrivate->dispatchCount == 0) || params->reset;

    FFX_ASSERT_MESSAGE(!contextPrivate->asyncSupported || bReset || (params->frameID > contextPrivate->previousFrameID),
                       "When async support is enabled, and the reset flag is not set, frame ID must increment in each dispatch");

    // Detect disjoint frameID values
    const bool bFrameID_Decreased   = params->frameID < contextPrivate->previousFrameID;
    const bool bFrameID_Skipped     = (params->frameID - contextPrivate->previousFrameID) > 1;
    const bool bDisjointFrameID     = bFrameID_Decreased || bFrameID_Skipped;
    contextPrivate->previousFrameID = params->frameID;
    contextPrivate->dispatchCount++;

    contextPrivate->constants.renderSize[0]         = params->renderSize.width;
    contextPrivate->constants.renderSize[1]         = params->renderSize.height;
    contextPrivate->constants.displaySize[0]        = params->displaySize.width;
    contextPrivate->constants.displaySize[1]        = params->displaySize.height;
    contextPrivate->constants.displaySizeRcp[0]     = 1.0f / params->displaySize.width;
    contextPrivate->constants.displaySizeRcp[1]     = 1.0f / params->displaySize.height;
    contextPrivate->constants.upscalerTargetSize[0] = params->interpolationRect.width;
    contextPrivate->constants.upscalerTargetSize[1] = params->interpolationRect.height;
    contextPrivate->constants.Mode                  = 0;
    contextPrivate->constants.Reset                 = bReset || bDisjointFrameID;
    contextPrivate->constants.deltaTime             = params->frameTimeDelta;
    contextPrivate->constants.HUDLessAttachedFactor = params->currentBackBuffer_HUDLess.resource ? 1 : 0;

    contextPrivate->constants.opticalFlowScale[0]   = params->opticalFlowScale.x;
    contextPrivate->constants.opticalFlowScale[1]   = params->opticalFlowScale.y;
    contextPrivate->constants.opticalFlowBlockSize  = params->opticalFlowBlockSize;// displaySize.width / params->opticalFlowBufferSize.width;
    contextPrivate->constants.dispatchFlags         = params->flags;

    contextPrivate->constants.cameraNear            = params->cameraNear;
    contextPrivate->constants.cameraFar             = params->cameraFar;

    contextPrivate->constants.interpolationRectBase[0] = params->interpolationRect.left;
    contextPrivate->constants.interpolationRectBase[1] = params->interpolationRect.top;
    contextPrivate->constants.interpolationRectSize[0] = params->interpolationRect.width;
    contextPrivate->constants.interpolationRectSize[1] = params->interpolationRect.height;

    // Debug bar
    static size_t dbgIdx = 0;
    memcpy(contextPrivate->constants.debugBarColor, &debugBarColorSequence[dbgIdx * 3], 3 * sizeof(float));
    dbgIdx = (dbgIdx + 1) % debugBarColorSequenceLength;

    contextPrivate->constants.backBufferTransferFunction = params->backBufferTransferFunction;
    contextPrivate->constants.minMaxLuminance[0]         = params->minMaxLuminance[0];
    contextPrivate->constants.minMaxLuminance[1]         = params->minMaxLuminance[1];

    const float aspectRatio                             = (float)params->renderSize.width / (float)params->renderSize.height;
    const float cameraAngleHorizontal                   = atan(tan(params->cameraFovAngleVertical / 2) * aspectRatio) * 2;
    contextPrivate->constants.fTanHalfFOV               = tanf(cameraAngleHorizontal * 0.5f);

    const bool bUseExternalDistortionFieldResource = !ffxFrameInterpolationResourceIsNull(params->distortionField);
    if (bUseExternalDistortionFieldResource)
    {
        contextPrivate->constants.distortionFieldSize[0] = params->distortionField.description.width;
        contextPrivate->constants.distortionFieldSize[1] = params->distortionField.description.height;
    }
    else
    {
        contextPrivate->constants.distortionFieldSize[0] = 1;
        contextPrivate->constants.distortionFieldSize[1] = 1;
    }

    contextPrivate->renderDescription.cameraFar                 = params->cameraFar;
    contextPrivate->renderDescription.cameraNear                = params->cameraNear;
    contextPrivate->renderDescription.viewSpaceToMetersFactor = (params->viewSpaceToMetersFactor > 0.0f) ? params->viewSpaceToMetersFactor : 1.0f;
    contextPrivate->renderDescription.cameraFovAngleVertical    = params->cameraFovAngleVertical;
    contextPrivate->renderDescription.renderSize                = params->renderSize;
    contextPrivate->renderDescription.upscaleSize               = params->displaySize;
    setupDeviceDepthToViewSpaceDepthParams(contextPrivate, renderDesc, &contextPrivate->constants);

    contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &contextPrivate->contextDescription.backendInterface,
        &contextPrivate->constants,
        sizeof(contextPrivate->constants),
        &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_CONSTANTBUFFER_IDENTIFIER]);

    if (contextPrivate->constants.HUDLessAttachedFactor == 1) {

        FFX_ASSERT_MESSAGE(contextPrivate->contextDescription.previousInterpolationSourceFormat == params->currentBackBuffer_HUDLess.description.format,
                           "Dispatch FI param currentBackBuffer_HUDLess format and Create FG Context's hudlessBackBufferFormat have to be identical. Otherwise, CopyTextureRegion from FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE to FI_PreviousInterpolationSource would fail");

        contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->currentBackBuffer, contextPrivate->effectContextId, &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PRESENT_BACKBUFFER]);
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->currentBackBuffer_HUDLess, contextPrivate->effectContextId, &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]);
    }
    else {
        FFX_ASSERT_MESSAGE(contextPrivate->contextDescription.previousInterpolationSourceFormat == params->currentBackBuffer.description.format,
                           "Dispatch FI param currentBackBuffer format and Create FG Context's backBufferFormat have to be identical. This assert can also be triggered if create FG Context with optional hudlessBackBufferFormat that is different from backBufferFormat and Dispatch FI param's currentBackBuffer_HUDLess is null.");
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->currentBackBuffer, contextPrivate->effectContextId, &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE]);
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PRESENT_BACKBUFFER] = contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE];
    }

    if (!ffxFrameInterpolationResourceIsNull(params->dilatedDepth))
    {
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->dilatedDepth,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_DEPTH]);
    }
    if (!ffxFrameInterpolationResourceIsNull(params->dilatedMotionVectors))
    {
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->dilatedMotionVectors,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS]);
    }
    if (!ffxFrameInterpolationResourceIsNull(params->reconstructedPrevDepth))
    {
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->reconstructedPrevDepth,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_PREVIOUS_FRAME]);
    }

    // Register output as SRV and UAV
    contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->output, contextPrivate->effectContextId, &contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT]);
    contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT] = contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OUTPUT];

    // set optical flow buffers
    if (params->opticalFlowScale.x > 0)
    {
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->opticalFlowVector, contextPrivate->effectContextId, &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(&contextPrivate->contextDescription.backendInterface, &params->opticalFlowSceneChangeDetection, contextPrivate->effectContextId, &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCENE_CHANGE_DETECTION]);
    }
    else
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_CONFIDENCE]             = {};
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_GLOBAL_MOTION]          = {};
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCENE_CHANGE_DETECTION] = {};
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR] = {};
    }

    if (bUseExternalDistortionFieldResource)
    {
        contextPrivate->contextDescription.backendInterface.fpRegisterResource(
            &contextPrivate->contextDescription.backendInterface,
            &params->distortionField,
            contextPrivate->effectContextId,
            &contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISTORTION_FIELD]);
    }
    else
    {
        contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISTORTION_FIELD] = contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DEFAULT_DISTORTION_FIELD];
    }

    uint32_t displayDispatchSizeX = uint32_t(params->displaySize.width + 7) / 8;
    uint32_t displayDispatchSizeY = uint32_t(params->displaySize.height + 7) / 8;

    uint32_t renderDispatchSizeX = uint32_t(params->renderSize.width + 7) / 8;
    uint32_t renderDispatchSizeY = uint32_t(params->renderSize.height + 7) / 8;

    uint32_t opticalFlowDispatchSizeX = uint32_t(params->displaySize.width / float(params->opticalFlowBlockSize) + 7) / 8;
    uint32_t opticalFlowDispatchSizeY = uint32_t(params->displaySize.height / float(params->opticalFlowBlockSize) + 7) / 8;

    const bool bExecutePreparationPasses = (false == contextPrivate->constants.Reset);

    // Schedule work for the interpolation command list
    {
        FfxResourceInternal aliasableResources[] = {
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_X],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_Y],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_MASK],
            contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK]
        };
        for (int i = 0; i < _countof(aliasableResources); ++i)
        {
            FfxGpuJobDescription discardJob        = {FFX_GPU_JOB_DISCARD};
            discardJob.discardJobDescriptor.target = aliasableResources[i];
            contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &discardJob);
        }

        scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiSetup, renderDispatchSizeX, renderDispatchSizeY);

            // game vector field inpainting pyramid
        auto scheduleDispatchGameVectorFieldInpaintingPyramid = [&]() {
            // Auto exposure
            uint32_t dispatchThreadGroupCountXY[2];
            uint32_t workGroupOffset[2];
            uint32_t numWorkGroupsAndMips[2];
            uint32_t rectInfo[4] = {0, 0, params->renderSize.width, params->renderSize.height};
            ffxSpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo);

            // downsample
            contextPrivate->inpaintingPyramidContants.numworkGroups      = numWorkGroupsAndMips[0];
            contextPrivate->inpaintingPyramidContants.mips               = numWorkGroupsAndMips[1];
            contextPrivate->inpaintingPyramidContants.workGroupOffset[0] = workGroupOffset[0];
            contextPrivate->inpaintingPyramidContants.workGroupOffset[1] = workGroupOffset[1];

            contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
                &contextPrivate->contextDescription.backendInterface,
                &contextPrivate->inpaintingPyramidContants,
                sizeof(contextPrivate->inpaintingPyramidContants),
                &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_INPAINTING_PYRAMID_CONSTANTBUFFER_IDENTIFIER]);

            scheduleDispatch(
                contextPrivate, &contextPrivate->pipelineGameVectorFieldInpaintingPyramid, dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1]);
        };

        // only execute FG data preparation passes when reset wasnt triggered
        if (bExecutePreparationPasses)
        {
            // clear estimated depth resources
            {
                FfxGpuJobDescription clearJob = {FFX_GPU_JOB_CLEAR_FLOAT};

                const bool bInverted =
                    (contextPrivate->contextDescription.flags & FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED) == FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
                const float clearDepthValue[]{bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f, bInverted ? 0.f : 1.f};
                memcpy(clearJob.clearJobDescriptor.color, clearDepthValue, 4 * sizeof(float));

                wcscpy_s(clearJob.jobLabel, L"Clear Reconstructed Depth Interpolated Frame");
                clearJob.clearJobDescriptor.target =
                    contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME];
                contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &clearJob);
            }

            scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiReconstructPreviousDepth, renderDispatchSizeX, renderDispatchSizeY);
            scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiGameMotionVectorField, renderDispatchSizeX, renderDispatchSizeY);

            scheduleDispatchGameVectorFieldInpaintingPyramid();

            scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiOpticalFlowVectorField, opticalFlowDispatchSizeX, opticalFlowDispatchSizeY);

            scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiDisocclusionMask, renderDispatchSizeX, renderDispatchSizeY);
        }

        scheduleDispatch(contextPrivate, &contextPrivate->pipelineFiScfi, displayDispatchSizeX, displayDispatchSizeY);

        // inpainting pyramid
        {
            // Auto exposure
            uint32_t dispatchThreadGroupCountXY[2];
            uint32_t workGroupOffset[2];
            uint32_t numWorkGroupsAndMips[2];
            uint32_t rectInfo[4] = { 0, 0, params->displaySize.width, params->displaySize.height };
            ffxSpdSetup(dispatchThreadGroupCountXY, workGroupOffset, numWorkGroupsAndMips, rectInfo);

            // downsample
            contextPrivate->inpaintingPyramidContants.numworkGroups      = numWorkGroupsAndMips[0];
            contextPrivate->inpaintingPyramidContants.mips               = numWorkGroupsAndMips[1];
            contextPrivate->inpaintingPyramidContants.workGroupOffset[0] = workGroupOffset[0];
            contextPrivate->inpaintingPyramidContants.workGroupOffset[1] = workGroupOffset[1];

            contextPrivate->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
                &contextPrivate->contextDescription.backendInterface,
                &contextPrivate->inpaintingPyramidContants,
                sizeof(contextPrivate->inpaintingPyramidContants),
                &contextPrivate->constantBuffers[FFX_FRAMEINTERPOLATION_INPAINTING_PYRAMID_CONSTANTBUFFER_IDENTIFIER]);

            scheduleDispatch(contextPrivate, &contextPrivate->pipelineInpaintingPyramid, dispatchThreadGroupCountXY[0], dispatchThreadGroupCountXY[1]);
        }

        scheduleDispatch(contextPrivate, &contextPrivate->pipelineInpainting, displayDispatchSizeX, displayDispatchSizeY);

        if (params->flags & FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW)
        {
            scheduleDispatchGameVectorFieldInpaintingPyramid();
            scheduleDispatch(contextPrivate, &contextPrivate->pipelineDebugView, displayDispatchSizeX, displayDispatchSizeY);
        }

        // store current buffer
        {
            FfxGpuJobDescription copyJobs[] = { {FFX_GPU_JOB_COPY} };
            FfxResourceInternal  copySources[_countof(copyJobs)] = { contextPrivate->srvResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_CURRENT_INTERPOLATION_SOURCE] };
            FfxResourceInternal destSources[_countof(copyJobs)] = { contextPrivate->uavResources[FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE] };

            for (int i = 0; i < _countof(copyJobs); ++i)
            {
                copyJobs[i].copyJobDescriptor.src = copySources[i];
                copyJobs[i].copyJobDescriptor.dst = destSources[i];
                contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &copyJobs[i]);
            }
        }

        // declare internal resources needed
        struct FfxInternalResourceStates
        {
            FfxUInt32 id;
            FfxResourceUsage usage;
        };
        const FfxInternalResourceStates internalSurfaceDesc[] = {

            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_RECONSTRUCTED_DEPTH_INTERPOLATED_FRAME, FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_X,             FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_GAME_MOTION_VECTOR_FIELD_Y,             FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_INPAINTING_PYRAMID,                     FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS,                               FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X,     FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y,     FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_PREVIOUS_INTERPOLATION_SOURCE,          FFX_RESOURCE_USAGE_UAV},
            {FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_DISOCCLUSION_MASK,                      FFX_RESOURCE_USAGE_UAV},
        };

        for (int32_t currentSurfaceIndex = 0; currentSurfaceIndex < FFX_ARRAY_ELEMENTS(internalSurfaceDesc); ++currentSurfaceIndex) {

            const FfxInternalResourceStates* currentSurfaceDescription = &internalSurfaceDesc[currentSurfaceIndex];
            FfxResourceStates initialState = FFX_RESOURCE_STATE_UNORDERED_ACCESS;
            if (currentSurfaceDescription->usage == FFX_RESOURCE_USAGE_READ_ONLY) initialState = FFX_RESOURCE_STATE_COMPUTE_READ;
            if (currentSurfaceDescription->usage == FFX_RESOURCE_USAGE_RENDERTARGET) initialState = FFX_RESOURCE_STATE_RENDER_TARGET;

            FfxGpuJobDescription barrier = {FFX_GPU_JOB_BARRIER};
            barrier.barrierDescriptor.resource = contextPrivate->srvResources[currentSurfaceDescription->id];
            barrier.barrierDescriptor.subResourceID = 0;
            barrier.barrierDescriptor.newState = (currentSurfaceDescription->id == FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_COUNTERS) ? FFX_RESOURCE_STATE_COPY_DEST : initialState;
            barrier.barrierDescriptor.barrierType = FFX_BARRIER_TYPE_TRANSITION;
            contextPrivate->contextDescription.backendInterface.fpScheduleGpuJob(&contextPrivate->contextDescription.backendInterface, &barrier);
        }

        // schedule optical flow and frame interpolation
        contextPrivate->contextDescription.backendInterface.fpExecuteGpuJobs(&contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId);
    }

    // release dynamic resources
    contextPrivate->contextDescription.backendInterface.fpUnregisterResources(&contextPrivate->contextDescription.backendInterface, params->commandList, contextPrivate->effectContextId);

    return FFX_OK;
}

FFX_API FfxVersionNumber ffxFrameInterpolationGetEffectVersion()
{
    return FFX_SDK_MAKE_VERSION(FFX_FRAMEINTERPOLATION_VERSION_MAJOR, FFX_FRAMEINTERPOLATION_VERSION_MINOR, FFX_FRAMEINTERPOLATION_VERSION_PATCH);
}

FFX_API FfxErrorCode ffxFrameInterpolationSetGlobalDebugMessage(ffxMessageCallback fpMessage, uint32_t debugLevel)
{
    ffxSetPrintMessageCallback(fpMessage, debugLevel);
    return FFX_OK;
}
