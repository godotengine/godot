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

#include <string.h>     // for memset
#include <stdlib.h>     // for _countof
#include <cmath>        // for fabs, abs, sinf, sqrt, etc.

#ifdef __clang__
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#ifdef _MSC_VER
#pragma warning(disable : 4505)
#endif

#include "ffx_fsr1.h"
#include "gpu/ffx_core.h"
#include "gpu/fsr1/ffx_fsr1.h"
#include "ffx_object_management.h"

#include "ffx_fsr1_private.h"

// lists to map shader resource bindpoint name to resource identifier
typedef struct ResourceBinding
{
    uint32_t    index;
    wchar_t     name[64];
}ResourceBinding;

static const ResourceBinding srvTextureBindingTable[] =
{
    {FFX_FSR1_RESOURCE_IDENTIFIER_INPUT_COLOR,                  L"r_input_color"},
    {FFX_FSR1_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,      L"r_internal_upscaled_color"},
    {FFX_FSR1_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT,              L"r_upscaled_output" },
};

static const ResourceBinding uavTextureBindingTable[] =
{
    {FFX_FSR1_RESOURCE_IDENTIFIER_INPUT_COLOR,                  L"rw_input_color"},
    {FFX_FSR1_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,      L"rw_internal_upscaled_color"},
    {FFX_FSR1_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT,              L"rw_upscaled_output"},
};

static const ResourceBinding cbResourceBindingTable[] =
{
    {FFX_FSR1_CONSTANTBUFFER_IDENTIFIER_FSR1,                   L"cbFSR1"},
};

static FfxErrorCode patchResourceBindings(FfxPipelineState* inoutPipeline)
{
    for (uint32_t srvIndex = 0; srvIndex < inoutPipeline->srvTextureCount; ++srvIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(srvTextureBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(srvTextureBindingTable[mapIndex].name, inoutPipeline->srvTextureBindings[srvIndex].name))
                break;
        }
        if (mapIndex == _countof(srvTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->srvTextureBindings[srvIndex].resourceIdentifier = srvTextureBindingTable[mapIndex].index;
    }

    for (uint32_t uavIndex = 0; uavIndex < inoutPipeline->uavTextureCount; ++uavIndex)
    {
        int32_t mapIndex = 0;
        for (mapIndex = 0; mapIndex < _countof(uavTextureBindingTable); ++mapIndex)
        {
            if (0 == wcscmp(uavTextureBindingTable[mapIndex].name, inoutPipeline->uavTextureBindings[uavIndex].name))
                break;
        }
        if (mapIndex == _countof(uavTextureBindingTable))
            return FFX_ERROR_INVALID_ARGUMENT;

        inoutPipeline->uavTextureBindings[uavIndex].resourceIdentifier = uavTextureBindingTable[mapIndex].index;
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

    return FFX_OK;
}

static uint32_t getPipelinePermutationFlags(uint32_t contextFlags, FfxFsr1Pass passId, bool fp16, bool force64)
{
    // work out what permutation to load.
    uint32_t flags = 0;
    flags |= (contextFlags & FFX_FSR1_RCAS_PASSTHROUGH_ALPHA) ? FSR1_SHADER_PERMUTATION_RCAS_PASSTHROUGH_ALPHA : 0;
    flags |= (contextFlags & FFX_FSR1_ENABLE_SRGB_CONVERSIONS) ? FSR1_SHADER_PERMUTATION_SRGB_CONVERSIONS : 0;
    flags |= (passId != FFX_FSR1_PASS_EASU) ? FSR1_SHADER_PERMUTATION_APPLY_RCAS : 0;
    flags |= (force64) ? FSR1_SHADER_PERMUTATION_FORCE_WAVE64 : 0;
#if defined(_GAMING_XBOX_SCARLETT)
    // Never got reports about NaNs on Xbox
    flags |= (fp16) ? FSR1_SHADER_PERMUTATION_ALLOW_FP16 : 0;
#else
    // Some NaNs have been observed on other hardware during Rcas with FP16
    flags |= (fp16 && (passId != FFX_FSR1_PASS_RCAS)) ? FSR1_SHADER_PERMUTATION_ALLOW_FP16 : 0;
#endif
    return flags;
}

static FfxErrorCode createPipelineStates(FfxFsr1Context_Private* context)
{
    FFX_ASSERT(context);

    FfxPipelineDescription pipelineDescription = {};
    pipelineDescription.contextFlags = context->contextDescription.flags;

    // Samplers
    pipelineDescription.samplerCount = 1;
    FfxSamplerDescription samplerDesc = { FFX_FILTER_TYPE_MINMAGMIP_LINEAR, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_ADDRESS_MODE_CLAMP, FFX_BIND_COMPUTE_SHADER_STAGE };
    pipelineDescription.samplers = &samplerDesc;

    // Root constants
    pipelineDescription.rootConstantBufferCount = 1;
    FfxRootConstantDescription rootConstantDesc = { sizeof(Fsr1Constants) / sizeof(uint32_t), FFX_BIND_COMPUTE_SHADER_STAGE };
    pipelineDescription.rootConstants = &rootConstantDesc;

    // Query device capabilities
    FfxDeviceCapabilities capabilities;
    context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &capabilities);

    // Setup a few options used to determine permutation flags
    bool haveShaderModel66 = capabilities.maximumSupportedShaderModel >= FFX_SHADER_MODEL_6_6;
    bool supportedFP16 = capabilities.fp16Supported;
    bool canForceWave64 = false;

    const uint32_t waveLaneCountMin = capabilities.waveLaneCountMin;
    const uint32_t waveLaneCountMax = capabilities.waveLaneCountMax;
    if (waveLaneCountMin <= 64 && waveLaneCountMax >= 64)
        canForceWave64 = haveShaderModel66;
    else
        canForceWave64 = false;

    // Work out what permutation to load.
    uint32_t contextFlags = context->contextDescription.flags;

    // Set up pipeline descriptors (basically RootSignature and binding)
    wcscpy_s(pipelineDescription.name, L"FSR1-EASU");
    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(&context->contextDescription.backendInterface, FFX_EFFECT_FSR1, FFX_FSR1_PASS_EASU,
        getPipelinePermutationFlags(contextFlags, FFX_FSR1_PASS_EASU, supportedFP16, canForceWave64),
        &pipelineDescription, context->effectContextId, &context->pipelineEASU));
    wcscpy_s(pipelineDescription.name, L"FSR1-EASU_RCAS");
    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(&context->contextDescription.backendInterface, FFX_EFFECT_FSR1, FFX_FSR1_PASS_EASU_RCAS,
        getPipelinePermutationFlags(contextFlags, FFX_FSR1_PASS_EASU_RCAS, supportedFP16, canForceWave64),
        &pipelineDescription, context->effectContextId, &context->pipelineEASU_RCAS));
    wcscpy_s(pipelineDescription.name, L"FSR1-RCAS");
    FFX_VALIDATE(context->contextDescription.backendInterface.fpCreatePipeline(&context->contextDescription.backendInterface, FFX_EFFECT_FSR1, FFX_FSR1_PASS_RCAS,
        getPipelinePermutationFlags(contextFlags, FFX_FSR1_PASS_RCAS, supportedFP16, canForceWave64),
        &pipelineDescription, context->effectContextId, &context->pipelineRCAS));

    // For each pipeline: re-route/fix-up IDs based on names
    patchResourceBindings(&context->pipelineEASU);
    patchResourceBindings(&context->pipelineEASU_RCAS);
    patchResourceBindings(&context->pipelineRCAS);

    return FFX_OK;
}

static void scheduleDispatch(FfxFsr1Context_Private* context, const FfxFsr1DispatchDescription*, const FfxPipelineState* pipeline, uint32_t dispatchX, uint32_t dispatchY)
{
    FfxGpuJobDescription dispatchJob = {FFX_GPU_JOB_COMPUTE};
    wcscpy_s(dispatchJob.jobLabel, pipeline->name);

    for (uint32_t currentShaderResourceViewIndex = 0; currentShaderResourceViewIndex < pipeline->srvTextureCount; ++currentShaderResourceViewIndex) {

        const uint32_t currentResourceId = pipeline->srvTextureBindings[currentShaderResourceViewIndex].resourceIdentifier;
        const FfxResourceInternal currentResource = context->srvResources[currentResourceId];
        dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].resource = currentResource;
#ifdef FFX_DEBUG
        wcscpy_s(dispatchJob.computeJobDescriptor.srvTextures[currentShaderResourceViewIndex].name,
                 pipeline->srvTextureBindings[currentShaderResourceViewIndex].name);
#endif
    }

    for (uint32_t currentUnorderedAccessViewIndex = 0; currentUnorderedAccessViewIndex < pipeline->uavTextureCount; ++currentUnorderedAccessViewIndex) {

        const uint32_t currentResourceId = pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].resourceIdentifier;
#ifdef FFX_DEBUG
        wcscpy_s(dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].name,
                 pipeline->uavTextureBindings[currentUnorderedAccessViewIndex].name);
#endif
        const FfxResourceInternal currentResource                       = context->uavResources[currentResourceId];
        dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].resource = currentResource;
        dispatchJob.computeJobDescriptor.uavTextures[currentUnorderedAccessViewIndex].mip = 0;
    }

    dispatchJob.computeJobDescriptor.dimensions[0] = dispatchX;
    dispatchJob.computeJobDescriptor.dimensions[1] = dispatchY;
    dispatchJob.computeJobDescriptor.dimensions[2] = 1;
    dispatchJob.computeJobDescriptor.pipeline      = *pipeline;

#ifdef FFX_DEBUG
    wcscpy_s(dispatchJob.computeJobDescriptor.cbNames[0], pipeline->constantBufferBindings[0].name);
#endif
    dispatchJob.computeJobDescriptor.cbs[0] = context->constantBuffer;


    context->contextDescription.backendInterface.fpScheduleGpuJob(&context->contextDescription.backendInterface, &dispatchJob);
}

static FfxErrorCode fsr1Dispatch(FfxFsr1Context_Private* context, const FfxFsr1DispatchDescription* params)
{
    // take a short cut to the command list
    FfxCommandList commandList = params->commandList;

    // Register resources for frame
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface, &params->color, context->effectContextId, &context->srvResources[FFX_FSR1_RESOURCE_IDENTIFIER_INPUT_COLOR]);
    context->contextDescription.backendInterface.fpRegisterResource(&context->contextDescription.backendInterface, &params->output, context->effectContextId, &context->uavResources[FFX_FSR1_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT]);

    // This value is the image region dimension that each thread group of the FSR shader operates on
    static const int threadGroupWorkRegionDim = 16;
    int dispatchX = FFX_DIVIDE_ROUNDING_UP(context->contextDescription.displaySize.width, threadGroupWorkRegionDim);
    int dispatchY = FFX_DIVIDE_ROUNDING_UP(context->contextDescription.displaySize.height, threadGroupWorkRegionDim);

    const bool doSharpen = params->enableSharpening && (context->contextDescription.flags & FFX_FSR1_ENABLE_RCAS);

    // Easu constants
    Fsr1Constants easuConst = {};
    ffxFsrPopulateEasuConstants(reinterpret_cast<FfxUInt32*>(&easuConst.const0),
        reinterpret_cast<FfxUInt32*>(&easuConst.const1),
        reinterpret_cast<FfxUInt32*>(&easuConst.const2),
        reinterpret_cast<FfxUInt32*>(&easuConst.const3),
        static_cast<FfxFloat32>(params->renderSize.width), static_cast<FfxFloat32>(params->renderSize.height),
        static_cast<FfxFloat32>(params->color.description.width), static_cast<FfxFloat32>(params->color.description.height),
        static_cast<FfxFloat32>(context->contextDescription.displaySize.width),
        static_cast<FfxFloat32>(context->contextDescription.displaySize.height));
    easuConst.sample[0] = context->contextDescription.flags & FFX_FSR1_ENABLE_HIGH_DYNAMIC_RANGE;
    context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
        &context->contextDescription.backendInterface,
        &easuConst,
        sizeof(Fsr1Constants),
        &context->constantBuffer);
    scheduleDispatch(context, params, doSharpen ? &context->pipelineEASU_RCAS : &context->pipelineEASU, dispatchX, dispatchY);

    if (doSharpen)
    {
        // Rcas constants
        Fsr1Constants rcasConst = {};
        const float sharpenessRemapped = (-2.0f * params->sharpness) + 2.0f;
        FsrRcasCon(reinterpret_cast<FfxUInt32*>(&rcasConst.const0), sharpenessRemapped);
        rcasConst.sample[0] = context->contextDescription.flags & FFX_FSR1_ENABLE_HIGH_DYNAMIC_RANGE;
        context->contextDescription.backendInterface.fpStageConstantBufferDataFunc(
            &context->contextDescription.backendInterface,
            &rcasConst,
            sizeof(Fsr1Constants),
            &context->constantBuffer);
        scheduleDispatch(context, params, &context->pipelineRCAS, dispatchX, dispatchY);
    }

    // Execute all the work for the frame
    context->contextDescription.backendInterface.fpExecuteGpuJobs(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    // Release dynamic resources
    context->contextDescription.backendInterface.fpUnregisterResources(&context->contextDescription.backendInterface, commandList, context->effectContextId);

    return FFX_OK;
}

static FfxErrorCode fsr1Create(FfxFsr1Context_Private* context, const FfxFsr1ContextDescription* contextDescription)
{
    FFX_ASSERT(context);
    FFX_ASSERT(contextDescription);

    // Setup the data for implementation.
    memset(context, 0, sizeof(FfxFsr1Context_Private));
    context->device = contextDescription->backendInterface.device;

    memcpy(&context->contextDescription, contextDescription, sizeof(FfxFsr1ContextDescription));

    // Check version info - make sure we are linked with the right backend version
    FfxVersionNumber version = context->contextDescription.backendInterface.fpGetSDKVersion(&context->contextDescription.backendInterface);
    FFX_RETURN_ON_ERROR(version == FFX_SDK_MAKE_VERSION(1, 1, 4), FFX_ERROR_INVALID_VERSION);

    // Setup constant buffer sizes.
    context->constantBuffer.num32BitEntries = sizeof(Fsr1Constants) / sizeof(uint32_t);

    // Create the context.
    FfxErrorCode errorCode =
        context->contextDescription.backendInterface.fpCreateBackendContext(&context->contextDescription.backendInterface, FFX_EFFECT_FSR1, nullptr, &context->effectContextId);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // Call out for device caps.
    errorCode = context->contextDescription.backendInterface.fpGetDeviceCapabilities(&context->contextDescription.backendInterface, &context->deviceCapabilities);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    // Create the intermediate upscale resource if RCAS is enabled
    const FfxInternalResourceDescription internalSurfaceDesc = {FFX_FSR1_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR,
                                                                L"FSR1_InternalUpscaledColor",
                                                                FFX_RESOURCE_TYPE_TEXTURE2D,
                                                                FFX_RESOURCE_USAGE_UAV,
                                                                contextDescription->outputFormat,
                                                                contextDescription->displaySize.width,
                                                                contextDescription->displaySize.height,
                                                                1,
                                                                FFX_RESOURCE_FLAGS_ALIASABLE,
                                                                {FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED}};

    // Clear the SRV resources to NULL.
    memset(context->srvResources, 0, sizeof(context->srvResources));

    if (contextDescription->flags & FFX_FSR1_ENABLE_RCAS)
    {
        const FfxResourceDescription       resourceDescription       = {FFX_RESOURCE_TYPE_TEXTURE2D,
                                                                        internalSurfaceDesc.format,
                                                                        internalSurfaceDesc.width,
                                                                        internalSurfaceDesc.height,
                                                                        1,
                                                                        internalSurfaceDesc.mipCount,
                                                                        internalSurfaceDesc.flags,
                                                                        internalSurfaceDesc.usage};

        const FfxCreateResourceDescription createResourceDescription = {FFX_HEAP_TYPE_DEFAULT,
                                                                        resourceDescription,
                                                                        FFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                                        internalSurfaceDesc.name,
                                                                        internalSurfaceDesc.id,
                                                                        internalSurfaceDesc.initData};

        FFX_VALIDATE(context->contextDescription.backendInterface.fpCreateResource(&context->contextDescription.backendInterface, &createResourceDescription, context->effectContextId, &context->srvResources[internalSurfaceDesc.id]));
    }

    // And copy resources to uavResrouces list
    memcpy(context->uavResources, context->srvResources, sizeof(context->srvResources));

    // Create shaders on initialize.
    errorCode = createPipelineStates(context);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

static FfxErrorCode fsr1Release(FfxFsr1Context_Private* context)
{
    FFX_ASSERT(context);

    // Release all pipelines
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineEASU, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineEASU_RCAS, context->effectContextId);
    ffxSafeReleasePipeline(&context->contextDescription.backendInterface, &context->pipelineRCAS, context->effectContextId);

    // Unregister resources not created internally
    context->srvResources[FFX_FSR1_RESOURCE_IDENTIFIER_INPUT_COLOR]     = { FFX_FSR1_RESOURCE_IDENTIFIER_NULL };
    context->srvResources[FFX_FSR1_RESOURCE_IDENTIFIER_UPSCALED_OUTPUT] = { FFX_FSR1_RESOURCE_IDENTIFIER_NULL };

    // Release internal resource
    ffxSafeReleaseResource(&context->contextDescription.backendInterface, context->srvResources[FFX_FSR1_RESOURCE_IDENTIFIER_INTERNAL_UPSCALED_COLOR], context->effectContextId);

    // Destroy the context
    context->contextDescription.backendInterface.fpDestroyBackendContext(&context->contextDescription.backendInterface, context->effectContextId);

    return FFX_OK;
}

FfxErrorCode ffxFsr1ContextCreate(FfxFsr1Context* context, const FfxFsr1ContextDescription* contextDescription)
{
    // Zero context memory
    memset(context, 0, sizeof(FfxFsr1Context));

    // Check pointers are valid.
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        contextDescription,
        FFX_ERROR_INVALID_POINTER);

    // Validate that all callbacks are set for the interface
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetSDKVersion, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
    FFX_RETURN_ON_ERROR(contextDescription->backendInterface.fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

    // If a scratch buffer is declared, then we must have a size
    if (contextDescription->backendInterface.scratchBuffer) {

        FFX_RETURN_ON_ERROR(contextDescription->backendInterface.scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
    }

    // Ensure the context is large enough for the internal context.
    FFX_STATIC_ASSERT(sizeof(FfxFsr1Context) >= sizeof(FfxFsr1Context_Private));

    // create the context.
    FfxFsr1Context_Private* contextPrivate = (FfxFsr1Context_Private*)(context);
    const FfxErrorCode errorCode = fsr1Create(contextPrivate, contextDescription);

    return errorCode;
}

FFX_API FfxErrorCode ffxFsr1ContextGetGpuMemoryUsage(FfxFsr1Context* context, FfxEffectMemoryUsage* vramUsage)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(vramUsage, FFX_ERROR_INVALID_POINTER);
    FfxFsr1Context_Private* contextPrivate = (FfxFsr1Context_Private*)(context);

    FFX_RETURN_ON_ERROR(contextPrivate->device, FFX_ERROR_NULL_DEVICE);

    FfxErrorCode errorCode = contextPrivate->contextDescription.backendInterface.fpGetEffectGpuMemoryUsage(
        &contextPrivate->contextDescription.backendInterface, contextPrivate->effectContextId, vramUsage);
    FFX_RETURN_ON_ERROR(errorCode == FFX_OK, errorCode);

    return FFX_OK;
}

FfxErrorCode ffxFsr1ContextDestroy(FfxFsr1Context* context)
{
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);

    // Destroy the context.
    FfxFsr1Context_Private* contextPrivate = (FfxFsr1Context_Private*)(context);
    const FfxErrorCode errorCode = fsr1Release(contextPrivate);
    return errorCode;
}

FfxErrorCode ffxFsr1ContextDispatch(FfxFsr1Context* context, const FfxFsr1DispatchDescription* dispatchDescription)
{
    // check pointers are valid
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchDescription, FFX_ERROR_INVALID_POINTER);

    FfxFsr1Context_Private* contextPrivate = (FfxFsr1Context_Private*)(context);

    // validate that renderSize is within the maximum.
    FFX_RETURN_ON_ERROR(
        dispatchDescription->renderSize.width <= contextPrivate->contextDescription.maxRenderSize.width,
        FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(
        dispatchDescription->renderSize.height <= contextPrivate->contextDescription.maxRenderSize.height,
        FFX_ERROR_OUT_OF_RANGE);
    FFX_RETURN_ON_ERROR(
        contextPrivate->device,
        FFX_ERROR_NULL_DEVICE);

    // dispatch the FSR2 passes.
    const FfxErrorCode errorCode = fsr1Dispatch(contextPrivate, dispatchDescription);
    return errorCode;
}

float ffxFsr1GetUpscaleRatioFromQualityMode(FfxFsr1QualityMode qualityMode)
{
    switch (qualityMode) {
    case FFX_FSR1_QUALITY_MODE_ULTRA_QUALITY:
        return 1.3f;
    case FFX_FSR1_QUALITY_MODE_QUALITY:
        return 1.5f;
    case FFX_FSR1_QUALITY_MODE_BALANCED:
        return 1.7f;
    case FFX_FSR1_QUALITY_MODE_PERFORMANCE:
        return 2.0f;
    default:
        return 0.0f;
    }
}

FfxErrorCode ffxFsr1GetRenderResolutionFromQualityMode(
    uint32_t* renderWidth,
    uint32_t* renderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    FfxFsr1QualityMode qualityMode)
{
    FFX_RETURN_ON_ERROR(renderWidth, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(renderHeight, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        FFX_FSR1_QUALITY_MODE_ULTRA_QUALITY <= qualityMode && qualityMode <= FFX_FSR1_QUALITY_MODE_PERFORMANCE,
        FFX_ERROR_INVALID_ENUM);

    // scale by the predefined ratios in each dimension.
    const float ratio = ffxFsr1GetUpscaleRatioFromQualityMode(qualityMode);
    const uint32_t scaledDisplayWidth = (uint32_t)((float)displayWidth / ratio);
    const uint32_t scaledDisplayHeight = (uint32_t)((float)displayHeight / ratio);
    *renderWidth = scaledDisplayWidth;
    *renderHeight = scaledDisplayHeight;

    return FFX_OK;
}

FFX_API FfxVersionNumber ffxFsr1GetEffectVersion()
{
    return FFX_SDK_MAKE_VERSION(FFX_FSR1_VERSION_MAJOR, FFX_FSR1_VERSION_MINOR, FFX_FSR1_VERSION_PATCH);
}
