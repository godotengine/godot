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
#include <string.h>     // for memset
#include <cfloat>       // for FLT_EPSILON
#include "ffx_fsr3.h"
#include "ffx_fsr3upscaler.h"
#define FFX_CPU
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#endif

#include "gpu/ffx_core.h"
#include "gpu/fsr3/ffx_fsr3_resources.h"
#include "ffx_object_management.h"
#include "ffx_frameinterpolation_private.h"

#include "ffx_fsr3_private.h"

// To track only one context is present, also used in fi dispatch callback
static FfxFsr3Context* s_Context = nullptr;

FfxErrorCode ffxFsr3ContextCreate(FfxFsr3Context* context, FfxFsr3ContextDescription* contextDescription)
{
    FFX_STATIC_ASSERT(sizeof(FfxFsr3Context) >= sizeof(FfxFsr3Context_Private));
    FfxErrorCode            ret            = FFX_OK;
    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    // Prepare backend
    memset(context, 0, sizeof(FfxFsr3Context_Private));

    // check pointers are valid.
    FFX_RETURN_ON_ERROR(
        context,
        FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(
        contextDescription,
        FFX_ERROR_INVALID_POINTER);

    contextPrivate->description = *contextDescription;

    contextPrivate->backendInterfaceSharedResources = contextDescription->backendInterfaceSharedResources;
    contextPrivate->backendInterfaceUpscaling = contextDescription->backendInterfaceUpscaling;
    contextPrivate->backendInterfaceFrameInterpolation = contextDescription->backendInterfaceFrameInterpolation;

    bool upscalingOnly                      = (contextDescription->flags & FFX_FSR3_ENABLE_UPSCALING_ONLY) != 0;
    bool interpolationOnly                  = (contextDescription->flags & FFX_FSR3_ENABLE_INTERPOLATION_ONLY) != 0;
    contextPrivate->asyncWorkloadSupported  = (contextDescription->flags & FFX_FSR3_ENABLE_ASYNC_WORKLOAD_SUPPORT) != 0;
    contextPrivate->sharedResourceCount     = contextPrivate->asyncWorkloadSupported ? FSR3_MAX_QUEUED_FRAMES : 1;

    // ensure upscalingOnly and interpolationOnly are not set simultaneously
    FFX_ASSERT(upscalingOnly == false || interpolationOnly == false);

    // validate that all callbacks are set for the backend interfaces
    if (contextPrivate->interpolationOnly)
    {
        const FfxUInt32 numBackendsToVerify = 2;
        FfxInterface* backendsToVerify[] = { &contextPrivate->backendInterfaceSharedResources,
                                            &contextPrivate->backendInterfaceFrameInterpolation };

        for (FfxUInt32 i = 0; i < numBackendsToVerify; i++)
        {
            FfxInterface* backend = backendsToVerify[i];
            FFX_RETURN_ON_ERROR(backend, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

            // if a scratch buffer is declared, then we must have a size
            if (backend->scratchBuffer)
            {
                FFX_RETURN_ON_ERROR(backend->scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
            }
        }
    }
    else
    {
        const FfxUInt32 numBackendsToVerify = contextPrivate->upscalingOnly ? 1 : 3;
        FfxInterface* backendsToVerify[] = { &contextPrivate->backendInterfaceUpscaling,
                                               &contextPrivate->backendInterfaceSharedResources,
                                               &contextPrivate->backendInterfaceFrameInterpolation };

        for (FfxUInt32 i = 0; i < numBackendsToVerify; i++)
        {
            FfxInterface* backend = backendsToVerify[i];
            FFX_RETURN_ON_ERROR(backend, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpGetSDKVersion, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpGetDeviceCapabilities, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpCreateBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);
            FFX_RETURN_ON_ERROR(backend->fpDestroyBackendContext, FFX_ERROR_INCOMPLETE_INTERFACE);

            // if a scratch buffer is declared, then we must have a size
            if (backend->scratchBuffer)
            {
                FFX_RETURN_ON_ERROR(backend->scratchBufferSize, FFX_ERROR_INCOMPLETE_INTERFACE);
            }
        }
    }

    if (!contextPrivate->upscalingOnly)
    {
        FFX_VALIDATE(contextPrivate->backendInterfaceSharedResources.fpCreateBackendContext(&contextPrivate->backendInterfaceSharedResources,
            FFX_EFFECT_SHAREDRESOURCES,
            nullptr,
            &contextPrivate->effectContextIdSharedResources));
    }
    else
    {
        contextPrivate->backendInterfaceSharedResources = contextPrivate->backendInterfaceUpscaling;
        contextDescription->backendInterfaceSharedResources = contextDescription->backendInterfaceUpscaling;
    }

    // set up FSR3 Upscaler
    // ensure we're actually creating an FSR3 Upscaler context, not the creationfunction that reroutes to ffxFsr3ContextCreate
    if (!interpolationOnly)
	{
		FfxFsr3UpscalerContextDescription upDesc = {};
        upDesc.flags = 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_HIGH_DYNAMIC_RANGE) ? FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DEPTH_INVERTED) ? FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DEPTH_INFINITE) ? FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_AUTO_EXPOSURE) ? FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DYNAMIC_RESOLUTION) ? FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DEBUG_CHECKING) ? FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING : 0;
        upDesc.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_HDR_UPSCALE_SDR_FINALOUTPUT) ? FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE : 0;
        upDesc.maxRenderSize = contextDescription->maxRenderSize;
        upDesc.maxUpscaleSize = contextDescription->maxUpscaleSize;
		upDesc.backendInterface = contextDescription->backendInterfaceUpscaling;
		upDesc.fpMessage = contextDescription->fpMessage;
		FFX_VALIDATE(ffxFsr3UpscalerContextCreate(&contextPrivate->upscalerContext, &upDesc));
	}

    if (!upscalingOnly)
    {

        FfxOpticalflowContextDescription ofDescription = {};
        ofDescription.backendInterface                 = contextDescription->backendInterfaceFrameInterpolation;
        ofDescription.resolution                       = contextDescription->displaySize;

        // set up Opticalflow
        FFX_VALIDATE(ffxOpticalflowContextCreate(&contextPrivate->ofContext, &ofDescription));

        FfxFrameInterpolationContextDescription fiDescription = {};
        fiDescription.backendInterface  = contextDescription->backendInterfaceFrameInterpolation;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) ? FFX_FRAMEINTERPOLATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS : 0;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) ? FFX_FRAMEINTERPOLATION_ENABLE_JITTER_MOTION_VECTORS : 0;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DEPTH_INVERTED) ? FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED : 0;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_DEPTH_INFINITE) ? FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE : 0;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_HIGH_DYNAMIC_RANGE) ? FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT : 0;
        fiDescription.flags |= (contextDescription->flags & FFX_FSR3_ENABLE_SDR_UPSCALE_HDR_FINALOUTPUT) ? FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT : 0;
        fiDescription.flags |= contextPrivate->asyncWorkloadSupported ? FFX_FRAMEINTERPOLATION_ENABLE_ASYNC_SUPPORT : 0;
        fiDescription.maxRenderSize = contextDescription->maxRenderSize;
        fiDescription.displaySize      = contextDescription->displaySize;
        fiDescription.backBufferFormat = contextDescription->backBufferFormat;
        // This is a new item exposed only through ffx API on PC
        fiDescription.previousInterpolationSourceFormat = contextDescription->backBufferFormat;

        // set up Frameinterpolation
        FFX_VALIDATE(ffxFrameInterpolationContextCreate(&contextPrivate->fiContext, &fiDescription));
        contextPrivate->effectContextIdFrameGeneration = reinterpret_cast<FfxFrameInterpolationContext_Private*>(&contextPrivate->fiContext)->effectContextId;

        // set up optical flow resources
        FfxOpticalflowSharedResourceDescriptions ofResourceDescs = {};
        FFX_VALIDATE(ffxOpticalflowGetSharedResourceDescriptions(&contextPrivate->ofContext, &ofResourceDescs));

        FFX_VALIDATE(contextDescription->backendInterfaceSharedResources.fpCreateResource(
            &contextDescription->backendInterfaceSharedResources, &ofResourceDescs.opticalFlowVector, contextPrivate->effectContextIdSharedResources, &contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]));
        FFX_VALIDATE(contextDescription->backendInterfaceSharedResources.fpCreateResource(
            &contextDescription->backendInterfaceSharedResources, &ofResourceDescs.opticalFlowSCD, contextPrivate->effectContextIdSharedResources, &contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT]));
    }

    // set up FSR3Upscaler resources
    if (!contextPrivate->interpolationOnly)
    {
        FfxFsr3UpscalerSharedResourceDescriptions fs3UpscalerResourceDescs = {};
        FFX_VALIDATE(ffxFsr3UpscalerGetSharedResourceDescriptions(&contextPrivate->upscalerContext, &fs3UpscalerResourceDescs));

        wchar_t Name[256] = {};
        for (FfxUInt32 i = 0; i < contextPrivate->sharedResourceCount; i++)
        {
            FfxCreateResourceDescription dilD = fs3UpscalerResourceDescs.dilatedDepth;
            swprintf(Name, 255, L"%s%d", fs3UpscalerResourceDescs.dilatedDepth.name, i);
            dilD.name = Name;
            FFX_VALIDATE(contextDescription->backendInterfaceSharedResources.fpCreateResource(
                &contextDescription->backendInterfaceSharedResources, &dilD, contextPrivate->effectContextIdSharedResources, &contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0 + (i * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]));

            FfxCreateResourceDescription dilMVs = fs3UpscalerResourceDescs.dilatedMotionVectors;
            swprintf(Name, 255, L"%s%d", fs3UpscalerResourceDescs.dilatedMotionVectors.name, i);
            dilMVs.name = Name;
            FFX_VALIDATE(contextDescription->backendInterfaceSharedResources.fpCreateResource(
                &contextDescription->backendInterfaceSharedResources, &dilMVs, contextPrivate->effectContextIdSharedResources, &contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_0 + (i * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]));

            FfxCreateResourceDescription recND = fs3UpscalerResourceDescs.reconstructedPrevNearestDepth;
            swprintf(Name, 255, L"%s%d", fs3UpscalerResourceDescs.reconstructedPrevNearestDepth.name, i);
            recND.name = Name;
            FFX_VALIDATE(contextDescription->backendInterfaceSharedResources.fpCreateResource(
                &contextDescription->backendInterfaceSharedResources, &recND, contextPrivate->effectContextIdSharedResources, &contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0 + (i * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]));
        }
    }

    return ret;
}

FfxErrorCode ffxFsr3ContextGetGpuMemoryUsage(
    FfxFsr3Context* context,
    FfxEffectMemoryUsage* pUpscalerUsage,
    FfxEffectMemoryUsage* pOpticalFlowUsage,
    FfxEffectMemoryUsage* pFrameGenerationUsage)
{
    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);

    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    if (nullptr != pUpscalerUsage)
    {
        memset(pUpscalerUsage, 0, sizeof(FfxEffectMemoryUsage));
        ffxFsr3UpscalerContextGetGpuMemoryUsage(&contextPrivate->upscalerContext, pUpscalerUsage);
    }


    if (nullptr != pOpticalFlowUsage)
    {
        memset(pOpticalFlowUsage, 0, sizeof(FfxEffectMemoryUsage));
        ffxOpticalflowContextGetGpuMemoryUsage(&contextPrivate->ofContext, pOpticalFlowUsage);
    }

    if (nullptr != pFrameGenerationUsage)
    {
        memset(pFrameGenerationUsage, 0, sizeof(FfxEffectMemoryUsage));
        ffxFrameInterpolationContextGetGpuMemoryUsage(&contextPrivate->fiContext, pFrameGenerationUsage);
    }

    return FFX_OK;
}

FfxErrorCode ffxFsr3ContextGenerateReactiveMask(FfxFsr3Context* context, const FfxFsr3GenerateReactiveDescription* params)
{
    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    FfxFsr3UpscalerGenerateReactiveDescription fsr3Params{};

	fsr3Params.commandList     = params->commandList;
	fsr3Params.colorOpaqueOnly = params->colorOpaqueOnly;
	fsr3Params.colorPreUpscale = params->colorPreUpscale;
	fsr3Params.outReactive     = params->outReactive;
	fsr3Params.renderSize      = params->renderSize;
	fsr3Params.scale           = params->scale;
	fsr3Params.cutoffThreshold = params->cutoffThreshold;
	fsr3Params.binaryValue     = params->binaryValue;
	fsr3Params.flags           = params->flags;

    return ffxFsr3UpscalerContextGenerateReactiveMask(&contextPrivate->upscalerContext, &fsr3Params);
}

FfxErrorCode ffxFsr3DispatchFrameGeneration(const FfxFrameGenerationDispatchDescription* callbackDesc)
{
    FfxErrorCode errorCode = FFX_OK;

    FFX_RETURN_ON_ERROR(s_Context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(callbackDesc, FFX_ERROR_INVALID_POINTER);

    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(s_Context);

    bool upscalingOnly     = (contextPrivate->description.flags & FFX_FSR3_ENABLE_UPSCALING_ONLY) != 0;
    FFX_ASSERT_MESSAGE(upscalingOnly == false, "Fsr3 context has not been initialized to support Frame Generation");

    const FfxFrameInterpolationPrepareDescription* prepareDesc = &contextPrivate->fgPrepareDescriptions[callbackDesc->frameID & 1];

    // Optical flow
    {
        FfxOpticalflowDispatchDescription ofDispatchDesc{};
        ofDispatchDesc.commandList = callbackDesc->commandList;
        ofDispatchDesc.color = callbackDesc->presentColor;
        if (contextPrivate->HUDLess_color.resource)
        {
            ofDispatchDesc.color = contextPrivate->HUDLess_color;
        }
        ofDispatchDesc.reset                      = callbackDesc->reset;
        ofDispatchDesc.backbufferTransferFunction = callbackDesc->backBufferTransferFunction;
        ofDispatchDesc.minMaxLuminance.x = callbackDesc->minMaxLuminance[0];
        ofDispatchDesc.minMaxLuminance.y = callbackDesc->minMaxLuminance[1];
        ofDispatchDesc.opticalFlowVector = contextPrivate->backendInterfaceSharedResources.fpGetResource(&contextPrivate->backendInterfaceSharedResources, contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);
        ofDispatchDesc.opticalFlowSCD = contextPrivate->backendInterfaceSharedResources.fpGetResource(&contextPrivate->backendInterfaceSharedResources, contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT]);

        errorCode |= ffxOpticalflowContextDispatch(&contextPrivate->ofContext, &ofDispatchDesc);
    }

    // Frame interpolation
    {
        FfxFrameInterpolationDispatchDescription fiDispatchDesc{0};

        // don't dispatch interpolation async for now: use the same commandlist for copy and interpolate
        fiDispatchDesc.commandList = callbackDesc->commandList;
        fiDispatchDesc.displaySize.width = callbackDesc->presentColor.description.width;
        fiDispatchDesc.displaySize.height = callbackDesc->presentColor.description.height;
        fiDispatchDesc.currentBackBuffer = callbackDesc->presentColor;
        fiDispatchDesc.currentBackBuffer_HUDLess = contextPrivate->HUDLess_color;

        fiDispatchDesc.renderSize               = prepareDesc->renderSize;
        fiDispatchDesc.output                   = callbackDesc->outputs[0];
        fiDispatchDesc.opticalFlowVector        = contextPrivate->backendInterfaceSharedResources.fpGetResource(&contextPrivate->backendInterfaceSharedResources, contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_VECTOR]);
        fiDispatchDesc.opticalFlowSceneChangeDetection = contextPrivate->backendInterfaceSharedResources.fpGetResource(&contextPrivate->backendInterfaceSharedResources, contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT]);
        fiDispatchDesc.opticalFlowBlockSize     = 8;
        fiDispatchDesc.opticalFlowScale         = { 1.f / fiDispatchDesc.displaySize.width, 1.f / fiDispatchDesc.displaySize.height };
        fiDispatchDesc.frameTimeDelta           = prepareDesc->frameTimeDelta;
        fiDispatchDesc.reset                    = callbackDesc->reset;
        fiDispatchDesc.cameraNear               = prepareDesc->cameraNear;
        fiDispatchDesc.cameraFar                = prepareDesc->cameraFar;
        fiDispatchDesc.viewSpaceToMetersFactor  = prepareDesc->viewSpaceToMetersFactor;
        fiDispatchDesc.cameraFovAngleVertical   = prepareDesc->cameraFovAngleVertical;
        fiDispatchDesc.interpolationRect.left   = callbackDesc->interpolationRect.left;
        fiDispatchDesc.interpolationRect.top    = callbackDesc->interpolationRect.top;
        fiDispatchDesc.interpolationRect.width  = callbackDesc->interpolationRect.width;
        fiDispatchDesc.interpolationRect.height = callbackDesc->interpolationRect.height;
        fiDispatchDesc.frameID                  = callbackDesc->frameID;

        // use the same surfaces that were specified in the upscale (or interpolation prepare)
        fiDispatchDesc.dilatedDepth             = contextPrivate->dilatedDepth;
        fiDispatchDesc.dilatedMotionVectors     = contextPrivate->dilatedMotionVectors;
        fiDispatchDesc.reconstructedPrevDepth   = contextPrivate->reconstructedPrevNearestDepth;

        if (contextPrivate->frameGenerationFlags & FFX_FSR3_FRAME_GENERATION_FLAG_DRAW_DEBUG_TEAR_LINES)
        {
            fiDispatchDesc.flags |= FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES;
        }

        if (contextPrivate->frameGenerationFlags & FFX_FSR3_FRAME_GENERATION_FLAG_DRAW_DEBUG_VIEW)
        {
            fiDispatchDesc.flags |= FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW;
        }

        fiDispatchDesc.backBufferTransferFunction = callbackDesc->backBufferTransferFunction;
        fiDispatchDesc.minMaxLuminance[0]         = callbackDesc->minMaxLuminance[0];
        fiDispatchDesc.minMaxLuminance[1]         = callbackDesc->minMaxLuminance[1];

        errorCode |= ffxFrameInterpolationDispatch(&contextPrivate->fiContext, &fiDispatchDesc);
    }

    return errorCode;
}

FfxErrorCode ffxFsr3ContextDispatchUpscale(FfxFsr3Context* context, const FfxFsr3DispatchUpscaleDescription* dispatchParams)
{
    FfxErrorCode ret = FFX_OK;

    FFX_RETURN_ON_ERROR(context, FFX_ERROR_INVALID_POINTER);
    FFX_RETURN_ON_ERROR(dispatchParams, FFX_ERROR_INVALID_POINTER);

    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    bool interpolationOnly = (contextPrivate->description.flags & FFX_FSR3_ENABLE_INTERPOLATION_ONLY) != 0;
    FFX_ASSERT_MESSAGE(interpolationOnly == false, "Fsr3 context has not been initialized to support Frame Generation");

    contextPrivate->deltaTime = FFX_MAXIMUM(0.0f, FFX_MINIMUM(1.0f, dispatchParams->frameTimeDelta / 1000.0f));

    FfxUInt32 sharedResourceIndexUpscaling = dispatchParams->frameID % contextPrivate->sharedResourceCount;

    contextPrivate->dilatedDepth = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);
    contextPrivate->dilatedMotionVectors = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);
    contextPrivate->reconstructedPrevNearestDepth = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);

    // dispatch FSR3
    FfxFsr3UpscalerDispatchDescription fsr3DispatchParams{};
    fsr3DispatchParams.commandList                      = dispatchParams->commandList;
    fsr3DispatchParams.color                            = dispatchParams->color;
    fsr3DispatchParams.depth                            = dispatchParams->depth;
    fsr3DispatchParams.motionVectors                    = dispatchParams->motionVectors;
    fsr3DispatchParams.exposure                         = dispatchParams->exposure;
    fsr3DispatchParams.reactive                         = dispatchParams->reactive;
    fsr3DispatchParams.transparencyAndComposition       = dispatchParams->transparencyAndComposition;
    fsr3DispatchParams.output                           = dispatchParams->upscaleOutput;
    fsr3DispatchParams.jitterOffset                     = dispatchParams->jitterOffset;
    fsr3DispatchParams.motionVectorScale                = dispatchParams->motionVectorScale;
    fsr3DispatchParams.renderSize                       = dispatchParams->renderSize;
    fsr3DispatchParams.enableSharpening                 = dispatchParams->enableSharpening;
    fsr3DispatchParams.sharpness                        = dispatchParams->sharpness;
    fsr3DispatchParams.frameTimeDelta                   = dispatchParams->frameTimeDelta;
    fsr3DispatchParams.preExposure                      = dispatchParams->preExposure;
    fsr3DispatchParams.reset                            = dispatchParams->reset;
    fsr3DispatchParams.cameraNear                       = dispatchParams->cameraNear;
    fsr3DispatchParams.cameraFar                        = dispatchParams->cameraFar;
    fsr3DispatchParams.cameraFovAngleVertical           = dispatchParams->cameraFovAngleVertical;
    fsr3DispatchParams.viewSpaceToMetersFactor          = dispatchParams->viewSpaceToMetersFactor;
    fsr3DispatchParams.dilatedDepth                     = contextPrivate->dilatedDepth;
    fsr3DispatchParams.dilatedMotionVectors             = contextPrivate->dilatedMotionVectors;
    fsr3DispatchParams.reconstructedPrevNearestDepth    = contextPrivate->reconstructedPrevNearestDepth;

    if (dispatchParams->flags & FFX_FSR3_UPSCALER_FLAG_DRAW_DEBUG_VIEW)
    {
        fsr3DispatchParams.flags |= FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW;
    }

    ret = ffxFsr3UpscalerContextDispatch(&contextPrivate->upscalerContext, &fsr3DispatchParams);

    return ret;
}

FfxErrorCode ffxFsr3ContextDispatchFrameGenerationPrepare(FfxFsr3Context* context, const FfxFsr3DispatchFrameGenerationPrepareDescription* dispatchParams)
{
    FfxErrorCode            ret            = FFX_OK;

    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    bool upscalingOnly     = (contextPrivate->description.flags & FFX_FSR3_ENABLE_UPSCALING_ONLY) != 0;
    FFX_ASSERT_MESSAGE(upscalingOnly == false, "Fsr3 context has not been initialized to support Frame Generation");

    // if not interpolationOnly there's no need to execute prepare as prepared resources from upscale can be used
    bool interpolationOnly = (contextPrivate->description.flags & FFX_FSR3_ENABLE_INTERPOLATION_ONLY) != 0;

    FfxUInt32 sharedResourceIndexUpscaling = dispatchParams->frameID % contextPrivate->sharedResourceCount;

    FfxFrameInterpolationPrepareDescription fiPrepareParams = {0};
    fiPrepareParams.commandList = dispatchParams->commandList;
    fiPrepareParams.renderSize = dispatchParams->renderSize;
    fiPrepareParams.depth = dispatchParams->depth;
    fiPrepareParams.motionVectors = dispatchParams->motionVectors;
    fiPrepareParams.jitterOffset = dispatchParams->jitterOffset;
    fiPrepareParams.motionVectorScale = dispatchParams->motionVectorScale;
    fiPrepareParams.frameTimeDelta = dispatchParams->frameTimeDelta;
    fiPrepareParams.cameraNear = dispatchParams->cameraNear;
    fiPrepareParams.cameraFar = dispatchParams->cameraFar;
    fiPrepareParams.viewSpaceToMetersFactor = dispatchParams->viewSpaceToMetersFactor;
    fiPrepareParams.cameraFovAngleVertical = dispatchParams->cameraFovAngleVertical;
    fiPrepareParams.frameID = dispatchParams->frameID;

    contextPrivate->dilatedDepth = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);
    contextPrivate->dilatedMotionVectors = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);
    contextPrivate->reconstructedPrevNearestDepth = contextPrivate->backendInterfaceSharedResources.fpGetResource(
        &contextPrivate->backendInterfaceSharedResources,
        contextPrivate->sharedResources[FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0 +
        (sharedResourceIndexUpscaling * FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT)]);

    fiPrepareParams.dilatedDepth = contextPrivate->dilatedDepth;
    fiPrepareParams.dilatedMotionVectors = contextPrivate->dilatedMotionVectors;
    fiPrepareParams.reconstructedPrevDepth = contextPrivate->reconstructedPrevNearestDepth;

    if (interpolationOnly)
    {
        ret = ffxFrameInterpolationPrepare(&contextPrivate->fiContext, &fiPrepareParams);
    }

    contextPrivate->fgPrepareDescriptions[dispatchParams->frameID & 1] = fiPrepareParams;

    return ret;
}

FfxErrorCode ffxFsr3ConfigureFrameGeneration(FfxFsr3Context* context, const FfxFrameGenerationConfig* config)
{
    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

    bool upscalingOnly     = (contextPrivate->description.flags & FFX_FSR3_ENABLE_UPSCALING_ONLY) != 0;
    FFX_ASSERT_MESSAGE(upscalingOnly == false, "Fsr3 context has not been initialized to support Frame Generation");

    FFX_ASSERT(config);
    FFX_ASSERT_MESSAGE(!contextPrivate->frameGenerationEnabled || !config->allowAsyncWorkloads || contextPrivate->asyncWorkloadSupported,
                       "Illegal to allow async workload when context was created without FFX_FSR3_ENABLE_ASYNC_WORKLOAD_SUPPORT flag set.");

    FfxFrameGenerationConfig patchedConfig = *config;

    contextPrivate->frameGenerationFlags    = patchedConfig.flags;
    contextPrivate->HUDLess_color           = patchedConfig.HUDLessColor;

    if (patchedConfig.flags & FFX_FSR3_FRAME_GENERATION_FLAG_DRAW_DEBUG_VIEW)
    {
        patchedConfig.onlyPresentInterpolated = true;
    }

    if (patchedConfig.flags & FFX_FSR3_FRAME_GENERATION_FLAG_DRAW_DEBUG_PACING_LINES)
    {
        patchedConfig.drawDebugPacingLines = true;
    }

    // reset shared resource indices
    if (contextPrivate->frameGenerationEnabled != patchedConfig.frameGenerationEnabled)
    {
        contextPrivate->frameGenerationEnabled = patchedConfig.frameGenerationEnabled;

        if (contextPrivate->frameGenerationEnabled) {
            FFX_ASSERT(nullptr == s_Context);
            s_Context = context;
        }
        else if (s_Context == context) {
            s_Context = nullptr;
        }
    }

    return contextPrivate->backendInterfaceFrameInterpolation.fpSwapChainConfigureFrameGeneration(&patchedConfig);
}

FfxErrorCode ffxFsr3ContextDestroy(FfxFsr3Context* context)
{
    FfxFsr3Context_Private* contextPrivate = (FfxFsr3Context_Private*)(context);

	for (FfxUInt32 i = 0; i < FFX_FSR3_RESOURCE_IDENTIFIER_COUNT; i++)
    {
        FFX_VALIDATE(contextPrivate->backendInterfaceSharedResources.fpDestroyResource(&contextPrivate->backendInterfaceSharedResources, contextPrivate->sharedResources[i], contextPrivate->effectContextIdSharedResources))
    }
    contextPrivate->backendInterfaceSharedResources.fpDestroyBackendContext(&contextPrivate->backendInterfaceSharedResources, contextPrivate->effectContextIdSharedResources);

    bool upscalingOnly     = (contextPrivate->description.flags & FFX_FSR3_ENABLE_UPSCALING_ONLY) != 0;
    bool interpolationOnly = (contextPrivate->description.flags & FFX_FSR3_ENABLE_INTERPOLATION_ONLY) != 0;

    if (!upscalingOnly)
    {
        FFX_VALIDATE(ffxFrameInterpolationContextDestroy(&contextPrivate->fiContext));
        FFX_VALIDATE(ffxOpticalflowContextDestroy(&contextPrivate->ofContext));
    }

    if (!interpolationOnly)
    {
        FFX_VALIDATE(ffxFsr3UpscalerContextDestroy(&contextPrivate->upscalerContext));
    }

    if (s_Context == context) {
        s_Context = nullptr;
    }

    return FFX_OK;
}

float ffxFsr3GetUpscaleRatioFromQualityMode(FfxFsr3QualityMode qualityMode)
{
    return ffxFsr3UpscalerGetUpscaleRatioFromQualityMode((FfxFsr3UpscalerQualityMode)qualityMode);
}

FfxErrorCode ffxFsr3GetRenderResolutionFromQualityMode(
    uint32_t* renderWidth, uint32_t* renderHeight, uint32_t displayWidth, uint32_t displayHeight, FfxFsr3QualityMode qualityMode)
{
    return ffxFsr3UpscalerGetRenderResolutionFromQualityMode( renderWidth, renderHeight, displayWidth, displayHeight, (FfxFsr3UpscalerQualityMode) qualityMode);
}

int32_t ffxFsr3GetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth)
{
	return ffxFsr3UpscalerGetJitterPhaseCount(renderWidth, displayWidth);
}

FfxErrorCode ffxFsr3GetJitterOffset(float* outX, float* outY, int32_t index, int32_t phaseCount)
{
	return ffxFsr3UpscalerGetJitterOffset(outX, outY, index, phaseCount);
}

FFX_API bool ffxFsr3ResourceIsNull(FfxResource resource)
{
	return ffxFsr3UpscalerResourceIsNull(resource);
}

FFX_API FfxVersionNumber ffxFsr3GetEffectVersion()
{
    return FFX_SDK_MAKE_VERSION(FFX_FSR3_VERSION_MAJOR, FFX_FSR3_VERSION_MINOR, FFX_FSR3_VERSION_PATCH);
}
