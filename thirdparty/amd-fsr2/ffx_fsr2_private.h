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

#pragma once

// Constants for FSR2 DX12 dispatches. Must be kept in sync with cbFSR2 in ffx_fsr2_callbacks_hlsl.h
typedef struct Fsr2Constants {

    int32_t                     renderSize[2];
    int32_t                     maxRenderSize[2];
    int32_t                     displaySize[2];
    int32_t                     inputColorResourceDimensions[2];
    int32_t                     lumaMipDimensions[2];
    int32_t                     lumaMipLevelToUse;
    int32_t                     frameIndex;
    
    float                       deviceToViewDepth[4];
    float                       jitterOffset[2];
    float                       motionVectorScale[2];
    float                       downscaleFactor[2];
    float                       motionVectorJitterCancellation[2];
    float                       preExposure;
    float                       previousFramePreExposure;
    float                       tanHalfFOV;
    float                       jitterPhaseCount;
    float                       deltaTime;
    float                       dynamicResChangeFactor;
    float                       viewSpaceToMetersFactor;

    float                       pad;
    float                       reprojectionMatrix[16];
} Fsr2Constants;

struct FfxFsr2ContextDescription;
struct FfxDeviceCapabilities;
struct FfxPipelineState;
struct FfxResource;

// FfxFsr2Context_Private
// The private implementation of the FSR2 context.
typedef struct FfxFsr2Context_Private {

    FfxFsr2ContextDescription   contextDescription;
    Fsr2Constants               constants;
    FfxDevice                   device;
    FfxDeviceCapabilities       deviceCapabilities;
    FfxPipelineState            pipelineDepthClip;
    FfxPipelineState            pipelineReconstructPreviousDepth;
    FfxPipelineState            pipelineLock;
    FfxPipelineState            pipelineAccumulate;
    FfxPipelineState            pipelineAccumulateSharpen;
    FfxPipelineState            pipelineRCAS;
    FfxPipelineState            pipelineComputeLuminancePyramid;
    FfxPipelineState            pipelineGenerateReactive;
    FfxPipelineState            pipelineTcrAutogenerate;

    // 2 arrays of resources, as e.g. FFX_FSR2_RESOURCE_IDENTIFIER_LOCK_STATUS will use different resources when bound as SRV vs when bound as UAV
    FfxResourceInternal         srvResources[FFX_FSR2_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal         uavResources[FFX_FSR2_RESOURCE_IDENTIFIER_COUNT];

    bool                        firstExecution;
    bool                        refreshPipelineStates;
    uint32_t                    resourceFrameIndex;
    float                       previousJitterOffset[2];
    int32_t                     jitterPhaseCountRemaining;
} FfxFsr2Context_Private;
