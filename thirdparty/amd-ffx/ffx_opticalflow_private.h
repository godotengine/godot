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

#define FFX_CPU
#include "gpu/opticalflow/ffx_opticalflow_resources.h"

typedef enum OpticalFlowBindingIdentifiers
{
    FFX_OF_BINDING_IDENTIFIER_NULL = 0,
    FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR,

    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_1,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_2,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_3,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_4,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_5,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_6,

    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS_INPUT,

    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT,

    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_NEXT_LEVEL,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS,

    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_1,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_2,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_3,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_4,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_5,
    FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_ALIAS_LEVEL_6,

    FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_VECTOR,
    FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_SCD_OUTPUT,

    FFX_OF_BINDING_IDENTIFIER_COUNT
} OpticalFlowBindingIdentifiers;

typedef enum OpticalflowShaderPermutationOptions
{
    OPTICALFLOW_SHADER_PERMUTATION_FORCE_WAVE64 = (1 <<  0),  ///< doesn't map to a define, selects different table
    OPTICALFLOW_SHADER_PERMUTATION_ALLOW_FP16   = (1 <<  1),  ///< Enables fast math computations where possible
    OPTICALFLOW_HDR_COLOR_INPUT                 = (1 << 2),
} OpticalflowShaderPermutationOptions;

typedef struct OpticalflowConstants
{
    int32_t inputLumaResolution[2];
    uint32_t opticalFlowPyramidLevel;
    uint32_t opticalFlowPyramidLevelCount;

    int32_t frameIndex;
    uint32_t backbufferTransferFunction;
    float minMaxLuminance[2];
} OpticalflowConstants;

typedef struct FfxOpticalflowContext_Private
{
    FfxOpticalflowContextDescription contextDescription;
    FfxUInt32 effectContextId;
    OpticalflowConstants constants;
    FfxDevice device;
    FfxDeviceCapabilities deviceCapabilities;

    FfxPipelineState pipelinePrepareLuma;
    FfxPipelineState pipelineGenerateOpticalFlowInputPyramid;
    FfxPipelineState pipelineGenerateSCDHistogram;
    FfxPipelineState pipelineComputeSCDDivergence;
    FfxPipelineState pipelineComputeOpticalFlowAdvancedV5;
    FfxPipelineState pipelineFilterOpticalFlowV5;
    FfxPipelineState pipelineScaleOpticalFlowAdvancedV5;

    FfxResourceInternal resources[FFX_OF_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal srvBindings[FFX_OF_BINDING_IDENTIFIER_COUNT];
    FfxResourceInternal uavBindings[FFX_OF_BINDING_IDENTIFIER_COUNT];

    FfxConstantBuffer constantBuffers[FFX_OPTICALFLOW_CONSTANTBUFFER_COUNT];

    bool firstExecution;
    bool refreshPipelineStates;
    uint32_t resourceFrameIndex;
} FfxOpticalflowContext_Private;
