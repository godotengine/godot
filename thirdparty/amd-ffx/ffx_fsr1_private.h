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
#include "gpu/fsr1/ffx_fsr1_resources.h"

/// An enumeration of all the permutations that can be passed to the FSR1 algorithm.
///
/// FSR1 features are organized through a set of pre-defined compile
/// permutation options that need to be specified. Which shader blob
/// is returned for pipeline creation will be determined by what combination
/// of shader permutations are enabled.
///
typedef enum Fs1ShaderPermutationOptions
{
    FSR1_SHADER_PERMUTATION_APPLY_RCAS              = (1 << 0),  ///< RCAS will be applied, outputs to correct intermediary target
    FSR1_SHADER_PERMUTATION_RCAS_PASSTHROUGH_ALPHA  = (1 << 1),  ///< Compile RCAS to pass through the input alpha value
    FSR1_SHADER_PERMUTATION_SRGB_CONVERSIONS        = (1 << 2),  ///< Handle necessary conversions for SRGB formats (de-gamma in and gamma out)
    FSR1_SHADER_PERMUTATION_FORCE_WAVE64            = (1 << 3),  ///< doesn't map to a define, selects different table
    FSR1_SHADER_PERMUTATION_ALLOW_FP16              = (1 << 4),  ///< Enables fast math computations where possible
} Fs1ShaderPermutationOptions;

// Constants for FSR1 dispatches. Must be kept in sync with cbFSR1 in ffx_fsr1_callbacks_hlsl.h
typedef struct Fsr1Constants
{
    FfxUInt32x4 const0;
    FfxUInt32x4 const1;
    FfxUInt32x4 const2;
    FfxUInt32x4 const3;
    FfxUInt32x4 sample;
} Fsr1Constants;

struct FfxFsr1ContextDescription;
struct FfxDeviceCapabilities;
struct FfxPipelineState;
struct FfxResource;

// FfxFsr1Context_Private
// The private implementation of the FSR1 context.
typedef struct FfxFsr1Context_Private {

    FfxFsr1ContextDescription   contextDescription;
    FfxUInt32                   effectContextId;
    Fsr1Constants               constants;
    FfxDevice                   device;
    FfxDeviceCapabilities       deviceCapabilities;
    FfxConstantBuffer           constantBuffer;

    FfxPipelineState            pipelineEASU;
    FfxPipelineState            pipelineEASU_RCAS;
    FfxPipelineState            pipelineRCAS;

    FfxResourceInternal         srvResources[FFX_FSR1_RESOURCE_IDENTIFIER_COUNT];
    FfxResourceInternal         uavResources[FFX_FSR1_RESOURCE_IDENTIFIER_COUNT];

} FfxFsr1Context_Private;
