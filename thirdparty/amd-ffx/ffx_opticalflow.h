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

// @defgroup OpticalFlow

#pragma once

// Include the interface for the backend of the OpticalFlow API.
#include "ffx_interface.h"

/// FidelityFX OpticalFlow major version.
///
/// @ingroup ffxOpticalflow
#define FFX_OPTICALFLOW_VERSION_MAJOR (1)

/// FidelityFX OpticalFlow minor version.
///
/// @ingroup ffxOpticalflow
#define FFX_OPTICALFLOW_VERSION_MINOR (1)

/// FidelityFX OpticalFlow patch version.
///
/// @ingroup ffxOpticalflow
#define FFX_OPTICALFLOW_VERSION_PATCH (2)

/// FidelityFX Optical Flow context count
///
/// Defines the number of internal effect contexts required by Optical Flow
///
/// @ingroup ffxOpticalFlow
#define FFX_OPTICALFLOW_CONTEXT_COUNT (1)

/// The size of the context specified in 32bit size units.
///
/// @ingroup ffxOpticalflow
#define FFX_OPTICALFLOW_CONTEXT_SIZE (FFX_SDK_DEFAULT_CONTEXT_SIZE)

#if defined(__cplusplus)
extern "C" {
#endif // #if defined(__cplusplus)

/// An enumeration of all the passes which constitute the OpticalFlow algorithm.
///
/// @ingroup ffxOpticalflow
typedef enum FfxOpticalflowPass
{
    FFX_OPTICALFLOW_PASS_PREPARE_LUMA = 0,
    FFX_OPTICALFLOW_PASS_GENERATE_OPTICAL_FLOW_INPUT_PYRAMID,
    FFX_OPTICALFLOW_PASS_GENERATE_SCD_HISTOGRAM,
    FFX_OPTICALFLOW_PASS_COMPUTE_SCD_DIVERGENCE,
    FFX_OPTICALFLOW_PASS_COMPUTE_OPTICAL_FLOW_ADVANCED_V5,
    FFX_OPTICALFLOW_PASS_FILTER_OPTICAL_FLOW_V5,
    FFX_OPTICALFLOW_PASS_SCALE_OPTICAL_FLOW_ADVANCED_V5,

    FFX_OPTICALFLOW_PASS_COUNT
} FfxOpticalflowPass;

/// An enumeration of bit flags used when creating a
/// <c><i>FfxOpticalflowContext</i></c>. See <c><i>FfxOpticalflowDispatchDescription</i></c>.
///
/// @ingroup ffxOpticalflow
typedef enum FfxOpticalflowInitializationFlagBits
{
    FFX_OPTICALFLOW_ENABLE_TEXTURE1D_USAGE = (1 << 0),

} FfxOpticalflowInitializationFlagBits;

/// A structure encapsulating the parameters required to initialize
/// FidelityFX OpticalFlow.
///
/// @ingroup ffxOpticalflow
typedef struct FfxOpticalflowContextDescription {

    FfxInterface                backendInterface;       ///< A set of pointers to the backend implementation for FidelityFX SDK
    uint32_t                    flags;                  ///< A collection of <c><i>FfxOpticalflowInitializationFlagBits</i></c>.
    FfxDimensions2D             resolution;
} FfxOpticalflowContextDescription;

/// A structure encapsulating the parameters for dispatching the various passes
/// of FidelityFX Opticalflow.
///
/// @ingroup ffxOpticalflow
typedef struct FfxOpticalflowDispatchDescription
{
    FfxCommandList   commandList;       ///< The <c><i>FfxCommandList</i></c> to record rendering commands into.
    FfxResource      color;             ///< A <c><i>FfxResource</i></c> containing the input color buffer
    FfxResource      opticalFlowVector; ///< A <c><i>FfxResource</i></c> containing the output motion buffer
    FfxResource      opticalFlowSCD;    ///< A <c><i>FfxResource</i></c> containing the output scene change detection buffer
    bool             reset;             ///< A boolean value which when set to true, indicates the camera has moved discontinuously.
    int              backbufferTransferFunction;
    FfxFloatCoords2D minMaxLuminance;
} FfxOpticalflowDispatchDescription;

typedef struct FfxOpticalflowSharedResourceDescriptions {

    FfxCreateResourceDescription opticalFlowVector;
    FfxCreateResourceDescription opticalFlowSCD;

} FfxOpticalflowSharedResourceDescriptions;

/// A structure encapsulating the FidelityFX OpticalFlow context.
///
/// This sets up an object which contains all persistent internal data and
/// resources that are required by OpticalFlow.
///
/// The <c><i>FfxOpticalflowContext</i></c> object should have a lifetime matching
/// your use of OpticalFlow. Before destroying the OpticalFlow context care should be taken
/// to ensure the GPU is not accessing the resources created or used by OpticalFlow.
/// It is therefore recommended that the GPU is idle before destroying OpticalFlow
/// OpticalFlow context.
///
/// @ingroup ffxOpticalflow
typedef struct FfxOpticalflowContext
{
    uint32_t data[FFX_OPTICALFLOW_CONTEXT_SIZE];  ///< An opaque set of <c>uint32_t</c> which contain the data for the context.
} FfxOpticalflowContext;


/// Create a FidelityFX OpticalFlow context from the parameters
/// programmed to the <c><i>FfxOpticalflowContextDescription</i></c> structure.
///
/// The context structure is the main object used to interact with the OpticalFlow
/// API, and is responsible for the management of the internal resources used
/// by the OpticalFlow algorithm. When this API is called, multiple calls will be
/// made via the pointers contained in the <c><i>callbacks</i></c> structure.
/// These callbacks will attempt to retreive the device capabilities, and
/// create the internal resources, and pipelines required by OpticalFlow's
/// frame-to-frame function. Depending on the precise configuration used when
/// creating the <c><i>FfxOpticalflowContext</i></c> a different set of resources and
/// pipelines might be requested via the callback functions.
///
/// The flags included in the <c><i>flags</i></c> field of
/// <c><i>FfxOpticalflowContext</i></c> how match the configuration of your
/// application as well as the intended use of OpticalFlow. It is important that these
/// flags are set correctly (as well as a correct programmed
/// <c><i>FfxOpticalflowContextDescription</i></c>) to ensure correct operation. It is
/// recommended to consult the overview documentation for further details on
/// how OpticalFlow should be integerated into an application.
///
/// When the <c><i>FfxOpticalflowContext</i></c> is created, you should use the
/// <c><i>ffxOpticalflowContextDispatch</i></c> function each frame where FSR3
/// upscaling should be applied. See the documentation of
/// <c><i>ffxOpticalflowContextDispatch</i></c> for more details.
///
/// The <c><i>FfxOpticalflowContext</i></c> should be destroyed when use of it is
/// completed, typically when an application is unloaded or OpticalFlow is
/// disabled by a user. To destroy the OpticalFlow context you should call
/// <c><i>ffxOpticalflowContextDestroy</i></c>.
///
/// @param [out] context                A pointer to a <c><i>FfxOpticalflowContext</i></c> structure to populate.
/// @param [in]  contextDescription     A pointer to a <c><i>FfxOpticalflowContextDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxOpticalflowContextDescription.callbacks</i></c>  was not fully specified.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup ffxOpticalflow
FFX_API FfxErrorCode ffxOpticalflowContextCreate(FfxOpticalflowContext* context, FfxOpticalflowContextDescription* contextDescription);

FFX_API FfxErrorCode ffxOpticalflowContextGetGpuMemoryUsage(FfxOpticalflowContext* pContext, FfxEffectMemoryUsage* vramUsage);

FFX_API FfxErrorCode ffxOpticalflowGetSharedResourceDescriptions(FfxOpticalflowContext* context, FfxOpticalflowSharedResourceDescriptions* SharedResources);

FFX_API FfxErrorCode ffxOpticalflowContextDispatch(FfxOpticalflowContext* context, const FfxOpticalflowDispatchDescription* dispatchDescription);

/// Destroy the FidelityFX OpticalFlow context.
///
/// @param [out] context                A pointer to a <c><i>FfxOpticalflowContext</i></c> structure to destroy.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>.
///
/// @ingroup ffxOpticalflow
FFX_API FfxErrorCode ffxOpticalflowContextDestroy(FfxOpticalflowContext* context);

/// Queries the effect version number.
///
/// @returns
/// The SDK version the effect was built with.
///
/// @ingroup ffxOpticalflow
FFX_API FfxVersionNumber ffxOpticalflowGetEffectVersion();

#if defined(__cplusplus)
}
#endif // #if defined(__cplusplus)
