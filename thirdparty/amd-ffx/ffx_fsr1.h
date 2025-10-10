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

/// @defgroup ffxFsr1 FidelityFX FSR1
/// FidelityFX Super Resolution 1 runtime library
///
/// @ingroup SDKComponents

#pragma once

/// Include the interface for the backend of the FSR 1.0 API.
///
/// @ingroup ffxFsr1
#include "ffx_interface.h"

/// FidelityFX Super Resolution 1.0 major version.
///
/// @ingroup ffxFsr1
#define FFX_FSR1_VERSION_MAJOR      (1)

/// FidelityFX Super Resolution 1.0 minor version.
///
/// @ingroup ffxFsr1
#define FFX_FSR1_VERSION_MINOR      (2)

/// FidelityFX Super Resolution 1.0 patch version.
///
/// @ingroup ffxFsr1
#define FFX_FSR1_VERSION_PATCH      (0)

/// FidelityFX Super Resolution 1.0 context count
///
/// Defines the number of internal effect contexts required by FSR1
///
/// @ingroup ffxFsr1
#define FFX_FSR1_CONTEXT_COUNT   2

/// The size of the context specified in 32bit values.
///
/// @ingroup ffxFsr1
#define FFX_FSR1_CONTEXT_SIZE       (27448)

#if defined(__cplusplus)
extern "C" {
#endif // #if defined(__cplusplus)

/// An enumeration of all the passes which constitute the FSR1 algorithm.
///
/// FSR1 is implemented as a composite of several compute passes each
/// computing a key part of the final result. Each call to the
/// <c><i>FfxFsr1ScheduleGpuJobFunc</i></c> callback function will
/// correspond to a single pass included in <c><i>FfxFsr1Pass</i></c>. For a
/// more comprehensive description of each pass, please refer to the FSR1
/// reference documentation.
///
/// @ingroup ffxFsr1
typedef enum FfxFsr1Pass
{
    FFX_FSR1_PASS_EASU = 0,         ///< A pass which upscales the color buffer using easu.
    FFX_FSR1_PASS_EASU_RCAS = 1,    ///< A pass which upscales the color buffer in preparation for rcas
    FFX_FSR1_PASS_RCAS = 2,         ///< A pass which performs rcas sharpening on the upscaled image.

    FFX_FSR1_PASS_COUNT             ///< The number of passes performed by FSR2.
} FfxFsr1Pass;

/// An enumeration of all the quality modes supported by FidelityFX Super
/// Resolution 1 upscaling.
///
/// In order to provide a consistent user experience across multiple
/// applications which implement FSR1. It is strongly recommended that the
/// following preset scaling factors are made available through your
/// application's user interface.
///
/// If your application does not expose the notion of preset scaling factors
/// for upscaling algorithms (perhaps instead implementing a fixed ratio which
/// is immutable) or implementing a more dynamic scaling scheme (such as
/// dynamic resolution scaling), then there is no need to use these presets.
///
/// @ingroup ffxFsr1
typedef enum FfxFsr1QualityMode {

    FFX_FSR1_QUALITY_MODE_ULTRA_QUALITY     = 0,    ///< Perform upscaling with a per-dimension upscaling ratio of 1.3x.
    FFX_FSR1_QUALITY_MODE_QUALITY           = 1,    ///< Perform upscaling with a per-dimension upscaling ratio of 1.5x.
    FFX_FSR1_QUALITY_MODE_BALANCED          = 2,    ///< Perform upscaling with a per-dimension upscaling ratio of 1.7x.
    FFX_FSR1_QUALITY_MODE_PERFORMANCE       = 3     ///< Perform upscaling with a per-dimension upscaling ratio of 2.0x.
} FfxFsr1QualityMode;

/// An enumeration of bit flags used when creating a
/// <c><i>FfxFsr1Context</i></c>. See <c><i>FfxFsr1ContextDescription</i></c>.
///
/// @ingroup ffxFsr1
typedef enum FfxFsr1InitializationFlagBits {

    FFX_FSR1_ENABLE_RCAS                = (1 << 0), ///< A bit indicating if we should use rcas.
    FFX_FSR1_RCAS_PASSTHROUGH_ALPHA     = (1 << 1), ///< A bit indicating if we should use passthrough alpha during rcas.
    FFX_FSR1_RCAS_DENOISE               = (1 << 2), ///< A bit indicating if denoising is invoked during rcas.
    FFX_FSR1_ENABLE_HIGH_DYNAMIC_RANGE  = (1 << 3), ///< A bit indicating if the input color data provided is using a high-dynamic range.
    FFX_FSR1_ENABLE_SRGB_CONVERSIONS    = (1 << 4), ///< A bit indicating that input/output resources require gamma conversions

} FfxFsr1InitializationFlagBits;

/// A structure encapsulating the parameters required to initialize FidelityFX
/// Super Resolution 1.0
///
/// @ingroup ffxFsr1
typedef struct FfxFsr1ContextDescription {

    uint32_t                    flags;                  ///< A collection of <c><i>FfxFsr1InitializationFlagBits</i></c>.
    FfxSurfaceFormat            outputFormat;           ///< Format of the output target used for creation of the internal upscale resource
    FfxDimensions2D             maxRenderSize;          ///< The maximum size that rendering will be performed at.
    FfxDimensions2D             displaySize;            ///< The size of the presentation resolution targeted by the upscaling process.
    FfxInterface                backendInterface;       ///< A set of pointers to the backend implementation for FSR1.
} FfxFsr1ContextDescription;

/// A structure encapsulating the parameters for dispatching the various passes
/// of FidelityFX Super Resolution 1.0
///
/// @ingroup ffxFsr1
typedef struct FfxFsr1DispatchDescription {

    FfxCommandList              commandList;        ///< The <c><i>FfxCommandList</i></c> to record FSR1 rendering commands into.
    FfxResource                 color;              ///< A <c><i>FfxResource</i></c> containing the color buffer for the current frame (at render resolution).
    FfxResource                 output;             ///< A <c><i>FfxResource</i></c> containing the output color buffer for the current frame (at presentation resolution).
    FfxDimensions2D             renderSize;         ///< The resolution that was used for rendering the input resource.
    bool                        enableSharpening;   ///< Enable an additional sharpening pass.
    float                       sharpness;          ///< The sharpness value between 0 and 1, where 0 is no additional sharpness and 1 is maximum additional sharpness.
} FfxFsr1DispatchDescription;

/// A structure encapsulating the FidelityFX Super Resolution 1.0 context.
///
/// This sets up an object which contains all persistent internal data and
/// resources that are required by FSR1.
///
/// The <c><i>FfxFsr1Context</i></c> object should have a lifetime matching
/// your use of FSR1. Before destroying the FSR1 context care should be taken
/// to ensure the GPU is not accessing the resources created or used by FSR1.
/// It is therefore recommended that the GPU is idle before destroying the
/// FSR1 context.
///
/// @ingroup ffxFsr1
typedef struct FfxFsr1Context {

    uint32_t                    data[FFX_FSR1_CONTEXT_SIZE];  ///< An opaque set of <c>uint32_t</c> which contain the data for the context.
} FfxFsr1Context;


/// Create a FidelityFX Super Resolution 1.0 context from the parameters
/// programmed to the <c><i>FfxFsr1ContextDescription</i></c> structure.
///
/// The context structure is the main object used to interact with the Super
/// Resoution 1.0 API, and is responsible for the management of the internal resources
/// used by the FSR1 algorithm. When this API is called, multiple calls
/// will be made via the pointers contained in the <c><i>callbacks</i></c>
/// structure. These callbacks will attempt to retreive the device capabilities,
/// and create the internal resources, and pipelines required by FSR1
/// frame-to-frame function. Depending on the precise configuration used when
/// creating the <c><i>FfxFsr1Context</i></c> a different set of resources and
/// pipelines might be requested via the callback functions.
///
/// The <c><i>FfxParallelSortContext</i></c> should be destroyed when use of it is
/// completed, typically when an application is unloaded or FSR1
/// upscaling is disabled by a user. To destroy the FSR1 context you
/// should call <c><i>ffxFsr1ContextDestroy</i></c>.
///
/// @param [out] pContext                A pointer to a <c><i>FfxFsr1Context</i></c> structure to populate.
/// @param [in]  pContextDescription     A pointer to a <c><i>FfxFsr1ContextDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxFsr1ContextDescription.callbacks</i></c>  was not fully specified.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup ffxFsr1
FFX_API FfxErrorCode ffxFsr1ContextCreate(FfxFsr1Context* pContext, const FfxFsr1ContextDescription* pContextDescription);

/// Get GPU memory usage of the FidelityFX Super Resolution context.
///
/// @param [in]  pContext                A pointer to a <c><i>FfxFsr1Context</i></c> structure.
/// @param [out] pVramUsage              A pointer to a <c><i>FfxEffectMemoryUsage</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>vramUsage</i></c> were <c><i>NULL</i></c>.
///
/// @ingroup ffxFsr1
FFX_API FfxErrorCode ffxFsr1ContextGetGpuMemoryUsage(FfxFsr1Context* pContext, FfxEffectMemoryUsage* pVramUsage);

/// @param [out] pContext                A pointer to a <c><i>FfxFsr1Context</i></c> structure to populate.
/// @param [in]  pDispatchDescription    A pointer to a <c><i>FfxFsr1DispatchDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>dispatchDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup ffxFsr1
FFX_API FfxErrorCode ffxFsr1ContextDispatch(FfxFsr1Context* pContext, const FfxFsr1DispatchDescription* pDispatchDescription);

/// Destroy the FidelityFX FSR 1 context.
///
/// @param [out] pContext                A pointer to a <c><i>FfxFsr1Context</i></c> structure to destroy.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>.
///
/// @ingroup ffxFsr1
FFX_API FfxErrorCode ffxFsr1ContextDestroy(FfxFsr1Context* pContext);

/// Get the upscale ratio from the quality mode.
///
/// The following table enumerates the mapping of the quality modes to
/// per-dimension scaling ratios.
///
/// Quality preset                                        | Scale factor
/// ----------------------------------------------------- | -------------
/// <c><i>FFX_FSR1_QUALITY_MODE_ULTRA_QUALITY</i></c>     | 1.3x
/// <c><i>FFX_FSR1_QUALITY_MODE_QUALITY</i></c>           | 1.5x
/// <c><i>FFX_FSR1_QUALITY_MODE_BALANCED</i></c>          | 1.7x
/// <c><i>FFX_FSR1_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x
///
/// Passing an invalid <c><i>qualityMode</i></c> will return 0.0f.
///
/// @param [in] qualityMode             The quality mode preset.
///
/// @returns
/// The upscaling the per-dimension upscaling ratio for
/// <c><i>qualityMode</i></c> according to the table above.
///
/// @ingroup ffxFsr1
FFX_API float ffxFsr1GetUpscaleRatioFromQualityMode(FfxFsr1QualityMode qualityMode);

/// A helper function to calculate the rendering resolution from a target
/// resolution and desired quality level.
///
/// This function applies the scaling factor returned by
/// <c><i>ffxFsr1GetUpscaleRatioFromQualityMode</i></c> to each dimension.
///
/// @param [out] pRenderWidth            A pointer to a <c>uint32_t</c> which will hold the calculated render resolution width.
/// @param [out] pRenderHeight           A pointer to a <c>uint32_t</c> which will hold the calculated render resolution height.
/// @param [in] displayWidth            The target display resolution width.
/// @param [in] displayHeight           The target display resolution height.
/// @param [in] qualityMode             The desired quality mode for FSR1 upscaling.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_INVALID_POINTER           Either <c><i>renderWidth</i></c> or <c><i>renderHeight</i></c> was <c>NULL</c>.
/// @retval
/// FFX_ERROR_INVALID_ENUM              An invalid quality mode was specified.
///
/// @ingroup ffxFsr1
FFX_API FfxErrorCode ffxFsr1GetRenderResolutionFromQualityMode(
    uint32_t* pRenderWidth,
    uint32_t* pRenderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    FfxFsr1QualityMode qualityMode);

/// Queries the effect version number.
///
/// @returns
/// The SDK version the effect was built with.
///
/// @ingroup ffxFsr1
FFX_API FfxVersionNumber ffxFsr1GetEffectVersion();

#if defined(__cplusplus)
}
#endif // #if defined(__cplusplus)
