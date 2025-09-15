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


// @defgroup FSR2

#pragma once

// Include the interface for the backend of the FSR2 API.
#include "ffx_fsr2_interface.h"

/// FidelityFX Super Resolution 2 major version.
///
/// @ingroup FSR2
#define FFX_FSR2_VERSION_MAJOR      (2)

/// FidelityFX Super Resolution 2 minor version.
///
/// @ingroup FSR2
#define FFX_FSR2_VERSION_MINOR      (2)

/// FidelityFX Super Resolution 2 patch version.
///
/// @ingroup FSR2
#define FFX_FSR2_VERSION_PATCH      (1)

/// The size of the context specified in 32bit values.
///
/// @ingroup FSR2
#define FFX_FSR2_CONTEXT_SIZE       (16536)

#if defined(__cplusplus)
extern "C" {
#endif // #if defined(__cplusplus)

/// An enumeration of all the quality modes supported by FidelityFX Super
/// Resolution 2 upscaling.
///
/// In order to provide a consistent user experience across multiple
/// applications which implement FSR2. It is strongly recommended that the
/// following preset scaling factors are made available through your
/// application's user interface.
///
/// If your application does not expose the notion of preset scaling factors
/// for upscaling algorithms (perhaps instead implementing a fixed ratio which
/// is immutable) or implementing a more dynamic scaling scheme (such as
/// dynamic resolution scaling), then there is no need to use these presets.
///
/// Please note that <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> is
/// an optional mode which may introduce significant quality degradation in the
/// final image. As such it is recommended that you evaluate the final results
/// of using this scaling mode before deciding if you should include it in your
/// application.
///
/// @ingroup FSR2
typedef enum FfxFsr2QualityMode {

    FFX_FSR2_QUALITY_MODE_QUALITY                       = 1,        ///< Perform upscaling with a per-dimension upscaling ratio of 1.5x.
    FFX_FSR2_QUALITY_MODE_BALANCED                      = 2,        ///< Perform upscaling with a per-dimension upscaling ratio of 1.7x.
    FFX_FSR2_QUALITY_MODE_PERFORMANCE                   = 3,        ///< Perform upscaling with a per-dimension upscaling ratio of 2.0x.
    FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE             = 4         ///< Perform upscaling with a per-dimension upscaling ratio of 3.0x.
} FfxFsr2QualityMode;

/// An enumeration of bit flags used when creating a
/// <c><i>FfxFsr2Context</i></c>. See <c><i>FfxFsr2ContextDescription</i></c>.
///
/// @ingroup FSR2
typedef enum FfxFsr2InitializationFlagBits {

    FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE                  = (1<<0),   ///< A bit indicating if the input color data provided is using a high-dynamic range.
    FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS   = (1<<1),   ///< A bit indicating if the motion vectors are rendered at display resolution.
    FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION  = (1<<2),   ///< A bit indicating that the motion vectors have the jittering pattern applied to them.
    FFX_FSR2_ENABLE_DEPTH_INVERTED                      = (1<<3),   ///< A bit indicating that the input depth buffer data provided is inverted [1..0].
    FFX_FSR2_ENABLE_DEPTH_INFINITE                      = (1<<4),   ///< A bit indicating that the input depth buffer data provided is using an infinite far plane.
    FFX_FSR2_ENABLE_AUTO_EXPOSURE                       = (1<<5),   ///< A bit indicating if automatic exposure should be applied to input color data.
    FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION                  = (1<<6),   ///< A bit indicating that the application uses dynamic resolution scaling.
    FFX_FSR2_ENABLE_TEXTURE1D_USAGE                     = (1<<7),   ///< A bit indicating that the backend should use 1D textures.
    FFX_FSR2_ENABLE_DEBUG_CHECKING                      = (1<<8),   ///< A bit indicating that the runtime should check some API values and report issues.
} FfxFsr2InitializationFlagBits;

/// A structure encapsulating the parameters required to initialize FidelityFX
/// Super Resolution 2 upscaling.
///
/// @ingroup FSR2
typedef struct FfxFsr2ContextDescription {

    uint32_t                    flags;                              ///< A collection of <c><i>FfxFsr2InitializationFlagBits</i></c>.
    FfxDimensions2D             maxRenderSize;                      ///< The maximum size that rendering will be performed at.
    FfxDimensions2D             displaySize;                        ///< The size of the presentation resolution targeted by the upscaling process.
    FfxFsr2Interface            callbacks;                          ///< A set of pointers to the backend implementation for FSR 2.0.
    FfxDevice                   device;                             ///< The abstracted device which is passed to some callback functions.

    FfxFsr2Message              fpMessage;                          ///< A pointer to a function that can recieve messages from the runtime.
} FfxFsr2ContextDescription;

/// A structure encapsulating the parameters for dispatching the various passes
/// of FidelityFX Super Resolution 2.
///
/// @ingroup FSR2
typedef struct FfxFsr2DispatchDescription {

    FfxCommandList              commandList;                        ///< The <c><i>FfxCommandList</i></c> to record FSR2 rendering commands into.
    FfxResource                 color;                              ///< A <c><i>FfxResource</i></c> containing the color buffer for the current frame (at render resolution).
    FfxResource                 depth;                              ///< A <c><i>FfxResource</i></c> containing 32bit depth values for the current frame (at render resolution).
    FfxResource                 motionVectors;                      ///< A <c><i>FfxResource</i></c> containing 2-dimensional motion vectors (at render resolution if <c><i>FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS</i></c> is not set).
    FfxResource                 exposure;                           ///< A optional <c><i>FfxResource</i></c> containing a 1x1 exposure value.
    FfxResource                 reactive;                           ///< A optional <c><i>FfxResource</i></c> containing alpha value of reactive objects in the scene.
    FfxResource                 transparencyAndComposition;         ///< A optional <c><i>FfxResource</i></c> containing alpha value of special objects in the scene.
    FfxResource                 output;                             ///< A <c><i>FfxResource</i></c> containing the output color buffer for the current frame (at presentation resolution).
    FfxFloatCoords2D            jitterOffset;                       ///< The subpixel jitter offset applied to the camera.
    FfxFloatCoords2D            motionVectorScale;                  ///< The scale factor to apply to motion vectors.
    FfxDimensions2D             renderSize;                         ///< The resolution that was used for rendering the input resources.
    bool                        enableSharpening;                   ///< Enable an additional sharpening pass.
    float                       sharpness;                          ///< The sharpness value between 0 and 1, where 0 is no additional sharpness and 1 is maximum additional sharpness.
    float                       frameTimeDelta;                     ///< The time elapsed since the last frame (expressed in milliseconds).
    float                       preExposure;                        ///< The pre exposure value (must be > 0.0f)
    bool                        reset;                              ///< A boolean value which when set to true, indicates the camera has moved discontinuously.
    float                       cameraNear;                         ///< The distance to the near plane of the camera.
    float                       cameraFar;                          ///< The distance to the far plane of the camera.
    float                       cameraFovAngleVertical;             ///< The camera angle field of view in the vertical direction (expressed in radians).
    float                       viewSpaceToMetersFactor;            ///< The scale factor to convert view space units to meters

    // EXPERIMENTAL reactive mask generation parameters
    bool                        enableAutoReactive;                 ///< A boolean value to indicate internal reactive autogeneration should be used
    FfxResource                 colorOpaqueOnly;                    ///< A <c><i>FfxResource</i></c> containing the opaque only color buffer for the current frame (at render resolution).
    float                       autoTcThreshold;                    ///< Cutoff value for TC
    float                       autoTcScale;                        ///< A value to scale the transparency and composition mask
    float                       autoReactiveScale;                  ///< A value to scale the reactive mask
    float                       autoReactiveMax;                    ///< A value to clamp the reactive mask

    float                       reprojectionMatrix[16];             ///< The matrix used for reprojecting pixels with invalid motion vectors by using the depth.
} FfxFsr2DispatchDescription;

/// A structure encapsulating the parameters for automatic generation of a reactive mask
///
/// @ingroup FSR2
typedef struct FfxFsr2GenerateReactiveDescription {

    FfxCommandList              commandList;                        ///< The <c><i>FfxCommandList</i></c> to record FSR2 rendering commands into.
    FfxResource                 colorOpaqueOnly;                    ///< A <c><i>FfxResource</i></c> containing the opaque only color buffer for the current frame (at render resolution).
    FfxResource                 colorPreUpscale;                    ///< A <c><i>FfxResource</i></c> containing the opaque+translucent color buffer for the current frame (at render resolution).
    FfxResource                 outReactive;                        ///< A <c><i>FfxResource</i></c> containing the surface to generate the reactive mask into.
    FfxDimensions2D             renderSize;                         ///< The resolution that was used for rendering the input resources.
    float                       scale;                              ///< A value to scale the output
    float                       cutoffThreshold;                    ///< A threshold value to generate a binary reactive mask
    float                       binaryValue;                        ///< A value to set for the binary reactive mask
    uint32_t                    flags;                              ///< Flags to determine how to generate the reactive mask
} FfxFsr2GenerateReactiveDescription;

/// A structure encapsulating the FidelityFX Super Resolution 2 context.
///
/// This sets up an object which contains all persistent internal data and
/// resources that are required by FSR2.
///
/// The <c><i>FfxFsr2Context</i></c> object should have a lifetime matching
/// your use of FSR2. Before destroying the FSR2 context care should be taken
/// to ensure the GPU is not accessing the resources created or used by FSR2.
/// It is therefore recommended that the GPU is idle before destroying the
/// FSR2 context.
///
/// @ingroup FSR2
typedef struct FfxFsr2Context {

    uint32_t                    data[FFX_FSR2_CONTEXT_SIZE];        ///< An opaque set of <c>uint32_t</c> which contain the data for the context.
} FfxFsr2Context;

/// Create a FidelityFX Super Resolution 2 context from the parameters
/// programmed to the <c><i>FfxFsr2CreateParams</i></c> structure.
///
/// The context structure is the main object used to interact with the FSR2
/// API, and is responsible for the management of the internal resources used
/// by the FSR2 algorithm. When this API is called, multiple calls will be
/// made via the pointers contained in the <c><i>callbacks</i></c> structure.
/// These callbacks will attempt to retreive the device capabilities, and
/// create the internal resources, and pipelines required by FSR2's
/// frame-to-frame function. Depending on the precise configuration used when
/// creating the <c><i>FfxFsr2Context</i></c> a different set of resources and
/// pipelines might be requested via the callback functions.
///
/// The flags included in the <c><i>flags</i></c> field of
/// <c><i>FfxFsr2Context</i></c> how match the configuration of your
/// application as well as the intended use of FSR2. It is important that these
/// flags are set correctly (as well as a correct programmed
/// <c><i>FfxFsr2DispatchDescription</i></c>) to ensure correct operation. It is
/// recommended to consult the overview documentation for further details on
/// how FSR2 should be integerated into an application.
///
/// When the <c><i>FfxFsr2Context</i></c> is created, you should use the
/// <c><i>ffxFsr2ContextDispatch</i></c> function each frame where FSR2
/// upscaling should be applied. See the documentation of
/// <c><i>ffxFsr2ContextDispatch</i></c> for more details.
///
/// The <c><i>FfxFsr2Context</i></c> should be destroyed when use of it is
/// completed, typically when an application is unloaded or FSR2 upscaling is
/// disabled by a user. To destroy the FSR2 context you should call
/// <c><i>ffxFsr2ContextDestroy</i></c>.
///
/// @param [out] context                A pointer to a <c><i>FfxFsr2Context</i></c> structure to populate.
/// @param [in]  contextDescription     A pointer to a <c><i>FfxFsr2ContextDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxFsr2ContextDescription.callbacks</i></c>  was not fully specified.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2ContextCreate(FfxFsr2Context* context, const FfxFsr2ContextDescription* contextDescription);

/// Dispatch the various passes that constitute FidelityFX Super Resolution 2.
///
/// FSR2 is a composite effect, meaning that it is compromised of multiple
/// constituent passes (implemented as one or more clears, copies and compute
/// dispatches). The <c><i>ffxFsr2ContextDispatch</i></c> function is the
/// function which (via the use of the functions contained in the
/// <c><i>callbacks</i></c> field of the <c><i>FfxFsr2Context</i></c>
/// structure) utlimately generates the sequence of graphics API calls required
/// each frame.
///
/// As with the creation of the <c><i>FfxFsr2Context</i></c> correctly
/// programming the <c><i>FfxFsr2DispatchDescription</i></c> is key to ensuring
/// the correct operation of FSR2. It is particularly important to ensure that
/// camera jitter is correctly applied to your application's projection matrix
/// (or camera origin for raytraced applications). FSR2 provides the
/// <c><i>ffxFsr2GetJitterPhaseCount</i></c> and
/// <c><i>ffxFsr2GetJitterOffset</i></c> entry points to help applications
/// correctly compute the camera jitter. Whatever jitter pattern is used by the
/// application it should be correctly programmed to the
/// <c><i>jitterOffset</i></c> field of the <c><i>dispatchDescription</i></c>
/// structure. For more guidance on camera jitter please consult the
/// documentation for <c><i>ffxFsr2GetJitterOffset</i></c> as well as the
/// accompanying overview documentation for FSR2.
///
/// @param [in] context                 A pointer to a <c><i>FfxFsr2Context</i></c> structure.
/// @param [in] dispatchDescription     A pointer to a <c><i>FfxFsr2DispatchDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>dispatchDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_OUT_OF_RANGE              The operation failed because <c><i>dispatchDescription.renderSize</i></c> was larger than the maximum render resolution.
/// @retval
/// FFX_ERROR_NULL_DEVICE               The operation failed because the device inside the context was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2ContextDispatch(FfxFsr2Context* context, const FfxFsr2DispatchDescription* dispatchDescription);

/// A helper function generate a Reactive mask from an opaque only texure and one containing translucent objects.
///
/// @param [in] context                 A pointer to a <c><i>FfxFsr2Context</i></c> structure.
/// @param [in] params                  A pointer to a <c><i>FfxFsr2GenerateReactiveDescription</i></c> structure
///
/// @retval
/// FFX_OK                              The operation completed successfully.
///
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2ContextGenerateReactiveMask(FfxFsr2Context* context, const FfxFsr2GenerateReactiveDescription* params);

/// Destroy the FidelityFX Super Resolution context.
///
/// @param [out] context                A pointer to a <c><i>FfxFsr2Context</i></c> structure to destroy.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>.
///
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2ContextDestroy(FfxFsr2Context* context);

/// Get the upscale ratio from the quality mode.
///
/// The following table enumerates the mapping of the quality modes to
/// per-dimension scaling ratios.
///
/// Quality preset                                        | Scale factor
/// ----------------------------------------------------- | -------------
/// <c><i>FFX_FSR2_QUALITY_MODE_QUALITY</i></c>           | 1.5x
/// <c><i>FFX_FSR2_QUALITY_MODE_BALANCED</i></c>          | 1.7x
/// <c><i>FFX_FSR2_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x
/// <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x
///
/// Passing an invalid <c><i>qualityMode</i></c> will return 0.0f.
///
/// @param [in] qualityMode             The quality mode preset.
///
/// @returns
/// The upscaling the per-dimension upscaling ratio for
/// <c><i>qualityMode</i></c> according to the table above.
///
/// @ingroup FSR2
FFX_API float ffxFsr2GetUpscaleRatioFromQualityMode(FfxFsr2QualityMode qualityMode);

/// A helper function to calculate the rendering resolution from a target
/// resolution and desired quality level.
///
/// This function applies the scaling factor returned by
/// <c><i>ffxFsr2GetUpscaleRatioFromQualityMode</i></c> to each dimension.
///
/// @param [out] renderWidth            A pointer to a <c>uint32_t</c> which will hold the calculated render resolution width.
/// @param [out] renderHeight           A pointer to a <c>uint32_t</c> which will hold the calculated render resolution height.
/// @param [in] displayWidth            The target display resolution width.
/// @param [in] displayHeight           The target display resolution height.
/// @param [in] qualityMode             The desired quality mode for FSR 2 upscaling.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_INVALID_POINTER           Either <c><i>renderWidth</i></c> or <c><i>renderHeight</i></c> was <c>NULL</c>.
/// @retval
/// FFX_ERROR_INVALID_ENUM              An invalid quality mode was specified.
///
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2GetRenderResolutionFromQualityMode(
    uint32_t* renderWidth,
    uint32_t* renderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    FfxFsr2QualityMode qualityMode);

/// A helper function to calculate the jitter phase count from display
/// resolution.
///
/// For more detailed information about the application of camera jitter to
/// your application's rendering please refer to the
/// <c><i>ffxFsr2GetJitterOffset</i></c> function.
/// 
/// The table below shows the jitter phase count which this function
/// would return for each of the quality presets.
///
/// Quality preset                                        | Scale factor  | Phase count
/// ----------------------------------------------------- | ------------- | ---------------
/// <c><i>FFX_FSR2_QUALITY_MODE_QUALITY</i></c>           | 1.5x          | 18
/// <c><i>FFX_FSR2_QUALITY_MODE_BALANCED</i></c>          | 1.7x          | 23
/// <c><i>FFX_FSR2_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x          | 32
/// <c><i>FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x          | 72
/// Custom                                                | [1..n]x       | ceil(8*n^2)
///
/// @param [in] renderWidth             The render resolution width.
/// @param [in] displayWidth            The display resolution width.
///
/// @returns
/// The jitter phase count for the scaling factor between <c><i>renderWidth</i></c> and <c><i>displayWidth</i></c>.
///
/// @ingroup FSR2
FFX_API int32_t ffxFsr2GetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth);

/// A helper function to calculate the subpixel jitter offset.
///
/// FSR2 relies on the application to apply sub-pixel jittering while rendering.
/// This is typically included in the projection matrix of the camera. To make
/// the application of camera jitter simple, the FSR2 API provides a small set
/// of utility function which computes the sub-pixel jitter offset for a
/// particular frame within a sequence of separate jitter offsets. To begin, the
/// index within the jitter phase must be computed. To calculate the
/// sequence's length, you can call the <c><i>ffxFsr2GetJitterPhaseCount</i></c>
/// function. The index should be a value which is incremented each frame modulo
/// the length of the sequence computed by <c><i>ffxFsr2GetJitterPhaseCount</i></c>.
/// The index within the jitter phase  is passed to
/// <c><i>ffxFsr2GetJitterOffset</i></c> via the <c><i>index</i></c> parameter.
///
/// This function uses a Halton(2,3) sequence to compute the jitter offset.
/// The ultimate index used for the sequence is <c><i>index</i></c> %
/// <c><i>phaseCount</i></c>.
///
/// It is important to understand that the values returned from the
/// <c><i>ffxFsr2GetJitterOffset</i></c> function are in unit pixel space, and
/// in order to composite this correctly into a projection matrix we must
/// convert them into projection offsets. This is done as per the pseudo code
/// listing which is shown below.
///
///     const int32_t jitterPhaseCount = ffxFsr2GetJitterPhaseCount(renderWidth, displayWidth);
///
///     float jitterX = 0;
///     float jitterY = 0;
///     ffxFsr2GetJitterOffset(&jitterX, &jitterY, index, jitterPhaseCount);
/// 
///     const float jitterX = 2.0f * jitterX / (float)renderWidth;
///     const float jitterY = -2.0f * jitterY / (float)renderHeight;
///     const Matrix4 jitterTranslationMatrix = translateMatrix(Matrix3::identity, Vector3(jitterX, jitterY, 0));
///     const Matrix4 jitteredProjectionMatrix = jitterTranslationMatrix * projectionMatrix;
/// 
/// Jitter should be applied to all rendering. This includes opaque, alpha
/// transparent, and raytraced objects. For rasterized objects, the sub-pixel
/// jittering values calculated by the <c><i>iffxFsr2GetJitterOffset</i></c>
/// function can be applied to the camera projection matrix which is ultimately
/// used to perform transformations during vertex shading. For raytraced
/// rendering, the sub-pixel jitter should be applied to the ray's origin,
/// often the camera's position.
/// 
/// Whether you elect to use the <c><i>ffxFsr2GetJitterOffset</i></c> function
/// or your own sequence generator, you must program the
/// <c><i>jitterOffset</i></c> field of the
/// <c><i>FfxFsr2DispatchParameters</i></c> structure in order to inform FSR2
/// of the jitter offset that has been applied in order to render each frame.
/// 
/// If not using the recommended <c><i>ffxFsr2GetJitterOffset</i></c> function,
/// care should be taken that your jitter sequence never generates a null vector;
/// that is value of 0 in both the X and Y dimensions.
///
/// @param [out] outX                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the x dimension.
/// @param [out] outY                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the y dimension.
/// @param [in] index                   The index within the jitter sequence.
/// @param [in] phaseCount              The length of jitter phase. See <c><i>ffxFsr2GetJitterPhaseCount</i></c>.
/// 
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_INVALID_POINTER           Either <c><i>outX</i></c> or <c><i>outY</i></c> was <c>NULL</c>.
/// @retval
/// FFX_ERROR_INVALID_ARGUMENT          Argument <c><i>phaseCount</i></c> must be greater than 0.
/// 
/// @ingroup FSR2
FFX_API FfxErrorCode ffxFsr2GetJitterOffset(float* outX, float* outY, int32_t index, int32_t phaseCount);

/// A helper function to check if a resource is
/// <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>.
///
/// @param [in] resource                A <c><i>FfxResource</i></c>.
///
/// @returns
/// true                                The <c><i>resource</i></c> was not <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>.
/// @returns
/// false                               The <c><i>resource</i></c> was <c><i>FFX_FSR2_RESOURCE_IDENTIFIER_NULL</i></c>.
///
/// @ingroup FSR2
FFX_API bool ffxFsr2ResourceIsNull(FfxResource resource);

#if defined(__cplusplus)
}
#endif // #if defined(__cplusplus)
