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

// Include the interface for the backend of the FSR3 API.
#include "ffx_interface.h"

/// @defgroup ffxFsr3Upscaler FidelityFX FSR3
/// FidelityFX Super Resolution 3 runtime library
///
/// @ingroup SDKComponents

/// FidelityFX Super Resolution 3 major version.
///
/// @ingroup ffxFsr3Upscaler
#define FFX_FSR3UPSCALER_VERSION_MAJOR      (3)

/// FidelityFX Super Resolution 3 minor version.
///
/// @ingroup ffxFsr3Upscaler
#define FFX_FSR3UPSCALER_VERSION_MINOR      (1)

/// FidelityFX Super Resolution 3 patch version.
///
/// @ingroup ffxFsr3Upscaler
#define FFX_FSR3UPSCALER_VERSION_PATCH      (4)

/// FidelityFX Super Resolution 3 context count
///
/// Defines the number of internal effect contexts required by FSR3
///
/// @ingroup ffxFsr3Upscaler
#define FFX_FSR3UPSCALER_CONTEXT_COUNT   1

/// The size of the context specified in 32bit values.
///
/// @ingroup ffxFsr3Upscaler
#define FFX_FSR3UPSCALER_CONTEXT_SIZE (FFX_SDK_DEFAULT_CONTEXT_SIZE)

#if defined(__cplusplus)
extern "C" {
#endif // #if defined(__cplusplus)

/// An enumeration of all the passes which constitute the FSR3 algorithm.
///
/// FSR3 is implemented as a composite of several compute passes each
/// computing a key part of the final result. Each call to the
/// <c><i>FfxFsr3UpscalerScheduleGpuJobFunc</i></c> callback function will
/// correspond to a single pass included in <c><i>FfxFsr3UpscalerPass</i></c>. For a
/// more comprehensive description of each pass, please refer to the FSR3
/// reference documentation.
///
/// Please note in some cases e.g.: <c><i>FFX_FSR3UPSCALER_PASS_ACCUMULATE</i></c>
/// and <c><i>FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN</i></c> either one pass or the
/// other will be used (they are mutually exclusive). The choice of which will
/// depend on the way the <c><i>FfxFsr3UpscalerContext</i></c> is created and the
/// precise contents of <c><i>FfxFsr3UpscalerDispatchParamters</i></c> each time a call
/// is made to <c><i>ffxFsr3UpscalerContextDispatch</i></c>.
///
/// @ingroup ffxFsr3Upscaler
typedef enum FfxFsr3UpscalerPass
{
    FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS,                           ///< A pass which prepares game inputs for later passes
    FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID,                             ///< A pass which generates the luminance mipmap chain for the current frame.
    FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID,                   ///< A pass which generates the shading change detection mipmap chain for the current frame.
    FFX_FSR3UPSCALER_PASS_SHADING_CHANGE,                           ///< A pass which estimates shading changes for the current frame
    FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY,                       ///< A pass which prepares accumulation relevant information
    FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY,                         ///< A pass which estimates temporal instability of the luminance changes.
    FFX_FSR3UPSCALER_PASS_ACCUMULATE,                               ///< A pass which performs upscaling.
    FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN,                       ///< A pass which performs upscaling when sharpening is used.
    FFX_FSR3UPSCALER_PASS_RCAS,                                     ///< A pass which performs sharpening.
    FFX_FSR3UPSCALER_PASS_DEBUG_VIEW,                               ///< A pass which draws some internal resources, for debugging purposes

    FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE,                        ///< An optional pass to generate a reactive mask.
    FFX_FSR3UPSCALER_PASS_TCR_AUTOGENERATE,                         ///< DEPRECATED - NO LONGER SUPPORTED
    FFX_FSR3UPSCALER_PASS_COUNT  ///< The number of passes performed by FSR3.
} FfxFsr3UpscalerPass;

/// An enumeration of all the quality modes supported by FidelityFX Super
/// Resolution 3 upscaling.
///
/// In order to provide a consistent user experience across multiple
/// applications which implement FSR3. It is strongly recommended that the
/// following preset scaling factors are made available through your
/// application's user interface.
///
/// If your application does not expose the notion of preset scaling factors
/// for upscaling algorithms (perhaps instead implementing a fixed ratio which
/// is immutable) or implementing a more dynamic scaling scheme (such as
/// dynamic resolution scaling), then there is no need to use these presets.
///
/// Please note that <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> is
/// an optional mode which may introduce significant quality degradation in the
/// final image. As such it is recommended that you evaluate the final results
/// of using this scaling mode before deciding if you should include it in your
/// application.
///
/// @ingroup ffxFsr3Upscaler
typedef enum FfxFsr3UpscalerQualityMode {
    FFX_FSR3UPSCALER_QUALITY_MODE_NATIVEAA                      = 0,        ///< Perform upscaling with a per-dimension upscaling ratio of 1.0x.
    FFX_FSR3UPSCALER_QUALITY_MODE_QUALITY                       = 1,        ///< Perform upscaling with a per-dimension upscaling ratio of 1.5x.
    FFX_FSR3UPSCALER_QUALITY_MODE_BALANCED                      = 2,        ///< Perform upscaling with a per-dimension upscaling ratio of 1.7x.
    FFX_FSR3UPSCALER_QUALITY_MODE_PERFORMANCE                   = 3,        ///< Perform upscaling with a per-dimension upscaling ratio of 2.0x.
    FFX_FSR3UPSCALER_QUALITY_MODE_ULTRA_PERFORMANCE             = 4         ///< Perform upscaling with a per-dimension upscaling ratio of 3.0x.
} FfxFsr3UpscalerQualityMode;

/// An enumeration of bit flags used when creating a
/// <c><i>FfxFsr3UpscalerContext</i></c>. See <c><i>FfxFsr3UpscalerContextDescription</i></c>.
///
/// @ingroup ffxFsr3Upscaler
typedef enum FfxFsr3UpscalerInitializationFlagBits {

    FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE                  = (1<<0),   ///< A bit indicating if the input color data provided is using a high-dynamic range.
    FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS   = (1<<1),   ///< A bit indicating if the motion vectors are rendered at display resolution.
    FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION  = (1<<2),   ///< A bit indicating that the motion vectors have the jittering pattern applied to them.
    FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED                      = (1<<3),   ///< A bit indicating that the input depth buffer data provided is inverted [1..0].
    FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE                      = (1<<4),   ///< A bit indicating that the input depth buffer data provided is using an infinite far plane.
    FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE                       = (1<<5),   ///< A bit indicating if automatic exposure should be applied to input color data.
    FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION                  = (1<<6),   ///< A bit indicating that the application uses dynamic resolution scaling.
    FFX_FSR3UPSCALER_ENABLE_TEXTURE1D_USAGE                     = (1<<7),   ///< This value is deprecated, but remains in order to aid upgrading from older versions of FSR3.
    FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING                      = (1<<8),   ///< A bit indicating that the runtime should check some API values and report issues.
} FfxFsr3UpscalerInitializationFlagBits;

/// Pass a string message
///
/// Used for debug messages.
///
/// @param [in] type                       The type of message.
/// @param [in] message                    A string message to pass.
///
///
/// @ingroup ffxFsr3Upscaler
typedef void(*FfxFsr3UpscalerMessage)(
    FfxMsgType type,
    const wchar_t* message);

/// A structure encapsulating the parameters required to initialize FidelityFX
/// Super Resolution 3 upscaling.
///
/// @ingroup ffxFsr3Upscaler
typedef struct FfxFsr3UpscalerContextDescription {

    uint32_t                    flags;                              ///< A collection of <c><i>FfxFsr3UpscalerInitializationFlagBits</i></c>.
    FfxDimensions2D             maxRenderSize;                      ///< The maximum size that rendering will be performed at.
    FfxDimensions2D             maxUpscaleSize;                     ///< The size of the output resolution targeted by the upscaling process.
    FfxFsr3UpscalerMessage      fpMessage;                          ///< A pointer to a function that can receive messages from the runtime.
    FfxInterface                backendInterface;                   ///< A set of pointers to the backend implementation for FidelityFX SDK

} FfxFsr3UpscalerContextDescription;

typedef enum FfxFsr3UpscalerDispatchFlags
{
    FFX_FSR3UPSCALER_DISPATCH_DRAW_DEBUG_VIEW = (1 << 0),  ///< A bit indicating that the interpolated output resource will contain debug views with relevant information.
} FfxFsr3UpscalerDispatchFlags;

typedef enum FfxFsr3UpscalerConfigureKey
{
    FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR = 0, //Override constant buffer fVelocityFactor. The float value is casted from void * ptr. Value of 0.0f can improve temporal stability of bright pixels. Default value is 1.0f. Value is clamped to [0.0f, 1.0f].
    FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE = 1, //Override constant buffer fReactivenessScale. The float value is casted from void * ptr. Meant for development purpose to test if writing a larger value to reactive mask, reduces ghosting. Default value is 1.0f. Value is clamped to [0.0f, +infinity].
    FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE =2, //Override fShadingChangeScale. Increasing this scales fsr3.1 computed shading change value at read to have higher reactiveness. Default value is 1.0f. Value is clamped to [0.0f, +infinity].
    FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME = 3, // Override constant buffer fAccumulationAddedPerFrame. Corresponds to amount of accumulation added per frame at pixel coordinate where disocclusion occured or when reactive mask value is > 0.0f. Decreasing this and drawing the ghosting object (IE no mv) to reactive mask with value close to 1.0f can decrease temporal ghosting. Decreasing this value could result in more thin feature pixels flickering. Default value is 0.333. Value is clamped to [0.0f, 1.0f].
    FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION = 4, //Override constant buffer fMinDisocclusionAccumulation. Increasing this value may reduce white pixel temporal flickering around swaying thin objects that are disoccluding one another often. Too high value may increase ghosting. Default value is -0.333. A sufficiently negative value means for pixel coordinate at frame N that is disoccluded, add fAccumulationAddedPerFrame starting at frame N+2. Default value is -0.333. Value is clamped to [-1.0f, 1.0f].
} FfxFsr3UpscalerConfigureKey;

/// A structure encapsulating the parameters for dispatching the various passes
/// of FidelityFX Super Resolution 3.
///
/// @ingroup ffxFsr3Upscaler
typedef struct FfxFsr3UpscalerDispatchDescription {

    FfxCommandList              commandList;                        ///< The <c><i>FfxCommandList</i></c> to record FSR3 rendering commands into.
    FfxResource                 color;                              ///< A <c><i>FfxResource</i></c> containing the color buffer for the current frame (at render resolution).
    FfxResource                 depth;                              ///< A <c><i>FfxResource</i></c> containing 32bit depth values for the current frame (at render resolution).
    FfxResource                 motionVectors;                      ///< A <c><i>FfxResource</i></c> containing 2-dimensional motion vectors (at render resolution if <c><i>FFX_FSR3UPSCALER_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS</i></c> is not set).
    FfxResource                 exposure;                           ///< A optional <c><i>FfxResource</i></c> containing a 1x1 exposure value.
    FfxResource                 reactive;                           ///< A optional <c><i>FfxResource</i></c> containing alpha value of reactive objects in the scene.
    FfxResource                 transparencyAndComposition;         ///< A optional <c><i>FfxResource</i></c> containing alpha value of special objects in the scene.
    FfxResource                 dilatedDepth;                       ///< A <c><i>FfxResource</i></c> allocated as described in <c><i>FfxFsr3UpscalerSharedResourceDescriptions</i></c> that is used to emit dilated depth and share with following effects.
    FfxResource                 dilatedMotionVectors;               ///< A <c><i>FfxResource</i></c> allocated as described in <c><i>FfxFsr3UpscalerSharedResourceDescriptions</i></c> that is used to emit dilated motion vectors and share with following effects.
    FfxResource                 reconstructedPrevNearestDepth;      ///< A <c><i>FfxResource</i></c> allocated as described in <c><i>FfxFsr3UpscalerSharedResourceDescriptions</i></c> that is used to emit reconstructed previous nearest depth and share with following effects.
    FfxResource                 output;                             ///< A <c><i>FfxResource</i></c> containing the output color buffer for the current frame (at presentation resolution).
    FfxFloatCoords2D            jitterOffset;                       ///< The subpixel jitter offset applied to the camera.
    FfxFloatCoords2D            motionVectorScale;                  ///< The scale factor to apply to motion vectors.
    FfxDimensions2D             renderSize;                         ///< The resolution that was used for rendering the input resources.
    FfxDimensions2D             upscaleSize;                        ///< The resolution that the upscaler will output.
    bool                        enableSharpening;                   ///< Enable an additional sharpening pass.
    float                       sharpness;                          ///< The sharpness value between 0 and 1, where 0 is no additional sharpness and 1 is maximum additional sharpness.
    float                       frameTimeDelta;                     ///< The time elapsed since the last frame (expressed in milliseconds).
    float                       preExposure;                        ///< The pre exposure value (must be > 0.0f)
    bool                        reset;                              ///< A boolean value which when set to true, indicates the camera has moved discontinuously.
    float                       cameraNear;                         ///< The distance to the near plane of the camera.
    float                       cameraFar;                          ///< The distance to the far plane of the camera.
    float                       cameraFovAngleVertical;             ///< The camera angle field of view in the vertical direction (expressed in radians).
    float                       viewSpaceToMetersFactor;            ///< The scale factor to convert view space units to meters
    uint32_t                    flags;                              ///< combination of FfxFsr3UpscalerDispatchFlags
} FfxFsr3UpscalerDispatchDescription;

/// A structure encapsulating the parameters for automatic generation of a reactive mask
///
/// @ingroup ffxFsr3Upscaler
typedef struct FfxFsr3UpscalerGenerateReactiveDescription {

    FfxCommandList              commandList;                        ///< The <c><i>FfxCommandList</i></c> to record FSR3 rendering commands into.
    FfxResource                 colorOpaqueOnly;                    ///< A <c><i>FfxResource</i></c> containing the opaque only color buffer for the current frame (at render resolution).
    FfxResource                 colorPreUpscale;                    ///< A <c><i>FfxResource</i></c> containing the opaque+translucent color buffer for the current frame (at render resolution).
    FfxResource                 outReactive;                        ///< A <c><i>FfxResource</i></c> containing the surface to generate the reactive mask into.
    FfxDimensions2D             renderSize;                         ///< The resolution that was used for rendering the input resources.
    float                       scale;                              ///< A value to scale the output
    float                       cutoffThreshold;                    ///< A threshold value to generate a binary reactive mask
    float                       binaryValue;                        ///< A value to set for the binary reactive mask
    uint32_t                    flags;                              ///< Flags to determine how to generate the reactive mask
} FfxFsr3UpscalerGenerateReactiveDescription;

/// A structure encapsulating the resource descriptions for shared resources for this effect.
///
/// @ingroup ffxFsr3Upscaler
typedef struct FfxFsr3UpscalerSharedResourceDescriptions {

    FfxCreateResourceDescription reconstructedPrevNearestDepth; ///< The <c><i>FfxCreateResourceDescription</i></c> for allocating the <c><i>reconstructedPrevNearestDepth</i></c> shared resource.
    FfxCreateResourceDescription dilatedDepth;					///< The <c><i>FfxCreateResourceDescription</i></c> for allocating the <c><i>dilatedDepth</i></c> shared resource.
    FfxCreateResourceDescription dilatedMotionVectors;			///< The <c><i>FfxCreateResourceDescription</i></c> for allocating the <c><i>dilatedMotionVectors</i></c> shared resource.
} FfxFsr3UpscalerSharedResourceDescriptions;

/// A structure encapsulating the FidelityFX Super Resolution 3 context.
///
/// This sets up an object which contains all persistent internal data and
/// resources that are required by FSR3.
///
/// The <c><i>FfxFsr3UpscalerContext</i></c> object should have a lifetime matching
/// your use of FSR3. Before destroying the FSR3 context care should be taken
/// to ensure the GPU is not accessing the resources created or used by FSR3.
/// It is therefore recommended that the GPU is idle before destroying the
/// FSR3 context.
///
/// @ingroup ffxFsr3Upscaler
typedef struct FfxFsr3UpscalerContext
{
    uint32_t data[FFX_FSR3UPSCALER_CONTEXT_SIZE];  ///< An opaque set of <c>uint32_t</c> which contain the data for the context.
} FfxFsr3UpscalerContext;


/// Create a FidelityFX Super Resolution 3 context from the parameters
/// programmed to the <c><i>FfxFsr3UpscalerCreateParams</i></c> structure.
///
/// The context structure is the main object used to interact with the FSR3
/// API, and is responsible for the management of the internal resources used
/// by the FSR3 algorithm. When this API is called, multiple calls will be
/// made via the pointers contained in the <c><i>callbacks</i></c> structure.
/// These callbacks will attempt to retreive the device capabilities, and
/// create the internal resources, and pipelines required by FSR3's
/// frame-to-frame function. Depending on the precise configuration used when
/// creating the <c><i>FfxFsr3UpscalerContext</i></c> a different set of resources and
/// pipelines might be requested via the callback functions.
///
/// The flags included in the <c><i>flags</i></c> field of
/// <c><i>FfxFsr3UpscalerContext</i></c> how match the configuration of your
/// application as well as the intended use of FSR3. It is important that these
/// flags are set correctly (as well as a correct programmed
/// <c><i>FfxFsr3UpscalerDispatchDescription</i></c>) to ensure correct operation. It is
/// recommended to consult the overview documentation for further details on
/// how FSR3 should be integerated into an application.
///
/// When the <c><i>FfxFsr3UpscalerContext</i></c> is created, you should use the
/// <c><i>ffxFsr3UpscalerContextDispatch</i></c> function each frame where FSR3
/// upscaling should be applied. See the documentation of
/// <c><i>ffxFsr3UpscalerContextDispatch</i></c> for more details.
///
/// The <c><i>FfxFsr3UpscalerContext</i></c> should be destroyed when use of it is
/// completed, typically when an application is unloaded or FSR3 upscaling is
/// disabled by a user. To destroy the FSR3 context you should call
/// <c><i>ffxFsr3UpscalerContextDestroy</i></c>.
///
/// @param [out] pContext                A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure to populate.
/// @param [in]  pContextDescription     A pointer to a <c><i>FfxFsr3UpscalerContextDescription</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>contextDescription</i></c> was <c><i>NULL</i></c>.
/// @retval
/// FFX_ERROR_INCOMPLETE_INTERFACE      The operation failed because the <c><i>FfxFsr3UpscalerContextDescription.callbacks</i></c>  was not fully specified.
/// @retval
/// FFX_ERROR_BACKEND_API_ERROR         The operation failed because of an error returned from the backend.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerContextCreate(FfxFsr3UpscalerContext* pContext, const FfxFsr3UpscalerContextDescription* pContextDescription);

/// Provides the descriptions for shared resources that must be allocated for this effect.
///
/// @param [in] context					A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure.
/// @param [out] SharedResources		A pointer to a <c><i>FfxFsr3UpscalerSharedResourceDescriptions</i></c> to populate.
///
/// @returns
/// FFX_OK								The operation completed successfully.
/// @returns
/// Anything else						The operation failed.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerGetSharedResourceDescriptions(FfxFsr3UpscalerContext* context, FfxFsr3UpscalerSharedResourceDescriptions* SharedResources);

/// Get GPU memory usage of the FidelityFX Super Resolution context.
///
/// @param [in]  pContext                A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure.
/// @param [out] pVramUsage              A pointer to a <c><i>FfxEffectMemoryUsage</i></c> structure.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> or <c><i>vramUsage</i></c> were <c><i>NULL</i></c>.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerContextGetGpuMemoryUsage(FfxFsr3UpscalerContext* pContext, FfxEffectMemoryUsage* pVramUsage);

/// Dispatch the various passes that constitute FidelityFX Super Resolution 3.
///
/// FSR3 is a composite effect, meaning that it is compromised of multiple
/// constituent passes (implemented as one or more clears, copies and compute
/// dispatches). The <c><i>ffxFsr3UpscalerContextDispatch</i></c> function is the
/// function which (via the use of the functions contained in the
/// <c><i>callbacks</i></c> field of the <c><i>FfxFsr3UpscalerContext</i></c>
/// structure) utlimately generates the sequence of graphics API calls required
/// each frame.
///
/// As with the creation of the <c><i>FfxFsr3UpscalerContext</i></c> correctly
/// programming the <c><i>FfxFsr3UpscalerDispatchDescription</i></c> is key to ensuring
/// the correct operation of FSR3. It is particularly important to ensure that
/// camera jitter is correctly applied to your application's projection matrix
/// (or camera origin for raytraced applications). FSR3 provides the
/// <c><i>ffxFsr3UpscalerGetJitterPhaseCount</i></c> and
/// <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> entry points to help applications
/// correctly compute the camera jitter. Whatever jitter pattern is used by the
/// application it should be correctly programmed to the
/// <c><i>jitterOffset</i></c> field of the <c><i>dispatchDescription</i></c>
/// structure. For more guidance on camera jitter please consult the
/// documentation for <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> as well as the
/// accompanying overview documentation for FSR3.
///
/// @param [in] pContext                 A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure.
/// @param [in] pDispatchDescription     A pointer to a <c><i>FfxFsr3UpscalerDispatchDescription</i></c> structure.
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
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerContextDispatch(FfxFsr3UpscalerContext* pContext, const FfxFsr3UpscalerDispatchDescription* pDispatchDescription);

/// A helper function generate a Reactive mask from an opaque only texure and one containing translucent objects.
///
/// @param [in] pContext                 A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure.
/// @param [in] pParams                  A pointer to a <c><i>FfxFsr3UpscalerGenerateReactiveDescription</i></c> structure
///
/// @retval
/// FFX_OK                              The operation completed successfully.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerContextGenerateReactiveMask(FfxFsr3UpscalerContext* pContext, const FfxFsr3UpscalerGenerateReactiveDescription* pParams);

/// Destroy the FidelityFX Super Resolution context.
///
/// @param [out] pContext                A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure to destroy.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_CODE_NULL_POINTER         The operation failed because either <c><i>context</i></c> was <c><i>NULL</i></c>.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerContextDestroy(FfxFsr3UpscalerContext* pContext);

/// Get the upscale ratio from the quality mode.
///
/// The following table enumerates the mapping of the quality modes to
/// per-dimension scaling ratios.
///
/// Quality preset                                        | Scale factor
/// ----------------------------------------------------- | -------------
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_QUALITY</i></c>           | 1.5x
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_BALANCED</i></c>          | 1.7x
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x
///
/// Passing an invalid <c><i>qualityMode</i></c> will return 0.0f.
///
/// @param [in] qualityMode             The quality mode preset.
///
/// @returns
/// The upscaling the per-dimension upscaling ratio for
/// <c><i>qualityMode</i></c> according to the table above.
///
/// @ingroup ffxFsr3Upscaler
FFX_API float ffxFsr3UpscalerGetUpscaleRatioFromQualityMode(FfxFsr3UpscalerQualityMode qualityMode);

/// A helper function to calculate the rendering resolution from a target
/// resolution and desired quality level.
///
/// This function applies the scaling factor returned by
/// <c><i>ffxFsr3UpscalerGetUpscaleRatioFromQualityMode</i></c> to each dimension.
///
/// @param [out] pRenderWidth            A pointer to a <c>uint32_t</c> which will hold the calculated render resolution width.
/// @param [out] pRenderHeight           A pointer to a <c>uint32_t</c> which will hold the calculated render resolution height.
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
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerGetRenderResolutionFromQualityMode(
    uint32_t* pRenderWidth,
    uint32_t* pRenderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    FfxFsr3UpscalerQualityMode qualityMode);

/// A helper function to calculate the jitter phase count from display
/// resolution.
///
/// For more detailed information about the application of camera jitter to
/// your application's rendering please refer to the
/// <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> function.
///
/// The table below shows the jitter phase count which this function
/// would return for each of the quality presets.
///
/// Quality preset                                        | Scale factor  | Phase count
/// ----------------------------------------------------- | ------------- | ---------------
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_QUALITY</i></c>           | 1.5x          | 18
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_BALANCED</i></c>          | 1.7x          | 23
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_PERFORMANCE</i></c>       | 2.0x          | 32
/// <c><i>FFX_FSR3UPSCALER_QUALITY_MODE_ULTRA_PERFORMANCE</i></c> | 3.0x          | 72
/// Custom                                                | [1..n]x       | ceil(8*n^2)
///
/// @param [in] renderWidth             The render resolution width.
/// @param [in] displayWidth            The display resolution width.
///
/// @returns
/// The jitter phase count for the scaling factor between <c><i>renderWidth</i></c> and <c><i>displayWidth</i></c>.
///
/// @ingroup ffxFsr3Upscaler
FFX_API int32_t ffxFsr3UpscalerGetJitterPhaseCount(int32_t renderWidth, int32_t displayWidth);

/// A helper function to calculate the subpixel jitter offset.
///
/// FSR3 relies on the application to apply sub-pixel jittering while rendering.
/// This is typically included in the projection matrix of the camera. To make
/// the application of camera jitter simple, the FSR3 API provides a small set
/// of utility function which computes the sub-pixel jitter offset for a
/// particular frame within a sequence of separate jitter offsets. To begin, the
/// index within the jitter phase must be computed. To calculate the
/// sequence's length, you can call the <c><i>ffxFsr3UpscalerGetJitterPhaseCount</i></c>
/// function. The index should be a value which is incremented each frame modulo
/// the length of the sequence computed by <c><i>ffxFsr3UpscalerGetJitterPhaseCount</i></c>.
/// The index within the jitter phase  is passed to
/// <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> via the <c><i>index</i></c> parameter.
///
/// This function uses a Halton(2,3) sequence to compute the jitter offset.
/// The ultimate index used for the sequence is <c><i>index</i></c> %
/// <c><i>phaseCount</i></c>.
///
/// It is important to understand that the values returned from the
/// <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> function are in unit pixel space, and
/// in order to composite this correctly into a projection matrix we must
/// convert them into projection offsets. This is done as per the pseudo code
/// listing which is shown below.
///
///     const int32_t jitterPhaseCount = ffxFsr3UpscalerGetJitterPhaseCount(renderWidth, displayWidth);
///
///     float jitterX = 0;
///     float jitterY = 0;
///     ffxFsr3UpscalerGetJitterOffset(&jitterX, &jitterY, index, jitterPhaseCount);
///
///     const float jitterX = 2.0f * jitterX / (float)renderWidth;
///     const float jitterY = -2.0f * jitterY / (float)renderHeight;
///     const Matrix4 jitterTranslationMatrix = translateMatrix(Matrix3::identity, Vector3(jitterX, jitterY, 0));
///     const Matrix4 jitteredProjectionMatrix = jitterTranslationMatrix * projectionMatrix;
///
/// Jitter should be applied to all rendering. This includes opaque, alpha
/// transparent, and raytraced objects. For rasterized objects, the sub-pixel
/// jittering values calculated by the <c><i>iffxFsr3UpscalerGetJitterOffset</i></c>
/// function can be applied to the camera projection matrix which is ultimately
/// used to perform transformations during vertex shading. For raytraced
/// rendering, the sub-pixel jitter should be applied to the ray's origin,
/// often the camera's position.
///
/// Whether you elect to use the <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> function
/// or your own sequence generator, you must program the
/// <c><i>jitterOffset</i></c> field of the
/// <c><i>FfxFsr3UpscalerDispatchParameters</i></c> structure in order to inform FSR3
/// of the jitter offset that has been applied in order to render each frame.
///
/// If not using the recommended <c><i>ffxFsr3UpscalerGetJitterOffset</i></c> function,
/// care should be taken that your jitter sequence never generates a null vector;
/// that is value of 0 in both the X and Y dimensions.
///
/// @param [out] pOutX                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the x dimension.
/// @param [out] pOutY                   A pointer to a <c>float</c> which will contain the subpixel jitter offset for the y dimension.
/// @param [in] index                   The index within the jitter sequence.
/// @param [in] phaseCount              The length of jitter phase. See <c><i>ffxFsr3UpscalerGetJitterPhaseCount</i></c>.
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_INVALID_POINTER           Either <c><i>outX</i></c> or <c><i>outY</i></c> was <c>NULL</c>.
/// @retval
/// FFX_ERROR_INVALID_ARGUMENT          Argument <c><i>phaseCount</i></c> must be greater than 0.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerGetJitterOffset(float* pOutX, float* pOutY, int32_t index, int32_t phaseCount);

/// A helper function to check if a resource is
/// <c><i>FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NULL</i></c>.
///
/// @param [in] resource                A <c><i>FfxResource</i></c>.
///
/// @returns
/// true                                The <c><i>resource</i></c> was not <c><i>FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NULL</i></c>.
/// @returns
/// false                               The <c><i>resource</i></c> was <c><i>FFX_FSR3UPSCALER_RESOURCE_IDENTIFIER_NULL</i></c>.
///
/// @ingroup ffxFsr3Upscaler
FFX_API bool ffxFsr3UpscalerResourceIsNull(FfxResource resource);

/// Queries the effect version number.
///
/// @returns
/// The SDK version the effect was built with.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxVersionNumber ffxFsr3UpscalerGetEffectVersion();

/// Override upscaler constant buffer value after upscaler context creation.
///
/// @param [in] context                  A pointer to a <c><i>FfxFsr3UpscalerContext</i></c> structure.
/// @param [in] key                      A key from <c><i>FfxFsr3UpscalerConfigureKey</i></c> enum
/// @param [in] valuePtr                 A pointer to value to pass to shader in Constant Buffer. See Fsr3UpscalerConstants
///
/// @retval
/// FFX_OK                              The operation completed successfully.
/// @retval
/// FFX_ERROR_INVALID_ENUM              An invalid FfxFsr3UpscalerConfigureKey was specified.
/// @retval
/// FFX_ERROR_INVALID_POINTER           <c><i>pContext</c></i> was NULL.
///
/// @ingroup ffxFsr3Upscaler
FFX_API FfxErrorCode ffxFsr3UpscalerSetConstant(FfxFsr3UpscalerContext* context, FfxFsr3UpscalerConfigureKey key, void* valuePtr);

/// Set global debug message settings
///
/// @param [in] fpMessage                A <c><i>ffxMessageCallback</i></ci>
/// @param [in] debugLevel               An unsigned integer. Unimplemented.
/// @retval
/// FFX_OK                               The operation completed successfully.
///
/// @ingroup FRAMEINTERPOLATION
FFX_API FfxErrorCode ffxFsr3UpscalerSetGlobalDebugMessage(ffxMessageCallback fpMessage, uint32_t debugLevel);

#if defined(__cplusplus)
}
#endif // #if defined(__cplusplus)
