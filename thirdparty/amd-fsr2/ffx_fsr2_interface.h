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

#include "ffx_assert.h"
#include "ffx_types.h"
#include "ffx_error.h"

// Include the FSR2 resources defined in the HLSL code. This shared here to avoid getting out of sync.
#define FFX_CPU
#include "shaders/ffx_fsr2_resources.h"
#include "shaders/ffx_fsr2_common.h"

#if defined(__cplusplus)
extern "C" {
#endif // #if defined(__cplusplus)

FFX_FORWARD_DECLARE(FfxFsr2Interface);

/// An enumeration of all the passes which constitute the FSR2 algorithm.
///
/// FSR2 is implemented as a composite of several compute passes each
/// computing a key part of the final result. Each call to the 
/// <c><i>FfxFsr2ScheduleGpuJobFunc</i></c> callback function will
/// correspond to a single pass included in <c><i>FfxFsr2Pass</i></c>. For a
/// more comprehensive description of each pass, please refer to the FSR2
/// reference documentation.
///
/// Please note in some cases e.g.: <c><i>FFX_FSR2_PASS_ACCUMULATE</i></c>
/// and <c><i>FFX_FSR2_PASS_ACCUMULATE_SHARPEN</i></c> either one pass or the
/// other will be used (they are mutually exclusive). The choice of which will
/// depend on the way the <c><i>FfxFsr2Context</i></c> is created and the
/// precise contents of <c><i>FfxFsr2DispatchParamters</i></c> each time a call
/// is made to <c><i>ffxFsr2ContextDispatch</i></c>.
/// 
/// @ingroup FSR2
typedef enum FfxFsr2Pass {

    FFX_FSR2_PASS_DEPTH_CLIP = 0,                                       ///< A pass which performs depth clipping.
    FFX_FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH = 1,                       ///< A pass which performs reconstruction of previous frame's depth.
    FFX_FSR2_PASS_LOCK = 2,                                             ///< A pass which calculates pixel locks.
    FFX_FSR2_PASS_ACCUMULATE = 3,                                       ///< A pass which performs upscaling.
    FFX_FSR2_PASS_ACCUMULATE_SHARPEN = 4,                               ///< A pass which performs upscaling when sharpening is used.
    FFX_FSR2_PASS_RCAS = 5,                                             ///< A pass which performs sharpening.
    FFX_FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID = 6,                        ///< A pass which generates the luminance mipmap chain for the current frame.
    FFX_FSR2_PASS_GENERATE_REACTIVE = 7,                                ///< An optional pass to generate a reactive mask
    FFX_FSR2_PASS_TCR_AUTOGENERATE = 8,                                 ///< An optional pass to generate a texture-and-composition and reactive masks

    FFX_FSR2_PASS_COUNT                                                 ///< The number of passes performed by FSR2.
} FfxFsr2Pass;

typedef enum FfxFsr2MsgType {
    FFX_FSR2_MESSAGE_TYPE_ERROR = 0,
    FFX_FSR2_MESSAGE_TYPE_WARNING = 1,
    FFX_FSR2_MESSAGE_TYPE_COUNT
} FfxFsr2MsgType;

/// Create and initialize the backend context.
///
/// The callback function sets up the backend context for rendering.
/// It will create or reference the device and create required internal data structures.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] device                              The FfxDevice obtained by ffxGetDevice(DX12/VK/...).
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
///
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2CreateBackendContextFunc)(
    FfxFsr2Interface* backendInterface,
    FfxDevice device);

/// Get a list of capabilities of the device.
///
/// When creating an <c><i>FfxFsr2Context</i></c> it is desirable for the FSR2
/// core implementation to be aware of certain characteristics of the platform
/// that is being targetted. This is because some optimizations which FSR2
/// attempts to perform are more effective on certain classes of hardware than
/// others, or are not supported by older hardware. In order to avoid cases
/// where optimizations actually have the effect of decreasing performance, or
/// reduce the breadth of support provided by FSR2, FSR2 queries the
/// capabilities of the device to make such decisions.
///
/// For target platforms with fixed hardware support you need not implement
/// this callback function by querying the device, but instead may hardcore
/// what features are available on the platform.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [out] outDeviceCapabilities              The device capabilities structure to fill out.
/// @param [in] device                              The device to query for capabilities.
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode(*FfxFsr2GetDeviceCapabilitiesFunc)(
    FfxFsr2Interface* backendInterface,
    FfxDeviceCapabilities* outDeviceCapabilities,
    FfxDevice device);

/// Destroy the backend context and dereference the device.
///
/// This function is called when the <c><i>FfxFsr2Context</i></c> is destroyed.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
///
/// @ingroup FSR2
typedef FfxErrorCode(*FfxFsr2DestroyBackendContextFunc)(
    FfxFsr2Interface* backendInterface);

/// Create a resource.
///
/// This callback is intended for the backend to create internal resources.
///
/// Please note: It is also possible that the creation of resources might
/// itself cause additional resources to be created by simply calling the
/// <c><i>FfxFsr2CreateResourceFunc</i></c> function pointer again. This is
/// useful when handling the initial creation of resources which must be
/// initialized. The flow in such a case would be an initial call to create the
/// CPU-side resource, another to create the GPU-side resource, and then a call
/// to schedule a copy render job to move the data between the two. Typically
/// this type of function call flow is only seen during the creation of an
/// <c><i>FfxFsr2Context</i></c>.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] createResourceDescription           A pointer to a <c><i>FfxCreateResourceDescription</i></c>.
/// @param [out] outResource                        A pointer to a <c><i>FfxResource</i></c> object.
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2CreateResourceFunc)(
    FfxFsr2Interface* backendInterface,
    const FfxCreateResourceDescription* createResourceDescription,
    FfxResourceInternal* outResource);

/// Register a resource in the backend for the current frame.
///
/// Since FSR2 and the backend are not aware how many different
/// resources will get passed to FSR2 over time, it's not safe 
/// to register all resources simultaneously in the backend.
/// Also passed resources may not be valid after the dispatch call.
/// As a result it's safest to register them as FfxResourceInternal 
/// and clear them at the end of the dispatch call.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] inResource                          A pointer to a <c><i>FfxResource</i></c>.
/// @param [out] outResource                        A pointer to a <c><i>FfxResourceInternal</i></c> object.
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode(*FfxFsr2RegisterResourceFunc)(
    FfxFsr2Interface* backendInterface,
    const FfxResource* inResource,
    FfxResourceInternal* outResource);

/// Unregister all temporary FfxResourceInternal from the backend.
///
/// Unregister FfxResourceInternal referencing resources passed to 
/// a function as a parameter.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
///
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode(*FfxFsr2UnregisterResourcesFunc)(
    FfxFsr2Interface* backendInterface);

/// Retrieve a <c><i>FfxResourceDescription</i></c> matching a
/// <c><i>FfxResource</i></c> structure. 
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] resource                            A pointer to a <c><i>FfxResource</i></c> object.
///
/// @returns
/// A description of the resource.
///
/// @ingroup FSR2
typedef FfxResourceDescription (*FfxFsr2GetResourceDescriptionFunc)(
    FfxFsr2Interface* backendInterface,
    FfxResourceInternal resource);

/// Destroy a resource
///
/// This callback is intended for the backend to release an internal resource.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] resource                            A pointer to a <c><i>FfxResource</i></c> object.
/// 
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2DestroyResourceFunc)(
    FfxFsr2Interface* backendInterface,
    FfxResourceInternal resource);

/// Create a render pipeline.
///
/// A rendering pipeline contains the shader as well as resource bindpoints
/// and samplers.
/// 
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] pass                                The identifier for the pass.
/// @param [in] pipelineDescription                 A pointer to a <c><i>FfxPipelineDescription</i></c> describing the pipeline to be created.
/// @param [out] outPipeline                        A pointer to a <c><i>FfxPipelineState</i></c> structure which should be populated.
/// 
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2CreatePipelineFunc)(
    FfxFsr2Interface* backendInterface,
    FfxFsr2Pass pass,
    const FfxPipelineDescription* pipelineDescription,
    FfxPipelineState* outPipeline);

/// Destroy a render pipeline.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [out] pipeline                           A pointer to a <c><i>FfxPipelineState</i></c> structure which should be released.
/// 
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2DestroyPipelineFunc)(
    FfxFsr2Interface* backendInterface,
    FfxPipelineState* pipeline);

/// Schedule a render job to be executed on the next call of
/// <c><i>FfxFsr2ExecuteGpuJobsFunc</i></c>.
///
/// Render jobs can perform one of three different tasks: clear, copy or
/// compute dispatches.
///
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] job                                 A pointer to a <c><i>FfxGpuJobDescription</i></c> structure.
/// 
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2ScheduleGpuJobFunc)(
    FfxFsr2Interface* backendInterface,
    const FfxGpuJobDescription* job);

/// Execute scheduled render jobs on the <c><i>comandList</i></c> provided.
/// 
/// The recording of the graphics API commands should take place in this
/// callback function, the render jobs which were previously enqueued (via
/// callbacks made to <c><i>FfxFsr2ScheduleGpuJobFunc</i></c>) should be
/// processed in the order they were received. Advanced users might choose to
/// reorder the rendering jobs, but should do so with care to respect the
/// resource dependencies.
/// 
/// Depending on the precise contents of <c><i>FfxFsr2DispatchDescription</i></c> a
/// different number of render jobs might have previously been enqueued (for
/// example if sharpening is toggled on and off).
/// 
/// @param [in] backendInterface                    A pointer to the backend interface.
/// @param [in] commandList                         A pointer to a <c><i>FfxCommandList</i></c> structure.
/// 
/// @retval
/// FFX_OK                                          The operation completed successfully.
/// @retval
/// Anything else                                   The operation failed.
/// 
/// @ingroup FSR2
typedef FfxErrorCode (*FfxFsr2ExecuteGpuJobsFunc)(
    FfxFsr2Interface* backendInterface,
    FfxCommandList commandList);

/// Pass a string message
///
/// Used for debug messages.
///
/// @param [in] type                       The type of message.
/// @param [in] message                    A string message to pass.
///
///
/// @ingroup FSR2
typedef void(*FfxFsr2Message)(
    FfxFsr2MsgType type,
    const wchar_t* message);

/// A structure encapsulating the interface between the core implentation of
/// the FSR2 algorithm and any graphics API that it should ultimately call.
/// 
/// This set of functions serves as an abstraction layer between FSR2 and the
/// API used to implement it. While FSR2 ships with backends for DirectX12 and
/// Vulkan, it is possible to implement your own backend for other platforms or
/// which sits ontop of your engine's own abstraction layer. For details on the
/// expectations of what each function should do you should refer the
/// description of the following function pointer types:
/// 
///     <c><i>FfxFsr2CreateDeviceFunc</i></c>
///     <c><i>FfxFsr2GetDeviceCapabilitiesFunc</i></c>
///     <c><i>FfxFsr2DestroyDeviceFunc</i></c>
///     <c><i>FfxFsr2CreateResourceFunc</i></c>
///     <c><i>FfxFsr2GetResourceDescriptionFunc</i></c>
///     <c><i>FfxFsr2DestroyResourceFunc</i></c>
///     <c><i>FfxFsr2CreatePipelineFunc</i></c>
///     <c><i>FfxFsr2DestroyPipelineFunc</i></c>
///     <c><i>FfxFsr2ScheduleGpuJobFunc</i></c>
///     <c><i>FfxFsr2ExecuteGpuJobsFunc</i></c>
///
/// Depending on the graphics API that is abstracted by the backend, it may be
/// required that the backend is to some extent stateful. To ensure that
/// applications retain full control to manage the memory used by FSR2, the
/// <c><i>scratchBuffer</i></c> and <c><i>scratchBufferSize</i></c> fields are
/// provided. A backend should provide a means of specifying how much scratch
/// memory is required for its internal implementation (e.g: via a function
/// or constant value). The application is that responsible for allocating that
/// memory and providing it when setting up the FSR2 backend. Backends provided
/// with FSR2 do not perform dynamic memory allocations, and instead
/// suballocate all memory from the scratch buffers provided.
///
/// The <c><i>scratchBuffer</i></c> and <c><i>scratchBufferSize</i></c> fields
/// should be populated according to the requirements of each backend. For
/// example, if using the DirectX 12 backend you should call the 
/// <c><i>ffxFsr2GetScratchMemorySizeDX12</i></c> function. It is not required
/// that custom backend implementations use a scratch buffer.
///
/// @ingroup FSR2
typedef struct FfxFsr2Interface {

    FfxFsr2CreateBackendContextFunc         fpCreateBackendContext;         ///< A callback function to create and initialize the backend context.
    FfxFsr2GetDeviceCapabilitiesFunc        fpGetDeviceCapabilities;        ///< A callback function to query device capabilites.
    FfxFsr2DestroyBackendContextFunc        fpDestroyBackendContext;        ///< A callback function to destroy the backendcontext. This also dereferences the device.
    FfxFsr2CreateResourceFunc               fpCreateResource;               ///< A callback function to create a resource.
    FfxFsr2RegisterResourceFunc             fpRegisterResource;             ///< A callback function to register an external resource.
    FfxFsr2UnregisterResourcesFunc          fpUnregisterResources;          ///< A callback function to unregister external resource.
    FfxFsr2GetResourceDescriptionFunc       fpGetResourceDescription;       ///< A callback function to retrieve a resource description.
    FfxFsr2DestroyResourceFunc              fpDestroyResource;              ///< A callback function to destroy a resource.
    FfxFsr2CreatePipelineFunc               fpCreatePipeline;               ///< A callback function to create a render or compute pipeline.
    FfxFsr2DestroyPipelineFunc              fpDestroyPipeline;              ///< A callback function to destroy a render or compute pipeline.
    FfxFsr2ScheduleGpuJobFunc               fpScheduleGpuJob;               ///< A callback function to schedule a render job.
    FfxFsr2ExecuteGpuJobsFunc               fpExecuteGpuJobs;               ///< A callback function to execute all queued render jobs.

    void*                                   scratchBuffer;                  ///< A preallocated buffer for memory utilized internally by the backend.
    size_t                                  scratchBufferSize;              ///< Size of the buffer pointed to by <c><i>scratchBuffer</i></c>.
} FfxFsr2Interface;

#if defined(__cplusplus)
}
#endif // #if defined(__cplusplus)
