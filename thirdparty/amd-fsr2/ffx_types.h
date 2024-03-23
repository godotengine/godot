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

#include <stdint.h>
// -- GODOT start --
#include <stdlib.h>
// -- GODOT end --

#if defined (FFX_GCC)
/// FidelityFX exported functions
#define FFX_API
#else
/// FidelityFX exported functions
#define FFX_API __declspec(dllexport)
#endif // #if defined (FFX_GCC)

/// Maximum supported number of simultaneously bound SRVs.
#define FFX_MAX_NUM_SRVS            16

/// Maximum supported number of simultaneously bound UAVs.
#define FFX_MAX_NUM_UAVS            8

/// Maximum number of constant buffers bound.
#define FFX_MAX_NUM_CONST_BUFFERS   2

/// Maximum size of bound constant buffers.
#define FFX_MAX_CONST_SIZE          64

/// Off by default warnings
#if defined(_MSC_VER)
#pragma warning(disable : 4365 4710 4820 5039)
#elif defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#ifdef __cplusplus
extern "C" {
#endif  // #ifdef __cplusplus

/// An enumeration of surface formats.
typedef enum FfxSurfaceFormat {

    FFX_SURFACE_FORMAT_UNKNOWN,                     ///< Unknown format
    FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS,       ///< 32 bit per channel, 4 channel typeless format
    FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT,          ///< 32 bit per channel, 4 channel float format
    FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,          ///< 16 bit per channel, 4 channel float format
    FFX_SURFACE_FORMAT_R16G16B16A16_UNORM,          ///< 16 bit per channel, 4 channel unsigned normalized format
    FFX_SURFACE_FORMAT_R32G32_FLOAT,                ///< 32 bit per channel, 2 channel float format
    FFX_SURFACE_FORMAT_R32_UINT,                    ///< 32 bit per channel, 1 channel float format
    FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS,           ///<  8 bit per channel, 4 channel float format
    FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,              ///<  8 bit per channel, 4 channel unsigned normalized format
    FFX_SURFACE_FORMAT_R11G11B10_FLOAT,             ///< 32 bit 3 channel float format
    FFX_SURFACE_FORMAT_R16G16_FLOAT,                ///< 16 bit per channel, 2 channel float format
    FFX_SURFACE_FORMAT_R16G16_UINT,                 ///< 16 bit per channel, 2 channel unsigned int format
    FFX_SURFACE_FORMAT_R16_FLOAT,                   ///< 16 bit per channel, 1 channel float format
    FFX_SURFACE_FORMAT_R16_UINT,                    ///< 16 bit per channel, 1 channel unsigned int format
    FFX_SURFACE_FORMAT_R16_UNORM,                   ///< 16 bit per channel, 1 channel unsigned normalized format
    FFX_SURFACE_FORMAT_R16_SNORM,                   ///< 16 bit per channel, 1 channel signed normalized format
    FFX_SURFACE_FORMAT_R8_UNORM,                    ///<  8 bit per channel, 1 channel unsigned normalized format
    FFX_SURFACE_FORMAT_R8_UINT,                     ///<  8 bit per channel, 1 channel unsigned int format
    FFX_SURFACE_FORMAT_R8G8_UNORM,                  ///<  8 bit per channel, 2 channel unsigned normalized format
    FFX_SURFACE_FORMAT_R32_FLOAT                    ///< 32 bit per channel, 1 channel float format
} FfxSurfaceFormat;

/// An enumeration of resource usage.
typedef enum FfxResourceUsage {

    FFX_RESOURCE_USAGE_READ_ONLY = 0,               ///< No usage flags indicate a resource is read only.
    FFX_RESOURCE_USAGE_RENDERTARGET = (1<<0),       ///< Indicates a resource will be used as render target.
    FFX_RESOURCE_USAGE_UAV = (1<<1),                ///< Indicates a resource will be used as UAV.
} FfxResourceUsage;

/// An enumeration of resource states.
typedef enum FfxResourceStates {

    FFX_RESOURCE_STATE_UNORDERED_ACCESS = (1<<0),   ///< Indicates a resource is in the state to be used as UAV.
    FFX_RESOURCE_STATE_COMPUTE_READ = (1 << 1),     ///< Indicates a resource is in the state to be read by compute shaders.
    FFX_RESOURCE_STATE_COPY_SRC = (1 << 2),         ///< Indicates a resource is in the state to be used as source in a copy command.
    FFX_RESOURCE_STATE_COPY_DEST = (1 << 3),        ///< Indicates a resource is in the state to be used as destination in a copy command.
    FFX_RESOURCE_STATE_GENERIC_READ = (FFX_RESOURCE_STATE_COPY_SRC | FFX_RESOURCE_STATE_COMPUTE_READ),  ///< Indicates a resource is in generic (slow) read state.
} FfxResourceStates;

/// An enumeration of surface dimensions.
typedef enum FfxResourceDimension {

    FFX_RESOURCE_DIMENSION_TEXTURE_1D,              ///< A resource with a single dimension.
    FFX_RESOURCE_DIMENSION_TEXTURE_2D,              ///< A resource with two dimensions.
} FfxResourceDimension;

/// An enumeration of surface dimensions.
typedef enum FfxResourceFlags {

    FFX_RESOURCE_FLAGS_NONE         = 0,            ///< No flags.
    FFX_RESOURCE_FLAGS_ALIASABLE    = (1<<0),       ///< A bit indicating a resource does not need to persist across frames.
} FfxResourceFlags;

/// An enumeration of all resource view types.
typedef enum FfxResourceViewType {

    FFX_RESOURCE_VIEW_UNORDERED_ACCESS,             ///< The resource view is an unordered access view (UAV).
    FFX_RESOURCE_VIEW_SHADER_READ,                  ///< The resource view is a shader resource view (SRV).
} FfxResourceViewType;

/// The type of filtering to perform when reading a texture.
typedef enum FfxFilterType {

    FFX_FILTER_TYPE_POINT,                          ///< Point sampling.
    FFX_FILTER_TYPE_LINEAR                          ///< Sampling with interpolation.
} FfxFilterType;

/// An enumeration of all supported shader models.
typedef enum FfxShaderModel {

    FFX_SHADER_MODEL_5_1,                           ///< Shader model 5.1.
    FFX_SHADER_MODEL_6_0,                           ///< Shader model 6.0.
    FFX_SHADER_MODEL_6_1,                           ///< Shader model 6.1.
    FFX_SHADER_MODEL_6_2,                           ///< Shader model 6.2.
    FFX_SHADER_MODEL_6_3,                           ///< Shader model 6.3.
    FFX_SHADER_MODEL_6_4,                           ///< Shader model 6.4.
    FFX_SHADER_MODEL_6_5,                           ///< Shader model 6.5.
    FFX_SHADER_MODEL_6_6,                           ///< Shader model 6.6.
    FFX_SHADER_MODEL_6_7,                           ///< Shader model 6.7.
} FfxShaderModel;

// An enumeration for different resource types
typedef enum FfxResourceType {

    FFX_RESOURCE_TYPE_BUFFER,                       ///< The resource is a buffer.
    FFX_RESOURCE_TYPE_TEXTURE1D,                    ///< The resource is a 1-dimensional texture.
    FFX_RESOURCE_TYPE_TEXTURE2D,                    ///< The resource is a 2-dimensional texture.
    FFX_RESOURCE_TYPE_TEXTURE3D,                    ///< The resource is a 3-dimensional texture.
} FfxResourceType;

/// An enumeration for different heap types
typedef enum FfxHeapType {

    FFX_HEAP_TYPE_DEFAULT = 0,                      ///< Local memory.
    FFX_HEAP_TYPE_UPLOAD                            ///< Heap used for uploading resources.
} FfxHeapType;

/// An enumberation for different render job types
typedef enum FfxGpuJobType {

    FFX_GPU_JOB_CLEAR_FLOAT = 0,                 ///< The GPU job is performing a floating-point clear.
    FFX_GPU_JOB_COPY = 1,                        ///< The GPU job is performing a copy.
    FFX_GPU_JOB_COMPUTE = 2,                     ///< The GPU job is performing a compute dispatch.
} FfxGpuJobType;

/// A typedef representing the graphics device.
typedef void* FfxDevice;

/// A typedef representing a command list or command buffer.
typedef void* FfxCommandList;

/// A typedef for a root signature.
typedef void* FfxRootSignature;

/// A typedef for a pipeline state object.
typedef void* FfxPipeline;

/// A structure encapasulating a collection of device capabilities.
typedef struct FfxDeviceCapabilities {

    FfxShaderModel                  minimumSupportedShaderModel;            ///< The minimum shader model supported by the device.
    uint32_t                        waveLaneCountMin;                       ///< The minimum supported wavefront width.
    uint32_t                        waveLaneCountMax;                       ///< The maximum supported wavefront width.
    bool                            fp16Supported;                          ///< The device supports FP16 in hardware.
    bool                            raytracingSupported;                    ///< The device supports raytracing.
} FfxDeviceCapabilities;

/// A structure encapsulating a 2-dimensional point, using 32bit unsigned integers.
typedef struct FfxDimensions2D {

    uint32_t                        width;                                  ///< The width of a 2-dimensional range.
    uint32_t                        height;                                 ///< The height of a 2-dimensional range.
} FfxDimensions2D;

/// A structure encapsulating a 2-dimensional point,
typedef struct FfxIntCoords2D {

    int32_t                         x;                                      ///< The x coordinate of a 2-dimensional point.
    int32_t                         y;                                      ///< The y coordinate of a 2-dimensional point.
} FfxIntCoords2D;

/// A structure encapsulating a 2-dimensional set of floating point coordinates.
typedef struct FfxFloatCoords2D {

    float                           x;                                      ///< The x coordinate of a 2-dimensional point.
    float                           y;                                      ///< The y coordinate of a 2-dimensional point.
} FfxFloatCoords2D;

/// A structure describing a resource.
typedef struct FfxResourceDescription {

    FfxResourceType                 type;                                   ///< The type of the resource.
    FfxSurfaceFormat                format;                                 ///< The surface format.
    uint32_t                        width;                                  ///< The width of the resource.
    uint32_t                        height;                                 ///< The height of the resource.
    uint32_t                        depth;                                  ///< The depth of the resource.
    uint32_t                        mipCount;                               ///< Number of mips (or 0 for full mipchain).
    FfxResourceFlags                flags;                                  ///< A set of <c><i>FfxResourceFlags</i></c> flags.
} FfxResourceDescription;

/// An outward facing structure containing a resource
typedef struct FfxResource {
    void*                           resource;                               ///< pointer to the resource.
    wchar_t                         name[64];
    FfxResourceDescription          description;
    FfxResourceStates               state;
    bool                            isDepth;
    uint64_t                        descriptorData;
} FfxResource;

/// An internal structure containing a handle to a resource and resource views
typedef struct FfxResourceInternal {
    int32_t                         internalIndex;                          ///< The index of the resource.
} FfxResourceInternal;


/// A structure defining a resource bind point
typedef struct FfxResourceBinding
{
    uint32_t    slotIndex;
    uint32_t    resourceIdentifier;
    wchar_t     name[64];
}FfxResourceBinding;

/// A structure encapsulating a single pass of an algorithm.
typedef struct FfxPipelineState {

    FfxRootSignature                rootSignature;                                  ///< The pipelines rootSignature
    FfxPipeline                     pipeline;                                       ///< The pipeline object
    uint32_t                        uavCount;                                       ///< Count of UAVs used in this pipeline
    uint32_t                        srvCount;                                       ///< Count of SRVs used in this pipeline
    uint32_t                        constCount;                                     ///< Count of constant buffers used in this pipeline

    FfxResourceBinding              uavResourceBindings[FFX_MAX_NUM_UAVS];          ///< Array of ResourceIdentifiers bound as UAVs
    FfxResourceBinding              srvResourceBindings[FFX_MAX_NUM_SRVS];          ///< Array of ResourceIdentifiers bound as SRVs
    FfxResourceBinding              cbResourceBindings[FFX_MAX_NUM_CONST_BUFFERS];  ///< Array of ResourceIdentifiers bound as CBs
} FfxPipelineState;

/// A structure containing the data required to create a resource.
typedef struct FfxCreateResourceDescription {
    
    FfxHeapType                     heapType;                               ///< The heap type to hold the resource, typically <c><i>FFX_HEAP_TYPE_DEFAULT</i></c>.
    FfxResourceDescription          resourceDescription;                    ///< A resource description.
    FfxResourceStates               initalState;                            ///< The initial resource state.
    uint32_t                        initDataSize;                           ///< Size of initial data buffer.
    void*                           initData;                               ///< Buffer containing data to fill the resource.
    const wchar_t*                  name;                                   ///< Name of the resource.
    FfxResourceUsage                usage;                                  ///< Resource usage flags.
    uint32_t                        id;                                     ///< Internal resource ID.
} FfxCreateResourceDescription;

/// A structure containing the description used to create a
/// <c><i>FfxPipeline</i></c> structure.
///
/// A pipeline is the name given to a shader and the collection of state that
/// is required to dispatch it. In the context of FSR2 and its architecture
/// this means that a <c><i>FfxPipelineDescription</i></c> will map to either a
/// monolithic object in an explicit API (such as a
/// <c><i>PipelineStateObject</i></c> in DirectX 12). Or a shader and some
/// ancillary API objects (in something like DirectX 11).
///
/// The <c><i>contextFlags</i></c> field contains a copy of the flags passed
/// to <c><i>ffxFsr2ContextCreate</i></c> via the <c><i>flags</i></c> field of
/// the <c><i>FfxFsr2InitializationParams</i></c> structure. These flags are
/// used to determine which permutation of a pipeline for a specific
/// <c><i>FfxFsr2Pass</i></c> should be used to implement the features required
/// by each application, as well as to acheive the best performance on specific
/// target hardware configurations.
/// 
/// When using one of the provided backends for FSR2 (such as DirectX 12 or
/// Vulkan) the data required to create a pipeline is compiled offline and
/// included into the backend library that you are using. For cases where the
/// backend interface is overriden by providing custom callback function
/// implementations care should be taken to respect the contents of the
/// <c><i>contextFlags</i></c> field in order to correctly support the options
/// provided by FSR2, and acheive best performance.
///
/// @ingroup FSR2
typedef struct FfxPipelineDescription {

    uint32_t                            contextFlags;                   ///< A collection of <c><i>FfxFsr2InitializationFlagBits</i></c> which were passed to the context.
    FfxFilterType*                      samplers;                       ///< Array of static samplers.
    size_t                              samplerCount;                   ///< The number of samples contained inside <c><i>samplers</i></c>.
    const uint32_t*                     rootConstantBufferSizes;        ///< Array containing the sizes of the root constant buffers (count of 32 bit elements).
    uint32_t                            rootConstantBufferCount;        ///< The number of root constants contained within <c><i>rootConstantBufferSizes</i></c>.
} FfxPipelineDescription;

/// A structure containing a constant buffer.
typedef struct FfxConstantBuffer {

    uint32_t                        uint32Size;                             ///< Size of 32 bit chunks used in the constant buffer
    uint32_t                        data[FFX_MAX_CONST_SIZE];               ///< Constant buffer data
}FfxConstantBuffer;

/// A structure describing a clear render job.
typedef struct FfxClearFloatJobDescription {

    float                           color[4];                               ///< The clear color of the resource.
    FfxResourceInternal             target;                                 ///< The resource to be cleared.
} FfxClearFloatJobDescription;

/// A structure describing a compute render job.
typedef struct FfxComputeJobDescription {

    FfxPipelineState                pipeline;                               ///< Compute pipeline for the render job.
    uint32_t                        dimensions[3];                          ///< Dispatch dimensions.
    FfxResourceInternal             srvs[FFX_MAX_NUM_SRVS];                 ///< SRV resources to be bound in the compute job.
    wchar_t                         srvNames[FFX_MAX_NUM_SRVS][64];
    FfxResourceInternal             uavs[FFX_MAX_NUM_UAVS];                 ///< UAV resources to be bound in the compute job.
    uint32_t                        uavMip[FFX_MAX_NUM_UAVS];               ///< Mip level of UAV resources to be bound in the compute job.
    wchar_t                         uavNames[FFX_MAX_NUM_UAVS][64];
    FfxConstantBuffer               cbs[FFX_MAX_NUM_CONST_BUFFERS];         ///< Constant buffers to be bound in the compute job.
    wchar_t                         cbNames[FFX_MAX_NUM_CONST_BUFFERS][64];
    uint32_t                        cbSlotIndex[FFX_MAX_NUM_CONST_BUFFERS]; ///< Slot index in the descriptor table
} FfxComputeJobDescription;

/// A structure describing a copy render job.
typedef struct FfxCopyJobDescription
{
    FfxResourceInternal                     src;                                    ///< Source resource for the copy.
    FfxResourceInternal                     dst;                                    ///< Destination resource for the copy.
} FfxCopyJobDescription;

/// A structure describing a single render job.
typedef struct FfxGpuJobDescription{

    FfxGpuJobType                jobType;                                    ///< Type of the job.

    union {
        FfxClearFloatJobDescription clearJobDescriptor;                     ///< Clear job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_CLEAR_FLOAT</i></c>.
        FfxCopyJobDescription       copyJobDescriptor;                      ///< Copy job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_COPY</i></c>.
        FfxComputeJobDescription    computeJobDescriptor;                   ///< Compute job descriptor. Valid when <c><i>jobType</i></c> is <c><i>FFX_RENDER_JOB_COMPUTE</i></c>.
    };
} FfxGpuJobDescription;

#ifdef __cplusplus
}
#endif  // #ifdef __cplusplus
