#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include <IOSurface/IOSurfaceRef.h>

namespace MTL {
    class AccelerationStructure;
    class AccelerationStructureDescriptor;
    class ArgumentEncoder;
    class BinaryArchive;
    class BinaryArchiveDescriptor;
    class Buffer;
    class BufferBinding;
    class CommandQueue;
    class CommandQueueDescriptor;
    class CompileOptions;
    class ComputePipelineDescriptor;
    class ComputePipelineState;
    class CounterSampleBuffer;
    class CounterSampleBufferDescriptor;
    class DepthStencilDescriptor;
    class DepthStencilState;
    class DynamicLibrary;
    class Event;
    class Fence;
    class Function;
    class FunctionHandle;
    class Heap;
    class HeapDescriptor;
    class IOCommandQueue;
    class IOCommandQueueDescriptor;
    class IOFileHandle;
    class IndirectCommandBuffer;
    class IndirectCommandBufferDescriptor;
    class Library;
    class LogState;
    class LogStateDescriptor;
    class MeshRenderPipelineDescriptor;
    class RasterizationRateMap;
    class RasterizationRateMapDescriptor;
    class RenderPipelineDescriptor;
    class RenderPipelineState;
    class ResidencySet;
    class ResidencySetDescriptor;
    class ResourceViewPoolDescriptor;
    class SamplerDescriptor;
    class SamplerState;
    class SharedEvent;
    class SharedEventHandle;
    class SharedTextureHandle;
    class StitchedLibraryDescriptor;
    class Tensor;
    class TensorDescriptor;
    class Texture;
    class TextureDescriptor;
    class TextureViewPool;
    class TileRenderPipelineDescriptor;
    enum BindingAccess : NS::UInteger;
    enum DataType : NS::UInteger;
    enum PixelFormat : NS::UInteger;
    using ResourceOptions = NS::UInteger;
    enum SparsePageSize : NS::Integer;
    enum TextureType : NS::UInteger;
}
namespace MTL4 {
    class Archive;
    class ArgumentTable;
    class ArgumentTableDescriptor;
    class BinaryFunction;
    class CommandAllocator;
    class CommandAllocatorDescriptor;
    class CommandBuffer;
    class CommandQueue;
    class CommandQueueDescriptor;
    class Compiler;
    class CompilerDescriptor;
    class CounterHeap;
    class CounterHeapDescriptor;
    class PipelineDataSetSerializer;
    class PipelineDataSetSerializerDescriptor;
    enum CounterHeapType : NS::Integer;
}
namespace NS {
    class Array;
    class Bundle;
    class Error;
    class String;
    class URL;
}

namespace MTL
{

using DeviceNotificationName = NS::String*;
extern DeviceNotificationName const DeviceWasAddedNotification __asm__("_MTLDeviceWasAddedNotification");
extern DeviceNotificationName const DeviceRemovalRequestedNotification __asm__("_MTLDeviceRemovalRequestedNotification");
extern DeviceNotificationName const DeviceWasRemovedNotification __asm__("_MTLDeviceWasRemovedNotification");
extern NS::ErrorDomain const DeviceErrorDomain __asm__("_MTLDeviceErrorDomain");
_MTL_ENUM(NS::Integer, IOCompressionMethod) {
    IOCompressionMethodZlib = 0,
    IOCompressionMethodLZFSE = 1,
    IOCompressionMethodLZ4 = 2,
    IOCompressionMethodLZMA = 3,
    IOCompressionMethodLZBitmap = 4,
};

_MTL_ENUM(NS::UInteger, FeatureSet) {
    FeatureSet_iOS_GPUFamily1_v1 = 0,
    FeatureSet_iOS_GPUFamily2_v1 = 1,
    FeatureSet_iOS_GPUFamily1_v2 = 2,
    FeatureSet_iOS_GPUFamily2_v2 = 3,
    FeatureSet_iOS_GPUFamily3_v1 = 4,
    FeatureSet_iOS_GPUFamily1_v3 = 5,
    FeatureSet_iOS_GPUFamily2_v3 = 6,
    FeatureSet_iOS_GPUFamily3_v2 = 7,
    FeatureSet_iOS_GPUFamily1_v4 = 8,
    FeatureSet_iOS_GPUFamily2_v4 = 9,
    FeatureSet_iOS_GPUFamily3_v3 = 10,
    FeatureSet_iOS_GPUFamily4_v1 = 11,
    FeatureSet_iOS_GPUFamily1_v5 = 12,
    FeatureSet_iOS_GPUFamily2_v5 = 13,
    FeatureSet_iOS_GPUFamily3_v4 = 14,
    FeatureSet_iOS_GPUFamily4_v2 = 15,
    FeatureSet_iOS_GPUFamily5_v1 = 16,
    FeatureSet_macOS_GPUFamily1_v1 = 10000,
    FeatureSet_OSX_GPUFamily1_v1 = FeatureSet_macOS_GPUFamily1_v1,
    FeatureSet_macOS_GPUFamily1_v2 = 10001,
    FeatureSet_OSX_GPUFamily1_v2 = FeatureSet_macOS_GPUFamily1_v2,
    FeatureSet_macOS_ReadWriteTextureTier2 = 10002,
    FeatureSet_OSX_ReadWriteTextureTier2 = FeatureSet_macOS_ReadWriteTextureTier2,
    FeatureSet_macOS_GPUFamily1_v3 = 10003,
    FeatureSet_macOS_GPUFamily1_v4 = 10004,
    FeatureSet_macOS_GPUFamily2_v1 = 10005,
    FeatureSet_tvOS_GPUFamily1_v1 = 30000,
    FeatureSet_TVOS_GPUFamily1_v1 = FeatureSet_tvOS_GPUFamily1_v1,
    FeatureSet_tvOS_GPUFamily1_v2 = 30001,
    FeatureSet_tvOS_GPUFamily1_v3 = 30002,
    FeatureSet_tvOS_GPUFamily2_v1 = 30003,
    FeatureSet_tvOS_GPUFamily1_v4 = 30004,
    FeatureSet_tvOS_GPUFamily2_v2 = 30005,
};

_MTL_ENUM(NS::Integer, GPUFamily) {
    GPUFamilyApple1 = 1001,
    GPUFamilyApple2 = 1002,
    GPUFamilyApple3 = 1003,
    GPUFamilyApple4 = 1004,
    GPUFamilyApple5 = 1005,
    GPUFamilyApple6 = 1006,
    GPUFamilyApple7 = 1007,
    GPUFamilyApple8 = 1008,
    GPUFamilyApple9 = 1009,
    GPUFamilyApple10 = 1010,
    GPUFamilyMac1 = 2001,
    GPUFamilyMac2 = 2002,
    GPUFamilyCommon1 = 3001,
    GPUFamilyCommon2 = 3002,
    GPUFamilyCommon3 = 3003,
    GPUFamilyMacCatalyst1 = 4001,
    GPUFamilyMacCatalyst2 = 4002,
    GPUFamilyMetal3 = 5001,
    GPUFamilyMetal4 = 5002,
};

_MTL_ENUM(NS::UInteger, DeviceLocation) {
    DeviceLocationBuiltIn = 0,
    DeviceLocationSlot = 1,
    DeviceLocationExternal = 2,
    DeviceLocationUnspecified = static_cast<NS::UInteger>(-1),
};

_MTL_OPTIONS(NS::UInteger, PipelineOption) {
    PipelineOptionNone = 0,
    PipelineOptionArgumentInfo = 1 << 0,
    PipelineOptionBindingInfo = 1 << 0,
    PipelineOptionBufferTypeInfo = 1 << 1,
    PipelineOptionFailOnBinaryArchiveMiss = 1 << 2,
};

_MTL_ENUM(NS::UInteger, ReadWriteTextureTier) {
    ReadWriteTextureTierNone = 0,
    ReadWriteTextureTier1 = 1,
    ReadWriteTextureTier2 = 2,
};

_MTL_ENUM(NS::UInteger, ArgumentBuffersTier) {
    ArgumentBuffersTier1 = 0,
    ArgumentBuffersTier2 = 1,
};

_MTL_ENUM(NS::UInteger, SparseTextureRegionAlignmentMode) {
    SparseTextureRegionAlignmentModeOutward = 0,
    SparseTextureRegionAlignmentModeInward = 1,
};

_MTL_ENUM(NS::UInteger, CounterSamplingPoint) {
    CounterSamplingPointAtStageBoundary = 0,
    CounterSamplingPointAtDrawBoundary = 1,
    CounterSamplingPointAtDispatchBoundary = 2,
    CounterSamplingPointAtTileDispatchBoundary = 3,
    CounterSamplingPointAtBlitBoundary = 4,
};

_MTL_ENUM(NS::Integer, DeviceError) {
    DeviceErrorNone = 0,
    DeviceErrorNotSupported = 1,
};


class ArgumentDescriptor;
class Architecture;
class Device;

class ArgumentDescriptor : public NS::Copying<ArgumentDescriptor>
{
public:
    static ArgumentDescriptor* alloc();
    ArgumentDescriptor*        init() const;

    static MTL::ArgumentDescriptor* argumentDescriptor();

    MTL::BindingAccess access() const;
    NS::UInteger       arrayLength() const;
    NS::UInteger       constantBlockAlignment() const;
    MTL::DataType      dataType() const;
    NS::UInteger       index() const;
    void               setAccess(MTL::BindingAccess access);
    void               setArrayLength(NS::UInteger arrayLength);
    void               setConstantBlockAlignment(NS::UInteger constantBlockAlignment);
    void               setDataType(MTL::DataType dataType);
    void               setIndex(NS::UInteger index);
    void               setTextureType(MTL::TextureType textureType);
    MTL::TextureType   textureType() const;

};

class Architecture : public NS::Copying<Architecture>
{
public:
    static Architecture* alloc();
    Architecture*        init() const;

    NS::String* name() const;

};

class Device : public NS::Referencing<Device>
{
public:
    MTL::AccelerationStructureSizes  accelerationStructureSizes(MTL::AccelerationStructureDescriptor* descriptor);
    MTL::Architecture*               architecture() const;
    bool                             areBarycentricCoordsSupported();
    bool                             areProgrammableSamplePositionsSupported();
    bool                             areRasterOrderGroupsSupported();
    MTL::ArgumentBuffersTier         argumentBuffersSupport() const;
    void                             convertSparsePixelRegions(const MTL::Region * pixelRegions, MTL::Region* tileRegions, MTL::Size tileSize, MTL::SparseTextureRegionAlignmentMode mode, NS::UInteger numRegions);
    void                             convertSparseTileRegions(const MTL::Region * tileRegions, MTL::Region* pixelRegions, MTL::Size tileSize, NS::UInteger numRegions);
    NS::Array*                       counterSets() const;
    NS::UInteger                     currentAllocatedSize() const;
    bool                             depth24Stencil8PixelFormatSupported() const;
    MTL::FunctionHandle*             functionHandle(MTL::Function* function);
    MTL::FunctionHandle*             functionHandle(MTL4::BinaryFunction* function);
    void                             getDefaultSamplePositions(MTL::SamplePosition* positions, NS::UInteger count);
    bool                             hasUnifiedMemory() const;
    bool                             headless() const;
    MTL::SizeAndAlign                heapAccelerationStructureSizeAndAlign(NS::UInteger size);
    MTL::SizeAndAlign                heapAccelerationStructureSizeAndAlign(MTL::AccelerationStructureDescriptor* descriptor);
    MTL::SizeAndAlign                heapBufferSizeAndAlign(NS::UInteger length, MTL::ResourceOptions options);
    MTL::SizeAndAlign                heapTextureSizeAndAlign(MTL::TextureDescriptor* desc);
    bool                             isDepth24Stencil8PixelFormatSupported();
    bool                             isHeadless();
    bool                             isLowPower();
    bool                             isRemovable();
    MTL::DeviceLocation              location() const;
    NS::UInteger                     locationNumber() const;
    bool                             lowPower() const;
    NS::UInteger                     maxArgumentBufferSamplerCount() const;
    NS::UInteger                     maxBufferLength() const;
    NS::UInteger                     maxThreadgroupMemoryLength() const;
    MTL::Size                        maxThreadsPerThreadgroup() const;
    uint64_t                         maxTransferRate() const;
    NS::UInteger                     maximumConcurrentCompilationTaskCount() const;
    NS::UInteger                     minimumLinearTextureAlignmentForPixelFormat(MTL::PixelFormat format);
    NS::UInteger                     minimumTextureBufferAlignmentForPixelFormat(MTL::PixelFormat format);
    NS::String*                      name() const;
    MTL::AccelerationStructure*      newAccelerationStructure(NS::UInteger size);
    MTL::AccelerationStructure*      newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    MTL4::Archive*                   newArchive(NS::URL* url, NS::Error** error);
    MTL::ArgumentEncoder*            newArgumentEncoder(NS::Array* arguments);
    MTL::ArgumentEncoder*            newArgumentEncoder(MTL::BufferBinding* bufferBinding);
    MTL4::ArgumentTable*             newArgumentTable(MTL4::ArgumentTableDescriptor* descriptor, NS::Error** error);
    MTL::BinaryArchive*              newBinaryArchive(MTL::BinaryArchiveDescriptor* descriptor, NS::Error** error);
    MTL::Buffer*                     newBuffer(NS::UInteger length, MTL::ResourceOptions options);
    MTL::Buffer*                     newBuffer(const void * pointer, NS::UInteger length, MTL::ResourceOptions options);
    MTL::Buffer*                     newBuffer(void * pointer, NS::UInteger length, MTL::ResourceOptions options, MTL::NewBufferBlock deallocator);
    MTL::Buffer*                     newBuffer(void * pointer, NS::UInteger length, MTL::ResourceOptions options, const MTL::NewBufferFunction& deallocator);
    MTL::Buffer*                     newBuffer(NS::UInteger length, MTL::ResourceOptions options, MTL::SparsePageSize placementSparsePageSize);
    MTL4::CommandAllocator*          newCommandAllocator();
    MTL4::CommandAllocator*          newCommandAllocator(MTL4::CommandAllocatorDescriptor* descriptor, NS::Error** error);
    MTL4::CommandBuffer*             newCommandBuffer();
    MTL::CommandQueue*               newCommandQueue();
    MTL::CommandQueue*               newCommandQueue(NS::UInteger maxCommandBufferCount);
    MTL::CommandQueue*               newCommandQueue(MTL::CommandQueueDescriptor* descriptor);
    MTL4::Compiler*                  newCompiler(MTL4::CompilerDescriptor* descriptor, NS::Error** error);
    MTL::ComputePipelineState*       newComputePipelineState(MTL::Function* computeFunction, NS::Error** error);
    MTL::ComputePipelineState*       newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, MTL::AutoreleasedComputePipelineReflection* reflection, NS::Error** error);
    void                             newComputePipelineState(MTL::Function* computeFunction, MTL::NewComputePipelineStateCompletionHandler completionHandler);
    void                             newComputePipelineState(MTL::Function* computeFunction, const MTL::NewComputePipelineStateCompletionHandlerFunction& completionHandler);
    void                             newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, MTL::NewComputePipelineStateWithReflectionCompletionHandler completionHandler);
    void                             newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, const MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction& completionHandler);
    MTL::ComputePipelineState*       newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedComputePipelineReflection* reflection, NS::Error** error);
    void                             newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewComputePipelineStateWithReflectionCompletionHandler completionHandler);
    void                             newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction& completionHandler);
    MTL4::CounterHeap*               newCounterHeap(MTL4::CounterHeapDescriptor* descriptor, NS::Error** error);
    MTL::CounterSampleBuffer*        newCounterSampleBuffer(MTL::CounterSampleBufferDescriptor* descriptor, NS::Error** error);
    MTL::Library*                    newDefaultLibrary();
    MTL::Library*                    newDefaultLibrary(NS::Bundle* bundle, NS::Error** error);
    MTL::DepthStencilState*          newDepthStencilState(MTL::DepthStencilDescriptor* descriptor);
    MTL::DynamicLibrary*             newDynamicLibrary(MTL::Library* library, NS::Error** error);
    MTL::DynamicLibrary*             newDynamicLibrary(NS::URL* url, NS::Error** error);
    MTL::Event*                      newEvent();
    MTL::Fence*                      newFence();
    MTL::Heap*                       newHeap(MTL::HeapDescriptor* descriptor);
    MTL::IOCommandQueue*             newIOCommandQueue(MTL::IOCommandQueueDescriptor* descriptor, NS::Error** error);
    MTL::IOFileHandle*               newIOFileHandle(NS::URL* url, NS::Error** error);
    MTL::IOFileHandle*               newIOFileHandle(NS::URL* url, MTL::IOCompressionMethod compressionMethod, NS::Error** error);
    MTL::IOFileHandle*               newIOHandle(NS::URL* url, NS::Error** error);
    MTL::IOFileHandle*               newIOHandle(NS::URL* url, MTL::IOCompressionMethod compressionMethod, NS::Error** error);
    MTL::IndirectCommandBuffer*      newIndirectCommandBuffer(MTL::IndirectCommandBufferDescriptor* descriptor, NS::UInteger maxCount, MTL::ResourceOptions options);
    MTL::Library*                    newLibrary(NS::String* filepath, NS::Error** error);
    MTL::Library*                    newLibrary(NS::URL* url, NS::Error** error);
    MTL::Library*                    newLibrary(dispatch_data_t data, NS::Error** error);
    MTL::Library*                    newLibrary(NS::String* source, MTL::CompileOptions* options, NS::Error** error);
    void                             newLibrary(NS::String* source, MTL::CompileOptions* options, MTL::NewLibraryCompletionHandler completionHandler);
    void                             newLibrary(NS::String* source, MTL::CompileOptions* options, const MTL::NewLibraryCompletionHandlerFunction& completionHandler);
    MTL::Library*                    newLibrary(MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error);
    void                             newLibrary(MTL::StitchedLibraryDescriptor* descriptor, MTL::NewLibraryCompletionHandler completionHandler);
    void                             newLibrary(MTL::StitchedLibraryDescriptor* descriptor, const MTL::NewLibraryCompletionHandlerFunction& completionHandler);
    MTL::LogState*                   newLogState(MTL::LogStateDescriptor* descriptor, NS::Error** error);
    MTL4::CommandQueue*              newMTL4CommandQueue();
    MTL4::CommandQueue*              newMTL4CommandQueue(MTL4::CommandQueueDescriptor* descriptor, NS::Error** error);
    MTL4::PipelineDataSetSerializer* newPipelineDataSetSerializer(MTL4::PipelineDataSetSerializerDescriptor* descriptor);
    MTL::RasterizationRateMap*       newRasterizationRateMap(MTL::RasterizationRateMapDescriptor* descriptor);
    MTL::RenderPipelineState*        newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, NS::Error** error);
    MTL::RenderPipelineState*        newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error);
    void                             newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    void                             newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, const MTL::NewRenderPipelineStateCompletionHandlerFunction& completionHandler);
    void                             newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler);
    void                             newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler);
    MTL::RenderPipelineState*        newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error);
    void                             newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler);
    void                             newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler);
    MTL::RenderPipelineState*        newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error);
    void                             newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler);
    void                             newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler);
    MTL::ResidencySet*               newResidencySet(MTL::ResidencySetDescriptor* desc, NS::Error** error);
    MTL::SamplerState*               newSamplerState(MTL::SamplerDescriptor* descriptor);
    MTL::SharedEvent*                newSharedEvent();
    MTL::SharedEvent*                newSharedEvent(MTL::SharedEventHandle* sharedEventHandle);
    MTL::Texture*                    newSharedTexture(MTL::TextureDescriptor* descriptor);
    MTL::Texture*                    newSharedTexture(MTL::SharedTextureHandle* sharedHandle);
    MTL::Tensor*                     newTensor(MTL::TensorDescriptor* descriptor, NS::Error** error);
    MTL::Texture*                    newTexture(MTL::TextureDescriptor* descriptor);
    MTL::Texture*                    newTexture(MTL::TextureDescriptor* descriptor, IOSurfaceRef iosurface, NS::UInteger plane);
    MTL::TextureViewPool*            newTextureViewPool(MTL::ResourceViewPoolDescriptor* descriptor, NS::Error** error);
    uint32_t                         peerCount() const;
    uint64_t                         peerGroupID() const;
    uint32_t                         peerIndex() const;
    uint64_t                         queryTimestampFrequency();
    MTL::ReadWriteTextureTier        readWriteTextureSupport() const;
    uint64_t                         recommendedMaxWorkingSetSize() const;
    uint64_t                         registryID() const;
    bool                             removable() const;
    void                             sampleTimestamps(MTL::Timestamp* cpuTimestamp, MTL::Timestamp* gpuTimestamp);
    void                             setShouldMaximizeConcurrentCompilation(bool shouldMaximizeConcurrentCompilation);
    bool                             shouldMaximizeConcurrentCompilation() const;
    NS::UInteger                     sizeOfCounterHeapEntry(MTL4::CounterHeapType type);
    MTL::Size                        sparseTileSize(MTL::TextureType textureType, MTL::PixelFormat pixelFormat, NS::UInteger sampleCount);
    MTL::Size                        sparseTileSize(MTL::TextureType textureType, MTL::PixelFormat pixelFormat, NS::UInteger sampleCount, MTL::SparsePageSize sparsePageSize);
    NS::UInteger                     sparseTileSizeInBytes() const;
    NS::UInteger                     sparseTileSizeInBytes(MTL::SparsePageSize sparsePageSize);
    bool                             supports32BitFloatFiltering() const;
    bool                             supports32BitMSAA() const;
    bool                             supportsBCTextureCompression() const;
    bool                             supportsCounterSampling(MTL::CounterSamplingPoint samplingPoint);
    bool                             supportsDynamicLibraries() const;
    bool                             supportsFamily(MTL::GPUFamily gpuFamily);
    bool                             supportsFeatureSet(MTL::FeatureSet featureSet);
    bool                             supportsFunctionPointers() const;
    bool                             supportsFunctionPointersFromRender() const;
    bool                             supportsPrimitiveMotionBlur() const;
    bool                             supportsPullModelInterpolation() const;
    bool                             supportsQueryTextureLOD() const;
    bool                             supportsRasterizationRateMap(NS::UInteger layerCount);
    bool                             supportsRaytracing() const;
    bool                             supportsRaytracingFromRender() const;
    bool                             supportsRenderDynamicLibraries() const;
    bool                             supportsShaderBarycentricCoordinates() const;
    bool                             supportsTextureSampleCount(NS::UInteger sampleCount);
    bool                             supportsVertexAmplificationCount(NS::UInteger count);
    MTL::SizeAndAlign                tensorSizeAndAlign(MTL::TensorDescriptor* descriptor);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLArgumentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLArchitecture;
extern "C" void *OBJC_CLASS_$_MTLDevice;

_MTL_INLINE MTL::ArgumentDescriptor* MTL::ArgumentDescriptor::alloc()
{
    return _MTL_msg_MTL__ArgumentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLArgumentDescriptor, nullptr);
}

_MTL_INLINE MTL::ArgumentDescriptor* MTL::ArgumentDescriptor::init() const
{
    return _MTL_msg_MTL__ArgumentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArgumentDescriptor* MTL::ArgumentDescriptor::argumentDescriptor()
{
    return _MTL_msg_MTL__ArgumentDescriptorp_argumentDescriptor((const void*)&OBJC_CLASS_$_MTLArgumentDescriptor, nullptr);
}

_MTL_INLINE MTL::DataType MTL::ArgumentDescriptor::dataType() const
{
    return _MTL_msg_MTL__DataType_dataType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setDataType(MTL::DataType dataType)
{
    _MTL_msg_v_setDataType__MTL__DataType((const void*)this, nullptr, dataType);
}

_MTL_INLINE NS::UInteger MTL::ArgumentDescriptor::index() const
{
    return _MTL_msg_NS__UInteger_index((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setIndex(NS::UInteger index)
{
    _MTL_msg_v_setIndex__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE NS::UInteger MTL::ArgumentDescriptor::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setArrayLength(NS::UInteger arrayLength)
{
    _MTL_msg_v_setArrayLength__NS__UInteger((const void*)this, nullptr, arrayLength);
}

_MTL_INLINE MTL::BindingAccess MTL::ArgumentDescriptor::access() const
{
    return _MTL_msg_MTL__BindingAccess_access((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setAccess(MTL::BindingAccess access)
{
    _MTL_msg_v_setAccess__MTL__BindingAccess((const void*)this, nullptr, access);
}

_MTL_INLINE MTL::TextureType MTL::ArgumentDescriptor::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setTextureType(MTL::TextureType textureType)
{
    _MTL_msg_v_setTextureType__MTL__TextureType((const void*)this, nullptr, textureType);
}

_MTL_INLINE NS::UInteger MTL::ArgumentDescriptor::constantBlockAlignment() const
{
    return _MTL_msg_NS__UInteger_constantBlockAlignment((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ArgumentDescriptor::setConstantBlockAlignment(NS::UInteger constantBlockAlignment)
{
    _MTL_msg_v_setConstantBlockAlignment__NS__UInteger((const void*)this, nullptr, constantBlockAlignment);
}

_MTL_INLINE MTL::Architecture* MTL::Architecture::alloc()
{
    return _MTL_msg_MTL__Architecturep_alloc((const void*)&OBJC_CLASS_$_MTLArchitecture, nullptr);
}

_MTL_INLINE MTL::Architecture* MTL::Architecture::init() const
{
    return _MTL_msg_MTL__Architecturep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Architecture::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Device::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE uint64_t MTL::Device::registryID() const
{
    return _MTL_msg_uint64_t_registryID((const void*)this, nullptr);
}

_MTL_INLINE MTL::Architecture* MTL::Device::architecture() const
{
    return _MTL_msg_MTL__Architecturep_architecture((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::Device::maxThreadsPerThreadgroup() const
{
    return _MTL_msg_MTL__Size_maxThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::lowPower() const
{
    return _MTL_msg_bool_lowPower((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::headless() const
{
    return _MTL_msg_bool_headless((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::removable() const
{
    return _MTL_msg_bool_removable((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::hasUnifiedMemory() const
{
    return _MTL_msg_bool_hasUnifiedMemory((const void*)this, nullptr);
}

_MTL_INLINE uint64_t MTL::Device::recommendedMaxWorkingSetSize() const
{
    return _MTL_msg_uint64_t_recommendedMaxWorkingSetSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::DeviceLocation MTL::Device::location() const
{
    return _MTL_msg_MTL__DeviceLocation_location((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::locationNumber() const
{
    return _MTL_msg_NS__UInteger_locationNumber((const void*)this, nullptr);
}

_MTL_INLINE uint64_t MTL::Device::maxTransferRate() const
{
    return _MTL_msg_uint64_t_maxTransferRate((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::depth24Stencil8PixelFormatSupported() const
{
    return _MTL_msg_bool_depth24Stencil8PixelFormatSupported((const void*)this, nullptr);
}

_MTL_INLINE MTL::ReadWriteTextureTier MTL::Device::readWriteTextureSupport() const
{
    return _MTL_msg_MTL__ReadWriteTextureTier_readWriteTextureSupport((const void*)this, nullptr);
}

_MTL_INLINE MTL::ArgumentBuffersTier MTL::Device::argumentBuffersSupport() const
{
    return _MTL_msg_MTL__ArgumentBuffersTier_argumentBuffersSupport((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supports32BitFloatFiltering() const
{
    return _MTL_msg_bool_supports32BitFloatFiltering((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supports32BitMSAA() const
{
    return _MTL_msg_bool_supports32BitMSAA((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsQueryTextureLOD() const
{
    return _MTL_msg_bool_supportsQueryTextureLOD((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsBCTextureCompression() const
{
    return _MTL_msg_bool_supportsBCTextureCompression((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsPullModelInterpolation() const
{
    return _MTL_msg_bool_supportsPullModelInterpolation((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsShaderBarycentricCoordinates() const
{
    return _MTL_msg_bool_supportsShaderBarycentricCoordinates((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::currentAllocatedSize() const
{
    return _MTL_msg_NS__UInteger_currentAllocatedSize((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::maxThreadgroupMemoryLength() const
{
    return _MTL_msg_NS__UInteger_maxThreadgroupMemoryLength((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::maxArgumentBufferSamplerCount() const
{
    return _MTL_msg_NS__UInteger_maxArgumentBufferSamplerCount((const void*)this, nullptr);
}

_MTL_INLINE uint64_t MTL::Device::peerGroupID() const
{
    return _MTL_msg_uint64_t_peerGroupID((const void*)this, nullptr);
}

_MTL_INLINE uint32_t MTL::Device::peerIndex() const
{
    return _MTL_msg_uint32_t_peerIndex((const void*)this, nullptr);
}

_MTL_INLINE uint32_t MTL::Device::peerCount() const
{
    return _MTL_msg_uint32_t_peerCount((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::sparseTileSizeInBytes() const
{
    return _MTL_msg_NS__UInteger_sparseTileSizeInBytes((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Device::maxBufferLength() const
{
    return _MTL_msg_NS__UInteger_maxBufferLength((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::Device::counterSets() const
{
    return _MTL_msg_NS__Arrayp_counterSets((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsDynamicLibraries() const
{
    return _MTL_msg_bool_supportsDynamicLibraries((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsRenderDynamicLibraries() const
{
    return _MTL_msg_bool_supportsRenderDynamicLibraries((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsRaytracing() const
{
    return _MTL_msg_bool_supportsRaytracing((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsFunctionPointers() const
{
    return _MTL_msg_bool_supportsFunctionPointers((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsFunctionPointersFromRender() const
{
    return _MTL_msg_bool_supportsFunctionPointersFromRender((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsRaytracingFromRender() const
{
    return _MTL_msg_bool_supportsRaytracingFromRender((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsPrimitiveMotionBlur() const
{
    return _MTL_msg_bool_supportsPrimitiveMotionBlur((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::shouldMaximizeConcurrentCompilation() const
{
    return _MTL_msg_bool_shouldMaximizeConcurrentCompilation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Device::setShouldMaximizeConcurrentCompilation(bool shouldMaximizeConcurrentCompilation)
{
    _MTL_msg_v_setShouldMaximizeConcurrentCompilation__bool((const void*)this, nullptr, shouldMaximizeConcurrentCompilation);
}

_MTL_INLINE NS::UInteger MTL::Device::maximumConcurrentCompilationTaskCount() const
{
    return _MTL_msg_NS__UInteger_maximumConcurrentCompilationTaskCount((const void*)this, nullptr);
}

_MTL_INLINE MTL::LogState* MTL::Device::newLogState(MTL::LogStateDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__LogStatep_newLogStateWithDescriptor_error__MTL__LogStateDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::CommandQueue* MTL::Device::newCommandQueue()
{
    return _MTL_msg_MTL__CommandQueuep_newCommandQueue((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandQueue* MTL::Device::newCommandQueue(NS::UInteger maxCommandBufferCount)
{
    return _MTL_msg_MTL__CommandQueuep_newCommandQueueWithMaxCommandBufferCount__NS__UInteger((const void*)this, nullptr, maxCommandBufferCount);
}

_MTL_INLINE MTL::CommandQueue* MTL::Device::newCommandQueue(MTL::CommandQueueDescriptor* descriptor)
{
    return _MTL_msg_MTL__CommandQueuep_newCommandQueueWithDescriptor__MTL__CommandQueueDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::SizeAndAlign MTL::Device::heapTextureSizeAndAlign(MTL::TextureDescriptor* desc)
{
    return _MTL_msg_MTL__SizeAndAlign_heapTextureSizeAndAlignWithDescriptor__MTL__TextureDescriptorp((const void*)this, nullptr, desc);
}

_MTL_INLINE MTL::SizeAndAlign MTL::Device::heapBufferSizeAndAlign(NS::UInteger length, MTL::ResourceOptions options)
{
    return _MTL_msg_MTL__SizeAndAlign_heapBufferSizeAndAlignWithLength_options__NS__UInteger_MTL__ResourceOptions((const void*)this, nullptr, length, options);
}

_MTL_INLINE MTL::Heap* MTL::Device::newHeap(MTL::HeapDescriptor* descriptor)
{
    return _MTL_msg_MTL__Heapp_newHeapWithDescriptor__MTL__HeapDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Buffer* MTL::Device::newBuffer(NS::UInteger length, MTL::ResourceOptions options)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithLength_options__NS__UInteger_MTL__ResourceOptions((const void*)this, nullptr, length, options);
}

_MTL_INLINE MTL::Buffer* MTL::Device::newBuffer(const void * pointer, NS::UInteger length, MTL::ResourceOptions options)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithBytes_length_options__constvoidp_NS__UInteger_MTL__ResourceOptions((const void*)this, nullptr, pointer, length, options);
}

_MTL_INLINE MTL::Buffer* MTL::Device::newBuffer(void * pointer, NS::UInteger length, MTL::ResourceOptions options, MTL::NewBufferBlock deallocator)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithBytesNoCopy_length_options_deallocator__voidp_NS__UInteger_MTL__ResourceOptions_MTL__NewBufferBlock((const void*)this, nullptr, pointer, length, options, deallocator);
}

_MTL_INLINE MTL::Buffer* MTL::Device::newBuffer(void * pointer, NS::UInteger length, MTL::ResourceOptions options, const MTL::NewBufferFunction& deallocator)
{
    __block MTL::NewBufferFunction blockFunction = deallocator;
    return newBuffer(pointer, length, options, ^(void * x0, NS::UInteger x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::DepthStencilState* MTL::Device::newDepthStencilState(MTL::DepthStencilDescriptor* descriptor)
{
    return _MTL_msg_MTL__DepthStencilStatep_newDepthStencilStateWithDescriptor__MTL__DepthStencilDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Device::newTexture(MTL::TextureDescriptor* descriptor)
{
    return _MTL_msg_MTL__Texturep_newTextureWithDescriptor__MTL__TextureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Device::newTexture(MTL::TextureDescriptor* descriptor, IOSurfaceRef iosurface, NS::UInteger plane)
{
    return _MTL_msg_MTL__Texturep_newTextureWithDescriptor_iosurface_plane__MTL__TextureDescriptorp_IOSurfaceRef_NS__UInteger((const void*)this, nullptr, descriptor, iosurface, plane);
}

_MTL_INLINE MTL::Texture* MTL::Device::newSharedTexture(MTL::TextureDescriptor* descriptor)
{
    return _MTL_msg_MTL__Texturep_newSharedTextureWithDescriptor__MTL__TextureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Device::newSharedTexture(MTL::SharedTextureHandle* sharedHandle)
{
    return _MTL_msg_MTL__Texturep_newSharedTextureWithHandle__MTL__SharedTextureHandlep((const void*)this, nullptr, sharedHandle);
}

_MTL_INLINE MTL::SamplerState* MTL::Device::newSamplerState(MTL::SamplerDescriptor* descriptor)
{
    return _MTL_msg_MTL__SamplerStatep_newSamplerStateWithDescriptor__MTL__SamplerDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Library* MTL::Device::newDefaultLibrary()
{
    return _MTL_msg_MTL__Libraryp_newDefaultLibrary((const void*)this, nullptr);
}

_MTL_INLINE MTL::Library* MTL::Device::newDefaultLibrary(NS::Bundle* bundle, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newDefaultLibraryWithBundle_error__NS__Bundlep_NS__Errorpp((const void*)this, nullptr, bundle, error);
}

_MTL_INLINE MTL::Library* MTL::Device::newLibrary(NS::String* filepath, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newLibraryWithFile_error__NS__Stringp_NS__Errorpp((const void*)this, nullptr, filepath, error);
}

_MTL_INLINE MTL::Library* MTL::Device::newLibrary(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newLibraryWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE MTL::Library* MTL::Device::newLibrary(dispatch_data_t data, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newLibraryWithData_error__dispatch_data_t_NS__Errorpp((const void*)this, nullptr, data, error);
}

_MTL_INLINE MTL::Library* MTL::Device::newLibrary(NS::String* source, MTL::CompileOptions* options, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newLibraryWithSource_options_error__NS__Stringp_MTL__CompileOptionsp_NS__Errorpp((const void*)this, nullptr, source, options, error);
}

_MTL_INLINE void MTL::Device::newLibrary(NS::String* source, MTL::CompileOptions* options, MTL::NewLibraryCompletionHandler completionHandler)
{
    _MTL_msg_v_newLibraryWithSource_options_completionHandler__NS__Stringp_MTL__CompileOptionsp_MTL__NewLibraryCompletionHandler((const void*)this, nullptr, source, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newLibrary(NS::String* source, MTL::CompileOptions* options, const MTL::NewLibraryCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewLibraryCompletionHandlerFunction blockFunction = completionHandler;
    newLibrary(source, options, ^(MTL::Library* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::Library* MTL::Device::newLibrary(MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__Libraryp_newLibraryWithStitchedDescriptor_error__MTL__StitchedLibraryDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE void MTL::Device::newLibrary(MTL::StitchedLibraryDescriptor* descriptor, MTL::NewLibraryCompletionHandler completionHandler)
{
    _MTL_msg_v_newLibraryWithStitchedDescriptor_completionHandler__MTL__StitchedLibraryDescriptorp_MTL__NewLibraryCompletionHandler((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL_INLINE void MTL::Device::newLibrary(MTL::StitchedLibraryDescriptor* descriptor, const MTL::NewLibraryCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewLibraryCompletionHandlerFunction blockFunction = completionHandler;
    newLibrary(descriptor, ^(MTL::Library* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::RenderPipelineState* MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_error__MTL__RenderPipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::RenderPipelineState* MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_options_reflection_error__MTL__RenderPipelineDescriptorp_MTL__PipelineOption_MTL__RenderPipelineReflectionpp_NS__Errorpp((const void*)this, nullptr, descriptor, options, reflection, error);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    _MTL_msg_v_newRenderPipelineStateWithDescriptor_completionHandler__MTL__RenderPipelineDescriptorp_MTL__NewRenderPipelineStateCompletionHandler((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, const MTL::NewRenderPipelineStateCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewRenderPipelineStateCompletionHandlerFunction blockFunction = completionHandler;
    newRenderPipelineState(descriptor, ^(MTL::RenderPipelineState* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler)
{
    _MTL_msg_v_newRenderPipelineStateWithDescriptor_options_completionHandler__MTL__RenderPipelineDescriptorp_MTL__PipelineOption_MTL__NewRenderPipelineStateWithReflectionCompletionHandler((const void*)this, nullptr, descriptor, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::RenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction blockFunction = completionHandler;
    newRenderPipelineState(descriptor, options, ^(MTL::RenderPipelineState* x0, void* x1, NS::Error* x2) { blockFunction(x0, x1, x2); });
}

_MTL_INLINE MTL::ComputePipelineState* MTL::Device::newComputePipelineState(MTL::Function* computeFunction, NS::Error** error)
{
    return _MTL_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithFunction_error__MTL__Functionp_NS__Errorpp((const void*)this, nullptr, computeFunction, error);
}

_MTL_INLINE MTL::ComputePipelineState* MTL::Device::newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, MTL::AutoreleasedComputePipelineReflection* reflection, NS::Error** error)
{
    return _MTL_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithFunction_options_reflection_error__MTL__Functionp_MTL__PipelineOption_MTL__ComputePipelineReflectionpp_NS__Errorpp((const void*)this, nullptr, computeFunction, options, reflection, error);
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::Function* computeFunction, MTL::NewComputePipelineStateCompletionHandler completionHandler)
{
    _MTL_msg_v_newComputePipelineStateWithFunction_completionHandler__MTL__Functionp_MTL__NewComputePipelineStateCompletionHandler((const void*)this, nullptr, computeFunction, completionHandler);
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::Function* computeFunction, const MTL::NewComputePipelineStateCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewComputePipelineStateCompletionHandlerFunction blockFunction = completionHandler;
    newComputePipelineState(computeFunction, ^(MTL::ComputePipelineState* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, MTL::NewComputePipelineStateWithReflectionCompletionHandler completionHandler)
{
    _MTL_msg_v_newComputePipelineStateWithFunction_options_completionHandler__MTL__Functionp_MTL__PipelineOption_MTL__NewComputePipelineStateWithReflectionCompletionHandler((const void*)this, nullptr, computeFunction, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::Function* computeFunction, MTL::PipelineOption options, const MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction blockFunction = completionHandler;
    newComputePipelineState(computeFunction, options, ^(MTL::ComputePipelineState* x0, void* x1, NS::Error* x2) { blockFunction(x0, x1, x2); });
}

_MTL_INLINE MTL::ComputePipelineState* MTL::Device::newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedComputePipelineReflection* reflection, NS::Error** error)
{
    return _MTL_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithDescriptor_options_reflection_error__MTL__ComputePipelineDescriptorp_MTL__PipelineOption_MTL__ComputePipelineReflectionpp_NS__Errorpp((const void*)this, nullptr, descriptor, options, reflection, error);
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewComputePipelineStateWithReflectionCompletionHandler completionHandler)
{
    _MTL_msg_v_newComputePipelineStateWithDescriptor_options_completionHandler__MTL__ComputePipelineDescriptorp_MTL__PipelineOption_MTL__NewComputePipelineStateWithReflectionCompletionHandler((const void*)this, nullptr, descriptor, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newComputePipelineState(MTL::ComputePipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewComputePipelineStateWithReflectionCompletionHandlerFunction blockFunction = completionHandler;
    newComputePipelineState(descriptor, options, ^(MTL::ComputePipelineState* x0, void* x1, NS::Error* x2) { blockFunction(x0, x1, x2); });
}

_MTL_INLINE MTL::Fence* MTL::Device::newFence()
{
    return _MTL_msg_MTL__Fencep_newFence((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::supportsFeatureSet(MTL::FeatureSet featureSet)
{
    return _MTL_msg_bool_supportsFeatureSet__MTL__FeatureSet((const void*)this, nullptr, featureSet);
}

_MTL_INLINE bool MTL::Device::supportsFamily(MTL::GPUFamily gpuFamily)
{
    return _MTL_msg_bool_supportsFamily__MTL__GPUFamily((const void*)this, nullptr, gpuFamily);
}

_MTL_INLINE bool MTL::Device::supportsTextureSampleCount(NS::UInteger sampleCount)
{
    return _MTL_msg_bool_supportsTextureSampleCount__NS__UInteger((const void*)this, nullptr, sampleCount);
}

_MTL_INLINE NS::UInteger MTL::Device::minimumLinearTextureAlignmentForPixelFormat(MTL::PixelFormat format)
{
    return _MTL_msg_NS__UInteger_minimumLinearTextureAlignmentForPixelFormat__MTL__PixelFormat((const void*)this, nullptr, format);
}

_MTL_INLINE NS::UInteger MTL::Device::minimumTextureBufferAlignmentForPixelFormat(MTL::PixelFormat format)
{
    return _MTL_msg_NS__UInteger_minimumTextureBufferAlignmentForPixelFormat__MTL__PixelFormat((const void*)this, nullptr, format);
}

_MTL_INLINE MTL::RenderPipelineState* MTL::Device::newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithTileDescriptor_options_reflection_error__MTL__TileRenderPipelineDescriptorp_MTL__PipelineOption_MTL__RenderPipelineReflectionpp_NS__Errorpp((const void*)this, nullptr, descriptor, options, reflection, error);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler)
{
    _MTL_msg_v_newRenderPipelineStateWithTileDescriptor_options_completionHandler__MTL__TileRenderPipelineDescriptorp_MTL__PipelineOption_MTL__NewRenderPipelineStateWithReflectionCompletionHandler((const void*)this, nullptr, descriptor, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::TileRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction blockFunction = completionHandler;
    newRenderPipelineState(descriptor, options, ^(MTL::RenderPipelineState* x0, void* x1, NS::Error* x2) { blockFunction(x0, x1, x2); });
}

_MTL_INLINE MTL::RenderPipelineState* MTL::Device::newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::AutoreleasedRenderPipelineReflection* reflection, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithMeshDescriptor_options_reflection_error__MTL__MeshRenderPipelineDescriptorp_MTL__PipelineOption_MTL__RenderPipelineReflectionpp_NS__Errorpp((const void*)this, nullptr, descriptor, options, reflection, error);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, MTL::NewRenderPipelineStateWithReflectionCompletionHandler completionHandler)
{
    _MTL_msg_v_newRenderPipelineStateWithMeshDescriptor_options_completionHandler__MTL__MeshRenderPipelineDescriptorp_MTL__PipelineOption_MTL__NewRenderPipelineStateWithReflectionCompletionHandler((const void*)this, nullptr, descriptor, options, completionHandler);
}

_MTL_INLINE void MTL::Device::newRenderPipelineState(MTL::MeshRenderPipelineDescriptor* descriptor, MTL::PipelineOption options, const MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction& completionHandler)
{
    __block MTL::NewRenderPipelineStateWithReflectionCompletionHandlerFunction blockFunction = completionHandler;
    newRenderPipelineState(descriptor, options, ^(MTL::RenderPipelineState* x0, void* x1, NS::Error* x2) { blockFunction(x0, x1, x2); });
}

_MTL_INLINE void MTL::Device::getDefaultSamplePositions(MTL::SamplePosition* positions, NS::UInteger count)
{
    _MTL_msg_v_getDefaultSamplePositions_count__MTL__SamplePositionp_NS__UInteger((const void*)this, nullptr, positions, count);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Device::newArgumentEncoder(NS::Array* arguments)
{
    return _MTL_msg_MTL__ArgumentEncoderp_newArgumentEncoderWithArguments__NS__Arrayp((const void*)this, nullptr, arguments);
}

_MTL_INLINE bool MTL::Device::supportsRasterizationRateMap(NS::UInteger layerCount)
{
    return _MTL_msg_bool_supportsRasterizationRateMapWithLayerCount__NS__UInteger((const void*)this, nullptr, layerCount);
}

_MTL_INLINE MTL::RasterizationRateMap* MTL::Device::newRasterizationRateMap(MTL::RasterizationRateMapDescriptor* descriptor)
{
    return _MTL_msg_MTL__RasterizationRateMapp_newRasterizationRateMapWithDescriptor__MTL__RasterizationRateMapDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::IndirectCommandBuffer* MTL::Device::newIndirectCommandBuffer(MTL::IndirectCommandBufferDescriptor* descriptor, NS::UInteger maxCount, MTL::ResourceOptions options)
{
    return _MTL_msg_MTL__IndirectCommandBufferp_newIndirectCommandBufferWithDescriptor_maxCommandCount_options__MTL__IndirectCommandBufferDescriptorp_NS__UInteger_MTL__ResourceOptions((const void*)this, nullptr, descriptor, maxCount, options);
}

_MTL_INLINE MTL::Event* MTL::Device::newEvent()
{
    return _MTL_msg_MTL__Eventp_newEvent((const void*)this, nullptr);
}

_MTL_INLINE MTL::SharedEvent* MTL::Device::newSharedEvent()
{
    return _MTL_msg_MTL__SharedEventp_newSharedEvent((const void*)this, nullptr);
}

_MTL_INLINE MTL::SharedEvent* MTL::Device::newSharedEvent(MTL::SharedEventHandle* sharedEventHandle)
{
    return _MTL_msg_MTL__SharedEventp_newSharedEventWithHandle__MTL__SharedEventHandlep((const void*)this, nullptr, sharedEventHandle);
}

_MTL_INLINE MTL::IOFileHandle* MTL::Device::newIOHandle(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_MTL__IOFileHandlep_newIOHandleWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE MTL::IOCommandQueue* MTL::Device::newIOCommandQueue(MTL::IOCommandQueueDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__IOCommandQueuep_newIOCommandQueueWithDescriptor_error__MTL__IOCommandQueueDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::IOFileHandle* MTL::Device::newIOHandle(NS::URL* url, MTL::IOCompressionMethod compressionMethod, NS::Error** error)
{
    return _MTL_msg_MTL__IOFileHandlep_newIOHandleWithURL_compressionMethod_error__NS__URLp_MTL__IOCompressionMethod_NS__Errorpp((const void*)this, nullptr, url, compressionMethod, error);
}

_MTL_INLINE MTL::IOFileHandle* MTL::Device::newIOFileHandle(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_MTL__IOFileHandlep_newIOFileHandleWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE MTL::IOFileHandle* MTL::Device::newIOFileHandle(NS::URL* url, MTL::IOCompressionMethod compressionMethod, NS::Error** error)
{
    return _MTL_msg_MTL__IOFileHandlep_newIOFileHandleWithURL_compressionMethod_error__NS__URLp_MTL__IOCompressionMethod_NS__Errorpp((const void*)this, nullptr, url, compressionMethod, error);
}

_MTL_INLINE MTL::Size MTL::Device::sparseTileSize(MTL::TextureType textureType, MTL::PixelFormat pixelFormat, NS::UInteger sampleCount)
{
    return _MTL_msg_MTL__Size_sparseTileSizeWithTextureType_pixelFormat_sampleCount__MTL__TextureType_MTL__PixelFormat_NS__UInteger((const void*)this, nullptr, textureType, pixelFormat, sampleCount);
}

_MTL_INLINE void MTL::Device::convertSparsePixelRegions(const MTL::Region * pixelRegions, MTL::Region* tileRegions, MTL::Size tileSize, MTL::SparseTextureRegionAlignmentMode mode, NS::UInteger numRegions)
{
    _MTL_msg_v_convertSparsePixelRegions_toTileRegions_withTileSize_alignmentMode_numRegions__constMTL__Regionp_MTL__Regionp_MTL__Size_MTL__SparseTextureRegionAlignmentMode_NS__UInteger((const void*)this, nullptr, pixelRegions, tileRegions, tileSize, mode, numRegions);
}

_MTL_INLINE void MTL::Device::convertSparseTileRegions(const MTL::Region * tileRegions, MTL::Region* pixelRegions, MTL::Size tileSize, NS::UInteger numRegions)
{
    _MTL_msg_v_convertSparseTileRegions_toPixelRegions_withTileSize_numRegions__constMTL__Regionp_MTL__Regionp_MTL__Size_NS__UInteger((const void*)this, nullptr, tileRegions, pixelRegions, tileSize, numRegions);
}

_MTL_INLINE NS::UInteger MTL::Device::sparseTileSizeInBytes(MTL::SparsePageSize sparsePageSize)
{
    return _MTL_msg_NS__UInteger_sparseTileSizeInBytesForSparsePageSize__MTL__SparsePageSize((const void*)this, nullptr, sparsePageSize);
}

_MTL_INLINE MTL::Size MTL::Device::sparseTileSize(MTL::TextureType textureType, MTL::PixelFormat pixelFormat, NS::UInteger sampleCount, MTL::SparsePageSize sparsePageSize)
{
    return _MTL_msg_MTL__Size_sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize__MTL__TextureType_MTL__PixelFormat_NS__UInteger_MTL__SparsePageSize((const void*)this, nullptr, textureType, pixelFormat, sampleCount, sparsePageSize);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::Device::newCounterSampleBuffer(MTL::CounterSampleBufferDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__CounterSampleBufferp_newCounterSampleBufferWithDescriptor_error__MTL__CounterSampleBufferDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE void MTL::Device::sampleTimestamps(MTL::Timestamp* cpuTimestamp, MTL::Timestamp* gpuTimestamp)
{
    _MTL_msg_v_sampleTimestamps_gpuTimestamp__uint64_tp_uint64_tp((const void*)this, nullptr, cpuTimestamp, gpuTimestamp);
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::Device::newArgumentEncoder(MTL::BufferBinding* bufferBinding)
{
    return _MTL_msg_MTL__ArgumentEncoderp_newArgumentEncoderWithBufferBinding__MTL__BufferBindingp((const void*)this, nullptr, bufferBinding);
}

_MTL_INLINE bool MTL::Device::supportsCounterSampling(MTL::CounterSamplingPoint samplingPoint)
{
    return _MTL_msg_bool_supportsCounterSampling__MTL__CounterSamplingPoint((const void*)this, nullptr, samplingPoint);
}

_MTL_INLINE bool MTL::Device::supportsVertexAmplificationCount(NS::UInteger count)
{
    return _MTL_msg_bool_supportsVertexAmplificationCount__NS__UInteger((const void*)this, nullptr, count);
}

_MTL_INLINE MTL::DynamicLibrary* MTL::Device::newDynamicLibrary(MTL::Library* library, NS::Error** error)
{
    return _MTL_msg_MTL__DynamicLibraryp_newDynamicLibrary_error__MTL__Libraryp_NS__Errorpp((const void*)this, nullptr, library, error);
}

_MTL_INLINE MTL::DynamicLibrary* MTL::Device::newDynamicLibrary(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_MTL__DynamicLibraryp_newDynamicLibraryWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE MTL::BinaryArchive* MTL::Device::newBinaryArchive(MTL::BinaryArchiveDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__BinaryArchivep_newBinaryArchiveWithDescriptor_error__MTL__BinaryArchiveDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::AccelerationStructureSizes MTL::Device::accelerationStructureSizes(MTL::AccelerationStructureDescriptor* descriptor)
{
    return _MTL_msg_MTL__AccelerationStructureSizes_accelerationStructureSizesWithDescriptor__MTL__AccelerationStructureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Device::newAccelerationStructure(NS::UInteger size)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithSize__NS__UInteger((const void*)this, nullptr, size);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Device::newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithDescriptor__MTL__AccelerationStructureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::SizeAndAlign MTL::Device::heapAccelerationStructureSizeAndAlign(NS::UInteger size)
{
    return _MTL_msg_MTL__SizeAndAlign_heapAccelerationStructureSizeAndAlignWithSize__NS__UInteger((const void*)this, nullptr, size);
}

_MTL_INLINE MTL::SizeAndAlign MTL::Device::heapAccelerationStructureSizeAndAlign(MTL::AccelerationStructureDescriptor* descriptor)
{
    return _MTL_msg_MTL__SizeAndAlign_heapAccelerationStructureSizeAndAlignWithDescriptor__MTL__AccelerationStructureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::ResidencySet* MTL::Device::newResidencySet(MTL::ResidencySetDescriptor* desc, NS::Error** error)
{
    return _MTL_msg_MTL__ResidencySetp_newResidencySetWithDescriptor_error__MTL__ResidencySetDescriptorp_NS__Errorpp((const void*)this, nullptr, desc, error);
}

_MTL_INLINE MTL::SizeAndAlign MTL::Device::tensorSizeAndAlign(MTL::TensorDescriptor* descriptor)
{
    return _MTL_msg_MTL__SizeAndAlign_tensorSizeAndAlignWithDescriptor__MTL__TensorDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Tensor* MTL::Device::newTensor(MTL::TensorDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__Tensorp_newTensorWithDescriptor_error__MTL__TensorDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::FunctionHandle* MTL::Device::functionHandle(MTL::Function* function)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithFunction__MTL__Functionp((const void*)this, nullptr, function);
}

_MTL_INLINE MTL4::CommandAllocator* MTL::Device::newCommandAllocator()
{
    return _MTL_msg_MTL4__CommandAllocatorp_newCommandAllocator((const void*)this, nullptr);
}

_MTL_INLINE MTL4::CommandAllocator* MTL::Device::newCommandAllocator(MTL4::CommandAllocatorDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL4__CommandAllocatorp_newCommandAllocatorWithDescriptor_error__MTL4__CommandAllocatorDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL4::CommandQueue* MTL::Device::newMTL4CommandQueue()
{
    return _MTL_msg_MTL4__CommandQueuep_newMTL4CommandQueue((const void*)this, nullptr);
}

_MTL_INLINE MTL4::CommandQueue* MTL::Device::newMTL4CommandQueue(MTL4::CommandQueueDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL4__CommandQueuep_newMTL4CommandQueueWithDescriptor_error__MTL4__CommandQueueDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL4::CommandBuffer* MTL::Device::newCommandBuffer()
{
    return _MTL_msg_MTL4__CommandBufferp_newCommandBuffer((const void*)this, nullptr);
}

_MTL_INLINE MTL4::ArgumentTable* MTL::Device::newArgumentTable(MTL4::ArgumentTableDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL4__ArgumentTablep_newArgumentTableWithDescriptor_error__MTL4__ArgumentTableDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL::TextureViewPool* MTL::Device::newTextureViewPool(MTL::ResourceViewPoolDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL__TextureViewPoolp_newTextureViewPoolWithDescriptor_error__MTL__ResourceViewPoolDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL4::Compiler* MTL::Device::newCompiler(MTL4::CompilerDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL4__Compilerp_newCompilerWithDescriptor_error__MTL4__CompilerDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE MTL4::Archive* MTL::Device::newArchive(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_MTL4__Archivep_newArchiveWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE MTL4::PipelineDataSetSerializer* MTL::Device::newPipelineDataSetSerializer(MTL4::PipelineDataSetSerializerDescriptor* descriptor)
{
    return _MTL_msg_MTL4__PipelineDataSetSerializerp_newPipelineDataSetSerializerWithDescriptor__MTL4__PipelineDataSetSerializerDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Buffer* MTL::Device::newBuffer(NS::UInteger length, MTL::ResourceOptions options, MTL::SparsePageSize placementSparsePageSize)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithLength_options_placementSparsePageSize__NS__UInteger_MTL__ResourceOptions_MTL__SparsePageSize((const void*)this, nullptr, length, options, placementSparsePageSize);
}

_MTL_INLINE MTL4::CounterHeap* MTL::Device::newCounterHeap(MTL4::CounterHeapDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_MTL4__CounterHeapp_newCounterHeapWithDescriptor_error__MTL4__CounterHeapDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE NS::UInteger MTL::Device::sizeOfCounterHeapEntry(MTL4::CounterHeapType type)
{
    return _MTL_msg_NS__UInteger_sizeOfCounterHeapEntry__MTL4__CounterHeapType((const void*)this, nullptr, type);
}

_MTL_INLINE uint64_t MTL::Device::queryTimestampFrequency()
{
    return _MTL_msg_uint64_t_queryTimestampFrequency((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionHandle* MTL::Device::functionHandle(MTL4::BinaryFunction* function)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithBinaryFunction__MTL4__BinaryFunctionp((const void*)this, nullptr, function);
}

_MTL_INLINE bool MTL::Device::isLowPower()
{
    return _MTL_msg_bool_isLowPower((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::isHeadless()
{
    return _MTL_msg_bool_isHeadless((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::isRemovable()
{
    return _MTL_msg_bool_isRemovable((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::isDepth24Stencil8PixelFormatSupported()
{
    return _MTL_msg_bool_isDepth24Stencil8PixelFormatSupported((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::areRasterOrderGroupsSupported()
{
    return _MTL_msg_bool_areRasterOrderGroupsSupported((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::areBarycentricCoordsSupported()
{
    return _MTL_msg_bool_areBarycentricCoordsSupported((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Device::areProgrammableSamplePositionsSupported()
{
    return _MTL_msg_bool_areProgrammableSamplePositionsSupported((const void*)this, nullptr);
}
