#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLAllocation.hpp"

namespace MTL {
    class AccelerationStructure;
    class AccelerationStructureDescriptor;
    class Buffer;
    class Device;
    class Texture;
    class TextureDescriptor;
    enum CPUCacheMode : NS::UInteger;
    enum HazardTrackingMode : NS::UInteger;
    enum PurgeableState : NS::UInteger;
    using ResourceOptions = NS::UInteger;
    enum SparsePageSize : NS::Integer;
    enum StorageMode : NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL
{

_MTL_ENUM(NS::Integer, HeapType) {
    HeapTypeAutomatic = 0,
    HeapTypePlacement = 1,
    HeapTypeSparse = 2,
};


class HeapDescriptor;
class Heap;

class HeapDescriptor : public NS::Copying<HeapDescriptor>
{
public:
    static HeapDescriptor* alloc();
    HeapDescriptor*        init() const;

    MTL::CPUCacheMode       cpuCacheMode() const;
    MTL::HazardTrackingMode hazardTrackingMode() const;
    MTL::SparsePageSize     maxCompatiblePlacementSparsePageSize() const;
    MTL::ResourceOptions    resourceOptions() const;
    void                    setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);
    void                    setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);
    void                    setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize maxCompatiblePlacementSparsePageSize);
    void                    setResourceOptions(MTL::ResourceOptions resourceOptions);
    void                    setSize(NS::UInteger size);
    void                    setSparsePageSize(MTL::SparsePageSize sparsePageSize);
    void                    setStorageMode(MTL::StorageMode storageMode);
    void                    setType(MTL::HeapType type);
    NS::UInteger            size() const;
    MTL::SparsePageSize     sparsePageSize() const;
    MTL::StorageMode        storageMode() const;
    MTL::HeapType           type() const;

};

class Heap : public NS::Referencing<Heap, MTL::Allocation>
{
public:
    MTL::CPUCacheMode           cpuCacheMode() const;
    NS::UInteger                currentAllocatedSize() const;
    MTL::Device*                device() const;
    MTL::HazardTrackingMode     hazardTrackingMode() const;
    NS::String*                 label() const;
    NS::UInteger                maxAvailableSize(NS::UInteger alignment);
    MTL::AccelerationStructure* newAccelerationStructure(NS::UInteger size);
    MTL::AccelerationStructure* newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor);
    MTL::AccelerationStructure* newAccelerationStructure(NS::UInteger size, NS::UInteger offset);
    MTL::AccelerationStructure* newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor, NS::UInteger offset);
    MTL::Buffer*                newBuffer(NS::UInteger length, MTL::ResourceOptions options);
    MTL::Buffer*                newBuffer(NS::UInteger length, MTL::ResourceOptions options, NS::UInteger offset);
    MTL::Texture*               newTexture(MTL::TextureDescriptor* descriptor);
    MTL::Texture*               newTexture(MTL::TextureDescriptor* descriptor, NS::UInteger offset);
    MTL::ResourceOptions        resourceOptions() const;
    void                        setLabel(NS::String* label);
    MTL::PurgeableState         setPurgeableState(MTL::PurgeableState state);
    NS::UInteger                size() const;
    MTL::StorageMode            storageMode() const;
    MTL::HeapType               type() const;
    NS::UInteger                usedSize() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLHeapDescriptor;
extern "C" void *OBJC_CLASS_$_MTLHeap;

_MTL_INLINE MTL::HeapDescriptor* MTL::HeapDescriptor::alloc()
{
    return _MTL_msg_MTL__HeapDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLHeapDescriptor, nullptr);
}

_MTL_INLINE MTL::HeapDescriptor* MTL::HeapDescriptor::init() const
{
    return _MTL_msg_MTL__HeapDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::HeapDescriptor::size() const
{
    return _MTL_msg_NS__UInteger_size((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setSize(NS::UInteger size)
{
    _MTL_msg_v_setSize__NS__UInteger((const void*)this, nullptr, size);
}

_MTL_INLINE MTL::StorageMode MTL::HeapDescriptor::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    _MTL_msg_v_setStorageMode__MTL__StorageMode((const void*)this, nullptr, storageMode);
}

_MTL_INLINE MTL::CPUCacheMode MTL::HeapDescriptor::cpuCacheMode() const
{
    return _MTL_msg_MTL__CPUCacheMode_cpuCacheMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    _MTL_msg_v_setCpuCacheMode__MTL__CPUCacheMode((const void*)this, nullptr, cpuCacheMode);
}

_MTL_INLINE MTL::SparsePageSize MTL::HeapDescriptor::sparsePageSize() const
{
    return _MTL_msg_MTL__SparsePageSize_sparsePageSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setSparsePageSize(MTL::SparsePageSize sparsePageSize)
{
    _MTL_msg_v_setSparsePageSize__MTL__SparsePageSize((const void*)this, nullptr, sparsePageSize);
}

_MTL_INLINE MTL::HazardTrackingMode MTL::HeapDescriptor::hazardTrackingMode() const
{
    return _MTL_msg_MTL__HazardTrackingMode_hazardTrackingMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    _MTL_msg_v_setHazardTrackingMode__MTL__HazardTrackingMode((const void*)this, nullptr, hazardTrackingMode);
}

_MTL_INLINE MTL::ResourceOptions MTL::HeapDescriptor::resourceOptions() const
{
    return _MTL_msg_MTL__ResourceOptions_resourceOptions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    _MTL_msg_v_setResourceOptions__MTL__ResourceOptions((const void*)this, nullptr, resourceOptions);
}

_MTL_INLINE MTL::HeapType MTL::HeapDescriptor::type() const
{
    return _MTL_msg_MTL__HeapType_type((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setType(MTL::HeapType type)
{
    _MTL_msg_v_setType__MTL__HeapType((const void*)this, nullptr, type);
}

_MTL_INLINE MTL::SparsePageSize MTL::HeapDescriptor::maxCompatiblePlacementSparsePageSize() const
{
    return _MTL_msg_MTL__SparsePageSize_maxCompatiblePlacementSparsePageSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::HeapDescriptor::setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize maxCompatiblePlacementSparsePageSize)
{
    _MTL_msg_v_setMaxCompatiblePlacementSparsePageSize__MTL__SparsePageSize((const void*)this, nullptr, maxCompatiblePlacementSparsePageSize);
}

_MTL_INLINE NS::String* MTL::Heap::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Heap::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::Heap::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::StorageMode MTL::Heap::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::CPUCacheMode MTL::Heap::cpuCacheMode() const
{
    return _MTL_msg_MTL__CPUCacheMode_cpuCacheMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::HazardTrackingMode MTL::Heap::hazardTrackingMode() const
{
    return _MTL_msg_MTL__HazardTrackingMode_hazardTrackingMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceOptions MTL::Heap::resourceOptions() const
{
    return _MTL_msg_MTL__ResourceOptions_resourceOptions((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Heap::size() const
{
    return _MTL_msg_NS__UInteger_size((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Heap::usedSize() const
{
    return _MTL_msg_NS__UInteger_usedSize((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Heap::currentAllocatedSize() const
{
    return _MTL_msg_NS__UInteger_currentAllocatedSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::HeapType MTL::Heap::type() const
{
    return _MTL_msg_MTL__HeapType_type((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Heap::maxAvailableSize(NS::UInteger alignment)
{
    return _MTL_msg_NS__UInteger_maxAvailableSizeWithAlignment__NS__UInteger((const void*)this, nullptr, alignment);
}

_MTL_INLINE MTL::Buffer* MTL::Heap::newBuffer(NS::UInteger length, MTL::ResourceOptions options)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithLength_options__NS__UInteger_MTL__ResourceOptions((const void*)this, nullptr, length, options);
}

_MTL_INLINE MTL::Texture* MTL::Heap::newTexture(MTL::TextureDescriptor* descriptor)
{
    return _MTL_msg_MTL__Texturep_newTextureWithDescriptor__MTL__TextureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::PurgeableState MTL::Heap::setPurgeableState(MTL::PurgeableState state)
{
    return _MTL_msg_MTL__PurgeableState_setPurgeableState__MTL__PurgeableState((const void*)this, nullptr, state);
}

_MTL_INLINE MTL::Buffer* MTL::Heap::newBuffer(NS::UInteger length, MTL::ResourceOptions options, NS::UInteger offset)
{
    return _MTL_msg_MTL__Bufferp_newBufferWithLength_options_offset__NS__UInteger_MTL__ResourceOptions_NS__UInteger((const void*)this, nullptr, length, options, offset);
}

_MTL_INLINE MTL::Texture* MTL::Heap::newTexture(MTL::TextureDescriptor* descriptor, NS::UInteger offset)
{
    return _MTL_msg_MTL__Texturep_newTextureWithDescriptor_offset__MTL__TextureDescriptorp_NS__UInteger((const void*)this, nullptr, descriptor, offset);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(NS::UInteger size)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithSize__NS__UInteger((const void*)this, nullptr, size);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithDescriptor__MTL__AccelerationStructureDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(NS::UInteger size, NS::UInteger offset)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithSize_offset__NS__UInteger_NS__UInteger((const void*)this, nullptr, size, offset);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(MTL::AccelerationStructureDescriptor* descriptor, NS::UInteger offset)
{
    return _MTL_msg_MTL__AccelerationStructurep_newAccelerationStructureWithDescriptor_offset__MTL__AccelerationStructureDescriptorp_NS__UInteger((const void*)this, nullptr, descriptor, offset);
}
