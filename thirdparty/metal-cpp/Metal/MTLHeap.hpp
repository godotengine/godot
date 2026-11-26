//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLHeap.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLAllocation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"

namespace MTL
{
class AccelerationStructure;
class AccelerationStructureDescriptor;
class Buffer;
class Device;
class HeapDescriptor;
class Texture;
class TextureDescriptor;
_MTL_ENUM(NS::Integer, HeapType) {
    HeapTypeAutomatic = 0,
    HeapTypePlacement = 1,
    HeapTypeSparse = 2,
};

class HeapDescriptor : public NS::Copying<HeapDescriptor>
{
public:
    static HeapDescriptor* alloc();

    CPUCacheMode           cpuCacheMode() const;

    HazardTrackingMode     hazardTrackingMode() const;

    HeapDescriptor*        init();

    SparsePageSize         maxCompatiblePlacementSparsePageSize() const;

    ResourceOptions        resourceOptions() const;

    void                   setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);

    void                   setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);

    void                   setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize maxCompatiblePlacementSparsePageSize);

    void                   setResourceOptions(MTL::ResourceOptions resourceOptions);

    void                   setSize(NS::UInteger size);

    void                   setSparsePageSize(MTL::SparsePageSize sparsePageSize);

    void                   setStorageMode(MTL::StorageMode storageMode);

    void                   setType(MTL::HeapType type);

    NS::UInteger           size() const;
    SparsePageSize         sparsePageSize() const;

    StorageMode            storageMode() const;

    HeapType               type() const;
};
class Heap : public NS::Referencing<Heap, Allocation>
{
public:
    CPUCacheMode           cpuCacheMode() const;

    NS::UInteger           currentAllocatedSize() const;

    Device*                device() const;

    HazardTrackingMode     hazardTrackingMode() const;

    NS::String*            label() const;

    NS::UInteger           maxAvailableSize(NS::UInteger alignment);

    AccelerationStructure* newAccelerationStructure(NS::UInteger size);
    AccelerationStructure* newAccelerationStructure(const MTL::AccelerationStructureDescriptor* descriptor);
    AccelerationStructure* newAccelerationStructure(NS::UInteger size, NS::UInteger offset);
    AccelerationStructure* newAccelerationStructure(const MTL::AccelerationStructureDescriptor* descriptor, NS::UInteger offset);

    Buffer*                newBuffer(NS::UInteger length, MTL::ResourceOptions options);
    Buffer*                newBuffer(NS::UInteger length, MTL::ResourceOptions options, NS::UInteger offset);

    Texture*               newTexture(const MTL::TextureDescriptor* descriptor);
    Texture*               newTexture(const MTL::TextureDescriptor* descriptor, NS::UInteger offset);

    ResourceOptions        resourceOptions() const;

    void                   setLabel(const NS::String* label);

    PurgeableState         setPurgeableState(MTL::PurgeableState state);

    NS::UInteger           size() const;

    StorageMode            storageMode() const;

    HeapType               type() const;

    NS::UInteger           usedSize() const;
};

}
_MTL_INLINE MTL::HeapDescriptor* MTL::HeapDescriptor::alloc()
{
    return NS::Object::alloc<MTL::HeapDescriptor>(_MTL_PRIVATE_CLS(MTLHeapDescriptor));
}

_MTL_INLINE MTL::CPUCacheMode MTL::HeapDescriptor::cpuCacheMode() const
{
    return Object::sendMessage<MTL::CPUCacheMode>(this, _MTL_PRIVATE_SEL(cpuCacheMode));
}

_MTL_INLINE MTL::HazardTrackingMode MTL::HeapDescriptor::hazardTrackingMode() const
{
    return Object::sendMessage<MTL::HazardTrackingMode>(this, _MTL_PRIVATE_SEL(hazardTrackingMode));
}

_MTL_INLINE MTL::HeapDescriptor* MTL::HeapDescriptor::init()
{
    return NS::Object::init<MTL::HeapDescriptor>();
}

_MTL_INLINE MTL::SparsePageSize MTL::HeapDescriptor::maxCompatiblePlacementSparsePageSize() const
{
    return Object::sendMessage<MTL::SparsePageSize>(this, _MTL_PRIVATE_SEL(maxCompatiblePlacementSparsePageSize));
}

_MTL_INLINE MTL::ResourceOptions MTL::HeapDescriptor::resourceOptions() const
{
    return Object::sendMessage<MTL::ResourceOptions>(this, _MTL_PRIVATE_SEL(resourceOptions));
}

_MTL_INLINE void MTL::HeapDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCpuCacheMode_), cpuCacheMode);
}

_MTL_INLINE void MTL::HeapDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setHazardTrackingMode_), hazardTrackingMode);
}

_MTL_INLINE void MTL::HeapDescriptor::setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize maxCompatiblePlacementSparsePageSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCompatiblePlacementSparsePageSize_), maxCompatiblePlacementSparsePageSize);
}

_MTL_INLINE void MTL::HeapDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResourceOptions_), resourceOptions);
}

_MTL_INLINE void MTL::HeapDescriptor::setSize(NS::UInteger size)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSize_), size);
}

_MTL_INLINE void MTL::HeapDescriptor::setSparsePageSize(MTL::SparsePageSize sparsePageSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSparsePageSize_), sparsePageSize);
}

_MTL_INLINE void MTL::HeapDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStorageMode_), storageMode);
}

_MTL_INLINE void MTL::HeapDescriptor::setType(MTL::HeapType type)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setType_), type);
}

_MTL_INLINE NS::UInteger MTL::HeapDescriptor::size() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(size));
}

_MTL_INLINE MTL::SparsePageSize MTL::HeapDescriptor::sparsePageSize() const
{
    return Object::sendMessage<MTL::SparsePageSize>(this, _MTL_PRIVATE_SEL(sparsePageSize));
}

_MTL_INLINE MTL::StorageMode MTL::HeapDescriptor::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}

_MTL_INLINE MTL::HeapType MTL::HeapDescriptor::type() const
{
    return Object::sendMessage<MTL::HeapType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE MTL::CPUCacheMode MTL::Heap::cpuCacheMode() const
{
    return Object::sendMessage<MTL::CPUCacheMode>(this, _MTL_PRIVATE_SEL(cpuCacheMode));
}

_MTL_INLINE NS::UInteger MTL::Heap::currentAllocatedSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(currentAllocatedSize));
}

_MTL_INLINE MTL::Device* MTL::Heap::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::HazardTrackingMode MTL::Heap::hazardTrackingMode() const
{
    return Object::sendMessage<MTL::HazardTrackingMode>(this, _MTL_PRIVATE_SEL(hazardTrackingMode));
}

_MTL_INLINE NS::String* MTL::Heap::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::Heap::maxAvailableSize(NS::UInteger alignment)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxAvailableSizeWithAlignment_), alignment);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(NS::UInteger size)
{
    return Object::sendMessage<MTL::AccelerationStructure*>(this, _MTL_PRIVATE_SEL(newAccelerationStructureWithSize_), size);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(const MTL::AccelerationStructureDescriptor* descriptor)
{
    return Object::sendMessage<MTL::AccelerationStructure*>(this, _MTL_PRIVATE_SEL(newAccelerationStructureWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(NS::UInteger size, NS::UInteger offset)
{
    return Object::sendMessage<MTL::AccelerationStructure*>(this, _MTL_PRIVATE_SEL(newAccelerationStructureWithSize_offset_), size, offset);
}

_MTL_INLINE MTL::AccelerationStructure* MTL::Heap::newAccelerationStructure(const MTL::AccelerationStructureDescriptor* descriptor, NS::UInteger offset)
{
    return Object::sendMessage<MTL::AccelerationStructure*>(this, _MTL_PRIVATE_SEL(newAccelerationStructureWithDescriptor_offset_), descriptor, offset);
}

_MTL_INLINE MTL::Buffer* MTL::Heap::newBuffer(NS::UInteger length, MTL::ResourceOptions options)
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(newBufferWithLength_options_), length, options);
}

_MTL_INLINE MTL::Buffer* MTL::Heap::newBuffer(NS::UInteger length, MTL::ResourceOptions options, NS::UInteger offset)
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(newBufferWithLength_options_offset_), length, options, offset);
}

_MTL_INLINE MTL::Texture* MTL::Heap::newTexture(const MTL::TextureDescriptor* descriptor)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Heap::newTexture(const MTL::TextureDescriptor* descriptor, NS::UInteger offset)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureWithDescriptor_offset_), descriptor, offset);
}

_MTL_INLINE MTL::ResourceOptions MTL::Heap::resourceOptions() const
{
    return Object::sendMessage<MTL::ResourceOptions>(this, _MTL_PRIVATE_SEL(resourceOptions));
}

_MTL_INLINE void MTL::Heap::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::PurgeableState MTL::Heap::setPurgeableState(MTL::PurgeableState state)
{
    return Object::sendMessage<MTL::PurgeableState>(this, _MTL_PRIVATE_SEL(setPurgeableState_), state);
}

_MTL_INLINE NS::UInteger MTL::Heap::size() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(size));
}

_MTL_INLINE MTL::StorageMode MTL::Heap::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}

_MTL_INLINE MTL::HeapType MTL::Heap::type() const
{
    return Object::sendMessage<MTL::HeapType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE NS::UInteger MTL::Heap::usedSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(usedSize));
}
