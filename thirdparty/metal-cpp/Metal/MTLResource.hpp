//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLResource.hpp
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
#include <mach/mach.h>

namespace MTL
{
class Device;
class Heap;
_MTL_ENUM(NS::UInteger, PurgeableState) {
    PurgeableStateKeepCurrent = 1,
    PurgeableStateNonVolatile = 2,
    PurgeableStateVolatile = 3,
    PurgeableStateEmpty = 4,
};

_MTL_ENUM(NS::UInteger, CPUCacheMode) {
    CPUCacheModeDefaultCache = 0,
    CPUCacheModeWriteCombined = 1,
};

_MTL_ENUM(NS::UInteger, StorageMode) {
    StorageModeShared = 0,
    StorageModeManaged = 1,
    StorageModePrivate = 2,
    StorageModeMemoryless = 3,
};

_MTL_ENUM(NS::UInteger, HazardTrackingMode) {
    HazardTrackingModeDefault = 0,
    HazardTrackingModeUntracked = 1,
    HazardTrackingModeTracked = 2,
};

_MTL_ENUM(NS::Integer, SparsePageSize) {
    SparsePageSize16 = 101,
    SparsePageSize64 = 102,
    SparsePageSize256 = 103,
};

_MTL_ENUM(NS::Integer, BufferSparseTier) {
    BufferSparseTierNone = 0,
    BufferSparseTier1 = 1,
};

_MTL_ENUM(NS::Integer, TextureSparseTier) {
    TextureSparseTierNone = 0,
    TextureSparseTier1 = 1,
    TextureSparseTier2 = 2,
};

_MTL_OPTIONS(NS::UInteger, ResourceOptions) {
    ResourceCPUCacheModeDefaultCache = 0,
    ResourceCPUCacheModeWriteCombined = 1,
    ResourceStorageModeShared = 0,
    ResourceStorageModeManaged = 1 << 4,
    ResourceStorageModePrivate = 1 << 5,
    ResourceStorageModeMemoryless = 1 << 5,
    ResourceHazardTrackingModeDefault = 0,
    ResourceHazardTrackingModeUntracked = 1 << 8,
    ResourceHazardTrackingModeTracked = 1 << 9,
    ResourceOptionCPUCacheModeDefault = 0,
    ResourceOptionCPUCacheModeWriteCombined = 1,
};

class Resource : public NS::Referencing<Resource, Allocation>
{
public:
    NS::UInteger       allocatedSize() const;

    CPUCacheMode       cpuCacheMode() const;

    Device*            device() const;

    HazardTrackingMode hazardTrackingMode() const;

    Heap*              heap() const;
    NS::UInteger       heapOffset() const;

    bool               isAliasable();

    NS::String*        label() const;

    void               makeAliasable();

    ResourceOptions    resourceOptions() const;

    void               setLabel(const NS::String* label);

    kern_return_t      setOwner(task_id_token_t task_id_token);

    PurgeableState     setPurgeableState(MTL::PurgeableState state);

    StorageMode        storageMode() const;
};

}
_MTL_INLINE NS::UInteger MTL::Resource::allocatedSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(allocatedSize));
}

_MTL_INLINE MTL::CPUCacheMode MTL::Resource::cpuCacheMode() const
{
    return Object::sendMessage<MTL::CPUCacheMode>(this, _MTL_PRIVATE_SEL(cpuCacheMode));
}

_MTL_INLINE MTL::Device* MTL::Resource::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::HazardTrackingMode MTL::Resource::hazardTrackingMode() const
{
    return Object::sendMessage<MTL::HazardTrackingMode>(this, _MTL_PRIVATE_SEL(hazardTrackingMode));
}

_MTL_INLINE MTL::Heap* MTL::Resource::heap() const
{
    return Object::sendMessage<MTL::Heap*>(this, _MTL_PRIVATE_SEL(heap));
}

_MTL_INLINE NS::UInteger MTL::Resource::heapOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(heapOffset));
}

_MTL_INLINE bool MTL::Resource::isAliasable()
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isAliasable));
}

_MTL_INLINE NS::String* MTL::Resource::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::Resource::makeAliasable()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(makeAliasable));
}

_MTL_INLINE MTL::ResourceOptions MTL::Resource::resourceOptions() const
{
    return Object::sendMessage<MTL::ResourceOptions>(this, _MTL_PRIVATE_SEL(resourceOptions));
}

_MTL_INLINE void MTL::Resource::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE kern_return_t MTL::Resource::setOwner(task_id_token_t task_id_token)
{
    return Object::sendMessage<kern_return_t>(this, _MTL_PRIVATE_SEL(setOwnerWithIdentity_), task_id_token);
}

_MTL_INLINE MTL::PurgeableState MTL::Resource::setPurgeableState(MTL::PurgeableState state)
{
    return Object::sendMessage<MTL::PurgeableState>(this, _MTL_PRIVATE_SEL(setPurgeableState_), state);
}

_MTL_INLINE MTL::StorageMode MTL::Resource::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}
