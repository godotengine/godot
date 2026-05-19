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
    class Device;
    class Heap;
}
namespace NS {
    class String;
}

namespace MTL
{

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

_MTL_OPTIONS(NS::UInteger, ResourceOptions) {
    ResourceCPUCacheModeDefaultCache = 0,
    ResourceCPUCacheModeWriteCombined = 1,
    ResourceStorageModeShared = 0,
    ResourceStorageModeManaged = 16,
    ResourceStorageModePrivate = 32,
    ResourceStorageModeMemoryless = 48,
    ResourceHazardTrackingModeDefault = 0,
    ResourceHazardTrackingModeUntracked = 256,
    ResourceHazardTrackingModeTracked = 512,
    ResourceOptionCPUCacheModeDefault = ResourceCPUCacheModeDefaultCache,
    ResourceOptionCPUCacheModeWriteCombined = ResourceCPUCacheModeWriteCombined,
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


class Resource : public NS::Referencing<Resource, MTL::Allocation>
{
public:
    NS::UInteger            allocatedSize() const;
    MTL::CPUCacheMode       cpuCacheMode() const;
    MTL::Device*            device() const;
    MTL::HazardTrackingMode hazardTrackingMode() const;
    MTL::Heap*              heap() const;
    NS::UInteger            heapOffset() const;
    bool                    isAliasable();
    NS::String*             label() const;
    void                    makeAliasable();
    MTL::ResourceOptions    resourceOptions() const;
    void                    setLabel(NS::String* label);
    void*                   setOwner(void* task_id_token);
    MTL::PurgeableState     setPurgeableState(MTL::PurgeableState state);
    MTL::StorageMode        storageMode() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLResource;

_MTL_INLINE NS::String* MTL::Resource::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Resource::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::Resource::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::CPUCacheMode MTL::Resource::cpuCacheMode() const
{
    return _MTL_msg_MTL__CPUCacheMode_cpuCacheMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::StorageMode MTL::Resource::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::HazardTrackingMode MTL::Resource::hazardTrackingMode() const
{
    return _MTL_msg_MTL__HazardTrackingMode_hazardTrackingMode((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceOptions MTL::Resource::resourceOptions() const
{
    return _MTL_msg_MTL__ResourceOptions_resourceOptions((const void*)this, nullptr);
}

_MTL_INLINE MTL::Heap* MTL::Resource::heap() const
{
    return _MTL_msg_MTL__Heapp_heap((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Resource::heapOffset() const
{
    return _MTL_msg_NS__UInteger_heapOffset((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Resource::allocatedSize() const
{
    return _MTL_msg_NS__UInteger_allocatedSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::PurgeableState MTL::Resource::setPurgeableState(MTL::PurgeableState state)
{
    return _MTL_msg_MTL__PurgeableState_setPurgeableState__MTL__PurgeableState((const void*)this, nullptr, state);
}

_MTL_INLINE void MTL::Resource::makeAliasable()
{
    _MTL_msg_v_makeAliasable((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Resource::isAliasable()
{
    return _MTL_msg_bool_isAliasable((const void*)this, nullptr);
}

_MTL_INLINE void* MTL::Resource::setOwner(void* task_id_token)
{
    return _MTL_msg_voidp_setOwnerWithIdentity__voidp((const void*)this, nullptr, task_id_token);
}
