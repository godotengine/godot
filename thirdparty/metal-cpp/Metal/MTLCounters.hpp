#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
    enum StorageMode : NS::UInteger;
}
namespace NS {
    class Array;
    class Data;
    class String;
}

namespace MTL
{

extern MTL::CommonCounter CommonCounterTimestamp __asm__("_MTLCommonCounterTimestamp");
extern MTL::CommonCounter CommonCounterTessellationInputPatches __asm__("_MTLCommonCounterTessellationInputPatches");
extern MTL::CommonCounter CommonCounterVertexInvocations __asm__("_MTLCommonCounterVertexInvocations");
extern MTL::CommonCounter CommonCounterPostTessellationVertexInvocations __asm__("_MTLCommonCounterPostTessellationVertexInvocations");
extern MTL::CommonCounter CommonCounterClipperInvocations __asm__("_MTLCommonCounterClipperInvocations");
extern MTL::CommonCounter CommonCounterClipperPrimitivesOut __asm__("_MTLCommonCounterClipperPrimitivesOut");
extern MTL::CommonCounter CommonCounterFragmentInvocations __asm__("_MTLCommonCounterFragmentInvocations");
extern MTL::CommonCounter CommonCounterFragmentsPassed __asm__("_MTLCommonCounterFragmentsPassed");
extern MTL::CommonCounter CommonCounterComputeKernelInvocations __asm__("_MTLCommonCounterComputeKernelInvocations");
extern MTL::CommonCounter CommonCounterTotalCycles __asm__("_MTLCommonCounterTotalCycles");
extern MTL::CommonCounter CommonCounterVertexCycles __asm__("_MTLCommonCounterVertexCycles");
extern MTL::CommonCounter CommonCounterTessellationCycles __asm__("_MTLCommonCounterTessellationCycles");
extern MTL::CommonCounter CommonCounterPostTessellationVertexCycles __asm__("_MTLCommonCounterPostTessellationVertexCycles");
extern MTL::CommonCounter CommonCounterFragmentCycles __asm__("_MTLCommonCounterFragmentCycles");
extern MTL::CommonCounter CommonCounterRenderTargetWriteCycles __asm__("_MTLCommonCounterRenderTargetWriteCycles");
extern MTL::CommonCounterSet CommonCounterSetTimestamp __asm__("_MTLCommonCounterSetTimestamp");
extern MTL::CommonCounterSet CommonCounterSetStageUtilization __asm__("_MTLCommonCounterSetStageUtilization");
extern MTL::CommonCounterSet CommonCounterSetStatistic __asm__("_MTLCommonCounterSetStatistic");
extern NS::ErrorDomain const CounterErrorDomain __asm__("_MTLCounterErrorDomain");
_MTL_ENUM(NS::Integer, CounterSampleBufferError) {
    CounterSampleBufferErrorOutOfMemory = 0,
    CounterSampleBufferErrorInvalid = 1,
    CounterSampleBufferErrorInternal = 2,
};


class Counter;
class CounterSet;
class CounterSampleBufferDescriptor;
class CounterSampleBuffer;

class Counter : public NS::Referencing<Counter>
{
public:
    NS::String* name() const;

};

class CounterSet : public NS::Referencing<CounterSet>
{
public:
    NS::Array*  counters() const;
    NS::String* name() const;

};

class CounterSampleBufferDescriptor : public NS::Copying<CounterSampleBufferDescriptor>
{
public:
    static CounterSampleBufferDescriptor* alloc();
    CounterSampleBufferDescriptor*        init() const;

    MTL::CounterSet* counterSet() const;
    NS::String*      label() const;
    NS::UInteger     sampleCount() const;
    void             setCounterSet(MTL::CounterSet* counterSet);
    void             setLabel(NS::String* label);
    void             setSampleCount(NS::UInteger sampleCount);
    void             setStorageMode(MTL::StorageMode storageMode);
    MTL::StorageMode storageMode() const;

};

class CounterSampleBuffer : public NS::Referencing<CounterSampleBuffer>
{
public:
    MTL::Device* device() const;
    NS::String*  label() const;
    NS::Data*    resolveCounterRange(NS::Range range);
    NS::UInteger sampleCount() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCounter;
extern "C" void *OBJC_CLASS_$_MTLCounterSet;
extern "C" void *OBJC_CLASS_$_MTLCounterSampleBufferDescriptor;
extern "C" void *OBJC_CLASS_$_MTLCounterSampleBuffer;

_MTL_INLINE NS::String* MTL::Counter::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::CounterSet::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::CounterSet::counters() const
{
    return _MTL_msg_NS__Arrayp_counters((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBufferDescriptor* MTL::CounterSampleBufferDescriptor::alloc()
{
    return _MTL_msg_MTL__CounterSampleBufferDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLCounterSampleBufferDescriptor, nullptr);
}

_MTL_INLINE MTL::CounterSampleBufferDescriptor* MTL::CounterSampleBufferDescriptor::init() const
{
    return _MTL_msg_MTL__CounterSampleBufferDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSet* MTL::CounterSampleBufferDescriptor::counterSet() const
{
    return _MTL_msg_MTL__CounterSetp_counterSet((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setCounterSet(MTL::CounterSet* counterSet)
{
    _MTL_msg_v_setCounterSet__MTL__CounterSetp((const void*)this, nullptr, counterSet);
}

_MTL_INLINE NS::String* MTL::CounterSampleBufferDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::StorageMode MTL::CounterSampleBufferDescriptor::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    _MTL_msg_v_setStorageMode__MTL__StorageMode((const void*)this, nullptr, storageMode);
}

_MTL_INLINE NS::UInteger MTL::CounterSampleBufferDescriptor::sampleCount() const
{
    return _MTL_msg_NS__UInteger_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    _MTL_msg_v_setSampleCount__NS__UInteger((const void*)this, nullptr, sampleCount);
}

_MTL_INLINE MTL::Device* MTL::CounterSampleBuffer::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::CounterSampleBuffer::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::CounterSampleBuffer::sampleCount() const
{
    return _MTL_msg_NS__UInteger_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE NS::Data* MTL::CounterSampleBuffer::resolveCounterRange(NS::Range range)
{
    return _MTL_msg_NS__Datap_resolveCounterRange__NS__Range((const void*)this, nullptr, range);
}
