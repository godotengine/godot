//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLCounters.hpp
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
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"
#include <cstdint>

namespace MTL
{
class CounterSampleBufferDescriptor;
class CounterSet;
class Device;
_MTL_ENUM(NS::Integer, CounterSampleBufferError) {
    CounterSampleBufferErrorOutOfMemory = 0,
    CounterSampleBufferErrorInvalid = 1,
    CounterSampleBufferErrorInternal = 2,
};

using CommonCounter = NS::String*;
using CommonCounterSet = NS::String*;

static const NS::UInteger CounterErrorValue = static_cast<NS::UInteger>(~0ULL);
static const NS::UInteger CounterDontSample = static_cast<NS::UInteger>(-1);
_MTL_CONST(NS::ErrorDomain, CounterErrorDomain);
_MTL_CONST(CommonCounter, CommonCounterTimestamp);
_MTL_CONST(CommonCounter, CommonCounterTessellationInputPatches);
_MTL_CONST(CommonCounter, CommonCounterVertexInvocations);
_MTL_CONST(CommonCounter, CommonCounterPostTessellationVertexInvocations);
_MTL_CONST(CommonCounter, CommonCounterClipperInvocations);
_MTL_CONST(CommonCounter, CommonCounterClipperPrimitivesOut);
_MTL_CONST(CommonCounter, CommonCounterFragmentInvocations);
_MTL_CONST(CommonCounter, CommonCounterFragmentsPassed);
_MTL_CONST(CommonCounter, CommonCounterComputeKernelInvocations);
_MTL_CONST(CommonCounter, CommonCounterTotalCycles);
_MTL_CONST(CommonCounter, CommonCounterVertexCycles);
_MTL_CONST(CommonCounter, CommonCounterTessellationCycles);
_MTL_CONST(CommonCounter, CommonCounterPostTessellationVertexCycles);
_MTL_CONST(CommonCounter, CommonCounterFragmentCycles);
_MTL_CONST(CommonCounter, CommonCounterRenderTargetWriteCycles);
_MTL_CONST(CommonCounterSet, CommonCounterSetTimestamp);
_MTL_CONST(CommonCounterSet, CommonCounterSetStageUtilization);
_MTL_CONST(CommonCounterSet, CommonCounterSetStatistic);
struct CounterResultTimestamp
{
    uint64_t timestamp;
} _MTL_PACKED;

struct CounterResultStageUtilization
{
    uint64_t totalCycles;
    uint64_t vertexCycles;
    uint64_t tessellationCycles;
    uint64_t postTessellationVertexCycles;
    uint64_t fragmentCycles;
    uint64_t renderTargetCycles;
} _MTL_PACKED;

struct CounterResultStatistic
{
    uint64_t tessellationInputPatches;
    uint64_t vertexInvocations;
    uint64_t postTessellationVertexInvocations;
    uint64_t clipperInvocations;
    uint64_t clipperPrimitivesOut;
    uint64_t fragmentInvocations;
    uint64_t fragmentsPassed;
    uint64_t computeKernelInvocations;
} _MTL_PACKED;

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

    CounterSet*                           counterSet() const;

    CounterSampleBufferDescriptor*        init();

    NS::String*                           label() const;

    NS::UInteger                          sampleCount() const;

    void                                  setCounterSet(const MTL::CounterSet* counterSet);

    void                                  setLabel(const NS::String* label);

    void                                  setSampleCount(NS::UInteger sampleCount);

    void                                  setStorageMode(MTL::StorageMode storageMode);
    StorageMode                           storageMode() const;
};
class CounterSampleBuffer : public NS::Referencing<CounterSampleBuffer>
{
public:
    Device*      device() const;

    NS::String*  label() const;

    NS::Data*    resolveCounterRange(NS::Range range);

    NS::UInteger sampleCount() const;
};

}

_MTL_PRIVATE_DEF_CONST(NS::ErrorDomain, CounterErrorDomain);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterTimestamp);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterTessellationInputPatches);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterVertexInvocations);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterPostTessellationVertexInvocations);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterClipperInvocations);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterClipperPrimitivesOut);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterFragmentInvocations);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterFragmentsPassed);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterComputeKernelInvocations);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterTotalCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterVertexCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterTessellationCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterPostTessellationVertexCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterFragmentCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounter, CommonCounterRenderTargetWriteCycles);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounterSet, CommonCounterSetTimestamp);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounterSet, CommonCounterSetStageUtilization);
_MTL_PRIVATE_DEF_CONST(MTL::CommonCounterSet, CommonCounterSetStatistic);

_MTL_INLINE NS::String* MTL::Counter::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE NS::Array* MTL::CounterSet::counters() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(counters));
}

_MTL_INLINE NS::String* MTL::CounterSet::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::CounterSampleBufferDescriptor* MTL::CounterSampleBufferDescriptor::alloc()
{
    return NS::Object::alloc<MTL::CounterSampleBufferDescriptor>(_MTL_PRIVATE_CLS(MTLCounterSampleBufferDescriptor));
}

_MTL_INLINE MTL::CounterSet* MTL::CounterSampleBufferDescriptor::counterSet() const
{
    return Object::sendMessage<MTL::CounterSet*>(this, _MTL_PRIVATE_SEL(counterSet));
}

_MTL_INLINE MTL::CounterSampleBufferDescriptor* MTL::CounterSampleBufferDescriptor::init()
{
    return NS::Object::init<MTL::CounterSampleBufferDescriptor>();
}

_MTL_INLINE NS::String* MTL::CounterSampleBufferDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::CounterSampleBufferDescriptor::sampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(sampleCount));
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setCounterSet(const MTL::CounterSet* counterSet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCounterSet_), counterSet);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleCount_), sampleCount);
}

_MTL_INLINE void MTL::CounterSampleBufferDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStorageMode_), storageMode);
}

_MTL_INLINE MTL::StorageMode MTL::CounterSampleBufferDescriptor::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}

_MTL_INLINE MTL::Device* MTL::CounterSampleBuffer::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::CounterSampleBuffer::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::Data* MTL::CounterSampleBuffer::resolveCounterRange(NS::Range range)
{
    return Object::sendMessage<NS::Data*>(this, _MTL_PRIVATE_SEL(resolveCounterRange_), range);
}

_MTL_INLINE NS::UInteger MTL::CounterSampleBuffer::sampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(sampleCount));
}
