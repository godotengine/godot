//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4ComputePipeline.hpp
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
#include "MTL4PipelineState.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL4
{
class ComputePipelineDescriptor;
class FunctionDescriptor;
class StaticLinkingDescriptor;

class ComputePipelineDescriptor : public NS::Copying<ComputePipelineDescriptor, PipelineDescriptor>
{
public:
    static ComputePipelineDescriptor* alloc();

    FunctionDescriptor*               computeFunctionDescriptor() const;

    ComputePipelineDescriptor*        init();

    NS::UInteger                      maxTotalThreadsPerThreadgroup() const;

    MTL::Size                         requiredThreadsPerThreadgroup() const;

    void                              reset();

    void                              setComputeFunctionDescriptor(const MTL4::FunctionDescriptor* computeFunctionDescriptor);

    void                              setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);

    void                              setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);

    void                              setStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* staticLinkingDescriptor);

    void                              setSupportBinaryLinking(bool supportBinaryLinking);

    void                              setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);

    void                              setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth);

    StaticLinkingDescriptor*          staticLinkingDescriptor() const;

    bool                              supportBinaryLinking() const;

    IndirectCommandBufferSupportState supportIndirectCommandBuffers() const;

    bool                              threadGroupSizeIsMultipleOfThreadExecutionWidth() const;
};

}
_MTL_INLINE MTL4::ComputePipelineDescriptor* MTL4::ComputePipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::ComputePipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4ComputePipelineDescriptor));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::ComputePipelineDescriptor::computeFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(computeFunctionDescriptor));
}

_MTL_INLINE MTL4::ComputePipelineDescriptor* MTL4::ComputePipelineDescriptor::init()
{
    return NS::Object::init<MTL4::ComputePipelineDescriptor>();
}

_MTL_INLINE NS::UInteger MTL4::ComputePipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE MTL::Size MTL4::ComputePipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setComputeFunctionDescriptor(const MTL4::FunctionDescriptor* computeFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputeFunctionDescriptor_), computeFunctionDescriptor);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerThreadgroup_), maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerThreadgroup_), requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* staticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStaticLinkingDescriptor_), staticLinkingDescriptor);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setSupportBinaryLinking(bool supportBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportBinaryLinking_), supportBinaryLinking);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE void MTL4::ComputePipelineDescriptor::setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadGroupSizeIsMultipleOfThreadExecutionWidth_), threadGroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::ComputePipelineDescriptor::staticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(staticLinkingDescriptor));
}

_MTL_INLINE bool MTL4::ComputePipelineDescriptor::supportBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportBinaryLinking));
}

_MTL_INLINE MTL4::IndirectCommandBufferSupportState MTL4::ComputePipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessage<MTL4::IndirectCommandBufferSupportState>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE bool MTL4::ComputePipelineDescriptor::threadGroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(threadGroupSizeIsMultipleOfThreadExecutionWidth));
}
