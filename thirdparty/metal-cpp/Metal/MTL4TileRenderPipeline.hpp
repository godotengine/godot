//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4TileRenderPipeline.hpp
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
class FunctionDescriptor;
class StaticLinkingDescriptor;
class TileRenderPipelineDescriptor;
}

namespace MTL
{
class TileRenderPipelineColorAttachmentDescriptorArray;
}

namespace MTL4
{
class TileRenderPipelineDescriptor : public NS::Copying<TileRenderPipelineDescriptor, PipelineDescriptor>
{
public:
    static TileRenderPipelineDescriptor*                   alloc();

    MTL::TileRenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    TileRenderPipelineDescriptor*                          init();

    NS::UInteger                                           maxTotalThreadsPerThreadgroup() const;

    NS::UInteger                                           rasterSampleCount() const;

    MTL::Size                                              requiredThreadsPerThreadgroup() const;

    void                                                   reset();

    void                                                   setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);

    void                                                   setRasterSampleCount(NS::UInteger rasterSampleCount);

    void                                                   setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);

    void                                                   setStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* staticLinkingDescriptor);

    void                                                   setSupportBinaryLinking(bool supportBinaryLinking);

    void                                                   setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize);

    void                                                   setTileFunctionDescriptor(const MTL4::FunctionDescriptor* tileFunctionDescriptor);

    StaticLinkingDescriptor*                               staticLinkingDescriptor() const;

    bool                                                   supportBinaryLinking() const;

    bool                                                   threadgroupSizeMatchesTileSize() const;

    FunctionDescriptor*                                    tileFunctionDescriptor() const;
};

}
_MTL_INLINE MTL4::TileRenderPipelineDescriptor* MTL4::TileRenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::TileRenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4TileRenderPipelineDescriptor));
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL4::TileRenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::TileRenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL4::TileRenderPipelineDescriptor* MTL4::TileRenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL4::TileRenderPipelineDescriptor>();
}

_MTL_INLINE NS::UInteger MTL4::TileRenderPipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE NS::UInteger MTL4::TileRenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE MTL::Size MTL4::TileRenderPipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerThreadgroup_), maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerThreadgroup_), requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* staticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStaticLinkingDescriptor_), staticLinkingDescriptor);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setSupportBinaryLinking(bool supportBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportBinaryLinking_), supportBinaryLinking);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupSizeMatchesTileSize_), threadgroupSizeMatchesTileSize);
}

_MTL_INLINE void MTL4::TileRenderPipelineDescriptor::setTileFunctionDescriptor(const MTL4::FunctionDescriptor* tileFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileFunctionDescriptor_), tileFunctionDescriptor);
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::TileRenderPipelineDescriptor::staticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(staticLinkingDescriptor));
}

_MTL_INLINE bool MTL4::TileRenderPipelineDescriptor::supportBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportBinaryLinking));
}

_MTL_INLINE bool MTL4::TileRenderPipelineDescriptor::threadgroupSizeMatchesTileSize() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(threadgroupSizeMatchesTileSize));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::TileRenderPipelineDescriptor::tileFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(tileFunctionDescriptor));
}
