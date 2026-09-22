//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4RenderPass.hpp
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
#include "MTLRenderPass.hpp"

namespace MTL4
{
class RenderPassDescriptor;
}

namespace MTL
{
class Buffer;
class RasterizationRateMap;
class RenderPassColorAttachmentDescriptorArray;
class RenderPassDepthAttachmentDescriptor;
class RenderPassStencilAttachmentDescriptor;
struct SamplePosition;
}

namespace MTL4
{
class RenderPassDescriptor : public NS::Copying<RenderPassDescriptor>
{
public:
    static RenderPassDescriptor*                   alloc();

    MTL::RenderPassColorAttachmentDescriptorArray* colorAttachments() const;

    NS::UInteger                                   defaultRasterSampleCount() const;

    MTL::RenderPassDepthAttachmentDescriptor*      depthAttachment() const;

    NS::UInteger                                   getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count);

    NS::UInteger                                   imageblockSampleLength() const;

    RenderPassDescriptor*                          init();

    MTL::RasterizationRateMap*                     rasterizationRateMap() const;

    NS::UInteger                                   renderTargetArrayLength() const;

    NS::UInteger                                   renderTargetHeight() const;

    NS::UInteger                                   renderTargetWidth() const;

    void                                           setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount);

    void                                           setDepthAttachment(const MTL::RenderPassDepthAttachmentDescriptor* depthAttachment);

    void                                           setImageblockSampleLength(NS::UInteger imageblockSampleLength);

    void                                           setRasterizationRateMap(const MTL::RasterizationRateMap* rasterizationRateMap);

    void                                           setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength);

    void                                           setRenderTargetHeight(NS::UInteger renderTargetHeight);

    void                                           setRenderTargetWidth(NS::UInteger renderTargetWidth);

    void                                           setSamplePositions(const MTL::SamplePosition* positions, NS::UInteger count);

    void                                           setStencilAttachment(const MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment);

    void                                           setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);

    void                                           setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength);

    void                                           setTileHeight(NS::UInteger tileHeight);

    void                                           setTileWidth(NS::UInteger tileWidth);

    void                                           setVisibilityResultBuffer(const MTL::Buffer* visibilityResultBuffer);

    void                                           setVisibilityResultType(MTL::VisibilityResultType visibilityResultType);

    MTL::RenderPassStencilAttachmentDescriptor*    stencilAttachment() const;

    bool                                           supportColorAttachmentMapping() const;

    NS::UInteger                                   threadgroupMemoryLength() const;

    NS::UInteger                                   tileHeight() const;

    NS::UInteger                                   tileWidth() const;

    MTL::Buffer*                                   visibilityResultBuffer() const;

    MTL::VisibilityResultType                      visibilityResultType() const;
};

}
_MTL_INLINE MTL4::RenderPassDescriptor* MTL4::RenderPassDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::RenderPassDescriptor>(_MTL_PRIVATE_CLS(MTL4RenderPassDescriptor));
}

_MTL_INLINE MTL::RenderPassColorAttachmentDescriptorArray* MTL4::RenderPassDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL::RenderPassColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::defaultRasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(defaultRasterSampleCount));
}

_MTL_INLINE MTL::RenderPassDepthAttachmentDescriptor* MTL4::RenderPassDescriptor::depthAttachment() const
{
    return Object::sendMessage<MTL::RenderPassDepthAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(depthAttachment));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::getSamplePositions(MTL::SamplePosition* positions, NS::UInteger count)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(getSamplePositions_count_), positions, count);
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::imageblockSampleLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(imageblockSampleLength));
}

_MTL_INLINE MTL4::RenderPassDescriptor* MTL4::RenderPassDescriptor::init()
{
    return NS::Object::init<MTL4::RenderPassDescriptor>();
}

_MTL_INLINE MTL::RasterizationRateMap* MTL4::RenderPassDescriptor::rasterizationRateMap() const
{
    return Object::sendMessage<MTL::RasterizationRateMap*>(this, _MTL_PRIVATE_SEL(rasterizationRateMap));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetArrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetArrayLength));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetHeight));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::renderTargetWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(renderTargetWidth));
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setDefaultRasterSampleCount(NS::UInteger defaultRasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDefaultRasterSampleCount_), defaultRasterSampleCount);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setDepthAttachment(const MTL::RenderPassDepthAttachmentDescriptor* depthAttachment)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthAttachment_), depthAttachment);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setImageblockSampleLength(NS::UInteger imageblockSampleLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setImageblockSampleLength_), imageblockSampleLength);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setRasterizationRateMap(const MTL::RasterizationRateMap* rasterizationRateMap)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationRateMap_), rasterizationRateMap);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setRenderTargetArrayLength(NS::UInteger renderTargetArrayLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetArrayLength_), renderTargetArrayLength);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setRenderTargetHeight(NS::UInteger renderTargetHeight)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetHeight_), renderTargetHeight);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setRenderTargetWidth(NS::UInteger renderTargetWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderTargetWidth_), renderTargetWidth);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setSamplePositions(const MTL::SamplePosition* positions, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplePositions_count_), positions, count);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setStencilAttachment(const MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilAttachment_), stencilAttachment);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportColorAttachmentMapping_), supportColorAttachmentMapping);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setThreadgroupMemoryLength(NS::UInteger threadgroupMemoryLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_), threadgroupMemoryLength);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setTileHeight(NS::UInteger tileHeight)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileHeight_), tileHeight);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setTileWidth(NS::UInteger tileWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileWidth_), tileWidth);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setVisibilityResultBuffer(const MTL::Buffer* visibilityResultBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultBuffer_), visibilityResultBuffer);
}

_MTL_INLINE void MTL4::RenderPassDescriptor::setVisibilityResultType(MTL::VisibilityResultType visibilityResultType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultType_), visibilityResultType);
}

_MTL_INLINE MTL::RenderPassStencilAttachmentDescriptor* MTL4::RenderPassDescriptor::stencilAttachment() const
{
    return Object::sendMessage<MTL::RenderPassStencilAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(stencilAttachment));
}

_MTL_INLINE bool MTL4::RenderPassDescriptor::supportColorAttachmentMapping() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportColorAttachmentMapping));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::threadgroupMemoryLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadgroupMemoryLength));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::tileHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileHeight));
}

_MTL_INLINE NS::UInteger MTL4::RenderPassDescriptor::tileWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileWidth));
}

_MTL_INLINE MTL::Buffer* MTL4::RenderPassDescriptor::visibilityResultBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(visibilityResultBuffer));
}

_MTL_INLINE MTL::VisibilityResultType MTL4::RenderPassDescriptor::visibilityResultType() const
{
    return Object::sendMessage<MTL::VisibilityResultType>(this, _MTL_PRIVATE_SEL(visibilityResultType));
}
